#!/usr/bin/env bash
# Run an ANNA forecast (the DANRA ML LAM model of arXiv:2504.09340, see
# configs/model.yaml) from a DINI control forecast.
#
# Intended to be run in the container, where the inference artifact (see
# inference-artifact/README.md) is in ./inference_artifact. It can also be run
# outside the container in the ANNA uv environment, with
# INFERENCE_ARTIFACT_PATH=inference-artifact/build/<artifact-name>.
#
# Steps (see INFERENCE_PLAN.md):
#   1. regrid DINI to the DANRA grid (interior) and, for the DINI boundary,
#      to the 0.25 deg ERA5 boundary box     (src/regrid_dini.py)
#   2. create the inference datastores and neural-lam config
#                                            (src/create_inference_dataset.py)
#   3. build the model's graph               (neural_lam.build_rectangular_graph)
#   4. run the forecast                      (neural_lam.train_model --eval)
#   5. convert the prediction to DANRA-like single/pressure level zarr datasets
#                                            (src/convert_output.py)
#
# Configuration through environment variables:
#   ANALYSIS_TIME      analysis time of the DINI forecast, ISO8601, e.g.
#                      2026-10-01T00:00Z (required)
#   FORECAST_DURATION  ISO8601 duration, multiple of 3h, >= PT12H (default PT18H).
#                      The first prediction is at +6h; neural-lam needs at least
#                      12h to load the model (see create_inference_dataset.py).
#                      The DINI forecast must cover FORECAST_DURATION + 6h, so
#                      at most PT30H for a 36h DINI run
#   BOUNDARY_SOURCE    "dini" (default) or "ifs"
#   DINI_ROOT          DINI forecast zarr root (default
#                      s3://harmonie-zarr/dini/control/{analysis}/)
#   IFS_BOUNDARY_PATH  IFS forecast zarr (required for BOUNDARY_SOURCE=ifs)
#   INFERENCE_WORKDIR  working directory (default ./inference_workdir)
#   INFERENCE_ARTIFACT_PATH  inference artifact (default ./inference_artifact)
#   MLWM_DEBUGGER      set to "ipdb" to debug the python steps on exceptions
#   SKIP_COMPLETED     "true" to skip the regridding, datastore and graph steps
#                      that already completed in INFERENCE_WORKDIR (marked by
#                      a `.completed` file), e.g. to rerun only the forecast.
#                      Only use with the same ANALYSIS_TIME, FORECAST_DURATION
#                      and BOUNDARY_SOURCE as the completed run
#
# Outputs: ${INFERENCE_WORKDIR}/outputs/{single_levels,pressure_levels}.zarr

set -euo pipefail

if [ -f .env ] ; then
    echo "Sourcing local .env file"
    set -a && source .env && set +a
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_UV=${USE_UV:-true}
if [ "$USE_UV" = true ] ; then
    # the ANNA project explicitly, as the forecast step runs in the workdir
    PYTHON="uv run --project ${SCRIPT_DIR} python"
else
    PYTHON="python"
fi

: "${ANALYSIS_TIME:?ANALYSIS_TIME must be set, e.g. 2026-10-01T00:00Z}"
FORECAST_DURATION=${FORECAST_DURATION:-PT18H}
BOUNDARY_SOURCE=${BOUNDARY_SOURCE:-dini}
INFERENCE_WORKDIR=${INFERENCE_WORKDIR:-./inference_workdir}
INFERENCE_ARTIFACT_PATH=${INFERENCE_ARTIFACT_PATH:-./inference_artifact}
SKIP_COMPLETED=${SKIP_COMPLETED:-false}

mkdir -p "${INFERENCE_WORKDIR}"
WORKDIR="$(cd "${INFERENCE_WORKDIR}" && pwd)"
ARTIFACT="$(cd "${INFERENCE_ARTIFACT_PATH}" && pwd)"

# normalised analysis time (UTC): ISO8601 for the python steps, and the
# compact form used in the DINI zarr paths
read -r ANALYSIS_ISO ANALYSIS_COMPACT < <(${PYTHON} - "${ANALYSIS_TIME}" <<'PY'
import datetime, sys
import isodate
t = isodate.parse_datetime(sys.argv[1])
if t.tzinfo is not None:
    t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
print(t.strftime("%Y-%m-%dT%H:%M"), t.strftime("%Y-%m-%dT%H%M%SZ"))
PY
)
DINI_ROOT=${DINI_ROOT:-"s3://harmonie-zarr/dini/control/${ANALYSIS_COMPACT}/"}

# number of autoregressive steps: the initial states are at the analysis time
# and +3h, the predictions at +6h .. +FORECAST_DURATION
if [[ "${FORECAST_DURATION}" =~ ^PT([0-9]+)H$ ]] && (( BASH_REMATCH[1] % 3 == 0 )) \
        && (( BASH_REMATCH[1] >= 12 )) ; then
    AR_STEPS=$(( BASH_REMATCH[1] / 3 - 1 ))
else
    echo "ERROR: FORECAST_DURATION must be PT{N}H with N a multiple of 3 and >= 12, got ${FORECAST_DURATION}"
    exit 1
fi

if [ "${BOUNDARY_SOURCE}" = "dini" ] ; then
    BOUNDARY_PATH="${WORKDIR}/inputs/boundary.zarr"
    REGRID_BOUNDARY_ARGS=()
elif [ "${BOUNDARY_SOURCE}" = "ifs" ] ; then
    : "${IFS_BOUNDARY_PATH:?IFS_BOUNDARY_PATH must be set for BOUNDARY_SOURCE=ifs}"
    BOUNDARY_PATH="${IFS_BOUNDARY_PATH}"
    REGRID_BOUNDARY_ARGS=(--no-boundary)
else
    echo "ERROR: BOUNDARY_SOURCE must be dini or ifs, got ${BOUNDARY_SOURCE}"
    exit 1
fi

echo "ANNA forecast:"
echo "  ANALYSIS_TIME=${ANALYSIS_ISO} FORECAST_DURATION=${FORECAST_DURATION} (${AR_STEPS} steps)"
echo "  DINI_ROOT=${DINI_ROOT}"
echo "  BOUNDARY_SOURCE=${BOUNDARY_SOURCE} BOUNDARY_PATH=${BOUNDARY_PATH}"
echo "  WORKDIR=${WORKDIR} ARTIFACT=${ARTIFACT}"
if grep -q "boundary_stats_placeholder: true" "${ARTIFACT}/artifact.yaml" ; then
    echo "WARNING: the inference artifact uses PLACEHOLDER boundary statistics"
fi

${PYTHON} - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# steps that completed write a marker file in their output directory, which
# SKIP_COMPLETED=true uses to skip them
completed() {
    [ "${SKIP_COMPLETED}" = true ] && [ -f "$1/.completed" ] \
        && echo "Skipping $2: already completed in $1"
}
mark_completed() {
    echo "${ANALYSIS_ISO} ${FORECAST_DURATION} ${BOUNDARY_SOURCE}" > "$1/.completed"
}

## 1. regrid DINI
if ! completed "${WORKDIR}/inputs" "regridding" ; then
    rm -f "${WORKDIR}/inputs/.completed"
    ${PYTHON} "${SCRIPT_DIR}/src/regrid_dini.py" regrid \
        --dini-root "${DINI_ROOT}" \
        --danra-grid "${ARTIFACT}/grids/danra_model1_config.grid.zarr" \
        --forecast-duration "${FORECAST_DURATION}" \
        --output "${WORKDIR}/inputs" \
        ${REGRID_BOUNDARY_ARGS[@]+"${REGRID_BOUNDARY_ARGS[@]}"}
    mark_completed "${WORKDIR}/inputs"
fi

## 2. inference datastores and neural-lam config
if ! completed "${WORKDIR}/datastores" "the inference datastores" ; then
    rm -f "${WORKDIR}/datastores/.completed"
    ${PYTHON} "${SCRIPT_DIR}/src/create_inference_dataset.py" \
        --artifact "${ARTIFACT}" \
        --interior-dir "${WORKDIR}/inputs" \
        --boundary "${BOUNDARY_PATH}" \
        --boundary-source "${BOUNDARY_SOURCE}" \
        --analysis-time "${ANALYSIS_ISO}" \
        --forecast-duration "${FORECAST_DURATION}" \
        --workdir "${WORKDIR}/datastores"
    mark_completed "${WORKDIR}/datastores"
fi
NL_CONFIG="${WORKDIR}/datastores/config.yaml"

## the model (graph recipe, train_model arguments, checkpoint) is described in
## the artifact's configs/model.yaml
MODEL_YAML="${ARTIFACT}/configs/model.yaml"
model_yaml() {
    # print a value (one list item per line) from model.yaml
    ${PYTHON} - "${MODEL_YAML}" "$1" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print("\n".join(map(str, value)) if isinstance(value, list) else value)
PY
}
GRAPH_NAME="$(model_yaml graph.name)"
GRAPH_BUILD_ARGS=()
while IFS= read -r arg ; do GRAPH_BUILD_ARGS+=("${arg}") ; done < <(model_yaml graph.build_args)
MODEL_ARGS=()
while IFS= read -r arg ; do MODEL_ARGS+=("${arg}") ; done < <(model_yaml train_model_args)
CHECKPOINT="${ARTIFACT}/$(model_yaml checkpoint.artifact_path)"
if [ ! -f "${CHECKPOINT}" ] ; then
    echo "ERROR: model checkpoint ${CHECKPOINT} not found, download it from $(model_yaml checkpoint.url)"
    exit 1
fi

## 3. graph (written to datastores/graphs/)
GRAPH_DIR="${WORKDIR}/datastores/graphs/${GRAPH_NAME}"
if ! completed "${GRAPH_DIR}" "the graph" ; then
    ${PYTHON} -m neural_lam.build_rectangular_graph \
        --config_path "${NL_CONFIG}" \
        --graph_name "${GRAPH_NAME}" \
        "${GRAPH_BUILD_ARGS[@]}"
    mark_completed "${GRAPH_DIR}"
fi

## 4. forecast
# - wandb offline (disabled crashes when saving the metric plots); neural-lam
#   writes its wandb run dir relative to the CWD, hence the `cd`
# - empty --eval_init_times: no restriction to forecasts from 00/12 UTC
# - neural-lam's main() only logs exceptions (exit code 0), so check the output
PREDICTION="${WORKDIR}/outputs/prediction.zarr"
rm -rf "${PREDICTION}"
mkdir -p "${WORKDIR}/outputs" "${WORKDIR}/wandb"
(
    cd "${WORKDIR}"
    WANDB_MODE=offline WANDB_DIR="${WORKDIR}/wandb" \
    ${PYTHON} -m neural_lam.train_model \
        --config_path "${NL_CONFIG}" \
        "${MODEL_ARGS[@]}" \
        --graph_name "${GRAPH_NAME}" \
        --eval test \
        --ar_steps_eval "${AR_STEPS}" \
        --val_steps_to_log $(seq 1 "${AR_STEPS}") \
        --eval_init_times \
        --n_example_pred 0 \
        --batch_size 1 \
        --num_workers 1 \
        --load "${CHECKPOINT}" \
        --save_eval_to_zarr_path "${PREDICTION}"
)
if [ ! -d "${PREDICTION}" ] ; then
    echo "ERROR: the forecast didn't produce ${PREDICTION}, see the neural-lam log above"
    exit 1
fi

## 5. DANRA-like output datasets
${PYTHON} "${SCRIPT_DIR}/src/convert_output.py" \
    --prediction "${PREDICTION}" \
    --danra-grid "${ARTIFACT}/grids/danra_model1_config.grid.zarr" \
    --interior-config "${ARTIFACT}/configs/danra_model1_config.yaml" \
    --analysis-time "${ANALYSIS_ISO}" \
    --output-dir "${WORKDIR}/outputs"

echo "ANNA forecast done: ${WORKDIR}/outputs/{single_levels,pressure_levels}.zarr"
