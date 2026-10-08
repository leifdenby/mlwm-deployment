# ANNA

ANNA runs the **DANRA machine-learning limited area model** of Adamov et al.
(2025), ["Building Machine Learning Limited Area Models: Kilometer-Scale
Weather Forecasting in Realistic Settings"](https://arxiv.org/abs/2504.09340)
operationally from **DINI** (interior initial states), with **DINI or IFS** on
the boundary. The model is a hierarchical graph-based model (neural-lam
`hi_lam`, graph `7deg_rect_hi4`) trained on DANRA (2.5 km, 3-hourly) with ERA5
boundary forcing on a 7.19° ring around the DANRA domain.

The full history, decisions and the reasons behind them are in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md).

## Model

Described in [configs/model.yaml](configs/model.yaml) (checkpoint, graph
recipe, neural-lam arguments), which `entry.sh` and the container build read.

- Checkpoint: the paper's DANRA model, Zenodo
  [10.5281/zenodo.15131838](https://doi.org/10.5281/zenodo.15131838)
  (`danra_model.ckpt`, CC-BY-4.0, by S. Adamov, J. Oskarsson and K. S. Hintz)
- Training: wandb [`jo-research-team/neural_lam/hfzfhiha`](https://wandb.ai/jo-research-team/neural_lam/runs/hfzfhiha)
  (`train-hi_lam-2x300-02_17_19-2867`), a 3-epoch fine-tune with 4-step
  rollouts of `train-hi_lam-2x300-02_13_14-1703`, with
  [`joeloskarsson/neural-lam-dev`](https://github.com/joeloskarsson/neural-lam-dev)
  commit `e7d11c9`
- Run without `--dynamic_time_deltas`: the model was trained before that flag
  existed, and leaving it off reproduces its behaviour

An earlier artifact, `gefion-1`, held an unpublished ablation run (graph
`7deg_rect_hi3`, trained with `--dynamic_time_deltas`), not the paper's model.
It was trained on the same datastores, so it still supplies the training
statistics and grids. See
[inference-artifact/README.md](inference-artifact/README.md#gefion-1-a-prior-incorrect-inference-artifact).

## Software

Pinned in [pyproject.toml](pyproject.toml):

- **neural-lam**: `joeloskarsson/neural-lam-dev@a44d432e` (`research` branch,
  contains the training code and forecast-format boundary forcing, needed for
  IFS; later commits break installing from git)
- **mllam-data-prep**: `sadamov/mllam-data-prep@dd9af481`, the version used
  for training. mllam-data-prep ≥ 0.7 doesn't keep per-point lat/lon for
  regular lat/lon grids, which neural-lam-dev needs for the boundary
- **zarr 2**, as in training; neural-lam-dev writes its output with a zarr v2
  compressor

## Configs

All in [configs/](configs/):

| File | What |
|---|---|
| `model.yaml` | the model to run: checkpoint (Zenodo URL, md5), graph recipe, neural-lam arguments, and which neural-lam config to use per boundary source (`neural_lam_configs`: `era5` as in training, `dini`, `ifs`) |
| `danra_model1_config.yaml` | interior (DANRA) datastore, as in training |
| `era_7deg_model1_config.yaml` | ERA5 boundary datastore the model was trained with; reference for the boundary features and their (ERA5) normalisation statistics |
| `ifs_7deg_model1_config.yaml` | operational IFS boundary. **Its header is the contract for the IFS GRIB → zarr conversion** (variables, units, dims, lead times, and the 0.25° box lat 40.0–72.0, lon −27.0–39.5; the east edge must be exactly 39.5°E) |
| `dini_7deg_model1_config.yaml` | DINI boundary, in the same layout as IFS |
| `7deg_config_{era5,ifs,dini}.yaml` | neural-lam configs; IFS and DINI are normalised with the ERA5 training statistics (`overload_stats_path`) |

The configs are linked in a chain, and each link is stated in one place only:
1. `model.yaml` names the neural-lam config for each boundary source;
2. that neural-lam config names its interior and boundary datastore configs
   (`datastore.config_path`, `datastore_boundary.config_path`);
3. for DINI/IFS, it also names the ERA5 training boundary datastore whose
   statistics normalise the boundary (`overload_stats_path`).

`src/model_configs.py` follows this chain for the scripts.

## Inference artifact

In mlwm-deployment, an **inference artifact** is a directory (shipped as a
zip) that holds everything needed to run a trained model's forecasts, apart
from the input data and the software. See the
[repository README](../../README.md). It normally has:
- the model checkpoint (weights);
- the configs the model was trained with (neural-lam and mllam-data-prep
  datastore configs) and its training arguments;
- the training dataset statistics, which inputs are normalised with and
  outputs de-normalised with.

Without the artifact, the checkpoint alone can't be run. The model expects
its inputs in a particular feature order and normalisation, and on a
particular grid and graph, and these only follow from the configs and
statistics it was trained with.

Usually the artifact is built with `mlwm.build_inference_artifact` on the
machine the model was trained on. ANNA's had to be reconstructed after the
fact: the boundary configs, statistics and grid were missing (see
[inference-artifact/README.md](inference-artifact/README.md)). ANNA's
artifact has:

| Path | What |
|---|---|
| `configs/` | the configs above, including `model.yaml` |
| `configs/era_7deg_model1_config.zarr` | the ERA5 boundary statistics in the form neural-lam reads them (`overload_stats_path`) |
| `stats/` | training statistics of the DANRA interior and the ERA5 boundary datastores (the ERA5 ones recomputed exactly) |
| `grids/` | the DANRA grid and statics (target grid for regridding DINI) and the 18,014 ERA5 boundary points the model was trained with |
| `artifact.yaml`, `README.md` | provenance and description |
| `danra_model.ckpt` | the checkpoint |

The checkpoint is already public (Zenodo 15131838), so it isn't re-published.
The artifact is therefore published **without the checkpoint**, as the
*inference package* zip on Zenodo (record to be added once published).

The container build puts the two back together:
- it fetches the package, or uses a locally built one;
- it downloads the checkpoint into it (md5-checked);
- the result is the complete artifact at `/workspace/inference_artifact`,
  which `entry.sh` reads (`INFERENCE_ARTIFACT_PATH`).

How the artifact is built, checked and published, and its provenance, are in
[inference-artifact/README.md](inference-artifact/README.md). That includes
the superseded `gefion-1` artifact, which held a different model. The README
shipped inside the package is
[inference-artifact/package-README.md](inference-artifact/package-README.md).

## Building the image

```bash
CONTAINER_APP=podman ./build_image.sh
```

- `ARTIFACT_SOURCE=local` (default) uses a locally built artifact directory,
  `LOCAL_ARTIFACT_DIR` (default `inference-artifact/build/anna-local`).
- `ARTIFACT_SOURCE=zenodo` downloads the published package (`PACKAGE_URL`,
  `PACKAGE_MD5`).

In both cases the model checkpoint is downloaded from Zenodo during the build
(md5-checked) if it isn't already there. No AWS credentials are needed. Set
`MLWM_PULL_PROXY` to pull the base image through DMI's proxy.

## Running a forecast

[entry.sh](entry.sh) runs the whole pipeline:

1. `src/regrid_dini.py`: DINI → DANRA grid (interior) and, for the DINI
   boundary, → the 0.25° ERA5 boundary box
2. `src/create_inference_dataset.py`: inference datastores and neural-lam
   config
3. `neural_lam.build_rectangular_graph`: the model's graph (`model.yaml`)
4. `neural_lam.train_model --eval test`: the forecast
5. `src/convert_output.py`: DANRA-like output datasets

It's configured through environment variables:

| Variable | Default | |
|---|---|---|
| `ANALYSIS_TIME` | (required) | DINI analysis time, e.g. `2026-10-01T00:00Z`; 00/06/12/18 UTC with the DINI boundary |
| `FORECAST_DURATION` | `PT18H` | multiple of 3 h, between 6 h and DINI forecast length − 6 h (30 h for a 36 h DINI run) |
| `BOUNDARY_SOURCE` | `dini` | `dini` or `ifs` |
| `DINI_ROOT` | `s3://harmonie-zarr/dini/control/{analysis}/` | DINI forecast zarrs |
| `IFS_BOUNDARY_PATH` | | IFS forecast zarr, required for `BOUNDARY_SOURCE=ifs` |
| `INFERENCE_WORKDIR` | `./inference_workdir` | |
| `INFERENCE_ARTIFACT_PATH` | `./inference_artifact` | the inference artifact (with checkpoint); outside the container e.g. `inference-artifact/build/<name>` |
| `MLWM_DEBUGGER` | | `ipdb` to debug the python steps on exceptions |

Outside the container, run it from `configurations/ANNA` in the uv
environment, with AWS credentials for reading DINI (e.g. `AWS_PROFILE`):

```bash
INFERENCE_ARTIFACT_PATH=inference-artifact/build/anna-local \
    ANALYSIS_TIME=2026-10-01T00:00Z FORECAST_DURATION=PT18H ./entry.sh
```

In the container: `run_inference_container.sh` still has the interface from
before these changes, and is to be updated (step 7 in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md)).

### Outputs and conventions

`${INFERENCE_WORKDIR}/outputs/single_levels.zarr` (pres_seasurface, t2m, u10m,
v10m, pres0m, lwavr0m, swavr0m) and `pressure_levels.zarr` (z, t, r, u, v, tw
at 100–1000 hPa) are on the DANRA grid at valid times T+6 h …
T+`FORECAST_DURATION` in 3 h steps. The initial states are DINI at T+0 and
T+3 h. The conventions are DANRA's:

- **winds are relative to the DANRA (Lambert) grid**, not east/north
- relative humidity `r` is a fraction (0–1)
- `lwavr0m`/`swavr0m` are net surface radiation fluxes
- `tw` is geometric vertical velocity (m/s)

## Known limitations

- **DINI boundary.** 25% of the model's boundary points (4,573 of 18,014) lie
  outside the DINI domain, mostly to the east and north. They are filled with
  the value of the nearest DINI grid point. IFS covers the whole ring and is
  the preferred boundary source once the IFS zarr is available.

  ![Boundary points vs the DINI domain](docs/anna_boundary_vs_dini.png)

- **DINI boundary: 00/06/12/18 UTC only.** The model was trained with a
  6-hourly ERA5 boundary at 00/06/12/18 UTC. The boundary is therefore given
  to neural-lam as valid times on that grid, which for each prediction uses
  the last boundary time at or before it ± 6 h. A forecast from, say, 03Z
  needs the boundary at 00Z, which its own DINI run doesn't have. IFS
  boundaries can come from an older cycle (the latest at or before the
  analysis time).

- **DINI zarr retention** is two weeks. Keep regridded development cases
  locally in `dev-data/` (gitignored).

## Development

- [inference-artifact/](inference-artifact/): building, checking and
  publishing the inference artifact
- [dev-utils/](dev-utils/)`plot_boundary_coverage.py`: the boundary coverage
  map above
- Tests:
  ```bash
  uv run --project configurations/ANNA --with pytest \
      python -m pytest configurations/ANNA/tests
  ```
- The `src/` scripts can also be run on their own; see their `--help` and
  docstrings.
