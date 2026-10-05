"""
Create the inference datastores and neural-lam config for running ANNA
(gefion-1) from regridded DINI (interior) and DINI or IFS (boundary) data.

Inputs:
- the inference artifact (`dev-utils/assemble_artifact.py`), providing the
  datastore configs, the neural-lam configs and the training statistics
- the regridded interior (`interior_{single,pressure}_levels.zarr` from
  `regrid_dini.py`)
- the boundary forecast in the IFS contract layout: `boundary.zarr` from
  `regrid_dini.py` (DINI boundary) or a converted IFS forecast

Written to the inference workdir (neural-lam opens `{name}.zarr` next to each
`{name}.yaml`):
- `danra_model1_config.{yaml,zarr}`: interior datastore covering
  analysis_time .. analysis_time + forecast_duration (+ one step for
  `num_future_forcing_steps=1`), with the DANRA *training* statistics merged
  in (statistics are not computed from the inference data)
- `{dini,ifs}_7deg_model1_config.{yaml,zarr}`: boundary datastore, cropped to
  the boundary points around the interior; normalised with the ERA5 training
  statistics in the artifact (`overload_stats_path`)
- `config.yaml`: neural-lam config referencing the above

NB: mllam-data-prep configs must never be written with sorted keys, that
would reorder inputs/variables and so the features the checkpoint expects.

Usage:
    python create_inference_dataset.py --artifact inference_artifact \\
        --interior-dir inference_workdir/inputs \\
        --boundary inference_workdir/inputs/boundary.zarr --boundary-source dini \\
        --analysis-time 2026-09-26T18:00 --forecast-duration PT18H \\
        --workdir inference_workdir
"""
import argparse
import datetime
import os
from pathlib import Path

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import numpy as np
import xarray as xr
import yaml
from loguru import logger

INTERIOR_NAME = "danra_model1_config"
BOUNDARY_NAMES = dict(
    dini="dini_7deg_model1_config", ifs="ifs_7deg_model1_config"
)
NL_CONFIG_NAMES = dict(
    dini="7deg_config_dini.yaml", ifs="7deg_config_ifs.yaml"
)
# interior time step and the boundary time step ANNA was trained with
INTERIOR_STEP = datetime.timedelta(hours=3)
BOUNDARY_STEP = datetime.timedelta(hours=6)
# interior time steps needed beyond analysis_time + forecast_duration: one for
# `num_future_forcing_steps=1`, and one more because neural-lam's
# WeatherDataset counts samples as
# `n_times - (2 + ar_steps) - num_future_forcing_steps`, one step more
# conservative than the data a sample actually uses (same as in
# regrid_dini.py)
INTERIOR_EXTRA_STEPS = 2


def ar_steps_for(forecast_duration):
    """
    Autoregressive steps for a forecast: the initial states are at the
    analysis time and +3h, the predictions at +6h .. +forecast_duration.
    """
    return int(forecast_duration / INTERIOR_STEP) - 1


SPLIT_NAMES = ["train", "val", "test"]
INTERIOR_INPUT_FILES = dict(
    danra_sl_state="interior_single_levels.zarr",
    danra_pl_state="interior_pressure_levels.zarr",
    danra_static="interior_single_levels.zarr",
    danra_forcing="interior_single_levels.zarr",
)


def _write_config(config, fp):
    # NB: sort_keys=False, sorting would change the feature order
    config.to_yaml_file(fp, sort_keys=False)


def _create_interior_datastore(
    artifact, interior_dir, analysis_time, interior_end, workdir
):
    config = mdp.Config.from_yaml_file(
        artifact / "configs" / f"{INTERIOR_NAME}.yaml"
    )
    for input_name, input_config in config.inputs.items():
        input_config.path = str(
            (interior_dir / INTERIOR_INPUT_FILES[input_name]).resolve()
        )

    start, end = analysis_time.isoformat(), interior_end.isoformat()
    config.output.coord_ranges["time"].start = start
    config.output.coord_ranges["time"].end = end
    # neural-lam requires train/val/test splits, the forecast is run on "test".
    # Statistics are not computed from the inference data, the training
    # statistics are merged in below
    config.output.splitting.splits = {
        name: mdp_config.Split(start=start, end=end) for name in SPLIT_NAMES
    }

    fp_config = workdir / f"{INTERIOR_NAME}.yaml"
    _write_config(config, fp_config)

    ds = mdp.create_dataset(config=config)
    ds_stats = xr.open_zarr(
        artifact / "stats" / f"{INTERIOR_NAME}.stats.zarr"
    ).load()
    for category in ["state", "forcing", "static"]:
        dim = f"{category}_feature"
        if dim in ds_stats.dims and list(ds_stats[dim].values) != list(
            ds[dim].values
        ):
            raise ValueError(
                f"{category} features of the inference dataset don't match "
                "those of the training statistics"
            )
    # only merge the statistics themselves, the feature metadata coordinates
    # (units, long names) describe the inference data
    stats_vars = [v for v in ds_stats.data_vars if "__train__" in v]
    ds_stats = ds_stats[stats_vars].reset_coords(drop=True)
    ds = xr.merge([ds, ds_stats], join="exact", combine_attrs="override")

    n_missing = {
        v: int(ds[v].isnull().sum()) for v in ["state", "forcing", "static"]
    }
    if any(n_missing.values()):
        raise ValueError(
            f"missing values in the interior datastore: {n_missing}"
        )
    ds.to_zarr(workdir / f"{INTERIOR_NAME}.zarr", mode="w", consolidated=True)
    logger.info(
        f"interior datastore: {ds.time.size} time steps {start} .. {end}, "
        f"{ds.grid_index.size} grid points"
    )
    return fp_config


def _create_boundary_datastore(
    artifact,
    boundary_source,
    fp_boundary,
    fp_interior_config,
    boundary_end,
    workdir,
):
    name = BOUNDARY_NAMES[boundary_source]
    config = mdp.Config.from_yaml_file(artifact / "configs" / f"{name}.yaml")
    for input_config in config.inputs.values():
        input_config.path = str(Path(fp_boundary).resolve())
    # crop around the (small) inference interior dataset, rather than the
    # training interior dataset. The path is resolved relative to the CWD by
    # mllam-data-prep, so make it absolute
    config.output.domain_cropping.interior_dataset_config_path = str(
        fp_interior_config.resolve()
    )

    fp_config = workdir / f"{name}.yaml"
    _write_config(config, fp_config)
    ds = mdp.create_dataset(config=config)

    valid_times = (
        ds.analysis_time.values[:, None] + ds.elapsed_forecast_duration.values
    )
    needed = np.datetime64(boundary_end, "ns")
    if valid_times.max() < needed:
        raise ValueError(
            f"boundary forecast ends at {valid_times.max()}, but must reach "
            f"{needed}"
        )
    n_missing = {v: int(ds[v].isnull().sum()) for v in ["forcing", "static"]}
    if any(n_missing.values()):
        raise ValueError(
            f"missing values in the boundary datastore: {n_missing}"
        )
    ds.to_zarr(workdir / f"{name}.zarr", mode="w", consolidated=True)
    logger.info(
        f"boundary datastore ({boundary_source}): analysis time(s) "
        f"{ds.analysis_time.values}, {ds.elapsed_forecast_duration.size} lead "
        f"times, {ds.grid_index.size} grid points"
    )
    return fp_config


def _create_neural_lam_config(
    artifact, boundary_source, fp_interior_config, fp_boundary_config, workdir
):
    nl_config = yaml.safe_load(
        (artifact / "configs" / NL_CONFIG_NAMES[boundary_source]).read_text()
    )
    nl_config["datastore"]["config_path"] = fp_interior_config.name
    boundary = nl_config["datastore_boundary"]
    boundary["config_path"] = fp_boundary_config.name
    # the ERA5 training statistics (stats datastore) in the artifact. neural-lam
    # joins this onto the config directory, so use an absolute path
    boundary["overload_stats_path"] = str(
        (artifact / "configs" / boundary["overload_stats_path"]).resolve()
    )
    fp_config = workdir / "config.yaml"
    fp_config.write_text(yaml.safe_dump(nl_config, sort_keys=False))
    logger.info(f"neural-lam config written to {fp_config}")
    return fp_config


def create_inference_datasets(
    artifact,
    interior_dir,
    boundary,
    boundary_source,
    analysis_time,
    forecast_duration,
    workdir,
):
    artifact, interior_dir, workdir = (
        Path(artifact),
        Path(interior_dir),
        Path(workdir),
    )
    workdir.mkdir(parents=True, exist_ok=True)
    # the initial states are at analysis_time and +3h, so the first prediction
    # is at +6h
    if (
        forecast_duration < 2 * INTERIOR_STEP
        or forecast_duration % INTERIOR_STEP
    ):
        raise ValueError(
            f"forecast duration must be a multiple of {INTERIOR_STEP} and at "
            f"least {2 * INTERIOR_STEP}"
        )
    interior_end = (
        analysis_time
        + forecast_duration
        + INTERIOR_EXTRA_STEPS * INTERIOR_STEP
    )
    # one boundary step after the forecast for `num_future_boundary_steps=1`
    boundary_end = analysis_time + forecast_duration + BOUNDARY_STEP

    fp_interior_config = _create_interior_datastore(
        artifact, interior_dir, analysis_time, interior_end, workdir
    )
    fp_boundary_config = _create_boundary_datastore(
        artifact,
        boundary_source,
        boundary,
        fp_interior_config,
        boundary_end,
        workdir,
    )
    logger.info(
        f"forecast {analysis_time} + {forecast_duration}: run neural-lam with "
        f"--ar_steps_eval {ar_steps_for(forecast_duration)} --eval_init_times "
        "(empty: no init-time filtering)"
    )
    return _create_neural_lam_config(
        artifact,
        boundary_source,
        fp_interior_config,
        fp_boundary_config,
        workdir,
    )


def _parse_analysis_time(s):
    t = isodate.parse_datetime(s)
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("inference_artifact")
    )
    parser.add_argument(
        "--interior-dir",
        required=True,
        type=Path,
        help="regrid_dini.py output",
    )
    parser.add_argument(
        "--boundary",
        required=True,
        help="boundary forecast zarr (IFS contract)",
    )
    parser.add_argument(
        "--boundary-source", choices=sorted(BOUNDARY_NAMES), default="dini"
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=_parse_analysis_time,
        help="ISO8601, UTC if no timezone",
    )
    parser.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()

    create_inference_datasets(
        artifact=args.artifact,
        interior_dir=args.interior_dir,
        boundary=args.boundary,
        boundary_source=args.boundary_source,
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    if os.getenv("MLWM_DEBUGGER", "") == "ipdb":
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
