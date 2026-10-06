"""
Check that the pinned neural-lam and mllam-data-prep can run the ANNA
checkpoint.

Small synthetic datasets with the variable names ANNA expects are created, an
interior on a DANRA-like Lambert grid (~1000 km at 5 km spacing) and an ERA5
layout boundary (0.25 deg), and run through the configs in `configs/` to
create the interior and boundary datastores. A `7deg_rect_hi{levels}` graph
is then built with the recipe from neural-lam-dev
`scripts/danra_build_graphs.sh`. If the checkpoint stores its training
arguments it is loaded strictly with `HiLAM.load_from_checkpoint(...)`, the
same code path `train_model --eval` uses. With `--run-eval` (implied for
checkpoints without stored arguments, e.g. the paper's DANRA checkpoint) a
one-step `train_model --eval test` is run, which also loads the checkpoint
strictly, writing predictions to zarr.

Defaults are for the paper's DANRA model (Zenodo 15131838: graph
`7deg_rect_hi4`, no `--dynamic_time_deltas`). For gefion-1 use
`--graph-levels 3 --dynamic-time-deltas`.

The data is random noise, so this only checks that the software stack and
checkpoint fit together, not forecast skill.

The (superseded) gefion-1 checkpoint should first be cleaned with
`sanitize_checkpoint.py`.

Usage (from the repository root), e.g. for an assembled package:
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/check_checkpoint_compat.py \\
        <package>/danra_model.ckpt <package>/configs <workdir> --run-eval
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The (trusted) checkpoint contains argparse.Namespace and NeuralLAMConfig
# objects, which torch>=2.6 refuses to unpickle by default (Lightning's
# `load_from_checkpoint` uses torch's default `weights_only`)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cartopy.crs as ccrs  # noqa: E402
import mllam_data_prep as mdp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]
INTERIOR_SL_VARS = [
    "pres_seasurface",
    "t2m",
    "u10m",
    "v10m",
    "pres0m",
    "lwavr0m",
    "swavr0m",
]
INTERIOR_PL_VARS = ["z", "t", "r", "u", "v", "tw"]
INTERIOR_STATIC_VARS = ["lsm", "orography"]
BOUNDARY_SL_VARS = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
BOUNDARY_PL_VARS = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
BOUNDARY_STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]
MODEL_ARGS = [
    "--model", "hi_lam",
    "--hidden_dim", "300",
    "--hidden_dim_grid", "150",
    "--time_delta_enc_dim", "32",
    "--processor_layers", "2",
    "--num_past_forcing_steps", "1",
    "--num_future_forcing_steps", "1",
    "--num_past_boundary_steps", "1",
    "--num_future_boundary_steps", "1",
]  # fmt: skip


def _set_time_range(config, dim, times):
    config.output.coord_ranges[dim].start = str(times[0])
    config.output.coord_ranges[dim].end = str(times[-1])
    for split in config.output.splitting.splits.values():
        split.start, split.end = str(times[0]), str(times[-1])


def _create_interior(configs_dir, workdir, times, rng):
    proj = ccrs.LambertConformal(
        central_longitude=25.0,
        central_latitude=56.7,
        standard_parallels=(56.7, 56.7),
        globe=ccrs.Globe(semimajor_axis=6367470.0, semiminor_axis=6367470.0),
    )
    x = np.arange(-1_500_000, -500_000, 5_000.0)
    y = np.arange(-500_000, 500_000, 5_000.0)
    xx, yy = np.meshgrid(x, y)
    lonlat = ccrs.PlateCarree().transform_points(proj, xx, yy)
    latlon_coords = dict(
        lat=(("y", "x"), lonlat[..., 1]), lon=(("y", "x"), lonlat[..., 0])
    )

    shape = (len(times), len(y), len(x))
    ds_sl = xr.Dataset(
        {
            v: (("time", "y", "x"), rng.normal(size=shape))
            for v in INTERIOR_SL_VARS
        },
        coords=dict(time=times, x=x, y=y, **latlon_coords),
    )
    for v in INTERIOR_STATIC_VARS:
        ds_sl[v] = (("y", "x"), rng.normal(size=shape[1:]))
    shape = (len(times), len(LEVELS), len(y), len(x))
    ds_pl = xr.Dataset(
        {
            v: (("time", "pressure", "y", "x"), rng.normal(size=shape))
            for v in INTERIOR_PL_VARS
        },
        coords=dict(
            time=times,
            pressure=("pressure", LEVELS, {"units": "hPa"}),
            x=x,
            y=y,
            **latlon_coords,
        ),
    )
    ds_sl.to_zarr(workdir / "interior_sl.zarr")
    ds_pl.to_zarr(workdir / "interior_pl.zarr")

    config = mdp.Config.from_yaml_file(
        configs_dir / "danra_model1_config.yaml"
    )
    for name, input_config in config.inputs.items():
        fn = (
            "interior_pl.zarr"
            if name == "danra_pl_state"
            else "interior_sl.zarr"
        )
        input_config.path = str(workdir / fn)
    _set_time_range(config, "time", times)
    # NB: sort_keys=False, sorted keys would change the feature order
    config.to_yaml_file(workdir / "danra_model1_config.yaml", sort_keys=False)
    return lonlat[..., 1], lonlat[..., 0]


def _create_boundary(configs_dir, workdir, times, lat2d, lon2d, rng):
    lats = np.arange(lat2d.max() + 8, lat2d.min() - 8, -0.25)
    lons = np.arange(lon2d.min() - 12, lon2d.max() + 12, 0.25)
    data_vars = {}
    for v in BOUNDARY_SL_VARS:
        data_vars[v] = (
            ("time", "latitude", "longitude"),
            rng.normal(size=(len(times), len(lats), len(lons))),
        )
    for v in BOUNDARY_PL_VARS:
        data_vars[v] = (
            ("time", "level", "latitude", "longitude"),
            rng.normal(size=(len(times), len(LEVELS), len(lats), len(lons))),
        )
    for v in BOUNDARY_STATIC_VARS:
        data_vars[v] = (
            ("latitude", "longitude"),
            rng.normal(size=(len(lats), len(lons))),
        )
    coords = dict(time=times, level=LEVELS, latitude=lats, longitude=lons)
    xr.Dataset(data_vars, coords=coords).to_zarr(workdir / "era5.zarr")

    config = mdp.Config.from_yaml_file(
        configs_dir / "era_7deg_model1_config.yaml"
    )
    for input_config in config.inputs.values():
        input_config.path = str(workdir / "era5.zarr")
    config.output.domain_cropping.interior_dataset_config_path = str(
        workdir / "danra_model1_config.yaml"
    )
    _set_time_range(config, "time", times)
    config.to_yaml_file(
        workdir / "era_7deg_model1_config.yaml", sort_keys=False
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("configs_dir", type=Path)
    parser.add_argument(
        "workdir", type=Path, help="must not exist or be empty"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="also run a one-step `train_model --eval test`",
    )
    parser.add_argument(
        "--graph-levels",
        type=int,
        default=4,
        help="hierarchical mesh levels (paper model: 4, gefion-1: 3)",
    )
    parser.add_argument(
        "--dynamic-time-deltas",
        action="store_true",
        help="model trained with --dynamic_time_deltas (gefion-1, not the "
        "paper model)",
    )
    args = parser.parse_args()
    graph_name = f"7deg_rect_hi{args.graph_levels}"
    model_args = MODEL_ARGS + ["--graph_name", graph_name]
    if args.dynamic_time_deltas:
        model_args.append("--dynamic_time_deltas")

    workdir = args.workdir.resolve()
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"workdir {workdir} is not empty")
    workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # three days: enough for complete 00/12 UTC samples after the interior
    # and (6-hourly) boundary data have been aligned
    times = pd.date_range("2000-01-01T00:00", "2000-01-04T00:00", freq="3h")
    lat2d, lon2d = _create_interior(args.configs_dir, workdir, times, rng)
    btimes = pd.date_range(times[0], times[-1], freq="6h")
    _create_boundary(args.configs_dir, workdir, btimes, lat2d, lon2d, rng)

    nl_config = workdir / "7deg_config_era5.yaml"
    shutil.copy(args.configs_dir / "7deg_config_era5.yaml", nl_config)
    subprocess.run(
        [
            sys.executable, "-m", "neural_lam.build_rectangular_graph",
            "--config_path", str(nl_config),
            "--mesh_node_distance", "12500",
            "--archetype", "hierarchical",
            "--max_num_levels", str(args.graph_levels),
            "--graph_name", graph_name,
        ],
        check=True,
    )  # fmt: skip

    from neural_lam.config import load_config_and_datastores
    from neural_lam.models import HiLAM

    config, datastore, datastore_boundary = load_config_and_datastores(
        config_path=str(nl_config)
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    n_tensors = len(ckpt["state_dict"])
    if "hyper_parameters" in ckpt:
        ckpt_args = ckpt["hyper_parameters"]["args"]
        ckpt_args.config_path = str(nl_config)
        ckpt_args.load = str(args.checkpoint)
        ckpt_args.eval = "test"
        model = HiLAM.load_from_checkpoint(
            str(args.checkpoint),
            map_location="cpu",
            args=ckpt_args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(
            "OK: checkpoint loaded strictly into HiLAM "
            f"({n_params:,} parameters)"
        )
    else:
        print(
            f"checkpoint ({n_tensors} tensors) has no stored training "
            "arguments, checking the strict load through train_model --eval"
        )
        args.run_eval = True

    if args.run_eval:
        fp_output = workdir / "eval_output.zarr"
        env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(workdir))
        subprocess.run(
            [
                sys.executable, "-m", "neural_lam.train_model",
                "--config_path", str(nl_config),
                *model_args,
                "--eval", "test",
                "--ar_steps_eval", "1",
                "--val_steps_to_log", "1",
                "--n_example_pred", "0",
                "--batch_size", "1",
                "--num_workers", "1",
                "--load", str(args.checkpoint),
                "--save_eval_to_zarr_path", str(fp_output),
            ],
            check=True,
            env=env,
            # neural-lam writes the wandb run dir relative to the CWD
            cwd=workdir,
        )  # fmt: skip
        # neural-lam-dev's main() catches (and only logs) exceptions, so check
        # for the output rather than relying on the exit code
        if not fp_output.exists():
            raise SystemExit(f"eval did not produce {fp_output}")
        ds = xr.open_zarr(fp_output)
        if not bool(np.isfinite(ds.state).all()):
            raise SystemExit("eval output contains non-finite values")
        print(f"OK: eval wrote {fp_output} with dims {dict(ds.state.sizes)}")


if __name__ == "__main__":
    main()
