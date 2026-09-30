"""
Assemble a complete local ANNA inference artifact directory from the
`gefion-1` inference artifact and the files it is missing.

`gefion-1.zip` (s3://mlwm-artifacts/inference-artifacts/gefion-1.zip) lacks
everything about the boundary datastore, and its checkpoint can't be loaded
at inference time as is (see INFERENCE_PLAN.md). The assembled directory
contains:

    checkpoint.pkl          gefion-1 checkpoint, sanitised (training boundary
                            datastore dropped from its hyper-parameters)
    configs/                interior and boundary (ERA5/IFS/DINI) datastore
                            configs and neural-lam configs from
                            configurations/ANNA/configs/
    configs/era_7deg_model1_config.zarr
                            ERA5 "stats datastore" that neural-lam opens via
                            `overload_stats_path` (statistics and splits only)
    configs/gefion-1/       the configs as shipped in gefion-1 (provenance)
    stats/danra_model1_config.stats.zarr
                            interior training statistics (from gefion-1)
    stats/era_7deg_model1_config.stats.zarr
                            ERA5 boundary training statistics (from
                            compute_era5_boundary_stats.py, or a placeholder)
    grids/era_7deg_model1_config.grid.zarr
                            lat/lon of the 18014 boundary points ANNA was
                            trained with, recovered from the checkpoint
    training_cli_args.yaml  from gefion-1
    gefion-1.artifact.yaml  artifact.yaml of gefion-1
    artifact.yaml           provenance of this assembled artifact

Usage (in the ANNA environment, from the repository root):

    aws s3 cp s3://mlwm-artifacts/inference-artifacts/gefion-1.zip .
    uv run --project configurations/ANNA \\
        python configurations/ANNA/dev-utils/assemble_artifact.py \\
        --gefion-1-zip gefion-1.zip \\
        --boundary-stats era_7deg_model1_config.stats.zarr
"""
import argparse
import datetime
import hashlib
import importlib.util
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from loguru import logger

ANNA_DIR = Path(__file__).parent.parent
DEV_UTILS_DIR = Path(__file__).parent

BOUNDARY_DATASTORE_NAME = "era_7deg_model1_config"
INTERIOR_DATASTORE_NAME = "danra_model1_config"
N_BOUNDARY_POINTS = 18014
CHECKPOINT_FILENAME = "checkpoint.pkl"


def _import_dev_util(name):
    spec = importlib.util.spec_from_file_location(
        name, DEV_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_describe():
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=ANNA_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _boundary_grid_from_checkpoint(ckpt, subset_lats, subset_lons):
    """
    Recover the lat/lon of the boundary points ANNA was trained with.

    The checkpoint's pickled training boundary datastore keeps its
    `grid_index` index in memory. mllam-data-prep stacked `[longitude,
    latitude]` of the ERA5 subset (latitude descending, longitude sorted
    ascending in 0..360) into `grid_index`, so each index is
    `lon_idx * n_lat + lat_idx`.
    """
    datastore = ckpt["hyper_parameters"]["datastore_boundary"]
    state = vars(datastore).get("_state", vars(datastore))
    grid_index = np.asarray(state["_ds"].grid_index.values)

    lats = np.asarray(subset_lats)
    lons = np.sort(np.asarray(subset_lons))  # as stored in the ERA5 subset
    lon_idx, lat_idx = np.divmod(grid_index, lats.size)
    if lon_idx.max() >= lons.size:
        raise ValueError(
            "boundary grid_index doesn't fit the ERA5 subset grid"
        )
    if grid_index.size != N_BOUNDARY_POINTS:
        raise ValueError(
            f"expected {N_BOUNDARY_POINTS} boundary points, got {grid_index.size}"
        )
    return xr.Dataset(
        coords=dict(
            grid_index=("grid_index", grid_index),
            latitude=("grid_index", lats[lat_idx]),
            longitude=("grid_index", lons[lon_idx]),
        ),
        attrs=dict(
            description=(
                "lat/lon of the ERA5 (0.25 deg) boundary points ANNA (gefion-1) "
                "was trained with, recovered from the checkpoint's pickled "
                "boundary datastore grid_index and the era_danra_model1_subset "
                "grid (grid_index stacked [longitude, latitude])"
            )
        ),
    )


def _create_stats_datastore(ds_stats, fp_boundary_config, fp_out):
    """
    Create the zarr neural-lam opens as boundary "stats datastore" (via
    `overload_stats_path`): the statistics, their feature coordinates and the
    `splits` of the boundary datastore config. neural-lam only uses the
    `{forcing,static}__train__{mean,std}` of it.
    """
    import mllam_data_prep as mdp

    config = mdp.Config.from_yaml_file(fp_boundary_config)
    splits = config.output.splitting.splits
    ds = ds_stats.copy()
    ds["splits"] = xr.DataArray(
        np.array([[str(s.start), str(s.end)] for s in splits.values()]),
        dims=["split_name", "split_part"],
        coords=dict(split_name=list(splits), split_part=["start", "end"]),
    )
    ds.to_zarr(fp_out, mode="w", consolidated=True)


def _check_stats_datastore(fp_boundary_config):
    """Open the stats datastore as neural-lam will and check the statistics."""
    from neural_lam.datastore.mdp import MDPDatastore

    fp_zarr = Path(fp_boundary_config).with_suffix(".zarr")
    if not fp_zarr.exists():
        # MDPDatastore would otherwise try to build the full training dataset
        raise FileNotFoundError(fp_zarr)
    datastore = MDPDatastore(config_path=fp_boundary_config)
    for category, n_features in [("forcing", 58), ("static", 2)]:
        ds = datastore.get_standardization_dataarray(category=category)
        for op in ["mean", "std"]:
            values = ds[f"{category}_{op}"].values
            if values.size != n_features or not np.isfinite(values).all():
                raise ValueError(f"bad {category}_{op} in stats datastore")
        if not (ds[f"{category}_std"].values > 0).all():
            raise ValueError(f"non-positive {category}_std in stats datastore")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gefion-1-zip", required=True, type=Path)
    parser.add_argument(
        "--boundary-stats",
        required=True,
        type=Path,
        help=f"{BOUNDARY_DATASTORE_NAME}.stats.zarr with the ERA5 boundary stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANNA_DIR / "inference_artifact",
        help="artifact directory to create (must not exist)",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=ANNA_DIR / "configs"
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists, remove it first")

    sanitize = _import_dev_util("sanitize_checkpoint")
    stats_script = _import_dev_util("compute_era5_boundary_stats")

    ds_boundary_stats = xr.open_zarr(args.boundary_stats).load()
    is_placeholder = ds_boundary_stats.attrs.get("PLACEHOLDER") == "true"
    if is_placeholder:
        logger.warning(
            "the boundary statistics are a PLACEHOLDER, replace them with the "
            "output of compute_era5_boundary_stats.py for real forecasts"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"unpacking {args.gefion_1_zip}")
        with zipfile.ZipFile(args.gefion_1_zip) as zf:
            zf.extractall(tmpdir)

        # build in a temporary directory and move into place at the end, so
        # that a failed run doesn't leave a half-assembled artifact behind
        out = tmpdir / "artifact"
        (out / "stats").mkdir(parents=True)
        (out / "grids").mkdir()

        logger.info("recovering boundary grid and sanitising checkpoint")
        ckpt = sanitize.load_checkpoint(tmpdir / "checkpoint.pkl")
        ds_grid = _boundary_grid_from_checkpoint(
            ckpt, stats_script.SUBSET_LATS, stats_script.SUBSET_LONS
        )
        ds_grid.to_zarr(out / "grids" / f"{BOUNDARY_DATASTORE_NAME}.grid.zarr")
        torch.save(
            sanitize.sanitize_checkpoint(ckpt), out / CHECKPOINT_FILENAME
        )

        shutil.copytree(tmpdir / "configs", out / "configs" / "gefion-1")
        for fp in sorted(args.configs_dir.glob("*.yaml")):
            shutil.copy(fp, out / "configs" / fp.name)
        shutil.copytree(
            tmpdir / "stats" / f"{INTERIOR_DATASTORE_NAME}.stats.zarr",
            out / "stats" / f"{INTERIOR_DATASTORE_NAME}.stats.zarr",
        )
        shutil.copytree(
            args.boundary_stats,
            out / "stats" / f"{BOUNDARY_DATASTORE_NAME}.stats.zarr",
        )
        shutil.copy(tmpdir / "training_cli_args.yaml", out)
        shutil.copy(tmpdir / "artifact.yaml", out / "gefion-1.artifact.yaml")

        # created after the configs, as neural-lam warns if the zarr is older
        # than its config
        fp_boundary_config = (
            out / "configs" / f"{BOUNDARY_DATASTORE_NAME}.yaml"
        )
        _create_stats_datastore(
            ds_boundary_stats,
            fp_boundary_config,
            out / "configs" / f"{BOUNDARY_DATASTORE_NAME}.zarr",
        )
        logger.info("checking the stats datastore loads in neural-lam")
        _check_stats_datastore(fp_boundary_config)

        meta = dict(
            artifact_name="gefion-1-local",
            description=(
                "gefion-1 completed with the boundary datastore configs and "
                "statistics, see configurations/ANNA/INFERENCE_PLAN.md"
            ),
            assembled_from=dict(
                gefion_1_zip=str(args.gefion_1_zip.resolve()),
                gefion_1_zip_sha256=_sha256(args.gefion_1_zip),
                boundary_stats=str(args.boundary_stats.resolve()),
                boundary_stats_placeholder=is_placeholder,
                boundary_stats_attrs={
                    k: str(v) for k, v in ds_boundary_stats.attrs.items()
                },
                configs_dir=str(args.configs_dir.resolve()),
                mlwm_deployment_git=_git_describe(),
            ),
            checkpoint_sanitized=True,
            created_on=datetime.datetime.now()
            .replace(microsecond=0)
            .isoformat(),
        )
        (out / "artifact.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False)
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(args.output))
    logger.info(f"assembled artifact in {args.output}")


if __name__ == "__main__":
    main()
