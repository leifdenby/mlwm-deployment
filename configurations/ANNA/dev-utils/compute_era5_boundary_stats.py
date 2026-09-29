"""
Recompute the ERA5 boundary training statistics of ANNA (gefion-1) from
WeatherBench2 ERA5.

The statistics ANNA's boundary was normalised with (`forcing__train__*` and
`static__train__*` of the `era_7deg_model1_config` datastore) were never
exported from Gefion. They can be reproduced exactly from the public
WeatherBench2 ERA5 store because we know
- the boundary datastore config (`configs/era_7deg_model1_config.yaml`),
- the ERA5 subset box the datastore was built from
  (`era_danra_model1_subset.zarr`: lat 79.25..32.75, lon -27.0..39.5 at 0.25
  deg, 187 x 267 points), and
- exactly how mllam-data-prep (sadamov/mllam-data-prep@dd9af481, the version
  used for training) computes statistics.

Two details of that mllam-data-prep version are replicated here:
1. statistics are computed *before* domain cropping, i.e. over all points of
   the ERA5 subset box, not only over the 18014 boundary points
2. `calc_stats()` overwrites the dataset when applying a `diff_` operation, so
   with `ops: [mean, std, diff_mean, diff_std]` `diff_mean` is the mean of the
   first time-differences but `diff_std` is the std of the *second*
   time-differences

Rather than building the (~300 GB) datastore, the training split is streamed
in blocks of consecutive time steps and per-feature (count, mean, M2)
accumulators are combined (Chan et al.), in float64 as in training (the
forcing features are promoted to float64 when stacked with the derived
features). Each block's accumulators are saved in `--partials-dir`, so an
interrupted run can be restarted and continues where it stopped.

WeatherBench2 stores one global chunk per time step (all 13 levels for
pressure level variables), so every time step requires reading ~345 MB
(uncompressed) whatever the box size. The full training split
(2000-01-01..2018-10-29, 6-hourly, ~27500 steps) is therefore ~9 TB of reads.
`--block-stride N` processes only every N-th block, which gives approximate
statistics much faster (differences are still exact within blocks); the
output attributes record this.

The output zarr has the same layout as the stats zarr in the inference
artifact (`stats/{datastore_name}.stats.zarr`).

Example (full, exact computation):

    uv run --project configurations/ANNA --with gcsfs \\
        python configurations/ANNA/dev-utils/compute_era5_boundary_stats.py \\
        --output era_7deg_model1_config.stats.zarr \\
        --partials-dir era5_stats_partials

Validation against a datastore built by mllam-data-prep (e.g. a small test
datastore built from a local ERA5 subset):

    ... compute_era5_boundary_stats.py --source era_danra_model1_subset.zarr \\
        --start 2010-01-01T00:00 --end 2010-01-02T00:00 \\
        --output test.stats.zarr --partials-dir test_partials
"""
import argparse
import datetime
import json
from pathlib import Path

import mllam_data_prep as mdp
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mllam_data_prep.ops.derive_variable import derive_variable

WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
DEFAULT_CONFIG = (
    Path(__file__).parent.parent / "configs" / "era_7deg_model1_config.yaml"
)

# ERA5 subset box the training boundary datastore was built from
# (`era_danra_model1_subset.zarr`, made with neural-lam-dev
# `scripts/era_download.py`)
SUBSET_LATS = np.arange(79.25, 32.75 - 0.125, -0.25)
SUBSET_LONS = np.arange(-27.0, 39.5 + 0.125, 0.25) % 360.0
assert SUBSET_LATS.size == 187 and SUBSET_LONS.size == 267

SPLIT_NAME = "train"


def _open_source(source):
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    ds = xr.open_zarr(source, storage_options=storage_options)
    return ds.sel(latitude=SUBSET_LATS, longitude=SUBSET_LONS)


def _feature_definitions(config):
    """
    Return the forcing and static feature definitions in the order
    mllam-data-prep creates them (inputs in config order, within an input the
    selected variables before the derived ones, pressure level variables
    stacked as "{var_name}{level}").
    """
    forcing, static = [], []
    for input_name, input_config in config.inputs.items():
        target = input_config.target_output_variable
        features = forcing if target == "forcing" else static
        variables = input_config.variables or []
        if isinstance(variables, dict):
            for var_name, coords_to_sample in variables.items():
                ((coord, selection),) = coords_to_sample.items()
                for value in selection.values:
                    features.append(
                        dict(
                            name=f"{var_name}{value}",
                            input=input_name,
                            var=var_name,
                            sel={coord: value},
                        )
                    )
        else:
            for var_name in variables:
                features.append(
                    dict(name=var_name, input=input_name, var=var_name)
                )
        for var_name, derived_variable in (
            input_config.derived_variables or {}
        ).items():
            features.append(
                dict(
                    name=var_name,
                    input=input_name,
                    derived=derived_variable,
                    dims=input_config.dims,
                )
            )
    return forcing, static


def _feature_values(ds_block, feature, chunking, template):
    """Values of one feature as float64 array of shape (time, grid_point)."""
    if "derived" in feature:
        da = derive_variable(
            ds=ds_block,
            derived_variable=feature["derived"],
            chunking=chunking,
            target_dims=feature["dims"],
        )
    else:
        da = ds_block[feature["var"]]
        if "sel" in feature:
            da = da.sel(feature["sel"])
    # time-only derived features (e.g. hour of day) are broadcast over the
    # grid, as they are when stacked into the `forcing` variable
    da = da.broadcast_like(template).transpose(*template.dims)
    return np.asarray(da.values, dtype=np.float64).reshape(
        template.sizes["time"], -1
    )


def _moments(values):
    """(count, mean, M2) of the finite values of an array."""
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return 0, 0.0, 0.0
    mean = finite.mean()
    return n, mean, float(((finite - mean) ** 2).sum())


def _combine(acc_a, acc_b):
    """Combine (count, mean, M2) accumulators (Chan et al. 1979)."""
    n_a, mean_a, m2_a = acc_a
    n_b, mean_b, m2_b = acc_b
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
    return n, mean, m2


def _process_block(
    ds, time_index, i0, i1, forcing_features, chunking, fp_partial
):
    """
    Accumulate moments for time steps [i0, i1) of the split. Two earlier time
    steps are loaded too, so first and second differences ending in the block
    are exact.
    """
    j0 = max(i0 - 2, 0)
    ds_block = ds.isel(time=slice(j0, i1)).load()
    assert (ds_block.time.values == time_index[j0:i1]).all()
    template = ds_block["2m_temperature"]
    n_own = i1 - i0
    acc = {k: np.zeros((len(forcing_features), 3)) for k in ["x", "d1", "d2"]}
    for i, feature in enumerate(forcing_features):
        x = _feature_values(ds_block, feature, chunking, template)
        d1 = np.diff(x, axis=0)
        d2 = np.diff(d1, axis=0)
        # x[k] is time step j0 + k, d1[k] ends at j0 + k + 1, d2[k] at j0 + k + 2
        acc["x"][i] = _moments(x[-n_own:])
        acc["d1"][i] = _moments(d1[-min(n_own, d1.shape[0]) :])
        acc["d2"][i] = _moments(d2[-min(n_own, d2.shape[0]) :])
    np.savez(fp_partial, **acc, i0=i0, i1=i1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="stats zarr to write"
    )
    parser.add_argument(
        "--partials-dir",
        required=True,
        type=Path,
        help="per-block accumulators",
    )
    parser.add_argument("--source", default=WB2_ERA5, help="ERA5 zarr store")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument(
        "--start", default=None, help="split start (default: from config)"
    )
    parser.add_argument(
        "--end", default=None, help="split end (default: from config)"
    )
    parser.add_argument(
        "--block-steps",
        type=int,
        default=40,
        help="time steps per block (10 days)",
    )
    parser.add_argument(
        "--block-stride",
        type=int,
        default=1,
        help="process every N-th block only (approximate statistics if > 1)",
    )
    args = parser.parse_args()

    config = mdp.Config.from_yaml_file(args.config)
    split = config.output.splitting.splits[SPLIT_NAME]
    stats_config = split.compute_statistics
    if list(stats_config.ops) != ["mean", "std", "diff_mean", "diff_std"]:
        raise NotImplementedError(
            f"unsupported statistics ops {stats_config.ops}"
        )
    start = pd.Timestamp(args.start or split.start)
    end = pd.Timestamp(args.end or split.end)
    step = pd.Timedelta(config.output.coord_ranges["time"].step)

    forcing_features, static_features = _feature_definitions(config)
    logger.info(
        f"{len(forcing_features)} forcing features: "
        f"{[f['name'] for f in forcing_features]}"
    )
    logger.info(f"{len(static_features)} static features")

    ds = _open_source(args.source)
    ds = ds.sel(time=slice(start, end))
    time_index = ds.time.values
    if not (np.diff(time_index) == step.to_timedelta64()).all():
        raise ValueError(f"source time steps in split are not all {step}")
    n_steps = time_index.size
    logger.info(
        f"split {SPLIT_NAME}: {start} .. {end}, {n_steps} steps of {step}"
    )

    blocks = list(range(0, n_steps, args.block_steps))[:: args.block_stride]
    args.partials_dir.mkdir(parents=True, exist_ok=True)
    run_info = dict(
        source=args.source,
        start=str(start),
        end=str(end),
        block_steps=args.block_steps,
        block_stride=args.block_stride,
        features=[f["name"] for f in forcing_features],
    )
    fp_run_info = args.partials_dir / "run_info.json"
    if fp_run_info.exists():
        if json.loads(fp_run_info.read_text()) != run_info:
            raise ValueError(
                f"{args.partials_dir} contains partials from a run with other "
                "arguments, use a new --partials-dir"
            )
    else:
        fp_run_info.write_text(json.dumps(run_info, indent=2))

    chunking = config.output.chunking
    for n, i0 in enumerate(blocks):
        i1 = min(i0 + args.block_steps, n_steps)
        fp_partial = args.partials_dir / f"block_{i0:06d}_{i1:06d}.npz"
        if fp_partial.exists():
            continue
        logger.info(
            f"block {n + 1}/{len(blocks)}: {time_index[i0]} .. {time_index[i1 - 1]}"
        )
        _process_block(
            ds, time_index, i0, i1, forcing_features, chunking, fp_partial
        )

    # combine the block accumulators
    totals = {
        k: [(0, 0.0, 0.0)] * len(forcing_features) for k in ["x", "d1", "d2"]
    }
    for i0 in blocks:
        i1 = min(i0 + args.block_steps, n_steps)
        partial = np.load(args.partials_dir / f"block_{i0:06d}_{i1:06d}.npz")
        for k in totals:
            totals[k] = [
                _combine(total, tuple(p))
                for total, p in zip(totals[k], partial[k])
            ]

    for k, total in totals.items():
        if any(n == 0 for n, _, _ in total):
            logger.warning(
                f"no values for `{k}` statistics (too few time steps?)"
            )

    def _mean(k):
        return np.array(
            [mean if n > 0 else np.nan for n, mean, _ in totals[k]]
        )

    def _std(k):
        # population std (ddof=0) as xarray's default
        return np.array(
            [np.sqrt(m2 / n) if n > 0 else np.nan for n, _, m2 in totals[k]]
        )

    ds_static = ds.isel(time=0)
    static_values = np.stack(
        [
            np.asarray(ds_static[f["var"]].values, dtype=np.float64).ravel()
            for f in static_features
        ]
    )
    forcing_names = [f["name"] for f in forcing_features]
    static_names = [f["name"] for f in static_features]
    ds_stats = xr.Dataset(
        {
            f"forcing__{SPLIT_NAME}__mean": ("forcing_feature", _mean("x")),
            f"forcing__{SPLIT_NAME}__std": ("forcing_feature", _std("x")),
            f"forcing__{SPLIT_NAME}__diff_mean": (
                "forcing_feature",
                _mean("d1"),
            ),
            # second differences, as in the mllam-data-prep version used for
            # training (see module docstring)
            f"forcing__{SPLIT_NAME}__diff_std": (
                "forcing_feature",
                _std("d2"),
            ),
            f"static__{SPLIT_NAME}__mean": (
                "static_feature",
                np.nanmean(static_values, axis=1),
            ),
            f"static__{SPLIT_NAME}__std": (
                "static_feature",
                np.nanstd(static_values, axis=1),
            ),
        },
        coords=dict(
            forcing_feature=forcing_names, static_feature=static_names
        ),
    )
    n_blocks_total = len(range(0, n_steps, args.block_steps))
    ds_stats.attrs = dict(
        description=(
            "ERA5 boundary training statistics for ANNA (gefion-1), recomputed "
            "from WeatherBench2 ERA5 over the era_danra_model1_subset box with "
            "configurations/ANNA/dev-utils/compute_era5_boundary_stats.py"
        ),
        source=args.source,
        split=SPLIT_NAME,
        split_start=str(start),
        split_end=str(end),
        n_time_steps=n_steps,
        n_grid_points=SUBSET_LATS.size * SUBSET_LONS.size,
        blocks_processed=f"{len(blocks)}/{n_blocks_total}",
        exact=str(args.block_stride == 1),
        diff_std_note="std of second time-differences (as in training)",
        created_on=datetime.datetime.now().replace(microsecond=0).isoformat(),
    )
    ds_stats.to_zarr(args.output, mode="w", consolidated=True)
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
