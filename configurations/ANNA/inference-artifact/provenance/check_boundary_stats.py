"""
Check ERA5 boundary statistics from `compute_era5_boundary_stats.py`: their
attributes, the feature order and that all values are finite (std > 0).
Optionally compare them with another stats zarr (e.g. the 3-day placeholder
used before the exact statistics were available).

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/provenance/check_boundary_stats.py \\
        era_7deg_model1_config.stats.zarr [--compare placeholder.stats.zarr]
"""
import argparse
import sys
from pathlib import Path

import mllam_data_prep as mdp
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from compute_era5_boundary_stats import (  # noqa: E402
    DEFAULT_CONFIG,
    _feature_definitions,
)

DERIVED = [
    "toa_radiation",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stats", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    ds = xr.open_zarr(args.stats).load()
    print("attrs:")
    for k, v in ds.attrs.items():
        print(f"  {k}: {v}")

    forcing, static = (
        [f["name"] for f in features]
        for features in _feature_definitions(
            mdp.Config.from_yaml_file(DEFAULT_CONFIG)
        )
    )
    ok = list(ds.forcing_feature.values) == forcing
    print(f"\nforcing features: {ds.forcing_feature.size}, as training: {ok}")
    ok = list(ds.static_feature.values) == static
    print(f"static features: {ds.static_feature.size}, as training: {ok}")
    for v in ds.data_vars:
        vals = ds[v].values
        extra = f", all > 0: {bool((vals > 0).all())}" if "std" in v else ""
        print(f"  {v}: finite={bool(np.isfinite(vals).all())}{extra}")

    if args.compare is None:
        return
    other = xr.open_zarr(args.compare).load()
    print(f"\ncompared with {args.compare}:")
    rows = []
    for f in ds.forcing_feature.values:
        a, b = ds.sel(forcing_feature=f), other.sel(forcing_feature=f)
        mean, std = float(a.forcing__train__mean), float(a.forcing__train__std)
        # shift of the other mean in units of this std, and std ratio
        rows.append(
            (
                f,
                (float(b.forcing__train__mean) - mean) / std,
                float(b.forcing__train__std) / std,
            )
        )
    for f, dz, ratio in sorted(rows, key=lambda r: -abs(r[1]))[:10]:
        kind = " (derived)" if f in DERIVED else ""
        print(
            f"  {f:26s} mean off by {dz:+.2f} std, std ratio {ratio:.2f}{kind}"
        )
    for f in ds.static_feature.values:
        a, b = ds.sel(static_feature=f), other.sel(static_feature=f)
        print(
            f"  {f:26s} static mean {float(a.static__train__mean):.5g} vs "
            f"{float(b.static__train__mean):.5g}"
        )


if __name__ == "__main__":
    main()
