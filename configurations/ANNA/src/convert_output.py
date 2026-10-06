"""
Convert the neural-lam evaluation output of ANNA into DANRA-like datasets.

neural-lam (`train_model --eval test --save_eval_to_zarr_path ...`) writes the
predicted (de-normalised) state as `state(start_time,
elapsed_forecast_duration, state_feature, x, y)`. This splits the state
features back into variables:

- `single_levels.zarr`: pres_seasurface, t2m, u10m, v10m, pres0m, lwavr0m,
  swavr0m with dims (time, y, x)
- `pressure_levels.zarr`: z, t, r, u, v, tw with dims (time, pressure, y, x)

where `time` is the valid time, on the DANRA grid (with lat/lon from the
DANRA grid in the inference artifact). Conventions are those of DANRA/ANNA:
winds are relative to the DANRA (Lambert) grid and relative humidity is a
fraction (0-1).

Usage:
    python convert_output.py --prediction inference_output.zarr \\
        --danra-grid inference_artifact/grids/danra_model1_config.grid.zarr \\
        --analysis-time 2026-10-01T00:00 --output-dir outputs/
"""
import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

SINGLE_LEVEL_VARS = [
    "pres_seasurface",
    "t2m",
    "u10m",
    "v10m",
    "pres0m",
    "lwavr0m",
    "swavr0m",
]
PRESSURE_LEVEL_VARS = ["z", "t", "r", "u", "v", "tw"]
UNITS = {
    "pres_seasurface": "Pa",
    "t2m": "K",
    "u10m": "m s**-1",
    "v10m": "m s**-1",
    "pres0m": "Pa",
    "lwavr0m": "W m**-2",
    "swavr0m": "W m**-2",
    "z": "m**2 s**-2",
    "t": "K",
    "r": "1",
    "u": "m s**-1",
    "v": "m s**-1",
    "tw": "m s**-1",
}
NOTES = {
    "u10m": "relative to the DANRA (Lambert) grid x-axis",
    "v10m": "relative to the DANRA (Lambert) grid y-axis",
    "u": "relative to the DANRA (Lambert) grid x-axis",
    "v": "relative to the DANRA (Lambert) grid y-axis",
    "r": "fraction (0-1), not percent",
    "lwavr0m": "net surface long-wave radiation flux",
    "swavr0m": "net surface short-wave radiation flux",
    "tw": "geometric vertical velocity",
}


def convert(ds_pred, ds_grid, analysis_time):
    """Split the predicted state features into single/pressure level datasets."""
    if ds_pred.start_time.size != 1:
        raise ValueError(
            f"expected a single forecast, got {ds_pred.start_time.size} start times"
        )
    da = ds_pred.state.isel(start_time=0)
    valid_time = (
        ds_pred.start_time.values[0] + da.elapsed_forecast_duration
    ).values
    da = da.assign_coords(time=("elapsed_forecast_duration", valid_time))
    da = da.swap_dims(elapsed_forecast_duration="time").drop_vars(
        "elapsed_forecast_duration"
    )
    features = [str(f) for f in da.state_feature.values]

    coords = dict(
        x=ds_grid.x.values,
        y=ds_grid.y.values,
        lat=(("y", "x"), ds_grid.lat.values),
        lon=(("y", "x"), ds_grid.lon.values),
    )
    attrs = dict(
        source="ANNA forecast (DANRA ML LAM model, arXiv:2504.09340)",
        analysis_time=str(np.datetime64(analysis_time, "s")),
        start_time=str(ds_pred.start_time.values[0]),
    )

    def _var(feature):
        return (
            da.sel(state_feature=feature)
            .drop_vars("state_feature")
            .transpose("time", "y", "x")
        )

    ds_sl = xr.Dataset({v: _var(v) for v in SINGLE_LEVEL_VARS}).assign_coords(
        **coords
    )

    levels = sorted(
        {
            int(m.group(1))
            for f in features
            if (m := re.fullmatch(r"z(\d+)", f))
        }
    )
    ds_pl = xr.Dataset(
        {
            v: xr.concat(
                [_var(f"{v}{p}") for p in levels],
                dim=xr.DataArray(levels, dims="pressure", name="pressure"),
            ).transpose("time", "pressure", "y", "x")
            for v in PRESSURE_LEVEL_VARS
        }
    ).assign_coords(**coords)
    ds_pl.pressure.attrs["units"] = "hPa"

    expected = set(SINGLE_LEVEL_VARS) | {
        f"{v}{p}" for v in PRESSURE_LEVEL_VARS for p in levels
    }
    if expected != set(features):
        raise ValueError(
            f"unexpected state features: {set(features) ^ expected}"
        )

    for ds in [ds_sl, ds_pl]:
        ds.attrs = attrs
        for v in ds.data_vars:
            ds[v].attrs["units"] = UNITS[v]
            if v in NOTES:
                ds[v].attrs["comment"] = NOTES[v]
    return ds_sl, ds_pl


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--danra-grid", required=True, type=Path)
    parser.add_argument("--analysis-time", required=True, type=np.datetime64)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ds_pred = xr.open_zarr(args.prediction)
    ds_grid = xr.open_zarr(args.danra_grid).load()
    ds_sl, ds_pl = convert(ds_pred, ds_grid, args.analysis_time)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in [("single_levels", ds_sl), ("pressure_levels", ds_pl)]:
        fp = args.output_dir / f"{name}.zarr"
        ds.to_zarr(fp, mode="w", consolidated=True)
        logger.info(
            f"wrote {fp} ({ds.time.size} valid times {ds.time.values})"
        )


if __name__ == "__main__":
    main()
