"""
Check that neural-lam can build the forecast sample from the inference
datastores written by `src/create_inference_dataset.py`.

The test dataset is set up as `train_model --eval test` does it in
`entry.sh` (`--ar_steps_eval` for the forecast duration, no init-time
filtering, no dynamic time deltas), and its single sample is checked:

- exactly one sample, starting at the analysis time;
- the times of the initial states (T+0, T+3h), targets (T+6h ..
  T+duration), interior forcing and boundary windows;
- tensor shapes, and that all values are finite;
- the standardised values per feature, flagging features whose mean or std
  are far from 0 and 1 (a sign of wrong units or conventions, e.g. % instead
  of a fraction, or wind directions).

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/dev-utils/check_inference_sample.py \\
        <workdir with config.yaml> --forecast-duration PT6H
"""
import argparse
import sys
from pathlib import Path

import isodate
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from create_inference_dataset import ar_steps_for  # noqa: E402

# standardised values: flag a feature if its mean is further than this from 0,
# or its std is outside this range
MAX_ABS_MEAN = 3.0
STD_RANGE = (0.05, 5.0)


def _times(da):
    for dim in ["time", "analysis_time"]:
        if dim in da.coords:
            return [str(t)[:16] for t in np.atleast_1d(da[dim].values)]
    return None


def _feature_summary(name, values, features):
    """Print and return the features with unusual standardised values."""
    # values: (..., grid_index, feature), averaged over all but the features
    flat = values.reshape(-1, values.shape[-1])
    means, stds = flat.mean(axis=0), flat.std(axis=0)
    flagged = [
        (f, m, s)
        for f, m, s in zip(features, means, stds)
        if abs(m) > MAX_ABS_MEAN
        # time-of-day/year features are uniform in space and nearly constant
        # over a few hours, so their spread says nothing
        or (
            not STD_RANGE[0] <= s <= STD_RANGE[1]
            and not str(f).startswith(("hour_of_day_", "day_of_year_"))
        )
    ]
    print(
        f"  {name}: standardised mean in [{means.min():+.2f}, "
        f"{means.max():+.2f}], std in [{stds.min():.2f}, {stds.max():.2f}]"
    )
    for f, m, s in flagged:
        print(f"    FLAG {f}: mean {m:+.2f}, std {s:.2f}")
    return flagged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("workdir", type=Path)
    parser.add_argument(
        "--forecast-duration", required=True, type=isodate.parse_duration
    )
    args = parser.parse_args()

    from neural_lam.config import load_config_and_datastores
    from neural_lam.weather_dataset import WeatherDataModule

    config, datastore, datastore_boundary = load_config_and_datastores(
        config_path=str(args.workdir / "config.yaml")
    )
    ar_steps = ar_steps_for(args.forecast_duration)
    # as train_model builds it for `--eval test` (see entry.sh)
    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        ar_steps_eval=ar_steps,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        batch_size=1,
        num_workers=1,
        eval_split="test",
        eval_init_times=[],
        dynamic_time_deltas=False,
        excluded_intervals=config.training.excluded_intervals,
    )
    data_module.setup(stage="test")
    dataset = data_module.test_dataset
    problems = []

    print(f"ar_steps_eval: {ar_steps}, test samples: {len(dataset)}")
    if len(dataset) != 1:
        problems.append(f"expected 1 test sample, got {len(dataset)}")
    if len(dataset) == 0:
        sys.exit("PROBLEM: no test sample")

    (
        da_init,
        da_target,
        da_forcing,
        da_boundary,
        da_target_times,
    ) = dataset._build_item_dataarrays(idx=0)
    init_states, target_states, forcing, boundary, target_times = dataset[0]

    print("\ntimes:")
    print(f"  initial states: {_times(da_init)}")
    print(f"  targets:        {[str(t)[:16] for t in da_target_times.values]}")
    for name, da in [("forcing", da_forcing), ("boundary", da_boundary)]:
        print(f"  {name} window dims {dict(da.sizes)}")
        print(f"    time: {[str(v)[:16] for v in da.time.values]}")
        # window time deltas (repeated for every feature), in hours; the
        # window has num_past + 1 + num_future = 3 steps
        deltas = da.window_time_deltas.values.reshape(-1)[:3]
        hours = (deltas / np.timedelta64(1, "h")).tolist()
        print(f"    window time deltas (h): {hours}")

    if not datastore_boundary.is_forecast:
        # as WeatherDataset selects them: the boundary time at or before each
        # target time, and the window around it
        boundary_times = dataset.da_boundary_forcing.time.values
        window = np.arange(-1, 2)
        # the times create_inference_dataset.py padded with copies, which
        # must never be used
        padded = set()
        for fp in args.workdir.glob("*.input.zarr"):
            attr = xr.open_zarr(fp).attrs.get("boundary_times_padded", "")
            padded |= {t.strip() for t in attr.split(",") if t.strip()}
        print(f"  padded (unused) boundary times: {sorted(padded)}")
        print("  boundary times used per target:")
        for t in da_target_times.values:
            idx = np.searchsorted(boundary_times, t, side="right") - 1
            used = [str(boundary_times[idx + w])[:16] for w in window]
            print(f"    {str(t)[:16]}: {used}")
            if padded & set(used):
                problems.append(
                    f"target {str(t)[:16]} uses padded boundary times "
                    f"{sorted(padded & set(used))}"
                )

    analysis_time = np.datetime64(_times(da_init)[0])
    expected_targets = [
        analysis_time + np.timedelta64(3 * (i + 2), "h")
        for i in range(ar_steps)
    ]
    if list(da_target_times.values.astype("datetime64[h]")) != [
        t.astype("datetime64[h]") for t in expected_targets
    ]:
        problems.append(
            f"unexpected target times, expected {expected_targets}"
        )

    print("\nshapes:")
    n_interior = datastore.num_grid_points
    n_boundary = datastore_boundary.num_grid_points
    for name, tensor, shape in [
        ("init_states", init_states, (2, n_interior, None)),
        ("target_states", target_states, (ar_steps, n_interior, None)),
        ("forcing", forcing, (ar_steps, n_interior, None)),
        ("boundary", boundary, (ar_steps, n_boundary, None)),
        ("target_times", target_times, (ar_steps,)),
    ]:
        finite = bool(np.isfinite(tensor.numpy()).all())
        print(f"  {name}: {tuple(tensor.shape)}, finite: {finite}")
        if not finite:
            problems.append(f"{name} has non-finite values")
        if any(e is not None and e != s for e, s in zip(shape, tensor.shape)):
            problems.append(f"{name} has shape {tuple(tensor.shape)}")

    print("\nstandardised values (training statistics):")
    state_features = list(datastore.get_vars_names(category="state"))
    flagged = _feature_summary(
        "interior state",
        np.concatenate([init_states.numpy(), target_states.numpy()]),
        state_features,
    )
    # the windowed forcing/boundary stack the window steps per feature, so
    # summarise the raw (standardised) datastore values instead
    for name, ds, category in [
        ("interior forcing", datastore, "forcing"),
        ("boundary forcing", datastore_boundary, "forcing"),
        ("boundary static", datastore_boundary, "static"),
    ]:
        da = ds.get_dataarray(
            category=category, split="test", standardize=True
        )
        da = da.transpose(..., "grid_index", f"{category}_feature")
        flagged += _feature_summary(
            name,
            np.asarray(da.values),
            list(da[f"{category}_feature"].values),
        )
    if flagged:
        problems.append(
            f"{len(flagged)} feature(s) with unusual standardised values"
        )

    print()
    for problem in problems:
        print("PROBLEM:", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    xr.set_options(keep_attrs=True)
    main()
