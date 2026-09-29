"""
Check that the ANNA boundary datastore configs are consistent with each other
and with the data contract documented in `configs/ifs_7deg_model1_config.yaml`.

Synthetic zarr datasets following the contract are created for the IFS/DINI
boundaries (and in ERA5 layout for the training boundary), and each config is
run through `mllam_data_prep.create_dataset` (with `domain_cropping` disabled,
as that would require building the full DANRA interior dataset). The resulting
boundary features must match the ERA5 training boundary exactly, in number and
order, as that is what the gefion-1 checkpoint expects.

Requires a mllam-data-prep version supporting `domain_cropping` and
`lead_time` in derived variables, i.e. the one pinned in the ANNA
`pyproject.toml` (sadamov/mllam-data-prep@building-ml-lams):

    uv run --project configurations/ANNA --with pytest \
        pytest configurations/ANNA/tests
"""
from pathlib import Path

import mllam_data_prep as mdp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]
SURFACE_VARS = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]
DERIVED_VARS = [
    "toa_radiation",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
N_FORCING_FEATURES = (
    len(SURFACE_VARS)
    + len(DERIVED_VARS)
    + len(PRESSURE_LEVEL_VARS) * len(LEVELS)
)

BOUNDARY_CONFIGS = {
    "era5": "era_7deg_model1_config.yaml",
    "ifs": "ifs_7deg_model1_config.yaml",
    "dini": "dini_7deg_model1_config.yaml",
}
NL_CONFIGS = {
    "era5": "7deg_config_era5.yaml",
    "ifs": "7deg_config_ifs.yaml",
    "dini": "7deg_config_dini.yaml",
}

# small regular lat/lon grid, a few points is enough
LATS = np.array([60.0, 59.75, 59.5])
LONS = np.array([10.0, 10.25, 10.5, 10.75])


def _spatial_coords():
    return dict(latitude=("latitude", LATS), longitude=("longitude", LONS))


def _make_forecast_contract_dataset(analysis_time, lead_hours):
    """
    Synthetic dataset following the IFS boundary contract (also used by the
    DINI boundary written by `src/regrid_dini.py`). An extra pressure level
    (500 hPa) is included to check that level selection works.
    """
    rng = np.random.default_rng(42)
    levels = sorted(LEVELS + [500])
    coords = dict(
        time=("time", [np.datetime64(analysis_time, "ns")]),
        prediction_timedelta=(
            "prediction_timedelta",
            pd.to_timedelta(lead_hours, unit="h").values,
        ),
        level=("level", levels, {"units": "hPa"}),
        **_spatial_coords(),
    )
    sfc_dims = ("time", "prediction_timedelta", "latitude", "longitude")
    pl_dims = (
        "time",
        "prediction_timedelta",
        "level",
        "latitude",
        "longitude",
    )
    sfc_shape = (1, len(lead_hours), len(LATS), len(LONS))
    pl_shape = (1, len(lead_hours), len(levels), len(LATS), len(LONS))

    data_vars = {}
    for var in SURFACE_VARS:
        data_vars[var] = (sfc_dims, rng.normal(size=sfc_shape))
    for var in PRESSURE_LEVEL_VARS:
        data_vars[var] = (pl_dims, rng.normal(size=pl_shape))
    # statics must not carry time/lead-time/level dims
    for var in STATIC_VARS:
        data_vars[var] = (
            ("latitude", "longitude"),
            rng.normal(size=(len(LATS), len(LONS))),
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _make_era5_layout_dataset(times):
    """Synthetic dataset in the (WeatherBench2) ERA5 layout used for training."""
    rng = np.random.default_rng(42)
    coords = dict(
        time=("time", pd.to_datetime(times).values),
        level=("level", LEVELS),
        **_spatial_coords(),
    )
    data_vars = {}
    for var in SURFACE_VARS:
        data_vars[var] = (
            ("time", "latitude", "longitude"),
            rng.normal(size=(len(times), len(LATS), len(LONS))),
        )
    for var in PRESSURE_LEVEL_VARS:
        data_vars[var] = (
            ("time", "level", "latitude", "longitude"),
            rng.normal(size=(len(times), len(LEVELS), len(LATS), len(LONS))),
        )
    for var in STATIC_VARS:
        data_vars[var] = (
            ("latitude", "longitude"),
            rng.normal(size=(len(LATS), len(LONS))),
        )
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _create_boundary_dataset(config_name, fp_input):
    config = mdp.Config.from_yaml_file(
        CONFIGS_DIR / BOUNDARY_CONFIGS[config_name]
    )
    for input_config in config.inputs.values():
        input_config.path = str(fp_input)
    # cropping needs the full interior (DANRA) dataset, which is out of scope
    # for this test
    config.output.domain_cropping = None
    return mdp.create_dataset(config=config)


@pytest.fixture(scope="module")
def boundary_datasets(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("anna_boundary")

    fp_era5 = tmp_path / "era5_layout.zarr"
    times = pd.date_range("2000-01-01T00:00", periods=4, freq="6h")
    ds_era5 = _make_era5_layout_dataset(times=times)
    ds_era5.to_zarr(fp_era5)

    datasets = {}
    for name in ["ifs", "dini"]:
        # the IFS/DINI configs are used as-is, only the input paths change
        ds = _make_forecast_contract_dataset(
            analysis_time="2026-09-26T12:00", lead_hours=[0, 6, 12]
        )
        fp = tmp_path / f"{name}.zarr"
        ds.to_zarr(fp)
        datasets[name] = _create_boundary_dataset(name, fp)

    # training config covers 2000-2020, the synthetic data only a day
    config = mdp.Config.from_yaml_file(CONFIGS_DIR / BOUNDARY_CONFIGS["era5"])
    for input_config in config.inputs.values():
        input_config.path = str(fp_era5)
    config.output.domain_cropping = None
    config.output.coord_ranges["time"].start = str(times[0])
    config.output.coord_ranges["time"].end = str(times[-1])
    for split in config.output.splitting.splits.values():
        split.start = str(times[0])
        split.end = str(times[-1])
    datasets["era5"] = mdp.create_dataset(config=config)

    return datasets


@pytest.mark.parametrize("name", list(BOUNDARY_CONFIGS))
def test_boundary_config_loads(name):
    config = mdp.Config.from_yaml_file(CONFIGS_DIR / BOUNDARY_CONFIGS[name])
    assert config.output.domain_cropping.margin_width_degrees == 7.19
    assert config.output.domain_cropping.include_interior_points is False
    assert (
        CONFIGS_DIR
        / config.output.domain_cropping.interior_dataset_config_path
    ).exists()


def test_interior_config_loads():
    mdp.Config.from_yaml_file(CONFIGS_DIR / "danra_model1_config.yaml")


@pytest.mark.parametrize("name", list(NL_CONFIGS))
def test_neural_lam_config_references(name):
    nl_config = yaml.safe_load((CONFIGS_DIR / NL_CONFIGS[name]).read_text())
    assert nl_config["datastore"]["config_path"] == "danra_model1_config.yaml"
    boundary = nl_config["datastore_boundary"]
    assert boundary["config_path"] == BOUNDARY_CONFIGS[name]
    if name == "era5":
        assert "overload_stats_path" not in boundary
    else:
        assert boundary["overload_stats_path"] == BOUNDARY_CONFIGS["era5"]
    for key in ["config_path", "overload_stats_path"]:
        if key in boundary:
            assert (CONFIGS_DIR / boundary[key]).exists()


@pytest.mark.parametrize("name", list(BOUNDARY_CONFIGS))
def test_boundary_feature_counts(boundary_datasets, name):
    ds = boundary_datasets[name]
    assert ds.forcing_feature.size == N_FORCING_FEATURES == 58
    assert ds.static_feature.size == len(STATIC_VARS) == 2
    assert ds.grid_index.size == len(LATS) * len(LONS)


@pytest.mark.parametrize("name", ["ifs", "dini"])
def test_boundary_features_match_training(boundary_datasets, name):
    ds_train = boundary_datasets["era5"]
    ds = boundary_datasets[name]
    assert list(ds.forcing_feature.values) == list(
        ds_train.forcing_feature.values
    )
    assert list(ds.static_feature.values) == list(
        ds_train.static_feature.values
    )


@pytest.mark.parametrize("name", ["ifs", "dini"])
def test_forecast_boundary_dims(boundary_datasets, name):
    ds = boundary_datasets[name]
    # dim order is up to mllam-data-prep, neural-lam selects dims by name
    assert set(ds.forcing.dims) == {
        "analysis_time",
        "elapsed_forecast_duration",
        "grid_index",
        "forcing_feature",
    }
    assert ds.elapsed_forecast_duration.size == 3
    assert not bool(ds.forcing.isnull().any())
