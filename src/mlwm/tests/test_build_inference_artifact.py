from pathlib import Path

import mlwm.build_inference_artifact as bia
import numpy as np
import pytest
import xarray as xr
import yaml


def _write_datastore(fp_config: Path, n_features: int):
    """Write a (dummy) datastore config and a zarr dataset with statistics"""
    fp_config.write_text(f"# dummy datastore config {fp_config.name}\n")
    ds = xr.Dataset(
        {
            "forcing": (("time", "forcing_feature"), np.ones((2, n_features))),
            "forcing__train__mean": ("forcing_feature", np.zeros(n_features)),
            "forcing__train__std": ("forcing_feature", np.ones(n_features)),
        },
        coords=dict(forcing_feature=[f"f{i}" for i in range(n_features)]),
    )
    ds.to_zarr(fp_config.with_suffix(".zarr"))


@pytest.fixture
def nl_config_with_boundary(tmp_path):
    """
    neural-lam config with an interior (relative path) and a boundary
    (absolute path) datastore, as used for ANNA (gefion-1)
    """
    datastore_dir = tmp_path / "datastores"
    datastore_dir.mkdir()
    _write_datastore(datastore_dir / "interior.yaml", n_features=3)
    _write_datastore(datastore_dir / "boundary.yaml", n_features=5)

    nl_config = dict(
        datastore=dict(kind="mdp", config_path="datastores/interior.yaml"),
        datastore_boundary=dict(
            kind="mdp", config_path=str(datastore_dir / "boundary.yaml")
        ),
        training=dict(),
    )
    fp_nl_config = tmp_path / "config.yaml"
    fp_nl_config.write_text(yaml.dump(nl_config))
    return fp_nl_config


def test_find_datastore_paths_with_boundary(nl_config_with_boundary):
    paths = bia._find_datastore_paths(str(nl_config_with_boundary))
    datastore_dir = nl_config_with_boundary.parent / "datastores"
    assert {k: Path(v) for k, v in paths.items()} == {
        ("datastore",): datastore_dir / "interior.yaml",
        ("datastore_boundary",): datastore_dir / "boundary.yaml",
    }


def test_find_datastore_paths_multiple_datastores(tmp_path):
    nl_config = dict(
        datastores=dict(
            danra=dict(kind="mdp", config_path="danra.yaml"),
            era5=dict(kind="mdp", config_path="/data/era5.yaml"),
        )
    )
    fp_nl_config = tmp_path / "config.yaml"
    fp_nl_config.write_text(yaml.dump(nl_config))
    paths = bia._find_datastore_paths(str(fp_nl_config))
    assert {k: str(v) for k, v in paths.items()} == {
        ("datastores", "danra"): str(tmp_path / "danra.yaml"),
        ("datastores", "era5"): "/data/era5.yaml",
    }


def test_find_datastore_paths_no_datastore(tmp_path):
    fp_nl_config = tmp_path / "config.yaml"
    fp_nl_config.write_text(yaml.dump(dict(training=dict())))
    with pytest.raises(ValueError):
        bia._find_datastore_paths(str(fp_nl_config))


def test_copy_yaml_configs_includes_boundary(
    nl_config_with_boundary, tmp_path
):
    artifact_path = tmp_path / "artifact"
    bia._copy_yaml_configs(str(nl_config_with_boundary), str(artifact_path))

    configs_dir = artifact_path / "configs"
    assert (configs_dir / "interior.yaml").exists()
    assert (configs_dir / "boundary.yaml").exists()

    # datastore config paths are rewritten to the packaged files
    nl_config = yaml.safe_load((configs_dir / "config.yaml").read_text())
    assert nl_config["datastore"]["config_path"] == "interior.yaml"
    assert nl_config["datastore_boundary"]["config_path"] == "boundary.yaml"


def test_extract_stats_includes_boundary(nl_config_with_boundary, tmp_path):
    artifact_path = tmp_path / "artifact"
    bia._extract_stats_for_all_datastores(
        str(nl_config_with_boundary), str(artifact_path)
    )
    for name, n_features in [("interior", 3), ("boundary", 5)]:
        ds_stats = xr.open_zarr(artifact_path / "stats" / f"{name}.stats.zarr")
        # only the statistics are extracted, not the data itself
        assert set(ds_stats.data_vars) == {
            "forcing__train__mean",
            "forcing__train__std",
        }
        assert ds_stats.forcing_feature.size == n_features
