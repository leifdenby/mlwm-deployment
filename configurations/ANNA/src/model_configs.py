"""
Resolve which configs ANNA uses for a boundary source.

`configs/model.yaml` names the neural-lam config per boundary source
(`neural_lam_configs`), and each neural-lam config names its interior and
boundary datastore configs (and, for DINI/IFS, the ERA5 training boundary
datastore whose statistics normalise the boundary). So the chain is
model.yaml -> neural-lam config -> datastore configs, each link stated once.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

TRAINING_BOUNDARY_SOURCE = "era5"


@dataclass
class ModelConfigs:
    """Config files (in the configs directory) for one boundary source."""

    neural_lam_config: str
    interior_datastore: str
    boundary_datastore: str
    # the boundary datastore whose training statistics normalise the
    # boundary (the boundary datastore itself when there's no override)
    boundary_stats_datastore: str

    @staticmethod
    def name(fn):
        """Datastore name: the config file name without `.yaml`."""
        return Path(fn).stem


def boundary_sources(configs_dir):
    """Boundary sources with a neural-lam config in model.yaml."""
    return list(_model_yaml(configs_dir)["neural_lam_configs"])


def model_configs(configs_dir, boundary_source):
    """The configs (file names in `configs_dir`) for `boundary_source`."""
    configs_dir = Path(configs_dir)
    nl_configs = _model_yaml(configs_dir)["neural_lam_configs"]
    if boundary_source not in nl_configs:
        raise KeyError(
            f"no neural-lam config for boundary source {boundary_source!r} "
            f"in model.yaml (have {sorted(nl_configs)})"
        )
    fn_nl_config = nl_configs[boundary_source]
    nl_config = yaml.safe_load((configs_dir / fn_nl_config).read_text())
    boundary = nl_config["datastore_boundary"]
    return ModelConfigs(
        neural_lam_config=fn_nl_config,
        interior_datastore=nl_config["datastore"]["config_path"],
        boundary_datastore=boundary["config_path"],
        boundary_stats_datastore=boundary.get(
            "overload_stats_path", boundary["config_path"]
        ),
    )


def _model_yaml(configs_dir):
    return yaml.safe_load((Path(configs_dir) / "model.yaml").read_text())
