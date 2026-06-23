import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import dataclass_wizard as dw

ENV_VAR_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass
class Schedule:
    cron: str
    timezone: str


@dataclass
class ManagedImage:
    published_as: str
    build: bool = True
    publish: bool = True
    build_secret_env: List[str] = field(default_factory=list)


@dataclass
class DeploymentJob:
    image: str
    schedule: Schedule
    runtime_context_script: str


@dataclass
class DeploymentConfig(dw.YAMLWizard):
    images: Dict[str, ManagedImage] = field(default_factory=dict)
    jobs: List[DeploymentJob] = field(default_factory=list)


def find_managed_image_model(
    config: DeploymentConfig, image_ref: str
) -> str | None:
    for model, image in config.images.items():
        if image.published_as == image_ref:
            return model
    return None


def load_deployments_config(config_path: str | Path) -> DeploymentConfig:
    return DeploymentConfig.from_yaml_file(config_path)


def validate_deployments_config(
    config: DeploymentConfig,
    repo_root: str | Path,
    *,
    image_exists: Callable[[str], bool] | None = None,
) -> None:
    repo_root = Path(repo_root)
    configurations_dir = repo_root / "configurations"

    published_images = {}
    for model, image in config.images.items():
        model_dir = configurations_dir / model
        if not model_dir.is_dir():
            raise ValueError(
                f"Managed image '{model}' does not map to an existing "
                f"configuration directory: {model_dir}"
            )

        containerfile = model_dir / "Containerfile"
        if image.build and not containerfile.is_file():
            raise ValueError(
                f"Managed image '{model}' is build-enabled but missing "
                f"Containerfile: {containerfile}"
            )

        if image.published_as in published_images:
            raise ValueError(
                f"Duplicate published_as image reference: {image.published_as}"
            )
        published_images[image.published_as] = model

        for env_var in image.build_secret_env:
            if not ENV_VAR_PATTERN.fullmatch(env_var):
                raise ValueError(
                    f"Managed image '{model}' has invalid build_secret_env "
                    f"entry: {env_var}"
                )

    for job in config.jobs:
        runtime_context_script = repo_root / job.runtime_context_script
        if not runtime_context_script.is_file():
            raise ValueError(
                f"Runtime context script does not exist: {job.runtime_context_script}"
            )

        if job.image in published_images:
            continue

        if image_exists is None:
            raise ValueError(
                f"Job image '{job.image}' is not declared under images and no "
                "registry lookup was provided"
            )

        if not image_exists(job.image):
            raise ValueError(
                f"Job image '{job.image}' is not declared under images and was "
                "not found in the container registry"
            )
