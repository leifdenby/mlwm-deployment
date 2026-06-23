## Implementation Plan: Repository-Wide Deployment Configuration

### Goal

Add a repository-wide mechanism for declaring which container images this repository manages, which model executions should run routinely, when they should run, and how each execution derives its runtime context.

The design should:

- keep per-model deployment details in `configurations/<model>/config.yaml`
- keep routine execution policy in a top-level manifest under `configurations/`
- avoid putting too much scheduling and context-derivation logic in YAML
- let each routine execution compute its runtime environment through a required bash script

### Proposed Configuration Layout

Add a new top-level file:

- `configurations/deployments.yaml`

Add one runtime context script per routine-enabled model, or allow reuse of shared scripts:

- `configurations/<model>/build_runtime_context.sh`
- optionally `configurations/scripts/<name>.sh`

### Proposed `deployments.yaml` Schema

Keep this file declarative and minimal. It should define which images this repository manages in CI, which jobs should run routinely, and which script to call for runtime context.

Example:

```yaml
images:
  surface-dummy-model_DINI:
    published_as: ghcr.io/dmidk/surface-dummy-model_dini:v2026.06.22
    build_secret_env:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_DEFAULT_REGION

jobs:
  - image: ghcr.io/dmidk/surface-dummy-model_dini:v2026.06.22
    schedule:
      cron: "15 */6 * * *"
      timezone: UTC
    runtime_context_script: configurations/surface-dummy-model_DINI/build_runtime_context.sh
```

Notes:

- `images` defines the image artifacts this repository is responsible for building and publishing in CI.
- the `images` key is the internal model identifier and maps to `configurations/<model>/config.yaml`.
- `published_as` is required and is the explicit container image reference that will be published and later executed.
- `build_secret_env` lists environment variables that must be provided for image builds.
- `jobs[*].image` is the explicit container image reference to run.
- `schedule` is repository metadata for the orchestrator.
- `runtime_context_script` is the source of runtime execution environment variables.
- images listed under `images` are built in CI for validation, but are only published when changes have been merged to `main`.

Validation rules for job images:

- every `jobs[*].image` must satisfy at least one of:
  - it matches an `images[*].published_as` entry in the same file
  - it already exists in the container registry
  - CI validation should check registry existence for any job image that is not declared under `images`

Additional validation and warning rules:

- warn if there is a `configurations/<model-name>/` directory with no matching `images.<model-name>` entry
- warn if an `images.<model-name>.published_as` value is not referenced by any `jobs[*].image`
- error if any `images.<model-name>` entry does not point to a valid model configuration in `configurations/<model-name>/config.yaml`

### Context Script Contract

Each configured routine job must provide a script that returns the execution context as strict `KEY=VALUE` lines on stdout.

Example output:

```bash
ANALYSIS_TIME=2026-06-22T060000Z
FORECAST_DURATION=PT18H
TIME_DIMENSIONS=time
DATASTORE_INPUT_PATHS=danra.danra_surface=s3://...,danra.danra_static=s3://...
```

Contract requirements:

- stdout must contain only `KEY=VALUE` lines
- diagnostics should go to stderr
- non-zero exit status means context generation failed
- script must be deterministic for the same inputs
- script should not mutate repository files or external state

Recommended invocation shape:

```bash
configurations/<model>/build_runtime_context.sh --scheduled-time 2026-06-22T07:15:00Z
```

This lets the script derive `ANALYSIS_TIME` from the scheduler trigger time without encoding date-rounding logic in YAML.

### How This Fits Current Code

Current relevant functionality already present in `src/mlwm/`:

- `config_spec.py` parses per-model `config.yaml`
- `run_models.py` already discovers model configs under `configurations/`
- `run_models.py` already prepares input and output paths from model config + `analysis_time`
- `run_models.py` is the natural place to add routine job loading and execution orchestration

Current gaps to fill:

- no top-level routine manifest parser
- no top-level image build manifest parser
- no runtime context abstraction
- no docker/podman launch implementation
- `analysis_time` is hardcoded
- all directories under `configurations/` are assumed to be runnable
- CI image orchestration is hardcoded in `.github/workflows/image.yml`
- CI image builds currently depend on `build_image.sh` rather than a default `Containerfile` build path

### Implementation Steps

#### 1. Add schema for deployment manifest

Create a new module, likely `src/mlwm/deployment_spec.py`, with dataclass-wizard-backed types for:

- `ManagedImage`
- `Schedule`
- `DeploymentJob`
- `DeploymentConfig`

Initial fields:

- `ManagedImage.published_as: str`
- `ManagedImage.build_secret_env: List[str]`
- `DeploymentJob.image: str`
- `DeploymentJob.schedule.cron: str`
- `DeploymentJob.schedule.timezone: str`
- `DeploymentJob.runtime_context_script: str`

Validation to add:

- each `images` key must reference an existing directory in `configurations/`
- each `images` key must contain a valid model configuration in `configurations/<model>/config.yaml`
- each managed image's `published_as` must be unique
- each `runtime_context_script` must exist
- each `jobs[*].image` must either match a managed image in `images` or already exist in the registry
- for managed images, the corresponding config directory should contain the expected build inputs such as `Containerfile`
- each `build_secret_env` value should be a valid environment variable name
- warn if a `configurations/<model>/` directory exists without a matching `images.<model>` entry
- warn if a managed image is not referenced by any job

#### 2. Add routines manifest file

Create `configurations/deployments.yaml` with the first managed image and the first routine-enabled job definition.

Initially include the existing `surface-dummy-model_DINI` example only.

#### 3. Add context script for the first model

Create `configurations/surface-dummy-model_DINI/build_runtime_context.sh`.

Responsibilities:

- accept a scheduled time argument
- derive `ANALYSIS_TIME`
- construct `DATASTORE_INPUT_PATHS`
- emit required runtime variables in `KEY=VALUE` format

This script should encapsulate logic currently embedded in `run_inference_container.sh` that is specific to DINI-driven execution.

#### 4. Define the default image build contract

Make `Containerfile` the standard build entrypoint for managed images.

The default path should be:

- local builds call `podman build` or `docker build` directly
- required build-time credentials are supplied by the caller environment
- CI builds call the same container build command directly
- CI provides required build-time credentials from GitHub Secrets
- CI only publishes managed images on merges to `main`

This removes `build_image.sh` from the default build path.

Host-specific wrappers may still exist later for exceptional environments, but they should not be required for standard CI or standard local builds.

#### 5. Extend `run_models.py` to load routine jobs

Refactor model discovery so that routine execution is driven by `configurations/deployments.yaml` instead of scanning every directory under `configurations/`.

Add functions to:

- load managed images and routine jobs from the manifest
- map managed images to model configs via the `images` keys
- map each managed image key to `configurations/<model>/config.yaml`
- resolve each job image either to a managed image in the manifest or to an already-published external image

This should replace the current assumption in `find_model_configurations()` that every config directory is enabled for execution.

#### 6. Add CI-facing image build resolution

Add a helper or small module that can enumerate which images CI should build from `configurations/deployments.yaml`.

This should eventually allow `.github/workflows/image.yml` to stop hardcoding a matrix of model directories and instead derive build targets from the manifest.

Responsibilities:

- identify all managed images listed under `images`
- resolve the internal model id to `configurations/<model>`
- expose the exact `published_as` image reference that CI should tag and push
- expose the `build_secret_env` entries that CI must source from GitHub Secrets
- validate that the managed image has the required build inputs

For managed images, CI should build directly from:

- build context: `configurations/<model>`
- containerfile: `configurations/<model>/Containerfile`

For local builds, the same image should be buildable by exporting the required variables listed in `build_secret_env` and then invoking `podman build` or `docker build` directly.

#### 7. Add runtime context loading from script

Add a helper in `run_models.py` or a small new module to:

- invoke the configured `runtime_context_script`
- pass standard arguments such as `--scheduled-time`
- parse stdout as strict `KEY=VALUE`
- return a dictionary of runtime env vars

Validation to add:

- required keys must be present, at minimum `ANALYSIS_TIME`
- reject malformed lines
- produce clear errors when the script fails

#### 8. Thread runtime context into execution flow

Update the model execution flow so that:

- `ANALYSIS_TIME` comes from the parsed runtime context
- other context values are available when launching the container
- existing input/output preparation still uses typed model config from `config.yaml`

This likely means splitting current logic into:

- repository policy loading
- model config loading
- runtime context loading
- container execution

#### 9. Implement container launch

Replace the `launch_docker_container()` stub in `run_models.py`.

Decisions to make during implementation:

- whether to support `docker` and `podman`, or only one initially
- how runtime env vars are passed into the container
- how volume mounts are assembled from prepared input/output directories

The minimum viable implementation should:

- launch the configured `jobs[*].image`
- mount the prepared directories
- inject runtime context as environment variables
- fail clearly on non-zero container exit

#### 10. Add tests

Add focused tests for:

- parsing `deployments.yaml`
- validation of missing model/config/script references
- validation that managed images point to valid `config.yaml` model configs
- validation that a job image either exists in `images` or is already available in the registry
- validation that `jobs[*].image` matches `images[*].published_as` when repo-managed
- validation of `build_secret_env`
- warning on configuration directories with no matching `images` entry
- warning on managed images not referenced by any job
- parsing `KEY=VALUE` output from context scripts
- failure behavior for malformed script output

Testing priority should be on pure Python parsing and validation logic first, not full container execution.

Registry existence checks should be structured so they can be mocked in unit tests.

#### 11. Update documentation

Update `README.md` to document:

- the purpose of `configurations/deployments.yaml`
- how `images` controls CI-managed build and publish targets
- that managed images are built directly from `configurations/<model>/Containerfile`
- how `build_secret_env` defines required build-time environment variables
- that managed images are only published from `main`
- how routine-enabled jobs are declared
- the required interface for `build_runtime_context.sh`
- how `ANALYSIS_TIME` is derived at execution time
- the rule that each job image must either already exist in the registry or be declared under `images`
- the warning behavior for configuration directories missing from `images` and for managed images unused by jobs

Also update any model-specific docs where the new context script replaces prior ad hoc execution logic.

### Suggested Order of Work

1. Add `deployment_spec.py`
2. Add `configurations/deployments.yaml`
3. Add `build_runtime_context.sh` for `surface-dummy-model_DINI`
4. Define and document the default `Containerfile` build contract
5. Refactor `run_models.py` to load managed images and jobs from the deployments manifest
6. Add CI-facing build target resolution from the manifest
7. Add context script execution and parsing
8. Implement container launch
9. Add tests
10. Update docs

### Initial Scope Boundary

To keep the first implementation small, do not add the following yet unless needed:

- secret management in the manifest
- multiple schedule types beyond cron
- fallback context sources besides `runtime_context_script`
- workflow-level orchestration inside Python
- backfill and ad hoc run semantics

The first version should make CI-managed images explicit, make routine jobs explicit, and make each job executable with runtime context returned by a script.
