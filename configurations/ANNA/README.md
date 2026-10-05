# ANNA (gefion-1)

ANNA is a hierarchical graph-based limited-area model (neural-lam `hi_lam`,
graph `7deg_rect_hi3`) trained on Gefion on DANRA (interior, 2.5 km, 3-hourly),
with ERA5 boundary forcing on a 7.19° ring around the DANRA domain. Here it is
run operationally from **DINI** (interior initial states) with **DINI or IFS**
on the boundary.

The full history, the decisions and the reasons behind them are in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md).

## Provenance

- Training run: wandb [`jo-research-team/neural_lam/n0o7jw5f`](https://wandb.ai/jo-research-team/neural_lam/runs/n0o7jw5f)
  (`train-hi_lam-2x300-02_27_15-4034`, 80 epochs), checkpoint packaged as
  `s3://mlwm-artifacts/inference-artifacts/gefion-1.zip`
- Training code: [`joeloskarsson/neural-lam-dev`](https://github.com/joeloskarsson/neural-lam-dev)
  commit `e58e334c` (`research` branch), with torch 2.6, mllam-data-prep 0.5.0
  (sadamov fork), weather-model-graphs 0.2.0 and zarr 2.18
- Model arguments relevant at inference: `--model hi_lam --graph_name
  7deg_rect_hi3 --hidden_dim 300 --hidden_dim_grid 150 --time_delta_enc_dim 32
  --processor_layers 2 --dynamic_time_deltas`, with 1 past and 1 future
  forcing and boundary step (all set in `entry.sh`). The full training
  arguments are in `inference_artifact/training_cli_args.yaml`
- The paper's DANRA checkpoint ([Zenodo 15131838](https://zenodo.org/records/15131838))
  may be a later fine-tune of this model; it hasn't been compared yet

## Software

Pinned in [pyproject.toml](pyproject.toml):

- **neural-lam**: `joeloskarsson/neural-lam-dev@a44d432e`. It contains the
  training commit and forecast-format boundary forcing (needed for IFS); later
  commits break installing from git
- **mllam-data-prep**: `sadamov/mllam-data-prep@dd9af481`, exactly the version
  used for training. mllam-data-prep ≥ 0.7 doesn't keep per-point lat/lon for
  regular lat/lon grids, which neural-lam-dev needs for the boundary
- **zarr 2**, as in training; neural-lam-dev writes its output with a zarr v2
  compressor

## Configs

All in [configs/](configs/):

| File | What |
|---|---|
| `danra_model1_config.yaml` | interior (DANRA) datastore, as in training |
| `era_7deg_model1_config.yaml` | ERA5 boundary datastore ANNA was trained with; reference for the boundary features and their (ERA5) normalisation statistics |
| `ifs_7deg_model1_config.yaml` | operational IFS boundary. **Its header is the contract for the IFS GRIB → zarr conversion** (variables, units, dims, lead times, and the 0.25° box lat 40.0–72.0, lon −27.0–39.5; the east edge must be exactly 39.5°E) |
| `dini_7deg_model1_config.yaml` | DINI boundary, in the same layout as IFS |
| `7deg_config_{era5,ifs,dini}.yaml` | neural-lam configs; IFS and DINI are normalised with the ERA5 training statistics (`overload_stats_path`) |

## Inference artifact

`gefion-1.zip` is incomplete (no boundary datastore or statistics), and its
checkpoint can't be loaded as is. The complete artifact is assembled locally
in `inference_artifact/` (gitignored):

```bash
aws s3 cp s3://mlwm-artifacts/inference-artifacts/gefion-1.zip .
uv run --project configurations/ANNA \
    python configurations/ANNA/dev-utils/assemble_artifact.py \
    --gefion-1-zip gefion-1.zip \
    --boundary-stats era_7deg_model1_config.stats.zarr
```

This adds the sanitised checkpoint, the configs above, the ERA5 boundary
statistics, the DANRA grid and statics, and the 18014 training boundary
points. See the docstring of `dev-utils/assemble_artifact.py` for the layout.

The **ERA5 boundary statistics** were never exported from the training
datastore. They are recomputed from WeatherBench2 ERA5 with
`dev-utils/compute_era5_boundary_stats.py`, which reproduces the training
computation exactly. It's restartable and has a progress bar, but needs about
9 TB of reads. Until that has run, the artifact uses **placeholder**
statistics (3 days of ERA5, with exact values for the derived and static
features). Re-run `assemble_artifact.py` with the exact statistics when they
are available.

## Building the image

```bash
CONTAINER_APP=podman ./build_image.sh
```

By default (`ARTIFACT_SOURCE=local`) the image includes the assembled
`inference_artifact/`, and no AWS credentials are needed. `ARTIFACT_SOURCE=s3`
downloads the artifact from S3 instead (needs `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`),
but that only makes sense once a complete artifact has been uploaded. Set
`MLWM_PULL_PROXY` to pull the base image through DMI's proxy.

## Running a forecast

[entry.sh](entry.sh) runs the whole pipeline:

1. `src/regrid_dini.py`: DINI → DANRA grid (interior) and, for the DINI
   boundary, → the 0.25° ERA5 boundary box
2. `src/create_inference_dataset.py`: inference datastores and neural-lam
   config
3. `neural_lam.build_rectangular_graph`: the `7deg_rect_hi3` graph
4. `neural_lam.train_model --eval test`: the forecast
5. `src/convert_output.py`: DANRA-like output datasets

It's configured through environment variables:

| Variable | Default | |
|---|---|---|
| `ANALYSIS_TIME` | (required) | DINI analysis time, e.g. `2026-10-01T00:00Z` |
| `FORECAST_DURATION` | `PT18H` | multiple of 3 h, between 6 h and DINI forecast length − 6 h (30 h for a 36 h DINI run) |
| `BOUNDARY_SOURCE` | `dini` | `dini` or `ifs` |
| `DINI_ROOT` | `s3://harmonie-zarr/dini/control/{analysis}/` | DINI forecast zarrs |
| `IFS_BOUNDARY_PATH` | | IFS forecast zarr, required for `BOUNDARY_SOURCE=ifs` |
| `INFERENCE_WORKDIR` | `./inference_workdir` | |
| `INFERENCE_ARTIFACT_PATH` | `./inference_artifact` | |
| `MLWM_DEBUGGER` | | `ipdb` to debug the python steps on exceptions |

Outside the container, run it from `configurations/ANNA` in the uv
environment, with AWS credentials for reading DINI (e.g. `AWS_PROFILE`):

```bash
ANALYSIS_TIME=2026-10-01T00:00Z FORECAST_DURATION=PT18H ./entry.sh
```

In the container: `run_inference_container.sh` still has the interface from
before these changes, and is to be updated (step 7 in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md)).

### Outputs and conventions

`${INFERENCE_WORKDIR}/outputs/single_levels.zarr` (pres_seasurface, t2m, u10m,
v10m, pres0m, lwavr0m, swavr0m) and `pressure_levels.zarr` (z, t, r, u, v, tw
at 100–1000 hPa) are on the DANRA grid at valid times T+6 h …
T+`FORECAST_DURATION` in 3 h steps. The initial states are DINI at T+0 and
T+3 h. The conventions are DANRA's:

- **winds are relative to the DANRA (Lambert) grid**, not east/north
- relative humidity `r` is a fraction (0–1)
- `lwavr0m`/`swavr0m` are net surface radiation fluxes
- `tw` is geometric vertical velocity (m/s)

## Known limitations

- **DINI boundary.** 25% of ANNA's boundary points (4,573 of 18,014) lie
  outside the DINI domain, mostly to the east and north. They are filled with
  the value of the nearest DINI grid point. IFS covers the whole ring and is
  the preferred boundary source once the IFS zarr is available.

  ![ANNA boundary points vs the DINI domain](docs/anna_boundary_vs_dini.png)

- **Placeholder boundary statistics** until the exact ERA5 statistics have
  been computed (see above).
- **DINI zarr retention** is two weeks. Keep regridded development cases
  locally in `dev-data/` (gitignored).

## Development

- [dev-utils/](dev-utils/):
  - `assemble_artifact.py`: build `inference_artifact/`
  - `compute_era5_boundary_stats.py`: exact ERA5 boundary training statistics
  - `sanitize_checkpoint.py`: make the gefion-1 checkpoint loadable
  - `check_checkpoint_compat.py`: strict checkpoint load and one-step forecast
    on synthetic data
  - `plot_boundary_coverage.py`: the boundary coverage map above
- Tests:
  ```bash
  uv run --project configurations/ANNA --with pytest \
      python -m pytest configurations/ANNA/tests
  ```
- The `src/` scripts can also be run on their own; see their `--help` and
  docstrings.
