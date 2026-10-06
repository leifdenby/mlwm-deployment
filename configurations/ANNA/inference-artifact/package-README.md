# ANNA inference package for the DANRA ML LAM model

This package contains everything needed to run the DANRA machine-learning
limited area model (LAM) of Adamov et al. (2025),
["Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings"](https://arxiv.org/abs/2504.09340),
as an operational forecast at DMI ("ANNA"). The model's own checkpoint is
not included:

- interior initial states come from DMI's **DINI** forecasts, regridded to the
  DANRA grid;
- boundary forcing comes from **DINI** or operational **IFS** forecasts, on
  the ERA5 boundary points the model was trained with.

**The model weights are not in this package.** They are the paper's published
DANRA checkpoint, Zenodo
[10.5281/zenodo.15131838](https://doi.org/10.5281/zenodo.15131838)
(`danra_model.ckpt`, CC-BY-4.0). Download them separately (URL and md5 in
`configs/model.yaml`) and put the file next to this README as
`danra_model.ckpt`.

The package's version, creation time and provenance are recorded in
`artifact.yaml`.

## Contents

| Path | What |
|---|---|
| `configs/model.yaml` | the model: checkpoint (Zenodo URL, md5), graph recipe (`7deg_rect_hi4`), neural-lam `train_model` arguments |
| `configs/danra_model1_config.yaml` | interior (DANRA) mllam-data-prep datastore config, as in training |
| `configs/era_7deg_model1_config.yaml` | ERA5 boundary datastore config the model was trained with: the reference for the boundary features and their normalisation |
| `configs/ifs_7deg_model1_config.yaml` | operational IFS boundary datastore config. Its header specifies the variables, units, lead times and grid an IFS boundary zarr must have |
| `configs/dini_7deg_model1_config.yaml` | DINI boundary datastore config (DINI regridded to the ERA5 boundary box, same layout as IFS) |
| `configs/7deg_config_{era5,ifs,dini}.yaml` | neural-lam configs. IFS and DINI boundaries are normalised with the ERA5 training statistics (`overload_stats_path`) |
| `configs/era_7deg_model1_config.zarr` | ERA5 boundary "stats datastore" that neural-lam opens through `overload_stats_path` (statistics and splits only) |
| `stats/danra_model1_config.stats.zarr` | training statistics of the DANRA interior datastore |
| `stats/era_7deg_model1_config.stats.zarr` | training statistics of the ERA5 boundary datastore, recomputed exactly from WeatherBench2 ERA5 (2000-01-01 to 2018-10-29, 6-hourly) |
| `grids/danra_model1_config.grid.zarr` | DANRA grid (x, y, lat, lon) and static fields (land-sea mask, orography) from DANRA v0.5.0, the target grid for DINI |
| `grids/era_7deg_model1_config.grid.zarr` | lat/lon of the 18,014 ERA5 (0.25°) boundary points the model was trained with, a 7.19° ring around the DANRA domain |
| `artifact.yaml` | version and provenance of this package |

## Using it

The package is made for the ANNA configuration of
[mlwm-deployment](https://github.com/leifdenby/mlwm-deployment)
(`configurations/ANNA`). Its container build downloads this package and the
checkpoint. Its `entry.sh` regrids DINI, creates the inference datastores
from these configs, builds the graph, runs the forecast and converts the
output.

Software as used in training:
- neural-lam from
  [joeloskarsson/neural-lam-dev](https://github.com/joeloskarsson/neural-lam-dev)
  (`research` branch);
- mllam-data-prep from `sadamov/mllam-data-prep` (`building-ml-lams` branch);
- zarr 2.

Exact pins are in `configurations/ANNA/pyproject.toml` of mlwm-deployment.

Conventions of the model's inputs and outputs (those of DANRA):

- **winds are relative to the DANRA (Lambert) grid**, not east/north;
- relative humidity `r` is a fraction (0–1);
- `tw` is geometric vertical velocity (m/s) in the interior. The boundary
  uses pressure vertical velocity ω (Pa/s), as in ERA5.

## Licence and acknowledgements

The package is licensed CC-BY-4.0. The model, its training setup and the
datastore configurations are the work of the authors of arXiv:2504.09340:

> Simon Adamov, Joel Oskarsson, Leif Denby, Tomas Landelius, Kasper Hintz,
> Simon Christiansen, Irene Schicker, Carlos Osuna, Fredrik Lindsten, Oliver
> Fuhrer and Sebastian Schemm

When you use the model, please cite the paper and the checkpoint record
(10.5281/zenodo.15131838).
