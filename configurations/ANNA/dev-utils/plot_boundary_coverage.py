"""
Map of the ANNA training boundary points coloured by whether they lie inside
the DINI domain, with the DINI and DANRA (interior) domain outlines.

Usage (from configurations/ANNA, cartopy downloads coastlines on first use):

    uv run --with matplotlib python dev-utils/plot_boundary_coverage.py \\
        src/regrid_dini.py \\
        inference_artifact/grids/era_7deg_model1_config.grid.zarr \\
        inference_artifact/grids/danra_model1_config.grid.zarr \\
        s3://harmonie-zarr/dini/control/2026-10-01T000000Z/ \\
        docs/anna_boundary_vs_dini.png
"""
import importlib.util
import sys

try:
    import truststore

    truststore.inject_into_ssl()  # cartopy downloads coastlines
except ImportError:
    pass

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

fp_rd, fp_bgrid, fp_dgrid, dini_root, fp_out = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("rd", fp_rd)
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

# palette: reference categorical slots 1 (blue) and 2 (orange), validated;
# text/outline ink from the reference text tokens
INSIDE, OUTSIDE = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

grid = xr.open_zarr(fp_bgrid).load()
lat = grid.latitude.values
lon = np.where(
    grid.longitude.values > 180,
    grid.longitude.values - 360,
    grid.longitude.values,
)

ds_sl, _ = rd._open_dini(dini_root)
transformer = rd._dini_transformer(ds_sl)
x, y = transformer.transform(lon, lat)
xs, ys = ds_sl.x.values, ds_sl.y.values
inside = (x >= xs[0]) & (x <= xs[-1]) & (y >= ys[0]) & (y <= ys[-1])

dini_lat, dini_lon = ds_sl.lat.values, ds_sl.lon.values
danra = xr.open_zarr(fp_dgrid).load()


def perimeter(lat2d, lon2d):
    lo = np.concatenate(
        [lon2d[0, :], lon2d[:, -1], lon2d[-1, ::-1], lon2d[::-1, 0]]
    )
    la = np.concatenate(
        [lat2d[0, :], lat2d[:, -1], lat2d[-1, ::-1], lat2d[::-1, 0]]
    )
    return np.where(lo > 180, lo - 360, lo), la


proj = ccrs.LambertConformal(
    central_longitude=8, central_latitude=56, standard_parallels=(56, 56)
)
fig = plt.figure(figsize=(10, 8.5), facecolor=SURFACE)
ax = plt.axes(projection=proj, facecolor=SURFACE)
ax.set_extent([-30, 45, 36, 74], crs=ccrs.PlateCarree())
ax.add_feature(
    cfeature.LAND.with_scale("50m"), facecolor="#efeeea", edgecolor="none"
)
ax.add_feature(
    cfeature.COASTLINE.with_scale("50m"), edgecolor="#a3a29c", linewidth=0.5
)
gl = ax.gridlines(
    draw_labels=True,
    x_inline=False,
    y_inline=False,
    linewidth=0.4,
    color="#d6d5d0",
)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = dict(color=INK_2, fontsize=8)

pc = ccrs.PlateCarree()
ax.scatter(
    lon[inside],
    lat[inside],
    s=1.2,
    color=INSIDE,
    transform=pc,
    linewidths=0,
    label=f"boundary point covered by DINI ({inside.sum():,})",
    rasterized=True,
)
ax.scatter(
    lon[~inside],
    lat[~inside],
    s=1.2,
    color=OUTSIDE,
    transform=pc,
    linewidths=0,
    label=f"boundary point outside DINI ({(~inside).sum():,}, "
    f"{100 * (~inside).mean():.0f}%)",
    rasterized=True,
)

lo, la = perimeter(dini_lat, dini_lon)
ax.plot(lo, la, color=INK, linewidth=2, transform=pc)
lo, la = perimeter(danra.lat.values, danra.lon.values)
ax.plot(lo, la, color=INK_2, linewidth=2, linestyle=(0, (4, 3)), transform=pc)

# direct labels on the outlines
ax.text(
    -24,
    44.0,
    "DINI domain",
    color=INK,
    fontsize=10,
    fontweight="bold",
    transform=pc,
)
ax.text(
    -1.5,
    58.2,
    "DANRA interior\n(ANNA's grid)",
    color=INK_2,
    fontsize=9,
    fontweight="bold",
    transform=pc,
    ha="left",
)

leg = ax.legend(
    loc="lower right",
    markerscale=8,
    frameon=True,
    fontsize=9,
    facecolor=SURFACE,
    edgecolor="#d6d5d0",
    labelcolor=INK,
)
ax.set_title(
    "ANNA training boundary points (ERA5 0.25°, 7.19° ring around DANRA)\n"
    "vs the DINI domain",
    color=INK,
    fontsize=12,
    loc="left",
)
fig.savefig(fp_out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {fp_out}: inside {inside.sum()}, outside {(~inside).sum()}")
