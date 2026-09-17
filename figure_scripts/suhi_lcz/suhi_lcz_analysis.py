#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUHI × LCZ Analysis — from lst-bolzano.nc
Combines the seasonal SUHI analysis with Local Climate Zone stratification.

Two LCZ inputs are used together:
  1. WUDAPT LCZ raster  (LCZ_cropped.tif, classes 1–17)
     → mean SUHI per LCZ class per (season, day/night) bucket
  2. Zone shapefiles    (city_core / urban_area / rural_area)
     → same zone masks as the parent SUHI script, drawn as boundaries on maps

Outputs (all saved to OUTPUT_DIR):
  Plot A  : SUHI maps (season x day/night) with LCZ raster underlay + zone boundaries
  Plot B  : Barplot — mean SUHI per LCZ class × (season, day/night) bucket
  Plot C  : KDE overlays — one curve per LCZ class (for each bucket)
  CSV     : Stats table (mean, median, p90, n_pixels) per LCZ class × bucket

ELEVATION / LAPSE-RATE CORRECTION (added 2026-08-03, flat term removed 2026-08-03):
previously this script applied only a flat, spatially-uniform ELEV_CORRECTION constant
(-2.275 degC everywhere) -- no actual per-pixel elevation correction at all. It now
applies a dynamic, per-(season, day/night) lapse-rate correction, matching the
methodology used across the WRF/UrbClim/Netatmo UHI scripts (seasonal_valley_lapse_rate.py,
fit purely from WRF, applied here to the LST/satellite data): DEM_10m_south_tirol.tif
is reprojected onto the LST grid to build a per-pixel elevation-diff field, corrected
using the rate for each scene's (season, astronomical day/night) bucket
(solar_daynight.classify_day_night). The old flat ELEV_CORRECTION constant was dropped
entirely -- it predated the per-pixel correction and was never anything other than a
crude flat proxy for the same elevation effect, so stacking it on top of the per-pixel
correction double-counted elevation (and it was applied even under LAPSE_MODE='none',
which is supposed to mean no elevation correction at all). The script also SPLITS every
season into day/night (previously season-only, all times pooled together), so all
outputs (Plot A/B/H, the stats CSV) carry roughly twice as many categories as before.
"""

#%% ── IMPORTS ─────────────────────────────────────────────────────────────────

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from scipy.stats import gaussian_kde
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import os, sys
import contextily as ctx
from shapely.geometry import box
from shapely.vectorized import contains

# Dynamic, per-(season, day/night) valley lapse rate + the same astronomical
# sunrise/sunset day/night classifier used across the WRF/UHI scripts -- see
# UHI/seasonal_valley_lapse_rate.py and UHI/solar_daynight.py.
# Replaces this script's old flat, spatially-uniform ELEV_CORRECTION-only approach
# with an actual per-pixel elevation correction, applied per (season, day/night) bucket.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT

#%% ── CONFIGURATION ──────────────────────────────────────────────────────────

INPUT_FILE   = "/home/gsimonet/Desktop/LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc"
LCZ_RASTER   = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/LCZ_clipped.tif"
DEM_RASTER   = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif"
URBAN_SHAPE  = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area_WGS84.shp"
RURAL_SHAPE  = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area.shp"
CITY_CORE_SHAPE = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_city_core.shp"
# Elevation-correction mode -- edit and re-run to get each version; see
# common/lapse_mode_utils.py. OUTPUT_DIR is tagged with this mode so the three
# don't overwrite each other's stats CSV / figures.
#   'none'     -- no elevation correction (raw urban-minus-rural difference)
#   'constant' -- classical fixed rate (LAPSE_RATE_CONSTANT), no season/day-night dependence (default)
#   'valley'   -- dynamic, per-(season, day/night) rate fitted from WRF -- WRF-only; this
#                 script has no WRF data at all (satellite/LST only), so 'valley' would be
#                 indistinguishable from 'constant' here. Kept at 'constant' for honest labeling.
LAPSE_MODE = 'constant'

OUTPUT_DIR   = f"suhi_lcz_analysis_{LAPSE_MODE}"

LST_VAR  = "__xarray_dataarray_variable__"
TIME_DIM = "time"
CRS      = "EPSG:32632"

MIN_LCZ_PIXELS  = 550       # discard LCZ classes with fewer pixels on the reprojected LST grid

# Season assignment by calendar month, matching the WRF valley-lapse-rate fit
# (seasonal_valley_lapse_rate.py) so bucket keys line up exactly.
SEASON_MAP_MONTH = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring',  4: 'Spring',  5: 'Spring',
                      6: 'Summer',  7: 'Summer',  8: 'Summer',
                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
PERIOD_ORDER = ['Day', 'Night']   # display case; classify_day_night() returns lowercase

# {(season, 'day'|'night'): K/m} -- fitted once from the WRF archive, applied here to
# the satellite/LST data exactly as to every other source (UrbClim, WRF, Netatmo).
DYNAMIC_LAPSE = load_seasonal_valley_lapse()

# Buckets iterated everywhere below: "All data" (mixed season/period) + one per
# (season, day/night) combo, ordered season-major to match SEASON_ORDER.
BUCKET_LABELS = ["All data"] + [f"{s} - {p}" for s in SEASON_ORDER for p in PERIOD_ORDER]

_SEASON_BASE_COLORS = {"Winter": "#1f77b4", "Spring": "#2ca02c",
                         "Summer": "#d62728", "Autumn": "#ff7f0e"}

def _shade(hex_color, factor):
    """Darken a hex color by `factor` (<1) -- used to give Night a darker shade
    of its season's Day color, so day/night pairs stay visually grouped by season."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)

BUCKET_COLORS = {"All data": "#7f7f7f"}
for _s, _base in _SEASON_BASE_COLORS.items():
    BUCKET_COLORS[f"{_s} - Day"]   = _base
    BUCKET_COLORS[f"{_s} - Night"] = _shade(_base, 0.55)

SUHI_VMIN      = -8
SUHI_VMAX      =  8
DIVERGING_CMAP = "RdBu_r"

# Standard WUDAPT LCZ definitions
LCZ_NAMES = {
    1: "Compact highrise",   2: "Compact midrise",   3: "Compact lowrise",
    4: "Open highrise",      5: "Open midrise",       6: "Open lowrise",
    7: "Lightweight lowrise", 8: "Large lowrise",     9: "Sparsely built",
    10: "Heavy industry",    11: "Dense trees",       12: "Scattered trees",
    13: "Bush/scrub",        14: "Low plants",        15: "Bare rock/paved",
    16: "Bare soil/sand",    17: "Water",
}

LCZ_COLORS = {
    1: "#8B0000", 2: "#CD0000", 3: "#FF0000",  4: "#FF6347",  5: "#FF8C00",
    6: "#FFD700", 7: "#FFFF99", 8: "#BEBEBE",  9: "#696969", 10: "#8B4513",
    11: "#006400",12: "#228B22",13: "#90EE90", 14: "#FFFF00",
    15: "#A9A9A9",16: "#F5DEB3",17: "#4169E1",
}

# (season, day/night) buckets to show in multi-panel plots (exclude "All data" to
# keep layout tight) -- 8 labels, ordered season-major: Winter Day/Night, Spring
# Day/Night, Summer Day/Night, Autumn Day/Night.
SEASON_LABELS = [f"{s} - {p}" for s in SEASON_ORDER for p in PERIOD_ORDER]
SEASON_COLORS = BUCKET_COLORS   # alias kept so downstream code reads unchanged

os.makedirs(OUTPUT_DIR, exist_ok=True)


#%% ── LOAD LST DATA ───────────────────────────────────────────────────────────

print("Loading LST dataset…")
ds   = xr.open_dataset(INPUT_FILE)
lst  = ds[LST_VAR]
times = pd.DatetimeIndex(ds[TIME_DIM].values)
print(f"  Shape : {lst.shape}")
print(f"  Period: {str(times[0])[:10]} → {str(times[-1])[:10]}")

x_coords = ds["x"].values
y_coords = ds["y"].values
xx, yy = np.meshgrid(x_coords, y_coords)

ny, nx = xx.shape


#%% ── SPATIAL MASKS FROM SHAPEFILES ─────────────────────────────────────────

print("\nBuilding shapefile masks…")
urban_gdf     = gpd.read_file(URBAN_SHAPE).to_crs(CRS)
rural_gdf     = gpd.read_file(RURAL_SHAPE).to_crs(CRS)
city_core_gdf = gpd.read_file(CITY_CORE_SHAPE).to_crs(CRS)

urban_union     = urban_gdf.unary_union
rural_union     = rural_gdf.unary_union
city_core_union = city_core_gdf.unary_union

urban_mask     = contains(urban_union,     xx.ravel(), yy.ravel()).reshape(xx.shape)
rural_mask     = contains(rural_union,     xx.ravel(), yy.ravel()).reshape(xx.shape)
city_core_mask = contains(city_core_union, xx.ravel(), yy.ravel()).reshape(xx.shape)

print(f"  Urban pixels    : {urban_mask.sum():,}")
print(f"  Rural pixels    : {rural_mask.sum():,}")
print(f"  City-core pixels: {city_core_mask.sum():,}")


#%% ── LOAD & REPROJECT LCZ RASTER TO LST GRID ───────────────────────────────

print("\nLoading and reprojecting LCZ raster to LST grid…")

# Build a rasterio-compatible transform for the LST grid
res_x = float(x_coords[1] - x_coords[0])
res_y = float(y_coords[1] - y_coords[0])   # negative (top→bottom)
from rasterio.transform import from_origin
lst_transform = from_origin(
    float(x_coords[0]) - res_x / 2,
    float(y_coords[0]) - res_y / 2,   # y_coords[0] is the topmost row
    abs(res_x), abs(res_y),
)

with rasterio.open(LCZ_RASTER) as src:
    lcz_src_crs   = src.crs
    lcz_src_data  = src.read(1).astype(np.float32)
    lcz_nodata    = src.nodata if src.nodata is not None else 0
    lcz_transform_src = src.transform

# Reproject to the LST grid (UTM 32N, 30 m)
import rasterio.crs as rcrs
lcz_reprojected = np.zeros((ny, nx), dtype=np.float32)
reproject(
    source=lcz_src_data,
    destination=lcz_reprojected,
    src_transform=lcz_transform_src,
    src_crs=lcz_src_crs,
    dst_transform=lst_transform,
    dst_crs=rcrs.CRS.from_epsg(32632),
    resampling=Resampling.nearest,
    src_nodata=lcz_nodata,
    dst_nodata=0,
)

lcz_grid = lcz_reprojected.astype(np.int16)
present_lcz = sorted([v for v in np.unique(lcz_grid)
                       if v in LCZ_NAMES and int(np.sum(lcz_grid == v)) >= MIN_LCZ_PIXELS])
print(f"  LCZ classes present in domain (≥{MIN_LCZ_PIXELS} px): {present_lcz}")
present_lcz_urban = sorted([v for v in np.unique(lcz_grid[urban_mask])
                             if v in LCZ_NAMES
                             and int(np.sum((lcz_grid == v) & urban_mask)) >= MIN_LCZ_PIXELS])
print(f"  LCZ classes in urban area    (≥{MIN_LCZ_PIXELS} px): {present_lcz_urban}")


#%% ── LOAD & REPROJECT DEM TO LST GRID (for the elevation/lapse correction) ────

print("\nLoading and reprojecting DEM raster to LST grid…")
with rasterio.open(DEM_RASTER) as src:
    dem_src_crs      = src.crs
    dem_src_data     = src.read(1).astype(np.float32)
    dem_nodata       = src.nodata
    dem_transform_src = src.transform

dem_reprojected = np.full((ny, nx), np.nan, dtype=np.float32)
reproject(
    source=dem_src_data,
    destination=dem_reprojected,
    src_transform=dem_transform_src,
    src_crs=dem_src_crs,
    dst_transform=lst_transform,
    dst_crs=rcrs.CRS.from_epsg(32632),
    resampling=Resampling.bilinear,   # continuous elevation, unlike the categorical LCZ raster
    src_nodata=dem_nodata,
    dst_nodata=np.nan,
)
dem_grid = dem_reprojected.astype(np.float64)

rural_elev_mean = float(np.nanmean(dem_grid[rural_mask]))
ediff2d = dem_grid - rural_elev_mean   # NOT multiplied by a rate here -- see mean_dynamic_rate()
print(f"  DEM reprojected: rural mean elev = {rural_elev_mean:.0f} m, "
      f"grid range [{np.nanmin(dem_grid):.0f}, {np.nanmax(dem_grid):.0f}] m")


#%% ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

# Per-scene season/day-night classification, computed once for the full time axis.
seasons_all = np.array([SEASON_MAP_MONTH[m] for m in times.month])
periods_all = classify_day_night(times)   # 'day'/'night', from that calendar day's real sunrise/sunset


def select_bucket(season=None, period=None):
    """Boolean index array into `times` for a given season name and/or 'day'/'night'."""
    mask = np.ones(len(times), dtype=bool)
    if season is not None:
        mask &= (seasons_all == season)
    if period is not None:
        mask &= (periods_all == period)
    return np.where(mask)[0]


def mean_dynamic_rate(idx):
    """Mean DYNAMIC_LAPSE rate (K/m) over a set of scene indices.

    Because the elevation-correction FIELD (ediff2d) doesn't vary with time, using
    this mean rate to build one correction field for a time-averaged map is exactly
    equivalent to applying each scene's own rate before averaging -- the two are
    separable for a rate(t)*ediff(px) term. Only matters for the mixed "All data"
    bucket; single (season, period) buckets already share one rate for every scene.
    """
    if len(idx) == 0:
        return 0.0
    if LAPSE_MODE == 'none':
        return 0.0
    if LAPSE_MODE == 'constant':
        return LAPSE_RATE_CONSTANT
    rates = [DYNAMIC_LAPSE.get((seasons_all[i], periods_all[i])) for i in idx]
    rates = [r for r in rates if r is not None]
    return float(np.mean(rates)) if rates else 0.0


def compute_suhi(lst, times_idx, urban_mask, rural_mask, corr_field=None):
    """SUHI(x,y) = (LST(x,y) + corr_field(x,y)) - RAW rural mean.

    Reference stays the raw rural mean -- corr_field (the dynamic, per-pixel lapse
    correction) is applied to the WHOLE field (urban included), not to the rural
    pixels feeding the reference, so it can't cancel to zero once re-averaged over
    that same rural mask.
    """
    subset = lst.isel({TIME_DIM: times_idx})
    arr    = subset.values                        # (t, y, x)

    rural_vals = arr[:, rural_mask]
    rural_mean = np.nanmean(rural_vals, axis=1)    # RAW, uncorrected

    corrected = arr + corr_field[np.newaxis, :, :] if corr_field is not None else arr
    suhi = corrected - rural_mean[:, np.newaxis, np.newaxis]

    urban_3d  = np.broadcast_to(urban_mask, suhi.shape)
    suhi_urban = np.where(urban_3d, suhi, np.nan)

    avg_map  = np.nanmean(suhi_urban, axis=0)
    n_scenes = np.sum(~np.all(np.isnan(suhi_urban.reshape(suhi.shape[0], -1)), axis=1))
    return avg_map, n_scenes


def lcz_stats(avg_map, lcz_grid, present_lcz):
    rows = []
    for cls in present_lcz:
        mask = lcz_grid == cls
        vals = avg_map[mask & ~np.isnan(avg_map)]
        if len(vals) < 5:
            continue
        rows.append({
            "lcz_class"  : cls,
            "lcz_name"   : LCZ_NAMES[cls],
            "n_pixels"   : len(vals),
            "mean"       : np.mean(vals),
            "median"     : np.median(vals),
            "std"        : np.std(vals),
            "p10"        : np.percentile(vals, 10),
            "p90"        : np.percentile(vals, 90),
            "pct_pos"    : 100 * np.sum(vals > 0) / len(vals),
        })
    return pd.DataFrame(rows)


#%% ── COMPUTE SUHI PER (SEASON, DAY/NIGHT) BUCKET ───────────────────────────

print(f"\nComputing SUHI per (season, day/night) bucket "
      f"(elevation correction: LAPSE_MODE={LAPSE_MODE!r})…")

results = {}

# "All data" -- mixed season/period; its correction field uses the mean per-scene
# dynamic rate across everything selected (see mean_dynamic_rate() docstring).
idx_all = np.arange(len(times))
rate_all = mean_dynamic_rate(idx_all)
avg_map, n = compute_suhi(lst, idx_all, urban_mask, rural_mask, rate_all * ediff2d)
results["All data"] = dict(avg_map=avg_map, n_scenes=n, times=times[idx_all])
results["All data"]["lcz_stats"] = lcz_stats(avg_map, lcz_grid, present_lcz)
mean_s = np.nanmean(avg_map[urban_mask])
print(f"  [{'All data':18s}]  n={n:3d} scenes  mean_urban={mean_s:+.2f}°C  "
      f"mean_lapse={rate_all*1000:.2f} K/km  lcz_classes={len(results['All data']['lcz_stats'])}")

for season in SEASON_ORDER:
    for period in PERIOD_ORDER:
        label = f"{season} - {period}"
        idx   = select_bucket(season=season, period=period.lower())
        if len(idx) == 0:
            print(f"  [{label}] — no data, skipping.")
            continue
        rate = resolve_rate(LAPSE_MODE, DYNAMIC_LAPSE.get((season, period.lower())))
        avg_map, n = compute_suhi(lst, idx, urban_mask, rural_mask, rate * ediff2d)
        results[label] = dict(avg_map=avg_map, n_scenes=n, times=times[idx])
        stats_df = lcz_stats(avg_map, lcz_grid, present_lcz)
        results[label]["lcz_stats"] = stats_df
        mean_s = np.nanmean(avg_map[urban_mask])
        print(f"  [{label:18s}]  n={n:3d} scenes  mean_urban={mean_s:+.2f}°C  "
              f"lapse={rate*1000:.2f} K/km  lcz_classes={len(stats_df)}")

print("Done.")


#%% ── WEB MERCATOR EXTENT FOR BASEMAP ────────────────────────────────────────

norm = TwoSlopeNorm(vmin=SUHI_VMIN, vcenter=0, vmax=SUHI_VMAX)
cmap_suhi = mpl.colormaps[DIVERGING_CMAP]

x_min, x_max = float(ds["x"].min()), float(ds["x"].max())
y_min, y_max = float(ds["y"].min()), float(ds["y"].max())
bbox_gdf = gpd.GeoDataFrame(
    geometry=[box(x_min, y_min, x_max, y_max)], crs=CRS
).to_crs("EPSG:3857")
bx = bbox_gdf.total_bounds   # (minx, miny, maxx, maxy) in Web Mercator

# Zone boundary GDFs in Web Mercator for overlay
urban_wm     = urban_gdf.to_crs("EPSG:3857")
rural_wm     = rural_gdf.to_crs("EPSG:3857")
city_core_wm = city_core_gdf.to_crs("EPSG:3857")

# LCZ raster in Web Mercator for the basemap panel
# (we reproject once for display purposes only)
with rasterio.open(LCZ_RASTER) as src:
    dst_transform_wm, dst_width, dst_height = calculate_default_transform(
        src.crs, "EPSG:3857", src.width, src.height, *src.bounds
    )
    lcz_wm = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=src.read(1).astype(np.float32),
        destination=lcz_wm,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform_wm,
        dst_crs=rasterio.crs.CRS.from_epsg(3857),
        resampling=Resampling.nearest,
        src_nodata=src.nodata if src.nodata is not None else 0,
        dst_nodata=0,
    )

lcz_bounds_wm = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform_wm)
lcz_extent_wm = [lcz_bounds_wm[0], lcz_bounds_wm[2],
                  lcz_bounds_wm[1], lcz_bounds_wm[3]]

# Build a discrete colormap for the LCZ classes that are present
lcz_vals_sorted = sorted(LCZ_COLORS.keys())
lcz_cmap_colors = [LCZ_COLORS[v] for v in lcz_vals_sorted]
lcz_listed_cmap = ListedColormap(lcz_cmap_colors, name="lcz")
lcz_boundaries  = [v - 0.5 for v in lcz_vals_sorted] + [lcz_vals_sorted[-1] + 0.5]
lcz_bnorm       = BoundaryNorm(lcz_boundaries, lcz_listed_cmap.N)


#%% ── PLOT 0 — LCZ MAP (NO SUHI OVERLAY) ────────────────────────────────────
# Clean reference map showing only the LCZ raster + urban boundary + legend.

print("\nPlot 0: LCZ reference map…")

fig_lcz, ax_lcz = plt.subplots(figsize=(7, 7))

ax_lcz.set_xlim(bx[0], bx[2])
ax_lcz.set_ylim(bx[1], bx[3])
ctx.add_basemap(
    ax_lcz, crs="EPSG:3857",
    source="https://tile.opentopomap.org/{z}/{x}/{y}.png",
    zoom="auto",
    attribution="© OpenTopoMap (CC-BY-SA)",
    attribution_size=6, zorder=1,
)

lcz_disp0 = lcz_wm.astype(float)
lcz_disp0[lcz_disp0 == 0] = np.nan
ax_lcz.imshow(
    lcz_disp0,
    cmap=lcz_listed_cmap, norm=lcz_bnorm,
    extent=lcz_extent_wm,
    origin="upper", alpha=0.65, zorder=2,
)

urban_wm.boundary.plot(ax=ax_lcz, color="black", linewidth=1.1, zorder=3)

ax_lcz.set_xticks([]); ax_lcz.set_yticks([])
ax_lcz.set_title("Local Climate Zones — Bolzano domain", fontsize=11, fontweight="bold")

lcz_legend_handles = [
    mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.1,
                   label="Urban area boundary"),
]
for cls in present_lcz:
    lcz_legend_handles.append(
        mpatches.Patch(facecolor=LCZ_COLORS.get(cls, "#888888"), edgecolor="none",
                       alpha=0.85, label=f"LCZ {cls} — {LCZ_NAMES.get(cls, '?')}")
    )

ax_lcz.legend(
    handles=lcz_legend_handles,
    loc="lower left", bbox_to_anchor=(0.01, 0.01),
    fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
    handlelength=1.2, handletextpad=0.5, labelspacing=0.45,
)

out_lcz = os.path.join(OUTPUT_DIR, "lcz_reference_map.png")
fig_lcz.savefig(out_lcz, dpi=300, bbox_inches="tight", facecolor="white")
fig_lcz.savefig(out_lcz.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
print(f"  Saved → {out_lcz}")
plt.show()


#%% ── PLOT A — SEASONAL SUHI MAPS WITH LCZ RASTER UNDERLAY ──────────────────
# Layout per panel column-pair:
#   col 0-1  :  [map]  [right KDE sidebar]     ← overall SUHI distribution
#   col 0-1  (top strip)  : [LCZ KDE strip spanning map+kde width]
#
# Full GridSpec rows:
#   0  : LCZ KDE strips row 1     (short, above map)
#   1  : map panels row 1         (tall)
#   2  : colorbar                 (very short)
#   3  : LCZ KDE strips row 2     (short, above map)
#   4  : map panels row 2         (tall)

print("\nPlot A: seasonal SUHI maps with LCZ underlay…")

ncols      = 2   # Day | Night
col_ratios = [1, 0.18] * ncols       # map | right-KDE | map | right-KDE

# row order: for each of the 4 seasons -- lcz-strip | map -- then one shared
# colorbar row at the very end (was a 2-season/2x2 layout; now 4 seasons x
# 2 periods, so rows grow from 2 to 4 season-rows instead of adding columns).
n_season_rows = len(SEASON_ORDER)
row_ratios = []
_MAP_ROWS, _DIST_ROWS = [], []
for _ in range(n_season_rows):
    row_ratios.append(0.28); _DIST_ROWS.append(len(row_ratios) - 1)
    row_ratios.append(1.0);  _MAP_ROWS.append(len(row_ratios) - 1)
_CBAR_ROW = len(row_ratios)
row_ratios.append(0.06)

fig_a = plt.figure(figsize=(5.5 * ncols + 0.8, 3.6 * n_season_rows + 1.0))
gs_a  = gridspec.GridSpec(
    len(row_ratios), len(col_ratios),
    figure=fig_a,
    width_ratios=col_ratios,
    height_ratios=row_ratios,
    wspace=0.04, hspace=0.10,
)

im_ref = None
all_plot_cls  = set()   # union of LCZ classes plotted across all panels → used for legend
_map_kde_pairs = []     # (ax_map, ax_kde) per panel → used to match kde height to map
_panel_letters = "abcdefgh"
_panel_idx = 0
for si, season in enumerate(SEASON_ORDER):
    for pi, period in enumerate(PERIOD_ORDER):
        label = f"{season} - {period}"
        if label not in results:
            continue
        gs_map_row  = _MAP_ROWS[si]
        gs_dist_row = _DIST_ROWS[si]
        gs_col      = pi * 2

        ax_map  = fig_a.add_subplot(gs_a[gs_map_row,  gs_col])
        ax_kde  = fig_a.add_subplot(gs_a[gs_map_row,  gs_col + 1])
        ax_dist = fig_a.add_subplot(gs_a[gs_dist_row, gs_col])   # top strip, map width only
        _map_kde_pairs.append((ax_map, ax_kde))

        res = results[label]

        # ── basemap ──────────────────────────────────────────────────────
        ax_map.set_xlim(bx[0], bx[2])
        ax_map.set_ylim(bx[1], bx[3])
        ctx.add_basemap(
            ax_map, crs="EPSG:3857",
            source="https://tile.opentopomap.org/{z}/{x}/{y}.png",
            zoom="auto",
            attribution="© OpenTopoMap (CC-BY-SA)",
            attribution_size=5, zorder=1,
        )

        # ── LCZ raster underlay ─────────────────────────────────────────────
        lcz_disp = lcz_wm.astype(float)
        lcz_disp[lcz_disp == 0] = np.nan
        ax_map.imshow(
            lcz_disp,
            cmap=lcz_listed_cmap, norm=lcz_bnorm,
            extent=lcz_extent_wm,
            origin="upper", alpha=0.35, zorder=2,
        )

        # ── SUHI overlay ─────────────────────────────────────────────────────
        im_ref = ax_map.imshow(
            res["avg_map"], cmap=DIVERGING_CMAP, norm=norm,
            extent=[bx[0], bx[2], bx[1], bx[3]],
            origin="upper", alpha=0.65, zorder=3,
        )

        # ── zone boundaries ──────────────────────────────────────────────────
        urban_wm.boundary.plot(ax=ax_map, color="black", linewidth=0.9, zorder=4)

        ax_map.set_xlabel(f"{label}  (n={res['n_scenes']})", fontsize=9, fontweight="bold", labelpad=4)
        ax_map.set_xticks([]); ax_map.set_yticks([])
        ax_map.text(0.02, 0.98, _panel_letters[_panel_idx], transform=ax_map.transAxes,
                    fontsize=10, fontweight="bold", va="top", ha="left", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                              edgecolor="black", linewidth=0.7, alpha=0.9))
        _panel_idx += 1

        # ── RIGHT KDE sidebar — overall SUHI distribution ───────────────────
        vals = res["avg_map"][~np.isnan(res["avg_map"])].ravel()
        if len(vals) >= 10:
            v_min = float(np.percentile(vals, 1))  - 0.5
            v_max = float(np.percentile(vals, 99)) + 0.5
            x_k   = np.linspace(v_min, v_max, 400)
            kde   = gaussian_kde(vals, bw_method=0.4)
            y_k   = kde(x_k); y_k /= y_k.max()

            for j in range(len(x_k) - 1):
                c = cmap_suhi(norm(0.5 * (x_k[j] + x_k[j + 1])))
                ax_kde.fill_betweenx(
                    [x_k[j], x_k[j + 1]], 0, [y_k[j], y_k[j + 1]],
                    color=c, alpha=0.85, linewidth=0,
                )
            ax_kde.plot(y_k, x_k, color="white", lw=1.2)
            ax_kde.plot(y_k, x_k, color="#333333", lw=0.6)
            med = float(np.median(vals))
            ax_kde.axhline(med, color="black", lw=1.0, alpha=0.8)
            ax_kde.text(0.05, med, f"{med:+.1f}°C",
                        transform=ax_kde.get_yaxis_transform(),
                        ha="left", va="bottom", fontsize=6.5)
            ax_kde.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
            ax_kde.set_ylim(v_min, v_max)

        ax_kde.set_facecolor("white")
        ax_kde.set_xlim(-0.55, 1.15)
        ax_kde.set_xticks([])
        ax_kde.yaxis.tick_right()
        ax_kde.tick_params(axis="y", labelsize=6.5)
        for sp in ["top", "bottom", "left"]:
            ax_kde.spines[sp].set_visible(False)
        ax_kde.spines["right"].set_color("#aaaaaa")

        # ── BOTTOM LCZ KDE STRIP — one curve per LCZ class (urban area only) ─
        avg_map  = res["avg_map"]
        plot_cls = [c for c in present_lcz_urban
                    if np.sum(urban_mask & (lcz_grid == c) & ~np.isnan(avg_map)) >= 10]

        if plot_cls:
            all_plot_cls.update(plot_cls)
            all_v   = np.concatenate([avg_map[urban_mask & (lcz_grid == c) & ~np.isnan(avg_map)]
                                       for c in plot_cls])
            x_range = np.linspace(all_v.min() - 0.5, all_v.max() + 0.5, 400)

            for cls in plot_cls:
                v = avg_map[urban_mask & (lcz_grid == cls) & ~np.isnan(avg_map)]
                if len(v) < 10:
                    continue
                yk  = gaussian_kde(v, bw_method=0.5)(x_range)
                yk /= yk.max()
                col = LCZ_COLORS.get(cls, "#888888")
                ax_dist.plot(x_range, yk, lw=1.4, color=col)
                ax_dist.fill_between(x_range, yk, alpha=0.12, color=col)

            ax_dist.axvline(0, color="black", lw=0.8, ls="--", alpha=0.45)
            ax_dist.set_xlim(x_range[0], x_range[-1])
            ax_dist.set_ylim(0, 1.35)
            ax_dist.set_yticks([])
            ax_dist.set_xlabel("SUHI (°C)", fontsize=7, labelpad=2)
            ax_dist.tick_params(axis="x", labelsize=6.5)
            ax_dist.spines[["top", "bottom", "right", "left"]].set_visible(False)
            ax_dist.set_facecolor("white")

# ── match KDE sidebar height to the actual map content area ──────────────
# contextily forces equal aspect on ax_map, which shrinks the map within
# its subplot box; we resize ax_kde to match the resulting map extent.
fig_a.canvas.draw()
for _am, _ak in _map_kde_pairs:
    _trans = _am.transData + fig_a.transFigure.inverted()
    _y0 = float(_trans.transform((_am.get_xlim()[0], _am.get_ylim()[0]))[1])
    _y1 = float(_trans.transform((_am.get_xlim()[0], _am.get_ylim()[1]))[1])
    _q  = _ak.get_position()
    _ak.set_position([_q.x0, _y0, _q.width, _y1 - _y0])

# ── colorbar strip (row 2) ────────────────────────────────────────────────
ax_cbar = fig_a.add_subplot(gs_a[_CBAR_ROW, :])
cb = fig_a.colorbar(im_ref, cax=ax_cbar, orientation="horizontal",
                    norm=norm, extend="both")
cb.set_label("SUHI Intensity (°C)", fontsize=9, labelpad=4)
cb.ax.tick_params(labelsize=8)
cb.outline.set_edgecolor("#aaaaaa"); cb.outline.set_linewidth(0.6)

# ── legend: urban boundary + only the LCZ classes actually plotted ────────
legend_handles = [
    mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.2,
                   label="Urban area boundary"),
]
for cls in sorted(all_plot_cls):
    legend_handles.append(
        mpatches.Patch(facecolor=LCZ_COLORS.get(cls, "#888888"), edgecolor="none",
                       alpha=0.8, label=f"LCZ {cls} — {LCZ_NAMES.get(cls, '?')}")
    )
fig_a.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.08),
    fontsize=7.5, framealpha=0.9,
    ncol=min(len(legend_handles), 6),
    handlelength=1.2, handletextpad=0.5, columnspacing=1.0,
)

out_a = os.path.join(OUTPUT_DIR, "suhi_lcz_maps_seasonal.png")
fig_a.savefig(out_a, dpi=300, bbox_inches="tight", facecolor="white")
fig_a.savefig(out_a.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
print(f"  Saved → {out_a}")
plt.show()


#%% ── PLOT B — BARPLOT: MEAN SUHI PER LCZ CLASS × SEASON ────────────────────

print("\nPlot B: mean SUHI per LCZ class × season…")

# Collect a tidy DataFrame
rows_bar = []
for label in [l for l in SEASON_LABELS if l in results]:
    df = results[label]["lcz_stats"].copy()
    df["season"] = label
    rows_bar.append(df)
df_bar = pd.concat(rows_bar, ignore_index=True)

# Keep only LCZ classes that appear in every season for a clean comparison
common_cls = (
    df_bar.groupby("lcz_class")["season"].nunique()
    == len([l for l in SEASON_LABELS if l in results])
)
common_cls = common_cls[common_cls].index.tolist()
df_bar_plot = df_bar[df_bar["lcz_class"].isin(common_cls)].copy()
df_bar_plot["label"] = df_bar_plot["lcz_class"].map(
    lambda c: f"LCZ{c}\n{LCZ_NAMES.get(c,'?')[:12]}"
)

n_cls = len(common_cls)
n_sea = len([l for l in SEASON_LABELS if l in results])
bar_w = 0.8 / n_sea
x_pos = np.arange(n_cls)

fig_b, ax_b = plt.subplots(figsize=(max(10, n_cls * 1.4), 6))

for k, label in enumerate([l for l in SEASON_LABELS if l in results]):
    sub = df_bar_plot[df_bar_plot["season"] == label].set_index("lcz_class")
    means  = [sub.loc[c, "mean"]  if c in sub.index else np.nan for c in common_cls]
    stds   = [sub.loc[c, "std"]   if c in sub.index else np.nan for c in common_cls]
    color  = SEASON_COLORS.get(label, f"C{k}")
    offset = (k - (n_sea - 1) / 2) * bar_w
    ax_b.bar(x_pos + offset, means, bar_w * 0.9,
             yerr=stds, capsize=3,
             color=color, alpha=0.85, label=label,
             error_kw=dict(elinewidth=0.8, ecolor="#333333"))

ax_b.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(
    [f"LCZ{c}\n{LCZ_NAMES.get(c,'?')[:14]}" for c in common_cls],
    fontsize=8,
)
ax_b.set_ylabel("Mean SUHI Intensity (°C)", fontsize=11)
ax_b.set_title("Mean SUHI per LCZ Class by Season — Bolzano", fontsize=12, fontweight="bold")
ax_b.legend(fontsize=9, frameon=True)
ax_b.grid(axis="y", alpha=0.3)
plt.tight_layout()

out_b = os.path.join(OUTPUT_DIR, "suhi_lcz_barplot.png")
fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
fig_b.savefig(out_b.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
print(f"  Saved → {out_b}")
plt.show()


#%% ── STATS TABLE ─────────────────────────────────────────────────────────────

print("\nBuilding stats table…")

all_rows = []
for label in results:
    df = results[label]["lcz_stats"].copy()
    df.insert(0, "period", label)
    all_rows.append(df)

df_stats = pd.concat(all_rows, ignore_index=True)
df_stats = df_stats.round(3)

csv_out = os.path.join(OUTPUT_DIR, "suhi_lcz_stats.csv")
df_stats.to_csv(csv_out, index=False)
print(f"  Saved → {csv_out}")

print("\nSummary (mean SUHI per LCZ × season):")
pivot = df_stats.pivot_table(
    index=["lcz_class", "lcz_name"],
    columns="period",
    values="mean",
).round(2)
print(pivot.to_string())

#%% ── PLOT H — LCZ URBAN AREA + MEAN SUHI BY SEASON ─────────────────────────
# Left  : urban area (km²) per LCZ class, sorted by summer mean SUHI
# Right : grouped horizontal bars — mean SUHI ± std per LCZ class × season
# Reading both panels together shows which classes are warm AND spatially large.

print("\nPlot H: LCZ urban area + mean SUHI by season…")

PIXEL_AREA_KM2 = (30 * 30) / 1e6
seasons_h = [l for l in SEASON_LABELS if l in results]

common_cls_h = sorted(set.intersection(*[
    {int(c) for c in results[l]["lcz_stats"]["lcz_class"]}
    for l in seasons_h if not results[l]["lcz_stats"].empty
]))

# Sort rows by summer (day) mean SUHI descending (warmest class on top)
ref_season_h = next((l for l in ["Summer - Day", "Summer - Night"] + seasons_h if l in results),
                     seasons_h[0])
ref_sdf_h    = results[ref_season_h]["lcz_stats"].set_index("lcz_class")
common_cls_h = sorted(
    common_cls_h,
    key=lambda c: ref_sdf_h.loc[c, "mean"] if c in ref_sdf_h.index else 0,
    reverse=True,
)

n_cls_h = len(common_cls_h)
n_sea_h = len(seasons_h)
y_pos   = np.arange(n_cls_h)
bar_h   = 0.7 / n_sea_h

fig_h, (ax_area, ax_suhi) = plt.subplots(
    1, 2,
    figsize=(13, max(4, n_cls_h * 0.6 + 2.0)),
    gridspec_kw={"width_ratios": [1, 2.5]},
)

# ── left: urban area per LCZ class ───────────────────────────────────────────
ref_lbl_h  = "All data" if "All data" in results else seasons_h[0]
ref_sdf2_h = results[ref_lbl_h]["lcz_stats"].set_index("lcz_class")

areas_h  = [ref_sdf2_h.loc[c, "n_pixels"] * PIXEL_AREA_KM2
            if c in ref_sdf2_h.index else 0.0
            for c in common_cls_h]
b_cols_h = [LCZ_COLORS.get(c, "#888888") for c in common_cls_h]

ax_area.barh(y_pos, areas_h, height=0.65, color=b_cols_h, alpha=0.85, zorder=2)
x_max_area = max(areas_h) if areas_h else 1
for yi, a in enumerate(areas_h):
    if a > 0:
        ax_area.text(a + x_max_area * 0.02, yi, f"{a:.2f}",
                     va="center", ha="left", fontsize=7, color="#333333")
ax_area.set_yticks(y_pos)
ax_area.set_yticklabels(
    [f"LCZ {c} — {LCZ_NAMES.get(c, '?')}" for c in common_cls_h],
    fontsize=8,
)
ax_area.set_xlabel("Urban area (km²)", fontsize=9)
ax_area.set_title("Urban area per LCZ class", fontsize=10, fontweight="bold")
ax_area.set_xlim(0, x_max_area * 1.3)
ax_area.spines[["top", "right"]].set_visible(False)
ax_area.grid(axis="x", alpha=0.25, linewidth=0.5)

# ── right: mean SUHI ± std per LCZ class × season ────────────────────────────
for si, label in enumerate(seasons_h):
    sdf    = results[label]["lcz_stats"].set_index("lcz_class")
    col    = SEASON_COLORS.get(label, f"C{si}")
    offset = (si - (n_sea_h - 1) / 2) * bar_h

    means = [sdf.loc[c, "mean"] if c in sdf.index else np.nan for c in common_cls_h]
    stds  = [sdf.loc[c, "std"]  if c in sdf.index else 0.0    for c in common_cls_h]

    ax_suhi.barh(
        y_pos + offset, means, height=bar_h * 0.9,
        xerr=stds, color=col, alpha=0.82, label=label,
        error_kw=dict(elinewidth=0.7, ecolor="#555555", capsize=2),
        zorder=2,
    )

ax_suhi.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax_suhi.set_yticks(y_pos)
ax_suhi.set_yticklabels([])
ax_suhi.set_xlabel("Mean SUHI Intensity (°C)", fontsize=9)
ax_suhi.set_title("Mean SUHI per LCZ class × season", fontsize=10, fontweight="bold")
ax_suhi.spines[["top", "right"]].set_visible(False)
ax_suhi.grid(axis="x", alpha=0.25, linewidth=0.5)
ax_suhi.legend(fontsize=8.5, frameon=True, framealpha=0.9,
               edgecolor="#cccccc", loc="lower right")

fig_h.suptitle(
    "LCZ Urban Area and Mean SUHI by Season — Bolzano",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()

out_h = os.path.join(OUTPUT_DIR, "suhi_lcz_area_suhi.png")
fig_h.savefig(out_h, dpi=300, bbox_inches="tight", facecolor="white")
fig_h.savefig(out_h.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
print(f"  Saved → {out_h}")
plt.show()


#%% end
