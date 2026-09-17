#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UHI Combined Figure — interannual heatmap + distribution + seasonal maps
Single figure combining:
  a) Interannual heatmap of spatio-temporally averaged UHI intensity per
     year / season / time window (24H, daytime, nighttime).
  b) Per-season, per-time-window distributions (KDE) of UHI intensity
     across all urban pixels and all years.
  c) Spatial maps of the mean UHI intensity averaged over the full record
     for each season, with urban/rural boundaries.

Built to replace two separate figures (interannual heatmap+distribution,
and seasonal maps) in the LaTeX document with one combined figure.

Cell structure
──────────────
  [0] Imports
  [1] Paths & parameters              ← edit here
  [2] Open dataset & shapefiles       ← cheap (lazy)
  [3] Lapse-rate correction           ← cheap
  [4] HEAVY: per-timestep CSV         ← asks before recomputing
  [5] HEAVY: seasonal spatial NPZ     ← asks before recomputing
  [6] Load caches                     ← fast, always safe to re-run
  [7] Combined plot function
  [8] Run plot
"""


#%% ── [0] IMPORTS ─────────────────────────────────────────────────────────────

import io
import os
import sys
import warnings
from datetime import datetime

import contextily as ctx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import reproject as rio_reproject, Resampling as RioResampling
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# Dynamic, per-(season, day/night) valley lapse rate + the same astronomical
# sunrise/sunset day/night classifier used across the WRF/UHI/SUHI scripts -- see
# UHI/seasonal_valley_lapse_rate.py and UHI/solar_daynight.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT


#%% ── [1] PATHS & PARAMETERS ─────────────────────────────────────────────────

UCB_FILE    = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc'
URBAN_SHAPE = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area_WGS84.shp'
RURAL_SHAPE = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'

# Elevation-correction mode -- edit and re-run to get each version; see
# common/lapse_mode_utils.py. CSV_TIMESTEP/NPZ_ALL/output filenames are all tagged
# with this mode: those two caches are looked up under a FIXED filename (via
# confirm_recompute()'s interactive gate, not a parameter hash), so without the
# mode in the filename, switching LAPSE_MODE and answering "n" (use cache) at the
# prompt would silently serve another mode's stale results.
#   'none'     -- no elevation correction (raw urban-minus-rural difference)
#   'constant' -- classical fixed rate (LAPSE_RATE_CONSTANT), no season/day-night dependence
#   'valley'   -- dynamic, per-(season, day/night) rate fitted from WRF (default)
LAPSE_MODE = 'valley'

# Re-use the existing caches from the two source scripts — no need to
# recompute anything that has already been extracted.
CSV_DIR   = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/ridge_plots'
NPZ_DIR   = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/spatial_maps'
OUTPUT_DIR = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/combined_figures'

CSV_TIMESTEP = os.path.join(CSV_DIR, f'uhi_per_timestep_{LAPSE_MODE}.csv')
NPZ_ALL      = os.path.join(NPZ_DIR, f'uhi_spatial_means_all_{LAPSE_MODE}.npz')   # panel c) uses the full-record, all-hours average

UTM_CRS    = 'EPSG:32632'

# DEM used for the elevation/lapse correction -- the UrbClim file itself carries no
# elevation variable (checked: only T2M/QV2M/WS2M/LST), so the same 10m south-Tirol
# DEM used by the SUHI/LCZ script is reprojected onto this grid instead.
DEM_RASTER = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif'

SEASON_MAP = {12: 'Winter', 1: 'Winter',  2: 'Winter',
               3: 'Spring',  4: 'Spring',  5: 'Spring',
               6: 'Summer',  7: 'Summer',  8: 'Summer',
               9: 'Fall',   10: 'Fall',   11: 'Fall'}
SEASON_ORDER  = ['Winter', 'Spring', 'Summer', 'Fall']
SEASON_TITLES = {'Winter': 'Winter (DJF)', 'Spring': 'Spring (MAM)',
                 'Summer': 'Summer (JJA)', 'Fall':   'Autumn (SON)'}

# This script names the fourth season 'Fall'; the shared valley-lapse-rate table
# (seasonal_valley_lapse_rate.py) uses 'Autumn' -- map between them.
_LAPSE_SEASON_KEY = {'Winter': 'Winter', 'Spring': 'Spring',
                      'Summer': 'Summer', 'Fall': 'Autumn'}

# {(season, 'day'|'night'): K/m} -- fitted once from the WRF archive, applied here to
# the UrbClim data exactly as to every other source (WRF, Netatmo, Satellite).
DYNAMIC_LAPSE = load_seasonal_valley_lapse()

BASEMAP_SOURCE = ctx.providers.OpenTopoMap

os.makedirs(OUTPUT_DIR, exist_ok=True)


#%% ── HELPER: ask before recomputing ────────────────────────────────────────

def confirm_recompute(cache_path, label):
    """Return True if the heavy computation should (re)run, False to use cache."""
    if os.path.exists(cache_path):
        print(f"\n✓ Found cached {label} → {cache_path}")
        resp = input(f"  Recompute {label} from raw data anyway? [y/N]: ").strip().lower()
        if resp not in ('y', 'yes'):
            print(f"  → Using cached {label} (fast).")
            return False
        print(f"  → Recomputing {label} — this reads the full netCDF, can take a while…")
        return True
    else:
        print(f"\n✗ No cache found for {label} at {cache_path}")
        print(f"  → Will compute from raw data — this can take a while…")
        return True


#%% ── [2] OPEN DATASET & SHAPEFILES ──────────────────────────────────────────
# Lazy open — cheap even if we end up using the caches below.

print("Opening dataset (lazy)…")
ds = xr.open_dataset(UCB_FILE, chunks={'t': 200})
ds.rio.write_crs(UTM_CRS, inplace=True)
times = pd.DatetimeIndex(ds.t.values)
print(f"  Time range : {times[0]}  →  {times[-1]}")
print(f"  Time steps : {len(times)}")

print("Loading shapefiles…")
urban_shape = gpd.read_file(URBAN_SHAPE).to_crs(UTM_CRS)
rural_shape = gpd.read_file(RURAL_SHAPE).to_crs(UTM_CRS)
print("  Done.")


#%% ── [3] LAPSE-RATE CORRECTION ──────────────────────────────────────────────
# Builds `elev_diff` (elevation minus rural mean elevation, on the UrbClim grid) --
# NOT multiplied by a rate here. The rate is season x day/night dependent
# (DYNAMIC_LAPSE), so the multiplication happens per-timestep, in process_chunk()
# and the seasonal-NPZ loop below, instead of being baked into a single field.
# Also: previously the correction (when an elevation var existed) was added only to
# the rural clip before averaging, which -- since it was referenced to that same
# rural clip's own mean -- cancelled to exactly zero once re-averaged; it's applied
# to the WHOLE field (urban included) below instead, with the reference kept raw.

print("Preparing DEM-based elevation field (reprojected onto UrbClim grid)…")

with rasterio.open(DEM_RASTER) as _dem_src:
    _dem_data          = _dem_src.read(1).astype(np.float32)
    _dem_nodata        = _dem_src.nodata
    _dem_src_transform = _dem_src.transform
    _dem_src_crs       = _dem_src.crs

_x_arr, _y_arr = ds.x.values, ds.y.values
_res_x = float(_x_arr[1] - _x_arr[0])
_res_y = float(_y_arr[1] - _y_arr[0])
# UrbClim's y axis is ascending (row 0 = south), unlike the usual north-up raster
# convention -- build the affine explicitly so row index tracks ascending y.
_ucb_transform = Affine(_res_x, 0, _x_arr[0] - _res_x / 2,
                         0, _res_y, _y_arr[0] - _res_y / 2)

_dem_on_ucb = np.full((len(_y_arr), len(_x_arr)), np.nan, dtype=np.float32)
rio_reproject(
    source=_dem_data, destination=_dem_on_ucb,
    src_transform=_dem_src_transform, src_crs=_dem_src_crs,
    dst_transform=_ucb_transform, dst_crs=UTM_CRS,
    resampling=RioResampling.bilinear,
    src_nodata=_dem_nodata, dst_nodata=np.nan,
)

dem_da = xr.DataArray(_dem_on_ucb.astype(np.float64), dims=('y', 'x'),
                      coords={'y': ds.y, 'x': ds.x})
dem_da.rio.write_crs(UTM_CRS, inplace=True)

rural_elev_clip = dem_da.rio.clip(rural_shape.geometry, rural_shape.crs)
mean_rural_elev = float(rural_elev_clip.mean(skipna=True))
elev_diff       = dem_da - mean_rural_elev
print(f"  Mean rural elevation = {mean_rural_elev:.1f} m (from {os.path.basename(DEM_RASTER)}) "
      f"→ dynamic per-(season, day/night) correction ready.")


#%% ── [4] HEAVY: PER-TIMESTEP CSV (panels a/b) ─────────────────────────────
# Only recomputes if the cache is missing or the user explicitly confirms.

def dynamic_rate_array(chunk_times):
    """Per-timestep rate (K/m) as a 1-D array aligned with chunk_times, resolved
    per LAPSE_MODE ('none'/'constant' both collapse to a uniform array; 'valley'
    uses each timestep's own (season, astronomical day/night) bucket)."""
    if LAPSE_MODE == 'none':
        return np.zeros(len(chunk_times))
    if LAPSE_MODE == 'constant':
        return np.full(len(chunk_times), LAPSE_RATE_CONSTANT)
    seasons = [_LAPSE_SEASON_KEY[SEASON_MAP[m]] for m in chunk_times.month]
    periods = classify_day_night(chunk_times)
    return np.array([DYNAMIC_LAPSE.get((s, p), 0.0) for s, p in zip(seasons, periods)])


def process_chunk(t2m_chunk, urban_shape, rural_shape, elev_diff):
    chunk_times = pd.DatetimeIndex(t2m_chunk.t.values)

    if elev_diff is not None:
        rates      = dynamic_rate_array(chunk_times)
        rate_da    = xr.DataArray(rates, dims='t', coords={'t': t2m_chunk.t})
        corrected  = t2m_chunk + rate_da * elev_diff
    else:
        corrected  = t2m_chunk

    urban = corrected.rio.clip(urban_shape.geometry, urban_shape.crs)
    # Reference stays the RAW rural mean (uncorrected) -- the correction was applied
    # to the whole field above (urban included), not to these rural pixels, so it
    # can't cancel to zero once re-averaged over this same rural mask.
    rural = t2m_chunk.rio.clip(rural_shape.geometry, rural_shape.crs)

    rural_mean = rural.mean(dim=['x', 'y'])
    uhi = urban - rural_mean

    arr = uhi.values.reshape(len(uhi.t), -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = dict(
            uhi_mean   = np.nanmean(arr,           axis=1),
            uhi_median = np.nanmedian(arr,          axis=1),
            uhi_std    = np.nanstd(arr,             axis=1),
            uhi_p10    = np.nanpercentile(arr, 10,  axis=1),
            uhi_p25    = np.nanpercentile(arr, 25,  axis=1),
            uhi_p75    = np.nanpercentile(arr, 75,  axis=1),
            uhi_p90    = np.nanpercentile(arr, 90,  axis=1),
            uhi_min    = np.nanmin(arr,             axis=1),
            uhi_max    = np.nanmax(arr,             axis=1),
            n_valid_px = np.sum(~np.isnan(arr),     axis=1),
        )

    df = pd.DataFrame(stats, index=chunk_times)
    df.index.name = 'time'
    df['year']        = chunk_times.year
    df['month']       = chunk_times.month
    df['hour']        = chunk_times.hour
    df['season']      = [SEASON_MAP[m] for m in chunk_times.month]
    df['time_period'] = ['day_time' if p == 'day' else 'night_time'
                         for p in classify_day_night(chunk_times)]
    return df


if confirm_recompute(CSV_TIMESTEP, 'per-timestep UHI stats (CSV)'):
    os.makedirs(CSV_DIR, exist_ok=True)
    write_header = True
    year_months  = sorted(set(zip(times.year, times.month)))
    n_chunks     = len(year_months)
    print(f"  Elevation correction : LAPSE_MODE={LAPSE_MODE!r}")
    print(f"  Output CSV           : {CSV_TIMESTEP}\n")

    for i, (yr, mo) in enumerate(year_months):
        mask  = (times.year == yr) & (times.month == mo)
        t_idx = np.where(mask)[0]

        chunk    = ds['T2M'].isel(t=t_idx).load()
        df_chunk = process_chunk(chunk, urban_shape, rural_shape, elev_diff)

        df_chunk.to_csv(CSV_TIMESTEP, mode='w' if write_header else 'a', header=write_header)
        write_header = False
        del chunk, df_chunk

        if (i + 1) % 12 == 0 or (i + 1) == n_chunks:
            print(f"  [{i+1:4d}/{n_chunks}]  {yr}-{mo:02d}  done")

    print(f"\n✓ Extraction complete  →  {CSV_TIMESTEP}")


#%% ── [5] HEAVY: SEASONAL SPATIAL NPZ (panel c, all-hours) ─────────────────
# Only recomputes if the cache is missing or the user explicitly confirms.

if confirm_recompute(NPZ_ALL, 'seasonal spatial UHI means, all-hours (NPZ)'):
    os.makedirs(NPZ_DIR, exist_ok=True)

    nx, ny = len(ds.x), len(ds.y)
    acc_sum   = {s: np.zeros((ny, nx), dtype=np.float64) for s in SEASON_ORDER}
    acc_count = {s: np.zeros((ny, nx), dtype=np.float64) for s in SEASON_ORDER}

    year_months = sorted(set(zip(times.year, times.month)))
    n_chunks    = len(year_months)
    print(f"  Elevation correction: LAPSE_MODE={LAPSE_MODE!r}\n")

    for i, (yr, mo) in enumerate(year_months):
        mask  = (times.year == yr) & (times.month == mo)
        t_idx = np.where(mask)[0]

        t2m = ds['T2M'].isel(t=t_idx).load()
        if t2m.dims != ('t', 'y', 'x'):
            t2m = t2m.transpose('t', 'y', 'x') if 'y' in t2m.dims else t2m.transpose('t', ...)

        chunk_times = pd.DatetimeIndex(t2m.t.values)

        # Reference stays the RAW rural mean; the dynamic per-timestep correction
        # (below) is applied to the whole field, not to these rural pixels, so it
        # can't cancel to zero once re-averaged over this same rural mask.
        rural_clip = t2m.rio.clip(rural_shape.geometry, rural_shape.crs)
        rural_mean = rural_clip.mean(dim=[d for d in rural_clip.dims if d != 't'])

        if elev_diff is not None:
            rates   = dynamic_rate_array(chunk_times)
            rate_da = xr.DataArray(rates, dims='t', coords={'t': t2m.t})
            t2m_corr = t2m + rate_da * elev_diff
        else:
            t2m_corr = t2m

        uhi_full = t2m_corr - rural_mean
        uhi_da   = uhi_full.rio.clip(urban_shape.geometry, urban_shape.crs, drop=False)
        uhi      = uhi_da.values

        season = SEASON_MAP[mo]
        valid  = ~np.isnan(uhi)
        acc_sum[season]   += np.nansum(uhi, axis=0)
        acc_count[season] += valid.sum(axis=0).astype(np.float64)

        del t2m, t2m_corr, rural_clip, rural_mean, uhi_full, uhi_da, uhi, valid

        if (i + 1) % 12 == 0 or (i + 1) == n_chunks:
            print(f"  [{i+1:4d}/{n_chunks}]  {yr}-{mo:02d}  done")

    print("\n✓ Accumulation complete. Computing seasonal means…")
    uhi_mean_computed = {}
    for s in SEASON_ORDER:
        cnt = acc_count[s].copy()
        cnt[cnt == 0] = np.nan
        uhi_mean_computed[s] = acc_sum[s] / cnt

    np.savez_compressed(
        NPZ_ALL,
        x=ds.x.values,
        y=ds.y.values,
        **{f'uhi_{s}': uhi_mean_computed[s] for s in SEASON_ORDER},
    )
    print(f"\n✓ Cache saved → {NPZ_ALL}")


#%% ── [6] LOAD CACHES ────────────────────────────────────────────────────────

print(f"\nLoading CSV cache: {CSV_TIMESTEP}")
stats_df = pd.read_csv(CSV_TIMESTEP, index_col='time', parse_dates=True)
stats_df['season']      = [SEASON_MAP[m] for m in stats_df['month']]
stats_df['time_period'] = ['day_time' if p == 'day' else 'night_time'
                           for p in classify_day_night(stats_df.index)]
print(f"  Loaded {len(stats_df):,} timesteps.")

print(f"Loading NPZ cache: {NPZ_ALL}")
_data    = np.load(NPZ_ALL)
x_arr    = _data['x']
y_arr    = _data['y']
uhi_maps = {s: _data[f'uhi_{s}'] for s in SEASON_ORDER}
x_min, x_max = x_arr.min(), x_arr.max()
y_min, y_max = y_arr.min(), y_arr.max()
print("✓ Ready to plot.")


#%% ── [7] COMBINED PLOT FUNCTION ────────────────────────────────────────────

def add_grayscale_basemap(ax, x_min, x_max, y_min, y_max,
                          source=BASEMAP_SOURCE, dpi=150):
    """Render a basemap tile to grayscale and place it under the data layer."""
    fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(6, 6))
    ax_tmp.set_xlim(x_min, x_max)
    ax_tmp.set_ylim(y_min, y_max)
    ctx.add_basemap(ax_tmp, crs=UTM_CRS, source=source, zoom='auto',
                    zorder=1, attribution_size=0)
    ax_tmp.axis('off')
    buf = io.BytesIO()
    fig_tmp.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig_tmp)
    buf.seek(0)
    tile_img = Image.open(buf).convert('L')
    tile_rgb = np.array(tile_img.convert('RGB'))
    ax.imshow(tile_rgb, extent=[x_min, x_max, y_min, y_max],
              origin='upper', aspect='auto', zorder=1, interpolation='bilinear')


def plot_combined_figure(stats_df, uhi_maps, x_arr, y_arr, urban_shape, rural_shape,
                         output_dir=None):
    """
    One figure, three panels:
      a) interannual heatmap (year × season × time window)
      b) per-season / per-time-window UHI distributions (KDE ridges)
      c) seasonal maps of the full-record mean UHI intensity
    """

    mpl.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
    })

    season_order  = SEASON_ORDER
    season_titles = SEASON_TITLES
    period_order  = ['full_time', 'day_time', 'night_time']
    period_labels = {'full_time': '24H', 'day_time': 'Day (sunrise–sunset)', 'night_time': 'Night (sunset–sunrise)'}
    period_ls     = {'full_time': '-',  'day_time': '--', 'night_time': ':'}
    period_lw     = {'full_time': 1.8,  'day_time': 1.4,  'night_time': 1.4}
    period_colors = {'full_time': '#222222', 'day_time': '#888888', 'night_time': '#bbbbbb'}

    df_full = stats_df.copy(); df_full['time_period'] = 'full_time'
    df_all  = pd.concat([stats_df, df_full])

    yearly = {}
    for p in period_order:
        yearly[p] = (df_all[df_all['time_period'] == p]
                     .groupby(['year', 'season'])['uhi_mean']
                     .mean()
                     .unstack('season')[season_order])

    years   = yearly['full_time'].index.tolist()
    n_years = len(years)

    all_vals  = np.concatenate([yearly[p].values.flatten() for p in period_order])
    heat_vmin = np.nanpercentile(all_vals, 2)
    heat_vmax = np.nanpercentile(all_vals, 98)
    heat_norm = mpl.colors.Normalize(vmin=heat_vmin, vmax=heat_vmax)

    # ── Panel c) map color scale ────────────────────────────────────────────
    map_vals = np.concatenate([uhi_maps[s].flatten() for s in season_order])
    map_vmin = 0
    map_vmax = np.nanpercentile(map_vals[map_vals > 0], 98)
    map_cmap = plt.cm.YlOrRd.copy()   # same colormap family as the heatmap (panel a) for visual consistency
    map_cmap.set_bad(alpha=0)

    tr_utm_to_wgs = Transformer.from_crs(UTM_CRS, 'EPSG:4326', always_xy=True)

    def format_lon(x_utm, pos):
        lon, _ = tr_utm_to_wgs.transform(x_utm, (y_min + y_max) / 2)
        return f"{lon:.2f}°E"

    def format_lat(y_utm, pos):
        _, lat = tr_utm_to_wgs.transform((x_min + x_max) / 2, y_utm)
        return f"{lat:.2f}°N"

    # ── Figure & outer layout: left = heatmap+distribution, right = maps ───
    fig = plt.figure(figsize=(24, 13))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.20)

    left_gs = outer[0].subgridspec(
        6, 2, width_ratios=[4.5, 1], height_ratios=[1, 1, 1, 1, 0.10, 0.06],
        hspace=0.18, wspace=0.03)
    right_gs = outer[1].subgridspec(
        3, 2, height_ratios=[1, 1, 0.07], hspace=0.22, wspace=0.16)

    ax_heat = [fig.add_subplot(left_gs[i, 0]) for i in range(4)]
    ax_kde  = [fig.add_subplot(left_gs[i, 1]) for i in range(4)]
    ax_cbar_left = fig.add_subplot(left_gs[5, :])

    # 2×2 grid: Winter, Spring / Summer, Fall
    ax_map  = [fig.add_subplot(right_gs[i // 2, i % 2]) for i in range(4)]
    ax_cbar_right = fig.add_subplot(right_gs[2, :])

    # ── a) + b) heatmap & KDE ridges (per season row) ──────────────────────
    im = None
    for si, season in enumerate(season_order):
        ax_h = ax_heat[si]
        ax_k = ax_kde[si]

        mat = np.array([yearly[p][season].values for p in period_order])
        im = ax_h.imshow(mat, aspect='auto', cmap='YlOrRd', norm=heat_norm,
                         origin='upper', interpolation='nearest')

        ax_h.set_yticks(range(3))
        ax_h.set_yticklabels([period_labels[p] for p in period_order], fontsize=8.5)
        ax_h.set_title(season_titles[season], loc='left', fontsize=10, fontweight='bold', pad=4)

        if si < 3:
            ax_h.set_xticks([])
        else:
            ax_h.set_xticks(range(n_years))
            ax_h.set_xticklabels(years, rotation=45, ha='right', fontsize=8)
            ax_h.set_xlabel('Year', fontsize=9, labelpad=4)

        for y in [0.5, 1.5]:
            ax_h.axhline(y, color='white', lw=0.8, alpha=0.5)

        TEXT_THRESH = 3.4
        for pi in range(3):
            for yi in range(n_years):
                val = mat[pi, yi]
                if not np.isnan(val):
                    txt_col = 'white' if val >= TEXT_THRESH else 'black'
                    ax_h.text(yi, pi, f"{val:.1f}", ha='center', va='center',
                              fontsize=5.5, fontweight='bold', color=txt_col)

        for spine in ax_h.spines.values():
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(0.8)

        ax_k.set_facecolor('white')
        x_all = []
        for period in period_order:
            vals = yearly[period][season].dropna().values
            if len(vals) >= 4:
                x_all.extend(np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300).tolist())

        cmap_ridge = mpl.colormaps['YlOrRd']
        for period in period_order:
            vals = yearly[period][season].dropna().values
            if len(vals) < 4:
                continue

            kde   = gaussian_kde(vals, bw_method=0.55)
            x_kde = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
            y_kde = kde(x_kde) / kde(x_kde).max() * 0.82

            offsets = {'full_time': 0.0, 'day_time': 0.88, 'night_time': 1.76}
            offset = offsets[period]

            for j in range(len(x_kde) - 1):
                x_mid = 0.5 * (x_kde[j] + x_kde[j + 1])
                c = cmap_ridge(heat_norm(x_mid))
                ax_k.fill_betweenx([x_kde[j], x_kde[j + 1]], offset,
                                   offset + np.array([y_kde[j], y_kde[j + 1]]),
                                   color=c, alpha=0.85, linewidth=0)

            ax_k.plot(offset + y_kde, x_kde, color=period_colors[period],
                      lw=period_lw[period], ls=period_ls[period])

            med = float(np.median(vals))
            ax_k.plot([offset, offset + 0.18], [med, med],
                      color=period_colors[period], lw=1.0, ls='-', alpha=0.8)

        ax_k.set_xlim(-0.05, 2.65)
        if x_all:
            ax_k.set_ylim(min(x_all) - 0.05, max(x_all) + 0.05)

        ax_k.set_xticks([])
        ax_k.yaxis.tick_right()
        ax_k.yaxis.set_label_position('right')
        ax_k.tick_params(axis='y', labelsize=7.5)
        ax_k.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        if si == 3:
            ax_k.set_ylabel('UHI Intensity (°C)', fontsize=8.5, labelpad=6)

        for spine in ax_k.spines.values():
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(0.6)
        ax_k.spines['left'].set_visible(False)
        ax_k.spines['bottom'].set_visible(False)
        ax_k.spines['top'].set_visible(False)

    y_lo = min(ax.get_ylim()[0] for ax in ax_kde)
    y_hi = max(ax.get_ylim()[1] for ax in ax_kde)
    for ax_k in ax_kde:
        ax_k.set_ylim(y_lo, y_hi)

    legend_lines = [
        Line2D([0], [0], color=period_colors[p], lw=period_lw[p], ls=period_ls[p],
               label=period_labels[p])
        for p in period_order
    ]
    ax_kde[0].legend(handles=legend_lines, fontsize=7, loc='lower right', frameon=True,
                     facecolor='white', framealpha=1.0, edgecolor='#aaaaaa', handlelength=2.5)
    ax_kde[3].set_xticks([0.41, 1.29, 2.17])
    ax_kde[3].set_xticklabels([period_labels[p] for p in period_order],
                              fontsize=7, rotation=30, ha='right')

    cb_left = fig.colorbar(im, cax=ax_cbar_left, orientation='horizontal', norm=heat_norm)
    cb_left.set_label('Spatially-averaged UHI Intensity (°C)', fontsize=9)
    cb_left.ax.tick_params(labelsize=8.5)
    cb_left.outline.set_edgecolor('#aaaaaa')
    cb_left.outline.set_linewidth(0.6)

    # ── c) seasonal maps ────────────────────────────────────────────────────
    im_map = None
    for si, season in enumerate(season_order):
        ax = ax_map[si]
        arr = uhi_maps[season]

        origin = 'lower' if y_arr[-1] > y_arr[0] else 'upper'
        extent = [x_min, x_max, y_min, y_max] if origin == 'lower' \
                 else [x_min, x_max, y_max, y_min]

        add_grayscale_basemap(ax, x_min, x_max, y_min, y_max)

        im_map = ax.imshow(arr, extent=extent, origin=origin, cmap=map_cmap,
                           vmin=map_vmin, vmax=map_vmax, zorder=2,
                           alpha=0.75, interpolation='bilinear')

        urban_shape.boundary.plot(ax=ax, color='#222222', linewidth=1.0, linestyle='-', zorder=3)
        rural_shape.boundary.plot(ax=ax, color='#555555', linewidth=0.8, linestyle='--', zorder=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(season_titles[season], fontsize=10, fontweight='bold', pad=4, loc='left')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
        ax.tick_params(labelsize=6.5)

        if si < 2:
            ax.set_xticklabels([])
        else:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha('right')

        if si % 2 == 1:
            ax.set_yticklabels([])

        if si == 0:
            handles = [
                mpatches_patch(edgecolor='#222222', label='Urban area'),
                mpatches_patch(edgecolor='#555555', label='Rural area', linestyle='--'),
            ]
            ax.legend(handles=handles, fontsize=6, loc='lower left',
                      framealpha=0.85, edgecolor='#aaaaaa')

    cb_right = fig.colorbar(im_map, cax=ax_cbar_right, orientation='horizontal', extend='max')
    cb_right.set_label('Temporally-averaged UHI Intensity (°C)', fontsize=9)
    cb_right.ax.tick_params(labelsize=8)
    cb_right.outline.set_edgecolor('#aaaaaa')
    cb_right.outline.set_linewidth(0.6)

    # ── Equalize colorbar thickness ─────────────────────────────────────────
    # left_gs and right_gs have different row-ratio totals, so their colorbar
    # rows map to different absolute heights even with matching ratios —
    # force both bars to the same physical thickness, anchored at their
    # existing top edge.
    fig.canvas.draw()
    pos_l = ax_cbar_left.get_position()
    pos_r = ax_cbar_right.get_position()
    cbar_height = min(pos_l.height, pos_r.height)
    ax_cbar_left.set_position([pos_l.x0, pos_l.y1 - cbar_height, pos_l.width, cbar_height])
    ax_cbar_right.set_position([pos_r.x0, pos_r.y1 - cbar_height, pos_r.width, cbar_height])

    # ── Panel labels a) b) c) — aligned on one common horizontal line ──────
    fig.canvas.draw()
    pos_a = ax_heat[0].get_position()
    pos_b = ax_kde[0].get_position()
    pos_c = ax_map[0].get_position()
    label_y = max(pos_a.y1, pos_b.y1, pos_c.y1) + 0.012

    fig.text(pos_a.x0 - 0.028, label_y, 'a)', fontsize=15, fontweight='bold')
    fig.text(pos_b.x0 - 0.010, label_y, 'b)', fontsize=15, fontweight='bold')
    fig.text(pos_c.x0 - 0.028, label_y, 'c)', fontsize=15, fontweight='bold')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for ext in ('pdf', 'png'):
            p = os.path.join(output_dir, f"uhi_distri_map_combined_{LAPSE_MODE}_{ts}.{ext}")
            plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"Saved → {p}")

    plt.show()
    return fig


def mpatches_patch(edgecolor, label, linestyle='-'):
    import matplotlib.patches as mpatches
    return mpatches.Patch(edgecolor=edgecolor, facecolor='none',
                          linewidth=1.2, linestyle=linestyle, label=label)


#%% ── [8] RUN PLOT ────────────────────────────────────────────────────────────

plot_combined_figure(stats_df, uhi_maps, x_arr, y_arr, urban_shape, rural_shape, OUTPUT_DIR)
