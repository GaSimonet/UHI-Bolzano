#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/meteotracker_vs_netatmo/combine_distribution_netatmo_MT_map_20260616.py
# Paper figure:       combined_uhi_ridge_map_2025-07-01_to_2025-08-01_12-15h.pdf
# Note: Unrelated to the valley-lapse-rate methodology -- included only because it feeds a paper figure. Set RIDGE_MODE='uhi' (not 'temperature') to reproduce this exact figure; 'temperature' mode produces the sibling combined_temp_ridge_map figure instead.
# UPDATE (2026-08-04): the ridge plot's day/night split previously used a fixed 06:00-18:00/
# 18:00-06:00 clock-hour window -- switched to the same astronomical sunrise/sunset
# classification (common/solar_daynight.classify_day_night) used across the UHI scripts.
# Ridge row labels no longer print a clock-hour range (e.g. "Daytime (6-18 h)"). The
# right-panel map's MAP_START_HOUR/MAP_END_HOUR window is untouched -- that's a fixed
# representative-snapshot period (the caption's "12:00-15:00 UTC"), not a day/night split.
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-
"""
Combined figure:
  a) Left  - Netatmo seasonal distributions (12-row ridge plot,
              colored by median value via shared colorbar)
              RIDGE_MODE = 'temperature' : air temperature (°C)
              RIDGE_MODE = 'uhi'         : UHI intensity (°C)
  b) Right - MeteoTracker temperature map with Netatmo station overlay
Panel labels in boxes, no titles, unified fonts.
"""

#%% Imports
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import math, time, datetime, os, sys
import fiona
from shapely.geometry import Point, shape
from fiona.transform import transform_geom
from matplotlib.colors import Normalize

# Astronomical day/night split (real sunrise/sunset per calendar day, Bolzano),
# replacing the old fixed 06:00-18:00/18:00-06:00 clock-hour window -- matches
# the convention used across the UHI scripts (common/solar_daynight.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from solar_daynight import classify_day_night

#%% ------------------------- CONFIGURATION -------------------------

METEOTRACKER_FILE = "/home/gsimonet/Desktop/Meteotrackers_package/meteotracker_incremental/netcdf_files/meteotracker_summer_2025.nc"
NETATMO_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
URBAN_SHP = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_urban_area.shp'
RURAL_SHP = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_rural_area.shp'

OUTPUT_DIR = 'combined_figure'

# --- Map panel settings ---
LAT_MIN_BOUND, LAT_MAX_BOUND = 46.4, 46.6
LON_MIN_BOUND, LON_MAX_BOUND = 11.1, 11.6
LAT_CENTER, LON_CENTER = 46.4983, 11.3548
RADIUS_KM = 6
GRIDSIZE = 65
ZOOM = 14
MIN_PTS = 20
DATE_START, DATE_END = '2025-07-01', '2025-08-01'
MAP_START_HOUR, MAP_END_HOUR = 12, 15

NETATMO_MARKER_SIZE = 55
NETATMO_MARKER_ALPHA = 0.9
NETATMO_EDGE_WIDTH = 1.2

# --- Ridge plot options ---
RIDGE_MODE      = 'uhi'   # 'temperature'  or  'uhi'
SHOW_ALL_HOURS  = False    # include the "All hours" row for each season
# Colormaps: temperature mirrors the map (YlOrRd); UHI uses RdBu_r diverging around 0
RIDGE_CMAP_TEMP      = 'YlOrRd'
RIDGE_CMAP_UHI       = 'RdBu_r'
RIDGE_SYNC_MAP_RANGE = False   # share vmin/vmax with the map colorbar (temperature mode only)

# --- Unified fonts ---
FS_LABEL = 13
FS_TICK = 11
FS_RIDGE = 9
FS_LEGEND = 9
FS_PANEL = 16
FONT_FAMILY = 'sans-serif'

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
})

FIGSIZE = (27, 9)
DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

#%% ------------------------- HELPERS -------------------------

def get_season(date):
    m = date.month
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]:  return 'Spring'
    if m in [6, 7, 8]:  return 'Summer'
    return 'Autumn'

def get_time_period(times):
    """Vectorized 'day_time'/'night_time' classification from each timestamp's own
    real astronomical sunrise/sunset (Bolzano), replacing the old fixed 6-18h split."""
    return np.where(classify_day_night(times) == 'day', 'day_time', 'night_time')

#%% ------------------------- LOAD SHAPEFILES & CLASSIFY STATIONS -------------------------

print("Loading shapefiles...")
with fiona.open(URBAN_SHP) as urban_src, fiona.open(RURAL_SHP) as rural_src:
    urban_geom = [shape(f['geometry']) for f in urban_src]
    rural_geom = [shape(f['geometry']) for f in rural_src]

print(f"Loading Netatmo data: {NETATMO_FILE}")
ds_netatmo = xr.open_dataset(NETATMO_FILE)

TEMP_VAR_CANDIDATES = ['T_lvl4', 'T0', 'temperature', 'T', 'temp', 'Ta']
NETATMO_TEMP_VAR = next((v for v in TEMP_VAR_CANDIDATES if v in ds_netatmo), None)
if NETATMO_TEMP_VAR is None:
    raise ValueError(
        "No temperature variable found in Netatmo file. "
        f"Available data variables: {list(ds_netatmo.data_vars)}"
    )
print(f"Using Netatmo temperature variable: '{NETATMO_TEMP_VAR}'")

netatmo_lats = ds_netatmo.latitude.values
netatmo_lons = ds_netatmo.longitude.values
netatmo_alts = ds_netatmo.altitude.values

def classify_point(lon, lat):
    g = transform_geom('EPSG:4326', 'EPSG:3035',
                       {'type': 'Point', 'coordinates': (lon, lat)})
    p = Point(g['coordinates'])
    if any(geom.contains(p) for geom in urban_geom):
        return 'urban'
    if any(geom.contains(p) for geom in rural_geom):
        return 'rural'
    return 'other'

# Per-station-index classification
urban_indices, rural_indices = [], []
all_valid_indices = []
for i, (lon, lat) in enumerate(zip(netatmo_lons, netatmo_lats)):
    if np.isnan(netatmo_alts[i]) or np.isnan(lat) or np.isnan(lon):
        continue
    cls = classify_point(lon, lat)
    all_valid_indices.append(i)
    if cls == 'urban':
        urban_indices.append(i)
    elif cls == 'rural':
        rural_indices.append(i)

print(f"Urban stations: {len(urban_indices)} | Rural stations: {len(rural_indices)}")

# Unique-location station list (for map overlay)
unique_stations = {}
for i, (lon, lat, alt) in enumerate(zip(netatmo_lons, netatmo_lats, netatmo_alts)):
    if not (np.isnan(lat) or np.isnan(lon)):
        key = (round(lat, 5), round(lon, 5))
        if key not in unique_stations:
            unique_stations[key] = {'lat': lat, 'lon': lon, 'alt': alt, 'index': i}

urban_stations, rural_stations, unclassified_stations = [], [], []
for (lat, lon), info in unique_stations.items():
    cls = classify_point(lon, lat)
    if cls == 'urban':
        urban_stations.append(info)
    elif cls == 'rural':
        rural_stations.append(info)
    else:
        unclassified_stations.append(info)

#%% ------------------------- TEMPERATURE TIME SERIES (RIDGE DATA) -------------------------

print("Computing temperature time series (all stations, altitude-corrected)...")
LAPSE_RATE = -0.0065

valid_altitudes = netatmo_alts[~np.isnan(netatmo_alts)]
reference_elevation = np.mean(valid_altitudes)

# Use all valid (urban + rural) stations for the temperature distribution
all_indices = urban_indices + rural_indices
all_temps = ds_netatmo[NETATMO_TEMP_VAR].isel(station=all_indices)
all_alts  = ds_netatmo.altitude.isel(station=all_indices).values

all_temps_corr = all_temps.copy()
for i, elevation in enumerate(all_alts):
    all_temps_corr[:, i] = all_temps[:, i] - LAPSE_RATE * (elevation - reference_elevation)

# City-wide mean temperature at each time step
city_mean_temp = all_temps_corr.mean('station')

temp_df = pd.DataFrame(
    {'temperature': city_mean_temp.values},
    index=pd.DatetimeIndex(ds_netatmo.time.values)
)
temp_df['season'] = temp_df.index.map(get_season)
temp_df['time_period'] = get_time_period(temp_df.index)

# Add 'full_time' duplicate rows
temp_full = temp_df.copy()
temp_full['time_period'] = 'full_time'
temp_all = pd.concat([temp_df, temp_full])
temp_all = temp_all[temp_all['temperature'].notna() & np.isfinite(temp_all['temperature'])]
temp_all['season_time'] = temp_all['season'] + '_' + temp_all['time_period']

#%% ------------------------- UHI TIME SERIES (RIDGE DATA – UHI MODE) -------------------------

print("Computing UHI time series...")
urban_temps = ds_netatmo[NETATMO_TEMP_VAR].isel(station=urban_indices)
rural_temps = ds_netatmo[NETATMO_TEMP_VAR].isel(station=rural_indices)

urban_temps_corr = urban_temps.copy()
rural_temps_corr = rural_temps.copy()
for i, elevation in enumerate(ds_netatmo.altitude.isel(station=urban_indices).values):
    urban_temps_corr[:, i] = urban_temps[:, i] - LAPSE_RATE * (elevation - reference_elevation)
for i, elevation in enumerate(ds_netatmo.altitude.isel(station=rural_indices).values):
    rural_temps_corr[:, i] = rural_temps[:, i] - LAPSE_RATE * (elevation - reference_elevation)

urban_mean = urban_temps_corr.mean('station')
rural_mean = rural_temps_corr.mean('station')

uhi = pd.DataFrame({'uhi': (urban_mean - rural_mean).values},
                   index=pd.DatetimeIndex(ds_netatmo.time.values))
uhi['season']      = uhi.index.map(get_season)
uhi['time_period'] = get_time_period(uhi.index)

uhi_full = uhi.copy()
uhi_full['time_period'] = 'full_time'
uhi_all = pd.concat([uhi, uhi_full])
uhi_all = uhi_all[uhi_all['uhi'].notna() & np.isfinite(uhi_all['uhi'])]
uhi_all['season_time'] = uhi_all['season'] + '_' + uhi_all['time_period']

#%% ------------------------- SELECT ACTIVE RIDGE DATASET -------------------------

row_order_global = [f"{s}_{tp}" for s in ['Winter', 'Spring', 'Summer', 'Autumn']
                    for tp in ['day_time', 'full_time', 'night_time']]

if RIDGE_MODE == 'uhi':
    ridge_df       = uhi_all.rename(columns={'uhi': 'value'})
    ridge_xlabel   = 'UHI Intensity (°C)'
    ridge_cbar_lbl = 'Median UHI intensity (°C)'
    ridge_cmap_name = RIDGE_CMAP_UHI
    x_min_ridge = min(-2, uhi_all['uhi'].quantile(0.001))
    x_max_ridge = max( 3, uhi_all['uhi'].quantile(0.999))
    ridge_mode_tag = 'uhi'
else:
    ridge_df       = temp_all.rename(columns={'temperature': 'value'})
    ridge_xlabel   = 'Air Temperature (°C)'
    ridge_cbar_lbl = 'Median temperature (°C)'
    ridge_cmap_name = RIDGE_CMAP_TEMP
    x_min_ridge = temp_all['temperature'].quantile(0.001)
    x_max_ridge = temp_all['temperature'].quantile(0.999)
    ridge_mode_tag = 'temp'

ridge_df['season_time'] = ridge_df['season'] + '_' + ridge_df['time_period']

#%% ------------------------- METEOTRACKER DATA -------------------------

print(f"Loading MeteoTracker data: {METEOTRACKER_FILE}")
ds_mt = xr.open_dataset(METEOTRACKER_FILE)

time_values = ds_mt.time.values
lat_values  = ds_mt.latitude.values
lon_values  = ds_mt.longitude.values
temp_values = ds_mt.T0.values

valid_mask = ~(np.isnan(time_values.astype('datetime64[ns]').view('int64')) |
               np.isnan(lat_values) | np.isnan(lon_values) | np.isnan(temp_values))
geo_mask = ((lat_values >= LAT_MIN_BOUND) & (lat_values <= LAT_MAX_BOUND) &
            (lon_values >= LON_MIN_BOUND) & (lon_values <= LON_MAX_BOUND))
mask = valid_mask & geo_mask

time_filtered = time_values[mask]
lat_filtered  = lat_values[mask]
lon_filtered  = lon_values[mask]
temp_filtered = temp_values[mask]

mt_dt    = pd.DatetimeIndex(time_filtered)
mt_dates = np.array([d.date() for d in mt_dt])
mt_hours = mt_dt.hour.values

# Pre-compute map temperature range (used to optionally sync ridge colorbar)
_map_start = datetime.datetime.strptime(DATE_START, '%Y-%m-%d').date()
_map_end   = datetime.datetime.strptime(DATE_END,   '%Y-%m-%d').date()
_map_mask  = ((mt_dates >= _map_start) & (mt_dates <= _map_end) &
              (mt_hours >= MAP_START_HOUR) & (mt_hours < MAP_END_HOUR))
_map_temps = temp_filtered[_map_mask]
MAP_VMIN = float(np.nanpercentile(_map_temps, 2))  if len(_map_temps) > 0 else None
MAP_VMAX = float(np.nanpercentile(_map_temps, 98)) if len(_map_temps) > 0 else None

#%% ------------------------- MAP TILES -------------------------

TILE_CACHE = {}

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    return (int((lon_deg + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_deg = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return (lat_deg, lon_deg)

def download_map_tiles_cached(lat_min, lat_max, lon_min, lon_max, zoom=14):
    cache_key = f"otm_{zoom}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}"
    if cache_key in TILE_CACHE:
        return TILE_CACHE[cache_key]

    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)
    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    if total > 12:
        zoom = max(10, zoom - 1)
        x_min, y_max = deg2num(lat_min, lon_min, zoom)
        x_max, y_min = deg2num(lat_max, lon_max, zoom)

    tiles, downloaded = {}, 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            try:
                time.sleep(0.15)
                r = requests.get(f'https://tile.opentopomap.org/{zoom}/{x}/{y}.png',
                                 timeout=10,
                                 headers={'User-Agent': 'MeteoTracker Scientific Visualization'})
                if r.status_code == 200:
                    tiles[(x, y)] = Image.open(BytesIO(r.content))
                    downloaded += 1
            except Exception:
                continue

    if downloaded == 0:
        return None, None, None, None, None

    ts = 256
    img = Image.new('RGB', ((x_max - x_min + 1) * ts, (y_max - y_min + 1) * ts),
                    color=(240, 240, 240))
    for (x, y), tile in tiles.items():
        img.paste(tile, ((x - x_min) * ts, (y - y_min) * ts))

    lat_max_a, lon_min_a = num2deg(x_min, y_min, zoom)
    lat_min_a, lon_max_a = num2deg(x_max + 1, y_max + 1, zoom)
    result = (img, lat_min_a, lat_max_a, lon_min_a, lon_max_a)
    TILE_CACHE[cache_key] = result
    return result

#%% ------------------------- PANEL a) RIDGE PLOT -------------------------

season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_names = {
    'Winter': 'Winter (DJF)', 'Spring': 'Spring (MAM)',
    'Summer': 'Summer (JJA)', 'Autumn': 'Autumn (SON)'
}
time_period_order = ['day_time', 'full_time', 'night_time'] if SHOW_ALL_HOURS \
                    else ['day_time', 'night_time']
time_period_names = {
    'day_time':   'Daytime',
    'full_time':  'All hours',
    'night_time': 'Nighttime'
}

row_order = [f"{s}_{tp}" for s in season_order for tp in time_period_order]

# Compute median value per row — drives the colormap
row_medians = {
    key: ridge_df.loc[ridge_df['season_time'] == key, 'value'].median()
    for key in row_order
}
all_medians = [v for v in row_medians.values() if np.isfinite(v)]
vmin_ridge = min(all_medians)
vmax_ridge = max(all_medians)

# For UHI: symmetric norm around 0
# For temperature with sync: share the map's vmin/vmax so colors are directly comparable
if RIDGE_MODE == 'uhi':
    bound = max(abs(vmin_ridge), abs(vmax_ridge))
    ridge_norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
elif RIDGE_MODE == 'temperature' and RIDGE_SYNC_MAP_RANGE and MAP_VMIN is not None:
    vmin_ridge, vmax_ridge = MAP_VMIN, MAP_VMAX
    ridge_norm = Normalize(vmin=vmin_ridge, vmax=vmax_ridge)
else:
    ridge_norm = Normalize(vmin=vmin_ridge, vmax=vmax_ridge)

ridge_cmap   = cm.get_cmap(ridge_cmap_name)
ridge_scalar = cm.ScalarMappable(norm=ridge_norm, cmap=ridge_cmap)
ridge_scalar.set_array([])

def draw_ridge_panel(fig, subspec):
    """Draw the 12-row ridge plot into the given SubplotSpec (no colorbar)."""
    n = len(row_order)
    inner = gridspec.GridSpecFromSubplotSpec(n, 1, subplot_spec=subspec, hspace=-0.65)
    axes = []

    for k, key in enumerate(row_order):
        ax = fig.add_subplot(inner[k])
        axes.append(ax)
        data  = ridge_df.loc[ridge_df['season_time'] == key, 'value']
        med   = row_medians[key]
        color = ridge_cmap(ridge_norm(med)) if np.isfinite(med) else 'gray'

        if len(data) > 1:
            sns.kdeplot(x=data, bw_adjust=0.7, clip_on=False, fill=True,
                        alpha=0.8, linewidth=1.5, color=color, ax=ax)
            sns.kdeplot(x=data, bw_adjust=0.7, clip_on=False, color='w',
                        lw=2, ax=ax)
            # Headroom above the peak -- without this, sharply-peaked rows (especially
            # the top row, which has no row above it to visually blend into) reach right
            # up to the axes/colorbar boundary with no margin.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)

        ax.axhline(0, lw=2, color=color, clip_on=False)
        if RIDGE_MODE == 'uhi':
            ax.axvline(0, ls='--', color='black', alpha=0.5, lw=1)

        # Row label — black for readability
        season, tp = key.split('_', 1)
        if tp == 'day_time':
            text = f"{season_names[season]} – {time_period_names[tp]}"
        else:
            text = time_period_names[tp]
        ax.text(-0.02, .2, text, fontweight='bold', color='black',
                ha='right', va='center', transform=ax.transAxes,
                fontsize=FS_RIDGE)

        # Median annotation
        if len(data) > 0:
            fmt = f"med={med:+.2f}°C" if RIDGE_MODE == 'uhi' else f"med={med:.1f}°C"
            ax.text(0.98, .2, fmt, color='black', alpha=0.7,
                    ha='right', va='center', transform=ax.transAxes,
                    fontsize=FS_RIDGE - 1)

        ax.set_xlim(x_min_ridge, x_max_ridge)
        ax.patch.set_alpha(0)
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        for s in ax.spines.values():
            s.set_visible(False)
        if k < n - 1:
            ax.set_xticks([])
        else:
            ax.tick_params(axis='x', labelsize=FS_TICK)
            ax.set_xlabel(ridge_xlabel, fontsize=FS_LABEL, fontweight='bold')

    return axes

#%% ------------------------- PANEL b) OVERLAY MAP -------------------------

def draw_map_panel(fig, ax, cax):
    """Draw the MeteoTracker hexbin + Netatmo overlay onto ax; cax receives the colorbar."""
    start_date = datetime.datetime.strptime(DATE_START, '%Y-%m-%d').date()
    end_date   = datetime.datetime.strptime(DATE_END,   '%Y-%m-%d').date()

    lat_radius = RADIUS_KM / 111.0
    lon_radius = RADIUS_KM / (111.0 * np.cos(np.radians(LAT_CENTER)))
    lat_min_f, lat_max_f = LAT_CENTER - lat_radius, LAT_CENTER + lat_radius
    lon_min_f, lon_max_f = LON_CENTER - lon_radius, LON_CENTER + lon_radius

    map_img, mlat_min, mlat_max, mlon_min, mlon_max = \
        download_map_tiles_cached(lat_min_f, lat_max_f, lon_min_f, lon_max_f, zoom=ZOOM)
    if map_img:
        g = ImageEnhance.Brightness(
            ImageEnhance.Contrast(map_img.convert('L')).enhance(1.2)).enhance(1.15)
        ax.imshow(g, extent=(mlon_min, mlon_max, mlat_min, mlat_max),
                  aspect='auto', cmap='gray', alpha=0.65, vmin=100, vmax=255)
    else:
        ax.set_facecolor('#f5f5f5')

    ax.set_xlim(lon_min_f, lon_max_f)
    ax.set_ylim(lat_min_f, lat_max_f)
    ax.set_aspect('equal', adjustable='box')

    tmask = ((mt_dates >= start_date) & (mt_dates <= end_date) &
             (mt_hours >= MAP_START_HOUR) & (mt_hours < MAP_END_HOUR) &
             (lat_filtered >= lat_min_f) & (lat_filtered <= lat_max_f) &
             (lon_filtered >= lon_min_f) & (lon_filtered <= lon_max_f))
    d_lons, d_lats, d_temps = lon_filtered[tmask], lat_filtered[tmask], temp_filtered[tmask]
    print(f"MeteoTracker measurements in map period: {len(d_lats):,}")

    _sync = (RIDGE_MODE == 'temperature' and RIDGE_SYNC_MAP_RANGE and MAP_VMIN is not None)
    _vmin = MAP_VMIN if _sync else None
    _vmax = MAP_VMAX if _sync else None

    if len(d_lats) >= 15:
        hb = ax.hexbin(d_lons, d_lats, C=d_temps, gridsize=GRIDSIZE,
                       cmap='OrRd', reduce_C_function=np.mean,
                       mincnt=MIN_PTS, alpha=0.80, edgecolors='none',
                       vmin=_vmin, vmax=_vmax)
        mappable = hb
    else:
        mappable = ax.scatter(d_lons, d_lats, c=d_temps, s=180, cmap='OrRd',
                              vmin=_vmin, vmax=_vmax,
                              alpha=0.85, edgecolors='black', linewidth=1.5, zorder=5)

    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(
        r'$\overline{T}$ period-mean air temperature (°C)' +
        f'\ntime-window avg: {MAP_START_HOUR:02d}:00–{MAP_END_HOUR:02d}:00 LT'
        f'\n{DATE_START} to {DATE_END}',
        fontsize=FS_LABEL, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=FS_TICK)
    vmin_use, vmax_use = mappable.get_clim()

    nat_dt    = pd.DatetimeIndex(ds_netatmo.time.values)
    nat_dates = np.array([d.date() for d in nat_dt])
    nat_hours = nat_dt.hour.values
    nat_tmask = ((nat_dates >= start_date) & (nat_dates <= end_date) &
                 (nat_hours >= MAP_START_HOUR) & (nat_hours < MAP_END_HOUR))

    nat_temp = ds_netatmo[NETATMO_TEMP_VAR].values

    norm = Normalize(vmin=vmin_use, vmax=vmax_use)
    cmap = plt.cm.YlOrRd

    groups = [('s', urban_stations, 'Urban Netatmo'),
              ('^', rural_stations, 'Rural Netatmo')]

    for marker, stations, label in groups:
        n_with_data = 0
        for st in stations:
            temps = nat_temp[nat_tmask, st['index']] if nat_temp.ndim == 2 \
                else nat_temp[nat_tmask]
            temps = temps[~np.isnan(temps)]
            if len(temps) == 0:
                continue
            n_with_data += 1
            ax.scatter(st['lon'], st['lat'], marker=marker,
                       s=NETATMO_MARKER_SIZE, c=[cmap(norm(np.mean(temps)))],
                       alpha=NETATMO_MARKER_ALPHA, edgecolors='black',
                       linewidth=NETATMO_EDGE_WIDTH, zorder=10)
        ax.scatter([], [], marker=marker, s=NETATMO_MARKER_SIZE, c='gray',
                   edgecolors='black', linewidth=NETATMO_EDGE_WIDTH,
                   label=f'{label} (n={n_with_data})')

    ax.legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor='black')

    ax.set_xlabel('Longitude (°E)', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=FS_LABEL, fontweight='bold')
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='white')

    ax.text(0.99, 0.01, '© OpenStreetMap', transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

#%% ------------------------- BUILD COMBINED FIGURE -------------------------

print("Building combined figure...")
fig = plt.figure(figsize=FIGSIZE, dpi=100)

# Two-group nested layout:
#   outer [left_group | right_group]  — controls the gap between the two panels
#   left_gs  [ridge KDEs | ridge cbar]
#   right_gs [map        | map cbar  ]
# Both colorbars live in the same outer row → identical height.
outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[1, 1],
    wspace=0.30, left=0.15, right=0.97,
    top=0.95, bottom=0.08,
)

left_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[0], width_ratios=[1, 0.07], wspace=0.05)
right_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1], width_ratios=[1, 0.07], wspace=0.05)

# Panel a) ridge KDEs
ridge_axes = draw_ridge_panel(fig, left_gs[0])

# Ridge colorbar
ax_ridge_cb = fig.add_subplot(left_gs[1])
cbar_ridge = fig.colorbar(ridge_scalar, cax=ax_ridge_cb)
cbar_ridge.set_label(ridge_cbar_lbl, fontsize=FS_RIDGE + 1, fontweight='bold', labelpad=8)
cbar_ridge.ax.tick_params(labelsize=FS_RIDGE)

# Panel b) map + its colorbar
ax_map    = fig.add_subplot(right_gs[0])
ax_map_cb = fig.add_subplot(right_gs[1])
draw_map_panel(fig, ax_map, ax_map_cb)

# Boxed panel labels
label_box = dict(boxstyle='square,pad=0.3', facecolor='white',
                 edgecolor='black', linewidth=1.2)
fig.text(0.025, 0.955, 'a)', fontsize=FS_PANEL, fontweight='bold',
         va='top', ha='left', bbox=label_box)

map_pos = ax_map.get_position()
fig.text(map_pos.x0 - 0.05, 0.955, 'b)', fontsize=FS_PANEL,
         fontweight='bold', va='top', ha='left', bbox=label_box)

out_base = os.path.join(OUTPUT_DIR,
    f'combined_{ridge_mode_tag}_ridge_map_{DATE_START}_to_{DATE_END}_{MAP_START_HOUR:02d}-{MAP_END_HOUR:02d}h')
plt.savefig(f'{out_base}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out_base}.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print(f"Saved: {out_base}.png/.pdf")
