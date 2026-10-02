#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-10-01 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/meteotracker_vs_netatmo/MT_vs_netAtmo_figF_avgrose_20260916.py
# Paper figure:       scatter_MT_netATMO_station_deviation.png (figF_4windows_rosemap_avgrose_* renamed)
# UPDATE (2026-10-01): scatter panels use the same tick positions on both axes (the
# evening-transition panel had x ticks every 2.5 °C but y ticks every 2 °C); grey
# 'other station' points darker with a thin black outline; map: each highlighted station
# drawn as a dot coloured by its mean bias (map colourbar, range now incl. these stations),
# dashed leader from each displaced rose centre to that dot, roses kept inside the map
# frame; output written to UHI-Bolzano/figures/.
# ─────────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
MeteoTracker vs Netatmo — 4-period scatter + bias-rose city map (Figure F only)
=================================================================

VARIANT (2026-09-16): forked from MT_vs_netAtmo_4windows_rosemap_20260612.py.
  (1) Figure G (the standalone city-map figure) is dropped entirely -- this
      script only produces Figure F (scatter panels + rose map + legend).
  (2) The legend panel's polar rose no longer plots illustrative example
      values -- it now plots the actual mean bias (MT-Netatmo) of the TOP_N
      stations for each time window (mean of bias_wide_v over top_ids), so it
      reads as a real "typical swinging station" profile alongside the
      per-station list below it.

Layout
------
  [Morning transition]  [Day      ]   |
  [06-09 h scatter   ]  [09-18 h  ]   |   big precise city-map panel:
  -------------------------------     |     - all Netatmo stations (dots)
  [Evening transition]  [Night    ]   |     - top-N "swinging" stations get
  [18-21 h scatter   ]  [21-24 h  ]   |       a 4-petal "bias rose":
                                       |         N = morning, E = day,
                                       |         S = evening, W = night
                                       |         petal length = |bias|
                                       |         petal colour = sign
                                       |         (red = MT warmer, blue = MT
                                       |          cooler than Netatmo)

The 4 scatter panels reuse the same outlier-station definition as
MT_vs_netAtmo_outlier_stations_20260612.py (stations ranked by bias_range =
max-min period-mean bias), now computed over 4 wider, physically-motivated
periods instead of 6. The rose map lets you see, at a glance and at each
station's real location, *which* time-of-day period drives its deviation
from the 1:1 line — the key step for relating it to local siting (sun
exposure, pavement/facade thermal inertia, shading, etc.).
"""

# %% Imports

import os
import math
import time
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr
import requests
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator

# %% Configuration — MODIFY HERE

METEOTRACKER_FILE = "/home/gsimonet/Desktop/Meteotrackers_package/meteotracker_incremental/netcdf_files/meteotracker_summer_2025.nc"
NETATMO_FILE      = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
STATION_LIST_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/station_list/stations_20250131_160200.csv'

LAT_MIN, LAT_MAX = 46.4, 46.6
LON_MIN, LON_MAX = 11.1, 11.6

DATE_START = '2025-07-01'
DATE_END   = '2025-08-01'

TIME_WINDOW_MIN = 30   # +/- minutes around each Netatmo timestamp
RADIUS_KM       = 0.5
MIN_MT_POINTS   = 3    # min MT points within radius/time-window to form a pair

# 4 physically-motivated periods covering the full day (06h -> next 06h):
# (display label, key, hour_start inclusive, hour_end exclusive)
TIME_WINDOWS = [
    ('Morning transition\n06-09 h', 'morning', 6,  9),
    ('Day\n09-18 h',                'day',     9, 18),
    ('Evening transition\n18-21 h', 'evening', 18, 21),
    ('Night\n21-24 h',              'night',   21, 24),
]
TRANSITION_KEYS = {'morning', 'evening'}
OTHER_KEYS      = {'day', 'night'}

# 6 finer-grained slots, used only for the scatter panels (the bias-rose /
# ranking pipeline above keeps using the 4 merged TIME_WINDOWS)
SCATTER_TIME_WINDOWS = [
    ('Morning transition\n06-09 h', '06-09', 6,  9),
    ('09-12 h',                      '09-12', 9, 12),
    ('12-15 h',                      '12-15', 12, 15),
    ('15-18 h',                      '15-18', 15, 18),
    ('Evening transition\n18-21 h', '18-21', 18, 21),
    ('Night\n21-24 h',               '21-24', 21, 24),
]
SCATTER_TRANSITION_KEYS = {'06-09', '18-21'}

# Scatter layout in figure F:
#   6 -> 3x2 grid using the finer SCATTER_TIME_WINDOWS slots above
#   4 -> 2x2 grid using the same 4 merged periods as the bias roses (TIME_WINDOWS)
SCATTER_WINDOW_MODE = 4

MIN_PAIRS_PER_WINDOW = 10   # min pairs in a (station, window) cell to trust its bias
MIN_TOTAL_PAIRS      = 30   # min total pairs for a station to enter the ranking
TOP_N                = 6    # number of "worst swinging" stations to highlight

ZOOM         = 16    # initial basemap zoom (auto-reduced if the tile count is too big)
ROSE_SIZE_IN   = 0.72  # bias-rose glyph diameter, in physical inches (kept square
                        # regardless of the figure's aspect ratio)

OUTPUT_DIR = 'outlier_station_diagnostics'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TAG = f'{DATE_START}_{DATE_END}'

# %% Helper: haversine distance (km)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

# %% Load MeteoTracker

ds_mt  = xr.open_dataset(METEOTRACKER_FILE)
t_raw  = ds_mt.time.values
la_raw = ds_mt.latitude.values
lo_raw = ds_mt.longitude.values
T_raw  = ds_mt['T0'].values

valid  = ~(np.isnan(la_raw) | np.isnan(lo_raw) | np.isnan(T_raw))
t_raw, la_raw, lo_raw, T_raw = t_raw[valid], la_raw[valid], lo_raw[valid], T_raw[valid]

geo     = ((la_raw >= LAT_MIN) & (la_raw <= LAT_MAX) &
           (lo_raw >= LON_MIN) & (lo_raw <= LON_MAX))
mt_lat  = la_raw[geo];  mt_lon  = lo_raw[geo]
mt_temp = T_raw[geo];   mt_time = t_raw[geo]

mt_dt    = pd.DatetimeIndex(mt_time)
mt_epoch = mt_dt.view('int64') / 1e9
mt_hours = mt_dt.hour.values
mt_dates = mt_dt.date

print(f"MT points loaded (in bounds): {len(mt_temp):,}")

# %% Load Netatmo (keep per-station identity: id, lat, lon, altitude)

ds_na   = xr.open_dataset(NETATMO_FILE)
na_lats = ds_na.latitude.values
na_lons = ds_na.longitude.values
na_alts = ds_na.altitude.values
na_ids  = ds_na.station.values.astype(str)

for vname in ('T_lvl4', 'T0', 'temperature'):
    if vname in ds_na:
        na_T = ds_na[vname].values   # shape (time, station)
        print(f"Netatmo variable: '{vname}', shape: {na_T.shape}")
        break

na_time  = ds_na.time.values
na_dt    = pd.DatetimeIndex(na_time)
na_epoch = na_dt.view('int64') / 1e9
na_hours = na_dt.hour.values
na_dates = na_dt.date

n_stations = len(na_ids)
print(f"Netatmo stations: {n_stations}")

# %% Station metadata (street / city, for the console report)

station_meta = {}
try:
    meta_df = pd.read_csv(STATION_LIST_FILE)
    meta_df['station_id'] = meta_df['station_id'].str.lower()
    station_meta = meta_df.set_index('station_id')[['city', 'street']].to_dict('index')
except Exception as exc:
    print(f"Could not load station metadata ({exc}); proceeding without street/city info.")

def meta_for(station_id):
    m = station_meta.get(station_id.lower(), {})
    return m.get('city', 'n/a'), m.get('street', 'n/a')


# Italian/German generic street-type words -> standard British-English equivalents.
# Prefixes are Italian-style (type word comes before the name, e.g. 'Via Roma'),
# suffixes are German-style compounds (type word glued to the end, e.g. 'Romstraße').
_STREET_TYPE_PREFIXES = {
    'via':    'Street',
    'viale':  'Avenue',
    'piazza': 'Square',
    'corso':  'Avenue',
    'strada': 'Road',
    'salita': 'Rise',
}
_STREET_TYPE_SUFFIXES = [   # checked longest-first, case-insensitive
    ('straße', 'Street'), ('strasse', 'Street'),
    ('allee', 'Avenue'),
    ('gasse', 'Lane'),
    ('platz', 'Square'),
    ('ufer', 'Embankment'),
    ('autobahn', 'Motorway'),
    ('weg', 'Way'),
]

def _to_english(name):
    """Translate the Italian/German street-type word to English (UK); e.g.
    'Via Innsbruck' -> 'Innsbruck Street', 'Innsbrucker Straße' -> 'Innsbrucker
    Street', 'Brennerautobahn' -> 'Brenner Motorway'. Leaves anything else
    (proper names, 'unnamed road', etc.) unchanged."""
    if name.lower().startswith('lungo '):
        return f'{name[6:].strip()} Embankment'
    head, _, tail = name.partition(' ')
    if tail and head.lower() in _STREET_TYPE_PREFIXES:
        return f'{tail} {_STREET_TYPE_PREFIXES[head.lower()]}'
    lname = name.lower()
    for suffix, english in _STREET_TYPE_SUFFIXES:
        if lname.endswith(suffix) and len(name) > len(suffix):
            stem = name[:-len(suffix)].replace('-', ' ').strip()
            return f'{stem} {english}' if stem else english
    return name

def _short(name):
    """Bilingual street/city names are 'Italian - German'; keep only the first,
    translated to English (UK) street-type wording."""
    return _to_english(name.split(' - ')[0].strip())

# Stations without a street name in the Netatmo station list, labelled by hand
STATION_LABEL_OVERRIDES = {
    '70:ee:50:52:db:3a': 'San Quirino district',   # 46.4969 N, 11.3353 E; Gries-San Quirino (OSM)
}

def display_name(row):
    """Human-readable station label (street/city) — never the raw MAC-like ID."""
    street, city = row['street'], row['city']
    street_ok = isinstance(street, str) and street not in ('Unknown', 'n/a')
    city_ok   = isinstance(city, str) and city not in ('Unknown', 'n/a')
    if street_ok and city_ok:
        return f"{_short(street)}, {_short(city)}"
    if street_ok:
        return _short(street)
    if city_ok:
        return _short(city)
    return f"Station (alt {row['alt']:.0f} m)"

# %% Date bounds

start_d = datetime.datetime.strptime(DATE_START, '%Y-%m-%d').date()
end_d   = datetime.datetime.strptime(DATE_END,   '%Y-%m-%d').date()
tw_sec  = TIME_WINDOW_MIN * 60.0

mt_date_mask = (mt_dates >= start_d) & (mt_dates <= end_d)

# %% Build matched pairs for every period, keeping station identity

def build_pairs(h_start, h_end, key, label):
    na_mask = ((na_dates >= start_d) & (na_dates <= end_d) &
               (na_hours >= h_start) & (na_hours < h_end))
    mt_mask = (mt_date_mask & (mt_hours >= h_start) & (mt_hours < h_end))

    na_ep = na_epoch[na_mask]
    na_Tp = na_T[na_mask, :]
    na_hp = na_hours[na_mask]

    ml, mo, mT, me = mt_lat[mt_mask], mt_lon[mt_mask], mt_temp[mt_mask], mt_epoch[mt_mask]

    pairs = []
    for ti, (na_ts, na_h) in enumerate(zip(na_ep, na_hp)):
        t_close = np.abs(me - na_ts) <= tw_sec
        if not t_close.any():
            continue
        ml_w, mo_w, mT_w = ml[t_close], mo[t_close], mT[t_close]

        for sidx in range(n_stations):
            na_val = na_Tp[ti, sidx]
            if np.isnan(na_val):
                continue
            dists = haversine(na_lats[sidx], na_lons[sidx], ml_w, mo_w)
            in_r  = dists <= RADIUS_KM
            if in_r.sum() < MIN_MT_POINTS:
                continue
            pairs.append({
                'window'     : key,
                'window_label': label,
                'hour'       : int(na_h),
                'station_id' : na_ids[sidx],
                'lat'        : na_lats[sidx],
                'lon'        : na_lons[sidx],
                'alt'        : na_alts[sidx],
                'netatmo_T'  : na_val,
                'mt_T'       : mT_w[in_r].mean(),
                'n_mt'       : int(in_r.sum()),
                'distance_km': dists[in_r].mean(),
            })
    return pd.DataFrame(pairs)

frames = []
for label, key, h0, h1 in TIME_WINDOWS:
    print(f"  Matching {label.splitlines()[0]} …", end=' ', flush=True)
    df = build_pairs(h0, h1, key, label)
    print(f"{len(df)} pairs")
    frames.append(df)

all_pairs = pd.concat(frames, ignore_index=True)
all_pairs['residual'] = all_pairs['mt_T'] - all_pairs['netatmo_T']  # MT - Netatmo
print(f"\nTotal matched pairs: {len(all_pairs)}")

# Same matching, but on the 6 finer slots -- used only for the scatter panels
scatter_frames = []
for label, key, h0, h1 in SCATTER_TIME_WINDOWS:
    print(f"  Matching {label.splitlines()[0]} …", end=' ', flush=True)
    df = build_pairs(h0, h1, key, label)
    print(f"{len(df)} pairs")
    scatter_frames.append(df)

scatter_pairs = pd.concat(scatter_frames, ignore_index=True)
scatter_pairs['residual'] = scatter_pairs['mt_T'] - scatter_pairs['netatmo_T']

# %% Per-station, per-period bias / RMSE

window_keys = [k for _, k, _, _ in TIME_WINDOWS]

g = all_pairs.groupby(['station_id', 'window'])['residual']
cell_stats = g.agg(n='count', bias='mean', std='std',
                    rmse=lambda x: np.sqrt((x ** 2).mean())).reset_index()

bias_wide = cell_stats.pivot(index='station_id', columns='window', values='bias').reindex(columns=window_keys)
n_wide    = cell_stats.pivot(index='station_id', columns='window', values='n').reindex(columns=window_keys).fillna(0)
rmse_wide = cell_stats.pivot(index='station_id', columns='window', values='rmse').reindex(columns=window_keys)

valid_cell  = n_wide >= MIN_PAIRS_PER_WINDOW
bias_wide_v = bias_wide.where(valid_cell)

# %% Overall per-station stats + bias_range ranking

overall = all_pairs.groupby('station_id')['residual'].agg(
    n_total='count',
    bias_all='mean',
    mae_all=lambda x: x.abs().mean(),
    rmse_all=lambda x: np.sqrt((x ** 2).mean()),
)

n_valid_windows = valid_cell.sum(axis=1)
bias_range      = (bias_wide_v.max(axis=1) - bias_wide_v.min(axis=1))

transition_cols = [k for k in window_keys if k in TRANSITION_KEYS]
other_cols      = [k for k in window_keys if k in OTHER_KEYS]
transition_bias = bias_wide_v[transition_cols].mean(axis=1)
other_bias      = bias_wide_v[other_cols].mean(axis=1)
transition_excess = transition_bias - other_bias

summary = overall.copy()
summary['lat'] = all_pairs.groupby('station_id')['lat'].first()
summary['lon'] = all_pairs.groupby('station_id')['lon'].first()
summary['alt'] = all_pairs.groupby('station_id')['alt'].first()
summary['city'], summary['street'] = zip(*[meta_for(sid) for sid in summary.index])
summary['n_valid_windows'] = n_valid_windows
summary['bias_range']      = bias_range
summary['transition_bias'] = transition_bias
summary['other_bias']      = other_bias
summary['transition_excess'] = transition_excess

for k in window_keys:
    summary[f'bias_{k}'] = bias_wide[k]
    summary[f'n_{k}']    = n_wide[k]

rankable = summary[(summary['n_valid_windows'] >= 2) &
                    (summary['n_total'] >= MIN_TOTAL_PAIRS)].copy()
rankable = rankable.sort_values('bias_range', ascending=False)
summary['rank'] = np.nan
summary.loc[rankable.index, 'rank'] = np.arange(1, len(rankable) + 1)

top_ids = rankable.index[:TOP_N].tolist()

# %% Save CSV

csv_path = os.path.join(OUTPUT_DIR, f'station_bias_4windows_{TAG}.csv')
summary.sort_values('bias_range', ascending=False).round(3).to_csv(csv_path)
print(f"\nSaved per-station bias table: {csv_path}")

# %% Console report

print("\n" + "=" * 78)
print(f"Top {TOP_N} stations by bias_range (max-min period-mean bias, MT-Netatmo)")
print("=" * 78)
for sid in top_ids:
    row = summary.loc[sid]
    print(f"\n#{int(row['rank'])}  {sid}  —  {row['street']}, {row['city']}  (alt {row['alt']:.0f} m)")
    print(f"    n_total={int(row['n_total'])}  bias_all={row['bias_all']:+.2f} C  "
          f"rmse_all={row['rmse_all']:.2f} C  bias_range={row['bias_range']:.2f} C")
    print(f"    transition (morning+evening) mean bias = {row['transition_bias']:+.2f} C  |  "
          f"other (day+night) mean bias = {row['other_bias']:+.2f} C  |  "
          f"excess = {row['transition_excess']:+.2f} C")
    win_str = "  ".join(
        f"{k}:{bias_wide_v.loc[sid, k]:+.2f}(n={int(n_wide.loc[sid, k])})"
        if valid_cell.loc[sid, k] else f"{k}: n/a (n={int(n_wide.loc[sid, k])})"
        for k in window_keys)
    print(f"    per-period bias (MT-Netatmo, C): {win_str}")

# %% Basemap tile download (OpenTopoMap, precise crop around the stations)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    return (int((lon_deg + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_deg = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lat_deg, lon_deg

def download_basemap(lat_min, lat_max, lon_min, lon_max, zoom, max_tiles=16):
    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)
    while (x_max - x_min + 1) * (y_max - y_min + 1) > max_tiles and zoom > 10:
        zoom -= 1
        x_min, y_max = deg2num(lat_min, lon_min, zoom)
        x_max, y_min = deg2num(lat_max, lon_max, zoom)

    ts = 256
    img = Image.new('RGB', ((x_max - x_min + 1) * ts, (y_max - y_min + 1) * ts), (240, 240, 240))
    headers = {'User-Agent': 'MeteoTracker Scientific Visualization (research use)'}
    n_ok = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            try:
                r = requests.get(f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png',
                                 headers=headers, timeout=10)
                if r.status_code == 200:
                    img.paste(Image.open(BytesIO(r.content)), ((x - x_min) * ts, (y - y_min) * ts))
                    n_ok += 1
                time.sleep(0.12)
            except Exception:
                pass

    lat_top, lon_left   = num2deg(x_min, y_min, zoom)
    lat_bot, lon_right  = num2deg(x_max + 1, y_max + 1, zoom)
    print(f"Basemap: {n_ok} tiles downloaded at zoom {zoom}")
    return img, lat_bot, lat_top, lon_left, lon_right

# Precise bounding box: the station cluster + a small margin
pad = 0.12
lon_span = summary['lon'].max() - summary['lon'].min()
lat_span = summary['lat'].max() - summary['lat'].min()
map_lon_min = summary['lon'].min() - pad * lon_span
map_lon_max = summary['lon'].max() + pad * lon_span
map_lat_min = summary['lat'].min() - pad * lat_span
map_lat_max = summary['lat'].max() + pad * lat_span
lat_center  = (map_lat_min + map_lat_max) / 2

basemap_img, mlat_min, mlat_max, mlon_min, mlon_max = download_basemap(
    map_lat_min, map_lat_max, map_lon_min, map_lon_max, ZOOM)

# %% Shared helpers for the rose-map panels

palette  = plt.cm.tab10(np.linspace(0, 1, 10))[:TOP_N]
color_of = {sid: palette[i] for i, sid in enumerate(top_ids)}

# Square-root radial scale for the bias roses: a few stations swing by ~8 C
# in one period while most periods/stations differ by <1 C. A linear radius
# would make the small (but still meaningful) petals invisible.
R_MAX = np.ceil(np.nanmax(np.abs(bias_wide_v.values)))
theta = [0, np.pi / 2, np.pi, 3 * np.pi / 2]   # N, E, S, W = morning, day, evening, night
petal_width = 2 * np.pi / 4 * 0.85

def petal_radius(abs_bias):
    return R_MAX * np.sqrt(np.minimum(abs_bias, R_MAX) / R_MAX)

# All Netatmo stations (incl. ones never matched to an MT point), with their
# overall mean bias (MT-Netatmo) where available -- used to colour the
# "other" (non-top-N) stations on the map
all_stations = pd.DataFrame({'station_id': na_ids, 'lat': na_lats,
                              'lon': na_lons, 'alt': na_alts}).set_index('station_id')
all_stations['bias_all'] = summary['bias_all']

other_ids = [sid for sid in all_stations.index if sid not in top_ids]
BIAS_CMAP = 'RdBu_r'   # blue = MT cooler, red = MT warmer (matches the rose petals)
bias_lim  = np.nanmax(np.abs(all_stations['bias_all']))   # incl. the top-N, whose dots use the same colours

def data_to_fig(ax, fig, x, y):
    disp = ax.transData.transform((x, y))
    return fig.transFigure.inverted().transform(disp)

def draw_station_map(ax_map, fig, rose_size_in, title):
    """Basemap + 'other' stations coloured by mean bias (with colourbar) +
    per-period bias roses for the top-N stations (with collision avoidance)."""

    gray = ImageEnhance.Contrast(basemap_img.convert('L')).enhance(1.1)
    ax_map.imshow(gray, extent=(mlon_min, mlon_max, mlat_min, mlat_max),
                  cmap='gray', alpha=0.8, vmin=80, vmax=255, zorder=0)
    ax_map.set_xlim(map_lon_min, map_lon_max)
    ax_map.set_ylim(map_lat_min, map_lat_max)
    ax_map.set_aspect(1.0 / np.cos(np.radians(lat_center)))

    # "other" stations: coloured by their overall mean bias (MT-Netatmo)
    oth = all_stations.loc[other_ids]
    has_bias = oth['bias_all'].notna()
    sc = ax_map.scatter(oth.loc[has_bias, 'lon'], oth.loc[has_bias, 'lat'],
                        c=oth.loc[has_bias, 'bias_all'], cmap=BIAS_CMAP,
                        vmin=-bias_lim, vmax=bias_lim, s=42,
                        edgecolors='k', linewidths=0.4, zorder=2)

    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.030, pad=0.02)
    cbar.set_label('mean bias, all periods (MT-Netatmo) [°C]', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_map.set_xlabel('Longitude (°E)', fontsize=10)
    ax_map.set_ylabel('Latitude (°N)', fontsize=10)
    ax_map.set_title(title, fontsize=11, fontweight='bold', pad=14)
    ax_map.tick_params(labelsize=8)
    ax_map.text(0.99, 0.01, '© OpenStreetMap contributors', transform=ax_map.transAxes,
                 fontsize=7, ha='right', va='bottom', style='italic',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # set_aspect() and the colourbar both resize/relocate the axes bbox; force
    # a draw now so transData reflects the *final* position before placing roses
    fig.canvas.draw()

    fig_w_in, fig_h_in = fig.get_figwidth(), fig.get_figheight()
    rose_w = rose_size_in / fig_w_in   # figure-fraction width
    rose_h = rose_size_in / fig_h_in   # figure-fraction height -- kept so the
                                        # rose is a physical square (circle) on
                                        # the page regardless of figure aspect

    def frac_to_in(p):
        return np.array([p[0] * fig_w_in, p[1] * fig_h_in])

    def in_to_frac(p):
        return np.array([p[0] / fig_w_in, p[1] / fig_h_in])

    # The top-N stations are often close together; push overlapping rose
    # centres apart (in physical inches) and keep a thin dashed leader line
    # back to the station's true location whenever a rose had to be displaced.
    true_pos_in = {sid: frac_to_in(data_to_fig(ax_map, fig, all_stations.loc[sid, 'lon'], all_stations.loc[sid, 'lat']))
                   for sid in top_ids}
    rose_pos_in = {sid: p.copy() for sid, p in true_pos_in.items()}

    # Map frame in inches: rose centres are kept at least one rose radius inside it
    bb = ax_map.get_position()
    r_in = rose_size_in / 2
    frame_lo = np.array([bb.x0 * fig_w_in + r_in, bb.y0 * fig_h_in + r_in])
    frame_hi = np.array([bb.x1 * fig_w_in - r_in, bb.y1 * fig_h_in - r_in])

    min_dist_in = rose_size_in * 1.05
    for _ in range(300):
        moved = False
        for i, sid1 in enumerate(top_ids):
            for sid2 in top_ids[i + 1:]:
                d = rose_pos_in[sid2] - rose_pos_in[sid1]
                dist = np.hypot(*d)
                if dist < min_dist_in:
                    direction = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    push = (min_dist_in - dist) / 2
                    rose_pos_in[sid1] -= direction * push
                    rose_pos_in[sid2] += direction * push
                    moved = True
        for sid in top_ids:
            clamped = np.clip(rose_pos_in[sid], frame_lo, frame_hi)
            if not np.allclose(clamped, rose_pos_in[sid]):
                rose_pos_in[sid] = clamped
                moved = True
        if not moved:
            break

    for sid in top_ids:
        row = summary.loc[sid]
        fx, fy = in_to_frac(rose_pos_in[sid])
        tx, ty = in_to_frac(true_pos_in[sid])

        # Dashed leader from the rose centre to the station's true location whenever
        # the rose was displaced
        if np.hypot(*(rose_pos_in[sid] - true_pos_in[sid])) > 0.01:
            leader = mlines.Line2D([tx, fx], [ty, fy], transform=fig.transFigure,
                                    color=color_of[sid], lw=1.4, ls='--', zorder=10)
            fig.add_artist(leader)

        # The station itself: dot coloured by its mean bias (same colourbar as the
        # other stations), always drawn on top
        station_dot = mlines.Line2D([tx], [ty], transform=fig.transFigure, marker='o',
                                    markersize=8,
                                    markerfacecolor=sc.cmap(sc.norm(row['bias_all'])),
                                    markeredgecolor='k', markeredgewidth=1.0,
                                    linestyle='none', zorder=20)
        fig.add_artist(station_dot)

        heights, colors = [], []
        for wk in window_keys:
            b = bias_wide_v.loc[sid, wk]
            if pd.isna(b):
                heights.append(0.0)
                colors.append('#d9d9d9')
            else:
                heights.append(petal_radius(abs(b)))
                colors.append('#d73027' if b > 0 else '#4575b4')

        rax = fig.add_axes([fx - rose_w / 2, fy - rose_h / 2, rose_w, rose_h], projection='polar')
        rax.set_theta_zero_location('N')
        rax.set_theta_direction(-1)
        rax.set_ylim(0, R_MAX)
        rax.set_xticks([]); rax.set_yticks([])
        rax.spines['polar'].set_visible(False)
        rax.patch.set_alpha(0)

        rax.bar(theta, heights, width=petal_width, bottom=0, color=colors,
                edgecolor='white', linewidth=0.6, zorder=3)
        ring_theta = np.linspace(0, 2 * np.pi, 100)
        rax.plot(ring_theta, [R_MAX] * 100, color=color_of[sid], lw=2.8, zorder=4)

def draw_legend_panel(ax_legend, fig):
    """Rose-scale legend + top-N station list.

    The legend's polar rose plots the actual mean bias (MT-Netatmo) of the
    TOP_N stations for each time window -- i.e. bias_wide_v.loc[top_ids,
    window_keys].mean(axis=0) -- rather than illustrative example values, so
    it summarizes the top-N group's typical diurnal bias profile.
    """

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis('off')

    legend_pos = ax_legend.get_position()
    fig_w_in, fig_h_in = fig.get_figwidth(), fig.get_figheight()

    # Rose-legend description text (axes-fraction coords, top of the panel --
    # placed as text rather than a sub-title so it can't collide with
    # ax_legend's own title above the axes box)
    ax_legend.text(0.5, 0.985,
                    r'Petal length $\propto\ \sqrt{|bias|}$, up to ' f'{R_MAX:.0f}°C   |   '
                    'red = MT warmer, blue = MT cooler\n'
                    f'(rose below: mean bias across the top {TOP_N} stations)',
                    ha='center', va='top', fontsize=9, transform=ax_legend.transAxes,
                    wrap=True)

    # Rose legend (polar), kept square, just below that description
    lax_h = legend_pos.height * 0.30
    lax_w = lax_h * fig_h_in / fig_w_in
    lx0 = legend_pos.x0 + (legend_pos.width - lax_w) / 2
    ly0 = legend_pos.y0 + legend_pos.height * 0.50
    lax = fig.add_axes([lx0, ly0, lax_w, lax_h], projection='polar')
    lax.set_theta_zero_location('N')
    lax.set_theta_direction(-1)
    lax.set_ylim(0, R_MAX)

    avg_bias    = bias_wide_v.loc[top_ids, window_keys].mean(axis=0, skipna=True)
    avg_heights = [petal_radius(abs(b)) if pd.notna(b) else 0.0 for b in avg_bias]
    avg_colors  = ['#d73027' if (pd.notna(b) and b > 0) else '#4575b4' for b in avg_bias]
    lax.bar(theta, avg_heights, width=petal_width, bottom=0,
            color=avg_colors, edgecolor='white', linewidth=0.6)
    lax.set_xticks(theta)
    lax.set_xticklabels(['Morning\n06-09h', 'Day\n09-18h', 'Evening\n18-21h', 'Night\n21-24h'], fontsize=8)
    # non-uniform radial scale (sqrt of |bias|): label gridlines in actual °C
    y_vals  = np.array([0.5, 1, 2, 4, R_MAX])
    y_ticks = petal_radius(y_vals)
    lax.set_yticks(y_ticks)
    lax.set_yticklabels([f'{v:.1f}' if v < 1 else f'{v:.0f}' for v in y_vals], fontsize=7)

    # Station list in the lower part of the panel
    row_h = 0.42 / TOP_N
    for i, sid in enumerate(top_ids):
        row = summary.loc[sid]
        y = 0.42 - (i + 0.5) * row_h
        ax_legend.scatter([0.035], [y], s=220, facecolors='none', edgecolors=color_of[sid],
                          linewidths=1.8, zorder=3, clip_on=False)
        street_ok = isinstance(row['street'], str) and row['street'] not in ('Unknown', 'n/a')
        if sid in STATION_LABEL_OVERRIDES:
            name_label = STATION_LABEL_OVERRIDES[sid]
        else:
            name_label = _short(row['street']) if street_ok else display_name(row)
        ax_legend.text(0.095, y, name_label, ha='left', va='center', fontsize=10)
        ax_legend.text(0.97, y, f"mean {row['bias_all']:+.1f}°C  |  range {row['bias_range']:.1f}°C",
                        ha='right', va='center', fontsize=8.5, style='italic', color='dimgrey')

# %% Figure F — scatter panels (4 or 6, see SCATTER_WINDOW_MODE), rose map, and a legend panel

if SCATTER_WINDOW_MODE == 6:
    scatter_windows, scatter_data, scatter_trans_keys = SCATTER_TIME_WINDOWS, scatter_pairs, SCATTER_TRANSITION_KEYS
    n_scatter_rows = 3
else:
    scatter_windows, scatter_data, scatter_trans_keys = TIME_WINDOWS, all_pairs, TRANSITION_KEYS
    n_scatter_rows = 2

fig_h = 16 if n_scatter_rows == 2 else 21
fig = plt.figure(figsize=(13, fig_h))
gs  = gridspec.GridSpec(n_scatter_rows + 1, 2, wspace=0.30, hspace=0.42,
                         left=0.075, right=0.975, top=0.94 if n_scatter_rows == 2 else 0.955,
                         bottom=0.045 if n_scatter_rows == 2 else 0.035)

# ---- scatter panels -------------------------------------------------------

for s_i, (label, key, h0, h1) in enumerate(scatter_windows):
    r_i, c_i = divmod(s_i, 2)
    ax = fig.add_subplot(gs[r_i, c_i])

    df = scatter_data[scatter_data['window'] == key]
    if len(df) < 2:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label, fontsize=11, fontweight='bold')
        continue

    tmin = min(df['netatmo_T'].min(), df['mt_T'].min()) - 0.5
    tmax = max(df['netatmo_T'].max(), df['mt_T'].max()) + 0.5

    others = df[~df['station_id'].isin(top_ids)]
    ax.scatter(others['netatmo_T'], others['mt_T'],
               color='#a6a6a6', s=40, alpha=0.6, edgecolors='k', linewidths=0.3, zorder=1)

    for sid in top_ids:
        sub = df[df['station_id'] == sid]
        if len(sub) == 0:
            continue
        ax.scatter(sub['netatmo_T'], sub['mt_T'],
                   color=color_of[sid], s=45, alpha=0.85,
                   edgecolors='k', linewidths=0.4, zorder=3)

    ax.plot([tmin, tmax], [tmin, tmax], color='grey', ls=':', lw=1.5, zorder=0)

    bias = df['residual'].mean()
    rmse = np.sqrt((df['residual'] ** 2).mean())
    r    = df['mt_T'].corr(df['netatmo_T'])

    ax.set_xlim(tmin, tmax)
    ax.set_ylim(tmin, tmax)
    ax.set_aspect('equal')
    # Same tick positions on both axes (otherwise matplotlib may pick e.g. 2.5 °C
    # steps on x and 2 °C steps on y, because the x tick labels are wider)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.set_xlabel('Netatmo T (°C)', fontsize=10)
    ax.set_ylabel('MeteoTracker T (°C)', fontsize=10)
    ax.set_title(f'{label}   (n={len(df)})\nr={r:.3f}, RMSE={rmse:.2f}°C, Bias={bias:+.2f}°C',
                  fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    if key in scatter_trans_keys:
        ax.text(0.02, 0.97, 'transition period', transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='#b35806', ha='left', va='top')

# ---- Map panel + legend panel ----------------------------------------------

ax_map = fig.add_subplot(gs[n_scatter_rows, 0])
draw_station_map(ax_map, fig, ROSE_SIZE_IN, '')

ax_legend = fig.add_subplot(gs[n_scatter_rows, 1])
draw_legend_panel(ax_legend, fig)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')   # UHI-Bolzano/figures
figF_path = os.path.join(FIGURES_DIR, 'scatter_MT_netATMO_station_deviation_20261001.png')
# plt.show()
plt.savefig(figF_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {figF_path}")
print("\nDone.")
