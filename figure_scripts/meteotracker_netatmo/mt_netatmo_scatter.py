#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/meteotracker_vs_netatmo/MT_vs_netAtmo_scatter_20260218.py
# Paper figure:       scatter_MT_netATMO_station_deviation.png
# Note: LIKELY match, not an exact filename match. This script saves 'scatter_mt_netatmo_{DATE_START}_{DATE_END}.png'; no script anywhere on disk saves the literal string 'station_deviation'. This is the best candidate by content (1:1 scatter + bias/RMSE annotation = 'station deviation') and the output file is confirmed present under this name at the original location -- almost certainly manually renamed when copied into the paper's figures/ folder.
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-
"""
MeteoTracker vs Netatmo - Scatter Plot (time-window + spatial matching)
Logic:
  For each Netatmo timestep → collect MT points within ±TIME_WINDOW_MIN
  and within RADIUS_KM of the station → average MT temps → one pair per
  (station × timestep) combination.
"""

#%% Imports

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import datetime
import fiona
from shapely.geometry import Point, shape
from fiona.transform import transform_geom

#%% Configuration — MODIFY HERE

METEOTRACKER_FILE = "/home/gsimonet/Desktop/Meteotrackers_package/meteotracker_incremental/netcdf_files/meteotracker_summer_2025.nc"
NETATMO_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
URBAN_SHP = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_urban_area.shp'
RURAL_SHP = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_rural_area.shp'

# Geographic bounds
LAT_MIN, LAT_MAX = 46.4, 46.6
LON_MIN, LON_MAX = 11.1, 11.6

# Date range and hour filter
DATE_START   = '2025-07-01'
DATE_END     = '2025-08-01'
HOUR_START   = 12    # inclusive
HOUR_END     = 18    # exclusive
PERIOD_LABEL = 'Afternoon (12–18 h)'

# Matching parameters
TIME_WINDOW_MIN = 30   # ± minutes around each Netatmo timestamp to collect MT points
RADIUS_KM       = 0.5  # spatial radius around each Netatmo station

# Minimum MT points required inside a window to form a valid pair
MIN_MT_POINTS = 3

# Colour points by: 'station_type' | 'hour' | 'distance' | 'n_mt_points'
COLOR_BY = 'distance'

OUTPUT_FILE = f'scatter_mt_netatmo_{DATE_START}_{DATE_END}.png'

#%% Helper: haversine distance (km)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

#%% Load MeteoTracker data

ds_mt = xr.open_dataset(METEOTRACKER_FILE)

t_raw  = ds_mt.time.values
la_raw = ds_mt.latitude.values
lo_raw = ds_mt.longitude.values
T_raw  = ds_mt['T0'].values

valid = ~(np.isnan(la_raw) | np.isnan(lo_raw) | np.isnan(T_raw))
t_raw, la_raw, lo_raw, T_raw = t_raw[valid], la_raw[valid], lo_raw[valid], T_raw[valid]

geo = (la_raw >= LAT_MIN) & (la_raw <= LAT_MAX) & (lo_raw >= LON_MIN) & (lo_raw <= LON_MAX)
mt_lat  = la_raw[geo]
mt_lon  = lo_raw[geo]
mt_temp = T_raw[geo]
mt_time = t_raw[geo]

# Convert MT timestamps to datetime objects (and seconds since epoch for fast arithmetic)
mt_dt  = np.array([datetime.datetime.fromtimestamp(
            dt.astype('datetime64[ns]').view('int64') / 1e9) for dt in mt_time])
mt_epoch = np.array([dt.timestamp() for dt in mt_dt])   # float seconds — fast comparison
mt_hours  = np.array([dt.hour for dt in mt_dt])
mt_dates  = np.array([dt.date() for dt in mt_dt])

print(f"MT points loaded (in bounds): {len(mt_temp):,}")

#%% Load Netatmo data

ds_na = xr.open_dataset(NETATMO_FILE)

na_lats = ds_na.latitude.values
na_lons = ds_na.longitude.values

for vname in ('T_lvl4', 'T0', 'temperature'):
    if vname in ds_na:
        na_T = ds_na[vname].values   # shape: (time, station)
        print(f"Netatmo temperature variable: '{vname}', shape: {na_T.shape}")
        break

# Netatmo timestamps → epoch seconds
na_time  = ds_na.time.values
na_dt    = np.array([datetime.datetime.fromtimestamp(
               dt.astype('datetime64[ns]').view('int64') / 1e9) for dt in na_time])
na_epoch = np.array([dt.timestamp() for dt in na_dt])
na_hours = np.array([dt.hour for dt in na_dt])
na_dates = np.array([dt.date() for dt in na_dt])

#%% Classify Netatmo stations (urban / rural / other)

with fiona.open(URBAN_SHP) as u, fiona.open(RURAL_SHP) as r:
    urban_geom = [shape(f['geometry']) for f in u]
    rural_geom = [shape(f['geometry']) for f in r]

def classify(lat, lon):
    g  = {'type': 'Point', 'coordinates': (lon, lat)}
    pt = Point(transform_geom('EPSG:4326', 'EPSG:3035', g)['coordinates'])
    if any(g.contains(pt) for g in urban_geom): return 'urban'
    if any(g.contains(pt) for g in rural_geom): return 'rural'
    return 'other'

# Build unique station list with their column index in na_T
unique_stations = {}
for i, (lon, lat) in enumerate(zip(na_lons, na_lats)):
    if not (np.isnan(lat) or np.isnan(lon)):
        key = (round(lat, 5), round(lon, 5))
        if key not in unique_stations:
            unique_stations[key] = {'lat': lat, 'lon': lon, 'index': i}

stations = list(unique_stations.values())
for s in stations:
    s['type'] = classify(s['lat'], s['lon'])

print(f"Netatmo stations: {len(stations)}  "
      f"(urban={sum(s['type']=='urban' for s in stations)}, "
      f"rural={sum(s['type']=='rural' for s in stations)}, "
      f"other={sum(s['type']=='other' for s in stations)})")

#%% Filter to date + hour range

start_d = datetime.datetime.strptime(DATE_START, '%Y-%m-%d').date()
end_d   = datetime.datetime.strptime(DATE_END,   '%Y-%m-%d').date()

# Netatmo timestep mask
na_mask = ((na_dates >= start_d) & (na_dates <= end_d) &
           (na_hours >= HOUR_START) & (na_hours < HOUR_END))
na_epoch_p = na_epoch[na_mask]
na_T_p     = na_T[na_mask, :]        # (n_timesteps_filtered, n_stations)
na_hours_p = na_hours[na_mask]

# MT mask (same date + hour range)
mt_mask = ((mt_dates >= start_d) & (mt_dates <= end_d) &
           (mt_hours >= HOUR_START) & (mt_hours < HOUR_END))
mt_lat_p   = mt_lat[mt_mask]
mt_lon_p   = mt_lon[mt_mask]
mt_temp_p  = mt_temp[mt_mask]
mt_epoch_p = mt_epoch[mt_mask]
mt_hours_p = mt_hours[mt_mask]

print(f"\nFiltered: {na_mask.sum()} Netatmo timesteps | {mt_mask.sum():,} MT points")
print(f"Time window: ±{TIME_WINDOW_MIN} min  |  Spatial radius: {RADIUS_KM} km  |  Min MT pts: {MIN_MT_POINTS}")

#%% Build matched pairs: (Netatmo timestep × station) → averaged MT

tw_sec = TIME_WINDOW_MIN * 60.0

pairs = []

for ti, (na_ts, na_h) in enumerate(zip(na_epoch_p, na_hours_p)):

    # MT points within the time window
    dt_mt   = np.abs(mt_epoch_p - na_ts)
    t_close = dt_mt <= tw_sec
    if t_close.sum() == 0:
        continue

    mt_lat_w  = mt_lat_p[t_close]
    mt_lon_w  = mt_lon_p[t_close]
    mt_temp_w = mt_temp_p[t_close]

    # For each station, check spatial proximity
    for s in stations:
        na_val = na_T_p[ti, s['index']]
        if np.isnan(na_val):
            continue

        dists  = np.array([haversine(s['lat'], s['lon'], la, lo)
                           for la, lo in zip(mt_lat_w, mt_lon_w)])
        in_r   = dists <= RADIUS_KM

        if in_r.sum() < MIN_MT_POINTS:
            continue

        pairs.append({
            'netatmo_T'  : na_val,
            'mt_T'       : mt_temp_w[in_r].mean(),
            'mt_T_std'   : mt_temp_w[in_r].std(),
            'n_mt'       : int(in_r.sum()),
            'distance_km': dists[in_r].mean(),
            'stype'      : s['type'],
            'hour'       : int(na_h),
        })

df = pd.DataFrame(pairs)
print(f"\nMatched pairs: {len(df)}")
if len(df) == 0:
    raise RuntimeError("No pairs found — try increasing TIME_WINDOW_MIN, RADIUS_KM or lowering MIN_MT_POINTS.")

#%% Statistics

r    = df['mt_T'].corr(df['netatmo_T'])
bias = (df['mt_T'] - df['netatmo_T']).mean()
rmse = np.sqrt(((df['mt_T'] - df['netatmo_T'])**2).mean())
print(f"r = {r:.3f}  |  RMSE = {rmse:.2f} °C  |  Bias (MT−Netatmo) = {bias:+.2f} °C")
print(f"\nPairs by station type:\n{df.groupby('stype')[['netatmo_T','mt_T']].describe().round(2)}")

#%% Scatter plot

fig, ax = plt.subplots(figsize=(8, 7))

tmin = min(df['netatmo_T'].min(), df['mt_T'].min()) - 0.5
tmax = max(df['netatmo_T'].max(), df['mt_T'].max()) + 0.5

# ---- Palette options (uncomment the one you like) ----
# Teal / Amber — complementary, colorblind-friendly (default)
# palette = {'urban': '#E8863A', 'rural': '#3A8EB5', 'other': '#A8A8A8'}
# Indigo / Coral
palette = {'urban': '#E07070', 'rural': '#5B6EB5', 'other': '#A8A8A8'}
# Olive / Plum
# palette = {'urban': '#7B5EA7', 'rural': '#7A9E3B', 'other': '#A8A8A8'}
# Muted viridis-inspired
# palette = {'urban': '#3E9E8F', 'rural': '#C4792A', 'other': '#A8A8A8'}

if COLOR_BY == 'station_type':
    for stype, grp in df.groupby('stype'):
        ax.scatter(grp['netatmo_T'], grp['mt_T'],
                   color=palette.get(stype, '#A8A8A8'),
                   label=f"{stype.capitalize()} (n={len(grp)})",
                   s=70, alpha=0.20, edgecolors='white', linewidths=0.6)
    ax.legend(title='Station type', fontsize=10)

elif COLOR_BY == 'hour':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['hour'],
                    cmap='twilight', norm=mcolors.Normalize(0, 23),
                    s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Hour of day')

elif COLOR_BY == 'distance':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['distance_km'],
                    cmap='plasma_r', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Mean distance (km)')

elif COLOR_BY == 'n_mt_points':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['n_mt'],
                    cmap='viridis', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='# MT points averaged')

# 1:1 line
ax.plot([tmin, tmax], [tmin, tmax], 'k--', lw=1.8, label='1:1', zorder=0)

# Regression
if len(df) >= 2:
    z    = np.polyfit(df['netatmo_T'], df['mt_T'], 1)
    xfit = np.linspace(tmin, tmax, 200)
    ax.plot(xfit, np.polyval(z, xfit), 'r-', lw=2,
            label=f'Fit: y = {z[0]:.2f}x {z[1]:+.2f}', zorder=1)

ax.set_xlim(tmin, tmax)
ax.set_ylim(tmin, tmax)
ax.set_aspect('equal')
ax.set_xlabel('Netatmo Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('MeteoTracker Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_title(
    f'MeteoTracker vs Netatmo — {PERIOD_LABEL}\n'
    f'{DATE_START} → {DATE_END}  |  n = {len(df)}\n'
    f'r = {r:.3f},  RMSE = {rmse:.2f} °C,  Bias = {bias:+.2f} °C  '
    f'[±{TIME_WINDOW_MIN} min, {RADIUS_KM} km]',
    fontsize=11, fontweight='bold'
)
ax.grid(True, alpha=0.3)
if COLOR_BY == 'station_type':
    ax.legend(title='Station type', fontsize=10)
else:
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"✅ Saved: {OUTPUT_FILE}")

#%% Scatter plot light
fig, ax = plt.subplots(figsize=(8, 7))
tmin = min(df['netatmo_T'].min(), df['mt_T'].min()) - 0.5
tmax = max(df['netatmo_T'].max(), df['mt_T'].max()) + 0.5

palette = {'urban': '#E07070', 'rural': '#5B6EB5', 'other': '#A8A8A8'}

if COLOR_BY == 'station_type':
    for stype, grp in df.groupby('stype'):
        ax.scatter(grp['netatmo_T'], grp['mt_T'],
                   color=palette.get(stype, '#A8A8A8'),
                   label=f"{stype.capitalize()} (n={len(grp)})",
                   s=70, alpha=0.20, edgecolors='white', linewidths=0.6)
elif COLOR_BY == 'hour':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['hour'],
                    cmap='twilight', norm=mcolors.Normalize(0, 23),
                    s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Hour of day')
elif COLOR_BY == 'distance':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['distance_km'],
                    cmap='plasma_r', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Mean distance (km)')
elif COLOR_BY == 'n_mt_points':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['n_mt'],
                    cmap='viridis', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='# MT points averaged')

# 1:1 line
ax.plot([tmin, tmax], [tmin, tmax], 'k--', lw=1.8, label='1:1 line', zorder=0)

# Regression + stats in legend
if len(df) >= 2:
    z    = np.polyfit(df['netatmo_T'], df['mt_T'], 1)
    xfit = np.linspace(tmin, tmax, 200)
    fit_label = (
        f'Fit: $y = {z[0]:.2f}x {z[1]:+.2f}$\n'
        f'$r = {r:.3f}$,  RMSE $= {rmse:.2f}$ °C,  Bias $= {bias:+.2f}$ °C'
    )
    ax.plot(xfit, np.polyval(z, xfit), 'r-', lw=2, label=fit_label, zorder=1)

ax.set_xlim(tmin, tmax)
ax.set_ylim(tmin, tmax)
ax.set_aspect('equal')
ax.set_xlabel('Netatmo Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('MeteoTracker Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_title('MeteoTracker vs. Netatmo', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

legend = ax.legend(
    title=f'Station type  |  n = {len(df)}' if COLOR_BY == 'station_type' else f'n = {len(df)}',
    fontsize=10,
    title_fontsize=10,
    framealpha=0.9,
    edgecolor='#cccccc',
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"✅ Saved: {OUTPUT_FILE}")

#%% Scatter plot all together
fig, ax = plt.subplots(figsize=(8, 7))
tmin = min(df['netatmo_T'].min(), df['mt_T'].min()) - 0.5
tmax = max(df['netatmo_T'].max(), df['mt_T'].max()) + 0.5

palette = {'urban': '#E07070', 'rural': '#5B6EB5', 'other': '#A8A8A8'}

if COLOR_BY == 'station_type':
    ax.scatter(df['netatmo_T'], df['mt_T'],
               color='#5B6EB5', s=70, alpha=0.20,
               edgecolors='white', linewidths=0.6)
elif COLOR_BY == 'hour':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['hour'],
                    cmap='twilight', norm=mcolors.Normalize(0, 23),
                    s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Hour of day')
elif COLOR_BY == 'distance':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['distance_km'],
                    cmap='plasma_r', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Mean distance (km)')
elif COLOR_BY == 'n_mt_points':
    sc = ax.scatter(df['netatmo_T'], df['mt_T'], c=df['n_mt'],
                    cmap='viridis', s=65, alpha=0.75, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='# MT points averaged')

# 1:1 line
ax.plot([tmin, tmax], [tmin, tmax], 'k--', lw=1.8, label='1:1 line', zorder=0)

# Regression + stats in legend
if len(df) >= 2:
    z    = np.polyfit(df['netatmo_T'], df['mt_T'], 1)
    xfit = np.linspace(tmin, tmax, 200)
    fit_label = (
        f'Fit: $y = {z[0]:.2f}x {z[1]:+.2f}$\n'
        f'$r = {r:.3f}$,  RMSE $= {rmse:.2f}$ °C,  Bias $= {bias:+.2f}$ °C'
    )
    ax.plot(xfit, np.polyval(z, xfit), 'k-', lw=2, label=fit_label, zorder=1)

ax.set_xlim(tmin, tmax)
ax.set_ylim(tmin, tmax)
ax.set_aspect('equal')
ax.set_xlabel('Netatmo Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('MeteoTracker Temperature (°C)', fontsize=13, fontweight='bold')
# ax.set_title('MeteoTracker vs. Netatmo', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

legend = ax.legend(
    title=f'n = {len(df)}',
    fontsize=10,
    title_fontsize=10,
    framealpha=0.9,
    edgecolor='#cccccc',
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"✅ Saved: {OUTPUT_FILE}")

#%%end cell