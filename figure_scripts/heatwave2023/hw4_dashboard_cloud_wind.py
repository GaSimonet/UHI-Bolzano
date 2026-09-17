#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW4 dashboard — 13-26 August 2023

Extends the UrbClim / WRF / Netatmo / Satellite UHI comparison
(plot_diurnal_uhi_hw4_detail_full_distri_v2_20260508.py) with two panels
below it, both from WRF d03 over the Bolzano urban shapefile:

  Row 1  — Full hourly UHI time series, all datasets  (unchanged from v2)
  Row 2  — Mean diurnal UHI composite, all datasets    (unchanged from v2)
  Row 3  — Cloud cover HEATMAP: column cloud fraction resolved by altitude
           (m AGL) and time -- shows cloud base/top evolution, not just a
           column-integrated number.
  Row 4  — 10 m -> bulk 0-500 m AGL wind "stick plot": each hourly arrow is
           the thickness-weighted mean wind in the lowest 500 m (boundary
           layer flow), not just the 10 m diagnostic -- direction (angle)
           and speed (length + color) both shown.

Rows 1, 3, 4 share a calendar-date x-axis; Row 2 keeps its own hour-of-day
x-axis (0-23), same convention as the original two-panel figure.

Data:
  UHI panels : same sources as plot_diurnal_uhi_hw4_detail_full_distri_v2_20260508.py
  Cloud/wind : /mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/
               HW_202308_13_26_subset/wrfout_d03_2023-08-{13..26}_00_24hr.nc
  Urban mask : LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp

NOTE: run under the ROI_python_38_env conda env (geopandas + shapely):
  conda activate ROI_python_38_env

WRF SOURCE CHANGE (2026-08-04): the UHI panel's WRF series previously loaded
a single annual file (WRF_RAW_Alto_Adige_2023.nc) via salem -- that file was
built by concatenating many short forecast-cycle runs WITHOUT dropping
spin-up, producing a visible jump in the WRF curve at/after each day's 00 UTC
boundary. Now uses common/wrf_daily.py (5-year daily-harmonized archive,
spin-up already trimmed per file); salem is no longer a dependency of this
script at all (the cloud/wind wrfout_d03_* panels never used it either).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, DayLocator, date2num
from matplotlib.legend_handler import HandlerTuple
from shapely.vectorized import contains as shp_contains
from shapely.geometry import Point
import geopandas as gpd
import xarray as xr
import os as _os
import warnings
warnings.filterwarnings('ignore')

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Dynamic, per-(season, day/night) valley lapse rate + the same astronomical
# sunrise/sunset day/night classifier used in the seasonal UHI scripts and in
# plot_diurnal_uhi_hw4_detail_full_distri_v2_20260508.py -- see
# UHI/seasonal_valley_lapse_rate.py and UHI/solar_daynight.py.
# Replaces this script's old fixed LAPSE_RATE = 0.0065 K/m and its hand-typed
# "18:00-04:00 UTC" night window.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night, sun_times_utc
from dem_utils import reproject_dem_to_grid
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT
from wrf_daily import load_wrf_uhi

# ── CONFIG (UHI panels — unchanged from full_distri_v2) ──────────────────────
UCB_FILE     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc'
NETATMO_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
LST_FILE     = '/home/gsimonet/Desktop/LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc'

URBAN_SHP  = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
RURAL_SHP  = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'
OUTPUT_DIR = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/All_combined_time_series/Combined_plots/figures'

HW4_START = '2023-08-13'
HW4_END   = '2023-08-26'

UTM_CRS    = 'EPSG:32632'
WGS84      = 'EPSG:4326'

# Elevation-correction mode -- edit and re-run to get each version; see
# common/lapse_mode_utils.py. Output filenames are tagged with this mode.
#   'none'     -- no elevation correction (raw urban-minus-rural difference)
#   'constant' -- classical fixed rate (LAPSE_RATE_CONSTANT), no season/day-night dependence
#   'valley'   -- dynamic, per-(season, day/night) rate fitted from WRF -- WRF ONLY (default);
#                 UrbClim/Netatmo/Satellite fall back to the constant rate under this mode too
#                 (see _non_wrf_rate() below), since the dynamic rate was fitted from WRF's own
#                 vertical profiles and isn't assumed representative of near-surface behavior
#                 for the other three sources.
LAPSE_MODE = 'valley'

SEASON_MAP = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring',  4: 'Spring',  5: 'Spring',
               6: 'Summer',  7: 'Summer',  8: 'Summer',
               9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

# {(season, period): K/m} -- HW4 (13-26 Aug) falls entirely in 'Summer'; the lookup is
# written generally in case HW4_START/HW4_END ever change to span a season boundary.
# Applied to WRF only -- see LAPSE_MODE note above.
DYNAMIC_LAPSE = load_seasonal_valley_lapse()


def _non_wrf_rate():
    """Constant-rate resolution for UrbClim/Netatmo/Satellite: 'valley' collapses to
    'constant' for these three (the dynamic rate is WRF-only), 'none'/'constant' behave
    exactly as resolve_rate() already defines them."""
    return resolve_rate('constant' if LAPSE_MODE == 'valley' else LAPSE_MODE, None)

# Representative sunrise/sunset (UTC) for the HW4 window, used for night shading
# below -- computed for the event midpoint (day length only shifts ~15 min across
# the 13-day window). Replaces the old hand-typed "sunset ~18:07-18:30 UTC, sunrise
# ~04:09-04:25 UTC" note with the actual formula behind those numbers.
_sr, _ss = sun_times_utc([pd.Timestamp(2023, 8, 19)])
HW4_SUNRISE_H, HW4_SUNSET_H = float(_sr[0]), float(_ss[0])

# CORRECTED 2026-07-28: UrbClim's raw 't' coord is already UTC, not local
# (empirically confirmed: its raw diurnal T2M peak matches WRF's confirmed-UTC
# peak hour exactly). offset was 2, is 0.
# CORRECTED 2026-07-29: netatmo offset also revised 2 -> 0 -- the original
# 2h-peak-lag reading didn't hold up under a per-season cross-check, and the
# ingestion pipeline converts epoch timestamps straight to UTC with no
# local-time step (see plot_diurnal_uhi_seasonal_20260728.py's UTC_OFFSET
# note for the full reasoning).
UTC_OFFSET   = {'urbclim': 0, 'wrf': 0, 'netatmo': 0, 'sat': 0}
SAT_HOUR_TOL = 1.5

COLORS = {
    'urbclim': '#E07B39',
    'wrf'    : '#3A7FC1',
    'netatmo': '#38A66C',
    'sat'    : '#9B3A9B',
}

ALPHA_SPATIAL  = 0.18
ALPHA_TEMPORAL = 0.22

os.makedirs(OUTPUT_DIR, exist_ok=True)
_shared_elev = None

# ── CONFIG (new cloud/wind panels — WRF d03 raw per-day files) ───────────────
WRF_HW_DIR = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/HW_202308_13_26_subset"
WRF_HW_FILE_TEMPLATE = "wrfout_d03_2023-08-{day:02d}_00_24hr.nc"
WRF_HW_DAYS = range(13, 27)

ALT_GRID_M = np.arange(0, 8001, 250)   # m AGL, cloud heatmap vertical bins
WIND_LAYER_TOP_M = 500.0               # bulk wind integration depth (m AGL)

# ── STYLE ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'         : 12,
    'axes.labelsize'    : 13,
    'axes.titlesize'    : 14,
    'xtick.labelsize'   : 12,
    'ytick.labelsize'   : 12,
    'legend.fontsize'   : 11,
    'axes.linewidth'    : 1.0,
    'xtick.major.size'  : 4,
    'ytick.major.size'  : 4,
    'xtick.major.width' : 1.0,
    'ytick.major.width' : 1.0,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# ── GEOMETRY ───────────────────────────────────────────────────────────────────

def build_mask(union, flat_x, flat_y, shape):
    b   = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) &
           (flat_y >= b[1]) & (flat_y <= b[3]))
    m   = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)

def elev_diff(elev2d, rural_m):
    """elev2d minus the rural mask's mean elevation, everywhere (urban included).

    NOT multiplied by a rate here -- the rate is now season x day/night dependent
    (DYNAMIC_LAPSE), so the rate multiplication happens per-timestep in extract_all()
    instead of being baked into a single precomputed field.
    """
    r = elev2d[rural_m];  r = r[~np.isnan(r)]
    if len(r) == 0: return np.zeros_like(elev2d)
    return elev2d - float(np.nanmean(r))

def to_utc(times, offset):
    return pd.DatetimeIndex(times) - pd.Timedelta(hours=offset)

# ── CORE EXTRACTOR (UHI panels) ─────────────────────────────────────────────

def extract_all(arr3d, times_utc, urban_m, rural_m, lapse2d=None):
    t0 = pd.Timestamp(HW4_START)
    t1 = pd.Timestamp(HW4_END) + pd.Timedelta('23h59m59s')
    in_period = (times_utc >= t0) & (times_utc <= t1)
    arr3d   = arr3d[in_period]
    times_p = times_utc[in_period]

    if len(arr3d) > 0 and np.nanmean(arr3d[0]) > 200:
        arr3d = arr3d - 273.15

    ts_rows  = []
    temporal = {h: [] for h in range(24)}

    for i, snap in enumerate(arr3d):
        # Reference stays the RAW rural mean (uncorrected) -- correction is applied to
        # the WHOLE field (urban included), not to the rural pixels feeding the
        # reference mean, so it can't cancel to zero once re-averaged over that same
        # rural mask (see elev_diff() docstring). UrbClim uses a constant rate (see
        # _non_wrf_rate()), so lapse2d is precomputed once, not resolved per timestep.
        r_mean_raw = float(np.nanmean(snap[rural_m]))
        corr_field = lapse2d if lapse2d is not None else 0.0
        uhi = (snap + corr_field) - r_mean_raw
        upx = uhi[urban_m];  upx = upx[np.isfinite(upx)]
        if len(upx) == 0:
            continue
        mean_uhi = float(np.nanmean(upx))
        ts_rows.append({
            'time':     times_p[i],
            'uhi_mean': mean_uhi,
            'uhi_p25':  float(np.percentile(upx, 25)),
            'uhi_p75':  float(np.percentile(upx, 75)),
        })
        temporal[times_p[i].hour].append(mean_uhi)

    df_ts    = pd.DataFrame(ts_rows)
    temporal = {h: np.array(v) for h, v in temporal.items()}
    return df_ts, temporal

# ── LOAD SHAPEFILES ────────────────────────────────────────────────────────────
print('Loading shapefiles...')
urban_utm = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
rural_utm = gpd.read_file(RURAL_SHP).to_crs(UTM_CRS)
urban_wgs = urban_utm.to_crs(WGS84)
rural_wgs = rural_utm.to_crs(WGS84)

# ── UrbClim ────────────────────────────────────────────────────────────────────
print('\n[UrbClim] Loading...')
uc_ts       = pd.DataFrame()
uc_temporal = {h: np.array([]) for h in range(24)}

if os.path.exists(UCB_FILE):
    ds  = xr.open_dataset(UCB_FILE)
    xd  = 'x' if 'x' in ds.dims else 'lon'
    yd  = 'y' if 'y' in ds.dims else 'lat'
    xx, yy = np.meshgrid(ds[xd].values, ds[yd].values)

    urban_m = build_mask(urban_utm.unary_union, xx.flatten(), yy.flatten(), xx.shape)
    rural_m = build_mask(rural_utm.unary_union, xx.flatten(), yy.flatten(), xx.shape)
    print(f'  Grid {xx.shape}  urban={urban_m.sum()} rural={rural_m.sum()} px')

    ev = next((v for v in ('HGT','elevation','dem','DEM') if v in ds), None)
    if ev:
        e = ds[ev].values;  e = e[0] if e.ndim == 3 else e
        print('  Elevation: native variable in UrbClim file')
    else:
        # UrbClim carries no native elevation (checked: Bolzano_UHI_RETURN_combined.nc
        # has only T2M/QV2M/WS2M/LST) -- reproject the shared 10m DEM onto this grid
        # instead of silently leaving the lapse correction at zero.
        e = reproject_dem_to_grid(ds[xd].values, ds[yd].values)
        print('  Elevation: reprojected from DEM_10m_south_tirol.tif (no native var)')
    lapse2d = elev_diff(e, rural_m) * _non_wrf_rate()
    _shared_elev = (xx.flatten().copy(), yy.flatten().copy(), e.flatten().copy())

    t0     = pd.Timestamp(HW4_START)
    t1_loc = pd.Timestamp(HW4_END) + pd.Timedelta(hours=23 + UTC_OFFSET['urbclim'])
    print('  Reading into RAM (HW4 period)...')
    arr3d     = ds['T2M'].sel(t=slice(t0, t1_loc)).load().values
    times_utc = to_utc(ds.t.sel(t=slice(t0, t1_loc)).values, UTC_OFFSET['urbclim'])
    ds.close()

    uc_ts, uc_temporal = extract_all(arr3d, times_utc, urban_m, rural_m, lapse2d)
    print(f'  TS rows={len(uc_ts)}')
else:
    print('  [MISSING]', UCB_FILE)

# ── WRF (daily-harmonized archive, spin-up already trimmed) ─────────────────
# Previously loaded the single annual WRF_RAW_Alto_Adige_2023.nc via salem --
# that file was built by concatenating many short forecast-cycle runs WITHOUT
# dropping spin-up, producing a visible jump in the WRF curve at/after each
# day's 00 UTC boundary. Now uses common/wrf_daily.py, which loops the 5-year
# wrf_harmonized_YYYY-MM-DD.nc daily archive (spin-up already trimmed).
print('\n[WRF] Loading (daily harmonized archive)...')
wrf_ts       = pd.DataFrame()
wrf_temporal = {h: np.array([]) for h in range(24)}

def _wrf_rate(t):
    period = classify_day_night(pd.DatetimeIndex([t]))[0]
    return resolve_rate(LAPSE_MODE, DYNAMIC_LAPSE.get((SEASON_MAP[t.month], period)))

times_wrf, mean_wrf, p25_wrf, p75_wrf = load_wrf_uhi(
    HW4_START, HW4_END, urban_wgs.unary_union, rural_wgs.unary_union,
    cache_key='hw4_dashboard', lapse_rate=_wrf_rate, lapse_rate_key=f'valley-{LAPSE_MODE}')

if len(times_wrf) > 0:
    wrf_ts = pd.DataFrame({
        'time':     times_wrf,
        'uhi_mean': mean_wrf,
        'uhi_p25':  p25_wrf,
        'uhi_p75':  p75_wrf,
    })
    temporal_dict = {h: [] for h in range(24)}
    for t, u in zip(times_wrf, mean_wrf):
        temporal_dict[t.hour].append(float(u))
    wrf_temporal = {h: np.array(v) for h, v in temporal_dict.items()}
    print(f'  TS rows={len(wrf_ts)}')
else:
    print('  [MISSING] no WRF daily files found for this window')

# ── Netatmo ────────────────────────────────────────────────────────────────────
print('\n[Netatmo] Loading...')
na_ts       = pd.DataFrame()
na_temporal = {h: np.array([]) for h in range(24)}

if os.path.exists(NETATMO_FILE):
    ds    = xr.open_dataset(NETATMO_FILE)
    T_VAR = next((v for v in ('T_lvl4','T0','temperature') if v in ds), None)

    if T_VAR:
        na_lats = ds.latitude.values
        na_lons = ds.longitude.values
        na_T    = ds[T_VAR].load().values
        if na_T.shape[0] == len(na_lats):
            na_T = na_T.T
        times_utc = to_utc(
            [pd.Timestamp(t.astype('datetime64[ns]').item()) for t in ds.time.values],
            UTC_OFFSET['netatmo'])
        ds.close()

        u_idx, r_idx = [], []
        u_union = urban_wgs.unary_union;  r_union = rural_wgs.unary_union
        for i, (lat, lon) in enumerate(zip(na_lats, na_lons)):
            if np.isnan(lat) or np.isnan(lon): continue
            pt = Point(lon, lat)
            if u_union.contains(pt):   u_idx.append(i)
            elif r_union.contains(pt): r_idx.append(i)
        u_idx = np.array(u_idx);  r_idx = np.array(r_idx)
        print(f'  Urban={len(u_idx)} Rural={len(r_idx)} stations')

        na_elev = None
        ev_var  = next((v for v in ('altitude','elevation','z','ALT')
                        if v in xr.open_dataset(NETATMO_FILE)), None)
        if ev_var:
            na_elev = xr.open_dataset(NETATMO_FILE)[ev_var].values.astype(float)
        elif _shared_elev is not None and HAS_PYPROJ:
            try:
                from scipy.interpolate import LinearNDInterpolator
                x_uc, y_uc, e_uc = _shared_elev
                ok = ~np.isnan(e_uc)
                interp = LinearNDInterpolator(list(zip(x_uc[ok], y_uc[ok])), e_uc[ok])
                tr = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)
                nx_utm, ny_utm = tr.transform(na_lons, na_lats)
                na_elev = interp(nx_utm, ny_utm)
            except Exception as ex:
                print(f'  [WARN] lapse skipped: {ex}')

        t0 = pd.Timestamp(HW4_START)
        t1 = pd.Timestamp(HW4_END) + pd.Timedelta('23h59m59s')
        in_period = (times_utc >= t0) & (times_utc <= t1)
        na_T_p    = na_T[in_period]
        times_p   = times_utc[in_period]

        r_e_mean = None
        if na_elev is not None:
            r_e_mean = float(np.nanmean(na_elev[r_idx][~np.isnan(na_elev[r_idx])]))

        na_rate = _non_wrf_rate()

        ts_rows      = []
        temporal_dict = {h: [] for h in range(24)}

        for i in range(len(times_p)):
            snap   = na_T_p[i]
            # Reference stays the RAW rural mean; correction applied to URBAN stations
            # instead, relative to the rural reference elevation, so it can't cancel to
            # zero once re-averaged over the rural set it used to be applied to.
            r_vals = snap[r_idx]
            r_vals = r_vals[~np.isnan(r_vals)]
            if len(r_vals) == 0: continue
            r_mean_raw = float(np.nanmean(r_vals))

            u_vals = snap[u_idx].copy()
            if na_elev is not None and r_e_mean is not None:
                u_vals = u_vals + na_rate * (na_elev[u_idx] - r_e_mean)

            uhi = u_vals - r_mean_raw
            uhi = uhi[np.isfinite(uhi)]
            if len(uhi) == 0: continue
            mean_uhi = float(np.nanmean(uhi))
            ts_rows.append({
                'time':     times_p[i],
                'uhi_mean': mean_uhi,
                'uhi_p25':  float(np.percentile(uhi, 25)) if len(uhi) >= 2 else mean_uhi,
                'uhi_p75':  float(np.percentile(uhi, 75)) if len(uhi) >= 2 else mean_uhi,
            })
            temporal_dict[times_p[i].hour].append(mean_uhi)

        na_ts       = pd.DataFrame(ts_rows)
        na_temporal = {h: np.array(v) for h, v in temporal_dict.items()}
        print(f'  TS rows={len(na_ts)}')
    else:
        print('  [WARN] Temperature variable not found')
else:
    print('  [MISSING]', NETATMO_FILE)

# ── Satellite ──────────────────────────────────────────────────────────────────
print('\n[Satellite] Loading...')
sat_ts_rows  = []
sat_temporal = {h: [] for h in range(24)}

if os.path.exists(LST_FILE):
    ds      = xr.open_dataset(LST_FILE)
    LST_VAR = '__xarray_dataarray_variable__'
    x_a = ds.x.values if 'x' in ds else ds.lon.values
    y_a = ds.y.values if 'y' in ds else ds.lat.values
    xx, yy  = np.meshgrid(x_a, y_a)
    urban_m = build_mask(urban_utm.unary_union, xx.flatten(), yy.flatten(), xx.shape)
    rural_m = build_mask(rural_utm.unary_union, xx.flatten(), yy.flatten(), xx.shape)

    ev_sat = next((v for v in ('HGT','elevation','dem','DEM') if v in ds), None)
    if ev_sat:
        e_sat = ds[ev_sat].values;  e_sat = e_sat[0] if e_sat.ndim==3 else e_sat
        print('  Elevation: native variable in LST file')
    else:
        # lst-bolzano.nc carries no native elevation either (checked: only the LST
        # DataArray + spatial_ref) -- reproject the shared 10m DEM directly onto this
        # grid's own resolution/extent, rather than interpolating from UrbClim's grid
        # (which was itself never populated, since UrbClim has no native elevation --
        # see the UrbClim block above).
        e_sat = reproject_dem_to_grid(x_a, y_a)
        print('  Elevation: reprojected from DEM_10m_south_tirol.tif (no native var)')
    sat_lapse2d = elev_diff(e_sat, rural_m) * _non_wrf_rate()

    times_all = pd.DatetimeIndex(
        [pd.Timestamp(t.astype('datetime64[ns]').item()) for t in ds.time.values])
    times_utc = to_utc(times_all, UTC_OFFSET['sat'])
    t0, t1    = pd.Timestamp(HW4_START), pd.Timestamp(HW4_END)
    in_period = (times_utc >= t0) & (times_utc <= t1)

    print('  Reading satellite scenes into RAM (HW4 period)...')
    arr3d     = ds[LST_VAR].isel(time=np.where(in_period)[0]).load().values
    times_utc = times_utc[in_period]
    ds.close()

    for i, snap in enumerate(arr3d):
        h_dec = times_utc[i].hour + times_utc[i].minute / 60.0
        h_int = round(h_dec) % 24
        if min(abs(h_dec - h_int), 24 - abs(h_dec - h_int)) > SAT_HOUR_TOL:
            continue
        # Reference stays the RAW rural mean; correction applied to the whole field.
        r_mean = float(np.nanmean(snap[rural_m]))
        suhi = (snap + sat_lapse2d) - r_mean
        upx  = suhi[urban_m];  upx = upx[np.isfinite(upx)]
        if len(upx) == 0: continue
        mean_suhi = float(np.nanmean(upx))
        sat_ts_rows.append({
            'time':      times_utc[i],
            'suhi_mean': mean_suhi,
            'suhi_p25':  float(np.percentile(upx, 25)),
            'suhi_p75':  float(np.percentile(upx, 75)),
        })
        sat_temporal[h_int].append(mean_suhi)

    df_sat_ts    = pd.DataFrame(sat_ts_rows)
    sat_temporal = {h: np.array(v) for h, v in sat_temporal.items()}
else:
    print('  [MISSING]', LST_FILE)
    df_sat_ts    = pd.DataFrame()
    sat_temporal = {h: np.array([]) for h in range(24)}

# ══════════════════════════════════════════════════════════════════════════════
# NEW: cloud-cover altitude heatmap + 0-500 m AGL wind stick, from WRF d03
# ══════════════════════════════════════════════════════════════════════════════

def wrf_hw_path(day):
    return _os.path.join(WRF_HW_DIR, WRF_HW_FILE_TEMPLATE.format(day=day))


def compute_cloud_wind_profiles():
    """
    For each hourly WRF timestep, spatially average (over the Bolzano urban
    mask) the CLDFRA and AGL-height profiles, then:
      - interpolate CLDFRA(agl) onto a fixed altitude grid -> cloud heatmap column
      - thickness-weight U,V over 0-500 m AGL -> bulk low-level wind vector
    """
    print('\n[WRF cloud/wind profiles] Loading...')
    urban_m = None
    times_all, cloud_cols, u_bulk, v_bulk = [], [], [], []

    for day in WRF_HW_DAYS:
        fpath = wrf_hw_path(day)
        if not _os.path.exists(fpath):
            print(f'  [!] missing {fpath}, skipping')
            continue
        ds = xr.open_dataset(fpath)

        if urban_m is None:
            lon2d = ds['lon'].values
            lat2d = ds['lat'].values
            urban = gpd.read_file(URBAN_SHP).to_crs(WGS84)
            urban_m = shp_contains(urban.unary_union, lon2d, lat2d)
            print(f'  urban mask: {urban_m.sum()} pixels')

        z = ds['Z'].values          # (time, z, y, x) m ASL
        hgt = ds['HGT'].values      # (time, y, x) m ASL (static, but keep per-time for safety)
        u = ds['U'].values
        v = ds['V'].values
        cldfra = ds['CLDFRA'].values
        times = pd.DatetimeIndex(ds['time'].values)

        for ti in range(len(times)):
            agl_profile = np.nanmean(z[ti][:, urban_m] - hgt[ti][urban_m][None, :], axis=1)  # (z,)
            cld_profile = np.nanmean(cldfra[ti][:, urban_m], axis=1)                          # (z,)
            u_profile = np.nanmean(u[ti][:, urban_m], axis=1)
            v_profile = np.nanmean(v[ti][:, urban_m], axis=1)

            cloud_cols.append(np.interp(ALT_GRID_M, agl_profile, cld_profile,
                                         left=np.nan, right=np.nan))

            # thickness-weighted bulk wind over 0-500 m AGL
            edges = np.concatenate([[0.0], (agl_profile[:-1] + agl_profile[1:]) / 2.0,
                                     [agl_profile[-1]]])
            dz = np.diff(edges)
            weight = np.clip(np.minimum(edges[1:], WIND_LAYER_TOP_M) -
                              np.minimum(edges[:-1], WIND_LAYER_TOP_M), 0, None)
            weight = np.where(dz > 0, weight, 0.0)
            if weight.sum() > 0:
                u_bulk.append(float(np.sum(u_profile * weight) / weight.sum()))
                v_bulk.append(float(np.sum(v_profile * weight) / weight.sum()))
            else:
                u_bulk.append(np.nan)
                v_bulk.append(np.nan)

            times_all.append(times[ti])
        ds.close()
        print(f'  processed day {day}')

    df_wind = pd.DataFrame({'time': times_all, 'u_bulk500': u_bulk, 'v_bulk500': v_bulk})
    df_wind['ws_bulk500'] = np.hypot(df_wind['u_bulk500'], df_wind['v_bulk500'])
    cloud_heatmap = np.array(cloud_cols).T   # (altitude, time)
    return pd.DatetimeIndex(times_all), cloud_heatmap, df_wind


times_wrf, cloud_heatmap, df_wind = compute_cloud_wind_profiles()

# ── DYNAMIC Y-RANGE (UHI panels) ─────────────────────────────────────────────
uhi_vals = []
for df, col in [(uc_ts, 'uhi_p25'), (uc_ts, 'uhi_p75'),
                (wrf_ts,'uhi_p25'), (wrf_ts,'uhi_p75'),
                (na_ts, 'uhi_p25'), (na_ts, 'uhi_p75')]:
    if len(df) and col in df.columns:
        uhi_vals.extend(df[col].dropna().tolist())
uhi_vals = np.array(uhi_vals);  uhi_vals = uhi_vals[np.isfinite(uhi_vals)]
if len(uhi_vals):
    ymin = np.floor((np.percentile(uhi_vals,  0.5) - 0.3) * 2) / 2
    ymax = np.ceil( (np.percentile(uhi_vals, 99.5) + 0.5) * 2) / 2
else:
    ymin, ymax = -1.0, 6.0

sat_vals = np.concatenate([v for v in sat_temporal.values() if len(v) > 0] or [np.array([])])
if len(df_sat_ts):
    sat_vals = np.concatenate([sat_vals,
                               df_sat_ts['suhi_p25'].dropna().values,
                               df_sat_ts['suhi_p75'].dropna().values])
sat_vals = sat_vals[np.isfinite(sat_vals)] if len(sat_vals) > 0 else sat_vals
if len(sat_vals):
    sat_ymin = np.floor((np.percentile(sat_vals,  1) - 0.5) * 2) / 2
    sat_ymax = np.ceil( (np.percentile(sat_vals, 99) + 0.5) * 2) / 2
else:
    sat_ymin, sat_ymax = 0.0, 12.0

def _align_zero(prim_min, prim_max, sec_min, sec_max):
    if not (prim_min < 0 < prim_max):
        return sec_min, sec_max
    f    = -prim_min / (prim_max - prim_min)
    span = max(sec_max / (1 - f), -sec_min / f)
    if not np.isfinite(span) or span == 0:
        return sec_min, sec_max
    return -f * span, (1 - f) * span

sat_ymin, sat_ymax = _align_zero(ymin, ymax, sat_ymin, sat_ymax)

# ── PLOT HELPERS ───────────────────────────────────────────────────────────────

def shade_nights_ts(ax, start, end, sunrise_h=HW4_SUNRISE_H, sunset_h=HW4_SUNSET_H):
    """Grey night bands (darker) and day bands (lighter) on a datetime x-axis, using
    the real Bolzano sunrise/sunset for the HW4 event midpoint (see sun_times_utc())."""
    day = pd.Timestamp(start)
    t1  = pd.Timestamp(end) + pd.Timedelta('23h59m59s')
    while day <= t1:
        ax.axvspan(day,                                day + pd.Timedelta(hours=sunrise_h),
                   color='#e8e8f0', alpha=0.55, zorder=0)
        ax.axvspan(day + pd.Timedelta(hours=sunrise_h), day + pd.Timedelta(hours=sunset_h),
                   color='#e8e8f0', alpha=0.18, zorder=0)
        ax.axvspan(day + pd.Timedelta(hours=sunset_h),  day + pd.Timedelta('24h'),
                   color='#e8e8f0', alpha=0.55, zorder=0)
        day += pd.Timedelta('1d')


def plot_ts(ax, df, mean_col, p25_col, p75_col, color, ls, lw=1.4, label=''):
    if len(df) == 0: return
    lo = np.minimum(df[p25_col], df[mean_col])
    hi = np.maximum(df[p75_col], df[mean_col])
    ax.fill_between(df['time'], lo, hi,
                    color=color, alpha=ALPHA_SPATIAL, linewidth=0, zorder=2)
    ax.plot(df['time'], df[mean_col],
            color=color, ls=ls, lw=lw, label=label, zorder=3)


def plot_diurnal(ax, temporal, color, ls, lw=1.6, label=''):
    hours = np.arange(24, dtype=float)
    means = np.full(24, np.nan)
    p25s  = np.full(24, np.nan)
    p75s  = np.full(24, np.nan)

    for h in range(24):
        v = temporal[h]
        v = v[np.isfinite(v)] if len(v) > 0 else v
        if len(v) >= 2:
            means[h] = float(np.nanmean(v))
            p25s[h]  = float(np.percentile(v, 25))
            p75s[h]  = float(np.percentile(v, 75))
        elif len(v) == 1:
            means[h] = p25s[h] = p75s[h] = float(v[0])

    ax.fill_between(hours, p25s, p75s,
                    color=color, alpha=ALPHA_TEMPORAL, linewidth=0, zorder=2)
    ax.plot(hours, means, color=color, ls=ls, lw=lw, label=label, zorder=3)


# ── FIGURE ─────────────────────────────────────────────────────────────────────
# Row order: a) full UHI ts, b) cloud heatmap, c) wind stick, d) diurnal composite.
# Rows a-c share one calendar-date x-axis; row d keeps its own hour-of-day axis.
# A dedicated colorbar column (col 1) keeps every row's main Axes (col 0) the
# same width -- fig.colorbar(..., ax=...) would otherwise shrink only the rows
# that carry a colorbar, breaking the shared x-axis alignment.
fig = plt.figure(figsize=(20, 15))
gs  = fig.add_gridspec(4, 2, height_ratios=[6.0, 2.6, 2.2, 4.0],
                        width_ratios=[1, 0.018], hspace=0.3, wspace=0.03)
ax_ts   = fig.add_subplot(gs[0, 0])
ax_cld  = fig.add_subplot(gs[1, 0], sharex=ax_ts)
ax_wind = fig.add_subplot(gs[2, 0], sharex=ax_ts)
ax_dc   = fig.add_subplot(gs[3, 0])
cax_cld  = fig.add_subplot(gs[1, 1])
cax_wind = fig.add_subplot(gs[2, 1])

# Twin axes for satellite
ax_ts2 = ax_ts.twinx()
ax_dc2 = ax_dc.twinx()
import matplotlib.ticker as mticker
for ax2, ax_prim in ((ax_ts2, ax_ts), (ax_dc2, ax_dc)):
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_edgecolor(COLORS['sat'])
    ax2.spines['right'].set_linewidth(1.0)
    ax2.set_ylim(sat_ymin, sat_ymax)
    ax2.set_ylabel('SUHI — satellite (°C)', fontsize=13, color=COLORS['sat'])
    ax2.tick_params(axis='y', labelcolor=COLORS['sat'], labelsize=12, width=1.0)
    ax2.axhline(0, color=COLORS['sat'], lw=0.9, ls=(0, (3, 5)), alpha=0.55, zorder=1)
    _loc   = mticker.AutoLocator()
    _ticks = _loc.tick_values(sat_ymin, sat_ymax)
    if 0.0 not in _ticks:
        _ticks = np.sort(np.concatenate([_ticks, [0.0]]))
    _ticks = _ticks[(_ticks >= sat_ymin) & (_ticks <= sat_ymax)]
    ax2.set_yticks(_ticks)
    ax_prim.set_zorder(ax2.get_zorder() + 1)
    ax_prim.patch.set_visible(False)

# ── ROW 1 — FULL UHI TIME SERIES ────────────────────────────────────────────
shade_nights_ts(ax_ts, HW4_START, HW4_END)
ax_ts.axhline(0, color='#333333', lw=0.8, ls='--', alpha=0.5, zorder=1)
for d in pd.date_range(HW4_START, HW4_END, freq='D'):
    ax_ts.axvline(d, color="gray", linestyle="--", linewidth=0.7, zorder=1)

for _t, _lbl in [('2023-08-13 12:00', 'Day'), ('2023-08-13 21:00', 'Night')]:
    ax_ts.text(pd.Timestamp(_t), 0.96, _lbl, ha='center', va='top',
               fontsize=10, color='#777777', style='italic', fontweight='semibold',
               transform=ax_ts.get_xaxis_transform(), zorder=4)

plot_ts(ax_ts, uc_ts,  'uhi_mean', 'uhi_p25', 'uhi_p75', COLORS['urbclim'], '-',  label='UrbClim')
plot_ts(ax_ts, wrf_ts, 'uhi_mean', 'uhi_p25', 'uhi_p75', COLORS['wrf'],     '--', label='WRF')
plot_ts(ax_ts, na_ts,  'uhi_mean', 'uhi_p25', 'uhi_p75', COLORS['netatmo'], ':',  lw=1.8, label='Netatmo')

if len(df_sat_ts):
    ax_ts2.errorbar(df_sat_ts['time'], df_sat_ts['suhi_mean'],
                    yerr=[df_sat_ts['suhi_mean'] - df_sat_ts['suhi_p25'],
                          df_sat_ts['suhi_p75']  - df_sat_ts['suhi_mean']],
                    fmt='D', color=COLORS['sat'], markersize=5, capsize=3,
                    markeredgewidth=0.8, markeredgecolor='white',
                    elinewidth=1.0, zorder=5, label='Satellite')

ax_ts.set_ylabel('UHI (°C)', fontsize=13)
ax_ts.set_ylim(ymin, ymax)
ax_ts.xaxis.set_major_locator(DayLocator(interval=2))
ax_ts.xaxis.set_major_formatter(DateFormatter('%d %b'))
ax_ts.grid(axis='y', alpha=0.25, lw=0.6)
ax_ts.grid(axis='x', alpha=0.15, lw=0.6)
ax_ts.set_xlim(pd.Timestamp(HW4_START), pd.Timestamp(HW4_END) + pd.Timedelta('23h59m59s'))
ax_ts.set_title('a) Hourly UHI time series', loc='left', fontsize=12, fontweight='bold')

iqr_handle_a = (
    mpatches.Patch(facecolor=COLORS['urbclim'], alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['wrf'],     alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['netatmo'], alpha=0.5, edgecolor='none'),
)
night_patch_a = mpatches.Patch(facecolor='#e8e8f0', edgecolor='none', alpha=0.8)
h_uhi = ax_ts.get_legend_handles_labels()
h_sat = ax_ts2.get_legend_handles_labels()
ax_ts.legend(
    h_uhi[0] + [iqr_handle_a, night_patch_a] + h_sat[0],
    h_uhi[1] + ['P25–P75 (spatial IQR)', 'Night (solar, mid-Aug)'] + h_sat[1],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    loc='upper left', fontsize=11, frameon=True, framealpha=0.9, ncol=2
)

# ── ROW 2 — CLOUD COVER HEATMAP (altitude x time) ───────────────────────────
shade_nights_ts(ax_cld, HW4_START, HW4_END)
mesh = ax_cld.pcolormesh(times_wrf, ALT_GRID_M, cloud_heatmap,
                          cmap='Blues', vmin=0, vmax=1, shading='auto', zorder=2)
for d in pd.date_range(HW4_START, HW4_END, freq='D'):
    ax_cld.axvline(d, color="gray", linestyle="--", linewidth=0.7, zorder=3)
ax_cld.set_ylabel('Altitude (m AGL)', fontsize=13)
ax_cld.set_ylim(0, 6000)
ax_cld.set_title('b) Cloud cover over Bolzano — fraction by altitude & time (WRF)',
                  loc='left', fontsize=12, fontweight='bold')
cbar_cld = fig.colorbar(mesh, cax=cax_cld)
cbar_cld.set_label('Cloud fraction (0-1)', fontsize=10)

# ── ROW 3 — 0-500 m AGL WIND STICK PLOT ─────────────────────────────────────
shade_nights_ts(ax_wind, HW4_START, HW4_END)
x = date2num(df_wind['time'])
ws_max = float(np.nanmax(df_wind['ws_bulk500']))
q = ax_wind.quiver(x, np.zeros(len(x)), df_wind['u_bulk500'], df_wind['v_bulk500'],
                    df_wind['ws_bulk500'], cmap='viridis',
                    angles='uv', scale_units='y', scale=1.0,
                    width=0.0018, headwidth=3.5, headlength=4.5, headaxislength=4,
                    pivot='tail', zorder=3)
for d in pd.date_range(HW4_START, HW4_END, freq='D'):
    ax_wind.axvline(d, color="gray", linestyle="--", linewidth=0.7, zorder=2)
ax_wind.axhline(0, color='#555555', lw=0.7, zorder=2)
ax_wind.set_ylim(-ws_max * 1.15, ws_max * 1.15)
ax_wind.set_ylabel('Wind vector\n(m s$^{-1}$)', fontsize=12)
ax_wind.set_title(f'c) Bulk wind, 0-{WIND_LAYER_TOP_M:.0f} m AGL, over Bolzano (WRF)',
                   loc='left', fontsize=12, fontweight='bold')
qk = ax_wind.quiverkey(q, 0.93, 1.1, 5, "5 m/s", labelpos='W', coordinates='axes',
                        fontproperties={'size': 9})
cbar_wind = fig.colorbar(q, cax=cax_wind)
cbar_wind.set_label('Wind speed (m s$^{-1}$)', fontsize=10)

# rows a-c share one calendar-date x-axis (enforced by sharex above)
ax_wind.xaxis.set_major_locator(DayLocator(interval=2))
ax_wind.xaxis.set_major_formatter(DateFormatter('%d %b'))
ax_wind.set_xlabel('Date (UTC)', fontsize=13)
plt.setp(ax_ts.get_xticklabels(), visible=False)
plt.setp(ax_cld.get_xticklabels(), visible=False)

# ── ROW 4 — MEAN DIURNAL CYCLE (own hour-of-day axis) ───────────────────────
ax_dc.axvspan(-0.7, HW4_SUNRISE_H, color='#e8e8f0', alpha=0.55, zorder=0)
ax_dc.axvspan(HW4_SUNSET_H, 23.7,  color='#e8e8f0', alpha=0.55, zorder=0)
ax_dc.axhline(0, color='#333333', lw=0.8, ls='--', alpha=0.5, zorder=1)

plot_diurnal(ax_dc, uc_temporal,  COLORS['urbclim'], '-',  label='UrbClim')
plot_diurnal(ax_dc, wrf_temporal, COLORS['wrf'],     '--', label='WRF')
plot_diurnal(ax_dc, na_temporal,  COLORS['netatmo'], ':',  lw=1.8, label='Netatmo')

first_sat = True
for h in range(24):
    vals = sat_temporal[h];  vals = vals[np.isfinite(vals)] if len(vals) > 0 else vals
    if len(vals) == 0: continue
    med = float(np.median(vals)); p25 = float(np.percentile(vals, 25)); p75 = float(np.percentile(vals, 75))
    ax_dc2.errorbar(h, med, yerr=[[med - p25], [p75 - med]],
                    fmt='D', color=COLORS['sat'], markersize=6, capsize=4,
                    markeredgewidth=0.8, markeredgecolor='white',
                    elinewidth=1.2, zorder=7, label='Satellite' if first_sat else '')
    first_sat = False

ax_dc.set_xlim(-0.7, 23.7)
ax_dc2.set_xlim(-0.7, 23.7)
ax_dc.set_ylim(ymin, ymax)
ax_dc.set_xticks(range(0, 24, 3))
ax_dc.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)])
ax_dc.set_xlabel('Hour (UTC)', fontsize=13)
ax_dc.set_ylabel('UHI (°C)', fontsize=13)
ax_dc.grid(axis='y', alpha=0.25, lw=0.6)
ax_dc.set_title('d) Mean diurnal UHI composite', loc='left', fontsize=12, fontweight='bold')

ls_map = {'urbclim': '-', 'wrf': '--', 'netatmo': ':'}
lw_map = {'urbclim': 1.6, 'wrf': 1.6,  'netatmo': 1.8}
iqr_handle_b = (
    mpatches.Patch(facecolor=COLORS['urbclim'], alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['wrf'],     alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['netatmo'], alpha=0.5, edgecolor='none'),
)
legend_handles = [
    mlines.Line2D([0], [0], color=COLORS[k], ls=ls_map[k], lw=lw_map[k])
    for k in ('urbclim', 'wrf', 'netatmo')
] + [iqr_handle_b, mpatches.Patch(facecolor='#e8e8f0', edgecolor='none', alpha=0.8),
     mlines.Line2D([0], [0], color=COLORS['sat'], lw=0, marker='D', markersize=6,
                   markeredgecolor='white', markeredgewidth=0.8)]
legend_labels = [k.capitalize() for k in ('urbclim', 'wrf', 'netatmo')] + \
                ['P25–P75 (day-to-day IQR)', 'Night (solar, mid-Aug)', 'Satellite']
ax_dc.legend(handles=legend_handles, labels=legend_labels,
             handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
             loc='upper left', fontsize=11, frameon=True, framealpha=0.9, ncol=2)

# ── SAVE ───────────────────────────────────────────────────────────────────────
from datetime import datetime
ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
for ext in ('pdf', 'png'):
    p = os.path.join(OUTPUT_DIR, f'hw4_dashboard_cloud_wind_{LAPSE_MODE}_{ts_str}.{ext}')
    plt.savefig(p, dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved -> {p}')

print('Done.')
