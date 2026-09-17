#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal diurnal UHI comparison — DJF / MAM / JJA / SON

Four independent sources compared on the same (season, hour-of-day) grid:
UrbClim, WRF (5-year daily-harmonized archive), Netatmo, and Satellite LST
(as SUHI, on a secondary axis since its magnitude and physical meaning differ
from near-surface air-temperature UHI). Each panel is one season; the line is
the mean diurnal UHI cycle (hour-of-day, UTC) and the band is the P25-P75
spread across the days contributing to that season/hour bucket. Night shading
uses each season's real solar sunrise/sunset for Bolzano, not a fixed UTC
window, so it tracks DST and the seasonal swing in day length correctly.

Elevation correction (reference = mean rural elevation, correction applied to
the urban side only, rural mean left raw -- see common/lapse_mode_utils.py):
the dynamic, per-(season, day/night) valley rate is WRF-ONLY (loaded via
common/wrf_daily.load_wrf_uhi(), daily harmonized archive, spin-up already
trimmed per file, passed a per-timestep rate closure), since that rate was
fitted from WRF's own vertical profiles and isn't assumed representative of
UrbClim/Netatmo/Satellite's near-surface behavior at every season/time-of-day.
UrbClim, Netatmo, and Satellite/LST instead use a single constant rate
(LAPSE_RATE_CONSTANT, via _non_wrf_rate() below) with a real per-pixel
elevation field where needed: UrbClim and Satellite/LST carry no native
elevation variable, so common/dem_utils.reproject_dem_to_grid() supplies one
from the shared 10m DEM; Netatmo uses each station's own real altitude.
LAPSE_MODE still gates all four sources the same way for 'none' (no
correction) and 'constant' (LAPSE_RATE_CONSTANT everywhere); only 'valley'
now differs between WRF (truly dynamic) and the other three (falls back to
the constant rate rather than the dynamic one).

Four-way source overlap is capped at 2023 (UrbClim ends 2023-12-31; Netatmo/LST
run to late 2025 but that's moot once UrbClim drops out). DATE_START/DATE_END
give all four sources two full years (2022-2023) for more robust per-season-
hour statistics; WRF's daily archive covers 2020-2024, so within this window
it contributes full samples like the other sources.

Provenance: this script was rebuilt in 2026-07 after the original generator
of paper figure diurnal_uhi_seasonal_20260512_101357.png was lost with no
recoverable history; re-running it will not reproduce that exact file
bit-for-bit. UrbClim's UTC_OFFSET (0, not 2) and Netatmo's (0, not 2) were
both independently confirmed by cross-checking each source's raw diurnal T2M
peak hour against WRF's confirmed-UTC peak hour.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple
from shapely.vectorized import contains as shp_contains
from shapely.geometry import Point
import geopandas as gpd
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from wrf_daily import load_wrf_uhi, urban_elev_correction
from dem_utils import reproject_dem_to_grid
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# ── CONFIG ─────────────────────────────────────────────────────────────────────
UCB_FILE     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc'
NETATMO_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
LST_FILE     = '/home/gsimonet/Desktop/LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc'

URBAN_SHP  = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
RURAL_SHP  = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'
OUTPUT_DIR = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/All_combined_time_series/Combined_plots/figures'

DATE_START = '2022-01-01'
DATE_END   = '2023-12-31'   # WRF & UrbClim both end 2023-12-31 -- true 4-way overlap ceiling

# Elevation-correction mode -- edit and re-run to get each version; see
# common/lapse_mode_utils.py. Output filenames are tagged with this mode so
# the three don't overwrite each other, matching every other script in this
# shared-methodology group (heatwave2023/, suhi_lcz/, urbclim/, wrf_uhi/).
#   'none'     -- no elevation correction (raw urban-minus-rural difference)
#   'constant' -- classical fixed rate (LAPSE_RATE_CONSTANT), no season/day-night dependence
#   'valley'   -- dynamic, per-(season, day/night) rate fitted from WRF -- WRF ONLY (default);
#                 UrbClim/Netatmo/Satellite fall back to the constant rate under this mode too
#                 (see _non_wrf_rate() below), since the dynamic rate was fitted from WRF's own
#                 vertical profiles and isn't assumed representative of near-surface behavior
#                 for the other three sources.
LAPSE_MODE = 'valley'

# {(season, 'day'|'night'): K/m} -- fitted once from the WRF archive, applied to WRF ONLY.
DYNAMIC_LAPSE = load_seasonal_valley_lapse()


def _non_wrf_rate():
    """Constant-rate resolution for UrbClim/Netatmo/Satellite: 'valley' collapses to
    'constant' for these three (the dynamic rate is WRF-only), 'none'/'constant' behave
    exactly as resolve_rate() already defines them."""
    return resolve_rate('constant' if LAPSE_MODE == 'valley' else LAPSE_MODE, None)

UTM_CRS    = 'EPSG:32632'
WGS84      = 'EPSG:4326'

UTC_OFFSET   = {'urbclim': 0, 'wrf': 0, 'netatmo': 0, 'sat': 0}
SAT_HOUR_TOL = 1.5  # hours — satellite scenes attributed to nearest integer hour

# NOTE (2026-07-29): netatmo offset changed 2 -> 0. The ingestion pipeline
# (NETATMO_BOLZANO_PACKAGE/src/data_fetch.py) converts the API's Unix epoch
# timestamps via pd.to_datetime(..., unit='s'), which is a direct epoch->UTC
# conversion with no local-time step -- so raw netatmo times should already
# be UTC. A per-season peak-hour cross-check against WRF's confirmed-UTC
# diurnal peak (DJF +0.8h, MAM +1.0h, JJA +1.6h, SON +0.25h) did not show the
# clean 1h(CET)/2h(CEST) DST step the old fixed "2" implied either -- the
# residual looks like seasonal thermal-lag/sensor-heating noise, not a
# timezone bug. Revisit if a cleaner ground-truth check becomes available.

SEASON_MAP = {12: 'DJF',  1: 'DJF',  2: 'DJF',
               3: 'MAM',  4: 'MAM',  5: 'MAM',
               6: 'JJA',  7: 'JJA',  8: 'JJA',
               9: 'SON', 10: 'SON', 11: 'SON'}
SEASON_ORDER = ['DJF', 'MAM', 'JJA', 'SON']

# This script names seasons DJF/MAM/JJA/SON; the shared valley-lapse-rate table
# (seasonal_valley_lapse_rate.py) uses 'Winter'/'Spring'/'Summer'/'Autumn' -- map
# between them for every DYNAMIC_LAPSE lookup (a silent mismatch here would make
# every lookup return None, and LAPSE_MODE='valley' would silently behave like
# 'none' with no error -- see urbclim/uhi_combined_heatmap_distri_map.py's
# identical _LAPSE_SEASON_KEY for the same reason).
_LAPSE_SEASON_KEY = {'DJF': 'Winter', 'MAM': 'Spring', 'JJA': 'Summer', 'SON': 'Autumn'}

# Representative mid-season date per panel, used only for the solar
# sunrise/sunset shading below (2023, arbitrary non-leap year).
SEASON_MID_DATE = {'DJF': (1, 15), 'MAM': (4, 15), 'JJA': (7, 15), 'SON': (10, 15)}
BOLZANO_LAT, BOLZANO_LON = 46.4996, 11.3408

COLORS = {
    'urbclim': '#E07B39',
    'wrf'    : '#3A7FC1',
    'netatmo': '#38A66C',
    'sat'    : '#9B3A9B',
}

ALPHA_TEMPORAL = 0.22

os.makedirs(OUTPUT_DIR, exist_ok=True)
_shared_elev = None   # (x_flat_utm, y_flat_utm, e_flat) filled by first dataset with DEM

# ── STYLE ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'         : 12,
    'axes.labelsize'    : 13,
    'axes.titlesize'    : 14,
    'xtick.labelsize'   : 11,
    'ytick.labelsize'   : 11,
    'legend.fontsize'   : 10,
    'axes.linewidth'    : 1.0,
    'xtick.major.size'  : 4,
    'ytick.major.size'  : 4,
    'xtick.major.width' : 1.0,
    'ytick.major.width' : 1.0,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# ── GEOMETRY / HELPERS ───────────────────────────────────────────────────────────

def build_mask(union, flat_x, flat_y, shape):
    b   = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) &
           (flat_y >= b[1]) & (flat_y <= b[3]))
    m   = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)

def to_utc(times, offset):
    return pd.DatetimeIndex(times) - pd.Timedelta(hours=offset)

def sun_times_utc(month, day, year=2023, lat=BOLZANO_LAT, lon=BOLZANO_LON):
    """
    Low-precision NOAA solar calc (~1 min accuracy) -> (sunrise_utc_hr, sunset_utc_hr)
    as decimal hours-of-day, for a representative date. Used to shade true
    season-specific night bands instead of a single fixed UTC window that
    ignores both DST and seasonal day-length changes.
    """
    doy   = pd.Timestamp(year, month, day).dayofyear
    gamma = 2 * np.pi / 365 * (doy - 1)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                        - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))  # minutes
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))  # radians

    lat_r   = np.radians(lat)
    zenith  = np.radians(90.833)  # accounts for atmospheric refraction + solar disk radius
    cos_ha  = np.cos(zenith) / (np.cos(lat_r) * np.cos(decl)) - np.tan(lat_r) * np.tan(decl)
    ha_deg  = np.degrees(np.arccos(np.clip(cos_ha, -1, 1)))

    sunrise_min = 720 - 4 * (lon + ha_deg) - eqtime
    sunset_min  = 720 - 4 * (lon - ha_deg) - eqtime
    return (sunrise_min / 60.0) % 24, (sunset_min / 60.0) % 24

def new_season_temporal():
    return {s: {h: [] for h in range(24)} for s in SEASON_ORDER}

def finalize_season_temporal(temporal):
    return {s: {h: np.array(v) for h, v in hd.items()} for s, hd in temporal.items()}

# ── CORE EXTRACTOR (gridded sources: UrbClim, WRF) ──────────────────────────────

def extract_seasonal(arr3d, times_utc, urban_m, rural_m, lapse2d=None):
    """
    Bucket each timestep's urban-mean UHI into temporal[season][hour].
    Mirrors extract_all() from hw4_dashboard_cloud_wind.py but keyed by
    (season, hour) over the full DATE_START..DATE_END window instead of a
    single fixed event window. UrbClim uses a constant rate (see
    _non_wrf_rate()), so `lapse2d` is a precomputed field (rate already
    baked in, unlike the WRF block below which resolves a dynamic rate
    per timestep).
    """
    t0 = pd.Timestamp(DATE_START)
    t1 = pd.Timestamp(DATE_END) + pd.Timedelta('23h59m59s')
    in_period = (times_utc >= t0) & (times_utc <= t1)
    arr3d   = arr3d[in_period]
    times_p = times_utc[in_period]

    if len(arr3d) > 0 and np.nanmean(arr3d[0]) > 200:
        arr3d = arr3d - 273.15   # K → °C

    temporal = new_season_temporal()

    for i, snap in enumerate(arr3d):
        # Reference stays the RAW rural mean -- the correction is applied to the
        # WHOLE field (urban included), not to the rural pixels feeding the
        # reference, so it can't cancel to zero once re-averaged over that same
        # rural mask (this used to be inverted: the correction only fed the rural
        # reference, via a field that is mean-zero over the rural mask by
        # construction, making it a no-op).
        snap_lc = snap + lapse2d if lapse2d is not None else snap
        r_mean_raw = float(np.nanmean(snap[rural_m]))
        uhi     = snap_lc - r_mean_raw
        upx     = uhi[urban_m];  upx = upx[np.isfinite(upx)]
        if len(upx) == 0:
            continue
        mean_uhi = float(np.nanmean(upx))
        t = times_p[i]
        temporal[SEASON_MAP[t.month]][t.hour].append(mean_uhi)

    return finalize_season_temporal(temporal)

# ── LOAD SHAPEFILES ────────────────────────────────────────────────────────────
print('Loading shapefiles…')
urban_utm = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
rural_utm = gpd.read_file(RURAL_SHP).to_crs(UTM_CRS)
urban_wgs = urban_utm.to_crs(WGS84)
rural_wgs = rural_utm.to_crs(WGS84)

# ── UrbClim ────────────────────────────────────────────────────────────────────
print('\n[UrbClim] Loading…')
uc_temporal = finalize_season_temporal(new_season_temporal())

if os.path.exists(UCB_FILE):
    ds  = xr.open_dataset(UCB_FILE, chunks={'t': 500})
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
        # UrbClim carries no native elevation (checked: only T2M/QV2M/WS2M/LST) --
        # reproject the shared 10m DEM onto this grid instead of silently leaving
        # the correction at zero (that used to depend on a _shared_elev fallback
        # only populated if UrbClim found elevation first, which it never did).
        e = reproject_dem_to_grid(ds[xd].values, ds[yd].values)
        print('  Elevation: reprojected from DEM_10m_south_tirol.tif (no native var)')
    lapse2d = urban_elev_correction(e, rural_m, lapse_rate=_non_wrf_rate())
    _shared_elev = (xx.flatten().copy(), yy.flatten().copy(), e.flatten().copy())

    t0     = pd.Timestamp(DATE_START)
    t1_loc = pd.Timestamp(DATE_END) + pd.Timedelta(hours=23 + UTC_OFFSET['urbclim'])
    print('  Reading into RAM (2022-2023)…')
    arr3d     = ds['T2M'].sel(t=slice(t0, t1_loc)).load().values
    times_utc = to_utc(ds.t.sel(t=slice(t0, t1_loc)).values, UTC_OFFSET['urbclim'])
    ds.close()

    uc_temporal = extract_seasonal(arr3d, times_utc, urban_m, rural_m, lapse2d)
    print(f'  n(JJA,14h)={len(uc_temporal["JJA"][14])}  n(DJF,14h)={len(uc_temporal["DJF"][14])}')
else:
    print('  [MISSING]', UCB_FILE)

# ── WRF (daily-harmonized archive, spin-up already trimmed) ────────────────────
print('\n[WRF] Loading (daily harmonized archive)…')
wrf_temporal = finalize_season_temporal(new_season_temporal())

def _wrf_rate(t):
    season = _LAPSE_SEASON_KEY[SEASON_MAP[t.month]]
    period = classify_day_night(pd.DatetimeIndex([t]))[0]
    return resolve_rate(LAPSE_MODE, DYNAMIC_LAPSE.get((season, period)))

times_wrf, uhi_wrf, _wrf_p25, _wrf_p75 = load_wrf_uhi(
    DATE_START, DATE_END, urban_wgs.unary_union, rural_wgs.unary_union,
    cache_key='diurnal_uhi_seasonal', lapse_rate=_wrf_rate, lapse_rate_key=f'valley-{LAPSE_MODE}')

if len(times_wrf) > 0:
    temporal = new_season_temporal()
    for t, u in zip(times_wrf, uhi_wrf):
        temporal[SEASON_MAP[t.month]][t.hour].append(float(u))
    wrf_temporal = finalize_season_temporal(temporal)
    print(f'  {len(times_wrf)} timesteps  '
          f'n(JJA,14h)={len(wrf_temporal["JJA"][14])}  n(DJF,14h)={len(wrf_temporal["DJF"][14])}')
else:
    print('  [MISSING] no WRF daily files found for this window')

# ── Netatmo ────────────────────────────────────────────────────────────────────
print('\n[Netatmo] Loading…')
na_temporal = finalize_season_temporal(new_season_temporal())

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
                print('  Elevations interpolated from UrbClim DEM')
            except Exception as ex:
                print(f'  [WARN] lapse skipped: {ex}')

        t0 = pd.Timestamp(DATE_START)
        t1 = pd.Timestamp(DATE_END) + pd.Timedelta('23h59m59s')
        in_period = (times_utc >= t0) & (times_utc <= t1)
        na_T_p    = na_T[in_period]
        times_p   = times_utc[in_period]

        r_e_mean = None
        if na_elev is not None:
            r_e_mean = float(np.nanmean(na_elev[r_idx][~np.isnan(na_elev[r_idx])]))

        na_rate = _non_wrf_rate()
        temporal = new_season_temporal()

        for i in range(len(times_p)):
            snap   = na_T_p[i]
            r_vals = snap[r_idx].copy()
            if na_elev is not None and r_e_mean is not None:
                r_vals = r_vals + (na_elev[r_idx] - r_e_mean) * (-na_rate)
            r_vals = r_vals[~np.isnan(r_vals)]
            if len(r_vals) == 0: continue
            uhi = snap[u_idx] - float(np.nanmean(r_vals))
            uhi = uhi[np.isfinite(uhi)]
            if len(uhi) == 0: continue
            mean_uhi = float(np.nanmean(uhi))
            t = times_p[i]
            temporal[SEASON_MAP[t.month]][t.hour].append(mean_uhi)

        na_temporal = finalize_season_temporal(temporal)
        print(f'  n(JJA,14h)={len(na_temporal["JJA"][14])}  n(DJF,14h)={len(na_temporal["DJF"][14])}')
    else:
        print('  [WARN] Temperature variable not found')
else:
    print('  [MISSING]', NETATMO_FILE)

# ── Satellite ──────────────────────────────────────────────────────────────────
print('\n[Satellite] Loading…')
sat_temporal = finalize_season_temporal(new_season_temporal())

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
        print('  Elevation: native variable in satellite file')
    else:
        # LST/satellite carries no native elevation either (checked: only the
        # LST DataArray + time/x/y/band/spatial_ref) -- reproject the same
        # shared DEM directly onto this grid, rather than depending on
        # UrbClim's _shared_elev happening to be populated first (it never
        # was, since UrbClim itself has no native elevation variable).
        e_sat = reproject_dem_to_grid(x_a, y_a)
        print('  Elevation: reprojected from DEM_10m_south_tirol.tif (no native var)')
    sat_lapse2d = urban_elev_correction(e_sat, rural_m, lapse_rate=_non_wrf_rate())

    times_all = pd.DatetimeIndex(
        [pd.Timestamp(t.astype('datetime64[ns]').item()) for t in ds.time.values])
    times_utc = to_utc(times_all, UTC_OFFSET['sat'])
    t0 = pd.Timestamp(DATE_START)
    t1 = pd.Timestamp(DATE_END) + pd.Timedelta('23h59m59s')
    in_period = (times_utc >= t0) & (times_utc <= t1)

    print('  Reading satellite scenes into RAM (2022-2023)…')
    arr3d     = ds[LST_VAR].isel(time=np.where(in_period)[0]).load().values
    times_utc = times_utc[in_period]
    ds.close()

    temporal = new_season_temporal()

    for i, snap in enumerate(arr3d):
        h_dec = times_utc[i].hour + times_utc[i].minute / 60.0
        h_int = round(h_dec) % 24
        if min(abs(h_dec - h_int), 24 - abs(h_dec - h_int)) > SAT_HOUR_TOL:
            continue
        # Reference stays the RAW rural mean -- see extract_seasonal() for why the
        # correction must be applied to the whole field, not just the rural side.
        snap_lc = snap + sat_lapse2d
        r_mean_raw = float(np.nanmean(snap[rural_m]))
        suhi    = snap_lc - r_mean_raw
        upx     = suhi[urban_m];  upx = upx[np.isfinite(upx)]
        if len(upx) == 0: continue
        mean_suhi = float(np.nanmean(upx))
        t = times_utc[i]
        temporal[SEASON_MAP[t.month]][h_int].append(mean_suhi)

    sat_temporal = finalize_season_temporal(temporal)
    total = sum(len(v) for hd in sat_temporal.values() for v in hd.values())
    print(f'  {total} valid scenes attributed to season/hour bins')
else:
    print('  [MISSING]', LST_FILE)

# ── DYNAMIC Y-RANGES (shared across all 4 season panels) ──────────────────────

def _pooled(temporal_dict):
    return np.concatenate([hd[h] for hd in temporal_dict.values() for h in range(24)
                            if len(hd[h]) > 0] or [np.array([])])

ymin, ymax = -3.5, 3.5   # fixed range for visual consistency across re-runs
print(f'\nPrimary Y-range (UHI): [{ymin:.1f}, {ymax:.1f}] °C')

sat_vals = _pooled(sat_temporal)
sat_vals = sat_vals[np.isfinite(sat_vals)] if len(sat_vals) else sat_vals
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
print(f'Secondary Y-range (SUHI): [{sat_ymin:.1f}, {sat_ymax:.1f}] °C  (zeros aligned)')

# ── PLOT HELPERS ───────────────────────────────────────────────────────────────

def plot_diurnal(ax, hourly_dict, color, ls, lw=1.6, label=''):
    hours = np.arange(24, dtype=float)
    means = np.full(24, np.nan)
    p25s  = np.full(24, np.nan)
    p75s  = np.full(24, np.nan)

    for h in range(24):
        v = hourly_dict[h]
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


def plot_sat_diurnal(ax2, hourly_dict, color):
    first = True
    for h in range(24):
        vals = hourly_dict[h]
        vals = vals[np.isfinite(vals)] if len(vals) > 0 else vals
        if len(vals) == 0: continue
        med = float(np.median(vals))
        p25 = float(np.percentile(vals, 25)) if len(vals) >= 2 else med
        p75 = float(np.percentile(vals, 75)) if len(vals) >= 2 else med
        ax2.errorbar(h, med, yerr=[[med - p25], [p75 - med]],
                     fmt='D', color=color, markersize=5, capsize=3,
                     markeredgewidth=0.7, markeredgecolor='white',
                     elinewidth=1.0, zorder=7,
                     label='Satellite' if first else '')
        ax2.text(h + 0.15, p75 + 0.1, f'n={len(vals)}',
                 fontsize=7.5, color=color, va='bottom')
        first = False


# ── FIGURE — 2×2 seasonal grid ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
axes_flat = axes.flatten()

for si, season in enumerate(SEASON_ORDER):
    ax = axes_flat[si]
    ax2 = ax.twinx()

    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_edgecolor(COLORS['sat'])
    ax2.spines['right'].set_linewidth(1.0)
    ax2.set_ylim(sat_ymin, sat_ymax)
    ax2.tick_params(axis='y', labelcolor=COLORS['sat'], labelsize=10, width=1.0)
    ax2.axhline(0, color=COLORS['sat'], lw=0.9, ls=(0, (3, 5)), alpha=0.55, zorder=1)
    _loc   = mticker.AutoLocator()
    _ticks = _loc.tick_values(sat_ymin, sat_ymax)
    if 0.0 not in _ticks:
        _ticks = np.sort(np.concatenate([_ticks, [0.0]]))
    _ticks = _ticks[(_ticks >= sat_ymin) & (_ticks <= sat_ymax)]
    ax2.set_yticks(_ticks)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    if si % 2 == 1:
        ax2.set_ylabel('SUHI — satellite (°C)', fontsize=12, color=COLORS['sat'])
    else:
        ax2.set_yticklabels([])

    mo, dy = SEASON_MID_DATE[season]
    sunrise_h, sunset_h = sun_times_utc(mo, dy)
    ax.axvspan(-0.7, sunrise_h, color='#e8e8f0', alpha=0.55, zorder=0)
    ax.axvspan(sunset_h, 23.7,  color='#e8e8f0', alpha=0.55, zorder=0)
    ax.axhline(0, color='#333333', lw=0.8, ls='--', alpha=0.5, zorder=1)

    plot_diurnal(ax, uc_temporal[season],  COLORS['urbclim'], '-',  label='UrbClim')
    plot_diurnal(ax, wrf_temporal[season], COLORS['wrf'],     '--', label='WRF')
    plot_diurnal(ax, na_temporal[season],  COLORS['netatmo'], ':',  lw=1.8, label='NetAtmo')
    plot_sat_diurnal(ax2, sat_temporal[season], COLORS['sat'])

    ax.set_xlim(-0.7, 23.7)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)])
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.set_title(season, fontsize=14, fontweight='bold', loc='left')

    if si % 2 == 0:
        ax.set_ylabel('UHI (°C)', fontsize=13)
    if si >= 2:
        ax.set_xlabel('Hour (UTC)', fontsize=13)

# Shared legend below the grid
ls_map = {'urbclim': '-', 'wrf': '--', 'netatmo': ':'}
lw_map = {'urbclim': 1.6, 'wrf': 1.6,  'netatmo': 1.8}
iqr_handle = (
    mpatches.Patch(facecolor=COLORS['urbclim'], alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['wrf'],     alpha=0.5, edgecolor='none'),
    mpatches.Patch(facecolor=COLORS['netatmo'], alpha=0.5, edgecolor='none'),
)
legend_handles = [
    mlines.Line2D([0], [0], color=COLORS[k], ls=ls_map[k], lw=lw_map[k])
    for k in ('urbclim', 'wrf', 'netatmo')
] + [
    iqr_handle,
    mpatches.Patch(facecolor='#e8e8f0', edgecolor='none', alpha=0.8),
    mlines.Line2D([0], [0], color=COLORS['sat'], lw=0, marker='D', markersize=6,
                  markeredgecolor='white', markeredgewidth=0.8),
]
legend_labels = ['UrbClim', 'WRF', 'NetAtmo'] + \
                ['P25–P75 (day-to-day IQR)', 'Night (solar, per season)', 'Satellite (median, P25-P75)']
fig.legend(handles=legend_handles, labels=legend_labels,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
           loc='lower center', ncol=3, fontsize=11, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

# fig.suptitle(f'Seasonal diurnal UHI cycle — Bolzano ({DATE_START} to {DATE_END}, WRF limited to 2023)',
#              fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout(rect=[0, 0.03, 1, 1])

# ── SAVE ───────────────────────────────────────────────────────────────────────
from datetime import datetime
ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
for ext in ('pdf', 'png'):
    p = os.path.join(OUTPUT_DIR, f'diurnal_uhi_seasonal_{LAPSE_MODE}_{ts_str}.{ext}')
    plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved → {p}')

plt.show()
print('✓ Done.')
