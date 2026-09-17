#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial UHI maps at 4 times of day + optional all-hours average.
Dual colorbar + mini in-panel distribution histogram version.

Layout: rows = [02h, 08h, 14h, 20h, (All hours)]
        cols = UrbClim | WRF | Netatmo | Satellite

Each panel carries a small horizontal histogram (lower-right inset) whose
bars are coloured with the same RdBu_r / norm as the main map.

Fixed vs. plot_spatial_uhi_maps_heatwave2023_dualcbar_distrib_20260504.py
(see Combined_plots/check_spatial_vs_diurnal_consistency_20260727.py for the
quantified diagnosis of both issues):

  (1) WRF rural reference / lapse correction now matches the project's
      canonical WRF pipeline (extractions/extraction_WRF_20260310.py and
      plot_diurnal_uhi_hw4_detail_full_distri_v2_20260508.py): rural
      reference = `ds.salem.subset(shape=rural_shp)` bounding-box subset
      (not a polygon mask) with lapse correction referenced to that subset's
      MIN elevation. Previously this script used a true rural-polygon mask
      + mean-elevation reference, which produced a ~2.4 degC systematic
      offset vs. every other WRF script in the project. Urban-pixel
      selection for the map DISPLAY is unaffected (both approaches agree
      exactly on which pixels are urban).

  (2) UrbClim & Netatmo timestamps are now converted to UTC (-2 h, matching
      UTC_OFFSET in the diurnal script) before bucketing by hour. Previously
      the row labels said "HH:00 UTC" but the raw (local, UTC+2) file hour
      was used unconverted, so the "same" nominal hour differed from the
      diurnal figure by up to 2 h of real time (~0.6 degC effect at some
      hours). WRF and Satellite are already UTC (offset 0, unaffected).

Dependencies: xarray, numpy, pandas, matplotlib, geopandas, shapely, pyproj,
              scipy, (salem for WRF).
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import geopandas as gpd
from shapely.geometry import Point
from shapely.vectorized import contains as shp_contains
import xarray as xr
import os, sys, warnings
warnings.filterwarnings('ignore')

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print('[WARN] pyproj not found — UTM grids cannot be reprojected to lat/lon')

# Dynamic, per-(season, day/night) valley lapse rate + the same astronomical
# sunrise/sunset day/night classifier used in the seasonal UHI scripts and the HW4
# diurnal/dashboard scripts -- see UHI/seasonal_valley_lapse_rate.py
# and UHI/solar_daynight.py. Replaces this script's old fixed
# LAPSE_RATE = 0.0065 K/m.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night
from dem_utils import reproject_dem_to_grid
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT

# ── CONFIG ────────────────────────────────────────────────────────────────────
UCB_FILE     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc'
WRF_FILE     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2023_subset/WRF_RAW_Alto_Adige_2023.nc'
NETATMO_FILE = '/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc'
LST_FILE     = '/home/gsimonet/Desktop/LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc'

URBAN_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
RURAL_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'

OUTPUT_DIR = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/All_combined_time_series/Combined_plots/figures'

DATE_START   = '2023-08-13'   # HW4 period, matches HW4_START in the diurnal script
DATE_END     = '2023-08-26'   # HW4_END
HOURS_OF_DAY = [2, 8, 14, 20]
SHOW_ALL_ROW = False    # set False to drop the full-period average row
ROW_KEYS     = HOURS_OF_DAY + (['all'] if SHOW_ALL_ROW else [])

SAT_HOUR_TOL = 3.0   # hours — only keep scenes strictly within 3 h of target hour

# Local -> UTC offset applied before bucketing by hour (matches the diurnal script)
# CORRECTED 2026-07-28: UrbClim's native 't' coord was assumed local (CEST,
# UTC+2) and shifted by -2h. Empirical check against WRF (confirmed UTC via
# SIMULATION_START_DATE) shows UrbClim's raw diurnal T2M peak lands on the
# SAME hour as WRF's UTC peak (both domain-mean and urban-masked), not 2h
# later as it would if UrbClim were local. So UrbClim's raw time is already
# UTC -> offset must be 0.
# CORRECTED 2026-07-29: netatmo offset also revised 2 -> 0. The original
# 2h-peak-lag check (comparing raw Netatmo vs WRF diurnal peaks) doesn't hold
# up under a per-season cross-check (see plot_diurnal_uhi_seasonal_20260728.py's
# UTC_OFFSET note), and the ingestion pipeline (NETATMO_BOLZANO_PACKAGE/src/
# data_fetch.py) converts epoch timestamps straight to UTC with no local-time
# step -- so raw netatmo times should already be UTC, same as UrbClim/WRF/sat.
UTC_OFFSET = {'urbclim': 0, 'wrf': 0, 'netatmo': 0, 'sat': 0}

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
# written generally in case DATE_START/DATE_END ever change to span a season boundary.
DYNAMIC_LAPSE = load_seasonal_valley_lapse()

CMAP = 'RdBu_r'

# Fixed symmetric colorbar limits (factor-3 difference between air T and LST)
VMIN_AIR, VMAX_AIR = -5,  5    # UrbClim / WRF / Netatmo (air T)
VMIN_SAT, VMAX_SAT = -15, 15   # Satellite LST (SUHI)

_shared_elev = None   # (x_flat_utm, y_flat_utm, e_flat) from first dataset with DEM
TRY_WRF = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'         : 12,
    'axes.linewidth'    : 0.6,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.spines.left'  : False,
    'axes.spines.bottom': False,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────

def utm_to_latlon(x2d, y2d):
    tr = Transformer.from_crs(UTM_CRS, WGS84, always_xy=True)
    lon, lat = tr.transform(x2d.flatten(), y2d.flatten())
    return lon.reshape(x2d.shape), lat.reshape(y2d.shape)


def unary(gdf):
    """geopandas >=0.14 renamed unary_union -> union_all(); support both."""
    return gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union


def build_mask(union, flat_x, flat_y, shape):
    b = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) &
           (flat_y >= b[1]) & (flat_y <= b[3]))
    m = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)


def to_utc(times, offset):
    """Shift local timestamps to UTC by subtracting the fixed offset (hours)."""
    return pd.DatetimeIndex(times) - pd.Timedelta(hours=offset)


def period_idx(times, hour=None):
    t0 = pd.Timestamp(DATE_START)
    t1 = pd.Timestamp(DATE_END) + pd.Timedelta('23h59m')
    mask = (times >= t0) & (times <= t1)
    if hour is not None:
        mask = mask & (times.hour == hour)
    return np.where(mask)[0]


def sat_closest_idx(times, target_hour):
    """Return (indices, label) of satellite scenes within SAT_HOUR_TOL of target_hour.
    label reports the actual HH:MM span of the selected overpasses (not a
    decimal-hour average), e.g. '09:42-10:15 UTC' or a single 'HH:MM UTC'."""
    in_period = period_idx(times, hour=None)
    if len(in_period) == 0:
        return np.array([], dtype=int), 'No data'
    dec_h  = times[in_period].hour + times[in_period].minute / 60.0
    dist   = np.minimum(np.abs(dec_h - target_hour), 24.0 - np.abs(dec_h - target_hour))
    close  = dist < SAT_HOUR_TOL
    if not close.any():
        return np.array([], dtype=int), f'No overpass within {SAT_HOUR_TOL:.0f} h'
    sel_times = times[in_period][close]
    t_min, t_max = sel_times.min(), sel_times.max()
    if t_min == t_max:
        label = f'{t_min.strftime("%H:%M")} UTC'
    else:
        label = f'{t_min.strftime("%H:%M")}-{t_max.strftime("%H:%M")} UTC'
    return in_period[close], label


def mean_dynamic_rate(times_subset):
    """Mean DYNAMIC_LAPSE rate (K/m) over a set of UTC timestamps, classified by
    (season, astronomical day/night) per timestamp then averaged.

    Because the elevation-correction FIELD (ediff2d) doesn't vary with time, using
    this mean rate to build one correction field for a time-averaged map is exactly
    equivalent to applying each timestep's own rate before averaging jointly over
    (time, pixel) -- the two operations are separable for a rate(t)*ediff(px) term.
    Only matters when a bucket (e.g. an 'all hours' row) spans both day and night;
    for the fixed HW4 hours (02/08/14/20 UTC) every contributing date already falls
    on the same side of sunrise/sunset, so this just reduces to that single rate.
    """
    if len(times_subset) == 0:
        return 0.0
    if LAPSE_MODE == 'none':
        return 0.0
    if LAPSE_MODE == 'constant':
        return LAPSE_RATE_CONSTANT
    periods = classify_day_night(times_subset)
    seasons = [SEASON_MAP[m] for m in times_subset.month]
    rates = [DYNAMIC_LAPSE.get((s, p)) for s, p in zip(seasons, periods)]
    rates = [r for r in rates if r is not None]
    return float(np.mean(rates)) if rates else 0.0


def _non_wrf_rate():
    """Constant-rate resolution for UrbClim/Netatmo/Satellite: 'valley' collapses to
    'constant' for these three (the dynamic rate is WRF-only), 'none'/'constant' behave
    exactly as resolve_rate() already defines them."""
    return resolve_rate('constant' if LAPSE_MODE == 'valley' else LAPSE_MODE, None)


def compute_uhi_map(mean_t, urban_m, rural_m, corr_field=None):
    """UHI(x,y) = (T(x,y) + corr_field(x,y)) − RAW rural mean; NaN outside urban.

    Reference stays the raw rural mean -- corr_field is applied to the WHOLE field
    (urban included), not to the rural pixels feeding the reference, so it can't
    cancel to zero once re-averaged over that same rural mask.
    """
    r_t = mean_t[rural_m]
    r_t = r_t[~np.isnan(r_t)]
    if len(r_t) == 0:
        return None
    r_mean_raw = float(np.nanmean(r_t))
    corrected = mean_t + corr_field if corr_field is not None else mean_t
    uhi = corrected - r_mean_raw
    uhi[~urban_m] = np.nan
    return uhi


def blank_panel(ax, msg='No data', facecolor='#f0f0f0'):
    ax.set_facecolor(facecolor)
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='#888888',
            multialignment='center')
    ax.set_xticks([]); ax.set_yticks([])


def draw_urban_boundary(ax, urban_gdf):
    urban_gdf.boundary.plot(ax=ax, color='black', linewidth=0.9, zorder=6)


def set_extent(ax, xlim, ylim):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])


def annotate_stats(ax, arr, extra=''):
    if arr is None or np.all(np.isnan(arr)):
        return
    mn, mx, mu = np.nanmin(arr), np.nanmax(arr), np.nanmean(arr)
    txt = f'μ={mu:.2f}  [{mn:.2f}, {mx:.2f}] °C'
    if extra:
        txt += f'\n{extra}'
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=10, color='#111111', va='top',
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))


def add_mini_distrib(ax, vals, norm):
    """
    Inset KDE in the lower-right corner of ax.
    The curve is filled segment-by-segment, each sliver coloured by its
    UHI value through the panel's norm/cmap (same technique as the
    inter-annual heatmap script).  A median tick and a zero reference
    line are added.  y-axis spans the full colorbar range.
    """
    from scipy.stats import gaussian_kde

    vals = np.asarray(vals).flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) < 4:
        return

    cmap_obj = mcm.get_cmap(CMAP)

    # Grid spanning the full data range (+ tail padding), not clipped to colorbar
    data_min, data_max = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad    = max((data_max - data_min) * 0.12, 0.3)
    y_lo   = min(norm.vmin, data_min - pad)
    y_hi   = max(norm.vmax, data_max + pad)
    y_grid = np.linspace(y_lo, y_hi, 400)
    try:
        density = gaussian_kde(vals, bw_method=0.4)(y_grid)
    except Exception:
        return
    density = density / density.max()   # normalise to [0, 1]

    # Inset: lower-right corner, 38 % wide × 33 % tall in axes fraction
    ax_in = ax.inset_axes([0.59, 0.03, 0.38, 0.33])

    # Fill segment by segment — each sliver gets the colour of its UHI value
    for j in range(len(y_grid) - 1):
        y_mid = 0.5 * (y_grid[j] + y_grid[j + 1])
        c = cmap_obj(norm(y_mid))
        ax_in.fill_betweenx(
            [y_grid[j], y_grid[j + 1]],
            0,
            [density[j], density[j + 1]],
            color=c, alpha=0.88, linewidth=0,
        )

    # Outline curve
    ax_in.plot(density, y_grid, color='#333333', lw=0.7, zorder=3)

    # Median tick
    med = float(np.median(vals))
    ax_in.axhline(med, color='#111111', lw=1.0, ls='-', alpha=0.85, zorder=4)

    # Zero reference
    ax_in.axhline(0, color='#555555', lw=0.6, ls='--', alpha=0.5, zorder=3)

    ax_in.set_ylim(y_lo, y_hi)
    ax_in.set_xlim(0, 1.3)
    # Tick marks at colorbar limits + zero so the reference range is visible
    tick_vals   = sorted({norm.vmin, 0, norm.vmax})
    tick_labels = [f'{v:.0f}' for v in tick_vals]
    ax_in.set_yticks(tick_vals)
    ax_in.set_yticklabels(tick_labels, fontsize=9)
    # Dashed lines at colorbar limits to show where clipping begins
    for lim in (norm.vmin, norm.vmax):
        ax_in.axhline(lim, color='#aaaaaa', lw=0.5, ls=':', zorder=2)
    ax_in.yaxis.tick_right()
    ax_in.yaxis.set_label_position('right')
    ax_in.set_xticks([])

    ax_in.set_facecolor('white')
    ax_in.patch.set_alpha(0.78)
    for sp in ('top', 'bottom', 'left'):
        ax_in.spines[sp].set_visible(False)
    ax_in.spines['right'].set_linewidth(0.4)
    ax_in.spines['right'].set_edgecolor('#aaaaaa')


# ── SHAPEFILES ────────────────────────────────────────────────────────────────
print('Loading shapefiles…')
urban_utm = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
rural_utm = gpd.read_file(RURAL_SHP).to_crs(UTM_CRS)
urban_wgs = urban_utm.to_crs(WGS84)
rural_wgs = rural_utm.to_crs(WGS84)

ub     = urban_wgs.total_bounds
MARGIN = 0.008
XLIM   = (ub[0] - MARGIN, ub[2] + MARGIN)
YLIM   = (ub[1] - MARGIN, ub[3] + MARGIN)
print(f'  Urban extent: lon {XLIM}  lat {YLIM}')

# ── URBCLIM ───────────────────────────────────────────────────────────────────
print('\n[UrbClim] Loading…')
uc_maps  = {}
uc_lon2d = uc_lat2d = None

if os.path.exists(UCB_FILE):
    ds  = xr.open_dataset(UCB_FILE, chunks={'t': 200})
    xd  = 'x' if 'x' in ds.dims else 'lon'
    yd  = 'y' if 'y' in ds.dims else 'lat'
    xx, yy = np.meshgrid(ds[xd].values, ds[yd].values)
    t_uc   = to_utc(ds.t.values, UTC_OFFSET['urbclim'])

    urban_m = build_mask(unary(urban_utm), xx.flatten(), yy.flatten(), xx.shape)
    rural_m = build_mask(unary(rural_utm), xx.flatten(), yy.flatten(), xx.shape)
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
    ediff2d = e - float(np.nanmean(e[rural_m]))
    _shared_elev = (xx.flatten().copy(), yy.flatten().copy(), e.flatten().copy())

    if HAS_PYPROJ:
        uc_lon2d, uc_lat2d = utm_to_latlon(xx, yy)

    for key in ROW_KEYS:
        idx = period_idx(t_uc, hour=None if key == 'all' else key)
        if len(idx) == 0: continue
        arr    = ds['T2M'].isel(t=idx).load()
        if arr.dims[0] != 't': arr = arr.transpose('t', yd, xd)
        mean_t = arr.values.mean(axis=0);  n = len(idx)
        corr_field = _non_wrf_rate() * ediff2d if ediff2d is not None else None
        uhi = compute_uhi_map(mean_t, urban_m, rural_m, corr_field)
        if uhi is not None:
            uc_maps[key] = uhi
            print(f'  {key}: {n} steps  [{np.nanmin(uhi):.2f}, {np.nanmax(uhi):.2f}] °C')
    ds.close()
else:
    print('  [MISSING]', UCB_FILE)

# ── WRF ───────────────────────────────────────────────────────────────────────
# Rural reference now matches extractions/extraction_WRF_20260310.py and the
# diurnal script exactly: bounding-box subset of the rural shape (NOT a
# polygon mask — see check_spatial_vs_diurnal_consistency_20260727.py), with
# the lapse-correction reference elevation = MIN elevation of that subset.
print('\n[WRF] Loading…')
wrf_maps  = {}
wrf_lon2d = wrf_lat2d = None

if TRY_WRF and os.path.exists(WRF_FILE):
    try:
        import salem
        ds    = salem.open_wrf_dataset(WRF_FILE)[['T2', 'HGT']]
        t_wrf = to_utc(ds.time.values, UTC_OFFSET['wrf'])

        for lname in ('lat','latitude','XLAT'):
            if lname in ds.coords or lname in ds:
                wrf_lat2d = ds[lname].values
                if wrf_lat2d.ndim == 3: wrf_lat2d = wrf_lat2d[0]
                break
        for lname in ('lon','longitude','XLONG'):
            if lname in ds.coords or lname in ds:
                wrf_lon2d = ds[lname].values
                if wrf_lon2d.ndim == 3: wrf_lon2d = wrf_lon2d[0]
                break
        if wrf_lat2d is None:
            raise RuntimeError('Cannot find lat/lon in WRF dataset')

        # K -> degC once, dataset-wide, so mean_t and the rural bbox subset
        # below are guaranteed to share the same units.
        if float(ds['T2'].isel(time=0).mean(skipna=True).values) > 200:
            ds['T2'] = ds['T2'] - 273.15
            print('  Converted T2 K -> degC')

        # Urban-pixel selection for the MAP DISPLAY only (unaffected by the
        # rural-reference bug — both masking approaches agree on these pixels).
        urban_m = build_mask(unary(urban_wgs),
                             wrf_lon2d.flatten(), wrf_lat2d.flatten(), wrf_lat2d.shape)
        print(f'  Grid {wrf_lat2d.shape}  urban(polygon mask)={urban_m.sum()} px')

        # Canonical rural reference (bbox subset, min-elevation lapse ref). The scalar
        # rate multiplying the elevation diff is now looked up per hour-bucket below
        # (mean_dynamic_rate), not fixed.
        ds_rural        = ds.salem.subset(shape=rural_wgs)
        ref_elev        = float(ds_rural['HGT'].isel(time=0).min().values)
        elev_diff_rural = ds_rural['HGT'].isel(time=0) - ref_elev
        n_rural_px      = int(np.sum(~np.isnan(ds_rural['HGT'].isel(time=0).values)))
        print(f'  Rural reference: bbox subset = {n_rural_px} px, '
              f'lapse ref (min elev) = {ref_elev:.1f} m')

        for key in ROW_KEYS:
            idx = period_idx(t_wrf, hour=None if key == 'all' else key)
            if len(idx) == 0: continue
            mean_t     = ds['T2'].isel(time=idx).values.mean(axis=0)
            rate       = mean_dynamic_rate(t_wrf[idx])
            rural_mean = float((ds_rural['T2'].isel(time=idx) + rate * elev_diff_rural)
                               .mean(skipna=True).values)
            uhi = np.where(urban_m, mean_t - rural_mean, np.nan)
            n   = len(idx)
            wrf_maps[key] = uhi
            print(f'  {key}: {n} steps  [{np.nanmin(uhi):.2f}, {np.nanmax(uhi):.2f}] °C')
        ds.close()
    except ImportError:
        print('  [SKIP] salem not installed')
    except Exception as e:
        print(f'  [ERROR] WRF: {e}')
elif not os.path.exists(WRF_FILE):
    print('  [MISSING]', WRF_FILE)

# ── NETATMO ───────────────────────────────────────────────────────────────────
print('\n[Netatmo] Loading…')
na_maps = {}

if os.path.exists(NETATMO_FILE):
    ds    = xr.open_dataset(NETATMO_FILE)
    T_VAR = next((v for v in ('T_lvl4','T0','temperature') if v in ds), None)
    if T_VAR:
        na_lats = ds.latitude.values
        na_lons = ds.longitude.values
        na_T    = ds[T_VAR].values
        if na_T.shape[0] == len(na_lats):
            na_T = na_T.T
        t_na = to_utc(
            [pd.Timestamp(t.astype('datetime64[ns]').item()) for t in ds.time.values],
            UTC_OFFSET['netatmo'])

        u_idx, r_idx = [], []
        u_union = unary(urban_wgs);  r_union = unary(rural_wgs)
        for i, (lat, lon) in enumerate(zip(na_lats, na_lons)):
            if np.isnan(lat) or np.isnan(lon): continue
            pt = Point(lon, lat)
            if u_union.contains(pt):   u_idx.append(i)
            elif r_union.contains(pt): r_idx.append(i)
        u_idx = np.array(u_idx);  r_idx = np.array(r_idx)
        print(f'  Urban={len(u_idx)} Rural={len(r_idx)} stations')

        na_elev = None
        elev_var = next((v for v in ('altitude','elevation','z','ALT') if v in ds), None)
        if elev_var:
            na_elev = ds[elev_var].values.astype(float)
            print(f'  Station elevations from dataset: '
                  f'{np.nanmin(na_elev):.0f}–{np.nanmax(na_elev):.0f} m')
        elif _shared_elev is not None and HAS_PYPROJ:
            try:
                from scipy.interpolate import LinearNDInterpolator
                x_uc, y_uc, e_uc = _shared_elev
                ok = ~np.isnan(e_uc)
                _interp = LinearNDInterpolator(list(zip(x_uc[ok], y_uc[ok])), e_uc[ok])
                tr_na = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)
                na_x_utm, na_y_utm = tr_na.transform(na_lons, na_lats)
                na_elev = _interp(na_x_utm, na_y_utm)
                print(f'  Station elevations interpolated from UrbClim DEM: '
                      f'{np.nanmin(na_elev):.0f}–{np.nanmax(na_elev):.0f} m')
            except Exception as _ex:
                print(f'  [WARN] Netatmo lapse correction skipped: {_ex}')
        else:
            print('  [WARN] No elevation source — Netatmo lapse correction skipped')

        for key in ROW_KEYS:
            idx = period_idx(t_na, hour=None if key == 'all' else key)
            if len(idx) == 0 or len(u_idx) == 0 or len(r_idx) == 0: continue
            mean_t = np.nanmean(na_T[idx, :], axis=0)
            # Reference stays the RAW rural mean; correction applied to URBAN stations
            # instead, relative to the rural reference elevation, so it can't cancel to
            # zero once re-averaged over the rural set it used to be applied to.
            r_vals_raw = mean_t[r_idx]
            r_vals_raw = r_vals_raw[~np.isnan(r_vals_raw)]
            if len(r_vals_raw) == 0: continue
            r_mean_raw = float(np.nanmean(r_vals_raw))

            u_vals = mean_t[u_idx].copy()
            if na_elev is not None:
                r_elev_mean = float(np.nanmean(na_elev[r_idx][~np.isnan(na_elev[r_idx])]))
                rate = _non_wrf_rate()
                u_vals = u_vals + rate * (na_elev[u_idx] - r_elev_mean)

            uhi = u_vals - r_mean_raw
            na_maps[key] = {'lon': na_lons[u_idx], 'lat': na_lats[u_idx],
                            'uhi': uhi,
                            'r_lon': na_lons[r_idx], 'r_lat': na_lats[r_idx]}
            print(f'  {key}: {len(idx)} steps  [{np.nanmin(uhi):.2f}, {np.nanmax(uhi):.2f}] °C')
    else:
        print('  [WARN] Temperature variable not found')
    ds.close()
else:
    print('  [MISSING]', NETATMO_FILE)

# ── SATELLITE ─────────────────────────────────────────────────────────────────
print('\n[Satellite] Loading…')
sat_maps   = {}
sat_labels = {}
sat_lon2d  = sat_lat2d = None

if os.path.exists(LST_FILE):
    ds      = xr.open_dataset(LST_FILE)
    LST_VAR = '__xarray_dataarray_variable__'
    lst     = ds[LST_VAR]
    t_sat   = to_utc(
        [pd.Timestamp(t.astype('datetime64[ns]').item()) for t in ds.time.values],
        UTC_OFFSET['sat'])

    x_a = ds.x.values if 'x' in ds else ds.lon.values
    y_a = ds.y.values if 'y' in ds else ds.lat.values
    xx, yy  = np.meshgrid(x_a, y_a)
    urban_m = build_mask(unary(urban_utm), xx.flatten(), yy.flatten(), xx.shape)
    rural_m = build_mask(unary(rural_utm), xx.flatten(), yy.flatten(), xx.shape)
    print(f'  Grid {xx.shape}  urban={urban_m.sum()} rural={rural_m.sum()} px')

    if HAS_PYPROJ:
        sat_lon2d, sat_lat2d = utm_to_latlon(xx, yy)

    ev_sat = next((v for v in ('HGT','elevation','dem','DEM') if v in ds), None)
    if ev_sat:
        e_sat = ds[ev_sat].values;  e_sat = e_sat[0] if e_sat.ndim == 3 else e_sat
        print('  Elevation: native variable in LST file')
    else:
        # lst-bolzano.nc carries no native elevation either (checked: only the LST
        # DataArray + spatial_ref) -- reproject the shared 10m DEM directly onto this
        # grid's own resolution/extent, rather than interpolating from UrbClim's grid
        # (which was itself never populated, since UrbClim has no native elevation --
        # see the UrbClim block above).
        e_sat = reproject_dem_to_grid(x_a, y_a)
        print('  Elevation: reprojected from DEM_10m_south_tirol.tif (no native var)')
    sat_ediff2d = e_sat - float(np.nanmean(e_sat[rural_m]))

    for key in ROW_KEYS:
        if key == 'all':
            idx   = period_idx(t_sat, hour=None)
            label = 'all hours'
        else:
            idx, label = sat_closest_idx(t_sat, key)

        if len(idx) == 0:
            print(f'  {key}: 0 scenes'); continue

        time_label = label
        label = f'n={len(idx)} scenes  ({time_label})'

        scenes = np.stack([lst.isel(time=int(i)).values for i in idx], axis=0)
        mean_t = np.nanmean(scenes, axis=0)
        corr_field = _non_wrf_rate() * sat_ediff2d if sat_ediff2d is not None else None
        uhi    = compute_uhi_map(mean_t, urban_m, rural_m, corr_field)
        if uhi is not None:
            sat_maps[key]   = uhi
            sat_labels[key] = label
            print(f'  {key}: {len(idx)} scenes  [{np.nanmin(uhi):.2f}, {np.nanmax(uhi):.2f}] °C  ({time_label})')
    ds.close()
else:
    print('  [MISSING]', LST_FILE)

# ── NORMS ─────────────────────────────────────────────────────────────────────
print(f'\n  Air colorbar : [{VMIN_AIR:.0f}, {VMAX_AIR:.0f}] °C')
print(f'  Sat colorbar : [{VMIN_SAT:.0f}, {VMAX_SAT:.0f}] °C')

norm_air = mcolors.TwoSlopeNorm(vmin=VMIN_AIR, vcenter=0, vmax=VMAX_AIR)
norm_sat = mcolors.TwoSlopeNorm(vmin=VMIN_SAT, vcenter=0, vmax=VMAX_SAT)
#%%PLOTTING
# ── FIGURE ────────────────────────────────────────────────────────────────────
print('\nPlotting…')

NCOLS = 4
NROWS = len(ROW_KEYS)

ROW_LABELS = [f'{h:02d}:00 UTC' for h in HOURS_OF_DAY]
if SHOW_ALL_ROW:
    ROW_LABELS += ['All hours\n(period avg.)']

COL_TITLES = ['UrbClim', 'WRF', 'NetAtmo', 'Satellite']

height_ratios = [1] * len(HOURS_OF_DAY) + ([1.12] if SHOW_ALL_ROW else [])

fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(4.2 * NCOLS, 3.6 * NROWS),
                         gridspec_kw={'height_ratios': height_ratios})
if NROWS == 1:
    axes = axes[np.newaxis, :]

ax_legend_panel = None   # will point to the blank satellite panel right after the row
                         # that actually has a satellite plot (so the SUHI colorbar sits
                         # just below it), if such a row/panel pair exists

sat_row_idx = next((ri for ri, key in enumerate(ROW_KEYS) if key in sat_maps), None)
legend_row_idx = None
if sat_row_idx is not None and sat_row_idx + 1 < len(ROW_KEYS) \
        and ROW_KEYS[sat_row_idx + 1] not in sat_maps:
    legend_row_idx = sat_row_idx + 1

for ri, key in enumerate(ROW_KEYS):
    is_avg = (key == 'all')

    for ci in range(NCOLS):
        ax = axes[ri, ci]
        if is_avg:
            ax.set_facecolor('#f8f8f4')

        norm = norm_sat if ci == 3 else norm_air

        if ci == 2:
            # ── Netatmo scatter ────────────────────────────────────────────
            if key in na_maps:
                nd = na_maps[key]
                ax.scatter(nd['r_lon'], nd['r_lat'],
                           c='#aaaaaa', marker='^', s=20, zorder=4,
                           edgecolors='k', linewidths=0.3)
                ax.scatter(nd['lon'], nd['lat'], c=nd['uhi'],
                           cmap=CMAP, norm=norm_air, s=55, marker='o',
                           edgecolors='k', linewidths=0.4, zorder=5)
                draw_urban_boundary(ax, urban_wgs)
                set_extent(ax, XLIM, YLIM)
                annotate_stats(ax, nd['uhi'],
                               f'n={np.sum(~np.isnan(nd["uhi"]))} sta')
                add_mini_distrib(ax, nd['uhi'], norm_air)
            else:
                blank_panel(ax)

        else:
            # ── Gridded datasets ───────────────────────────────────────────
            maps_dict = [uc_maps, wrf_maps, None, sat_maps][ci]
            lon2d     = [uc_lon2d, wrf_lon2d, None, sat_lon2d][ci]
            lat2d     = [uc_lat2d, wrf_lat2d, None, sat_lat2d][ci]

            if key in maps_dict and lon2d is not None:
                ax.pcolormesh(lon2d, lat2d, maps_dict[key],
                              cmap=CMAP, norm=norm, shading='auto', zorder=1)
                draw_urban_boundary(ax, urban_wgs)
                set_extent(ax, XLIM, YLIM)
                extra = sat_labels.get(key, '') if ci == 3 else ''
                annotate_stats(ax, maps_dict[key], extra)
                add_mini_distrib(ax, maps_dict[key], norm)
            else:
                if ci == 3 and ri == legend_row_idx:
                    blank_panel(ax, msg='', facecolor='white')
                    ax_legend_panel = ax
                else:
                    msg = ('No overpass\nnear this hour'
                           if ci == 3 and not is_avg else 'No data')
                    blank_panel(ax, msg, facecolor='white' if ci == 3 else '#f0f0f0')

        if ri == 0:
            ax.set_title(COL_TITLES[ci], fontsize=14, fontweight='bold', pad=4)
        if ci == 0:
            ax.set_ylabel(ROW_LABELS[ri], fontsize=15, fontweight='bold', labelpad=4)

# Separator above the all-hours row
if SHOW_ALL_ROW:
    from matplotlib.lines import Line2D
    for ci in range(NCOLS):
        ax = axes[len(HOURS_OF_DAY), ci]
        ax.add_artist(Line2D([0, 1], [1.04, 1.04],
                             transform=ax.transAxes, color='#999999',
                             lw=1.0, clip_on=False))

# ── COLORBARS ─────────────────────────────────────────────────────────────────
# Air UHI: big vertical colorbar on the far left (shared by UrbClim/WRF/Netatmo)
# SUHI   : small horizontal colorbar below the Satellite column
fig.subplots_adjust(left=0.105, right=0.975, top=0.965, bottom=0.04,
                    hspace=0.12, wspace=0.08)

pos_tl  = axes[0, 0].get_position()
pos_bl  = axes[-1, 0].get_position()

cbar_ax_air = fig.add_axes([0.028, pos_bl.y0, 0.015, pos_tl.y1 - pos_bl.y0])
sm_air = mcm.ScalarMappable(cmap=CMAP, norm=norm_air)
sm_air.set_array([])
cb_air = fig.colorbar(sm_air, cax=cbar_ax_air, extend='both')
cb_air.ax.yaxis.set_ticks_position('left')
cb_air.ax.yaxis.set_label_position('left')
cb_air.set_label('UHI (°C)', fontsize=12, labelpad=4)
cb_air.ax.tick_params(labelsize=11)

# SUHI colorbar: inset inside the blank satellite panel just below the row that
# actually has a satellite plot (legend_row_idx), with the same physical thickness
# as the air bar.  Air bar is 0.015 figure-fraction wide; panel height ≈ 0.23
# figure-fraction → equivalent panel-fraction height = 0.015/0.23 ≈ 0.065.
# Falls back to a fig-level bar below the last row if no such panel is found.
sm_sat = mcm.ScalarMappable(cmap=CMAP, norm=norm_sat)
sm_sat.set_array([])
if ax_legend_panel is not None:
    cbar_ax_sat = ax_legend_panel.inset_axes([0.08, 0.80, 0.84, 0.065])
else:
    pos_sat = axes[-1, 3].get_position()
    cbar_w  = pos_sat.width * 0.65
    cbar_x0 = pos_sat.x0 + (pos_sat.width - cbar_w) / 2
    cbar_ax_sat = fig.add_axes([cbar_x0, pos_bl.y0 - 0.06, cbar_w, 0.016])
cb_sat = fig.colorbar(sm_sat, cax=cbar_ax_sat, orientation='horizontal', extend='both')
cb_sat.set_label('SUHI (°C)', fontsize=11, labelpad=3)
cb_sat.ax.tick_params(labelsize=10)

# # ── CAPTION ────────────────────────────────────────────────────────────────────
# _legend_lines = [
#     # 'In-panel annotation key:',
#     '  μ: spatial mean UHI/SUHI over the urban area',
#     '  [min, max]: full range across urban area',
#     '  n: number of valid stations (Netatmo) or scenes (Satellite)',
#     '  ▲: rural reference stations',
#     # '  KDE inset   distribution of UHI values within the urban area;'
#     # '  y-axis spans the colorbar range, fill colour matches the UHI value;'
#     # '  horizontal tick = median',
# ]
# if ax_legend_panel is not None:
#     ax_legend_panel.text(
#         0.5, 0.72, '\n'.join(_legend_lines),
#         ha='center', va='center',
#         transform=ax_legend_panel.transAxes,
#         fontsize=10, color='#333333', linespacing=1.6,
#         bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='#aaaaaa', lw=0.8),
#     )
# else:
#     fig.text(
#         0.08, 0.005,
#         '\n'.join(_legend_lines),
#         fontsize=10.5, color='#333333', va='bottom', linespacing=1.55,
#         bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#aaaaaa', lw=0.8),
#     )

# ── SAVE ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
for ext in ('pdf', 'png'):
    p = os.path.join(OUTPUT_DIR,
                     f'spatial_uhi_maps_heatwave2023_dualcbar_distrib_fixed_{LAPSE_MODE}_{ts}.{ext}')
    plt.savefig(p, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved → {p}')

plt.show()
print('✓ Done.')
