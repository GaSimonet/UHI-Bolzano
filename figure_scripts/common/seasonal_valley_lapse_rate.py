#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal x day/night typical VALLEY lapse rate — standalone fit, WRF daily harmonized archive

Extracted out of seasonal_UHI_WRF_harmonized_dynamiclapse_20260731.py's Pass 1 so the
fitted rates live in ONE stable, non-timestamped CSV that any other figure/analysis
script can `pd.read_csv()` deterministically, instead of being recomputed inline and
buried in a timestamped diagnostic file inside that script's private OUT_DIR.

METHOD (Simonet et al. 2025 convention): one empirical lapse rate per SEASON x DAY/NIGHT
(8 values), computed as a spatio-temporal average of LOCAL, per-column lapse rates over a
"representative valley" polygon: the along-river valley transect
(valley_x_sec_along_river_bolzano_v2.shp) buffered VALLEY_BUFFER_M either side, restricted
to pixels from the valley floor up to VALLEY_ELEV_MAX_M m a.s.l.

For every valley pixel i (an individual WRF atmospheric column) and every hourly
timestep, the local lapse rate is the direct two-point slope between that column's own
near-surface temperature and the temperature at a fixed reference altitude of 3000 m
a.s.l. taken from THAT SAME column's vertical profile (via the full 3-D WRF temperature
field TK, interpolated in height using Z):

    Gamma_i(t) = (T2_i(t) - T3000_i(t)) / (3000 - HGT_i)

e.g. a valley-floor pixel at HGT=250 m uses (T3000 - T2(250m))/(3000-250); a
higher pixel near the valley wall at HGT=1500 m uses (T3000 - T2(1500m))/(3000-1500) --
each column supplies its own two-point vertical slope, rather than pooling many
columns' (elevation, T2) pairs into one regression. Those per-column, per-timestep
rates are then averaged spatially over all valley pixels and temporally over all
timesteps in a (season, day/night) bucket to give the final Gamma_{s,p}.

DAY/NIGHT WINDOW: NOT a fixed UTC hour range. Each hourly timestep is classified using
that CALENDAR DAY's real astronomical sunrise/sunset for Bolzano (see
solar_daynight.classify_day_night) -- so the "day" window is ~8.5h wide around
the Dec solstice and ~15.8h wide around the Jun solstice, tracking actual daylight rather
than a season-blind clock convention. seasonal_UHI_WRF_harmonized_dynamiclapse_20260731.py
imports the SAME classifier for its Pass-2 map compositing, so the (season, period) bucket
a rate is fit on is exactly the bucket it gets applied to.

OUTPUT
------
- SEASONAL_LAPSE_CSV (stable path, overwritten every run): season, period, lapse_rate_K_per_m,
  lapse_rate_std_K_per_m, n_pixel_timesteps, plus the fit's date range / valley params for
  provenance. This is the file other scripts should load.
- A timestamped archival copy + a diagnostic bar plot, both under ARCHIVE_DIR, kept only
  for run-to-run history/QC -- not meant to be read back programmatically.

Run this whenever the underlying WRF archive changes; downstream scripts should treat
SEASONAL_LAPSE_CSV as a checked-in input and just read it (see load_seasonal_valley_lapse()
below, importable from other scripts in this directory).
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.vectorized import contains as shp_contains
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solar_daynight import classify_day_night, BOLZANO_LAT, BOLZANO_LON

# ── CONFIG ─────────────────────────────────────────────────────────────────────
WRF_DIR     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_consolidated_intermediate'
WRF_PATTERN = 'wrf_harmonized_{date}.nc'

VALLEY_LINE_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/valley_x_sec_along_river_bolzano_v2.shp'

# The stable CSV (the "common file" every other script reads) now lives alongside
# this script, in figure_scripts/common/. The heavy .npz fit-cache and timestamped
# archival copies are NOT duplicated here -- they stay at the original location so
# a re-run reuses the existing multi-year WRF fit cache instead of re-reading the
# full 2020-2024 archive.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SEASONAL_LAPSE_CSV = os.path.join(_THIS_DIR, 'seasonal_valley_lapse_rates.csv')   # stable -- read by other scripts
ARCHIVE_DIR = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/UHI/seasonal_valley_lapse_rate_archive'
CACHE_DIR   = os.path.join(ARCHIVE_DIR, 'cache')
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
FORCE_RECOMPUTE = False

DATE_START = '2020-01-01'
DATE_END   = '2024-12-31'

LAPSE_RATE_FALLBACK = 0.0065   # K/m, used ONLY if a bucket has too few valid valley pixel-timesteps

VALLEY_BUFFER_M   = 5000.0   # half-width buffer (m, either side) around the valley transect line
VALLEY_ELEV_MAX_M = 3000.0   # m a.s.l. -- valley-floor-to-3000m cutoff: pixels above this are
                              # excluded from the valley mask entirely (different microclimate)
REF_ALT_M = VALLEY_ELEV_MAX_M   # m a.s.l. -- the fixed upper reference altitude each valley
                                 # column's local lapse rate is computed against (T2 vs T at
                                 # REF_ALT_M from that same column's TK/Z profile). Set equal to
                                 # VALLEY_ELEV_MAX_M by design: "ground to 3000m" is both the
                                 # inclusion cutoff and the reference altitude.

UTM_CRS = 'EPSG:32632'
WGS84   = 'EPSG:4326'

SEASON_MAP = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring',  4: 'Spring',  5: 'Spring',
               6: 'Summer',  7: 'Summer',  8: 'Summer',
               9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
})

# ── PUBLIC HELPER for other scripts ─────────────────────────────────────────────

def load_seasonal_valley_lapse(csv_path=SEASONAL_LAPSE_CSV):
    """Return {(season, period): lapse_rate_K_per_m} loaded from the stable CSV this
    script writes, period in {'day', 'night'}.

    Import this from other figure scripts instead of duplicating the fit, e.g.:
        from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
        dynamic_lapse = load_seasonal_valley_lapse()
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'{csv_path} not found -- run seasonal_valley_lapse_rate.py first '
            'to generate the seasonal valley lapse-rate table.')
    df = pd.read_csv(csv_path)
    if 'period' not in df.columns:
        raise ValueError(
            f'{csv_path} is a stale season-only table (no "period" column) -- '
            're-run seasonal_valley_lapse_rate.py to regenerate the '
            'season x day/night (8-value) table.')
    return {(s, p): v for s, p, v in zip(df['season'], df['period'], df['lapse_rate_K_per_m'])}

# ── HELPERS ────────────────────────────────────────────────────────────────────

def build_mask(union, flat_x, flat_y, shape):
    b = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) & (flat_y >= b[1]) & (flat_y <= b[3]))
    m = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)

def _dir_sig(dir_path, pattern, day_list):
    parts = []
    for day in day_list:
        fpath = os.path.join(dir_path, pattern.format(date=day.strftime('%Y-%m-%d')))
        if os.path.exists(fpath):
            st = os.stat(fpath)
            parts.append(f'{day:%Y%m%d}:{st.st_mtime:.0f}:{st.st_size}')
        else:
            parts.append(f'{day:%Y%m%d}:missing')
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]

def cache_path(name, *key_parts):
    key = hashlib.sha1('|'.join(str(p) for p in key_parts).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f'{name}_{key}.npz')

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading valley shapefile…')
    _ll_to_utm = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)
    valley_line_utm = gpd.read_file(VALLEY_LINE_SHP).to_crs(UTM_CRS)
    valley_utm = gpd.GeoSeries(valley_line_utm.buffer(VALLEY_BUFFER_M), crs=UTM_CRS)
    valley_wgs_union = valley_utm.to_crs(WGS84).unary_union

    grid_cache = {}   # native shape -> valley_fit_m (bool mask), hgt

    def get_grid_info(ds):
        lon2d = ds['lon'].values
        lat2d = ds['lat'].values
        shape = lon2d.shape
        if shape in grid_cache:
            return grid_cache[shape]
        valley_m = build_mask(valley_wgs_union, lon2d.flatten(), lat2d.flatten(), shape)
        hgt = ds['HGT'].isel(time=0).values
        valley_fit_m = valley_m & (hgt <= VALLEY_ELEV_MAX_M)
        info = dict(valley_fit_m=valley_fit_m, hgt=hgt)
        grid_cache[shape] = info
        print(f'  New native grid {shape}: valley_fit={valley_fit_m.sum()}px '
              f'(<= {VALLEY_ELEV_MAX_M:.0f}m asl, +/-{VALLEY_BUFFER_M/1000:.0f}km of transect)')
        return info

    day_list = pd.date_range(DATE_START, DATE_END, freq='D')
    print(f'\n{len(day_list)} calendar days requested ({DATE_START} .. {DATE_END})')

    print('\nComputing local per-column valley lapse rate per season x day/night '
          '(astronomical sunrise/sunset window, per calendar day)…')
    buckets = [(s, p) for s in SEASON_ORDER for p in ('day', 'night')]
    # Cache key includes a method tag so a stale OLS-fit cache from the previous
    # methodology is never silently reused under the same name.
    lapse_cache = cache_path('lapse_localcolumn_valley_daynight', DATE_START, DATE_END,
                              VALLEY_BUFFER_M, VALLEY_ELEV_MAX_M, REF_ALT_M, BOLZANO_LAT, BOLZANO_LON,
                              _dir_sig(WRF_DIR, WRF_PATTERN, day_list))

    if not FORCE_RECOMPUTE and os.path.exists(lapse_cache):
        _d = np.load(lapse_cache)
        rows = []
        for season, period in buckets:
            key = f'{season}_{period}'
            rows.append({'season': season, 'period': period,
                         'lapse_rate_K_per_m': float(_d[f'{key}__lapse']),
                         'lapse_rate_std_K_per_m': float(_d[f'{key}__std']),
                         'n_pixel_timesteps': int(_d[f'{key}__n'])})
        print(f'  [CACHE] loaded {os.path.basename(lapse_cache)}')
    else:
        acc = {b: np.zeros(3, dtype=np.float64) for b in buckets}   # n, sum(gamma), sumsq(gamma)

        for di, day in enumerate(day_list):
            fpath = os.path.join(WRF_DIR, WRF_PATTERN.format(date=day.strftime('%Y-%m-%d')))
            if not os.path.exists(fpath):
                continue
            try:
                ds = xr.open_dataset(fpath)[['T2', 'HGT', 'TK', 'Z', 'lat', 'lon']]
                info = get_grid_info(ds)
                valley_fit_m, hgt = info['valley_fit_m'], info['hgt']
                py, px = np.where(valley_fit_m)              # valley pixel row/col indices
                hgt_valley = hgt[py, px].astype(np.float64)    # (n_px,)

                t2_full = ds['T2'].load().values                       # (time, sn, we)
                tk_full = ds['TK'].isel(south_north=xr.DataArray(py, dims='px'),
                                         west_east=xr.DataArray(px, dims='px')).load().values   # (time, lev, n_px)
                z_full  = ds['Z'].isel(south_north=xr.DataArray(py, dims='px'),
                                        west_east=xr.DataArray(px, dims='px')).load().values     # (time, lev, n_px)
                times = pd.DatetimeIndex(ds['time'].values)
                ds.close()

                t2_valley = t2_full[:, py, px].astype(np.float64)   # (time, n_px)

                # Vertically interpolate TK to the fixed REF_ALT_M altitude, per column,
                # per timestep -- fully vectorized (no python loop over pixels/levels).
                # Z increases monotonically with model level, so the first level at or
                # above REF_ALT_M brackets the target altitude together with the level
                # just below it.
                above = z_full >= REF_ALT_M                              # (time, lev, n_px)
                idx_above = np.argmax(above, axis=1)                     # (time, n_px)
                idx_below = np.clip(idx_above - 1, 0, z_full.shape[1] - 1)
                ti_idx = np.arange(z_full.shape[0])[:, None]
                px_idx = np.arange(z_full.shape[2])[None, :]
                z_above  = z_full[ti_idx, idx_above, px_idx]
                z_below  = z_full[ti_idx, idx_below, px_idx]
                tk_above = tk_full[ti_idx, idx_above, px_idx]
                tk_below = tk_full[ti_idx, idx_below, px_idx]
                denom_z  = z_above - z_below
                frac = np.where(denom_z > 0, (REF_ALT_M - z_below) / np.maximum(denom_z, 1e-6), 0.0)
                frac = np.clip(frac, 0.0, 1.0)
                t_ref = tk_below + frac * (tk_above - tk_below)          # (time, n_px), same units as TK

                # Gamma_i(t) = (T2_i(t) - T_ref_i(t)) / (REF_ALT_M - HGT_i) -- standard
                # convention: positive = cooling with height. T2/TK share the same
                # (Kelvin) units from the raw file, and only the DIFFERENCE matters, so
                # no K->degC conversion is needed here.
                denom_h = REF_ALT_M - hgt_valley[None, :]                # (1, n_px), > 0 by construction (valley_fit_m <= REF_ALT_M)
                gamma = (t2_valley - t_ref) / denom_h                    # (time, n_px)

                periods = classify_day_night(times)   # per-timestep, real sunrise/sunset for that date
                for ti in range(gamma.shape[0]):
                    bucket = (SEASON_MAP[times[ti].month], periods[ti])
                    g = gamma[ti]
                    g = g[np.isfinite(g)]
                    n = len(g)
                    if n == 0:
                        continue
                    acc[bucket] += np.array([n, g.sum(), (g * g).sum()])
            except Exception as e:
                print(f'  [ERROR] {os.path.basename(fpath)}: {e}')

            if (di + 1) % 300 == 0:
                print(f'  …{di + 1}/{len(day_list)} days scanned')

        rows = []
        cache_payload = {}
        for (season, period), a in acc.items():
            n, sg, sgg = a
            if n < 100:
                lapse = LAPSE_RATE_FALLBACK
                std = float('nan')
                print(f'  [WARN] {season}/{period}: too few valley pixel-timesteps (n={int(n)}), '
                      f'falling back to fixed {LAPSE_RATE_FALLBACK} K/m')
            else:
                lapse = sg / n
                std = float(np.sqrt(max(sgg / n - lapse ** 2, 0.0)))
            rows.append({'season': season, 'period': period, 'lapse_rate_K_per_m': lapse,
                         'lapse_rate_std_K_per_m': std, 'n_pixel_timesteps': int(n)})
            key = f'{season}_{period}'
            cache_payload[f'{key}__lapse'] = np.array(lapse)
            cache_payload[f'{key}__std'] = np.array(std)
            cache_payload[f'{key}__n'] = np.array(n)

        np.savez_compressed(lapse_cache, **cache_payload)
        print(f'  [CACHE] saved {os.path.basename(lapse_cache)}')

    lapse_df = pd.DataFrame(rows)
    lapse_df['season'] = pd.Categorical(lapse_df['season'], categories=SEASON_ORDER, ordered=True)
    lapse_df['period'] = pd.Categorical(lapse_df['period'], categories=['day', 'night'], ordered=True)
    lapse_df = lapse_df.sort_values(['season', 'period']).reset_index(drop=True)
    lapse_df['fit_start'] = DATE_START
    lapse_df['fit_end'] = DATE_END
    lapse_df['valley_buffer_m'] = VALLEY_BUFFER_M
    lapse_df['valley_elev_max_m'] = VALLEY_ELEV_MAX_M
    lapse_df['daynight_window'] = 'astronomical (Bolzano sunrise/sunset, per calendar day)'
    lapse_df['computed_at'] = pd.Timestamp.now().isoformat()

    print('\n  Empirical typical valley lapse rate by season x day/night:')
    print(lapse_df.to_string(index=False))

    # Stable path -- this is the one other scripts should read.
    lapse_df.to_csv(SEASONAL_LAPSE_CSV, index=False)
    print(f'\nSaved (stable, for reuse) → {SEASONAL_LAPSE_CSV}')

    # Timestamped archival copy, for run-history only.
    archive_csv = os.path.join(ARCHIVE_DIR, f'seasonal_valley_lapse_rates_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv')
    lapse_df.to_csv(archive_csv, index=False)
    print(f'Saved (archive) → {archive_csv}')

    # Diagnostic bar plot (day vs night per season) vs the fixed standard-atmosphere rate.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SEASON_ORDER))
    w = 0.35
    idx     = lapse_df.set_index(['season', 'period'])['lapse_rate_K_per_m']
    idx_std = lapse_df.set_index(['season', 'period'])['lapse_rate_std_K_per_m']
    day_vals   = [idx[(s, 'day')]   for s in SEASON_ORDER]
    night_vals = [idx[(s, 'night')] for s in SEASON_ORDER]
    day_std    = [idx_std[(s, 'day')]   for s in SEASON_ORDER]
    night_std  = [idx_std[(s, 'night')] for s in SEASON_ORDER]
    ax.bar(x - w/2, day_vals,   width=w, yerr=day_std,   capsize=3,
           label='Day (sunrise–sunset)',   color='#E8A33D')
    ax.bar(x + w/2, night_vals, width=w, yerr=night_std, capsize=3,
           label='Night (sunset–sunrise)', color='#3A5FA0')
    ax.axhline(LAPSE_RATE_FALLBACK, color='black', ls='--', lw=1.2,
               label=f'Fixed standard-atmosphere rate ({LAPSE_RATE_FALLBACK} K/m)')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SEASON_ORDER)
    ax.set_ylabel('Empirical lapse rate (K/m)')
    ax.set_title(f'Valley lapse rate by season x day/night: mean of local per-column\n'
                 f'(T2 - T@{REF_ALT_M:.0f}m) / ({REF_ALT_M:.0f}m - HGT), +/-1 std across valley pixels & time\n'
                 f'(astronomical sunrise/sunset window, valley polygon <= {VALLEY_ELEV_MAX_M:.0f}m asl)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = os.path.join(ARCHIVE_DIR, f'seasonal_valley_lapse_rate_diagnostic_{pd.Timestamp.now():%Y%m%d_%H%M%S}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Saved → {plot_path}')
    plt.close(fig)

    print('\n✓ Done.')

if __name__ == '__main__':
    main()
