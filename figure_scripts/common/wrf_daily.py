#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared WRF loader for the 5-year daily-harmonized archive
(wrf_harmonized_YYYY-MM-DD.nc, WRF_consolidated_intermediate/), replacing the
single annual WRF_RAW_Alto_Adige_2023.nc file the heatwave2023/ scripts used
to load via salem.

Why: WRF_RAW_Alto_Adige_2023.nc was built by concatenating many short
forecast-cycle runs without dropping spin-up, producing a visible jump in
the WRF curve at/after each day's 00 UTC boundary and inflating the apparent
day-to-day spread. The daily harmonized files already have the first 6h of
each run dropped as spin-up (each file spans 06:00 UTC that day through
05:00 UTC the next), and carry plain 2-D lat/lon coordinates, so masking
against the urban/rural shapefiles no longer needs salem.

Elevation correction here uses the validated convention already proven out
in plot_diurnal_uhi_seasonal_wrfdaily_20260731.py and
plot_wrf_winter_timeseries_20260731.py: reference = MEAN elevation over the
rural mask, correction applied to the URBAN side only, rural mean left raw
(a rural-side correction referenced to that same rural mask's own mean
cancels to exactly zero once re-averaged over that mask, so it would always
be a no-op).
"""

import os
import glob
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from shapely.vectorized import contains as shp_contains

from lapse_mode_utils import LAPSE_RATE_CONSTANT

WRF_DIR     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_consolidated_intermediate'
WRF_PATTERN = 'wrf_harmonized_{date}.nc'

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'wrf_daily')


def build_mask(union, flat_x, flat_y, shape):
    """Boolean point-in-polygon mask over flattened coords, bbox-prefiltered."""
    b   = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) &
           (flat_y >= b[1]) & (flat_y <= b[3]))
    m   = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)


def elev_diff_field(elev2d, rural_m):
    """elev2d minus the rural mask's MEAN elevation -- the reusable, rate-free
    part of urban_elev_correction() (below), split out so a per-timestep-
    varying rate (e.g. the dynamic valley lapse rate) can be multiplied in
    per timestep without recomputing this field every time."""
    r = elev2d[rural_m]; r = r[~np.isnan(r)]
    if len(r) == 0:
        return np.zeros_like(elev2d)
    ref_elev = float(np.nanmean(r))
    return elev2d - ref_elev


def urban_elev_correction(elev2d, rural_m, lapse_rate=LAPSE_RATE_CONSTANT):
    """Per-pixel correction for URBAN pixels only -- see module docstring."""
    return elev_diff_field(elev2d, rural_m) * lapse_rate


def wrf_bounds():
    """(start, end) UTC timestamps covered by the daily archive, or (None, None)."""
    files = sorted(glob.glob(os.path.join(WRF_DIR, WRF_PATTERN.format(date='*'))))
    if not files:
        return None, None
    ds0 = xr.open_dataset(files[0])[['time']]
    t_start = pd.Timestamp(ds0.time.values[0]); ds0.close()
    dsN = xr.open_dataset(files[-1])[['time']]
    t_end = pd.Timestamp(dsN.time.values[-1]); dsN.close()
    return t_start, t_end


def _file_sig(path):
    if not os.path.exists(path):
        return 'missing'
    st = os.stat(path)
    return f'{st.st_mtime:.0f}:{st.st_size}'


def _dir_sig(day_list):
    parts = [f'{day:%Y%m%d}:'
             f'{_file_sig(os.path.join(WRF_DIR, WRF_PATTERN.format(date=day.strftime("%Y-%m-%d"))))}'
             for day in day_list]
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]


def iter_wrf_uhi(date_start, date_end, urban_union_wgs, rural_union_wgs,
                  lapse_rate=LAPSE_RATE_CONSTANT, verbose=True):
    """
    Yield (times, uhi_mean, uhi_p25, uhi_p75) per calendar day in
    [date_start, date_end] from the daily-harmonized WRF archive: `times` is
    a DatetimeIndex (UTC, no offset); the three UHI arrays are same-length
    np.ndarrays of elevation-corrected UHI (deg C) across urban pixels at
    each timestep -- mean, and the 25th/75th percentile spatial spread (for a
    shaded time-series band; a bucketed seasonal composite can just ignore
    the last two). Missing days are skipped. Masks and the correction field
    are cached per grid shape, so the mid-archive domain regrid (south_north
    82->113 px around 2022-10-26) is handled transparently.

    Callers concatenate the yielded arrays for a flat continuous time series,
    or bucket each (t, u) pair into temporal[season][hour] for a seasonal
    composite -- same extraction either way. See load_wrf_uhi() for a cached,
    already-concatenated convenience wrapper.

    Each daily file spans 06:00 UTC on its named day through 05:00 UTC the
    NEXT day (spin-up already trimmed), not 00:00-23:59 of its own day -- so
    covering date_start's own 00:00-05:59 UTC requires also opening the file
    named one day earlier, and the requested window is trimmed to exactly
    [date_start 00:00, date_end 23:59] afterward (otherwise the series would
    both be missing date_start's first 6h AND run ~5h past date_end).

    `lapse_rate` may be a plain float (applied uniformly), or a callable
    `rate_fn(pd.Timestamp) -> float` for a per-timestep-varying rate (e.g. the
    dynamic per-(season, day/night) valley rate other sources in the same
    figure already use via lapse_mode_utils.resolve_rate() -- pass a closure
    built from that, so WRF stays on the SAME LAPSE_MODE as everything else
    in a given script instead of silently defaulting to the fixed constant).
    """
    t0 = pd.Timestamp(date_start)
    t1 = pd.Timestamp(date_end) + pd.Timedelta('23h59m59s')
    day_list = pd.date_range(t0.floor('D') - pd.Timedelta(days=1), date_end, freq='D')
    rate_is_dynamic = callable(lapse_rate)
    grid_cache = {}   # shape -> (urban_m, rural_m, ediff2d)
    n_found = n_missing = 0

    for di, day in enumerate(day_list):
        fpath = os.path.join(WRF_DIR, WRF_PATTERN.format(date=day.strftime('%Y-%m-%d')))
        if not os.path.exists(fpath):
            n_missing += 1
            continue
        n_found += 1

        try:
            ds = xr.open_dataset(fpath)[['T2', 'HGT', 'lat', 'lon']]
            lon2d = ds['lon'].values
            shape = lon2d.shape
            if shape not in grid_cache:
                lat2d   = ds['lat'].values
                urban_m = build_mask(urban_union_wgs, lon2d.flatten(), lat2d.flatten(), shape)
                rural_m = build_mask(rural_union_wgs, lon2d.flatten(), lat2d.flatten(), shape)
                hgt0    = ds['HGT'].isel(time=0).values
                ediff2d = elev_diff_field(hgt0, rural_m)
                grid_cache[shape] = (urban_m, rural_m, ediff2d)
                if verbose:
                    print(f'  [wrf_daily] new grid {shape}  urban={urban_m.sum()} rural={rural_m.sum()} px '
                          f'(from {os.path.basename(fpath)})')
            else:
                urban_m, rural_m, ediff2d = grid_cache[shape]
            if not rate_is_dynamic:
                urban_corr2d = ediff2d * lapse_rate

            t2    = ds['T2'].load().values   # (time, south_north, west_east), K
            times = pd.DatetimeIndex(ds['time'].values)   # WRF is UTC -- no offset
            ds.close()

            if np.nanmean(t2[0]) > 200:
                t2 = t2 - 273.15   # K -> degC

            n = t2.shape[0]
            day_mean = np.full(n, np.nan)
            day_p25  = np.full(n, np.nan)
            day_p75  = np.full(n, np.nan)
            for ti in range(n):
                snap      = t2[ti]
                r_mean    = float(np.nanmean(snap[rural_m]))
                if rate_is_dynamic:
                    urban_corr2d = ediff2d * lapse_rate(times[ti])
                snap_corr = snap + urban_corr2d
                upx       = (snap_corr - r_mean)[urban_m]; upx = upx[np.isfinite(upx)]
                if len(upx) == 0:
                    continue
                day_mean[ti] = float(np.nanmean(upx))
                if len(upx) >= 2:
                    day_p25[ti] = float(np.percentile(upx, 25))
                    day_p75[ti] = float(np.percentile(upx, 75))
                else:
                    day_p25[ti] = day_p75[ti] = day_mean[ti]

            valid = ~np.isnan(day_mean) & (times >= t0) & (times <= t1)
            if valid.any():
                yield times[valid], day_mean[valid], day_p25[valid], day_p75[valid]

        except Exception as e:
            print(f'  [wrf_daily] [ERROR] {os.path.basename(fpath)}: {e}')

        if verbose and (di + 1) % 200 == 0:
            print(f'  [wrf_daily] …{di + 1}/{len(day_list)} days scanned ({n_found} found, {n_missing} missing)')

    if verbose:
        print(f'  [wrf_daily] done: {n_found} daily files read, {n_missing} missing '
              f'over {len(day_list)} calendar days')


def load_wrf_uhi(date_start, date_end, urban_union_wgs, rural_union_wgs,
                  cache_key, lapse_rate=LAPSE_RATE_CONSTANT, lapse_rate_key=None,
                  force_recompute=False, verbose=True):
    """
    Cached convenience wrapper around iter_wrf_uhi(): returns the full
    concatenated (times, uhi_mean, uhi_p25, uhi_p75) for [date_start,
    date_end] as a DatetimeIndex + three np.ndarrays, instead of a per-day
    generator. Callers that only need the mean (e.g. a seasonal composite
    bucketing by season/hour) can just ignore the last two arrays.

    `cache_key` should identify the caller (e.g. 'hw4_dashboard',
    'diurnal_uhi_seasonal') so different scripts/date windows don't collide;
    combined internally with a signature over every day's WRF file
    (mtime+size) and the requested date range, so a stale cache is never
    silently reused after the underlying archive changes.

    If `lapse_rate` is callable (see iter_wrf_uhi()), a closure can't be
    hashed into the cache key by itself -- pass a short, stable
    `lapse_rate_key` string identifying which dynamic-rate variant this is
    (e.g. 'valley', matching the caller's own LAPSE_MODE), so switching modes
    can't silently reuse another mode's cached result.
    """
    if callable(lapse_rate) and lapse_rate_key is None:
        raise ValueError('lapse_rate is callable -- pass lapse_rate_key to identify it for caching')
    day_list = pd.date_range(date_start, date_end, freq='D')
    rate_sig = lapse_rate_key if callable(lapse_rate) else lapse_rate
    sig = hashlib.sha1(
        f'{cache_key}|{date_start}|{date_end}|{rate_sig}|{_dir_sig(day_list)}'.encode()
    ).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'wrf_uhi_{cache_key}_{sig}.npz')

    if not force_recompute and os.path.exists(cache_path):
        data = np.load(cache_path)
        if verbose:
            print(f'  [wrf_daily] [CACHE] loaded {os.path.basename(cache_path)}')
        return pd.DatetimeIndex(data['times']), data['uhi_mean'], data['uhi_p25'], data['uhi_p75']

    times_all, mean_all, p25_all, p75_all = [], [], [], []
    for t, u_mean, u_p25, u_p75 in iter_wrf_uhi(date_start, date_end, urban_union_wgs, rural_union_wgs,
                                                 lapse_rate=lapse_rate, verbose=verbose):
        times_all.append(t)
        mean_all.append(u_mean)
        p25_all.append(u_p25)
        p75_all.append(u_p75)

    times = pd.DatetimeIndex(np.concatenate(times_all)) if times_all else pd.DatetimeIndex([])
    uhi_mean = np.concatenate(mean_all) if mean_all else np.array([])
    uhi_p25  = np.concatenate(p25_all) if p25_all else np.array([])
    uhi_p75  = np.concatenate(p75_all) if p75_all else np.array([])

    np.savez_compressed(cache_path, times=times.values, uhi_mean=uhi_mean, uhi_p25=uhi_p25, uhi_p75=uhi_p75)
    if verbose:
        print(f'  [wrf_daily] [CACHE] saved {os.path.basename(cache_path)}')
    return times, uhi_mean, uhi_p25, uhi_p75
