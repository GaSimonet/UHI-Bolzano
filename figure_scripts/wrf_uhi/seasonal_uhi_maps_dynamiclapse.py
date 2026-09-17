#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal / day-night UHI maps — WRF, daily harmonized archive, dynamic lapse rate
Rewrite of seasonal_UHI_WRF_KEEPME_20260223.py (kept untouched, not overwritten).

WHAT CHANGED vs the KEEPME original
------------------------------------
1. SOURCE: the original picked ONE year at a time from WRF_RAW_<year>_subset
   single-file archives via `salem.open_wrf_dataset` (whichever WRF_FILE
   assignment was left uncommented last -- effectively just 2024). This
   version loops over WRF_consolidated_intermediate/wrf_harmonized_YYYY-MM-DD.nc,
   the same daily, spin-up-trimmed archive used by
   plot_diurnal_uhi_seasonal_wrfdaily_20260731.py, covering the full
   2020-01-01 .. 2024-12-31 archive (5 years instead of 1) in a single run.
   No `salem` dependency: the daily files already carry plain 2-D lat/lon, so
   urban/rural masking uses shapely against the same WGS84 shapefiles.

2. DYNAMIC LAPSE RATE: the original applied one fixed LAPSE_RATE = 0.0065 K/m
   (the standard-atmosphere value) everywhere, and used it as
   `(elev - mean(rural_elev))`, which -- see plot_diurnal_uhi_seasonal_wrfdaily's
   lapse_corr() fix -- cancels to zero once spatially averaged over that same
   rural mask, silently making the correction a no-op regardless of the
   constant's value. This version instead uses a typical VALLEY lapse rate per
   SEASON x DAY/NIGHT (8 values -- following the Simonet et al. 2025
   methodology), fit as the OLS slope of T2 vs HGT over a VALLEY-representative
   polygon (the along-river valley transect, buffered 5 km either side,
   restricted to pixels from the valley floor up to 3000 m a.s.l.). As of
   2026-08-03 that fit was pulled OUT of this script into
   seasonal_valley_lapse_rate.py, which writes the stable
   UHI/seasonal_valley_lapse_rates.csv other figure scripts can also read
   (see load_seasonal_valley_lapse() there) -- this script now just loads it.
   The day/night split itself is NOT a fixed UTC hour range: both this script
   and the fit script classify each hourly timestep using that CALENDAR DAY's
   real astronomical sunrise/sunset for Bolzano (solar_daynight.py),
   so the window is ~8.5h wide in December and ~15.8h wide in June rather than
   a season-blind 06-17 UTC block -- the fit bucket and the bucket a rate gets
   applied to are always the same real window.
   The correction is still anchored to the rural mask's mean elevation
   (unchanged from before) so it isn't a no-op once spatially averaged.

3. GRID REGRID MID-ARCHIVE: the domain was regridded on 2022-10-26
   (south_north 82 -> 113 px). Unlike the diurnal script (which only needs
   scalar urban/rural means, trivially robust to a shape change), this
   script also builds SPATIAL composite maps across the full 5-year period,
   which requires per-pixel alignment. Checking the actual coordinates
   showed the two regimes are NOT a simple crop/extension of the same
   lattice -- the origin is offset by ~500m in both x and y, i.e. roughly
   half a grid cell, not a whole one -- so pixels don't correspond 1:1 by
   index OR by rounding lat/lon. Every day's native grid is instead
   resampled (nearest-neighbor, in UTM32632 meters) onto one fixed canonical
   output grid built once at the start; the two distinct native shapes each
   get their own precomputed index mapping (cheap: built once per shape, not
   per file).

Runs under any recent env with xarray/geopandas/shapely/scipy/pyproj
(the harmonized files need no salem/WRF-python stack, unlike the original).
"""

import os
import io
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.vectorized import contains as shp_contains
import geopandas as gpd
import xarray as xr
import contextily as ctx
from PIL import Image
from scipy.spatial import cKDTree
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seasonal_valley_lapse_rate import load_seasonal_valley_lapse
from solar_daynight import classify_day_night
from lapse_mode_utils import resolve_rate, LAPSE_RATE_CONSTANT

# ── CONFIG ─────────────────────────────────────────────────────────────────────
WRF_DIR     = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_consolidated_intermediate'
WRF_PATTERN = 'wrf_harmonized_{date}.nc'

URBAN_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
RURAL_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'
# Valley shapefile / buffer / elevation-cutoff constants used to FIT the seasonal lapse
# rate now live in seasonal_valley_lapse_rate.py -- this script only
# loads the resulting per-season rates (see load_seasonal_valley_lapse() import above).

OUT_DIR   = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/UHI/harmonized_dynamiclapse_outputs'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
FORCE_RECOMPUTE = False

DATE_START = '2020-01-01'
DATE_END   = '2024-12-31'   # full harmonized archive -- WRF-only script, no other source to intersect against

# Day/night is no longer a fixed UTC hour range -- see classify_day_night() import above:
# each timestep is classified by its own calendar day's real Bolzano sunrise/sunset.

LAPSE_RATE_FALLBACK = 0.0065   # K/m, only ever used inside the lapse-fit script itself as its fallback

# Elevation-correction mode -- edit and re-run to get each version; see
# common/lapse_mode_utils.py. Output filenames are tagged with this mode.
#   'none'     -- no elevation correction (raw urban-minus-rural difference)
#   'constant' -- classical fixed rate (LAPSE_RATE_CONSTANT), no season/day-night dependence
#   'valley'   -- dynamic, per-(season, day/night) rate fitted from WRF (default)
LAPSE_MODE = 'valley'

UTM_CRS = 'EPSG:32632'
WGS84   = 'EPSG:4326'

CANON_RES_M   = 1000.0   # canonical output-grid resolution, meters -- matches native WRF pixel size
                          # (1:1, no oversampling) so the map shows the real ~15-pixel WRF grid over
                          # Bolzano, same as the KEEPME original -- not an artificially finer grid.
                          # A common canonical grid is still needed because the two native-grid
                          # regimes (pre/post 2022-10-26) are offset by ~500m, not a whole cell.
CANON_BUFFER_M = 1000.0  # buffer around the URBAN shapefile bounds only, meters

SEASON_MAP = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring',  4: 'Spring',  5: 'Spring',
               6: 'Summer',  7: 'Summer',  8: 'Summer',
               9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
PERIOD_ORDER = ['full', 'day', 'night']

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
})

_ll_to_utm = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)

# ── HELPERS ────────────────────────────────────────────────────────────────────

def build_mask(union, flat_x, flat_y, shape):
    b = union.bounds
    pre = ((flat_x >= b[0]) & (flat_x <= b[2]) & (flat_y >= b[1]) & (flat_y <= b[3]))
    m = np.zeros(len(flat_x), dtype=bool)
    if pre.any():
        m[pre] = shp_contains(union, flat_x[pre], flat_y[pre])
    return m.reshape(shape)

def _file_sig(path):
    if not os.path.exists(path):
        return 'missing'
    st = os.stat(path)
    return f'{st.st_mtime:.0f}:{st.st_size}'

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

# ── SHAPEFILES + CANONICAL OUTPUT GRID ──────────────────────────────────────────
print('Loading shapefiles…')
urban_utm = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
rural_utm = gpd.read_file(RURAL_SHP).to_crs(UTM_CRS)
urban_wgs = urban_utm.to_crs(WGS84)
rural_wgs = rural_utm.to_crs(WGS84)
urban_wgs_union = urban_wgs.unary_union
rural_wgs_union = rural_wgs.unary_union

# Bounds from the URBAN shapefile only (+ a modest buffer for smooth edges/context),
# matching the KEEPME original: its maps plot ds_urban_roi -- an ROI'd dataset where
# only Bolzano-urban pixels carry data and everything else is NaN -- not the wider
# rural buffer. The rural mask is still used for the lapse-rate fit and the rural
# reference mean; it's just not part of what gets drawn on the map.
urban_bounds = urban_utm.total_bounds
cx0, cy0, cx1, cy1 = (urban_bounds[0] - CANON_BUFFER_M, urban_bounds[1] - CANON_BUFFER_M,
                      urban_bounds[2] + CANON_BUFFER_M, urban_bounds[3] + CANON_BUFFER_M)
canon_x = np.arange(cx0, cx1, CANON_RES_M)
canon_y = np.arange(cy0, cy1, CANON_RES_M)
canon_xx, canon_yy = np.meshgrid(canon_x, canon_y)
CANON_SHAPE = canon_xx.shape
print(f'  Canonical output grid: {CANON_SHAPE}  ({CANON_RES_M:.0f}m res, '
      f'{(cx1-cx0)/1000:.1f}x{(cy1-cy0)/1000:.1f} km extent)')

# plot_maps() draws canon_x/canon_y in KM (canon_x / 1000, etc.), so the urban
# boundary overlay needs its geometry in the same km units -- plotting
# urban_utm's raw meter coordinates directly against a km-scaled axis put the
# boundary ~1000x outside the visible range (x ~677000 vs an axis range of
# ~676-683), i.e. it was silently invisible. affine_transform just rescales
# coordinates (no re-projection), so shapes stay correct, only the units change.
urban_utm_km = urban_utm.copy()
urban_utm_km['geometry'] = urban_utm_km.geometry.affine_transform([1/1000, 0, 0, 1/1000, 0, 0])

# Urban/rural masks on the canonical grid itself, for scalar summaries computed
# directly from the composited maps and for the plotted urban boundary overlay.
_utm_to_ll = Transformer.from_crs(UTM_CRS, WGS84, always_xy=True)
canon_lon, canon_lat = _utm_to_ll.transform(canon_xx.flatten(), canon_yy.flatten())
canon_lon = canon_lon.reshape(CANON_SHAPE); canon_lat = canon_lat.reshape(CANON_SHAPE)
canon_urban_m = build_mask(urban_wgs_union, canon_lon.flatten(), canon_lat.flatten(), CANON_SHAPE)
canon_rural_m = build_mask(rural_wgs_union, canon_lon.flatten(), canon_lat.flatten(), CANON_SHAPE)
print(f'  Canonical grid urban px={canon_urban_m.sum()}  rural px={canon_rural_m.sum()}')

# ── PER-SHAPE GRID CACHE (masks, HGT, nearest-neighbor mapping onto canonical grid) ──
# The domain was regridded on 2022-10-26 and the two regimes' lattices are offset by
# ~500m (not a whole 1km cell) -- see module docstring -- so native pixels never line
# up across the boundary. Everything is instead resampled onto the fixed canonical
# grid above via nearest-neighbor (in UTM meters), with the NN index mapping built
# once per distinct native shape and reused for every file sharing that shape.
grid_cache = {}   # native shape -> dict(urban_m, rural_m, hgt, ref_elev, nn_idx)

def get_grid_info(ds):
    lon2d = ds['lon'].values
    lat2d = ds['lat'].values
    shape = lon2d.shape
    if shape in grid_cache:
        return grid_cache[shape]

    urban_m = build_mask(urban_wgs_union, lon2d.flatten(), lat2d.flatten(), shape)
    rural_m = build_mask(rural_wgs_union, lon2d.flatten(), lat2d.flatten(), shape)
    hgt = ds['HGT'].isel(time=0).values
    # MEAN, not min -- matches the corrected convention now used in
    # plot_diurnal_uhi_seasonal_wrfdaily_20260731.py: the rural mean elevation is an
    # external reference that the OTHER side (every non-rural pixel, incl. urban) gets
    # corrected toward, while the rural mean itself stays raw/uncorrected. Because the
    # reference sits outside the set being averaged for the diff, there's no
    # self-cancellation risk (unlike referencing rural to its own mean and then
    # correcting rural itself, which cancels to zero once re-averaged over that mask).
    ref_elev = float(np.nanmean(hgt[rural_m]))

    nx, ny = _ll_to_utm.transform(lon2d.flatten(), lat2d.flatten())
    tree = cKDTree(np.column_stack([nx, ny]))
    _, nn_idx = tree.query(np.column_stack([canon_xx.flatten(), canon_yy.flatten()]))
    nn_idx = nn_idx.reshape(CANON_SHAPE)

    info = dict(urban_m=urban_m, rural_m=rural_m, hgt=hgt, ref_elev=ref_elev, nn_idx=nn_idx)
    grid_cache[shape] = info
    print(f'  New native grid {shape}: urban={urban_m.sum()}px rural={rural_m.sum()}px '
          f'ref_elev={ref_elev:.0f}m -- NN mapping to canonical grid built')
    return info

# ── DAY LIST ───────────────────────────────────────────────────────────────────
day_list = pd.date_range(DATE_START, DATE_END, freq='D')
print(f'\n{len(day_list)} calendar days requested ({DATE_START} .. {DATE_END})')

# ── PASS 1 — load the pre-fit, typical VALLEY lapse rate per season ────────────
# The fit itself (OLS T2-vs-HGT over the valley polygon) now lives in
# seasonal_valley_lapse_rate.py, which writes the stable
# UHI/seasonal_valley_lapse_rates.csv this loads -- run that script first (or after
# the WRF archive changes) to (re)generate it. Loading it here keeps this script and
# any other figure script consuming the exact same seasonal rates.
print('\n[Pass 1/2] Loading pre-fit seasonal x day/night valley lapse rate…')
dynamic_lapse = load_seasonal_valley_lapse()   # {(season, period): K/m}
lapse_df = pd.DataFrame([(s, p, v) for (s, p), v in sorted(dynamic_lapse.items())],
                         columns=['season', 'period', 'lapse_rate_K_per_m'])
print(lapse_df.to_string(index=False))

# ── PASS 2 — apply dynamic lapse correction, accumulate scalars + spatial maps ──
print('\n[Pass 2/2] Computing corrected/uncorrected UHI (scalars + canonical spatial maps)…')

results_cache = cache_path('results', DATE_START, DATE_END, LAPSE_MODE,
                            tuple(sorted(dynamic_lapse.items())), CANON_SHAPE, CANON_RES_M, CANON_BUFFER_M,
                            _dir_sig(WRF_DIR, WRF_PATTERN, day_list))

buckets = [(s, p) for s in SEASON_ORDER for p in PERIOD_ORDER]

if not FORCE_RECOMPUTE and os.path.exists(results_cache):
    _d = np.load(results_cache)
    spatial_sum = {}; spatial_cnt = {}; scalar_vals = {}
    for s, p in buckets:
        for kind in ('corrected', 'uncorrected'):
            key = f'{s}_{p}_{kind}'
            spatial_sum[(s, p, kind)] = _d[f'{key}__sum']
            spatial_cnt[(s, p, kind)] = _d[f'{key}__cnt']
            scalar_vals[(s, p, kind)] = _d[f'{key}__scalar']
    print(f'  [CACHE] loaded {os.path.basename(results_cache)}')
else:
    spatial_sum = {(s, p, k): np.zeros(CANON_SHAPE) for s, p in buckets for k in ('corrected', 'uncorrected')}
    spatial_cnt = {(s, p, k): np.zeros(CANON_SHAPE) for s, p in buckets for k in ('corrected', 'uncorrected')}
    scalar_lists = {(s, p, k): [] for s, p in buckets for k in ('corrected', 'uncorrected')}

    for di, day in enumerate(day_list):
        fpath = os.path.join(WRF_DIR, WRF_PATTERN.format(date=day.strftime('%Y-%m-%d')))
        if not os.path.exists(fpath):
            continue
        try:
            ds = xr.open_dataset(fpath)[['T2', 'HGT', 'lat', 'lon']]
            info = get_grid_info(ds)
            urban_m, rural_m, hgt, ref_elev, nn_idx = (
                info['urban_m'], info['rural_m'], info['hgt'], info['ref_elev'], info['nn_idx'])

            t2 = ds['T2'].load().values
            times = pd.DatetimeIndex(ds['time'].values)
            ds.close()
            if np.nanmean(t2[0]) > 200:
                t2 = t2 - 273.15

            periods = classify_day_night(times)   # per-timestep, real sunrise/sunset for that date

            for ti in range(t2.shape[0]):
                t = times[ti]
                season, period = SEASON_MAP[t.month], periods[ti]
                snap = t2[ti]

                # Reference stays the RAW rural mean (uncorrected) -- matches the
                # convention in plot_diurnal_uhi_seasonal_wrfdaily_20260731.py: the
                # correction is applied to the non-reference side (here: every pixel
                # in the diff field, urban included) via a per-pixel elevation term
                # relative to the rural mask's mean elevation, not to the rural
                # reference itself (which would cancel out once re-averaged over the
                # same mask it's referenced to).
                rural_mean_raw = float(np.nanmean(snap[rural_m]))

                lapse = resolve_rate(LAPSE_MODE, dynamic_lapse.get((season, period)))
                elev_corr_field = lapse * (hgt - ref_elev)

                diff_uncorr = snap - rural_mean_raw
                diff_corr   = (snap + elev_corr_field) - rural_mean_raw

                flat_uncorr = diff_uncorr.flatten()
                flat_corr   = diff_corr.flatten()
                resamp_uncorr = flat_uncorr[nn_idx]
                resamp_corr   = flat_corr[nn_idx]

                urban_uncorr = float(np.nanmean(diff_uncorr[urban_m]))
                urban_corr   = float(np.nanmean(diff_corr[urban_m]))

                for period_key in ({period, 'full'}):
                    spatial_sum[(season, period_key, 'uncorrected')] += np.nan_to_num(resamp_uncorr)
                    spatial_cnt[(season, period_key, 'uncorrected')] += np.isfinite(resamp_uncorr)
                    spatial_sum[(season, period_key, 'corrected')]   += np.nan_to_num(resamp_corr)
                    spatial_cnt[(season, period_key, 'corrected')]   += np.isfinite(resamp_corr)
                    scalar_lists[(season, period_key, 'uncorrected')].append(urban_uncorr)
                    scalar_lists[(season, period_key, 'corrected')].append(urban_corr)

        except Exception as e:
            print(f'  [ERROR] {os.path.basename(fpath)}: {e}')

        if (di + 1) % 300 == 0:
            print(f'  …{di + 1}/{len(day_list)} days scanned (pass 2)')

    scalar_vals = {k: np.array(v, dtype=np.float64) for k, v in scalar_lists.items()}

    cache_payload = {}
    for s, p in buckets:
        for kind in ('corrected', 'uncorrected'):
            key = f'{s}_{p}_{kind}'
            cache_payload[f'{key}__sum'] = spatial_sum[(s, p, kind)]
            cache_payload[f'{key}__cnt'] = spatial_cnt[(s, p, kind)]
            cache_payload[f'{key}__scalar'] = scalar_vals[(s, p, kind)]
    np.savez_compressed(results_cache, **cache_payload)
    print(f'  [CACHE] saved {os.path.basename(results_cache)}')

# Finalize composite maps (mean = sum/count, NaN where no samples) and scalar summary table
composite_map = {}
for (s, p, kind), summ in spatial_sum.items():
    cnt = spatial_cnt[(s, p, kind)]
    with np.errstate(invalid='ignore', divide='ignore'):
        composite_map[(s, p, kind)] = np.where(cnt > 0, summ / np.maximum(cnt, 1), np.nan)

summary_rows = []
for s, p in buckets:
    for kind in ('corrected', 'uncorrected'):
        v = scalar_vals[(s, p, kind)]
        v = v[np.isfinite(v)]
        summary_rows.append({'season': s, 'period': p, 'type': kind,
                              'mean_UHI_C': float(np.mean(v)) if len(v) else np.nan,
                              'std_UHI_C': float(np.std(v)) if len(v) else np.nan,
                              'n_timesteps': int(len(v))})
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, f'seasonal_uhi_stats_harmonized_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv')
summary_df.to_csv(summary_csv, index=False)
print(f'\nSaved scalar summary → {summary_csv}')
print(summary_df.to_string(index=False))

# ── PLOTTING ───────────────────────────────────────────────────────────────────
# Grey OpenTopoMap background, same technique as add_grayscale_basemap() in
# UHI_combined_heatmap_distri_map_20260709.py: render a basemap tile over the
# canonical extent, convert to grayscale, and place it under the pcolormesh
# data. Every panel (4 seasons x 3 periods x 2 kinds) shares the exact same
# canonical bounds (cx0,cy0,cx1,cy1), so the tile is rendered/cached ONCE here
# and reused for all 24 panels instead of re-fetching per subplot.

BASEMAP_SOURCE = ctx.providers.OpenTopoMap
BASEMAP_CACHE  = os.path.join(CACHE_DIR, 'basemap_grayscale_opentopomap.npy')

def render_grayscale_basemap_tile(x0_m, x1_m, y0_m, y1_m, source=BASEMAP_SOURCE, dpi=150):
    fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(6, 6))
    ax_tmp.set_xlim(x0_m, x1_m)
    ax_tmp.set_ylim(y0_m, y1_m)
    ctx.add_basemap(ax_tmp, crs=UTM_CRS, source=source, zoom='auto', zorder=1, attribution_size=0)
    ax_tmp.axis('off')
    buf = io.BytesIO()
    fig_tmp.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig_tmp)
    buf.seek(0)
    tile_img = Image.open(buf).convert('L')
    return np.array(tile_img.convert('RGB'))

print('\nFetching grayscale OpenTopoMap basemap (rendered once, reused across all panels)…')
if not FORCE_RECOMPUTE and os.path.exists(BASEMAP_CACHE):
    basemap_tile = np.load(BASEMAP_CACHE)
    print(f'  [CACHE] loaded {os.path.basename(BASEMAP_CACHE)}')
else:
    try:
        basemap_tile = render_grayscale_basemap_tile(cx0, cx1, cy0, cy1)
        np.save(BASEMAP_CACHE, basemap_tile)
        print(f'  [CACHE] saved {os.path.basename(BASEMAP_CACHE)}')
    except Exception as e:
        basemap_tile = None
        print(f'  [WARN] basemap fetch failed ({e}) -- panels will have a plain white background')

def plot_maps(kind):
    # Only Bolzano-urban pixels are drawn -- matches the KEEPME original, which
    # plotted ds_urban_roi (urban pixels only, rest NaN), not the wider rural buffer.
    urban_only = {(s, p): np.where(canon_urban_m, composite_map[(s, p, kind)], np.nan)
                  for s, p in buckets}

    settings = dict(vmin=-1.5, vmax=1.5, title=f'{"Lapse-Corrected (dynamic)" if kind == "corrected" else "Uncorrected"}')
    vals = np.concatenate([v[np.isfinite(v)] for v in urban_only.values()])
    if len(vals):
        vmax = float(np.percentile(np.abs(vals), 98))
        settings['vmin'], settings['vmax'] = -vmax, vmax

    fig = plt.figure(figsize=(15, 17))
    gs = GridSpec(len(SEASON_ORDER), len(PERIOD_ORDER), figure=fig, hspace=0.15, wspace=0.05)
    period_labels = {'full': '24h', 'day': 'Day (sunrise–sunset)', 'night': 'Night (sunset–sunrise)'}

    for i, season in enumerate(SEASON_ORDER):
        for j, period in enumerate(PERIOD_ORDER):
            ax = fig.add_subplot(gs[i, j])
            data = urban_only[(season, period)]
            if basemap_tile is not None:
                ax.imshow(basemap_tile, extent=[cx0 / 1000, cx1 / 1000, cy0 / 1000, cy1 / 1000],
                          origin='upper', aspect='auto', zorder=0, interpolation='bilinear')
            im = ax.pcolormesh(canon_x / 1000, canon_y / 1000, data,
                                cmap='RdBu_r', vmin=settings['vmin'], vmax=settings['vmax'],
                                shading='auto', alpha=0.75, zorder=1)
            urban_utm_km.boundary.plot(ax=ax, color='black', linewidth=1.3, zorder=2)
            ax.set_xlim(cx0 / 1000, cx1 / 1000)
            ax.set_ylim(cy0 / 1000, cy1 / 1000)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(season, fontsize=14, fontweight='bold')
            if i == len(SEASON_ORDER) - 1:
                ax.set_xlabel(period_labels[period], fontsize=12)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.015])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='UHI intensity (°C)')
    # fig.suptitle(f'Seasonal / day-night UHI — {settings["title"]}\n'
    #              f'WRF daily harmonized archive, {DATE_START} .. {DATE_END}',
    #              fontsize=15, fontweight='bold', y=0.985)

    p = os.path.join(OUT_DIR, f'seasonal_uhi_maps_{kind}_{LAPSE_MODE}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'Saved → {p}')
    plt.close(fig)

plot_maps('corrected')
plot_maps('uncorrected')

# The lapse-rate-by-season diagnostic bar plot now lives in
# seasonal_valley_lapse_rate.py (it's the script that owns the fit).

print('\n✓ Done.')
