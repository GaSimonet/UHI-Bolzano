#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: WRF_alto_adige_local/script/analysis/Horizontal_X_section/seasonal_wind_arrows_low_terrain.py
# Paper figure:       seasonal_wind_arrows_low_terrain.png
# Note: Unrelated to the valley-lapse-rate methodology (wind vectors, not UHI) -- included only because it feeds a paper figure.
# UPDATE (2026-08-04): the day/night split previously used a fixed DAYTIME_HOURS=range(8,20)
# clock-hour window (the same window for all four seasons) -- switched to the same
# astronomical sunrise/sunset classification (common/solar_daynight.classify_day_night) used
# across the UHI scripts, which properly tracks each season's real day length instead of one
# fixed window. Panel titles no longer print a clock-hour range (e.g. "Day (08-19 h)").
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-

"""
Seasonal × Day/Night Surface Wind Arrows (terrain ≤ 1500 m)
4 seasons (Winter / Spring / Summer / Autumn)  ×  2 periods (Day / Night)
= 8 map panels.

Layers per panel (identical style to surface_wind_arrows_low_terrain_HW_20260302.py):
  1. Grey terrain relief  — full domain
  2. Orange/red wind-speed fill — terrain ≤ MAX_TERRAIN only
  3. Terrain contour lines — full domain
  4. Black quiver arrows  — terrain ≤ MAX_TERRAIN only

Data: 4 annual WRF runs (2021–2024) concatenated along the time axis.
  - 2021, 2023, 2024 → single annual .nc files
  - 2022             → 12 monthly .nc files (opened per-file to avoid coord clash)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import salem
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# Astronomical day/night split (real sunrise/sunset per calendar day, Bolzano),
# replacing the old fixed DAYTIME_HOURS=range(8,20) clock-hour window -- matches
# the convention used across the UHI scripts (common/solar_daynight.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from solar_daynight import classify_day_night

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 13,
    'axes.titlesize' : 15,
    'axes.labelsize' : 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# =====================================================
# USER SETTINGS
# =====================================================

WRF_BASE = "/home/gsimonet/Desktop/WRF_ALTO_ADIGE_on_CEPH"

WRF_FILES = {
    2020: "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2020_subset/WRF_2020_RAW_subset.nc",
    2021: f"{WRF_BASE}/WRF_RAW_2021_subset/WRF_RAW_Alto_Adige_2021.nc",
    # 2022: sorted(glob.glob(f"{WRF_BASE}/WRF_RAW_2022_subset/months/WRF_2022_*.nc")),
    2023: f"{WRF_BASE}/WRF_RAW_2023_subset/WRF_RAW_Alto_Adige_2023.nc",
    2024: f"{WRF_BASE}/WRF_RAW_2024_subset/WRF_RAW_Alto_Adige_2024.nc",
}

SHAPEFILE_PATH = (
    "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/"
    "LCZ_shapefile_analysis/shapefiles/valley_x_sec_along_river_bolzano_v2.dbf"
)

XSEC_INDICES   = [50, 120]
XSEC_AB_LABELS = [('A', 'B'), ('C', 'D')]
XSEC_NAMES     = ['Cross-section North', 'Cross-section South']
CROSS_LEN_KM   = 15
SHOW_XSEC      = False  # set False to hide cross-section lines, labels and legend

MAX_TERRAIN  = 1500    # wind masked above this elevation (m)
MAX_TOPO     = 3600    # upper bound for terrain shading (m)
ARROW_STRIDE = 3       # grid subsampling for quiver (lower = more arrows)
ARROW_SCALE  = 50      # quiver scale (lower = longer arrows)
WS_VMIN      = 0.0
WS_VMAX      = 5.0     # colorbar / contourf upper bound (m s-1)

WIND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'orange_red', ['#FEE8C8', '#FDBB84', '#FC8D59', '#E34A33', '#B30000']
)

BOLZANO = (46.47, 11.32)   # (lat, lon)

SEASON_DEF = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
}
SEASONS = list(SEASON_DEF.keys())

PERIODS = [
    ('Day',   'Day'),
    ('Night', 'Night'),
]

OUTPUT_FILE = "seasonal_wind_arrows_low_terrain.png"

# Cache file — named after the years used so different year sets get fresh caches. The
# "_astro" tag distinguishes this from any pre-2026-08-04 cache keyed only by year, which
# would have been built with the old fixed-hour day/night split -- without the distinct
# name, a stale cache from before this method change could get silently reused.
years_str  = '_'.join(str(y) for y in WRF_FILES.keys())
CACHE_FILE = f"cache_seasonal_avg_{years_str}_astro.nc"

# =====================================================
# HELPERS
# =====================================================

def get_valley_coordinates(shapefile_path, spacing_km=0.5):
    valley      = gpd.read_file(shapefile_path)
    line        = valley.geometry.iloc[0]
    spacing_deg = spacing_km / 111
    n_points    = int(line.length / spacing_deg)
    return [(line.interpolate(i * spacing_deg).y,
             line.interpolate(i * spacing_deg).x)
            for i in range(n_points)]


def cross_section_endpoints(coords, idx, half_len_km=15):
    lat, lon = coords[idx]
    if idx < len(coords) - 1:
        next_lat, next_lon = coords[idx + 1]
    else:
        prev_lat, prev_lon = coords[idx - 1]
        next_lat = lat + (lat - prev_lat)
        next_lon = lon + (lon - prev_lon)

    dx = next_lon - lon
    dy = next_lat - lat
    valley_angle = np.degrees(np.arctan2(dy, dx))
    cross_dir_m  = (((90 - valley_angle) + 180) % 360 - 90) % 360
    cross_rad    = np.radians(90 - cross_dir_m)
    dx_c, dy_c   = np.cos(cross_rad), np.sin(cross_rad)

    lat_scale = 1 / 111
    lon_scale = 1 / (111 * np.cos(np.radians(lat)))

    return (
        (lat - dy_c * half_len_km * lat_scale,
         lon - dx_c * half_len_km * lon_scale),
        (lat + dy_c * half_len_km * lat_scale,
         lon + dx_c * half_len_km * lon_scale),
    )


def draw_panel(ax, d, lons_1d, lats_1d, xsec_ep,
               ws_ref_holder, qv_ref_holder,
               show_left_labels, show_bottom_labels):
    """Draw one map panel. Fills ws_ref_holder[0] and qv_ref_holder[0] on first call."""

    terrain = d.HGT.values
    u_wind  = d.U10.values
    v_wind  = d.V10.values
    ws      = d.WS10.values

    low_mask = terrain <= MAX_TERRAIN
    u_plot   = np.where(low_mask, u_wind, np.nan)
    v_plot   = np.where(low_mask, v_wind, np.nan)
    ws_plot  = np.where(low_mask, ws,     np.nan)

    ax.set_extent([lons_1d.min(), lons_1d.max(),
                   lats_1d.min(), lats_1d.max()], crs=proj)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS,  linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.LAKES,   alpha=0.4)

    # Layer 1: grey terrain — full domain
    ax.contourf(
        lons_1d, lats_1d, terrain,
        levels=np.linspace(0, MAX_TOPO, 35),
        cmap='Greys', alpha=0.55,
        transform=proj, extend='neither',
    )

    # Layer 2: wind-speed fill (orange → red) — ≤ MAX_TERRAIN only
    ws_cf = ax.contourf(
        lons_1d, lats_1d,
        np.ma.masked_invalid(ws_plot),
        levels=np.linspace(WS_VMIN, WS_VMAX, 20),
        cmap=WIND_CMAP, norm=norm_ws, alpha=0.75,
        transform=proj, extend='max',
    )
    if ws_ref_holder[0] is None:
        ws_ref_holder[0] = ws_cf

    # Layer 3: terrain contour lines — full domain
    ax.contour(
        lons_1d, lats_1d, terrain,
        levels=np.arange(0, MAX_TOPO + 1, 200),
        colors='#444444', linewidths=0.5, alpha=0.55,
        transform=proj,
    )

    # Layer 4: black quiver arrows — ≤ MAX_TERRAIN only
    s     = ARROW_STRIDE
    lon_q = lons_1d[::s]
    lat_q = lats_1d[::s]
    u_q   = np.nan_to_num(u_plot[::s, ::s])
    v_q   = np.nan_to_num(v_plot[::s, ::s])

    qv = ax.quiver(
        lon_q, lat_q, u_q, v_q,
        transform=proj,
        color='black',
        scale=ARROW_SCALE,
        scale_units='width',
        width=0.003,
        headwidth=4,
        headlength=4,
        headaxislength=3.5,
        zorder=6,
    )
    if qv_ref_holder[0] is None:
        qv_ref_holder[0] = qv

    # Cross-section lines
    if SHOW_XSEC:
        for (idx, (lbl_B, lbl_A)) in zip(XSEC_INDICES, XSEC_AB_LABELS):
            pt_A, pt_B = xsec_ep[idx]
            ax.plot([pt_A[1], pt_B[1]], [pt_A[0], pt_B[0]],
                    color='black', linewidth=2, transform=proj, zorder=10)
            for pt, lbl in [(pt_A, lbl_A), (pt_B, lbl_B)]:
                ax.text(pt[1], pt[0], lbl,
                        fontsize=9, fontweight='bold', color='white',
                        ha='center', va='center', transform=proj, zorder=11,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1,
                                  edgecolor='none'))
            lat_c, lon_c = valley_coords[idx]
            ax.scatter(lon_c, lat_c, c='black', s=40, zorder=12, transform=proj)

    # Bolzano marker
    ax.scatter(BOLZANO[1], BOLZANO[0], c='white', s=220, marker='*',
               edgecolors='black', linewidths=0.5, zorder=15, transform=proj)
    ax.text(BOLZANO[1] + 0.035, BOLZANO[0] + 0.035, "Bolzano",
            fontsize=12, fontweight='bold', color='white', transform=proj,
            zorder=20,
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = show_left_labels
    gl.bottom_labels = show_bottom_labels
    gl.xlabel_style  = {'size': 11}
    gl.ylabel_style  = {'size': 11}

    return ws_plot

# =====================================================
# CACHE — load pre-computed averages or recompute
# =====================================================

import gc

def _fix_time(yr_ds):
    """Vectorised YYYYMMDD.fractional_day float → datetime64[ns]."""
    if np.issubdtype(yr_ds.time.dtype, np.datetime64):
        return yr_ds
    raw   = yr_ds.time.values.astype(float)
    int_p = raw.astype(np.int64)
    dates = pd.to_datetime({'year': int_p // 10000,
                            'month': (int_p % 10000) // 100,
                            'day':   int_p % 100})
    secs = np.round((raw - int_p) * 86400).astype(np.int64)
    return yr_ds.assign_coords(time=dates + pd.to_timedelta(secs, unit='s'))


def _open_year(path):
    if isinstance(path, list):
        parts = []
        for p in path:
            m = salem.open_wrf_dataset(p)[['U10', 'V10', 'HGT']]
            if 'xtime' in m.coords:
                m = m.drop_vars('xtime')
            parts.append(_fix_time(m))
        return xr.concat(parts, dim='time')
    return _fix_time(salem.open_wrf_dataset(path)[['U10', 'V10', 'HGT']])


def _get_lat_lon_1d(path):
    """Read only the 1-D lat/lon arrays for a file (fast, no data loaded)."""
    p  = path[0] if isinstance(path, list) else path
    ds = salem.open_wrf_dataset(p)[['HGT']]
    lo = ds.lon.values; la = ds.lat.values
    lo1d = lo[0, :] if lo.ndim == 2 else lo
    la1d = la[:, 0] if la.ndim == 2 else la
    del ds
    return la1d, lo1d


def _common_lat_lon_slices(wrf_files):
    """
    Pre-scan all files to find the geographic intersection of their domains.
    Returns per-year (r0,r1,c0,c1) index slices and the common 1-D lat/lon arrays.
    Using lat/lon intersection instead of minimum shape avoids mixing data from
    different geographic areas when files have different domain extents.
    """
    lat_arrs, lon_arrs = {}, {}
    for year, path in wrf_files.items():
        lat_arrs[year], lon_arrs[year] = _get_lat_lon_1d(path)

    # Intersection: most-restrictive southern/northern/western/eastern bound
    lat_min = max(a.min() for a in lat_arrs.values())
    lat_max = min(a.max() for a in lat_arrs.values())
    lon_min = max(a.min() for a in lon_arrs.values())
    lon_max = min(a.max() for a in lon_arrs.values())

    tol = 0.02  # °  — handles small floating-point differences between files
    slices = {}
    ref_la = ref_lo = None
    for year in wrf_files:
        la = lat_arrs[year]; lo = lon_arrs[year]
        ri = np.where((la >= lat_min - tol) & (la <= lat_max + tol))[0]
        ci = np.where((lo >= lon_min - tol) & (lo <= lon_max + tol))[0]
        if len(ri) == 0 or len(ci) == 0:
            raise ValueError(
                f"Year {year} has no lat/lon overlap with other files.\n"
                f"  Domain: lat [{la.min():.2f}…{la.max():.2f}]  "
                f"lon [{lo.min():.2f}…{lo.max():.2f}]\n"
                f"  Common: lat [{lat_min:.2f}…{lat_max:.2f}]  "
                f"lon [{lon_min:.2f}…{lon_max:.2f}]\n"
                f"  → Comment out year {year} in WRF_FILES."
            )
        r0, r1 = int(ri[0]), int(ri[-1]) + 1
        c0, c1 = int(ci[0]), int(ci[-1]) + 1
        slices[year] = (r0, r1, c0, c1)
        if ref_la is None:
            ref_la = la[r0:r1]; ref_lo = lo[c0:c1]

    return slices, ref_la, ref_lo


def _compute_seasonal_avgs(wrf_files, season_def, pkeys):
    """One-year-at-a-time accumulation — avoids loading all years into RAM."""
    print("  Pre-scanning domains for geographic alignment...")
    spatial_slices, ref_lats, ref_lons = _common_lat_lon_slices(wrf_files)
    ny_ref, nx_ref = len(ref_lats), len(ref_lons)
    print(f"  Common grid: ({ny_ref} × {nx_ref})  "
          f"lat [{ref_lats.min():.2f}…{ref_lats.max():.2f}]  "
          f"lon [{ref_lons.min():.2f}…{ref_lons.max():.2f}]")

    running_sum, running_count = {}, {}
    hgt_ref = None
    lons_1d = ref_lons
    lats_1d = ref_lats
    sp_dims = None

    for year, path in wrf_files.items():
        r0, r1, c0, c1 = spatial_slices[year]
        print(f"  Processing {year}  (rows {r0}:{r1}, cols {c0}:{c1})...")
        yr_ds = _open_year(path)

        # Extract to numpy immediately, sliced to the common geographic area
        u_np  = yr_ds.U10.values[:, r0:r1, c0:c1][:, :ny_ref, :nx_ref]
        v_np  = yr_ds.V10.values[:, r0:r1, c0:c1][:, :ny_ref, :nx_ref]
        ws_np = np.sqrt(u_np**2 + v_np**2)

        if hgt_ref is None:
            hgt_ref = yr_ds['HGT'].values[0, r0:r1, c0:c1][:ny_ref, :nx_ref]
            sp_dims = list(yr_ds.U10.dims)[1:]  # skip first (time-like) dim

        times  = pd.to_datetime(yr_ds.time.values)
        months = np.array([t.month for t in times])
        day_b  = classify_day_night(times) == 'day'   # WRF is UTC -- no offset
        pmasks = {'Day': day_b, 'Night': ~day_b}

        del yr_ds; gc.collect()   # free raw dataset — only numpy arrays remain

        for season, mlist in season_def.items():
            s_mask = np.isin(months, mlist)
            for pkey in pkeys:
                idx = np.where(s_mask & pmasks[pkey])[0]
                n   = len(idx)
                if n == 0:
                    continue
                su = np.nansum(u_np[idx], axis=0)
                sv = np.nansum(v_np[idx], axis=0)
                sw = np.nansum(ws_np[idx], axis=0)
                key = (season, pkey)
                if key not in running_sum:
                    running_sum[key]   = [su, sv, sw]
                    running_count[key] = n
                else:
                    running_sum[key][0] += su
                    running_sum[key][1] += sv
                    running_sum[key][2] += sw
                    running_count[key]  += n

        del u_np, v_np, ws_np; gc.collect()
        print(f"    → done")

    avg_grid = {}
    for season in season_def:
        avg_grid[season] = {}
        for pkey in pkeys:
            key = (season, pkey)
            if key not in running_sum:
                print(f"  WARNING: {season} {pkey} → 0 timesteps")
                avg_grid[season][pkey] = None
            else:
                n = running_count[key]; su, sv, sw = running_sum[key]
                print(f"  {season:6s} {pkey:5s}: {n:5d} total timesteps")
                avg_grid[season][pkey] = xr.Dataset({
                    'U10':  xr.DataArray(su / n, dims=sp_dims),
                    'V10':  xr.DataArray(sv / n, dims=sp_dims),
                    'WS10': xr.DataArray(sw / n, dims=sp_dims),
                    'HGT':  xr.DataArray(hgt_ref, dims=sp_dims),
                })
    return avg_grid, lons_1d, lats_1d


if os.path.exists(CACHE_FILE):
    print(f"Loading cached averages from {CACHE_FILE}...")
    _c = xr.open_dataset(CACHE_FILE).load()
    avg_grid = {}
    for season in SEASONS:
        avg_grid[season] = {}
        for pkey, *_ in PERIODS:
            tag = f'{season}_{pkey}'
            if f'U10_{tag}' in _c:
                avg_grid[season][pkey] = xr.Dataset(
                    {v: _c[f'{v}_{tag}'] for v in ['U10', 'V10', 'HGT', 'WS10']}
                )
            else:
                avg_grid[season][pkey] = None
    lons_1d = _c['lons_1d'].values
    lats_1d = _c['lats_1d'].values
    print("  Done.")

else:
    print("Computing seasonal × day/night averages (one year at a time)...")
    pkeys = [pk for pk, *_ in PERIODS]
    avg_grid, lons_1d, lats_1d = _compute_seasonal_avgs(
        WRF_FILES, SEASON_DEF, pkeys
    )

    print(f"\nSaving cache to {CACHE_FILE}...")
    cache_vars = {
        'lons_1d': xr.DataArray(lons_1d, dims=['west_east']),
        'lats_1d': xr.DataArray(lats_1d, dims=['south_north']),
    }
    for season in SEASONS:
        for pkey, *_ in PERIODS:
            d = avg_grid[season][pkey]
            if d is not None:
                tag = f'{season}_{pkey}'
                for var in ['U10', 'V10', 'HGT', 'WS10']:
                    cache_vars[f'{var}_{tag}'] = d[var]
    xr.Dataset(cache_vars).to_netcdf(CACHE_FILE)
    print(f"  Saved → {CACHE_FILE}")

valley_coords = get_valley_coordinates(SHAPEFILE_PATH)
xsec_ep       = {idx: cross_section_endpoints(valley_coords, idx, CROSS_LEN_KM)
                 for idx in XSEC_INDICES}

# =====================================================
# DIAGNOSTIC — print data ranges to diagnose NaN issues
# =====================================================

_d = next((avg_grid[s][pk] for s in SEASONS for pk, *_ in PERIODS
           if avg_grid[s][pk] is not None), None)
if _d is not None:
    _h = _d.HGT.values
    _w = _d.WS10.values
    print(f"\n--- DIAGNOSTIC ---")
    print(f"HGT  shape={_h.shape}  min={np.nanmin(_h):.1f}  max={np.nanmax(_h):.1f}  NaN={np.isnan(_h).sum()}")
    print(f"WS10 shape={_w.shape}  min={np.nanmin(_w):.3f}  max={np.nanmax(_w):.3f}  NaN={np.isnan(_w).sum()}")
    print(f"lons_1d [{lons_1d.min():.2f} … {lons_1d.max():.2f}]")
    print(f"lats_1d [{lats_1d.min():.2f} … {lats_1d.max():.2f}]")
    print(f"Cells with HGT ≤ {MAX_TERRAIN} m : {(_h <= MAX_TERRAIN).sum()} / {_h.size}")
    print(f"------------------\n")

# =====================================================
# FIGURE — GridSpec(5, 2)
#   rows 0–3 : 4 season rows,  cols 0–1 : Day / Night
#   row 4    : wind-speed colorbar
# =====================================================

proj    = ccrs.PlateCarree()
norm_ws = plt.Normalize(WS_VMIN, WS_VMAX)

fig = plt.figure(figsize=(8, 10), layout='constrained')
gs  = fig.add_gridspec(
    5, 2,
    width_ratios  = [1, 1],
    height_ratios = [1, 1, 1, 1, 0.06],
    hspace=0.005,
    wspace=0.05,
)

ws_ref_holder = [None]
qv_ref_holder = [None]
axes_dict     = {}

for row, season in enumerate(SEASONS):
    for col, (pkey, plabel) in enumerate(PERIODS):

        d  = avg_grid[season][pkey]
        ax = fig.add_subplot(gs[row, col], projection=proj)
        axes_dict[(season, pkey)] = ax

        if d is None:
            ax.set_title(f"{season} – {plabel}\n(no data)", fontsize=11)
            continue

        show_left   = (col == 0)
        show_bottom = (row == len(SEASONS) - 1)

        ws_plot = draw_panel(
            ax, d, lons_1d, lats_1d, xsec_ep,
            ws_ref_holder, qv_ref_holder,
            show_left, show_bottom,
        )

        # Column header on top row; season label on left column
        if row == 0:
            ax.set_title(plabel, fontweight='bold', fontsize=12, pad=8)
        else:
            ax.set_title('')

        if col == 0:
            ax.text(-0.22, 0.5, season,
                    transform=ax.transAxes,
                    fontsize=13, fontweight='bold',
                    va='center', ha='center', rotation=90)

        print(f"  {season} {pkey} — WS10 (≤{MAX_TERRAIN} m) "
              f"min: {np.nanmin(ws_plot):.2f}  max: {np.nanmax(ws_plot):.2f} m s⁻¹")

# =====================================================
# COLORBAR — single horizontal bar at bottom
# =====================================================

cax_wind  = fig.add_subplot(gs[4, 0:2])
cbar_wind = plt.colorbar(ws_ref_holder[0], cax=cax_wind,
                         orientation='horizontal', extend='max')
cbar_wind.set_label("Wind Speed at 10 m a.g.l. (m s$^{-1}$)", fontsize=11)
cbar_wind.set_ticks(np.arange(WS_VMIN, WS_VMAX + 0.1, 1))
cbar_wind.ax.tick_params(labelsize=10)

# =====================================================
# LEGEND + QUIVERKEY (top-left panel)
# =====================================================

first_ax = axes_dict.get((SEASONS[0], PERIODS[0][0]))
if first_ax is not None:
    if SHOW_XSEC:
        legend_handles = [
            Line2D([0], [0], color='black', linewidth=2,
                   label=f'{XSEC_NAMES[0]} (A–B)'),
            Line2D([0], [0], color='black', linewidth=2,
                   label=f'{XSEC_NAMES[1]} (C–D)'),
        ]
        first_ax.legend(handles=legend_handles, loc='upper left',
                        fontsize=9, framealpha=0.95)

    if qv_ref_holder[0] is not None:
        qk = first_ax.quiverkey(
            qv_ref_holder[0], X=0.07, Y=1.1, U=2,
            label='  2 m s$^{-1}$', labelpos='E',
            fontproperties={'size': 9},
            coordinates='axes',
            zorder=20,
        )
        qk.text.set_backgroundcolor('white')

# =====================================================
# SAVE
# =====================================================

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved → {OUTPUT_FILE}")
