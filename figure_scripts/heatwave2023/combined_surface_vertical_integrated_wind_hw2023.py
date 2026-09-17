#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: WRF_alto_adige_local/script/analysis/combined_surface_vertical_integrated_wind_HW2023_20260727.py
# Paper figure:       HW2023_combined_surface_vertical_integrated_wind.png
# Note: Unrelated to the valley-lapse-rate methodology (no urban/rural UHI correction here) -- included only because it feeds a paper figure.
# UPDATE (2026-08-04): the daytime/nighttime split previously used a fixed DAYTIME_HOURS=range(8,20)
# clock-hour window -- switched to the same astronomical sunrise/sunset classification
# (common/solar_daynight.classify_day_night) used across the UHI scripts, and panel titles no
# longer print a specific clock-hour range (e.g. "Daytime (08-19)") since the window itself now
# varies by calendar day.
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-

"""
HW2023 – Combined Figure: Surface 0-500m Integrated Wind Arrows + Vertical Cross-Sections
6 panels (a-f):
  a, b : surface 0-500 m AGL layer-mean wind (arrows) + topography (Daytime / Nighttime)
  c, d : vertical cross-section Bolzano-north (Daytime / Nighttime)
  e, f : vertical cross-section Bolzano-south (Daytime / Nighttime)

Difference vs combined_surface_vertical_HW2023_20260302.py:
  Panels a,b no longer show 10 m streamlines. Instead the near-surface wind
  is layer-thickness-weighted averaged between the ground and 500 m AGL
  (per grid column, using local terrain height) and drawn as quiver arrows.
  This represents the wind that actually carries air in/out of the valley,
  rather than the shallow 10 m diagnostic level alone.

Note on wind rotation: like the original script, U/V (destaggered, salem-
provided) are used as-is for the surface map, without an explicit
COSALPHA/SINALPHA earth-rotation correction. The domain is small and close
to the LCC projection's central meridian (lon_0=11, Bolzano at 11.32), so
grid-relative and true-north-relative directions differ by a fraction of a
degree here -- the same approximation the original script already makes for
U10/V10 and for the cross-section rotation (uvrot2wrf).

Panel style matches Horizontal_X_section/surface_wind_arrows_low_terrain_
HW_20260302.py (the paper figure): 4 layers per panel —
  1. grey terrain relief, full domain
  2. orange->red wind-speed fill, terrain <= MAX_TERRAIN only
  3. terrain contour lines, full domain
  4. black arrows, terrain <= MAX_TERRAIN only
fed by the 0-500 m layer-mean wind instead of the paper's 10 m U10/V10.

Domain: zoomed to a ZOOM_HALF_GP*2 x ZOOM_HALF_GP*2 grid-point window
centered on the model grid point nearest Bolzano (~90x90 grid points, i.e.
~90x90 km at this domain's ~1 km spacing) rather than the full WRF subset
used previously -- wide enough to see the valley-scale circulation, tighter
than the full domain so it isn't lost among far-away terrain. The window is
selected by index (isel), not a lat/lon box, so it stays exactly
ZOOM_HALF_GP grid points wide regardless of projection distortion.

Cell layout (Spyder/Jupyter #%% blocks): the LOAD DATA cell does every
expensive step once (opening the 11 GB file, building cross-sections,
computing the 0-500 m layer-mean wind) and stores results in plain
variables. The FIGURE cell only reads from those variables, so replotting
with different colors/labels/skip factors does not require rerunning the
load cell.
"""

#%%
# =====================================================
# IMPORTS, SETTINGS & HELPERS
# =====================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import salem
import geopandas as gpd

from mygaspypack.wrfutils import uvrot2wrf, latlon2xy, intpxsecv2

# Astronomical day/night split (real sunrise/sunset per calendar day, Bolzano),
# replacing the old fixed DAYTIME_HOURS=range(8,20) clock-hour window -- matches
# the convention used across the UHI scripts (common/solar_daynight.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from solar_daynight import classify_day_night

plt.rcParams['font.family'] = 'DejaVu Sans'

# =====================================================
# USER SETTINGS
# =====================================================

WRF_DIR   = "/home/gsimonet/Desktop/WRF_ALTO_ADIGE_on_CEPH/HW_202308_13_26_subset"
WRF_FILE  = "HW_WRF_202308_13_26.nc"

SHAPEFILE_PATH = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/valley_x_sec_along_river_bolzano_v2.dbf"

XSEC_INDICES   = [50, 120]
XSEC_LABELS    = ["Bolzano-north", "Bolzano-south"]
XSEC_AB_LABELS = [('A', 'B'), ('C', 'D')]
CROSS_LEN_KM   = 15

BOLZANO = (46.47, 11.32)

ZOOM_HALF_GP = 45   # grid points each direction from Bolzano -> ~90x90 GP window

XLIM     = [-15, 15]
CROSSNUM = 400
YLIM     = [0, 3200]

VROT_MIN = -6
VROT_MAX = 6

THETA_LEVELS   = np.arange(298, 320, 0.5)
THETA_COLOR    = 'black'
THETA_LW       = 0.8
THETA_LABEL_FS = 9

QUIVER_SKIP_X = 10
QUIVER_SKIP_Z = 6
W_SCALE       = 1
QUIVER_SCALE  = 30
QUIVER_WIDTH  = 0.005

# ---- 0-500 m integrated-wind panel settings ----
INTWIND_ZMIN = 0.0     # m AGL
INTWIND_ZMAX = 500.0   # m AGL

# ---- Surface arrow panel style (paper style, matches
# surface_wind_arrows_low_terrain_HW_20260302.py) ----
MAX_TERRAIN  = 1500   # mask wind fill/arrows above this elevation (m)
MAX_TOPO     = 3600   # upper bound for grey terrain background (m)
ARROW_STRIDE = 2      # grid subsampling for quiver (zoomed domain -> denser than the old wide-domain 3)
ARROW_SCALE  = 50     # quiver scale_units='width' (lower = longer arrows)
WS_VMIN      = 0.0
WS_VMAX      = 5.0    # wind-speed colorbar / contourf upper bound (m/s)

WIND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'orange_red', ['#FEE8C8', '#FDBB84', '#FC8D59', '#E34A33', '#B30000']
)

# =====================================================
# HELPERS
# =====================================================

def get_valley_coordinates(shapefile_path, spacing_km=0.5):
    valley = gpd.read_file(shapefile_path)
    line   = valley.geometry.iloc[0]
    spacing_deg = spacing_km / 111
    n_points    = int(line.length / spacing_deg)
    coords = []
    for i in range(n_points):
        pt = line.interpolate(i * spacing_deg)
        coords.append((pt.y, pt.x))
    return coords


def cross_section_endpoints(coords, idx, half_len_km=25):
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
    valley_dir_m = ((90 - valley_angle) + 180) % 360
    cross_dir_m  = (valley_dir_m - 90) % 360

    cross_rad = np.radians(90 - cross_dir_m)
    dx_c = np.cos(cross_rad)
    dy_c = np.sin(cross_rad)

    lat_scale = 1 / 111
    lon_scale = 1 / (111 * np.cos(np.radians(lat)))

    return (
        (lat - dy_c * half_len_km * lat_scale,
         lon - dx_c * half_len_km * lon_scale),
        (lat + dy_c * half_len_km * lat_scale,
         lon + dx_c * half_len_km * lon_scale)
    )


def build_cross_section(ds, valley_coords, valley_index):
    lat, lon = valley_coords[valley_index]
    next_lat, next_lon = valley_coords[valley_index + 1]
    dx = next_lon - lon
    dy = next_lat - lat
    valley_angle = np.degrees(np.arctan2(dy, dx))
    valley_dir_m = ((90 - valley_angle) + 180) % 360
    cross_dir_m  = (valley_dir_m - 90) % 360
    print(f"  Valley dir: {valley_dir_m:.1f}°  |  Cross dir: {cross_dir_m:.1f}°")
    x0, y0, _ = latlon2xy(ds, lat, lon)
    ds_rot = ds.copy()
    uvrot2wrf(ds_rot, cross_dir_m)
    dsi = intpxsecv2(ds_rot, x0, y0, cross_dir_m, xlim=XLIM, crossnum=CROSSNUM)
    return dsi


def interp_to_regular_grid(dtmp, var, x, z_levels):
    z  = dtmp['Z'].values
    nx = len(x)
    nz = len(z_levels)
    out = np.zeros((nz, nx))
    for j in range(nx):
        zcol = z[:, j]
        vcol = dtmp[var][:, j].values
        idx  = np.argsort(zcol)
        f    = interp1d(zcol[idx], vcol[idx], bounds_error=False, fill_value=np.nan)
        out[:, j] = f(z_levels)
    return out


def mask_terrain(arr, z_levels, terrain_interp):
    for j in range(arr.shape[1]):
        arr[z_levels < terrain_interp[j], j] = np.nan
    return arr


def layer_mean_agl(field, z_agl, zmin=0.0, zmax=500.0):
    """
    Thickness-weighted mean of `field` over the AGL layer [zmin, zmax].

    field, z_agl : ndarray, shape (bottom_top, south_north, west_east)
                   z_agl = height of each level's midpoint above LOCAL ground
                   (i.e. Z - HGT), increasing with the bottom_top index.
    Returns      : ndarray, shape (south_north, west_east)

    Layer boundaries are placed at the midpoints between consecutive level
    heights (interfaces), with the surface as the bottom boundary of level 0.
    Each level's weight is the overlap of its [lower, upper] interface with
    [zmin, zmax], so a partial layer at the 500 m cutoff is weighted
    proportionally rather than included/excluded wholesale.
    """
    mid_interfaces = 0.5 * (z_agl[:-1] + z_agl[1:])   # (nz-1, ny, nx)

    lower = np.empty_like(z_agl)
    upper = np.empty_like(z_agl)
    lower[0]   = 0.0
    lower[1:]  = mid_interfaces
    upper[:-1] = mid_interfaces
    upper[-1]  = z_agl[-1] + (z_agl[-1] - z_agl[-2])   # extrapolated top interface

    weight = np.clip(np.minimum(upper, zmax) - np.maximum(lower, zmin), 0.0, None)
    weight_sum = weight.sum(axis=0)

    return np.sum(field * weight, axis=0) / weight_sum


def plot_vertical_panel(ax, dtmp, show_xlabel=False, show_ylabel=True, ab_labels=('A', 'B')):
    """Plot one vertical cross-section panel. Returns (cf, qv)."""

    z       = dtmp['Z'].values
    x       = dtmp['xcross'].values
    terrain = dtmp['HGT'].values

    x_regular      = np.linspace(XLIM[0], XLIM[1], len(x))
    z_levels       = np.linspace(YLIM[0], YLIM[1], 120)
    terrain_interp = np.interp(x_regular, x, terrain)

    X = np.tile(x_regular, (z.shape[0], 1))

    cf = ax.contourf(
        X, z, dtmp['Vrot'].values,
        levels=np.linspace(VROT_MIN, VROT_MAX, 25),
        cmap='PuOr_r',
        extend='both'
    )

    theta = dtmp['T'].values + 300.0
    iso = ax.contour(
        X, z, theta,
        levels=THETA_LEVELS,
        colors=THETA_COLOR,
        linewidths=THETA_LW,
        linestyles='solid'
    )
    ax.clabel(iso, inline=True, fontsize=THETA_LABEL_FS, fmt='%.1f K')

    Urot_reg = mask_terrain(
        interp_to_regular_grid(dtmp, 'Urot', x, z_levels), z_levels, terrain_interp)
    W_reg    = mask_terrain(
        interp_to_regular_grid(dtmp, 'W',    x, z_levels), z_levels, terrain_interp)

    xs = x_regular[::QUIVER_SKIP_X]
    zs = z_levels [::QUIVER_SKIP_Z]
    Us = Urot_reg [::QUIVER_SKIP_Z, ::QUIVER_SKIP_X]
    Ws = W_reg    [::QUIVER_SKIP_Z, ::QUIVER_SKIP_X] * W_SCALE

    XQ, ZQ = np.meshgrid(xs, zs)
    qv = ax.quiver(
        XQ, ZQ, Us, Ws,
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH,
        color='dimgray',
        alpha=0.75,
        zorder=5
    )

    ax.set_ylim(YLIM)
    ax.set_xlim(XLIM)

    ax.fill_between(x_regular, YLIM[0], terrain, color='black', zorder=10)

    ax.text(x_regular[0]  + 0.2, terrain[0]  - 200, ab_labels[0],
            color='yellow', fontweight='bold', fontsize=12, ha='left',  zorder=11)
    ax.text(x_regular[-1] - 0.2, terrain[-1] - 200, ab_labels[1],
            color='yellow', fontweight='bold', fontsize=12, ha='right', zorder=11)

    xticks = np.linspace(XLIM[0], XLIM[1], 7)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(v * 1000)}" for v in xticks])

    if show_ylabel:
        ax.set_ylabel("Height (m a.s.l.)", fontsize=11)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("Distance from valley axis (m)", fontsize=11)

    return cf, qv


def grid_point_window(ds, lat0, lon0, half_gp):
    """Isel window of (2*half_gp) x (2*half_gp) grid points centered on the
    model grid point nearest (lat0, lon0). Index-based (not a lat/lon box),
    so it stays exactly the requested size regardless of projection distortion."""
    lat2d = ds.lat.values
    lon2d = ds.lon.values
    dist2 = (lat2d - lat0)**2 + (lon2d - lon0)**2
    iy0, ix0 = np.unravel_index(np.argmin(dist2), dist2.shape)
    ny, nx = lat2d.shape
    iy_lo, iy_hi = max(0, iy0 - half_gp), min(ny, iy0 + half_gp)
    ix_lo, ix_hi = max(0, ix0 - half_gp), min(nx, ix0 + half_gp)
    return ds.isel(south_north=slice(iy_lo, iy_hi), west_east=slice(ix_lo, ix_hi))


def add_panel_label(ax, letter):
    ax.text(0.02, 0.97, f'({letter})',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.3', linewidth=1.0),
            zorder=20)


#%%
# =====================================================
# LOAD DATA  (run once — skip re-running this cell when only tweaking the plot)
# =====================================================

print("Opening dataset...")
ds_full = salem.open_wrf_dataset(f"{WRF_DIR}/{WRF_FILE}")
ds_full = grid_point_window(ds_full, BOLZANO[0], BOLZANO[1], ZOOM_HALF_GP)
print(f"  Zoomed domain: {ds_full.sizes['south_north']} x {ds_full.sizes['west_east']} "
      f"grid points around Bolzano")

valley_coords = get_valley_coordinates(SHAPEFILE_PATH)

periods_sfc = classify_day_night(ds_full.time.values)   # WRF is UTC -- no offset
day_mask    = periods_sfc == 'day'
night_mask  = ~day_mask
time_titles = ["Daytime", "Nighttime"]

xsec_endpoints = {idx: cross_section_endpoints(valley_coords, idx, CROSS_LEN_KM)
                  for idx in XSEC_INDICES}

# ---- 0-500 m AGL layer-mean wind (drives panels a, b) ----
print("\nComputing 0-500 m AGL layer-mean wind...")
ds_wind3d = ds_full[['U', 'V', 'Z', 'HGT']]

wind3d_day   = ds_wind3d.isel(time=day_mask).mean("time")
wind3d_night = ds_wind3d.isel(time=night_mask).mean("time")

sfc_datasets = []
for d in [wind3d_day, wind3d_night]:
    HGT_np = d['HGT'].values                       # reused below for both z_agl and the topo background
    z_agl  = d['Z'].values - HGT_np[None, :, :]     # computed once, reused for U_int and V_int
    U_int  = layer_mean_agl(d['U'].values, z_agl, INTWIND_ZMIN, INTWIND_ZMAX)
    V_int  = layer_mean_agl(d['V'].values, z_agl, INTWIND_ZMIN, INTWIND_ZMAX)
    WS_int = np.sqrt(U_int**2 + V_int**2)

    entry = d[['HGT']].copy()
    entry['U_int']  = (('south_north', 'west_east'), U_int)
    entry['V_int']  = (('south_north', 'west_east'), V_int)
    entry['WS_int'] = (('south_north', 'west_east'), WS_int)
    sfc_datasets.append(entry)

# ---- Vertical cross-section data ----
ds_vert = ds_full[['U', 'V', 'W', 'Z', 'HGT', 'T']]

all_vert_datasets = []
for vi in XSEC_INDICES:
    print(f"\nBuilding cross-section at index {vi}...")
    dsi        = build_cross_section(ds_vert, valley_coords, vi)
    periods    = classify_day_night(dsi.time.values)   # WRF is UTC -- no offset
    dmask      = periods == 'day'
    nmask      = ~dmask
    all_vert_datasets.append([
        dsi.isel(time=dmask).mean("time"),
        dsi.isel(time=nmask).mean("time")
    ])

 #%%
# =====================================================
# FIGURE  — GridSpec(4, 3)  (uses only the variables computed above — safe to
# rerun this cell repeatedly while tuning colors/skip/labels)
#   row 0      : 2 surface map panels (a, b) + topo cbar
#   row 1      : wind speed cbar spanning cols 0-1
#   row 2      : Bolzano-north vertical panels (c, d)
#   row 3      : Bolzano-south vertical panels (e, f) + Vrot cbar spanning rows 2-3
# =====================================================

proj    = ccrs.PlateCarree()
norm_ws = plt.Normalize(WS_VMIN, WS_VMAX)

fig = plt.figure(figsize=(13, 14))
gs  = GridSpec(
    4, 3,
    width_ratios=[1, 1, 0.04],
    height_ratios=[1, 0.06, 0.85, 0.85],
    hspace=0.35,
    wspace=0.15
)

# ---- Row 0: surface panels ----
ws_ref     = None
qv_ref_sfc = None
map_axes   = []
xsection_names = ['Cross_section_north', 'Cross_section_south']

for col, (dtmp, title) in enumerate(zip(sfc_datasets, time_titles)):

    ax = fig.add_subplot(gs[0, col], projection=proj)
    map_axes.append(ax)

    lons    = dtmp.lon.values
    lats    = dtmp.lat.values
    lons_1d = lons[0, :] if lons.ndim == 2 else lons
    lats_1d = lats[:, 0] if lats.ndim == 2 else lats

    terrain = dtmp.HGT.values
    u_wind  = dtmp.U_int.values
    v_wind  = dtmp.V_int.values
    ws      = dtmp.WS_int.values

    low_mask = terrain <= MAX_TERRAIN
    u_plot   = np.where(low_mask, u_wind, np.nan)
    v_plot   = np.where(low_mask, v_wind, np.nan)
    ws_plot  = np.where(low_mask, ws,     np.nan)

    ax.set_extent([lons_1d.min(), lons_1d.max(),
                   lats_1d.min(), lats_1d.max()], crs=proj)
    ax.set_aspect('auto')

    ax.add_feature(cfeature.BORDERS,  linewidth=0.5)
    ax.add_feature(cfeature.RIVERS,   linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.LAKES,    alpha=0.5)

    # ---- layer 1 : grey terrain relief — full domain (incl. > MAX_TERRAIN) ----
    ax.contourf(
        lons_1d, lats_1d, terrain,
        levels=np.linspace(0, MAX_TOPO, 35),
        cmap='Greys', alpha=0.55,
        transform=proj, extend='neither',
    )

    # ---- layer 2 : wind speed fill (orange -> red) — <= MAX_TERRAIN only ----
    ws_cf = ax.contourf(
        lons_1d, lats_1d,
        np.ma.masked_invalid(ws_plot),
        levels=np.linspace(WS_VMIN, WS_VMAX, 20),
        cmap=WIND_CMAP, norm=norm_ws, alpha=0.75,
        transform=proj, extend='max',
    )
    if ws_ref is None:
        ws_ref = ws_cf

    # ---- layer 3 : terrain contour lines — full domain ----
    ax.contour(
        lons_1d, lats_1d, terrain,
        levels=np.arange(0, MAX_TOPO + 1, 200),
        colors='#444444', linewidths=0.5, alpha=0.55,
        transform=proj,
    )

    # ---- layer 4 : black arrows (length ∝ wind speed) — <= MAX_TERRAIN only ----
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
    if qv_ref_sfc is None:
        qv_ref_sfc = qv

    for k, (idx, name) in enumerate(zip(XSEC_INDICES, xsection_names)):
        pt_A, pt_B = xsec_endpoints[idx]
        lbl_B, lbl_A = XSEC_AB_LABELS[k]

        ax.plot([pt_A[1], pt_B[1]], [pt_A[0], pt_B[0]],
                color='black', linewidth=2, transform=proj, zorder=10)

        for pt, lbl in [(pt_A, lbl_A), (pt_B, lbl_B)]:
            ax.text(pt[1], pt[0], lbl,
                    fontsize=10, fontweight='bold', color='white',
                    ha='center', va='center', transform=proj, zorder=11,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

        lat_c, lon_c = valley_coords[idx]
        ax.scatter(lon_c, lat_c, c='black', s=40, zorder=12, transform=proj)

    ax.scatter(BOLZANO[1], BOLZANO[0], c='white', s=220, marker='*',
               edgecolors='black', linewidths=0.5, zorder=15, transform=proj)
    ax.text(BOLZANO[1] + 0.035, BOLZANO[0] + 0.035, "Bolzano",
            fontsize=9, fontweight='bold', color='white', transform=proj,
            zorder=20,
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = (col == 0)
    gl.bottom_labels = True

    ax.set_title(title, fontweight='bold', fontsize=12)
    add_panel_label(ax, chr(ord('a') + col))

    print(f"{title} — 0-500m WS (≤{MAX_TERRAIN} m) min: {np.nanmin(ws_plot):.2f}  max: {np.nanmax(ws_plot):.2f} m/s")

cax_wind  = fig.add_subplot(gs[1, 0:2])
cbar_wind = plt.colorbar(ws_ref, cax=cax_wind, orientation='horizontal', extend='max')
cbar_wind.set_label(f"{INTWIND_ZMIN:.0f}–{INTWIND_ZMAX:.0f} m AGL Layer-Mean Wind Speed (m s$^{-1}$)", fontsize=10)
cbar_wind.set_ticks(np.arange(WS_VMIN, WS_VMAX + 0.1, 1))

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label=f'{xsection_names[0]} (A–B)'),
    Line2D([0], [0], color='black', linewidth=2, label=f'{xsection_names[1]} (C–D)'),
]
map_axes[0].legend(handles=legend_elements, loc='upper left',
                    fontsize=9, framealpha=0.95)

qk = map_axes[0].quiverkey(
    qv_ref_sfc, X=0.09, Y=1.04, U=2,
    label='  2 m s$^{-1}$', labelpos='E',
    fontproperties={'size': 9},
    coordinates='axes',
    zorder=20,
)
qk.text.set_backgroundcolor('white')

# ---- Rows 2-3: vertical cross-section panels ----
cf_ref = None
qv_ref = None
panel_letters = ['c', 'd', 'e', 'f']

for row, (vert_datasets, xsec_label, ab) in enumerate(zip(all_vert_datasets, XSEC_LABELS, XSEC_AB_LABELS)):
    for col, (dtmp, time_title) in enumerate(zip(vert_datasets, time_titles)):

        ax = fig.add_subplot(gs[2 + row, col])

        cf, qv = plot_vertical_panel(
            ax, dtmp,
            show_xlabel=(row == 1),
            show_ylabel=(col == 0),
            ab_labels=ab
        )

        plt.quiverkey(qv, X=0.88, Y=1.04, U=5,
                      label='5 m s$^{-1}$',
                      labelpos='E',
                      coordinates='axes',
                      fontproperties={'size': 10},
                      zorder=10)

        letter = panel_letters[row * 2 + col]
        add_panel_label(ax, letter)

        if cf_ref is None: cf_ref = cf
        if qv_ref is None: qv_ref = qv

        if row == 0:
            ax.set_title(time_title, fontweight='bold', fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{xsec_label}\nHeight (m)", fontsize=10)

        print(f"{xsec_label} | {time_title} — "
              f"Vrot min: {dtmp['Vrot'].values.min():.2f}  "
              f"max: {dtmp['Vrot'].values.max():.2f} m/s")

cax_vrot  = fig.add_subplot(gs[2:4, 2])
cbar_vrot = plt.colorbar(cf_ref, cax=cax_vrot)
cbar_vrot.set_label("Along-Valley Wind (m s$^{-1}$)", fontsize=10)

# =====================================================
# SAVE
# =====================================================

plt.savefig("HW2023_combined_surface_vertical_integrated_wind.png", dpi=300, bbox_inches='tight')
plt.show()
print("✓ DONE")
