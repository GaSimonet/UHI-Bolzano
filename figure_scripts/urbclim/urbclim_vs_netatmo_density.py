#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/UrbClim_vs_netatmo/UrbClim_vs_netatmo_density_20260515.py
# Paper figure:       UrbClim_vs_Netatmo_exact_pixel_density.png
# Note: Unrelated to the valley-lapse-rate methodology -- included only because it feeds a paper figure. Copied as-is, no changes.
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-
"""
UrbClim vs NetAtmo — hexbin density plots with fraction-of-data contour overlay

Two colour modes (set COLOR_MODE at the top of CONFIG):

  'fraction' – CDF transform: each hex colour = fraction of data in hexes
               denser than this one.  Colorbar ticks and contour labels share
               the same scale (10 % … 90 %).  Dense core = dark, sparse = light.

  'density'  – Per-panel peak normalization: hex colour = count / panel_max.
               Colorbar ticks annotated at fraction-of-data positions computed
               from the ALL-data histogram.

Input
-----
  csv_exports/urbclim_netatmo_pairs.csv

Output
------
  figures/UrbClim_vs_Netatmo_exact_pixel_density.png
  figures/UrbClim_vs_Netatmo_surrounding_density.png
  figures/UrbClim_vs_Netatmo_combined_density.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

os.makedirs('figures', exist_ok=True)
# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_FILE              = '/home/gsimonet//Desktop/multi_source_UHI_Bolzano_package/UrbClim_vs_netatmo/csv_exports/urbclim_netatmo_pairs.csv'
DATE_START            = '2022-01-01'
DATE_END              = '2023-12-31'
SURROUNDING_RADIUS_KM = 0.3
HEXBIN_GRIDSIZE = 60
CMAP            = 'YlOrRd'

FRACTIONS    = [0.10, 0.25, 0.50, 0.75, 0.90]
NBINS        = 120
SMOOTH_SIGMA = 1.5

# ── Toggle ────────────────────────────────────────────────────────────────────
# 'fraction' : colour = fraction of data enclosed (colorbar ≡ contour labels)
# 'density'  : colour = count / panel peak  (0 → 100 % of panel max)
COLOR_MODE = 'fraction'
# ─────────────────────────────────────────────────────────────────────────────

SEASON_ORDER = ['DJF', 'MAM', 'JJA', 'SON', 'ALL']

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading pair CSV…")
df = pd.read_csv(CSV_FILE)
print(f"  {len(df)} pairs loaded")
print(df[['netatmo_T', 'ucb_exact', 'ucb_surr']].describe().round(3))

# ── DENSITY HELPERS ───────────────────────────────────────────────────────────

def _smooth_hist2d(x, y, tmin, tmax):
    """Smoothed 2-D histogram normalised to [0, 1] by panel peak."""
    H, xe, ye = np.histogram2d(x, y, bins=NBINS, range=[[tmin, tmax], [tmin, tmax]])
    H  = gaussian_filter(H.T, sigma=SMOOTH_SIGMA)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H / H.max(), xc, yc


def _smooth_hist2d_frac(x, y, tmin, tmax):
    """Smoothed 2-D histogram in fraction-of-data-enclosed space [0=dense, 1=empty]."""
    H, xe, ye = np.histogram2d(x, y, bins=NBINS, range=[[tmin, tmax], [tmin, tmax]])
    H  = gaussian_filter(H.T, sigma=SMOOTH_SIGMA)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    flat  = H.flatten()
    desc  = np.sort(flat)[::-1]
    csum  = np.cumsum(desc)
    total = csum[-1]
    if total > 0:
        idx    = np.searchsorted(-desc, -flat, side='left')
        idx    = np.clip(idx, 0, len(csum) - 1)
        H_frac = (csum[idx] / total).reshape(H.shape)
    else:
        H_frac = np.ones_like(H)
    return H_frac, xc, yc


def _fraction_levels(H, fractions):
    """Density thresholds enclosing each fraction; returns (levels, level→frac dict)."""
    flat  = H.flatten()
    desc  = np.sort(flat)[::-1]
    csum  = np.cumsum(desc)
    total = csum[-1]
    l2f   = {}
    for f in fractions:
        idx = int(np.searchsorted(csum, f * total))
        idx = min(idx, len(desc) - 1)
        lv  = float(desc[idx])
        if lv not in l2f or f > l2f[lv]:
            l2f[lv] = f
    return sorted(l2f), l2f


def _hexbin_to_frac(hb):
    """Transform hexbin counts in-place to fraction-of-data-enclosed space."""
    counts = hb.get_array().copy()
    desc   = np.sort(counts)[::-1]
    csum   = np.cumsum(desc)
    total  = csum[-1]
    if total > 0:
        idx = np.searchsorted(-desc, -counts, side='left')
        idx = np.clip(idx, 0, len(csum) - 1)
        hb.set_array(csum[idx] / total)
    hb.set_clim(0, 1)


def _colorbar(fig, mappable, cax, orientation, x_all, y_all):
    """Add colorbar with ticks appropriate for the current COLOR_MODE."""
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)

    if COLOR_MODE == 'fraction':
        ticks  = FRACTIONS
        labels = [f'{int(round(f * 100))} %' for f in FRACTIONS]
        label  = 'Fraction of data enclosed by contour'
    else:
        tmin = min(x_all.min(), y_all.min()) - 0.5
        tmax = max(x_all.max(), y_all.max()) + 0.5
        H, _, _ = _smooth_hist2d(x_all, y_all, tmin, tmax)
        lvs, l2f = _fraction_levels(H, FRACTIONS)
        ticks  = lvs
        labels = [f'{int(round(l2f[lv] * 100))} %' for lv in lvs]
        label  = 'Density (% of panel peak)  |  ticks = % of data enclosed'

    cb.set_ticks(ticks)
    if orientation == 'horizontal':
        cb.ax.set_xticklabels(labels, fontsize=8)
    else:
        cb.ax.set_yticklabels(labels, fontsize=8)
    cb.set_label(label, fontsize=8)
    return cb


# ── PANEL FUNCTION ────────────────────────────────────────────────────────────

def density_panel(ax, x, y, season_label, mode_label):
    """Draw one hexbin density panel with fraction-of-data contour overlay."""
    if len(x) < 10:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.set_title(season_label, fontsize=11, fontweight='bold')
        return None

    tmin = min(x.min(), y.min()) - 0.5
    tmax = max(x.max(), y.max()) + 0.5

    if COLOR_MODE == 'fraction':
        cmap_use = plt.get_cmap(CMAP).reversed()   # dense = dark
        hb = ax.hexbin(x, y, gridsize=HEXBIN_GRIDSIZE, cmap=cmap_use,
                       mincnt=1, linewidths=0.2)
        _hexbin_to_frac(hb)
        # Contour overlay in the same space
        H, xc, yc = _smooth_hist2d_frac(x, y, tmin, tmax)
        levels = list(FRACTIONS)
        fmt    = {f: f'{int(round(f * 100))} %' for f in FRACTIONS}
    else:  # 'density'
        cmap_use = plt.get_cmap(CMAP)              # dense = dark (high value)
        hb = ax.hexbin(x, y, gridsize=HEXBIN_GRIDSIZE, cmap=cmap_use,
                       mincnt=1, linewidths=0.2)
        counts = hb.get_array().copy()
        hb.set_array(counts / counts.max())
        hb.set_clim(0, 1)
        # Contour overlay — levels from per-panel fraction computation
        H, xc, yc       = _smooth_hist2d(x, y, tmin, tmax)
        levels, level_to_frac = _fraction_levels(H, FRACTIONS)
        if len(levels) < 2:
            levels = [0.01, 0.90]; level_to_frac = {0.01: 0.90, 0.90: 0.10}
        fmt = {lv: f'{int(round(level_to_frac[lv] * 100))} %'
               for lv in levels if lv in level_to_frac}

    if len(levels) >= 2:
        cs = ax.contour(xc, yc, H, levels=levels,
                        colors='k', linewidths=0.8, alpha=0.7, zorder=4)
        ax.clabel(cs, fmt=fmt, fontsize=6.5, inline=True, inline_spacing=4)

    ax.plot([tmin, tmax], [tmin, tmax], 'k--', lw=1.2, zorder=3)
    z    = np.polyfit(x, y, 1)
    xfit = np.linspace(tmin, tmax, 300)
    ax.plot(xfit, np.polyval(z, xfit), color='steelblue', lw=1.8, zorder=5)

    r    = float(np.corrcoef(x, y)[0, 1])
    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae  = float(np.mean(np.abs(y - x)))
    n    = len(x)
    stats_txt = (f'$y = {z[0]:.2f}x {z[1]:+.2f}$\n'
                 f'$r = {r:.3f}$\n'
                 f'RMSE $= {rmse:.2f}$ °C\n'
                 f'Bias $= {bias:+.2f}$ °C\n'
                 f'MAE $= {mae:.2f}$ °C')
    ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#cccccc'))
    ax.text(0.97, 0.04, f'$n = {n}$', transform=ax.transAxes,
            fontsize=8, va='bottom', ha='right', color='#444444')

    ax.set_xlim(tmin, tmax); ax.set_ylim(tmin, tmax)
    ax.set_aspect('equal')
    ax.set_xlabel('NetAtmo T (°C)', fontsize=9)
    ax.set_ylabel(f'UrbClim T2M — {mode_label} (°C)', fontsize=9)
    ax.set_title(season_label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, lw=0.6)
    return hb


def build_figure(df, y_col, mode_label, outfile):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    axes = axes.flatten()

    hb_last = None
    for idx, season in enumerate(SEASON_ORDER):
        ax  = axes[idx]
        sub = df if season == 'ALL' else df[df['season'] == season]
        hb  = density_panel(ax, sub['netatmo_T'].values, sub[y_col].values,
                            season, mode_label)
        if hb is not None: hb_last = hb

    ax_leg = axes[5]
    ax_leg.axis('off')

    # ax_leg.text(0.5, 0.65, 'Summary statistics', transform=ax_leg.transAxes,
    #             ha='center', va='bottom', fontsize=13, fontweight='bold')

    if hb_last is not None:
        pos     = ax_leg.get_position()
        cbar_ax = fig.add_axes([pos.x0 + pos.width * 0.1,
                                pos.y0 + pos.height * 0.68,
                                pos.width * 0.9, pos.height * 0.095])
        cb = _colorbar(fig, hb_last, cbar_ax, 'horizontal',
                       df['netatmo_T'].values, df[y_col].values)
        cb.ax.tick_params(labelsize=9)
        cb.ax.xaxis.label.set_fontsize(9.5)

    rows_tbl = []
    for season in SEASON_ORDER:
        sub = df if season == 'ALL' else df[df['season'] == season]
        if len(sub) < 2: continue
        x, y = sub['netatmo_T'].values, sub[y_col].values
        r    = float(np.corrcoef(x, y)[0, 1])
        bias = float(np.mean(y - x))
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        mae  = float(np.mean(np.abs(y - x)))
        rows_tbl.append([season, f'{len(sub)}', f'{r:.3f}',
                         f'{bias:+.2f}', f'{rmse:.2f}', f'{mae:.2f}'])

    tbl = ax_leg.table(cellText=rows_tbl,
                       colLabels=['Season', 'n', 'r', 'Bias (°C)', 'RMSE (°C)', 'MAE (°C)'],
                       cellLoc='center', loc='lower center',
                       bbox=[0.04, 0.03, 0.92, 0.58])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
        if row == 0:
            cell.set_facecolor('#DDDDDD'); cell.set_text_props(fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f7f7f7')

    radius_txt = (f'Surrounding radius: {SURROUNDING_RADIUS_KM} km\n'
                  if 'surrounding' in mode_label.lower() else '')
    # fig.suptitle(
    #     f'UrbClim T2M ({mode_label}) vs NetAtmo temperature — density [{COLOR_MODE} mode]\n'
    #     f'{DATE_START} → {DATE_END}  |  {radius_txt}'
    #     f'{df["na_idx"].nunique()} stations  |  {len(df)} pairs',
    #     fontsize=12, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig)
    print(f"✓ Saved: {outfile}")


# ── GENERATE FIGURES ──────────────────────────────────────────────────────────
print("\nGenerating density figures…")

build_figure(df, y_col='ucb_exact', mode_label='exact pixel',
             outfile='figures/UrbClim_vs_Netatmo_exact_pixel_density.png')

build_figure(df, y_col='ucb_surr',
             mode_label=f'surrounding ≤{SURROUNDING_RADIUS_KM} km mean',
             outfile='figures/UrbClim_vs_Netatmo_surrounding_density.png')

# ── COMBINED FIGURE ───────────────────────────────────────────────────────────
print("Generating combined density figure…")

fig, axes = plt.subplots(2, 5, figsize=(26, 11))
hb_ref = None
for col, season in enumerate(SEASON_ORDER):
    sub = df if season == 'ALL' else df[df['season'] == season]
    x   = sub['netatmo_T'].values
    hb  = density_panel(axes[0, col], x, sub['ucb_exact'].values,
                        season, 'exact pixel')
    density_panel(axes[1, col], x, sub['ucb_surr'].values,
                  season, f'surrounding ≤{SURROUNDING_RADIUS_KM} km')
    if hb is not None: hb_ref = hb

for row_i, lbl in enumerate(['Exact pixel', f'Surrounding ≤{SURROUNDING_RADIUS_KM} km']):
    axes[row_i, 0].annotate(lbl, xy=(-0.22, 0.5), xycoords='axes fraction',
                            fontsize=11, fontweight='bold', rotation=90,
                            ha='center', va='center')

if hb_ref is not None:
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    _colorbar(fig, hb_ref, cbar_ax, 'vertical',
              df['netatmo_T'].values, df['ucb_exact'].values)

fig.suptitle(
    f'UrbClim T2M vs NetAtmo — density [{COLOR_MODE} mode]\n'
    f'Exact pixel (top) / surrounding ≤{SURROUNDING_RADIUS_KM} km (bottom)  |  '
    f'{DATE_START} → {DATE_END}  |  '
    f'{df["na_idx"].nunique()} stations  |  {len(df)} pairs',
    fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.91, 0.95])
out_combined = 'figures/UrbClim_vs_Netatmo_combined_density.png'
plt.savefig(out_combined, dpi=200, bbox_inches='tight', facecolor='white')
plt.show(); plt.close(fig)
print(f"✓ Saved: {out_combined}")

print("\nDone.")
