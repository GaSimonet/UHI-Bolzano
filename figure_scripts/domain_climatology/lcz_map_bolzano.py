#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/LCZ_map_bolzano_20260604.py
# Paper figure:       LCZ_map_city_combined_20260604.png (right panel)
# Note: Produces the right 'LCZ classification + area bar chart' panel of a MANUALLY composited figure -- see bz_city_map.py in this same folder for the left panel. Not to be confused with suhi_lcz/suhi_lcz_analysis.py, which is a different script producing a different figure (suhi_lcz_maps_seasonal.pdf).
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-
"""
LCZ Map — Bolzano
Two-panel figure:
  Left  : OpenTopoMap basemap + LCZ raster overlay + urban & rural boundaries
  Right : Horizontal bar chart — area (km²) per LCZ class
"""

#%% ── IMPORTS ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import rasterio.crs
from rasterio.warp import reproject, Resampling, calculate_default_transform
import contextily as ctx
import os

#%% ── CONFIGURATION ──────────────────────────────────────────────────────────

LCZ_RASTER   = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/LCZ_clipped.tif"
URBAN_SHAPE  = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area_WGS84.shp"
RURAL_SHAPE  = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area.shp"
OUTPUT_DIR   = "."
DPI          = 300

PIXEL_AREA_KM2  = None   # auto-computed from raster resolution
MIN_LCZ_PIXELS  = 50    # discard LCZ classes with fewer native pixels in the domain

# Standard WUDAPT LCZ definitions
LCZ_NAMES = {
    1:  "Compact highrise",    2:  "Compact midrise",    3:  "Compact lowrise",
    4:  "Open highrise",       5:  "Open midrise",       6:  "Open lowrise",
    7:  "Lightweight lowrise", 8:  "Large lowrise",      9:  "Sparsely built",
    10: "Heavy industry",      11: "Dense trees",        12: "Scattered trees",
    13: "Bush / scrub",        14: "Low plants",         15: "Bare rock / paved",
    16: "Bare soil / sand",    17: "Water",
}

LCZ_COLORS = {
    1:  "#8B0000", 2:  "#CD0000", 3:  "#FF0000",  4:  "#FF6347",  5:  "#FF8C00",
    6:  "#FFD700", 7:  "#FFFF99", 8:  "#BEBEBE",  9:  "#696969",  10: "#8B4513",
    11: "#006400", 12: "#228B22", 13: "#90EE90",  14: "#FFFF00",
    15: "#A9A9A9", 16: "#F5DEB3", 17: "#4169E1",
}

# Urban boundary style
URBAN_STYLE = dict(color="#1a1a1a", linewidth=1.4, linestyle="-",  zorder=5)
RURAL_STYLE = dict(color="#1a1a1a", linewidth=1.1, linestyle="--", zorder=5)


#%% ── LOAD LCZ RASTER ────────────────────────────────────────────────────────

print("Loading LCZ raster…")
with rasterio.open(LCZ_RASTER) as src:
    lcz_data_src  = src.read(1).astype(np.float32)
    lcz_src_crs   = src.crs
    lcz_src_trans = src.transform
    lcz_nodata    = src.nodata if src.nodata is not None else 0

    # native pixel area for the area chart
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    PIXEL_AREA_KM2 = (res_x * res_y) / 1e6

    # reproject to Web Mercator for display
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src.crs, "EPSG:3857", src.width, src.height, *src.bounds
    )

lcz_wm = np.zeros((dst_h, dst_w), dtype=np.float32)
reproject(
    source=lcz_data_src,
    destination=lcz_wm,
    src_transform=lcz_src_trans,
    src_crs=lcz_src_crs,
    dst_transform=dst_transform,
    dst_crs=rasterio.crs.CRS.from_epsg(3857),
    resampling=Resampling.nearest,
    src_nodata=lcz_nodata,
    dst_nodata=0,
)

lcz_bounds_wm = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
lcz_extent_wm = [lcz_bounds_wm[0], lcz_bounds_wm[2],
                  lcz_bounds_wm[1], lcz_bounds_wm[3]]   # (left, right, bottom, top)

# Mask nodata
lcz_disp = lcz_wm.astype(float)
lcz_disp[lcz_disp == 0] = np.nan

lcz_int_src = lcz_data_src.astype(np.int16)
present_lcz = sorted([int(v) for v in np.unique(lcz_int_src)
                       if v > 0 and int(v) in LCZ_NAMES
                       and int(np.sum(lcz_int_src == v)) >= MIN_LCZ_PIXELS])
print(f"  LCZ classes present (≥{MIN_LCZ_PIXELS} px): {present_lcz}")
print(f"  Pixel area: {PIXEL_AREA_KM2*1e6:.0f} m²  ({PIXEL_AREA_KM2:.4f} km²)")


#%% ── LCZ COLOURMAP ──────────────────────────────────────────────────────────

all_lcz_vals  = sorted(LCZ_COLORS.keys())
lcz_cmap      = ListedColormap([LCZ_COLORS[v] for v in all_lcz_vals], name="lcz")
lcz_boundaries = [v - 0.5 for v in all_lcz_vals] + [all_lcz_vals[-1] + 0.5]
lcz_bnorm     = BoundaryNorm(lcz_boundaries, lcz_cmap.N)


#%% ── LOAD SHAPEFILES ────────────────────────────────────────────────────────

print("Loading shapefiles…")
urban_wm  = gpd.read_file(URBAN_SHAPE).to_crs("EPSG:3857")
rural_wm  = gpd.read_file(RURAL_SHAPE).to_crs("EPSG:3857")
urban_lcz = gpd.read_file(URBAN_SHAPE).to_crs(lcz_src_crs)
rural_lcz = gpd.read_file(RURAL_SHAPE).to_crs(lcz_src_crs)

# Cropped map extent: trim 8 % from each side of the LCZ raster bounds
_dx = (lcz_bounds_wm[2] - lcz_bounds_wm[0]) * 0.08
_dy = (lcz_bounds_wm[3] - lcz_bounds_wm[1]) * 0.08
bx = (
    lcz_bounds_wm[0] + _dx,
    lcz_bounds_wm[1] + _dy,
    lcz_bounds_wm[2] - _dx,
    lcz_bounds_wm[3] - _dy,
)   # (minx, miny, maxx, maxy) in Web Mercator


#%% ── URBAN / RURAL MASKS ON LCZ GRID ───────────────────────────────────────

from rasterio.features import rasterize
from rasterio.transform import from_bounds as transform_from_bounds
from shapely.geometry import mapping

lcz_h, lcz_w = lcz_data_src.shape

# Reconstruct the native raster transform from the source metadata
with rasterio.open(LCZ_RASTER) as src:
    native_transform = src.transform

def _rasterize_gdf(gdf, shape, transform):
    geoms = [mapping(g) for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(geoms, out_shape=shape, transform=transform,
                       fill=0, default_value=1, dtype=np.uint8)
    return burned.astype(bool)

print("Building urban / rural masks on LCZ grid…")
urban_mask = _rasterize_gdf(urban_lcz, (lcz_h, lcz_w), native_transform)
rural_mask = _rasterize_gdf(rural_lcz, (lcz_h, lcz_w), native_transform)
print(f"  Urban pixels: {urban_mask.sum():,}   Rural pixels: {rural_mask.sum():,}")


#%% ── AREA PER LCZ CLASS (URBAN / RURAL SPLIT) ───────────────────────────────

lcz_area_urban = {}
lcz_area_rural = {}
for cls in present_lcz:
    cls_mask = lcz_int_src == cls
    lcz_area_urban[cls] = int(np.sum(cls_mask & urban_mask)) * PIXEL_AREA_KM2
    lcz_area_rural[cls] = int(np.sum(cls_mask & rural_mask)) * PIXEL_AREA_KM2

# Sort by total area descending
lcz_area_total = {c: lcz_area_urban[c] + lcz_area_rural[c] for c in present_lcz}
cls_sorted   = sorted(present_lcz, key=lambda c: lcz_area_total[c], reverse=True)
names_sorted = [f"LCZ {c} — {LCZ_NAMES[c]}" for c in cls_sorted]
cols_sorted  = [LCZ_COLORS.get(c, "#888888") for c in cls_sorted]


#%% ── FIGURE ─────────────────────────────────────────────────────────────────

print("Rendering figure…")

fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[2.4, 1],
    wspace=0.06,
)
ax_map  = fig.add_subplot(gs[0])
ax_bar  = fig.add_subplot(gs[1])


# ── MAP PANEL ─────────────────────────────────────────────────────────────────

ax_map.set_xlim(bx[0], bx[2])
ax_map.set_ylim(bx[1], bx[3])   # cropped extent

ctx.add_basemap(
    ax_map, crs="EPSG:3857",
    source="https://tile.opentopomap.org/{z}/{x}/{y}.png",
    zoom="auto",
    attribution="© OpenTopoMap (CC-BY-SA)",
    attribution_size=6, zorder=1,
)

ax_map.imshow(
    lcz_disp,
    cmap=lcz_cmap, norm=lcz_bnorm,
    extent=lcz_extent_wm,
    origin="upper", alpha=0.60, zorder=2,
)

urban_wm.boundary.plot(ax=ax_map, **URBAN_STYLE)
rural_wm.boundary.plot(ax=ax_map, **RURAL_STYLE)

ax_map.set_xticks([]); ax_map.set_yticks([])
ax_map.set_title("Local Climate Zones — Bolzano", fontsize=13, fontweight="bold", pad=8)

# legend: zone boundaries + LCZ colours
legend_handles = [
    mpatches.Patch(facecolor="none", edgecolor="#1a1a1a", linewidth=1.4,
                   label="Urban area"),
    mpatches.Patch(facecolor="none", edgecolor="#1a1a1a", linewidth=1.1,
                   linestyle="--", label="Rural area"),
]
for cls in present_lcz:
    legend_handles.append(
        mpatches.Patch(facecolor=LCZ_COLORS.get(cls, "#888888"), edgecolor="none",
                       alpha=0.85, label=f"LCZ {cls} — {LCZ_NAMES.get(cls, '?')}")
    )

ax_map.legend(
    handles=legend_handles,
    loc="lower left", bbox_to_anchor=(0.01, 0.01),
    fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
    handlelength=1.3, handletextpad=0.5, labelspacing=0.45,
)


# ── BAR PANEL — urban / rural area per LCZ class ─────────────────────────────

n_cls   = len(cls_sorted)
bar_h   = 0.35
y_pos   = np.arange(n_cls)

urban_areas = [lcz_area_urban[c] for c in cls_sorted]
rural_areas = [lcz_area_rural[c] for c in cls_sorted]
x_max = max(u + r for u, r in zip(urban_areas, rural_areas))

# Urban bars (offset up)
ax_bar.barh(y_pos + bar_h / 2, urban_areas, height=bar_h,
            color="#c0392b", alpha=0.82, zorder=2, label="Urban")
# Rural bars (offset down)
ax_bar.barh(y_pos - bar_h / 2, rural_areas, height=bar_h,
            color="#27ae60", alpha=0.82, zorder=2, label="Rural")

# Value labels on urban bars
for yi, (u, r) in enumerate(zip(urban_areas, rural_areas)):
    if u > 0:
        ax_bar.text(u + x_max * 0.015, yi + bar_h / 2,
                    f"{u:.2f}", va="center", ha="left", fontsize=6.5, color="#7b241c")
    if r > 0:
        ax_bar.text(r + x_max * 0.015, yi - bar_h / 2,
                    f"{r:.2f}", va="center", ha="left", fontsize=6.5, color="#1e8449")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(names_sorted, fontsize=8)
ax_bar.set_xlabel("Area (km²)", fontsize=10)
ax_bar.set_title("LCZ area — urban vs rural", fontsize=11, fontweight="bold", pad=8)
ax_bar.invert_yaxis()
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.grid(axis="x", alpha=0.25, linewidth=0.5)
ax_bar.set_xlim(0, x_max * 1.32)
ax_bar.legend(fontsize=9, frameon=True, framealpha=0.9,
              loc="lower right", handlelength=1.2)


# ── SAVE ──────────────────────────────────────────────────────────────────────

out = os.path.join(OUTPUT_DIR, "LCZ_map_bolzano_20260604.png")
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.show()

#%% end
