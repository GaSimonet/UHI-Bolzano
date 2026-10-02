#!/usr/bin/env python3
# ── Consolidated into figure_scripts/ on 2026-09-30 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/LCZ_map_bolzano_urban_20260604.py
# Paper figure:       LCZ_map_city_combined_20260604.png (right panel: LCZ map + URBAN-only
#                     LCZ area bar chart) -- a MANUALLY composited figure; see bz_city_map.py
#                     in this same folder for the left panel.
# UPDATE (2026-09-30): rural outline now the 5 km round buffer around the urban area
# (common/build_rural_ring.py) instead of the hand-drawn Bolzano_rural_area.shp; output
# written to UHI-Bolzano/figures/ as two separate files (map panel, bar panel), larger
# legend and thicker boundary lines. The bar chart (urban area only) is unaffected.
# ─────────────────────────────────────────────────────────────────────────────

# -*- coding: utf-8 -*-
"""
LCZ Map — Bolzano urban area only
Two separate panels:
  Map  : OpenTopoMap basemap + LCZ raster + urban & rural boundaries
  Bars : Horizontal bar chart — LCZ area (km²) within the urban area only
"""

#%% ── IMPORTS ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import rasterio.crs
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize
from shapely.geometry import mapping, box
import contextily as ctx
import os

#%% ── CONFIGURATION ──────────────────────────────────────────────────────────

LCZ_RASTER  = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/LCZ_clipped.tif"
URBAN_SHAPE = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area_WGS84.shp"
RURAL_SHAPE = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_ring_gap0m_w5000m_nodz.shp"
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")   # UHI-Bolzano/figures
DPI         = 300

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


#%% ── LOAD LCZ RASTER ────────────────────────────────────────────────────────

print("Loading LCZ raster…")
with rasterio.open(LCZ_RASTER) as src:
    lcz_data_src  = src.read(1).astype(np.float32)
    lcz_src_crs   = src.crs
    lcz_src_trans = src.transform
    lcz_nodata    = src.nodata if src.nodata is not None else 0
    lcz_h, lcz_w  = src.shape
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    PIXEL_AREA_KM2 = (res_x * res_y) / 1e6

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
                  lcz_bounds_wm[1], lcz_bounds_wm[3]]

lcz_disp = lcz_wm.astype(float)
lcz_disp[lcz_disp == 0] = np.nan


#%% ── LCZ COLOURMAP ──────────────────────────────────────────────────────────

all_lcz_vals   = sorted(LCZ_COLORS.keys())
lcz_cmap       = ListedColormap([LCZ_COLORS[v] for v in all_lcz_vals], name="lcz")
lcz_boundaries = [v - 0.5 for v in all_lcz_vals] + [all_lcz_vals[-1] + 0.5]
lcz_bnorm      = BoundaryNorm(lcz_boundaries, lcz_cmap.N)


#%% ── LOAD URBAN SHAPEFILE + DERIVE EXTENT ───────────────────────────────────

print("Loading shapefiles…")
urban_wm  = gpd.read_file(URBAN_SHAPE).to_crs("EPSG:3857")
rural_wm  = gpd.read_file(RURAL_SHAPE).to_crs("EPSG:3857")
urban_lcz = gpd.read_file(URBAN_SHAPE).to_crs(lcz_src_crs)

# Map extent = union of urban + rural bounds + 6 % padding, so rural borders are visible
ub = urban_wm.total_bounds
rb = rural_wm.total_bounds
combined = (
    min(ub[0], rb[0]), min(ub[1], rb[1]),
    max(ub[2], rb[2]), max(ub[3], rb[3]),
)
pad_x = (combined[2] - combined[0]) * 0.06
pad_y = (combined[3] - combined[1]) * 0.06
bx = (combined[0] - pad_x, combined[1] - pad_y,
      combined[2] + pad_x, combined[3] + pad_y)


#%% ── URBAN MASK ON LCZ GRID ─────────────────────────────────────────────────

print("Building urban mask on LCZ grid…")
geoms_lcz  = [mapping(g) for g in urban_lcz.geometry if g is not None and not g.is_empty]
urban_mask = rasterize(
    geoms_lcz, out_shape=(lcz_h, lcz_w),
    transform=lcz_src_trans, fill=0, default_value=1, dtype=np.uint8,
).astype(bool)
print(f"  Urban pixels: {urban_mask.sum():,}")


#%% ── AREA PER LCZ CLASS WITHIN URBAN AREA ───────────────────────────────────

lcz_int     = lcz_data_src.astype(np.int16)
present_lcz = sorted([int(v) for v in np.unique(lcz_int[urban_mask])
                       if int(v) > 0 and int(v) in LCZ_NAMES])

lcz_area_urban = {}
for cls in present_lcz:
    lcz_area_urban[cls] = int(np.sum((lcz_int == cls) & urban_mask)) * PIXEL_AREA_KM2

# Sort by area descending
cls_sorted   = sorted(present_lcz, key=lambda c: lcz_area_urban[c], reverse=True)
areas_sorted = [lcz_area_urban[c] for c in cls_sorted]
names_sorted = [LCZ_NAMES[c] for c in cls_sorted]
cols_sorted  = [LCZ_COLORS.get(c, "#888888") for c in cls_sorted]

print(f"  LCZ classes in urban area: {cls_sorted}")
print(f"  Total urban LCZ area: {sum(areas_sorted):.2f} km²")


#%% ── FIGURE STYLE ───────────────────────────────────────────────────────────

URBAN_LW      = 2.6    # urban boundary line width
RURAL_LW      = 2.2    # rural boundary line width
LEGEND_FS     = 12     # map legend font size
BAR_LABEL_FS  = 14     # bar panel: class names, axis label and ticks
BAR_VALUE_FS  = 12     # bar panel: 'x.xx km² (y %)' annotations


#%% ── MAP PANEL (separate figure) ───────────────────────────────────────────

print("Rendering map panel…")

fig_map, ax_map = plt.subplots(figsize=(9.5, 7))

ax_map.set_xlim(bx[0], bx[2])
ax_map.set_ylim(bx[1], bx[3])

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
    origin="upper", alpha=0.62, zorder=2,
)

urban_wm.boundary.plot(ax=ax_map, color="#1a1a1a", linewidth=URBAN_LW, zorder=3)
rural_wm.boundary.plot(ax=ax_map, color="#1a1a1a", linewidth=RURAL_LW, linestyle="--", zorder=3)

ax_map.set_xticks([]); ax_map.set_yticks([])

# Legend: urban + rural boundaries + LCZ classes present in urban area
legend_handles = [
    mlines.Line2D([], [], color="#1a1a1a", linewidth=URBAN_LW, label="Urban area"),
    mlines.Line2D([], [], color="#1a1a1a", linewidth=RURAL_LW, linestyle="--",
                  label="Rural area (5 km buffer)"),
]
for cls in present_lcz:
    legend_handles.append(
        mpatches.Patch(facecolor=LCZ_COLORS.get(cls, "#888888"), edgecolor="none",
                       alpha=0.85, label=LCZ_NAMES.get(cls, f"LCZ {cls}"))
    )

# Legend below the map (outside), so it does not hide the city or the ring
ax_map.legend(
    handles=legend_handles,
    loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
    fontsize=LEGEND_FS, frameon=False,
    handlelength=2.2, handleheight=1.0, handletextpad=0.6,
    labelspacing=0.5, columnspacing=1.4,
)


#%% ── BAR PANEL — urban LCZ area distribution (separate figure) ─────────────

print("Rendering bar panel…")

fig_bar, ax_bar = plt.subplots(figsize=(8, 7))

y_pos = np.arange(len(cls_sorted))
x_max = max(areas_sorted)

bars = ax_bar.barh(y_pos, areas_sorted, height=0.6,
                   color=cols_sorted, alpha=0.85, zorder=2)
for bar in bars:
    bar.set_edgecolor("#444444")
    bar.set_linewidth(0.5)

# Percentage of total urban area annotated on each bar
total_urban = sum(areas_sorted)
for yi, (area, cls) in enumerate(zip(areas_sorted, cls_sorted)):
    pct = 100 * area / total_urban if total_urban > 0 else 0
    ax_bar.text(area + x_max * 0.02, yi,
                f"{area:.2f} km²  ({pct:.1f} %)",
                va="center", ha="left", fontsize=BAR_VALUE_FS, color="#333333")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(names_sorted, fontsize=BAR_LABEL_FS)
ax_bar.set_xlabel("Area (km²)", fontsize=BAR_LABEL_FS)
ax_bar.tick_params(axis="x", labelsize=BAR_LABEL_FS - 1)
ax_bar.invert_yaxis()
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.grid(axis="x", alpha=0.25, linewidth=0.5)
ax_bar.set_xlim(0, x_max * 1.75)   # room for labels + percentages


# ── SAVE ──────────────────────────────────────────────────────────────────────

for fig, name in ((fig_map, "LCZ_map_bolzano_urban_map_20260930"),
                  (fig_bar, "LCZ_map_bolzano_urban_bars_20260930")):
    out = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
plt.show()

#%% end
