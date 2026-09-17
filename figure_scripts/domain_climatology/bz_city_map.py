# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/Bolzano_CITY_maps/BZ_city_map_20260430.py
# Paper figure:       LCZ_map_city_combined_20260604.png (left panel)
# Note: The paper figure is a MANUAL composite (a 9.1 MB hand-edited SVG exists at the original location, created ~11 min after lcz_map_bolzano.py's own run). This script produces only the left 'urban morphology' panel (OSMnx buildings/water/parks/roads); see lcz_map_bolzano.py in this same folder for the right LCZ-classification panel.
# ─────────────────────────────────────────────────────────────────────────────

"""
Beautiful publication-quality map of Bolzano
Requirements: pip install osmnx geopandas matplotlib contextily shapely
"""

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx
from shapely.geometry import box

# ── 1. Configuration ──────────────────────────────────────────────────────────
PLACE   = "Bolzano, South Tyrol, Italy"
CRS     = "EPSG:25832"   # UTM zone 32N — good for northern Italy
DPI     = 300
FIGSIZE = (14, 18)

# ── 2. Download OSM data ───────────────────────────────────────────────────────
print("Downloading OSM data …")

# Street network (drive + walk combined)
G = ox.graph_from_place(PLACE, network_type="all", retain_all=False)
edges = ox.graph_to_gdfs(G, nodes=False)

# Footprint layers
tags = {
    "buildings":  {"building": True},
    "water":      {"natural": ["water", "riverbank"], "waterway": True},
    "parks":      {"leisure": ["park", "garden", "nature_reserve"]},
    "forest":     {"landuse": ["forest"], "natural": ["wood"]},
    "grass":      {"landuse": ["grass", "meadow", "recreation_ground"],
                   "natural": ["grassland"]},
    "farmland":   {"landuse": ["farmland", "orchard", "vineyard"]},
    "industrial": {"landuse": ["industrial", "commercial", "retail"]},
    "railway":    {"railway": ["rail", "subway", "tram"]},
}
layers = {}
for name, t in tags.items():
    try:
        layers[name] = ox.features_from_place(PLACE, tags=t)
    except Exception as e:
        print(f"  ⚠ Could not fetch {name}: {e}")

# Administrative boundary
boundary = ox.geocode_to_gdf(PLACE)

# ── 3. Reproject to metric CRS ────────────────────────────────────────────────
edges    = edges.to_crs(CRS)
boundary = boundary.to_crs(CRS)
for k in layers:
    layers[k] = layers[k].to_crs(CRS)

# Clip to boundary with a small buffer
buf = boundary.geometry.iloc[0].buffer(200)
for k in layers:
    layers[k] = layers[k].clip(buf)

# ── 4. Road classification palette ───────────────────────────────────────────
ROAD_STYLE = {
    "motorway":       dict(color="#E8604C", lw=2.2, zorder=5),
    "trunk":          dict(color="#E8814C", lw=1.8, zorder=5),
    "primary":        dict(color="#F0A500", lw=1.4, zorder=4),
    "secondary":      dict(color="#F0C040", lw=1.0, zorder=4),
    "tertiary":       dict(color="#FFFFFF", lw=0.7, zorder=3),
    "residential":    dict(color="#FFFFFF", lw=0.4, zorder=2),
    "footway":        dict(color="#CCCCCC", lw=0.3, zorder=2, linestyle="--"),
    "cycleway":       dict(color="#88CC88", lw=0.4, zorder=2, linestyle=":"),
    "_default":       dict(color="#DDDDDD", lw=0.3, zorder=1),
}

def road_style(highway):
    if isinstance(highway, list):
        highway = highway[0]
    return ROAD_STYLE.get(highway, ROAD_STYLE["_default"])

# ── 5. Figure setup ───────────────────────────────────────────────────────────
BG = "#FFFFFF"          # white background
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
ax.set_facecolor(BG)
ax.set_aspect("equal")

# Boundary mask (white halo around city edge)
boundary.plot(ax=ax, facecolor="none", edgecolor="#FFFFFF", linewidth=1.2, zorder=10)

# ── 6. Layer rendering ────────────────────────────────────────────────────────
# Parks / green areas
if "parks" in layers:
    poly = layers["parks"][layers["parks"].geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    poly.plot(ax=ax, facecolor="#A8D5A2", edgecolor="none", alpha=0.85)

# Forest
if "forest" in layers:
    poly = layers["forest"][layers["forest"].geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    poly.plot(ax=ax, facecolor="#4A8C4A", edgecolor="none", alpha=0.85)

# Grass / meadow
if "grass" in layers:
    poly = layers["grass"][layers["grass"].geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    poly.plot(ax=ax, facecolor="#C8E6A0", edgecolor="none", alpha=0.85)

# Farmland / orchards / vineyards
if "farmland" in layers:
    poly = layers["farmland"][layers["farmland"].geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    poly.plot(ax=ax, facecolor="#EDE8C0", edgecolor="none", alpha=0.85)

# Industrial / commercial
if "industrial" in layers:
    poly = layers["industrial"][layers["industrial"].geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    poly.plot(ax=ax, facecolor="#D4B8C8", edgecolor="none", alpha=0.85)

# Water
if "water" in layers:
    w = layers["water"]
    poly_w = w[w.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    line_w = w[w.geometry.geom_type.isin(["LineString","MultiLineString"])]
    poly_w.plot(ax=ax, facecolor="#1F3A5F", edgecolor="#3A7ABD", linewidth=0.5, alpha=0.9)
    line_w.plot(ax=ax, color="#3A7ABD", linewidth=1.2, alpha=0.8)

# Buildings
if "buildings" in layers:
    bld = layers["buildings"][layers["buildings"].geometry.geom_type.isin(
        ["Polygon","MultiPolygon"])]
    bld.plot(ax=ax, facecolor="#2C2C4A", edgecolor="#44446A", linewidth=0.2, alpha=0.9)

# Railways
if "railway" in layers:
    rly = layers["railway"][layers["railway"].geometry.geom_type.isin(
        ["LineString","MultiLineString"])]
    rly.plot(ax=ax, color="#AAAAFF", linewidth=0.8, linestyle="-.", alpha=0.7)

# Roads (grouped by highway type for performance)
for hw_type, style in ROAD_STYLE.items():
    if hw_type == "_default":
        mask = ~edges["highway"].apply(
            lambda h: any(k in (h if isinstance(h,list) else [h])
                         for k in ROAD_STYLE if k != "_default"))
    else:
        mask = edges["highway"].apply(
            lambda h: hw_type in (h if isinstance(h,list) else [h]))
    subset = edges[mask]
    if not subset.empty:
        subset.plot(ax=ax,
                    color=style["color"],
                    linewidth=style["lw"],
                    zorder=style["zorder"],
                    linestyle=style.get("linestyle","-"),
                    alpha=0.85)

# ── 7. Extent & framing ───────────────────────────────────────────────────────
b = boundary.total_bounds          # minx, miny, maxx, maxy
pad = 300                          # metres padding
ax.set_xlim(b[0]-pad, b[2]+pad)
ax.set_ylim(b[1]-pad, b[3]+pad)
ax.axis("off")

# ── 8. Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#A8D5A2", label="Parks & gardens"),
    mpatches.Patch(facecolor="#4A8C4A", label="Forest & woodland"),
    mpatches.Patch(facecolor="#C8E6A0", label="Grass & meadow"),
    mpatches.Patch(facecolor="#EDE8C0", label="Farmland & orchards"),
    mpatches.Patch(facecolor="#D4B8C8", label="Industrial & commercial"),
    mpatches.Patch(facecolor="#1F3A5F", edgecolor="#3A7ABD", label="Water bodies"),
    mpatches.Patch(facecolor="#2C2C4A", edgecolor="#44446A", label="Buildings"),
    Line2D([0],[0], color="#E8604C", lw=2,   label="Motorway"),
    Line2D([0],[0], color="#F0A500", lw=1.4, label="Primary road"),
    Line2D([0],[0], color="#FFFFFF", lw=0.7, label="Tertiary road"),
    Line2D([0],[0], color="#88CC88", lw=0.8, linestyle=":", label="Cycleway"),
    Line2D([0],[0], color="#AAAAFF", lw=0.8, linestyle="-.", label="Railway"),
]
legend = ax.legend(handles=legend_items,
                   loc="lower right", framealpha=0.4,
                   facecolor="#0D0D1A", edgecolor="#444466",
                   fontsize=10, labelcolor="white",
                   borderpad=1.2, labelspacing=0.8,
                   handlelength=2.2, handleheight=1.4)
legend.get_title().set_color("white")

# ── 9. Scale bar (top-right, where compass was) ───────────────────────────────
scale_m = 1000
x0 = b[2] + pad - 1400   # near top-right
y0 = b[3] + pad - 400
ax.plot([x0, x0+scale_m], [y0, y0], color="white", lw=2.5)
ax.plot([x0, x0], [y0-50, y0+50], color="white", lw=2)
ax.plot([x0+scale_m, x0+scale_m], [y0-50, y0+50], color="white", lw=2)
ax.text(x0 + scale_m/2, y0+100, "1 km",
        color="white", fontsize=11, ha="center", fontweight="bold")

# ── 11. Export ────────────────────────────────────────────────────────────────
out = "bolzano_map.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.show()