# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/ITALY_map_20260421.py
# Paper figure:       topo_map_BZ_alto_adige_italy.pdf (panel a)
# Note: The paper figure is a MANUAL two-panel composite (Illustrator/Inkscape -- a 'Page 1.pdf' export artifact sits alongside the final file at the original location). This script produces only panel (a), the Italy-location inset; see south_tirol_topo_map.py in this same folder for panel (b). No single script produces the combined PDF.
# ─────────────────────────────────────────────────────────────────────────────

# %% Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth, Reader
import matplotlib.patheffects as pe
import geopandas as gpd

# %% Config
LON_BOZ, LAT_BOZ = 11.354, 46.498
LON_ROM, LAT_ROM = 12.496, 41.902

PROJ = ccrs.LambertConformal(central_longitude=12, central_latitude=44)
DATA = ccrs.PlateCarree()

ALTO_ADIGE_SHAPEFILE     = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/TrentinoAltoAdige.shp"
ALTO_ADIGE_SHAPEFILE_CRS = "EPSG:4326"   # fallback if shapefile has no .prj

# %% Figure setup
fig = plt.figure(figsize=(9, 7))
ax = fig.add_axes([0.05, 0.05, 0.88, 0.90], projection=PROJ)
ax.set_extent([6.5, 18.5, 36.5, 47.8], crs=DATA)
ax.set_facecolor("#d6eaf8")

# %% Physical features
land = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", facecolor="#e8e8e8", edgecolor="none", zorder=1)
ax.add_feature(land)

borders = cfeature.NaturalEarthFeature(
    "cultural", "admin_0_countries", "10m",
    facecolor="none", edgecolor="#aaaaaa", linewidth=0.6, zorder=2)
ax.add_feature(borders)

ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
               linewidth=0.8, edgecolor="#666666", zorder=4)
ax.add_feature(cfeature.RIVERS.with_scale("10m"),
               linewidth=0.4, edgecolor="#89cff0", zorder=4)

# %% Italy highlight
italy_shp = Reader(natural_earth(resolution="10m", category="cultural",
                                  name="admin_0_countries"))
italy_geom = None
for rec in italy_shp.records():
    if rec.attributes["NAME"] == "Italy":
        italy_geom = rec.geometry
        ax.add_geometries([italy_geom], DATA,
                          facecolor="#fff9c4", edgecolor="#888888",
                          linewidth=0.8, zorder=3)
        break

# %% Country labels
country_labels = {
    "France":      (2.5,  46.5),
    "Switzerland": (8.2,  46.9),
    "Austria":     (14.0, 47.4),
    "Slovenia":    (14.8, 46.0),
    "Germany":     (11.5, 47.65),
    "Tunisia":     (9.5,  33.8),
    "Algeria":     (3.5,  36.5),
    "Libya":       (13.5, 31.5),
    "Greece":      (22.0, 39.5),
    "Albania":     (20.0, 41.0),
    "Montenegro":  (19.0, 42.5),
    "Bosnia":      (17.5, 44.0),
    "Serbia":      (20.5, 44.0),
    "Croatia":     (16.2, 45.2),
}

italy_bbox = italy_geom.bounds   # (minx, miny, maxx, maxy)
MAP_EXTENT = (6.5, 18.5, 36.5, 47.8)   # lonmin, lonmax, latmin, latmax

for name, (lon, lat) in country_labels.items():
    if not (MAP_EXTENT[0] <= lon <= MAP_EXTENT[1] and
            MAP_EXTENT[2] <= lat <= MAP_EXTENT[3]):
        continue
    ax.text(lon, lat, name, transform=DATA, fontsize=7, color="#555555",
            ha="center", va="center", style="italic",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            zorder=6)

# %% Trentino-Alto Adige region (from shapefile)
aa_gdf = gpd.read_file(ALTO_ADIGE_SHAPEFILE)
if aa_gdf.crs is None:
    aa_gdf = aa_gdf.set_crs(ALTO_ADIGE_SHAPEFILE_CRS)
aa_gdf = aa_gdf.to_crs("EPSG:4326")
ax.add_geometries(aa_gdf.geometry, DATA,
                  facecolor="#e8b88a", edgecolor="#c0392b",
                  linewidth=1.2, zorder=5, alpha=0.85)

# %% Italy label
ax.text(13.5, 42.5, "Italy", transform=DATA, fontsize=13,
        fontweight="bold", color="#7a6a00", ha="center", style="italic",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=7)

# %% City markers
# Roma — capital square
ax.plot(LON_ROM, LAT_ROM, transform=DATA,
        marker="s", markersize=6, color="black",
        markeredgecolor="white", markeredgewidth=0.7, zorder=10)
ax.text(LON_ROM + 0.22, LAT_ROM + 0.10, "Roma", transform=DATA,
        fontsize=8, color="black", fontweight="bold", zorder=11,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Bolzano — star marker
ax.plot(LON_BOZ, LAT_BOZ, transform=DATA,
        marker="*", markersize=14, color="black",
        markeredgecolor="white", markeredgewidth=0.8, zorder=10)
ax.text(LON_BOZ + 0.32, LAT_BOZ - 0.15, "Bolzano",
        transform=DATA, fontsize=10, fontweight="bold",
        color="black", zorder=11,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Reference cities
refs = {"Trento": (11.12, 46.07), "Innsbruck": (11.39, 47.27),
        "Merano": (11.16, 46.67), "Verona": (10.99, 45.44)}
for city, (lon, lat) in refs.items():
    ax.plot(lon, lat, transform=DATA, marker="o", markersize=5,
            color="#444444", markeredgecolor="white",
            markeredgewidth=0.6, zorder=10)
    ax.text(lon + 0.18, lat + 0.06, city, transform=DATA,
            fontsize=8, color="#333333", zorder=11,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# %% Graticule
gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                  color="gray", alpha=0.5, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 8}
gl.ylabel_style = {"size": 8}

# %% Legend
italy_patch = mpatches.Patch(facecolor="#fff9c4", edgecolor="#888888",
                              linewidth=0.8, label="Italy")
st_patch    = mpatches.Patch(facecolor="#e8b88a", edgecolor="#c0392b",
                              linewidth=1.2, alpha=0.85, label="Trentino-Alto Adige")
star_marker = plt.Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="black", markersize=10,
                          label="Bolzano")

ax.legend(handles=[italy_patch, st_patch, star_marker],
          loc="lower left", fontsize=9, framealpha=0.9)

# %% Save
plt.savefig("bolzano_location_map.pdf", dpi=300, bbox_inches="tight")
plt.savefig("bolzano_location_map.png", dpi=200, bbox_inches="tight")
plt.show()
