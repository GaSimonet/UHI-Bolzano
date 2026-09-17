# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/south_tirol_topo_20260421.py
# Paper figure:       topo_map_BZ_alto_adige_italy.pdf (panel b)
# Note: Produces panel (b), the South Tyrol/Trentino topography, of a MANUALLY composited two-panel PDF -- see italy_location_map.py in this same folder for panel (a).
# ─────────────────────────────────────────────────────────────────────────────

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import rasterio.coords
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import array_bounds
from pyproj import Transformer
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter, zoom

# %% [optional] Diagnose shapefiles — run this cell alone to inspect CRS / geometry / bounds
if False:   # ← flip to True and run this cell to diagnose
    import geopandas as gpd
    for label, path in [
        ("CLIP_SHAPEFILE",    "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/TrentinoAltoAdige.shp"),
        ("BOLZANO_SHAPEFILE", "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/Bolzano_urban_area.shp"),
    ]:
        gdf = gpd.read_file(path)
        print(f"\n── {label} ──────────────────────────")
        print(f"  CRS          : {gdf.crs}")
        print(f"  # rows       : {len(gdf)}")
        print(f"  geom types   : {gdf.geometry.geom_type.unique().tolist()}")
        print(f"  total bounds : {gdf.total_bounds}")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf4326 = gdf.to_crs("EPSG:4326")
        print(f"  bounds WGS84 : {gdf4326.total_bounds}")
        print(gdf.head(3))

# %% Config
DEM_PATH         = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol_WGS84.tif"
DOWNSAMPLE       = 4      # 1 = full 10 m res (slow), 4 = ~40 m (fast)
SUN_AZ           = 315    # sun azimuth  (NW = classic cartographic)
SUN_ALT          = 35     # sun altitude (lower = more dramatic valley shadows)
ALPHA_HS         = 0.55   # hillshade blend opacity
CONTOURS         = True   # set False to disable contour lines
CONTOUR_INTERVAL = 100    # metres

LON_BOZ, LAT_BOZ = 11.354, 46.498

# ── Bolzano municipality highlight ───────────────────────────────────────────
SHOW_BOLZANO_BOUNDARY = True
BOLZANO_SHAPEFILE     = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/Bolzano_urban_area.shp"
BOLZANO_SHAPEFILE_CRS = "EPSG:3035"
BOLZANO_FILL_COLOR    = "#ff7043"
BOLZANO_FILL_ALPHA    = 0.38
BOLZANO_EDGE_COLOR    = "#bf360c"
BOLZANO_EDGE_WIDTH    = 0.9

# ── clipping config ──────────────────────────────────────────────────────────
# CLIP_MODE options:
#   "full"      – use the entire DEM extent (default)
#   "lonlat"    – clip to CLIP_LONLAT bounding box [lon_min, lon_max, lat_min, lat_max]
#   "shapefile" – clip to polygon(s) in CLIP_SHAPEFILE (e.g. Alto Adige boundary)
CLIP_MODE          = "shapefile"
CLIP_LONLAT        = [10.8, 12.0, 46.2, 47.2]   # [lon_min, lon_max, lat_min, lat_max]
CLIP_SHAPEFILE     = "/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/TrentinoAltoAdige.shp"
CLIP_SHAPEFILE_CRS = "EPSG:4326"   # fallback if shapefile has no .prj

# %% Load & clip DEM
with rasterio.open(DEM_PATH) as src:
    crs_dem = src.crs
    nodata  = src.nodata

    if CLIP_MODE == "lonlat":
        tr_inv = Transformer.from_crs("EPSG:4326", crs_dem, always_xy=True)
        x_min, y_min = tr_inv.transform(CLIP_LONLAT[0], CLIP_LONLAT[2])
        x_max, y_max = tr_inv.transform(CLIP_LONLAT[1], CLIP_LONLAT[3])
        win = window_from_bounds(x_min, y_min, x_max, y_max, src.transform)
        win = win.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        w = max(1, round(win.width  / DOWNSAMPLE))
        h = max(1, round(win.height / DOWNSAMPLE))
        dem    = src.read(1, window=win, out_shape=(h, w),
                          resampling=Resampling.bilinear).astype(float)
        bounds = src.window_bounds(win)

    elif CLIP_MODE == "shapefile":
        if CLIP_SHAPEFILE is None:
            raise ValueError("CLIP_MODE='shapefile' requires a CLIP_SHAPEFILE path.")
        import geopandas as gpd
        from rasterio.mask import mask as rio_mask
        gdf = gpd.read_file(CLIP_SHAPEFILE)
        if gdf.crs is None:
            gdf = gdf.set_crs(CLIP_SHAPEFILE_CRS)
        gdf = gdf.to_crs(crs_dem)
        geometries = [geom.__geo_interface__ for geom in gdf.geometry]
        dem_full, clip_transform = rio_mask(src, geometries, crop=True)
        dem_full = dem_full[0].astype(float)
        ch, cw   = dem_full.shape
        h = max(1, ch // DOWNSAMPLE)
        w = max(1, cw // DOWNSAMPLE)
        dem    = zoom(dem_full, (h / ch, w / cw), order=1)
        bounds = rasterio.coords.BoundingBox(
            *array_bounds(ch, cw, clip_transform)
        )

    else:  # "full"
        bounds = src.bounds
        w = max(1, src.width  // DOWNSAMPLE)
        h = max(1, src.height // DOWNSAMPLE)
        dem = src.read(1, out_shape=(h, w),
                       resampling=Resampling.bilinear).astype(float)

if nodata is not None:
    dem[dem == nodata] = np.nan
dem[dem < 0] = np.nan

tr = Transformer.from_crs(crs_dem, "EPSG:4326", always_xy=True)
lon_min, lat_min = tr.transform(bounds.left,  bounds.bottom)
lon_max, lat_max = tr.transform(bounds.right, bounds.top)
MARGIN     = 0.01
extent_map = [lon_min + MARGIN, lon_max + MARGIN,
              lat_min - MARGIN, lat_max - MARGIN]

print(f"DEM shape   : {dem.shape}")
print(f"DEM CRS     : {crs_dem}")
print(f"Elev range  : {np.nanmin(dem):.0f} – {np.nanmax(dem):.0f} m")
print(f"Extent WGS84: {extent_map}")

# %% Colormap & hillshade
cmap_topo = mcolors.LinearSegmentedColormap.from_list(
    "topo_base", [
        (0.00, "#4a7c59"),
        (0.08, "#7fac6e"),
        (0.18, "#b8d48a"),
        (0.30, "#d4b87a"),
        (0.45, "#c49a6c"),
        (0.62, "#a07850"),
        (0.78, "#c8b8a2"),
        (0.90, "#e8e0d8"),
        (1.00, "#ffffff"),
    ]
)
cmap_topo.set_bad(color="white")
vmin, vmax = 65, np.nanmax(dem)   # 65 m ≈ Lake Garda surface; shows all valley floors

def hillshade(arr, az_deg=315, alt_deg=45, res=1.0):
    az  = np.radians(360 - az_deg + 90)
    alt = np.radians(alt_deg)
    filled = np.where(np.isnan(arr), 0, arr)
    dy, dx = np.gradient(filled, res)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)

dem[dem < vmin] = np.nan
dem_sm = uniform_filter(np.where(np.isnan(dem), 0, dem), size=3)
px_m   = 10 * DOWNSAMPLE
hs     = hillshade(dem_sm, SUN_AZ, SUN_ALT, res=px_m)

# %% Figure & plot
PROJ = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": PROJ})
ax.set_extent(extent_map, crs=PROJ)
ax.spines["geo"].set_visible(False)
ax.set_facecolor("white")
ax.set_rasterization_zorder(6)   # rasterize DEM+contours (zorder≤5) at save DPI; labels/boundaries stay vector

# composite elevation + hillshade into one RGBA array (halves embedded raster data)
norm_topo = mcolors.Normalize(vmin=vmin, vmax=vmax)
nan_mask  = np.isnan(dem)
rgba = cmap_topo(norm_topo(np.ma.masked_invalid(dem)))   # (H, W, 4)
rgba[..., :3] *= (1 - ALPHA_HS + ALPHA_HS * hs[..., np.newaxis])
rgba = np.clip(rgba, 0, 1)
rgba[nan_mask] = [1.0, 1.0, 1.0, 1.0]   # NaN areas stay pure white, not shaded grey
ax.imshow(rgba, origin="upper", extent=extent_map,
          transform=PROJ, zorder=1, interpolation="bilinear")

# ── contours ──────────────────────────────────────────────────────────────────
if CONTOURS:
    lons = np.linspace(extent_map[0], extent_map[1], dem.shape[1])
    lats = np.linspace(extent_map[3], extent_map[2], dem.shape[0])  # top→bottom
    levels = np.arange(np.floor(vmin / CONTOUR_INTERVAL) * CONTOUR_INTERVAL,
                       vmax + CONTOUR_INTERVAL, CONTOUR_INTERVAL)
    dem_filled = np.where(np.isnan(dem), 0, dem)
    ax.contour(lons, lats, dem_filled, levels=levels,
               colors="black", linewidths=0.25, alpha=0.35,
               transform=PROJ, zorder=5)
    levels_bold = levels[levels % 500 == 0]
    ax.contour(lons, lats, dem_filled, levels=levels_bold,
               colors="black", linewidths=0.6, alpha=0.45,
               transform=PROJ, zorder=5)

# ── helper: draw polygon/multipolygon boundary via ax.plot ────────────────────
def _plot_geom_boundary(ax, geom, proj, color, lw, ls="-", zorder=8):
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for part in parts:
        x, y = part.exterior.xy
        ax.plot(x, y, transform=proj, color=color,
                linewidth=lw, linestyle=ls, zorder=zorder)

def _fill_geom(ax, geom, proj, color, alpha, zorder=15):
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for part in parts:
        x, y = part.exterior.xy
        ax.fill(x, y, transform=proj, color=color, alpha=alpha, zorder=zorder)

# ── Bolzano municipality: soft fill + thin edge ───────────────────────────────
if SHOW_BOLZANO_BOUNDARY:
    import geopandas as gpd
    boz_gdf = gpd.read_file(BOLZANO_SHAPEFILE)
    if boz_gdf.crs is None:
        boz_gdf = boz_gdf.set_crs(BOLZANO_SHAPEFILE_CRS)
    boz_gdf = boz_gdf.to_crs("EPSG:4326")
    for geom in boz_gdf.geometry:
        _fill_geom(ax, geom, PROJ,
                   color=BOLZANO_FILL_COLOR, alpha=BOLZANO_FILL_ALPHA, zorder=15)
        _plot_geom_boundary(ax, geom, PROJ,
                            color=BOLZANO_EDGE_COLOR, lw=BOLZANO_EDGE_WIDTH, zorder=16)

# ── cities ────────────────────────────────────────────────────────────────────
cities = {
    "Merano":   (11.159, 46.670),
    "Vipiteno": (11.434, 46.895),
    "Trento":   (11.121, 46.067),
    "Rovereto": (11.042, 45.891),
}
for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, transform=PROJ, marker="o", markersize=8,
            color="black", markeredgecolor="white", markeredgewidth=0.6, zorder=10)
    ax.text(lon + 0.05, lat + 0.02, city, transform=PROJ,
            fontsize=14, color="black", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Lake Garda — label only (no marker); northern end is within Trentino
ax.text(10.76, 45.78, "Lake\nGarda", transform=PROJ,
        fontsize=9, color="#1a5276", fontstyle="italic",
        ha="center", va="center", zorder=11,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Bolzano — star marker + bold label
ax.plot(LON_BOZ, LAT_BOZ, transform=PROJ,
        marker="*", markersize=16, color="black",
        markeredgecolor="white", markeredgewidth=0.8, zorder=17)
ax.text(LON_BOZ + 0.07, LAT_BOZ - 0.05, "Bolzano", transform=PROJ,
        fontsize=16, fontweight="bold", color="black", zorder=18,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ── colorbar ──────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap_topo, norm=norm_topo)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                    fraction=0.04, pad=0.04, shrink=0.6, aspect=40)
cbar.set_label("Elevation (m a.s.l.)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()

# %% Save
plt.savefig("south_tyrol_topo.pdf", dpi=200, bbox_inches="tight")
plt.savefig("south_tyrol_topo.png", dpi=200, bbox_inches="tight")
plt.show()
