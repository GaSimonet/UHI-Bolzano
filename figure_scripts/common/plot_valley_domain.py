#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map of the "representative valley" domain used to fit the seasonal x day/night
lapse rate in seasonal_valley_lapse_rate.py -- the along-valley transect buffered
+/-VALLEY_BUFFER_M, split into the pixels actually used for the fit (elevation
<= VALLEY_ELEV_MAX_M m a.s.l.) vs. the higher-terrain pixels inside the buffer
that get excluded.

Purely a documentation/diagnostic figure -- not currently referenced by
UHI_Bolzano.tex, but built from the exact same shapefile/DEM/config this
project's lapse-rate fit uses, so it can be dropped straight into the paper if
useful (or just used as a QC check that the domain looks right).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, ListedColormap
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from shapely.vectorized import contains as shp_contains

VALLEY_LINE_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/valley_x_sec_along_river_bolzano_v2.shp'
URBAN_SHP       = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
RURAL_SHP       = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'
DEM_RASTER      = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif'

VALLEY_BUFFER_M   = 5000.0   # must match seasonal_valley_lapse_rate.py
VALLEY_ELEV_MAX_M = 3000.0   # must match seasonal_valley_lapse_rate.py

UTM_CRS = 'EPSG:32632'
MARGIN_M = 3000.0   # extra margin around the buffer extent, for map context
PLOT_RES_M = 30.0   # DEM display resolution

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_km(gdf_or_series):
    """Rescale a GeoDataFrame/GeoSeries's geometry (in metres) to kilometres, so it
    plots correctly against the km-scaled map axes below. affine_transform only
    rescales coordinates (no re-projection)."""
    out = gdf_or_series.copy()
    if isinstance(out, gpd.GeoDataFrame):
        out['geometry'] = out.geometry.affine_transform([1 / 1000, 0, 0, 1 / 1000, 0, 0])
        return out
    return out.affine_transform([1 / 1000, 0, 0, 1 / 1000, 0, 0])


def main():
    print('Loading shapefiles…')
    valley_line = gpd.read_file(VALLEY_LINE_SHP).to_crs(UTM_CRS)
    valley_buffer = gpd.GeoSeries(valley_line.buffer(VALLEY_BUFFER_M), crs=UTM_CRS)
    valley_buffer_union = valley_buffer.union_all()
    urban = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
    rural = gpd.read_file(RURAL_SHP).to_crs(UTM_CRS)

    x0, y0, x1, y1 = valley_buffer_union.bounds
    x0 -= MARGIN_M; y0 -= MARGIN_M; x1 += MARGIN_M; y1 += MARGIN_M

    print('Reprojecting DEM onto plot grid…')
    x_coords = np.arange(x0, x1, PLOT_RES_M)
    y_coords = np.arange(y0, y1, PLOT_RES_M)          # ascending (south -> north)
    # North-up destination transform for reproject() (row 0 = north / y1)
    dst_transform = Affine(PLOT_RES_M, 0, x_coords[0], 0, -PLOT_RES_M, y_coords[-1])

    with rasterio.open(DEM_RASTER) as src:
        dem_northup = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dem_northup,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=UTM_CRS,
            resampling=Resampling.bilinear,
        )
    dem = dem_northup[::-1, :]   # flip to south-up so row index tracks ascending y_coords

    # Which plot-grid pixels are inside the buffer, and which of those are within
    # the elevation cutoff (i.e. actually used for the lapse-rate fit).
    xx, yy = np.meshgrid(x_coords, y_coords)
    in_buffer = shp_contains(valley_buffer_union, xx, yy)
    used_for_fit = in_buffer & (dem <= VALLEY_ELEV_MAX_M)
    excluded_ridge = in_buffer & (dem > VALLEY_ELEV_MAX_M)
    print(f'  Buffer pixels: {in_buffer.sum()}  used (<= {VALLEY_ELEV_MAX_M:.0f}m): {used_for_fit.sum()}  '
          f'excluded (ridge, > {VALLEY_ELEV_MAX_M:.0f}m): {excluded_ridge.sum()}')

    print('Plotting…')
    fig, ax = plt.subplots(figsize=(9, 11))
    extent = [x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000]

    # Base layer: grayscale hillshade over the WHOLE plot extent, for context outside
    # the buffer too.
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=1.5, dx=PLOT_RES_M, dy=PLOT_RES_M)
    ax.imshow(hillshade, cmap='gray', extent=extent, origin='lower', zorder=1, alpha=0.6)

    # Pixels actually used for the fit: colored by elevation.
    dem_used = np.where(used_for_fit, dem, np.nan)
    im = ax.imshow(dem_used, cmap='terrain', vmin=0, vmax=VALLEY_ELEV_MAX_M,
                   extent=extent, origin='lower', zorder=2, alpha=0.8)

    # Pixels inside the buffer but excluded by the elevation cutoff: flat red overlay.
    red_rgba = np.zeros((*excluded_ridge.shape, 4))
    red_rgba[excluded_ridge] = [0.75, 0.15, 0.15, 0.45]
    ax.imshow(red_rgba, extent=extent, origin='lower', zorder=3)

    # Valley transect line + buffer outline (rescaled to km first -- see to_km()).
    valley_line_km = to_km(valley_line)
    for geom in valley_line_km.geometry:
        xs, ys = geom.xy
        ax.plot(xs, ys, color='#1f6fd6', lw=1.8, zorder=5, label='Along-valley transect')

    valley_buffer_km = to_km(gpd.GeoSeries(valley_buffer_union, crs=UTM_CRS))
    valley_buffer_km.boundary.plot(ax=ax, color='#1f6fd6', lw=1.2, ls='--', zorder=5)

    # Urban/rural UHI reference areas, for geographic context.
    to_km(urban).boundary.plot(ax=ax, color='black', lw=1.3, zorder=6)
    to_km(rural).boundary.plot(ax=ax, color='#444444', lw=0.9, ls=':', zorder=6)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('UTM32N Easting (km)')
    ax.set_ylabel('UTM32N Northing (km)')
    ax.set_title('Representative valley domain for the WRF lapse-rate fit\n'
                f'(along-valley transect ±{VALLEY_BUFFER_M/1000:.0f} km, '
                f'≤{VALLEY_ELEV_MAX_M:.0f} m a.s.l.)', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
    cbar.set_label('Elevation (m a.s.l.) — pixels used in the fit')

    legend_handles = [
        mpatches.Patch(facecolor='none', edgecolor='#1f6fd6', linewidth=1.2, linestyle='--',
                       label=f'Valley buffer (±{VALLEY_BUFFER_M/1000:.0f} km)'),
        mpatches.Patch(facecolor=(0.75, 0.15, 0.15, 0.45), edgecolor='none',
                       label=f'Excluded (buffer, but > {VALLEY_ELEV_MAX_M:.0f} m a.s.l.)'),
        mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.3,
                       label='Urban area (UHI reference)'),
        mpatches.Patch(facecolor='none', edgecolor='#444444', linewidth=0.9, linestyle=':',
                       label='Rural area (UHI reference)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'valley_lapse_domain_map.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'valley_lapse_domain_map.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved → {out_png}')
    print(f'Saved → {out_pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()
