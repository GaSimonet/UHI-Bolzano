#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a standardized rural reference area as a buffer ring around the urban shapefile,
restricted to the valley floor, instead of the hand-drawn Bolzano_rural_area.shp.

    rural = ring(urban, GAP_M .. GAP_M + WIDTH_M)   (WIDTH_M solved so that area ~ AREA_RATIO x urban)
            ∩ { DEM elevation within MAX_DZ_M of the urban mean elevation }
            minus the urban area, minus patches smaller than MIN_PATCH_KM2

The urban mean elevation is taken from the 10 m DEM inside the urban polygon.
The result is written as new shapefiles (UTM and WGS84) next to the existing ones,
with the parameters in the filename; the existing shapefiles are not modified. To use
it, point RURAL_SHP / RURAL_SHP_WGS84 (common/paths.py or a script's CONFIG) to the
new files.

A preview figure compares the new ring with the current rural shapefile over the DEM,
and reports area, mean elevation and the number of WRF grid-cell centres inside each.

    python build_rural_ring.py            # default: 5 km round buffer from the urban edge
    python build_rural_ring.py --gap 1000 --area-ratio 10 --max-dz 100
    python build_rural_ring.py --area-ratio 0 --width 4000      # fixed width
    python build_rural_ring.py --method scale --gap 0 --width 5000 --max-dz 0 --area-ratio 0
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.features
import rasterio.windows
import xarray as xr
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import shape, JOIN_STYLE
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import URBAN_SHP, RURAL_SHP, UTM_CRS, WGS84, DEM_RASTER
from wrf_daily import build_mask, WRF_DIR, WRF_PATTERN

# ── CONFIG (defaults, overridable from the command line) ───────────────────────
GAP_M = 0.0             # distance between the urban edge and the start of the ring
WIDTH_M = 5000.0        # ring width
MAX_DZ_M = 0.0          # keep only |DEM - urban mean elevation| <= this (0 = no elevation filter)
MIN_PATCH_KM2 = 0.25    # drop isolated valley-floor patches smaller than this
AREA_RATIO = 0.0        # rural area ~ this x urban area; the ring width is then solved for
                        # (set to 0 to use WIDTH_M as given)
METHOD = 'buffer'       # how the urban shape is enlarged:
                        #   'buffer' -- all points within a distance of the urban edge (round corners)
                        #   'mitre'  -- same distance, but corners kept sharp
                        #   'scale'  -- urban polygon scaled around its centroid (exact urban shape),
                        #               scale factor chosen so its area equals the 'buffer' shape's

OUT_SHP_DIR = os.path.dirname(URBAN_SHP)
WRF_SAMPLE_DAY = '2023-07-15'   # any archive day, only used to count WRF cells in the masks


def urban_mean_elevation(urban_utm, dem):
    geoms = [g.__geo_interface__ for g in urban_utm.to_crs(dem.crs).geometry]
    data, _ = rasterio.mask.mask(dem, geoms, crop=True, filled=True, nodata=np.nan)
    return float(np.nanmean(np.where(data[0] == dem.nodata, np.nan, data[0])))


def enlarge(urban_union, dist, method):
    """Urban shape enlarged by `dist` (m) with the given method."""
    if dist <= 0:
        return urban_union
    if method == 'buffer':
        return urban_union.buffer(dist)
    if method == 'mitre':
        return urban_union.buffer(dist, join_style=JOIN_STYLE.mitre, mitre_limit=5.0)
    if method == 'scale':
        factor = np.sqrt(urban_union.buffer(dist).area / urban_union.area)
        return affinity.scale(urban_union, factor, factor, origin=urban_union.centroid)
    raise ValueError(f'Unknown method {method!r}')


def build_ring(gap, width, max_dz, min_patch_km2, method=METHOD):
    urban = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS)
    urban_union = unary_union(urban.geometry)
    ring = enlarge(urban_union, gap + width, method).difference(enlarge(urban_union, gap, method))

    with rasterio.open(DEM_RASTER) as dem:
        z_urban = urban_mean_elevation(urban, dem)
        if max_dz <= 0:   # no elevation filter: plain ring
            return (gpd.GeoDataFrame(geometry=[ring.difference(urban_union)], crs=UTM_CRS),
                    urban, z_urban)
        ring_dem = gpd.GeoSeries([ring], crs=UTM_CRS).to_crs(dem.crs)
        data, transform = rasterio.mask.mask(dem, [ring_dem.iloc[0].__geo_interface__], crop=True,
                                             filled=True, nodata=np.nan)
        elev = data[0].astype(np.float64)
        if dem.nodata is not None:
            elev[elev == dem.nodata] = np.nan
        keep = (np.abs(elev - z_urban) <= max_dz).astype(np.uint8)
        polys = [shape(geom) for geom, val in rasterio.features.shapes(keep, mask=keep == 1, transform=transform)
                 if val == 1]
        floor = gpd.GeoSeries([unary_union(polys)], crs=dem.crs).to_crs(UTM_CRS).iloc[0]

    rural = floor.intersection(ring).difference(urban_union)
    parts = gpd.GeoSeries([rural], crs=UTM_CRS).explode(index_parts=False)
    parts = parts[parts.area >= min_patch_km2 * 1e6]
    return gpd.GeoDataFrame(geometry=[unary_union(parts.values)], crs=UTM_CRS), urban, z_urban


def build_ring_for_area_ratio(gap, ratio, max_dz, min_patch_km2, method=METHOD, tol=0.02, max_width=40000.0):
    """Find the ring width giving a rural area of ~ratio x urban area (bisection on width)."""
    urban_area = gpd.read_file(URBAN_SHP).to_crs(UTM_CRS).area.sum()
    target = ratio * urban_area
    lo, hi = 0.0, max_width
    result = build_ring(gap, hi, max_dz, min_patch_km2, method)
    if result[0].area.sum() < target:
        print(f'  [WARN] even a {max_width/1000:.0f} km ring gives only '
              f'{result[0].area.sum() / urban_area:.1f}x the urban area')
        return result + (hi,)
    for _ in range(20):
        mid = (lo + hi) / 2
        result = build_ring(gap, mid, max_dz, min_patch_km2, method)
        area = result[0].area.sum()
        print(f'  width {mid/1000:6.2f} km -> {area / urban_area:5.2f}x urban area')
        if abs(area - target) <= tol * target:
            break
        lo, hi = (mid, hi) if area < target else (lo, mid)
    return result + (mid,)


def describe(name, gdf, dem, wrf):
    """Area, mean DEM elevation and WRF cell-centre count/mean HGT inside `gdf`."""
    g = gdf.to_crs(dem.crs)
    data, _ = rasterio.mask.mask(dem, [geom.__geo_interface__ for geom in g.geometry], crop=True,
                                 filled=True, nodata=np.nan)
    elev = data[0].astype(np.float64)
    if dem.nodata is not None:
        elev[elev == dem.nodata] = np.nan
    out = {'name': name, 'area_km2': gdf.to_crs(UTM_CRS).area.sum() / 1e6,
           'dem_mean_m': np.nanmean(elev), 'dem_p10_m': np.nanpercentile(elev, 10),
           'dem_p90_m': np.nanpercentile(elev, 90)}
    if wrf is not None:
        union = unary_union(gdf.to_crs(WGS84).geometry)
        m = build_mask(union, wrf['lon'].ravel(), wrf['lat'].ravel(), wrf['lon'].shape)
        out.update(wrf_cells=int(m.sum()), wrf_hgt_mean_m=float(np.nanmean(wrf['hgt'][m])) if m.any() else np.nan)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gap', type=float, default=GAP_M, help='gap between urban edge and ring (m)')
    parser.add_argument('--width', type=float, default=WIDTH_M, help='ring width (m)')
    parser.add_argument('--max-dz', type=float, default=MAX_DZ_M, help='max |elevation - urban mean| (m), 0 = no elevation filter')
    parser.add_argument('--min-patch', type=float, default=MIN_PATCH_KM2, help='min patch area (km2)')
    parser.add_argument('--area-ratio', type=float, default=AREA_RATIO,
                        help='target rural/urban area ratio; sets the ring width automatically '
                             '(0 = use --width as given)')
    parser.add_argument('--method', choices=('buffer', 'mitre', 'scale'), default=METHOD,
                        help='how the urban shape is enlarged (see METHOD in the script)')
    args = parser.parse_args()

    if args.area_ratio > 0:
        rural_ring, urban, z_urban, args.width = build_ring_for_area_ratio(
            args.gap, args.area_ratio, args.max_dz, args.min_patch, args.method)
    else:
        rural_ring, urban, z_urban = build_ring(args.gap, args.width, args.max_dz, args.min_patch, args.method)
    ratio_tag = f'_x{args.area_ratio:g}' if args.area_ratio > 0 else ''
    dz_tag = f'dz{args.max_dz:.0f}m' if args.max_dz > 0 else 'nodz'
    method_tag = f'_{args.method}' if args.method != 'buffer' else ''
    tag = f'gap{args.gap:.0f}m_w{args.width:.0f}m_{dz_tag}{ratio_tag}{method_tag}'
    out_utm = os.path.join(OUT_SHP_DIR, f'Bolzano_rural_ring_{tag}.shp')
    out_wgs = os.path.join(OUT_SHP_DIR, f'Bolzano_rural_ring_{tag}_WGS84.shp')
    rural_ring.assign(gap_m=args.gap, width_m=args.width, max_dz_m=args.max_dz,
                      z_urban_m=round(z_urban, 1)).to_file(out_utm)
    rural_ring.to_crs(WGS84).assign(gap_m=args.gap, width_m=args.width, max_dz_m=args.max_dz,
                                    z_urban_m=round(z_urban, 1)).to_file(out_wgs)

    wrf_path = os.path.join(WRF_DIR, WRF_PATTERN.format(date=WRF_SAMPLE_DAY))
    wrf = None
    if os.path.exists(wrf_path):
        ds = xr.open_dataset(wrf_path)[['HGT', 'lat', 'lon']]
        wrf = {'lat': ds['lat'].values, 'lon': ds['lon'].values, 'hgt': ds['HGT'].isel(time=0).values}
        ds.close()

    current = gpd.read_file(RURAL_SHP)
    with rasterio.open(DEM_RASTER) as dem:
        rows = [describe('urban', urban, dem, wrf),
                describe('rural current (shapefile)', current, dem, wrf),
                describe(f'rural ring ({tag})', rural_ring, dem, wrf)]

        # preview over the DEM
        bounds = urban.buffer(args.gap + args.width + 2000).to_crs(dem.crs).total_bounds
        window = rasterio.windows.from_bounds(*bounds, transform=dem.transform)
        elev = dem.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float64)
        if dem.nodata is not None:
            elev[elev == dem.nodata] = np.nan
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        dem_crs = dem.crs

    print(f'\nUrban mean elevation (DEM): {z_urban:.0f} m')
    print(f'{"":32s}{"area km2":>10s}{"DEM mean":>10s}{"DEM p10":>9s}{"DEM p90":>9s}{"WRF cells":>11s}{"WRF HGT":>9s}')
    for r in rows:
        print(f'{r["name"]:32s}{r["area_km2"]:10.1f}{r["dem_mean_m"]:10.0f}{r["dem_p10_m"]:9.0f}{r["dem_p90_m"]:9.0f}'
              f'{r.get("wrf_cells", np.nan):11.0f}{r.get("wrf_hgt_mean_m", np.nan):9.0f}')

    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(elev, extent=extent, cmap='terrain', vmin=np.nanpercentile(elev, 1),
                   vmax=np.nanpercentile(elev, 99))
    fig.colorbar(im, ax=ax, label='DEM elevation (m)', shrink=0.7)
    current.to_crs(dem_crs).boundary.plot(ax=ax, color='#d62728', lw=1.5, label='rural current')
    rural_ring.to_crs(dem_crs).plot(ax=ax, color='#2ca02c', alpha=0.5)
    rural_ring.to_crs(dem_crs).boundary.plot(ax=ax, color='#1a6e1a', lw=1, label='rural ring (new)')
    urban.to_crs(dem_crs).boundary.plot(ax=ax, color='k', lw=1.5, label='urban')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    dz_txt = f'|Δz| ≤ {args.max_dz:.0f} m' if args.max_dz > 0 else 'no elevation filter'
    ax.set_title(f'Rural ring ({args.method}): gap {args.gap/1000:g} km, width {args.width/1000:g} km, '
                 f'{dz_txt} (urban mean {z_urban:.0f} m)')
    ax.legend(loc='lower left')
    out_png = os.path.join(OUT_SHP_DIR, f'Bolzano_rural_ring_{tag}_preview.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')

    print(f'\nSaved → {out_utm}\nSaved → {out_wgs}\nSaved → {out_png}')


if __name__ == '__main__':
    main()
