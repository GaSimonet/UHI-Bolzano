#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared DEM-reprojection helper.

Several sources used across these scripts carry no native elevation variable at
all (checked: UrbClim's Bolzano_UHI_RETURN_combined.nc has only T2M/QV2M/WS2M/LST;
the LST/satellite file lst-bolzano.nc has only the LST DataArray + spatial_ref) --
so the lapse-rate correction needs an external DEM reprojected onto each source's
own grid, rather than relying on one source happening to carry elevation that
others can borrow.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject as _rio_reproject, Resampling as _RioResampling
from rasterio.transform import Affine

DEM_RASTER = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif'


def reproject_dem_to_grid(x_coords, y_coords, dst_crs='EPSG:32632', dem_raster=DEM_RASTER):
    """Reproject DEM_RASTER onto an arbitrary regular grid.

    x_coords, y_coords: 1-D arrays of cell-center coordinates (ascending OR
    descending -- both handled) in `dst_crs` meters.

    Returns a 2-D array shaped (len(y_coords), len(x_coords)), NaN outside the
    DEM's coverage.
    """
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[1] - y_coords[0])
    # Affine built from the actual index-0 coordinate and the actual (possibly
    # negative) step, so this works whether the grid's y axis ascends (like
    # UrbClim's) or descends (the usual north-up raster convention).
    dst_transform = Affine(res_x, 0, x_coords[0] - res_x / 2,
                            0, res_y, y_coords[0] - res_y / 2)

    with rasterio.open(dem_raster) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        src_transform = src.transform
        src_crs = src.crs

    dst = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float32)
    _rio_reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=_RioResampling.bilinear,
        src_nodata=src_nodata, dst_nodata=np.nan,
    )
    return dst.astype(np.float64)
