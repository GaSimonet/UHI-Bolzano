#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared constants used across the WRF/UrbClim/SUHI dynamic-lapse-rate scripts
(wrf_uhi/, heatwave2023/, suhi_lcz/, urbclim/) -- pulled out here because these
exact values were previously duplicated, verbatim, in every one of those scripts.

Not used by the scripts under domain_climatology/, spatiotemporal_scale/, or
meteotracker_netatmo/ -- those are unrelated one-off figure scripts that don't
share this methodology, so they keep their own local constants unchanged.
"""

URBAN_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp'
URBAN_SHP_WGS84 = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area_WGS84.shp'
RURAL_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area.shp'
RURAL_SHP_WGS84 = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area_WGS84.shp'

VALLEY_LINE_SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/valley_x_sec_along_river_bolzano_v2.shp'
DEM_RASTER = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif'

UTM_CRS = 'EPSG:32632'
WGS84   = 'EPSG:4326'
