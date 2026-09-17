# External data sources and paths

The scripts in `figure_scripts/` read their inputs from **absolute paths on the
authors' workstation and on the Eurac Research CEPH project storage**. Those paths
are left exactly as they were when the paper's figures were produced — nothing has
been rewritten or parameterised — so that the released code is a faithful record of
what was run.

**To re-run any script you must substitute your own paths.** This document lists
every external input, grouped by storage root, so you know what to substitute and
what each file is.

A practical starting point: `figure_scripts/common/paths.py` holds the shapefile,
DEM and CRS constants shared by the six dynamic-lapse-rate scripts. Editing that one
file repoints the largest group of scripts. The remaining scripts carry their paths
inline, near the top of each file under a `CONFIG` block.

---

## 1. `/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/`

Eurac Research project storage (RETURN project). Not publicly mounted; access is
governed by the data-availability statement in the paper.

### WRF mesoscale simulation output

| Path (relative to the root above) | Contents |
|---|---|
| `WRF_ALTO_ADIGE/WRF_RAW_2020_subset/WRF_2020_RAW_subset.nc` | Raw WRF output, 2020 |
| `WRF_ALTO_ADIGE/WRF_RAW_2021_subset/WRF_RAW_Alto_Adige_2021.nc` | Raw WRF output, 2021 |
| `WRF_ALTO_ADIGE/WRF_RAW_2022_subset/WRF_RAW_Alto_Adige_2022.nc` | Raw WRF output, 2022 |
| `WRF_ALTO_ADIGE/WRF_RAW_2023_subset/WRF_RAW_Alto_Adige_2023.nc` | Raw WRF output, 2023 |
| `WRF_ALTO_ADIGE/WRF_RAW_2024_subset/WRF_RAW_Alto_Adige_2024.nc` | Raw WRF output, 2024 |
| `WRF_ALTO_ADIGE/WRF_consolidated_intermediate` | Harmonised daily intermediate archive derived from the five yearly files |
| `WRF_ALTO_ADIGE/HW_202308_13_26_subset` | High-frequency subset covering the 13–26 August 2023 heatwave |

The 2020–2024 raw files carry the full 3-D `TK`/`Z` fields and are what
`common/seasonal_valley_lapse_rate.py` reads to fit the seasonal valley lapse rates.
They are large; the fit is cached (see "Caches" below).

### Other CEPH inputs

| Path | Contents |
|---|---|
| `UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc` | UrbClim urban-climate model output for Bolzano, interannual |
| `TASMAX_HEATWAVES/Heatwaves95p_fixed_1980_2025_Bolzano_Grid.csv` | Heatwave chronology, 95th-percentile fixed threshold, 1980–2025, gridded regional record |
| `AWS_alto_Adige/scripts/01_process/bolzano_heatmap_only.py` | Referenced in a header comment as the original location of a processing step; not imported |

### CEPH output directories

These are write targets, not inputs; scripts create them if missing. Repoint them
anywhere writable.

- `UrbClim/combined_figures`
- `UrbClim/ridge_plots`
- `UrbClim/spatial_maps`

---

## 2. `/home/gsimonet/Desktop/` — local project trees

The figure scripts were consolidated into this repository from six separate local
project trees. The originals remain in place on the authors' workstation; the paths
below are what the scripts still point at.

### Shapefiles (urban/rural masks, valley transect)

| Path | Contents |
|---|---|
| `WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp` | Urban-area mask, UTM 32N |
| `WRF_alto_adige_local/.../shapefiles/Bolzano_urban_area_WGS84.shp` | Same, WGS84 |
| `WRF_alto_adige_local/.../shapefiles/Bolzano_rural_area.shp` | Rural reference mask, UTM 32N |
| `WRF_alto_adige_local/.../shapefiles/Bolzano_rural_area_WGS84.shp` | Same, WGS84 |
| `WRF_alto_adige_local/.../shapefiles/Bolzano_city_core.shp` | City-core polygon |
| `WRF_alto_adige_local/.../shapefiles/valley_x_sec_along_river_bolzano_v2.shp` (`.dbf`) | Along-valley transect line, used to build the representative valley polygon for the lapse-rate fit |
| `NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_urban_area.shp` | Second copy of the urban mask, used by the Netatmo-side scripts |
| `NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_rural_area.shp` | Second copy of the rural mask |
| `multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/Bolzano_urban_area.shp` | Urban mask used by the Italy location map |
| `multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location/TrentinoAltoAdige.shp` | Regional administrative boundary |

### Rasters

| Path | Contents |
|---|---|
| `multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol.tif` | 10 m DEM, South Tyrol, UTM 32N. Reprojected onto other grids by `common/dem_utils.py` because neither the UrbClim nor the LST NetCDFs carry elevation |
| `multi_source_UHI_Bolzano_package/topo_map_bolzano/DEM_10m_south_tirol_WGS84.tif` | Same, WGS84 |
| `multi_source_UHI_Bolzano_package/topo_map_bolzano/LCZ_clipped.tif` | Local Climate Zone raster, clipped to the Bolzano domain |

### Observational datasets

| Path | Contents |
|---|---|
| `NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc` | Quality-controlled crowdsourced Netatmo temperatures, Dec 2021 – Nov 2025 |
| `Meteotrackers_package/meteotracker_incremental/netcdf_files/meteotracker_summer_2025.nc` | MeteoTracker mobile transects, summer 2025 |
| `LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc` | Satellite land surface temperature, Bolzano domain |
| `AWS_alto_Adige/CSV/organized/8320LTV0010A_Bozen_20030101_20240224.csv` | Bolzano automatic weather station record, 2003–2024 |
| `multi_source_UHI_Bolzano_package/UrbClim_vs_netatmo/csv_exports/urbclim_netatmo_pairs.csv` | Pre-matched UrbClim/Netatmo pixel–station pairs |

### Local WRF derivatives and archives

| Path | Contents |
|---|---|
| `WRF_alto_adige_local/script/analysis/UHI/harmonized_dynamiclapse_outputs` | Output directory for the dynamic-lapse-rate WRF UHI maps |
| `WRF_alto_adige_local/script/analysis/UHI/seasonal_valley_lapse_rate_archive` | `.npz` fit cache and timestamped archive copies for the lapse-rate fit. **Deliberately not duplicated into this repository** so re-runs on the original workstation reuse the cache instead of re-reading five years of WRF output |
| `WRF_ALTO_ADIGE_on_CEPH` | Local mount point mirroring the CEPH WRF tree |
| `WRF_ALTO_ADIGE_on_CEPH/HW_202308_13_26_subset` | Heatwave subset via that mount |

---

## 3. Network services

`meteotracker_netatmo/combined_uhi_ridge_map.py` imports `requests`. It fetches
basemap/context tiles at plot time; it does not require credentials. Running it
offline will fail at that call.

`domain_climatology/bz_city_map.py` and `italy_location_map.py` use `osmnx` and
`contextily`, which likewise download OpenStreetMap data and basemap tiles on demand.

---

## 4. Caches

Three scripts cache expensive intermediates and either prompt interactively
(`confirm_recompute()`) or gate on a `FORCE_RECOMPUTE` flag:

- `common/seasonal_valley_lapse_rate.py`
- `wrf_uhi/seasonal_uhi_maps_dynamiclapse.py`
- `urbclim/uhi_combined_heatmap_distri_map.py`

The cache artefacts (`.npz`, intermediate CSVs, and the hashed JSON caches under
`domain_climatology/cache/`) are **excluded from this repository** — they are
machine-specific, large, and fully regenerable. Expect the first run of any of these
three scripts to be slow.

The one cached product that *is* included is
`figure_scripts/common/seasonal_valley_lapse_rates.csv`: the eight fitted
(season × day/night) lapse rates that every other script reads via
`load_seasonal_valley_lapse()`. It is small, stable, and is the numerical result
reported in the paper, so it ships with the code.

---

## 5. Non-public Python dependency

`heatwave2023/combined_surface_vertical_integrated_wind_hw2023.py` imports:

```python
from mygaspypack.wrfutils import uvrot2wrf, latlon2xy, intpxsecv2
```

`mygaspypack` is an internal helper package (WRF wind rotation, lat/lon-to-grid-index
lookup, and vertical cross-section interpolation) that is not on PyPI and is not
included here. That single script cannot run without it; the other 26 can.
