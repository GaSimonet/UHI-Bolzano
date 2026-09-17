# Bolzano Urban Heat Island — figure code

Code and figures accompanying "Multi-source observations and high-resolution
modelling to investigate the Urban Heat Island in a mountain city: the case of
Bolzano in the eastern Italian Alps" (Simonet, Crespi, Zandonella Callegher,
Giovannini & Pittore).

Bolzano's UHI is characterized by combining five independent sources — WRF
mesoscale simulation, UrbClim urban climate modelling, crowdsourced (Netatmo)
and mobile (MeteoTracker) observations, and satellite thermal imagery,
2001–2025.

- **`figure_scripts/`** — the 27 scripts that generate the paper's figures, grouped
  by data source. See its own [README](figure_scripts/README.md) for the
  dynamic-lapse-rate methodology shared across several of them and the required
  conda environments.
- **`figures/`** — the 16 figures as they appear in the paper.

## Figures

| # | Figure | Script |
|---|---|---|
| 1 | Alps/Italy location map | `domain_climatology/italy_location_map.py` + `south_tirol_topo_map.py` *(manual composite)* |
| 2 | LCZ / city map | `domain_climatology/bz_city_map.py` + `lcz_map_bolzano.py` *(manual composite)* |
| 3 | Spatiotemporal scale sketch | `spatiotemporal_scale/spatiotemporal_scale_sketch.py` |
| 4 | AWS temperature heatmap | `domain_climatology/aws_temperature_heatmap.py` |
| 5 | Heatwave chronology 1980–2023 | `domain_climatology/heatwave_chronology_1980_2023.py` |
| 6 | Seasonal wind arrows | `domain_climatology/seasonal_wind_arrows_low_terrain.py` |
| 7 | Seasonal UHI maps (WRF) | `wrf_uhi/seasonal_uhi_maps_keepme_legacy.py` *(fixed lapse rate; dynamic-rate version in `seasonal_uhi_maps_dynamiclapse.py`)* |
| 8 | UrbClim UHI heatmap + distribution | `urbclim/uhi_combined_heatmap_distri_map.py` |
| 9 | MeteoTracker/Netatmo UHI ridge map | `meteotracker_netatmo/combined_uhi_ridge_map.py` |
| 10 | MeteoTracker/Netatmo scatter | `meteotracker_netatmo/mt_netatmo_scatter.py` |
| 11 | SUHI × LCZ analysis | `suhi_lcz/suhi_lcz_analysis.py` |
| 12 | Diurnal UHI, seasonal | `heatwave2023/diurnal_uhi_seasonal.py` *(rebuilt; original script lost)* |
| 13 | August 2023 heatwave, surface + vertical | `heatwave2023/combined_surface_vertical_integrated_wind_hw2023.py` |
| 14 | Spatial UHI maps, 2023 heatwave | `heatwave2023/spatial_uhi_maps_hw4_dualcbar.py` |
| 15 | Heatwave dashboard (cloud/wind) | `heatwave2023/hw4_dashboard_cloud_wind.py` |
| 16 | UrbClim vs. Netatmo density | `urbclim/urbclim_vs_netatmo_density.py` |

## Data

Input datasets (WRF, UrbClim, Netatmo, MeteoTracker, satellite LST, DEM,
shapefiles) are not included — they're large and access to some is governed by
the paper's data-availability statement. Scripts read them from their original
absolute paths on the authors' systems; check each script's `CONFIG` section
(or `common/paths.py` for the shared ones) for what to substitute.

## License

Code: [MIT](LICENSE). Figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [`CITATION.cff`](CITATION.cff) for how to cite.
