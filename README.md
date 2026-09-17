# Multi-source observations and high-resolution modelling to investigate the Urban Heat Island in a mountain city: the case of Bolzano in the eastern Italian Alps

Figure-generation code and final figures accompanying the paper.

**Gaspard Simonet**¹, Alice Crespi¹, Claudio Zandonella Callegher³, Lorenzo Giovannini², Massimiliano Pittore¹

¹ Center for Climate Change and Transformation, Eurac Research, Bolzano, Italy
² Department of Civil, Environmental and Mechanical Engineering, University of Trento, Trento, Italy
³ Institute for Renewable Energy, Eurac Research, Bolzano, Italy

---

## What this repository is

Mountain cities test the limits of classical urban heat island (UHI) theory. This
study characterises the UHI of Bolzano, a mid-size city in the Adige Valley, by
combining five independent observational and modelling sources — WRF mesoscale
simulation, UrbClim urban climate modelling, crowdsourced (Netatmo) and mobile
(MeteoTracker) near-surface observations, and satellite thermal imagery — spanning
2001–2025.

This repository contains **the 27 Python scripts that produced the paper's 16
figures**, together with **the figures themselves**. It is a code- and
figure-availability record, not a turnkey pipeline: the scripts read large input
datasets that live outside this repository (see below).

```
.
├── figures/            the 16 figures as they appear in the paper
├── figure_scripts/     the scripts that produce them (see its own README for the
│                       methodology, especially the dynamic lapse-rate correction)
├── environments/       conda environment exports for the three envs the scripts need
└── docs/
    └── DATA_SOURCES.md every external input path, what it holds, and what to substitute
```

## Reproducing a figure

**The input data is not in this repository.** The scripts point at absolute paths on
the authors' workstation and on Eurac Research project storage, left unmodified so
the released code matches what was actually run. Before running anything, read
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md), which lists every external input and
what it contains. Availability of the underlying datasets is governed by the paper's
data-availability statement.

Three conda environments are needed — a pre-existing constraint of the source
projects, not an artefact of this release:

```bash
conda env create -f environments/UHI_map_env.yml       # most scripts
conda env create -f environments/ROI_python_38_env.yml # needs salem (WRF loading)
conda env create -f environments/RETURN_py_env.yml     # needs rioxarray
```

`figure_scripts/README.md` has the per-script environment table, the `LAPSE_MODE`
toggle, and the cache/recompute behaviour of the three expensive scripts.

## Methodology in one paragraph

The six scripts under `common/`, `wrf_uhi/`, `heatwave2023/`, `suhi_lcz/` and
`urbclim/` share one methodology: UHI/SUHI intensity corrected for elevation using a
**dynamic lapse rate**, fit empirically per (season × day/night) from the WRF
2020–2024 archive over a representative valley polygon, rather than a fixed constant
such as the standard-atmosphere −6.5 K/km. Day/night is real astronomical
sunrise/sunset for Bolzano (46.50°N, 11.35°E) per calendar day, not a fixed UTC-hour
split. The fitted rates (K/km) ship as
`figure_scripts/common/seasonal_valley_lapse_rates.csv`:

| Season | Day | Night |
|---|---|---|
| Winter | 3.94 | 3.19 |
| Spring | 6.83 | 5.37 |
| Summer | 7.18 | 5.58 |
| Autumn | 5.59 | 4.51 |

The full derivation, including why each grid column supplies its own two-point
vertical slope rather than a pooled regression, is in
[`figure_scripts/README.md`](figure_scripts/README.md).

## Figure → script map

| # | Figure in `figures/` | Script(s) | Notes |
|---|---|---|---|
| 1 | `topo_map_BZ_alto_adige_italy.pdf` | `domain_climatology/italy_location_map.py` (a) + `south_tirol_topo_map.py` (b) | **Manual composite** — the final PDF was hand-assembled in a vector editor; no single script emits it |
| 2 | `LCZ_map_city_combined_20260604.png` | `domain_climatology/bz_city_map.py` (left) + `lcz_map_bolzano.py` (right) | **Manual composite**, as #1 |
| 3 | `spatiotemporal_sketch.png` | `spatiotemporal_scale/spatiotemporal_scale_sketch.py` | **Unconfirmed** best candidate; no script writes this exact filename, so a manual rename/touch-up is likely |
| 4 | `AWS_temperature_heatmap_hour_month_panelab_20260804.pdf` | `domain_climatology/aws_temperature_heatmap.py` | Exact match; filename prefix renamed after generation |
| 5 | `HW_fixed_threshold_1980_2023_grid_only_panelab_20260804.pdf` | `domain_climatology/heatwave_chronology_1980_2023.py` | Exact match |
| 6 | `seasonal_wind_arrows_low_terrain.png` | `domain_climatology/seasonal_wind_arrows_low_terrain.py` | Exact match |
| 7 | `seasonal_uhi_maps_corrected.png` | `wrf_uhi/seasonal_uhi_maps_keepme_legacy.py` | **The paper's version uses a fixed 0.0065 K/m rate**, not the dynamic methodology. `wrf_uhi/seasonal_uhi_maps_dynamiclapse.py` is the dynamic-rate replacement |
| 8 | `uhi_distri_map_combined.png` | `urbclim/uhi_combined_heatmap_distri_map.py` | Exact match, dynamic rate |
| 9 | `combined_uhi_ridge_map_2025-07-01_to_2025-08-01_12-15h_astro_20260804.pdf` | `meteotracker_netatmo/combined_uhi_ridge_map.py` (`RIDGE_MODE='uhi'`) | Exact match |
| 10 | `scatter_MT_netATMO_station_deviation.png` | `meteotracker_netatmo/mt_netatmo_scatter.py` | **Likely match**; filename appears manually renamed from `scatter_mt_netatmo_{dates}.png` |
| 11 | `suhi_lcz_maps_seasonal.pdf` | `suhi_lcz/suhi_lcz_analysis.py` | Exact match, dynamic rate + day/night split |
| 12 | `diurnal_uhi_seasonal.png` | `heatwave2023/diurnal_uhi_seasonal.py` | **Original script lost.** The script here is a later rebuild that post-dates the cited figure — see its docstring for provenance |
| 13 | `HW2023_combined_surface_vertical_integrated_wind_valley.png` | `heatwave2023/combined_surface_vertical_integrated_wind_hw2023.py` | Shipped under its generated filename; the manuscript cites it without the `_valley` suffix. Requires `mygaspypack` (see `docs/DATA_SOURCES.md` §5) |
| 14 | `spatial_uhi_maps_heatwave2023_dualcbar_distrib.png` | `heatwave2023/spatial_uhi_maps_hw4_dualcbar.py` | **The original, unfixed script is lost**; only its corrected successor exists here, so re-running it will not reproduce the embedded figure bit-for-bit |
| 15 | `hw4_dashboard_cloud_wind.png` | `heatwave2023/hw4_dashboard_cloud_wind.py` | Shipped as PNG; the manuscript cites a timestamped PDF (`..._20260730_094811.pdf`) that was not retained. Exact match otherwise, dynamic rate |
| 16 | `UrbClim_vs_Netatmo_exact_pixel_density.png` | `urbclim/urbclim_vs_netatmo_density.py` | Exact match |

**Read the Notes column before assuming a script reproduces its figure bit-for-bit.**
Six entries carry a caveat: the generator is a manual composite (#1, #2), unconfirmed
(#3), lost and rebuilt or superseded (#12, #14), or the paper embeds output from a
version that has since been revised (#7).

## What is deliberately not here

- **Input datasets** — WRF/UrbClim/Netatmo/MeteoTracker/satellite files, DEM rasters
  and shapefiles. Documented in `docs/DATA_SOURCES.md`.
- **Caches** — `.npz` fit caches, hashed JSON caches, intermediate CSVs. Large,
  machine-specific, regenerable.
- **Intermediate script outputs** — the working plots each script wrote next to
  itself. Only the 16 figures used in the paper are kept, under `figures/`.
- **The manuscript source** — the LaTeX, bibliography and journal class files are
  not part of this release.
- **`mygaspypack`** — an internal, unpublished helper package that one script
  (#13) imports.

## Citing

If you use this code, please cite the paper. See [`CITATION.cff`](CITATION.cff).

## License

Code in `figure_scripts/` is released under the MIT License (see [`LICENSE`](LICENSE)).
The figures in `figures/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

Gaspard Simonet — Center for Climate Change and Transformation, Eurac Research, Bolzano, Italy
