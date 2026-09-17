# Bolzano UHI paper — figure-generation scripts

Every script that produces a figure in the paper, consolidated here from six
separate project trees on the authors' workstation (`WRF_alto_adige_local/`,
`multi_source_UHI_Bolzano_package/`, `LST_remote_sensing/`, `UrbClim/`,
`Heat_waves_bolzano/`, plus one on the `/mnt/CEPH_PROJECTS` mount). The originals
remain in place there; this folder is the consolidated copy going forward.

This file documents the **methodology and the per-script detail**. For the
repository overview, the figure-to-script map and how to get started, see the
[top-level README](../README.md); for every external data path the scripts read,
see [`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

## Two tiers of scripts in here

**`common/` + `wrf_uhi/` + `heatwave2023/` + `suhi_lcz/` + `urbclim/`** — six scripts
(eight files) share one methodology: UHI/SUHI intensity corrected for elevation using
a **dynamic lapse rate**, empirically fit per (season × day/night) from the WRF
2020–2024 archive over a representative valley polygon, instead of a single fixed
constant (e.g. the standard-atmosphere −6.5 K/km). Day/night is the real astronomical
sunrise/sunset for Bolzano (46.50°N, 11.35°E) per calendar day, not a fixed UTC-hour
split. These six scripts got real edits during consolidation (import paths repointed
at `common/`, shared constants deduplicated) — see "Shared methodology" below.

**`domain_climatology/` + `spatiotemporal_scale/` + `meteotracker_netatmo/`** — a
further fourteen scripts that feed other figures in the same paper but share none of
that methodology (domain maps, LCZ composites, AWS climatology, heatwave chronology,
wind arrows, ridge plots, MT/Netatmo comparisons). These were copied **as-is**, with
only a short header comment added (original location + which paper figure + any
caveats) — no internal edits, no forced common-module usage.

## Shared methodology (the six-script group)

The lapse rate is fit **once**, from WRF only (the only source with a full 3-D
atmospheric profile), then reused everywhere else as a plain scalar via the
classical linear correction `T_corrected = T + Γ·(elevation − reference elevation)`.
No other script re-derives Γ — UrbClim/Netatmo/Satellite couldn't anyway, since
none of them carry a vertical profile.

**How Γ itself is fit** (`common/seasonal_valley_lapse_rate.py`): for every WRF grid
column inside the representative valley domain (a 5&nbsp;km buffer around the
along-valley transect, restricted to elevations up to 3000&nbsp;m&nbsp;a.s.l.) and every
hourly timestep, the **local** lapse rate is the direct two-point slope between that
column's own near-surface temperature and the temperature at a fixed 3000&nbsp;m
a.s.l. reference altitude taken from **that same column's** vertical profile (WRF's
full 3-D `TK`/`Z` fields):

```
Γ_i(t) = (T2_i(t) − T@3000m_i(t)) / (3000 − HGT_i)
```

e.g. a valley-floor pixel at 250&nbsp;m uses `(T@3000m − T2(250m)) / (3000−250)`; a
pixel near the valley wall at 1500&nbsp;m uses `(T@3000m − T2(1500m)) / (3000−1500)` —
each column supplies its own two-point vertical slope (**not** a regression pooling
many columns' elevation/T2 pairs, which was this script's method before
2026-08-03). Those per-column, per-timestep rates are then averaged spatially over
all valley pixels and temporally within each (season, day/night) bucket. Day/night
is the real astronomical sunrise/sunset for Bolzano (46.50°N, 11.35°E) per calendar
day, not a fixed UTC-hour split.

Current fitted values (2020–2024, K/km):

| Season | Day | Night |
|---|---|---|
| Winter | 3.94 | 3.19 |
| Spring | 6.83 | 5.37 |
| Summer | 7.18 | 5.58 |
| Autumn | 5.59 | 4.51 |

- `common/seasonal_valley_lapse_rate.py` — fits the 8 rates above, writes the stable
  `common/seasonal_valley_lapse_rates.csv` every other script reads via
  `load_seasonal_valley_lapse()`. Heavy — reads the full WRF archive (T2, HGT, plus
  the 3-D TK/Z fields); its `.npz` fit-cache and timestamped archive copies are
  **not** duplicated into this folder, they stay at the original
  `WRF_alto_adige_local/.../UHI/seasonal_valley_lapse_rate_archive/` path so re-runs
  reuse the existing cache instead of re-reading 5 years of WRF output.
- `common/solar_daynight.py` — self-contained (no `astral`/`ephem` dependency)
  NOA-style sunrise/sunset formula; `classify_day_night(times)` returns `'day'`/
  `'night'` per timestamp.
- `common/dem_utils.py` — `reproject_dem_to_grid()`, reprojecting
  `DEM_10m_south_tirol.tif` onto an arbitrary regular grid. Needed because neither
  UrbClim's `Bolzano_UHI_RETURN_combined.nc` nor the LST/satellite
  `lst-bolzano.nc` carry a native elevation variable.
- `common/lapse_mode_utils.py` — `resolve_rate(mode, dynamic_rate)` and
  `LAPSE_RATE_CONSTANT` (0.0065 K/m), backing the `LAPSE_MODE` toggle described
  below.
- `common/paths.py` — the `URBAN_SHP`/`RURAL_SHP`/`UTM_CRS`/`WGS84` constants that
  were duplicated verbatim across all six scripts before consolidation.
- `wrf_uhi/seasonal_uhi_maps_dynamiclapse.py` — seasonal × day/night UHI maps from
  the WRF harmonized daily archive, using the fitted rate. **Run
  `seasonal_valley_lapse_rate.py` first** (or after the WRF archive changes) to
  (re)generate the CSV this depends on.
- `wrf_uhi/seasonal_uhi_maps_keepme_legacy.py` — the OLD, still-current source of
  the paper's `seasonal_uhi_maps_corrected.png` figure. Uses a **fixed** 0.0065 K/m
  rate, not the dynamic methodology, and has no LAPSE_MODE toggle. Regenerate that
  figure from `seasonal_uhi_maps_dynamiclapse.py` (with `LAPSE_MODE='valley'`)
  instead to pick up the new methodology.
- `heatwave2023/diurnal_uhi_hw4_full_distri.py`,
  `heatwave2023/hw4_dashboard_cloud_wind.py`,
  `heatwave2023/spatial_uhi_maps_hw4_dualcbar.py` — the August 2023 heatwave (HW4)
  UHI comparison across UrbClim/WRF/Netatmo/Satellite.
- `suhi_lcz/suhi_lcz_analysis.py` — satellite SUHI × LCZ analysis, split into
  (season, day/night) buckets (9 total incl. "All data"), with elevation from
  `DEM_10m_south_tirol.tif` reprojected onto the LST grid.
- `urbclim/uhi_combined_heatmap_distri_map.py` — interannual UrbClim UHI heatmap +
  distribution + seasonal maps combined figure. Needs `rioxarray` — see env table.

### LAPSE_MODE: none / constant / valley

All six methodology scripts (except the frozen `seasonal_uhi_maps_keepme_legacy.py`)
expose a `LAPSE_MODE` config constant near their other CONFIG values:

- `'none'` — no elevation correction at all (raw urban-minus-rural difference).
- `'constant'` — the classical, spatio-temporally fixed rate (`LAPSE_RATE_CONSTANT`
  = 0.0065 K/m), applied uniformly regardless of season or day/night.
- `'valley'` (default) — the dynamic, per-(season, day/night) rate above.

Edit `LAPSE_MODE` and re-run a script to get that mode's version of the figure;
every script tags its output filename (or, for `suhi_lcz_analysis.py`, its
`OUTPUT_DIR`) with the mode so the three don't overwrite each other. **For
`urbclim/uhi_combined_heatmap_distri_map.py` specifically**, `CSV_TIMESTEP` and
`NPZ_ALL` are also tagged with the mode — those two are looked up under a fixed
filename via an interactive `confirm_recompute()` prompt rather than a parameter
hash, so without the mode in the filename, switching `LAPSE_MODE` and answering
"n" (use cache) would silently serve a different mode's stale results.

## Env requirements

Different scripts here need different conda environments (a real, pre-existing
constraint, not introduced by this consolidation):

| Script | Env | Why |
|---|---|---|
| `wrf_uhi/*.py` | `UHI_map_env` | xarray/geopandas/shapely/scipy/pyproj |
| `common/seasonal_valley_lapse_rate.py` | `UHI_map_env` | same |
| `heatwave2023/diurnal_uhi_hw4_full_distri.py` | `ROI_python_38_env` | needs `salem` for WRF loading |
| `heatwave2023/hw4_dashboard_cloud_wind.py` | `ROI_python_38_env` | needs `salem` |
| `heatwave2023/spatial_uhi_maps_hw4_dualcbar.py` | `ROI_python_38_env` | needs `salem` (WRF panel only) |
| `suhi_lcz/suhi_lcz_analysis.py` | `UHI_map_env` | rasterio-only, no salem/rioxarray |
| `urbclim/uhi_combined_heatmap_distri_map.py` | `RETURN_py_env` | needs `rioxarray` for `.rio.clip`/`.rio.write_crs` |

The other 14 (domain_climatology/, spatiotemporal_scale/, meteotracker_netatmo/,
plus the two heatwave2023/urbclim light-touch scripts) weren't part of this
session's environment audit — check each script's own imports if you hit one.

## Cache/recompute prompts

`common/seasonal_valley_lapse_rate.py`, `wrf_uhi/seasonal_uhi_maps_dynamiclapse.py`,
and `urbclim/uhi_combined_heatmap_distri_map.py` each cache expensive intermediate
results (`.npz`/CSV) and either prompt interactively before recomputing
(`confirm_recompute()`) or gate on a `FORCE_RECOMPUTE` flag — check the top of each
script before a long run. The other scripts have no cache (they read their much
smaller source files fresh every run).

## All 16 figures in the paper → source script

| # | Paper figure | Script(s) in this folder | Status |
|---|---|---|---|
| 1 | `topo_map_BZ_alto_adige_italy.pdf` | `domain_climatology/italy_location_map.py` (panel a) + `south_tirol_topo_map.py` (panel b) | **Manual composite** — final PDF is hand-assembled (vector editor), no single script produces it |
| 2 | `LCZ_map_city_combined_20260604.png` | `domain_climatology/bz_city_map.py` (left panel) + `lcz_map_bolzano.py` (right panel) | **Manual composite** — same situation as #1 |
| 3 | `spatiotemporal_sketch.png` | `spatiotemporal_scale/spatiotemporal_scale_sketch.py` | **Unconfirmed** best-candidate; no script saves this exact filename, likely a manual rename/touch-up |
| 4 | `AWS_temperature_heatmap_hour_month.pdf` | `domain_climatology/aws_temperature_heatmap.py` | Found, exact match (base filename; prefix renamed) |
| 5 | `HW_fixed_threshold_1980_2023_grid_only.pdf` | `domain_climatology/heatwave_chronology_1980_2023.py` | Found, exact match |
| 6 | `seasonal_wind_arrows_low_terrain.png` | `domain_climatology/seasonal_wind_arrows_low_terrain.py` | Found, exact match |
| 7 | `seasonal_uhi_maps_corrected.png` | `wrf_uhi/seasonal_uhi_maps_keepme_legacy.py` (current source, **fixed** rate) / `wrf_uhi/seasonal_uhi_maps_dynamiclapse.py` (new dynamic-rate replacement) | Found; **paper currently uses the old fixed-rate version** |
| 8 | `uhi_distri_map_combined.png` | `urbclim/uhi_combined_heatmap_distri_map.py` | Found, exact match; dynamic-rate methodology applied this session |
| 9 | `combined_uhi_ridge_map_2025-07-01_to_2025-08-01_12-15h.pdf` | `meteotracker_netatmo/combined_uhi_ridge_map.py` (`RIDGE_MODE='uhi'`) | Found, exact match |
| 10 | `scatter_MT_netATMO_station_deviation.png` | `meteotracker_netatmo/mt_netatmo_scatter.py` | **Likely match**, filename appears manually renamed from `scatter_mt_netatmo_{dates}.png` |
| 11 | `suhi_lcz_maps_seasonal.pdf` | `suhi_lcz/suhi_lcz_analysis.py` | Found, exact match; dynamic-rate + day/night split added this session |
| 12 | `diurnal_uhi_seasonal_20260512_101357.png` | `heatwave2023/diurnal_uhi_seasonal.py` | **Original script lost** (deleted, unrecoverable); this later rebuild post-dates the cited figure — see its own docstring for provenance |
| 13 | `HW2023_combined_surface_vertical_integrated_wind.png` | `heatwave2023/combined_surface_vertical_integrated_wind_hw2023.py` | Found, exact match |
| 14 | `spatial_uhi_maps_heatwave2023_dualcbar_distrib.png` | `heatwave2023/spatial_uhi_maps_hw4_dualcbar.py` | **Original (unfixed) script lost**; only the `_fixed` successor exists — regenerate the paper figure from this one |
| 15 | `hw4_dashboard_cloud_wind_20260730_094811.pdf` | `heatwave2023/hw4_dashboard_cloud_wind.py` | Found, exact match; dynamic-rate methodology applied this session |
| 16 | `UrbClim_vs_Netatmo_exact_pixel_density.png` | `urbclim/urbclim_vs_netatmo_density.py` | Found, exact match |

**Read the "Status" column before assuming any script reproduces its cited figure
bit-for-bit** — five entries (#1, #2, #3, #12, #14) have a caveat: either the paper
currently embeds output from a script that's since been fixed/replaced (#7, #14), or
the true original generator is gone or unconfirmed (#1, #2, #3, #12).

## Folder layout

```
figure_scripts/
├── README.md                  (this file)
├── common/                    shared methodology modules + the stable lapse-rate CSV
├── wrf_uhi/                    WRF seasonal UHI maps (figure #7)
├── heatwave2023/                August 2023 heatwave UHI comparison + vertical/diurnal (#12, #13, #14, #15)
├── suhi_lcz/                    Satellite SUHI × LCZ analysis (#11)
├── urbclim/                     UrbClim interannual heatmap + UrbClim/Netatmo density (#8, #16)
├── domain_climatology/          Domain/LCZ/AWS/heatwave-chronology/wind maps (#1, #2, #4, #5, #6)
├── spatiotemporal_scale/         Spatio-temporal scale sketch (#3)
└── meteotracker_netatmo/         MT/Netatmo ridge + scatter comparisons (#9, #10)
```
