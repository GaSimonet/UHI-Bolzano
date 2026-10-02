# External input data

The figure scripts read their inputs from outside this repository (CEPH archive and
local data packages). These inputs are too large for git (WRF and UrbClim: tens to
hundreds of GB) or cannot be redistributed (NetAtmo, MeteoTracker), so they are
versioned here as follows:

- `data_manifest.csv`: every input file or directory with its size, modification time
  and SHA-256 hash (partial hash for files > 5 GB; for directories, a fingerprint of the
  sorted file list with sizes and modification times), plus the figure it is used for.
- `shapefiles/`: copies of the self-made vector files (urban area, hand-drawn rural
  polygon, 5 km rural buffer, valley transect), grouped by their original folder:
  `wrf_analysis/` (WRF_alto_adige_local/.../LCZ_shapefile_analysis/shapefiles),
  `netatmo_package/` (NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis),
  `italy_map/` (multi_source_UHI_Bolzano_package/MAP_of_ITALY_for_BZ_location).

Regenerate after any input change and commit the result:

    python figure_scripts/common/make_data_manifest.py

A changed hash or fingerprint in the diff of `data_manifest.csv` shows which input changed.
The scripts still read the original paths; the copies here are for archiving.
