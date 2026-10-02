#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version the external input data used by the paper-figure scripts.

The inputs live outside the repository (CEPH archive, local packages) and are either
too large for git (WRF/UrbClim: 10^1-10^2 GB) or not redistributable (NetAtmo,
MeteoTracker). This script records, for every input:

  - files      : size, modification time, SHA-256 (full hash; for files larger than
                 FULL_HASH_MAX_BYTES a 'partial' hash of size + first/last 64 MB)
  - directories: number of files, total size, and a fingerprint (SHA-256 over the
                 sorted list of name/size/mtime) -- e.g. the 1827-file WRF daily archive

and copies the small, self-made vector files (shapefiles and their sidecars) into
data/shapefiles/ so they are versioned directly in git.

Output: data/data_manifest.csv (+ data/shapefiles/). Re-run after any input changes
and commit the result; a changed hash/fingerprint flags a changed input.

    python make_data_manifest.py
"""

import os
import glob
import shutil
import hashlib
import datetime
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT_CSV = os.path.join(REPO, 'data', 'data_manifest.csv')
SHP_DIR = os.path.join(REPO, 'data', 'shapefiles')
FULL_HASH_MAX_BYTES = 5 * 1024 ** 3
CHUNK = 64 * 1024 ** 2

SHP = '/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles'
CEPH = '/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE'
PKG = '/home/gsimonet/Desktop/multi_source_UHI_Bolzano_package'

# (path, kind, used for) -- inputs read by the paper-figure scripts
INPUTS = [
    # NetAtmo / MeteoTracker / AWS (not redistributable or large -> manifest only)
    ('/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_SCT_20211231_2300_20251112_1000.nc',
     'file', 'NetAtmo QC with corrected SCT (Fig. 8, 9, 11, 13, 14)'),
    ('/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/qc_output/temperature_qc_filtered_20211231_2300_20251112_1000.nc',
     'file', 'NetAtmo QC without SCT (previous figure versions, Appendix)'),
    ('/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/raw_nc_files/NetAtmo_Bolzano_temperature_20251112.nc',
     'file', 'NetAtmo raw data, input of the QC'),
    ('/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/station_list/stations_20250131_160200.csv',
     'file', 'NetAtmo station metadata (street names)'),
    ('/home/gsimonet/Desktop/Meteotrackers_package/meteotracker_incremental/netcdf_files/meteotracker_summer_2025.nc',
     'file', 'MeteoTracker July 2025 campaign (Fig. 8, 9)'),
    ('/home/gsimonet/Desktop/AWS_alto_Adige/CSV/organized/8320LTV0010A_Bozen_20030101_20240224.csv',
     'file', 'AWS Bolzano hourly temperature (Fig. 4a)'),
    (f'{PKG}/UrbClim_vs_netatmo/csv_exports/urbclim_netatmo_pairs.csv',
     'file', 'UrbClim-NetAtmo collocated pairs (Appendix)'),
    # Satellite, UrbClim, heatwaves
    ('/home/gsimonet/Desktop/LST_remote_sensing/lst-bolzano-all/lst-bolzano.nc', 'file', 'Fused Landsat/MODIS LST'),
    (f'{CEPH}/UrbClim/Bolzano/Bolzano_UHI_RETURN_combined.nc', 'file', 'UrbClim 2001-2023 hourly fields'),
    (f'{CEPH}/TASMAX_HEATWAVES/Heatwaves95p_fixed_1980_2025_Bolzano_Grid.csv', 'file', 'Heatwave events (Fig. 4b)'),
    # WRF
    (f'{CEPH}/WRF_ALTO_ADIGE/WRF_consolidated_intermediate', 'dir', 'WRF daily harmonized archive 2020-2024'),
    (f'{CEPH}/WRF_ALTO_ADIGE/HW_202308_13_26_subset', 'dir', 'WRF HW 2023 subset (wind/cross-sections)'),
    (f'{CEPH}/WRF_ALTO_ADIGE/WRF_RAW_2020_subset/WRF_2020_RAW_subset.nc', 'file', 'WRF raw 2020 (seasonal wind)'),
    (f'{CEPH}/WRF_ALTO_ADIGE/WRF_RAW_2021_subset/WRF_RAW_Alto_Adige_2021.nc', 'file', 'WRF raw 2021 (seasonal wind)'),
    (f'{CEPH}/WRF_ALTO_ADIGE/WRF_RAW_2023_subset/WRF_RAW_Alto_Adige_2023.nc', 'file', 'WRF raw 2023 (seasonal wind)'),
    (f'{CEPH}/WRF_ALTO_ADIGE/WRF_RAW_2024_subset/WRF_RAW_Alto_Adige_2024.nc', 'file', 'WRF raw 2024 (seasonal wind)'),
    # Rasters / boundaries from third parties (manifest only)
    (f'{PKG}/topo_map_bolzano/DEM_10m_south_tirol.tif', 'file', 'DEM 10 m (elevation corrections, maps)'),
    (f'{PKG}/topo_map_bolzano/DEM_10m_south_tirol_WGS84.tif', 'file', 'DEM 10 m WGS84 (Fig. 1b)'),
    (f'{PKG}/topo_map_bolzano/LCZ_clipped.tif', 'file', 'European LCZ map, Demuzere et al. 2022 (Fig. 2)'),
    (f'{PKG}/MAP_of_ITALY_for_BZ_location/TrentinoAltoAdige.shp', 'shp', 'Regional boundary (Fig. 1)'),
]

# Self-made vector files: copied into the repo as well
OWN_SHAPEFILES = [
    (f'{SHP}/Bolzano_urban_area.shp', 'Urban area (EPSG:3035)'),
    (f'{SHP}/Bolzano_urban_area_WGS84.shp', 'Urban area (WGS84)'),
    (f'{SHP}/Bolzano_rural_area.shp', 'Hand-drawn rural polygon (EPSG:3035), gridded-source UHI'),
    (f'{SHP}/Bolzano_rural_area_WGS84.shp', 'Hand-drawn rural polygon (WGS84)'),
    (f'{SHP}/Bolzano_rural_ring_gap0m_w5000m_nodz.shp', '5 km rural buffer (UTM 32N), build_rural_ring.py'),
    (f'{SHP}/Bolzano_rural_ring_gap0m_w5000m_nodz_WGS84.shp', '5 km rural buffer (WGS84)'),
    (f'{SHP}/Bolzano_rural_ring_gap0m_w5000m_nodz_EPSG3035.shp', '5 km rural buffer (EPSG:3035), Fig. 8'),
    (f'{SHP}/valley_x_sec_along_river_bolzano_v2.shp', 'Valley transect for the lapse-rate fit'),
    (f'{SHP}/Bolzano_city_core.shp', 'City core polygon'),
    ('/home/gsimonet/Desktop/NETATMO_BOLZANO_PACKAGE/visualization/LCZ_shapefile_analysis/Bolzano_urban_area.shp',
     'Urban area, copy used by the NetAtmo/MeteoTracker scripts'),
    (f'{PKG}/MAP_of_ITALY_for_BZ_location/Bolzano_urban_area.shp', 'Urban area, copy used by Fig. 1b'),
]


def sha256_file(path, size):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if size <= FULL_HASH_MAX_BYTES:
            for block in iter(lambda: f.read(CHUNK), b''):
                h.update(block)
            return h.hexdigest(), 'full'
        h.update(str(size).encode())
        h.update(f.read(CHUNK))
        f.seek(-CHUNK, os.SEEK_END)
        h.update(f.read(CHUNK))
        return h.hexdigest(), 'partial (size + first/last 64 MB)'


def iso(t):
    return datetime.datetime.fromtimestamp(t).isoformat(timespec='seconds')


def describe_file(path, used_for, kind='file', repo_copy=''):
    if not os.path.exists(path):
        return {'path': path, 'kind': kind, 'used_for': used_for, 'status': 'MISSING'}
    st = os.stat(path)
    digest, method = sha256_file(path, st.st_size)
    return {'path': path, 'kind': kind, 'used_for': used_for, 'status': 'ok', 'n_files': 1,
            'size_bytes': st.st_size, 'mtime': iso(st.st_mtime), 'sha256': digest,
            'hash_method': method, 'repo_copy': repo_copy}


def describe_dir(path, used_for):
    if not os.path.isdir(path):
        return {'path': path, 'kind': 'dir', 'used_for': used_for, 'status': 'MISSING'}
    entries = []
    for root, _, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            st = os.stat(p)
            entries.append((os.path.relpath(p, path), st.st_size, int(st.st_mtime)))
    entries.sort()
    h = hashlib.sha256('\n'.join(f'{n}\t{s}\t{m}' for n, s, m in entries).encode())
    return {'path': path, 'kind': 'dir', 'used_for': used_for, 'status': 'ok', 'n_files': len(entries),
            'size_bytes': sum(e[1] for e in entries),
            'mtime': iso(max(e[2] for e in entries)) if entries else '',
            'sha256': h.hexdigest(), 'hash_method': 'fingerprint of sorted name/size/mtime list'}


def shapefile_parts(shp):
    stem = os.path.splitext(shp)[0]
    return sorted(p for p in glob.glob(stem + '.*') if os.path.splitext(p)[0] == stem)


def main():
    rows = []
    os.makedirs(SHP_DIR, exist_ok=True)

    for path, kind, used_for in INPUTS:
        print('  ', path)
        if kind == 'dir':
            rows.append(describe_dir(path, used_for))
        elif kind == 'shp':
            rows += [describe_file(p, used_for, 'shapefile part') for p in shapefile_parts(path)] or \
                    [describe_file(path, used_for, 'shapefile part')]
        else:
            rows.append(describe_file(path, used_for))

    for shp, used_for in OWN_SHAPEFILES:
        parts = shapefile_parts(shp)
        if not parts:
            rows.append({'path': shp, 'kind': 'shapefile part', 'used_for': used_for, 'status': 'MISSING'})
            continue
        sub = {'shapefiles': 'wrf_analysis', 'LCZ_shapefile_analysis': 'netatmo_package',
               'MAP_of_ITALY_for_BZ_location': 'italy_map'}.get(os.path.basename(os.path.dirname(shp)),
                                                                os.path.basename(os.path.dirname(shp)))
        dest_dir = os.path.join(SHP_DIR, sub)
        os.makedirs(dest_dir, exist_ok=True)
        for p in parts:
            dest = os.path.join(dest_dir, os.path.basename(p))
            shutil.copy2(p, dest)
            rows.append(describe_file(p, used_for, 'shapefile part', os.path.relpath(dest, REPO)))

    df = pd.DataFrame(rows)
    df.insert(0, 'manifest_created', datetime.datetime.now().isoformat(timespec='seconds'))
    df.to_csv(OUT_CSV, index=False)
    print(f'\n{len(df)} entries, {int((df.status == "MISSING").sum())} missing -> {OUT_CSV}')


if __name__ == '__main__':
    main()
