# fetch_lidar_ept.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 2 (Task 2.1)
# PURPOSE: Download the LiDAR point cloud of the study tile from the USGS 3DEP
#          public archive. The 3DEP collections are published on AWS as Entwine
#          Point Tiles (EPT); PDAL's EPT reader streams only the points inside a
#          bounding box, so a 1 km tile is fetched without touching the full
#          30-billion-point collection. The tile is reprojected from the archive's
#          Web Mercator grid to UTM 18N (metres) and saved as a LAZ file - the raw
#          file that every later Phase 3 script starts from.
#
# INPUT : STUDY_AREA_BBOX_LIDAR in .env  (min_lon,min_lat,max_lon,max_lat, WGS84)
# OUTPUT: data/lidar/raw/study_area.laz  (EPSG:32618, metres, NAVD88 heights)
#
# Run from the project root, in the environment that has PDAL:
#   conda activate geosense_lidar
#   python src/phase3_lidar/fetch_lidar_ept.py
# ============================================================
import json
import math
import os
import time

import pdal
from dotenv import load_dotenv

load_dotenv()

# USGS 3DEP collection covering the National Capital Region (Washington, D.C.)
DATASET = 'USGS_LPC_MD_VA_Sandy_NCR_2014_LAS_2015'
EPT_URL = f'https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{DATASET}/ept.json'
OUT_CRS = 'EPSG:32618'                      # WGS 84 / UTM zone 18N - metres
OUT_LAZ = 'data/lidar/raw/study_area.laz'

R_EARTH = 6378137.0                         # Web Mercator sphere radius (metres)


def lonlat_to_mercator(lon, lat):
    """Convert a WGS84 lon/lat pair to EPSG:3857 metres (the EPT archive grid)."""
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def read_bbox():
    """Read the LiDAR study box from .env: (min_lon, min_lat, max_lon, max_lat)."""
    raw = os.getenv('STUDY_AREA_BBOX_LIDAR')
    if not raw:
        raise SystemExit('STUDY_AREA_BBOX_LIDAR is not set in .env')
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in raw.split(','))
    return min_lon, min_lat, max_lon, max_lat


def build_pipeline(bbox, out_laz):
    """PDAL pipeline: EPT subset by bounds -> reproject to UTM -> write LAZ."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = lonlat_to_mercator(min_lon, min_lat)
    x1, y1 = lonlat_to_mercator(max_lon, max_lat)
    return [
        {
            'type': 'readers.ept',
            'filename': EPT_URL,
            'bounds': f'([{x0:.1f}, {x1:.1f}], [{y0:.1f}, {y1:.1f}])',
        },
        {
            'type': 'filters.reprojection',
            'in_srs': 'EPSG:3857',
            'out_srs': OUT_CRS,
        },
        {
            'type': 'writers.las',
            'filename': out_laz,
            'minor_version': 4,
            'a_srs': OUT_CRS,
        },
    ]


def main():
    os.makedirs(os.path.dirname(OUT_LAZ), exist_ok=True)
    bbox = read_bbox()
    print('Dataset :', DATASET)
    print('Box     :', bbox, '(min_lon, min_lat, max_lon, max_lat)')
    pipeline_def = build_pipeline(bbox, OUT_LAZ)
    print('EPT bounds (EPSG:3857):', pipeline_def[0]['bounds'])

    t0 = time.time()
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    count = pipeline.execute()
    elapsed = time.time() - t0

    size_mb = os.path.getsize(OUT_LAZ) / 1e6
    print(f'Fetched {count:,} points in {elapsed:.1f} s')
    print(f'Saved   {OUT_LAZ} ({size_mb:.1f} MB, {OUT_CRS})')

    # Quick look at what was fetched: extent and height range of the tile
    arr = pipeline.arrays[0]
    x, y, z = arr['X'], arr['Y'], arr['Z']
    print(f'Easting  : {x.min():.1f} - {x.max():.1f}')
    print(f'Northing : {y.min():.1f} - {y.max():.1f}')
    print(f'Elevation: {z.min():.1f} - {z.max():.1f} m')
    print('Returns  : up to', int(arr['NumberOfReturns'].max()), 'per pulse')
    area_m2 = float((x.max() - x.min()) * (y.max() - y.min()))
    print(f'Density  : {count / area_m2:.2f} points per m^2 over {area_m2 / 1e6:.2f} km^2')


if __name__ == '__main__':
    main()
