# fetch_aerial_naip.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 3 (Task E.3.3, data preparation)
# PURPOSE: Download a high-resolution aerial image of the LiDAR study tile for
#          the SAM rooftop step. The USDA NAIP programme photographs the whole of
#          the United States at 0.6 m and the imagery is in Google Earth Engine,
#          so the same Earth Engine account used in Phase 2 supplies it.
#
# INPUT : STUDY_AREA_BBOX_LIDAR and GEE_PROJECT from .env
# OUTPUT: data/raw/aerial/study_area_ortho.tif  (GeoTIFF, EPSG:32618, 0.6 m, RGB)
#         data/raw/aerial/study_area_ortho.png  (the same image for SAM / viewing)
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/fetch_aerial_naip.py
# ============================================================
import os
import time
from datetime import datetime, timezone

import ee
import numpy as np
import rasterio
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
PROJECT = os.getenv('GEE_PROJECT')
OUT_DIR = 'data/raw/aerial'
OUT_TIF = os.path.join(OUT_DIR, 'study_area_ortho.tif')
OUT_PNG = os.path.join(OUT_DIR, 'study_area_ortho.png')
CRS = 'EPSG:32618'                 # the LiDAR tile's CRS, so pixels and points line up
SCALE_M = 0.6                      # NAIP native resolution since 2018


def read_bbox():
    raw = os.getenv('STUDY_AREA_BBOX_LIDAR')
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in raw.split(','))
    return min_lon, min_lat, max_lon, max_lat


def main():
    ee.Initialize(project=PROJECT)
    min_lon, min_lat, max_lon, max_lat = read_bbox()
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    naip = (ee.ImageCollection('USDA/NAIP/DOQQ')
            .filterBounds(region)
            .filterDate('2018-01-01', '2025-12-31')
            .sort('system:time_start'))                     # latest image drawn on top
    dates = naip.aggregate_array('system:time_start').getInfo()
    print('NAIP scenes touching the tile:', len(dates))
    for t in sorted(set(dates)):
        print('  acquired', datetime.fromtimestamp(t / 1000, tz=timezone.utc).date())

    image = naip.mosaic().select(['R', 'G', 'B']).clip(region)
    url = image.getDownloadURL({
        'region': region, 'scale': SCALE_M, 'crs': CRS, 'format': 'GEO_TIFF'})
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    data = requests.get(url, timeout=600).content
    with open(OUT_TIF, 'wb') as f:
        f.write(data)
    print(f'Downloaded {len(data) / 1e6:.1f} MB in {time.time() - t0:.1f} s -> {OUT_TIF}')

    with rasterio.open(OUT_TIF) as src:
        rgb = src.read([1, 2, 3])                             # bands x rows x cols
        px, py = src.res
        print(f'Image size : {src.width} x {src.height} pixels, {src.count} bands')
        print(f'Pixel size : {px:.2f} x {py:.2f} m  ->  pixel area {px * py:.2f} m^2')
        print(f'Extent     : E {src.bounds.left:.1f}-{src.bounds.right:.1f}, '
              f'N {src.bounds.bottom:.1f}-{src.bounds.top:.1f} ({src.crs})')
    Image.fromarray(np.moveaxis(rgb, 0, -1).astype(np.uint8)).save(OUT_PNG)
    print('Saved', OUT_PNG)


if __name__ == '__main__':
    main()
