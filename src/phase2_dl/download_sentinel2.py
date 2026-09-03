# download_sentinel2.py
# Purpose: Download cloud-free Sentinel-2 composites for the study area as GeoTIFF
# Run from project root: python src/phase2_dl/download_sentinel2.py
import math
import os
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from dotenv import load_dotenv
from pyproj import Transformer

load_dotenv()

# Study area - Kolkata bounding box, read from .env (min_lon, min_lat, max_lon, max_lat)
BBOX = [
    float(os.getenv('BBOX_MIN_LON', '88.30')),
    float(os.getenv('BBOX_MIN_LAT', '22.45')),
    float(os.getenv('BBOX_MAX_LON', '88.42')),
    float(os.getenv('BBOX_MAX_LAT', '22.67')),
]
STUDY_NAME = os.getenv('STUDY_AREA', 'Kolkata')
GEE_PROJECT = os.getenv('GEE_PROJECT', 'ee-anubg2000')
EXPORT_CRS = 'EPSG:32645'      # WGS 84 / UTM zone 45N - the project's metric grid, 10 m square pixels
SCALE = 10                     # metres per pixel - Sentinel-2 resolution

# Two years for change detection. The harmonised surface-reflectance collection
# begins in 2017, so 2018 (its first complete year) is the baseline against 2023.
YEARS = [2018, 2023]
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']   # Blue, Green, Red, NIR, SWIR-1, SWIR-2

# Earth Engine's direct download accepts at most 50,331,648 bytes per request and
# counts 3 bytes per 16-bit band pixel (2 data + 1 mask): 18 bytes per 6-band pixel.
# The 10 m grid of the study area (1257 x 2448 px) needs 55.4 MB, so the export is
# cut into horizontal strips that are downloaded one by one and stitched locally.
MAX_REQUEST_BYTES = 50_331_648
BYTES_PER_PIXEL = 3 * len(BANDS)

OUT_DIR = Path('data/satellite/raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_study_area():
    """Returns a GEE geometry for the study area bounding box (WGS 84)."""
    return ee.Geometry.Rectangle(BBOX)


def utm_grid():
    """Bounding box of the study area in the export CRS, snapped to the 10 m grid."""
    tr = Transformer.from_crs('EPSG:4326', EXPORT_CRS, always_xy=True)
    n, pts = 50, []
    for i in range(n + 1):                          # sample every edge, not just the corners
        f = i / n
        pts.append((BBOX[0] + f * (BBOX[2] - BBOX[0]), BBOX[1]))
        pts.append((BBOX[0] + f * (BBOX[2] - BBOX[0]), BBOX[3]))
        pts.append((BBOX[0], BBOX[1] + f * (BBOX[3] - BBOX[1])))
        pts.append((BBOX[2], BBOX[1] + f * (BBOX[3] - BBOX[1])))
    xs, ys = zip(*[tr.transform(x, y) for x, y in pts])
    minx = math.floor(min(xs) / SCALE) * SCALE
    miny = math.floor(min(ys) / SCALE) * SCALE
    maxx = math.ceil(max(xs) / SCALE) * SCALE
    maxy = math.ceil(max(ys) / SCALE) * SCALE
    return minx, miny, maxx, maxy


def plan_tiles():
    """Split the grid into horizontal strips that each fit one download request."""
    minx, miny, maxx, maxy = utm_grid()
    width = int((maxx - minx) / SCALE)
    height = int((maxy - miny) / SCALE)
    max_rows = int(MAX_REQUEST_BYTES * 0.9 / (width * BYTES_PER_PIXEL))
    n_tiles = math.ceil(height / max_rows)
    rows_per_tile = math.ceil(height / n_tiles)
    tiles = []
    for i in range(n_tiles):
        top = maxy - i * rows_per_tile * SCALE
        bottom = max(miny, top - rows_per_tile * SCALE)
        tiles.append((minx, bottom, maxx, top))
    print(f'  Grid {width} x {height} px at {SCALE} m = {width * height * BYTES_PER_PIXEL / 1e6:.1f} MB '
          f'-> {n_tiles} strip(s) of <= {rows_per_tile} rows')
    return tiles


def get_clean_image(year: int) -> ee.Image:
    """
    Cloud-free Sentinel-2 composite for one year: the median of every scene
    with < 10% cloud cover. Median = the middle value at each pixel, which
    drops clouds and shadows that appear in only some of the scenes.
    """
    region = get_study_area()
    start, end = f'{year}-01-01', f'{year}-12-31'
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'QA60']))
    count = collection.size().getInfo()
    print(f'  Year {year}: found {count} clean images')
    if count == 0:
        raise ValueError(f'No clean images found for {year}. Try a different year.')
    median = collection.median().clip(region)    # combines all clean images
    return median


def download_image(image: ee.Image, year: int, bands: list) -> str:
    """Download a GEE image to local disk as one GeoTIFF (strip by strip, then stitched)."""
    out_path = OUT_DIR / f'sentinel2_{STUDY_NAME}_{year}.tif'
    if out_path.exists():
        print(f'  Already downloaded: {out_path}')
        return str(out_path)
    print(f'  Downloading {year} image to: {out_path}')
    # Reflectance is an integer 0-10000, so the composite is exported as 16-bit.
    export = image.select(bands).toInt16()
    tile_paths = []
    for i, (minx, miny, maxx, maxy) in enumerate(plan_tiles(), start=1):
        tile_path = OUT_DIR / f'_tile_{year}_{i}.tif'
        region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], EXPORT_CRS, False)
        print(f'  Strip {i}: rows {int((maxy - miny) / SCALE)} ...')
        geemap.ee_export_image(export, filename=str(tile_path), scale=SCALE,
                               crs=EXPORT_CRS, region=region, file_per_band=False)
        if not tile_path.exists():
            raise RuntimeError(f'Export of strip {i} failed - no file was written (see the message above)')
        tile_paths.append(tile_path)
    stitch(tile_paths, out_path)
    for p in tile_paths:
        p.unlink()
    size_mb = out_path.stat().st_size / 1e6
    print(f'  Downloaded: {out_path} ({size_mb:.1f} MB)')
    return str(out_path)


def stitch(tile_paths, out_path):
    """Place the strips on one 10 m grid (by their georeferencing) and write a single GeoTIFF."""
    strips = []
    for p in tile_paths:
        with rasterio.open(str(p)) as src:
            strips.append((src.read(), src.transform, src.profile))
    top = max(tr.f for _, tr, _ in strips)
    left = min(tr.c for _, tr, _ in strips)
    height = max(int(round((top - tr.f) / SCALE)) + arr.shape[1] for arr, tr, _ in strips)
    width = max(int(round((tr.c - left) / SCALE)) + arr.shape[2] for arr, tr, _ in strips)
    # Pixels outside the clipped study area come back masked; Earth Engine writes
    # them as -32768 in a 16-bit export, so that value is kept as the nodata tag.
    NODATA = -32768
    mosaic = np.full((len(BANDS), height, width), NODATA, dtype='int16')
    for arr, tr, _ in strips:
        r0 = int(round((top - tr.f) / SCALE))
        c0 = int(round((tr.c - left) / SCALE))
        mosaic[:, r0:r0 + arr.shape[1], c0:c0 + arr.shape[2]] = arr.astype('int16')
    profile = strips[0][2].copy()
    profile.update(driver='GTiff', height=height, width=width, count=len(BANDS), dtype='int16',
                   crs=EXPORT_CRS, transform=rasterio.Affine(SCALE, 0, left, 0, -SCALE, top),
                   nodata=NODATA, compress='deflate', tiled=False)
    with rasterio.open(str(out_path), 'w', **profile) as dst:
        dst.write(mosaic)
        dst.descriptions = tuple(BANDS)
    print(f'  Stitched {len(strips)} strip(s) -> {width} x {height} px, {len(BANDS)} bands, {EXPORT_CRS}')


def main():
    print('Initialising Google Earth Engine...')
    ee.Initialize(project=GEE_PROJECT)
    print(f'Study area: {STUDY_NAME} | BBOX: {BBOX} | export CRS: {EXPORT_CRS} at {SCALE} m')
    for year in YEARS:
        print(f'\nProcessing year {year}...')
        try:
            image = get_clean_image(year)
            path = download_image(image, year, BANDS)
            print(f'  SUCCESS: {path}')
        except Exception as e:
            print(f'  ERROR for {year}: {e}')
    print('\nDownload complete!')
    print(f'Files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
