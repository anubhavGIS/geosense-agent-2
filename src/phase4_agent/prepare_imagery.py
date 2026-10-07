# prepare_imagery.py
# Purpose: the imagery layers of the digital twin
#   1. Sentinel-2 composites of the time-slider years: 2018 and 2023 come from Phase II;
#      any other year is downloaded with the Phase II functions (same bands, 10 m grid,
#      cloud filter and annual median)
#   2. a true-colour 8-bit GeoTIFF per year (B4 B3 B2 + alpha) with ONE stretch for all
#      years, so that a change of colour on the slider is a change on the ground
#   3. an NDVI colour overlay of 2023 (BrBG, -0.2 to 0.8, transparent outside the data)
#   4. z/x/y PNG tile pyramids (zoom 10-15) in frontend/digital_twin/public/imagery/<name>/
#      cut with gdal2tiles (OSGeo4W Shell of QGIS), or with --tile by this script
# Run from the project root (geosense environment):
#   python src/phase4_agent/prepare_imagery.py          GeoTIFFs + the gdal2tiles commands
#   python src/phase4_agent/prepare_imagery.py --tile   GeoTIFFs + tiles cut here
import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

RAW = Path('data/satellite/raw')
OUT = Path('data/twin')
TILES = Path('frontend/digital_twin/public/imagery')
YEARS = [2018, 2021, 2023, 2025]
REFERENCE_YEAR = 2023                 # the stretch is measured on this year
ZOOM = (10, 15)                       # zoom 15 = 4.4 m per pixel at 22.5 N
NODATA = -32768                       # Phase II nodata value
ORIGIN = 20037508.342789244           # half the width of the Web Mercator world (m)
B2, B3, B4, B8 = 1, 2, 3, 4           # band order of the Phase II GeoTIFFs


def composite(year):
    """Path of the Phase II composite of a year, downloaded first if it is missing."""
    path = RAW / f'sentinel2_Kolkata_{year}.tif'
    if not path.exists():
        sys.path.append('src/phase2_dl')
        import download_sentinel2 as s2                              # Phase II functions
        s2.ee.Initialize(project=s2.GEE_PROJECT)
        print(f'{year}: composite not found - downloading with the Phase II script')
        s2.download_image(s2.get_clean_image(year), year, s2.BANDS)
    return path


def stretch_limits(path):
    """2nd and 98th percentile of B4, B3, B2 over the valid pixels of one composite."""
    with rasterio.open(path) as src:
        bands = src.read([B4, B3, B2]).astype('float32')
    valid = (bands != NODATA).all(axis=0)
    return [np.percentile(b[valid], (2, 98)) for b in bands]


def write_rgba(path, rgba, src):
    """Write a 4-band 8-bit GeoTIFF (red, green, blue, alpha) on the grid of src."""
    profile = src.profile.copy()
    profile.update(count=4, dtype='uint8', nodata=None, compress='deflate',
                   photometric='RGB')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(rgba)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                           ColorInterp.alpha]


def true_colour(year, limits):
    """data/twin/rgb_<year>.tif - B4 B3 B2 stretched with the common limits."""
    out = OUT / f'rgb_{year}.tif'
    with rasterio.open(composite(year)) as src:
        bands = src.read([B4, B3, B2]).astype('float32')
        valid = (bands != NODATA).all(axis=0)
        rgb = [np.clip((b - lo) / (hi - lo), 0, 1) * 255 for b, (lo, hi) in
               zip(bands, limits)]
        alpha = np.where(valid, 255, 0)
        write_rgba(out, np.stack(rgb + [alpha]).astype('uint8'), src)
    print(f'{year}: {out} ({100 * valid.mean():.1f} % of the grid has data)')
    return out


def ndvi_overlay(year):
    """data/twin/ndvi_<year>.tif - NDVI coloured with BrBG, transparent without data."""
    out = OUT / f'ndvi_{year}.tif'
    with rasterio.open(composite(year)) as src:
        red, nir = src.read(B4).astype('float32'), src.read(B8).astype('float32')
        valid = (red != NODATA) & (nir != NODATA) & (red + nir > 0)
        ndvi = np.where(valid, (nir - red) / np.where(valid, nir + red, 1), np.nan)
        colours = matplotlib.colormaps['BrBG'](np.clip((ndvi + 0.2) / 1.0, 0, 1))
        rgba = (colours * 255).astype('uint8').transpose(2, 0, 1)
        rgba[3] = np.where(valid, 210, 0)
        write_rgba(out, rgba, src)
    print(f'NDVI {year}: {out} (mean NDVI {np.nanmean(ndvi):.3f})')
    return out


def tile_range(bounds, z):
    """Columns and rows of the XYZ tiles that cover lon/lat bounds at zoom z."""
    west, south, east, north = bounds
    n = 2 ** z

    def column(lon):
        return int((lon + 180) / 360 * n)

    def row(lat):
        r = math.radians(lat)
        return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)

    return column(west), column(east), row(north), row(south)


def cut_tiles(tif, name):
    """
    Cut a 4-band GeoTIFF into 256 px PNG tiles (Web Mercator, XYZ numbering).

    Args:
        tif (Path): an RGBA GeoTIFF written by true_colour() or ndvi_overlay()
        name (str): folder name under public/imagery (e.g. '2018' or 'ndvi_2023')

    Returns:
        int: the number of tiles written

    Example:
        cut_tiles(Path('data/twin/rgb_2018.tif'), '2018')
    """
    with rasterio.open(tif) as src:
        data, crs, transform = src.read(), src.crs, src.transform
        bounds = transform_bounds(crs, 'EPSG:4326', *src.bounds)
    count = 0
    for z in range(ZOOM[0], ZOOM[1] + 1):
        x0, x1, y0, y1 = tile_range(bounds, z)
        size = 2 * ORIGIN / 2 ** z
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                tile = np.zeros((4, 256, 256), dtype='uint8')
                left, top = -ORIGIN + x * size, ORIGIN - y * size
                reproject(data, tile, src_transform=transform, src_crs=crs,
                          dst_transform=from_bounds(left, top - size, left + size, top,
                                                    256, 256),
                          dst_crs='EPSG:3857', resampling=Resampling.bilinear)
                if not tile[3].any():
                    continue                                   # outside the study area
                path = TILES / name / str(z) / str(x) / f'{y}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile.transpose(1, 2, 0), 'RGBA').save(path, optimize=True)
                count += 1
    return count


def main(years, tile):
    OUT.mkdir(parents=True, exist_ok=True)
    limits = stretch_limits(composite(REFERENCE_YEAR))
    print('Stretch (2nd-98th percentile of ' + str(REFERENCE_YEAR) + '): ' + ', '.join(
        f'{b} {lo:.0f}-{hi:.0f}' for b, (lo, hi) in zip(('B4', 'B3', 'B2'), limits)))
    layers = [(true_colour(year, limits), str(year)) for year in years]
    layers.append((ndvi_overlay(REFERENCE_YEAR), f'ndvi_{REFERENCE_YEAR}'))
    if tile:
        for tif, name in layers:
            t0 = time.time()
            n = cut_tiles(tif, name)
            print(f'  tiles {name}: {n} PNG tiles, zoom {ZOOM[0]}-{ZOOM[1]} '
                  f'({time.time() - t0:.0f} s)')
    else:
        print('\nCut the tiles in the OSGeo4W Shell (QGIS), from the project root:')
        for tif, name in layers:
            print(f'gdal2tiles --xyz --zoom={ZOOM[0]}-{ZOOM[1]} --webviewer=none '
                  f'--resampling=bilinear --processes=4 {tif} {TILES / name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imagery layers of the digital twin.')
    parser.add_argument('--years', type=int, nargs='+', default=YEARS)
    parser.add_argument('--tile', action='store_true', help='cut the tiles with Python')
    args = parser.parse_args()
    main(args.years, args.tile)
