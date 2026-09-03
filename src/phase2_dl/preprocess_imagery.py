# preprocess_imagery.py
# Purpose: Clean and normalise satellite imagery for deep learning
# Run from project root: python src/phase2_dl/preprocess_imagery.py
import json
from pathlib import Path

import numpy as np
import rasterio

SAT_DIR = Path('data/satellite')
RAW_DIR = SAT_DIR / 'raw'
CLEAN_DIR = SAT_DIR / 'clean'
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'B11_SWIR1', 'B12_SWIR2']
BAND_INDICES = {'B2': 0, 'B3': 1, 'B4': 2, 'B8': 3, 'B11': 4, 'B12': 5}


def remove_clouds(image_array: np.ndarray, cloud_threshold: float = 3000.0) -> np.ndarray:
    """
    Mark cloud pixels as NaN so they are excluded from analysis.
    Cloud pixels are very bright in every band: reflectance > 3000 in all
    six Sentinel-2 surface-reflectance bands is treated as cloud.
    Pixels outside the exported footprint are nodata: Earth Engine writes them as
    -32768 in a 16-bit export (an all-zero pixel is treated the same way).
    """
    cloud_mask = np.all(image_array > cloud_threshold, axis=0)
    nodata_mask = np.any(image_array <= -32768, axis=0) | np.all(image_array == 0, axis=0)
    cloud_mask &= ~nodata_mask
    clean = image_array.astype(float)
    clean[:, cloud_mask | nodata_mask] = np.nan
    cloud_pct = cloud_mask.mean() * 100
    nodata_pct = nodata_mask.mean() * 100
    print(f'  Cloud coverage removed: {cloud_pct:.2f}% of pixels '
          f'(nodata edge: {nodata_pct:.2f}%)')
    return clean


def normalise_to_01(image_array: np.ndarray) -> np.ndarray:
    """
    Normalise pixel values from the Sentinel-2 range (0-10000) to 0-1.
    Neural networks train best on small values.
    """
    normalised = np.clip(image_array / 10000.0, 0, 1)
    print(f'  Value range after normalisation: {np.nanmin(normalised):.3f} '
          f'to {np.nanmax(normalised):.3f}')
    return normalised


def compute_ndvi(image_array: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    NIR = Band 8 (index 3), Red = Band 4 (index 2)
    """
    nir = image_array[BAND_INDICES['B8']].astype(float)
    red = image_array[BAND_INDICES['B4']].astype(float)
    denominator = nir + red
    denominator[denominator == 0] = np.nan          # avoid division by zero
    ndvi = (nir - red) / denominator
    return np.clip(ndvi, -1, 1)


def compute_ndwi(image_array: np.ndarray) -> np.ndarray:
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Green = Band 3 (index 1), NIR = Band 8 (index 3). High NDWI = water.
    """
    green = image_array[BAND_INDICES['B3']].astype(float)
    nir = image_array[BAND_INDICES['B8']].astype(float)
    denom = green + nir
    denom[denom == 0] = np.nan
    return np.clip((green - nir) / denom, -1, 1)


def process_file(tif_path: Path) -> dict:
    """Full preprocessing pipeline for one GeoTIFF file."""
    print(f'\nProcessing: {tif_path.name}')
    with rasterio.open(str(tif_path)) as src:
        image = src.read()                           # shape: (bands, rows, cols)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    print(f'  Shape: {image.shape} | dtype: {image.dtype} | CRS: {crs} | '
          f'pixel: {transform.a:.1f} x {-transform.e:.1f} m')

    clean = remove_clouds(image)                     # step 1: remove clouds
    normalised = normalise_to_01(clean)              # step 2: normalise to 0-1
    ndvi = compute_ndvi(clean)                       # step 3: spectral indices
    ndwi = compute_ndwi(clean)

    year = tif_path.stem.split('_')[-1]
    clean_path = CLEAN_DIR / f'clean_{year}.npy'     # step 4: save cleaned image
    np.save(str(clean_path), normalised)
    np.save(str(CLEAN_DIR / f'ndvi_{year}.npy'), ndvi)   # step 5: save indices
    np.save(str(CLEAN_DIR / f'ndwi_{year}.npy'), ndwi)

    stats = {                                        # step 6: metadata
        'year': year,
        'source': tif_path.name,
        'shape': list(image.shape),
        'crs': str(crs),
        'pixel_size_m': [float(transform.a), float(-transform.e)],
        'origin': [float(transform.c), float(transform.f)],
        'ndvi_mean': float(np.nanmean(ndvi)),
        'ndvi_std': float(np.nanstd(ndvi)),
        'ndvi_min': float(np.nanmin(ndvi)),
        'ndvi_max': float(np.nanmax(ndvi)),
        'ndwi_mean': float(np.nanmean(ndwi)),
        'ndwi_std': float(np.nanstd(ndwi)),
        'cloud_free_pct': float(np.mean(~np.isnan(normalised[0])) * 100),
    }
    meta_path = CLEAN_DIR / f'meta_{year}.json'
    with open(str(meta_path), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  NDVI mean: {stats["ndvi_mean"]:.3f} (std {stats["ndvi_std"]:.3f}) | '
          f'NDWI mean: {stats["ndwi_mean"]:.3f} | Cloud-free: {stats["cloud_free_pct"]:.1f}%')
    print(f'  Saved: {clean_path}, ndvi_{year}.npy, ndwi_{year}.npy, meta_{year}.json')
    return stats


def main():
    tif_files = sorted(RAW_DIR.glob('sentinel2_*.tif'))
    if not tif_files:
        print('No .tif files found in data/satellite/raw/')
        return
    print(f'Found {len(tif_files)} files to process: {[f.name for f in tif_files]}')
    results = [process_file(f) for f in tif_files]
    print('\nSummary:')
    print(f'  {"year":>6} {"ndvi_mean":>10} {"ndwi_mean":>10} {"cloud_free_%":>13}')
    for s in results:
        print(f'  {s["year"]:>6} {s["ndvi_mean"]:>10.3f} {s["ndwi_mean"]:>10.3f} '
              f'{s["cloud_free_pct"]:>13.1f}')
    print('\nPreprocessing complete!')


if __name__ == '__main__':
    main()
