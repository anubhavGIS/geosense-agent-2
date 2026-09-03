# generate_labels.py
# Purpose: Create land-cover labels for every pixel of the cleaned composites using
#          spectral rules (rule-based pseudo-labelling), so that image chips can be
#          paired with label patches for U-Net training.
# Land cover classes: 0=Urban, 1=Vegetation, 2=Water, 3=Bare Land, 4=Agriculture
# Run from project root: python src/phase2_dl/generate_labels.py
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLEAN_DIR = Path('data/satellite/clean')
LABEL_DIR = Path('data/satellite/labels')
LABEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    0: 'Urban / Built-up',
    1: 'Dense Vegetation / Forest',
    2: 'Water Body',
    3: 'Bare Land / Soil',
    4: 'Agriculture / Crops',
}
NODATA = 255                          # label for pixels outside the study area (ignored in training)
YEARS = ['2018', '2023']

# Two rule sets share the same class list and the same order of tests.
#   generic : the textbook thresholds (NDWI > 0.3 water; NDVI > 0.4 vegetation;
#             NDVI > 0.15 agriculture; NDVI < 0.05 and SWIR > 0.2 urban; else bare)
#   kolkata : thresholds calibrated on the study-area composites. Kolkata's built-up
#             fabric has NDVI 0.05-0.35 at 10 m (street trees inside every pixel) and
#             a SWIR-1 reflectance around 0.17, so the generic urban test (NDVI < 0.05
#             and SWIR > 0.2) matches almost nothing. Built-up land is instead taken
#             where SWIR-1 exceeds NIR - the Normalised Difference Built-up Index
#             NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) is positive - and NDVI is below 0.40.
RULE_SET = 'kolkata'


def classify_pixel(ndvi: float, ndwi: float, nir: float, swir: float, rule_set: str = RULE_SET) -> int:
    """
    Classify a single pixel using spectral rules (reflectance inputs are 0-1).
    Returns the class id; NODATA (255) when the inputs are not finite.
    """
    if not (np.isfinite(ndvi) and np.isfinite(ndwi) and np.isfinite(nir) and np.isfinite(swir)):
        return NODATA
    if rule_set == 'generic':
        if ndwi > 0.3:                       return 2   # Water - high NDWI
        if ndvi > 0.4:                       return 1   # Dense vegetation - high NDVI
        if ndvi > 0.15:                      return 4   # Agriculture - moderate NDVI
        if ndvi < 0.05 and swir > 0.2:       return 0   # Urban - low NDVI, high SWIR
        return 3                                        # Default: bare land
    # kolkata rule set
    ndbi = (swir - nir) / (swir + nir) if (swir + nir) != 0 else 0.0
    if ndwi > 0.3 or ndvi < 0.0:             return 2   # Water - high NDWI or negative NDVI
    if ndvi > 0.4:                           return 1   # Dense vegetation / tree canopy
    if ndbi > 0.0:                           return 0   # Built-up - SWIR-1 above NIR, NDVI below 0.4
    if ndvi > 0.15:                          return 4   # Vegetated but not built-up and not dense
    return 3                                            # Bare soil, sand, construction ground


def classify_array(ndvi, ndwi, nir, swir, rule_set: str = RULE_SET) -> np.ndarray:
    """Vectorised form of classify_pixel - identical rules applied to whole arrays."""
    valid = np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(nir) & np.isfinite(swir)
    label = np.full(ndvi.shape, NODATA, dtype=np.uint8)
    if rule_set == 'generic':
        label[valid] = 3
        label[valid & (ndvi < 0.05) & (swir > 0.2)] = 0
        label[valid & (ndvi > 0.15)] = 4
        label[valid & (ndvi > 0.4)] = 1
        label[valid & (ndwi > 0.3)] = 2
        return label
    with np.errstate(divide='ignore', invalid='ignore'):
        ndbi = np.where((swir + nir) != 0, (swir - nir) / (swir + nir), 0.0)
    label[valid] = 3
    label[valid & (ndvi > 0.15)] = 4
    label[valid & (ndbi > 0.0)] = 0
    label[valid & (ndvi > 0.4)] = 1
    label[valid & ((ndwi > 0.3) | (ndvi < 0.0))] = 2
    return label


PALETTE = {0: (170, 60, 60), 1: (20, 110, 40), 2: (40, 90, 200), 3: (215, 190, 130), 4: (150, 205, 90), 255: (255, 255, 255)}


def save_preview(label_mask: np.ndarray, path: Path):
    """Colour PNG of a label mask (for quick visual checking and the lab record)."""
    rgb = np.zeros(label_mask.shape + (3,), dtype=np.uint8)
    for cls, colour in PALETTE.items():
        rgb[label_mask == cls] = colour
    Image.fromarray(rgb).save(str(path))


def class_distribution(label_mask: np.ndarray) -> dict:
    valid = label_mask != NODATA
    return {cls_id: float((label_mask == cls_id).sum() / valid.sum() * 100) for cls_id in CLASSES}


def create_label_mask(year: str, rule_set: str = RULE_SET, save: bool = True) -> np.ndarray:
    ndvi = np.load(str(CLEAN_DIR / f'ndvi_{year}.npy'))
    ndwi = np.load(str(CLEAN_DIR / f'ndwi_{year}.npy'))
    clean = np.load(str(CLEAN_DIR / f'clean_{year}.npy'))
    nir = clean[3]                    # B8
    swir = clean[4]                   # B11 (SWIR-1)
    label_mask = classify_array(ndvi, ndwi, nir, swir, rule_set)

    # spot-check: the vectorised rules must agree with classify_pixel everywhere sampled
    rng = np.random.default_rng(0)
    rows, cols = ndvi.shape
    rs, cs = rng.integers(0, rows, 5000), rng.integers(0, cols, 5000)
    mismatches = sum(classify_pixel(ndvi[r, c], ndwi[r, c], nir[r, c], swir[r, c], rule_set) != label_mask[r, c]
                     for r, c in zip(rs, cs))
    print(f'  [{rule_set}] classify_pixel spot-check on 5,000 random pixels: {mismatches} mismatches')

    dist = class_distribution(label_mask)
    for cls_id, cls_name in CLASSES.items():
        print(f'  Class {cls_id} ({cls_name:25s}): {dist[cls_id]:5.1f}%')
    print(f'  Nodata (outside study area): {(label_mask == NODATA).mean() * 100:.1f}% of the grid')

    if save:
        np.save(str(LABEL_DIR / f'labels_{year}.npy'), label_mask)
        save_preview(label_mask, LABEL_DIR / f'labels_{year}_preview.png')
        print(f'  Saved: labels_{year}.npy, labels_{year}_preview.png')
    return label_mask


if __name__ == '__main__':
    report = {}
    for year in YEARS:
        print(f'\nGenerating labels for {year}...')
        print(f' Generic thresholds (for comparison):')
        generic = create_label_mask(year, 'generic', save=False)
        np.save(str(LABEL_DIR / f'labels_{year}_generic.npy'), generic)
        print(f' Kolkata-calibrated thresholds (used for training):')
        mask = create_label_mask(year, 'kolkata', save=True)
        report[year] = {'generic': class_distribution(generic), 'kolkata': class_distribution(mask)}
    with open(str(LABEL_DIR / 'class_distribution.json'), 'w') as f:
        json.dump({'classes': CLASSES, 'rule_set_used': RULE_SET, 'distribution_pct': report}, f, indent=2)
    print('\nLabel generation complete!')
