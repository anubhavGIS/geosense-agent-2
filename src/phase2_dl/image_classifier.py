# image_classifier.py
# ============================================================
# GeoSense Agent 2.0 - Phase 2 Core Module
# PURPOSE: Classify satellite imagery for any location in the study area
#
# INPUT : lat (float), lon (float), radius_m (int)
# OUTPUT: dict with land_cover, class_id, confidence_pct, ndvi, ndwi, ndvi_label,
#         class_distribution, change_flag, change_description (and a few extras)
#
# Run from the project root:
#   python src/phase2_dl/image_classifier.py                     (seven test locations)
#   python src/phase2_dl/image_classifier.py --hotspots          (largest NDVI changes)
#   python src/phase2_dl/image_classifier.py --lat 22.557 --lon 88.346 --radius 500
# ============================================================
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')      # two OpenMP runtimes on Windows
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
import torch
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / '.env')
sys.path.insert(0, str(Path(__file__).resolve().parent))

_CLEAN = _ROOT / 'data' / 'satellite' / 'clean'
_MODEL = _ROOT / 'models' / 'saved' / 'unet_best.pth'
_PMODEL = _ROOT / 'models' / 'saved' / 'prithvi_finetuned.pth'
_PMETA = _ROOT / 'models' / 'saved' / 'prithvi_finetuned_meta.json'
_COMPARISON = _ROOT / 'models' / 'evaluation' / 'model_comparison.json'

CLASSES = {0: 'Urban', 1: 'Vegetation', 2: 'Water', 3: 'Bare Land', 4: 'Agriculture'}
CURRENT_YEAR = os.getenv('GEOSENSE_CURRENT_YEAR', '2023')
BASELINE_YEAR = os.getenv('GEOSENSE_BASELINE_YEAR', '2018')  # harmonised S2 starts 2017
CHANGE_THRESHOLD = 0.15                                       # |dNDVI| above this = change
PATCH = 224                                                   # model input size (pixels)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BBOX = [float(os.getenv('BBOX_MIN_LON', '88.30')),
        float(os.getenv('BBOX_MIN_LAT', '22.45')),
        float(os.getenv('BBOX_MAX_LON', '88.42')),
        float(os.getenv('BBOX_MAX_LAT', '22.67'))]

_model, _model_name = None, None
_images, _meta, _transformer = {}, {}, None


# ── Model ────────────────────────────────────────────────────────────────────
def _choose_model():
    """Which checkpoint to load: GEOSENSE_MODEL=unet|prithvi overrides; otherwise the
    model with the higher mean IoU in the Exercise 4 comparison; otherwise the
    fine-tuned model if it exists, else the U-Net."""
    choice = os.getenv('GEOSENSE_MODEL', '').lower()
    if choice not in ('unet', 'prithvi'):
        choice = 'prithvi' if _PMODEL.exists() else 'unet'
        if _COMPARISON.exists():
            best = json.load(open(_COMPARISON)).get('higher_mean_iou', '')
            choice = 'unet' if 'U-Net' in best else 'prithvi'
    if choice == 'prithvi' and not _PMODEL.exists():
        choice = 'unet'
    if choice == 'unet' and not _MODEL.exists() and _PMODEL.exists():
        choice = 'prithvi'
    return choice


def _load_model():
    """Load the segmentation model once (lazy) and keep it in memory."""
    global _model, _model_name
    if _model is None:
        import segmentation_models_pytorch as smp
        choice = _choose_model()
        if choice == 'prithvi':
            from finetune_prithvi import (PrithviSegmenter, build_prithvi_encoder,
                                          get_prithvi_files)
            meta = json.load(open(_PMETA)) if _PMETA.exists() else {'backbone': 'prithvi'}
            if meta['backbone'] == 'prithvi':
                code_path = get_prithvi_files(weights=False)[0]
                m = PrithviSegmenter(build_prithvi_encoder(code_path))
                name = 'Prithvi-100M fine-tuned'
            else:
                m = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                             in_channels=6, classes=5)
                name = 'ResNet-50 U-Net fine-tuned'
            model_path = _PMODEL
        else:
            m = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                         in_channels=6, classes=5)
            name = 'U-Net ResNet-34'
            model_path = _MODEL
        if not model_path.exists():
            raise FileNotFoundError(f'No trained model found at {model_path}')
        m.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        m.eval().to(DEVICE)
        _model, _model_name = m, f'{name} ({model_path.name})'
        print(f'Loaded model: {_model_name} on {DEVICE}')
    return _model


# ── Imagery and geo-referencing ─────────────────────────────────────────────
def _load_image(year):
    """The cleaned composite of a year as a memory-mapped (bands, rows, cols) array."""
    if year not in _images:
        path = _CLEAN / f'clean_{year}.npy'
        if not path.exists():
            raise FileNotFoundError(f'Satellite data not found: {path}')
        _images[year] = np.load(str(path), mmap_mode='r')
        _meta[year] = json.load(open(_CLEAN / f'meta_{year}.json'))
    return _images[year], _meta[year]


def _to_pixel(lat, lon, meta):
    """lat/lon -> (row, col) of the composite, through its projected CRS (EPSG:32645)."""
    global _transformer
    if _transformer is None:
        from pyproj import Transformer
        _transformer = Transformer.from_crs('EPSG:4326', meta['crs'], always_xy=True)
    east, north = _transformer.transform(lon, lat)
    px, py = meta['pixel_size_m']
    x0, y0 = meta['origin']                                   # top-left corner
    return int((y0 - north) / py), int((east - x0) / px)


def _get_image_patch(lat, lon, radius_m, year=CURRENT_YEAR):
    """
    The 224 x 224 x 6 patch centred on a lat/lon location (shifted inwards at the edges).
    Returns the patch (bands, 224, 224) with NaN filled by the band mean, the window origin
    (r1, c1), a boolean mask of the pixels inside radius_m of the location, and the mask of
    pixels that held real data.
    """
    image, meta = _load_image(year)
    bands, rows, cols = image.shape
    r_centre, c_centre = _to_pixel(lat, lon, meta)
    if not (0 <= r_centre < rows and 0 <= c_centre < cols):
        raise ValueError(f'({lat}, {lon}) is outside the study area '
                         f'{BBOX} / {rows} x {cols} px composite')
    half = PATCH // 2
    r1 = min(max(0, r_centre - half), rows - PATCH)
    c1 = min(max(0, c_centre - half), cols - PATCH)
    patch = np.array(image[:, r1:r1 + PATCH, c1:c1 + PATCH], dtype=np.float32)
    valid = ~np.isnan(patch[0])
    for b in range(bands):
        m = np.nanmean(patch[b]) if valid.any() else 0.0
        patch[b] = np.where(np.isnan(patch[b]), m, patch[b])
    rr, cc = np.mgrid[0:PATCH, 0:PATCH]
    radius_px = max(1.0, radius_m / float(meta['pixel_size_m'][0]))
    centre = (r_centre - r1, c_centre - c1)
    inside = ((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2) <= radius_px ** 2
    return patch, (r1, c1), centre, inside & valid, valid


def _compute_ndvi_ndwi(patch):
    """Per-pixel NDVI and NDWI from a (bands, H, W) patch (B2, B3, B4, B8, B11, B12)."""
    nir, red = patch[3].astype(float), patch[2].astype(float)
    green = patch[1].astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
        ndwi = np.where((green + nir) > 0, (green - nir) / (green + nir), 0.0)
    return np.clip(ndvi, -1, 1), np.clip(ndwi, -1, 1)


def _ndvi_label(ndvi):
    if ndvi > 0.5:
        return 'Dense healthy forest/vegetation'
    if ndvi > 0.3:
        return 'Moderate healthy vegetation'
    if ndvi > 0.1:
        return 'Sparse or stressed vegetation'
    if ndvi > 0.0:
        return 'Very sparse vegetation / degraded land'
    return 'Urban surface, water, or bare rock'


# ── The core function ───────────────────────────────────────────────────────
def classify_imagery(lat: float, lon: float, radius_m: int = 1000) -> dict:
    """
    Classify satellite imagery around a location.

    Args:
        lat, lon : location (WGS 84)
        radius_m : radius of the area summarised (default 1000 m; the model always sees
                   the full 224 x 224 patch = 2.24 km, so radii above 1120 m are clipped)
    Returns a dict with:
        land_cover / class_id  - dominant class inside the radius
        confidence_pct         - mean softmax probability of the predicted class (pixels
                                 inside the radius)
        ndvi, ndwi, ndvi_label - mean spectral indices inside the radius (current year)
        class_distribution     - % of each class inside the radius
        change_flag            - True when |NDVI(current) - NDVI(baseline)| > 0.15
        change_description     - what changed, with the share of pixels that changed
        + model, patch window, centre-pixel class, inference time
    """
    t0 = time.perf_counter()
    model = _load_model()
    patch, (r1, c1), centre, inside, valid = _get_image_patch(lat, lon, radius_m,
                                                              CURRENT_YEAR)

    with torch.no_grad():                                  # model inference
        tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)      # (1, 6, 224, 224)
        outputs = model(tensor)                                       # (1, 5, 224, 224)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()        # (5, 224, 224)
        pred = probs.argmax(0)                                        # (224, 224)

    n_inside = int(inside.sum())
    counts = np.bincount(pred[inside], minlength=5)
    class_counts = {CLASSES[i]: round(float(counts[i]) / n_inside * 100, 1)
                    for i in range(5)}
    dominant_id = int(counts.argmax())
    confidence = round(float(probs.max(0)[inside].mean() * 100), 1)
    centre_id = int(pred[centre])

    ndvi_map, ndwi_map = _compute_ndvi_ndwi(patch)          # spectral indices
    ndvi = float(ndvi_map[inside].mean())
    ndwi = float(ndwi_map[inside].mean())

    change = {'change_flag': False,
              'change_description': f'No {BASELINE_YEAR} image available'}
    if (_CLEAN / f'clean_{BASELINE_YEAR}.npy').exists():    # change detection
        old_patch, _, _, old_inside, _ = _get_image_patch(lat, lon, radius_m,
                                                          BASELINE_YEAR)
        old_ndvi_map, _ = _compute_ndvi_ndwi(old_patch)
        both = inside & old_inside
        old_ndvi = float(old_ndvi_map[both].mean())
        diff = ndvi_map - old_ndvi_map
        ndvi_change = float(diff[both].mean())
        changed_pct = float((np.abs(diff[both]) > CHANGE_THRESHOLD).mean() * 100)
        flag = bool(abs(ndvi_change) > CHANGE_THRESHOLD)
        years = f'{BASELINE_YEAR}->{CURRENT_YEAR}'
        pixels = f'{changed_pct:.0f}% of pixels changed by more than {CHANGE_THRESHOLD}'
        if flag:
            direction = 'vegetation loss' if ndvi_change < 0 else 'vegetation gain'
            desc = (f'Significant land change detected {years}: {direction}, '
                    f'mean NDVI {old_ndvi:+.3f} -> {ndvi:+.3f} ({pixels})')
        else:
            desc = (f'No significant change detected {years} '
                    f'(mean NDVI {old_ndvi:+.3f} -> {ndvi:+.3f}, {pixels})')
        change = {'ndvi_baseline': round(old_ndvi, 3),
                  'ndvi_change': round(ndvi_change, 3),
                  'changed_pixel_pct': round(changed_pct, 1), 'change_flag': flag,
                  'change_description': desc}

    return {
        'latitude': lat, 'longitude': lon, 'radius_m': min(radius_m, PATCH // 2 * 10),
        'land_cover': CLASSES[dominant_id], 'class_id': dominant_id,
        'confidence_pct': confidence,
        'centre_pixel_class': CLASSES[centre_id],
        'ndvi': round(ndvi, 3), 'ndwi': round(ndwi, 3), 'ndvi_label': _ndvi_label(ndvi),
        'class_distribution': class_counts,
        **change,
        'model': _model_name, 'year': CURRENT_YEAR,
        'patch_window_rc': [r1, c1], 'pixels_in_radius': n_inside,
        'inference_time_s': round(time.perf_counter() - t0, 3),
    }


# ── Where did the land change most? ─────────────────────────────────────────
def find_change_hotspots(window_m=500, top=3):
    """Rank window_m x window_m blocks of the study area by the mean NDVI change between
    the baseline and the current composite; returns the strongest losses and gains."""
    from pyproj import Transformer
    _, meta = _load_image(CURRENT_YEAR)
    now = np.load(str(_CLEAN / f'ndvi_{CURRENT_YEAR}.npy'))
    old = np.load(str(_CLEAN / f'ndvi_{BASELINE_YEAR}.npy'))
    diff = now - old
    n = int(window_m / meta['pixel_size_m'][0])
    rows, cols = diff.shape[0] // n, diff.shape[1] // n
    blocks = np.nanmean(diff[:rows * n, :cols * n].reshape(rows, n, cols, n), axis=(1, 3))
    inv = Transformer.from_crs(meta['crs'], 'EPSG:4326', always_xy=True)
    px, py = meta['pixel_size_m']; x0, y0 = meta['origin']
    out = {'loss': [], 'gain': []}
    order = np.argsort(np.nan_to_num(blocks, nan=0.0), axis=None)
    for key, idx in (('loss', order[:top]), ('gain', order[::-1][:top])):
        for k in idx:
            r, c = divmod(int(k), cols)
            east = x0 + (c * n + n / 2) * px
            north = y0 - (r * n + n / 2) * py
            lon, lat = inv.transform(east, north)
            out[key].append({'lat': round(lat, 4), 'lon': round(lon, 4),
                             'mean_ndvi_change': round(float(blocks[r, c]), 3)})
    return out


# ── Test run ─────────────────────────────────────────────────────────────────
TEST_LOCATIONS = [                  # Kolkata study area (88.30-88.42 E, 22.45-22.67 N)
    ('Maidan (Brigade Parade Ground)', 22.5570, 88.3460),
    ('B.B.D. Bagh (central business district)', 22.5710, 88.3480),
    ('Rabindra Sarobar lake', 22.5115, 88.3610),
    ('Hooghly river off Howrah', 22.5900, 88.3440),
    ('East Kolkata Wetlands (bheries)', 22.5470, 88.4150),
    ('Garden Reach / Kolkata port', 22.5370, 88.3110),
    ('Southern fringe near Joka', 22.4600, 88.3050),
]


def _print_result(result):
    for k, v in result.items():
        print(f'  {k:22s}: {v}')


if __name__ == '__main__':
    args = sys.argv[1:]
    print('=' * 60)
    print('GeoSense Agent 2.0 - image_classifier.py Test')
    print('=' * 60)
    if '--hotspots' in args:
        hs = find_change_hotspots(window_m=500, top=3)
        for key in ('loss', 'gain'):
            print(f'Largest NDVI {key} {BASELINE_YEAR}->{CURRENT_YEAR} (500 m blocks):')
            for h in hs[key]:
                print(f"  lat {h['lat']:.4f}, lon {h['lon']:.4f}: "
                      f"mean NDVI change {h['mean_ndvi_change']:+.3f}")
        locations = [(f"NDVI {key} hotspot", h['lat'], h['lon'])
                     for key in ('loss', 'gain') for h in hs[key][:1]]
    elif '--lat' in args:
        lat = float(args[args.index('--lat') + 1])
        lon = float(args[args.index('--lon') + 1])
        locations = [(f'({lat}, {lon})', lat, lon)]
    else:
        locations = TEST_LOCATIONS
    radius = int(args[args.index('--radius') + 1]) if '--radius' in args else 500
    _load_model()                                          # cold start outside the timing
    for name, lat, lon in locations:
        print(f'\n{name} - classify_imagery({lat}, {lon}, radius_m={radius})')
        t0 = time.perf_counter()
        result = classify_imagery(lat, lon, radius_m=radius)
        elapsed = time.perf_counter() - t0
        _print_result(result)
        print(f'  -> {elapsed:.3f} s {"(under 3 s)" if elapsed < 3 else "(over 3 s!)"}')
