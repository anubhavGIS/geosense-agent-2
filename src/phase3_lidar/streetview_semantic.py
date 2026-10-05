# streetview_semantic.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 4 (Task E.4.1); Exercise 5 additions: the
#          person class, frozen query snapshots + nearest-first selection (so that a
#          location analysed twice gives the same answer), enclosed scenes (tunnels,
#          car parks) skipped, smaller box on a 5xx answer
# PURPOSE: Describe a place from street level. Every pixel of a street photograph
#          is labelled with the pretrained SegFormer-B0 (Cityscapes classes), the
#          share of road / sidewalk / building / vegetation / sky / person / car is
#          counted, the shares of the images near a coordinate are averaged, and a
#          transparent walkability score is computed from them.
#
# INPUT : street images (data/streetview/images/*.jpg) or a coordinate, for which
#         up to 10 Mapillary images are fetched through mapillary_download.py
# OUTPUT: outputs/streetview/<name>.json and, with --figure, a PNG of the images
#         beside their segmentation
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/streetview_semantic.py --images data/streetview/images/A.jpg
#   python src/phase3_lidar/streetview_semantic.py                 (the default locations)
#   python src/phase3_lidar/streetview_semantic.py --lat 22.553 --lon 88.352
# ============================================================
import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')              # Windows OpenMP clash

import numpy as np                                                # noqa: E402
import requests                                                   # noqa: E402
import torch                                                      # noqa: E402
from PIL import Image                                             # noqa: E402
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapillary_download import find_images_in_bbox, download_image  # noqa: E402

MODEL_NAME = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
MODEL_VARIANTS = {'b0': MODEL_NAME,
                  'b3': 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
                  'b5': 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'}
CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'building', 8: 'vegetation', 10: 'sky',
               11: 'person', 13: 'car'}        # person: evidence of walking (Ex 4 Note 2)
ALL_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle']
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32]], dtype=np.uint8)
DEFAULT_WEIGHTS = {'sidewalk': 0.4, 'vegetation': 0.3, 'car': -0.2}
OUT_DIR = 'outputs/streetview'
IMAGE_DIR = 'data/streetview/images'
QUERY_DIR = 'data/streetview/results'     # frozen Mapillary answers, one per box
# Default locations: the Chennai example of the manual and three Kolkata places
# covered by the images of Exercise 2 (Park Street area) and by Phase I/II.
DEFAULT_LOCATIONS = [('Chennai centre', 13.0827, 80.2707),
                     ('Kolkata - Park Street / Camac Street', 22.5530, 88.3540),
                     ('Kolkata - Chowringhee (J. L. Nehru Road)', 22.5535, 88.3515),
                     ('Kolkata - Shakespeare Sarani', 22.5495, 88.3540)]

_processor, _model, _device = None, None, None                   # loaded once, reused


def _load_model(variant='b0', device=None):
    """Load the processor and the SegFormer model once (lazy, cached in globals)."""
    global _processor, _model, _device
    if _model is None:
        name = MODEL_VARIANTS[variant]
        _device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        t0 = time.time()
        try:                                  # the cached copy needs no network at all
            _processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
            _model = SegformerForSemanticSegmentation.from_pretrained(
                name, local_files_only=True)
        except Exception:                     # first use on this machine: download
            _processor = AutoImageProcessor.from_pretrained(name)
            _model = SegformerForSemanticSegmentation.from_pretrained(name)
        _model = _model.to(_device).eval()
        n_params = sum(p.numel() for p in _model.parameters())
        print(f'Loaded {name} ({n_params / 1e6:.1f} M parameters) on {_device} '
              f'in {time.time() - t0:.1f} s')
    return _processor, _model


def segment_image(image_path, variant='b0', device=None):
    """Label every pixel: returns the class-id map at the image's own resolution."""
    processor, model = _load_model(variant, device)
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt').to(_device)
    with torch.no_grad():
        logits = model(**inputs).logits                           # 1 x 19 x H/4 x W/4
    logits = torch.nn.functional.interpolate(logits, size=image.size[::-1],
                                             mode='bilinear', align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8), image


def classify_batch(image_paths, variant='b0', device=None):
    """One forward pass for several images (used to show the memory cost of a batch)."""
    processor, model = _load_model(variant, device)
    images = [Image.open(p).convert('RGB') for p in image_paths]
    inputs = processor(images=images, return_tensors='pt').to(_device)
    with torch.no_grad():
        logits = model(**inputs).logits                           # N x 19 x h x w
    labels = logits.argmax(dim=1).cpu().numpy()
    return [{name: round(100.0 * float((lab == cid).sum()) / lab.size, 1)
             for cid, name in CLASS_NAMES.items()} for lab in labels]


def classify_image(image_path, variant='b0', device=None):
    """Percentage of pixels in each of the seven classes of interest (one decimal)."""
    labels, _ = segment_image(image_path, variant, device)
    total = labels.size
    return {name: round(100.0 * float((labels == cid).sum()) / total, 1)
            for cid, name in CLASS_NAMES.items()}


def compute_walkability_score(composition, weights=None):
    """(sidewalk x 0.4 + vegetation x 0.3 - car x 0.2) / 5, clipped to 0-10."""
    w = weights or DEFAULT_WEIGHTS
    raw = sum(composition.get(k, 0.0) * v for k, v in w.items()) / 5.0
    return round(float(min(10.0, max(0.0, raw))), 2)


def _query_images(lat, lon, radius_deg, candidates=50, refresh=False):
    """
    Mapillary's answer for the box around (lat, lon), frozen on first use.

    The archive grows and the API returns a varying subset of a box when it holds more
    images than one answer carries, so the same question would otherwise get a
    different set of photographs every time. The first answer is saved to
    data/streetview/results/query_<lat>_<lon>_r<radius>.json with its time stamp and
    reused until refresh=True. Returns (images, radius actually used, time of the query).
    """
    os.makedirs(QUERY_DIR, exist_ok=True)
    path = os.path.join(QUERY_DIR, f'query_{lat:.4f}_{lon:.4f}_r{radius_deg:g}.json')
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            snap = json.load(f)
        return snap['images'], snap['radius_deg'], snap['queried_at']
    radius, images, error = radius_deg, [], None
    for attempt in range(3):              # dense archives answer 500: shrink the box
        bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
        try:
            images = find_images_in_bbox(*bbox, limit=candidates)
            break
        except requests.HTTPError as err:
            if err.response is None or err.response.status_code < 500:
                raise
            error, radius = err, radius / 2
            print(f'Mapillary answered {err.response.status_code}; retrying with a '
                  f'{radius} deg box')
    else:
        raise error
    queried_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(path, 'w') as f:
        json.dump({'latitude': lat, 'longitude': lon, 'requested_radius_deg': radius_deg,
                   'radius_deg': radius, 'candidates': candidates,
                   'queried_at': queried_at, 'images': images}, f, indent=1)
    return images, radius, queried_at


def _nearest_first(images, lat, lon, max_images):
    """The max_images images closest to the point, ties broken by id."""
    cos_lat = np.cos(np.radians(lat))

    def distance2(meta):
        x, y = meta['geometry']['coordinates']
        return ((x - lon) * cos_lat) ** 2 + (y - lat) ** 2

    return sorted(images, key=lambda m: (distance2(m), m['id']))[:max_images]


def analyze_location(lat, lon, radius_deg=0.0015, max_images=10, variant='b0',
                     candidates=50, refresh=False):
    """
    Average the scene composition of the max_images open-air images nearest (lat, lon).

    The candidates are walked outward from the point; an image with neither sky nor
    vegetation in it (a tunnel, an underpass, a car park, a shop) is counted as
    'enclosed' and skipped - it photographs the inside of something, not the street.
    """
    images, radius, queried_at = _query_images(lat, lon, radius_deg, candidates, refresh)
    if not images:
        return {'latitude': lat, 'longitude': lon, 'images_analyzed': 0,
                'archive_snapshot': queried_at,
                'error': f'no Mapillary imagery within {radius} deg of the point'}
    comps, used, enclosed = [], [], []
    for meta in _nearest_first(images, lat, lon, len(images)):
        if len(comps) >= max_images:
            break
        path = download_image(meta, IMAGE_DIR)
        try:
            comp = classify_image(path, variant)
        except OSError:                       # not an image: the signed link had expired
            os.remove(path)
            if refresh:
                raise
            return analyze_location(lat, lon, radius_deg, max_images, variant,
                                    candidates, refresh=True)
        if comp['sky'] + comp['vegetation'] == 0.0:
            enclosed.append(meta['id'])
            continue
        comps.append(comp)
        used.append(meta['id'])
    if not comps:
        return {'latitude': lat, 'longitude': lon, 'images_analyzed': 0,
                'images_enclosed': len(enclosed), 'archive_snapshot': queried_at,
                'error': f'only enclosed scenes ({len(enclosed)} images) near the point'}
    composition = {k: round(float(np.mean([c[k] for c in comps])), 1) for k in comps[0]}
    return {'latitude': lat, 'longitude': lon, 'images_analyzed': len(comps),
            'images_enclosed': len(enclosed),
            'scene_composition': composition,
            'walkability_score': compute_walkability_score(composition),
            'radius_deg': radius, 'archive_snapshot': queried_at,
            'image_ids': used, 'enclosed_ids': enclosed}


def save_figure(image_paths, out_png, variant='b0'):
    """Each image beside its segmentation, with the class shares in the title."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(image_paths), 2, figsize=(10, 3.4 * len(image_paths)))
    axes = np.atleast_2d(axes)
    for row, path in enumerate(image_paths):
        labels, image = segment_image(path, variant)
        comp = classify_image(path, variant)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(os.path.basename(path), fontsize=9)
        axes[row, 1].imshow(PALETTE[labels])
        axes[row, 1].set_title('  '.join(f'{k} {v}%' for k, v in comp.items()), fontsize=8)
        for ax in axes[row]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print('Saved', out_png)


def parse_weights(text):
    """'sidewalk=0.3,vegetation=0.4,car=-0.3' -> dict."""
    return {k.strip(): float(v) for k, v in (item.split('=') for item in text.split(','))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='*', help='classify these images and stop')
    ap.add_argument('--lat', type=float)
    ap.add_argument('--lon', type=float)
    ap.add_argument('--radius', type=float, default=0.0015, help='degrees')
    ap.add_argument('--max-images', type=int, default=10)
    ap.add_argument('--model', default='b0', choices=list(MODEL_VARIANTS))
    ap.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    ap.add_argument('--alt-weights',
                    default='sidewalk=0.3,vegetation=0.4,person=0.2,car=-0.3',
                    help='second weight set for the comparison, name=value,...')
    ap.add_argument('--figure', action='store_true', help='save the segmentation PNG')
    ap.add_argument('--batch', type=int, default=1,
                    help='images per forward pass in --images mode (1 = one by one)')
    ap.add_argument('--refresh', action='store_true',
                    help='ask Mapillary again instead of reusing the frozen answer')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    if args.images:
        # cmd.exe does not expand wildcards, so the script does it itself
        args.images = [p for pat in args.images for p in (sorted(glob.glob(pat)) or [pat])]
        rows = []
        if args.batch > 1:                                        # whole batches at once
            for i in range(0, len(args.images), args.batch):
                chunk = args.images[i:i + args.batch]
                t1 = time.time()
                comps = classify_batch(chunk, args.model, args.device)
                dt = (time.time() - t1) / len(chunk)
                rows += [(os.path.basename(p), c, dt) for p, c in zip(chunk, comps)]
        else:
            for path in args.images:
                t1 = time.time()
                comp = classify_image(path, args.model, args.device)
                rows.append((os.path.basename(path), comp, time.time() - t1))
        head = f'{"image":22s}' + ''.join(f'{k:>11s}' for k in CLASS_NAMES.values())
        print(head + f'{"score":>7s}{"s":>6s}')
        for name, comp, dt in rows:
            print(f'{name:22s}' + ''.join(f'{comp[k]:11.1f}' for k in CLASS_NAMES.values())
                  + f'{compute_walkability_score(comp):7.2f}{dt:6.2f}')
        with open(os.path.join(OUT_DIR, 'image_compositions.json'), 'w') as f:
            json.dump([{'image': n, 'composition': c, 'seconds': round(dt, 2),
                        'walkability_score': compute_walkability_score(c)}
                       for n, c, dt in rows], f, indent=1)
        if args.figure:
            save_figure(args.images, os.path.join(OUT_DIR, 'segformer_examples.png'),
                        args.model)
        print(f'Done in {time.time() - t0:.1f} s')
        sys.exit(0)

    if args.lat is not None and args.lon is not None:
        locations = [(f'{args.lat:.4f}, {args.lon:.4f}', args.lat, args.lon)]
    else:
        locations = DEFAULT_LOCATIONS
    alt = parse_weights(args.alt_weights)
    results = []
    for name, lat, lon in locations:
        try:
            res = analyze_location(lat, lon, args.radius, args.max_images, args.model,
                                   refresh=args.refresh)
        except requests.RequestException as err:      # no network, DNS, 5xx three times
            msg = re.sub(r'access_token=[^&\s]+', 'access_token=***', str(err))[:160]
            res = {'latitude': lat, 'longitude': lon, 'images_analyzed': 0,
                   'error': f'{type(err).__name__}: {msg}'}
        res['name'] = name
        if 'error' not in res:
            res['walkability_score_alt'] = compute_walkability_score(
                res['scene_composition'], alt)
        results.append(res)
        print(json.dumps(res, indent=1))
    print(f'\n{"location":42s}{"imgs":>5s}{"road":>7s}{"sidewk":>7s}{"bldg":>7s}'
          f'{"veg":>7s}{"sky":>7s}{"person":>7s}{"car":>7s}{"score":>7s}{"alt":>7s}')
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:42s}{0:5d}  {r["error"]}')
            continue
        c = r['scene_composition']
        print(f'{r["name"]:42s}{r["images_analyzed"]:5d}'
              + ''.join(f'{c[k]:7.1f}' for k in CLASS_NAMES.values())
              + f'{r["walkability_score"]:7.2f}{r["walkability_score_alt"]:7.2f}')
    print(f'default weights {DEFAULT_WEIGHTS}; alternative weights {alt}')
    tag = 'default_locations' if args.lat is None else f'{args.lat:.4f}_{args.lon:.4f}'
    out = os.path.join(OUT_DIR, f'locations_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'radius_deg': args.radius, 'max_images': args.max_images,
                   'default_weights': DEFAULT_WEIGHTS, 'alt_weights': alt,
                   'results': results}, f, indent=1)
    print('Saved', out)
    print(f'Done in {time.time() - t0:.1f} s')
