# sam_rooftop_extraction.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 3 (Task E.3.3)
# PURPOSE: Outline rooftops in the aerial image with Meta's Segment Anything Model
#          (automatic mask generator), keep the rooftop-sized masks, match every
#          footprint to the LiDAR heights of the building points that fall inside it,
#          and estimate building volume = footprint area x height.
#
# INPUT : data/raw/aerial/study_area_ortho.png   (the image SAM sees)
#         data/raw/aerial/study_area_ortho.tif   (pixel size and extent, EPSG:32618)
#         data/lidar/processed/study_area_classified.laz (heights above ground)
#         SAM_CHECKPOINT_PATH in .env (models/saved/sam/sam_vit_b_01ec64.pth)
# OUTPUT: outputs/rooftops/rooftop_candidates.json  (one record per candidate)
#         outputs/rooftops/sam_rooftops.png          (outlines + zoom on rooftop 1)
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/sam_rooftop_extraction.py                 (GPU if available)
#   python src/phase3_lidar/sam_rooftop_extraction.py --device cpu    (timing comparison)
#   python src/phase3_lidar/sam_rooftop_extraction.py --max-area 14000 (larger roofs too)
#   python src/phase3_lidar/sam_rooftop_extraction.py --min-area-m2 10 --max-area-m2 5000
#       (Exercise 5: the window in square metres, converted with the image's pixel size)
# ============================================================
import argparse
import json
import os
import sys
import time

# Windows/conda: torch and the MKL in numpy each bring an OpenMP runtime (OMP Error #15)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2                                                        # noqa: E402
import matplotlib                                                 # noqa: E402
import numpy as np                                                # noqa: E402
import rasterio                                                   # noqa: E402
import torch                                                      # noqa: E402
from dotenv import load_dotenv                                    # noqa: E402
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lidar_features import load_classified_cloud, classify_points     # noqa: E402

load_dotenv()
CHECKPOINT = os.getenv('SAM_CHECKPOINT_PATH', 'models/saved/sam/sam_vit_b_01ec64.pth')
IMAGE_PNG = 'data/raw/aerial/study_area_ortho.png'
IMAGE_TIF = 'data/raw/aerial/study_area_ortho.tif'
CLOUD = 'data/lidar/processed/study_area_classified.laz'
OUT_DIR = 'outputs/rooftops'
MIN_AREA_PX, MAX_AREA_PX = 40, 4000      # rooftop-sized masks (manual's thresholds)
POINTS_PER_SIDE = 48                     # prompt grid: 48 x 48 points over the image
ROOF_HAG_MIN = 2.5                       # LiDAR points below this are not roof points


def load_sam_generator(device='cpu'):
    """ViT-B SAM from the checkpoint in .env, moved to `device`, wrapped as a generator."""
    sam = sam_model_registry['vit_b'](checkpoint=CHECKPOINT)
    sam.to(device)                        # raises if a CUDA device is asked for but absent
    return SamAutomaticMaskGenerator(sam, points_per_side=POINTS_PER_SIDE)


def detect_rooftops(image_path, generator, min_area=MIN_AREA_PX, max_area=MAX_AREA_PX):
    """Run SAM on the aerial image; keep medium-sized masks (likely rooftops)."""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    t0 = time.time()
    masks = generator.generate(image)
    elapsed = time.time() - t0
    candidates = [m for m in masks if min_area < m['area'] < max_area]
    candidates.sort(key=lambda m: m['area'], reverse=True)
    print(f'SAM produced {len(masks)} masks in {elapsed:.1f} s')
    print(f'Detected {len(candidates)} candidate rooftop segments '
          f'({min_area} < area < {max_area} px)')
    return candidates, image, elapsed


def estimate_building_volume(rooftop_mask, avg_height_m, pixel_area_m2=0.25):
    """Footprint area (pixels x pixel area) times a height = building volume."""
    footprint_area_m2 = rooftop_mask['area'] * pixel_area_m2
    volume_m3 = round(footprint_area_m2 * avg_height_m, 1)
    return {'footprint_area_m2': round(footprint_area_m2, 1),
            'height_m': round(float(avg_height_m), 2), 'volume_m3': volume_m3}


def image_georeference(tif_path):
    """Pixel size (m) and the affine transform of the orthophoto from its GeoTIFF."""
    with rasterio.open(tif_path) as src:
        px, py = src.res
        return px * py, src.transform, (src.height, src.width)


def lidar_heights_per_mask(candidates, transform, shape, cloud_path):
    """Median HAG of the roof-level LiDAR points inside each footprint."""
    x, y, z, hag, num_returns, return_num = load_classified_cloud(cloud_path)
    labels = classify_points(hag, num_returns, return_num)
    tile_avg = float(hag[labels == 2].mean())             # Task 3.1 tile average
    # rows/cols of every point on the image grid (transform maps col,row -> x,y)
    cols = ((x - transform.c) / transform.a).astype(int)
    rows = ((y - transform.f) / transform.e).astype(int)
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    roof = inside & (hag >= ROOF_HAG_MIN)
    owner = np.full(shape, -1, dtype=np.int32)             # which candidate owns a pixel
    for i, m in enumerate(candidates):                     # small masks painted last
        owner[m['segmentation']] = i
    ids = owner[rows[roof], cols[roof]]
    heights = hag[roof]
    order = np.argsort(ids, kind='stable')
    ids, heights = ids[order], heights[order]
    starts = np.searchsorted(ids, np.arange(len(candidates)))
    ends = np.searchsorted(ids, np.arange(len(candidates)), side='right')
    stats = []
    for i in range(len(candidates)):
        pts = heights[starts[i]:ends[i]]
        stats.append((int(len(pts)), float(np.median(pts)) if len(pts) else None))
    return stats, tile_avg


def plot_results(image, candidates, records, pixel_area, out_png, min_area, max_area):
    """Left: every candidate outlined on the ortho. Right: zoom on rooftop 1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
    ax1.imshow(image)
    for m in candidates:
        contours, _ = cv2.findContours(m['segmentation'].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.reshape(-1, 2)
            ax1.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color='yellow',
                     linewidth=0.5)
    ax1.set_title(f'{len(candidates)} candidate rooftops (SAM ViT-B, '
                  f'{min_area}-{max_area} px = '
                  f'{min_area * pixel_area:.0f}-{max_area * pixel_area:.0f} m2)',
                  fontsize=10)
    ax1.axis('off')
    first = candidates[0]
    x0, y0, w, h = first['bbox']
    pad = 60
    r0, r1 = max(0, int(y0) - pad), min(image.shape[0], int(y0 + h) + pad)
    c0, c1 = max(0, int(x0) - pad), min(image.shape[1], int(x0 + w) + pad)
    ax2.imshow(image[r0:r1, c0:c1])
    ax2.imshow(np.ma.masked_where(~first['segmentation'][r0:r1, c0:c1],
                                  first['segmentation'][r0:r1, c0:c1]),
               cmap='autumn', alpha=0.45)
    rec = records[0]
    ax2.set_title(f"rooftop 1: {rec['footprint_area_m2']:.0f} m2 x "
                  f"{rec['height_m']} m (LiDAR, {rec['lidar_points']} pts) = "
                  f"{rec['volume_m3']:,.0f} m3", fontsize=10)
    ax2.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print('Saved', out_png)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--min-area', type=int, default=MIN_AREA_PX, help='pixels')
    ap.add_argument('--max-area', type=int, default=MAX_AREA_PX, help='pixels')
    ap.add_argument('--min-area-m2', type=float, default=None,
                    help='window in m2 instead of pixels (resolution-independent)')
    ap.add_argument('--max-area-m2', type=float, default=None)
    args = ap.parse_args()
    pixel_area, transform, shape = image_georeference(IMAGE_TIF)
    print(f'Pixel area from the GeoTIFF: {pixel_area:.2f} m2 '
          f'(pixel {transform.a:.2f} x {-transform.e:.2f} m)')
    if args.min_area_m2 is not None or args.max_area_m2 is not None:
        # the window is a size in the world, not in the image: convert it here
        if args.min_area_m2 is not None:
            args.min_area = int(round(args.min_area_m2 / pixel_area))
        if args.max_area_m2 is not None:
            args.max_area = int(round(args.max_area_m2 / pixel_area))
        print(f'Area window {args.min_area_m2}-{args.max_area_m2} m2 = '
              f'{args.min_area}-{args.max_area} px at {pixel_area:.2f} m2 per pixel')
        tag = f'_m2_{int(args.max_area_m2 or 0)}'
    else:
        tag = '' if args.max_area == MAX_AREA_PX else f'_max{args.max_area}'
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f'Device: {device} ({name}); checkpoint {CHECKPOINT} '
          f'({os.path.getsize(CHECKPOINT) / 1e6:.1f} MB)')

    t0 = time.time()
    generator = load_sam_generator(device)
    print(f'SAM ViT-B loaded in {time.time() - t0:.1f} s')
    candidates, image, sam_seconds = detect_rooftops(IMAGE_PNG, generator,
                                                     args.min_area, args.max_area)
    stats, tile_avg = lidar_heights_per_mask(candidates, transform, shape, CLOUD)
    print(f'Tile-average building height from the LiDAR (Task 3.1): {tile_avg:.2f} m')

    records = []
    for i, (m, (n_pts, h_lidar)) in enumerate(zip(candidates, stats), start=1):
        height = h_lidar if h_lidar is not None else tile_avg
        rec = estimate_building_volume(m, height, pixel_area_m2=pixel_area)
        rec.update({'rooftop': i, 'area_px': int(m['area']),
                    'bbox_px': [int(v) for v in m['bbox']],
                    'lidar_points': n_pts,
                    'height_source': 'lidar' if h_lidar is not None else 'tile average',
                    'predicted_iou': round(float(m['predicted_iou']), 3)})
        records.append(rec)

    print(f'{"#":>3} {"px":>5} {"m2":>7} {"pts":>5} {"height":>7} {"volume m3":>10}  '
          'source')
    for rec in records[:10]:
        print(f"{rec['rooftop']:3d} {rec['area_px']:5d} {rec['footprint_area_m2']:7.1f} "
              f"{rec['lidar_points']:5d} {rec['height_m']:7.2f} "
              f"{rec['volume_m3']:10,.1f}  {rec['height_source']}")
    first = records[0]
    print('First rooftop with its own LiDAR height :', {k: first[k] for k in
          ('footprint_area_m2', 'height_m', 'volume_m3')})
    print('Same footprint with the tile average    :',
          estimate_building_volume(candidates[0], tile_avg, pixel_area_m2=pixel_area))
    print("Same footprint with the manual's 8.4 m  :",
          estimate_building_volume(candidates[0], 8.4, pixel_area_m2=pixel_area))
    with_lidar = sum(1 for r in records if r['height_source'] == 'lidar')
    print(f'{with_lidar} of {len(records)} candidates have roof-level LiDAR points inside')

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f'rooftop_candidates{tag}.json')
    with open(json_path, 'w') as f:
        json.dump({'device': device, 'sam_seconds': round(sam_seconds, 1),
                   'pixel_area_m2': pixel_area, 'tile_avg_building_height_m': tile_avg,
                   'area_limits_px': [args.min_area, args.max_area],
                   'area_limits_m2': [round(args.min_area * pixel_area, 1),
                                      round(args.max_area * pixel_area, 1)],
                   'candidates': records}, f, indent=1)
    print('Saved', json_path)
    plot_results(image, candidates, records, pixel_area,
                 os.path.join(OUT_DIR, f'sam_rooftops{tag}.png'), args.min_area,
                 args.max_area)
    print(f'Done in {time.time() - t0:.1f} s (SAM {sam_seconds:.1f} s on {device})')
