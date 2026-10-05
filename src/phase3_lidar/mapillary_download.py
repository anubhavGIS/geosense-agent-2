# mapillary_download.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 2 (Task 2.2)
# PURPOSE: Find and download street-level images inside a bounding box using
#          the Mapillary API v4 (crowd-sourced, free; 360-degree panoramas are
#          flagged by the API as is_pano).
#
# INPUT : MAPILLARY_TOKEN in .env; a bounding box min_lon, min_lat, max_lon, max_lat
# OUTPUT: data/streetview/images/<image_id>.jpg for every image found
#         data/streetview/results/images_<bbox>.json  (the metadata of the batch)
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/mapillary_download.py
#   python src/phase3_lidar/mapillary_download.py --bbox 88.345 22.548 88.355 22.558 \
#          --limit 20
# ============================================================
import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('MAPILLARY_TOKEN')
BASE_URL = 'https://graph.mapillary.com/images'
FIELDS = 'id,thumb_2048_url,geometry,compass_angle,captured_at,is_pano'

# Default test area: central Kolkata - Park Street / Chowringhee, about 1 km x 1 km
DEFAULT_BBOX = (88.345, 22.548, 88.355, 22.558)
IMAGE_DIR = 'data/streetview/images'
RESULT_DIR = 'data/streetview/results'


def find_images_in_bbox(min_lon, min_lat, max_lon, max_lat, limit=50):
    """Query Mapillary for image metadata inside a bounding box (WGS84 degrees)."""
    params = {
        'access_token': TOKEN,
        'fields': FIELDS,
        'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
        'limit': limit,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()                 # 401 here means the token is wrong
    return resp.json().get('data', [])


def download_image(image_meta, save_dir=IMAGE_DIR):
    """Download one image (the 2048-px thumbnail) to <save_dir>/<image_id>.jpg."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{image_meta['id']}.jpg")
    if os.path.exists(filepath):            # never download the same image twice
        return filepath
    img_data = requests.get(image_meta['thumb_2048_url'], timeout=120).content
    with open(filepath, 'wb') as f:
        f.write(img_data)
    return filepath


def describe(image_meta):
    """One readable line per image: id, date, position, heading, panorama flag."""
    lon, lat = image_meta['geometry']['coordinates']
    captured = datetime.fromtimestamp(image_meta['captured_at'] / 1000, tz=timezone.utc)
    pano = '360' if image_meta.get('is_pano') else 'flat'
    heading = image_meta.get('compass_angle', 0)
    return (f"{image_meta['id']}  {captured:%Y-%m-%d}  "
            f"lat {lat:.5f} lon {lon:.5f}  heading {heading:5.1f}  {pano}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Mapillary images for a bbox')
    parser.add_argument('--bbox', nargs=4, type=float, default=DEFAULT_BBOX,
                        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'))
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    if not TOKEN:
        raise SystemExit('MAPILLARY_TOKEN is missing from .env')

    t0 = time.time()
    images = find_images_in_bbox(*args.bbox, limit=args.limit)
    print(f'Found {len(images)} images in bbox {args.bbox}')
    for img in images:
        print(' ', describe(img))

    saved = []
    for img in images:
        path = download_image(img)
        saved.append(path)
        print('Saved:', path, f'({os.path.getsize(path) / 1e3:.0f} kB)')

    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = '_'.join(f'{v:.3f}' for v in args.bbox)
    meta_path = os.path.join(RESULT_DIR, f'images_{tag}.json')
    with open(meta_path, 'w') as f:
        json.dump(images, f, indent=2)

    n_pano = sum(1 for img in images if img.get('is_pano'))
    print(f'{len(saved)} images saved to {IMAGE_DIR} ({n_pano} are 360-degree panoramas)')
    print(f'Metadata written to {meta_path}; total time {time.time() - t0:.1f} s')
