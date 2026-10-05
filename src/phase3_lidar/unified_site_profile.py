# unified_site_profile.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 5 (Task E.5.2)
# PURPOSE: One structured record per location that merges what every phase knows
#          about it: Phase 1 (site_scorer - opportunity score with its SHAP
#          explanation), Phase 2 (image_classifier - land cover, NDVI, change since
#          2018), Phase 3 (lidar_analyzer - building height, canopy, impervious
#          surface, slope risk; streetview_semantic - scene composition and
#          walkability). Each source is called only where its data exist, and the
#          profile records which sources contributed and why the others did not: the
#          agent of Phase 4 must know what it does not know.
#
# INPUT : latitude, longitude (WGS84); bbox_radius_deg for the LiDAR box (0.005)
# OUTPUT: outputs/reports/enhanced_site_profile_<lat>_<lon>.json
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/unified_site_profile.py               (three demo sites)
#   python src/phase3_lidar/unified_site_profile.py --lat 38.9097 --lon -77.0434 ^
#          --name "Washington - Dupont Circle"
# ============================================================
import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')              # Windows OpenMP clash

from dotenv import load_dotenv                                    # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]                                          # the project folder
for sub in ('phase1_ml', 'phase2_dl', 'phase3_lidar'):            # sibling phase modules
    sys.path.append(str(_ROOT / 'src' / sub))
load_dotenv(_ROOT / '.env')

from lidar_analyzer import analyze_bbox                           # noqa: E402
from streetview_semantic import analyze_location                  # noqa: E402

OUT_DIR = 'outputs/reports'
_TOKEN = re.compile(r'access_token=[^&\s]+')                      # never store a token


def _redact(err):
    """An exception as a short string with any API token blanked out."""
    return _TOKEN.sub('access_token=***', f'{type(err).__name__}: {err}')[:160]
# Phases 1 and 2 were built for the Kolkata study area (BBOX_* in .env, Phase 2 Ex 1)
STUDY_BBOX = [float(os.getenv('BBOX_MIN_LON', '88.30')),
              float(os.getenv('BBOX_MIN_LAT', '22.45')),
              float(os.getenv('BBOX_MAX_LON', '88.42')),
              float(os.getenv('BBOX_MAX_LAT', '22.67'))]
DEMO_SITES = [('Chennai centre (manual example)', 13.0827, 80.2707),
              ('Washington - Dupont Circle', 38.9097, -77.0434),
              ('Kolkata - Park Street / Camac Street', 22.5530, 88.3540)]


@dataclass
class EnhancedSiteProfile:
    """The unified profile: the eight fields of Task E.5.2 first, then the extensions."""
    latitude: float
    longitude: float
    avg_building_height_m: float
    canopy_cover_pct: float
    impervious_surface_pct: float
    slope_risk_score: float
    scene_composition: dict
    walkability_score: Optional[float]
    # --- Phase 3 provenance ---
    site_name: str = ''
    lidar_coverage: bool = False
    lidar_coverage_pct: float = 0.0
    lidar_points_analyzed: int = 0
    lidar_bbox: Optional[list] = None
    street_images_analyzed: int = 0
    street_images_enclosed: int = 0                     # tunnels, car parks: skipped
    street_archive_snapshot: Optional[str] = None       # when Mapillary was asked
    # --- Phase 1: site_scorer.py (Random Forest + SHAP) ---
    phase1_opportunity_score: Optional[float] = None
    phase1_risk_score: Optional[float] = None
    phase1_verdict: Optional[str] = None
    phase1_features: Optional[dict] = None
    phase1_shap_explanation: Optional[dict] = None
    # --- Phase 2: image_classifier.py (U-Net on Sentinel-2) ---
    phase2_land_cover: Optional[str] = None
    phase2_confidence_pct: Optional[float] = None
    phase2_ndvi: Optional[float] = None
    phase2_ndvi_label: Optional[str] = None
    phase2_class_distribution: Optional[dict] = None
    phase2_change_flag: Optional[bool] = None
    phase2_change_description: Optional[str] = None
    # --- what contributed ---
    sources: dict = field(default_factory=dict)
    bbox_radius_deg: float = 0.005
    generated_at: str = ''


def _in_study_area(lat, lon):
    """Is the point inside the Kolkata box that Phases 1 and 2 were built for?"""
    w, s, e, n = STUDY_BBOX
    return w <= lon <= e and s <= lat <= n


def phase1_score(lat, lon):
    """Phase 1 opportunity score, or (None, reason) when it cannot be computed here."""
    if not _in_study_area(lat, lon):
        return None, f'outside the Phase 1 study area {STUDY_BBOX}'
    try:
        from site_scorer import score_location                    # PostGIS + model .pkl
        return score_location(lat, lon), 'ok'
    except Exception as err:                                      # DB down, model missing
        return None, f'unavailable: {_redact(err)}'


def phase2_classify(lat, lon, radius_m=500):
    """Phase 2 land cover around the point, or (None, reason)."""
    if not _in_study_area(lat, lon):
        return None, f'outside the Phase 2 composite {STUDY_BBOX}'
    try:
        from image_classifier import classify_imagery             # U-Net, Sentinel-2
        return classify_imagery(lat, lon, radius_m=radius_m), 'ok'
    except Exception as err:
        return None, f'unavailable: {_redact(err)}'


def build_profile(lat, lon, bbox_radius_deg=0.005, site_name='', street_radius_deg=0.0015,
                  max_images=10):
    """
    Build the unified enhanced site profile of a location.

    Args:
        lat, lon (float): the location in WGS84
        bbox_radius_deg (float): half-width of the LiDAR box around the point (0.005
            deg = about 1.1 km north-south)
        site_name (str): a label stored in the profile
        street_radius_deg (float), max_images (int): the Mapillary search of Exercise 4

    Returns:
        dict: asdict(EnhancedSiteProfile) - the eight fields of Task E.5.2, the Phase 1
              and Phase 2 results where the point lies in their study area, and
              'sources', which says for every phase whether it contributed

    Example:
        profile = build_profile(38.9097, -77.0434, site_name='Dupont Circle')
        print(profile['avg_building_height_m'], profile['walkability_score'])
    """
    sources = {}
    lidar = analyze_bbox(lat - bbox_radius_deg, lon - bbox_radius_deg,
                         lat + bbox_radius_deg, lon + bbox_radius_deg)
    sources['phase3_lidar_analyzer'] = (f'ok ({lidar["coverage_pct"]} % of the box '
                                        f'covered)' if lidar['lidar_coverage']
                                        else lidar.get('note', 'no coverage'))
    try:
        street = analyze_location(lat, lon, street_radius_deg, max_images)
    except Exception as err:                                      # token, network
        street = {'images_analyzed': 0, 'error': _redact(err)}
    sources['phase3_streetview_semantic'] = ('ok (%d images)' % street['images_analyzed']
                                             if 'error' not in street else street['error'])
    p1, why1 = phase1_score(lat, lon)
    sources['phase1_site_scorer'] = why1
    p2, why2 = phase2_classify(lat, lon)
    sources['phase2_image_classifier'] = why2

    profile = EnhancedSiteProfile(
        latitude=lat, longitude=lon,
        avg_building_height_m=lidar['avg_building_height_m'],
        canopy_cover_pct=lidar['canopy_cover_pct'],
        impervious_surface_pct=lidar['impervious_surface_pct'],
        slope_risk_score=lidar['slope_risk_score'],
        scene_composition=street.get('scene_composition', {}),
        walkability_score=street.get('walkability_score'),
        site_name=site_name,
        lidar_coverage=lidar['lidar_coverage'], lidar_coverage_pct=lidar['coverage_pct'],
        lidar_points_analyzed=lidar['total_points_analyzed'],
        lidar_bbox=[round(v, 6) for v in lidar['bbox']],
        street_images_analyzed=street.get('images_analyzed', 0),
        street_images_enclosed=street.get('images_enclosed', 0),
        street_archive_snapshot=street.get('archive_snapshot'),
        sources=sources, bbox_radius_deg=bbox_radius_deg,
        generated_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    if p1 is not None:
        profile.phase1_opportunity_score = p1['opportunity_score']
        profile.phase1_risk_score = p1['risk_score']
        profile.phase1_verdict = p1['verdict']
        profile.phase1_features = {k: round(float(v), 1)
                                   for k, v in p1['features'].items()}
        profile.phase1_shap_explanation = p1['shap_explanation']
    if p2 is not None:
        profile.phase2_land_cover = p2['land_cover']
        profile.phase2_confidence_pct = p2['confidence_pct']
        profile.phase2_ndvi = p2['ndvi']
        profile.phase2_ndvi_label = p2['ndvi_label']
        profile.phase2_class_distribution = p2['class_distribution']
        profile.phase2_change_flag = p2.get('change_flag')
        profile.phase2_change_description = p2.get('change_description')
    return asdict(profile)


def save_profile(profile, out_dir=OUT_DIR):
    """outputs/reports/enhanced_site_profile_<lat>_<lon>.json (Section I.1 naming)."""
    os.makedirs(out_dir, exist_ok=True)
    lat, lon = profile['latitude'], profile['longitude']
    name = f'enhanced_site_profile_{lat:.2f}_{lon:.2f}.json'
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    print('Saved', path)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Unified enhanced site profile')
    ap.add_argument('--lat', type=float)
    ap.add_argument('--lon', type=float)
    ap.add_argument('--name', default='')
    ap.add_argument('--radius', type=float, default=0.005,
                    help='LiDAR box half-width in degrees')
    ap.add_argument('--street-radius', type=float, default=0.0015,
                    help='Mapillary search box half-width in degrees')
    args = ap.parse_args()
    sites = ([(args.name or f'{args.lat:.4f}, {args.lon:.4f}', args.lat, args.lon)]
             if args.lat is not None and args.lon is not None else DEMO_SITES)
    t0 = time.time()
    profiles = []
    for name, lat, lon in sites:
        t1 = time.time()
        profile = build_profile(lat, lon, args.radius, site_name=name,
                                street_radius_deg=args.street_radius)
        print(f'\n=== {name} ({time.time() - t1:.1f} s) ===')
        print(json.dumps(profile, indent=2))
        save_profile(profile)
        profiles.append(profile)

    print(f'\n{"site":38s}{"P1 score":>9s}{"P2 cover":>12s}{"bldg m":>7s}{"canopy":>7s}'
          f'{"imperv":>7s}{"slope":>6s}{"imgs":>5s}{"walk":>6s}')
    for p in profiles:
        p1 = p['phase1_opportunity_score']
        walk = p['walkability_score']
        print(f'{p["site_name"][:37]:38s}{("-" if p1 is None else f"{p1:.2f}"):>9s}'
              f'{(p["phase2_land_cover"] or "-"):>12s}'
              f'{p["avg_building_height_m"]:7.2f}{p["canopy_cover_pct"]:7.1f}'
              f'{p["impervious_surface_pct"]:7.1f}{p["slope_risk_score"]:6.2f}'
              f'{p["street_images_analyzed"]:5d}'
              f'{("-" if walk is None else f"{walk:.2f}"):>6s}')
    print(f'Done in {time.time() - t0:.1f} s')
