# test_phase3_modules.py
# Purpose: Automated tests to verify Phase 3 modules work correctly
# Run with: python -m pytest tests/test_phase3_modules.py -v     (from the project root)
#           python -m pytest tests/ -v                            (Phase 1 tests as well)
import json
import sys

sys.path.append('src/phase3_lidar')

import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

from lidar_analyzer import _compute_slope_risk, analyze_bbox      # noqa: E402
from lidar_features import classify_points                        # noqa: E402
from streetview_semantic import compute_walkability_score         # noqa: E402
from unified_site_profile import build_profile                    # noqa: E402

CHENNAI_BOX = (13.05, 80.24, 13.06, 80.25)            # the manual's example: no LiDAR
CIRCLE_BOX = (38.9085, -77.0450, 38.9110, -77.0420)   # Dupont Circle park (Washington)
OFFICE_BOX = (38.9050, -77.0420, 38.9075, -77.0390)   # office blocks south-east of it


# ── The four tests of Section G.1 ───────────────────────────────────────────
def test_lidar_analyzer_returns_dict():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    assert isinstance(result, dict), 'Result should be a dictionary'


def test_lidar_analyzer_required_keys():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    required = ['avg_building_height_m', 'canopy_cover_pct', 'impervious_surface_pct',
                'slope_risk_score']
    for key in required:
        assert key in result, f'Missing key: {key}'


def test_canopy_cover_in_valid_range():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    assert 0 <= result['canopy_cover_pct'] <= 100, 'Canopy cover must be 0-100'


def test_unified_profile_has_all_fields():
    profile = build_profile(13.0827, 80.2707)
    required = ['avg_building_height_m', 'canopy_cover_pct', 'scene_composition',
                'walkability_score']
    for key in required:
        assert key in profile, f'Missing key: {key}'


# ── Own tests ───────────────────────────────────────────────────────────────
def test_slope_risk_zero_for_fewer_than_two_points():
    assert _compute_slope_risk(np.array([])) == 0.0
    assert _compute_slope_risk(np.array([27.3])) == 0.0


def test_slope_risk_is_capped_and_flat_ground_scores_zero():
    assert _compute_slope_risk(np.array([0.0, 100.0] * 50)) == 10.0   # std 50 -> capped
    assert _compute_slope_risk(np.full(1000, 25.0)) == 0.0            # perfectly flat
    assert _compute_slope_risk(np.array([24.0, 26.0] * 50)) == 2.0    # std 1 m -> 2.0


def test_walkability_score_always_between_0_and_10():
    compositions = [
        {},                                                        # nothing recognised
        {'car': 100.0},                                            # a car park
        {'sidewalk': 100.0},                                       # all pavement
        {'sidewalk': 100.0, 'vegetation': 100.0},                  # impossible but legal
        {'road': 34.3, 'sidewalk': 0.2, 'building': 22.0, 'vegetation': 23.6,
         'sky': 3.1, 'car': 10.3},                                 # Exercise 4, image 1
    ]
    scores = [compute_walkability_score(c) for c in compositions]
    assert all(0.0 <= s <= 10.0 for s in scores), scores
    assert scores[1] == 0.0 and scores[3] == 10.0 and scores[4] == 1.02
    quiet = compute_walkability_score({'sidewalk': 5.0, 'vegetation': 40.0, 'car': 2.0})
    busy = compute_walkability_score({'sidewalk': 5.0, 'vegetation': 10.0, 'car': 15.0})
    assert quiet > busy, 'more trees and fewer cars must score higher'


def test_classify_points_follows_the_height_and_return_rules():
    hag = np.array([0.1, 1.0, 5.0, 5.0, 30.0, -2.0], dtype=np.float32)
    num_returns = np.array([1, 1, 3, 1, 2, 1])
    labels = classify_points(hag, num_returns, np.ones(6, dtype=int))
    assert labels.tolist() == [0, 1, 1, 2, 1, 0]       # ground shrub tree roof tree ground


def test_bbox_outside_the_cloud_is_reported_not_faked():
    result = analyze_bbox(*CHENNAI_BOX)
    assert result['lidar_coverage'] is False
    assert result['total_points_analyzed'] == 0 and result['coverage_pct'] == 0.0
    assert 'note' in result and 'no LiDAR points' in result['note']


def test_two_boxes_inside_the_tile_give_different_results():
    park, office = analyze_bbox(*CIRCLE_BOX), analyze_bbox(*OFFICE_BOX)
    for r in (park, office):
        assert r['lidar_coverage'] and r['total_points_analyzed'] > 100_000
        assert 0 <= r['impervious_surface_pct'] <= 100 and 0 <= r['slope_risk_score'] <= 10
    assert park['total_points_analyzed'] != office['total_points_analyzed']
    assert park['avg_building_height_m'] != office['avg_building_height_m']
    assert park['canopy_cover_pct'] > office['canopy_cover_pct']   # the park is greener
    assert park['bbox_projected'] != office['bbox_projected']


@pytest.fixture(scope='module')
def chennai_profile():
    """Built once for the tests below (Mapillary + SegFormer are the slow part)."""
    return build_profile(13.0827, 80.2707, site_name='Chennai centre')


def test_profile_records_its_sources_and_is_json_serialisable(chennai_profile):
    text = json.dumps(chennai_profile)                 # no numpy types may leak through
    assert 'sources' in chennai_profile
    for phase in ('phase1_site_scorer', 'phase2_image_classifier',
                  'phase3_lidar_analyzer', 'phase3_streetview_semantic'):
        assert phase in chennai_profile['sources'], phase
    assert chennai_profile['lidar_coverage'] is False  # Chennai has no LiDAR tile
    assert len(text) > 200


def test_profile_walkability_matches_its_composition(chennai_profile):
    comp = chennai_profile['scene_composition']
    score = chennai_profile['walkability_score']
    if comp:                                           # imagery was found
        assert score == compute_walkability_score(comp)
    else:                                              # or the profile says so
        assert score is None and chennai_profile['street_images_analyzed'] == 0
