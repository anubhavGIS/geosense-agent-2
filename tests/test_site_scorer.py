# test_site_scorer.py
# Purpose: Automated tests verifying the site scorer's contract.
# Run from the project root with:  python -m pytest tests/ -v
import sys
sys.path.append('src/phase1_ml')

from site_scorer import score_location

# Central Kolkata (Park Street area). The manual tests with Chennai
# (13.0827, 80.2707) -- meaningless against a Kolkata database: the
# nearest road would be ~1,660 km away.
TEST_LAT, TEST_LON = 22.5535, 88.3520

def test_score_returns_dict():
    result = score_location(TEST_LAT, TEST_LON)
    assert isinstance(result, dict), 'Result should be a dictionary'

def test_score_has_required_keys():
    result = score_location(TEST_LAT, TEST_LON)
    required_keys = ['latitude', 'longitude', 'opportunity_score',
                     'risk_score', 'features', 'shap_explanation', 'verdict']
    for key in required_keys:
        assert key in result, f'Missing key: {key}'

def test_scores_in_valid_range():
    result = score_location(TEST_LAT, TEST_LON)
    assert 0 <= result['opportunity_score'] <= 10, 'Score must be 0-10'
    assert 0 <= result['risk_score'] <= 10, 'Score must be 0-10'

def test_verdict_is_valid():
    result = score_location(TEST_LAT, TEST_LON)
    assert result['verdict'] in ['GOOD SITE', 'POOR SITE']