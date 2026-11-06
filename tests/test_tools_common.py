# test_tools_common.py
# Purpose: offline tests of the helpers every MCP tool relies on (src/phase4_agent/tools/
#          common.py) - the study-area gate, the explicit "no data here" answer, the
#          redaction of credentials in error text and the conversion to plain JSON.
#          They need no database, data file, server or key, so the CI pipeline runs them
#          (Phase 5, Exercise 4).
# Run with: python -m pytest tests/test_tools_common.py -v     (from the project root)
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.append('src/phase4_agent')
from tools.common import in_study_area, jsonable, outside_study_area, redact  # noqa: E402

PARK_STREET = (22.5530, 88.3540)          # inside the Kolkata box of Phases I and II
CHENNAI = (13.0827, 80.2707)              # Chennai centre, outside it


def test_study_area_gate():
    assert in_study_area(*PARK_STREET)
    assert not in_study_area(*CHENNAI)
    assert in_study_area(22.45, 88.30) and in_study_area(22.67, 88.42)   # the corners count
    assert not in_study_area(22.6701, 88.35) and not in_study_area(22.50, 88.4201)


def test_outside_answer_is_explicit_not_an_error():
    answer = outside_study_area(*CHENNAI, 'Phase I site scorer',
                                ('opportunity_score', 'verdict'))
    assert answer['available'] is False
    assert 'outside the study area of the Phase I site scorer' in answer['reason']
    assert answer['opportunity_score'] is None and answer['verdict'] is None
    assert (answer['latitude'], answer['longitude']) == CHENNAI


def test_redact_hides_credentials_in_error_text():
    text = ('GET https://graph.mapillary.com/images?access_token=MLY|123|abc&fields=id '
            'failed; key=AIzaSyFAKE password=not-a-real-one token=t0k3n')
    out = redact(text)
    for secret in ('MLY|123|abc', 'AIzaSyFAKE', 'not-a-real-one', 't0k3n'):
        assert secret not in out
    assert 'access_token=***&fields=id' in out and 'password=***' in out


def test_jsonable_gives_plain_json():
    value = {'a': np.float32(1.5), 'b': np.int64(7), 'c': np.array([1, 2]),
             'd': Decimal('2.5'), 'e': float('nan'), 'f': float('inf'), 'g': (1, 2),
             'h': Path('report.json'), 3: np.bool_(True)}
    out = jsonable(value)
    assert out == {'a': 1.5, 'b': 7, 'c': [1, 2], 'd': 2.5, 'e': None, 'f': None,
                   'g': [1, 2], 'h': 'report.json', '3': True}
    json.dumps(out, allow_nan=False)      # no numpy type, NaN or infinity is left
