# common.py
# Purpose: helpers shared by the MCP tools
#   - the study-area gate (Phases I and II were built for the Kolkata box in .env)
#   - conversion of numpy values, decimals and NaN into plain JSON
#   - ToolError, raised when a data source is unreachable (the server answers HTTP 503)
#   - one PostGIS engine and one Earth Engine session, opened on first use
#   - a small JSON cache in data/phase4_cache for slow open-data queries
#   Phase 5, Exercise 2: Earth Engine through a service account when the container has
#   one (GEE_SERVICE_ACCOUNT_KEY = path to its JSON key, or GEE_SERVICE_ACCOUNT_JSON =
#   the key itself); on the laptop, without either, the personal login of Phase II is
#   used as before. The key is never printed - redact() also hides it in error text.
import json
import math
import os
import re
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]                   # ...\GeoSense_Agent
load_dotenv(ROOT / '.env')

# Phases I and II were built for this box (BBOX_* in .env, Phase 2 Exercise 1)
STUDY_BBOX = [float(os.getenv('BBOX_MIN_LON', '88.30')),
              float(os.getenv('BBOX_MIN_LAT', '22.45')),
              float(os.getenv('BBOX_MAX_LON', '88.42')),
              float(os.getenv('BBOX_MAX_LAT', '22.67'))]
STUDY_AREA = os.getenv('STUDY_AREA', 'Kolkata')
CACHE_DIR = ROOT / 'data' / 'phase4_cache'
_SECRET = re.compile(r'(access_token|key|token|password)=[^&\s]+', re.IGNORECASE)
_engine, _ee = None, None


class ToolError(RuntimeError):
    """A data source the tool depends on is unavailable (database, Earth Engine...)."""


def in_study_area(lat, lon):
    """True when (lat, lon) lies inside the study-area box of Phases I and II."""
    w, s, e, n = STUDY_BBOX
    return w <= lon <= e and s <= lat <= n


def study_area_text():
    w, s, e, n = STUDY_BBOX
    return f'{STUDY_AREA}: {w:.2f}-{e:.2f} E, {s:.2f}-{n:.2f} N'


def outside_study_area(lat, lon, source, empty_keys=()):
    """The answer of a study-area tool for a point outside the box: not an error,
    but an explicit 'no data here' that an agent can explain to the user."""
    result = {'latitude': lat, 'longitude': lon, 'available': False,
              'reason': f'outside the study area of the {source} ({study_area_text()})'}
    result.update({k: None for k in empty_keys})
    return result


def redact(text):
    """Remove anything that looks like a credential from an error message."""
    return _SECRET.sub(lambda m: m.group(1) + '=***', str(text))


def jsonable(obj):
    """Recursively turn numpy values, decimals, tuples and paths into plain JSON
    values; NaN and infinity (which JSON cannot carry) become None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, Decimal):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def db_engine():
    """The PostGIS engine of Phase I (db_connection.py), created once and reused."""
    global _engine
    if _engine is None:
        from db_connection import get_engine          # src/phase1_ml
        _engine = get_engine()
    return _engine


def _ee_credentials(ee):
    """
    Service-account credentials for Earth Engine when the environment provides a key,
    otherwise None (the personal login stored by 'earthengine authenticate' is used).

    Args:
        ee: the imported earthengine-api module

    Returns:
        ee.ServiceAccountCredentials, or None when no key is configured

    Example:
        creds = _ee_credentials(ee)   # GEE_SERVICE_ACCOUNT_KEY=/secrets/gee-key.json
    """
    key_file = os.getenv('GEE_SERVICE_ACCOUNT_KEY')        # a mounted secret file (cloud)
    key_json = os.getenv('GEE_SERVICE_ACCOUNT_JSON')       # or the key as one env value
    if key_file:
        email = json.loads(Path(key_file).read_text())['client_email']
        return ee.ServiceAccountCredentials(email, key_file=key_file)
    if key_json:
        email = json.loads(key_json)['client_email']
        return ee.ServiceAccountCredentials(email, key_data=key_json)
    return None


def earth_engine():
    """The Earth Engine API, initialised once with GEE_PROJECT (as in Phase II); in a
    container with a service-account key the session belongs to that account. Without a
    key, credentials is left out: ee.Initialize then uses its default, the login stored
    by 'earthengine authenticate' (an explicit None would mean no credentials at all)."""
    global _ee
    if _ee is None:
        try:
            import ee
            options = {'project': os.getenv('GEE_PROJECT')}
            credentials = _ee_credentials(ee)
            if credentials is not None:
                options['credentials'] = credentials
            ee.Initialize(**options)
        except Exception as err:
            raise ToolError(f'Google Earth Engine is not available ({redact(err)}); '
                            'run "earthengine authenticate" in the geosense environment, '
                            'or give the container a service-account key') from err
        _ee = ee
    return _ee


def cache_get(name, max_age_days=None):
    """A cached value, or None when it is missing or older than max_age_days."""
    path = CACHE_DIR / f'{name}.json'
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    if max_age_days and time.time() - entry['saved_unix'] > max_age_days * 86400:
        return None
    return entry['value']


def cache_put(name, value):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {'saved': time.strftime('%Y-%m-%d %H:%M:%S'), 'saved_unix': time.time(),
             'value': value}
    (CACHE_DIR / f'{name}.json').write_text(json.dumps(entry, indent=1))
