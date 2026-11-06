"""
test_ee_service_account.py - Phase 5, Exercise 3: check the Earth Engine service-account key
on the laptop before it goes into Secret Manager.

Initialises Earth Engine exactly as the Cloud Run container will - tools/common.py with
GEE_SERVICE_ACCOUNT_KEY pointing at the key file, so the personal login of Phase II is not
used - and runs the climate tool's ERA5-Land query for Park Street, bypassing the cache.
The key's contents are never printed; only its file name, size and service account.

Run from the project root in the geosense environment:
    python deploy\\test_ee_service_account.py
    python deploy\\test_ee_service_account.py --key C:\\path\\to\\other-key.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY = Path.home() / '.config' / 'geosense' / 'gee-sa-key.json'
PARK_STREET = (22.5530, 88.3540)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', type=Path, default=DEFAULT_KEY, help='service-account key file')
    args = ap.parse_args()
    if not args.key.exists():
        sys.exit(f'key file not found: {args.key}')
    info = json.loads(args.key.read_text(encoding='utf-8'))
    if info.get('type') != 'service_account':
        sys.exit(f'{args.key} is not a service-account key file')
    print(f'key file {args.key} ({args.key.stat().st_size:,} bytes), service account '
          f'{info["client_email"]}')

    os.environ['GEE_SERVICE_ACCOUNT_KEY'] = str(args.key)      # the container's path
    os.environ.pop('GEE_SERVICE_ACCOUNT_JSON', None)
    sys.path[:0] = [str(ROOT / 'src' / 'phase4_agent'), str(ROOT / 'src' / 'phase1_ml')]
    from tools.common import ToolError, earth_engine           # noqa: E402 (needs sys.path)
    from tools import climate                                  # noqa: E402

    t0 = time.time()
    try:
        ee = earth_engine()
    except ToolError as err:
        sys.exit(f'Earth Engine refused the service account:\n  {err}')
    print(f'ee.Initialize with the service account: OK in {time.time() - t0:.1f} s '
          f'(project {os.getenv("GEE_PROJECT")}, earthengine-api {ee.__version__})')

    t0 = time.time()
    try:
        data = climate._query(*PARK_STREET)                    # no cache: a real request
    except Exception as err:                                   # noqa: BLE001 (report it)
        sys.exit(f'the ERA5-Land request failed: {type(err).__name__}: {err}')
    v = data['values']
    print(f'ERA5-Land request for Park Street: OK in {time.time() - t0:.1f} s - 12 months to '
          f'{data["end"]}: mean {v["t_recent"] - 273.15:.2f} C, normal '
          f'{v["t_normal"] - 273.15:.2f} C, precipitation {v["p_recent"] * 1000:.0f} mm')
    print('the key works - next: python deploy\\cloud_setup.py prepare')


if __name__ == '__main__':
    main()
