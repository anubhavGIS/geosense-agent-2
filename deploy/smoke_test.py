"""
smoke_test.py - Phase 5: call a running GeoSense backend once per tool and report.

Checks /health and the OpenAPI document, then POSTs one request to each of the seven
tools - the Park Street point for the Kolkata tools, the Dupont Circle box for LiDAR, the
Chennai point for the worldwide tools - and prints the status code, the time and the key
values of every answer. Written for the containerised stack of Exercise 2
(http://localhost:8000), reused against the Cloud Run URL in Exercise 3 and by the
deployment workflow in Exercise 4. The report is also saved as JSON in
outputs/deploy_logs/smoke_<stamp>.json. Exit code 1 when any call fails.

Run from the project root in the geosense environment:
    python deploy\\smoke_test.py                                   (localhost:8000)
    python deploy\\smoke_test.py --url https://geosense-backend-....run.app
    python deploy\\smoke_test.py --skip streetview_scene            (no Mapillary access)
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'outputs' / 'deploy_logs'
PARK_STREET = {'latitude': 22.5530, 'longitude': 88.3540}
CHENNAI = {'latitude': 13.0827, 'longitude': 80.2707}
DUPONT = {'min_lat': 38.9085, 'min_lon': -77.0450, 'max_lat': 38.9110, 'max_lon': -77.0420}
# tool -> (payload, the fields worth printing)
CALLS = {
    'score_location': (PARK_STREET, ['opportunity_score', 'verdict']),
    'classify_imagery': (PARK_STREET, ['land_cover', 'confidence_pct', 'model']),
    'lidar_profile': (DUPONT, ['avg_building_height_m', 'canopy_cover_pct',
                               'total_points_analyzed']),
    'streetview_scene': (PARK_STREET, ['walkability_score', 'images_analyzed']),
    'get_climate_risk': (CHENNAI, ['temperature_anomaly_c', 'drought_class', 'flood_zone']),
    'get_demographics': (PARK_STREET, ['population_density_per_km2',
                                       'nearest_hospital_km']),
    'find_similar_sites': (PARK_STREET, ['similar_sites', 'search']),
}


def summarise(tool, body):
    """The key values of one answer as a short string."""
    if not body.get('available', True):
        return f'available=false: {body.get("reason", "")[:70]}'
    parts = []
    for field in CALLS[tool][1]:
        value = body.get(field)
        if field == 'similar_sites':
            value = f'{len(value)} sites'
        elif field == 'search':
            value = f'{value.get("candidates_in_radius")} candidates'
        elif field == 'model' and isinstance(value, str):
            value = value.split(' (')[0]
        parts.append(f'{field}={value}')
    return ', '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8000', help='backend address')
    ap.add_argument('--skip', nargs='*', default=[], help='tools not to call')
    ap.add_argument('--timeout', type=float, default=180, help='seconds per call')
    args = ap.parse_args()
    url = args.url.rstrip('/')
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {'url': url, 'started': stamp, 'calls': []}
    failed = 0
    print(f'GeoSense backend smoke test - {url}  ({stamp})')
    with httpx.Client(timeout=args.timeout) as client:
        for path in ('/health', '/openapi.json'):
            t0 = time.perf_counter()
            try:
                r = client.get(url + path)
                body = r.json()
                if path == '/health':
                    detail = (f'tools={body.get("tools")} device={body.get("device")} '
                              f'uptime={body.get("uptime_s")}s')
                else:
                    n_routes = sum(p.startswith('/tools/') for p in body['paths'])
                    detail = f'{n_routes} tool routes, title "{body["info"]["title"]}"'
                ok = r.status_code == 200
            except httpx.HTTPError as err:
                ok, r, detail = False, None, f'{type(err).__name__}: {err}'
            failed += not ok
            status = r.status_code if r is not None else '---'
            print(f'  {"OK  " if ok else "FAIL"} GET  {path:22s} {status}  '
                  f'{time.perf_counter() - t0:6.2f} s  {detail}')
            report['calls'].append({'method': 'GET', 'path': path, 'ok': ok,
                                    'status': status, 'detail': detail})
        for tool, (payload, _) in CALLS.items():
            if tool in args.skip:
                print(f'  skip POST /tools/{tool}')
                continue
            t0 = time.perf_counter()
            try:
                r = client.post(f'{url}/tools/{tool}', json=payload)
                body = r.json()
                ok = r.status_code == 200
                detail = summarise(tool, body) if ok else str(body.get('detail'))[:90]
            except httpx.HTTPError as err:
                ok, r, body, detail = False, None, {}, f'{type(err).__name__}: {err}'
            seconds = round(time.perf_counter() - t0, 2)
            failed += not ok
            status = r.status_code if r is not None else '---'
            print(f'  {"OK  " if ok else "FAIL"} POST /tools/{tool:19s} {status}  '
                  f'{seconds:6.2f} s  {detail}')
            report['calls'].append({'method': 'POST', 'path': f'/tools/{tool}', 'ok': ok,
                                    'status': status, 'seconds': seconds,
                                    'detail': detail, 'response': body})
    report['failed'] = failed
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f'smoke_{stamp}.json'
    out.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    verdict = 'all calls passed' if not failed else f'{failed} call(s) FAILED'
    print(f'{verdict}; report: {out}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
