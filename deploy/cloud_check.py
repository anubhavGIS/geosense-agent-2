"""
cloud_check.py - Phase 5, Exercise 3: three checks of the live Cloud Run backend that the
smoke test does not make.

  1. /health twice - the first answer after an idle period includes the cold start (a new
     container is started), the second shows a warm instance; both times are printed
  2. Earth Engine from the cloud - get_climate_risk and get_demographics at a point that
     is in no cache of the image (New Delhi), so the answer must come from Earth Engine
     through the service account (from_cache = False)
  3. CORS - GET /agent/latest_result with the Origin header of each browser page that
     polls the backend (the twin on GitHub Pages by default): allowed when the answer
     carries Access-Control-Allow-Origin for that origin, blocked otherwise

Run from the project root in the geosense environment (the URL is read from gcloud when
--url is not given); the report goes to outputs/deploy_logs/cloud_check_<stamp>.json:
    python deploy\\cloud_check.py
    python deploy\\cloud_check.py --no-fresh            (cold start + CORS only)
    python deploy\\cloud_check.py --origin https://anubhavgis.github.io http://localhost:4173
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'outputs' / 'deploy_logs'
NEW_DELHI = {'latitude': 28.6139, 'longitude': 77.2090}         # outside every cache
TWIN_ORIGIN = 'https://anubhavgis.github.io'


def service_url():
    """The Cloud Run URL of geosense-backend, read from gcloud: the
    https://geosense-backend-PROJECTNUMBER.us-central1.run.app address (gcloud's Service URL),
    else status.url (the older ...-uc.a.run.app address of the same service)."""
    cmd = ('gcloud run services describe geosense-backend --region=us-central1 '
           '--project=geosense-agent-2 --format=json --quiet')
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    svc = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip().startswith('{') else {}
    urls = json.loads(svc.get('metadata', {}).get('annotations', {})
                      .get('run.googleapis.com/urls', '[]'))
    url = next((u for u in urls if u.endswith('.us-central1.run.app')),
               svc.get('status', {}).get('url') or '')
    if not url.startswith('https://'):
        sys.exit('service URL not found - pass --url https://geosense-backend-....run.app')
    return url


def timed(client, method, url, **kw):
    t0 = time.perf_counter()
    r = client.request(method, url, **kw)
    return r, round(time.perf_counter() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', help='backend address (default: from gcloud)')
    ap.add_argument('--origin', nargs='*', default=[TWIN_ORIGIN], help='browser origins')
    ap.add_argument('--no-fresh', action='store_true', help='skip the Earth Engine calls')
    args = ap.parse_args()
    url = (args.url or service_url()).rstrip('/')
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {'url': url, 'started': stamp, 'health': [], 'fresh': [], 'cors': []}
    print(f'GeoSense cloud check - {url}  ({stamp})')
    with httpx.Client(timeout=300) as client:
        print('1. /health (first request after idle = cold start)')
        for label in ('first', 'second'):
            r, s = timed(client, 'GET', f'{url}/health')
            body = r.json()
            print(f'   {label:6s} {r.status_code}  {s:6.2f} s  device={body.get("device")} '
                  f'port={body.get("port")} uptime={body.get("uptime_s")} s')
            report['health'].append({'call': label, 'status': r.status_code, 'seconds': s,
                                     **body})
        if not args.no_fresh:
            print(f'2. Earth Engine from the cloud at New Delhi {tuple(NEW_DELHI.values())}')
            for tool, fields in (('get_climate_risk', ['temperature_anomaly_c', 'period']),
                                 ('get_demographics', ['population_density_per_km2'])):
                r, s = timed(client, 'POST', f'{url}/tools/{tool}', json=NEW_DELHI)
                body = r.json()
                values = ', '.join(f'{f}={body.get(f)}' for f in fields)
                state = r.status_code == 200 and body.get('from_cache') is False
                print(f'   {"OK  " if state else "FAIL"} {tool:17s} {r.status_code}  {s:6.2f} s  '
                      f'from_cache={body.get("from_cache")}  {values or body.get("detail")}')
                report['fresh'].append({'tool': tool, 'status': r.status_code, 'seconds': s,
                                        'response': body})
        print('3. CORS for the browser pages that poll /agent/latest_result')
        for origin in args.origin:
            r, s = timed(client, 'GET', f'{url}/agent/latest_result',
                         headers={'Origin': origin})
            allowed = r.headers.get('access-control-allow-origin')
            verdict = 'allowed' if allowed in (origin, '*') else \
                'BLOCKED by the browser (no Access-Control-Allow-Origin for this origin)'
            print(f'   {origin:32s} {r.status_code}  {verdict}')
            report['cors'].append({'origin': origin, 'status': r.status_code,
                                   'allow_origin': allowed})
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f'cloud_check_{stamp}.json'
    out.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(f'report: {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
