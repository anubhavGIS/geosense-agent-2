# test_load.py
# Purpose: Part G.1 - a small load test of the live backend (Phase 5, Exercise 4)
#   1. GET /health once: the uptime of the instance that answers shows whether the run
#      starts on a cold instance (started for this request) or a warm one, and which
#      revision serves the traffic
#   2. 20 sequential POST /tools/score_location at Park Street, Kolkata - a point of the
#      study area, so every call does the tool's real work (PostGIS distances, the Random
#      Forest, SHAP); the first call is also reported apart from calls 2-20
#   3. a burst of 10 concurrent calls of the same request, released together
#   For each run: errors (an HTTP status other than 200, or no answer within the timeout),
#   average, p95 (nearest rank) and maximum latency, and the verdict against the targets
#   of Section 6 of the blueprint: 0 errors and an average under 5 seconds.
#   A point outside the study area, such as Chennai (13.0827, 80.2707), is answered at
#   once without running the model; --point 13.0827,80.2707 measures that path.
# Run with (project root, geosense environment):
#   python tests/test_load.py                                  (the live URL below)
#   python tests/test_load.py --url http://localhost:8000 --sequential 20 --burst 10
#   python -m pytest -m live tests/test_load.py                (sequential run, asserts the
#                                                               targets; LOAD_TEST_URL)
# A JSON report goes to outputs/deploy_logs/load_test_<stamp>.json.
import argparse
import datetime as dt
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_URL = 'https://geosense-backend-917306622260.us-central1.run.app'
PARK_STREET = (22.5530, 88.3540)
TARGET_AVG_S = 5.0                  # Section 6 of the blueprint: API response under 5 s
TIMEOUT_S = 120                     # long enough to measure a queued call, not to lose it


def p95(values):
    """95th percentile by the nearest-rank method: the smallest value that at least 95 %
    of the values do not exceed (of 20 values the 19th smallest, of 10 the largest)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def one_call(client, url, body):
    """One POST /tools/score_location: its latency in seconds and its HTTP status (or the
    name of the error when no answer came)."""
    start = time.perf_counter()
    try:
        status = client.post(f'{url}/tools/score_location', json=body,
                             timeout=TIMEOUT_S).status_code
    except httpx.HTTPError as err:
        status = type(err).__name__
    return {'seconds': round(time.perf_counter() - start, 3), 'status': status}


def sequential(url, body, n):
    """n calls one after another on one connection (one user asking again and again)."""
    with httpx.Client() as client:
        return [one_call(client, url, body) for _ in range(n)]


def burst(url, body, n):
    """n calls released together by a barrier, each on its own connection (n users at the
    same moment)."""
    gate = threading.Barrier(n)

    def worker(_):
        with httpx.Client() as client:
            gate.wait()
            return one_call(client, url, body)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(worker, range(n)))


def summary(calls):
    seconds = [c['seconds'] for c in calls]
    errors = sum(1 for c in calls if c['status'] != 200)
    average = sum(seconds) / len(seconds)
    return {'requests': len(calls), 'errors': errors, 'avg_s': round(average, 2),
            'p95_s': round(p95(seconds), 2), 'max_s': round(max(seconds), 2),
            'min_s': round(min(seconds), 2),
            'meets_target': errors == 0 and average < TARGET_AVG_S}


def health(url):
    """GET /health: an instance whose uptime is shorter than this request's own duration
    was started for it - a cold start."""
    start = time.perf_counter()
    answer = httpx.get(f'{url}/health', timeout=TIMEOUT_S)
    seconds = time.perf_counter() - start
    body = answer.json() if answer.status_code == 200 else {}
    uptime = body.get('uptime_s')
    return {'status': answer.status_code, 'seconds': round(seconds, 2), 'uptime_s': uptime,
            'revision': body.get('revision', '(not reported)'),
            'cold_start': uptime is not None and uptime <= seconds + 1.0}


def main():
    ap = argparse.ArgumentParser(description='Load test of the GeoSense backend (Part G.1)')
    ap.add_argument('--url', default=os.getenv('LOAD_TEST_URL', LIVE_URL))
    ap.add_argument('--point', default=f'{PARK_STREET[0]},{PARK_STREET[1]}',
                    help='latitude,longitude of the score_location request')
    ap.add_argument('--sequential', type=int, default=20)
    ap.add_argument('--burst', type=int, default=10)
    args = ap.parse_args()
    url = args.url.rstrip('/')
    lat, lon = (float(v) for v in args.point.split(','))
    body = {'latitude': lat, 'longitude': lon}
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'GeoSense load test - {url}  ({stamp})')
    print(f'request: POST /tools/score_location {json.dumps(body)}  timeout {TIMEOUT_S} s')

    h = health(url)
    print(f'GET /health  {h["status"]}  {h["seconds"]:.2f} s  revision={h["revision"]}  '
          f'uptime={h["uptime_s"]} s  -> '
          + ('a cold start (the instance was started for this request)' if h['cold_start']
             else 'a warm instance'))

    started = time.perf_counter()
    seq = sequential(url, body, args.sequential)
    seq_wall = time.perf_counter() - started
    started = time.perf_counter()
    bur = burst(url, body, args.burst)
    bur_wall = time.perf_counter() - started

    rows = [(f'sequential ({args.sequential})', summary(seq), seq_wall),
            ('  first call', summary(seq[:1]), None),
            (f'  calls 2-{args.sequential}', summary(seq[1:]), None) if len(seq) > 1 else None,
            (f'burst ({args.burst} concurrent)', summary(bur), bur_wall)]
    rows = [r for r in rows if r]
    print(f'\n{"run":24s} {"requests":>8s} {"errors":>6s} {"avg s":>7s} {"p95 s":>7s} '
          f'{"max s":>7s}  target: 0 errors, avg < {TARGET_AVG_S:.0f} s')
    for name, s, wall in rows:
        verdict = ('met' if s['meets_target'] else 'NOT met') if wall is not None else ''
        print(f'{name:24s} {s["requests"]:8d} {s["errors"]:6d} {s["avg_s"]:7.2f} '
              f'{s["p95_s"]:7.2f} {s["max_s"]:7.2f}  {verdict}')
    statuses = sorted({str(c['status']) for c in seq + bur})
    print(f'\nstatuses seen: {", ".join(statuses)}; wall time: sequential {seq_wall:.1f} s, '
          f'burst {bur_wall:.1f} s')
    print('burst latencies (s): ' + ', '.join(f'{c["seconds"]:.1f}' for c in
                                               sorted(bur, key=lambda c: c['seconds'])))

    report = {'url': url, 'stamp': stamp, 'request': body, 'timeout_s': TIMEOUT_S,
              'target_avg_s': TARGET_AVG_S, 'health': h,
              'sequential': {'summary': summary(seq), 'wall_s': round(seq_wall, 2),
                             'calls': seq},
              'burst': {'summary': summary(bur), 'wall_s': round(bur_wall, 2), 'calls': bur}}
    out = ROOT / 'outputs' / 'deploy_logs' / f'load_test_{stamp}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'report: {out.relative_to(ROOT)}')


@pytest.mark.live
def test_sequential_load_meets_the_targets():
    """Part G.1's success check: 20 sequential calls, 0 errors, average under 5 s."""
    url = os.getenv('LOAD_TEST_URL', LIVE_URL).rstrip('/')
    s = summary(sequential(url, {'latitude': PARK_STREET[0], 'longitude': PARK_STREET[1]},
                           20))
    assert s['errors'] == 0, s
    assert s['avg_s'] < TARGET_AVG_S, s


if __name__ == '__main__':
    main()
