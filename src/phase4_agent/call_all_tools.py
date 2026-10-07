# call_all_tools.py
# Purpose: call every tool of the running MCP server for the test locations, save all the
#          JSON responses in outputs/agent_logs/ and print one summary line per call
# Run from the project root while mcp_server.py runs in another window:
#   python src/phase4_agent/call_all_tools.py --label stage1
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
SERVER = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')

KOLKATA = {'latitude': 22.5530, 'longitude': 88.3540}       # Park Street, study area
CHENNAI = {'latitude': 13.0827, 'longitude': 80.2707}       # the manual's test point
DUPONT = {'min_lat': 38.9085, 'min_lon': -77.0450,          # LiDAR tile of Phase III
          'max_lat': 38.9110, 'max_lon': -77.0420}
CHENNAI_BOX = {'min_lat': 13.05, 'min_lon': 80.24, 'max_lat': 13.06, 'max_lon': 80.25}
CASES = [  # tool, case, JSON body, expected HTTP status
    ('score_location', 'Kolkata', KOLKATA, 200),
    ('score_location', 'Chennai', CHENNAI, 200),
    ('classify_imagery', 'Kolkata', KOLKATA, 200),
    ('classify_imagery', 'Chennai', CHENNAI, 200),
    ('lidar_profile', 'Dupont box', DUPONT, 200),
    ('lidar_profile', 'Chennai box', CHENNAI_BOX, 200),
    ('streetview_scene', 'Kolkata', KOLKATA, 200),
    ('streetview_scene', 'Chennai', CHENNAI, 200),
    ('get_climate_risk', 'Kolkata', KOLKATA, 200),
    ('get_climate_risk', 'Chennai', CHENNAI, 200),
    ('get_demographics', 'Kolkata', KOLKATA, 200),
    ('get_demographics', 'Chennai', CHENNAI, 200),
    ('find_similar_sites', 'Kolkata', KOLKATA, 200),
    ('find_similar_sites', 'Chennai', CHENNAI, 200),
    ('score_location', 'no longitude', {'latitude': 22.5530}, 422),
    ('score_location', 'latitude text', {'latitude': 'north', 'longitude': 88.354}, 422),
    ('lidar_profile', 'box reversed', {**DUPONT, 'min_lat': 38.9200}, 422),
]


def note(r, real):
    """' - PLACEHOLDER' for fixed example values, otherwise the real detail."""
    return ' - PLACEHOLDER' if 'placeholder' in r.get('source', '') else real


def summary(tool, r):
    """One line with the key values of a tool response."""
    if not r.get('available', True):
        return 'not available: ' + r.get('reason', '')
    if tool == 'score_location':
        return (f"opportunity {r['opportunity_score']}, risk {r['risk_score']} - "
                f"{r['verdict']}")
    if tool == 'classify_imagery':
        return (f"{r['land_cover']} ({r['confidence_pct']} %), NDVI {r['ndvi']}, "
                f"change {r['change_flag']}")
    if tool == 'lidar_profile':
        return (f"buildings {r['avg_building_height_m']} m, canopy "
                f"{r['canopy_cover_pct']} %, {r['total_points_analyzed']:,} points")
    if tool == 'streetview_scene':
        return f"walkability {r['walkability_score']} from {r['images_analyzed']} images"
    if tool == 'get_climate_risk':
        return (f"anomaly {r['temperature_anomaly_c']:+} C, drought index "
                f"{r['drought_index']}" + note(r, f" ({r.get('drought_class')})"))
    if tool == 'get_demographics':
        density, km = r['population_density_per_km2'], r['nearest_hospital_km']
        hospital = ('n/a (study area only)' if km is None else
                    f'{km} km' + note(r, f", {r.get('hospitals_within_2km')} within 2 km"))
        return (f"{density if density is None else format(density, ',')} people/km2, "
                f"nearest hospital {hospital}")
    if tool == 'find_similar_sites':
        scores = [str(s['opportunity_score']) for s in r['similar_sites']]
        return (f"reference {r['reference']['opportunity_score']}, {len(scores)} similar"
                + note(r, ': ' + ', '.join(scores)))
    return ''


def main(label, only):
    """
    Call every test case, print one line per call and save the full responses.

    Args:
        label (str): label of the run, used in the file name (e.g. stage1)
        only (str): run only the tools whose name contains this text ('' = all)

    Returns:
        int: number of calls whose HTTP status was not the expected one

    Example:
        main('stage2', 'climate')
    """
    log, failures = [], 0
    print(f'{"tool":<19}{"case":<14}{"HTTP":>4}{"time":>8}  result')
    with httpx.Client(base_url=SERVER, timeout=300) as client:
        for tool, case, body, expected in [c for c in CASES if only in c[0]]:
            t0 = time.perf_counter()
            try:
                resp = client.post(f'/tools/{tool}', json=body)
            except httpx.ConnectError:
                raise SystemExit(f'Cannot reach the MCP server at {SERVER} - start it '
                                 'first: python src/phase4_agent/mcp_server.py')
            seconds = time.perf_counter() - t0
            data = resp.json()
            if resp.status_code == 200:
                text = summary(tool, data)
            elif resp.status_code == 422:
                text = f"{data['detail'][0]['loc'][-1]}: {data['detail'][0]['msg']}"
            else:
                text = str(data.get('detail'))
            failures += resp.status_code != expected
            flag = '' if resp.status_code == expected else f'  <-- expected {expected}'
            print(f'{tool:<19}{case:<14}{resp.status_code:>4}{seconds:>7.1f}s  '
                  f'{text[:64]}{flag}')
            log.append({'tool': tool, 'case': case, 'request': body,
                        'status': resp.status_code, 'seconds': round(seconds, 2),
                        'response': data})
    out = ROOT / 'outputs' / 'agent_logs' / (
        f'tool_responses_{label}_{datetime.now():%Y%m%d_%H%M%S}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2))
    print(f'\n{len(log)} calls, {failures} with an unexpected status. '
          f'Responses saved to {out.relative_to(ROOT)}')
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call every tool of the MCP server.')
    parser.add_argument('--label', default='run', help='label for the output file')
    parser.add_argument('--only', default='', help='only tools containing this text')
    args = parser.parse_args()
    raise SystemExit(main(args.label, args.only))
