"""
inspect_deploy_env.py - Phase 5, Exercise 2: the facts the container build depends on.

Prints, and writes to outputs/deploy_logs/deploy_inspect_<stamp>.txt:
  1. Python, PyTorch and the device the laptop runs the models on
  2. the trained site-scorer model (its class, and the scikit-learn that wrote it)
  3. the files the backend image copies, with their sizes (what the image will carry)
  4. the database: PostgreSQL and PostGIS versions, rows and size of every table the
     server reads (the dump of export_db.py), and the tables it does not read
  5. the tools of the exercise: Docker (with the CPUs and memory it gives containers),
     docker compose, pg_dump, the Earth Engine login, the free disk space, whether the
     ports of the stack are free, and the .env keys the containers need
No value of .env is printed - only whether a key is set, and the NAME of any key whose
value contains a $ sign (docker compose would read it as a variable).

Run from the project root in the geosense environment:
    python deploy\\inspect_deploy_env.py
"""
import datetime as dt
import glob
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'outputs' / 'deploy_logs'
IMAGE_FILES = ['models/saved/site_scorer_model.pkl', 'models/saved/unet_best.pth',
               'models/evaluation/model_comparison.json',
               'data/satellite/clean/clean_2023.npy', 'data/satellite/clean/meta_2023.json',
               'data/satellite/clean/clean_2018.npy', 'data/satellite/clean/meta_2018.json',
               'data/lidar/processed/study_area_classified.laz',
               'data/streetview/results', 'data/streetview/images', 'data/phase4_cache',
               'src/phase1_ml', 'src/phase2_dl', 'src/phase3_lidar', 'src/phase4_agent']
READ_TABLES = ['osm_roads', 'osm_poi', 'osm_poi_area', 'candidate_sites', 'flood_zones']
OTHER_TABLES = ['osm_buildings', 'osm_landuse']
NEEDED_KEYS = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'GEE_PROJECT',
               'MAPILLARY_TOKEN', 'GOOGLE_API_KEY', 'AGENT_MODEL']
PORTS = {5433: 'PostGIS container', 8000: 'backend container', 8501: 'frontend container'}
SECRET = re.compile(r'(password|token|key)=\S+', re.IGNORECASE)
lines = []


def say(text=''):
    print(text)
    lines.append(text)


def run(cmd):
    """Run a shell command; (ok, first line of its output)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        out = (p.stdout + p.stderr).strip()
        return p.returncode == 0, out.splitlines()[0] if out else ''
    except (subprocess.TimeoutExpired, OSError) as err:
        return False, str(err)


def folder_size(path):
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()
               and '__pycache__' not in f.parts)


def port_free(port):
    """True when nothing on the laptop listens on the port yet."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False


def main():
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    say(f'GeoSense Agent 2.0 - Phase 5 deployment inspection ({stamp})')
    say(f'project root: {ROOT}')

    say('\n1. Python and PyTorch')
    say(f'  Python {sys.version.split()[0]}  ({sys.executable})')
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    say(f'  torch {torch.__version__}, device {device}; the container gets the CPU build '
        f'{torch.__version__.split("+")[0]}')

    say('\n2. The site-scorer model')
    import joblib
    import sklearn
    model = joblib.load(ROOT / 'models/saved/site_scorer_model.pkl')
    say(f'  {type(model).__module__}.{type(model).__name__} - scikit-learn '
        f'{sklearn.__version__} here; the pinned requirements keep it in the image')

    say('\n3. Files the backend image copies')
    total = 0
    for rel in IMAGE_FILES:
        p = ROOT / rel
        if not p.exists():
            say(f'  MISSING {rel}')
            continue
        size = folder_size(p) if p.is_dir() else p.stat().st_size
        n = f' ({sum(1 for f in p.iterdir() if f.is_file())} files)' if p.is_dir() else ''
        total += size
        say(f'  {size / 1e6:8.1f} MB  {rel}{"/" if p.is_dir() else ""}{n}')
    say(f'  {total / 1e6:8.1f} MB  total')

    say('\n4. The database')
    env = dotenv_values(ROOT / '.env')
    try:
        sys.path.append(str(ROOT / 'src' / 'phase1_ml'))
        from sqlalchemy import text
        from db_connection import get_engine
        with get_engine().connect() as conn:
            say('  ' + conn.execute(text('SELECT version()')).scalar().split(',')[0])
            say('  PostGIS ' + conn.execute(text('SELECT PostGIS_Full_Version()')).scalar()
                .split('GEOS')[0].strip())
            groups = (('read by the server', READ_TABLES),
                      ('loaded in Phase I, not read by the server', OTHER_TABLES))
            for group, tables in groups:
                say(f'  tables {group}:')
                for t in tables:
                    missing = conn.execute(text('SELECT to_regclass(:t) IS NULL'),
                                           {'t': t}).scalar()
                    if missing:
                        say(f'    {t:16s} missing')
                        continue
                    rows = conn.execute(text(f'SELECT count(*) FROM {t}')).scalar()
                    size = conn.execute(text('SELECT pg_total_relation_size(:t)'),
                                        {'t': t}).scalar()
                    say(f'    {t:16s} {rows:>9,} rows {size / 1e6:8.1f} MB')
            hospitals = conn.execute(text(
                "SELECT (SELECT count(*) FROM osm_poi WHERE fclass = 'hospital') + "
                "(SELECT count(*) FROM osm_poi_area WHERE fclass = 'hospital')")).scalar()
            say(f'  hospitals (points + areas): {hospitals}')
    except Exception as err:                                     # noqa: BLE001
        first = SECRET.sub(r'\1=***', str(err).splitlines()[0])[:150]
        say(f'  database not reachable: {type(err).__name__}: {first}')

    say('\n5. Tools, resources and credentials')
    for name, cmd in (('docker', 'docker --version'),
                      ('compose', 'docker compose version')):
        ok, out = run(cmd)
        say(f'  {name:9s} {out if ok else "not available - " + out[:80]}')
    ok, out = run('docker info --format "{{.ServerVersion}}|{{.OSType}}|{{.NCPU}}|'
                  '{{.MemTotal}}"')
    if ok and out.count('|') == 3:
        version, os_type, ncpu, mem = out.split('|')
        say(f'  daemon    engine {version} ({os_type}); containers may use {ncpu} CPUs and '
            f'{int(mem) / 2**30:.1f} GB of memory')
    else:
        say('  daemon    not running - start Docker Desktop and wait for "Engine running"')
    pg_dump = shutil.which('pg_dump') or next(iter(sorted(
        glob.glob(r'C:\Program Files\PostgreSQL\*\bin\pg_dump.exe'))[::-1]), None)
    pg_dump_version = run(f'"{pg_dump}" --version')[1] if pg_dump else 'not found'
    say(f'  pg_dump   {pg_dump_version}  ({pg_dump})')
    ee_file = Path.home() / '.config' / 'earthengine' / 'credentials'
    say(f'  Earth Engine login file {"present" if ee_file.exists() else "MISSING"} '
        f'({ee_file})')
    free = shutil.disk_usage(ROOT).free / 1e9
    say(f'  disk      {free:.0f} GB free on the project drive (the images need about 4 GB)')
    for port, use in PORTS.items():
        state = 'free' if port_free(port) else 'IN USE - close the program that holds it'
        say(f'  port {port:<5d}{state} ({use})')
    for key in NEEDED_KEYS:
        say(f'  .env {key:16s} {"set" if env.get(key) else "MISSING"}')
    dollar = [k for k, v in env.items() if v and '$' in v]
    verdict = (', '.join(dollar) + ' - tell me before step 6' if dollar
               else 'none (docker compose reads them as written)')
    say(f'  .env values with a $ sign: {verdict}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f'deploy_inspect_{stamp}.txt'
    log.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nreport: {log}')


if __name__ == '__main__':
    main()
