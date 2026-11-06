"""
restore_cloud_db.py - Phase 5, Exercise 3: load the project tables into the cloud database.

Reads CLOUD_DATABASE_URL from .env - the connection string of the Neon project, copied from
its Connect dialog (the password inside it is never printed) - enables PostGIS there,
restores deploy/geosense_db.sql.gz (the dump of Exercise 2) with the laptop's psql in one
transaction, and compares the row counts of the five tables and one spatial answer with the
laptop database. The cloud database is the one the Cloud Run backend queries; the laptop
keeps its own PostgreSQL.

Run from the project root in the geosense environment:
    python deploy\\restore_cloud_db.py              (enable PostGIS, restore, verify)
    python deploy\\restore_cloud_db.py --check      (verify only: versions and row counts)
    python deploy\\restore_cloud_db.py --replace    (drop the five tables first, then restore)
The report is written to outputs/deploy_logs/cloud_db_<stamp>.txt.
"""
import argparse
import contextlib
import datetime as dt
import glob
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import psycopg2
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / 'deploy' / 'geosense_db.sql.gz'
LOG_DIR = ROOT / 'outputs' / 'deploy_logs'
TABLES = ['osm_roads', 'osm_poi', 'osm_poi_area', 'candidate_sites', 'flood_zones']
PARK_STREET = (88.3540, 22.5530)                  # lon, lat: one spatial answer to compare
TAG = re.compile(r'^(SET|CREATE TABLE|ALTER TABLE|CREATE INDEX|COPY \d+)$')
lines = []                                        # the report


def say(text=''):
    print(text)
    lines.append(text)


def cloud_params(url):
    """libpq parameters from a postgresql:// URL (the password stays in this dict)."""
    u = urlparse(url.strip())
    if u.scheme not in ('postgres', 'postgresql') or not u.hostname:
        sys.exit('CLOUD_DATABASE_URL must look like postgresql://user:password@host/dbname')
    query = {k: v[-1] for k, v in parse_qs(u.query).items()}
    return {'host': u.hostname, 'port': u.port or 5432,
            'dbname': unquote(u.path.lstrip('/')) or 'postgres',
            'user': unquote(u.username or ''), 'password': unquote(u.password or ''),
            'sslmode': query.get('sslmode', 'require'),
            'channel_binding': query.get('channel_binding')}


def masked_host(host):
    """ep-cool-name-a1b2c3d4.us-east-2.aws.neon.tech -> ep-...c3d4.us-east-2.aws.neon.tech"""
    first, dot, rest = host.partition('.')
    if len(first) > 10:
        first = first[:3] + '...' + first[-4:]
    return first + dot + rest


def connect(p, app='geosense-restore'):
    """An open connection (closed by the caller's 'with closing(...)')."""
    kwargs = {k: v for k, v in p.items() if v not in (None, '')}
    return psycopg2.connect(connect_timeout=30, application_name=app, **kwargs)


def server_facts(conn):
    """PostgreSQL version and the installed PostGIS version (None when not enabled)."""
    with conn.cursor() as cur:
        cur.execute('SHOW server_version')
        version = cur.fetchone()[0].split()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'postgis'")
        row = cur.fetchone()
    return version, (row[0] if row else None)


def row_counts(conn):
    """Rows of each of the five tables, None for a table that does not exist."""
    counts = {}
    with conn.cursor() as cur:
        for table in TABLES:
            cur.execute('SELECT to_regclass(%s) IS NOT NULL', (f'public.{table}',))
            if not cur.fetchone()[0]:
                counts[table] = None
                continue
            cur.execute(f'SELECT count(*) FROM public.{table}')
            counts[table] = cur.fetchone()[0]
    return counts


def park_street_zone(conn):
    """The flood-risk zone of Park Street - both databases must give the same answer."""
    with conn.cursor() as cur:
        cur.execute('SELECT to_regclass(%s) IS NOT NULL', ('public.flood_zones',))
        if not cur.fetchone()[0]:
            return 'no flood_zones table'
        cur.execute('SELECT risk_level FROM public.flood_zones WHERE ST_Contains(geometry, '
                    'ST_SetSRID(ST_MakePoint(%s, %s), 4326)) LIMIT 1', PARK_STREET)
        row = cur.fetchone()
    return row[0] if row else 'outside every zone'


def find_psql():
    """The newest psql under Program Files (Windows), else psql on PATH: (path, major)."""
    found = glob.glob(r'C:\Program Files\PostgreSQL\*\bin\psql.exe')
    found.sort(key=lambda p: int(re.search(r'PostgreSQL\\(\d+)', p).group(1)))
    path = found[-1] if found else shutil.which('psql')
    if not path:
        sys.exit('psql not found - install the PostgreSQL client or add its bin folder to PATH')
    text = subprocess.run([path, '--version'], capture_output=True, text=True).stdout.strip()
    return path, int(re.search(r'PostgreSQL\)\s*(\d+)', text).group(1)), text


def restore(p, psql, psql_major, replace=False):
    """Feed the decompressed dump to psql in one transaction (with replace=True the five
    tables are dropped inside the same transaction) and print the command tags."""
    sql = gzip.decompress(DUMP.read_bytes())
    dumped_by = re.search(rb'Dumped by pg_dump version (\d+)', sql[:4000])
    if dumped_by and psql_major < int(dumped_by.group(1)):
        sys.exit(f'the dump was written by pg_dump {int(dumped_by.group(1))} - restore it with '
                 f'psql {int(dumped_by.group(1))} or newer (found psql {psql_major})')
    say(f'restoring {DUMP.relative_to(ROOT)} ({DUMP.stat().st_size / 1e6:.1f} MB, '
        f'{len(sql) / 1e6:.1f} MB of SQL) ...')
    env = {**os.environ, 'PGHOST': p['host'], 'PGPORT': str(p['port']),
           'PGDATABASE': p['dbname'], 'PGUSER': p['user'], 'PGPASSWORD': p['password'],
           'PGSSLMODE': p['sslmode'], 'PGAPPNAME': 'geosense-restore',
           'PGCLIENTENCODING': 'UTF8'}
    if p.get('channel_binding'):
        env['PGCHANNELBINDING'] = p['channel_binding']
    if replace:                    # dropped and reloaded in one transaction, or not at all
        drop = 'DROP TABLE IF EXISTS ' + ', '.join(f'public.{t}' for t in TABLES) + ' CASCADE;\n'
        sql = drop.encode() + sql
    t0 = time.time()
    result = subprocess.run([psql, '-X', '-v', 'ON_ERROR_STOP=1', '--single-transaction',
                             '-f', '-'], input=sql, env=env, capture_output=True)
    seconds = time.time() - t0
    if result.returncode != 0:
        say(result.stderr.decode('utf-8', 'replace')[-1500:])
        sys.exit('restore failed - the transaction was rolled back, the cloud database is '
                 'as it was before')
    tags = [m.group(1) for m in (TAG.match(line.strip()) for line in
                                 result.stdout.decode('utf-8', 'replace').splitlines()) if m]
    copies = [t.split()[1] for t in tags if t.startswith('COPY')]
    summary = ', '.join(f'{name} {sum(t == name for t in tags)}' for name in
                        ('SET', 'CREATE TABLE', 'ALTER TABLE', 'CREATE INDEX'))
    say(f'restored in {seconds:.1f} s: {summary}, COPY {len(copies)} ({", ".join(copies)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='verify only, restore nothing')
    ap.add_argument('--replace', action='store_true', help='drop the five tables first')
    args = ap.parse_args()
    env = {**dotenv_values(ROOT / '.env'), **os.environ}
    if not env.get('CLOUD_DATABASE_URL'):
        sys.exit('CLOUD_DATABASE_URL missing from .env - add the Neon connection string')
    cloud = cloud_params(env['CLOUD_DATABASE_URL'])
    laptop = {'host': env.get('DB_HOST', 'localhost'), 'port': int(env.get('DB_PORT', 5432)),
              'dbname': env.get('DB_NAME'), 'user': env.get('DB_USER'),
              'password': env.get('DB_PASSWORD'), 'sslmode': env.get('DB_SSLMODE') or 'prefer'}
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    say(f'GeoSense cloud database - {"check" if args.check else "restore and check"} ({stamp})')
    say(f'cloud:  {masked_host(cloud["host"])}  db {cloud["dbname"]}  user {cloud["user"]}  '
        f'sslmode {cloud["sslmode"]}  (password: {len(cloud["password"])} characters)')
    say(f'laptop: {laptop["host"]}:{laptop["port"]}  db {laptop["dbname"]}')

    with contextlib.closing(connect(cloud)) as conn:
        conn.autocommit = True
        version, postgis = server_facts(conn)
        if postgis is None and not args.check:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS postgis')
            postgis = f'{server_facts(conn)[1]} (enabled now)'
        say(f'cloud server: PostgreSQL {version}, PostGIS {postgis or "not enabled"}')
        existing = [t for t, n in row_counts(conn).items() if n is not None]
        if existing and not args.check and not args.replace:
            sys.exit(f'already in the cloud database: {", ".join(existing)} - run with '
                     '--check to verify, or --replace to load the dump again')

    if not args.check:
        psql, major, text = find_psql()
        say(f'psql: {psql} ({text})')
        if args.replace and existing:
            say(f'--replace: {len(existing)} existing table(s) are dropped in the same '
                'transaction as the restore')
        restore(cloud, psql, major, replace=args.replace)

    with contextlib.closing(connect(cloud)) as conn_c, \
            contextlib.closing(connect(laptop, 'geosense-check')) as conn_l:
        cloud_counts, laptop_counts = row_counts(conn_c), row_counts(conn_l)
        zones = park_street_zone(conn_l), park_street_zone(conn_c)
    say()
    say(f'{"table":18s} {"laptop":>9s} {"cloud":>9s}  match')
    matches = 0
    for table in TABLES:
        a, b = laptop_counts[table], cloud_counts[table]
        same = a is not None and a == b
        matches += same
        fmt = lambda n: f'{n:,}' if n is not None else 'missing'
        say(f'{table:18s} {fmt(a):>9s} {fmt(b):>9s}  {"yes" if same else "NO"}')
    say(f'flood zone at Park Street: laptop {zones[0]}, cloud {zones[1]}')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f'cloud_db_{stamp}.txt'
    verdict = ('all five tables match' if matches == len(TABLES) and zones[0] == zones[1]
               else 'MISMATCH - see the table above')
    say(f'{verdict}; report: {out.relative_to(ROOT)}')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sys.exit(0 if verdict.startswith('all') else 1)


if __name__ == '__main__':
    main()
