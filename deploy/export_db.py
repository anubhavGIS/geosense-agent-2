"""
export_db.py - Phase 5, Exercise 2: dump the project tables the backend reads to
deploy/geosense_db.sql.gz, the file the db container restores on its first start (Part C.4)
and the cloud database is loaded from in Exercise 3.

Only the five tables the MCP server queries are exported - osm_roads, osm_poi,
osm_poi_area (the site scorer and the demographics tool), candidate_sites (similar sites)
and flood_zones (the climate tool); osm_buildings and osm_landuse were loaded in Phase I
but no tool reads them. pg_dump writes plain SQL without owners or privileges, so that the
tables are created by whichever user restores them, and Python compresses the result.
The connection settings come from .env (python-dotenv); the password is passed to pg_dump
through the environment of its process only and is never printed.

Run from the project root in the geosense environment:
    python deploy\\export_db.py                 (dump + compress + report)
    python deploy\\export_db.py --keep-sql      (also keep the uncompressed .sql)
"""
import argparse
import glob
import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deploy' / 'geosense_db.sql.gz'
TABLES = ['osm_roads', 'osm_poi', 'osm_poi_area', 'candidate_sites', 'flood_zones']


def find_pg_dump():
    """pg_dump on PATH, else the newest PostgreSQL installation under Program Files."""
    found = shutil.which('pg_dump')
    if found:
        return found
    candidates = sorted(glob.glob(r'C:\Program Files\PostgreSQL\*\bin\pg_dump.exe'))
    if not candidates:
        sys.exit('pg_dump not found - add PostgreSQL\\<version>\\bin to PATH')
    return candidates[-1]


def table_report(env):
    """Rows and size of each exported table, and the server versions, from the database."""
    url = (f"postgresql://{env['DB_USER']}:{env['DB_PASSWORD']}@{env['DB_HOST']}:"
           f"{env['DB_PORT']}/{env['DB_NAME']}")
    engine = create_engine(url)
    with engine.connect() as conn:
        version = conn.execute(text('SHOW server_version')).scalar()
        postgis = conn.execute(text('SELECT PostGIS_Version()')).scalar()
        print(f'PostgreSQL {version}, PostGIS {postgis}')
        total = 0
        for table in TABLES:
            exists = conn.execute(text('SELECT to_regclass(:t) IS NOT NULL'),
                                  {'t': table}).scalar()
            if not exists:
                sys.exit(f'table {table} does not exist - run the Phase I/IV loaders first')
            rows = conn.execute(text(f'SELECT count(*) FROM {table}')).scalar()
            size = conn.execute(text('SELECT pg_total_relation_size(:t)'),
                                {'t': table}).scalar()
            total += size
            print(f'  {table:16s} {rows:>9,} rows  {size / 1e6:8.1f} MB')
        print(f'  {"total":16s} {"":>9}       {total / 1e6:8.1f} MB')
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep-sql', action='store_true', help='keep the uncompressed .sql')
    args = ap.parse_args()
    env = {**dotenv_values(ROOT / '.env'), **os.environ}
    for key in ('DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'):
        if not env.get(key):
            sys.exit(f'{key} missing from .env')
    version = table_report(env)
    pg_dump = find_pg_dump()
    print(f'pg_dump: {pg_dump}')
    sql = OUT.with_suffix('')                             # deploy/geosense_db.sql
    cmd = [pg_dump, '-h', env['DB_HOST'], '-p', env['DB_PORT'], '-U', env['DB_USER'],
           '-d', env['DB_NAME'], '--no-owner', '--no-privileges', '--no-tablespaces',
           '--encoding=UTF8', '-f', str(sql)]
    for table in TABLES:
        cmd += ['-t', f'public.{table}']
    t0 = time.time()
    run_env = {**os.environ, 'PGPASSWORD': env['DB_PASSWORD']}   # this process only
    result = subprocess.run(cmd, env=run_env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'pg_dump failed:\n{result.stderr}')
    with open(sql, 'rb') as src, gzip.open(OUT, 'wb', compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    plain, packed = sql.stat().st_size, OUT.stat().st_size
    if not args.keep_sql:
        sql.unlink()
    print(f'{OUT.relative_to(ROOT)}: {packed / 1e6:.1f} MB compressed from '
          f'{plain / 1e6:.1f} MB of SQL in {time.time() - t0:.1f} s (PostgreSQL {version} '
          f'dump - restore it with the same or a newer major version)')


if __name__ == '__main__':
    main()
