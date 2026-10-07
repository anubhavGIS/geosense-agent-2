# build_flood_zones.py
# Purpose: create the PostGIS table flood_zones for the Kolkata study area, the source of
#          the flood-risk layer of the digital twin (and the table that the Phase I feature
#          engineering reads when it exists). The zones are derived from two open global
#          datasets in Google Earth Engine:
#            MERIT Hydro, band hnd - height above the nearest drainage (HAND), 3 arc-sec
#            JRC Global Surface Water, band occurrence - % of observations 1984-2021 in
#              which a 30 m cell was water
#          Every 30 m cell of the study area gets a risk level:
#            high     HAND <= 2 m, or seasonal water (occurrence 10-89 %)
#            medium   2 m < HAND <= 5 m
#            low      HAND > 5 m
#          Permanent water (occurrence >= 90 %) - the river and the lakes - is left out.
#          The cells are turned into polygons, dissolved by level, simplified by 15 m and
#          written to PostGIS (EPSG:4326, columns risk_level, area_km2, geometry).
# Run from the project root (geosense environment):
#   python src/phase4_agent/build_flood_zones.py
#   python src/phase4_agent/build_flood_zones.py --high 1.5 --medium 4
#   python src/phase4_agent/build_flood_zones.py --geojson   (only the export, see below)
# The layer of the twin is exported with the ST_AsGeoJSON query of Section D.2 in
# DBeaver; --geojson runs the same query from Python and writes the same file.
import argparse
import json
import os
import sys
from pathlib import Path

import ee
import geopandas as gpd
import httpx
import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src' / 'phase1_ml'))
from db_connection import get_engine                                  # noqa: E402

load_dotenv(ROOT / '.env')
BBOX = [float(os.getenv(name, default)) for name, default in
        (('BBOX_MIN_LON', '88.30'), ('BBOX_MIN_LAT', '22.45'),
         ('BBOX_MAX_LON', '88.42'), ('BBOX_MAX_LAT', '22.67'))]
GEE_PROJECT = os.getenv('GEE_PROJECT', 'ee-anubg2000')
CRS, SCALE = 'EPSG:32645', 30                 # the project's metric grid (UTM 45N), 30 m
HAND_ASSET = 'MERIT/Hydro/v1_0_1'
WATER_ASSET = 'JRC/GSW1_4/GlobalSurfaceWater'
LEVELS = {1: 'low', 2: 'medium', 3: 'high'}
PARK_STREET = (88.3540, 22.5530)               # lon, lat - checked after the build
GEOJSON_OUT = ROOT / 'frontend' / 'digital_twin' / 'public' / 'flood_risk.geojson'
GEOJSON_SQL = """\
SELECT jsonb_build_object(
  'type', 'FeatureCollection',
  'features', jsonb_agg(
    jsonb_build_object(
      'type', 'Feature',
      'geometry', ST_AsGeoJSON(geometry)::jsonb,
      'properties', jsonb_build_object('risk_level', risk_level)
    )
  )
) AS geojson
FROM flood_zones"""


def download_layers():
    """
    HAND (m) and water occurrence (%) on a 30 m grid of the study area from Earth Engine.

    Args:
        none - the study area comes from BBOX_* in .env, the project from GEE_PROJECT

    Returns:
        tuple: hand (2-D array, -9999 = no data), occurrence (2-D array, 0-100) and the
               affine transform of the grid (EPSG:32645)

    Example:
        hand, occ, transform = download_layers()
    """
    ee.Initialize(project=GEE_PROJECT)
    region = ee.Geometry.Rectangle(BBOX)
    image = (ee.Image(HAND_ASSET).select('hnd').unmask(-9999).rename('hand')
             .addBands(ee.Image(WATER_ASSET).select('occurrence').unmask(0)
                       .rename('occurrence'))
             .toFloat())
    url = image.getDownloadURL({'region': region, 'scale': SCALE, 'crs': CRS,
                                'format': 'GEO_TIFF'})
    data = httpx.get(url, timeout=300, follow_redirects=True).content
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        return src.read(1), src.read(2), src.transform


def classify(hand, occurrence, high, medium):
    """Risk level per cell: 0 none, 1 low, 2 medium, 3 high (see the header)."""
    levels = np.zeros(hand.shape, dtype=np.uint8)
    valid = hand > -9000
    levels[valid & (hand > medium)] = 1
    levels[valid & (hand > high) & (hand <= medium)] = 2
    levels[valid & (hand <= high)] = 3
    levels[(occurrence >= 10) & (occurrence < 90)] = 3              # seasonal water
    levels[occurrence >= 90] = 0                                    # permanent water
    return levels


def to_polygons(levels, transform):
    """Polygons of the levels, dissolved and simplified, in WGS84."""
    parts = [(shape(geom), int(value)) for geom, value in
             shapes(levels, mask=levels > 0, transform=transform)]
    gdf = gpd.GeoDataFrame({'level': [v for _, v in parts]},
                           geometry=[g for g, _ in parts], crs=CRS)
    gdf = gdf.dissolve(by='level').reset_index()
    gdf['geometry'] = [g if g.geom_type == 'MultiPolygon' else MultiPolygon([g])
                       for g in gdf.geometry.simplify(15)]          # 15 m tolerance
    gdf['risk_level'] = gdf['level'].map(LEVELS)
    gdf['area_km2'] = (gdf.geometry.area / 1e6).round(2)
    gdf = gdf.sort_values('level', ascending=False)
    return gdf.to_crs(epsg=4326)[['risk_level', 'area_km2', 'geometry']]


def write_postgis(zones):
    """Replace the table flood_zones and give it a spatial index."""
    engine = get_engine()
    zones.to_postgis('flood_zones', engine, if_exists='replace', index=False)
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX flood_zones_geom_idx ON flood_zones '
                          'USING GIST (geometry)'))
        level = conn.execute(text(
            'SELECT risk_level FROM flood_zones WHERE ST_Contains(geometry, '
            'ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))'),
            {'lon': PARK_STREET[0], 'lat': PARK_STREET[1]}).scalar()
    return level


def export_geojson(path=GEOJSON_OUT):
    """
    Write the table flood_zones as a GeoJSON FeatureCollection for the digital twin.

    Args:
        path (Path): the file to write (frontend/digital_twin/public/flood_risk.geojson)

    Returns:
        int: the number of features written

    Example:
        export_geojson()
    """
    with get_engine().connect() as conn:
        collection = conn.execute(text(GEOJSON_SQL)).scalar()     # jsonb -> dict
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection), encoding='utf-8')
    count = len(collection['features'])
    print(f'Wrote {count} features to {path.relative_to(ROOT)} '
          f'({path.stat().st_size / 1e6:.2f} MB)')
    return count


def main(high, medium):
    print(f'Study area {BBOX}; HAND from {HAND_ASSET}, water from {WATER_ASSET}')
    hand, occurrence, transform = download_layers()
    valid = hand[hand > -9000]
    p = np.percentile(valid, [5, 25, 50, 75, 95])
    print(f'Grid {hand.shape[1]} x {hand.shape[0]} cells of {SCALE} m; HAND percentiles '
          f'5/25/50/75/95: ' + ' / '.join(f'{v:.1f}' for v in p) + ' m')
    print(f'Water: permanent {100 * (occurrence >= 90).mean():.1f} %, seasonal '
          f'{100 * ((occurrence >= 10) & (occurrence < 90)).mean():.1f} % of the cells')
    levels = classify(hand, occurrence, high, medium)
    land = (levels > 0).sum()
    for code, name in sorted(LEVELS.items(), reverse=True):
        share = 100 * (levels == code).sum() / land
        print(f'  {name:<6} {share:5.1f} % of the land cells')
    zones = to_polygons(levels, transform)
    level = write_postgis(zones)
    print(f'\nWrote {len(zones)} zones to PostGIS table flood_zones:')
    for _, row in zones.iterrows():
        n = len(getattr(row.geometry, 'geoms', [row.geometry]))
        print(f'  {row.risk_level:<6} {row.area_km2:7.2f} km2 in {n:,} polygons')
    print(f'Park Street ({PARK_STREET[1]}, {PARK_STREET[0]}) lies in a '
          f'{level or "no"} flood-risk zone')
    print('Next: export the layer with the ST_AsGeoJSON query in DBeaver (or --geojson)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the flood_zones table.')
    parser.add_argument('--high', type=float, default=2.0, help='HAND limit of high (m)')
    parser.add_argument('--medium', type=float, default=5.0,
                        help='HAND limit of medium (m)')
    parser.add_argument('--geojson', action='store_true',
                        help='only write flood_risk.geojson from the existing table')
    args = parser.parse_args()
    if args.geojson:
        export_geojson()
    else:
        main(args.high, args.medium)
