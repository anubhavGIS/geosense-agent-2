# demographics.py
# Purpose: the get_demographics tool - population density and access to healthcare.
#   Population: WorldPop 2020 (people per ~100 m cell, Google Earth Engine) summed
#   within 1 km of the location - available worldwide.
#   Healthcare: the nearest hospital and the hospitals within 2 km, from the
#   OpenStreetMap hospitals of Phase I in PostGIS (points and area centroids, as in
#   feature_engineering.py) - loaded for the study area only.
import math

from sqlalchemy import text

from tools.common import (cache_get, cache_put, db_engine, earth_engine, in_study_area,
                          study_area_text)

POP_DATASET = 'WorldPop/GP/100m/pop'
POP_YEAR = 2020
RADIUS_KM = 1.0
NEAR_M = 2000
HOSPITALS = text('''
WITH ref AS (SELECT ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography AS g),
     hospital AS (SELECT name, geometry FROM osm_poi WHERE fclass = 'hospital'
                  UNION ALL
                  SELECT name, ST_Centroid(geometry) FROM osm_poi_area
                  WHERE fclass = 'hospital')
SELECT h.name, ST_Distance(h.geometry::geography, ref.g) AS distance_m,
       count(*) FILTER (WHERE ST_DWithin(h.geometry::geography, ref.g, :near_m))
           OVER () AS within_near
FROM hospital h, ref
ORDER BY distance_m
LIMIT 1
''')


def _population(lat, lon):
    """People living within RADIUS_KM of the point (WorldPop), cached: (people, cached)."""
    key = f'worldpop{POP_YEAR}_{lat:.4f}_{lon:.4f}_r{RADIUS_KM:g}km'
    people = cache_get(key)
    if people is not None:
        return people, True
    ee = earth_engine()
    point = ee.Geometry.Point([lon, lat])
    images = (ee.ImageCollection(POP_DATASET).filterBounds(point)
              .filter(ee.Filter.eq('year', POP_YEAR)))
    if images.size().getInfo() == 0:              # no WorldPop raster here (open sea)
        return None, False
    native = images.first().select('population').projection()      # the ~100 m grid
    people = (images.mosaic().select('population')
              .reduceRegion(reducer=ee.Reducer.sum(), crs=native, maxPixels=1e9,
                            geometry=point.buffer(RADIUS_KM * 1000))
              .get('population').getInfo())
    cache_put(key, people)
    return people, False


def _healthcare(lat, lon):
    """Nearest hospital and hospitals within NEAR_M, from PostGIS (study area only)."""
    if not in_study_area(lat, lon):
        return {'nearest_hospital_km': None, 'nearest_hospital_name': None,
                'hospitals_within_2km': None,
                'healthcare_note': 'OpenStreetMap hospitals are loaded for the study '
                                   f'area only ({study_area_text()})'}
    with db_engine().connect() as conn:
        row = conn.execute(HOSPITALS, {'lat': lat, 'lon': lon,
                                       'near_m': NEAR_M}).mappings().first()
    return {'nearest_hospital_km': round(row['distance_m'] / 1000, 2),
            'nearest_hospital_name': row['name'],
            'hospitals_within_2km': int(row['within_near'])}


def get_demographics(lat, lon):
    """
    Population density and access to healthcare around a location.

    Args:
        lat (float): latitude in WGS84 decimal degrees
        lon (float): longitude in WGS84 decimal degrees

    Returns:
        dict: population_density_per_km2 and population_within_1km (WorldPop 2020),
              nearest_hospital_km, nearest_hospital_name and hospitals_within_2km
              (OpenStreetMap; None outside the study area), from_cache, source

    Example:
        get_demographics(22.5530, 88.3540)['population_density_per_km2']
    """
    people, cached = _population(lat, lon)
    area_km2 = math.pi * RADIUS_KM ** 2
    return {'latitude': lat, 'longitude': lon,
            'population_density_per_km2': (round(people / area_km2)
                                           if people is not None else None),
            'population_within_1km': round(people) if people is not None else None,
            'population_year': POP_YEAR,
            **_healthcare(lat, lon),
            'from_cache': cached,
            'source': f'population: WorldPop {POP_YEAR}, 100 m (Google Earth Engine '
                      f'{POP_DATASET}); hospitals: OpenStreetMap in PostGIS (Phase I)'}
