# similar_sites.py
# Purpose: the find_similar_sites tool - candidate sites with a profile like a reference.
#   The reference is scored with the Phase I site scorer. PostGIS then selects the
#   candidates of the Phase I grid (table candidate_sites, load_candidate_sites.py)
#   within radius_km of it and ranks them: the smallest difference in opportunity score
#   first; equal differences by the road/hospital profile (log distances in standard
#   units of the whole grid), then by distance from the reference.
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from site_scorer import score_location            # src/phase1_ml (path set by the server)
from tools.common import ToolError, db_engine, in_study_area, outside_study_area

MIN_SEPARATION_M = 250        # a grid point closer than this is the reference itself
SQL = text('''
WITH ref AS (SELECT ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography AS g),
     spread AS (SELECT stddev_pop(ln(1 + dist_road_m)) AS s_road,
                       stddev_pop(ln(1 + dist_hospital_m)) AS s_hosp
                FROM candidate_sites)
SELECT c.location_id, c.latitude, c.longitude, c.opportunity_score, c.verdict,
       c.dist_road_m, c.dist_hospital_m,
       ST_Distance(c.geometry::geography, ref.g) AS distance_m,
       round(abs(c.opportunity_score - :score)::numeric, 2) AS score_difference,
       sqrt((power((ln(1 + c.dist_road_m) - ln(1 + :road)) / s_road, 2)
             + power((ln(1 + c.dist_hospital_m) - ln(1 + :hosp)) / s_hosp, 2)) / 2)
           AS profile_difference,
       count(*) OVER () AS candidates_in_radius
FROM candidate_sites c, ref, spread
WHERE ST_DWithin(c.geometry::geography, ref.g, :radius_m)
  AND NOT ST_DWithin(c.geometry::geography, ref.g, :min_sep_m)
ORDER BY score_difference, profile_difference, distance_m
LIMIT :top_k
''')


def find_similar_sites(lat, lon, radius_km=5.0, top_k=5):
    """
    Find candidate sites with a profile similar to a reference location.

    Args:
        lat (float): latitude of the reference location in WGS84 decimal degrees
        lon (float): longitude of the reference location in WGS84 decimal degrees
        radius_km (float): search radius around the reference (default 5 km)
        top_k (int): number of similar sites to return (default 5)

    Returns:
        dict: reference (the Phase I score of the location), search (radius, number
              of candidates in it, ranking rule) and similar_sites, best first:
              rank, location_id, latitude, longitude, distance_km, opportunity_score,
              verdict, dist_road_m, dist_hospital_m, score_difference,
              profile_difference

    Example:
        find_similar_sites(22.5530, 88.3540, radius_km=3)['similar_sites'][0]
    """
    if not in_study_area(lat, lon):
        return outside_study_area(lat, lon, 'Phase I site scorer',
                                  ('reference', 'similar_sites'))
    reference = score_location(lat, lon)
    features = reference['features']
    params = {'lat': lat, 'lon': lon, 'score': float(reference['opportunity_score']),
              'road': float(features['dist_road_m']),
              'hosp': float(features['dist_hospital_m']),
              'radius_m': radius_km * 1000, 'min_sep_m': MIN_SEPARATION_M, 'top_k': top_k}
    try:
        with db_engine().connect() as conn:
            rows = conn.execute(SQL, params).mappings().all()
    except ProgrammingError as err:
        code = getattr(err.orig, 'pgcode', None) or getattr(err.orig, 'sqlstate', None)
        if code == '42P01':                                       # undefined table
            raise ToolError('table candidate_sites not found - run python '
                            'src/phase4_agent/load_candidate_sites.py once') from err
        raise
    sites = [{'rank': rank, 'location_id': int(r['location_id']),
              'latitude': round(r['latitude'], 4), 'longitude': round(r['longitude'], 4),
              'distance_km': round(r['distance_m'] / 1000, 2),
              'opportunity_score': r['opportunity_score'], 'verdict': r['verdict'],
              'dist_road_m': round(r['dist_road_m'], 1),
              'dist_hospital_m': round(r['dist_hospital_m'], 1),
              'score_difference': float(r['score_difference']),
              'profile_difference': round(r['profile_difference'], 3)}
             for rank, r in enumerate(rows, start=1)]
    return {'latitude': lat, 'longitude': lon, 'reference': reference,
            'search': {'radius_km': radius_km, 'top_k': top_k,
                       'candidates_in_radius': int(rows[0]['candidates_in_radius'])
                       if rows else 0,
                       'ranked_by': 'difference in opportunity score, then road and '
                                    'hospital profile, then distance'},
            'similar_sites': sites,
            'source': 'PostGIS table candidate_sites - the 1,125 points of the Phase I '
                      'grid scored with the Phase I model'}
