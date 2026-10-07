# load_candidate_sites.py
# Purpose: score the candidate locations of Phase I (the 0.005-degree grid with its
#          road and hospital distances in data/processed/features.csv) with the
#          Phase I model and store them in PostGIS as table candidate_sites, which the
#          find_similar_sites tool searches
# Run once from the project root (geosense environment):
#   python src/phase4_agent/load_candidate_sites.py
import sys

sys.path.append('src/phase1_ml')

import geopandas as gpd
import joblib
import pandas as pd
from sqlalchemy import text

from db_connection import get_engine
from site_scorer import FEATURE_NAMES

FEATURES_CSV = 'data/processed/features.csv'
MODEL_PATH = 'models/saved/site_scorer_model.pkl'          # as in site_scorer.py
TABLE = 'candidate_sites'


def load_candidate_sites():
    """
    Score every candidate location and write the table candidate_sites to PostGIS.

    Args:
        none - reads FEATURES_CSV and MODEL_PATH relative to the project root

    Returns:
        geopandas.GeoDataFrame: location_id, latitude, longitude, the three features,
            opportunity_score, risk_score, verdict and the point geometry (EPSG:4326)

    Example:
        sites = load_candidate_sites()
        print(sites['verdict'].value_counts())
    """
    sites = pd.read_csv(FEATURES_CSV)
    proba = joblib.load(MODEL_PATH).predict_proba(sites[FEATURE_NAMES])
    sites['opportunity_score'] = (proba[:, 1] * 10).round(2)    # same scale as
    sites['risk_score'] = (proba[:, 0] * 10).round(2)           # score_location()
    sites['verdict'] = sites['opportunity_score'].gt(5).map({True: 'GOOD SITE',
                                                             False: 'POOR SITE'})
    gdf = gpd.GeoDataFrame(sites, crs='EPSG:4326',
                           geometry=gpd.points_from_xy(sites.longitude, sites.latitude))
    engine = get_engine()
    gdf.to_postgis(TABLE, engine, if_exists='replace', index=False)
    with engine.begin() as conn:                 # key + index for distance searches
        conn.execute(text(f'ALTER TABLE {TABLE} ADD PRIMARY KEY (location_id)'))
        conn.execute(text(f'CREATE INDEX {TABLE}_geog_idx ON {TABLE} '
                          f'USING GIST ((geometry::geography))'))
    return gdf


if __name__ == '__main__':
    sites = load_candidate_sites()
    good = int((sites['verdict'] == 'GOOD SITE').sum())
    print(f'Wrote {len(sites):,} candidate sites to PostGIS table {TABLE}')
    print(f'  GOOD SITE {good:,}  |  POOR SITE {len(sites) - good:,}')
    quartiles = sites['opportunity_score'].describe()[['min', '25%', '50%', '75%', 'max']]
    print('  opportunity_score: ' + ', '.join(f'{name} {value:.2f}'
                                            for name, value in quartiles.items()))
    print(f'  distinct score values: {sites["opportunity_score"].nunique()}')
