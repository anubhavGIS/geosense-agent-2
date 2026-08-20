# site_scorer.py
# Purpose: Package the trained model into a reusable scoring function.
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from db_connection import get_engine
import shap

METRIC_CRS = 32645   # WGS 84 / UTM zone 45N (Kolkata)
FEATURE_NAMES = ['dist_road_m', 'dist_hospital_m', 'flood_risk']

# Load the model once, when first needed - not on every call
_model = None
_explainer = None

def _load_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load('models/saved/site_scorer_model.pkl')
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer

def _get_features_for_point(lat, lon, engine):
    """Compute the model's features for a single lat/lon point,
    identically to how feature_engineering.py computed them."""
    pt_gdf = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs='EPSG:4326')
    pt_proj = pt_gdf.to_crs(epsg=METRIC_CRS).geometry.iloc[0]

    roads = gpd.read_postgis('SELECT geometry FROM osm_roads',
                             engine, geom_col='geometry').to_crs(epsg=METRIC_CRS)
    dist_road = float(roads.geometry.distance(pt_proj).min())

    pts = gpd.read_postgis("SELECT geometry FROM osm_poi "
                           "WHERE fclass = 'hospital'",
                           engine, geom_col='geometry').to_crs(epsg=METRIC_CRS)
    areas = gpd.read_postgis("SELECT geometry FROM osm_poi_area "
                             "WHERE fclass = 'hospital'",
                             engine, geom_col='geometry').to_crs(epsg=METRIC_CRS)
    areas['geometry'] = areas.geometry.centroid
    hospitals = pd.concat([pts, areas], ignore_index=True)
    dist_hosp = float(hospitals.geometry.distance(pt_proj).min())

    return {'dist_road_m': dist_road,
            'dist_hospital_m': dist_hosp,
            'flood_risk': 0.0}   # placeholder, as in training (Ex. 4 Note 7)

def score_location(lat: float, lon: float) -> dict:
    """Score any location in the study area. Returns a dict with scores,
    features, SHAP explanation and verdict."""
    engine = get_engine()
    model, explainer = _load_model()

    features = _get_features_for_point(lat, lon, engine)
    X = pd.DataFrame([features])[FEATURE_NAMES]

    proba = model.predict_proba(X)[0]
    opportunity_score = round(float(proba[1]) * 10, 2)   # 0-10
    risk_score = round(float(proba[0]) * 10, 2)          # 0-10

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):          # old SHAP: list of 2 arrays
        sv = shap_vals[1][0]
    elif getattr(shap_vals, 'ndim', 2) == 3: # new SHAP: (n, features, classes)
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]
    explanation = {name: round(float(val), 4)
                   for name, val in zip(FEATURE_NAMES, sv)}

    return {
        'latitude': lat,
        'longitude': lon,
        'opportunity_score': opportunity_score,
        'risk_score': risk_score,
        'features': features,
        'shap_explanation': explanation,
        'verdict': 'GOOD SITE' if opportunity_score > 5 else 'POOR SITE',
    }

if __name__ == '__main__':
    for name, (lat, lon) in {
        'Park Street area (central)': (22.5535, 88.3520),
        'North-west fringe':          (22.6550, 88.3050),
    }.items():
        print(f'\n=== {name} ===')
        for key, value in score_location(lat, lon).items():
            print(f'{key:20s}: {value}')