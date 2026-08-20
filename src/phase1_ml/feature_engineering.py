# feature_engineering.py
# Purpose: Calculate geospatial features for every candidate location.
import geopandas as gpd
import pandas as pd
import numpy as np
from db_connection import get_engine

# WGS 84 / UTM zone 45N - the metric CRS for the Kolkata study area
# (Exercise 3). Degrees are never used for distance measurement.
METRIC_CRS = 32645

def load_layer(sql, engine):
    return gpd.read_postgis(sql, engine, geom_col='geometry')

def calc_distance_to_nearest(candidates_gdf, layer_gdf, col_name):
    """Distance in metres from each candidate to the nearest feature."""
    cands_proj = candidates_gdf.to_crs(epsg=METRIC_CRS)
    layer_proj = layer_gdf.to_crs(epsg=METRIC_CRS)
    distances = cands_proj.geometry.apply(
        lambda pt: layer_proj.geometry.distance(pt).min()
    )
    return distances

def load_hospitals(engine):
    """Hospitals from BOTH POI tables - points as-is, areas as centroids."""
    pts = load_layer("SELECT geometry FROM osm_poi "
                     "WHERE fclass = 'hospital'", engine)
    areas = load_layer("SELECT geometry FROM osm_poi_area "
                       "WHERE fclass = 'hospital'", engine)
    areas = areas.to_crs(epsg=METRIC_CRS)          # centroid computed in
    areas['geometry'] = areas.geometry.centroid    # metres, not degrees
    areas = areas.to_crs(epsg=4326)
    both = pd.concat([pts, areas], ignore_index=True)
    print(f'  ({len(pts)} hospital points + {len(areas)} hospital areas '
          f'= {len(both)} hospitals)')
    return gpd.GeoDataFrame(both, geometry='geometry', crs='EPSG:4326')

def assign_flood_risk(candidates_gdf, engine):
    """Flood risk 0-10 from a flood_zones table when available; else 0."""
    try:
        flood = load_layer('SELECT geometry, risk_level FROM flood_zones',
                           engine)
        joined = gpd.sjoin(candidates_gdf, flood, how='left',
                           predicate='within')
        risk_map = {'high': 8, 'medium': 5, 'low': 2}
        return joined['risk_level'].map(risk_map).fillna(0)
    except Exception:
        print('  flood_zones table not found - flood_risk set to 0 '
              '(placeholder)')
        return pd.Series(np.zeros(len(candidates_gdf)))

def build_feature_table(candidates_gdf, engine):
    df = candidates_gdf[['location_id', 'latitude', 'longitude']].copy()

    print('Calculating: distance to nearest road ...')
    roads = load_layer('SELECT geometry FROM osm_roads', engine)
    df['dist_road_m'] = calc_distance_to_nearest(candidates_gdf, roads,
                                                 'dist_road_m')

    print('Calculating: distance to nearest hospital ...')
    hospitals = load_hospitals(engine)
    df['dist_hospital_m'] = calc_distance_to_nearest(candidates_gdf,
                                                     hospitals,
                                                     'dist_hospital_m')

    print('Calculating: flood risk score ...')
    df['flood_risk'] = assign_flood_risk(candidates_gdf, engine)

    print('Feature table built with', len(df), 'locations and',
          len(df.columns), 'columns')
    return df

if __name__ == '__main__':
    engine = get_engine()
    cands = gpd.read_file('data/processed/candidate_locations.shp')
    cands = cands.rename(columns={'location_i': 'location_id'})  # undo the
    features = build_feature_table(cands, engine)                # dBase truncation
    features.to_csv('data/processed/features.csv', index=False)
    print('Features saved to data/processed/features.csv')