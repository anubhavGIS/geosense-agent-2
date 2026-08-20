# load_osm_data.py
# Purpose: Load the OSM shapefile layers into PostGIS, clipped to the study area.
import os
import geopandas as gpd
from db_connection import get_engine

DATA_DIR = 'data/shapefiles/osm/'

# Study area padded by 0.02 deg on every side (same padding as the DEM),
# so candidates at the grid edge still have nearby features to measure to.
# The Eastern Zone extract covers four states; the bbox filter makes
# GeoPandas read ONLY features intersecting this box instead of all of it.
CLIP_BBOX = (88.28, 22.43, 88.44, 22.69)

LAYERS_TO_LOAD = [
    ('gis_osm_roads_free_1.shp',       'osm_roads'),
    ('gis_osm_buildings_a_free_1.shp', 'osm_buildings'),  # _a_ suffix: actual filename in the extract
    ('gis_osm_landuse_a_free_1.shp',   'osm_landuse'),
    ('gis_osm_pois_free_1.shp',        'osm_poi'),
    ('gis_osm_pois_a_free_1.shp',      'osm_poi_area'),   # POIs mapped as areas (hospitals, malls, campuses)
]

def load_layer(filename, table_name, engine):
    filepath = os.path.join(DATA_DIR, filename)
    print(f'Loading {filename} ...')
    gdf = gpd.read_file(filepath, bbox=CLIP_BBOX)
    gdf = gdf.to_crs(epsg=4326)   # already WGS 84; kept as an explicit guarantee
    gdf.to_postgis(table_name, engine, if_exists='replace', index=False)
    print(f'  Loaded {len(gdf)} features into table: {table_name}')

def main():
    engine = get_engine()
    for filename, table_name in LAYERS_TO_LOAD:
        load_layer(filename, table_name, engine)
    print('All layers loaded successfully!')

if __name__ == '__main__':
    main()