# create_candidates.py
# Purpose: Generate a regular grid of candidate locations across the study area.
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# (min_longitude, min_latitude, max_longitude, max_latitude)
STUDY_AREA_BBOX = (88.30, 22.45, 88.42, 22.67)   # Kolkata (manual example: Chennai)
GRID_SPACING_DEGREES = 0.005   # ~554 m N-S, ~514 m E-W at this latitude

def create_grid():
    min_lon, min_lat, max_lon, max_lat = STUDY_AREA_BBOX
    lons = np.arange(min_lon, max_lon, GRID_SPACING_DEGREES)
    lats = np.arange(min_lat, max_lat, GRID_SPACING_DEGREES)
    points = [Point(lon, lat) for lon in lons for lat in lats]
    gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')
    gdf['location_id'] = range(len(gdf))
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y
    print(f'Created {len(gdf)} candidate locations '
          f'({len(lons)} columns x {len(lats)} rows)')
    return gdf

if __name__ == '__main__':
    grid = create_grid()
    grid.to_file('data/processed/candidate_locations.shp')
    print('Saved candidate locations to data/processed/candidate_locations.shp')