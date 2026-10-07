# export_lidar_geojson.py
# Purpose: Convert LiDAR grid metrics into a GeoJSON file the twin can extrude in 3D
# Version of step 53 (Exercise 4): every grid cell becomes a square polygon in real map
#          coordinates - the cell corners are computed in the projected CRS of the point
#          cloud (UTM 18N for the Dupont Circle tile) and reprojected to WGS84 with pyproj.
#          Only cells in which at least --min-building-pct % of the points are building
#          points are written, so that a cell of trees or street does not become a tower.
# Run from the project root:
#   python src/phase4_agent/export_lidar_geojson.py                      (10 m cells)
#   python src/phase4_agent/export_lidar_geojson.py --cell-size 50 --min-building-pct 0
import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer

sys.path.append('src/phase3_lidar')
from lidar_features import load_classified_cloud, classify_points, compute_grid_metrics

CLOUD = 'data/lidar/processed/study_area_classified.laz'
OUTPUT = 'frontend/digital_twin/public/lidar_buildings.geojson'


def cloud_crs(input_laz):
    """The coordinate reference system written in the LAZ header (read without points)."""
    with laspy.open(input_laz) as reader:
        crs = reader.header.parse_crs()
    return crs if crs is not None else 'EPSG:32618'


def export_geojson(input_laz, output_path, cell_size_m=10, min_building_pct=50):
    """
    Write one square polygon (WGS84) per grid cell that is mostly building.

    Args:
        input_laz (str): the classified point cloud of Phase III
        output_path (str): the GeoJSON file to write
        cell_size_m (float): size of a grid cell in metres
        min_building_pct (float): minimum share of building points in a cell (0-100)

    Returns:
        int: the number of features written

    Example:
        export_geojson(CLOUD, OUTPUT, cell_size_m=10, min_building_pct=50)
    """
    x, y, z, hag, num_returns, return_num = load_classified_cloud(input_laz)
    labels = classify_points(hag, num_returns, return_num)
    metrics = compute_grid_metrics(x, y, hag, labels, cell_size_m)
    x0, y0 = float(x.min()), float(y.min())      # compute_grid_metrics counts from here
    crs = cloud_crs(input_laz)
    to_wgs84 = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    features, heights = [], []
    for cell in metrics:
        if cell['avg_building_height_m'] <= 0 or cell['building_pct'] < min_building_pct:
            continue
        west = x0 + cell['grid_x'] * cell_size_m
        south = y0 + cell['grid_y'] * cell_size_m
        corners = [(west, south), (west + cell_size_m, south),
                   (west + cell_size_m, south + cell_size_m), (west, south + cell_size_m),
                   (west, south)]
        ring = [[round(lon, 7), round(lat, 7)] for lon, lat in
                (to_wgs84.transform(e, n) for e, n in corners)]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
            'properties': {'height_m': cell['avg_building_height_m'],
                           'canopy_pct': cell['canopy_cover_pct'],
                           'building_pct': cell['building_pct'],
                           'cell_size_m': cell_size_m},
        })
        heights.append(cell['avg_building_height_m'])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f)
    lons = [p[0] for f in features for p in f['geometry']['coordinates'][0]]
    lats = [p[1] for f in features for p in f['geometry']['coordinates'][0]]
    print(f'Exported {len(features)} building features to {output_path}')
    print(f'  cells of {cell_size_m} m with >= {min_building_pct} % building points, '
          f'{crs} -> EPSG:4326')
    print(f'  height {min(heights):.1f} - {max(heights):.1f} m (median '
          f'{np.median(heights):.1f} m); lon {min(lons):.5f} to {max(lons):.5f}, '
          f'lat {min(lats):.5f} to {max(lats):.5f}')
    return len(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export LiDAR buildings as GeoJSON.')
    parser.add_argument('--cell-size', type=float, default=10, help='cell size in metres')
    parser.add_argument('--min-building-pct', type=float, default=50,
                        help='minimum share of building points in a cell (%%)')
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    export_geojson(CLOUD, args.output, args.cell_size, args.min_building_pct)
