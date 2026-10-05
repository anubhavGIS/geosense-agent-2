# visualize_cloud.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 3 (Task E.3.2)
# PURPOSE: Colour-code the classified point cloud - ground brown, vegetation
#          green, buildings grey - and save it as a PLY file that opens in the
#          Open3D viewer or CloudCompare.
#
# INPUT : data/lidar/processed/study_area_classified.laz
# OUTPUT: outputs/point_clouds/study_area_colored.ply
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/visualize_cloud.py          (write the PLY)
#   python src/phase3_lidar/visualize_cloud.py --show   (write it and open the viewer)
# ============================================================
import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lidar_features import load_classified_cloud, classify_points     # noqa: E402

IN_LAZ = 'data/lidar/processed/study_area_classified.laz'
OUT_PLY = 'outputs/point_clouds/study_area_colored.ply'
# ground = brown, vegetation = green, building = grey
COLOR_MAP = {0: [0.55, 0.35, 0.15], 1: [0.10, 0.60, 0.10], 2: [0.60, 0.60, 0.60]}


def save_colored_ply(input_laz, output_ply):
    """Classify the cloud, colour every point by its label and write a PLY."""
    x, y, z, hag, num_returns, return_num = load_classified_cloud(input_laz)
    labels = classify_points(hag, num_returns, return_num)
    points = np.column_stack((x, y, z))
    palette = np.array([COLOR_MAP[k] for k in range(3)])          # 3 x 3 lookup table
    colors = palette[labels]                                      # one row per point
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f'Saved colour-coded point cloud to {output_ply} '
          f'({len(points):,} points, {os.path.getsize(output_ply) / 1e6:.1f} MB)')
    return pcd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true', help='open the cloud in Open3D')
    args = ap.parse_args()
    t0 = time.time()
    pcd = save_colored_ply(IN_LAZ, OUT_PLY)
    print(f'Done in {time.time() - t0:.1f} s')
    if args.show:
        # shift to a local origin so the viewer does not lose precision on UTM numbers
        pcd.translate(-pcd.get_min_bound())
        print('Opening the Open3D viewer - close the window (or press Q) to finish')
        o3d.visualization.draw_geometries([pcd], window_name='GeoSense - classified cloud',
                                          width=1400, height=900)
