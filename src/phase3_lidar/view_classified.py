# view_classified.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 2 (check of Task E.2.1)
# PURPOSE: Confirm that the processed cloud separates ground from elevated
#          points. Saves a two-panel figure (plan view coloured by height above
#          ground, and an east-west cross-section through the tile centre) and,
#          with --show, opens the cloud in the Open3D viewer: ground brown,
#          everything else coloured by HeightAboveGround.
#
# INPUT : data/lidar/processed/study_area_classified.laz
# OUTPUT: outputs/point_clouds/study_area_classified_preview.png
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/view_classified.py --show
# ============================================================
import argparse
import os

import laspy
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

IN_LAZ = 'data/lidar/processed/study_area_classified.laz'
OUT_PNG = 'outputs/point_clouds/study_area_classified_preview.png'
GROUND_RGB = (0.55, 0.35, 0.15)                                   # brown


def load(path):
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    return x, y, z, cls, hag


def save_figure(x, y, z, cls, hag, out_png, step=4):
    """Plan view (every `step`-th point) and a 20 m wide east-west section."""
    ground = cls == 2
    sel = slice(None, None, step)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11),
                                   gridspec_kw={'height_ratios': [3, 1.2]})
    ax1.scatter(x[sel][ground[sel]], y[sel][ground[sel]], s=0.2,
                c=[GROUND_RGB], label='ground (class 2)')
    sc = ax1.scatter(x[sel][~ground[sel]], y[sel][~ground[sel]], s=0.2,
                     c=np.clip(hag[sel][~ground[sel]], 0, 40), cmap='viridis')
    ax1.set_aspect('equal')
    ax1.set_title('Classified tile: ground (brown), other points by height above ground',
                  fontsize=10)
    ax1.set_xlabel('Easting (m, UTM 18N)')
    ax1.set_ylabel('Northing (m)')
    cb = fig.colorbar(sc, ax=ax1, fraction=0.03, pad=0.02)
    cb.set_label('Height above ground (m, clipped at 40)')
    ax1.legend(loc='upper right', markerscale=20)

    y_mid = (y.min() + y.max()) / 2
    band = np.abs(y - y_mid) < 10
    ax2.scatter(x[band & ground], z[band & ground], s=0.3, c=[GROUND_RGB])
    ax2.scatter(x[band & ~ground], z[band & ~ground], s=0.3,
                c=np.clip(hag[band & ~ground], 0, 40), cmap='viridis')
    ax2.set_title(f'East-west section, 20 m wide, through northing {y_mid:.0f}')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Elevation (m)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print('Saved', out_png)


def show_open3d(x, y, z, cls, hag):
    import open3d as o3d
    pts = np.column_stack((x - x.min(), y - y.min(), z))
    colors = np.empty((len(pts), 3))
    ground = cls == 2
    colors[ground] = GROUND_RGB
    t = np.clip(hag[~ground], 0, 40) / 40.0                         # 0 .. 1
    colors[~ground] = plt.get_cmap('viridis')(t)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print('Opening the Open3D viewer - rotate with the left mouse button, '
          'zoom with the wheel, close the window to finish')
    o3d.visualization.draw_geometries([pcd], window_name='GeoSense - classified cloud',
                                      width=1400, height=900)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true', help='open the Open3D viewer')
    args = ap.parse_args()

    x, y, z, cls, hag = load(IN_LAZ)
    n = len(x)
    print(f'{n:,} points; ground {int((cls == 2).sum()):,}; '
          f'HAG < -0.5 m: {int((hag < -0.5).sum()):,}; '
          f'HAG > 2.5 m: {int((hag > 2.5).sum()):,}')
    save_figure(x, y, z, cls, hag, OUT_PNG)
    if args.show:
        show_open3d(x, y, z, cls, hag)
