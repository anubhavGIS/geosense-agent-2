# lidar_features.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 3 (Task E.3.1)
# PURPOSE: Turn the classified point cloud into measurements. Every point is
#          labelled ground / vegetation / building from its height above ground
#          and the number of returns of its laser pulse, and the labels are then
#          aggregated into a 50 m grid: average building height and tree-canopy
#          cover for every cell.
#
# Heuristic classifier (applied to HeightAboveGround = HAG):
#   HAG < 0.5 m                          -> 0 ground
#   0.5 m <= HAG < 2.5 m                 -> 1 low vegetation / shrub
#   HAG >= 2.5 m and more than 1 return  -> 1 tree canopy (light passes through leaves)
#   HAG >= 2.5 m and a single return     -> 2 building (solid roof surface)
#
# INPUT : data/lidar/processed/study_area_classified.laz
# OUTPUT: outputs/lidar_grid_metrics.json
#         outputs/point_clouds/study_area_grid_metrics.png
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/lidar_features.py
# ============================================================
import json
import os
import time

import laspy
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

CLOUD = 'data/lidar/processed/study_area_classified.laz'
OUT_JSON = 'outputs/lidar_grid_metrics.json'
OUT_PNG = 'outputs/point_clouds/study_area_grid_metrics.png'
LABEL_NAMES = {0: 'ground', 1: 'vegetation', 2: 'building'}


def load_classified_cloud(filepath):
    """Read the processed LAZ: x, y, z, HeightAboveGround, returns per pulse, return no."""
    las = laspy.read(filepath)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    num_returns = np.asarray(las.number_of_returns)
    return_num = np.asarray(las.return_number)
    return x, y, z, hag, num_returns, return_num


def classify_points(hag, num_returns, return_num):
    """Heuristic classification: 0 = ground, 1 = vegetation, 2 = building."""
    labels = np.zeros(len(hag), dtype=np.int8)                    # default: ground
    labels[(hag >= 0.5) & (hag < 2.5)] = 1                        # low vegetation
    tree_mask = (hag >= 2.5) & (num_returns > 1)                  # light passes through
    labels[tree_mask] = 1
    building_mask = (hag >= 2.5) & (num_returns == 1)             # one return: a roof
    labels[building_mask] = 2
    return labels


def compute_grid_metrics(x, y, hag, labels, cell_size_m=50):
    """Aggregate the points into cell_size_m x cell_size_m cells (vectorised)."""
    grid_x = ((x - x.min()) // cell_size_m).astype(int)
    grid_y = ((y - y.min()) // cell_size_m).astype(int)
    ncol = int(grid_x.max()) + 1
    key = grid_y * ncol + grid_x                                  # one integer per cell
    n_cells = int(key.max()) + 1
    total = np.bincount(key, minlength=n_cells)
    tree_pts = np.bincount(key, weights=(labels == 1), minlength=n_cells)
    bld_pts = np.bincount(key, weights=(labels == 2), minlength=n_cells)
    bld_hag = np.bincount(key, weights=np.where(labels == 2, hag, 0.0), minlength=n_cells)
    results = []
    for k in np.nonzero(total)[0]:
        avg_height = bld_hag[k] / bld_pts[k] if bld_pts[k] else 0.0
        results.append({
            'grid_x': int(k % ncol),
            'grid_y': int(k // ncol),
            'avg_building_height_m': round(float(avg_height), 2),
            'canopy_cover_pct': round(100 * float(tree_pts[k]) / float(total[k]), 1),
            'building_pct': round(100 * float(bld_pts[k]) / float(total[k]), 1),
            'n_points': int(total[k]),
        })
    return results


def plot_grid(metrics, cell_size_m, out_png):
    """Two maps of the grid: average building height and canopy cover."""
    nx = max(m['grid_x'] for m in metrics) + 1
    ny = max(m['grid_y'] for m in metrics) + 1
    height = np.full((ny, nx), np.nan)
    canopy = np.full((ny, nx), np.nan)
    for m in metrics:
        height[m['grid_y'], m['grid_x']] = m['avg_building_height_m']
        canopy[m['grid_y'], m['grid_x']] = m['canopy_cover_pct']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    im1 = ax1.imshow(height, origin='lower', cmap='magma')
    ax1.set_title(f'Average building height per {cell_size_m} m cell (m)')
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    im2 = ax2.imshow(canopy, origin='lower', cmap='Greens', vmin=0, vmax=100)
    ax2.set_title(f'Canopy cover per {cell_size_m} m cell (% of points)')
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    for ax in (ax1, ax2):
        ax.set_xlabel('cell column (west to east)')
        ax.set_ylabel('cell row (south to north)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print('Saved', out_png)


if __name__ == '__main__':
    t0 = time.time()
    x, y, z, hag, num_returns, return_num = load_classified_cloud(CLOUD)
    labels = classify_points(hag, num_returns, return_num)
    n = len(labels)
    print(f'{n:,} points loaded in {time.time() - t0:.1f} s')
    for code, name in LABEL_NAMES.items():
        k = int((labels == code).sum())
        print(f'  {name:11s}: {k:10,} points ({100 * k / n:5.1f} %)')
    print(f'  building points: mean HAG {hag[labels == 2].mean():.2f} m, '
          f'95th percentile {np.percentile(hag[labels == 2], 95):.2f} m')

    metrics = compute_grid_metrics(x, y, hag, labels)
    print(f'Computed metrics for {len(metrics)} grid cells')
    print(metrics[:3])
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(metrics, f, indent=1)
    print('Saved', OUT_JSON)
    plot_grid(metrics, 50, OUT_PNG)
    print(f'Done in {time.time() - t0:.1f} s')
