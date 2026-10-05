# lidar_analyzer.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 5 (Task E.5.1) - the main LiDAR deliverable
# PURPOSE: One call, one bounding box, one evidence-based summary. The classified
#          point cloud of Exercise 2 is loaded once and labelled ground / vegetation /
#          building with the Exercise 3 heuristic; the WGS84 box is reprojected into
#          the cloud's own CRS, the points inside it are selected and summarised:
#          average building height, canopy cover, impervious surface, slope risk.
#          Three refinements over the per-point measures of Exercise 3:
#            - canopy and impervious shares are measured as AREA (1 m cells labelled
#              by their highest return), the point shares are kept beside them;
#            - the slope risk is computed on a 5 m ground model (median ground
#              elevation per cell), not on the raw ground points;
#            - large flat roofs that the ground filter accepted as ground appear in
#              that model as plateaus; their points are given back to the buildings.
#
# INPUT : bounding box (min_lat, min_lon, max_lat, max_lon) in WGS84 decimal degrees
#         data/lidar/processed/study_area_classified.laz  (PROCESSED_CLOUD_PATH in .env)
# OUTPUT: dict - avg_building_height_m, canopy_cover_pct, impervious_surface_pct,
#         slope_risk_score, total_points_analyzed, lidar_coverage and a few extras
#
# Run from the project root (geosense environment):
#   python src/phase3_lidar/lidar_analyzer.py                       (the demo boxes)
#   python src/phase3_lidar/lidar_analyzer.py --bbox 38.9085 -77.0450 38.9110 -77.0420
#   python src/phase3_lidar/lidar_analyzer.py --bbox ... --name "Dupont Circle" --figure
# ============================================================
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pyproj import CRS, Transformer
from scipy.ndimage import minimum_filter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lidar_features import load_classified_cloud, classify_points     # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]                       # the project folder
load_dotenv(_ROOT / '.env')
PROCESSED_CLOUD_PATH = os.getenv('PROCESSED_CLOUD_PATH',
                                 'data/lidar/processed/study_area_classified.laz')
FALLBACK_CRS = os.getenv('LIDAR_CRS', 'EPSG:32618')              # if the header has none
CELL_SIZE_M = 1.0            # area raster: one label per 1 m cell (its highest return)
DTM_CELL_M = 5.0             # ground model: median ground elevation per 5 m cell
PLATEAU_WINDOW_M = 50.0      # a ground cell this much above the lowest ground cell
PLATEAU_SLOPE = 0.15         # within 50 m (3 m + 0.15 x 25 m = 6.75 m) is a roof that
PLATEAU_MIN_M = 3.0          # the ground filter accepted as ground, not terrain
LABEL_NAMES = {0: 'ground', 1: 'vegetation', 2: 'building'}

# The manual's Chennai example (no LiDAR there) and four boxes inside the
# Dupont Circle tile of Exercise 2: (name, min_lat, min_lon, max_lat, max_lon)
DEMO_BOXES = [
    ('Chennai centre (manual example)', 13.05, 80.24, 13.06, 80.25),
    ('Whole Dupont Circle tile', 38.905, -77.048, 38.913, -77.038),
    ('Dupont Circle park and radial avenues', 38.9085, -77.0450, 38.9110, -77.0420),
    ('Office blocks south-east of the circle', 38.9050, -77.0420, 38.9075, -77.0390),
    ('Row houses north-west of the circle', 38.9105, -77.0480, 38.9130, -77.0450),
]

_cloud = None                                                     # loaded once, reused


def _resolve(path):
    """Paths in .env are relative to the project root, wherever Python was started."""
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _load_cloud(path=None):
    """Read the processed cloud once: coordinates, HAG, heuristic labels and the CRS."""
    global _cloud
    if _cloud is None or (path is not None and _cloud['path'] != str(path)):
        import laspy
        full = _resolve(path or PROCESSED_CLOUD_PATH)
        if not full.exists():
            raise FileNotFoundError(f'Processed point cloud not found: {full}')
        t0 = time.time()
        x, y, z, hag, num_returns, return_num = load_classified_cloud(str(full))
        labels = classify_points(hag, num_returns, return_num)
        with laspy.open(str(full)) as f:                          # header only
            crs = f.header.parse_crs()
        crs_source = 'LAS header'
        if crs is None:                                           # header without a VLR
            crs, crs_source = CRS.from_user_input(FALLBACK_CRS), 'LIDAR_CRS fallback'
        _cloud = {'path': str(path or PROCESSED_CLOUD_PATH), 'x': x, 'y': y, 'z': z,
                  'hag': hag, 'labels': labels, 'crs': crs, 'crs_source': crs_source,
                  'extent': (float(x.min()), float(y.min()),
                             float(x.max()), float(y.max())),
                  'to_cloud': Transformer.from_crs('EPSG:4326', crs, always_xy=True),
                  'to_wgs84': Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)}
        epsg = crs.to_epsg()
        print(f'Loaded {len(x):,} points in {time.time() - t0:.1f} s; '
              f'CRS {"EPSG:" + str(epsg) if epsg else crs.name} ({crs_source})')
    return _cloud


def _compute_slope_risk(z_values):
    """A simple proxy: higher elevation variance = steeper terrain = higher risk (0-10)."""
    if len(z_values) < 2:
        return 0.0
    std = float(np.std(z_values))
    return round(min(10.0, std * 2), 2)


def bbox_to_cloud_crs(min_lat, min_lon, max_lat, max_lon, transformer):
    """
    Reproject a WGS84 box into the point cloud's CRS.

    The four corners are transformed (a geographic rectangle is a slightly rotated
    quadrilateral in UTM) and their enclosing rectangle is returned, so that no corner
    of the requested area is cut off.
    """
    lons = [min_lon, max_lon, max_lon, min_lon]
    lats = [min_lat, min_lat, max_lat, max_lat]
    xs, ys = transformer.transform(lons, lats)
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _coverage_pct(request, extent):
    """Share (%) of the requested rectangle that lies inside the cloud's extent."""
    ix = max(0.0, min(request[2], extent[2]) - max(request[0], extent[0]))
    iy = max(0.0, min(request[3], extent[3]) - max(request[1], extent[1]))
    area = (request[2] - request[0]) * (request[3] - request[1])
    return round(100.0 * ix * iy / area, 1) if area > 0 else 0.0


def _cell_index(x, y, cell_m):
    """Row/column of every point on a grid anchored at the points' own minimum."""
    col = ((x - x.min()) // cell_m).astype(np.int64)
    row = ((y - y.min()) // cell_m).astype(np.int64)
    return row, col, int(row.max()) + 1, int(col.max()) + 1


def ground_model(x, y, z, labels, cell_m=DTM_CELL_M):
    """
    A small digital terrain model of the box from the ground-labelled points.

    Every cell takes the MEDIAN elevation of its ground points (robust against the few
    noise points below the surface). A cell more than PLATEAU_MIN_M + PLATEAU_SLOPE x
    half-window above the lowest cell within PLATEAU_WINDOW_M cannot be terrain at the
    slopes of a city: it is a large flat roof that the ground filter accepted as ground.
    Returns a dict with the model (plateaus set to NaN), the plateau mask, the local
    minimum surface and the grid geometry.
    """
    ground = labels == 0
    if ground.sum() == 0:
        return None
    xg, yg, zg = x[ground], y[ground], z[ground]
    row, col, nrow, ncol = _cell_index(xg, yg, cell_m)
    key = row * ncol + col
    order = np.lexsort((zg, key))                                 # by cell, then by z
    key_s, z_s = key[order], zg[order]
    cells, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    dtm = np.full(nrow * ncol, np.nan)
    dtm[cells] = z_s[starts + counts // 2]                        # median per cell
    dtm = dtm.reshape(nrow, ncol)
    window = int(round(PLATEAU_WINDOW_M / cell_m)) | 1            # odd number of cells
    local_min = minimum_filter(np.where(np.isnan(dtm), np.inf, dtm), size=window,
                               mode='nearest')
    plateau = (dtm - local_min) > PLATEAU_MIN_M + PLATEAU_SLOPE * PLATEAU_WINDOW_M / 2
    dtm[plateau] = np.nan
    return {'dtm': dtm, 'plateau': plateau, 'local_min': local_min, 'cell_m': cell_m,
            'x0': float(xg.min()), 'y0': float(yg.min()), 'shape': (nrow, ncol)}


def recover_roofs(x, y, z, hag, labels, gm):
    """
    Give the plateau cells back to the buildings.

    Ground-labelled points that fall in a plateau cell are relabelled building and their
    height above ground becomes elevation minus the local minimum of the ground model.
    Returns (hag, labels, number of relabelled points) - copies, the cloud is untouched.
    """
    hag, labels = hag.copy(), labels.copy()
    if gm is None or not gm['plateau'].any():
        return hag, labels, 0
    nrow, ncol = gm['shape']
    row = np.clip(((y - gm['y0']) // gm['cell_m']).astype(np.int64), 0, nrow - 1)
    col = np.clip(((x - gm['x0']) // gm['cell_m']).astype(np.int64), 0, ncol - 1)
    on_plateau = gm['plateau'][row, col] & (labels == 0)
    labels[on_plateau] = 2
    local_ground = gm['local_min'][row, col][on_plateau]
    hag[on_plateau] = np.maximum(0.0, z[on_plateau] - local_ground)
    return hag, labels, int(on_plateau.sum())


def area_shares(x, y, z, labels, cell_m=CELL_SIZE_M):
    """
    Label shares by AREA instead of by point count.

    The points are binned into cell_m cells; in every cell the highest return decides
    the label of the cell (what a map viewed from above shows there), and the cells are
    counted. A tree crown with 30 returns per m2 and a roof with 5 then weigh the same
    per square metre. Returns ({label: % of cells}, number of cells, the label raster).
    """
    if len(x) == 0:
        return {name: 0.0 for name in LABEL_NAMES.values()}, 0, None
    row, col, nrow, ncol = _cell_index(x, y, cell_m)
    key = row * ncol + col
    order = np.lexsort((z, key))                                  # by cell, then by z
    key_s = key[order]
    last = np.r_[np.nonzero(np.diff(key_s))[0], len(key_s) - 1]  # highest return/cell
    raster = np.full(nrow * ncol, -1, dtype=np.int8)
    raster[key_s[last]] = labels[order][last]
    top = raster[raster >= 0]
    counts = np.bincount(top, minlength=3)
    shares = {name: round(100.0 * float(counts[code]) / len(top), 1)
              for code, name in LABEL_NAMES.items()}
    return shares, int(len(top)), raster.reshape(nrow, ncol)


def analyze_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                 fix_roofs: bool = True, keep_rasters: bool = False) -> dict:
    """
    Analyse LiDAR-derived metrics for a bounding box.

    Args:
        min_lat, min_lon, max_lat, max_lon (float): bounding box corners in WGS84
            decimal degrees (south, west, north, east)
        fix_roofs (bool): relabel the flat roofs that the ground filter accepted as
            ground (default True); False reproduces the Exercise 3 labels exactly
        keep_rasters (bool): also return the label raster and the ground model
            (for figures; not JSON-serialisable)

    Returns:
        dict: bbox, bbox_projected, crs, lidar_coverage (bool), coverage_pct,
              avg_building_height_m (mean HAG of the building points),
              canopy_cover_pct (% of 1 m cells whose highest return is vegetation),
              impervious_surface_pct (% of 1 m cells whose highest return is
              building - roofs only; roads and pavements are 'ground' here),
              slope_risk_score (std of the 5 m ground model x 2, capped at 10),
              ground_relief_m (5th-95th percentile range of the ground model),
              median_slope_pct (median gradient of the ground model),
              point_shares_pct (the Exercise 3 measure: % of points per label),
              roofs_recovered (plateau cells, relabelled points, % of the area),
              total_points_analyzed, cells_1m
            When the box holds no points every metric is 0.0, lidar_coverage is
            False and a note says what the processed cloud covers.

    Example:
        result = analyze_bbox(38.9085, -77.0450, 38.9110, -77.0420)
        print(result['avg_building_height_m'])      # e.g., 16.3
    """
    cloud = _load_cloud()
    request = bbox_to_cloud_crs(min_lat, min_lon, max_lat, max_lon, cloud['to_cloud'])
    xmin, ymin, xmax, ymax = request
    inside = ((cloud['x'] >= xmin) & (cloud['x'] <= xmax) &
              (cloud['y'] >= ymin) & (cloud['y'] <= ymax))
    n = int(inside.sum())
    epsg = cloud['crs'].to_epsg()
    result = {'bbox': [min_lat, min_lon, max_lat, max_lon],
              'bbox_projected': [round(v, 1) for v in request],
              'crs': f'EPSG:{epsg}' if epsg else cloud['crs'].name,
              'lidar_coverage': n > 0,
              'coverage_pct': _coverage_pct(request, cloud['extent'])}
    if n == 0:
        w, s = cloud['to_wgs84'].transform(cloud['extent'][0], cloud['extent'][1])
        e, nn = cloud['to_wgs84'].transform(cloud['extent'][2], cloud['extent'][3])
        result.update({'avg_building_height_m': 0.0, 'canopy_cover_pct': 0.0,
                       'impervious_surface_pct': 0.0, 'slope_risk_score': 0.0,
                       'ground_relief_m': 0.0, 'median_slope_pct': 0.0,
                       'point_shares_pct': {k: 0.0 for k in LABEL_NAMES.values()},
                       'roofs_recovered': {'cells_5m': 0, 'points': 0, 'area_pct': 0.0},
                       'total_points_analyzed': 0, 'cells_1m': 0,
                       'note': (f'no LiDAR points inside the box; the processed cloud'
                                f' covers lon {w:.4f} to {e:.4f}, '
                                f'lat {s:.4f} to {nn:.4f}')})
        return result

    x, y, z = cloud['x'][inside], cloud['y'][inside], cloud['z'][inside]
    hag, labels = cloud['hag'][inside], cloud['labels'][inside]
    gm = ground_model(x, y, z, labels)
    n_fixed = 0
    if fix_roofs:
        hag, labels, n_fixed = recover_roofs(x, y, z, hag, labels, gm)
    shares, n_cells, raster = area_shares(x, y, z, labels)
    counts = np.bincount(labels, minlength=3)
    building = hag[labels == 2]

    slope_risk, relief, median_slope, n_plateau = 0.0, 0.0, 0.0, 0
    if gm is not None:
        cells = gm['dtm'][~np.isnan(gm['dtm'])]
        n_plateau = int(gm['plateau'].sum())
        if len(cells) >= 2:
            slope_risk = _compute_slope_risk(cells)
            relief = float(np.percentile(cells, 95) - np.percentile(cells, 5))
            gy, gx = np.gradient(gm['dtm'], gm['cell_m'])
            median_slope = float(np.nanmedian(np.hypot(gx, gy)) * 100)
    plateau_area = n_plateau * DTM_CELL_M ** 2
    box_area = (xmax - xmin) * (ymax - ymin)
    result.update({
        'avg_building_height_m': (round(float(building.mean()), 2) if len(building)
                                  else 0.0),
        'canopy_cover_pct': shares['vegetation'],
        'impervious_surface_pct': shares['building'],
        'slope_risk_score': slope_risk,
        'ground_relief_m': round(relief, 2),
        'median_slope_pct': round(median_slope, 2),
        'point_shares_pct': {name: round(100.0 * float(counts[code]) / n, 1)
                             for code, name in LABEL_NAMES.items()},
        'roofs_recovered': {'cells_5m': n_plateau, 'points': n_fixed,
                            'area_pct': round(100.0 * plateau_area / box_area, 1)},
        'total_points_analyzed': n,
        'cells_1m': n_cells,
    })
    if keep_rasters:
        result['_raster'] = raster
        result['_ground_model'] = gm
    return result


def plot_box(result, out_png):
    """Three panels: label raster (area), ground model with the recovered roofs, legend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    raster, gm = result['_raster'], result['_ground_model']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    cmap = ListedColormap([[1, 1, 1], [0.55, 0.35, 0.15], [0.1, 0.6, 0.1],
                           [0.6, 0.6, 0.6]])
    ax1.imshow(raster + 1, origin='lower', cmap=cmap, vmin=0, vmax=3, interpolation='none')
    ax1.set_title(f'Highest return per 1 m cell: canopy {result["canopy_cover_pct"]} %, '
                  f'buildings {result["impervious_surface_pct"]} % of the area',
                  fontsize=9)
    ax1.legend(handles=[Patch(color=[0.55, 0.35, 0.15], label='ground'),
                        Patch(color=[0.1, 0.6, 0.1], label='vegetation'),
                        Patch(color=[0.6, 0.6, 0.6], label='building')],
               loc='lower right', fontsize=8)
    if gm is not None:
        valid = gm['dtm'][~np.isnan(gm['dtm'])]
        lo, hi = np.percentile(valid, [1, 99]) if len(valid) else (0, 1)
        im = ax2.imshow(gm['dtm'], origin='lower', cmap='terrain', interpolation='none',
                        vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax2, fraction=0.046, label='ground elevation (m)')
        overlay = np.ma.masked_where(~gm['plateau'], gm['plateau'].astype(float))
        ax2.imshow(overlay, origin='lower', cmap=ListedColormap(['red']),
                   interpolation='none', alpha=0.9)
        rr = result['roofs_recovered']
        ax2.set_title(f'5 m ground model: relief {result["ground_relief_m"]} m, '
                      f'slope risk {result["slope_risk_score"]}; red = {rr["cells_5m"]} '
                      f'roof cells given back to the buildings', fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xlabel('cells (west to east)')
        ax.set_ylabel('cells (south to north)')
    fig.suptitle(f'{result.get("name", "box")}: {result["total_points_analyzed"]:,} '
                 f'points, building height {result["avg_building_height_m"]} m',
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('Saved', out_png)


def print_result(result, name=None):
    """Every returned field, one per line (the format of the manual's main block)."""
    if name:
        print(f'\n=== {name} ===')
    for key, value in result.items():
        if not key.startswith('_'):
            print(f'{key:24s}: {value}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='LiDAR metrics for a WGS84 bounding box')
    ap.add_argument('--bbox', nargs=4, type=float,
                    metavar=('MIN_LAT', 'MIN_LON', 'MAX_LAT', 'MAX_LON'))
    ap.add_argument('--name', default=None, help='label printed above the result')
    ap.add_argument('--no-roof-fix', action='store_true',
                    help='keep the Exercise 3 labels (no plateau relabelling)')
    ap.add_argument('--figure', action='store_true',
                    help='save outputs/point_clouds/lidar_analyzer_<box>.png per box')
    ap.add_argument('--json', nargs='?', const='outputs/reports/lidar_bbox_results.json',
                    default=None, help='also save the results to this JSON file')
    args = ap.parse_args()
    t0 = time.time()

    boxes = ([(args.name or 'requested box', *args.bbox)] if args.bbox else DEMO_BOXES)
    results = []
    for name, *bbox in boxes:
        res = analyze_bbox(*bbox, fix_roofs=not args.no_roof_fix, keep_rasters=args.figure)
        res['name'] = name
        print_result(res, name)
        if args.figure and res['lidar_coverage']:
            tag = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())[:40]
            plot_box(res, f'outputs/point_clouds/lidar_analyzer_{tag}.png')
        results.append({k: v for k, v in res.items() if not k.startswith('_')})

    if len(results) > 1:                                          # side-by-side summary
        print(f'\n{"box":40s}{"points":>10s}{"bldg h m":>9s}{"canopy%":>8s}'
              f'{"imperv%":>8s}{"slope":>6s}{"relief m":>9s}{"veg pts%":>9s}'
              f'{"bldg pts%":>10s}')
        for r in results:
            p = r['point_shares_pct']
            print(f'{r["name"][:39]:40s}{r["total_points_analyzed"]:10,d}'
                  f'{r["avg_building_height_m"]:9.2f}{r["canopy_cover_pct"]:8.1f}'
                  f'{r["impervious_surface_pct"]:8.1f}{r["slope_risk_score"]:6.2f}'
                  f'{r["ground_relief_m"]:9.2f}{p["vegetation"]:9.1f}'
                  f'{p["building"]:10.1f}')
    if args.json:
        out = _resolve(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        print('Saved', args.json)
    print(f'Done in {time.time() - t0:.1f} s')
