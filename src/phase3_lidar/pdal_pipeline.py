# pdal_pipeline.py
# ============================================================
# GeoSense Agent 2.0 - Phase 3 / Exercise 2 (Task E.2.1)
# PURPOSE: Clean the raw LiDAR tile and prepare it for analysis:
#            1. filters.outlier  - flag isolated noise points (birds, multipath,
#                                  sensor errors) as LAS class 7
#            2. filters.range    - drop the flagged noise so it cannot distort
#                                  any later height statistic
#            3. filters.smrf     - Simple Morphological Filter: label bare-earth
#                                  points as ground (class 2)
#            4. filters.hag_nn   - add HeightAboveGround to every point from the
#                                  nearest ground points
#            5. writers.las      - write a LAZ with the extra HAG dimension kept
#
# INPUT : data/lidar/raw/study_area.laz            (UTM 18N, metres)
# OUTPUT: data/lidar/processed/study_area_classified.laz
#
# Run from the project root, in the environment that has PDAL:
#   conda activate geosense_lidar
#   python src/phase3_lidar/pdal_pipeline.py
# ============================================================
import json
import os
import time

import numpy as np
import pdal

RAW_LAZ = 'data/lidar/raw/study_area.laz'
OUT_LAZ = 'data/lidar/processed/study_area_classified.laz'


def build_pipeline(input_laz, output_laz):
    """Return the PDAL pipeline definition (a list of stages) as a Python object."""
    return [
        input_laz,
        {
            'type': 'filters.outlier',       # statistical outlier removal
            'method': 'statistical',
            'mean_k': 8,                     # neighbours used for the mean distance
            'multiplier': 2.5,               # threshold = mean + 2.5 * std. dev.
        },
        {
            'type': 'filters.range',         # outliers were given class 7 - drop them
            'limits': 'Classification![7:7]',
        },
        {
            'type': 'filters.smrf',          # ground (class 2) vs non-ground
        },
        {
            'type': 'filters.hag_nn',        # HeightAboveGround from nearest ground
        },
        {
            'type': 'writers.las',
            'filename': output_laz,
            'minor_version': 4,              # LAS 1.4 - needed for the extra dimension
            'extra_dims': 'all',             # keep HeightAboveGround in the file
        },
    ]


def run_pipeline(input_laz, output_laz):
    """Execute the pipeline, print what it did, and return the PDAL pipeline object."""
    pipeline_def = build_pipeline(input_laz, output_laz)
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    t0 = time.time()
    count = pipeline.execute()
    print(f'Processed {count:,} points in {time.time() - t0:.1f} s.')
    print(f'Output saved to {output_laz}')

    arr = pipeline.arrays[0]
    cls = arr['Classification']
    hag = arr['HeightAboveGround']
    n_ground = int((cls == 2).sum())
    print(f'Ground points      : {n_ground:,} ({100 * n_ground / count:.1f} %)')
    n_above = count - n_ground
    print(f'Non-ground points  : {n_above:,} ({100 * n_above / count:.1f} %)')
    print(f'HeightAboveGround  : min {hag.min():.2f} m, median {np.median(hag):.2f} m, '
          f'max {hag.max():.2f} m')
    for lo, hi in [(0.5, 2.5), (2.5, 10), (10, 25), (25, 60), (60, 1000)]:
        n = int(((hag >= lo) & (hag < hi)).sum())
        print(f'  {lo:5.1f} - {hi:6.1f} m above ground : {n:9,} points '
              f'({100 * n / count:5.1f} %)')
    return pipeline


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT_LAZ), exist_ok=True)
    raw_mb = os.path.getsize(RAW_LAZ) / 1e6
    print(f'Input : {RAW_LAZ} ({raw_mb:.1f} MB)')
    run_pipeline(RAW_LAZ, OUT_LAZ)
    print(f'Output: {OUT_LAZ} ({os.path.getsize(OUT_LAZ) / 1e6:.1f} MB)')
