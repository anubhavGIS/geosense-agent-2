# chip_images.py
# Purpose: Cut the cleaned satellite images into 224 x 224 training chips
# Run from project root: python src/phase2_dl/chip_images.py
import json
from pathlib import Path

import numpy as np

CLEAN_DIR = Path('data/satellite/clean')
CHIPS_DIR = Path('data/satellite/chips')
CHIPS_DIR.mkdir(parents=True, exist_ok=True)

CHIP_SIZE = 224          # pixels - standard deep-learning input size
STRIDE = 174             # overlap = 224 - 174 = 50 pixels between neighbouring chips
YEARS = ['2018', '2023']


def create_chips(year: str) -> int:
    clean_path = CLEAN_DIR / f'clean_{year}.npy'
    if not clean_path.exists():
        print(f'File not found: {clean_path}')
        return 0
    image = np.load(str(clean_path))                 # shape: (bands, rows, cols)
    bands, rows, cols = image.shape
    print(f'Year {year}: image shape = {image.shape}')

    chip_dir = CHIPS_DIR / year
    chip_dir.mkdir(exist_ok=True)
    chip_count, skipped = 0, 0
    row_starts = range(0, rows - CHIP_SIZE + 1, STRIDE)
    col_starts = range(0, cols - CHIP_SIZE + 1, STRIDE)
    print(f'  Grid: {len(row_starts)} rows x {len(col_starts)} cols of windows '
          f'({CHIP_SIZE} px, stride {STRIDE})')

    for row_start in row_starts:
        for col_start in col_starts:
            chip = image[:, row_start:row_start + CHIP_SIZE,
                            col_start:col_start + CHIP_SIZE].copy()
            valid_pct = np.mean(~np.isnan(chip[0]))
            if valid_pct < 0.8:                      # skip if more than 20% NaN
                skipped += 1
                continue
            for b in range(bands):                   # fill remaining NaN with the band mean
                band_mean = np.nanmean(chip[b])
                chip[b] = np.where(np.isnan(chip[b]), band_mean, chip[b])
            chip_path = chip_dir / f'chip_{year}_{row_start:05d}_{col_start:05d}.npy'
            np.save(str(chip_path), chip.astype(np.float32))
            chip_count += 1

    print(f'  Created {chip_count} chips for year {year} '
          f'(skipped {skipped} with > 20% missing pixels) -> {chip_dir}')
    return chip_count


if __name__ == '__main__':
    summary = {}
    for year in YEARS:
        summary[year] = create_chips(year)
    with open(str(CHIPS_DIR / 'chip_summary.json'), 'w') as f:
        json.dump({'chip_size': CHIP_SIZE, 'stride': STRIDE, 'counts': summary}, f, indent=2)
    print(f'\nChip summary: {summary} (total {sum(summary.values())})')
    print('Chipping complete!')
