# Phase 3 Completion Checklist (Part H.1) — with evidence

Project: GeoSense Agent 2.0 · LiDAR tile: Dupont Circle, Washington, D.C. (77.048–77.038 W, 38.905–38.913 N, EPSG:32618) · Street imagery: Kolkata (Park Street area), Chennai centre, Dupont Circle · Environments: conda `geosense` (Python 3.11, PyTorch 2.11.0+cu128, RTX 5060 Laptop 8 GB) and `geosense_lidar` (PDAL 2.10.2, python-pdal 3.5.5).
Record references (Ex. n) point to the Phase III lab records, Exercises 1–5.

## Software Installation

| Done | Item | Evidence |
|---|---|---|
| ☑ | OpenTopography account created | account and API key (myOpenTopo); the Find Data map shows no coverage over Kolkata and 3DEP coverage over Washington–Baltimore; portal downloads of 3DEP are restricted to .edu / OT+ users, so the same 3DEP data is streamed from the public Entwine archive (Ex. 1) |
| ☑ | PDAL installed via conda — `pdal --version` shows a version number | `pdal 2.10.2` in `geosense_lidar` (the solver could not fit PDAL into the deep-learning environment; PDAL has its own) (Ex. 1) |
| ☑ | Open3D installed — `import open3d` works | open3d 0.20.0 (Ex. 1) |
| ☑ | GPU access configured — local CUDA or Colab T4 | `nvidia-smi`: driver 617.14, CUDA 13.4, RTX 5060 Laptop GPU 8151 MiB; `torch.cuda.is_available()` → True (Ex. 1) |
| ☑ | SAM installed with the ViT-B checkpoint in `models/saved/sam/` | segment_anything 1.0 (commit dca509f); `sam_vit_b_01ec64.pth` 375.0 MB (Ex. 1); 212 masks in 21.2 s on the GPU (Ex. 3) |
| ☑ | HuggingFace Transformers installed — SegFormer downloads and caches | transformers 5.16.1, accelerate 1.15.0; `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` (3.7 M parameters, ~15 MB) cached on first use, loaded in 1–3 s afterwards (Ex. 4) |
| ☑ | Mapillary developer account and access token saved in `.env` | app `GeoSense-Agent-Phase3` (READ scope); `MAPILLARY_TOKEN` in `.env`, never in a script (Ex. 1) |

## Environment Setup

| Done | Item | Evidence |
|---|---|---|
| ☑ | Phase 3 folder structure created | `data/lidar/{raw,processed}`, `data/streetview/{images,results}`, `src/phase3_lidar`, `models/saved/sam`, `outputs/point_clouds` (Ex. 1) |
| ☑ | `.env` updated with MAPILLARY_TOKEN and OPENTOPOGRAPHY_API_KEY | plus `SAM_CHECKPOINT_PATH` and `STUDY_AREA_BBOX_LIDAR=-77.048,38.905,-77.038,38.913` — 16 keys in all (Ex. 1) |
| ☑ | `.gitignore` excludes LiDAR raw files, street images and model checkpoints | `data/lidar/raw/`, `data/lidar/processed/*.laz`, `data/streetview/images/`, `models/saved/sam/*.pth`, `outputs/point_clouds/*.ply`; `git check-ignore -v` confirms (Ex. 1) |

## Data Collection

| Done | Item | Evidence |
|---|---|---|
| ☑ | LiDAR LAZ point cloud for the study area in `data/lidar/raw/` | `study_area.laz` — USGS_LPC_MD_VA_Sandy_NCR_2014_LAS_2015 via `readers.ept`, 6,418,741 points in 17.5 s, 35.7 MB, LAS 1.4, EPSG:32618, 7.98 pts/m² over 0.80 km² (Ex. 2) |
| ☑ | At least 10–20 Mapillary street images in `data/streetview/images/` | 29 images of the Park Street area (19 + 11, one overlap), all flat, 2018–2019; later queries added the Chennai and Dupont Circle photographs (Ex. 2, Ex. 5) |

## LiDAR Processing

| Done | Item | Evidence |
|---|---|---|
| ☑ | `pdal_pipeline.py` runs without error — classified LAZ produced | `study_area_classified.laz`: 6,277,069 points in 68.3 s (2.2 % removed as noise), ground 43.8 %, `HeightAboveGround` median 2.68 m, max 60.40 m (Ex. 2) |
| ☑ | `lidar_features.py` runs — building height and canopy per grid cell | 331 cells of 50 m; ground 45.3 %, vegetation 16.9 %, building 37.7 % of the points; building mean HAG 17.71 m; `outputs/lidar_grid_metrics.json`, `study_area_grid_metrics.png` (Ex. 3) |
| ☑ | `visualize_cloud.py` runs — colour-coded PLY viewable in Open3D | `study_area_colored.ply` 169.5 MB in 9.8 s; top-down and oblique screenshots (`outputs/point_clouds/study_area_colored_*.png`) (Ex. 3) |
| ☑ | `sam_rooftop_extraction.py` detects rooftop segments and estimates volumes | 212 masks; 180 candidates in the 40–4,000 px window, 212 in the 10–5,000 m² window; rooftop 1: 4,028.4 m² × 33.30 m (LiDAR median of 14,085 roof points) = 134,145.7 m³; `outputs/rooftops/rooftop_candidates*.json` (Ex. 3, Ex. 5) |

## Street Semantic Analysis

| Done | Item | Evidence |
|---|---|---|
| ☑ | `streetview_semantic.py` classifies at least one street image | three Kolkata photographs: road 34.3 / 38.0 / 8.8 %, vegetation 23.6 / 19.9 / 19.5 %, scores 1.02 / 0.98 / 1.11; `outputs/streetview/segformer_examples.png` (Ex. 4) |
| ☑ | `analyze_location()` returns a scene_composition dictionary with sensible percentages | Shakespeare Sarani 46 % vegetation and 2 % car against Park Street 11 % car; Dupont Circle 29 % road, 16.5 % vegetation, 8.5 % sky once the 12 underpass photographs were skipped (Ex. 4, Ex. 5) |
| ☑ | walkability_score calculated and returned | 0–10 scale, (sidewalk × 0.4 + vegetation × 0.3 − car × 0.2) / 5; alternative weights compared; Chennai 1.69, Dupont Circle 1.13, Park Street 1.12 in the final profiles (Ex. 5) |

## Core Modules

| Done | Item | Evidence |
|---|---|---|
| ☑ | `lidar_analyzer.py` works — `analyze_bbox()` returns a complete result dictionary | five demo boxes: the Chennai example (no coverage, flagged), the whole tile (18.18 m, 23.9 % canopy, 43.2 % impervious, slope risk 6.38) and three sub-boxes with different results; `outputs/reports/lidar_bbox_results.json` (Ex. 5) |
| ☑ | `streetview_semantic.py` works — `analyze_location()` returns a complete result dictionary | latitude, longitude, images_analyzed, images_enclosed, scene_composition, walkability_score, radius_deg, archive_snapshot, image_ids (Ex. 5) |
| ☑ | `unified_site_profile.py` works — `build_profile()` merges both into one JSON profile | `enhanced_site_profile_13.08_80.27.json`, `_38.91_-77.04.json`, `_22.55_88.35.json` — the Kolkata profile carries the Phase 1 score (10.0, GOOD SITE), the Phase 2 land cover (Urban, 83.6 %) and the street scene; two consecutive runs identical (Ex. 5) |
| ☑ | All 4 unit tests pass — `pytest tests/ -v` | 12 Phase 3 tests (the four of Section G.1 and eight own) + 4 Phase 1 tests: 16 passed in 27.2 s (Ex. 5) |

## GitHub Portfolio

| Done | Item | Evidence |
|---|---|---|
| ☑ | All Phase 3 code committed to the existing geosense-agent-2 repository | `src/phase3_lidar/` (11 scripts), `tests/test_phase3_modules.py`, outputs, snapshots and this checklist |
| ☑ | README.md updated with a Phase 3 section | pipeline table, `lidar_analyzer.py`, `streetview_semantic.py` and `unified_site_profile.py` sections with results |
| ☑ | Screenshot of the colour-coded point cloud in the README | `outputs/point_clouds/study_area_colored_oblique.png`; the `lidar_analyzer` figure of the whole tile beside it |
| ☑ | At least 5 meaningful Phase 3 commits with descriptive messages | eight commits with the Add: / Improve: / Complete: prefixes (see `git log`) |

## Common errors of Section G.2 — the fixes in my own words

| Error | What it means | Fix |
|---|---|---|
| `MemoryError` when laspy reads a LAZ | laspy decompresses the whole file into RAM — about 36 bytes per point plus the arrays the script makes from them; a tile of tens of millions of points exceeds 16 GB | ask for a smaller box when fetching (the EPT reader streams only the points inside `bounds`), or cut the cloud into spatial chunks with PDAL `filters.splitter` / `filters.crop` and process them one at a time; the 6.3 M-point tile used here needs about 1 GB |
| `RuntimeError: No CUDA GPUs are available` | the code asked for a CUDA device that PyTorch cannot see — either a CPU-only torch build, or the GPU hidden (`CUDA_VISIBLE_DEVICES=-1` reproduced it in Ex. 3) | let the script choose the device (`'cuda' if torch.cuda.is_available() else 'cpu'`); on a CPU-only machine install the CPU wheels or move the heavy scripts to Colab; SAM ran in 79.9 s on the CPU against 21.2 s on the GPU — slower, not broken |
| empty result from `find_images_in_bbox()` | the box holds no photographs, or its corners are in the wrong order — Mapillary wants `min_lon,min_lat,max_lon,max_lat` and a swapped box is a box in the Arabian Sea | check the order (a swapped box returned 0 images in Ex. 2), enlarge the radius (0.00005° returned nothing, 0.0030° ten images in Ex. 4) or move to a well-covered street; `analyze_location()` now also halves the box when the API answers 500 for a very dense area |
| `'pdal' command not found` / `No module named 'pdal'` | the environment that has PDAL is not active — PDAL is a conda-forge package with C++ libraries and cannot be pip-installed | `conda activate geosense_lidar` (PDAL has its own environment here); reproduced and fixed in Ex. 1 |

Errors met in practice and their fixes are in the records: `OMP: Error #15` (two OpenMP runtimes — `KMP_DUPLICATE_LIB_OK` guard), `torch.OutOfMemoryError` with SegFormer-B5 on ten images at once (back to B0, one image per pass), `OSError: [Errno 22]` from a literal `*.jpg` on the Windows prompt (the script expands wildcards itself), HTTP 400 `OAuthException` for a wrong Mapillary token, HTTP 500 for a dense Mapillary box (smaller box), and a day of `getaddrinfo failed` that was the laptop's DNS, not the code.

## Section I in practice

- **Naming** — scripts `lowercase_underscore.py`; point clouds `study_area.laz`, `study_area_classified.laz`, `study_area_colored.ply`; profiles `enhanced_site_profile_<lat>_<lon>.json`; snapshots `query_<lat>_<lon>_r<radius>.json`; commit prefixes Add:, Improve:, Complete:.
- **Daily routine** — Anaconda Prompt → `conda activate geosense` (or `geosense_lidar` for the two PDAL scripts) → `cd Desktop\GeoSense_Agent` → one script at a time, tested before the next → commit and push at the end of the session.
- **Three-layer backup** — GitHub for the code and small outputs (this repository); Google Drive for `data/lidar/`, `data/streetview/images/` and `models/saved/sam/` (expensive to re-download); an external drive for a monthly copy of the whole project folder including the lab records.
- **Troubleshooting process (Section I.5) on a real error** — `OMP: Error #15: Initializing libiomp5md.dll, but found libomp.dll already initialized` ended the first SAM runs after the JSON was written: (1) the last line named two OpenMP runtimes; (2) a search showed it is the known clash between the MKL inside numpy and the OpenMP that torch brings on Windows; (3) the environment was the right one and the script was running from the project root, so the basics were fine; (4) a print before the figure showed the error came at the first matplotlib call after torch; (5) the documented workaround — `os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')` before the imports, already used in Phase 2 — fixed it, and every Phase 3 script that imports torch now carries it.
