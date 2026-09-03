# Phase 2 Completion Checklist (Part F) — with evidence

Project: GeoSense Agent 2.0 · Study area: Kolkata (88.30–88.42 E, 22.45–22.67 N) · Environment: conda `geosense`, Python 3.11, Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), PyTorch 2.11.0+cu128.
Record references (Fig./Table) point to the Phase II lab records, Exercises 1–5.

## Software Setup

| Done | Item | Evidence |
|---|---|---|
| ☑ | PyTorch installed — `import torch` works | torch 2.11.0+cu128; `torch.cuda.is_available()` → True, `Capability (12, 0)` (Ex. 1) |
| ☑ | TorchGeo installed — `import torchgeo` works | torchgeo 0.8.1 after pinning stringzilla==5.1.1 (Ex. 1) |
| ☑ | HuggingFace Transformers installed — `import transformers` works | transformers 5.16.1, huggingface_hub 1.29.0 (Ex. 1 verification screenshot) |
| ☑ | Google Earth Engine authenticated — `test_gee.py` prints an image ID | 54 scenes for 2023; ID `20230305T043701_20230305T044808_T45QXF`, 26 bands (Ex. 1) |
| ☑ | segmentation-models-pytorch installed | smp 0.5.0 — used by `train_unet.py`, the `finetune_prithvi.py` fallback and `image_classifier.py` |
| ☑ | Albumentations installed | albumentations 2.0.8 — augmentation pipeline in `train_unet.py` |
| ☑ | Google Colab account with GPU enabled | Colab T4 runtime verified with `nvidia-smi` (Ex. 1); training was finally run on the laptop GPU |

## Data Pipeline

| Done | Item | Evidence |
|---|---|---|
| ☑ | Baseline-year Sentinel-2 image downloaded | `data/satellite/raw/sentinel2_Kolkata_2018.tif` (25.0 MB; 28 scenes < 10 % cloud) — 2018 is the earliest full year of the harmonised collection (Ex. 2) |
| ☑ | 2023 Sentinel-2 image downloaded | `data/satellite/raw/sentinel2_Kolkata_2023.tif` (25.1 MB; 55 scenes) (Ex. 2) |
| ☑ | Both images visible and correct in QGIS | True-colour composites over OSM with matching extent (Ex. 2, QGIS screenshots) |
| ☑ | Preprocessing complete — `clean_2023.npy` exists | `data/satellite/clean/clean_2018.npy`, `clean_2023.npy` (6 × 2448 × 1257, float32, reflectance 0–1); cloud-free 97.8 % (Ex. 2) |
| ☑ | NDVI computed, values in −1…1 | `ndvi_2023.npy`: mean 0.284, min −0.434, max 0.870; `ndvi_2018.npy`: mean 0.281 (`data/satellite/clean/meta_*.json`) |
| ☑ | NDWI computed | `ndwi_2023.npy` mean −0.294; `ndwi_2018.npy` mean −0.299 |
| ☑ | Image chips created | `data/satellite/chips/2018/` and `/2023/`: 78 chips per year (13 × 6 grid, stride 174), 156 total (Ex. 2) |
| ☑ | Training labels created — `labels_2023.npy` exists | `data/satellite/labels/labels_2018.npy`, `labels_2023.npy` (Kolkata-calibrated rules: 50 % Urban, 27 % Vegetation, 4 % Water, 3 % Bare, 15 % Agriculture) (Ex. 3) |

## Model Training

| Done | Item | Evidence |
|---|---|---|
| ☑ | U-Net trained — `models/saved/unet_best.pth` exists | 97.96 MB, ResNet-34 encoder, 24,446,357 parameters (Ex. 3) |
| ☑ | Training accuracy above 75 % — `unet_history.json` | validation accuracy 87.17 % at epoch 50 (passed 75 % at epoch 5) |
| ☑ | Training loss decreased consistently | train loss 1.182 → 0.339, val loss 1.294 → 0.323 over 50 epochs (`outputs/plots/training_curves_unet.png`) |
| ☑ | Prithvi / foundation model downloaded | `ibm-nasa-geospatial/Prithvi-EO-1.0-100M` — `prithvi_mae.py` + `Prithvi_EO_V1_100M.pt` (454 MB), 149 of 254 checkpoint tensors loaded (Ex. 4) |
| ☑ | Fine-tuned model saved — `models/saved/prithvi_finetuned.pth` exists | 352.6 MB; 16,906,469 of 88,966,373 parameters trained (19.0 %); best val loss 0.6140 at epoch 39, val acc 76.17 % (Ex. 4) |
| ☑ | Confusion matrix generated — `outputs/plots/confusion_matrix_unet.png` | plus `confusion_matrix_finetuned.png` (Ex. 4) |
| ☑ | Training curves plotted | `outputs/plots/training_curves_unet.png`, `training_curves_finetuned.png`, `training_curves_comparison.png` |
| ☑ | Model comparison documented — Prithvi vs U-Net | `models/evaluation/model_comparison.json`, `outputs/model_comparison.md`: U-Net 87.17 % / mIoU 66.63 %; Prithvi 76.11 % / 44.66 % (Ex. 4, §4.3) |

## Core Module

| Done | Item | Evidence |
|---|---|---|
| ☑ | `image_classifier.py` written at `src/phase2_dl/image_classifier.py` | `classify_imagery()`, `_load_model()`, `_get_image_patch()`, `_compute_ndvi_ndwi()`, `find_change_hotspots()` |
| ☑ | `classify_imagery()` runs without error | seven Kolkata test locations — Maidan, B.B.D. Bagh, Rabindra Sarobar, the Hooghly, the East Kolkata Wetlands, Garden Reach, Joka (Ex. 5 transcript) |
| ☑ | Output contains land_cover, ndvi, ndwi, confidence_pct, class_distribution, change_flag | all fields printed for every location, plus class_id, ndvi_label, change_description, ndvi_baseline, ndvi_change |
| ☑ | NDVI values are realistic (0.0–0.4 for urban / mixed areas) | B.B.D. Bagh 0.223, Garden Reach 0.229, river bank 0.109; Maidan 0.458, wetlands 0.432 |
| ☑ | Change detection flag returns correctly for unchanged and changed areas | stable locations False (ΔNDVI ≤ 0.033); wetlands gain hotspot True (ΔNDVI +0.164, 45 % of pixels changed) |
| ☑ | Inference time under 3 s | 0.59 s for the first call (CUDA warm-up), 0.04–0.07 s afterwards on the RTX 5060 |

## GitHub Portfolio

| Done | Item | Evidence |
|---|---|---|
| ☑ | All Phase 2 code committed to GitHub | `src/phase2_dl/` (8 scripts), `test_gee.py`, `models/evaluation/*.json`, `outputs/plots/*.png`, `docs/`, `requirements.txt` in `anubhavGIS/geosense-agent-2` |
| ☑ | README updated with Phase 2 results: accuracy, NDVI mean, change-detection examples | `README.md` — Phase 2 section |
| ☑ | Sample confusion matrix image added to README | `outputs/plots/confusion_matrix_unet.png` embedded in the README |
| ☑ | Phase 2 section documenting model performance numbers | README tables: U-Net vs Prithvi metrics, NDVI means 2018/2023, test-location table, change example |
