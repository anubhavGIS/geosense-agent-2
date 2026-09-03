# Model comparison - U-Net vs fine-tuned foundation model

Held-out set: 32 chips, 1,586,574 labelled pixels (the validation split of train_unet.py).

| Metric | U-Net (from scratch) | Prithvi-100M (fine-tuned) |
|---|---|---|
| Pixel accuracy | 87.17% | 76.11% |
| Mean IoU | 66.63% | 44.66% |
| Macro F1 | 78.37% | 54.94% |
| IoU Urban | 86.73% | 74.02% |
| IoU Vegetation | 81.64% | 65.57% |
| IoU Water | 75.58% | 66.99% |
| IoU Bare Land | 39.10% | 3.76% |
| IoU Agriculture | 50.10% | 12.96% |
| Trainable parameters | 24,446,357 | 16,906,469 of 88,966,373 |
| Epochs / best epoch | 50 / 50 | 40 / 39 |
| Training time | 1.3 min | 2.3 min |
| Inference per chip (cuda) | 2.1 ms | 6.4 ms |

**Higher mean IoU: U-Net (ResNet-34, trained from scratch)** (66.63% vs 44.66%).