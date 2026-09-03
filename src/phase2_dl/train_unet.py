# train_unet.py
# Purpose: Build a U-Net (ResNet-34 encoder, 6 input bands, 5 classes) and train it on
#          the Sentinel-2 chips against the rule-based label masks.
# Run from project root: python src/phase2_dl/train_unet.py   (uses the GPU when available)
import json
import os
import time
from pathlib import Path

# Windows + conda: PyTorch ships its own OpenMP runtime (libomp.dll) and NumPy/MKL ship
# libiomp5md.dll; when both are loaded the second one aborts the process (OMP Error #15)
# unless duplicates are explicitly allowed. Must be set before torch is imported.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ── Configuration ──────────────────────────────────────────────────────────
NUM_CLASSES = 5                   # Urban, Vegetation, Water, Bare, Agriculture
IN_CHANNELS = 6                   # B2, B3, B4, B8, B11, B12
CHIP_SIZE = 224
BATCH_SIZE = 8                    # reduce to 4 if the GPU runs out of memory
NUM_EPOCHS = int(os.getenv('UNET_EPOCHS', '50'))
LEARNING_RATE = 0.001
IGNORE_INDEX = 255                # label of pixels outside the study area
YEARS = ['2018', '2023']
SEED = 42
CLASS_NAMES = ['Urban', 'Vegetation', 'Water', 'Bare', 'Agriculture']

CHIPS_DIR = Path('data/satellite/chips')
LABEL_DIR = Path('data/satellite/labels')
MODEL_DIR = Path('models/saved')
EVAL_DIR = Path('models/evaluation')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE == 'cuda' else ''))


# ── Dataset class ──────────────────────────────────────────────────────────
class SatelliteDataset(Dataset):
    """Pairs every chip of one year with the matching 224 x 224 patch of that year's label mask."""

    def __init__(self, chip_dir, label_dir, year, transform=None):
        self.chip_paths = sorted(Path(chip_dir).glob(f'chip_{year}_*.npy'))
        self.label_arr = np.load(str(Path(label_dir) / f'labels_{year}.npy'))
        self.transform = transform
        self.chip_size = CHIP_SIZE

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        chip = np.load(str(self.chip_paths[idx]))                     # (bands, H, W), float32 0-1
        parts = self.chip_paths[idx].stem.split('_')                   # chip_<year>_<row>_<col>
        r, c = int(parts[-2]), int(parts[-1])
        label = self.label_arr[r:r + self.chip_size, c:c + self.chip_size].astype(np.int64)
        chip_hwc = np.transpose(chip, (1, 2, 0)).astype(np.float32)    # (H, W, bands) for albumentations
        if self.transform:
            aug = self.transform(image=chip_hwc, mask=label)
            return aug['image'].float(), aug['mask'].long()
        return torch.tensor(chip_hwc).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)


# ── Augmentation pipeline ──────────────────────────────────────────────────
# Augmentation = random variations of the training chips so the network learns the
# land cover, not the orientation: a park is still a park when mirrored or rotated.
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ToTensorV2(),
])
val_transform = A.Compose([ToTensorV2()])


def build_model():
    # segmentation-models-pytorch builds the U-Net; ResNet-34 is the encoder backbone.
    # No pretrained weights: ImageNet encoders expect 3 RGB channels, our chips have 6.
    return smp.Unet(encoder_name='resnet34', encoder_weights=None,
                    in_channels=IN_CHANNELS, classes=NUM_CLASSES)


def main():
    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: U-Net / ResNet-34 encoder | parameters: {n_params:,}')

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)     # 255 = pixels outside the study area
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)

    # ── Data: both years, 80/20 split fixed by the seed ───────────────────
    full = ConcatDataset([SatelliteDataset(CHIPS_DIR / y, LABEL_DIR, y) for y in YEARS])
    n_total = len(full)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=gen)
    train_ds = ConcatDataset([SatelliteDataset(CHIPS_DIR / y, LABEL_DIR, y, train_transform) for y in YEARS])
    val_ds = ConcatDataset([SatelliteDataset(CHIPS_DIR / y, LABEL_DIR, y, val_transform) for y in YEARS])
    train_subset = torch.utils.data.Subset(train_ds, list(train_idx))
    val_subset = torch.utils.data.Subset(val_ds, list(val_idx))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_names = [p.stem for y in YEARS for p in sorted((CHIPS_DIR / y).glob(f'chip_{y}_*.npy'))]
    split = {'train': [all_names[i] for i in train_idx], 'val': [all_names[i] for i in val_idx]}
    with open(str(EVAL_DIR / 'unet_split.json'), 'w') as f:
        json.dump(split, f, indent=2)
    print(f'Chips: {n_total} ({" + ".join(YEARS)}) | Train: {n_train} | Val: {n_val} | '
          f'batch {BATCH_SIZE} | epochs {NUM_EPOCHS} | lr {LEARNING_RATE}')

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()                                   # TRAINING phase
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        model.eval()                                    # VALIDATION phase
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                valid = labels != IGNORE_INDEX          # accuracy over study-area pixels only
                correct += ((preds == labels) & valid).sum().item()
                total += valid.sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total * 100
        lr_now = optimiser.param_groups[0]['lr']
        for k, v in zip(history, [epoch, avg_train, avg_val, val_acc, lr_now]):
            history[k].append(v)
        scheduler.step()
        flag = ''
        if avg_val < best_val_loss:                    # save best model
            best_val_loss = avg_val
            torch.save(model.state_dict(), str(MODEL_DIR / 'unet_best.pth'))
            flag = '  <- best, saved'
        print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | '
              f'Val Acc: {val_acc:.2f}% | lr {lr_now:.5f} | {time.time() - t0:5.0f}s{flag}')

    # ── Save final model and history ──────────────────────────────────────
    torch.save(model.state_dict(), str(MODEL_DIR / 'unet_final.pth'))
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = int(np.argmin(history['val_loss']) + 1)
    history['train_time_s'] = time.time() - t0
    history['device'] = DEVICE + (f' {torch.cuda.get_device_name(0)}' if DEVICE == 'cuda' else '')
    history['parameters'] = n_params
    with open(str(EVAL_DIR / 'unet_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Per-class check of the best model on the validation chips ────────
    model.load_state_dict(torch.load(str(MODEL_DIR / 'unet_best.pth'), map_location=DEVICE))
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy().ravel()
            labs = labels.numpy().ravel()
            keep = labs != IGNORE_INDEX
            conf += np.bincount(labs[keep] * NUM_CLASSES + preds[keep], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    print('\nBest model on the validation chips (rows = label, columns = prediction):')
    print('  ' + ' ' * 12 + ''.join(f'{n[:6]:>9}' for n in CLASS_NAMES) + '   recall   IoU')
    ious = []
    for i, name in enumerate(CLASS_NAMES):
        tp = conf[i, i]; fn = conf[i].sum() - tp; fp = conf[:, i].sum() - tp
        recall = tp / max(conf[i].sum(), 1) * 100
        iou = tp / max(tp + fn + fp, 1) * 100
        ious.append(iou)
        print(f'  {name:12s}' + ''.join(f'{v:9d}' for v in conf[i]) + f'  {recall:6.1f}%  {iou:5.1f}%')
    overall = np.trace(conf) / conf.sum() * 100
    print(f'  Overall pixel accuracy: {overall:.2f}% | mean IoU: {np.mean(ious):.2f}%')
    np.save(str(EVAL_DIR / 'unet_val_confusion.npy'), conf)

    # ── Training curves ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(history['epoch'], history['train_loss'], color='#2C6FB3', lw=2, label='train loss')
        ax1.plot(history['epoch'], history['val_loss'], color='#D97A1B', lw=2, label='validation loss')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('cross-entropy loss'); ax1.legend(loc='upper right', frameon=False)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_title(f'U-Net training on {n_total} Kolkata chips ({history["device"]})', fontsize=10)
        fig.tight_layout(); fig.savefig(str(EVAL_DIR / 'unet_loss_curves.png'), dpi=160); plt.close(fig)
        fig, ax2 = plt.subplots(figsize=(8, 3.2))
        ax2.plot(history['epoch'], history['val_acc'], color='#2E7D32', lw=2)
        ax2.set_xlabel('epoch'); ax2.set_ylabel('validation pixel accuracy (%)')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_title('Validation accuracy per epoch', fontsize=10)
        fig.tight_layout(); fig.savefig(str(EVAL_DIR / 'unet_accuracy_curve.png'), dpi=160); plt.close(fig)
        print(f'Curves saved: {EVAL_DIR / "unet_loss_curves.png"}, {EVAL_DIR / "unet_accuracy_curve.png"}')
    except Exception as e:
        print(f'(plotting skipped: {e})')

    print(f'\nTraining complete! Best validation loss: {best_val_loss:.4f} at epoch {history["best_epoch"]} | '
          f'final val acc: {history["val_acc"][-1]:.2f}% | time {history["train_time_s"] / 60:.1f} min')
    print('Model saved to: models/saved/unet_best.pth (best) and models/saved/unet_final.pth (final)')


if __name__ == '__main__':
    main()
