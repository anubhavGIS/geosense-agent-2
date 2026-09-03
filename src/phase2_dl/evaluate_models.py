# evaluate_models.py
# Purpose: Compare the from-scratch U-Net (train_unet.py) and the fine-tuned foundation
#          model (finetune_prithvi.py) on the same held-out chips: per-class IoU,
#          confusion matrices, classification reports, training-curve plots and a
#          written comparison.
# Run from project root: python src/phase2_dl/evaluate_models.py
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from train_unet import (build_model, CHIPS_DIR, LABEL_DIR, CHIP_SIZE, NUM_CLASSES,
                        IGNORE_INDEX)
from finetune_prithvi import (PrithviSegmenter, build_prithvi_encoder, get_prithvi_files,
                              IN_CHANNELS)
from generate_labels import PALETTE

CLASSES = ['Urban', 'Vegetation', 'Water', 'Bare Land', 'Agriculture']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = Path('models/saved'); EVAL_DIR = Path('models/evaluation')
PLOT_DIR = Path('outputs/plots'); PLOT_DIR.mkdir(parents=True, exist_ok=True)
UNET_COLOUR, FT_COLOUR = '#2C6FB3', '#D97A1B'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> list:
    """
    Compute Intersection over Union for each class.
    IoU = how well the predicted region overlaps with the actual region.
    IoU of 1.0 = perfect prediction. IoU of 0.5 = 50% overlap.
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        iou = intersection / (union + 1e-10)
        ious.append(float(iou))
    return ious


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(row_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right'); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)', ha='center',
                    va='center', fontsize=7.5,
                    color='white' if row_pct[i, j] > 55 else '#222222')
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('Actual (rule-based label)'); ax.set_xlabel('Predicted')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('% of actual class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')
    return cm


def plot_training_curves(history_path, model_name, save_path,
                         colour_loss=('#2C6FB3', '#D97A1B')):
    with open(history_path) as f:
        h = json.load(f)
    epochs = h.get('epoch', list(range(1, len(h['train_loss']) + 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, h['train_loss'], label='Train Loss', color=colour_loss[0], lw=2)
    ax1.plot(epochs, h['val_loss'], label='Val Loss', color=colour_loss[1], lw=2)
    ax1.set_title(f'{model_name} - Training Curves')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(frameon=False)
    ax2.plot(epochs, h['val_acc'], label='Val Accuracy', color='#2E7D32', lw=2)
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy %'); ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_curve_comparison(hist_unet, hist_ft, ft_name, save_path):
    """Both models on the same axes (validation loss and validation accuracy per epoch)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(hist_unet['epoch'], hist_unet['val_loss'], color=UNET_COLOUR, lw=2,
             label='U-Net (from scratch)')
    ax1.plot(hist_ft['epoch'], hist_ft['val_loss'], color=FT_COLOUR, lw=2, label=ft_name)
    ax1.set_title('Validation loss per epoch'); ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss'); ax1.legend(frameon=False)
    ax2.plot(hist_unet['epoch'], hist_unet['val_acc'], color=UNET_COLOUR, lw=2,
             label='U-Net (from scratch)')
    ax2.plot(hist_ft['epoch'], hist_ft['val_acc'], color=FT_COLOUR, lw=2, label=ft_name)
    ax2.set_title('Validation pixel accuracy per epoch'); ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy %'); ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_iou_comparison(iou_unet, iou_ft, ft_name, save_path):
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    b1 = ax.bar(x - w / 2, [v * 100 for v in iou_unet], w, color=UNET_COLOUR,
                label='U-Net (from scratch)')
    b2 = ax.bar(x + w / 2, [v * 100 for v in iou_ft], w, color=FT_COLOUR, label=ft_name)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f'{b.get_height():.1f}', ha='center', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel('IoU (%)'); ax.set_ylim(0, 105)
    ax.set_title('Per-class IoU on the held-out chips'); ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_prediction_examples(chips, labels, preds_unet, preds_ft, names, ft_name,
                             save_path):
    cols = [tuple(v / 255 for v in PALETTE[i]) for i in range(5)] + [(1, 1, 1)]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 255.5], cmap.N)
    legend = [Patch(facecolor=cols[i], edgecolor='#888', label=n)
              for i, n in enumerate(CLASSES)]
    n = len(names)
    fig, axes = plt.subplots(4, n, figsize=(1.8 * n, 7.4))
    for k in range(n):
        rgb = np.dstack([np.clip((chips[k][i] - 0.015) / 0.205, 0, 1) ** 0.75
                         for i in (2, 1, 0)])
        axes[0, k].imshow(rgb, interpolation='nearest')
        axes[0, k].set_title(names[k], fontsize=7, fontweight='normal')
        for row, arr in ((1, labels[k]), (2, preds_unet[k]), (3, preds_ft[k])):
            a = arr.astype(float); a[arr == IGNORE_INDEX] = 255
            axes[row, k].imshow(a, cmap=cmap, norm=norm, interpolation='nearest')
        for ax in axes[:, k]:
            ax.set_xticks([]); ax.set_yticks([])
    for row, t in enumerate(['chip (true colour)', 'rule-based label', 'U-Net', ft_name]):
        axes[row, 0].set_ylabel(t, fontsize=8)
    fig.legend(handles=legend, loc='lower center', ncol=5, frameon=False, fontsize=7.5)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ── Models and data ──────────────────────────────────────────────────────────
def load_unet():
    model = build_model()
    model.load_state_dict(torch.load(str(MODEL_DIR / 'unet_best.pth'), map_location='cpu'))
    return model.to(DEVICE).eval()


def load_finetuned():
    meta = json.load(open(MODEL_DIR / 'prithvi_finetuned_meta.json'))
    if meta['backbone'] == 'prithvi':
        code_path = get_prithvi_files(weights=False)[0]
        model = PrithviSegmenter(build_prithvi_encoder(code_path))
        name = 'Prithvi-100M (fine-tuned)'
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=IN_CHANNELS, classes=NUM_CLASSES)
        name = 'ResNet-50 U-Net (fine-tuned)'
    model.load_state_dict(torch.load(str(MODEL_DIR / 'prithvi_finetuned.pth'),
                                     map_location='cpu'))
    return model.to(DEVICE).eval(), name, meta['backbone']


def load_test_set():
    """The 32 held-out chips of unet_split.json (never used to train either model)."""
    split = json.load(open(EVAL_DIR / 'unet_split.json'))
    label_arrs = {y: np.load(str(LABEL_DIR / f'labels_{y}.npy')) for y in ('2018', '2023')}
    chips, labels, names = [], [], []
    for name in split['val']:
        _, year, r, c = name.split('_'); r, c = int(r), int(c)
        chips.append(np.load(str(CHIPS_DIR / year / f'{name}.npy')).astype(np.float32))
        labels.append(label_arrs[year][r:r + CHIP_SIZE, c:c + CHIP_SIZE])
        names.append(name)
    return np.stack(chips), np.stack(labels), names


def predict(model, chips, batch_size=4):
    """Class map for every chip; returns predictions and the mean time per chip (ms)."""
    preds = []
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(chips), batch_size):
            x = torch.from_numpy(chips[i:i + batch_size]).to(DEVICE)
            preds.append(model(x).argmax(1).cpu().numpy().astype(np.uint8))
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    ms_per_chip = (time.perf_counter() - t0) / len(chips) * 1000
    return np.concatenate(preds), ms_per_chip


def evaluate(model, model_name, chips, labels):
    predict(model, chips[:4])                    # warm-up (CUDA kernels, cuDNN autotuning)
    preds, ms = predict(model, chips)
    valid = labels != IGNORE_INDEX
    y_true, y_pred = labels[valid].astype(np.int64), preds[valid].astype(np.int64)
    ious = compute_iou(y_pred, y_true, NUM_CLASSES)
    acc = float((y_true == y_pred).mean() * 100)
    per_chip_acc = [float((preds[k][valid[k]] == labels[k][valid[k]]).mean() * 100)
                    for k in range(len(chips))]
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASSES, digits=3, zero_division=0,
                                   output_dict=True)
    print(f'\n{model_name}')
    print(f'  Pixel accuracy: {acc:.2f}% | mean IoU: {np.mean(ious) * 100:.2f}% | '
          f'{ms:.1f} ms per chip on {DEVICE}')
    for cls, iou in zip(CLASSES, ious):
        r = report[cls]
        print(f'  IoU {cls:12s}: {iou * 100:6.2f}%   '
              f'precision {r["precision"] * 100:6.2f}%   '
              f'recall {r["recall"] * 100:6.2f}%   F1 {r["f1-score"] * 100:6.2f}%')
    print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                target_names=CLASSES, digits=3, zero_division=0))
    result = {'name': model_name, 'pixel_accuracy': acc,
              'mean_iou': float(np.mean(ious)) * 100,
              'iou': {c: v * 100 for c, v in zip(CLASSES, ious)},
              'precision': {c: report[c]['precision'] * 100 for c in CLASSES},
              'recall': {c: report[c]['recall'] * 100 for c in CLASSES},
              'f1': {c: report[c]['f1-score'] * 100 for c in CLASSES},
              'macro_f1': report['macro avg']['f1-score'] * 100, 'ms_per_chip': ms,
              'per_chip_accuracy': per_chip_acc,
              'chips_above_75': int(sum(a >= 75 for a in per_chip_acc))}
    return result, preds, y_true, y_pred


def main():
    print(f'Device: {DEVICE}')
    chips, labels, names = load_test_set()
    n18 = sum(n.startswith('chip_2018') for n in names)
    n23 = sum(n.startswith('chip_2023') for n in names)
    print(f'Held-out chips: {len(names)} ({n18} from 2018, {n23} from 2023) | '
          f'valid pixels: {(labels != IGNORE_INDEX).sum():,}')

    unet = load_unet()
    ft_model, ft_name, backbone = load_finetuned()
    res_u, pred_u, yt, yp_u = evaluate(unet, 'U-Net (ResNet-34, trained from scratch)',
                                       chips, labels)
    res_f, pred_f, _, yp_f = evaluate(ft_model, ft_name, chips, labels)

    cm_u = plot_confusion_matrix(yt, yp_u, 'U-Net (from scratch)',
                                 PLOT_DIR / 'confusion_matrix_unet.png')
    cm_f = plot_confusion_matrix(yt, yp_f, ft_name,
                                 PLOT_DIR / 'confusion_matrix_finetuned.png')
    plot_training_curves(EVAL_DIR / 'unet_history.json', 'U-Net (from scratch)',
                         PLOT_DIR / 'training_curves_unet.png')
    plot_training_curves(EVAL_DIR / 'prithvi_history.json', ft_name,
                         PLOT_DIR / 'training_curves_finetuned.png')
    hist_u = json.load(open(EVAL_DIR / 'unet_history.json'))
    hist_f = json.load(open(EVAL_DIR / 'prithvi_history.json'))
    plot_curve_comparison(hist_u, hist_f, ft_name,
                          PLOT_DIR / 'training_curves_comparison.png')
    plot_iou_comparison([res_u['iou'][c] / 100 for c in CLASSES],
                        [res_f['iou'][c] / 100 for c in CLASSES], ft_name,
                        PLOT_DIR / 'iou_comparison.png')
    wanted = ('chip_2023_01044_00696', 'chip_2023_00174_00348',
              'chip_2018_02088_00174', 'chip_2023_01740_00522')
    picks = [k for k, n in enumerate(names) if n in wanted][:4] or list(range(4))
    plot_prediction_examples([chips[k] for k in picks], [labels[k] for k in picks],
                             [pred_u[k] for k in picks], [pred_f[k] for k in picks],
                             [names[k] for k in picks], ft_name,
                             PLOT_DIR / 'prediction_examples.png')

    # ── Written comparison ─────────────────────────────────────────────────
    better = res_u if res_u['mean_iou'] >= res_f['mean_iou'] else res_f
    other = res_f if better is res_u else res_u
    n_epochs_u, n_epochs_f = len(hist_u['epoch']), len(hist_f['epoch'])
    summary = {
        'test_chips': len(names), 'valid_pixels': int((labels != IGNORE_INDEX).sum()),
        'device': DEVICE,
        'unet': {**res_u, 'parameters': hist_u.get('parameters'),
                 'trainable_parameters': hist_u.get('parameters'), 'epochs': n_epochs_u,
                 'best_epoch': hist_u.get('best_epoch'),
                 'train_time_s': hist_u.get('train_time_s'),
                 'confusion_matrix': cm_u.tolist()},
        'finetuned': {**res_f, 'backbone': backbone,
                      'parameters': hist_f.get('parameters'),
                      'trainable_parameters': hist_f.get('trainable_parameters'),
                      'frozen_tensors': hist_f.get('frozen_tensors'), 'epochs': n_epochs_f,
                      'best_epoch': hist_f.get('best_epoch'),
                      'train_time_s': hist_f.get('train_time_s'),
                      'confusion_matrix': cm_f.tolist()},
        'higher_mean_iou': better['name'],
    }
    json.dump(summary, open(EVAL_DIR / 'model_comparison.json', 'w'), indent=2)
    lines = ['# Model comparison - U-Net vs fine-tuned foundation model', '',
             f'Held-out set: {len(names)} chips, {summary["valid_pixels"]:,} labelled '
             f'pixels (the validation split of train_unet.py).', '',
             '| Metric | U-Net (from scratch) | ' + ft_name + ' |', '|---|---|---|',
             f'| Pixel accuracy | {res_u["pixel_accuracy"]:.2f}% | '
             f'{res_f["pixel_accuracy"]:.2f}% |',
             f'| Mean IoU | {res_u["mean_iou"]:.2f}% | {res_f["mean_iou"]:.2f}% |',
             f'| Macro F1 | {res_u["macro_f1"]:.2f}% | {res_f["macro_f1"]:.2f}% |']
    lines += [f'| IoU {c} | {res_u["iou"][c]:.2f}% | {res_f["iou"][c]:.2f}% |'
              for c in CLASSES]
    lines += [f'| Trainable parameters | {hist_u.get("parameters", 0):,} | '
              f'{hist_f.get("trainable_parameters", 0):,} of '
              f'{hist_f.get("parameters", 0):,} |',
              f'| Epochs / best epoch | {n_epochs_u} / {hist_u.get("best_epoch")} | '
              f'{n_epochs_f} / {hist_f.get("best_epoch")} |',
              f'| Training time | {hist_u.get("train_time_s", 0) / 60:.1f} min | '
              f'{hist_f.get("train_time_s", 0) / 60:.1f} min |',
              f'| Inference per chip ({DEVICE}) | {res_u["ms_per_chip"]:.1f} ms | '
              f'{res_f["ms_per_chip"]:.1f} ms |', '',
              f'**Higher mean IoU: {better["name"]}** '
              f'({better["mean_iou"]:.2f}% vs {other["mean_iou"]:.2f}%).']
    Path('outputs/model_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[4:]))
    print('\nEvaluation complete - check outputs/plots/ for all charts')
    print('Comparison saved to: models/evaluation/model_comparison.json and '
          'outputs/model_comparison.md')


if __name__ == '__main__':
    main()
