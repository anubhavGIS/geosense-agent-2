# finetune_prithvi.py
# Purpose: Fine-tune the NASA/IBM Prithvi-100M geospatial foundation model on the
#          Kolkata chips (5 land-cover classes), with a documented fallback to an
#          ImageNet-pretrained ResNet-50 U-Net if the Prithvi weights cannot be obtained.
# Run from project root:
#   python src/phase2_dl/finetune_prithvi.py            (full fine-tuning)
#   python src/phase2_dl/finetune_prithvi.py --check    (download + inspect only)
import json
import os
import sys
import time
from pathlib import Path

# two OpenMP runtimes on Windows (see train_unet.py); silence the HF symlink notice
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from train_unet import (SatelliteDataset, train_transform, val_transform, CHIPS_DIR,
                        LABEL_DIR, CLASS_NAMES, NUM_CLASSES, IN_CHANNELS, IGNORE_INDEX,
                        YEARS, SEED)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M'   # the Prithvi-100M repository
WEIGHTS_FILE = 'Prithvi_EO_V1_100M.pt'
CODE_FILE = 'prithvi_mae.py'
BATCH_SIZE = 4                                   # smaller: Prithvi is a larger model
NUM_EPOCHS = int(os.getenv('FT_EPOCHS', '40'))   # fewer epochs than a from-scratch run
LR_BACKBONE = 1e-5                               # very small LR for the pretrained weights
LR_HEAD = 1e-4                                   # larger LR for the new segmentation head
WEIGHT_DECAY = 0.01
MODEL_DIR = Path('models/saved'); EVAL_DIR = Path('models/evaluation')
MODEL_DIR.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)

# Prithvi was pre-trained on HLS reflectance (x 10,000) with these per-band statistics
# (config.json of the repository); the chips hold reflectance / 10,000, so the wrapper
# rescales them before normalising.
PRITHVI_MEAN = [775.2290211032589, 1080.992780391705, 1228.5855250417867,
                2497.2022620507532, 2204.2139147975554, 1610.8324823273745]
PRITHVI_STD = [1281.526139861424, 1270.0297974547493, 1399.4802505642526,
               1368.3446143747644, 1291.6764008585435, 1154.505683480695]


# ── Prithvi encoder + segmentation head ─────────────────────────────────────
class PrithviSegmenter(nn.Module):
    """Prithvi ViT-B/16 encoder (frozen but for its top blocks) + a light segmentation
    head. The tokens of the last block (14 x 14 patches of 768 features) are reshaped to
    a feature map and decoded to 224 x 224 along two paths whose class scores are summed:
      - context path: four stride-2 transposed convolutions (14 -> 28 -> 56 -> 112 -> 224)
        that blend neighbouring patches and give smooth, coherent regions;
      - detail path: a linear layer through which every token predicts the class scores
        of its own 16 x 16 pixels directly (the 'un-patchify' idea of the MAE
        reconstruction head), which keeps roads, canals and building edges that are
        narrower than one patch."""

    def __init__(self, encoder, num_classes=NUM_CLASSES, embed_dim=768, patch=16):
        super().__init__()
        self.encoder = encoder
        mean, std = torch.tensor(PRITHVI_MEAN), torch.tensor(PRITHVI_STD)
        self.register_buffer('mean', mean.view(1, -1, 1, 1) / 10000.0)
        self.register_buffer('std', std.view(1, -1, 1, 1) / 10000.0)

        def up(cin, cout):
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.head = nn.Sequential(up(embed_dim, 256), up(256, 128), up(128, 64),
                                  up(64, 32), nn.Conv2d(32, num_classes, kernel_size=1))
        self.pixel = nn.Sequential(
            nn.Conv2d(embed_dim, num_classes * patch * patch, kernel_size=1),
            nn.PixelShuffle(patch))

    def forward(self, x):
        x = (x - self.mean) / self.std                       # Prithvi normalisation
        tokens = self.encoder.forward_features(x)[-1]        # (B, 1 + 196, 768), CLS first
        tokens = tokens[:, 1:, :]                            # drop the CLS token
        b, n, c = tokens.shape
        s = int(round(n ** 0.5))
        fmap = tokens.transpose(1, 2).reshape(b, c, s, s)    # (B, 768, 14, 14)
        return self.head(fmap) + self.pixel(fmap)            # (B, classes, 224, 224)


def get_prithvi_files(weights=True):
    """Download (or fetch from the local HuggingFace cache) the Prithvi code + weights."""
    from huggingface_hub import hf_hub_download
    code_path = hf_hub_download(repo_id=MODEL_ID, filename=CODE_FILE)
    weights_path = None
    if weights:
        weights_path = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHTS_FILE)
    return code_path, weights_path


def build_prithvi_encoder(code_path):
    """Instantiate the Prithvi ViT encoder (random weights) from prithvi_mae.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('prithvi_mae', code_path)
    prithvi_mae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prithvi_mae)
    return prithvi_mae.PrithviViT(img_size=224, patch_size=(1, 16, 16), num_frames=1,
                                  in_chans=IN_CHANNELS, embed_dim=768, depth=12,
                                  num_heads=12, mlp_ratio=4.0, encoder_only=True)


def load_prithvi(verbose=True):
    """Download the Prithvi code and weights from HuggingFace and build the encoder."""
    code_path, weights_path = get_prithvi_files(weights=True)
    if verbose:
        print(f'  Prithvi code   : {code_path}')
        size_mb = os.path.getsize(weights_path) / 1e6
        print(f'  Prithvi weights: {weights_path} ({size_mb:.0f} MB)')
    encoder = build_prithvi_encoder(code_path)
    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    for wrapper in ('model', 'state_dict'):
        if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
            state = state[wrapper]
    keys = list(state.keys())
    # keep encoder weights only; strip the 'encoder.' prefix used by the MAE wrapper
    enc_state = {}
    for k, v in state.items():
        if k.startswith('decoder') or k.startswith('mask_token'):
            continue
        enc_state[k[len('encoder.'):] if k.startswith('encoder.') else k] = v
    # the checkpoint's positional embedding is for 3 time frames; ours is recomputed for 1
    dropped = [k for k in enc_state
               if k == 'pos_embed' and enc_state[k].shape != encoder.pos_embed.shape]
    for k in dropped:
        del enc_state[k]
    result = encoder.load_state_dict(enc_state, strict=False)
    if verbose:
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        more = '...' if len(missing) > 4 else ''
        print(f'  Checkpoint keys: {len(keys)} (first: {keys[0]}) | loaded into encoder: '
              f'{len(enc_state)} | missing: {missing[:4]}{more} | '
              f'unexpected: {unexpected[:4]}')
    n_missing = [k for k in result.missing_keys if k != 'pos_embed']
    if n_missing:
        raise RuntimeError(f'Prithvi weights did not match the encoder: '
                           f'missing {n_missing[:5]}')
    return encoder


def build_finetune_model(pretrained=True):
    """Prithvi if it can be downloaded and loaded, else the documented ResNet-50 fallback.
    pretrained=False builds the same architecture with random weights (to load a
    saved model)."""
    print(f'Loading {MODEL_ID} from HuggingFace...')
    try:
        if pretrained:
            encoder = load_prithvi()
        else:
            encoder = build_prithvi_encoder(get_prithvi_files(weights=False)[0])
        model = PrithviSegmenter(encoder)
        backbone = 'prithvi'
        print('  Prithvi-EO-1.0-100M encoder loaded (ViT-B/16, 6 bands, 224 x 224, '
              '1 frame).')
    except Exception as e:
        print(f'Prithvi download/load failed: {type(e).__name__}: {str(e)[:200]}')
        print('Using an ImageNet-pretrained ResNet-50 U-Net as the foundation-model '
              'substitute...')
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50',
                         encoder_weights='imagenet' if pretrained else None,
                         in_channels=IN_CHANNELS, classes=NUM_CLASSES)
        backbone = 'resnet50'
    return model, backbone


def freeze_backbone(model, backbone):
    """Freeze the pretrained layers except the ones that adapt to the task.
    Prithvi: all transformer blocks but the top two (and the final norm) are frozen.
    ResNet-50: every encoder layer except layer1 / layer2 is frozen."""
    frozen = 0
    for name, param in model.named_parameters():
        if backbone == 'prithvi':
            trainable = (not name.startswith('encoder.')
                         or name.startswith('encoder.blocks.10')
                         or name.startswith('encoder.blocks.11')
                         or name.startswith('encoder.norm'))
        else:
            in_encoder = 'encoder' in name
            trainable = not (in_encoder and 'layer1' not in name and 'layer2' not in name)
        param.requires_grad = trainable
        frozen += 0 if trainable else 1
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in model.parameters())
    what = ('patch embedding, CLS token and transformer blocks 0-9'
            if backbone == 'prithvi' else 'encoder stem, layer3 and layer4')
    print(f'Trainable parameters: {trainable_n:,} / {total_n:,} '
          f'({trainable_n / total_n * 100:.1f}%)')
    print(f'Frozen layers: {frozen} of {n_tensors} parameter tensors ({what})')
    return trainable_n, total_n, frozen


def make_loaders():
    """Same 124 / 32 split as train_unet.py (read from unet_split.json), so that both
    models are judged on the same chips."""
    split = json.load(open(EVAL_DIR / 'unet_split.json'))
    train_ds = ConcatDataset([SatelliteDataset(CHIPS_DIR / y, LABEL_DIR, y,
                                               train_transform) for y in YEARS])
    val_ds = ConcatDataset([SatelliteDataset(CHIPS_DIR / y, LABEL_DIR, y,
                                             val_transform) for y in YEARS])
    names = [p.stem for y in YEARS
             for p in sorted((CHIPS_DIR / y).glob(f'chip_{y}_*.npy'))]
    idx = {n: i for i, n in enumerate(names)}
    train_idx = [idx[n] for n in split['train']]; val_idx = [idx[n] for n in split['val']]
    return (DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0),
            DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0),
            len(train_idx), len(val_idx))


def main():
    gpu = f' ({torch.cuda.get_device_name(0)})' if DEVICE == 'cuda' else ''
    print(f'Device: {DEVICE}{gpu}')
    model, backbone = build_finetune_model()
    model = model.to(DEVICE)
    trainable_n, total_n, frozen = freeze_backbone(model, backbone)
    if '--check' in sys.argv:
        x = torch.zeros(2, IN_CHANNELS, 224, 224, device=DEVICE)
        with torch.no_grad():
            y = model(x)
        print(f'Forward check: input {tuple(x.shape)} -> output {tuple(y.shape)}  '
              f'[backbone: {backbone}]')
        return

    # ── Two learning rates: tiny for the backbone, larger for the head ──────
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    backbone_params = [p for n, p in params if 'encoder' in n]
    head_params = [p for n, p in params if 'encoder' not in n]
    optimiser = torch.optim.AdamW([{'params': backbone_params, 'lr': LR_BACKBONE},
                                   {'params': head_params, 'lr': LR_HEAD}],
                                  weight_decay=WEIGHT_DECAY)
    # same schedule as the U-Net: both rates halved every 15 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    train_loader, val_loader, n_train, n_val = make_loaders()
    print(f'Backbone: {backbone} | train {n_train} chips | val {n_val} chips | '
          f'batch {BATCH_SIZE} | epochs {NUM_EPOCHS} | '
          f'lr backbone {LR_BACKBONE} / head {LR_HEAD}')
    print(f'Fine-tuning for {NUM_EPOCHS} epochs...')

    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimiser.step()
            train_loss += loss.item()
        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1); valid = labels != IGNORE_INDEX
                correct += ((preds == labels) & valid).sum().item()
                total += valid.sum().item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total * 100
        lr_head_now = optimiser.param_groups[1]['lr']
        for k, v in zip(history, [epoch, avg_train, avg_val, val_acc, lr_head_now]):
            history[k].append(v)
        scheduler.step()
        flag = ''
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), str(MODEL_DIR / 'prithvi_finetuned.pth'))
            flag = '  <- best, saved'
        print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {avg_train:.4f} | '
              f'Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.2f}% | '
              f'{time.time() - t0:5.0f}s{flag}')

    history.update({'best_val_loss': best_val_loss,
                    'best_epoch': int(np.argmin(history['val_loss']) + 1),
                    'train_time_s': time.time() - t0, 'backbone': backbone,
                    'model_id': MODEL_ID,
                    'parameters': total_n, 'trainable_parameters': trainable_n,
                    'frozen_tensors': frozen, 'batch_size': BATCH_SIZE,
                    'lr_backbone': LR_BACKBONE, 'lr_head': LR_HEAD,
                    'device': DEVICE + gpu})
    json.dump(history, open(EVAL_DIR / 'prithvi_history.json', 'w'), indent=2)
    json.dump({'backbone': backbone, 'model_id': MODEL_ID, 'weights_file': WEIGHTS_FILE,
               'in_channels': IN_CHANNELS, 'num_classes': NUM_CLASSES},
              open(MODEL_DIR / 'prithvi_finetuned_meta.json', 'w'), indent=2)
    print(f'\nFine-tuning complete! Best validation loss: {best_val_loss:.4f} at epoch '
          f'{history["best_epoch"]} | final val acc: {history["val_acc"][-1]:.2f}% | '
          f'time {history["train_time_s"] / 60:.1f} min')
    print('Fine-tuned model saved to: models/saved/prithvi_finetuned.pth')


if __name__ == '__main__':
    main()
