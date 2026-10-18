"""NWRD patch dataset: split discovery, background subsampling, transforms.

The dataset (Kaggle `abdur548/nwrd-patched`) holds 512x512 native-resolution patches of the 100
NWRD photographs, split by photograph: <root>/{train,val,test}/{images,masks}/<photo>_patch_*.png.
"""

import glob
import json
import os
import random

import cv2
import numpy as np
import torch

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics
DATASET_DIRNAME = 'wheat_rust_patches'


def find_data_root(hint: str | None = None) -> str:
    """Returns the directory containing train/ val/ test/, searching `hint`, then Kaggle's
    input mount."""
    candidates = [hint] if hint else []
    candidates.append('/kaggle/input')
    for base in candidates:
        if os.path.isdir(os.path.join(base, 'train', 'images')):
            return base
        hits = glob.glob(os.path.join(base, '**', DATASET_DIRNAME), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f'no {DATASET_DIRNAME}/ found under {candidates}; attach the Kaggle dataset '
        f'abdur548/nwrd-patched or pass --data-root'
    )


def split_dirs(root: str, split: str) -> tuple[str, str]:
    return os.path.join(root, split, 'images'), os.path.join(root, split, 'masks')


def read_mask(path: str) -> np.ndarray:
    return (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)


def mask_stats(root: str, split: str, cache_dir: str | None = None) -> dict:
    """Per-file rust pixel counts for a split, cached as JSON (reading 8k masks is slow)."""
    cache = os.path.join(cache_dir, f'mask_stats_{split}.json') if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    _, mask_dir = split_dirs(root, split)
    stats = {}
    for name in sorted(os.listdir(mask_dir)):
        m = read_mask(os.path.join(mask_dir, name))
        stats[name] = {'pos': int(m.sum()), 'total': int(m.size)}
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, 'w') as f:
            json.dump(stats, f)
    return stats


def select_train_files(stats: dict, bg_keep_ratio: float, seed: int) -> tuple[list[str], dict]:
    """All rust-positive patches plus a seeded random fraction of background-only ones.

    The seed is the *data* seed, fixed across models so every model sees the same patches."""
    disease = sorted(f for f, s in stats.items() if s['pos'] > 0)
    background = sorted(f for f, s in stats.items() if s['pos'] == 0)
    kept = random.Random(seed).sample(background, int(len(background) * bg_keep_ratio))
    info = {'disease': len(disease), 'bg_kept': len(kept), 'bg_total': len(background)}
    return disease + sorted(kept), info


def pos_weight_from_stats(stats: dict, files: list[str]) -> float:
    """Background-to-rust pixel ratio over `files`: the BCE positive-class weight."""
    pos = sum(stats[f]['pos'] for f in files)
    total = sum(stats[f]['total'] for f in files)
    return (total - pos) / max(pos, 1)


def train_transform(size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def eval_transform(size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


class RustDataset(torch.utils.data.Dataset):
    """(image tensor, mask tensor of shape (1, H, W), index) for each file of a split."""

    def __init__(self, root: str, split: str, files, transform):
        self.img_dir, self.mask_dir = split_dirs(root, split)
        self.files, self.transform = list(files), transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)), cv2.COLOR_BGR2RGB)
        out = self.transform(image=img, mask=read_mask(os.path.join(self.mask_dir, name)))
        return out['image'], out['mask'].unsqueeze(0), idx
