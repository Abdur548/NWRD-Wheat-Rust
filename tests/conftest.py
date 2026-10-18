"""Shared fixtures: tiny synthetic datasets laid out and named like the Kaggle NWRD patches."""

import os

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_dataset(root, size, photos, per_photo=2):
    """<root>/<split>/{images,masks}/<photo>_patch_<n>_x<col>_y0.png. In every photograph all
    patches but the last have a rust square; the last is background only."""
    rng = np.random.default_rng(0)
    for split, ids in photos.items():
        for sub in ('images', 'masks'):
            os.makedirs(os.path.join(root, split, sub))
        for p in ids:
            for n in range(per_photo):
                name = f'{p}_patch_{n:04d}_x{n * size}_y0.png'
                img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
                mask = np.zeros((size, size), np.uint8)
                if n < per_photo - 1:
                    mask[size // 8 : size // 4, size // 8 : size // 4] = 255
                    img[mask > 0] = (30, 80, 200)
                cv2.imwrite(os.path.join(root, split, 'images', name), img)
                cv2.imwrite(os.path.join(root, split, 'masks', name), mask)
    return root


@pytest.fixture
def patches_256(tmp_path):
    """Photos 1-3 train, 4 val, 5-6 test; three 256px patches each (two with rust)."""
    photos = {'train': (1, 2, 3), 'val': (4,), 'test': (5, 6)}
    return write_dataset(str(tmp_path / 'wheat_rust_patches'), 256, photos, per_photo=3)


@pytest.fixture
def patches_512(tmp_path):
    """Photos 1-3 train, 4 val, 5-6 test; two 512px patches each (one with rust)."""
    photos = {'train': (1, 2, 3), 'val': (4,), 'test': (5, 6)}
    return write_dataset(str(tmp_path / 'wheat_rust_patches'), 512, photos)
