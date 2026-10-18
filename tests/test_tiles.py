"""Tiling at the NWRD paper's scale: 512px patch -> 0.5x Lanczos -> 2x2 tiles of 128px."""

import json
import os

import cv2
import numpy as np
import pytest

from nwrd.compare import source_photo
from nwrd.tiles import build_tiles, tile_name

from .conftest import write_dataset


def test_tiles_cover_the_paper_patch_footprint(patches_512, tmp_path):
    dst = str(tmp_path / 'tiles')
    build_tiles(patches_512, dst, scale=0.5, tile=128, workers=1)

    # A 512px patch halves to 256px and splits 2x2: four 128px tiles per patch.
    names = sorted(os.listdir(os.path.join(dst, 'test', 'images')))
    assert len(names) == 4 * 4
    assert names == sorted(os.listdir(os.path.join(dst, 'test', 'masks')))
    assert {source_photo(n) for n in names} == {'5', '6'}  # tiles still group by photograph

    patch = '5_patch_0000_x0_y0.png'
    img = cv2.imread(os.path.join(patches_512, 'test', 'images', patch))
    half = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        t = cv2.imread(os.path.join(dst, 'test', 'images', tile_name(patch, r, c)))
        assert t.shape == (128, 128, 3)
        assert np.array_equal(t, half[r * 128 : (r + 1) * 128, c * 128 : (c + 1) * 128])
    # The rust square (64-128 px in the patch) lands at 32-64 px in the top-left tile only.
    masks = {
        (r, c): cv2.imread(
            os.path.join(dst, 'test', 'masks', tile_name(patch, r, c)), cv2.IMREAD_GRAYSCALE
        )
        for r in (0, 1)
        for c in (0, 1)
    }
    assert set(np.unique(masks[0, 0])) == {0, 255}
    assert (masks[0, 0] > 0).sum() == 32 * 32 and masks[0, 0][32:64, 32:64].all()
    assert all((m == 0).all() for k, m in masks.items() if k != (0, 0))

    with open(os.path.join(dst, 'tiling.json')) as f:
        stats = json.load(f)
    assert stats['splits']['train'] == {
        'patches': 6,
        'tiles': 24,
        'rust_tiles': 3,
        'rust_pixel_fraction_patches': pytest.approx(64 * 64 / 512**2 / 2),
        'rust_pixel_fraction_tiles': pytest.approx(32 * 32 / 128**2 / 8),
    }


def test_tiles_are_reused_but_never_mixed(tmp_path):
    photos = {'train': (1,), 'val': (2,), 'test': (3,)}
    src = write_dataset(str(tmp_path / 'patches'), 256, photos)
    dst = str(tmp_path / 'tiles')
    build_tiles(src, dst, scale=0.5, tile=64, workers=1)
    logs = []
    assert build_tiles(src, dst, scale=0.5, tile=64, workers=1, log=logs.append) == dst
    assert logs == [f'tiles: reusing {dst}']
    with pytest.raises(RuntimeError):
        build_tiles(src, dst, scale=0.25, tile=64, workers=1)
    with pytest.raises(ValueError):
        build_tiles(src, str(tmp_path / 'other'), scale=0.5, tile=96, workers=1)  # 128/96
    with pytest.raises(ValueError):
        build_tiles(src, src)
