"""Significance testing and the pre-registered decision rule."""

import numpy as np
import pytest

from nwrd.compare import paired_bootstrap, source_photo, verdict


def _runs(counts, files):
    return {1: {'files': files, 'counts': counts}}


def test_photo_bootstrap_is_wider_when_patches_of_a_photo_move_together():
    # 4 photographs x 10 patches. Ours wins every patch of photos 1-2 and loses every patch of
    # photos 3-4: patch-level resampling sees 40 consistent votes, photo-level only 4.
    files = np.array([f'{p}_patch_{i:04d}_x0_y0.png' for p in range(1, 5) for i in range(10)])
    ours, base = np.zeros((40, 4), np.int64), np.zeros((40, 4), np.int64)
    for i in range(40):
        good = i < 20
        ours[i] = (90, 5, 5, 900) if good else (70, 15, 15, 900)
        base[i] = (70, 15, 15, 900) if good else (88, 6, 6, 900)
    patch = paired_bootstrap(_runs(ours, files), _runs(base, files), [1])['IoU']
    photo = paired_bootstrap(_runs(ours, files), _runs(base, files), [1], by_photo=True)['IoU']
    assert patch['delta'] == photo['delta']
    assert (patch['units'], photo['units']) == (40, 4)

    def width(b):
        return b['ci95'][1] - b['ci95'][0]

    assert width(photo) > 2 * width(patch)


def test_bootstrap_refuses_unpaired_test_sets():
    files = np.array(['1_patch_0000_x0_y0.png', '2_patch_0000_x0_y0.png'])
    counts = np.ones((2, 4), np.int64)
    with pytest.raises(ValueError):
        paired_bootstrap(_runs(counts, files), _runs(counts, files[::-1]), [1])


def test_source_photo_groups_patches_and_tiles():
    assert source_photo('17_patch_0042_x1024_y512.png') == '17'
    assert source_photo('17_patch_0042_x1024_y512_t01.png') == '17'


@pytest.mark.parametrize(
    'delta, ci, wins, expected',
    [
        (0.02, (0.005, 0.04), 3, 'improves'),
        (0.02, (0.005, 0.04), 1, 'comparable'),  # CI excludes 0 but too few seed wins
        (0.01, (-0.01, 0.03), 3, 'comparable'),  # CI includes 0
        (-0.04, (-0.08, -0.006), 0, 'worse'),
        (-0.02, (-0.042, 0.019), 0, 'comparable'),
    ],
)
def test_decision_rule(delta, ci, wins, expected):
    assert verdict({'delta': delta, 'ci95': list(ci)}, wins) == expected
