"""Pooled metrics: the histogram gives the same counts as thresholding every pixel."""

import numpy as np
import torch

from nwrd.metrics import ProbHistogram, dice, iou, per_image_counts, scores


def test_histogram_counts_match_direct_thresholding():
    torch.manual_seed(0)
    probs, target = torch.rand(4, 1, 32, 32), (torch.rand(4, 1, 32, 32) > 0.8).float()
    target[0] = 0  # an all-background image
    hist = ProbHistogram()
    hist.update(probs, target)
    for t in (0.3, 0.5, 0.71):
        direct = per_image_counts(probs, target, t).sum(0)
        assert tuple(direct) == tuple(int(v) for v in hist.counts(t))


def test_iou_and_dice_are_consistent():
    s = scores(90, 10, 20, 880)
    assert 0 < s['IoU'] < s['Dice'] < 1
    assert abs(s['IoU'] - s['Dice'] / (2 - s['Dice'])) < 1e-9
    counts = np.array([[90, 10, 20, 880], [0, 0, 0, 1000]])
    assert np.allclose(iou(counts), [90 / 120, 0.0])
    assert np.allclose(dice(counts), [180 / 210, 0.0])


def test_best_threshold_maximises_iou_on_the_grid():
    torch.manual_seed(1)
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()
    probs = (0.6 * target + 0.4 * torch.rand_like(target)).clamp(0, 1)
    hist = ProbHistogram()
    hist.update(probs, target)
    thr, rows = hist.best_threshold(lo=0.05, hi=0.95, step=0.01)
    assert len(rows) == 91 and rows[0]['threshold'] == 0.05 and rows[-1]['threshold'] == 0.95
    assert max(r['IoU'] for r in rows) == next(r['IoU'] for r in rows if r['threshold'] == thr)
