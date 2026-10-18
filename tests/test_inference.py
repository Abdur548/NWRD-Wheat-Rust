"""Whole-photograph inference: predicted patch by patch exactly as the test set was scored."""

import numpy as np
import pytest
import torch

from nwrd import data as D
from nwrd import inference
from nwrd.models import build_model

PATCH = 64  # small patches keep the CPU cost low; the app uses 512


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return build_model('unet', pretrained=False).eval()


def _image(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _forward(model, tile):
    x = D.eval_transform(tile.shape[0])(image=tile)['image'][None]
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].numpy()


def test_each_patch_is_predicted_as_at_test_time(model):
    img = _image(2 * PATCH, 3 * PATCH)
    probs = inference.predict(model, img, patch=PATCH, batch_size=4)
    assert probs.shape == img.shape[:2] and probs.dtype == np.float32
    for r in range(2):
        for c in range(3):
            ys, xs = slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH)
            np.testing.assert_allclose(probs[ys, xs], _forward(model, img[ys, xs]), atol=1e-5)


@pytest.mark.parametrize('shape', [(100, 150), (30, 40), (PATCH, PATCH + 1)])
def test_any_size_is_padded_then_cropped(model, shape):
    img = _image(*shape)
    probs = inference.predict(model, img, patch=PATCH)
    assert probs.shape == shape
    padded = inference.pad_to_patches(img, PATCH)
    assert padded.shape[0] % PATCH == 0 and padded.shape[1] % PATCH == 0
    # Padding is added after the image, never mixed into it.
    np.testing.assert_array_equal(padded[: shape[0], : shape[1]], img)
    np.testing.assert_allclose(probs[: min(shape[0], PATCH), : min(shape[1], PATCH)],
                               _forward(model, padded[:PATCH, :PATCH])[: shape[0], : shape[1]],
                               atol=1e-5)  # fmt: skip


def test_rejects_anything_but_rgb_uint8(model):
    with pytest.raises(ValueError):
        inference.predict(model, np.zeros((64, 64), np.uint8), patch=PATCH)
    with pytest.raises(ValueError):
        inference.predict(model, np.zeros((64, 64, 3), np.float32), patch=PATCH)


@pytest.mark.parametrize(
    'coverage, label',
    [(0.0, 'healthy'), (0.01, 'early'), (4.99, 'early'), (5.0, 'moderate'), (19.9, 'moderate'),
     (20.0, 'severe'), (80.0, 'severe')],
)  # fmt: skip
def test_severity_bands(coverage, label):
    assert inference.severity(coverage) == label


def test_overlay_tints_only_rust_pixels():
    img = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    out = inference.overlay(img, mask)
    assert (out[0, 0] != 100).any() and (out[1:, 1:] == 100).all() and (img == 100).all()


def test_thumbnail_keeps_aspect_and_never_enlarges():
    assert inference.thumbnail(_image(2000, 1000), 1024).shape == (1024, 512, 3)
    assert inference.thumbnail(_image(300, 200), 1024).shape == (300, 200, 3)
