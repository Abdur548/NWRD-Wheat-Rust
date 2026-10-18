"""Whole-photograph inference for the app (backend and dashboard).

The model was trained and tested on 512x512 patches cut at native resolution from padded
photographs, never on downscaled photographs. A photograph is therefore predicted the same way
the test set was scored: padded to a whole number of patches, cut into non-overlapping 512px
patches, each preprocessed exactly as at test time (data.eval_transform), predicted, and
stitched back; the padding is then cropped away. Padding mirrors the photograph's border, so
edge patches see image content rather than a black band.
"""

import io

import cv2
import numpy as np
import torch

from . import data as D
from .models import build_model

PATCH = 512
MAX_PIXELS = 40_000_000  # default upload limit: NWRD photographs are 16-24 MP
# Upper bound (exclusive, % of image pixels predicted as rust) of each label below 'severe'.
# Illustrative bands for the demo app only: they are not a validated agronomic severity scale,
# and a photograph may include soil or sky as well as leaves.
SEVERITY_BANDS = ((5.0, 'early'), (20.0, 'moderate'))


class ImageError(ValueError):
    """The upload is not a readable image."""


class ImageTooLarge(ImageError):
    """The image has more pixels than allowed."""


def load_model(checkpoint: str, device=None) -> tuple[torch.nn.Module, dict]:
    """The checkpoint's model in eval mode, and the checkpoint's metadata (without weights).
    Only tensors and plain data are unpickled."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_model(ckpt.get('model_name', 'canet_b4'), pretrained=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), {k: v for k, v in ckpt.items() if k != 'model'}


def read_image(data: bytes, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Decodes an uploaded image to RGB uint8, turned upright by its camera (EXIF) orientation.
    The pixel count is checked from the header, before anything is decoded."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise ImageTooLarge('Image too large') from None
    except Exception:
        raise ImageError('Not a readable image file') from None
    width, height = img.size
    if width * height > max_pixels:
        raise ImageTooLarge(
            f'Image has {width * height / 1e6:.0f} MP; the limit is {max_pixels / 1e6:.0f} MP'
        )
    try:
        return np.asarray(ImageOps.exif_transpose(img).convert('RGB'))
    except Exception:
        raise ImageError('Not a readable image file') from None


def encode_png(array: np.ndarray) -> bytes:
    """PNG bytes of a uint8 image (grayscale or RGB)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format='PNG')
    return buf.getvalue()


def pad_to_patches(image: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """Mirror-pads the bottom and right edges up to a whole number of patches."""
    h, w = image.shape[:2]
    ph, pw = -h % patch, -w % patch
    if not (ph or pw):
        return image
    return cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)


@torch.no_grad()
def predict(
    model: torch.nn.Module, image: np.ndarray, patch: int = PATCH, batch_size: int = 4
) -> np.ndarray:
    """Rust probability for every pixel of an RGB uint8 image of shape (H, W, 3).

    Each patch's probabilities match a test-time forward pass of that patch alone; batching
    patches together changes them only by floating-point rounding (about 1e-6 on a CPU)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f'expected an RGB uint8 image of shape (H, W, 3), got {image.shape} {image.dtype}'
        )
    device = next(model.parameters()).device
    h, w = image.shape[:2]
    padded = pad_to_patches(image, patch)
    rows, cols = padded.shape[0] // patch, padded.shape[1] // patch
    transform = D.eval_transform(patch)
    origins = [(r * patch, c * patch) for r in range(rows) for c in range(cols)]
    probs = np.zeros(padded.shape[:2], np.float32)
    for i in range(0, len(origins), batch_size):
        chunk = origins[i : i + batch_size]
        batch = torch.stack(
            [transform(image=padded[y : y + patch, x : x + patch])['image'] for y, x in chunk]
        ).to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            p = torch.sigmoid(model(batch).float())[:, 0].cpu().numpy()
        for (y, x), tile in zip(chunk, p, strict=True):
            probs[y : y + patch, x : x + patch] = tile
    return probs[:h, :w]


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 60, 60), alpha: float = 0.6):
    """The image with predicted rust pixels tinted by `color`."""
    out = image.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def severity(coverage_pct: float) -> str:
    """Illustrative label for the share of the image predicted as rust (SEVERITY_BANDS)."""
    if coverage_pct <= 0:
        return 'healthy'
    for upper, label in SEVERITY_BANDS:
        if coverage_pct < upper:
            return label
    return 'severe'


def thumbnail(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """A copy no larger than max_side on its longer side, for display."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
