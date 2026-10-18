"""Re-cuts the 512px NWRD patches at the NWRD paper's scale.

The paper (Anwar et al., Sensors 2023, §3.4.1-3.4.2) halves every photograph with a Lanczos
filter, then cuts 128x128 patches. The Kaggle dataset pads each photograph and cuts it, at
native resolution, into 512x512 patches. Halving a patch and cutting it 2x2 therefore gives
128px tiles covering the same 256x256 native pixels as one of the paper's patches.

Each tile keeps its patch's name plus a _t<row><col> suffix, so compare.source_photo still
groups tiles by photograph. The output is deterministic, and a finished build is reused.

    python -m nwrd.tiles --src /path/to/wheat_rust_patches --dst /tmp/nwrd_tiles
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .data import split_dirs

SPLITS = ('train', 'val', 'test')


def tile_name(name: str, row: int, col: int) -> str:
    stem, ext = os.path.splitext(name)
    return f'{stem}_t{row}{col}{ext}'


def downsample_and_tile(img: np.ndarray, mask: np.ndarray, scale: float, tile: int):
    """Yields (row, col, image tile, mask tile). The image is resized with Lanczos, as in the
    paper. A mask pixel is rust when at least half of the area it replaces was rust."""
    h, w = mask.shape
    size = (round(w * scale), round(h * scale))
    if size[0] % tile or size[1] % tile:
        raise ValueError(
            f'{w}x{h} scaled by {scale} is {size[0]}x{size[1]}, '
            f'not a whole number of {tile}px tiles'
        )
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    rust = cv2.resize((mask > 127).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    mask = (rust >= 0.5).astype(np.uint8) * 255
    for r in range(size[1] // tile):
        for c in range(size[0] // tile):
            ys, xs = slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)
            yield r, c, img[ys, xs], mask[ys, xs]


def _tile_patch(job) -> tuple[int, int, int, int, int]:
    """Tiles one patch; returns (tiles, rust tiles, rust px before, px before, rust px after)."""
    name, src_img, src_mask, dst_img, dst_mask, scale, tile = job
    img = cv2.imread(os.path.join(src_img, name), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(src_mask, name), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError(f'cannot read {name} from {src_img} and {src_mask}')
    n = rust_tiles = rust_px = 0
    for r, c, ti, tm in downsample_and_tile(img, mask, scale, tile):
        out = tile_name(name, r, c)
        cv2.imwrite(os.path.join(dst_img, out), ti)
        cv2.imwrite(os.path.join(dst_mask, out), tm)
        pos = int((tm > 0).sum())
        n, rust_tiles, rust_px = n + 1, rust_tiles + (pos > 0), rust_px + pos
    return n, rust_tiles, int((mask > 127).sum()), int(mask.size), rust_px


def build_tiles(
    src: str, dst: str, scale: float = 0.5, tile: int = 128, workers: int | None = None, log=print
) -> str:
    """Writes <dst>/<split>/{images,masks}/ from the patches under `src` and returns `dst`.

    A finished build (marked by <dst>/tiling.json) with the same settings is reused, since
    tiling is deterministic. A build with other settings is refused rather than overwritten."""
    if os.path.abspath(dst) == os.path.abspath(src):
        raise ValueError('the tile directory must differ from the patch directory')
    params = {'source': os.path.abspath(src), 'scale': scale, 'tile': tile}
    marker = os.path.join(dst, 'tiling.json')
    if os.path.exists(marker):
        with open(marker) as f:
            done = json.load(f)
        if {k: done.get(k) for k in params} != params:
            raise RuntimeError(f'{dst} holds tiles built with {done}; use another --tile-dir')
        log(f'tiles: reusing {dst}')
        return dst

    stats = {}
    for split in SPLITS:
        src_img, src_mask = split_dirs(src, split)
        dst_img, dst_mask = split_dirs(dst, split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_mask, exist_ok=True)
        jobs = [
            (f, src_img, src_mask, dst_img, dst_mask, scale, tile)
            for f in sorted(os.listdir(src_img))
        ]
        if workers == 1:
            res = [_tile_patch(j) for j in jobs]
        else:
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_tile_patch, jobs, chunksize=32))
        n, rust_tiles, pos_before, px_before, pos_after = np.sum(res, axis=0).tolist()
        stats[split] = {
            'patches': len(jobs),
            'tiles': n,
            'rust_tiles': rust_tiles,
            'rust_pixel_fraction_patches': pos_before / px_before,
            'rust_pixel_fraction_tiles': pos_after / (n * tile * tile),
        }
        log(f'tiles: {split} {len(jobs)} patches -> {n} tiles ({rust_tiles} with rust)')
    with open(marker, 'w') as f:
        json.dump({**params, 'splits': stats}, f, indent=1)
    return dst


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--src', required=True, help='directory holding train/ val/ test/')
    p.add_argument('--dst', required=True)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--tile', type=int, default=128)
    p.add_argument('--workers', type=int, default=None)
    a = p.parse_args(argv)
    build_tiles(a.src, a.dst, a.scale, a.tile, a.workers)


if __name__ == '__main__':
    main()
