"""Runs one experiment: [tile] -> train -> evaluate -> benchmark -> figures -> compare -> export.

    python -m nwrd.run --preset main --data-root /path/to/wheat_rust_patches --out runs
    python -m nwrd.run --preset 128px --data-root /path/to/wheat_rust_patches --out runs_128px
    python -m nwrd.run --stage compare --out results/512px/final     # re-aggregate only

Finished stages are skipped, and an interrupted training run resumes from last.pt, so the same
command can be repeated across Kaggle sessions until everything is done. Exit code 3 means the
time budget stopped training: run the same command again to continue.
"""

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import typing
from dataclasses import fields

from . import __version__
from .config import PRESETS, Config

STAGES = ('train', 'eval', 'bench', 'figures', 'compare', 'export')
# Stages that create run records. Only these write config.json and log.txt, so re-aggregating
# a finished results folder (compare, export) never overwrites what the original run recorded.
RECORDING_STAGES = {'train', 'eval', 'bench', 'figures'}


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ('1', 'true', 'yes'):
        return True
    if v in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, got {s!r}')


def parse_args(argv=None) -> argparse.Namespace:
    """Every Config field is a flag; the preset's values are the defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--preset', choices=sorted(PRESETS), default='main')
    preset = pre.parse_known_args(argv)[0].preset
    defaults = {**Config().to_dict(), **PRESETS[preset]}
    hints = typing.get_type_hints(Config)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--preset',
        choices=sorted(PRESETS),
        default='main',
        help='the experiment whose settings are the defaults (nwrd/config.py)',
    )
    p.add_argument('--stage', default='all', choices=('all',) + STAGES)
    p.add_argument('--out', dest='out_dir', default=defaults['out_dir'])
    for f in fields(Config):
        if f.name == 'out_dir':
            continue
        flag, hint, default = '--' + f.name.replace('_', '-'), hints[f.name], defaults[f.name]
        if hint is bool:
            p.add_argument(
                flag, dest=f.name, type=_parse_bool, default=default, metavar='{true,false}'
            )
        elif typing.get_origin(hint) is list:
            p.add_argument(
                flag, dest=f.name, nargs='+', type=typing.get_args(hint)[0], default=default
            )
        else:
            p.add_argument(flag, dest=f.name, type=hint, default=default)
    return p.parse_args(argv)


def _environment() -> str:
    """Library versions and device, recorded at the top of every session's log."""
    from importlib.metadata import PackageNotFoundError, version

    import cv2
    import numpy
    import torch

    def v(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return 'not installed'

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return (
        f'python {platform.python_version()}, torch {torch.__version__}, '
        f'numpy {numpy.__version__}, opencv {cv2.__version__}, '
        f'albumentations {v("albumentations")}, '
        f'segmentation-models-pytorch {v("segmentation_models_pytorch")}, '
        f'efficientnet-pytorch {v("efficientnet_pytorch")}, device {device}'
    )


def main(argv=None) -> int:
    t_session = time.time()  # the time budget counts from here, across all runs
    a = parse_args(argv)
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
    cfg.validate()
    stages = STAGES if a.stage == 'all' else (a.stage,)
    os.makedirs(cfg.out_dir, exist_ok=True)
    recording = bool(RECORDING_STAGES & set(stages))
    log_file = (
        open(os.path.join(cfg.out_dir, 'log.txt'), 'a', encoding='utf-8') if recording else None
    )

    def log(msg: str) -> None:
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        if log_file:
            log_file.write(line + '\n')
            log_file.flush()

    try:
        return _run(cfg, a.preset, stages, log, t_session, recording)
    except Exception:
        # On Kaggle the notebook's own output cannot be downloaded, but runs/log.txt can.
        if log_file:
            log_file.write(f'{time.strftime("%H:%M:%S")} CRASHED:\n{traceback.format_exc()}')
        raise
    finally:
        if log_file:
            log_file.close()


def _run(cfg: Config, preset: str, stages: tuple, log, t_session: float, recording: bool) -> int:
    log(f'nwrd {__version__} | preset {preset} | stages {",".join(stages)} | {_environment()}')
    if {'train', 'eval', 'figures'} & set(stages):
        from .data import find_data_root

        cfg.data_root = find_data_root(cfg.data_root or None)
        log(f'data: {cfg.data_root}')
        if cfg.tile_size:
            # Tiles are rebuilt each session outside out_dir, so they never enter the saved output.
            from .tiles import build_tiles

            tile_dir = cfg.tile_dir or os.path.join(
                tempfile.gettempdir(), f'nwrd_tiles_{cfg.tile_size}px_x{cfg.tile_scale:g}'
            )
            cfg.data_root = build_tiles(
                cfg.data_root, tile_dir, cfg.tile_scale, cfg.tile_size, log=log
            )
            shutil.copy(
                os.path.join(tile_dir, 'tiling.json'), os.path.join(cfg.out_dir, 'tiling.json')
            )
            log(f'data: {cfg.tile_size}px tiles at {cfg.tile_scale:g}x in {cfg.data_root}')
    if recording:
        with open(os.path.join(cfg.out_dir, 'config.json'), 'w') as f:
            json.dump({'preset': preset, **cfg.to_dict()}, f, indent=2)

    # Seed-major order: after each seed finishes, every model has a paired run to compare.
    # unet_tuned is a control (is the shared lr fair to the scratch UNet?), so its runs go last:
    # if sessions or quota run out, the main comparison is the part that is complete.
    runs = [(m, s, os.path.join(cfg.out_dir, m, f'seed{s}')) for s in cfg.seeds for m in cfg.models]
    runs.sort(key=lambda r: r[0] == 'unet_tuned')
    complete, session_start = True, None
    for model, seed, run_dir in runs:
        if 'train' in stages and not os.path.exists(os.path.join(run_dir, 'TRAINED')):
            from .train import train_run

            if not train_run(cfg, model, seed, run_dir, log=log, session_start=session_start):
                complete = False
                break  # out of time budget: stop here, resume next session
            session_start = t_session
        if (
            'eval' in stages
            and os.path.exists(os.path.join(run_dir, 'TRAINED'))
            and not os.path.exists(os.path.join(run_dir, 'metrics.json'))
        ):
            from .evaluate import evaluate_run

            evaluate_run(cfg, model, run_dir, log=log)

    if 'bench' in stages and not os.path.exists(os.path.join(cfg.out_dir, 'benchmark.json')):
        from .benchmark import run_benchmarks

        run_benchmarks(
            cfg.bench_models or cfg.models,
            os.path.join(cfg.out_dir, 'benchmark.json'),
            size=cfg.img_size,
            quick=cfg.bench_quick,
            log=log,
            batches=cfg.bench_batches,
        )

    if 'figures' in stages:
        from .evaluate import plot_qualitative

        first = {m: os.path.join(cfg.out_dir, m, f'seed{cfg.seeds[0]}') for m in cfg.models}
        first = {m: d for m, d in first.items() if os.path.exists(os.path.join(d, 'metrics.json'))}
        if first:
            os.makedirs(os.path.join(cfg.out_dir, 'figures'), exist_ok=True)
            plot_qualitative(
                cfg, first, os.path.join(cfg.out_dir, 'figures', 'qualitative_results.png')
            )

    if 'compare' in stages:
        from .compare import compare

        compare(cfg.out_dir, log=log)

    if 'export' in stages:
        if cfg.tile_size:
            # The app serves full-resolution photographs to a model trained on 512px patches; a
            # model trained on tiles of half-size photographs must never replace it.
            log('[export] skipped: models trained on tiles are not for the app')
        else:
            from .export import export_for_app

            export_for_app(cfg.out_dir, log=log)

    if not complete:
        log('INCOMPLETE: re-run the same command in a new session to continue.')
    return 0 if complete else 3


if __name__ == '__main__':
    sys.exit(main())
