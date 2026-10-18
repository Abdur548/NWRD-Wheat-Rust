"""End-to-end runs of the pipeline on CPU with tiny synthetic datasets (slow: minutes)."""

import json
import os

import numpy as np
import pytest
import torch

from nwrd.compare import source_photo
from nwrd.run import main

pytestmark = pytest.mark.slow

FAST = [
    '--img-size', '128', '--batch-size', '2', '--num-workers', '0',
    '--pretrained', 'false', '--bench-quick', 'true',
]  # fmt: skip


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _history(run_dir):
    h = json.loads(_read(os.path.join(run_dir, 'history.json')))
    h['history'].pop('epoch_seconds')  # wall clock
    return h


def _weights(run_dir):
    return torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=True)[
        'model'
    ]


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_full_pipeline(patches_256, tmp_path):
    out = tmp_path / 'runs'
    args = ['--data-root', patches_256, '--out', str(out), *FAST,
            '--models', 'canet_b4', 'unet', 'unet_tuned', '--seeds', '1', '2', '--epochs', '2',
            '--freeze-encoder-epochs', '1', '--bg-keep-ratio', '0.5',
            '--bench-models', 'canet_b4']  # fmt: skip
    assert main(args) == 0
    started = [
        line.split('] data:')[0].split('[')[1]
        for line in _read(out / 'log.txt').splitlines()
        if '] data:' in line
    ]
    assert started == [
        'canet_b4 s1',
        'unet s1',
        'canet_b4 s2',
        'unet s2',
        'unet_tuned s1',
        'unet_tuned s2',
    ]  # fmt: skip  (the lr control runs last)
    m = json.loads(_read(out / 'canet_b4' / 'seed1' / 'metrics.json'))
    assert 0.05 <= m['threshold'] <= 0.95 and m['test_images'] == 6
    assert 'test_rust_patches@thr' in m
    assert not (out / 'canet_b4' / 'seed1' / 'last.pt').exists()

    comp = json.loads(_read(out / 'comparison.json'))
    sig = comp['significance']['unet']
    assert sig['seeds'] == [1, 2]
    assert sig['verdict'] in ('improves', 'comparable', 'worse') and 0 <= sig['canet_wins'] <= 2
    assert set(sig) >= {'bootstrap', 'bootstrap_photo', 'seed_ttest'}
    tuned = json.loads(_read(out / 'unet_tuned' / 'seed1' / 'history.json'))['history']
    shared = json.loads(_read(out / 'unet' / 'seed1' / 'history.json'))['history']
    assert tuned['lr'][0] == pytest.approx(1e-3) and shared['lr'][0] == pytest.approx(5e-5)
    md = _read(out / 'comparison.md')
    assert 'paper-architecture baseline' in md and 'Against the NWRD paper' in md
    assert '| Verdict |' in md

    exp = json.loads(_read(out / 'export' / 'results.json'))
    assert set(exp['final_metrics']) >= {'IoU', 'F1 / Dice'} and exp['split'] == 'test'
    for f in ('best_model.pth', 'confusion_matrix.png', 'training_curves.png',
              'qualitative_results.png'):  # fmt: skip
        assert (out / 'export' / f).exists(), f
    assert json.loads(_read(out / 'config.json'))['preset'] == 'main'
    assert [r['model'] for r in json.loads(_read(out / 'benchmark.json'))['results']] == [
        'canet_b4'
    ]

    # Re-running is a no-op that re-aggregates.
    assert main(args) == 0
    # Re-aggregating alone never rewrites what the run recorded.
    config, log = _read(out / 'config.json'), _read(out / 'log.txt')
    assert main(['--stage', 'compare', '--out', str(out)]) == 0
    assert _read(out / 'config.json') == config and _read(out / 'log.txt') == log


def test_training_is_reproducible_and_resumes_exactly(patches_256, tmp_path):
    """Same seed, same outputs; a run stopped by the time budget and resumed sees the same
    batches and augmentations as one that ran straight through."""
    base = ['--data-root', patches_256, *FAST, '--models', 'canet_b4', '--seeds', '1',
            '--epochs', '3', '--freeze-encoder-epochs', '1', '--stage', 'train']  # fmt: skip
    straight, again, resumed = (tmp_path / n for n in ('straight', 'again', 'resumed'))
    assert main(base + ['--out', str(straight)]) == 0
    assert main(base + ['--out', str(again)]) == 0
    assert main(base + ['--out', str(resumed), '--time-budget-hours', '1e-9']) == 3  # 1 epoch
    run = resumed / 'canet_b4' / 'seed1'
    assert (run / 'last.pt').exists() and not (run / 'TRAINED').exists()
    assert len(_history(run)['history']['train_loss']) == 1
    assert main(base + ['--out', str(resumed)]) == 0

    ref = straight / 'canet_b4' / 'seed1'
    for other in (again, resumed):
        assert _history(other / 'canet_b4' / 'seed1') == _history(ref)
        assert _same_weights(_weights(other / 'canet_b4' / 'seed1'), _weights(ref))


def test_augmentation_is_seeded_in_worker_processes(patches_256, tmp_path):
    base = ['--data-root', patches_256, *FAST, '--models', 'canet_b4', '--seeds', '1',
            '--epochs', '1', '--stage', 'train', '--num-workers', '2']  # fmt: skip
    a, b = tmp_path / 'a', tmp_path / 'b'
    assert main(base + ['--out', str(a)]) == 0
    assert main(base + ['--out', str(b)]) == 0
    assert _history(a / 'canet_b4' / 'seed1') == _history(b / 'canet_b4' / 'seed1')


def test_resume_survives_checkpoints_loaded_off_the_cpu(patches_256, tmp_path, monkeypatch):
    # On a GPU, loading last.pt onto the compute device put the saved RNG state on the GPU,
    # and torch.set_rng_state rejects that, so every resume crashed. Emulate it: whatever is
    # not loaded explicitly to 'cpu' lands on another device ('meta' stands in for cuda).
    from nwrd import train

    out = tmp_path / 'runs'
    base = ['--data-root', patches_256, '--out', str(out), *FAST, '--models', 'canet_b4',
            '--seeds', '1', '--epochs', '2', '--stage', 'train']  # fmt: skip
    assert main(base + ['--time-budget-hours', '1e-9']) == 3
    real_load = torch.load

    def load_like_on_gpu(path, map_location=None, **kw):
        ck = real_load(path, map_location='cpu', **kw)
        if map_location != 'cpu' and isinstance(ck, dict) and 'rng_cpu' in ck:
            ck['rng_cpu'] = ck['rng_cpu'].to('meta')
        return ck

    monkeypatch.setattr(train.torch, 'load', load_like_on_gpu)
    assert main(base) == 0
    assert (out / 'canet_b4' / 'seed1' / 'TRAINED').exists()


def test_time_budget_counts_from_session_start(patches_256, tmp_path):
    # Each run fits the budget on its own clock, but the session does not: the second run
    # must not start. Measuring per run let a Kaggle session overrun its 12 h limit.
    out = tmp_path / 'runs'
    args = ['--data-root', patches_256, '--out', str(out), *FAST, '--models', 'canet_b4', 'unet',
            '--seeds', '1', '--epochs', '1', '--stage', 'train',
            '--time-budget-hours', '1e-9']  # fmt: skip
    assert main(args) == 3
    assert (out / 'canet_b4' / 'seed1' / 'TRAINED').exists()
    assert not (out / 'unet' / 'seed1').exists() or not any((out / 'unet' / 'seed1').iterdir())
    assert main(args) == 0  # the next session's first run always makes progress
    assert (out / 'unet' / 'seed1' / 'TRAINED').exists()


def test_full_pipeline_on_tiles(patches_512, tmp_path):
    tiles, out = tmp_path / 'tiles', tmp_path / 'runs'
    args = ['--preset', '128px', '--data-root', patches_512, '--out', str(out),
            '--tile-dir', str(tiles), *FAST, '--seeds', '1', '--epochs', '1',
            '--freeze-encoder-epochs', '1', '--bg-keep-ratio', '0.5']  # fmt: skip
    assert main(args) == 0
    m = json.loads(_read(out / 'canet_b4' / 'seed1' / 'metrics.json'))
    assert m['test_images'] == 16 and m['test_rust_images'] == 2
    files = np.load(out / 'canet_b4' / 'seed1' / 'test_counts.npz')['files']
    assert {source_photo(f) for f in files} == {'5', '6'}
    assert json.loads(_read(out / 'tiling.json'))['tile'] == 128
    config = json.loads(_read(out / 'config.json'))
    assert config['preset'] == '128px' and config['models'] == ['canet_b4', 'unet']
    assert config['tile_size'] == 128 and config['bench_batches'] == [8, 64]
    bench = json.loads(_read(out / 'benchmark.json'))['results']
    assert {r['input'] for r in bench} == {'128x128'}
    assert "128px tiles re-cut at 0.5× scale, the paper's own patch size" in _read(
        out / 'comparison.md'
    )
    assert not (out / 'export').exists()  # a tile-trained model never goes to the app

    with pytest.raises(ValueError):  # resizing tiles to 512px would undo the tiling
        main(['--preset', '128px', '--img-size', '512', '--out', str(tmp_path / 'x')])
