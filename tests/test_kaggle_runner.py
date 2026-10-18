"""The Kaggle runner must never push a resume that can silently restart from scratch, and never
mix up kernels, versions or accounts."""

import os
import sys

import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import run_on_kaggle as k  # noqa: E402


def test_resume_push_requires_a_restore():
    nb = k.experiment('main')['notebook']
    fresh, resume = k.notebook_source(nb, False), k.notebook_source(nb, True)
    assert k.EXPECT_OFF in fresh and k.EXPECT_ON not in fresh
    assert k.EXPECT_ON in resume and k.EXPECT_OFF not in resume


def test_kernels_keep_their_slugs():
    # Earlier versions live under these kernels; resuming and fetching depend on the slugs.
    assert k.experiment('main')['slug'] == 'nwrd-canet-b4-vs-baselines'
    assert k.experiment('128px')['slug'] == 'nwrd-128px-canet-vs-unet'
    e = k.experiment('main')
    meta = k.kernel_metadata(e, 'u/' + e['slug'], resume=True)
    assert meta['title'] == 'NWRD CANet-B4 vs baselines' and meta['kernel_sources'] == [
        'u/' + e['slug']
    ]
    assert meta['code_file'] == 'nwrd_experiments.ipynb' and meta['dataset_sources'] == [k.DATASET]
    assert meta['is_private'] and meta['enable_gpu']
    assert k.kernel_metadata(e, 'u/x', resume=False)['kernel_sources'] == []


def test_experiments_never_share_a_kernel_or_folder():
    exps = [k.experiment(n) for n in k.EXPERIMENTS]
    for key in ('slug', 'notebook', 'push_dir'):
        assert len({e[key] for e in exps}) == len(exps), key
    assert len({k.output_dir(e, 'u', 1) for e in exps}) == len(exps)
    assert all(os.path.exists(e['notebook']) for e in exps)


@pytest.mark.parametrize('name', sorted(k.EXPERIMENTS))
def test_budget_override(name):
    nb = k.experiment(name)['notebook']
    assert k.BUDGET_LINE in k.notebook_source(nb, False)
    text = k.notebook_source(nb, False, budget_hours=6)
    assert 'KAGGLE_TIME_BUDGET_HOURS = 6.0' in text and k.BUDGET_LINE not in text


def test_accounts_never_overwrite_each_other(tmp_path, monkeypatch):
    # Each account's kernel numbers its versions from 1; their outputs must not collide.
    e = k.experiment('128px')
    assert k.output_dir(e, 'alice', 1) != k.output_dir(e, 'bob', 1)
    assert k.version_file(e, 'alice') != k.version_file(e, 'bob')
    cfg = tmp_path / 'kaggle.json'
    cfg.write_text('{"username": "someone", "key": "x"}')
    monkeypatch.setenv('KAGGLE_CONFIG_DIR', str(tmp_path))
    assert k.username() == 'someone'  # the CLI's own precedence, so pushes and fetches agree


@pytest.mark.parametrize(
    'text, state',
    [
        ('u/slug has status "KernelWorkerStatus.COMPLETE"', 'complete'),
        ('u/slug has status "KernelWorkerStatus.RUNNING"', 'running'),
        ('u/slug has status "KernelWorkerStatus.ERROR"', 'error'),
        ('u/slug has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', 'cancelled'),
        ('u/slug has status "QUEUED"', 'queued'),
        # A failed CLI call mentions "error" but says nothing about the kernel.
        ('ConnectionError: HTTPSConnectionPool: Max retries exceeded', 'unknown'),
    ],
)
def test_status_is_parsed_not_guessed(text, state):
    assert k.parse_status(text) == state


def test_downloads_exclude_weights_unless_asked():
    import re

    files = ['runs/log.txt', 'runs/canet_b4/seed42/best.pt', 'runs/export/best_model.pth']
    kept = {p: [f for f in files if re.match(p, f)] for p in (k.NO_WEIGHTS, k.NO_CHECKPOINTS)}
    assert kept[k.NO_WEIGHTS] == ['runs/log.txt']
    assert kept[k.NO_CHECKPOINTS] == ['runs/log.txt', 'runs/export/best_model.pth']
