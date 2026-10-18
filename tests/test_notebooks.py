"""The Kaggle notebooks must embed the current nwrd/ package; Kaggle runs the notebook, not the
repository, so a stale notebook would silently run old code."""

import json
import os
import sys

import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import build_notebooks  # noqa: E402
import run_on_kaggle  # noqa: E402


@pytest.mark.parametrize('name', sorted(run_on_kaggle.EXPERIMENTS))
def test_notebook_is_up_to_date(name):
    path = os.path.join(ROOT, 'notebooks', run_on_kaggle.EXPERIMENTS[name]['notebook'])
    with open(path, encoding='utf-8') as f:
        on_disk = f.read()
    assert on_disk == build_notebooks.render(build_notebooks.build(name)), (
        f'{path} is stale: run python scripts/build_notebooks.py'
    )


@pytest.mark.parametrize('name', sorted(run_on_kaggle.EXPERIMENTS))
def test_notebook_markers_the_runner_rewrites(name):
    """--resume and --budget-hours each rewrite exactly one line of the pushed copy."""
    nb = build_notebooks.render(build_notebooks.build(name))
    assert nb.count(run_on_kaggle.EXPECT_OFF) == 1
    assert nb.count(run_on_kaggle.BUDGET_LINE) == 1
    cells = [''.join(c['source']) for c in json.loads(nb)['cells'] if c['cell_type'] == 'code']
    config = next(c for c in cells if c.startswith('# ── Experiment settings'))
    assert f"PRESET = '{name}'" in config


def test_notebooks_embed_every_pipeline_module():
    cells = [''.join(c['source']) for c in build_notebooks.build('main')['cells']]
    embedded = [c.split('\n', 1)[0] for c in cells if c.startswith('%%writefile')]
    assert embedded == [f'%%writefile code/nwrd/{m}.py' for m in build_notebooks.MODULES]
    pipeline = {f[:-3] for f in os.listdir(os.path.join(ROOT, 'nwrd')) if f.endswith('.py')}
    assert pipeline - set(build_notebooks.MODULES) == {'inference'}  # app only
