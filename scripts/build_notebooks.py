"""Builds the Kaggle notebooks, one per experiment, each embedding the nwrd/ package so it runs
without cloning the repository:

    notebooks/nwrd_experiments.ipynb   main: 512px native-resolution patches, all four models
    notebooks/nwrd_128px.ipynb         128px: the NWRD paper's scale, CANet-B4 vs UNet

    python scripts/build_notebooks.py

Re-run after changing anything in nwrd/; tests/test_notebooks.py fails while a notebook is
stale. The experiment's settings live in nwrd/config.py (PRESETS); a notebook only names its
preset.
"""

import json
import os

from run_on_kaggle import BUDGET_LINE, DATASET, EXPECT_OFF, EXPERIMENTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Everything the pipeline imports; nwrd/inference.py serves the app and is not needed here.
MODULES = [
    '__init__', 'config', 'data', 'tiles', 'models', 'metrics', 'train', 'evaluate',
    'benchmark', 'compare', 'export', 'run',
]  # fmt: skip

KAGGLE_SETUP = f"""**Kaggle setup**
1. *Add Input* → dataset **`{DATASET}`**.
2. *Settings* → Accelerator **GPU T4** (one GPU is used), **Internet on** (pip and ImageNet weights).
3. *Save Version* → **Save & Run All (Commit)**. A session runs for up to 12 h.

**If a session stops at its time budget** (the log ends in `INCOMPLETE`), continue it with
`python scripts/run_on_kaggle.py{{flag}} --resume --watch` from the repository, or add this
notebook's latest version as an input and commit again. Repeat until the log has no `INCOMPLETE`."""

INTRO = {
    'main': """# NWRD wheat-rust segmentation: CANet-B4 vs baselines (512px)

Trains CANet-B4 (ours), the NWRD paper's UNet architecture, DeepLabV3+ R50 and a UNet
learning-rate control under one protocol on 512×512 native-resolution patches. Each run's
threshold is chosen on **val**; the held-out **test** split is scored once; efficiency is
benchmarked; the comparison is written to `runs/comparison.md`.

{setup}

Rough cost on a T4: 1.5–2.5 GPU-hours per run, 4 models × 3 seeds, so plan on two or three
sessions (Kaggle's weekly quota is about 30 GPU-hours). Runs go seed by seed, so a paired
comparison exists as soon as the first seed finishes; the `unet_tuned` control runs last.""",
    '128px': """# NWRD at the paper's scale: CANet-B4 vs UNet on 128×128 patches

The NWRD paper (Anwar et al., Sensors 2023) trained its UNet on **128×128 patches of photographs
halved in size** with a Lanczos filter (§3.4.1–3.4.2). This notebook trains CANet-B4 and the
paper's UNet architecture at that scale, so accuracy and efficiency are measured where the paper
measured them; the main notebook (`nwrd_experiments.ipynb`) uses 512×512 native-resolution
patches.

**The data.** `abdur548/nwrd-patched` pads each photograph and cuts it, at native resolution,
into 512×512 patches. Each patch is halved (Lanczos) and cut 2×2 into 128×128 tiles, so a tile
covers 256×256 native pixels: the footprint of one of the paper's patches. Tiles go to local disk
(not the saved output) and are rebuilt identically every session, in a few minutes.

**Everything else is the main experiment's protocol**: the same split by photograph, every rust
tile plus 20% of background tiles, 30 epochs, batch 8, AdamW lr 5e-5, checkpoint and threshold
chosen on validation, 3 seeds, photograph-level bootstrap. Only the input scale changes.

{setup}

Rough cost on a T4: 2–3 GPU-hours per seed for the two models, so 6–9 GPU-hours in all.""",
}

OUTRO = {
    'main': """## Taking the results back to the repository

`python scripts/run_on_kaggle.py --status` downloads this version's output to
`runs_kaggle/main/<account>_v<N>/` (weights stay on Kaggle; add `--with-export` for the app
model). Copy its `runs/` to `results/512px/`, without `export/`, checkpoints or `TRAINED` markers.

`export/` holds the CANet-B4 seed with the best **validation** IoU (test metrics are never used to
choose it); its files go to `app/assets/`.""",
    '128px': """## Taking the results back to the repository

`python scripts/run_on_kaggle.py --experiment 128px --status` downloads this version's output to
`runs_kaggle/128px/<account>_v<N>/` (weights stay on Kaggle). Copy its `runs/` to
`results/128px/`, without checkpoints or `TRAINED` markers.

`runs/tiling.json` records how the tiles were made and their rust-pixel share before and after
downsampling. No model is exported: these models expect 128px tiles of half-size photographs,
while the app serves full-resolution photographs to the 512px model.""",
}

SETUP = """import glob, os, shutil, subprocess, sys
!pip install -q segmentation-models-pytorch==0.5.0 efficientnet_pytorch==0.7.1 "albumentations>=2.0,<3"
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'enable a GPU accelerator'
os.makedirs('code/nwrd', exist_ok=True)
sys.path.insert(0, os.path.abspath('code'))"""

CONFIG = f"""# ── Experiment settings ─────────────────────────────────────────────────────
PRESET = '{{preset}}'   # models, input scale and benchmark batches: PRESETS in nwrd/config.py
NUM_WORKERS = 4
# Kaggle kills a session at 12 h. Lower this when the weekly GPU quota has less than ~12 h left,
# so the session stops cleanly (and can be resumed) instead of being killed.
# scripts/run_on_kaggle.py --budget-hours sets it on the pushed copy.
{BUDGET_LINE}
# True = this session must continue earlier runs; stop at once if none are attached rather than
# silently retraining everything. scripts/run_on_kaggle.py --resume sets it on the pushed copy.
{EXPECT_OFF}"""

STORAGE = """# Runs go to /kaggle/working/runs, which Kaggle keeps with each committed version.
OUT = '/kaggle/working/runs' if os.path.isdir('/kaggle/working') else os.path.abspath('runs')
os.makedirs(OUT, exist_ok=True)
print('runs ->', OUT, '| time budget', KAGGLE_TIME_BUDGET_HOURS, 'h')"""

DATA = """from nwrd.data import find_data_root

DATA_ROOT = find_data_root(None)   # the attached dataset, under /kaggle/input
print('dataset:', DATA_ROOT, '|',
      {s: len(os.listdir(os.path.join(DATA_ROOT, s, 'images'))) for s in ('train', 'val', 'test')})
# The dataset's own notes say how it was made (padding, split by photograph).
for f in ('README.md', 'processing_stats.json'):
    p = os.path.join(DATA_ROOT, f)
    if os.path.exists(p):
        print('--- %s ---' % f)
        print(open(p, encoding='utf-8', errors='replace').read()[:3000])"""

RESTORE = """# Continue an earlier session. Kaggle offers the output of a finished version as an input
# (not of one killed at the session limit); its runs/ is copied into place here.
prev = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/runs/config.json', recursive=True))
if prev and not os.path.exists(os.path.join(OUT, 'config.json')):
    shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
    print('restored previous runs from', prev[0])
    with open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write('--- new session: restored previous runs from %s\\n' % prev[0])
states = {}
for d in sorted(glob.glob(os.path.join(OUT, '*', 'seed*'))):
    states[d] = ('done' if os.path.exists(os.path.join(d, 'metrics.json')) else
                 'trained' if os.path.exists(os.path.join(d, 'TRAINED')) else
                 'partial' if os.path.exists(os.path.join(d, 'last.pt')) else 'empty')
    print(' ', os.path.relpath(d, OUT), states[d])
print('%d run directories found' % len(states))
if EXPECT_RESTORE and not any(s != 'empty' for s in states.values()):
    print('/kaggle/input:', sorted(glob.glob('/kaggle/input/*/*/*'))[:30])
    raise RuntimeError('Resume was requested but no earlier runs/ was found under /kaggle/input. '
                       'Stopping now instead of retraining every run from scratch.')"""

RUN = """cmd = [sys.executable, '-m', 'nwrd.run', '--preset', PRESET, '--data-root', DATA_ROOT,
       '--out', OUT, '--time-budget-hours', str(KAGGLE_TIME_BUDGET_HOURS),
       '--num-workers', str(NUM_WORKERS)]
print(' '.join(cmd))
proc = subprocess.Popen(cmd, cwd='code', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='')
code = proc.wait()
print('exit code', code, '(3 = stopped at the time budget; commit again to resume)')
# Fail the version, so Kaggle reports ERROR instead of a COMPLETE run that trained nothing.
if code not in (0, 3):
    raise RuntimeError('the pipeline failed (exit code %d); the traceback is in runs/log.txt' % code)"""

SHOW = """from IPython.display import Image, Markdown, display
if os.path.exists(f'{OUT}/comparison.md'):
    display(Markdown(open(f'{OUT}/comparison.md', encoding='utf-8').read()))
    for f in ('training_curves.png', 'qualitative_results.png'):
        if os.path.exists(f'{OUT}/figures/{f}'):
            display(Image(f'{OUT}/figures/{f}'))"""


def md(src: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(True)}


def code(src: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': src.splitlines(True),
    }


def build(name: str) -> dict:
    """The notebook of experiment `name` as a JSON-ready dict."""
    flag = '' if name == 'main' else f' --experiment {name}'
    cells = [
        md(INTRO[name].format(setup=KAGGLE_SETUP.format(flag=flag))),
        code(SETUP),
        code(CONFIG.format(preset=name)),
        code(STORAGE),
        md(
            '### Pipeline source (generated from `nwrd/` by `scripts/build_notebooks.py`; edit there)'
        ),
    ]
    for m in MODULES:
        with open(os.path.join(ROOT, 'nwrd', f'{m}.py'), encoding='utf-8') as f:
            cells.append(code(f'%%writefile code/nwrd/{m}.py\n' + f.read()))
    cells += [
        md('### Data and resume'),
        code(DATA),
        code(RESTORE),
        md('### Run the experiment'),
        code(RUN),
        code(SHOW),
        md(OUTRO[name]),
    ]
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    return {
        'cells': cells,
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU',
            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True},
        },
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + '\n'


def main() -> None:
    for name, e in EXPERIMENTS.items():
        out = os.path.join(ROOT, 'notebooks', e['notebook'])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(render(build(name)))
        print('wrote', out)


if __name__ == '__main__':
    main()
