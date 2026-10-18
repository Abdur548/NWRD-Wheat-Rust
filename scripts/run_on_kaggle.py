"""Runs an experiment notebook on a Kaggle GPU from the command line.

Needs a Kaggle API token once: kaggle.com -> Settings -> API -> Create New Token, saved as
~/.kaggle/kaggle.json (or in the folder KAGGLE_CONFIG_DIR names).

    python scripts/run_on_kaggle.py --watch                      # main experiment: push, poll
    python scripts/run_on_kaggle.py --resume --watch             # continue from the last version
    python scripts/run_on_kaggle.py --status                     # check, download a stopped run
    python scripts/run_on_kaggle.py --experiment 128px --watch   # the 128px experiment
    python scripts/run_on_kaggle.py --budget-hours 6 --watch     # little weekly quota left

Each experiment is its own notebook and private kernel (EXPERIMENTS), GPU-enabled, with internet
on and the dataset attached. `--resume` attaches the kernel's previous output, which the
notebook's restore cell copies into place, so training continues instead of restarting; if
nothing can be restored, the notebook stops at once instead of retraining from scratch. Kaggle
only attaches the output of a version that finished (complete or error), so a version killed at
the 12 h limit, or when the weekly GPU quota runs out, cannot be resumed from: `--budget-hours`
stops a session cleanly before that. Repeat until the log has no INCOMPLETE line.

Output lands in runs_kaggle/<experiment>/<account>_v<version>/, one folder per pushed version,
so sessions and accounts never overwrite each other. Weights stay on Kaggle unless
--with-export (the app model, export/best_model.pth) or --with-checkpoints (everything).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = 'abdur548/nwrd-patched'
# Kaggle derives each kernel's slug from its title, so the title decides where it lives:
# renaming one moves the kernel, and --resume / --status would look in the wrong place.
EXPERIMENTS = {
    'main': {'notebook': 'nwrd_experiments.ipynb', 'title': 'NWRD CANet-B4 vs baselines'},
    '128px': {'notebook': 'nwrd_128px.ipynb', 'title': 'NWRD 128px CANet vs UNet'},
}
PUSH_ROOT = os.path.join(ROOT, 'build', 'kaggle_push')
OUT_ROOT = os.path.join(ROOT, 'runs_kaggle')
EXPECT_OFF, EXPECT_ON = 'EXPECT_RESTORE = False', 'EXPECT_RESTORE = True'
BUDGET_LINE = 'KAGGLE_TIME_BUDGET_HOURS = 11.0'
NO_WEIGHTS = r'^(?!.*\.pth?$).*$'  # neither checkpoints (*.pt) nor the exported model (*.pth)
NO_CHECKPOINTS = r'^(?!.*\.pt$).*$'  # everything but checkpoints: includes export/best_model.pth
STATES = {
    'COMPLETE': 'complete',
    'ERROR': 'error',
    'RUNNING': 'running',
    'QUEUED': 'queued',
    'CANCEL_REQUESTED': 'cancelled',
    'CANCEL_ACKNOWLEDGED': 'cancelled',
}
STOPPED = ('complete', 'error', 'cancelled')  # a stopped run's output is still worth fetching


def slugify(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def experiment(name: str) -> dict:
    e = EXPERIMENTS[name]
    return {
        'name': name,
        'title': e['title'],
        'slug': slugify(e['title']),
        'notebook': os.path.join(ROOT, 'notebooks', e['notebook']),
        'push_dir': os.path.join(PUSH_ROOT, name),
    }


def version_file(exp: dict, user: str) -> str:
    return os.path.join(exp['push_dir'], f'last_pushed_version_{user}.txt')


def output_dir(exp: dict, user: str, version) -> str:
    """Each account numbers its kernel versions from 1, so folders carry the account name."""
    return os.path.join(OUT_ROOT, exp['name'], f'{user}_v{version}')


def kaggle(*args: str, check: bool = True) -> str:
    """Runs the kaggle CLI through the current interpreter, so no PATH setup is needed."""
    p = subprocess.run([sys.executable, '-m', 'kaggle', *args], capture_output=True, text=True)
    out = (p.stdout or '') + (p.stderr or '')
    if check and p.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} failed:\n{out.strip()}')
    return out.strip()


def username() -> str:
    """The account the kaggle CLI will use: KAGGLE_CONFIG_DIR wins over ~/.kaggle, as in the CLI."""
    config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
    for path in (
        os.path.join(config_dir, 'kaggle.json') if config_dir else None,
        os.path.expanduser('~/.kaggle/kaggle.json'),
    ):
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)['username']
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    raise SystemExit(
        'No Kaggle credentials. Save kaggle.json to ~/.kaggle/ '
        '(kaggle.com -> Settings -> API -> Create New Token).'
    )


def notebook_source(notebook: str, resume: bool, budget_hours: float | None = None) -> str:
    """The notebook as pushed. With resume, the restore cell must find earlier runs or stop.
    With budget_hours, the session stops cleanly after that many hours of training."""
    with open(notebook, encoding='utf-8') as f:
        text = f.read()
    if resume:
        if text.count(EXPECT_OFF) != 1:
            raise SystemExit(
                f'expected one "{EXPECT_OFF}" line in {notebook}; rebuild it with '
                'scripts/build_notebooks.py'
            )
        text = text.replace(EXPECT_OFF, EXPECT_ON)
    if budget_hours is not None:
        if text.count(BUDGET_LINE) != 1:
            raise SystemExit(f'--budget-hours needs one "{BUDGET_LINE}" line in {notebook}')
        text = text.replace(BUDGET_LINE, f'KAGGLE_TIME_BUDGET_HOURS = {float(budget_hours)}')
    return text


def kernel_metadata(exp: dict, kernel_id: str, resume: bool) -> dict:
    return {
        'id': kernel_id,
        'title': exp['title'],
        'code_file': os.path.basename(exp['notebook']),
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [DATASET],
        'competition_sources': [],
        # Attaching the kernel's own previous output lets the notebook resume.
        'kernel_sources': [kernel_id] if resume else [],
    }


def push(exp: dict, user: str, resume: bool, budget_hours: float | None = None) -> None:
    kernel_id = f'{user}/{exp["slug"]}'
    os.makedirs(exp['push_dir'], exist_ok=True)
    with open(os.path.join(exp['push_dir'], 'kernel-metadata.json'), 'w') as f:
        json.dump(kernel_metadata(exp, kernel_id, resume), f, indent=2)
    pushed = os.path.join(exp['push_dir'], os.path.basename(exp['notebook']))
    with open(pushed, 'w', encoding='utf-8') as f:
        f.write(notebook_source(exp['notebook'], resume, budget_hours))
    out = kaggle('kernels', 'push', '-p', exp['push_dir'])
    print(out)
    m = re.search(r'version (\d+)', out)
    if m:
        with open(version_file(exp, user), 'w') as f:
            f.write(m.group(1))


def parse_status(text: str) -> str:
    """Maps `kaggle kernels status` output to complete / error / running / queued /
    cancelled, or unknown (for instance when the CLI itself failed)."""
    m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', text) or re.search(
        r'has status "([A-Z_]+)"', text
    )
    return STATES.get(m.group(1), 'unknown') if m else 'unknown'


def status(kernel_id: str) -> tuple[str, str]:
    text = kaggle('kernels', 'status', kernel_id, check=False)
    return parse_status(text), text


def fetch_output(exp: dict, user: str, pattern: str | None = NO_WEIGHTS) -> str:
    """Downloads the latest version's output into its own folder and prints the log tail.
    The CLI always serves the latest version, so the folder is named after the last version
    this script pushed from this account."""
    vfile = version_file(exp, user)
    version = 'latest'
    if os.path.exists(vfile):
        with open(vfile) as f:
            version = f.read().strip()
    out_dir = output_dir(exp, user, version)
    os.makedirs(out_dir, exist_ok=True)
    args = ['kernels', 'output', f'{user}/{exp["slug"]}', '-p', out_dir, '-q']
    if pattern:
        args += ['--file-pattern', pattern]
    print(kaggle(*args, check=False) or f'downloaded to {out_dir}')
    comparison = os.path.join(out_dir, 'runs', 'comparison.md')
    if os.path.exists(comparison):
        with open(comparison, encoding='utf-8') as f:
            print('\n' + f.read())
    else:
        print(f'no comparison.md in {out_dir}')
    log = os.path.join(out_dir, 'runs', 'log.txt')
    if os.path.exists(log):
        with open(log, encoding='utf-8') as f:
            tail = f.read().splitlines()[-15:]
        print('--- log tail ---\n' + '\n'.join(tail))
        if any('INCOMPLETE' in line for line in tail):
            flag = '' if exp['name'] == 'main' else f' --experiment {exp["name"]}'
            print(f'\nStopped at the time budget: re-run with{flag} --resume --watch to continue.')
    return out_dir


def main() -> None:
    sys.stdout.reconfigure(
        encoding='utf-8'
    )  # comparison.md has ≈ and ±; Windows consoles are cp1252
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--experiment', choices=sorted(EXPERIMENTS), default='main')
    ap.add_argument('--resume', action='store_true', help="attach the kernel's previous output")
    ap.add_argument('--watch', action='store_true', help='poll until the run finishes')
    ap.add_argument('--status', action='store_true', help='report status only, do not push')
    ap.add_argument('--poll-minutes', type=float, default=10)
    ap.add_argument(
        '--budget-hours',
        type=float,
        default=None,
        help='stop training cleanly after this many hours, e.g. when the weekly GPU quota has '
        'less than a full session left',
    )
    weights = ap.add_mutually_exclusive_group()
    weights.add_argument('--with-export', action='store_true', help='also download the app model')
    weights.add_argument('--with-checkpoints', action='store_true', help='download every file')
    a = ap.parse_args()

    exp, user = experiment(a.experiment), username()
    kernel_id = f'{user}/{exp["slug"]}'
    pattern = None if a.with_checkpoints else NO_CHECKPOINTS if a.with_export else NO_WEIGHTS
    if a.status:
        state, text = status(kernel_id)
        print(text)
        if state in STOPPED:
            fetch_output(exp, user, pattern)
        return

    push(exp, user, a.resume, a.budget_hours)
    print(f'https://www.kaggle.com/code/{kernel_id}')
    if not a.watch:
        flag = '' if exp['name'] == 'main' else f' --experiment {exp["name"]}'
        print(f'Check later:  python scripts/run_on_kaggle.py{flag} --status')
        return
    while True:
        time.sleep(a.poll_minutes * 60)
        state, text = status(kernel_id)
        print(f'{time.strftime("%H:%M")} {state}')
        if state in STOPPED:
            print(text)
            fetch_output(exp, user, pattern)
            return


if __name__ == '__main__':
    main()
