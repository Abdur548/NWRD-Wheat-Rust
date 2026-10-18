"""Aggregates the finished runs of one experiment into its comparison report.

Outputs in <out_dir>/: comparison.md, comparison.json and figures/training_curves.png. Every
number is computed from files the pipeline saves per run (metrics.json, history.json,
test_counts.npz), so a report can be rebuilt without checkpoints or images.
"""

import glob
import json
import os

import numpy as np

from .metrics import dice, iou
from .models import MODELS

METRICS = ('IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'Accuracy')

# Published NWRD paper results (Anwar et al., Sensors 2023, doi:10.3390/s23156942), UNet + APF.
# Evaluated on 128px patches of 10 held-out full images; their IoU and F1 are not mutually
# consistent (likely per-image averaging), so treat them as approximate reference points.
PAPER = [
    {
        'table': 'Table 3',
        'test_set': 'full test set',
        'split': 'test@thr',  # our matching test set
        'scores': {'Precision': 0.506, 'Recall': 0.624, 'Dice': 0.557},
    },
    {
        'table': 'Table 4',
        'test_set': 'background-only patches removed',
        'split': 'test_rust_patches@thr',
        'scores': {'Precision': 0.593, 'Recall': 0.552, 'Dice': 0.564, 'IoU': 0.438},
    },
]
OURS = 'canet_b4'

VERDICTS = {
    'improves': 'improves on',
    'comparable': 'is comparable to',
    'worse': 'is worse than',
}


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_runs(out_dir: str) -> dict:
    """{model: {seed: run}} for every run with a metrics.json. A run holds its directory and
    metrics, plus 'history' (the whole history.json), 'sweep' (val threshold sweep) and
    'files' / 'counts' / 'counts05' (per-patch test confusion counts) where those files exist."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*', 'seed*', 'metrics.json'))):
        run_dir = os.path.dirname(path)
        model = os.path.basename(os.path.dirname(run_dir))
        seed = int(os.path.basename(run_dir)[len('seed') :])
        run = {'dir': run_dir, 'metrics': _read_json(path)}
        for key, name in (('history', 'history.json'), ('sweep', 'val_threshold_sweep.json')):
            value = _read_json(os.path.join(run_dir, name))
            if value is not None:
                run[key] = value
        counts_path = os.path.join(run_dir, 'test_counts.npz')
        if os.path.exists(counts_path):
            with np.load(counts_path) as c:
                run['files'] = c['files']
                run['counts'] = c['at_thr'].astype(np.int64)
                run['counts05'] = c['at_05'].astype(np.int64)
        runs.setdefault(model, {})[seed] = run
    return runs


def source_photo(filename) -> str:
    """NWRD patches are named <photo>_patch_<n>_x<col>_y<row>.png (tiles add _t<rc>)."""
    return str(filename).split('_patch_')[0]


def paired_bootstrap(
    ours: dict,
    base: dict,
    seeds: list,
    n_boot: int = 5000,
    rng_seed: int = 0,
    by_photo: bool = False,
) -> dict:
    """Paired bootstrap of the seed-averaged difference in pooled test IoU/Dice.

    Each resample draws test patches with replacement, or with by_photo whole source
    photographs with all their patches. Patches of one photograph are correlated (same plant,
    light and annotator), so by_photo is the honest interval; the patch-level one is narrower
    than the evidence supports. The same draws are used for both models and every seed."""
    for s in seeds:
        if not np.array_equal(ours[s]['files'], base[s]['files']):
            raise ValueError(f'seed {s}: the two models were tested on different files')
    files = ours[seeds[0]]['files']
    n = len(files)
    rng = np.random.default_rng(rng_seed)
    if by_photo:
        photos, inv = np.unique([source_photo(f) for f in files], return_inverse=True)
        draws = rng.integers(0, len(photos), size=(n_boot, len(photos)))
        w = np.stack([np.bincount(d, minlength=len(photos)) for d in draws])[:, inv]
        units = len(photos)
    else:
        w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
        units = n
    out = {}
    for name, fn in (('IoU', iou), ('Dice', dice)):
        d = np.mean([fn(w @ ours[s]['counts']) - fn(w @ base[s]['counts']) for s in seeds], 0)
        point = np.mean(
            [fn(ours[s]['counts'].sum(0)) - fn(base[s]['counts'].sum(0)) for s in seeds]
        )
        out[name] = {
            'delta': float(point),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'p_two_sided': float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            'units': int(units),
        }
    return out


def seed_ttest(ours: dict, base: dict, seeds: list) -> dict | None:
    """Paired t-test over seeds of test IoU; it captures training-run variance."""
    if len(seeds) < 2:
        return None
    from scipy import stats

    a = [ours[s]['metrics']['test@thr']['IoU'] for s in seeds]
    b = [base[s]['metrics']['test@thr']['IoU'] for s in seeds]
    t, p = stats.ttest_rel(a, b)
    return {'t': float(t), 'p': float(p), 'n_seeds': len(seeds)}


def seed_wins(ours: dict, base: dict, seeds: list) -> int:
    """Number of seeds on which `ours` has the higher test IoU."""
    return sum(
        ours[s]['metrics']['test@thr']['IoU'] > base[s]['metrics']['test@thr']['IoU'] for s in seeds
    )


def verdict(boot_photo_iou: dict, wins: int) -> str:
    """The decision rule, fixed before the results were seen (docs/protocol.md).

    'improves': higher mean test IoU, the 95% CI of the IoU difference that resamples test
    photographs excludes 0, and CANet wins on at least 2 of 3 seeds. 'worse': that CI lies
    entirely below 0. 'comparable': otherwise. The rule assumes all three seeds; with fewer,
    the reading is interim."""
    lo, hi = boot_photo_iou['ci95']
    if boot_photo_iou['delta'] > 0 and lo > 0 and wins >= 2:
        return 'improves'
    if hi < 0:
        return 'worse'
    return 'comparable'


def significance(runs: dict) -> dict:
    """CANet-B4 against every other model, on the seeds both have."""
    tests = {}
    if OURS not in runs:
        return tests
    for base in (m for m in runs if m != OURS):
        seeds = sorted(set(runs[OURS]) & set(runs[base]))
        if not seeds:
            continue
        photo = paired_bootstrap(runs[OURS], runs[base], seeds, by_photo=True)
        wins = seed_wins(runs[OURS], runs[base], seeds)
        tests[base] = {
            'seeds': seeds,
            'bootstrap': paired_bootstrap(runs[OURS], runs[base], seeds),
            'bootstrap_photo': photo,
            'seed_ttest': seed_ttest(runs[OURS], runs[base], seeds),
            'canet_wins': wins,
            'verdict': verdict(photo['IoU'], wins),
        }
    return tests


def mean_std(vals) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    vals = np.asarray(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def _fmt(m: float, s: float, n: int) -> str:
    return f'{m:.4f} ± {s:.4f}' if n > 1 else f'{m:.4f}'


def summarise(runs: dict) -> dict:
    summary = {}
    for model, seeds in runs.items():
        ms = [seeds[s]['metrics'] for s in sorted(seeds)]
        hs = [seeds[s]['history']['history'] for s in sorted(seeds)]
        row = {'seeds': sorted(seeds), 'n': len(ms)}
        for split in ('test@thr', 'test@0.5', 'val@thr', 'test_rust_patches@thr'):
            row[split] = {k: mean_std([m[split][k] for m in ms]) for k in METRICS}
        row['threshold'] = mean_std([m['threshold'] for m in ms])
        row['threshold_at_grid_edge'] = any(m['threshold_at_grid_edge'] for m in ms)
        row['best_epoch'] = [m['best_epoch'] for m in ms]
        row['train_min_per_epoch'] = mean_std([np.mean(h['epoch_seconds']) / 60 for h in hs])
        summary[model] = row
    return summary


def write_markdown(path: str, summary: dict, tests: dict, bench, test_info: dict, tiling=None):
    order = [m for m in (OURS,) if m in summary] + sorted(m for m in summary if m != OURS)
    L = [
        '# CANet-B4 vs baselines: comparison',
        '',
        f'Test split: {test_info["test_images"]} patches, {test_info["test_rust_images"]} with '
        f'rust, {100 * test_info["test_rust_pixel_fraction"]:.1f}% rust pixels. Threshold chosen '
        'per run on val (IoU-maximising, grid 0.05–0.95), then frozen for test. Metrics pooled '
        'over all test pixels; mean ± sample std over seeds.',
        '',
        '## Accuracy (test split, val-selected threshold)',
        '',
        '| Model | Seeds | ' + ' | '.join(METRICS) + ' | IoU @0.5 | Threshold |',
        '|---|---|' + '---|' * (len(METRICS) + 2),
    ]
    for m in order:
        r = summary[m]
        cells = [_fmt(*r['test@thr'][k], r['n']) for k in METRICS]
        thr = f'{r["threshold"][0]:.2f}' + (' ⚠ edge' if r['threshold_at_grid_edge'] else '')
        name = f'**{m}** (ours)' if m == OURS else m
        L.append(
            f'| {name} | {r["n"]} | '
            + ' | '.join(cells)
            + f' | {_fmt(*r["test@0.5"]["IoU"], r["n"])} | {thr} |'
        )
    L += [
        '',
        '## Against the NWRD paper (UNet + APF)',
        '',
        '| Result | Precision | Recall | F1 / Dice | IoU |',
        '|---|---|---|---|---|',
    ]
    for p in PAPER:
        s = p['scores']
        L.append(
            f'| Paper, {p["table"]}, {p["test_set"]} | '
            + ' | '.join(
                f'{s[k]:.3f}' if k in s else '–' for k in ('Precision', 'Recall', 'Dice', 'IoU')
            )
            + ' |'
        )
    for m in order:
        r = summary[m]
        for split, label in (
            ('test@thr', 'all test patches'),
            ('test_rust_patches@thr', 'rust-positive test patches (≈ Table 4)'),
        ):
            L.append(
                f'| {m}, re-run: {label} | '
                + ' | '.join(
                    f'{r[split][k][0]:.3f}' for k in ('Precision', 'Recall', 'Dice', 'IoU')
                )
                + ' |'
            )
    if tiling:
        L += [
            '',
            f'Every model here sees {tiling["tile"]}px tiles re-cut at {tiling["scale"]:g}× '
            "scale, the paper's own patch size and field of view (nwrd/tiles.py). The paper's "
            'test photographs, training (APF, 500 epochs, focal loss) and patch sampling still '
            'differ, so its rows are reference points, not a paired comparison. The `unet` row is '
            "the paper's architecture re-trained under this protocol, so CANet vs `unet` is the "
            'like-for-like comparison.',
        ]
    else:
        L += [
            '',
            'Paper figures are as published, on a different test unit (full-image patches, '
            "128px). The `unet` row is the paper's architecture re-trained under this protocol, so "
            'CANet vs `unet` is the like-for-like comparison.',
        ]
    if tests:

        def ci(b):
            return f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}] (p {b["p_two_sided"]:.4f})'

        L += [
            '',
            '## Is the difference real? (CANet-B4 minus baseline, test split)',
            '',
            '| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | '
            '95% CI, resampling patches | ΔDice | ΔDice 95% CI, photographs | '
            'seed-level t-test p | CANet higher (seeds) | Verdict |',
            '|---|---|---|---|---|---|---|---|---|---|',
        ]
        for base, t in tests.items():
            bp, tt = t['bootstrap_photo'], t['seed_ttest']
            n = len(t['seeds'])
            reading = f'CANet-B4 {VERDICTS[t["verdict"]]} {base}'
            if n < 3:
                reading += f' (interim: {n} of 3 seeds)'
            p_seeds = f'{tt["p"]:.4f}' if tt else 'n/a (1 seed)'
            L.append(
                f'| {base} | {n} | {bp["IoU"]["delta"]:+.4f} | {ci(bp["IoU"])} | '
                f'{ci(t["bootstrap"]["IoU"])} | {bp["Dice"]["delta"]:+.4f} | {ci(bp["Dice"])} | '
                f'{p_seeds} | {t["canet_wins"]} of {n} | {reading} |'
            )
        unets = [m for m in ('unet', 'unet_tuned') if m in summary]
        if len(unets) == 2:
            best = max(unets, key=lambda m: summary[m]['val@thr']['IoU'][0])
            L += [
                '',
                "Two runs of the paper's UNet exist (shared lr and `unet_tuned`). The stronger on "
                f'**validation** IoU, `{best}`, is the paper-architecture baseline; choosing by '
                'test IoU would bias the comparison.',
            ]
        L += [
            '',
            'Verdict: the decision rule fixed before the results were seen. CANet-B4 *improves on* '
            'a baseline if its mean test IoU is higher, the photograph-resampling 95% CI of ΔIoU '
            'excludes 0, and it is higher on at least 2 of 3 seeds; it *is worse* if that CI lies '
            'below 0; otherwise the two are *comparable*.',
            '',
            'A CI that excludes 0 means the gap is larger than test-set sampling noise. Use the '
            'photograph interval: the test patches are cut from a few photographs, and patches of '
            'one photograph are not independent, so resampling patches understates the '
            'uncertainty. Neither bootstrap captures training-run variance; the seed-level t-test '
            'does, but has little power with 3 seeds.',
        ]
    if bench:
        env, rows = bench['env'], {r['model']: r for r in bench['results']}
        first = next(iter(rows.values()))
        gpu_keys = [k for k in first if k.startswith('gpu_')]
        L += [
            '',
            f'## Efficiency ({env["device"]}, {first["input"]} input, torch {env["torch"]})',
            '',
            '| Model | Params (M) | GFLOPs | '
            + ' | '.join(k[4:] for k in gpu_keys)
            + ' | CPU latency bs1 (ms) | Train min/epoch |',
            '|---|---|---|' + '---|' * (len(gpu_keys) + 2),
        ]
        for m in [x for x in dict.fromkeys(order + list(rows)) if x in rows]:
            r = rows[m]
            tr = summary.get(m, {}).get('train_min_per_epoch')
            L.append(
                f'| {m} | {r["params_M"]:.2f} | {r["GFLOPs"]:.1f} | '
                + ' | '.join(f'{r[k]:.1f}' for k in gpu_keys)
                + f' | {r["cpu_latency_ms_bs1_fp32"]:.0f} | {f"{tr[0]:.1f}" if tr else "–"} |'
            )
        if 'canet_b4_twopass' in rows:
            L += [
                '',
                '`canet_b4_twopass` is CANet as first implemented, running its encoder twice per '
                'forward pass; its outputs are bit-identical to `canet_b4`, so only the cost '
                'differs. It was measured in this run and is no longer part of the code.',
            ]
    L += ['', '## Models', ''] + [f'- `{m}` — {MODELS.get(m, m)}' for m in order]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def plot_figures(runs: dict, fig_dir: str) -> None:
    """Validation loss and IoU per epoch: mean over seeds, band from min to max."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for model, seeds in sorted(runs.items()):
        for key, ax in (('val_loss', axes[0]), ('val_iou', axes[1])):
            curves = [seeds[s]['history']['history'][key] for s in sorted(seeds)]
            n = min(map(len, curves))
            arr = np.array([c[:n] for c in curves])
            x = np.arange(1, n + 1)
            (line,) = ax.plot(x, arr.mean(0), label=model, lw=2 if model == OURS else 1.4)
            if len(curves) > 1:
                ax.fill_between(x, arr.min(0), arr.max(0), color=line.get_color(), alpha=0.15)
    axes[0].set_title('Validation loss')
    axes[1].set_title('Validation IoU (pooled, threshold 0.5)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150)
    plt.close(fig)


def compare(out_dir: str, log=print) -> dict | None:
    """Writes comparison.md, comparison.json and the training-curve figure for `out_dir`."""
    runs = load_runs(out_dir)
    if not runs:
        log('[compare] no finished runs yet')
        return None
    incomplete = [
        r['dir']
        for seeds in runs.values()
        for r in seeds.values()
        if not {'history', 'counts'} <= set(r)
    ]
    if incomplete:
        raise FileNotFoundError(f'history.json or test_counts.npz missing in {incomplete}')
    summary = summarise(runs)
    tests = significance(runs)
    bench = _read_json(os.path.join(out_dir, 'benchmark.json'))
    tiling = _read_json(os.path.join(out_dir, 'tiling.json'))
    any_run = next(iter(next(iter(runs.values())).values()))['metrics']
    plot_figures(runs, os.path.join(out_dir, 'figures'))
    md_path = os.path.join(out_dir, 'comparison.md')
    write_markdown(md_path, summary, tests, bench, any_run, tiling)
    with open(os.path.join(out_dir, 'comparison.json'), 'w') as f:
        json.dump({'summary': summary, 'significance': tests, 'benchmark': bench}, f, indent=2)
    with open(md_path, encoding='utf-8') as f:
        log(f.read())
    return summary
