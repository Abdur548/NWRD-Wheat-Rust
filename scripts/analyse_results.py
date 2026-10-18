"""Deeper analysis of one experiment's results, beyond its comparison.md.

    python scripts/analyse_results.py results/512px/final --replicate results/512px/replicate
    python scripts/analyse_results.py results/128px/final

Writes analysis.json and figures to results/analysis/<scale>/ (or --out). Everything is computed
from files the pipeline saves per run (metrics.json, history.json, test_counts.npz,
val_threshold_sweep.json) plus benchmark.json and log.txt, so it needs no GPU, checkpoints or
images. It works with any number of seeds; per-seed values are averaged.

What it adds to comparison.md:
- scores per test photograph, and averaged per photograph as the NWRD paper likely did;
- scores by lesion size, and false alarms on rust-free patches;
- the photograph- and patch-level bootstraps side by side, with photographs won per model;
- training dynamics, threshold sensitivity, and accuracy against measured cost;
- with --replicate, the same seeds from another session, which measures run-to-run noise.
"""

import argparse
import json
import os
import re

import numpy as np

from nwrd.compare import OURS, load_runs, paired_bootstrap, source_photo
from nwrd.metrics import scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ['canet_b4', 'unet', 'deeplabv3plus_r50']  # the comparison proper
ALL = MAIN + ['unet_tuned']  # plus the learning-rate control
NAMES = {
    'canet_b4': 'CANet-B4 (ours)',
    'unet': 'UNet (paper architecture)',
    'deeplabv3plus_r50': 'DeepLabV3+ R50',
    'unet_tuned': 'UNet, lr 1e-3 (control)',
}
# Categorical slots in fixed order (validated for colour-vision deficiency), one per model.
COLORS = {
    'canet_b4': '#2a78d6',
    'unet': '#eb6834',
    'deeplabv3plus_r50': '#1baf7a',
    'unet_tuned': '#eda100',
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7',
)  # fmt: skip
BINS = [
    ('rust-free', 0.0, 0.0),
    ('< 1%', 0.0, 0.01),
    ('1–5%', 0.01, 0.05),
    ('5–20%', 0.05, 0.20),
    ('> 20%', 0.20, 1.01),
]
FALSE_ALARM_SHARE = 0.001  # 0.1% of a patch predicted rust: a visible blob, not a stray pixel
N_BOOT = 10_000
LOG_LINE = re.compile(
    r'\[(\S+) seed(\d+)\] thr ([\d.]+).*test IoU ([\d.]+) Dice ([\d.]+) P ([\d.]+) R ([\d.]+)'
)


def log_results(run_root: str) -> dict:
    """Per-run test results as logged; they survive even where run files were lost."""
    out, path = {}, os.path.join(run_root, 'log.txt')
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = LOG_LINE.search(line)
            if m:
                keys = ('threshold', 'IoU', 'Dice', 'Precision', 'Recall')
                out[(m[1], int(m[2]))] = dict(zip(keys, map(float, m.groups()[2:]), strict=True))
    return out


def count_scores(c) -> dict:
    """Pooled scores of summed confusion counts, plus the false-positive rate."""
    s = scores(*c)
    return {
        'IoU': s['IoU'],
        'Dice': s['Dice'],
        'Precision': s['Precision'],
        'Recall': s['Recall'],
        'FP rate': 1 - s['Specificity'],
    }


def mean_over_seeds(model_runs: dict, fn):
    vals = [fn(r) for r in model_runs.values()]
    if isinstance(vals[0], dict):
        return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return float(np.mean(vals))


def _photo_key(photo: str):
    return (0, int(photo)) if photo.isdigit() else (1, photo)


def split_check(run_root: str, test_files) -> dict | None:
    """Which source photographs each split was cut from, and whether any is in two splits.
    Train and val file names come from the pipeline's mask-statistics cache."""
    ids = {'test': sorted({source_photo(f) for f in test_files}, key=_photo_key)}
    for split in ('train', 'val'):
        path = os.path.join(run_root, '_cache', f'mask_stats_{split}.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            ids[split] = sorted({source_photo(name) for name in json.load(f)}, key=_photo_key)
    return {
        'source_images': {s: len(v) for s, v in ids.items()},
        'id_range': {s: [v[0], v[-1]] for s, v in ids.items()},
        'shared': {
            f'{a}/{b}': sorted(set(ids[a]) & set(ids[b]), key=_photo_key)
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))
        },
    }


def analyse(runs: dict, bench, log_now: dict, log_rep: dict) -> dict:
    models = [m for m in ALL if m in runs]
    any_run = next(iter(runs[OURS].values()))
    files = any_run['files']
    src = np.array([source_photo(f) for f in files])
    images = sorted(set(src), key=_photo_key)
    truth_px = any_run['counts'][:, 0] + any_run['counts'][:, 2]  # rust pixels per patch
    patch_px = any_run['counts'].sum(1)
    frac = truth_px / patch_px
    false_alarm_px = int(round(FALSE_ALARM_SHARE * patch_px[0]))
    A = {
        'test_patches': int(len(files)),
        'test_source_images': images,
        'patches_per_image': int(round(len(files) / len(images))),
        'patch_pixels': int(patch_px[0]),
        'false_alarm_pixels': false_alarm_px,
        'seeds': {m: sorted(runs[m]) for m in models},
    }

    # Headline, averaged over seeds.
    A['headline'] = {
        m: {
            'val_IoU': mean_over_seeds(runs[m], lambda r: r['metrics']['val@thr']['IoU']),
            'test': mean_over_seeds(runs[m], lambda r: r['metrics']['test@thr']),
            'test_IoU@0.5': mean_over_seeds(runs[m], lambda r: r['metrics']['test@0.5']['IoU']),
            'test_rust_patches_IoU': mean_over_seeds(
                runs[m], lambda r: r['metrics']['test_rust_patches@thr']['IoU']
            ),
            'threshold': mean_over_seeds(runs[m], lambda r: r['metrics']['threshold']),
            'best_epoch': [runs[m][s]['metrics']['best_epoch'] for s in sorted(runs[m])],
        }
        for m in models
    }

    # Per source photograph: pooled within each photograph, then averaged over seeds.
    per_image = {
        m: {
            img: mean_over_seeds(
                runs[m], lambda r, img=img: count_scores(r['counts'][src == img].sum(0))
            )
            for img in images
        }
        for m in models
    }
    A['per_image'] = {
        m: {img: {'IoU': v['IoU'], 'Dice': v['Dice']} for img, v in per_image[m].items()}
        for m in models
    }
    A['image_rust_fraction'] = {
        img: float(truth_px[src == img].sum() / patch_px[src == img].sum()) for img in images
    }
    # Averaged per photograph rather than pooled: likely how the NWRD paper averaged.
    A['per_image_mean'] = {
        m: {
            k: float(np.mean([per_image[m][i][k] for i in images]))
            for k in ('IoU', 'Dice', 'Precision', 'Recall')
        }
        for m in models
    }

    # Is CANet's difference real? Photograph-level vs patch-level resampling.
    A['significance'] = {}
    for base in (m for m in models if m != OURS):
        seeds = sorted(set(runs[OURS]) & set(runs[base]))
        wins = [per_image[OURS][i]['IoU'] > per_image[base][i]['IoU'] for i in images]
        A['significance'][base] = {
            'paired_seeds': seeds,
            'image_bootstrap': paired_bootstrap(
                runs[OURS], runs[base], seeds, n_boot=N_BOOT, by_photo=True
            )['IoU'],
            'patch_bootstrap': paired_bootstrap(runs[OURS], runs[base], seeds, n_boot=N_BOOT)[
                'IoU'
            ],
            'images_won_by_canet': int(sum(wins)),
            'images': len(images),
        }

    # By lesion size, pooled over the patches in each bin.
    A['lesion_bins'] = {}
    for name, lo, hi in BINS:
        idx = (frac == 0) if hi == 0 else (frac > lo) & (frac <= hi)
        row = {
            'patches': int(idx.sum()),
            'rust_pixel_share': float(truth_px[idx].sum() / truth_px.sum()),
        }
        for m in models:
            row[m] = mean_over_seeds(
                runs[m], lambda r, idx=idx: count_scores(r['counts'][idx].sum(0))
            )
            if hi == 0:
                row[m]['false_alarm_patches'] = mean_over_seeds(
                    runs[m],
                    lambda r, idx=idx: float((r['counts'][idx, 1] >= false_alarm_px).mean()),
                )
        A['lesion_bins'][name] = row

    # Training dynamics and threshold sensitivity.
    A['training'] = {}
    for m in models:
        rows = []
        for s in sorted(runs[m]):
            h = runs[m][s].get('history', {}).get('history')
            if not h:
                continue
            v, lr = np.array(h['val_iou']), np.array(h['lr'])
            rows.append(
                {
                    'seed': s,
                    'epochs': len(v),
                    'best_epoch': int(v.argmax()) + 1,
                    'best_val_IoU@0.5': float(v.max()),
                    'final_val_IoU@0.5': float(v[-1]),
                    'collapsed_epochs': int((v < 0.05).sum()),
                    'lr_restarts_after_epoch': [int(i) for i in np.where(np.diff(lr) > 0)[0] + 1],
                    'min_per_epoch': float(np.mean(h['epoch_seconds']) / 60),
                }
            )
        A['training'][m] = rows
    A['threshold_sweep'] = {}
    for m in models:
        for s in sorted(runs[m]):
            sw = runs[m][s].get('sweep')
            if not sw:
                continue
            best = max(sw, key=lambda r: r['IoU'])
            near = [r['IoU'] for r in sw if abs(r['threshold'] - best['threshold']) <= 0.10 + 1e-9]
            A['threshold_sweep'].setdefault(m, []).append(
                {
                    'seed': s,
                    'best_threshold': best['threshold'],
                    'best_val_IoU': best['IoU'],
                    'val_IoU_at_0.5': next(
                        (r['IoU'] for r in sw if abs(r['threshold'] - 0.5) < 1e-9), None
                    ),
                    'min_val_IoU_within_0.10': float(min(near)),
                }
            )

    # The same seeds in another session: the spread run-to-run noise alone produces.
    if log_rep:
        common = sorted(set(log_now) & set(log_rep))
        A['replicate'] = {
            f'{m} seed{s}': {
                'this_session': log_now[(m, s)]['IoU'],
                'other_session': log_rep[(m, s)]['IoU'],
                'abs_diff': abs(log_now[(m, s)]['IoU'] - log_rep[(m, s)]['IoU']),
            }
            for m, s in common
        }
        A['replicate_extra_runs'] = {
            f'{m} seed{s}': v for (m, s), v in log_rep.items() if (m, s) not in log_now
        }

    if bench:
        A['efficiency'] = {'env': bench['env'], 'rows': {r['model']: r for r in bench['results']}}
    return A


def style(ax, grid_axis: str = 'y') -> None:
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend(ax_or_fig, handles, **kw) -> None:
    ax_or_fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, **kw)


def _offsets(n: int, step: float) -> np.ndarray:
    """Offsets that centre n grouped marks or bars on their tick."""
    return (np.arange(n) - (n - 1) / 2) * step


def _padded_range(values, pad: float, lo_bound: float = 0.0, hi_bound: float = 1.0):
    lo, hi = min(values), max(values)
    return max(lo_bound, lo - pad), min(hi_bound, hi + pad)


def figures(A: dict, runs: dict, out: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter

    plt.rcParams.update(
        {
            'font.family': ['Segoe UI', 'DejaVu Sans'],
            'figure.facecolor': SURFACE,
            'savefig.facecolor': SURFACE,
            'text.color': INK,
            'axes.labelcolor': INK2,
            'axes.titlesize': 10,
            'axes.titlecolor': INK,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 9,
        }
    )
    main = [m for m in MAIN if m in A['headline']]
    models = [m for m in ALL if m in runs]

    def dot(m):
        return Line2D(
            [], [], marker='o', ls='', ms=7, color=COLORS[m], mec=SURFACE, mew=1.5, label=NAMES[m]
        )

    def line(m):
        return Line2D([], [], color=COLORS[m], lw=1.6, label=NAMES[m])

    # 1. Per source photograph: dot plot, photographs ordered by rust share.
    imgs = sorted(A['test_source_images'], key=lambda i: A['image_rust_fraction'][i])
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(imgs) + 0.8))
    style(ax, 'x')
    offsets = _offsets(len(main), 0.22)
    for k, m in enumerate(main):
        y = np.arange(len(imgs)) + offsets[k]
        ax.plot(
            [A['per_image'][m][i]['IoU'] for i in imgs],
            y, 'o', ms=7, color=COLORS[m], mec=SURFACE, mew=1.5, zorder=3,
        )  # fmt: skip
    ax.set_yticks(
        np.arange(len(imgs)),
        [f'Image {i}  ·  {100 * A["image_rust_fraction"][i]:.1f}% rust' for i in imgs],
    )
    ax.set_xlabel(f'Test IoU within the image (pooled over its {A["patches_per_image"]} patches)')
    ax.set_xlim(*_padded_range([A['per_image'][m][i]['IoU'] for m in main for i in imgs], 0.03))
    ax.set_title('Per test image: IoU of each model', loc='left', pad=24)
    legend(ax, [dot(m) for m in main], loc='lower left', bbox_to_anchor=(-0.01, 0.99), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'per_image_iou.png'), dpi=200)
    plt.close(fig)

    # 2. By lesion size (rust-positive bins) and false alarms on rust-free patches.
    bins = [b for b, *_ in BINS if b != 'rust-free']
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), gridspec_kw={'width_ratios': [3, 3, 3, 1.6]})
    width = 0.24
    offsets = _offsets(len(main), width + 0.02)  # the 0.02 is the surface gap between bars
    for ax, metric in zip(axes[:3], ('IoU', 'Recall', 'Precision'), strict=True):
        style(ax)
        for k, m in enumerate(main):
            x = np.arange(len(bins)) + offsets[k]
            ax.bar(x, [A['lesion_bins'][b][m][metric] for b in bins], width, color=COLORS[m])
        ax.set_xticks(np.arange(len(bins)), bins)
        ax.set_ylim(0, 1)
        ax.set_title(metric, loc='left')
        ax.set_xlabel('Rust share of the patch')
    ax = axes[3]
    style(ax)
    fa = [100 * A['lesion_bins']['rust-free'][m]['false_alarm_patches'] for m in main]
    ax.bar(np.arange(len(main)), fa, 0.6, color=[COLORS[m] for m in main])
    for x, v in enumerate(fa):
        ax.text(x, v, f'{v:.0f}%', ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks([])
    ax.set_ylim(0, max(fa) * 1.25 + 1)
    ax.set_title('False alarms', loc='left')
    ax.set_xlabel(f'Rust-free patches with\n≥ {A["false_alarm_pixels"]} false-positive pixels')
    n = A['lesion_bins']
    fig.suptitle(
        'Accuracy by lesion size  (patches per bin: '
        + ', '.join(f'{b} {n[b]["patches"]}' for b, *_ in BINS)
        + ')',
        x=0.01, ha='left', fontsize=10, fontweight='semibold', color=INK,
    )  # fmt: skip
    legend(
        fig,
        [Patch(color=COLORS[m], label=NAMES[m]) for m in main],
        loc='lower center', ncol=len(main), bbox_to_anchor=(0.5, -0.02),
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(out, 'lesion_size.png'), dpi=200)
    plt.close(fig)

    # 3. Validation IoU per epoch (first seed of each model), with the lr warm restarts.
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    style(ax)
    restarts, curves = set(), {}
    for m in models:
        seeds = [s for s in sorted(runs[m]) if 'history' in runs[m][s]]
        if not seeds:
            continue
        h = runs[m][seeds[0]]['history']['history']
        curves[m] = h['val_iou']
        restarts |= set(np.where(np.diff(h['lr']) > 0)[0] + 1)
    healthy = [v for c in curves.values() for v in c if v >= 0.05]
    lo, hi = _padded_range(healthy, 0.03)
    epochs = max(len(c) for c in curves.values())
    for m, v in curves.items():
        ax.plot(np.arange(1, len(v) + 1), v, color=COLORS[m], lw=1.6, solid_capstyle='round')
        b = int(np.argmax(v))
        ax.plot(b + 1, v[b], 'o', ms=7, color=COLORS[m], mec=SURFACE, mew=1.5, zorder=3)
    for r in sorted(restarts):
        ax.axvline(r + 0.5, color=AXIS, lw=0.8, zorder=0)
        ax.text(r + 0.7, lo + 0.005, 'lr restart', fontsize=7.5, color=MUTED, rotation=90,
                va='bottom')  # fmt: skip
    for m, v in curves.items():
        if any(x < 0.05 for x in v):
            first = next(i for i, x in enumerate(v) if x < 0.05) + 1
            ax.annotate(
                f'{NAMES[m].split(" (")[0]}: IoU 0 from epoch {first}\n(predicts no rust at all)',
                (first, lo), xytext=(first + 0.4, lo + 0.06), fontsize=7.5, color=INK2,
                arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 0.8},
            )  # fmt: skip
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation IoU at threshold 0.5')
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.5, epochs + 0.5)
    ax.set_title('Validation IoU per epoch (dot = checkpoint kept)', loc='left', pad=24)
    legend(ax, [line(m) for m in curves], loc='lower left', bbox_to_anchor=(-0.01, 0.99), ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'training_dynamics.png'), dpi=200)
    plt.close(fig)

    # 4. Threshold sensitivity on validation.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    drawn = []
    for m in reversed(models):  # ours drawn last, on top
        sw = runs[m][min(runs[m])].get('sweep')
        if not sw:
            continue
        t, v = [r['threshold'] for r in sw], [r['IoU'] for r in sw]
        ax.plot(t, v, color=COLORS[m], lw=1.6)
        b = int(np.argmax(v))
        ax.plot(t[b], v[b], 'o', ms=7, color=COLORS[m], mec=SURFACE, mew=1.5, zorder=3)
        drawn.append(m)
    ax.axvline(0.5, color=AXIS, lw=0.8, zorder=0)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Validation IoU')
    ax.set_title('Threshold sweep on validation (dot = threshold frozen for test)', loc='left',
                 pad=24)  # fmt: skip
    legend(ax, [line(m) for m in reversed(drawn)], loc='lower left', bbox_to_anchor=(-0.01, 0.99),
           ncol=4)  # fmt: skip
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'threshold_sweep.png'), dpi=200)
    plt.close(fig)

    # 5. Accuracy against cost: FLOPs and measured T4 latency, side by side (one axis each).
    if 'efficiency' in A:
        rows = A['efficiency']['rows']
        size = rows[main[0]]['input'].replace('x', '×')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, key, label in (
            (axes[0], 'GFLOPs', f'GFLOPs per {size} input (log scale)'),
            (axes[1], 'gpu_latency_ms_bs1_fp16', 'T4 latency, batch 1, fp16 (ms)'),
        ):
            style(ax, 'both')
            for m in main:
                x, y = rows[m][key], A['headline'][m]['test']['IoU']
                ax.plot(x, y, 'o', ms=8, color=COLORS[m], mec=SURFACE, mew=1.5, zorder=3)
                ax.annotate(NAMES[m].split(' (')[0], (x, y), textcoords='offset points',
                            xytext=(7, 4), fontsize=8, color=INK2)  # fmt: skip
            ax.set_xlabel(label)
        axes[0].set_xscale('log')
        axes[0].xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))  # 1-2-5 ticks
        axes[0].xaxis.set_major_formatter(ScalarFormatter())
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].set_ylabel('Test IoU')
        lo = min(A['headline'][m]['test']['IoU'] for m in main)
        hi = max(A['headline'][m]['test']['IoU'] for m in main)
        axes[0].set_ylim(lo - 0.03, hi + 0.03)
        axes[0].set_title('Accuracy vs compute', loc='left')
        axes[1].set_title('Accuracy vs measured latency', loc='left')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'accuracy_vs_cost.png'), dpi=200)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('runs', help="an experiment's results folder (a runs/ directory)")
    ap.add_argument('--replicate', help='results of another session with the same seeds')
    ap.add_argument('--out', help='output folder (default: results/analysis/<scale>)')
    a = ap.parse_args()

    runs = {m: v for m, v in load_runs(a.runs).items() if m in ALL}
    runs = {m: {s: r for s, r in v.items() if 'counts' in r} for m, v in runs.items()}
    bench_path = os.path.join(a.runs, 'benchmark.json')
    bench = None
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench = json.load(f)
    A = analyse(runs, bench, log_results(a.runs), log_results(a.replicate) if a.replicate else {})
    A['splits'] = split_check(a.runs, next(iter(runs[OURS].values()))['files'])
    scale = '128px' if os.path.exists(os.path.join(a.runs, 'tiling.json')) else '512px'
    out = a.out or os.path.join(ROOT, 'results', 'analysis', scale)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'analysis.json'), 'w') as f:
        json.dump(A, f, indent=1)
    figures(A, runs, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
