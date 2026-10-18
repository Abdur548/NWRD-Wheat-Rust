"""Packages the model the app serves: the CANet-B4 seed with the best *validation* IoU.

Writes <out_dir>/export/: best_model.pth, results.json (test metrics and training history),
confusion_matrix.png, and copies of the experiment's figures. Selection never looks at test
metrics. Copy the folder's files to app/assets/ to update the app.
"""

import json
import os
import shutil

import numpy as np

from .compare import OURS, load_runs


def plot_confusion(counts: np.ndarray, thr: float, path: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tp, fp, fn, tn = counts.sum(0)
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j, i, f'{v:,}', ha='center', va='center', color='white' if v > cm.max() / 2 else 'black'
        )
    ax.set_xticks([0, 1], ['Pred background', 'Pred rust'])
    ax.set_yticks([0, 1], ['True background', 'True rust'])
    ax.set_title(f'Test confusion matrix (threshold {thr:.2f})')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_for_app(out_dir: str, log=print) -> int | None:
    """Exports the best-on-validation CANet-B4 seed; returns that seed, or None without runs."""
    runs = load_runs(out_dir).get(OURS)
    if not runs:
        log(f'[export] no finished {OURS} runs')
        return None
    seed = max(runs, key=lambda s: runs[s]['metrics']['val@thr']['IoU'])
    run = runs[seed]
    m, h = run['metrics'], run['history']['history']
    exp = os.path.join(out_dir, 'export')
    os.makedirs(exp, exist_ok=True)
    t = m['test@thr']
    results = {
        'model': OURS,
        'seed': seed,
        'split': 'test',
        'best_epoch': m['best_epoch'],
        'best_threshold': m['threshold'],
        'final_metrics': {
            'IoU': round(t['IoU'], 4),
            'F1 / Dice': round(t['Dice'], 4),
            'Precision': round(t['Precision'], 4),
            'Recall': round(t['Recall'], 4),
            'Specificity': round(t['Specificity'], 4),
            'Accuracy': round(t['Accuracy'], 4),
        },
        'history': {k: h[k] for k in ('train_loss', 'val_loss', 'val_iou', 'val_dice')},
    }
    with open(os.path.join(exp, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    shutil.copy(os.path.join(run['dir'], 'best.pt'), os.path.join(exp, 'best_model.pth'))
    plot_confusion(run['counts'], m['threshold'], os.path.join(exp, 'confusion_matrix.png'))
    fig_dir = os.path.join(out_dir, 'figures')
    for name in ('training_curves.png', 'qualitative_results.png'):
        if os.path.exists(os.path.join(fig_dir, name)):
            shutil.copy(os.path.join(fig_dir, name), os.path.join(exp, name))
    log(f'[export] {OURS} seed {seed} (best val IoU) -> {exp}')
    return seed
