# NWRD_final: pipeline, decisions and invariants

**Updated:** 2026-10-18

## What it is

A study of CANet with an ImageNet-pretrained EfficientNet-B4 encoder (CANet-B4, "ours") for
binary wheat-stripe-rust segmentation on NWRD, against the NWRD paper's UNet architecture and
DeepLabV3+ R50, under one protocol (`docs/protocol.md`). Two experiments, both **final**:

| Experiment | Preset | Kaggle kernel (slug from title) | Results |
|---|---|---|---|
| main, 512px native patches | `main` | `abdur548/nwrd-canet-b4-vs-baselines` | `results/512px/final/` (+ `replicate/`) |
| paper scale, 128px tiles | `128px` | `abdur548/nwrd-128px-canet-vs-unet` | `results/128px/final/` |

Verdicts (pre-registered rule): comparable to UNet and DeepLabV3+ at 512px; worse than UNet at
128px. Efficiency: 10.4× fewer FLOPs than UNet; faster at 512px on every device; not faster than
DeepLabV3+ R50 on a GPU.

## Flow

```
nwrd/ ── scripts/build_notebooks.py ──> notebooks/*.ipynb (embed nwrd/, PRESET = main | 128px)
   scripts/run_on_kaggle.py [--experiment 128px] [--resume] [--budget-hours H] ──> Kaggle T4
   nwrd.run: [tile] → train → eval → bench → figures → compare → export (512px only)
   runs_kaggle/<exp>/<account>_v<N>/ (ignored) ──copy by hand──> results/<scale>/final/
   nwrd.run --stage compare | scripts/analyse_results.py | scripts/compare_scales.py → results/
   docs/*.md ── scripts/md_to_pdf.py ──> pdf/ (ignored)
   export/ → app/assets/ (best_model.pth in Git LFS) → app/backend (FastAPI) + app/frontend
   (React) + app/dashboard.py (Streamlit), all predicting through nwrd/inference.py
```

## Invariants

- `nwrd/` is the source of truth; notebooks are generated (`tests/test_notebooks.py`).
- Kernel titles in `scripts/run_on_kaggle.py` must not change (Kaggle derives the slug from them).
- Each notebook has exactly one `EXPECT_RESTORE = False` and one `KAGGLE_TIME_BUDGET_HOURS = 11.0`
  line (rewritten by `--resume` / `--budget-hours`).
- The protocol is identical for every model; the test split is never used for selection.
- Tile names keep `_patch_` (`compare.source_photo` groups by photograph for the bootstrap).
- CANet module names are frozen (checkpoints load strictly); context pooling is capped at the
  feature-map size (no effect at ≥256px, tested).
- Checkpoints load with `weights_only=True` everywhere.
- A model trained on 128px tiles is never exported to the app (enforced in `nwrd/run.py`).
- `--stage compare`/`export` alone never rewrites a results folder's `config.json` or `log.txt`.
- The app predicts on native-resolution 512px patches with mirror padding, as the test set was
  scored (`nwrd/inference.py`); never resize a whole photograph to 512.

## Decisions

- 2026-10-17: replicated `Plant_Segmentation` into this repo (plan: `../NWRD_final_PLAN.md`).
  User decisions: keep both UIs (D1), main experiment final, `unet_tuned` seeds 43–44 dropped (D2),
  repo local only (D3), `canet_b4_twopass` removed (D4), Colab support dropped (D5), PDFs not
  versioned (D6).
- Equivalence gates (2026-10-17/18) proved the cleaned pipeline gives byte-identical outputs to
  the old one on CPU (training, models, aggregation of the published results).
- Augmentation is now seeded from (seed, epoch, worker) (`nwrd/train.py`); albumentations 2
  otherwise seeds from the OS. The published runs may have had unseeded augmentation.
- Patch bootstrap in the analysis now uses the pipeline's scheme (multinomial); the photograph
  bootstrap, which the decision rule uses, is unchanged.

## External dependencies

| What | Used for | Notes |
|---|---|---|
| Kaggle (T4 kernels, dataset `abdur548/nwrd-patched`) | Training | Token in `~/.kaggle/kaggle.json`; a second account via `KAGGLE_CONFIG_DIR`. The CLI clears its `-p` target folder on single-file downloads: give each download its own folder |
| Git LFS | `app/assets/best_model.pth` (118 MB) | |
| jsDelivr | mermaid 10.9.1 for PDFs | SHA-256 pinned in `scripts/md_to_pdf.py` |
