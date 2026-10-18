# Wheat stripe-rust segmentation on NWRD: CANet-B4 against the paper's baselines

Pixel-level segmentation of wheat stripe rust in field photographs of the NWRD dataset (Anwar et
al., *The NWRD Dataset*, Sensors 23(15):6942, 2023). The method studied is **CANet with an
ImageNet-pretrained EfficientNet-B4 encoder (CANet-B4)**. It is compared with the NWRD paper's
UNet architecture and with DeepLabV3+ R50 under one protocol: the same data, split by photograph,
training schedule and threshold rule, three seeds per model, and a decision rule fixed before the
results were seen. Accuracy and efficiency are both measured. A FastAPI service with a React
interface, and a Streamlit dashboard, serve the trained model.

## Results

| | 512px (native resolution) | 128px (the paper's scale) |
|---|---|---|
| Test IoU, CANet-B4 (ours) | 0.729 ± 0.013 | 0.704 ± 0.004 |
| Test IoU, UNet (paper architecture, re-trained) | 0.747 ± 0.012 | 0.749 ± 0.009 |
| Test IoU, DeepLabV3+ R50 | 0.739 ± 0.021 | – |
| CANet-B4 − UNet, 95% CI (resampling test photographs) | −0.019 [−0.042, +0.019] | −0.045 [−0.081, −0.006] |
| Verdict against UNet (pre-registered rule) | comparable | worse |
| GFLOPs per input, CANet-B4 vs UNet | 36.9 vs 385.3 | 2.3 vs 24.1 |

CANet-B4 is not more accurate than the paper's UNet architecture: comparable at 512px and worse
at the paper's 128px scale. It needs a tenth of UNet's computation, and at 512px runs 3.1× faster
on a T4 GPU and 4.4× faster on a CPU. Both re-trained models far exceed the paper's published
scores (F1 0.826–0.857 against 0.557 at the same scale), so that gain comes from the training and
evaluation protocol, not from CANet-B4's architecture. Full results:
[`docs/results.md`](docs/results.md). Protocol: [`docs/protocol.md`](docs/protocol.md). The
baseline paper, with its figures and a check of its claims:
[`docs/paper_summary.md`](docs/paper_summary.md).

## Repository layout

```
nwrd/            the pipeline: models, data, tiling, training, evaluation, benchmark, comparison,
                 app export, whole-photograph inference
scripts/         Kaggle runner, notebook builder, result analysis, cross-scale report, PDF and
                 paper-figure tools
notebooks/       the two Kaggle notebooks, generated from nwrd/ by scripts/build_notebooks.py
results/         curated outputs of the Kaggle runs and their analysis (see results/README.md)
app/             backend/ (FastAPI), frontend/ (React + Vite), dashboard.py (Streamlit),
                 assets/ (the served model, Git LFS, with its metrics and figures)
docs/            protocol, results and paper summary
tests/           unit and end-to-end tests on synthetic data (CPU)
```

## Installation

Python 3.10 or newer. The model file is stored with [Git LFS](https://git-lfs.com):

```bash
git lfs pull
pip install -e ".[app,dev,docs]"            # or pip install -e . for the pipeline only
pip install -e ".[app,dev,docs]" -c constraints.txt   # the exact versions last tested
```

## Usage

**Run the experiments.** Training runs on Kaggle GPUs. With a Kaggle API token in
`~/.kaggle/kaggle.json`:

```bash
python scripts/run_on_kaggle.py --watch                     # main experiment (512px)
python scripts/run_on_kaggle.py --experiment 128px --watch  # the paper's scale
python scripts/run_on_kaggle.py --resume --watch            # continue after a time-budget stop
```

Each notebook embeds `nwrd/`; after changing the package, rebuild them with
`python scripts/build_notebooks.py` (a test fails while one is stale). On a local GPU:
`python -m nwrd.run --preset main --data-root <path to wheat_rust_patches> --out runs`.

**Rebuild the reports** from the saved results, without a GPU:

```bash
python -m nwrd.run --stage compare --out results/512px/final
python scripts/analyse_results.py results/512px/final --replicate results/512px/replicate
python scripts/analyse_results.py results/128px/final
python scripts/compare_scales.py results/128px/final results/512px/final --out results/analysis/scales.md
python scripts/md_to_pdf.py                                 # docs/*.md -> pdf/ (needs Edge or Chrome)
```

**Run the app.** The model predicts a photograph at native resolution in 512px patches, the way
the test set was scored (about 1.5 s per patch on a CPU).

```bash
uvicorn app.backend.main:app                                # API at http://localhost:8000
cd app/frontend && npm install && npm run dev               # interface at http://localhost:5173
streamlit run app/dashboard.py                              # or the Streamlit dashboard
```

The API's settings (model folder, allowed origins, upload limits) are environment variables,
documented in `app/backend/main.py`. The interface reads the API address from `VITE_API_URL`.

**Run the tests:**

```bash
python -m pytest -m "not slow"     # unit, app and notebook tests, about a minute
python -m pytest                   # all 75, with the end-to-end pipeline runs: about 4 minutes on a CPU
python -m ruff check . && python -m ruff format --check .
```

## Data

The experiments use the Kaggle dataset `abdur548/nwrd-patched`: the 100 NWRD photographs cut
into 512×512 patches at native resolution and split by photograph (train 1–85, validation
86–90, test 91–100). The original NWRD dataset is published by its authors at
[github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD](https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD).

## Credits

- **Dataset and baseline:** Anwar, H. et al. *The NWRD Dataset: An Open-Source Annotated
  Segmentation Dataset of Diseased Wheat Crop.* Sensors 2023, 23(15), 6942.
  doi:10.3390/s23156942.
- **CANet design:** adapted from Tang, Q. et al. *Attention-guided Chained Context Aggregation for
  Semantic Segmentation*, arXiv:2002.12041; the project's reference is Shoaib, M. et al.,
  Frontiers in Plant Science 13:1095547 (2022). See [`docs/protocol.md`](docs/protocol.md) §2.1
  for what was changed.
- **Libraries:** EfficientNet-PyTorch (Apache-2.0) for the encoder and its ImageNet weights,
  segmentation-models-pytorch (MIT) for DeepLabV3+, PyTorch, albumentations.

**Project team:** Syed Muhammad Abdur Rahman, Abdul Hadi Sheikh, Abdullah Salim Nizami.
**Supervisor:** Dr. Imran Malik. NUST SEECS.

## License

No license has been chosen yet, so the code is not yet open source: until a `LICENSE` file is
added, all rights are reserved by the authors. MIT and Apache-2.0 both fit the dependencies;
Apache-2.0 matches the encoder library.
