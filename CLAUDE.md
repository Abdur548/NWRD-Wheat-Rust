# NWRD_final

The maintained codebase of the NWRD wheat-rust segmentation study (CANet-B4 against the NWRD
paper's UNet and DeepLabV3+ R50). It supports an official research paper and is reviewed by the
supervisor: keep it at professional standards (tested, linted, no dead code, accurate docs).
`../Plant_Segmentation/` is the read-only archive it was cleaned from; don't develop there.

## Running it

From this folder, with `python` (C:\Python313; it has torch CPU, pytest, kaggle, ruff, streamlit):

- Tests: `python -m pytest -m "not slow"` (fast tier) or `python -m pytest` (all 75, ~4 min CPU).
  Streamlit's AppTest replaces `sys.modules['__main__']`; `tests/test_dashboard.py` restores it,
  or later DataLoader workers hang on Windows.
- Lint: `python -m ruff check . && python -m ruff format --check .`
- Scripts import `nwrd`: run them with `PYTHONPATH=.` unless the package is installed
  (`pip install -e ".[app,dev,docs]"`).
- After any `nwrd/` edit: `python scripts/build_notebooks.py` (a test fails on a stale notebook).
- Reports from saved results: see README "Rebuild the reports"; PDFs: `python scripts/md_to_pdf.py`.
- App: `uvicorn app.backend.main:app`; `cd app/frontend && npm run dev`; `streamlit run app/dashboard.py`.

## Conventions

- No local GPU: training happens only on Kaggle (`scripts/run_on_kaggle.py`). Both experiments are
  final; don't push to Kaggle without the user's go-ahead.
- CANet-B4 is "ours" in every table. Report results honestly even when it loses: it is
  *comparable* to UNet at 512px and *worse* at 128px.
- Every number in `docs/` must trace to a file in `results/` (docs/results.md, Appendix).
- Remote: `github.com/Abdur548/NWRD-Wheat-Rust` (public). Commit and push only when the user
  asks, and never publish credentials or session-private notes.

<!-- pipeline-continuity:start -->
## Session continuity

Read `.claude/state/` first: `handoff.md` (act on it if ACTIVE, then mark it CONSUMED),
`progress.md`, `pipeline.md` (architecture, decisions, invariants), `suggestions.md`. Check them
against `git status` and the files; where they disagree, reality wins: fix the file and say so.
<!-- pipeline-continuity:end -->
