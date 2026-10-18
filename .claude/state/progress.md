# Progress

**Status:** NWRD_final is built, verified, and pushed (2026-10-18) to the public repository
`github.com/Abdur548/NWRD-Wheat-Rust`. Both experiments are documented in `docs/results.md`;
DeepLabV3+ R50 at 128px is being added at the user's request.
**Updated:** 2026-10-18
**Verified by:** `python -m pytest` (75 passed, 3 min 22 s), `ruff check` / `ruff format --check`,
`npm run lint` / `npm run build` / `npm audit` (0 vulnerabilities), a real-data smoke test of the
API, the dashboard (Streamlit AppTest) and the React UI in Chrome.

## Done (2026-10-17/18)

- Clean package `nwrd/` with presets, safe checkpoint loading, seeded augmentation, verdict in
  reports, export as its own stage, shared whole-photograph inference.
- Equivalence gates against the old code: training (163 checks), models, aggregation: identical.
- Results imported: `results/512px/{final,replicate}`, `results/128px/final`; comparisons
  regenerated (numbers identical, verdicts added); analysis for both scales; `scales.md`.
- App: FastAPI backend (limits, CORS for localhost and 127.0.0.1, EXIF, threadpool), Streamlit
  dashboard, React UI; v4 export (CANet-B4 seed 43, chosen on validation) as the served model.
- Docs consolidated to `docs/{protocol,results,paper_summary}.md`; README; PDFs render.

## Next

1. DeepLabV3+ R50 at 128px (3 seeds) on Kaggle, logged like the other runs (handoff.md).
2. Optional, needs the user: a LICENSE (MIT or Apache-2.0), a CITATION.cff.
