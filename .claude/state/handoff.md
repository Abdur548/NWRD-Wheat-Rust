# Handoff

**Status:** ACTIVE (written 2026-10-18)

NWRD_final was built from `Plant_Segmentation` per `../NWRD_final_PLAN.md` (approved 2026-10-17),
verified (see progress.md), committed and pushed on 2026-10-18 to the public repository
`github.com/Abdur548/NWRD-Wheat-Rust` at the user's request.

What the next session needs that the files don't show:

- **In progress:** the user asked (2026-10-18) to run DeepLabV3+ at both scales and log the runs
  like the others. DeepLabV3+ R50 already has 3 seeds at 512px (`results/512px/final`); the new
  work is 128px: add `deeplabv3plus_r50` to the `128px` preset, rebuild the notebooks, resume the
  128px kernel, then copy the runs into `results/128px/final` and regenerate the reports and docs.
- **D1 was answered "keep"** and read as "keep both UIs". The user did not object; the dashboard
  is ported. If they say React-only, delete `app/dashboard.py`, `tests/test_dashboard.py` and
  streamlit from `pyproject.toml`/`constraints.txt`, and the dashboard lines in README/CLAUDE.md.
- **Open wording question:** `docs/protocol.md` calls `deeplabv3plus_r50` "the strongest earlier
  in-house model"; it is a standard re-implementation (the internship's "Custom DeepLabV3+" code
  was never preserved). The user was asked whether to reword it; no answer yet.
