# Suggestions

Everything below is outside the pre-registered comparisons; each would need its own protocol.

## Open

### Evaluate robustness to JPEG compression — 2026-10-18
**What:** Score the test set after re-encoding each patch as JPEG at several qualities (95, 85, 75).
**Why it's worth it:** In an app smoke test, JPEG quality 95 (mean change 0.8 grey levels) lowered
CANet-B4's predicted rust on one region from 61% to 49%. Field photographs are usually JPEG.
**Cost:** CPU-feasible (~25 min per quality for CANet-B4); needs the 960 test patches downloaded.
**Why not yet:** Not asked for; would add a section to `docs/results.md`.

### Recover lesion extent in CANet's output
**What:** A 1/8-resolution skip or learned upsampling instead of ×4 bilinear.
**Why it's worth it:** CANet-B4 under-segments patches >20% rust (89% of rust pixels); it has the
lowest recall in every lesion-size bin. **Cost:** model change + 3-seed re-run.

### Same-encoder ablation: DeepLabV3+ B4
**What:** Train `deeplabv3plus_b4` (already defined) under the protocol.
**Why it's worth it:** Separates the context module from the encoder; it costs the same FLOPs
(35.7 vs 36.9 GFLOPs). **Cost:** ~1.5 GPU-h per seed at 512px.

### Cross-validate by photograph
**What:** 10 folds of 10 photographs instead of the fixed split.
**Why it's worth it:** The 10-photo test set and 5-photo validation bound every estimate.
**Cost:** ~10× training compute.

### Add a LICENSE and a CITATION.cff
**What:** MIT or Apache-2.0 (Apache-2.0 matches EfficientNet-PyTorch); a citation file once the
paper's title and authors are fixed. **Why not yet:** the owners' choice.

## Rejected

### The paper's own training recipe for the 128px run — 2026-09-29
APF + RMSprop 1e-6 + 500 epochs + batch 64 + focal loss: beyond the GPU budget, and it changes
many variables at once; the 128px run changes only the input scale.

## Implemented

- Photograph-level bootstrap (2026-09-28); seeded augmentation, verdict in reports, 128px analysis,
  report cost per photograph (`results/analysis/scales.md`) (2026-10-17/18).
