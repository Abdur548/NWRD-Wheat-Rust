# CANet-B4 vs baselines: comparison

Test split: 960 patches, 266 with rust, 8.0% rust pixels. Threshold chosen per run on val (IoU-maximising, grid 0.05–0.95), then frozen for test. Metrics pooled over all test pixels; mean ± sample std over seeds.

## Accuracy (test split, val-selected threshold)

| Model | Seeds | IoU | Dice | Precision | Recall | Specificity | Accuracy | IoU @0.5 | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| **canet_b4** (ours) | 3 | 0.7286 ± 0.0132 | 0.8429 ± 0.0088 | 0.8402 ± 0.0123 | 0.8459 ± 0.0153 | 0.9859 ± 0.0014 | 0.9746 ± 0.0014 | 0.7125 ± 0.0098 | 0.79 |
| deeplabv3plus_r50 | 3 | 0.7394 ± 0.0205 | 0.8500 ± 0.0135 | 0.8336 ± 0.0431 | 0.8689 ± 0.0227 | 0.9846 ± 0.0053 | 0.9753 ± 0.0032 | 0.7188 ± 0.0265 | 0.70 |
| unet | 3 | 0.7474 ± 0.0117 | 0.8554 ± 0.0076 | 0.8350 ± 0.0134 | 0.8770 ± 0.0015 | 0.9848 ± 0.0014 | 0.9761 ± 0.0014 | 0.7211 ± 0.0226 | 0.70 |
| unet_tuned | 1 | 0.6165 | 0.7628 | 0.6846 | 0.8610 | 0.9653 | 0.9569 | 0.6165 | 0.50 |

## Against the NWRD paper (UNet + APF)

| Result | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|
| Paper, Table 3, full test set | 0.506 | 0.624 | 0.557 | – |
| Paper, Table 4, background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4, re-run: all test patches | 0.840 | 0.846 | 0.843 | 0.729 |
| canet_b4, re-run: rust-positive test patches (≈ Table 4) | 0.917 | 0.846 | 0.880 | 0.785 |
| deeplabv3plus_r50, re-run: all test patches | 0.834 | 0.869 | 0.850 | 0.739 |
| deeplabv3plus_r50, re-run: rust-positive test patches (≈ Table 4) | 0.893 | 0.869 | 0.880 | 0.786 |
| unet, re-run: all test patches | 0.835 | 0.877 | 0.855 | 0.747 |
| unet, re-run: rust-positive test patches (≈ Table 4) | 0.888 | 0.877 | 0.883 | 0.790 |
| unet_tuned, re-run: all test patches | 0.685 | 0.861 | 0.763 | 0.616 |
| unet_tuned, re-run: rust-positive test patches (≈ Table 4) | 0.825 | 0.861 | 0.843 | 0.728 |

Paper figures are as published, on a different test unit (full-image patches, 128px). The `unet` row is the paper's architecture re-trained under this protocol, so CANet vs `unet` is the like-for-like comparison.

## Is the difference real? (CANet-B4 minus baseline, test split)

| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | 95% CI, resampling patches | ΔDice | ΔDice 95% CI, photographs | seed-level t-test p | CANet higher (seeds) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| deeplabv3plus_r50 | 3 | -0.0108 | [-0.0241, +0.0079] (p 0.2092) | [-0.0306, +0.0070] (p 0.2508) | -0.0071 | [-0.0165, +0.0052] (p 0.2188) | 0.1631 | 0 of 3 | CANet-B4 is comparable to deeplabv3plus_r50 |
| unet | 3 | -0.0189 | [-0.0421, +0.0186] (p 0.2612) | [-0.0419, +0.0033] (p 0.1004) | -0.0125 | [-0.0279, +0.0125] (p 0.2612) | 0.1964 | 0 of 3 | CANet-B4 is comparable to unet |
| unet_tuned | 1 | +0.1019 | [+0.0056, +0.1886] (p 0.0324) | [+0.0673, +0.1360] (p 0.0000) | +0.0734 | [+0.0038, +0.1472] (p 0.0324) | n/a (1 seed) | 1 of 1 | CANet-B4 is comparable to unet_tuned (interim: 1 of 3 seeds) |

Two runs of the paper's UNet exist (shared lr and `unet_tuned`). The stronger on **validation** IoU, `unet`, is the paper-architecture baseline; choosing by test IoU would bias the comparison.

Verdict: the decision rule fixed before the results were seen. CANet-B4 *improves on* a baseline if its mean test IoU is higher, the photograph-resampling 95% CI of ΔIoU excludes 0, and it is higher on at least 2 of 3 seeds; it *is worse* if that CI lies below 0; otherwise the two are *comparable*.

A CI that excludes 0 means the gap is larger than test-set sampling noise. Use the photograph interval: the test patches are cut from a few photographs, and patches of one photograph are not independent, so resampling patches understates the uncertainty. Neither bootstrap captures training-run variance; the seed-level t-test does, but has little power with 3 seeds.

## Efficiency (Tesla T4, 512x512 input, torch 2.10.0+cu128)

| Model | Params (M) | GFLOPs | latency_ms_bs1_fp32 | latency_ms_bs1_fp16 | throughput_img_s_bs8_fp16 | peak_mem_MB_bs8_fp16 | CPU latency bs1 (ms) | Train min/epoch |
|---|---|---|---|---|---|---|---|---|
| canet_b4 | 29.39 | 36.9 | 30.6 | 26.3 | 72.6 | 743.7 | 725 | 3.2 |
| deeplabv3plus_r50 | 26.68 | 73.1 | 27.3 | 10.8 | 95.4 | 834.7 | 579 | 2.1 |
| unet | 31.04 | 385.3 | 94.9 | 40.1 | 28.0 | 1873.9 | 3157 | 6.6 |
| unet_tuned | 31.04 | 385.3 | 96.3 | 41.0 | 27.7 | 1872.6 | 3024 | 7.0 |
| canet_b4_twopass | 29.39 | 52.5 | 53.7 | 48.8 | 38.4 | 707.6 | 1042 | – |

`canet_b4_twopass` is CANet as first implemented, running its encoder twice per forward pass; its outputs are bit-identical to `canet_b4`, so only the cost differs. It was measured in this run and is no longer part of the code.

## Models

- `canet_b4` — CANet, EfficientNet-B4 encoder (ours)
- `deeplabv3plus_r50` — DeepLabV3+, ResNet-50 encoder (baseline)
- `unet` — UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)
- `unet_tuned` — UNet (paper architecture) with a from-scratch lr (config.unet_tuned_lr)
