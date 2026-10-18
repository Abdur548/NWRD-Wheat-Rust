Tiles: 128px at 0.5× scale. Test: 3840 tiles (722 with rust) from 960 patches. Rust pixel share, train: 9.77% of patches, 9.79% of tiles.

## Accuracy at 128px (test tiles, threshold chosen on validation)

| Model | Seeds | IoU | Dice / F1 | Precision | Recall | IoU per seed |
|---|---|---|---|---|---|---|
| canet_b4 | 3 | 0.704 ± 0.004 | 0.826 ± 0.003 | 0.817 ± 0.019 | 0.837 ± 0.013 | 42: 0.701, 43: 0.702, 44: 0.709 |
| unet | 3 | 0.749 ± 0.009 | 0.857 ± 0.006 | 0.850 ± 0.012 | 0.863 ± 0.006 | 42: 0.758, 43: 0.752, 44: 0.739 |

CANet − UNet: ΔIoU -0.0454, 95% CI [-0.0809, -0.0056] resampling test photographs (p 0.028); CANet higher on 0 of 3 seeds; seed-level t-test p 0.030.
Pre-registered decision rule at 128px: CANet-B4 **is worse than** the paper's UNet.

## Against the paper's published UNet + APF

| Source | Test set | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|---|
| Paper, Table 3 | full test set | 0.506 | 0.624 | 0.557 | – |
| canet_b4, 128px | all test tiles | 0.817 | 0.837 | 0.826 | 0.704 |
| unet, 128px | all test tiles | 0.850 | 0.863 | 0.857 | 0.749 |
| Paper, Table 4 | background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4, 128px | rust tiles only | 0.918 | 0.837 | 0.875 | 0.778 |
| unet, 128px | rust tiles only | 0.914 | 0.863 | 0.888 | 0.798 |

Same patch size and field of view as the paper. Still different: the paper's test photographs (random 80/10/10 split), its training (APF, RMSprop, 500 epochs, focal loss) and patch sampling, so these rows are context, not a paired comparison.

## Effect of scale on each model (test IoU, mean over seeds)

| Model | 512px native | 128px at 0.5× | Change | Seeds (512 / 128) |
|---|---|---|---|---|
| canet_b4 | 0.729 ± 0.013 | 0.704 ± 0.004 | -0.024 | 3 / 3 |
| unet | 0.747 ± 0.012 | 0.749 ± 0.009 | +0.002 | 3 / 3 |

## Efficiency

Measured on Tesla T4 (torch 2.10.0+cu128), random weights.

| Scale | Model | Params (M) | GFLOPs | GPU fp32 bs1 (ms) | GPU fp16 bs1 (ms) | fp16 throughput (img/s) | CPU bs1 (ms) | Train min/epoch |
|---|---|---|---|---|---|---|---|---|
| 512px | canet_b4 | 29.4 | 36.87 | 30.6 | 26.3 | 73 @bs8 | 725 | 3.2 |
| 512px | unet | 31.0 | 385.34 | 94.9 | 40.1 | 28 @bs8 | 3157 | 6.6 |
| 128px | canet_b4 | 29.4 | 2.31 | 23.8 | 25.4 | 299 @bs8, 865 @bs64 | 72 | 3.1 |
| 128px | unet | 31.0 | 24.08 | 7.3 | 3.7 | 468 @bs8, 530 @bs64 | 162 | 1.7 |

Per 4608×3456 photograph (60.8 native 512px patches, or 243 tiles of 128px at 0.5×, by area; GPU batched time uses the largest batch measured):

| Scale | Model | TFLOPs / photo | GPU fp16 batched (ms / photo) | GPU fp32 bs1 (ms / photo) | CPU bs1 (s / photo) |
|---|---|---|---|---|---|
| 512px | canet_b4 | 2.24 | 837 | 1857 | 44.0 |
| 512px | unet | 23.41 | 2172 | 5767 | 191.8 |
| 128px | canet_b4 | 0.56 | 281 | 5787 | 17.4 |
| 128px | unet | 5.85 | 459 | 1779 | 39.5 |
