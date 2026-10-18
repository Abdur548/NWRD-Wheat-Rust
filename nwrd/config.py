"""Experiment protocol. Every model in an experiment is trained with exactly these settings.

Each field is also a command-line flag of `python -m nwrd.run` (`data_root` -> `--data-root`).
A preset sets the fields that define one experiment; flags given explicitly override it.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    data_root: str = ''
    out_dir: str = 'runs'
    models: list[str] = field(
        default_factory=lambda: ['canet_b4', 'unet', 'deeplabv3plus_r50', 'unet_tuned']
    )
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    data_seed: int = 42  # background subsample; fixed so all runs see the same patches

    img_size: int = 512
    # >0: train and evaluate on tiles re-cut from every patch the way the NWRD paper cut its
    # patches: resized by tile_scale (Lanczos), then split into tile_size squares. 128 and 0.5
    # reproduce the paper's patches (nwrd/tiles.py). img_size must then equal tile_size.
    tile_size: int = 0
    tile_scale: float = 0.5
    tile_dir: str = ''  # where tiles are written; default: under the temp directory
    bg_keep_ratio: float = 0.20
    epochs: int = 30
    freeze_encoder_epochs: int = 3
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 5e-5
    # 'unet_tuned' only: the shared lr suits fine-tuning pretrained encoders and may
    # under-train a from-scratch UNet, so this control variant gets its own.
    unet_tuned_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dice_weight: float = 0.7
    bce_weight: float = 0.3
    dropout: float = 0.5  # CANet head dropout; the other models have none
    sched_t0: int = 5
    sched_tmult: int = 2
    sched_eta_min: float = 1e-7
    amp: bool = True
    pretrained: bool = True

    # threshold chosen on val over this grid, then frozen and applied to test
    thr_lo: float = 0.05
    thr_hi: float = 0.95
    thr_step: float = 0.01

    # efficiency benchmark: models to measure (empty = the trained models), fp16 batch sizes
    bench_models: list[str] = field(default_factory=list)
    bench_batches: list[int] = field(default_factory=lambda: [8])
    bench_quick: bool = False  # few timing iterations (smoke tests)

    time_budget_hours: float = 0.0  # >0: stop cleanly before a platform session limit
    limit: int = 0  # >0: use only N patches per split (smoke tests)

    def validate(self) -> None:
        if self.tile_size and self.img_size != self.tile_size:
            raise ValueError(
                f'img_size {self.img_size} would resize the {self.tile_size}px tiles; '
                f'set --img-size {self.tile_size}'
            )

    def to_dict(self) -> dict:
        return asdict(self)


# The two experiments. `main` is the protocol as defined above; `128px` changes only the input
# scale, to the NWRD paper's 128px patches of half-size photographs, so 512px vs 128px is a
# one-variable comparison. fp16 throughput at 128px is also measured at the paper's batch of 64.
PRESETS = {
    'main': {},
    '128px': {
        'models': ['canet_b4', 'unet'],
        'img_size': 128,
        'tile_size': 128,
        'tile_scale': 0.5,
        'bench_batches': [8, 64],
    },
}
