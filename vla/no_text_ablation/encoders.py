"""Frozen image encoder: images -> CNN features.

Identical to ../defussion_text_generator/encoders.py with the sentence
encoder removed — this ablation never touches text.

The CNN is pretrained and frozen, so running it over the whole dataset
before cross-validation introduces NO leakage: it is not fitted on this
data and sees no labels. Features are cached (shared with the main model's
cache when present, so both runs see byte-identical inputs).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from config import Config
from data import Sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- images
def _build_cnn(cfg: Config):
    import os

    import torchvision.models as tvm

    if cfg.cnn_name != "mobilenet_v3_small":
        raise ValueError(f"unsupported cnn_name {cfg.cnn_name!r}")

    # VLA_CNN_WEIGHTS=none skips the ImageNet download — for offline smoke tests
    # only; features are then random and the numbers are meaningless.
    if os.environ.get("VLA_CNN_WEIGHTS", "").lower() == "none":
        print("  [warn] VLA_CNN_WEIGHTS=none -> UNTRAINED backbone (smoke test only)")
        net = tvm.mobilenet_v3_small(weights=None)
    else:
        net = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.features, 576


def _preprocess(images_uint8: np.ndarray, size: int) -> torch.Tensor:
    """(B, H, W, 3) uint8  ->  (B, 3, size, size) normalised float tensor."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def image_features(samples: list[Sample], cfg: Config) -> np.ndarray:
    """(N, 576) pooled CNN features, one row per sample. Cached."""
    cache = cfg.cache_dir / f"img_{cfg.cnn_name}_{cfg.image_size}.npz"
    key = np.array([f"{s.scenario}:{s.frame}" for s in samples])

    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        if len(blob["key"]) == len(key) and (blob["key"] == key).all():
            print(f"  [cache] image features {blob['feat'].shape} <- {cache.name}")
            return blob["feat"]

    backbone, dim = _build_cnn(cfg)
    feats = np.zeros((len(samples), dim), dtype=np.float32)

    # group by scenario so each .h5 is opened exactly once
    by_scenario: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_scenario.setdefault(s.scenario, []).append(i)

    for scenario, idxs in by_scenario.items():
        h5_path = cfg.dataset_dir / "data" / f"{scenario}.h5"
        with h5py.File(h5_path, "r") as f:
            rows = [samples[i].frame for i in idxs]
            images = f["rgb_images"][:][rows]        # (n, 480, 640, 3) uint8
        for start in range(0, len(idxs), 16):
            chunk = images[start : start + 16]
            x = _preprocess(chunk, cfg.image_size)
            fmap = backbone(x)                        # (b, 576, 7, 7)
            pooled = fmap.mean(dim=(2, 3))            # global average pool
            feats[idxs[start : start + 16]] = pooled.numpy()
        print(f"  [cnn] {scenario}: {len(idxs)} frames")

    np.savez_compressed(cache, feat=feats, key=key)
    return feats
