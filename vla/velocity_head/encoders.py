"""Frozen encoders: image -> CNN features, text -> sentence embedding.

Neither is trained here and neither sees a label, so running them over the
whole dataset before cross-validation is not leakage. The only data-fitted
component is the PCA in latent.py, which lives inside the fold loop.

The image cache is keyed by "scenario:frame" and is looked up per key rather
than by array order, so this package reuses the cache written by
../defussion_text_generator even though its sample list is a different length
(windowing drops hist + horizon frames per run).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from config import Config
from data import Sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- images
def _build_cnn(cfg: Config):
    import torch
    import torchvision.models as tvm

    if cfg.cnn_name != "mobilenet_v3_small":
        raise ValueError(f"unsupported cnn_name {cfg.cnn_name!r}")

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


def _preprocess(images_uint8: np.ndarray, size: int):
    import torch

    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def image_features(samples: list[Sample], cfg: Config) -> np.ndarray:
    """(N, 576) pooled CNN features, one row per window (its CURRENT frame)."""
    import torch

    cache = cfg.cache_dir / f"img_{cfg.cnn_name}_{cfg.image_size}.npz"
    want = [f"{s.scenario}:{s.frame}" for s in samples]

    known: dict[str, np.ndarray] = {}
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        known = {str(k): f for k, f in zip(blob["key"], blob["feat"])}
        if all(k in known for k in want):
            print(f"  [cache] image features for {len(want)} frames <- {cache.name}")
            return np.stack([known[k] for k in want]).astype(np.float32)

    missing = sorted({k for k in want if k not in known})
    print(f"  [cnn] extracting {len(missing)} frames not in the cache")

    import h5py

    backbone, dim = _build_cnn(cfg)
    by_scenario: dict[str, list[int]] = {}
    for key in missing:
        scenario, frame = key.rsplit(":", 1)
        by_scenario.setdefault(scenario, []).append(int(frame))

    with torch.no_grad():
        for scenario, frames in by_scenario.items():
            frames = sorted(frames)
            h5_path = cfg.dataset_dir / "data" / f"{scenario}.h5"
            with h5py.File(h5_path, "r") as f:
                images = f["rgb_images"][:][frames]
            for start in range(0, len(frames), 16):
                chunk = images[start:start + 16]
                pooled = backbone(_preprocess(chunk, cfg.image_size)).mean(dim=(2, 3))
                for j, fr in enumerate(frames[start:start + 16]):
                    known[f"{scenario}:{fr}"] = pooled[j].numpy().astype(np.float32)
            print(f"  [cnn] {scenario}: {len(frames)} frames")

    all_keys = sorted(known)
    np.savez_compressed(
        cache,
        feat=np.stack([known[k] for k in all_keys]).astype(np.float32),
        key=np.array(all_keys),
    )
    return np.stack([known[k] for k in want]).astype(np.float32)


# ---------------------------------------------------------------- text
def text_embeddings(texts: list[str], cfg: Config) -> np.ndarray:
    """(n_unique, 384) L2-normalised sentence embeddings. Cached on disk."""
    unique = sorted(set(texts))
    cache = cfg.cache_dir / f"text_{Path(cfg.text_encoder).name}_{cfg.text_field}.npz"

    known: dict[str, np.ndarray] = {}
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        known = {str(t): e for t, e in zip(blob["texts"], blob["emb"])}
        if all(t in known for t in unique):
            print(f"  [cache] text embeddings for {len(unique)} strings <- {cache.name}")
            return np.stack([known[t] for t in unique])

    missing = [t for t in unique if t not in known]

    if os.environ.get("VLA_TEXT_WEIGHTS", "").lower() == "none":
        # Offline smoke test only. Deterministic pseudo-embeddings: the code
        # path runs end to end but every text number it produces is meaningless.
        print("  [warn] VLA_TEXT_WEIGHTS=none -> FAKE text embeddings (smoke test only)")
        for t in missing:
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            e = rng.normal(size=384).astype(np.float32)
            known[t] = e / np.linalg.norm(e)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required to embed the text targets.\n"
                "    pip install sentence-transformers"
            ) from exc
        print(f"  [text] embedding {len(missing)} new strings with {cfg.text_encoder}")
        model = SentenceTransformer(cfg.text_encoder)
        new = model.encode(missing, convert_to_numpy=True, normalize_embeddings=True)
        known.update({t: e.astype(np.float32) for t, e in zip(missing, new)})

    all_texts = sorted(known)
    np.savez_compressed(
        cache,
        emb=np.stack([known[t] for t in all_texts]),
        texts=np.array(all_texts, dtype=object),
    )
    return np.stack([known[t] for t in unique])


def text_matrix(samples: list[Sample], cfg: Config):
    """Per-window (N, 384) embeddings of cfg.text_field, plus the vocabulary."""
    texts = [s.text for s in samples]
    unique = sorted(set(texts))
    emb = text_embeddings(texts, cfg)
    lookup = {t: emb[i] for i, t in enumerate(unique)}
    return np.stack([lookup[t] for t in texts]), unique
