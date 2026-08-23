"""Evaluation metrics: trajectory (ADE/FDE), action, and text-embedding fidelity."""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ trajectory
def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """
    pred, true : (N, T, 2) in metres.
    ADE = mean displacement over all waypoints; FDE = displacement at the last one.
    """
    d = np.linalg.norm(pred - true, axis=-1)          # (N, T)
    return float(d.mean()), float(d[:, -1].mean())


def min_ade_fde(preds: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """
    preds : (K, N, T, 2) — K samples from the generative head.
    Standard best-of-K forecasting metric: credit the closest sample.
    """
    d = np.linalg.norm(preds - true[None], axis=-1)   # (K, N, T)
    ade = d.mean(axis=2).min(axis=0)                  # (N,)
    fde = d[:, :, -1].min(axis=0)                     # (N,)
    return float(ade.mean()), float(fde.mean())


# ---------------------------------------------------------------- action
def action_report(pred: np.ndarray, true: np.ndarray, classes: tuple) -> dict:
    acc = float((pred == true).mean())
    per_class = {}
    for i, name in enumerate(classes):
        mask = true == i
        if mask.any():
            per_class[name] = {
                "support": int(mask.sum()),
                "recall": round(float((pred[mask] == i).mean()), 4),
            }
    return {"accuracy": round(acc, 4), "per_class_recall": per_class}


# ------------------------------------------------------- text embedding
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return (a * b).sum(-1)


def text_report(
    pred_emb: np.ndarray,      # (N, 384) generated, decoded back to sentence space
    true_emb: np.ndarray,      # (N, 384) ground truth
    ceiling_emb: np.ndarray,   # (N, 384) truth round-tripped through PCA
    bank_emb: np.ndarray,      # (B, 384) unique embeddings seen in TRAIN
    bank_texts: list[str],
    true_texts: list[str],
    mean_emb: np.ndarray,      # (384,) train mean — the "predict the average" baseline
) -> dict:
    """How close is the generated embedding to the real trajectory_text?"""
    cos_model = cosine(pred_emb, true_emb)
    cos_ceiling = cosine(ceiling_emb, true_emb)
    cos_mean = cosine(np.repeat(mean_emb[None], len(true_emb), axis=0), true_emb)

    # nearest-neighbour retrieval in the TRAIN text bank
    sim = pred_emb @ bank_emb.T                        # both L2-normalised
    nearest = sim.argmax(axis=1)
    retrieved = [bank_texts[i] for i in nearest]

    covered = np.array([t in set(bank_texts) for t in true_texts])
    correct = np.array([r == t for r, t in zip(retrieved, true_texts)])

    # baseline: always retrieve the most common training text
    from collections import Counter

    return {
        "cosine_model": round(float(cos_model.mean()), 4),
        "cosine_pca_ceiling": round(float(cos_ceiling.mean()), 4),
        "cosine_mean_baseline": round(float(cos_mean.mean()), 4),
        "retrieval_accuracy": round(float(correct.mean()), 4),
        "retrieval_accuracy_covered": (
            round(float(correct[covered].mean()), 4) if covered.any() else None
        ),
        "bank_coverage": round(float(covered.mean()), 4),
        "bank_size": len(bank_texts),
        "n_unique_true": len(set(true_texts)),
    }


# ---------------------------------------------------------------- summary
def aggregate(per_fold: list[dict]) -> dict:
    """Mean and std of every scalar metric across folds."""
    out = {}
    for key in per_fold[0]:
        vals = [f[key] for f in per_fold if isinstance(f.get(key), (int, float))]
        if len(vals) == len(per_fold):
            out[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
    return out
