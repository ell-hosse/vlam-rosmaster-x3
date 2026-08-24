"""Metrics for the two stages.

Stage 1 (this model)      : per-channel command error, in physical units.
Stage 2 (the calculator)  : ADE / FDE of the trajectory those commands produce.

Both are reported per fold, plus the oracle floor -- what the calculator scores
when it is handed the TRUE commands. The gap between the oracle and the model
row is the part of the trajectory error this network is responsible for.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ commands
def velocity_report(pred: np.ndarray, true: np.ndarray,
                    prototypes: np.ndarray | None = None) -> dict:
    """
    pred, true : (N, F, 2) command chunks in physical units (m/s, rad/s).

    Linear and angular are reported separately and never pooled -- they are
    different units, and pooling them would hide which one is failing.
    """
    err = pred - true
    v_err, w_err = err[..., 0], err[..., 1]

    def stats(e, t):
        denom = np.maximum(t.std(), 1e-8)
        return {
            "mae": round(float(np.abs(e).mean()), 4),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
            "max_abs": round(float(np.abs(e).max()), 4),
            "r2": round(float(1.0 - (e ** 2).mean() / denom ** 2), 4),
            "mae_per_step": [round(float(x), 4) for x in np.abs(e).mean(axis=0)],
        }

    out = {
        "v": stats(v_err, true[..., 0]),
        "w": stats(w_err, true[..., 1]),
    }

    if prototypes is not None and len(prototypes):
        # The dataset holds only ~10 distinct (v, w) commands, so "did it pick
        # the right one" is a fair and much more legible score than MAE.
        d = np.linalg.norm(
            pred.reshape(-1, 1, 2) - prototypes.reshape(1, -1, 2), axis=2
        )
        snapped = prototypes[d.argmin(axis=1)].reshape(pred.shape)
        exact = np.all(np.isclose(snapped, true, atol=1e-3), axis=2)
        out["snapped_exact_match"] = round(float(exact.mean()), 4)
        out["snapped_exact_match_step0"] = round(float(exact[:, 0].mean()), 4)
        out["snapped_v_mae"] = round(float(np.abs(snapped[..., 0] - true[..., 0]).mean()), 4)
        out["snapped_w_mae"] = round(float(np.abs(snapped[..., 1] - true[..., 1]).mean()), 4)
    return out


# ---------------------------------------------------------- trajectory
def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """pred, true : (N, T, 2) in metres."""
    d = np.linalg.norm(pred - true, axis=-1)
    return float(d.mean()), float(d[:, -1].mean())


def min_ade_fde(preds: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """preds : (K, N, T, 2) — best-of-K over samples from the generative head."""
    d = np.linalg.norm(preds - true[None], axis=-1)
    return float(d.mean(axis=2).min(axis=0).mean()), float(d[:, :, -1].min(axis=0).mean())


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


def text_report(pred_emb, true_emb, ceiling_emb, bank_emb, bank_texts,
                true_texts, mean_emb) -> dict:
    cos_model = cosine(pred_emb, true_emb)
    cos_ceiling = cosine(ceiling_emb, true_emb)
    cos_mean = cosine(np.repeat(mean_emb[None], len(true_emb), axis=0), true_emb)

    sim = pred_emb @ bank_emb.T
    retrieved = [bank_texts[i] for i in sim.argmax(axis=1)]
    covered = np.array([t in set(bank_texts) for t in true_texts])
    correct = np.array([r == t for r, t in zip(retrieved, true_texts)])

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
    """Mean and std of every scalar metric across folds (nested dicts included)."""
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}{k}"] = float(v)
        return out

    rows = [flat(f) for f in per_fold]
    keys = set(rows[0])
    for r in rows[1:]:
        keys &= set(r)
    return {
        k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
            "std": round(float(np.std([r[k] for r in rows])), 4)}
        for k in sorted(keys)
    }
