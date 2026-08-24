"""Make the `cos=` number interpretable.

    python vla/defussion_text_generator/explain_text_metric.py

Cosine between two sentence embeddings is not a 0..1 scale where 0 means
"unrelated". When every target string is a near-clone of the others, even a
completely wrong prediction scores high. This script measures that floor
directly, then rescales the reported cosine against it.

Reads only the caches and whatever fold metrics already exist, so it is safe
to run against a partially finished cross-validation.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config          # noqa: E402

RULE = "-" * 74


def load_text_cache(cfg: Config):
    path = cfg.cache_dir / f"text_{Path(cfg.text_encoder).name}_{cfg.text_field}.npz"
    if not path.exists():
        raise SystemExit(
            f"no text cache at {path}\nRun run_cv.py at least once first."
        )
    blob = np.load(path, allow_pickle=True)
    return [str(t) for t in blob["texts"]], blob["emb"]


def main():
    cfg = Config()
    texts, emb = load_text_cache(cfg)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    n = len(texts)

    print(f"\n{RULE}\nHOW SIMILAR ARE THE TARGET STRINGS TO EACH OTHER?\n{RULE}")
    print(f"{n} unique '{cfg.text_field}' strings\n")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    sim = emb @ emb.T
    off = sim[~np.eye(n, dtype=bool)]
    print(f"\n  pairwise cosine between DIFFERENT strings:")
    print(f"    min {off.min():.4f}   mean {off.mean():.4f}   max {off.max():.4f}")
    print(f"\n  -> {off.mean():.4f} is roughly the score a WRONG prediction still gets.")
    print(f"     Read every cosine in the report against this floor, not against 0.")

    print(f"\n  closest confusable pair:")
    iu = np.triu_indices(n, 1)
    k = np.argmax(sim[iu])
    a, b = iu[0][k], iu[1][k]
    print(f"    {sim[a, b]:.4f}   [{a}] vs [{b}]")

    # ---------------------------------------------------------------- folds
    fold_files = sorted(glob.glob(str(cfg.out_dir / "*" / "metrics.json")))
    if not fold_files:
        print(f"\n  (no fold metrics yet under {cfg.out_dir})")
        return

    print(f"\n{RULE}\nWHERE THE MODEL SITS BETWEEN BASELINE AND CEILING\n{RULE}")
    print(f"  {'fold':<22} {'model':>8} {'mean-bl':>8} {'ceiling':>8} "
          f"{'scaled':>8} {'retr':>7} {'cover':>7}")

    rows = []
    for fp in fold_files:
        m = json.loads(Path(fp).read_text())
        model = m.get("text_cosine_model")
        base = m.get("text_cosine_mean_baseline")
        ceil = m.get("text_cosine_pca_ceiling")
        if None in (model, base, ceil):
            continue
        span = ceil - base
        scaled = (model - base) / span if abs(span) > 1e-6 else float("nan")
        rows.append((scaled, m.get("text_retrieval_accuracy")))
        print(f"  {m['held_out']:<22} {model:8.4f} {base:8.4f} {ceil:8.4f} "
              f"{scaled:8.3f} {m.get('text_retrieval_accuracy', float('nan')):7.3f} "
              f"{m.get('text_bank_coverage', float('nan')):7.3f}")

    if rows:
        sc = np.array([r[0] for r in rows], dtype=float)
        rt = np.array([r[1] for r in rows], dtype=float)
        print(f"\n  scaled score = (model - mean_baseline) / (ceiling - mean_baseline)")
        print(f"    1.0 = perfect;  0.0 = no better than predicting the average;")
        print(f"    negative = worse than predicting the average.")
        print(f"\n  mean over {len(rows)} folds:  scaled {np.nanmean(sc):.3f}"
              f"   retrieval {np.nanmean(rt):.3f}")

    # majority-text baseline for retrieval, from the full dataset
    counts = {}
    for fp in sorted(glob.glob(str(cfg.dataset_dir / "annotations" / "*.json"))):
        for rec in json.loads(Path(fp).read_text()):
            key = (rec["trajectory"][cfg.text_field]
                   if cfg.text_field.startswith("trajectory") else rec[cfg.text_field])
            counts[key] = counts.get(key, 0) + 1
    if counts:
        total = sum(counts.values())
        top = max(counts.values())
        print(f"\n  retrieval baseline (always answer the most common string): "
              f"{top / total:.3f}")
        print(f"  -> retrieval accuracy is the honest headline here: it is discrete,")
        print(f"     so the compressed cosine scale cannot flatter it.")


if __name__ == "__main__":
    main()
