"""Put the full model and the no-text ablation side by side.

    python vla/no_text_ablation/compare.py

Reads both summary.json files. Because the folds are leave-one-scenario-out
with the same ordering and seeds in both runs, the two are paired on the same
held-out scenarios — so a paired test is valid and much more sensitive than
comparing the two means.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "defussion_text_generator" / "runs" / "summary.json"
ABLA = HERE / "runs" / "summary.json"

RULE = "-" * 74


def load(path: Path, label: str):
    if not path.exists():
        sys.exit(f"missing {label} results: {path}\nRun that variant first.")
    return json.loads(path.read_text())


def paired(full_folds, abl_folds, key):
    """Align folds by held-out scenario and return the two aligned arrays."""
    a = {f["held_out"]: f[key] for f in full_folds if key in f}
    b = {f["held_out"]: f[key] for f in abl_folds if key in f}
    shared = sorted(set(a) & set(b))
    return shared, np.array([a[s] for s in shared]), np.array([b[s] for s in shared])


def main():
    full = load(FULL, "full model")
    abl = load(ABLA, "no-text ablation")

    print(f"\n{RULE}")
    print("DOES THE GENERATED TEXT EMBEDDING HELP?")
    print(RULE)
    print(f"  full model      : {full['model_size']['total_params']:>10,} params")
    print(f"  no-text ablation: {abl['model_size']['total_params']:>10,} params"
          + (f"   (--match-params {abl['match_params']:,})" if abl.get("match_params") else ""))
    if not abl.get("match_params"):
        print("\n  NOTE: the ablation is the smaller network. Re-run it with")
        print(f"        --match-params {full['model_size']['total_params']}")
        print("        to rule out capacity as the explanation.")

    for key, label, lower_is_better in (
        ("ade", "ADE (m)", True),
        ("fde", "FDE (m)", True),
        ("accuracy", "action accuracy", False),
    ):
        shared, a, b = paired(full["folds"], abl["folds"], key)
        if not shared:
            continue
        diff = a - b                      # full minus ablation
        better = (diff < 0) if lower_is_better else (diff > 0)

        print(f"\n{RULE}\n{label}\n{RULE}")
        print(f"  {'held-out scenario':<24} {'full':>9} {'no-text':>9} {'delta':>9}")
        for s, x, y in zip(shared, a, b):
            mark = "  <-- text helps" if (
                (x < y) if lower_is_better else (x > y)
            ) else ""
            print(f"  {s:<24} {x:9.4f} {y:9.4f} {x - y:+9.4f}{mark}")

        print(f"\n  mean            full {a.mean():.4f}   no-text {b.mean():.4f}"
              f"   delta {diff.mean():+.4f}")
        print(f"  folds where text helps: {int(better.sum())}/{len(shared)}")

        # paired t-test, computed directly (no scipy dependency)
        n = len(diff)
        if n > 1 and diff.std(ddof=1) > 0:
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
            print(f"  paired t({n - 1}) = {t:+.3f}"
                  f"   |t| > ~2.26 is p < 0.05 at n=10")
        else:
            print("  (no variation in the differences)")

    print(f"\n{RULE}")
    print("  Read this together with min-ADE vs ADE in the full model's summary:")
    print("  if best-of-5 sampling barely beats a single sample, the generative")
    print("  head has collapsed to a point estimate, and any gap here is coming")
    print("  from the extra supervision, not from sampling.")
    print(RULE + "\n")


if __name__ == "__main__":
    main()
