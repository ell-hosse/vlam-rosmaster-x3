"""NO-TEXT ablation: the same pipeline with the text embedding removed.

    python vla/no_text_ablation/run_cv.py
    python vla/no_text_ablation/run_cv.py --match-params 1073332

Same inputs (frozen CNN features + telemetry), same folds, same seeds, same
epochs, same optimiser, same head shape. The only difference is that the
trajectory and action heads no longer receive a generated text embedding.

--match-params widens the head trunk until the trainable parameter count
matches the full model, so a gap in results cannot be explained by this
network simply being smaller.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                    # noqa: E402
from data import describe, load_samples, scenarios_of        # noqa: E402
from encoders import image_features                          # noqa: E402
from metrics import action_report, ade_fde, aggregate        # noqa: E402
from model import (                                          # noqa: E402
    VLAModelNoText, parameter_report, solve_hidden_for,
)


class Standardiser:
    """Zero-mean unit-variance, fitted on training rows only."""

    def __init__(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def train_fold(cfg, model, tensors, class_weight, rng_seed):
    torch.manual_seed(rng_seed)
    img, tel, traj, action = tensors
    n = len(img)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            traj_pred, logits = model(img[idx], tel[idx])
            loss = (
                cfg.w_traj * F.smooth_l1_loss(traj_pred, traj[idx])
                + cfg.w_action * F.cross_entropy(logits, action[idx], weight=class_weight)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None, help="limit folds (debug)")
    ap.add_argument(
        "--match-params", type=int, default=0,
        help="widen the head trunk to hit this trainable parameter count "
             "(use the full model's total, e.g. 1073332)",
    )
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.match_params:
        cfg.match_params = args.match_params

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("NO-TEXT ABLATION")
    print("=" * 72)
    samples = load_samples(cfg)
    print(describe(samples, cfg))
    print(f"  cache dir         : {cfg.cache_dir}")

    print("\nFROZEN ENCODER")
    img_all = image_features(samples, cfg)
    tel_all = np.stack([s.telemetry for s in samples])
    wp_all = np.stack([s.waypoints for s in samples])
    act_all = np.array([s.action for s in samples])
    scen_all = np.array([s.scenario for s in samples])

    folds = scenarios_of(samples)
    if args.folds:
        folds = folds[: args.folds]

    print(f"\n  image feats {img_all.shape} | telemetry {tel_all.shape} "
          f"| waypoints {wp_all.shape}")
    print(f"  {len(folds)} folds (leave-one-scenario-out)")
    print("  NOTE: no text embedding anywhere — not as a target, not as an input.")

    hidden = cfg.hidden_dim
    if cfg.match_params:
        hidden = solve_hidden_for(
            cfg.match_params, img_all.shape[1], tel_all.shape[1], cfg
        )
        print(f"  --match-params {cfg.match_params:,} -> trunk width {hidden}")

    per_fold, size_info = [], None

    for fold_i, held_out in enumerate(folds):
        te = np.where(scen_all == held_out)[0]
        tr = np.where(scen_all != held_out)[0]

        # ---- everything below is fitted on TRAIN ONLY ----
        img_std = Standardiser(img_all[tr])
        tel_std = Standardiser(tel_all[tr])
        wp_std = Standardiser(wp_all[tr].reshape(len(tr), -1))

        counts = np.bincount(act_all[tr], minlength=cfg.n_actions).astype(np.float32)
        weight = torch.tensor(
            np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
            / max((1.0 / np.maximum(counts[counts > 0], 1.0)).mean(), 1e-8),
            dtype=torch.float32,
        )

        def pack(idx):
            return (
                torch.tensor(img_std(img_all[idx])),
                torch.tensor(tel_std(tel_all[idx])),
                torch.tensor(wp_std(wp_all[idx].reshape(len(idx), -1))),
                torch.tensor(act_all[idx], dtype=torch.long),
            )

        model = VLAModelNoText(img_all.shape[1], tel_all.shape[1], cfg, hidden=hidden)
        if size_info is None:
            size_info = parameter_report(model)
            print("\nMODEL SIZE (trainable; the frozen CNN is not counted)")
            print(f"  total            : {size_info['total_params']:,} parameters")
            print(f"  size             : {size_info['size_fp32_mb']} MB fp32 / "
                  f"{size_info['size_fp16_mb']} MB fp16")
            for name, n in size_info["by_module"].items():
                print(f"    {name:<10s} {n:>10,}")
            print()

        t0 = time.time()
        train_fold(cfg, model, pack(tr), weight, cfg.seed + fold_i)
        train_secs = time.time() - t0

        model.eval()
        img_te, tel_te, _, _ = pack(te)
        traj_hat, logits = model.predict(img_te, tel_te)
        traj_pred = (
            traj_hat.numpy() * wp_std.std + wp_std.mean
        ).reshape(len(te), cfg.n_waypoints, 2)
        ade, fde = ade_fde(traj_pred, wp_all[te])
        act_pred = logits.argmax(1).numpy()
        rep = action_report(act_pred, act_all[te], cfg.action_classes)

        result = {
            "fold": fold_i,
            "held_out": held_out,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "ade": round(ade, 4),
            "fde": round(fde, 4),
            "accuracy": rep["accuracy"],
            "action": rep,
            "train_seconds": round(train_secs, 1),
        }
        per_fold.append(result)

        fold_dir = cfg.out_dir / f"fold_{fold_i:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "trunk_hidden": hidden,
                "img_mean": img_std.mean, "img_std": img_std.std,
                "tel_mean": tel_std.mean, "tel_std": tel_std.std,
                "wp_mean": wp_std.mean, "wp_std": wp_std.std,
                "config": {k: str(v) for k, v in vars(cfg).items()},
            },
            fold_dir / "model.pt",
        )
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

        print(f"  fold {fold_i:2d}  hold-out {held_out:<20s} "
              f"n_test={len(te):3d}  ADE={ade:.3f}  FDE={fde:.3f}  "
              f"acc={rep['accuracy']:.3f}  [{train_secs:.0f}s]")

    summary = {
        "variant": "no_text_ablation",
        "match_params": cfg.match_params,
        "n_folds": len(per_fold),
        "model_size": size_info,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "average": aggregate(per_fold),
        "folds": per_fold,
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    avg = summary["average"]
    maj = max(np.bincount(act_all, minlength=cfg.n_actions)) / len(act_all)

    print("\n" + "=" * 72)
    print(f"AVERAGE OVER {len(per_fold)} FOLDS   (mean +/- std)")
    print("=" * 72)
    for label, key in (("ADE (m)", "ade"), ("FDE (m)", "fde"),
                       ("action accuracy", "accuracy")):
        if key in avg:
            print(f"  {label:<34s} {avg[key]['mean']:8.4f} +/- {avg[key]['std']:.4f}")
    print(f"  {'majority-class baseline':<34s} {maj:8.4f}")
    print(f"\n  trainable params : {size_info['total_params']:,}")
    print(f"  size             : {size_info['size_fp32_mb']} MB fp32")
    print(f"\n  summary -> {cfg.out_dir / 'summary.json'}")
    print("  compare with the full model:  python vla/no_text_ablation/compare.py")


if __name__ == "__main__":
    main()
