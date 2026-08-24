"""Leave-one-scenario-out cross-validation: 10 runs, 9 scenarios train, 1 test.

    python vla/defussion_text_generator/run_cv.py

Everything fitted on data — PCA, feature standardisation, waypoint scaling,
class weights — is refit inside each fold from the training scenarios only.
The frozen CNN and sentence encoder are pretrained and never see a label, so
running them over the whole dataset up front is not leakage.
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
from encoders import image_features, text_matrix             # noqa: E402
from latent import TextLatent                                # noqa: E402
from metrics import (                                        # noqa: E402
    action_report, ade_fde, aggregate, min_ade_fde, text_report,
)
from model import VLAModel, parameter_report                 # noqa: E402


class Standardiser:
    """Zero-mean unit-variance, fitted on training rows only."""

    def __init__(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def train_fold(cfg, model, tensors, class_weight, rng_seed):
    torch.manual_seed(rng_seed)
    img, tel, z_true, traj, action = tensors
    n = len(img)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            bi, bt = img[idx], tel[idx]
            bz, btr, ba = z_true[idx], traj[idx], action[idx]
            b = len(idx)

            cond = model.condition(bi, bt)

            # ---- rectified flow: straight line from noise to data ----
            s = torch.rand(b, 1)
            eps = torch.randn_like(bz)
            z_s = (1 - s) * eps + s * bz
            v_target = bz - eps
            v_pred = model.flow(z_s, s.squeeze(1), cond)
            loss_flow = F.mse_loss(v_pred, v_target)

            # ---- downstream heads, fed the true z or a sampled one ----
            if torch.rand(()) < cfg.p_sampled_z:
                with torch.no_grad():
                    z_in = model.flow.sample(cond, cfg.flow_steps)
            else:
                z_in = bz
            traj_pred, logits = model.head(cond, z_in)
            loss_traj = F.smooth_l1_loss(traj_pred, btr)
            loss_act = F.cross_entropy(logits, ba, weight=class_weight)

            loss = (
                cfg.w_flow * loss_flow
                + cfg.w_traj * loss_traj
                + cfg.w_action * loss_act
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
    ap.add_argument("--text-field", type=str, default=None)
    ap.add_argument("--folds", type=int, default=None, help="limit folds (debug)")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.text_field:
        cfg.text_field = args.text_field

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("DATA")
    print("=" * 72)
    samples = load_samples(cfg)
    print(describe(samples, cfg))

    print("\nFROZEN ENCODERS")
    img_all = image_features(samples, cfg)
    txt_all, vocab = text_matrix(samples, cfg)
    tel_all = np.stack([s.telemetry for s in samples])
    wp_all = np.stack([s.waypoints for s in samples])          # (N, T, 2)
    act_all = np.array([s.action for s in samples])
    scen_all = np.array([s.scenario for s in samples])
    texts_all = [s.text for s in samples]

    folds = scenarios_of(samples)
    if args.folds:
        folds = folds[: args.folds]

    print(f"\n  image feats {img_all.shape} | telemetry {tel_all.shape} "
          f"| text {txt_all.shape} | waypoints {wp_all.shape}")
    print(f"  {len(folds)} folds (leave-one-scenario-out)")

    per_fold, size_info = [], None

    for fold_i, held_out in enumerate(folds):
        te = np.where(scen_all == held_out)[0]
        tr = np.where(scen_all != held_out)[0]

        # ---- everything below is fitted on TRAIN ONLY ----
        img_std = Standardiser(img_all[tr])
        tel_std = Standardiser(tel_all[tr])
        wp_flat_tr = wp_all[tr].reshape(len(tr), -1)
        wp_std = Standardiser(wp_flat_tr)
        latent = TextLatent(txt_all[tr], cfg.pca_dim)

        bank_texts = sorted(set(texts_all[i] for i in tr))
        bank_emb = np.stack(
            [txt_all[[i for i in tr if texts_all[i] == t][0]] for t in bank_texts]
        )
        train_mean_emb = txt_all[tr].mean(axis=0)
        train_mean_emb /= max(np.linalg.norm(train_mean_emb), 1e-8)

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
                torch.tensor(latent.encode(txt_all[idx]), dtype=torch.float32),
                torch.tensor(wp_std(wp_all[idx].reshape(len(idx), -1))),
                torch.tensor(act_all[idx], dtype=torch.long),
            )

        model = VLAModel(img_all.shape[1], tel_all.shape[1], latent.dim, cfg)
        if size_info is None:
            size_info = parameter_report(model)
            # printed before the first fold trains, not only in the summary
            print("\nMODEL SIZE (trainable; the frozen CNN is not counted)")
            print(f"  total            : {size_info['total_params']:,} parameters")
            print(f"  size             : {size_info['size_fp32_mb']} MB fp32 / "
                  f"{size_info['size_fp16_mb']} MB fp16")
            for name, n in size_info["by_module"].items():
                print(f"    {name:<10s} {n:>10,}")
            print(f"  flow head cost   : ~{2 * size_info['flow_per_step_params'] / 1e6:.2f} "
                  f"MFLOP per Euler step after the first\n")

        t0 = time.time()
        train_fold(cfg, model, pack(tr), weight, cfg.seed + fold_i)
        train_secs = time.time() - t0

        # ------------------------------- evaluate -------------------------------
        model.eval()
        img_te, tel_te, _, _, act_te = pack(te)
        gen = torch.Generator().manual_seed(cfg.seed + fold_i)

        z_hat, traj_hat, logits = model.predict(img_te, tel_te, cfg.flow_steps, gen)
        traj_pred = (
            traj_hat.numpy() * wp_std.std + wp_std.mean
        ).reshape(len(te), cfg.n_waypoints, 2)
        ade, fde = ade_fde(traj_pred, wp_all[te])
        act_pred = logits.argmax(1).numpy()

        # best-of-K from the generative head
        multi = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator().manual_seed(1000 * (fold_i + 1) + k)
            _, t_k, _ = model.predict(img_te, tel_te, cfg.flow_steps, g)
            multi.append((t_k.numpy() * wp_std.std + wp_std.mean)
                         .reshape(len(te), cfg.n_waypoints, 2))
        m_ade, m_fde = min_ade_fde(np.stack(multi), wp_all[te])

        # text-embedding fidelity
        pred_emb = latent.decode(z_hat.numpy())
        txt = text_report(
            pred_emb=pred_emb,
            true_emb=txt_all[te],
            ceiling_emb=latent.reconstruct(txt_all[te]),
            bank_emb=bank_emb,
            bank_texts=bank_texts,
            true_texts=[texts_all[i] for i in te],
            mean_emb=train_mean_emb,
        )

        # how quality varies with the number of Euler steps
        sweep = {}
        for steps in cfg.step_sweep:
            g = torch.Generator().manual_seed(cfg.seed + fold_i)
            t_start = time.time()
            z_s, tr_s, lg_s = model.predict(img_te, tel_te, steps, g)
            ms = (time.time() - t_start) * 1000 / max(len(te), 1)
            p = (tr_s.numpy() * wp_std.std + wp_std.mean).reshape(len(te), cfg.n_waypoints, 2)
            a_s, f_s = ade_fde(p, wp_all[te])
            sweep[str(steps)] = {
                "ade": round(a_s, 4),
                "fde": round(f_s, 4),
                "acc": round(float((lg_s.argmax(1).numpy() == act_all[te]).mean()), 4),
                "cosine": round(float(np.mean(
                    (latent.decode(z_s.numpy()) * txt_all[te]).sum(-1))), 4),
                "ms_per_sample": round(ms, 3),
            }

        result = {
            "fold": fold_i,
            "held_out": held_out,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "latent_dim": latent.dim,
            "pca_explained": round(latent.explained, 4),
            "ade": round(ade, 4),
            "fde": round(fde, 4),
            f"min_ade_k{cfg.n_samples_eval}": round(m_ade, 4),
            f"min_fde_k{cfg.n_samples_eval}": round(m_fde, 4),
            "train_seconds": round(train_secs, 1),
            **{f"text_{k}": v for k, v in txt.items()},
            "action": action_report(act_pred, act_all[te], cfg.action_classes),
            "accuracy": action_report(act_pred, act_all[te], cfg.action_classes)["accuracy"],
            "step_sweep": sweep,
        }
        per_fold.append(result)

        fold_dir = cfg.out_dir / f"fold_{fold_i:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "latent_dim": latent.dim,
                "pca_mean": latent.mean, "pca_basis": latent.basis,
                "pca_scale": latent.scale,
                "img_mean": img_std.mean, "img_std": img_std.std,
                "tel_mean": tel_std.mean, "tel_std": tel_std.std,
                "wp_mean": wp_std.mean, "wp_std": wp_std.std,
                "bank_texts": bank_texts, "config": vars(cfg),
            },
            fold_dir / "model.pt",
        )
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

        print(
            f"  fold {fold_i:2d}  hold-out {held_out:<20s} "
            f"n_test={len(te):3d}  ADE={ade:.3f}  FDE={fde:.3f}  "
            f"acc={result['accuracy']:.3f}  cos={txt['cosine_model']:.3f} "
            f"(ceiling {txt['cosine_pca_ceiling']:.3f})  [{train_secs:.0f}s]"
        )

    # ------------------------------- summary -------------------------------
    summary = {
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

    def row(label, key, unit=""):
        if key in avg:
            print(f"  {label:<34s} {avg[key]['mean']:8.4f} +/- {avg[key]['std']:.4f}{unit}")

    row("ADE (m)", "ade")
    row("FDE (m)", "fde")
    row(f"min-ADE over {cfg.n_samples_eval} samples (m)", f"min_ade_k{cfg.n_samples_eval}")
    row(f"min-FDE over {cfg.n_samples_eval} samples (m)", f"min_fde_k{cfg.n_samples_eval}")
    print()
    row("action accuracy", "accuracy")
    print(f"  {'majority-class baseline':<34s} {maj:8.4f}")
    print()
    print("  --- generated text embedding vs. real ---")
    row("cosine, model", "text_cosine_model")
    row("cosine, PCA ceiling", "text_cosine_pca_ceiling")
    row("cosine, train-mean baseline", "text_cosine_mean_baseline")
    row("retrieval accuracy", "text_retrieval_accuracy")
    row("bank coverage", "text_bank_coverage")
    print()
    print(f"  trainable params : {size_info['total_params']:,}")
    print(f"  size             : {size_info['size_fp32_mb']} MB fp32 / "
          f"{size_info['size_fp16_mb']} MB fp16")
    print(f"  by module        : {size_info['by_module']}")
    print(f"\n  models + per-fold metrics -> {cfg.out_dir}")
    print(f"  summary                   -> {cfg.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
