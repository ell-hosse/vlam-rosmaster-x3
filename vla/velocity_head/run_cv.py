"""Leave-one-scenario-out CV for the velocity head + frozen trajectory calculator.

    python vla/velocity_head/run_cv.py

Fold i holds out scenario S. Everything fitted on data is refit inside the fold
from the other nine scenarios only: the text PCA, the feature standardisers,
the command standardiser, the class weights, the command prototypes -- AND the
stage-2 plant model, which is loaded from
vla/trajectory_predictor/models/fold_*_S.json, i.e. the calculator that was
also identified without ever seeing S. That pairing is the whole reason the
calculator was exported per fold.

Per fold you get, separately:
    loss / error on v      (m/s)
    loss / error on w      (rad/s)
    ADE and FDE of the trajectory the predicted commands produce (m)
    the same ADE/FDE with the TRUE commands  -- the calculator's own floor
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                              # noqa: E402
from data import (                                                     # noqa: E402
    command_prototypes, describe, load_samples, scenarios_of, tc,
)
from encoders import image_features, text_matrix                       # noqa: E402
from latent import TextLatent                                          # noqa: E402
from metrics import (                                                  # noqa: E402
    action_report, ade_fde, aggregate, min_ade_fde, text_report, velocity_report,
)
from model import VelocityVLA, parameter_report                        # noqa: E402


class Standardiser:
    """Zero-mean unit-variance, fitted on training rows only."""

    def __init__(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)

    def __call__(self, x): return ((x - self.mean) / self.std).astype(np.float32)

    def invert(self, x): return x * self.std + self.mean


def load_fold_calculator(cfg: Config, held_out: str):
    """The plant model identified WITHOUT the held-out scenario."""
    hits = sorted(glob.glob(str(cfg.calculator_dir / f"fold_*_{held_out}.json")))
    if not hits:
        present = sorted(p.name for p in cfg.calculator_dir.glob("*.json")) \
            if cfg.calculator_dir.is_dir() else []
        raise FileNotFoundError(
            f"no stage-2 model for held-out scenario {held_out!r} in "
            f"{cfg.calculator_dir}\n"
            f"    directory exists : {cfg.calculator_dir.is_dir()}\n"
            f"    .json files found: {present if present else 'none'}\n"
            "    Export them, or point at an existing folder:\n"
            "        cd vla/trajectory_predictor\n"
            "        python trajectory_calculator.py --export models\n"
            "        python run_cv.py --calculator-dir "
            "../trajectory_predictor/<your-folder>"
        )
    blob = json.loads(Path(hits[0]).read_text())
    return tc.PlantModel.from_dict(blob), Path(hits[0]).name


def rollout(plant, chunks: np.ndarray, warms: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """(N, F, 2) commands -> (N, F, 2) waypoints, through the frozen calculator."""
    return np.stack([plant(c, w, d) for c, w, d in zip(chunks, warms, dts)])


def train_fold(cfg, model, tensors, class_weight, rng_seed):
    torch.manual_seed(rng_seed)
    img, tel, z_true, v_true, w_true, action = tensors
    n = len(img)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    model.train()
    last = {}
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        acc = {"flow": 0.0, "v": 0.0, "w": 0.0, "act": 0.0}
        nb = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            bi, bt, bz = img[idx], tel[idx], z_true[idx]
            bv, bw, ba = v_true[idx], w_true[idx], action[idx]
            b = len(idx)

            cond = model.condition(bi, bt)

            # ---- rectified flow: straight line from noise to the text latent ----
            s = torch.rand(b, 1)
            eps = torch.randn_like(bz)
            z_s = (1 - s) * eps + s * bz
            loss_flow = Fn.mse_loss(model.flow(z_s, s.squeeze(1), cond), bz - eps)

            # ---- command head, fed the true latent or a sampled one ----
            if torch.rand(()) < cfg.p_sampled_z:
                with torch.no_grad():
                    z_in = model.flow.sample(cond, cfg.flow_steps)
            else:
                z_in = bz
            v_pred, w_pred, logits = model.head(cond, z_in)

            loss_v = Fn.smooth_l1_loss(v_pred, bv)        # linear velocity chunk
            loss_w = Fn.smooth_l1_loss(w_pred, bw)        # angular velocity chunk
            loss_act = Fn.cross_entropy(logits, ba, weight=class_weight)

            loss = (cfg.w_flow * loss_flow + cfg.w_vel * loss_v
                    + cfg.w_ang * loss_w + cfg.w_action * loss_act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            acc["flow"] += loss_flow.detach().item()
            acc["v"] += loss_v.detach().item()
            acc["w"] += loss_w.detach().item()
            acc["act"] += loss_act.detach().item()
            nb += 1
        sched.step()
        last = {k: round(x / max(nb, 1), 5) for k, x in acc.items()}
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--text-field", type=str, default=None)
    ap.add_argument("--hist", type=int, default=None)
    ap.add_argument("--calculator-dir", type=str, default=None,
                    help="folder holding the exported stage-2 fold_*.json "
                         "(auto-discovered under vla/trajectory_predictor/)")
    ap.add_argument("--folds", type=int, default=None, help="limit folds (debug)")
    args = ap.parse_args()

    cfg = Config()
    if args.calculator_dir:
        cfg.calculator_dir = Path(args.calculator_dir).expanduser().resolve()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.text_field:
        cfg.text_field = args.text_field
    if args.hist:
        cfg.hist = args.hist

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("DATA")
    print("=" * 72)
    samples = load_samples(cfg)
    print(describe(samples, cfg))

    print("\nFROZEN ENCODERS")
    img_all = image_features(samples, cfg)
    txt_all, _ = text_matrix(samples, cfg)
    tel_all = np.stack([s.tel for s in samples])
    chunk_all = np.stack([s.chunk for s in samples]).reshape(len(samples), cfg.horizon, 2)
    wp_all = np.stack([s.wp for s in samples])
    dt_all = np.stack([s.dt for s in samples])
    warm_all = np.stack([s.warm for s in samples])
    act_all = np.array([s.action for s in samples])
    scen_all = np.array([s.scenario for s in samples])
    texts_all = [s.text for s in samples]

    folds = scenarios_of(samples)
    if args.folds:
        folds = folds[:args.folds]

    print(f"\n  stage-2 models <- {cfg.calculator_dir}")
    print(f"  image {img_all.shape} | history {tel_all.shape} | text {txt_all.shape}")
    print(f"  command chunks {chunk_all.shape} | waypoints {wp_all.shape}")
    print(f"  {len(folds)} folds (leave-one-scenario-out)")

    per_fold, size_info = [], None

    for fold_i, held_out in enumerate(folds):
        te = np.where(scen_all == held_out)[0]
        tr = np.where(scen_all != held_out)[0]

        plant, plant_file = load_fold_calculator(cfg, held_out)

        # ---------- everything below is fitted on TRAIN ONLY ----------
        img_std = Standardiser(img_all[tr])
        tel_std = Standardiser(tel_all[tr])
        v_std = Standardiser(chunk_all[tr, :, 0])
        w_std = Standardiser(chunk_all[tr, :, 1])
        latent = TextLatent(txt_all[tr], cfg.pca_dim)
        protos = command_prototypes(samples, tr)

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
                torch.tensor(v_std(chunk_all[idx, :, 0])),
                torch.tensor(w_std(chunk_all[idx, :, 1])),
                torch.tensor(act_all[idx], dtype=torch.long),
            )

        model = VelocityVLA(img_all.shape[1], tel_all.shape[1], latent.dim, cfg)
        if size_info is None:
            size_info = parameter_report(model)
            print("\nMODEL SIZE (trainable; the frozen CNN is not counted)")
            print(f"  total            : {size_info['total_params']:,} parameters")
            print(f"  size             : {size_info['size_fp32_mb']} MB fp32 / "
                  f"{size_info['size_fp16_mb']} MB fp16")
            for name, k in size_info["by_module"].items():
                print(f"    {name:<10s} {k:>10,}")
            print(f"  stage-2 plant    : 4 parameters, frozen, no gradient\n")

        t0 = time.time()
        train_losses = train_fold(cfg, model, pack(tr), weight, cfg.seed + fold_i)
        train_secs = time.time() - t0

        # ------------------------------ evaluate ------------------------------
        model.eval()
        img_te, tel_te, _, _, _, _ = pack(te)
        gen = torch.Generator().manual_seed(cfg.seed + fold_i)
        z_hat, v_hat, w_hat, logits = model.predict(img_te, tel_te, cfg.flow_steps, gen)

        pred_chunk = np.stack(
            [v_std.invert(v_hat.numpy()), w_std.invert(w_hat.numpy())], axis=2
        )
        true_chunk = chunk_all[te]

        vel = velocity_report(pred_chunk, true_chunk, protos)

        # ---- stage 2: commands -> trajectory, through the fold's calculator ----
        traj_pred = rollout(plant, pred_chunk, warm_all[te], dt_all[te])
        ade, fde = ade_fde(traj_pred, wp_all[te])

        traj_oracle = rollout(plant, true_chunk, warm_all[te], dt_all[te])
        ade_o, fde_o = ade_fde(traj_oracle, wp_all[te])

        # baseline: keep issuing the last command the robot actually sent
        hold = np.repeat(warm_all[te][:, -1:, :], cfg.horizon, axis=1)
        ade_h, fde_h = ade_fde(rollout(plant, hold, warm_all[te], dt_all[te]), wp_all[te])

        # best-of-K from the generative head
        multi = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator().manual_seed(1000 * (fold_i + 1) + k)
            _, v_k, w_k, _ = model.predict(img_te, tel_te, cfg.flow_steps, g)
            c_k = np.stack([v_std.invert(v_k.numpy()), w_std.invert(w_k.numpy())], axis=2)
            multi.append(rollout(plant, c_k, warm_all[te], dt_all[te]))
        m_ade, m_fde = min_ade_fde(np.stack(multi), wp_all[te])

        txt = text_report(
            pred_emb=latent.decode(z_hat.numpy()),
            true_emb=txt_all[te],
            ceiling_emb=latent.reconstruct(txt_all[te]),
            bank_emb=bank_emb, bank_texts=bank_texts,
            true_texts=[texts_all[i] for i in te],
            mean_emb=train_mean_emb,
        )

        sweep = {}
        for steps in cfg.step_sweep:
            g = torch.Generator().manual_seed(cfg.seed + fold_i)
            t_start = time.time()
            _, v_s, w_s, _ = model.predict(img_te, tel_te, steps, g)
            ms = (time.time() - t_start) * 1000 / max(len(te), 1)
            c_s = np.stack([v_std.invert(v_s.numpy()), w_std.invert(w_s.numpy())], axis=2)
            a_s, f_s = ade_fde(rollout(plant, c_s, warm_all[te], dt_all[te]), wp_all[te])
            sweep[str(steps)] = {
                "ade": round(a_s, 4), "fde": round(f_s, 4),
                "v_mae": round(float(np.abs(c_s[..., 0] - true_chunk[..., 0]).mean()), 4),
                "w_mae": round(float(np.abs(c_s[..., 1] - true_chunk[..., 1]).mean()), 4),
                "ms_per_sample": round(ms, 3),
            }

        result = {
            "fold": fold_i,
            "held_out": held_out,
            "stage2_model": plant_file,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "latent_dim": latent.dim,
            "pca_explained": round(latent.explained, 4),
            "train_loss": train_losses,
            "velocity": vel,
            "ade": round(ade, 4),
            "fde": round(fde, 4),
            "ade_oracle_commands": round(ade_o, 4),
            "fde_oracle_commands": round(fde_o, 4),
            "ade_hold_last_command": round(ade_h, 4),
            "fde_hold_last_command": round(fde_h, 4),
            f"min_ade_k{cfg.n_samples_eval}": round(m_ade, 4),
            f"min_fde_k{cfg.n_samples_eval}": round(m_fde, 4),
            "action": action_report(logits.argmax(1).numpy(), act_all[te],
                                    cfg.action_classes),
            "accuracy": action_report(logits.argmax(1).numpy(), act_all[te],
                                      cfg.action_classes)["accuracy"],
            **{f"text_{k}": v for k, v in txt.items()},
            "step_sweep": sweep,
            "train_seconds": round(train_secs, 1),
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
                "v_mean": v_std.mean, "v_std": v_std.std,
                "w_mean": w_std.mean, "w_std": w_std.std,
                "prototypes": protos,
                "bank_texts": bank_texts,
                "stage2": plant.to_dict(), "stage2_file": plant_file,
                "config": vars(cfg),
            },
            fold_dir / "model.pt",
        )
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

        print(
            f"  fold {fold_i:2d} {held_out:<20s} n={len(te):3d}  "
            f"v_mae={vel['v']['mae']:.4f}  w_mae={vel['w']['mae']:.4f}  "
            f"ADE={ade:.4f} (oracle {ade_o:.4f})  FDE={fde:.4f}  "
            f"acc={result['accuracy']:.3f}  [{train_secs:.0f}s]"
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

    def row(label, key, unit=""):
        if key in avg:
            print(f"  {label:<40s} {avg[key]['mean']:8.4f} +/- {avg[key]['std']:.4f}{unit}")

    print("\n" + "=" * 72)
    print(f"AVERAGE OVER {len(per_fold)} FOLDS   (mean +/- std)")
    print("=" * 72)
    print("  --- stage 1: commands ---")
    row("v  MAE (m/s)", "velocity.v.mae")
    row("v  RMSE (m/s)", "velocity.v.rmse")
    row("v  R^2", "velocity.v.r2")
    row("w  MAE (rad/s)", "velocity.w.mae")
    row("w  RMSE (rad/s)", "velocity.w.rmse")
    row("w  R^2", "velocity.w.r2")
    row("exact command match (snapped)", "velocity.snapped_exact_match")
    row("exact match, current step only", "velocity.snapped_exact_match_step0")
    print()
    print("  --- stage 2: trajectory through the frozen calculator ---")
    row("ADE (m)", "ade")
    row("FDE (m)", "fde")
    row(f"min-ADE over {cfg.n_samples_eval} samples (m)", f"min_ade_k{cfg.n_samples_eval}")
    print()
    row("ADE with TRUE commands (floor)", "ade_oracle_commands")
    row("ADE holding the last command", "ade_hold_last_command")
    print()
    print("  --- auxiliary ---")
    row("action accuracy (current step)", "accuracy")
    row("text cosine, model", "text_cosine_model")
    row("text cosine, PCA ceiling", "text_cosine_pca_ceiling")
    row("text retrieval accuracy", "text_retrieval_accuracy")
    print()
    print(f"  trainable params : {size_info['total_params']:,} "
          f"({size_info['size_fp16_mb']} MB fp16)  + 4 frozen plant parameters")
    print(f"  per-fold metrics -> {cfg.out_dir}")
    print(f"  summary          -> {cfg.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
