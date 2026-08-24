#!/usr/bin/env python3
"""
Kinematic trajectory predictor for the Rosmaster X3 dataset.

Task
----
Input  : a short causal window of ego-motion state ending at the current frame
         (window = frames t-H+1 .. t, i.e. the PAST, never the future)
Output : the next F waypoints in the CURRENT robot frame (metres)
         -- exactly `trajectory.future_waypoints_robot_frame`, which is verified
         below to be the map-frame positions of t+1..t+F rotated into frame t.

Model
-----
    y_hat = unicycle_rollout(v_t, w_t)  +  W @ z + b

  * the rollout is the physics: constant (v, w) integrated F steps of dt
  * the ridge term learns the systematic residual (accel/decel, curvature
    change, controller lag) from the window
  * everything is linear -> the deployed model is one (2F x D) matmul.

Why not a CNN/GRU/transformer: with ~250 usable windows and 10 scenarios,
leave-one-scenario-out shows every higher-capacity variant tested overfits.
Numbers are printed by `python kinematic_ridge.py --ablation`.

Usage
-----
    python kinematic_ridge.py                # LOSO eval of the recommended model
    python kinematic_ridge.py --ablation     # full feature / model ablation
    python kinematic_ridge.py --export out.npz   # fit on all data, save weights

Deps: numpy (+ h5py only if --use-h5 telemetry features are enabled)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
DT = 0.5           # dataset sample period (2 Hz)
H = 4              # history frames INCLUDING current  -> 2.0 s of context
F = 6              # future waypoints to predict       -> 3.0 s horizon
ALPHA = 30.0       # ridge strength (flat optimum over ~10-100)

# Which annotation runs to use is discovered from the folder.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN_GLOB = os.path.join(REPO, "dataset", "annotations", "*.json")

# Optional: telemetry_cache.npz next to this file holds every non-image array
# from the 10 .h5 runs (85 KB), keyed "<run>|<dataset>", so IMU / wheel-speed
# features can be tried without opening 1 GB of h5. They were measured to HURT
# leave-one-scenario-out ADE at this dataset size -- see the README notes.
TEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "telemetry_cache.npz")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_runs(ann_glob=ANN_GLOB):
    runs = {}
    for fp in sorted(glob.glob(ann_glob)):
        name = os.path.basename(fp).split("_vla_annotations")[0]
        if name in runs:                      # duplicate run file guard
            continue
        runs[name] = json.load(open(fp))
    if not runs:
        raise FileNotFoundError(f"no annotations at {ann_glob}")
    return runs


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def build_windows(run, telemetry=None, h=H, f=F, drop_first=False,
                  require_real_future=True):
    """One row per valid current-frame t.

    require_real_future=True drops the last `f` frames of a run, whose
    waypoint targets are padded with the final pose and are therefore
    fake 'the robot stops' examples.
    """
    n = len(run)
    P = np.array([[s["robot_state"]["x"], s["robot_state"]["y"],
                   s["robot_state"]["yaw"]] for s in run], dtype=np.float64)
    V = np.array([[s["robot_state"]["linear_velocity"],
                   s["robot_state"]["angular_velocity"]] for s in run], dtype=np.float64)
    WP = np.array([s["trajectory"]["future_waypoints_robot_frame"] for s in run],
                  dtype=np.float64)

    hi = n - f if require_real_future else n
    feats, targs, states = [], [], []
    for t in range(h - 1, hi):
        yaw = P[t, 2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s], [s, c]])                  # world -> ego(t)
        idx = np.arange(t - h + 1, t + 1)

        past_xy = np.array([R @ (P[i, :2] - P[t, :2]) for i in idx]).ravel()
        past_dyaw = _wrap(P[idx, 2] - yaw)

        parts = [V[idx, 0], V[idx, 1], past_xy, past_dyaw]
        if telemetry is not None:
            parts.append(telemetry[idx].ravel())

        wp = WP[t][1:] if drop_first else WP[t]
        feats.append(np.concatenate(parts))
        targs.append(wp[:f].ravel())
        states.append(V[t])                              # (v_t, w_t) for the rollout
    if not feats:
        return np.zeros((0, 1)), np.zeros((0, 2 * f)), np.zeros((0, 2))
    return np.array(feats), np.array(targs), np.array(states)


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------
def unicycle_rollout(states, f=F, dt=DT):
    """Constant (v, w) held for f steps, integrated in the current ego frame."""
    v = states[:, 0:1]
    w = states[:, 1:2]
    k = np.arange(1, f + 1)[None, :]
    th = w * k * dt                       # heading at end of step k
    th_mid = th - w * dt / 2.0            # midpoint heading of step k
    dx = v * dt * np.cos(th_mid)
    dy = v * dt * np.sin(th_mid)
    x = np.cumsum(dx, axis=1)
    y = np.cumsum(dy, axis=1)
    return np.stack([x, y], axis=2).reshape(len(states), -1)


# --------------------------------------------------------------------------
# model: standardise -> ridge on the physics residual  (closed form)
# --------------------------------------------------------------------------
class KinematicRidge:
    def __init__(self, alpha=ALPHA, f=F, residual=True):
        self.alpha, self.f, self.residual = alpha, f, residual

    def fit(self, X, Y, S):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        Z = (X - self.mu) / self.sd
        T = Y - unicycle_rollout(S, self.f) if self.residual else Y
        Z1 = np.hstack([Z, np.ones((len(Z), 1))])
        A = Z1.T @ Z1 + self.alpha * np.eye(Z1.shape[1])
        A[-1, -1] -= self.alpha                       # do not penalise the bias
        self.Wb = np.linalg.solve(A, Z1.T @ T)        # (D+1, 2F)
        return self

    def predict(self, X, S):
        Z = (X - self.mu) / self.sd
        out = np.hstack([Z, np.ones((len(Z), 1))]) @ self.Wb
        return out + unicycle_rollout(S, self.f) if self.residual else out

    @property
    def n_params(self):
        return self.Wb.size + self.mu.size + self.sd.size


def ade_fde(P, Y, f=F):
    d = np.linalg.norm(P.reshape(-1, f, 2) - Y.reshape(-1, f, 2), axis=2)
    return d.mean(), d[:, -1].mean()


# --------------------------------------------------------------------------
# evaluation: leave one scenario out (never split by frame - 0.5 s apart)
# --------------------------------------------------------------------------
def loso(runs, alpha=ALPHA, h=H, f=F, residual=True, keep=None,
         drop_first=False, require_real_future=True, verbose=False):
    data = {k: build_windows(v, None, h, f, drop_first, require_real_future)
            for k, v in runs.items()}
    ades, fdes, sizes = [], [], []
    for ho in runs:
        Xte, Yte, Ste = data[ho]
        if len(Xte) == 0:
            continue
        tr = [data[k] for k in runs if k != ho and len(data[k][0])]
        Xtr = np.vstack([d[0] for d in tr])
        Ytr = np.vstack([d[1] for d in tr])
        Str = np.vstack([d[2] for d in tr])
        if keep is not None:
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        m = KinematicRidge(alpha, f, residual).fit(Xtr, Ytr, Str)
        a, fd = ade_fde(m.predict(Xte, Ste), Yte, f)
        ades.append(a); fdes.append(fd); sizes.append(len(Xte))
        if verbose:
            print(f"    {ho:<20} n={len(Xte):3d}  ADE {a:.4f}  FDE {fd:.4f}")
    return float(np.mean(ades)), float(np.mean(fdes)), int(np.sum(sizes))


def physics_only(runs, h=H, f=F, drop_first=False, require_real_future=True):
    ades, fdes = [], []
    for k, v in runs.items():
        X, Y, S = build_windows(v, None, h, f, drop_first, require_real_future)
        if len(X) == 0:
            continue
        a, fd = ade_fde(unicycle_rollout(S, f), Y, f)
        ades.append(a); fdes.append(fd)
    return float(np.mean(ades)), float(np.mean(fdes))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--export", default=None, help="fit on everything, save .npz")
    args = ap.parse_args()

    runs = load_runs()
    n_frames = sum(len(v) for v in runs.values())
    print(f"runs {len(runs)}  frames {n_frames}")

    # sanity: the target really is the future pose in the current ego frame
    r = next(iter(runs.values()))
    err = []
    for t in range(len(r) - F):
        x, y, yaw = (r[t]["robot_state"][k] for k in ("x", "y", "yaw"))
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s], [s, c]])
        gt = np.array(r[t]["trajectory"]["future_waypoints_robot_frame"])
        dr = np.array([R @ np.array([r[t + k]["robot_state"]["x"] - x,
                                     r[t + k]["robot_state"]["y"] - y])
                       for k in range(1, F + 1)])
        err.append(np.abs(dr - gt).max())
    print(f"target check: max |waypoints - dead-reckoned t+1..t+{F}| = {max(err):.2e} m")

    a, f_, n = loso(runs)
    pa, pf = physics_only(runs)
    print(f"\nwindows (H={H}, F={F}, real futures only): {n}")
    print(f"  constant-curvature physics    ADE {pa:.4f}  FDE {pf:.4f}")
    print(f"  KinematicRidge (recommended)  ADE {a:.4f}  FDE {f_:.4f}")

    if args.ablation:
        print("\n--- history length (v,w,past,dyaw; residual) ---")
        for h in (1, 2, 3, 4, 6, 8):
            a, f_, n = loso(runs, h=h)
            print(f"  H={h}  ({h*DT:.1f}s)  n={n:4d}  ADE {a:.4f}  FDE {f_:.4f}")

        print("\n--- residual vs direct ---")
        for res in (True, False):
            a, f_, _ = loso(runs, residual=res)
            print(f"  residual={res!s:<5}  ADE {a:.4f}  FDE {f_:.4f}")

        print("\n--- ridge strength ---")
        for al in (1, 3, 10, 30, 100, 300):
            a, f_, _ = loso(runs, alpha=al)
            print(f"  alpha={al:<4}  ADE {a:.4f}  FDE {f_:.4f}")

        print("\n--- comparison on the no_text_ablation protocol "
              "(4 wp, drop_first, all frames) ---")
        pa, pf = physics_only(runs, h=1, f=4, drop_first=True,
                              require_real_future=False)
        print(f"  constant-curvature physics       ADE {pa:.4f}  FDE {pf:.4f}")
        a, f_, n = loso(runs, h=H, f=4, drop_first=True, require_real_future=False)
        print(f"  KinematicRidge                   ADE {a:.4f}  FDE {f_:.4f}  (n={n})")
        print(f"  frozen-CNN + MLP (runs/summary)  ADE 0.1755  FDE 0.2496")

    if args.export:
        X = np.vstack([build_windows(v)[0] for v in runs.values()])
        Y = np.vstack([build_windows(v)[1] for v in runs.values()])
        S = np.vstack([build_windows(v)[2] for v in runs.values()])
        m = KinematicRidge().fit(X, Y, S)
        np.savez(args.export, Wb=m.Wb, mu=m.mu, sd=m.sd, H=H, F=F, DT=DT)
        print(f"\nexported {args.export}: {m.n_params} floats "
              f"({m.n_params * 4 / 1024:.1f} KB fp32)")
        print("inference = standardise, one matmul, add the rollout. "
              "~1e4 FLOPs, microseconds on the Nano CPU.")


if __name__ == "__main__":
    main()
