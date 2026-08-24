#!/usr/bin/env python3
"""
Stage-2 trajectory calculator for the Rosmaster X3 VLA.

    VLA  ->  action chunk  u[t..t+5] = (v_cmd, w_cmd) x 6
                      |
                      v
    this module  ->  6 waypoints in the current robot frame, 3 s ahead

WHAT IS ACTUALLY BEING LEARNED
------------------------------
`robot_state.linear_velocity / angular_velocity` in the annotations are byte-
identical to the h5 `cmd_vel` -- they are the COMMANDS, not measurements.
So the map "commands -> path" is not pure kinematics; it is the robot's
actuator response (lag + slip) followed by exact integration.

The kinematic half is known exactly and is imposed, not learned:

    theta_k = theta_{k-1} + w_k * dt_k
    x_k     = x_{k-1} + v_k * dt_k * cos(theta_{k-1} + w_k*dt_k/2)
    y_k     = y_{k-1} + v_k * dt_k * sin(theta_{k-1} + w_k*dt_k/2)

The dynamic half is identified from data (this is the "equation extraction"):

    v_k = a_v * v_{k-1} + b_v * u_v,k          first-order lag
    w_k = a_w * w_{k-1} + b_w * u_w,k          first-order lag

FOUR learned numbers. Fitted on 9 runs, tested on the 10th; the fold-to-fold
spread of every coefficient is under 3%.

A discovered third term  w_k += c * u_w,k * |u_v,k|  (yaw authority grows with
forward speed: spin-in-place loses ~38% of the commanded rate, driving-and-
turning slightly overshoots) raises one-step R^2 but does NOT improve
trajectory ADE, so it is off by default -- keep it as an ablation row.

INPUTS AT INFERENCE
-------------------
Only commands. 2 past commands (to warm the lag state) + the 6-step chunk.
No IMU, no wheel encoders, no odometry, no image. Measured odometry velocity
was tested as the initial state and was *worse* than replaying commands,
because it is a finite difference over a jittery dt.

Usage
-----
    python trajectory_calculator.py                 # LOSO evaluation table
    python trajectory_calculator.py --equation      # print the identified model
    python trajectory_calculator.py --export models/   # 10 fold models + all-data
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DT_NOMINAL = 0.5      # fallback when true timestamps are unavailable
F = 6                 # waypoints predicted
J = 2                 # past commands replayed to initialise the lag state

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN_GLOB = os.path.join(REPO, "dataset", "annotations", "*.json")
TEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "telemetry_cache.npz")


# ==========================================================================
# the model
# ==========================================================================
class PlantModel:
    """v_k = a_v v_{k-1} + b_v u_v ;  w_k = a_w w_{k-1} + b_w u_w [+ c u_w|u_v|]"""

    def __init__(self, a_v, b_v, a_w, b_w, c_w=0.0):
        self.a_v, self.b_v, self.a_w, self.b_w, self.c_w = a_v, b_v, a_w, b_w, c_w

    # ---- identification ---------------------------------------------------
    @staticmethod
    def identify(steps, rich=False):
        """steps: list of (v_prev, w_prev, u_v, u_w, v_real, w_real)."""
        s = np.asarray(steps, dtype=np.float64)
        Av = np.stack([s[:, 0], s[:, 2]], 1)
        Aw = (np.stack([s[:, 1], s[:, 3], s[:, 3] * np.abs(s[:, 2])], 1) if rich
              else np.stack([s[:, 1], s[:, 3]], 1))
        cv, *_ = np.linalg.lstsq(Av, s[:, 4], rcond=None)
        cw, *_ = np.linalg.lstsq(Aw, s[:, 5], rcond=None)
        return PlantModel(cv[0], cv[1], cw[0], cw[1], cw[2] if rich else 0.0)

    # ---- physical read-out ------------------------------------------------
    def describe(self, dt=DT_NOMINAL):
        tau_v = -dt / np.log(abs(self.a_v))
        tau_w = -dt / np.log(abs(self.a_w))
        k_v = self.b_v / (1 - self.a_v)
        k_w0 = self.b_w / (1 - self.a_w)
        k_w5 = (self.b_w + self.c_w * 0.5) / (1 - self.a_w)
        return (f"v[k] = {self.a_v:.3f}*v[k-1] + {self.b_v:.3f}*u_v"
                f"    (tau {tau_v:.2f}s, steady-state gain {k_v:.3f})\n"
                f"w[k] = {self.a_w:.3f}*w[k-1] + {self.b_w:.3f}*u_w"
                + (f" + {self.c_w:.3f}*u_w*|u_v|" if self.c_w else "")
                + f"    (tau {tau_w:.2f}s, gain {k_w0:.2f} at rest"
                + (f", {k_w5:.2f} at 0.5 m/s" if self.c_w else "") + ")")

    # ---- the actual calculator -------------------------------------------
    def __call__(self, u_chunk, u_past=None, dts=None):
        """u_chunk: (F,2) planned commands. u_past: (J,2) commands already sent.
        Returns (F,2) waypoints in the robot frame at t, metres."""
        u_chunk = np.asarray(u_chunk, dtype=np.float64)
        f = len(u_chunk)
        dts = np.full(f, DT_NOMINAL) if dts is None else np.asarray(dts, float)
        v = w = 0.0
        for uv, uw in (np.asarray(u_past, float) if u_past is not None else []):
            v, w = self._step(v, w, uv, uw)
        th = x = y = 0.0
        out = np.empty((f, 2))
        for k in range(f):
            v, w = self._step(v, w, u_chunk[k, 0], u_chunk[k, 1])
            h = dts[k]
            th_mid = th + w * h / 2.0
            th += w * h
            x += v * h * np.cos(th_mid)
            y += v * h * np.sin(th_mid)
            out[k] = (x, y)
        return out

    def _step(self, v, w, uv, uw):
        return (self.a_v * v + self.b_v * uv,
                self.a_w * w + self.b_w * uw + self.c_w * uw * abs(uv))

    # ---- serialisation: the whole model is 5 floats -----------------------
    def to_dict(self):
        return dict(a_v=self.a_v, b_v=self.b_v, a_w=self.a_w,
                    b_w=self.b_w, c_w=self.c_w, dt_nominal=DT_NOMINAL, J=J, F=F)

    @staticmethod
    def from_dict(d):
        return PlantModel(d["a_v"], d["b_v"], d["a_w"], d["b_w"], d.get("c_w", 0.0))


# ==========================================================================
# data
# ==========================================================================
def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def load(ann_glob=ANN_GLOB, tel_cache=TEL_CACHE):
    tel = np.load(tel_cache, allow_pickle=True) if os.path.exists(tel_cache) else None
    out = {}
    for fp in sorted(glob.glob(ann_glob)):
        name = os.path.basename(fp).split("_vla_annotations")[0]
        if name in out:
            continue
        r = json.load(open(fp))
        P = np.array([[s["robot_state"][k] for k in ("x", "y", "yaw")] for s in r])
        U = np.array([[s["robot_state"]["linear_velocity"],
                       s["robot_state"]["angular_velocity"]] for s in r])
        WP = np.array([s["trajectory"]["future_waypoints_robot_frame"] for s in r])
        key = name + "|sample_wall_times"
        if tel is not None and key in tel.files:
            dt = np.diff(tel[key])
        else:                                   # nominal spacing if no h5 cache
            dt = np.full(len(r) - 1, DT_NOMINAL)
        # realised body-frame velocity over each step (what the waypoints encode)
        d = np.diff(P[:, :2], axis=0)
        c, s = np.cos(-P[:-1, 2]), np.sin(-P[:-1, 2])
        v_real = (c * d[:, 0] - s * d[:, 1]) / dt
        w_real = _wrap(np.diff(P[:, 2])) / dt
        out[name] = dict(U=U, WP=WP, dt=dt, v=v_real, w=w_real, n=len(r))
    if not out:
        raise FileNotFoundError(f"no annotations at {ann_glob}")
    return out


def steps_of(run):
    U, v, w = run["U"], run["v"], run["w"]
    return [(v[k - 1], w[k - 1], U[k, 0], U[k, 1], v[k], w[k])
            for k in range(1, len(v))]


# ==========================================================================
# evaluation
# ==========================================================================
def evaluate(model, run, f=F, j=J, noise=0.0, rng=None, true_dt=True):
    U, WP, dt, n = run["U"], run["WP"], run["dt"], run["n"]
    errs = []
    for t in range(j, n - f):
        if len(dt[t:t + f]) < f:
            continue
        u = U[t:t + f].astype(float)
        if noise:
            u = u + rng.normal(0, noise, u.shape)
        pred = model(u, U[t - j:t], dt[t:t + f] if true_dt else None)
        errs.append(np.linalg.norm(pred - WP[t][:f], axis=1))
    e = np.array(errs)
    return e.mean(), e[:, -1].mean(), len(e)


def loso(data, rich=False, **kw):
    ades, fdes, models = [], [], {}
    for ho in data:
        steps = [s for k in data if k != ho for s in steps_of(data[k])]
        m = PlantModel.identify(steps, rich)
        models[ho] = m
        a, f_, _ = evaluate(m, data[ho], **kw)
        ades.append(a); fdes.append(f_)
    return float(np.mean(ades)), float(np.mean(fdes)), models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", action="store_true")
    ap.add_argument("--export", default=None, metavar="DIR")
    args = ap.parse_args()
    data = load()
    print(f"runs {len(data)}  frames {sum(d['n'] for d in data.values())}")

    print("\nLOSO: identify on 9 runs, predict the held-out run's waypoints")
    print("input = 2 past commands + 6-step action chunk, nothing else\n")
    ident = PlantModel(0, 1, 0, 1)                     # v_real = v_cmd
    a = np.mean([evaluate(ident, d)[0] for d in data.values()])
    f_ = np.mean([evaluate(ident, d)[1] for d in data.values()])
    print(f"  {'perfect tracking (no plant model)':38s} ADE {a:.4f}  FDE {f_:.4f}")
    a, f_, models = loso(data)
    print(f"  {'identified lag  (4 parameters)':38s} ADE {a:.4f}  FDE {f_:.4f}")
    ar, fr, _ = loso(data, rich=True)
    print(f"  {'+ u_w*|u_v| term (5 parameters)':38s} ADE {ar:.4f}  FDE {fr:.4f}")
    an, fn, _ = loso(data, true_dt=False)
    print(f"  {'4-param, nominal dt=0.5':38s} ADE {an:.4f}  FDE {fn:.4f}")

    print("\n  warm-up length (past commands replayed):")
    for j in (0, 1, 2, 3, 4):
        a, f_, _ = loso(data, j=j)
        print(f"    J={j}   ADE {a:.4f}  FDE {f_:.4f}")

    print("\n  tolerance to VLA command error (gaussian, m/s and rad/s):")
    for s in (0.02, 0.05, 0.10, 0.20):
        a, f_, _ = loso(data, noise=s, rng=np.random.default_rng(0))
        print(f"    sigma={s:.2f}   ADE {a:.4f}  FDE {f_:.4f}")

    if args.equation:
        print("\n--- identified equation, all 10 runs ---")
        allm = PlantModel.identify([s for d in data.values() for s in steps_of(d)],
                                   rich=True)
        print("  " + allm.describe().replace("\n", "\n  "))
        print("\n--- the equation each fold discovered on its own 9 runs ---")
        for i, (ho, m) in enumerate(models.items()):
            a, f_, n = evaluate(m, data[ho])
            print(f"\n  fold {i}  (held out {ho}, n={n})")
            print(f"    v[k] = {m.a_v:.4f}*v[k-1] + {m.b_v:.4f}*u_v"
                  f"   [tau {-DT_NOMINAL/np.log(abs(m.a_v)):.3f}s, gain {m.b_v/(1-m.a_v):.4f}]")
            print(f"    w[k] = {m.a_w:.4f}*w[k-1] + {m.b_w:.4f}*u_w"
                  f"   [tau {-DT_NOMINAL/np.log(abs(m.a_w)):.3f}s, gain {m.b_w/(1-m.a_w):.4f}]")
            print(f"    held-out ADE {a:.4f}  FDE {f_:.4f}")

        print("\n--- agreement across the 10 folds ---")
        rows = {"a_v": [], "b_v": [], "a_w": [], "b_w": [],
                "tau_v": [], "K_v": [], "tau_w": [], "K_w": []}
        for m in models.values():
            rows["a_v"].append(m.a_v); rows["b_v"].append(m.b_v)
            rows["a_w"].append(m.a_w); rows["b_w"].append(m.b_w)
            rows["tau_v"].append(-DT_NOMINAL / np.log(abs(m.a_v)))
            rows["K_v"].append(m.b_v / (1 - m.a_v))
            rows["tau_w"].append(-DT_NOMINAL / np.log(abs(m.a_w)))
            rows["K_w"].append(m.b_w / (1 - m.a_w))
        print(f"  {'coefficient':<12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"
              f"{'spread%':>9}")
        for k, v in rows.items():
            v = np.array(v)
            print(f"  {k:<12}{v.mean():9.4f}{v.std():9.4f}{v.min():9.4f}"
                  f"{v.max():9.4f}{100*(v.max()-v.min())/abs(v.mean()):9.1f}")
        print("\n  A spread of a few percent means the plant is a property of the"
              "\n  robot, not of the scenario -- the calculator transfers.")

        print("\n--- cross-application: every fold model on every held-out run ---")
        mats = np.array([[evaluate(m, data[ho])[0] for ho in data]
                         for m in models.values()])
        own = np.array([mats[i, i] for i in range(len(mats))])
        print(f"  ADE with the correct (unleaked) fold model : {own.mean():.4f}")
        print(f"  ADE with the worst fold model per run      : {mats.max(0).mean():.4f}")
        print(f"  largest ADE change from swapping models    : "
              f"{(mats.max(0) - mats.min(0)).max():.4f} m")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        for i, (ho, m) in enumerate(models.items()):
            d = m.to_dict(); d["held_out_scenario"] = ho; d["fold"] = i
            json.dump(d, open(os.path.join(args.export, f"fold_{i:02d}_{ho}.json"), "w"),
                      indent=2)
        allm = PlantModel.identify([s for d in data.values() for s in steps_of(d)])
        d = allm.to_dict(); d["held_out_scenario"] = None; d["fold"] = "all"
        json.dump(d, open(os.path.join(args.export, "all_data.json"), "w"), indent=2)
        print(f"\nwrote {len(models)} fold models + all_data.json to {args.export}/")
        print("Pair fold k of the VLA with fold k here: the calculator for a held-out")
        print("scenario must not have been identified on that scenario either.")
        print("For deployment use all_data.json -- the folds agree to within 3%.")


if __name__ == "__main__":
    main()
