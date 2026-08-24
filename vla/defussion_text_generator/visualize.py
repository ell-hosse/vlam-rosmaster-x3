"""Render a held-out fold as a video: predicted vs ground-truth trajectories,
drawn on the actual camera frames, with the true and predicted action per frame.

    # last fold (default), full render
    python vla/defussion_text_generator/visualize.py

    # a specific fold
    python vla/defussion_text_generator/visualize.py --fold 9

    # tune the camera projection first (see CALIBRATION below)
    python vla/defussion_text_generator/visualize.py --calib-sweep

Each output frame has three parts:

    +-----------------------------+-------------------+
    |  camera image               |  bird's-eye view  |
    |  + projected trajectories   |  (exact, no       |
    |                             |   calibration)    |
    +-----------------------------+-------------------+
    |  TRUE: RIGHT_TURN   PRED: RIGHT_TURN   ADE 0.12 |
    +-------------------------------------------------+

GREEN = ground truth, ORANGE = model prediction.

CALIBRATION — three ways, easiest first
---------------------------------------
Drawing the trajectory *on the image* needs the camera height and pitch, which
this dataset does not ship. The --cam-* defaults are a guess, not a measurement.

1. CONTACT SHEET (no thinking required)

       python visualize.py --calib-sweep

   Renders one frame under two dozen height/pitch combinations on a single
   sheet. Find the panel where the coloured distance lines land on the mat
   where they should, and use that panel's numbers. This is the recommended
   route.

2. SOLVE IT (if you can measure two distances on the mat)

       python visualize.py --solve "455,0.3;352,1.0"

   Two ground points straight ahead of the robot, each given as
   "image_row,distance_in_metres". Open a frame in any image viewer, put the
   cursor on a feature whose distance you know — a crosswalk stripe, a lane
   edge — and read off the row. Prints exact --cam-height / --cam-pitch.

3. CHECK ONE SETTING

       python visualize.py --calib-grid --cam-height 0.18 --cam-pitch 8

   Draws a full ground grid on a few frames so you can confirm a candidate.

Which way do the knobs move?
    MORE pitch  -> every distance line moves UP the image (the camera is
                   tilted further down, so the ground near you fills the frame
                   and 1 m appears higher).
    MORE height -> the lines move DOWN and spread apart (a taller camera looks
                   further over the ground, so a given distance sits lower).
Nudge pitch first: it dominates.

The bird's-eye panel is exact and never depends on any of this.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                       # noqa: E402
from data import load_samples                   # noqa: E402
from encoders import image_features             # noqa: E402
from model import VLAModel                      # noqa: E402

GT_COLOR = "#2E9E4F"        # green  — ground truth
PR_COLOR = "#E07B24"        # orange — prediction
OK_COLOR = "#2E9E4F"
BAD_COLOR = "#C43D3D"


# ------------------------------------------------------------------ camera
def project_ground(points_xy: np.ndarray, args) -> np.ndarray:
    """Robot-frame ground points (x forward, y left, metres) -> pixel (u, v).

    Camera coords: x right, y down, z along the optical axis. The camera sits
    `h` above the ground, tilted down by `pitch`.

        y_c = h*cos(p) - X*sin(p)
        z_c = h*sin(p) + X*cos(p)
        u   = cx + fx * (-Y) / z_c
        v   = cy + fy * y_c  / z_c

    Points at or behind the camera plane come back as NaN.
    """
    X = points_xy[:, 0].astype(np.float64)
    Y = points_xy[:, 1].astype(np.float64)
    p = np.deg2rad(args.cam_pitch)
    h = args.cam_height

    y_c = h * np.cos(p) - X * np.sin(p)
    z_c = h * np.sin(p) + X * np.cos(p)

    with np.errstate(divide="ignore", invalid="ignore"):
        u = args.cam_cx + args.cam_fx * (-Y) / z_c
        v = args.cam_cy + args.cam_fy * y_c / z_c
    bad = z_c <= 1e-3
    u[bad] = np.nan
    v[bad] = np.nan
    return np.stack([u, v], axis=1)


def solve_camera(obs, fy: float, cy: float):
    """Solve (height, pitch) from two ground points of known forward distance.

    Each observation is (image_row, forward_distance_metres) for a point on
    the ground straight ahead of the robot. With t = (row - cy) / fy and
    T = tan(pitch):

        X = h * (1 - t*T) / (t + T)

    Two observations give a quadratic in T:

        (X2*t1 - X1*t2) T^2 + (X1 - X2)(1 - t1*t2) T + (X1*t1 - X2*t2) = 0
    """
    (v1, x1), (v2, x2) = obs
    t1, t2 = (v1 - cy) / fy, (v2 - cy) / fy

    a = x2 * t1 - x1 * t2
    b = (x1 - x2) * (1 - t1 * t2)
    c = x1 * t1 - x2 * t2

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            raise ValueError("degenerate observations — pick two clearly different rows")
        roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError("no real solution — check the rows and distances")
        r = np.sqrt(disc)
        roots = [(-b + r) / (2 * a), (-b - r) / (2 * a)]

    best = None
    for T in roots:
        pitch = np.degrees(np.arctan(T))
        if not (-30.0 <= pitch <= 89.0):
            continue
        denom = 1 - t1 * T
        if abs(denom) < 1e-12:
            continue
        h = x1 * (t1 + T) / denom
        if 0.01 < h < 1.5:
            if best is None or abs(pitch) < abs(best[1]):
                best = (h, pitch)
    if best is None:
        raise ValueError(f"no physically sensible solution (roots {roots})")
    return best


# ------------------------------------------------------------------ model
def load_fold(fold_dir: Path, cfg: Config):
    ckpt = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
    model = VLAModel(576, 16, int(ckpt["latent_dim"]), cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def decode_text(z: np.ndarray, ckpt) -> np.ndarray:
    """Latent -> 384-d sentence space, using the fold's own PCA."""
    out = (z * ckpt["pca_scale"]) @ ckpt["pca_basis"] + ckpt["pca_mean"]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


# ------------------------------------------------------------------ render
def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg = Config()

    fold_dirs = sorted(glob.glob(str(cfg.out_dir / "fold_*")))
    if not fold_dirs:
        sys.exit(f"no folds under {cfg.out_dir} — run run_cv.py first")
    if args.fold is None:
        fold_dir = Path(fold_dirs[-1])
    else:
        match = [d for d in fold_dirs if Path(d).name.startswith(f"fold_{args.fold:02d}")]
        if not match:
            sys.exit(f"fold {args.fold} not found in {cfg.out_dir}")
        fold_dir = Path(match[0])

    scenario = fold_dir.name.split("_", 2)[2]
    print(f"fold      : {fold_dir.name}")
    print(f"scenario  : {scenario}")

    model, ckpt = load_fold(fold_dir, cfg)

    # ---- features for this scenario (cache hit: nothing re-extracted) ----
    samples = load_samples(cfg)
    feats = image_features(samples, cfg)
    keep = [i for i, s in enumerate(samples) if s.scenario == scenario]
    if not keep:
        sys.exit(f"no samples for {scenario}")
    order = sorted(keep, key=lambda i: samples[i].frame)

    img_f = (feats[order] - ckpt["img_mean"]) / ckpt["img_std"]
    tel = np.stack([samples[i].telemetry for i in order])
    tel_f = (tel - ckpt["tel_mean"]) / ckpt["tel_std"]
    wp_true = np.stack([samples[i].waypoints for i in order])
    act_true = np.array([samples[i].action for i in order])
    txt_true = [samples[i].text for i in order]

    gen = torch.Generator().manual_seed(cfg.seed)
    z, traj, logits = model.predict(
        torch.tensor(img_f, dtype=torch.float32),
        torch.tensor(tel_f, dtype=torch.float32),
        cfg.flow_steps, gen,
    )
    wp_pred = (
        traj.numpy() * ckpt["wp_std"] + ckpt["wp_mean"]
    ).reshape(len(order), cfg.n_waypoints, 2)
    act_pred = logits.argmax(1).numpy()

    # retrieved caption from the fold's own training bank
    bank_texts = list(ckpt["bank_texts"])
    bank_lookup = {t: i for i, t in enumerate(bank_texts)}
    txt_all_emb = None
    try:
        from encoders import text_embeddings
        uniq = sorted(set(t for s in samples for t in [s.text]))
        emb = text_embeddings([s.text for s in samples], cfg)
        lookup = {t: emb[i] for i, t in enumerate(uniq)}
        bank_emb = np.stack([lookup[t] for t in bank_texts])
        pred_emb = decode_text(z.numpy(), ckpt)
        retrieved = [bank_texts[i] for i in (pred_emb @ bank_emb.T).argmax(1)]
        txt_all_emb = True
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [warn] caption retrieval unavailable ({exc}); skipping that line")
        retrieved = [""] * len(order)

    # ---- raw frames ----
    with h5py.File(cfg.dataset_dir / "data" / f"{scenario}.h5", "r") as f:
        images = f["rgb_images"][:][[samples[i].frame for i in order]]

    H, W = images.shape[1:3]
    out_dir = fold_dir / ("calib_grid" if args.calib_grid else "video")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- contact sheet: pick the panel that matches ----------
    if args.calib_sweep:
        import copy as _copy

        heights = [float(x) for x in args.sweep_heights.split(",")]
        pitches = [float(x) for x in args.sweep_pitches.split(",")]
        kf = args.sweep_frame if args.sweep_frame is not None else len(order) // 2
        kf = max(0, min(kf, len(order) - 1))
        bands = ((0.3, "#3A86FF"), (0.6, "#D62828"), (1.0, "#00A878"), (2.0, "#8338EC"))

        fig, axes = plt.subplots(
            len(heights), len(pitches),
            figsize=(2.35 * len(pitches), 2.0 * len(heights)), dpi=args.dpi,
            squeeze=False,
        )
        for i, hh in enumerate(heights):
            for j, pp in enumerate(pitches):
                ax = axes[i][j]
                ax.imshow(images[kf])
                ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
                a2 = _copy.copy(args)
                a2.cam_height, a2.cam_pitch = hh, pp
                for dist, col in bands:
                    line = np.stack(
                        [np.full(31, dist), np.linspace(-0.7, 0.7, 31)], 1
                    )
                    uv = project_ground(line, a2)
                    ax.plot(uv[:, 0], uv[:, 1], color=col, lw=1.6)
                ax.set_title(f"h={hh:.2f}  pitch={pp:g}°", fontsize=8, pad=2)

        legend = "   ".join(f"{d:g} m" for d, _ in bands)
        fig.suptitle(
            f"CALIBRATION SWEEP  —  frame {samples[order[kf]].frame}   "
            f"lines: {legend}   (blue / red / green / purple)\n"
            "pick the panel where the lines sit on the mat at the right "
            "distances, then pass its h and pitch to the real render",
            fontsize=10,
        )
        fig.subplots_adjust(left=.01, right=.99, top=.88, bottom=.01,
                            wspace=.04, hspace=.16)
        sweep_path = fold_dir / "calib_sweep.png"
        fig.savefig(sweep_path, facecolor="white")
        plt.close(fig)
        print(f"\n  contact sheet -> {sweep_path}")
        print("  then:  --cam-height <h> --cam-pitch <pitch>")
        return

    n = len(order) if not args.calib_grid else min(4, len(order))
    print(f"frames    : {n}   image {W}x{H}")
    print(f"camera    : h={args.cam_height} m  pitch={args.cam_pitch} deg  "
          f"f=({args.cam_fx}, {args.cam_fy})  c=({args.cam_cx}, {args.cam_cy})")

    ade_all = []
    rendered = []
    off_below = 0        # too close to see — expected, geometry is fine
    off_bad = 0          # off to the side or above the horizon — calibration

    # ---- how far below the image do the near waypoints land? ----
    # A level camera cannot see the ground right in front of it, so the first
    # waypoints legitimately project below row H. Rather than clip them away,
    # extend the canvas and shade the strip as outside the field of view.
    all_pts = np.concatenate(
        [np.vstack([[0.0, 0.0], wp_true[i]]) for i in range(len(order))]
        + [np.vstack([[0.0, 0.0], wp_pred[i]]) for i in range(len(order))]
    )
    proj_v = project_ground(all_pts, args)[:, 1]
    proj_v = proj_v[np.isfinite(proj_v)]
    if args.pad_bottom == "auto":
        need = float(proj_v.max()) - H if len(proj_v) else 0.0
        pad = int(min(max(0.0, need + 18.0), 0.38 * H))
    else:
        pad = int(args.pad_bottom)
    if pad:
        print(f"  near waypoints fall {pad} px below the image "
              f"(a level camera cannot see closer than "
              f"{args.cam_fy * args.cam_height / (H - args.cam_cy):.2f} m)")
        print(f"  -> extending the canvas and shading that strip; "
              f"pass --pad-bottom 0 to clip instead")

    # one fixed bird's-eye scale for the whole scenario, so the view does not
    # rescale between frames and motion stays comparable
    bev_lim = max(
        0.6, float(np.abs(np.concatenate([wp_true, wp_pred], axis=0)).max()) * 1.15
    )

    for k in range(n):
        fig = plt.figure(figsize=(11.6, 5.4), dpi=args.dpi)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0],
                              height_ratios=[1.0, 0.15], hspace=0.06, wspace=0.12)

        # ---------------- camera panel ----------------
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(images[k])
        ax.set_xlim(0, W); ax.set_ylim(H + pad, 0); ax.axis("off")
        if pad:
            ax.add_patch(Rectangle((0, H), W, pad, facecolor="#E4E4DF",
                                   edgecolor="none", zorder=1))
            ax.axhline(H, color="#9A9A93", lw=1.0, ls="--", zorder=2)
            ax.text(8, H + 15, "below the camera's field of view",
                    fontsize=8, color="#6E6E66", ha="left", va="top", zorder=4)

        if args.calib_grid:
            for dist in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
                line = np.stack([np.full(41, dist), np.linspace(-1.0, 1.0, 41)], 1)
                uv = project_ground(line, args)
                ax.plot(uv[:, 0], uv[:, 1], color="#3A86FF", lw=1.2, alpha=.9)
                mid = uv[len(uv) // 2]
                if np.isfinite(mid).all():
                    ax.text(mid[0] + 6, mid[1] - 6, f"{dist:g} m",
                            color="#3A86FF", fontsize=9, weight="bold")
            for lat in (-0.5, -0.25, 0.0, 0.25, 0.5):
                line = np.stack([np.linspace(0.15, 3.0, 41), np.full(41, lat)], 1)
                uv = project_ground(line, args)
                ax.plot(uv[:, 0], uv[:, 1], color="#3A86FF", lw=0.8, alpha=.5)
            ax.set_title("CALIBRATION GRID — tune --cam-height / --cam-pitch "
                         "until this matches the mat", fontsize=10, color="#3A86FF")
        else:
            for pts, color, label in (
                (wp_true[k], GT_COLOR, "ground truth"),
                (wp_pred[k], PR_COLOR, "predicted"),
            ):
                path = np.vstack([[0.0, 0.0], pts])       # start at the robot
                uv = project_ground(path, args)
                ok = np.isfinite(uv).all(axis=1)
                below = ok & (uv[:, 1] >= H + pad)
                bad = ok & ~below & (
                    (uv[:, 0] <= 0) | (uv[:, 0] >= W) | (uv[:, 1] <= 0)
                )
                off_below += int(below.sum())
                off_bad += int(bad.sum())
                ax.plot(uv[ok, 0], uv[ok, 1], "-o", color=color, lw=2.6,
                        ms=6, mec="white", mew=1.1, label=label, zorder=3)
            ax.legend(loc="upper right", fontsize=9, framealpha=.85)

        # ---------------- bird's-eye panel ----------------
        bev = fig.add_subplot(gs[0, 1])
        for pts, color, label in (
            (wp_true[k], GT_COLOR, "ground truth"),
            (wp_pred[k], PR_COLOR, "predicted"),
        ):
            path = np.vstack([[0.0, 0.0], pts])
            bev.plot(-path[:, 1], path[:, 0], "-o", color=color, lw=2.4,
                     ms=5, mec="white", mew=1.0, label=label)
        bev.plot(0, 0, marker="^", ms=11, color="#333333", zorder=5)
        bev.set_xlim(-bev_lim, bev_lim); bev.set_ylim(-0.15 * bev_lim, bev_lim)
        bev.set_aspect("equal")
        bev.grid(alpha=.25, lw=.6)
        bev.set_xlabel("left  <-  y (m)  ->  right", fontsize=8)
        bev.set_ylabel("forward  x (m)", fontsize=8)
        bev.tick_params(labelsize=8)
        bev.set_title("bird's-eye (exact)", fontsize=10)

        # ---------------- banner ----------------
        ban = fig.add_subplot(gs[1, :]); ban.axis("off")
        t_name = cfg.action_classes[act_true[k]]
        p_name = cfg.action_classes[act_pred[k]]
        correct = act_true[k] == act_pred[k]
        ade = float(np.linalg.norm(wp_pred[k] - wp_true[k], axis=-1).mean())
        ade_all.append(ade)

        ban.add_patch(Rectangle((0, 0), 1, 1, transform=ban.transAxes,
                                facecolor="#F2F2EF", edgecolor="none"))
        ban.text(0.012, 0.62, f"frame {samples[order[k]].frame:03d} / {len(order) - 1}",
                 fontsize=10, va="center", family="monospace", color="#555555")
        ban.text(0.135, 0.62, f"TRUE  {t_name}", fontsize=12, va="center",
                 family="monospace", weight="bold", color="#222222")
        ban.text(0.360, 0.62, f"PRED  {p_name}", fontsize=12, va="center",
                 family="monospace", weight="bold",
                 color=OK_COLOR if correct else BAD_COLOR)
        ban.text(0.585, 0.62, "correct" if correct else "wrong", fontsize=11,
                 va="center", family="monospace",
                 color=OK_COLOR if correct else BAD_COLOR)
        ban.text(0.70, 0.62, f"ADE {ade:.3f} m", fontsize=11, va="center",
                 family="monospace", color="#222222")
        if retrieved[k]:
            hit = retrieved[k] == txt_true[k]
            ban.text(0.012, 0.18, f"text  {retrieved[k][:96]}", fontsize=8.5,
                     va="center", family="monospace",
                     color=OK_COLOR if hit else BAD_COLOR)

        fig.subplots_adjust(left=.015, right=.985, top=.93, bottom=.02)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        rendered.append(frame)
        fig.savefig(out_dir / f"frame_{k:03d}.png", facecolor="white")
        plt.close(fig)

    print(f"  mean ADE over this scenario: {np.mean(ade_all):.4f} m")
    print(f"  action accuracy            : {(act_pred == act_true).mean():.4f}")

    if not args.calib_grid:
        total_pts = max(n * 2 * (cfg.n_waypoints + 1), 1)
        if off_below:
            print(f"\n  {100 * off_below / total_pts:.0f}% of waypoints are nearer than "
                  f"the camera can see and run off the bottom.\n"
                  f"  That is correct geometry for a level camera, not a calibration "
                  f"problem —\n  the lines still show their direction, and the "
                  f"bird's-eye panel shows them exactly.")
        if off_bad / total_pts > 0.15:
            print(f"\n  [warn] {100 * off_bad / total_pts:.0f}% of points land off to "
                  f"the side or above the horizon.\n         That does suggest the "
                  f"camera model is wrong — try --calib-sweep.")

    if args.calib_grid:
        print(f"\n  grid frames -> {out_dir}")
        print("  adjust --cam-height / --cam-pitch, re-run, then render for real.")
        return

    write_video(rendered, fold_dir / f"{scenario}.mp4", args.fps, out_dir)


def write_video(frames, path: Path, fps: int, png_dir: Path):
    """Try mp4 via OpenCV, then imageio, then fall back to an animated GIF."""
    h, w = frames[0].shape[:2]
    try:
        import cv2
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter would not open")
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"\n  video -> {path}")
        return
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [info] OpenCV path unavailable ({exc})")

    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(path), frames, fps=fps)
        print(f"\n  video -> {path}")
        return
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [info] imageio path unavailable ({exc})")

    try:
        from PIL import Image
        gif = path.with_suffix(".gif")
        Image.fromarray(frames[0]).save(
            gif, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
            duration=int(1000 / fps), loop=0,
        )
        print(f"\n  GIF -> {gif}   (install opencv-python or imageio for mp4)")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [warn] could not write a video ({exc})")

    print(f"  frames -> {png_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=None,
                    help="fold index; default = the last one in runs/")
    ap.add_argument("--fps", type=int, default=2, help="data is 2 Hz; 2 = real time")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--calib-grid", action="store_true",
                    help="draw a ground grid on a few frames to check the projection")
    ap.add_argument("--calib-sweep", action="store_true",
                    help="EASIEST: one contact sheet of many height/pitch combos — "
                         "just pick the panel that matches the mat")
    ap.add_argument("--sweep-frame", type=int, default=None,
                    help="which frame to use for the sweep (default: middle)")
    ap.add_argument("--sweep-heights", default="0.08,0.13,0.18,0.24")
    ap.add_argument("--sweep-pitches", default="0,8,16,24,32,40")
    ap.add_argument("--solve", default=None,
                    help="solve height+pitch from two known ground points: "
                         "\"row1,dist_m;row2,dist_m\"  e.g. \"455,0.3;352,1.0\"")
    # camera model — GUESSES, tune with --calib-grid
    ap.add_argument("--cam-height", type=float, default=0.13, help="metres above ground")
    ap.add_argument("--cam-pitch", type=float, default=0.0, help="degrees, down positive")
    ap.add_argument("--cam-fx", type=float, default=525.0)
    ap.add_argument("--cam-fy", type=float, default=525.0)
    ap.add_argument("--cam-cx", type=float, default=320.0)
    ap.add_argument("--cam-cy", type=float, default=240.0)
    ap.add_argument("--pad-bottom", default="auto",
                    help="pixels of shaded canvas below the image so near "
                         "waypoints outside the camera's view still render; "
                         "'auto' (default) fits them, '0' clips them")
    args = ap.parse_args()

    if args.solve:
        try:
            obs = []
            for part in args.solve.split(";"):
                row, dist = part.split(",")
                obs.append((float(row), float(dist)))
            if len(obs) != 2:
                raise ValueError("need exactly two points")
            h, pitch = solve_camera(obs, args.cam_fy, args.cam_cy)
        except ValueError as exc:
            sys.exit(f"could not solve: {exc}")
        print(f"\n  solved from {obs}")
        print(f"    --cam-height {h:.4f} --cam-pitch {pitch:.2f}\n")
        print("  sanity-check it with:")
        print(f"    python {Path(__file__).name} --calib-grid "
              f"--cam-height {h:.4f} --cam-pitch {pitch:.2f}\n")
        return

    render(args)


if __name__ == "__main__":
    main()
