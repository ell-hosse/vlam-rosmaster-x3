"""Windowed records for the velocity head.

One record = one *current frame* t inside a run, carrying

    tel   : u[t-H .. t-1]      the H past commands            (H, 2)  -> input
    chunk : u[t   .. t+F-1]    the F commands to predict      (F, 2)  -> target
    wp    : the F recorded waypoints in the robot frame at t  (F, 2)  -> eval
    dt    : the F true step durations                         (F,)    -> eval
    text  : cfg.text_field of frame t                                 -> flow target
    action: action_label of frame t                                   -> aux target

Why the chunk starts at t and the history ends at t-1
-----------------------------------------------------
`trajectory.future_waypoints_robot_frame[k]` is the pose at t+k+1. The pose at
t+1 is produced by executing the command logged at t, so the chunk that
generates the recorded 3 s trajectory is u[t .. t+5]. Deciding u[t] is the
model's job -- that is the same quantity the existing action head predicts --
so it must not appear in the input. The 3 s of context is therefore the six
commands the robot issued before now, all of which it genuinely knows at
inference time.

Commands, not measurements
--------------------------
`action.linear_velocity_target` / `angular_velocity_target` are bit-identical to
the h5 `cmd_vel` channels (verified: max |difference| = 0 over all 331 frames),
and `robot_state.linear_velocity` / `angular_velocity` are the same numbers
again. They are what was SENT to the base, not what the wheels did -- the
stage-2 calculator is what converts one into the other.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import Config

# The stage-2 package owns the canonical (commands, waypoints, dt) loader and
# the plant model. Importing it keeps the two stages byte-identical on the
# things they share instead of re-deriving them here.
CALC_DIR = Path(__file__).resolve().parents[1] / "trajectory_predictor"
if str(CALC_DIR) not in sys.path:
    sys.path.insert(0, str(CALC_DIR))

try:
    import trajectory_calculator as tc          # noqa: E402
except ImportError as exc:                      # pragma: no cover
    raise ImportError(
        f"cannot import the stage-2 calculator from {CALC_DIR}.\n"
        "    Expected vla/trajectory_predictor/trajectory_calculator.py"
    ) from exc


@dataclass
class Sample:
    scenario: str
    frame: int                # index of the CURRENT frame t inside the run
    tel: np.ndarray           # (H*2,) float32  past commands, flattened
    chunk: np.ndarray         # (F*2,) float32  target commands, flattened
    wp: np.ndarray            # (F, 2) float32  recorded waypoints, robot frame
    dt: np.ndarray            # (F,)   float32  true step durations
    warm: np.ndarray          # (J, 2) float32  last J past commands (plant warm-up)
    action: int
    text: str


def _get_field(record: dict, field: str):
    """'action_text' or 'action.action_text' or 'trajectory.trajectory_text'."""
    if "." in field:
        node = record
        for part in field.split("."):
            node = node[part]
        return node
    for parent in (None, "action", "trajectory", "map_context"):
        node = record if parent is None else record.get(parent, {})
        if isinstance(node, dict) and field in node:
            return node[field]
    raise KeyError(f"field {field!r} not found in the annotation record")


def _scenario_name(path: Path) -> str:
    return path.name.split("_vla_annotations")[0]


def load_samples(cfg: Config) -> list[Sample]:
    H, F, J = cfg.hist, cfg.horizon, tc.J

    runs = tc.load(str(cfg.dataset_dir / "annotations" / "*.json"))

    ann_dir = cfg.dataset_dir / "annotations"
    records: dict[str, list] = {}
    for fp in sorted(ann_dir.glob("*.json")):
        name = _scenario_name(fp)
        if name in records:
            print(f"  [skip] duplicate scenario file {fp.name}")
            continue
        records[name] = json.loads(fp.read_text())

    action_index = {a: i for i, a in enumerate(cfg.action_classes)}
    samples: list[Sample] = []

    for scenario, d in runs.items():
        U, WP, dt, n = d["U"], d["WP"], d["dt"], d["n"]
        recs = records.get(scenario)
        if recs is None:
            print(f"  [skip] {scenario}: no annotation file")
            continue
        if len(recs) != n:
            raise ValueError(f"{scenario}: {len(recs)} annotations vs {n} h5 rows")

        # t needs H commands behind it and F real (non-padded) waypoints ahead:
        # WP[t] holds the poses at t+1..t+F, so t must satisfy t + F <= n - 1.
        for t in range(max(H, J), n - F):
            label = recs[t]["action"]["action_label"]
            if label not in action_index:
                raise ValueError(f"unknown action_label {label!r} in {scenario}")
            samples.append(
                Sample(
                    scenario=scenario,
                    frame=t,
                    tel=U[t - H:t].astype(np.float32).ravel(),
                    chunk=U[t:t + F].astype(np.float32).ravel(),
                    wp=WP[t][:F].astype(np.float32),
                    dt=dt[t:t + F].astype(np.float32),
                    warm=U[t - J:t].astype(np.float32),
                    action=action_index[label],
                    text=str(_get_field(recs[t], cfg.text_field)),
                )
            )

    if not samples:
        raise RuntimeError(
            "no windows built -- check dataset paths, or lower cfg.hist "
            "(each run loses hist + horizon frames)"
        )
    return samples


def scenarios_of(samples: list[Sample]) -> list[str]:
    out, seen = [], set()
    for s in samples:
        if s.scenario not in seen:
            seen.add(s.scenario)
            out.append(s.scenario)
    return out


def describe(samples: list[Sample], cfg: Config) -> str:
    import collections

    per_scenario = collections.Counter(s.scenario for s in samples)
    per_action = collections.Counter(cfg.action_classes[s.action] for s in samples)
    texts = collections.Counter(s.text for s in samples)
    majority = per_action.most_common(1)[0]

    chunks = np.stack([s.chunk for s in samples]).reshape(len(samples), -1, 2)
    v, w = chunks[..., 0], chunks[..., 1]

    lines = [
        f"windows           : {len(samples)}  "
        f"(input u[t-{cfg.hist}..t-1], target u[t..t+{cfg.horizon - 1}])",
        f"per scenario      : {dict(per_scenario)}",
        f"current action    : {dict(per_action)}",
        f"majority baseline : {majority[0]} = {100 * majority[1] / len(samples):.1f}%",
        f"unique '{cfg.text_field}' : {len(texts)}",
        f"v_cmd  range      : [{v.min():+.2f}, {v.max():+.2f}]  "
        f"mean {v.mean():+.3f}  std {v.std():.3f}",
        f"w_cmd  range      : [{w.min():+.2f}, {w.max():+.2f}]  "
        f"mean {w.mean():+.3f}  std {w.std():.3f}",
    ]
    return "\n".join("  " + ln for ln in lines)


def command_prototypes(samples: list[Sample], idx) -> np.ndarray:
    """The distinct (v, w) commands present in the given rows.

    The dataset contains only 10 distinct command pairs, so 'how often is the
    predicted command exactly right' is a meaningful metric. Built from TRAIN
    rows only, then used to snap test predictions.
    """
    rows = np.concatenate([samples[i].chunk.reshape(-1, 2) for i in idx])
    return np.unique(np.round(rows, 4), axis=0)
