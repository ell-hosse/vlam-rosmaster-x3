"""Load the Rosmaster X3 dataset into flat per-frame records.

One record = one annotated frame = one row of an .h5 file.
Scenario name is kept so the cross-validation can split on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from config import Config


@dataclass
class Sample:
    scenario: str          # e.g. "scenario_1_run_03"
    frame: int             # row index inside the .h5
    telemetry: np.ndarray  # (D_tel,) float32
    waypoints: np.ndarray  # (n_waypoints, 2) float32, robot frame, metres
    action: int            # index into cfg.action_classes
    text: str              # the raw target string


def _scenario_name(path: Path) -> str:
    """'scenario_1_run_03_vla_annotations_FINAL_APPROVED.json' -> 'scenario_1_run_03'"""
    return path.name.split("_vla_annotations")[0]


def load_samples(cfg: Config) -> list[Sample]:
    ann_dir = cfg.dataset_dir / "annotations"
    h5_dir = cfg.dataset_dir / "data"

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"no annotation .json files under {ann_dir}")

    action_index = {name: i for i, name in enumerate(cfg.action_classes)}
    samples: list[Sample] = []
    seen: set[str] = set()

    for ann_path in ann_files:
        scenario = _scenario_name(ann_path)
        if scenario in seen:          # guards against duplicated run files
            print(f"  [skip] duplicate scenario {scenario} ({ann_path.name})")
            continue
        seen.add(scenario)

        h5_path = h5_dir / f"{scenario}.h5"
        if not h5_path.exists():
            print(f"  [skip] {scenario}: no matching .h5")
            continue

        records = json.loads(ann_path.read_text())

        with h5py.File(h5_path, "r") as f:
            n_h5 = len(f["sample_wall_times"])
            if n_h5 != len(records):
                raise ValueError(
                    f"{scenario}: {len(records)} annotations but {n_h5} h5 frames"
                )
            telemetry = np.concatenate(
                [np.asarray(f[k][:], dtype=np.float32).reshape(n_h5, -1)
                 for k in cfg.telemetry_keys],
                axis=1,
            )

        for i, rec in enumerate(records):
            wp = np.asarray(
                rec["trajectory"]["future_waypoints_robot_frame"], dtype=np.float32
            )
            if cfg.drop_first_waypoint:
                wp = wp[1:]
            wp = wp[: cfg.n_waypoints]
            if len(wp) != cfg.n_waypoints:
                raise ValueError(
                    f"{scenario}[{i}]: need {cfg.n_waypoints} waypoints, got {len(wp)}"
                )

            label = rec["action"]["action_label"]
            if label not in action_index:
                raise ValueError(f"unknown action label {label!r} in {scenario}")

            samples.append(
                Sample(
                    scenario=scenario,
                    frame=i,
                    telemetry=telemetry[i],
                    waypoints=wp,
                    action=action_index[label],
                    text=rec["trajectory"][cfg.text_field]
                    if cfg.text_field.startswith("trajectory")
                    else rec[cfg.text_field],
                )
            )

    if not samples:
        raise RuntimeError("no samples loaded — check dataset paths")
    return samples


def scenarios_of(samples: list[Sample]) -> list[str]:
    """Unique scenario names, in a stable order."""
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

    lines = [
        f"samples           : {len(samples)}",
        f"scenarios         : {len(per_scenario)}  {dict(per_scenario)}",
        f"action balance    : {dict(per_action)}",
        f"majority baseline : {majority[0]} = {100 * majority[1] / len(samples):.1f}%",
        f"unique '{cfg.text_field}': {len(texts)}",
    ]
    return "\n".join("  " + ln for ln in lines)
