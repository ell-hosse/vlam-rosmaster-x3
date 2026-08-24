# velocity_head — VLA that predicts commands, not waypoints

Two stages, trained separately, evaluated together:

```
 RGB frame  --frozen MobileNetV3-Small-->  f  --.
                                                |--> cond --> FLOW HEAD --> z
 u[t-6 .. t-1]  (3 s of past commands) ------->  '                          |
                                                                            v
                                             cond + z --> VELOCITY head --> u[t .. t+5]
                                                          ACTION  head  --> class of u[t]
                                                                            |
                              ../trajectory_predictor  (4 frozen parameters) |
                                                                            v
                                                       6 waypoints, robot frame, 3 s
```

Stage 1 is this package: ~1.07 M trainable parameters, same skeleton as
`../defussion_text_generator` (rectified flow, FiLM conditioning, PCA text
latent). Stage 2 is the plant model in `../trajectory_predictor`: four numbers,
frozen, no gradient ever flows into it.

## Run it

Stage 2 must be exported first — stage 1 loads one calculator per fold:

```bash
cd vla/trajectory_predictor
python trajectory_calculator.py --export models

cd ../velocity_head
python run_cv.py
```

The export folder can be called anything — `run_cv.py` looks for whichever
directory under `vla/trajectory_predictor/` actually contains `fold_*.json`
(`models/`, `trajectory_predictors/`, …) and prints the one it picked. Override
with `--calculator-dir <path>` if you keep them somewhere else entirely.

Useful flags: `--epochs N`, `--folds N` (debug), `--hist N`,
`--calculator-dir PATH`, `--text-field caption_detailed`.

Fold *i* holds out scenario *S*, trains on the other nine, **and** loads
`../trajectory_predictor/models/fold_*_S.json` — the calculator that was also
identified without ever seeing *S*. Nothing in the pipeline has touched the
held-out scenario. That pairing is the entire reason the calculator was
exported per fold rather than once.

## The three alignments that had to be right

**Text field.** `action.action_text` — 8 unique strings in this dataset (vs 9
for `trajectory_text`, 240 for `caption_detailed`). `_get_field` accepts
`"action_text"`, `"action.action_text"`, or any dotted path.

**Input window.** `u[t-6 .. t-1]`: the six commands the robot issued before
now — 3 s at 2 Hz. All are genuinely known at inference; the robot sent them.

**Output chunk.** `u[t .. t+5]`, *starting at the current frame*. This is not a
style choice. `future_waypoints_robot_frame[k]` is the pose at `t+k+1`, and the
pose at `t+1` is produced by the command logged at `t`. Fitting the plant with
the chunk shifted one step later doubles its residual:

| chunk | plant residual |
|---|---|
| `u[t .. t+5]` | **0.037 m** |
| `u[t+1 .. t+6]` | 0.072 m |
| `u[t-1 .. t+4]` | 0.044 m |

So deciding `u[t]` is part of the model's job — the same quantity the existing
action head predicts — which is why `u[t]` must not appear in the input.

**Cost of the window.** Each run loses `hist + horizon` = 12 frames, leaving
**211 windows** from 331 frames. `--hist 4` buys back 20 windows; the history
mattered less than the count above `hist=3` when I checked on the stage-2 side.

## What each fold reports

`runs/fold_XX_<scenario>/metrics.json`, and averaged in `runs/summary.json`:

* `velocity.v` — MAE, RMSE, R², max error, and **per-step** MAE, in m/s
* `velocity.w` — the same, in rad/s (never pooled with `v`: different units,
  and pooling hides which channel is failing)
* `velocity.snapped_exact_match` — the dataset contains only **10 distinct
  `(v, w)` command pairs**, so predictions are snapped to the nearest training
  prototype and scored for exact agreement. More legible than MAE.
* `ade` / `fde` — the trajectory those predicted commands produce
* `ade_oracle_commands` — the same calculator fed the **true** commands. This is
  the floor (~0.04 m). The gap between it and `ade` is stage 1's contribution.
* `ade_hold_last_command` — baseline: keep issuing the last command sent (~0.18 m).
  Stage 1 has to beat this to have earned anything.
* `train_loss` — `{flow, v, w, act}` separately, last epoch
* action accuracy, text cosine vs the PCA ceiling, retrieval accuracy, and the
  Euler-step sweep

Console line per fold:

```
fold  4 scenario_4_run_03  n= 23  v_mae=0.1102  w_mae=0.1651  ADE=0.1095 (oracle 0.0215)  FDE=0.2187  acc=0.522
```

## Read the oracle row first

`ade_oracle_commands ≈ 0.04 m` says the calculator is essentially exact. Any
ADE above that is a stage-1 error, and the error budget is roughly linear:
σ = 0.05 on both channels costs ~0.011 m of ADE, σ = 0.10 costs ~0.032 m.
A *systematic* yaw bias is worse than noise — +0.1 rad/s of constant oversteer
costs 0.005 m on its own — so watch `velocity.w.mae` against the 46 %
RIGHT_TURN class imbalance.

## The first ablation to run

The command space is discrete — 10 pairs, one per action label plus six
near-identical RIGHT_TURN variants:

| label | v | w | frames |
|---|---|---|---|
| FORWARD | 0.50 | 0.00 | 100 |
| STOP | 0.00 | 0.00 | 58 |
| RIGHT_TURN | 0.12 | −0.433 (6 variants) | 153 |
| LEFT_TURN | 0.12 | +0.40 | 11 |
| REVERSE_LEFT | −0.20 | +0.44 | 9 |

Replacing every command with its class prototype changes stage-2 ADE by
0.0001 m — the six RIGHT_TURN variants are noise. So an **`F × 5`
classification head** over action labels is very likely stronger than the
regression head used here: L1 regression on a multi-modal discrete target
produces blurry averages that sit between FORWARD and RIGHT_TURN and belong to
neither. The `snapped_exact_match` metric is in the output specifically so the
two can be compared on equal footing.

Keep the regression head as the ablation row; it is the honest comparison.

## Notes

* `VLA_CNN_WEIGHTS=none` and `VLA_TEXT_WEIGHTS=none` skip the pretrained
  downloads for offline smoke tests. Both print a warning, and every number
  produced under them is meaningless.
* Image features are cached by `scenario:frame` and shared with
  `../defussion_text_generator/cache`, so windowing does not force a
  re-extraction.
* `action.linear_velocity_target` / `angular_velocity_target` are bit-identical
  to the h5 `cmd_vel` channels (max difference 0.0 over all 331 frames) and to
  `robot_state.linear_velocity` / `angular_velocity`. They are **commands**, not
  measurements — converting one into the other is exactly stage 2's job. Do not
  train this head on odometry.
