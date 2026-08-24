# Diffusion Text-Embedding VLA — first experiment

A vision-language-action model that never writes a word. A frozen CNN encodes
the frame; a **flow-matching head generates the *embedding* of the caption it
would have written**; that embedding, plus telemetry, drives the trajectory and
action heads.

```
   RGB frame ──[frozen MobileNetV3-Small]──► f  (576)
                                             │
   telemetry ────────────────────────────►  t  (16)
                                             │
                                    cond = [f, t]
                                             │
                            ┌────────────────┴─────────────────┐
                            │  FLOW HEAD  (the generative part)│
                            │  noise ──1–4 Euler steps──► z    │   z: text latent (≤8)
                            └────────────────┬─────────────────┘
                                             │
                              [cond, z] ─────┴──► 4 waypoints (x,y)
                                                └► action class (5-way)
```

No VLM is called at any point — training or inference.

## Run it

```bash
pip install -r vla/defussion_text_generator/requirements.txt
python vla/defussion_text_generator/run_cv.py
```

First run downloads the ImageNet MobileNetV3 weights and the MiniLM sentence
encoder, then caches the extracted features under `cache/` so the ten folds
share one extraction pass. Roughly 2–5 minutes on a laptop CPU.

Useful flags: `--epochs N`, `--text-field caption_detailed`, `--folds N` (debug).

## What it evaluates

**Leave-one-scenario-out cross-validation: 10 runs, 9 scenarios train, 1 test.**
Never split by frame — frames 0.5 s apart are near-duplicates and a random split
would report an inflated, meaningless number.

Reported as mean ± std over the 10 folds:

| Metric | Meaning |
|---|---|
| **ADE** | mean displacement error over the 4 waypoints, metres |
| **FDE** | displacement error at the last waypoint, metres |
| **min-ADE / min-FDE (K=5)** | best of 5 samples — the standard forecasting metric for a generative head |
| **action accuracy** | 5-way, printed against the majority-class baseline |
| per-class recall | printed per fold; `LEFT_TURN` (11) and `REVERSE_LEFT` (9) are the ones that matter |
| **cosine, model** | generated embedding vs. the true `trajectory_text` embedding |
| **cosine, PCA ceiling** | the truth round-tripped through the fold's PCA — the best any model could score |
| **cosine, train-mean baseline** | always predict the average training embedding |
| **retrieval accuracy** | nearest text in the *training* bank == the true text |
| **bank coverage** | fraction of test samples whose true text appears in the training bank at all |
| **step sweep** | ADE/FDE/accuracy/cosine and ms-per-sample at 1, 2, 4, 8 Euler steps |

Per fold: `runs/fold_XX_<scenario>/model.pt` and `metrics.json`.
Overall: `runs/summary.json`, plus model size in parameters and MB.

## Read the text metric against its baselines, not on its own

`trajectory_text` has **only 9 unique strings** across all 331 samples. Two
consequences you must state when reporting:

1. **The PCA ceiling is ~1.0 and means nothing here.** With ≤9 unique targets, a
   PCA of dim 8 reconstructs them exactly. That number becomes informative only
   with a richer target — try `--text-field caption_detailed` (~240 unique).
2. **Predicting the text is effectively 9-way classification.** A cosine score
   is only meaningful next to the train-mean baseline, and retrieval accuracy is
   only meaningful next to the majority-text rate. Both baselines are printed.

`bank_coverage < 1.0` means a held-out scenario contained a sentence that never
appeared in training — retrieval cannot possibly get those right, which is the
honest version of the "retrieval doesn't scale" problem.

## Where leakage could enter, and why it doesn't

Refit inside **every fold**, from training scenarios only:

- the **PCA** for the text latent space (`latent.py`) — including its mean
- image-feature and telemetry standardisation
- waypoint scaling
- action class weights
- the retrieval bank and the train-mean baseline embedding

The MobileNet and MiniLM encoders are pretrained and frozen: they are never
fitted on this data and see no labels, so extracting features for all 331
samples up front is not leakage. Only the PCA is data-fitted, and it lives
inside the fold loop.

## The diffusion part, and what makes it cheap

- **Rectified flow, not DDPM.** The training path is a straight line
  `z_s = (1−s)·ε + s·z₀` and the target is the constant direction `z₀ − ε`. A
  straight path integrates exactly in one Euler step, so 1–4 steps replace the
  ~100 a curved noise schedule needs.
- **FiLM conditioning hoisted out of the loop.** The conditioning term
  `film_cond(cond)` depends only on the frame, so `sample()` computes it once
  and reuses it across every step; only a small time projection and the residual
  blocks re-run. This is why the step sweep shows the marginal step costing far
  less than the first.
- **A small latent.** The head generates ≤8 PCA coefficients, not 384 raw
  embedding dimensions.
- **Check the sweep before fixing `flow_steps`.** On this dataset 1–2 steps
  typically matches 8. Take the cheapest setting the sweep supports.

## Watching a fold

```bash
python vla/defussion_text_generator/visualize.py            # last fold
python vla/defussion_text_generator/visualize.py --fold 9
```

Writes `runs/fold_XX_<scenario>/<scenario>.mp4` (plus PNGs): each camera frame
with the ground-truth and predicted trajectories drawn on it, an exact
bird's-eye panel beside it, and a banner with the true and predicted action,
the per-frame ADE, and the retrieved caption.

**Calibrate the overlay first.** Projecting waypoints onto the image needs the
camera height and pitch, which the dataset does not ship. Easiest route:

```bash
python vla/defussion_text_generator/visualize.py --calib-sweep
```

One contact sheet, two dozen height/pitch combinations, pick the panel whose
distance lines land on the mat correctly. If you can measure two distances,
`--solve "row,dist;row,dist"` computes the exact values instead. The knobs:
**more pitch moves the lines up, more height moves them down** — pitch
dominates, so nudge it first. The bird's-eye panel needs no calibration.

## Deliberately not here (yet)

- **Mirror augmentation.** It needs the class set to be closed under mirroring —
  `REVERSE_LEFT` has no `REVERSE_RIGHT` counterpart, so mirroring those 9 frames
  would teach the model something false. Add the class or exclude them first.
- **Warm starting between frames.** Needs sequential rollout; this harness
  evaluates frames independently.
- **The gated VLM lane.** Out of scope by request.

## Files

| File | Role |
|---|---|
| `config.py` | every setting, one dataclass |
| `data.py` | `.h5` + annotations → flat per-frame records |
| `encoders.py` | frozen CNN and sentence encoder, with disk caching |
| `latent.py` | the per-fold PCA text latent space |
| `model.py` | flow head + trajectory/action heads, parameter report |
| `metrics.py` | ADE/FDE, action, text-embedding fidelity |
| `run_cv.py` | the 10-fold loop — entry point |
