# No-text ablation

The same model with the text embedding deleted. This is the control that says
whether the generated text embedding is doing any work.

```
   full model                          this ablation
   ──────────                          ─────────────
   image ──CNN──► f  ─┐                image ──CNN──► f  ─┐
   telemetry ────► t ─┤                telemetry ────► t ─┤
                cond ─┤                             cond ─┤
                      ▼                                   │
              FLOW HEAD ──► z                              │   (nothing here)
                      │                                   │
           [cond, z] ─┴──► waypoints        [cond] ───────┴──► waypoints
                       └──► action                         └──► action
```

Everything else is held fixed: the same cached CNN features, the same
telemetry, the same leave-one-scenario-out folds in the same order, the same
seeds, epochs, optimiser, schedule, class weights and head shape. One variable
changes.

## Run it

```bash
python vla/no_text_ablation/run_cv.py
python vla/no_text_ablation/compare.py
```

It reuses `../defussion_text_generator/cache/` when that exists, so both runs
see byte-identical image features and nothing is re-extracted. It needs no
sentence encoder at all — text is never loaded as an input or a target.

Fast: no flow head means no sampler in the training loop, so folds take
seconds rather than a minute.

## Run it twice — the second run is the one that matters

Deleting the flow head also deletes 889K parameters. If the full model wins,
that could be language *or* it could be capacity. So:

```bash
# 1. strict ablation: text removed, nothing added back
python vla/no_text_ablation/run_cv.py

# 2. capacity-matched: head trunk widened to 916 units so the parameter
#    count matches the full model (1,073,332)
python vla/no_text_ablation/run_cv.py --match-params 1073332
```

| Variant | Trainable params |
|---|---|
| full model (with text) | 1,073,332 |
| strict ablation | 184,749 |
| capacity-matched ablation (`hidden=916`) | 1,073,769 |

If the full model beats the strict ablation but ties the capacity-matched one,
the story was parameters, not language. Report both rows.

## Reading `compare.py`

The folds are paired — both runs hold out the same scenarios in the same order
— so the comparison is per-scenario, not mean-against-mean. That is far more
sensitive with only 10 folds. The script prints, for ADE, FDE and accuracy:

- the per-scenario numbers and their difference,
- how many folds the text version wins,
- a paired t-statistic (|t| > ~2.26 is p < 0.05 at n=10).

A win on 6/10 folds with |t| < 1 is **not** evidence that language helps. Say
so plainly if that is what comes out — it is a perfectly publishable negative
result on a 331-frame dataset, and much better than an overclaim.

## What the full model's numbers already suggest

From the completed 10-fold run of the main model:

| | value |
|---|---|
| ADE | 0.1722 ± 0.0525 |
| **min-ADE over 5 samples** | **0.1705 ± 0.0525** |
| cosine, model | 0.9221 ± 0.0227 |
| cosine, train-mean baseline | 0.9109 ± 0.0166 |
| cosine, PCA ceiling | 0.9963 ± 0.0059 |
| retrieval accuracy | 0.6524 ± 0.0884 |

Two things stand out before this ablation is even run.

**The generative head has collapsed to a point estimate.** Drawing five samples
improves ADE by 0.0017 m — under two millimetres. A working generative head
produces meaningfully different trajectories across samples; this one does not.
Whatever the text pathway contributes, it is not coming from sampling.

**The cosine is barely above the do-nothing baseline.** Scaled into the
available headroom, `(0.9221 − 0.9109) / (0.9963 − 0.9109) = 0.13` — the model
captures 13% of the distance between "predict the average embedding" and
"perfect". On three folds (`scenario_10`, `5`, `6`) it scores *below* the
baseline. Retrieval accuracy at 0.652 against a 0.483 majority-text baseline is
the one number that clearly carries signal.

That is the context this ablation lands in. Do not assume it will show the text
helping.

## Files

| File | Difference from the full model |
|---|---|
| `data.py` | byte-identical |
| `encoders.py` | sentence encoder removed |
| `config.py` | text/PCA/flow settings removed; `match_params` added |
| `model.py` | flow head removed; head takes `cond` only |
| `metrics.py` | text report removed; min-ADE/FDE removed (deterministic model) |
| `run_cv.py` | no PCA, no flow loss, no sampling |
| `compare.py` | new — paired comparison against the full model |

`diff -r` the two folders to see the whole ablation at once.
