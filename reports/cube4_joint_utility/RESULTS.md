# Cube4 persistent auxiliary training: complete fixed comparison

10 October 2026 (America/New_York). Run `cube4_joint_utility/joint_v1`.

**Decision: retain DIRECT for this tested workflow.** Continuing JEPA prediction and SIGReg throughout scalar learning did not improve solver utility. PREDICT_EXACT qualified 114/768 requests versus DIRECT_EXACT 181/768; the paired difference was −8.724 percentage points (97.5% crossed-bootstrap interval [−14.844, −3.125]). All training and primary cost gates passed. The separate H2 reuse decision also failed: no verified solution and insufficient CPU saving. This is a negative result for a fixed, short-budget native Cube4 recipe, not a universal conclusion about JEPA or larger puzzles.

## What changed and why

The earlier native experiment froze its predictor after pretraining while the encoder and scalar head continued learning. Here, the original one-step auxiliary loss continues during every scalar update, keeping P trainable through the final PREDICT update. This addresses a concrete implementation concern. There is no concurrently trained frozen-P arm, so this study cannot identify why the previous predictor failed or estimate the causal effect of unfreezing it. Old outcomes are not pooled into these estimates.

Three recipes use identical paired encoder/head initialization, fresh data seeds and equal nominal 120 process-CPU-second API budgets. DIRECT learns scalar values; ALIGN adds action-free adjacent-state alignment plus SIGReg; PREDICT adds action-conditioned latent prediction plus SIGReg. Auxiliary recipes spend 20% of the numerical envelope in pretraining, 20% in scalar warmup and the remainder in Bellman learning; DIRECT uses 20% warmup and the remainder in Bellman learning. Numerical work stops against 108 CPU seconds from API entry, with checkpoint persistence still charged. Equal CPU does not mean equal optimizer updates, samples or BatchNorm exposure.

The architecture is unchanged: 96 physical-color facelets become 576 one-hot inputs; E is 576→256→64 with BatchNorm/ReLU, two width-64 residual blocks and a 64-dimensional projection. h consumes `[z,z_goal,z−z_goal,z*z_goal]` through 256→128 ReLU→1. PREDICT alone adds a 24×32 action embedding and residual predictor `z + MLP([z,e(a)])`, 96→256 ReLU→64.

ALIGN uses `MSE(z,z′)+0.1R`; PREDICT uses `MSE(P(z,a),z′)+0.1R`. R is the mean of native SIGReg on both endpoint batches (64 random projections, 17 knots). E is shared, both endpoints receive gradients, and no EMA or detached auxiliary target is used. h is frozen during pretraining. Scalar stages update E/h and, for PREDICT, P; scalar MSE is added to the same auxiliary loss with weight 1. DIRECT has scalar MSE only. Scalar targets use exact encodings, not predicted embeddings. Warmup uses minimum witnessed prefixes; Bellman tables use the current model in eval/no-grad mode over all 24 exact successors, terminal zero, and the historical clip [0,k]. Completed target tables are detached before fitting. No EMA or target network is added. Incomplete tables are retained without fitting. Adam 0.001, batch 256 and gradient norm cap 10 are fixed.

Sources: frozen protocol (historical artifact omitted from this compact release), config (historical artifact omitted from this compact release), joint training (historical artifact omitted from this compact release), and the original native modules bound in source freeze (historical artifact omitted from this compact release).

## Data, search and complete results

Six fresh paired seeds (71401–71406) × three recipes produced all 18 final models before evaluation. All six recipe orders occurred once. The panel contains 128 fresh length-14 walk roots, with true distance certified in [8,14] using the bound complete B4 ball and radius-three checks plus replayed witnesses. These are neither exact-distance-14 nor uniformly sampled Cube4 states. Training excludes protected roots and one-step neighborhoods, checking each training parent and all 24 successors; 35,200 states are protected. No panel distance labels enter training.

All 3,072 requests completed, with no late requests. Search executes exact color states, actions, duplicate checks, goals and independent output-path replay. Beam width is 32, parent-expansion cap 2,048, depth limit 80, score batch 512, and the complete warm-request deadline is 10 seconds. A **qualified** request returns a verified path of at most 14 moves on time; this is not certified optimality. The H2 arm uses the same final PREDICT checkpoint, scores two predicted transitions, then re-encodes the retained parent before the next expansion. E/P/h and BatchNorm state are frozen during every search.

| Arm | Qualified / 768 | Qualified rate | Any verified path / 768 | Request CPU seconds |
|---|---:|---:|---:|---:|
| DIRECT_EXACT | 181 | 23.568% | 202 | 97.355 |
| ALIGN_EXACT | 146 | 19.010% | 165 | 102.056 |
| PREDICT_EXACT | 114 | 14.844% | 132 | 106.982 |
| PREDICT_H2 | 0 | 0% | 0 | 105.126 |

Qualified counts by paired seed, in the fixed order above: DIRECT `[36,25,32,29,32,27]`; ALIGN `[28,27,22,23,24,22]`; PREDICT `[18,18,20,16,21,21]`; H2 `[0,0,0,0,0,0]`.

All unsuccessful requests stopped at the parent-expansion cap: DIRECT 566, ALIGN 603, PREDICT 636 and H2 768; the remaining requests returned verified solutions. Among verified solutions only, path length (minimum, median, maximum) was DIRECT (8,14,64), ALIGN (8,14,66) and PREDICT (8,14,64); H2 has no conditional length distribution. These selected-success summaries are not paired path-quality effects. Nine incomplete Bellman target tables (DIRECT 3, ALIGN 4, PREDICT 2) were retained without fitting, as prescribed; they are deadline-bound partial tables rather than failed model fits. Detailed stage/update counts are retained in the independent statistics review.

The fixed 10,000-draw bootstrap independently resamples six paired seed units and 128 shared query identities, retaining matched arms. It does not treat all 768 observations as independent. Draws are saved; six seeds limit precision. The two declared decisions use 97.5% intervals; other contrasts are descriptive.

| Comparison | Qualified difference, pp | Interval, pp | Role |
|---|---:|---:|---|
| PREDICT_EXACT − DIRECT_EXACT | −8.724 | [−14.844, −3.125], 97.5% | Primary |
| PREDICT_H2 − PREDICT_EXACT | −14.844 | [−21.354, −9.115], 97.5% | Separate reuse decision |
| ALIGN_EXACT − DIRECT_EXACT | −4.557 | [−9.375, 0.000], 95% | Descriptive |
| PREDICT_EXACT − ALIGN_EXACT | −4.167 | [−8.203, −0.521], 95% | Descriptive |

The primary required gain ≥3 pp and interval lower >0. Both fail, with the entire primary interval below zero. The primary classification is **`joint_recipe_utility_not_established`**. The reuse decision required interval lower >−2 pp and request CPU ratio ≤0.9. Both fail: CPU ratio is 0.982654 (about 1.735% saving), with a large quality loss. Its separate classification is **`joint_h2_reuse_not_established`**. The primary training and cost guards pass; H2 separately fails both retention and the 10% request-CPU-saving requirement.

Sources: panel preparation (historical artifact omitted from this compact release), input audit (historical artifact omitted from this compact release), runner (historical artifact omitted from this compact release), complete analysis (historical artifact omitted from this compact release), paired outcomes (historical artifact omitted from this compact release) and saved bootstrap draws (historical artifact omitted from this compact release).

## Actual costs and integrity

All 18 API CPU durations are 108.015–108.051 seconds, within the preset [105.6,120] interval. PREDICT/DIRECT aggregate API CPU ratio is 1.000022. Full charged training-plus-evaluation CPU is 746.419 seconds for DIRECT_EXACT and 755.906 for PREDICT_EXACT, ratio **1.012711**, below 1.1. ALIGN_EXACT and PREDICT_H2 charges are 750.936 and 754.053 seconds. These are separate deployment alternatives: each receives its own full training cell, measured cold load including validation, arm wrapper including row persistence, and the entire shared group overhead. They must not be summed as observed whole-study cost. The H2 saving gate specifically uses full warm-request CPU, excluding wrapper persistence, as fixed before dispatch.

| Guarded stage | Wall seconds | Child plus monitor-subprocess CPU | Guardian CPU | Peak sampled RSS, MiB |
|---|---:|---:|---:|---:|
| Preparation | 3.063 | 2.286 | 0.007 | 367.375 |
| Independent input audit | 3.064 | 2.621 | 0.007 | 385.781 |
| Training | 1950.807 | 1970.150 | 2.562 | 310.953 |
| Evaluation | 418.706 | 419.531 | 0.532 | 233.641 |
| Independent saved-result audit | 488.834 | 485.502 | 0.609 | 286.109 |
| Fixed analysis | 3.133 | 2.271 | 0.007 | 53.969 |

All six stage guards passed. These totals include their imports, persistence and process monitoring; they do not include earlier engineering, later independent statistics/report review or packaging. Timed numerical work was serial with one CPU thread and no competing experimental worker or scheduling-policy change; the desktop was not exclusive. Resource limits are sampled and cannot establish zero transient overshoot.

Twenty-four fabricated engineering tests and independent source and prospective analysis-code reviews preceded the frozen run. Two earlier order-fixture test failures were corrected before freeze and retained in the engineering record. Eight later fabricated packaging tests passed after hardening special-file and path checks; these tests do not add scientific trials.

The saved-result audit passed for all 18 models, 13,634 saved epochs, 163,217 updates, 3,072 requests, 499 returned-path replays and 128 input witnesses, binding 27,658 files. It reconstructed saved streams and target bounds, paired initialization, training/freeze commitments and search paths. It did **not** regenerate numerical Bellman predictions, gradients, optimizer trajectories or search rankings; SIGReg random projection draws were not replayed. The active audit log and its not-yet-written guardian receipt were intentionally excluded there and must be included in final closure, together with the completed analysis log/receipt.

Evidence: engineering record (historical artifact omitted from this compact release), independent result audit (historical artifact omitted from this compact release), all guardian receipts (historical artifact omitted from this compact release), independent statistics review (historical artifact omitted from this compact release), report review (historical artifact omitted from this compact release), and packaging code review (historical artifact omitted from this compact release). Source freeze SHA256: `f7899bcc20cbdaa2dc7bd5b8fc30088e63a8a351b5c8d84a84fa35d364d79c00`; input freeze: `723fe6c4900bee06e78dc7fe2c42e8e5bf562ca10c2745d1ea5a55b9dbcf9d2a`.

## Interpretation and reproducibility

Persistent one-step predictive training was insufficient for useful two-step latent scoring in this recipe. The study gives no support for adopting it over DIRECT at the tested budget. It does not establish that stale predictors were the earlier failure's cause, that prediction is inherently harmful, that SIGReg caused collapse, that reinforcement learning uniquely supplies the benefit, or that all JEPA architectures fail. PREDICT versus ALIGN changes predictor computation and optimization exposure as well as the action-conditioned objective; it is not a correct-action-specific causal ablation. No latent-error or collapse diagnostic was introduced after seeing these results.

The next adoption decision for this exact workflow is to keep DIRECT. A distinct future study would need a concrete workload or a different, prospectively justified intervention; this closed result does not authorize an automatic sweep, longer retry, selected-seed extension or retrospective cost normalization. Historical Cube3 gains and prior Cube4/W/RL negative or cost-failed evidence remain unchanged.

Frozen commands were `run_guarded.py prepare`, `inputs`, `train`, `evaluate`, `results`, `analyze` under `PYTHONPATH=src .venv/bin/python scripts/cube4_joint_utility/`. Existing identities reject overwrite/retry. A true reproduction requires a separate namespace and newly declared resources, not rerunning into this archive. For saved-file-only verification use:

```sh
.venv/bin/python reports/cube4_joint_utility/verify_closed.py
```

This standard-library verifier checks exact inventories, file hashes, source copies, review bindings, all completed guards and both classifications without loading models or rerunning science. Final closure destinations are manifest (historical artifact omitted from this compact release) and completion (historical artifact omitted from this compact release); the separate external receipt location is closed verification (historical artifact omitted from this compact release). This scientific report is finalized before packaging; archival completion requires the external verification receipt to report `passed`. Mutable state and STATUS/NEXT_STEPS are excluded. Only included Pro allowance was authorized; the minimum 10% reserve and no-credit/no-reset restrictions remain in force.
