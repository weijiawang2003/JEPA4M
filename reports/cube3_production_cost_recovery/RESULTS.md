# Cube3 production adjacency recipe: reviewed execution recovery

This report concerns the separately frozen recovery namespace. The original attempt failed at an input metadata guard before pool RNG initialization and produced no pool, labels, model fits, timing calibration or searches. It remains closed as engineering_prepare_failed_before_rng; its failed guard is not counted as a passed recovery stage.

The original preparation guard consumed 1.624369 wall seconds, reported separately from recovery API costs. Its115-file manifest and66-file calibration source freeze remain unchanged. The reviewed fix supplies the explicit timing-only disposable-support purpose marker; planned seeds, numerical recipes, selection, endpoints and budgets are unchanged. New caller-to-real-pool regression tests and fresh freezes preceded this recovery.

Original engineering closure (historical artifact omitted from this compact release); Failure diagnosis (historical artifact omitted from this compact release); Original sealed inventory (historical artifact omitted from this compact release).

**Saved decision: practical_gain_at_comparable_cpu.** The fixed ALIGN recipe meets every prespecified quality and realized-cost gate against the independently selected direct recipe.

Against **WALK_BELLMAN**, ALIGN's exact 8/10 optimal coverage difference is **+5.4565 [+3.7476, +7.2266] percentage points**, with the prespecified 95% crossed-bootstrap interval. Verified coverage differs by **+5.2734 [+3.5886, +7.0190] pp**. The selected-direct/ALIGN ratio of full confirmation API CPU sums is **1.000209**.

| Prespecified primary criterion | Observed | Threshold | Passed |
| --- | --- | --- | --- |
| Optimal point gain | +5.4565 pp | ≥ +3 pp | True |
| Optimal lower95 | +3.7476 pp | > 0 pp | True |
| Verified lower95 | +3.5886 pp | > −2 pp | True |
| Aggregate DIRECT/ALIGN CPU ratio | 1.000209 | inclusive [0.90,1.10] | True |

This is one conjunction on a fresh confirmation set conditional on independent development selection. The practical point threshold is not a requirement that the lower confidence bound exceed three pp. A failed realized-cost gate takes priority in the classification. Harder states, a different checkpoint, another candidate or additional samples cannot rescue a failure.

## Independent direct recipe selection

All three direct allocations passed timing before development. Nine fresh fits—three paired replicas per candidate—completed before their 1,152 searches on 128 exact queries. Selection used only the total exact8/10 optimal successes (192 attempts per candidate), with verified successes and then the fixed order WALK_PRE, WALK_BELLMAN, ANCHOR_WARM breaking ties. ALIGN was not fitted or evaluated in development.

| Candidate | Optimal / 192 | Verified / 192 | Selected |
| --- | --- | --- | --- |
| WALK_PRE | 76 | 77 | False |
| WALK_BELLMAN | 79 | 81 | True |
| ANCHOR_WARM | 77 | 78 | False |

The winner was bound before confirmation training: **WALK_BELLMAN**, with prehistory/warmup/Bellman updates **1000/1000/6050**. The selector is noisy and limited; it does not identify the globally best direct baseline. Walk-only warmup-heavy and balanced allocations were omitted prospectively for tractability.

## Fresh confirmation results

Sixteen fresh paired model/data units supply 32 final models. Every final existed and passed integrity checks before any confirmation search. The common panel has 1,024 newly certified roots, 256 each at exact QTM8/10/11/12, producing 32,768 confirmation searches. Each condition/depth cell contains 4,096 attempts. Failures remain in all verified and optimal denominators. Development and historical results do not enter confirmation estimates.

The saved 10,000 common crossed-bootstrap draws resample 16 paired units and independently resample 256 queries within each depth. The same draws apply to both metrics and both conditions; each population weights its two depth strata equally. Exact8/10 is primary and exact11/12 descriptive. Percentile limits are the fixed linear .025/.975 quantiles. This approximate procedure has no finite-sample or power guarantee.

| Population | Metric | ALIGN−selected direct [95% interval], pp | Scope |
| --- | --- | --- | --- |
| easy8_10 | optimal | +5.4565 [+3.7476, +7.2266] | primary |
| easy8_10 | verified | +5.2734 [+3.5886, +7.0190] | primary |
| hard11_12 | optimal | +0.6714 [-0.0732, +1.4648] | descriptive |
| hard11_12 | verified | +0.6592 [-0.1465, +1.5015] | descriptive |

| Condition | Exact depth | Verified / 4,096 | Optimal / 4,096 | Verified | Optimal |
| --- | --- | --- | --- | --- | --- |
| ALIGN | 8 | 2955 | 2933 | 72.1436% | 71.6064% |
| WALK_BELLMAN | 8 | 2684 | 2655 | 65.5273% | 64.8193% |
| ALIGN | 10 | 646 | 629 | 15.7715% | 15.3564% |
| WALK_BELLMAN | 10 | 485 | 460 | 11.8408% | 11.2305% |
| ALIGN | 11 | 263 | 254 | 6.4209% | 6.2012% |
| WALK_BELLMAN | 11 | 219 | 205 | 5.3467% | 5.0049% |
| ALIGN | 12 | 56 | 49 | 1.3672% | 1.1963% |
| WALK_BELLMAN | 12 | 46 | 43 | 1.1230% | 1.0498% |

## Architecture, training and scope

Inputs are 54 goal-relative facelet colors, expanded to 324 flattened one-hot entries. Shared encoder E uses 324→256→64 BatchNorm/ReLU layers, two 64-wide residual blocks and a final 64-dimensional projection. Scalar h consumes [z,zgoal,z−zgoal,z*zgoal] through 256→128 ReLU→1. Legacy CubeModel initialization is consumed once to preserve E/h tensors and Torch RNG position; only E/h remain. Construction is charged. No predictor forward, backward, Adam state or routine checkpoint tensor remains.

ALIGN prehistory updates E using unit mean-squared latent alignment of actual adjacent states plus 0.1 times the mean of two endpoint SIGReg terms. E is shared and both endpoints are live; there is no detach, separate target encoder or EMA. SIGReg uses 64 random directions per endpoint. Direct prehistory instead updates E/h on witnessed walk bounds plus identity. ALIGN retains its fixed 1000/1000/4000 recipe selected by the predeclared T late-usefulness rule, not by ranking T point estimates or choosing a best active duration.

All recipes use Adam0.001, separate module norm10 clipping and resets only at stage boundaries. Walk warmup uses 256 parents plus identity. ANCHOR_WARM uses 256 parents, 256 exact anchors and identity with three unit MSEs, and a private warmup-anchor RNG. Continuation updates E/h on the same three-part scalar loss. Each recipe constructs its own single detached hard-copy post-warmup teacher over all12 exact successors, clipped to witnessed upper bounds and forced identity zero; targets never refresh. Common downstream parent/anchor streams are paired. Prehistory streams need not be identical because ALIGN consumes action draws.

The anchor-aware comparator changes label timing, label exposure, BatchNorm batches and teacher quality together. Equal access to128 exact labels per pool does not mean equal presentations. A positive result supports this compact adjacency+SIGReg recipe against one selected member of three direct families. It does not isolate enduring representation benefit, require a JEPA predictor, establish correct-action/group structure, optimize the full baseline frontier, or show Cube4 transfer. A failed gate is not a proof of no benefit.

The fresh supports contain 4,096 parents and128 independently preselected shallow exact labels per paired unit. All19 index selections preceded any scientific anchor lookup, giving2,432 label records before deduplication. New panel roots and all12 successors exclude the named142,202-state historical union and the disposable calibration footprint, and the two new panels have disjoint footprints. Scientific parents and successors avoid both new panels. These exclusions do not cover every historical training state or search frontier. Shell-conditioned nonbacktracking proposals are not uniform Cube3 state sampling.

Search uses exact symbolic successors and frozen E/h, a cached identity encoding per model, and forced identity zero. Beam width32, expanded-parent cap512, depth cap60 and batch512 are common. Every successful solution was replayed independently in relative and original frames. Predictor and exact-distance oracle calls during search are zero.

## Scalar heuristic diagnostics

These compare the learned scalar h with exact root distance; they are not raw latent-distance or successor-ranking measurements.

| Condition | Exact depth | Mean score | Signed bias | MAE |
| --- | --- | --- | --- | --- |
| ALIGN | 8 | 9.699588 | 1.699588 | 2.555929 |
| WALK_BELLMAN | 8 | 9.576856 | 1.576856 | 2.544957 |
| ALIGN | 10 | 11.105462 | 1.105462 | 2.398064 |
| WALK_BELLMAN | 10 | 10.902647 | 0.902647 | 2.413400 |
| ALIGN | 11 | 11.760287 | 0.760287 | 2.162192 |
| WALK_BELLMAN | 11 | 11.551516 | 0.551516 | 2.135209 |
| ALIGN | 12 | 12.218553 | 0.218553 | 1.890703 |
| WALK_BELLMAN | 12 | 12.067178 | 0.067178 | 1.861707 |

## Scientific resource accounting

| Scope | Fresh fits | Training API CPU seconds | Summed API wall seconds |
| --- | --- | --- | --- |
| Development | 9 | 166.996054 | 167.256844 |
| Confirmation ALIGN | 16 | 295.555464 | 296.002295 |
| Confirmation WALK_BELLMAN | 16 | 295.617165 | 295.900028 |

There are41 scientific fits and35 separate calibration fits, with33,920 total scientific searches (1,152 development plus32,768 confirmation). Scientific training APIs total 758.168683 CPU seconds. Only the32 confirmation APIs determine the realized cost guard; calibration/development are reported sunk work, not extra confirmation replicas.

| Phase | Recipe | Fits | bellman_anchor_presentations | bellman_updates | encoder_examples | encoder_forwards | exact_label_presentations | goal_label_presentations | predictor_forwards | prehistory_updates | transition_presentations | walk_label_presentations | warmup_anchor_presentations | warmup_updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | WALK_PRE | 3 | 768000 | 3000 | 8747850 | 31050 | 768000 | 31050 | 0 | 25050 | 0 | 7180800 | 0 | 3000 |
| development | WALK_BELLMAN | 3 | 4646400 | 18150 | 10852950 | 24150 | 4646400 | 24150 | 0 | 3000 | 0 | 1536000 | 0 | 3000 |
| development | ANCHOR_WARM | 3 | 768000 | 3000 | 11159250 | 23250 | 5184000 | 23250 | 0 | 3000 | 0 | 5184000 | 4416000 | 17250 |
| confirmation | ALIGN | 16 | 16384000 | 64000 | 45136000 | 112000 | 16384000 | 80000 | 0 | 16000 | 4096000 | 4096000 | 0 | 16000 |
| confirmation | WALK_BELLMAN | 16 | 24780800 | 96800 | 57882400 | 128800 | 24780800 | 128800 | 0 | 16000 | 0 | 8192000 | 0 | 16000 |

The counters above expose update, parent/walk, exact-anchor, identity, neighbor-transition and encoder work separately. Static-teacher construction is part of the complete API and reported in its stage timers. Exact anchor presentations can differ despite the shared128-label identities.

| Evaluation phase | Searches | Cell API CPU seconds | Summed cell API wall seconds | Offline-root CPU subset |
| --- | --- | --- | --- | --- |
| development | 1152 | 15.819465 | 15.850328 | 0.004634 |
| confirmation | 32768 | 440.359687 | 441.085233 | 0.043558 |

## Independent timing calibration

The timing phase completed 35 fresh fits: one excluded priming fit, 18 paired discovery fits and 16 independent validation fits.

Calibration used one disposable 4,096-parent support with 128 preselected shallow exact anchors. These models were never evaluated on search quality. Timing was serial with one CPU thread, using fixed cyclic orders. The active recipe was ALIGN 1000/1000/4000; direct bases were 1000/1000/1000. Walk and anchor-aware warmup used separate measured base gaps and slopes. Counts were rounded upward to 50 updates within the prospective 1,000–20,000 adjustable-stage bound.

| Slot | Role | Recipe | Pre / warm / Bellman | CPU seconds | API wall seconds |
| --- | --- | --- | --- | --- | --- |
| priming | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.958411 | 18.037577 |
| discovery0 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.665167 | 17.682413 |
| discovery0 | WALK_BASE | WALK_PRE | 1000 / 1000 / 1000 | 8.416708 | 8.423968 |
| discovery0 | WALK_PRE_PROBE | WALK_PRE | 4000 / 1000 / 1000 | 12.232700 | 12.244342 |
| discovery0 | WALK_BELLMAN_PROBE | WALK_BELLMAN | 1000 / 1000 / 4000 | 14.060275 | 14.072077 |
| discovery0 | ANCHOR_BASE | ANCHOR_WARM | 1000 / 1000 / 1000 | 8.843565 | 8.847599 |
| discovery0 | ANCHOR_WARM_PROBE | ANCHOR_WARM | 1000 / 4000 / 1000 | 14.445821 | 14.455067 |
| discovery1 | WALK_PRE_PROBE | WALK_PRE | 4000 / 1000 / 1000 | 12.314357 | 12.324964 |
| discovery1 | WALK_BELLMAN_PROBE | WALK_BELLMAN | 1000 / 1000 / 4000 | 13.912861 | 13.924146 |
| discovery1 | ANCHOR_BASE | ANCHOR_WARM | 1000 / 1000 / 1000 | 9.039032 | 9.046615 |
| discovery1 | ANCHOR_WARM_PROBE | ANCHOR_WARM | 1000 / 4000 / 1000 | 14.525993 | 14.536969 |
| discovery1 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.714389 | 17.727575 |
| discovery1 | WALK_BASE | WALK_PRE | 1000 / 1000 / 1000 | 8.324126 | 8.328272 |
| discovery2 | ANCHOR_BASE | ANCHOR_WARM | 1000 / 1000 / 1000 | 8.864539 | 8.869652 |
| discovery2 | ANCHOR_WARM_PROBE | ANCHOR_WARM | 1000 / 4000 / 1000 | 14.447079 | 14.457628 |
| discovery2 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.823015 | 17.838237 |
| discovery2 | WALK_BASE | WALK_PRE | 1000 / 1000 / 1000 | 8.400990 | 8.428556 |
| discovery2 | WALK_PRE_PROBE | WALK_PRE | 4000 / 1000 / 1000 | 12.256804 | 12.263670 |
| discovery2 | WALK_BELLMAN_PROBE | WALK_BELLMAN | 1000 / 1000 / 4000 | 13.722236 | 13.730819 |
| validation0 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.862165 | 17.876198 |
| validation0 | WALK_PRE | WALK_PRE | 8350 / 1000 / 1000 | 18.030307 | 18.046069 |
| validation0 | WALK_BELLMAN | WALK_BELLMAN | 1000 / 1000 / 6050 | 17.390969 | 17.399260 |
| validation0 | ANCHOR_WARM | ANCHOR_WARM | 1000 / 5750 / 1000 | 17.580796 | 17.592332 |
| validation1 | WALK_PRE | WALK_PRE | 8350 / 1000 / 1000 | 18.280995 | 18.295968 |
| validation1 | WALK_BELLMAN | WALK_BELLMAN | 1000 / 1000 / 6050 | 17.648673 | 17.663820 |
| validation1 | ANCHOR_WARM | ANCHOR_WARM | 1000 / 5750 / 1000 | 17.617573 | 17.631043 |
| validation1 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.844059 | 17.855707 |
| validation2 | WALK_BELLMAN | WALK_BELLMAN | 1000 / 1000 / 6050 | 17.496812 | 17.507265 |
| validation2 | ANCHOR_WARM | ANCHOR_WARM | 1000 / 5750 / 1000 | 17.373679 | 17.382173 |
| validation2 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.628477 | 17.642022 |
| validation2 | WALK_PRE | WALK_PRE | 8350 / 1000 / 1000 | 18.211142 | 18.228859 |
| validation3 | ANCHOR_WARM | ANCHOR_WARM | 1000 / 5750 / 1000 | 17.617902 | 17.631780 |
| validation3 | ACTIVE | ALIGN | 1000 / 1000 / 4000 | 17.460057 | 17.468991 |
| validation3 | WALK_PRE | WALK_PRE | 8350 / 1000 / 1000 | 17.913916 | 17.924511 |
| validation3 | WALK_BELLMAN | WALK_BELLMAN | 1000 / 1000 / 6050 | 17.882961 | 17.907911 |

| Direct family | Measured gap (CPU s) | Slope (CPU s/update) | Raw total stage count | Rounded total stage count | Pre / warm / Bellman |
| --- | --- | --- | --- | --- | --- |
| WALK_PRE | 9.390263 | 0.001285271 | 8306.054960 | 8350 | 8350 / 1000 / 1000 |
| WALK_BELLMAN | 9.390263 | 0.001862912 | 6040.637819 | 6050 | 1000 / 1000 / 6050 |
| ANCHOR_WARM | 8.821602 | 0.001860847 | 5740.638849 | 5750 | 1000 / 5750 / 1000 |

| Family | Median DIRECT/ALIGN CPU | 95% interval | All-bound gate passed |
| --- | --- | --- | --- |
| WALK_PRE | 1.025240 | [1.009413, 1.033052] | True |
| WALK_BELLMAN | 0.990791 | [0.973620, 1.024221] | True |
| ANCHOR_WARM | 0.986427 | [0.984248, 1.009040] | True |

All three candidate medians and their entire 95% intervals must lie in [0.90,1.10]. Intervals use the saved 20,000 shared paired four-block bootstrap draws. Four blocks provide an approximate engineering guard, not guaranteed equivalence. No family may be dropped or recalibrated.

Calibration training APIs total 528.808551 CPU seconds and 529.294057 summed API wall seconds, including the excluded priming fit. The enclosing calibration process recorded 528.927485 CPU seconds and 529.424539 wall seconds. These scopes overlap and must not be added.

The full API clock includes validation, compatibility initialization, source/config archives, updates, the own static teacher, prescribed checkpoint serialization and final integrity checks. Timing-ledger serialization, input preparation and later independent audit are outside the API. Stage timers overlap the API. Local measured CPU cost is not FLOP equality or hardware-independent efficiency.

## Preparation, guards and audit

Calibration support contains 128 exact-anchor records and 128 unique anchor states. Independent calibration input proof checked 1 pool(s) and reports zero new production oracle/model calls. The final saved-result audit took 37.436303 CPU seconds and 38.117572 wall seconds; it introduced no training, neural search evaluation or oracle queries.

| Panel phase | Proposals | Accepted roots | Certificate queries | Certificate CPU seconds | Certificate wall seconds |
| --- | --- | --- | --- | --- | --- |
| development | 159 | 128 | 157 | 2.230268 | 2.232197 |
| confirmation | 1230 | 1024 | 1214 | 17.619385 | 17.634915 |

Scientific preparation records 2432 shallow labels, 2113 unique anchor states, 77824 parents and 933888 successor records. Independent scientific input proof used 22.881528 CPU seconds and 23.094403 wall seconds. Full preparation accounting is linked below.

| Completed bounded stage | Outer wall seconds | Sampled process-group peak MiB |
| --- | --- | --- |
| calibration_prepare | 9.489967 | 729.641 |
| calibration_audit_inputs | 1.369060 | 670.875 |
| calibrate | 530.320822 | 316.250 |
| prepare | 247.733778 | 762.266 |
| audit_inputs | 23.832801 | 702.312 |
| development_train | 168.328306 | 302.531 |
| development_evaluate | 7.689715 | 1062.234 |
| select | 1.907256 | 217.016 |
| confirmation_train | 593.290761 | 319.438 |
| confirmation_evaluate | 114.775495 | 1069.203 |
| analyze | 16.262568 | 900.312 |

Full API CPU sums, summed per-cell wall times, parallel evaluation dispatch, preparation and outer guard wall times have different scopes. Component clocks overlap enclosing APIs and must not be added twice. Training is serial; four single-thread evaluation workers are allowed only after its timed phase ends. Sparse full snapshots occur at250/500/1000/end per stage, with deduplicated steps and records at1/every250/end; no overwritten latest checkpoint.

## Sources and complete tables

- Authoritative configuration (historical artifact omitted from this compact release); Prospective protocol (historical artifact omitted from this compact release); Calibration source freeze (historical artifact omitted from this compact release).
- Calibration decision (historical artifact omitted from this compact release); Disposable support (historical artifact omitted from this compact release); Calibration input proof (historical artifact omitted from this compact release); Independent saved-result audit (historical artifact omitted from this compact release).
- Scientific source freeze (historical artifact omitted from this compact release); Independent development selection (historical artifact omitted from this compact release); Confirmation analysis (historical artifact omitted from this compact release); Paired outcomes and fixed draws (historical artifact omitted from this compact release); Scientific input proof (historical artifact omitted from this compact release).
- calibration_fits: 35 rows (historical artifact omitted from this compact release); calibration_allocations: 3 rows (historical artifact omitted from this compact release); calibration_validation: 3 rows (historical artifact omitted from this compact release); contrasts: 4 rows (historical artifact omitted from this compact release); cells: 8 rows (historical artifact omitted from this compact release); per_seed: 128 rows (historical artifact omitted from this compact release); root_calibration: 8 rows (historical artifact omitted from this compact release); training_costs: 32 rows (historical artifact omitted from this compact release); evaluation_costs: 32 rows (historical artifact omitted from this compact release); development_candidates: 3 rows (historical artifact omitted from this compact release); development_cells: 12 rows (historical artifact omitted from this compact release); development_per_seed: 36 rows (historical artifact omitted from this compact release); development_root_calibration: 12 rows (historical artifact omitted from this compact release); development_training_costs: 9 rows (historical artifact omitted from this compact release); development_evaluation_costs: 9 rows (historical artifact omitted from this compact release).

This report adds no statistics, resampling, training, neural forwards, searches or oracle queries. Immutable reviewed closure ends this fixed U design. The broader user-authorized research program may consider a distinct prospective question afterward; closure does not claim that the entire program has stopped.
