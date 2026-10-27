# Direct budget allocation: independent confirmation complete

JEPA retains a positive practical search gain against the direct allocation selected by the fixed development rule. On 16 fresh paired replicas and 320 fresh Cube2 QTM tasks, primary verified coverage is **70.371% versus 66.992%**, a **+3.379 percentage-point** difference with crossed 95% interval **[1.719, 5.039]**. Exact-optimal coverage improves **+4.453 points [2.598, 6.309]**. The actual selected-direct/JEPA training CPU ratio is **0.994597**, within the prespecified [0.90, 1.10] range. All practical, quality and cost gates pass. Independent audit reproduces the decisions and replays every successful path.

This supports the fixed Cube2 recipe at 128 unique exact anchors and matched local CPU training cost after bounded direct-allocation selection. The candidate selected by the development rule is close to the preceding study's direct baseline; the new evidence is a fresh confirmation after comparing three budget allocations. It does not establish that the selected allocation is globally optimal, that all direct allocations lose on confirmation, or that JEPA improves Cube3 or rewriting.

## Calibration and development choice

The prospective protocol (historical artifact omitted from this compact release), precision rationale (historical artifact omitted from this compact release), and master `configs/bellman_allocation/study_v1.json` were frozen before scientific data preparation. A separate 29-pipeline timing calibration used old data and no quality outcomes. It fixed the allocations below; every candidate's median paired CPU ratio and complete 95% interval lay within [0.90, 1.10]. Common walk warmup is 1,000 updates for every model.

| Direct candidate | Prehistory updates | Static Bellman updates | Calibration CPU ratio, median [95% interval] | Development solved /144 | Development optimal /144 |
|---|---:|---:|---:|---:|---:|
| Prehistory-heavy, selected | 4,200 | 1,000 | 0.9928 [0.9921, 0.9964] | 93 | 71 |
| Downstream-heavy | 1,000 | 3,400 | 1.0046 [1.0013, 1.0142] | 92 | 76 |
| Balanced | 2,600 | 2,200 | 1.0159 [1.0005, 1.0523] | 90 | 73 |

The nine fresh development models reused the old 48-task panel with three paired seeds. The rule prioritizes verified coverage, then exact-optimal counts, then fixed candidate order. Prehistory-heavy wins on the first criterion by one success over downstream-heavy; no statistical superiority is claimed for this adaptive choice. Selection was saved before any confirmation fit. Downstream-heavy's larger optimal count did not override the prespecified primary objective. Both confirmed arms ultimately use the same 1,000-update downstream stage; the extra downstream label presentations occur only in the two development alternatives.

Evidence: `results/bellman_allocation/calibration_v1/{selection,validation}.json`, `development_v1/selection.json`, and `calibration_review.json`. All three candidate outcomes remain reported, including both unselected allocations.

## Fresh confirmation

Each of 16 fresh model/data pairs shares initial weights, a 4,096-parent walk pool, 128 exact anchors and paired warmup/downstream draws. JEPA uses 1,000 predictive updates, 1,000 walk warmup updates and 1,000 static-target updates. The selected direct model uses 4,200 direct walk-prehistory updates, followed by the same warmup and downstream stages. Architecture, optimizer, search and the numerical training cores are unchanged. One detached teacher table is computed from each warmup model; it is never refreshed. Encoder/scorer train during continuation and the predictor remains unchanged.

All 32 final checkpoints were frozen before evaluation. Search uses the scalar encoder/scorer on exact symbolic successors, with beam width32, maximum depth60 and fixed expansion budgets; the predictor and distance oracle are absent online. Returned paths are checked in the original frame. The 320 unique queries include80 at each exact depth4/6/8/10;24 separate geometry pairs are descriptive only.

| Search budget | Metric | JEPA | Selected direct | Paired difference, pp [95% interval] |
|---|---|---:|---:|---:|
| 256, primary | Verified coverage | 3,603/5,120 = 70.371% | 3,430/5,120 = 66.992% | +3.379 [1.719, 5.039] |
| 256, quality guard | Exact-optimal coverage | 3,127/5,120 = 61.074% | 2,899/5,120 = 56.621% | +4.453 [2.598, 6.309] |
| 512, descriptive | Verified coverage | 4,181/5,120 = 81.660% | 4,012/5,120 = 78.359% | +3.301 [1.250, 5.273] |
| 512, descriptive | Exact-optimal coverage | 3,127/5,120 = 61.074% | 2,899/5,120 = 56.621% | +4.453 [2.598, 6.309] |

Coverage and optimality differences at256 are positive in15 replica pairs and tied inone, with none negative. Each interval uses20,000 common crossed bootstrap draws over16 replica units and320 task units, stratified by depth. The5,120 observations per arm are not independent samples. The primary coverage point estimate exceeds3pp and its lower bound exceeds zero; the optimality lower bound exceeds the−2pp guard. Actual aggregate CPU is155.737791s for JEPA and154.896283s for selected direct. This is a local full-training-API CPU comparison, not exact FLOP or hardware-independent equivalence.

The new panel is smaller than the preceding1,024-task panel because only87 eligible unused depth4 states and4 depth3 geometry states remained. No role/depth restriction was relaxed, sample size extended, or result-driven retuning performed. Wider task uncertainty is disclosed in PRECISION.md (historical artifact omitted from this compact release). The interval's lower end does not establish a gain of at least3pp; the frozen practical criterion uses the point estimate together with evidence of a positive effect.

Evidence: `results/bellman_allocation/confirmation_v1/analysis/{summary.json,bootstrap_draws.npz,bootstrap_effects.npz}`, saved source/input/training rosters and the two confirmation configurations. Figures: primary effects and cost (historical artifact omitted from this compact release), development choice and confirmation quality (historical artifact omitted from this compact release). Both PNG/PDF pairs passed visual review.

## Separation, verification and resources

All19 new pools and the reserved344 query identities were frozen before development training. Every new parent and its12 immediate successors excludes the3,911-state union of prior and new evaluated relative queries. The new queries are distinct from prior evaluated queries. Of the344 new queries,139 overlap older studies' training parents or neighbors; those models are not reused. Thus novelty concerns evaluated queries and current-model exposure, not all historical graph states.

The independent audit verified all41 fresh prefixes/final fits, paired initial weights and common streams, full variable-length streams,41 static teacher tables and unchanged continuation predictors. It checked all20,912 saved search records and independently replayed all15,501 successful paths:275 development and15,226 confirmation. It reproduced the exact development choice, all four bootstrap endpoint arrays and the statistical/cost decisions. All34 focused tests passed. The42 pre-development and47 confirmation frozen current/archive bindings remain unchanged; the14-file calibration freeze also remains intact. The previous2,871-file artifact inventory is preserved, including copies of historical mutable handovers.

| Scope | Complete pipelines | Optimizer updates | Training CPU seconds | Training API wall seconds | Evaluation wall seconds |
|---|---:|---:|---:|---:|---:|
| Timing calibration, separate | 29 | 132,600 | 261.170 | 261.444 | 0 |
| Adaptive development | 9 | 52,200 | 87.510 | 87.683 | 2.564 |
| Independent confirmation | 32 | 147,200 | 310.634 | 310.950 | 101.472 |

Scientific training plus evaluation totals502.669 wall seconds under the recorded API boundaries. Calibration and development tuning overhead are disclosed separately, not hidden in the per-model confirmation cost match. There are2,432 anchor records across19 pools, representing2,425 distinct states:384 development records/distinct states,2,048 confirmation records/2,042 distinct states. Repeated exact-anchor presentations total13,260,800 across scientific training; unique labels and repeated presentations are different quantities. Each confirmed model sees256,000 anchor presentations from its128 unique anchors. No new labels are acquired during fitting. The41 teacher tables contain167,936 scalar target outputs; teacher forward time is included within continuation cost, not added a second time.

Authoritative evidence: `confirmation_v1/{independent_audit,resource_accounting}.json`, `prior_preservation.json`, and reproduction notes (historical artifact omitted from this compact release). This bounded study is complete. No label-efficiency curve, new architecture sweep or subsequent experiment is queued.
