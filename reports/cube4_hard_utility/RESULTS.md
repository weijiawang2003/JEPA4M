# Native Cube4 hard-population utility

## Outcome

**The charged predictive pretraining prefix did not establish useful native Cube4 solver benefit in this fixed study.** The final classification is `native_cube4_hard_utility_not_established`. All training, evaluation, audit and analysis stages completed; all governing CPU-cost gates passed. This is an efficacy result under the stated execution limitations, not a cost-guard failure or an incomplete experiment.

| Recipe | Verified solves / 6,144 | Solve rate | Verified returned path ≤14 / 6,144 | ≤14 coverage |
|---|---:|---:|---:|---:|
| DIRECT | 1,751 | 28.4993% | 1,606 | 26.1393% |
| ALIGN | 1,825 | 29.7038% | 1,685 | 27.4251% |
| PREDICT | 1,755 | 28.5645% | 1,614 | 26.2695% |

PREDICT−DIRECT is **+0.065104 percentage points**, with a **97.5% crossed-bootstrap interval [-1.937052, +2.116089] pp**: just four additional solved model–query attempts. Both efficacy gates fail. The interval includes zero, and its upper endpoint (+2.116089 pp) is below the predeclared +3 pp practical target. This constrains that target under this study's approximate uncertainty model; it is not a proof of exact equivalence or absence of all smaller benefits.

| Descriptive comparison | Difference, pp | 95% interval, pp |
|---|---:|---:|
| ALIGN−DIRECT | +1.204427 | [-0.325521, +2.718099] |
| PREDICT−ALIGN | -1.139323 | [-2.587891, +0.276693] |

Both descriptive intervals include zero. ALIGN has the highest point estimate, but its benefit is not established and cannot replace the primary comparison. All 18,432 attempts comprise 5,331 verified solves and 13,101 node-budget stops, with zero cooperative wall overruns. PREDICT returns eight more paths within the 14-step witness bound than DIRECT; this descriptive endpoint does not change the primary decision.

| Recipe | Mean successful path length | Median | Range | Successful paths longer than 14 |
|---|---:|---:|---:|---:|
| DIRECT | 14.667047 | 14 | 8–66 | 145 |
| ALIGN | 14.353973 | 14 | 8–66 | 140 |
| PREDICT | 14.349858 | 14 | 8–66 | 141 |

These path lengths condition on success, so differing solved subsets prevent a clean unconditional quality comparison. Full length histograms and stop counts are retained in analysis.json (historical artifact omitted from this compact release). The independent statistical review (historical artifact omitted from this compact release) reconstructed every count and the intervals from the saved common draws without new resampling.

**Practical decision:** keep DIRECT as the default for this fixed 120-CPU-second native solver. The evidence does not justify adding this PREDICT prefix. The approximately 28.5% DIRECT solve rate leaves room for improvement, so the earlier easy-population ceiling is no longer the explanation for failing to see useful benefit here. This does not establish whether advantages grow with difficulty or cube order, and it does not settle other JEPA architectures, joint objectives, longer budgets or latent search. No next numerical study is selected.


## Question and provenance

This study asks whether an action-conditioned predictive pretraining prefix improves verified native Cube4 solving when its computation is charged to the same training budget as direct fitted value learning. The single primary comparison is PREDICT−DIRECT. ALIGN is a descriptive adjacent-embedding control. This is a prospective, development-informed test of one complete recipe, population and search budget.

The preceding native difficulty pilot remains closed as `design_inconclusive`: its primary exact7 and walk10 populations exceeded the prespecified DIRECT ceiling gates, and it trained no ALIGN or PREDICT models. Its descriptive walk14 result, 18/64 solved model–query attempts, motivated this separately reviewed study. Those 64 attempts involved two models on 32 shared roots. None enters this study's effect estimate. The old pilot selected no budget; the new study separately prespecified 120 CPU seconds using the prior development and feasibility evidence. See the candidate review (historical artifact omitted from this compact release), prior result (historical artifact omitted from this compact release), current protocol (historical artifact omitted from this compact release) and configuration (historical artifact omitted from this compact release).

## Population and exact environment

The new panel contains 512 inverse-pruned, 14-move goal-walk roots, each certified to have true distance in **[8,14]** in the fixed single-layer QTM metric. A complete local radius-three enumeration is disjoint from the verified goal radius-four ball, ruling out every path of length at most seven; a replayed inverse generating word supplies the upper bound. These are not exact-distance-14 states or uniform unrestricted Cube4 scrambles. Preparation retained all 513 proposals, including one rejection below the minimum distance.

The state is 96 physical-color facelets in a fixed canonical frame. There are 24 outer/adjacent-inner single-layer quarter-turn generators; action inverses use index XOR 1. This is native Cube4, with no wide-move shortcut, rotation quotient, arbitrary-goal inversion or transferred Cube3 weights. New root/successor families avoid one another and the complete named historical/pilot/unused-formal protected union. The actual protected union contains 32,000 distinct states. This count is reconstructed from sets, not assumed from a free group action. Training rejects a parent when it or any immediate successor touches that union. The exclusion does not cover every historical training state or every deeper search frontier. Sources: environment (historical artifact omitted from this compact release), new data and proof (historical artifact omitted from this compact release), input audit (historical artifact omitted from this compact release).

## Actual model and learning procedure

E one-hot encodes the 96 six-class facelets and flattens them to 576 inputs. It applies 576→256 BN/ReLU, 256→64 BN/ReLU, two width-64 residual blocks (Linear/BN/ReLU/Linear/BN, skip addition and ReLU), and a final 64→64 linear projection. There is no pooling or latent normalization. The scalar head h receives `[z,z_goal,z−z_goal,z*z_goal]` and applies 256→128 ReLU→1. E/h have 219,137 parameters, with byte-identical initialization within each paired unit.

PREDICT adds a training-only 24×32 action embedding and residual predictor `z + MLP([z,e(a)])`, with layers 96→256 ReLU→64. Its extra 42,048 parameters bring training capacity to 261,185; inference capacity remains unchanged. DIRECT has no latent pretraining. ALIGN minimizes adjacent-embedding MSE; PREDICT minimizes next-embedding prediction MSE. Both add 0.1 times the mean endpoint SIGReg statistic, using 64 random projections and 17 integration knots. The endpoint branches share E and both receive gradients; there is no detached pretraining target or EMA. This is the existing Epps–Pulley characteristic-function SIGReg implementation, not a variance-only penalty. The experiment does not isolate SIGReg causality or diagnose collapse. Sources: [encoder, head and SIGReg](../../src/jepa4math/models.py), native model and losses (historical artifact omitted from this compact release).

h is frozen during pretraining; E and h update during warmup and Bellman training, while P is frozen. Each epoch draws 128 fresh inverse-pruned goal walks of length 24, deduplicates states in first-seen order and assigns minimum observed prefix lengths as witnessed upper bounds. These are not exact distance labels. Pretraining uses this protected-safe support, independently drawing one uniform action from the 24 generators for each retained state; its exact successor supplies the paired endpoint. Warmup fits the bounds. Each subsequent Bellman epoch constructs a fresh detached, eval-mode table over all 24 exact successors: `min(k, max(0, 1 + min_a h(T_a(s))))`, with goal parent/child boundary values zero. Both scalar stages minimize mean-squared error. Target forwards preserve BN buffers and RNG. Only complete target tables are eligible for fitting, at most once per row in a shuffled pass subject to the deadline; all partial tables and fitting epochs are retained. Adam uses learning rate 0.001, batch 256 and per-module gradient clipping at 10; it resets between pretraining and warmup, then continues into Bellman training. Phase-specific data/RNG streams share the paired seeds, while the amount consumed can differ. Sources: fresh support (historical artifact omitted from this compact release), targets and training stages (historical artifact omitted from this compact release).

All arms receive a nominal 120-second complete-API CPU budget. The numerical-loop stopping deadline is set at 90% of that budget, or 108 process-CPU seconds elapsed from API entry. Pretraining, when present, has a deadline at 20% of this envelope; warmup ends after an additional 20%, or at 20% for DIRECT. These timers include initialization, sampling and streamed persistence, and deadlines are checked between work units rather than enforcing exact optimization-time shares. Actual API cost and any overshoot remain charged. DIRECT therefore receives a longer Bellman allocation. Pretraining is charged, with no idle padding or update-count matching. Twelve new paired model/data units (model seeds 71201–71212) use all six arm orders twice. All 36 final checkpoints existed before learned evaluation. No checkpoint, seed, architecture, duration or panel was selected from the new outcomes.

## Search, statistics and decision rule

Every final model attempts the same 512 roots at a single cap of 2,048 parent expansions: 18,432 searches. This cap does not count neural scores, generated successors or distinct visited states. Search uses exact simulator successors and frozen E/h scores, retaining candidates with lower learned values, with immediate-inverse pruning and a global visited set. Goal detection precedes scoring. Beam width is 32, maximum depth 80, score batch 512 and the cooperative wall deadline is 10 seconds. P is absent from inference; no distance oracle or generating witness guides the search. Successful action sequences are replayed independently. The returned path may exceed the known 14-step witness. Sources: runner (historical artifact omitted from this compact release), [search](../../src/jepa4math/search.py).

The primary endpoint is verified solving. Useful support requires PREDICT−DIRECT ≥3 percentage points and the lower endpoint of its two-sided 97.5% interval >0, with every governing cost gate passing. The fixed analysis uses 10,000 crossed paired bootstrap draws over 12 model/data units and 512 common query indices, shared across arms. This is approximate uncertainty conditional on the selected sampler and development-informed recipe, not inference from 18,432 independent models. ALIGN−DIRECT and PREDICT−ALIGN use descriptive 95% intervals. The draws and full outcome arrays are saved. Sources: fixed analysis (historical artifact omitted from this compact release) and the configuration above.

Every DIRECT/PREDICT full training API must be in [105.6,120] CPU seconds; their summed API CPU ratio must be in [0.9,1.1]; their summed full training-cell plus full evaluation-cell CPU ratio must be ≤1.1. A governing cost failure takes precedence as `realized_cost_guard_failed`. ALIGN costs are reported separately. Common preparation, development and audit costs cannot dilute the treatment ratio. API, cell and stage scopes overlap and must not be added together.

## Operational limitation

At 2026-10-08 03:47:50.748461 UTC, after the first DIRECT fit completed and during the first ALIGN fit, the coordinator ran `taskpolicy -B -p 25055` on this study's training worker, requesting removal of PRIO_DARWIN_BG in response to slow wall progress. It returned 0. The prior background status and any causal speed effect were unknown. A limited process snapshot showed ordinary desktop/cloud-file-provider activity; no unrelated applications or services were stopped.

No scientific source, seed, thread count, budget, guard or checkpoint changed, no model was restarted, and no learned evaluation result was then available. All earlier work and CPU/wall measurements remain charged. Nevertheless, this was an asymmetric within-run operational intervention: the first DIRECT preceded it, ALIGN straddled it, and later fits followed it. Balanced arm ordering and the bootstrap do not remove that limitation. CPU seconds are not fixed FLOPs. Any result applies to the actual execution and cannot establish an unchanged scheduling environment, matched wall-time efficiency or a causal speedup from that command. See the observation (historical artifact omitted from this compact release) and independent runtime review (historical artifact omitted from this compact release).

## Interpretation and audit limits

This comparison can support or constrain the complete charged predictive-prefix recipe on this bounded native Cube4 population. It cannot establish a difficulty interaction, a Cube4-versus-Cube3 interaction, shortest-path optimality, universal JEPA benefit or failure, or an isolated prediction/SIGReg mechanism. Solved-only path lengths are selection affected; returned length ≤14 is witness-bound quality rather than optimality. A failure of the success rule is not an equivalence or zero-effect result.

The saved-only scientific audit checks source/input/checkpoint lineage, the complete fixed grid, replayable action/RNG streams and support filtering, stage freezing, target provenance, retained partial work, counters/costs and successful paths. It reconstructs initial tensors without model forwards. It does not rerun neural Bellman values or learned search rankings, and is not a training replication. Numerical stage failures cannot be relabeled as efficacy evidence. The historical pilot and all previously closed studies remain unchanged. No result automatically selects another experiment.

## Realized computation and full retained grid

| Recipe | Sum API CPU, s | Sum full training-cell CPU, s | Sum full evaluation-cell CPU, s | Training-cell wall, s | Evaluation-cell wall, s |
|---|---:|---:|---:|---:|---:|
| DIRECT | 1296.229054 | 1297.768201 | 581.386480 | 1595.846107 | 586.685331 |
| ALIGN | 1296.228578 | 1297.966053 | 576.299512 | 1446.175904 | 581.536490 |
| PREDICT | 1296.235197 | 1297.855437 | 576.361295 | 1431.944538 | 580.912874 |

All 36 full APIs are in [108.012419, 108.074356] CPU seconds; DIRECT, ALIGN and PREDICT each pass their API bands. PREDICT/DIRECT summed API CPU is **1.000004739** (allowed 0.9–1.1). The full training-plus-evaluation-cell CPU ratio is **0.997372250** (maximum 1.1). These are the fixed CPU-accounting gates, not an equal-wall-time or fixed-FLOPs claim. Cell scopes include setup/imports/bindings/load/persistence up to their own final caller/summary JSON write; whole-stage records retain that remaining overhead.

| Model seed | DIRECT solved /512 | ALIGN solved /512 | PREDICT solved /512 | DIRECT complete Bellman epochs | ALIGN | PREDICT |
|---|---:|---:|---:|---:|---:|---:|
| 71201 | 65 | 71 | 56 | 101 | 109 | 94 |
| 71202 | 155 | 166 | 160 | 593 | 430 | 437 |
| 71203 | 148 | 168 | 162 | 587 | 449 | 442 |
| 71204 | 149 | 160 | 157 | 585 | 436 | 434 |
| 71205 | 168 | 157 | 151 | 577 | 435 | 431 |
| 71206 | 148 | 155 | 148 | 600 | 433 | 426 |
| 71207 | 153 | 152 | 141 | 557 | 420 | 425 |
| 71208 | 148 | 155 | 162 | 574 | 430 | 434 |
| 71209 | 146 | 152 | 145 | 584 | 438 | 426 |
| 71210 | 162 | 166 | 161 | 597 | 430 | 431 |
| 71211 | 155 | 162 | 165 | 571 | 432 | 432 |
| 71212 | 154 | 161 | 147 | 587 | 430 | 433 |

All twelve units remain in the primary estimate and the bootstrap resampling pool. The first unit completed substantially fewer Bellman epochs at similar CPU seconds, illustrating why CPU seconds are not identical amounts of numerical work. The documented scheduling request and shared-host conditions preclude attributing that variation to a single cause. No seed was dropped, rerun, analyzed as a replacement post-intervention subset, or cost-normalized.

| Guarded stage | Wall, s | Child + monitoring-subprocess CPU, s | Guardian CPU, s | Peak sampled process-tree MiB | Status |
|---|---:|---:|---:|---:|---|
| prepare | 65.275212 | 25.676736 | 0.166490 | 339.844 | passed |
| inputs | 67.168758 | 29.106532 | 0.182101 | 363.000 | passed |
| train | 4481.515233 | 3966.613860 | 6.452339 | 297.078 | passed |
| evaluate | 1749.950623 | 1755.590467 | 2.034375 | 230.141 | passed |
| results | 1335.245511 | 1327.172646 | 1.489051 | 277.625 | passed |
| analyze | 20.353866 | 12.028381 | 0.025750 | 74.000 | passed |

The six nonoverlapping guarded stages sum to 7719.509203 wall seconds (128.66 minutes). This excludes earlier pilot/development/engineering and later statistical/report/packaging review; it is not total project cost. Do not add API or cell timers to these stage totals. Shared overhead is reported here and is not used to improve the primary treatment ratio. Fixed wall/RSS guards were neither extended nor retried.

## Verification and reproducibility

The input audit verified 512 new roots and the actual 32,000-state protected union. The saved-result audit covered **36 training cells, 36 evaluation cells, 32,735 saved epochs, 391,833 updates, 18,432 searches and 5,331 replayed successful paths**, binding 66,027 evidence files. All complete and partial training records remain. The fixed analysis and independent arithmetic review agree on all counts, intervals, cost gates and classification. The review used the existing 10,000 draws; it loaded no models, trained nothing and issued no new searches or oracle queries.

Before scientific generation, 23 fabricated engineering checks and an independent 33-file source review passed. The 142-file scientific source freeze is `366b61c3a3f8cd32ddda0d8b763e0e7e3c1362749a48ad155e016b9c28a04704`; input freeze is `4e0fb247d4feff78f908fd765f9c64f3859edf172467020bd0deb3f04cc0e877`. Configuration, source snapshots, inputs, checkpoint/epoch records, rosters, guards, saved draws and full analysis are retained under hard14_v1 (historical artifact omitted from this compact release).

After timed scientific work ended, nine fabricated closure-contract tests passed in 1.92 seconds, including rejection of omitted audit lineage/partial-file bindings, invalid guard CPU and missing report/statistical evidence. These test packaging logic, not empirical efficacy. See closure engineering receipt (historical artifact omitted from this compact release) and closure verifier source review (historical artifact omitted from this compact release).

Final report reviews, the complete scoped manifest/completion metadata and a separate read-only verification receipt accompany closure. They verify retained-file integrity and evidence lineage; they do not constitute a second training run. Historical study artifacts and README remain unchanged.
