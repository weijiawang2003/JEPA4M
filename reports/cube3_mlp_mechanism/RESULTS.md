# Cube3 MLP prediction mechanism: fixed prospective results

**Saved decision: useful_action_unresolved.** The fixed criteria establish a practically useful TRUE-minus-OFF gain; correct-action specificity is not established.

Primary exact 8/10 TRUE−OFF optimal coverage is **+8.3496 [+6.5671, +10.1929] percentage points** (95% crossed-bootstrap interval). TRUE−WRONG optimal coverage is **+1.0132 [-0.4883, +2.4780] pp**. The fixed run ends here in every decision category; there is no sample extension, threshold change or rescue by harder-query outcomes.

| Fixed easy-population criterion | Observed | Threshold | Passed |
| --- | --- | --- | --- |
| TRUE−OFF optimal point | +8.3496 pp | ≥ +2 pp | True |
| TRUE−OFF optimal lower95 | +6.5671 pp | > 0 pp | True |
| TRUE−OFF verified lower95 | +6.3721 pp | > −2 pp | True |
| TRUE−WRONG optimal lower95 | -0.4883 pp | > 0 pp; only additional specificity | False |

## Population and all prespecified effects

Sixteen fresh paired model/data units produce 48 models, with identical initialization and sampled inputs within each TRUE/OFF/WRONG triplet. Every final model runs the same 1,024 newly certified queries, 256 each at exact distances 8/10/11/12: 49,152 searches. Each condition/depth row has 16 × 256 = 4,096 attempts; the easy and hard aggregate populations each have 512 distinct queries. Failed searches remain in both coverage denominators.

The 10,000 common crossed-bootstrap draws resample 16 paired units and queries independently within each of four exact-depth strata. Aggregate depth strata receive equal weight. The first three usefulness criteria form one conjunction; the additional action-specific criterion is evaluated only with usefulness. The other intervals are descriptive, unadjusted, and do not have simultaneous 95% coverage. A point estimate of at least two pp is required; the protocol does not require the lower interval bound to exceed two pp. No power guarantee was made.

| Population | Contrast | Metric | Delta [95% interval], pp |
| --- | --- | --- | --- |
| easy8_10 | TRUE_OFF | optimal | +8.3496 [+6.5671, +10.1929] |
| easy8_10 | TRUE_OFF | verified | +8.1543 [+6.3721, +9.9854] |
| easy8_10 | TRUE_WRONG | optimal | +1.0132 [-0.4883, +2.4780] |
| easy8_10 | TRUE_WRONG | verified | +0.9277 [-0.5127, +2.3560] |
| easy8_10 | WRONG_OFF | optimal | +7.3364 [+5.7983, +8.9478] |
| easy8_10 | WRONG_OFF | verified | +7.2266 [+5.6641, +8.8745] |
| hard11_12 | TRUE_OFF | optimal | +1.5259 [+0.7199, +2.3560] |
| hard11_12 | TRUE_OFF | verified | +1.4038 [+0.5493, +2.2461] |
| hard11_12 | TRUE_WRONG | optimal | +0.2930 [-0.4153, +1.0254] |
| hard11_12 | TRUE_WRONG | verified | +0.2197 [-0.5249, +0.9644] |
| hard11_12 | WRONG_OFF | optimal | +1.2329 [+0.4761, +2.0020] |
| hard11_12 | WRONG_OFF | verified | +1.1841 [+0.3662, +2.0020] |

WRONG−OFF and absolute rates are retained below: deterioration in WRONG cannot substitute for a useful TRUE−OFF gain. A failed practical threshold, interval crossing zero, or failed quality guard is reported as such, without asserting equivalence or universal failure.

## Absolute coverage

| Condition | Exact depth | Verified / 4,096 | Optimal / 4,096 | Verified | Optimal |
| --- | --- | --- | --- | --- | --- |
| TRUE | 8 | 2914 | 2894 | 71.1426% | 70.6543% |
| OFF | 8 | 2502 | 2467 | 61.0840% | 60.2295% |
| WRONG | 8 | 2850 | 2822 | 69.5801% | 68.8965% |
| TRUE | 10 | 756 | 743 | 18.4570% | 18.1396% |
| OFF | 10 | 500 | 486 | 12.2070% | 11.8652% |
| WRONG | 10 | 744 | 732 | 18.1641% | 17.8711% |
| TRUE | 11 | 308 | 301 | 7.5195% | 7.3486% |
| OFF | 11 | 230 | 219 | 5.6152% | 5.3467% |
| WRONG | 11 | 302 | 290 | 7.3730% | 7.0801% |
| TRUE | 12 | 83 | 79 | 2.0264% | 1.9287% |
| OFF | 12 | 46 | 36 | 1.1230% | 0.8789% |
| WRONG | 12 | 71 | 66 | 1.7334% | 1.6113% |

## Intervention and limits

All 48 models use the established Cube3 MLP: 324 one-hot inputs, width 256, latent dimension 64, two residual blocks, the action-conditioned residual predictor and the original scalar goal head. Each stage has 1,000 updates at Adam learning rate 0.001: prehistory uses 256 transitions, warmup 256 parents plus identity, and continuation 256 parents plus 256 anchors plus identity. Prehistory uses prediction MSE plus 0.1 endpoint SIGReg and two sequential shared-encoder BatchNorm forwards. TRUE differentiates through both real encoder endpoints with the true action. OFF detaches both endpoints for prediction only: the predictor still trains, and encoder SIGReg gradients and BN updates remain live. WRONG differentiates both endpoints but receives a freshly sampled other action; the real successor and common input/SIGReg randomness stay fixed. WRONG conveys action-exclusion information and is not independent noise.

Warmup trains E/h on walk bounds; continuation trains E/h on a once-computed own-model static teacher, exact shallow anchors and zero identity. The predictor and unused heads are frozen downstream. The exact-anchor BN batch contains 256 parents, 256 anchors and one identity. Each arm uses its own post-warmup teacher, so teacher differences mediate the total pretraining effect. This experiment does not isolate only the final encoder geometry or a common-teacher direct effect.

Every pair has a fresh 4,096-parent pool and 128 shallow exact labels selected before distance lookup. The new test root-plus-successor footprint is excluded from every new parent and successor input; roots/footprints also exclude the named prior query union. This does not exclude all historical training states or all search frontiers. Queries follow the fixed shell-conditioned walk generator, not a uniform Cube3 distribution. Recipe selection used earlier research, while models, supports and evaluation roots here are fresh. Results do not establish Cube4 transfer, a general group representation or a label-efficiency law.

Search uses exact symbolic successors, fixed beam 32, expanded-parent cap 512, depth limit 60 and score batches 512. It scores frozen E/h with one cached identity encoding per model and forces the identity score to zero. No predictor or oracle is called online. Every successful solution was independently replayed in relative and original frames.

## Scalar heuristic diagnostics

These compare h(state, identity) with exact distance. They are not raw latent-distance geometry and include no successor-ranking diagnostic.

| Condition | Depth | Mean score | Signed bias | MAE |
| --- | --- | --- | --- | --- |
| TRUE | 8 | 9.369555 | 1.369555 | 2.518222 |
| OFF | 8 | 9.095860 | 1.095860 | 2.387828 |
| WRONG | 8 | 9.479379 | 1.479379 | 2.591890 |
| TRUE | 10 | 11.130323 | 1.130323 | 2.338863 |
| OFF | 10 | 10.669502 | 0.669502 | 2.135997 |
| WRONG | 10 | 11.232697 | 1.232697 | 2.406914 |
| TRUE | 11 | 11.744706 | 0.744706 | 2.262033 |
| OFF | 11 | 11.243004 | 0.243004 | 2.143817 |
| WRONG | 11 | 11.849803 | 0.849803 | 2.335113 |
| TRUE | 12 | 12.311130 | 0.311130 | 1.852432 |
| OFF | 12 | 11.697146 | -0.302854 | 1.843925 |
| WRONG | 12 | 12.439356 | 0.439356 | 1.926891 |

## Recorded resources

| Fresh full-recipe training APIs | Instances | CPU seconds | Summed API wall seconds |
| --- | --- | --- | --- |
| TRUE | 16 | 241.712061 | 242.400222 |
| OFF | 16 | 242.086997 | 242.997949 |
| WRONG | 16 | 241.894091 | 242.713524 |

All 48 scientific pipelines account for 725.693149 CPU seconds; inherited training cost is zero. TRUE/OFF realized API CPU ratio is 0.998451, descriptive only. Matched update/exposure counts do not imply equal gradient FLOPs or CPU cost, and no cost-matching efficacy claim is made.

| Evaluation scope | CPU seconds | Wall seconds |
| --- | --- | --- |
| All 48 evaluation cell APIs | 674.679817 | 675.826691 |
| Search subset | 661.137940 | See raw per-search records |
| Offline root-score subset | 0.065600 | 0.065764 |

| Offline input/proof scope | Cases | CPU seconds | Wall seconds |
| --- | --- | --- | --- |
| Panel B6 index load (preparation subset) | 1 | 0.264464 | 0.264890 |
| All panel certificate APIs (preparation subset) | 1262 | 18.736835 | 18.758304 |
| Shallow-anchor preparation (preparation subset) | 2048 | 37.579969 | 37.623113 |
| Preparation through accounting snapshot | 1 | 212.938294 | 213.200106 |
| Independent input proof; zero production oracle calls | 1262 | 21.811427 | 22.133977 |
| Independent saved-result audit; zero neural/search calls | 1 | 21.992768 | 22.412188 |

Preparation retained 1,024 roots from 1275 proposals and 1262 complete certificate queries, with zero child-label queries. There are 88954 named prior states, 13312 unique protected states, 2048 shared anchor records and 1797 unique anchor states across pools. The three method copies do not create additional label records.

| Counted work | Count |
| --- | --- |
| Scientific optimizer updates | 144000 |
| Searches | 49152 |
| Expanded parents | 21599126 |
| Search scored states | 217427638 |
| Search E/h forward calls | 1460872 |
| Hooked score calls including root diagnostics | 730532 |
| Hooked scored states including diagnostics | 217476790 |
| Hooked E/h forwards | 1461064 |
| Additional cached-identity encoder forwards | 48 |
| Online P/oracle calls | 0 / 0 |

| Distinct wall-time scope | Seconds |
| --- | --- |
| Training parallel dispatch | 185.700761 |
| Evaluation parallel dispatch | 171.262380 |
| Outer prepare guard | 214.056624 |
| Outer audit_inputs guard | 22.987823 |
| Outer train guard | 186.608453 |
| Outer evaluate guard | 173.474319 |
| Outer analyze guard | 18.599323 |

API CPU sums, summed API wall times, parallel dispatch and outer guards measure different scopes. Subset timers overlap their enclosing stages and must not be added twice. Engineering tests, including tiny TRUE/OFF numerical parity and actual spawn serialization checks, precede science and are separate from the 48 scientific pipelines. Their bound test record and log are linked below; no engineering timing is mislabeled as a scientific fit.

The combined preflight test command passed with exit code 0 and recorded 3.499644 wall seconds, including engineering fixtures; this duration is not added to any scientific training API total.

## Evidence and complete tables

- Fixed configuration (historical artifact omitted from this compact release); protocol (historical artifact omitted from this compact release); frozen source/archive (historical artifact omitted from this compact release).
- Independent input proof (historical artifact omitted from this compact release); Independent saved-result audit (historical artifact omitted from this compact release); Engineering test record (historical artifact omitted from this compact release); Engineering test log (historical artifact omitted from this compact release).
- Saved analysis (historical artifact omitted from this compact release); Paired outcomes and draws (historical artifact omitted from this compact release); Input accounting (historical artifact omitted from this compact release).
- contrasts: 12 rows (historical artifact omitted from this compact release); cells: 12 rows (historical artifact omitted from this compact release); per_seed: 192 rows (historical artifact omitted from this compact release); root_calibration: 12 rows (historical artifact omitted from this compact release); training_costs: 48 rows (historical artifact omitted from this compact release); evaluation_costs: 48 rows (historical artifact omitted from this compact release).

This report adds no resampling, neural calls, training, search or oracle queries. The bounded mechanism study stops after this result.
