# Cube3 adjacency alignment with and without explicit SIGReg: prospective results

**Saved decision: sigreg_free_gain_not_established.** The reference adjacency benefit is established; useful adjacency without SIGReg is not established by its conjunction.

| Primary exact 8/10 comparison | Optimal delta [95% CI], pp | Verified delta [95% CI], pp |
| --- | --- | --- |
| FULL − SIG_ONLY | +7.1167 [+5.3833, +8.9725] | +6.9946 [+5.2734, +8.8382] |
| ALIGN_ONLY − BN_ONLY | +1.5747 [+0.2686, +2.8687] | +1.2573 [-0.1102, +2.6245] |
| ALIGN_ONLY − FULL | -4.5654 [-6.2015, -3.0151] | -4.5898 [-6.2744, -3.0151] |

| Fixed primary criterion | Observed | Threshold | Passed |
| --- | --- | --- | --- |
| FULL − SIG_ONLY optimal point | +7.1167 pp | ≥ +2 pp | True |
| FULL − SIG_ONLY optimal lower95 | +5.3833 pp | > 0 pp | True |
| FULL − SIG_ONLY verified lower95 | +5.2734 pp | > −2 pp | True |
| ALIGN_ONLY − BN_ONLY optimal point | +1.5747 pp | ≥ +2 pp | False |
| ALIGN_ONLY − BN_ONLY optimal lower95 | +0.2686 pp | > 0 pp | True |
| ALIGN_ONLY − BN_ONLY verified lower95 | -0.1102 pp | > −2 pp | True |
| ALIGN_ONLY − FULL optimal lower95 | -6.2015 pp | > −2 pp | False |
| ALIGN_ONLY − FULL verified lower95 | -6.2744 pp | > −2 pp | False |

All eight gates must pass for the single omission-retention claim. Six metric endpoints supply those gates; the other 18 endpoints are descriptive and cannot rescue a failure. Two-point practical thresholds apply to point estimates, whereas positive lower bounds need only exceed zero. The two-point tolerated retention loss is a prospective practical margin, not equivalence. Intervals are approximate and carry no simultaneous-coverage, finite-sample coverage or power guarantee.

A reference-gate failure means adjacency-under-SIGReg usefulness was not established on this fresh population. Failure of omission usefulness or noninferiority does not prove that SIGReg is necessary, that ALIGN_ONLY is inferior, or that an effect is zero. This fixed study ends in all five classifications; no coefficient, seed, checkpoint, duration or population rescue is permitted.

## Fixed intervention and population

| Condition | Prehistory objective |
| --- | --- |
| FULL | A + 0.1 R |
| ALIGN_ONLY | A + 0 R |
| SIG_ONLY | 0 A + 0.1 R |
| BN_ONLY | 0 A + 0 R |

For true neighbors z=E(s), n=E(s′), A=mean((z−n)²) and R=0.5(SIGReg(z)+SIGReg(n)). All four arms compute both differentiable terms, use the same two sequential training-mode E forwards and two ordered 64-direction SIGReg draws, and freeze h during prehistory. One shared E receives both endpoint gradients; there is no detached target encoder or EMA. BN_ONLY has exact-zero encoder gradients and unchanged prehistory parameters while its BN buffers update. It is an exposure/buffer control, not an untouched random encoder or a direct-supervision baseline.

Sixteen fresh paired model/data units produce 64 full models and 64 own static teacher tables. Conditions share initial E/h tensors, literal transition/supervision streams, 4,096-parent supports and 128 shallow exact anchors per pair. All 16 anchor selections precede label lookup; 2,048 anchor records are shared across conditions. No historical model supplies a control or initialization.

All 64 final checkpoints pass bindings before learned evaluation. Each searches the same 1,024 newly certified queries, 256 at each exact QTM depth 8/10/11/12: 65,536 searches and 4,096 attempts per condition/depth cell. Failures stay in both verified and exact-optimal coverage denominators. The fresh shell-conditioned panel is not uniform over Cube3. Roots and 12 successors exclude the named 18-source union of 185,466 previously observed states; new training parents and successors exclude the new panel footprint. This does not exclude every historical training state or deeper search frontier.

## All saved effects and absolute coverage

Common crossed-bootstrap draws resample 16 paired units and queries independently within each exact-depth stratum, carrying every condition and both metrics together. Each population equally weights its two depths. There are 10,000 draws with seed 26100550301 and linear 2.5th/97.5th percentiles.

| Population | Contrast | Formula | Metric | Scope | Delta [95% CI], pp |
| --- | --- | --- | --- | --- | --- |
| easy8_10 | FULL_SIG | FULL − SIG_ONLY | optimal | primary | +7.1167 [+5.3833, +8.9725] |
| easy8_10 | FULL_SIG | FULL − SIG_ONLY | verified | primary | +6.9946 [+5.2734, +8.8382] |
| easy8_10 | ALIGN_BN | ALIGN_ONLY − BN_ONLY | optimal | primary | +1.5747 [+0.2686, +2.8687] |
| easy8_10 | ALIGN_BN | ALIGN_ONLY − BN_ONLY | verified | primary | +1.2573 [-0.1102, +2.6245] |
| easy8_10 | ALIGN_FULL | ALIGN_ONLY − FULL | optimal | primary | -4.5654 [-6.2015, -3.0151] |
| easy8_10 | ALIGN_FULL | ALIGN_ONLY − FULL | verified | primary | -4.5898 [-6.2744, -3.0151] |
| easy8_10 | SIG_BN | SIG_ONLY − BN_ONLY | optimal | descriptive | -0.9766 [-2.4170, +0.4517] |
| easy8_10 | SIG_BN | SIG_ONLY − BN_ONLY | verified | descriptive | -1.1475 [-2.6611, +0.3052] |
| easy8_10 | FULL_BN | FULL − BN_ONLY | optimal | descriptive | +6.1401 [+4.5410, +7.7881] |
| easy8_10 | FULL_BN | FULL − BN_ONLY | verified | descriptive | +5.8472 [+4.2725, +7.4951] |
| easy8_10 | interaction | FULL − ALIGN_ONLY − SIG_ONLY + BN_ONLY | optimal | descriptive | +5.5420 [+3.3569, +7.8738] |
| easy8_10 | interaction | FULL − ALIGN_ONLY − SIG_ONLY + BN_ONLY | verified | descriptive | +5.7373 [+3.4668, +8.1665] |
| hard11_12 | FULL_SIG | FULL − SIG_ONLY | optimal | descriptive | +1.1597 [+0.5005, +1.9043] |
| hard11_12 | FULL_SIG | FULL − SIG_ONLY | verified | descriptive | +1.0986 [+0.4150, +1.8677] |
| hard11_12 | ALIGN_BN | ALIGN_ONLY − BN_ONLY | optimal | descriptive | +0.0000 [-0.5249, +0.5005] |
| hard11_12 | ALIGN_BN | ALIGN_ONLY − BN_ONLY | verified | descriptive | -0.1099 [-0.6836, +0.4395] |
| hard11_12 | ALIGN_FULL | ALIGN_ONLY − FULL | optimal | descriptive | -1.0254 [-1.7215, -0.3540] |
| hard11_12 | ALIGN_FULL | ALIGN_ONLY − FULL | verified | descriptive | -1.0498 [-1.7822, -0.3662] |
| hard11_12 | SIG_BN | SIG_ONLY − BN_ONLY | optimal | descriptive | -0.1343 [-0.7568, +0.5005] |
| hard11_12 | SIG_BN | SIG_ONLY − BN_ONLY | verified | descriptive | -0.1587 [-0.8179, +0.4883] |
| hard11_12 | FULL_BN | FULL − BN_ONLY | optimal | descriptive | +1.0254 [+0.3540, +1.7334] |
| hard11_12 | FULL_BN | FULL − BN_ONLY | verified | descriptive | +0.9399 [+0.2441, +1.6602] |
| hard11_12 | interaction | FULL − ALIGN_ONLY − SIG_ONLY + BN_ONLY | optimal | descriptive | +1.1597 [+0.2686, +2.0874] |
| hard11_12 | interaction | FULL − ALIGN_ONLY − SIG_ONLY + BN_ONLY | verified | descriptive | +1.2085 [+0.2808, +2.1851] |

For readability, the same removal contrast can be shown as the contribution of retaining SIGReg. The following entries negate ALIGN_FULL and reverse its interval bounds; they introduce no new endpoint or resampling.

| Population | Metric | FULL − ALIGN_ONLY [same 95% CI], pp |
| --- | --- | --- |
| easy8_10 | optimal | +4.5654 [+3.0151, +6.2015] |
| easy8_10 | verified | +4.5898 [+3.0151, +6.2744] |
| hard11_12 | optimal | +1.0254 [+0.3540, +1.7215] |
| hard11_12 | verified | +1.0498 [+0.3662, +1.7822] |

The factorial interaction is descriptive. Its sign does not identify collapse prevention or another pure representation mechanism. Every condition uses its own post-warmup scalar teacher, so effects include changes in initialization, downstream optimization and teacher targets.

| Condition | Exact depth | Verified / 4,096 | Optimal / 4,096 | Verified | Optimal |
| --- | --- | --- | --- | --- | --- |
| FULL | 8 | 3007 | 2983 | 73.4131% | 72.8271% |
| ALIGN_ONLY | 8 | 2796 | 2781 | 68.2617% | 67.8955% |
| SIG_ONLY | 8 | 2635 | 2608 | 64.3311% | 63.6719% |
| BN_ONLY | 8 | 2701 | 2670 | 65.9424% | 65.1855% |
| FULL | 10 | 755 | 750 | 18.4326% | 18.3105% |
| ALIGN_ONLY | 10 | 590 | 578 | 14.4043% | 14.1113% |
| SIG_ONLY | 10 | 554 | 542 | 13.5254% | 13.2324% |
| BN_ONLY | 10 | 582 | 560 | 14.2090% | 13.6719% |
| FULL | 11 | 189 | 175 | 4.6143% | 4.2725% |
| ALIGN_ONLY | 11 | 126 | 115 | 3.0762% | 2.8076% |
| SIG_ONLY | 11 | 126 | 107 | 3.0762% | 2.6123% |
| BN_ONLY | 11 | 131 | 111 | 3.1982% | 2.7100% |
| FULL | 12 | 68 | 63 | 1.6602% | 1.5381% |
| ALIGN_ONLY | 12 | 45 | 39 | 1.0986% | 0.9521% |
| SIG_ONLY | 12 | 41 | 36 | 1.0010% | 0.8789% |
| BN_ONLY | 12 | 49 | 43 | 1.1963% | 1.0498% |

## Architecture, optimization and search

The compact model consumes the legacy CubeModel construction RNG, then retains only E/h. The input is 54 goal-relative facelet colors flattened into 324 one-hot values. E uses 324→256→64 with BN/ReLU, two 64-wide residual blocks and a final unconstrained 64-dimensional linear projection. h maps [z,zgoal,z−zgoal,z*zgoal] through 256→128 ReLU→1. No predictor is called, optimized or stored. Internal BN does not prohibit a constant final projection.

Every model receives 1,000 E-only prehistory steps, 1,000 E/h walk-bound warmup steps and 4,000 uninterrupted E/h static Bellman steps. Adam uses learning rate .001, separate module clipping at norm 10 and resets only between stages. Warmup uses 256 parents plus identity with unit parent and identity MSEs. Continuation uses 256 parents, 256 sampled exact anchors and identity with three unit-weight MSEs. Training-mode BN remains active in both supervised stages.

Each post-warmup model supplies one frozen eval-mode hard-copy teacher table: clip(1+minimum over all 12 exact-successor scores,0,witnessed parent upper bound), forcing identity-successor score zero. Targets are detached and never refreshed; no EMA is used. There are 384,000 optimizer steps, including 16,000 exact-zero BN_ONLY prehistory steps, and 576 scheduled audit snapshots. Initial and stage-end aliases preserve full model, Adam, RNG and input/source bindings.

Search uses exact symbolic successors and frozen E/h, with a cached identity encoding and forced identity score zero. Beam width is 32, expanded-parent cap 512, depth cap 60 and score batch size 512. Online predictor and exact-distance oracle calls are zero. Every successful action word is independently replayed in both relative and original frames.

The findings concern this finite total recipe. They do not establish universal SIGReg necessity or dispensability, absence of other anti-collapse influences, pure-encoder causality, converged behavior, direct-supervision superiority, matched CPU cost, Cube4 transfer or rewriting generalization. Closed U/V/W conclusions are not changed by this study.

## Descriptive representation and scalar diagnostics

At initial, prehistory-end and final stages, each model encodes the first 256 saved training parents once in eval mode while preserving model/BN tensors, all mode flags and RNG. The 192 saved latent matrices account for 192 encoder calls and 49,152 examples, separately counted and included in full API cost. The representation_diagnostics CSV retains all 192 rows, model hashes, mean L2 norm, centered mean square, entropy effective rank, participation ratio and preservation flags.

Centering and covariance Gram/n use float64. Negative roundoff eigenvalues are clipped to zero; an exactly zero spectrum gives both ranks zero, otherwise spectral probabilities define ranks. These are train-support descriptions, with no collapse threshold, held-out geometry claim or outcome-based selection. Small scale alone is not information loss, and later supervised learning may recover from a shrunken prehistory. The final search conjunction determines the conclusion.

The following scalar errors compare h(state,identity) with exact distance; they are not raw latent-distance or successor-ranking diagnostics.

| Condition | Exact depth | Mean score | Signed bias | MAE |
| --- | --- | --- | --- | --- |
| FULL | 8 | 10.012404 | 2.012404 | 2.879854 |
| ALIGN_ONLY | 8 | 9.672407 | 1.672407 | 2.712375 |
| SIG_ONLY | 8 | 9.584815 | 1.584815 | 2.647075 |
| BN_ONLY | 8 | 9.637711 | 1.637711 | 2.689350 |
| FULL | 10 | 11.066043 | 1.066043 | 2.305488 |
| ALIGN_ONLY | 10 | 10.669544 | 0.669544 | 2.194968 |
| SIG_ONLY | 10 | 10.539154 | 0.539154 | 2.108059 |
| BN_ONLY | 10 | 10.618906 | 0.618906 | 2.144085 |
| FULL | 11 | 11.785974 | 0.785974 | 2.052527 |
| ALIGN_ONLY | 11 | 11.420363 | 0.420363 | 1.993775 |
| SIG_ONLY | 11 | 11.236812 | 0.236812 | 1.878947 |
| BN_ONLY | 11 | 11.312821 | 0.312821 | 1.913424 |
| FULL | 12 | 12.109124 | 0.109124 | 1.890226 |
| ALIGN_ONLY | 12 | 11.710495 | -0.289505 | 1.918643 |
| SIG_ONLY | 12 | 11.522047 | -0.477953 | 1.841245 |
| BN_ONLY | 12 | 11.608591 | -0.391409 | 1.856120 |

## Recorded resources

| Complete training APIs | Fits | CPU seconds | Summed API wall seconds |
| --- | --- | --- | --- |
| FULL | 16 | 306.091859 | 306.738302 |
| ALIGN_ONLY | 16 | 305.341773 | 305.892000 |
| SIG_ONLY | 16 | 304.933668 | 305.397681 |
| BN_ONLY | 16 | 312.393239 | 312.928454 |
| Total | 64 | 1228.760539 | 1230.956436 |

Each full API includes initialization, validation, source archiving, diagnostics, own teacher, updates, checkpoint serialization and integrity checks; only final timing-ledger serialization is excluded. These are 64 fits with 16 independent paired units. Stage timers overlap enclosing APIs. Both SIGReg terms are computed even at zero weight, so these costs neither estimate optimized omission savings nor establish CPU matching.

| Evaluation scope | CPU seconds | Summed API wall seconds |
| --- | --- | --- |
| All 64 evaluation APIs | 895.335824 | 897.776971 |
| Search subset | 877.956785 | See per-search records |
| Offline root-score subset | 0.087147 | 0.087468 |

| Preparation/proof scope | Cases | CPU seconds | Wall seconds |
| --- | --- | --- | --- |
| Panel B6 load subset | 1 | 0.281324 | 0.281659 |
| Panel certificate APIs subset | 1250 | 18.531264 | 18.566197 |
| Anchor preparation subset | 2048 | 37.294444 | 37.339092 |
| Preparation through accounting snapshot | 1 | 213.451807 | 213.806894 |
| Independent input proof | 1250 | 21.272293 | 21.297196 |
| Independent saved-result audit | 1 | 28.186166 | 29.639086 |

Preparation retained 1,024 roots from 1257 proposals and 1250 complete certificate queries, with zero child-label queries. Accounting records 185466 prior states, 13312 protected states, 2048 shared anchor records and 1784 unique anchor states across pools.

| Distinct wall-time scope | Seconds |
| --- | --- |
| Training serial dispatch | 1231.153903 |
| Evaluation parallel dispatch | 228.398065 |
| prepare outer guard | 214.501416 |
| audit_inputs outer guard | 22.029818 |
| train outer guard | 1232.554345 |
| evaluate outer guard | 231.626395 |
| analyze outer guard | 25.227879 |
| independent_audit outer guard | 30.435554 |

Subset timers overlap their enclosing scopes and must not be added twice. Every scientific stage passed its frozen wall/RSS guard. The independent saved-result audit passed its 1,800-second/8,000-MiB guard with zero new neural, training, search or oracle calls. Saved tensor/RNG/accounting checks complement frozen source and engineering evidence; they do not reproduce training.

## Reproducible evidence

- Fixed configuration (historical artifact omitted from this compact release); protocol (historical artifact omitted from this compact release); source freeze and archive (historical artifact omitted from this compact release).
- Trainer (historical artifact omitted from this compact release); evaluation (historical artifact omitted from this compact release); analysis (historical artifact omitted from this compact release).
- Input proof (historical artifact omitted from this compact release); independent saved-result audit (historical artifact omitted from this compact release); audit resource guard (historical artifact omitted from this compact release); engineering test record (historical artifact omitted from this compact release).
- Saved summary (historical artifact omitted from this compact release); paired outcomes and bootstrap draws (historical artifact omitted from this compact release); input accounting (historical artifact omitted from this compact release).
- contrasts: 24 rows (historical artifact omitted from this compact release); cells: 16 rows (historical artifact omitted from this compact release); per_seed: 256 rows (historical artifact omitted from this compact release); root_calibration: 16 rows (historical artifact omitted from this compact release); training_costs: 64 rows (historical artifact omitted from this compact release); evaluation_costs: 64 rows (historical artifact omitted from this compact release); representation_diagnostics: 192 rows (historical artifact omitted from this compact release).

The seven CSV tables contain 632 saved rows. Rendering introduces no resampling, neural calls, training, search or oracle queries. Audit, report review and immutable closure finish this fixed study. The broader user-authorized program may select a distinct justified next question afterward.
