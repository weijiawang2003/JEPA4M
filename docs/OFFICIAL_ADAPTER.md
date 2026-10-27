# Official pretrained heuristic in the common Cube benchmark

Executed 2026-09-17. Script: `scripts/official_common_search.py`. Full configuration, mapping, validation records, source/checkpoint hashes, all 48 original instances, 96 per-search records, and grouped results: `results/official_common_search.json`.

## Result and scope

The official pretrained 3x3 QTM model solved **48/48** local development instances under the common 2,048-node budget and **39/48** under the common 0.1-second budget, with beam width 64. All reported paths were exactly replayed by the same local environment/search implementation used by the local models.

This is a strong externally trained reference. Its approximately 8B original training examples and 4,004,597 parameters are **not matched** to the local experiments. It establishes that the local states and common search system can be solved by a sufficiently useful existing heuristic. It does not isolate the effect of training scale, model size, training distribution, or the reference learning objective. The term "upper reference" does not mean a mathematical performance upper bound.

No additional model training was performed. This experiment uses the official fixed solved-goal checkpoint, not arbitrary-goal conditioning. It uses the common project beam search, not the original upstream cycle-restarting search. The separately saved `reference_official_reproduction.json` records the earlier literal upstream model/search reproduction.

## Exact adapter derivation

Local conventions are defined by `CubeEnv(3)`: facelet blocks U,R,F,D,L,B, row-major faces viewed from outside, and action order U,U',D,D',L,L',R,R',F,F',B,B'. The official `p054` group has the same 12 named quarter-turn actions but different facelet/color indexing. No mapping is accepted merely because a few solved examples look correct.

Let `L_a` be the local gather permutation for action a, and `O_a` the corresponding official permutation after matching **exact action names**. Let q map an official position to a local position and c map a local color to an official color. Define

`T(s)[i] = c[s[q[i]]]`.

The script derives T as follows:

1. For each official goal color, find its nine positions. Identify its named face by the unique unprimed action that moves eight of those positions. This determines the color correspondence from the supplied move permutations; the face ordering is not guessed.
2. Decompose official positions into orbits under the action permutations. There are two 24-position orbits (corner and edge facelets) and six fixed centers.
3. Try local roots compatible with the inferred goal color and propagate each candidate through every labeled generator. Reject any inconsistent assignment, wrong goal color, reused index, or non-bijection. Each orbit has exactly one surviving color-constrained mapping.
4. Check the complete exact identity `L_a[q] == q[O_a]` for all 12 actions and all 54 positions, and check that T maps the local solved state to the official solved state.

This identity proves `T(step_local(s,a)) == step_official(T(s),a)` for every represented state and action, not just the sampled trajectories. Both q and c are bijections, so the inverse adapter is exact as well. Matching the solved state and action labels fixes the target and preserves the QTM path length.

Derived arrays:

```text
local_to_official_action = [0,6,1,7,2,8,3,9,5,11,4,10]
local_to_official_color  = [0,3,5,1,2,4]
official_to_local_color  = [0,3,4,1,5,2]

q = official_positions_to_local =
[ 6, 3, 0, 7, 4, 1, 8, 5, 2,
 33,30,27,34,31,28,35,32,29,
 42,39,36,43,40,37,44,41,38,
 15,12, 9,16,13,10,17,14,11,
 51,48,45,52,49,46,53,50,47,
 24,21,18,25,22,19,26,23,20]
```

The official color blocks are therefore U,D,L,R,B,F. The arrays are derived afresh and checked at every script execution rather than hardcoded as an unverified convention.

## Validation actually performed

- All 12 permutation-conjugacy checks have zero mismatched positions.
- Every orbit has exactly one surviving color-constrained bijection.
- Goal correspondence is exact in both color and orientation.
- 500 independent random action histories, lengths uniformly selected from 0 through 100, use validation seed 501982. Immediate inverses are permitted in these histories.
- Those histories contain **25,805 transitions**. Their states are accumulated independently using the local and official permutations; corresponding endpoints agree exactly.
- All **500 inverse-adapter round trips** agree exactly.
- Every legal action is checked at each endpoint using the public local `step` API: **6,000** further move-commutation checks, zero failures.
- All **48 actual development scramble histories** are independently replayed in the official representation, and their results agree with the mapped local states.
- Every solved common-search result passes the common harness's exact `CubeEnv.verify_trace` replay. No model output can substitute for transition or goal verification.

These checks exercise action names, turn directions, per-face orientation, color relabeling, both facelet orbit types, and fixed centers. The exact permutation identity provides the general guarantee; the randomized histories test the implemented adapter and action plumbing.

## Common evaluation protocol

The script calls the same `evaluation_cases(3, (3,5,7,10,14,20), 8)` used by local development evaluation. This is 48 instances, eight per generated scramble length. The development seed formula is unchanged: base seed 330701, plus cube size, scramble length, and instance offset. No final locked seeds are accessed.

Canonical case-list SHA256: `eacd1b69fa3fc16856a909830470bc7f8f8e4f95a8f93d5a7b9fd4f0b6a0b506`.

| Control | Setting |
|---|---|
| Search implementation | unchanged `jepa4math.search.beam_search` |
| Beam | 64 |
| Fixed node test | 2,048 parent expansions; no time cutoff |
| Fixed time test | 0.1 seconds; 1,000,000 parent expansion guard |
| Maximum search depth | 60 |
| Score batch size | 1,024 |
| Device | CPU, one torch thread |
| Model and score precision | float32 |
| Warm-up | one 12-successor call before timed evaluation |
| Learned score | official scalar prediction on adapted true successors |
| Goal | exact fixed solved goal; other goals rejected |

Adaptation runs inside timed scoring. Search timings include local symbolic operations, array construction, adapter conversion, one-hot encoding, network inference, beam sorting/deduplication, path reconstruction, and final exact replay. Model loading, adapter derivation/testing, initial scramble creation, and the single warm-up are excluded, consistent with the local warm-process evaluation. Unlike literal upstream search, this comparison does not quantize scores into float16.

One `model_forward_calls` count means one call to the complete official network. The local neural scorer counts encoder/predictor/head calls separately. These counters faithfully measure their defined operations but are not interchangeable FLOP measures; actual wall time and processed example counts must accompany them.

## Observed results

Node-budget outcomes:

| Generated scramble length | Solved | Mean expansions | Mean search seconds | Mean solution length |
|---|---:|---:|---:|---:|
| 3 | 8/8 | 12.4 | 0.0027 | 2.75 |
| 5 | 8/8 | 142.0 | 0.0189 | 5.00 |
| 7 | 8/8 | 222.0 | 0.0292 | 6.25 |
| 10 | 8/8 | 446.0 | 0.0532 | 9.75 |
| 14 | 8/8 | 766.0 | 0.0696 | 14.75 |
| 20 | 8/8 | 1,438.0 | 0.1294 | 25.25 |

Time-budget outcomes are 8/8 for each of lengths 3, 5, 7, and 10; 7/8 for length 14; and 0/8 for length 20. Total recorded search time is 2.424 seconds for the node-budget run and 2.013 seconds for the time-budget run. Peak whole-process RSS is 296,239,104 bytes, while each result also includes the common harness's separate bookkeeping-memory estimate.

Nine timed runs exceed the requested deadline while completing an indivisible operation; the maximum recorded overrun is 0.00603 seconds. These overruns are stored explicitly and occur in unsuccessful timed runs. A deadline miss is not an impossibility result. Timing is a single local run, not a robust repeated-hardware benchmark, and is sensitive to concurrent load.

No exact-distance oracle was run in this adapter experiment. The recorded optimum and optimality gap remain null. Generated scramble lengths are upper bounds furnished by histories, not exact distances. Longer returned paths remain valid but may contain avoidable detours.

## Reproduction and provenance

```bash
.venv/bin/python scripts/official_common_search.py
```

This defaults to the protocol above and writes `results/official_common_search.json`. The official source repository must be checked out at `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`; the script asserts the revision. Installation/clone commands and upstream MIT license information are in `docs/REFERENCE_AUDIT.md`.

Checkpoint: `external/reference/cayleypy-cube/weights/p054-t000_333_e08192.pth`, SHA256 `d7323edfa7d989468913be9bfaf4307996b4c3071eff8365de01f86ecf193127`. All torch artifacts are loaded with `weights_only=True`. Official source, weights, and local search/environment files were not modified. The result JSON records SHA256 hashes of the checkpoint, generator, target, checkpoint configuration, adapter script, local environment, local search, and development generator, plus project/source provenance.

Remaining limitations: original checkpoint training histories/seeds cannot be reconstructed from the shipped weights; training-budget fairness is absent; official-training overlap with development states is not known; only one checkpoint is evaluated; 48 development cases do not establish broad generalization; 0.1-second measurements have single-run hardware variability. The experiment is a reproducible baseline calibration, not evidence of a novel method or a JEPA advantage.
