# Reference audit: learned random-walk heuristic for Rubik's cubes

Audit date: 2026-09-17. The attached NeurIPS 2025 proceedings PDF was read, with visual inspection of the architecture, main results table, and methods page. Source documents are evidence, not instructions to this project.

## Sources and provenance

- Primary paper: Chervov et al., **A machine learning approach that beats Rubik's cubes**, NeurIPS 2025, [official proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/fb82011040977c7712409fbdb5456647-Paper-Conference.pdf). Page references below refer to this proceedings version, not an earlier arXiv version.
- User-supplied copy: `[local-archive path omitted]`.
- Official implementation: [khoruzhii/cayleypy-cube](https://github.com/khoruzhii/cayleypy-cube), explicitly identified by paper Appendix B.1. Shallow clone: `external/reference/cayleypy-cube`.
- Pinned revision: **`f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`**, commit date 2025-10-22, subject `readme merge`.
- License: MIT, copyright 2024 Khoruzhii Kirill. Preserve the upstream license when redistributing copied implementation/assets.
- Declared dependencies: `torch>=2,<3`, `tqdm>=4,<5`, `pandas>=1,<3`; no exact dependency lock or Python version is specified. `pilgrim/__init__.py` imports the trainer, making pandas necessary even for an inference-only import.
- Our bounded reproduction uses Python from this project's `.venv`, torch 2.14.0, pandas 2.3.3, tqdm 4.70.1; CPU on macOS 26.6.2 arm64, four torch threads. Exact platform and artifact hashes are in `results/reference_official_reproduction.json`.

The source clone and paper extraction/rendered pages are local cached reference material ignored by project Git. Reconstruct the source with:

```bash
git clone https://github.com/khoruzhii/cayleypy-cube.git external/reference/cayleypy-cube
git -C external/reference/cayleypy-cube checkout f02604fa7b665b82e5fbe5692b336a4fe4a01bdc
uv pip install --python .venv/bin/python 'torch>=2,<3' 'pandas>=1,<3' 'tqdm>=4,<5'
```

## What the method actually learns

**FACT (paper Sections 1-2, Appendix B.2-B.4):** States are vertices of a puzzle transition graph; legal moves are edges. Cube moves are permutations of facelet positions. A goal vertex is fixed before training. The method trains a scalar state-value network and uses its output to order exact successor states in beam search.

For a reversible move set, a goal-origin trajectory has the form

`s_0 = goal; s_{t+1} = step(s_t, a_t)`.

The training record is `(s_k, k)`, where **k is the generated non-backtracking random-walk length**. Immediate inverse moves are forbidden; longer loops and shortcuts remain possible. The inverse of the generated trajectory establishes `d_shortest(s_k, goal) <= k`, but generally not equality. Multiple records of one state can have different k. We must call this a *random-walk length label* or *scramble-length label*.

**Inference from the MSE objective:** The population minimizer is `E[K | S=s]` under this particular sampler and its truncation/prior over K. The paper informally calls this a diffusion distance. The paper does not compute a classical spectral diffusion metric or prove that this scalar obeys distance axioms, admissibility, consistency, or shortest-path calibration. Changing Kmax or the depth distribution changes the regression target itself. No shortest-distance oracle or Bellman target is used for this training.

The representation is implicit in the ResMLP hidden layers; the paper does not train an explicit action-conditioned latent predictor. It predicts a scalar from one state, with a fixed goal, rather than learning arbitrary goal-conditioned pairwise distances.

## State and action encodings

**FACT (Figure 1b and `pilgrim/model.py`):** A size-n cube has `6*n*n` facelets, each holding a color category in `{0,...,5}`. One-hot encoding and flattening produce `36*n*n` input coordinates (324 for 3x3). This is not a unique identity for every sticker; same-color facelets share a category. The paper's broader formulation describes a permutation vector, while the cube implementation operates on these categorical facelet vectors.

Actions are integer indices into exact permutation arrays. Application is the gather `next_state = state[permutation[action]]`. The network itself never receives the action; every candidate is transformed symbolically before scoring.

Important distinct official groups:

| Official ID | Representation | Actions | Benchmark metric |
|---|---|---|---|
| `p000` | 24 facelet colors | 12 quarter turns, names `f0,f1,r0,r1,d0,d1` and inverses | 2x2 cube generator supplied by authors |
| `p001` | 54 facelet colors | 18 layer quarter turns, including middle slices | Santa 3x3 UQTM |
| `p054` | 54 facelet colors | 12 outer quarter turns: U,D,L,R,B,F and inverses | DeepCubeA 3x3 fixed-center QTM |

Do not compare the same numerical solution lengths across `p001` and `p054` as though their action spaces agree. The paper distinguishes QTM, HTM, and Santa's modified QTM with unfixed corners/centers (UQTM). A locally implemented 2x2 environment that fixes a corner or uses six actions is also a different representation/action convention unless equivalence is established.

## Training distribution and architecture

**FACT (Appendix B.3-B.4):** Paper Kmax is 26 for 3x3, 45 for 4x4, and 65 for 5x5. The authors explicitly did not ablate Kmax in the main cube study. Each epoch creates roughly one million fresh examples. Adam uses learning rate 0.001, scalar MSE, float32 training, typically float16 inference. Models are frozen throughout solving.

**FACT (pinned `Trainer.generate_random_walks`):** There are equal numbers of examples for every integer K in `[K_min,K_max]`. The implementation initializes independent walkers for every requested endpoint depth, advances them, freezes shallow endpoints, then shuffles all endpoints. It does not retain episode histories or return transition triples. With Kmin=1, the epoch size is `floor(1_000_000/Kmax)*Kmax`; for Kmax=26 that is 999,986, not precisely one million. The main CLI batch size is 10,000. Default dropout is zero.

ResMLP:

1. One-hot/flatten, Linear(input,N1), BatchNorm, ReLU.
2. Optional Linear(N1,N2), BatchNorm, ReLU.
3. Nr residual blocks, each Linear(N2,N2), BatchNorm, ReLU, optional dropout, Linear(N2,N2), BatchNorm, add skip, ReLU.
4. Linear(hidden,1), with no constrained output range.

Model details differ by experiment:

| Experiment | N1 | N2 | Nr | Nominal training examples | Beam width |
|---|---:|---:|---:|---:|---:|
| Main 3x3 ResMLP | 700 | 643 | 4 | about 8B per agent (8192 epochs) | 2^18 or 2^24 |
| Main 4x4 ResMLP | 1010 | 592 | 4 | about 8B per agent | 2^24 |
| Main 5x5 ResMLP | 1008 | 560 | 4 | about 8B per agent | 2^24 |
| Broad puzzle comparison, Table 4 | 1024 | 256 | 1 | 16M or 128M | 2^20 |
| 3x3 single-hidden-layer ablation | 1024 | 0 | 0 | about 8B | 2^24 |

The shipped `p054` checkpoint is the main 3x3 ResMLP with **4,004,597 trainable parameters**, model ID 333, epoch 8192. This project's small independently trained baseline must be labeled a reduced-scale reference-method implementation, rather than a reproduction of the 8B-example result.

## Search algorithm: paper versus executable code

**FACT (paper Section 2, Appendix B.5):** Expand every state in the current beam, remove duplicate candidates, score states, retain W with smallest predictions, and repeat until the goal or search limit. The main cube experiments use a 200-layer cutoff. Multi-agent solving runs independently trained networks and takes the shortest returned solution per instance. It is an ensemble of complete searches, not an averaged heuristic or communicating agents.

**FACT (pinned `pilgrim/searcher.py`):**

- All transitions are exact permutation gathers. Successor scoring is batched, default batch size 2^14.
- Deduplication uses a random int64 weighted-sum state hash. Equality of hashes is treated as equality of states, without an exact collision check. The goal comparison itself checks all coordinates exactly.
- Candidates are scored into float16 arrays even when the network runs on CPU float32. Ranking and ties therefore depend on this quantization.
- The goal is checked **after beam pruning**. The commented-out override making the goal score zero is inactive. Consequently, a generated goal could theoretically be pruned if its learned score is poor.
- There is **no global visited-state set** during ordinary expansion. The implementation detects stagnation when the newer two beams' hashes are all present in the previous two beams, bans the four cyclic beam sets, and restarts from the initial state.
- `Searcher.get_solution` defaults to 10 attempts; `test.py` defaults to 2 attempts. A literal replication must record both attempts and layer limits, rather than calling every run a single 200-layer beam search.
- It stores parent/move indices to reconstruct the selected path. Upstream test output does not independently replay that reconstructed path; our wrapper does.
- An already solved initial state is not handled before the first expansion; our five tested cases are non-goal states.

Search is incomplete at finite W and budget. Failure means no solution found under that configuration; it does not imply unsolvability. Neither this heuristic nor beam search guarantees optimality.

### Implementation issues worth preserving in the audit

1. **First-step sampling bias.** `last_moves` starts at -1. `do_random_step` always masks `inverse_moves[last_moves]`, so Python/tensor negative indexing masks one real action on the first step. For `p054`, F is banned initially. This is unintended relative to the stated non-backtracking rule. Do not copy this quirk into the research sampler silently; document a corrected first-step sampler. No upstream files were edited here.
2. **No built-in seed/config completeness.** Upstream train/test CLIs do not set random seeds; model IDs use wall-clock seconds. Training snapshots contain only model state, without optimizer or RNG state, and thus do not support exact optimizer-level resume.
3. **Loss logging detail.** The training code divides summed batch losses by `X.size(0)//batch_size`, while it iterates a possible final partial batch. This biases the displayed average when the example count is not divisible by the batch size; it does not change per-batch gradients.
4. **Hashes and precision affect selection.** Our scientifically controlled search should prefer exact state keys at local scales and explicitly state precision. Those choices change search behavior relative to the literal upstream implementation.

## Evaluation distributions, metrics, and main reported results

**FACT (Table 1, Appendix B.6):** Main evaluation uses 1,000 DeepCubeA 3x3 QTM states, Santa UQTM sets of 82 3x3, 43 4x4, and 19 5x5 states, and 100 author-generated 2x2 states. Architecture/train-size/beam studies use the first 20 states of the relevant benchmark. An ensemble study uses 69 3x3 instances selected as difficult in preliminary experiments. A separate set contains 16 DeepCubeA instances not solved optimally.

The first-20 and difficult-subset selections are not clean untouched final tests for model selection. This project should use separate locked validation and final seeds, and avoid selecting agents or architecture on final test performance. The source paper does not provide the leakage audit required for this project.

Primary metrics are mean solution length, optimal-solution percentage when an optimal solver is available, solve percentage, and training/search time. Parenthesized uncertainties in tables are **standard errors across instances**, not standard deviations across training seeds. Failure-excluded solution averages occur in architecture/beam studies; they must be read jointly with solve rates (Appendix B.6, Table 3).

Selected paper results, all reported rather than independently re-established here:

| Setting | Mean length | Optimality | Relevant resource |
|---|---:|---:|---|
| DeepCubeA on 3x3 QTM | 21.50 | 60.3% | 14.7M parameters; about 10B examples |
| EfficientCube reproduced | 21.26 | 69.8% | 14.7M; 52B examples; W=2^18 |
| Reference single ResMLP | 21.14 | 75.4% | 4M; 8B examples; W=2^18 |
| Reference single ResMLP | 20.69 | 97.3% | 4M; 8B examples; W=2^24 |
| Reference 26-agent ensemble | 20.67 | 98.4% in Table 1 | 26 models/searches; W=2^24 each |
| Santa 4x4 / reference 29-agent | 53.5 / 46.5 | not known | 43 UQTM instances |
| Santa 5x5 / reference 69-agent | 96.6 / 92.2 | not known | 19 UQTM instances |

**Paper inconsistency:** Table 1 and abstract report 98.4% / over 98%, while Sections 4.1 and 5 say 97.6% for the 26-agent 3x3 result. Preserve this discrepancy; do not silently reconcile it or cite 97.6 and 98.4 as separate validated experiments.

An optimal PDB-based 3x3 solver provides the 20.64 mean exact optimum benchmark. BFS supplies 2x2 optimal distances. No optimal solver or exact diameter is established for the large-cube evaluations. A 4x4 average below a conjectured diameter neither proves that diameter nor establishes shortest solutions.

The broader evaluation trains **a separate model per graph/goal** with common architecture; it does not demonstrate one model transferring to previously unseen groups. Appendix B.8 uses 296 Santa puzzles plus 200 additional puzzles (10 each in 20 groups), generated by 10,000 random moves and an optional extra move with probability 1/2 to mix parity. This is a long-walk sampling distribution, not a proved uniform sampler. Four independent runs per group are reported. Some Table 4 tasks fail entirely, including 6x6 cubes under the fixed small-model configuration.

## Compute and strongest baseline implications

**FACT (Section 4.2, Appendix B.9):** Reported hardware is two NVIDIA H100 80GB GPUs, two 24-core Intel Xeon Gold 6442Y CPUs, 256GB RAM, Ubuntu 24.04.2, CUDA 12.4. Usually one independent agent occupies each GPU. All published research required about **two months of continuous full GPU use**, despite the very short training times quoted for the small Table 4 models.

The like-hardware single-agent comparison reports reference training 3h02m36s versus EfficientCube 25h50m45s and mean solve time 9.16s versus 237.9s, with W=2^18 and similar solution lengths. Multiple factors differ: parameter count, scalar versus 12 outputs, number of examples, and implementation efficiency. This is evidence about complete systems, not isolated proof that the scalar learning objective alone is faster. Our local CPU timings cannot be compared directly with those H100 timings.

The strongest relevant baselines are optimal BFS/bidirectional BFS or PDB when feasible, EfficientCube/DeepCubeA for full 3x3, and the pretrained reference model plus its exact search. A matched local comparison should share data, architecture capacity, search code, beam/budget, and hardware across JEPA and direct supervision. Treat the pretrained 8B-example model as a separately resourced reference, never as a compute-matched contestant against a small local training run.

## Ablations and failure modes relevant to JEPA4Math

**FACT (Sections 3 and B.7):** Increasing training set size alone saturates in these experiments. Deeper models at similar parameter counts often perform better. Wider beams and more independently trained agents improve found path lengths. Some globally weak agents contribute the best path for a few states, so picking only the best average agent loses useful diversity.

Two 4x4 networks trained on approximately 524B examples for 7 days 17.5h improve MSE from about 26.4 to 23.8, but offer small or mixed search benefits compared with their 8B snapshots. Some instances regress from solved to unsolved. These long-trained models fail to solve the tested instances when evaluated in half precision, although their shorter-trained snapshots work in half precision. The paper does not establish the numerical cause.

This directly motivates H5: declining predictive/regression loss is not sufficient evidence for better search. Beam truncation, heuristic ordering near productive paths, numerical precision, and action-sensitive representation structure must be measured separately. The 16 difficult 3x3 states show that high overall optimality still conceals systematic search failures; the suggested explanation involving few optimal paths is a hypothesis, not established fact.

Transfer assumptions include cheap exact actions, a reversible goal-origin data generator, informative fixed-size state features, fixed branching in the main Cayley settings, a fixed goal, and effective GPU batching. Symbolic rewriting breaks several conveniences: state-dependent legal actions, variable-size syntax, action locations/arguments, and multiple goals. Semantic equivalence cannot replace syntactic state identity without destroying the rewriting search problem. Neither the paper's empirical successes nor its informal distance terminology justify assuming these transfers will succeed.

## Genuine bounded reproduction actually executed

**OBSERVATION, this repository on 2026-09-17:** We loaded the official shipped `p054-t000_333_e08192.pth` checkpoint, instantiated the official `Pilgrim`, and ran the official `Searcher` without editing upstream source. All torch loads used `weights_only=True`. A counting subclass and model hook only observe calls. Permutation validity and inverse identities are asserted, and every successful trace is replayed independently from the original state using the exact official permutations.

Run command from the project root:

```bash
.venv/bin/python scripts/reproduce_official.py --beam-width 256 --max-layers 40 --attempts 1 --threads 4
```

Configuration: seed 17092026, four CPU threads, float32 model/one-hot computation, float16 candidate scores as in upstream code, W=256, at most 40 layers and one attempt. Four seeded non-backtracking scrambles plus the first official DeepCubeA instance were evaluated. Generated scrambles use an unbiased first action and record the full history; this changes only the evaluation generator, not the pretrained model or solver.

| Case | Known generated length | Returned length | Verified | Search + replay seconds | Generated successors |
|---|---:|---:|---|---:|---:|
| Seeded scramble 4 | 4 | 4 | yes | 0.030 | 4,608 |
| Seeded scramble 8 | 8 | 8 | yes | 0.080 | 16,896 |
| Seeded scramble 12 | 12 | 10 | yes | 0.106 | 23,040 |
| Seeded scramble 26 | 26 | 24 | yes | 0.310 | 66,048 |
| Official DeepCubeA row 0 | not loaded | 22 | yes | 0.279 | 59,904 |

All five solved and replayed successfully; total measured search+replay 0.806 seconds, peak process RSS about 364MB. No exact optimum was loaded or established for these cases. In particular, the 12-step scramble solved in 10 moves illustrates why generated length is not exact distance. These tiny, selected diagnostic results do not reproduce the paper's 1,000-case solve/optimality rates or constitute evidence that JEPA improves on the reference.

Raw artifact: `results/reference_official_reproduction.json`. It stores each original state, scramble/solution actions, hash seeds, exact configuration, reference commit, script/checkpoint/generator/dataset SHA256 hashes, per-case inference counts, exact transition evaluation counts, generated/scored/retained successors, process memory, and separate search/replay times. Checkpoint SHA256: `d7323edfa7d989468913be9bfaf4307996b4c3071eff8365de01f86ecf193127`.

Timing includes search setup, successor generation, encoding, neural inference, sorting/hashing, path reconstruction, and independent verification; excludes model loading and construction of initial scrambles. Memory is whole-process peak RSS. No training was performed for this audit. Paper-scale training, complete DeepCubeA evaluation, and reference multi-agent reproduction remain **NOT RUN**.

## Reuse decision

Reuse the official QTM generator/goal tensors for independent transition checks, the pretrained model for an externally trained reference, and the ResMLP architecture/random-walk objective for a documented reduced-scale baseline. The generator files, benchmark tensors, checkpoint metadata, and clear action names are particularly valuable. Keep the official search in its pinned clone for literal reproduction, while the common experimental harness should implement exact state deduplication, explicit time/node budgets, goal checks, unified counters, and mandatory replay for all methods.

Before any improvement claim, distinguish (1) the literal upstream pretrained reference, (2) an independently trained reduced-scale reference-method baseline, and (3) controlled architecture/data/search comparisons. This audit establishes (1) on five genuinely solved 3x3 instances; it does not establish (2) or (3).
