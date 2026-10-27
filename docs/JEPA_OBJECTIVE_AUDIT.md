# JEPA / SIGReg objective audit

Audited 2026-09-17. This document separates inspected source behavior from recommendations for JEPA4Math. No experiments are reported here.

## Actionable objective

For a batch of **exact** transitions `(s_i, a_i, s'_i)`, use a shared encoder, an action-conditioned predictor, and

\[
z_i=E_\theta(s_i),\quad y_i=E_\theta(s'_i),\quad
\hat y_i=P_\psi(z_i,A_\eta(a_i)),
\]
\[
L_{\rm pred}=\frac1{Nd}\sum_{i,k}(\hat y_{ik}-y_{ik})^2,\qquad
L=L_{\rm pred}+\beta\,\frac{R(z)+R(y)}2.
\]

**Gradients propagate through both `E(s)` and `E(s')`, through the predictor, and through both regularizer inputs. No target `detach`, EMA encoder, or encoder output L2 normalization is part of this objective.** Only sampled projection directions are normalized. This is a one-step, two-view specialization of the inspected LeWorldModel training implementation, which uses additive `prediction + weight * regularization`. Its regularizer is evaluated separately across batch for each time position, then averaged over time. [Official LeWorldModel training code](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py#L17)

LeJEPA instead writes `(1-lambda)*prediction + lambda*regularization`; the objectives are proportional when `beta=lambda/(1-lambda)`, but gradient magnitudes and optimizer behavior need not be identical. Log the convention. The inspected LeWorldModel config uses `beta=0.09`; this is a starting reference, **not a validated cube/rewrite hyperparameter**. [LeWM config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)

## Exact regularizer and reductions

For `Z` of shape `(N,d)`, draw `M` independent vectors `b_j ~ N(0,I_d)` and set `u_j=b_j/||b_j||_2`. Redraw directions each training call. Define

\[
c_j(t)=\frac1N\sum_i\cos(tu_j^Tz_i),\qquad
q_j(t)=\frac1N\sum_i\sin(tu_j^Tz_i),\qquad
\phi(t)=e^{-t^2/2}.
\]

The audited current implementation computes

\[
R(Z)=\frac{N}{M}\sum_{j=1}^{M}\sum_{\ell=0}^{16}
w_\ell\left[(c_j(t_\ell)-\phi(t_\ell))^2+q_j(t_\ell)^2\right],
\]

where `t_l=3*l/16`, `dt=3/16`, interior `w_l=2*dt*phi(t_l)`, and endpoint `w_0=w_16=dt*phi(t_l)`. This is doubled trapezoidal integration on `[0,3]`, equivalent by symmetry to the corresponding grid on `[-3,3]`. It retains the **N factor** of the Epps–Pulley statistic. For `(V,N,d)` inputs, compute ECF means over `N`, not over `V*N`, and average final statistics over both `V` and `M`. [Official Epps–Pulley code](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py#L15), [official slicing code](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py#L15)

LeJEPA's paper pseudocode instead uses 17 knots over `[-5,5]`; prose also discusses a Gaussian window convention different from its displayed implementation. These change finite quadrature values. Our reference above follows the **current executable default**: `exp(-t^2/2)` for both target CF and window. The minimal official example explicitly explains its change to positive-half quadrature. [LeJEPA paper §4–5](https://arxiv.org/html/2511.08544v3), [official minimal example](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)

## Compact implementation recommendation

The following is an independently expressed, single-device implementation of the audited formula. It accepts `(N,d)` or `(V,N,d)` and keeps the gradient graph on both views. `M=256` is a reasonable compact first implementation; save it in each run config. Current LeWM defaults to 1024. Compute the trigonometric loss in FP32; this function should be called outside autocast or under disabled autocast.

```python
import torch
import torch.nn.functional as F

def sigreg(encoded, directions=256):
    values = encoded.float()
    n, width = values.shape[-2:]
    axes = F.normalize(torch.randn(width, directions,
                                   device=values.device), dim=0)
    knots = torch.linspace(0., 3., 17, device=values.device)
    reference = (-knots.square() / 2).exp()
    phase = (values @ axes)[..., None] * knots
    real = phase.cos().mean(dim=-3)
    imag = phase.sin().mean(dim=-3)
    integrand = ((real - reference).square() + imag.square()) * reference
    return (2 * n * torch.trapezoid(integrand, knots, dim=-1)).mean()

z = encoder(current)
y = encoder(successor)
prediction = predictor(z, action_encoder(action))
pred_loss = F.mse_loss(prediction, y)
reg_loss = sigreg(torch.stack((z, y)))
loss = pred_loss + beta * reg_loss
```

For exact reproducibility, save Torch CPU/device RNG state as well as training seeds; validation SIGReg should use fixed diagnostic directions or a separately seeded generator so validation does not perturb subsequent training projections. A shared draw across the two views matches the reference multi-view implementation. A distributed version must synchronize directions, reduce the differentiable ECF means, and multiply by global sample count; the simple code above does **not** implement DDP.

Do not substitute `mean(square(z-z_next))`: on a connected cube graph, perfect action-independent invariance forces every state to share an embedding. Likewise, the image-view LeJEPA centroid loss must not be imported as the transition prediction loss. In the official image example it is `mean((view_mean - each_view)^2)`, including feature averaging. As an algebraic caution, for two views this is one quarter of their ordinary pairwise MSE; formulas that silently interchange these losses alter regularization balance.

## Transfer caveats and falsification checks

The points below are our deductions, not performance guarantees from the cited papers.

1. **Global distribution matching does not ensure useful action structure.** An encoder can preserve irrelevant state distinctions while collapsing the distinctions between legal successors. For rewriting, variance across semantic families can hide collapse inside each semantic family. Measure both.
2. **Finite deterministic state spaces cannot be exactly Gaussian.** Treat Gaussian matching as a regularizer. It is not an admissibility guarantee, a sufficient statistic theorem for shortest paths, or a guarantee of optimization success. The LeWM paper itself reports a limitation on low-complexity TwoRoom, where the Gaussian prior may be mismatched to intrinsic dimension. [LeWM discussion](https://arxiv.org/html/2603.19312v3)
3. **SIGReg is not expected to be zero on a finite Gaussian batch.** For IID standard-normal projected samples, `E[N*|ECF-phi|^2]=1-phi^2`. Thus its expected quadrature loss has a positive sampling floor. Report reference Gaussian draws at the same batch size and dimension. Do not compare raw statistics across different batch sizes without recording N.
4. **Exact all-zero embeddings can have zero regularizer gradient.** At zero, cosine derivatives vanish and the sine-error derivative is multiplied by a zero sine mean. A positive penalty does not guarantee escape from every stationary collapsed initialization. Ordinary random initialization and explicit variance/rank monitoring matter.
5. **Correlated trajectories are not IID samples.** Batch across independently generated episodes where practical. Never interpret the resulting training statistic as a calibrated Gaussianity-test p-value. Pooling all time positions is a different objective and changes both correlations and N scaling.
6. **Output normalization changes the target family.** L2-normalized embeddings lie on a sphere, inconsistent with `N(0,I)`. Per-example LayerNorm fixes an aggregate mean/norm and can constrain the final latent distribution; use a final unconstrained linear layer if necessary. BatchNorm inside a projector is present in LeJEPA's example and is a separate architectural choice.

Minimum diagnostic panel, all on held-out states and deterministic model evaluation:

| Diagnostic | Definition / interpretation |
|---|---|
| Prediction versus identity | Report MSE of `P(z,a)` to `y` alongside MSE of `z` to `y`; also report their ratio. Low prediction loss alone is insufficient. |
| Same-state wrong-action gap | For fixed `(s,s')`, replace the correct action by every other legal action. Report wrong-minus-correct MSE and retrieval margins. Global batch shuffling is a weaker check because state/action correlations can confound it. |
| Hard successor retrieval | Rank `E(step(s,b))` by distance to `P(E(s),a)` over all legal `b`; report top-1, MRR, and target separation. Deduplicate identical successor states and use tie-aware ranks so action aliases do not create false errors or successes. |
| Predicted and true action spread | Compare average pairwise distances between predictions for different actions to corresponding true-successor distances; inspect correlation of the two distance matrices. A predictor can depend on actions in an irrelevant way. |
| Rank and scale | Per-feature variance, mean norm, covariance eigenvalues, and entropy effective rank `exp(-sum(p*log p))`, `p=eigenvalues/sum(eigenvalues)`. Declare rank zero when total variance is zero; also report participation ratio. |
| Search utility | Fit the same scorer to each encoder with matched labels, then compare exact-successor scoring and predicted-successor scoring on identical instances and budgets. |
| Multi-step consistency | Compose only recorded, exactly legal action sequences; report latent error versus step and compare repeated exact re-encoding. Never treat a low continuous rollout error as proof of valid symbolic transitions. |
| Rewrite conditional collapse | Within semantic-family variance/rank, distinct-AST nearest-neighbor collisions, and same-state hard-successor retrieval; separately report cross-family metrics. |

A strong positive H3 result requires correct-action retrieval and a clear wrong-action penalty **with noncollapsed true-successor embeddings**. For the training control, randomize actions independently of transitions throughout predictive pretraining, keeping examples, model, optimizer, and scorer-label budget matched. Record separately an inference-only shuffled-action control.

## Directly relevant prior work and positioning

| Primary work | Why it changes the research design |
|---|---|
| [LeJEPA](https://arxiv.org/abs/2511.08544), Balestriero & LeCun | Source of Gaussian distribution matching. Its vision representation results do not establish search improvement. |
| [LeWorldModel](https://arxiv.org/abs/2603.19312), Maes et al. | Already combines action-conditioned latent prediction and SIGReg end to end, without target stop-gradient. Our potential contribution must be the controlled combinatorial/math-search evidence and structural generalization, not that objective alone. |
| [DINO-WM](https://dino-wm.github.io/), Zhou et al. | Frozen pretrained visual features plus action-conditioned dynamics and model-predictive goal reaching. Motivates separating encoder quality from dynamics quality and accounting for encoder inference cost. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985), Assran et al. | Action-conditioned post-training and planning are established JEPA uses; differs substantially in pretraining scale and observation domain. |
| [DeepCubeAI: Learning Discrete World Models for Heuristic Search](https://rlj.cs.umass.edu/2024/papers/Paper225.html), Agostinelli & Soltani | Direct combinatorial precedent: discrete learned world model, goal-generalizing learned heuristic, search, including Rubik's Cube. Discreteness addresses error accumulation and state re-identification. This is a necessary related-work comparison before claiming novelty for latent cube search. [Author code](https://github.com/misaghsoltani/DeepCubeAI). |
| [Latplan: Classical Planning in Deep Latent Space](https://arxiv.org/abs/2107.00110), Asai et al. | Learns propositional planning representations and action models from image transitions. Useful contrast with keeping exact symbolic dynamics as the authority. [Author code](https://github.com/guicho271828/latplan). |
| [Plausibility-Based Heuristics for Latent Space Classical Planning](https://arxiv.org/abs/2306.11434) | Explicitly addresses invalid states/plans in learned latent planning. Supports measuring exact replay validity separately from latent goal matching. |

This bounded review did not reproduce DINO-WM, V-JEPA 2, DeepCubeAI, or Latplan, and their source trees were not audited. The user-specified NeurIPS 2025 baseline is covered by a separate audit.

## Provenance and reuse

| Repository inspected | Commit | License file observed | Files read |
|---|---|---|---|
| [galilai-group/lejepa](https://github.com/galilai-group/lejepa) | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | CC BY-NC 4.0 | `MINIMAL.md`, `lejepa/univariate/epps_pulley.py`, `lejepa/univariate/base.py`, `lejepa/multivariate/slicing.py` |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | MIT | `module.py`, `train.py`, `jepa.py`, `config/train/lewm.yaml`, `README.md` |

These were read from shallow clones under `/tmp/jepa4m-lejepa-audit` and `/tmp/jepa4m-lewm-audit`; temporary copies are disposable and the commit-pinned URLs above are the durable references. LeWM's published setup uses Python 3.10 plus `stable-worldmodel[train,env]`; its full training stack is unnecessary for a small standalone Torch loss. Copying MIT source requires retaining its notice. The independently expressed pseudocode above is a specification suggestion, not a claim that repository training code has already passed this audit.
