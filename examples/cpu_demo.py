"""Tiny synthetic latent-prediction and verified-search smoke demonstration.

This is newly added portfolio code, not a replication of a historical study.
Training learns successor latents with shared live endpoint gradients and SIGReg.
Search is a separate shallow exact BFS check, not a learned-solver benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from jepa4math.cube import CubeEnv
from jepa4math.data import generate_walks
from jepa4math.models import CubeModel, sigreg
from jepa4math.search import bidirectional_bfs


def transition_error(model, arrays):
    model.eval()
    with torch.no_grad():
        state = torch.from_numpy(arrays["state"])
        successor = torch.from_numpy(arrays["next_state"])
        action = torch.from_numpy(arrays["action"])
        predicted = model.predictor(model.encoder(state), action)
        target = model.encoder(successor)
        return float((predicted - target).square().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local_demo.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000")

    torch.set_num_threads(1)
    torch.manual_seed(20261026)
    generator = np.random.default_rng(20261026)
    train = generate_walks(2, episodes=64, max_walk=6, seed=101)
    heldout = generate_walks(2, episodes=16, max_walk=6, seed=202)
    model = CubeModel(size=2, width=32, latent=16, blocks=1).cpu()
    initial = transition_error(model, heldout)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=0.001)
    losses = []
    model.train()
    for _ in range(args.steps):
        indices = generator.integers(0, len(train["action"]), size=64)
        state = torch.from_numpy(train["state"][indices])
        successor = torch.from_numpy(train["next_state"][indices])
        action = torch.from_numpy(train["action"][indices])
        source_latent = model.encoder(state)
        successor_latent = model.encoder(successor)
        prediction_loss = (model.predictor(source_latent, action) - successor_latent).square().mean()
        regularizer = (sigreg(source_latent, projections=16)
                       + sigreg(successor_latent, projections=16)) / 2
        loss = prediction_loss + 0.1 * regularizer
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite demonstration loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        losses.append(float(loss.detach()))
    final = transition_error(model, heldout)

    env = CubeEnv(2)
    initial_state = env.goal
    for action in ("U", "R", "F"):
        initial_state = env.step(initial_state, action)
    search = bidirectional_bfs(env, initial_state, max_nodes=5000, max_depth=6)
    if not search["verified"] or not env.verify_trace(initial_state, search["actions"]):
        raise RuntimeError("Exact search demonstration failed path replay")

    report = {
        "purpose": "Small synthetic runtime demonstration, not historical-study replication",
        "seed": 20261026, "device": "cpu", "training_steps": args.steps,
        "training_transitions": len(train["action"]),
        "heldout_episode_transitions": len(heldout["action"]),
        "split_note": "Distinct episode seeds; shallow states may overlap, so no unseen-state claim",
        "initial_heldout_successor_latent_mse": initial,
        "final_heldout_successor_latent_mse": final,
        "first_training_loss": losses[0], "last_training_loss": losses[-1],
        "latent_metric_note": "Target encoder changes during training; MSE is not search utility or a fixed-target benchmark",
        "search": search,
        "search_note": "Separate exact BFS, without learned scores; replay proves this returned path only",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
