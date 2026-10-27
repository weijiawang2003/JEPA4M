"""Episode-disjoint Cube walks. Step indices are walk labels, never distances."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .cube import CubeEnv
from .io import ROOT, arrays_hash, write_json

# The locked seeds are declared up front and NOT evaluated during development.
SPLIT_SEEDS = {"train": 110701, "validation": 220701, "development": 330701,
               "final_locked": 990701}

def generate_walks(size, episodes, max_walk, seed):
    env = CubeEnv(size)
    rng = np.random.default_rng(seed)
    current = np.repeat(env.goal[None], episodes, axis=0)
    last = np.full(episodes, -1, dtype=np.int64)
    arrays = {k: [] for k in ("state", "next_state", "action", "walk_step", "episode_id")}
    for k in range(max_walk):
        actions = rng.integers(0, 12, size=episodes)
        # All 12 actions are possible at the first step; reject only true inverses.
        bad = (last >= 0) & (actions == (last ^ 1))
        while bad.any():
            actions[bad] = rng.integers(0, 12, size=int(bad.sum()))
            bad = (last >= 0) & (actions == (last ^ 1))
        successor = np.take_along_axis(current, env.permutations[actions], axis=1)
        arrays["state"].append(current.copy())
        arrays["next_state"].append(successor.copy())
        arrays["action"].append(actions.astype(np.int64))
        arrays["walk_step"].append(np.full(episodes, k + 1, dtype=np.float32))
        arrays["episode_id"].append(np.arange(episodes, dtype=np.int64))
        current, last = successor, actions
    return {k: np.concatenate(v) for k,v in arrays.items()}

def load_dataset(size, episodes, max_walk, split="train"):
    if split == "final_locked":
        raise ValueError("Locked final set requires a separately preregistered final evaluation")
    seed = SPLIT_SEEDS[split]
    stem = f"cube{size}_{split}_e{episodes}_k{max_walk}_s{seed}"
    path = ROOT / "data" / f"{stem}.npz"
    if path.exists():
        with np.load(path) as f:
            arrays = {k: f[k] for k in f.files}
    else:
        arrays = generate_walks(size, episodes, max_walk, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    digest = arrays_hash(arrays)
    manifest_path=ROOT / "data" / "manifests" / f"{stem}.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous["arrays_sha256"]!=digest:
            raise ValueError(f"Cached dataset hash mismatch: {path}; retain original artifacts before regenerating")
    manifest = {"name": stem, "path": str(path.relative_to(ROOT)), "size": size,
                "episodes": episodes, "max_walk": max_walk, "seed": seed,
                "split": split, "label": "nonbacktracking_walk_step",
                "label_is_shortest_distance": False, "samples": len(arrays["action"]),
                "arrays_sha256": digest, "split_seeds": SPLIT_SEEDS,
                "model_inputs": ["state", "action", "goal_state"],
                "note": "Independent episode seeds. Shallow state overlap is possible and audited; no unseen-state claim without audit."}
    write_json(manifest_path, manifest)
    return arrays, manifest

def evaluation_cases(size, lengths, per_length, seed=SPLIT_SEEDS["development"]):
    env = CubeEnv(size)
    cases = []
    for length in lengths:
        for i in range(per_length):
            case_seed = seed + size * 100000 + length * 1000 + i
            state, history = env.scramble(length, seed=case_seed)
            cases.append({"id": f"cube{size}_walk{length}_{i}", "seed": case_seed,
                          "scramble_length": length, "history": list(history),
                          "state": state.tolist(), "label_provenance": "generated_walk_length"})
    return cases
