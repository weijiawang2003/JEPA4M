#!/usr/bin/env python3
"""Generate a locked syntax-preserving rewrite pilot and run verified search.

Run: PYTHONPATH=src .venv/bin/python scripts/rewrite_benchmark.py
No trained model is used. Development families are exploratory, not final test.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from jepa4math.rewriting import (RewriteAction, RewriteEnv, RewriteState, ast_from_json,
    node_count, renamed_semantic_key, semantic_polynomial, structural_heuristic)

ROOT = Path(__file__).resolve().parents[1]


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _renamed(expr, mapping):
    return mapping.get(expr, expr) if isinstance(expr, str) else (expr[0], _renamed(expr[1], mapping), _renamed(expr[2], mapping))


def _goal(degree, coefficient):
    factor = "a"
    for _ in range(degree - 1):
        factor = ("Mul", factor, "a")
    base = ("Mul", factor, ("Add", "b", "c"))
    goal = base
    for _ in range(coefficient - 1):
        goal = ("Add", goal, base)
    return goal


def generate_dataset(config):
    env = RewriteEnv(config["max_nodes"])
    episodes = []
    family_records = []
    for family_number, (degree, coefficient) in enumerate((d,c) for d in range(1,5) for c in range(1,4)):
        split = ("train", "validation", "development")[family_number // 4]
        template = _goal(degree, coefficient)
        semantic_key = renamed_semantic_key(template)
        family_id = "semantic_" + semantic_key[:16]
        family_records.append({"family_id": family_id, "semantic_group_id": semantic_key, "split": split,
                               "template_ast": template, "degree_parameter": degree, "coefficient": coefficient})
        for episode_index in range(config["cases_per_family"]):
            seed = config["split_seeds"][split] + family_number * 100 + episode_index
            rng = random.Random(seed)
            names = ("p", "q", "r") if episode_index == 0 else ("gamma", "alpha", "beta")
            goal = _renamed(template, dict(zip(("a","b","c"),names)))
            state = RewriteState(goal, goal)
            seen = {goal}
            transitions = []
            history = []
            walk_length = 3 + family_number // 4 + episode_index
            for step in range(walk_length):
                candidates = []
                for action in env.legal_actions(state):
                    updated = env.step(state, action)
                    if updated.current not in seen:
                        candidates.append((action, updated))
                # First expose distribution; later favor neutral rearrangements
                # and factoring while retaining occasional identity operations.
                preferred = [(a,s) for a,s in candidates if (a.rule_id.startswith("distribute") and a.direction == 1)] if step == 0 else [(a,s) for a,s in candidates if "zero" not in a.rule_id and "one" not in a.rule_id]
                if not candidates:
                    break
                action, updated = rng.choice(preferred or candidates)
                transitions.append({"current_ast": state.current, "action": action.to_dict(), "next_ast": updated.current,
                                    "goal_ast": goal, "label_provenance": "exact_rule_execution; no distance label", "is_explicit_detour": False})
                history.append(action)
                seen.add(updated.current)
                state = updated
            # An explicit expansion/reduction detour records a non-monotone
            # trajectory and distinguishes walk length from shortest distance.
            detours = [a for a in env.legal_actions(state) if a.rule_id == "add_zero_right" and a.direction == 1]
            if detours:
                detour = rng.choice(detours)
                for action in (detour, detour.inverse()):
                    updated = env.step(state, action)
                    transitions.append({"current_ast": state.current, "action": action.to_dict(), "next_ast": updated.current,
                                        "goal_ast": goal, "label_provenance": "exact_rule_execution; no distance label", "is_explicit_detour": True})
                    history.append(action)
                    state = updated
            for transition in transitions:
                assert semantic_polynomial(transition["current_ast"]) == semantic_polynomial(transition["next_ast"])
            witness = [a.inverse() for a in reversed(history)]
            assert env.verify_trace(state, witness)
            episode_id = f"rewrite_{split}_{family_number:02d}_{episode_index:02d}"
            metadata = {"episode_id": episode_id, "family_id": family_id, "semantic_group_id": semantic_key, "seed": seed}
            for transition in transitions:
                transition.update(metadata)
            episodes.append({**metadata, "split": split, "current_ast": state.current, "goal_ast": goal,
                             "generator_walk_length": len(history), "distance_label": None,
                             "distance_label_provenance": "unknown; generator length is an upper bound only",
                             "verified_witness": [a.to_dict() for a in witness], "transitions": transitions,
                             "state_pair_sha256": _hash([state.current,goal]),
                             "history_sha256": _hash([a.to_dict() for a in history])})
    audit = audit_dataset(episodes)
    assert audit["cross_split_semantic_groups"] == 0
    assert audit["cross_split_state_pairs"] == 0
    assert audit["cross_split_shared_histories"] == 0
    return {"protocol": config["protocol"], "config_sha256": _hash(config), "episodes": episodes,
            "families": family_records, "leakage_audit": audit,
            "reserved_final_test_seed": config["reserved_final_test_seed"], "final_test_generated": False,
            "model_input_fields": ["current_ast", "goal_ast", "action (when action-conditioned)"],
            "metadata_never_model_input": ["episode_id", "family_id", "semantic_group_id", "seed", "next_ast", "distance_label_provenance", "verified_witness"],
            "dataset_sha256": _hash(episodes)}


def audit_dataset(episodes):
    def shared(field):
        groups = {}
        for episode in episodes:
            groups.setdefault(episode[field],set()).add(episode["split"])
        return sum(len(splits)>1 for splits in groups.values())
    # Canonicalized semantic groups also catch duplicated templates, algebraic
    # equivalents, and variable-renamed copies across split boundaries.
    return {"cross_split_semantic_groups": shared("semantic_group_id"),
            "cross_split_state_pairs": shared("state_pair_sha256"),
            "cross_split_shared_histories": shared("history_sha256"),
            "duplicate_state_pairs": len(episodes)-len({e["state_pair_sha256"] for e in episodes}),
            "split_counts": dict(Counter(e["split"] for e in episodes)),
            "semantic_group_counts": {split: len({e["semantic_group_id"] for e in episodes if e["split"]==split}) for split in ("train","validation","development")},
            "held_out_rule_compositions_claimed": False,
            "structural_overlap_limitation": "Families share operator/rule vocabulary and a power-times-sum generator; this is semantic-family holdout, not broad template generalization."}


def search(env, initial, method, *, node_budget, wall_seconds, beam_width, max_depth):
    if method not in ("exact_bfs", "structural_beam"):
        raise ValueError("Unknown search method")
    if node_budget < 0 or wall_seconds < 0 or beam_width < 1 or max_depth < 0:
        raise ValueError("Invalid search budget")
    # A prior algorithm's generated ASTs must not prewarm the next method.
    # Cache disposal belongs to run isolation, outside the search timer.
    node_count.cache_clear()
    started = time.perf_counter()
    frontier = deque([(initial,())])
    visited = {initial.current}
    expansions = generated = exact_calls = 0
    peak_frontier = 1
    goal_tests = 1
    solution = () if env.is_goal(initial) else None
    reason = "solved" if solution is not None else "frontier_exhausted"
    while frontier and solution is None:
        if expansions >= node_budget:
            reason = "node_budget"
            break
        if time.perf_counter()-started >= wall_seconds:
            reason = "wall_budget"
            break
        if method == "exact_bfs":
            state,path = frontier.popleft()
            working = [(state,path)]
        else:
            working = list(frontier)
            frontier.clear()
        candidates = []
        for state,path in working:
            goal_tests += 1
            if env.is_goal(state):
                solution = path
                break
            if expansions >= node_budget or time.perf_counter()-started >= wall_seconds:
                reason = "node_budget" if expansions >= node_budget else "wall_budget"
                break
            if method != "exact_bfs" and len(path) >= max_depth:
                reason = "depth_budget"
                continue
            expansions += 1
            exact_calls += 1  # legal_actions enumeration
            for action in env.legal_actions(state):
                if time.perf_counter()-started >= wall_seconds:
                    reason = "wall_budget"
                    break
                updated = env.step(state, action)
                exact_calls += 1
                generated += 1
                if updated.current in visited:
                    continue
                visited.add(updated.current)
                updated_path = path+(action,)
                goal_tests += 1
                if env.is_goal(updated):
                    solution = updated_path
                    break
                if method == "exact_bfs":
                    frontier.append((updated,updated_path))
                else:
                    candidates.append((structural_heuristic(updated.current,updated.goal), generated, updated, updated_path))
            peak_frontier = max(peak_frontier,len(frontier)+len(candidates))
            if solution is not None:
                break
        if solution is not None:
            reason = "solved"
            break
        if method != "exact_bfs":
            candidates.sort(key=lambda item:(item[0],item[1]))
            frontier = deque((state,path) for _,_,state,path in candidates[:beam_width])
    verified = solution is not None and env.verify_trace(initial,solution)
    if solution is not None and not verified:
        raise RuntimeError("Search returned an invalid solution")
    # Trace replay and final exact verification are included in wall timing.
    verification_transitions = len(solution) if solution is not None else 0
    exact_calls += verification_transitions + goal_tests
    elapsed = time.perf_counter()-started
    return {"method": method, "solved": verified, "termination": reason,
            "expansions": expansions, "generated_successors": generated, "model_forward_passes": 0,
            "exact_environment_calls": exact_calls, "wall_seconds_including_verification": elapsed,
            "exact_transition_calls": generated,
            "action_enumeration_calls": expansions,
            "goal_test_calls": goal_tests,
            "verification_transition_calls": verification_transitions,
            "node_count_cache": "cleared before each separately timed search",
            "solution_length": len(solution) if solution is not None else None,
            "exact_shortest_distance": len(solution) if verified and method=="exact_bfs" else None,
            "path": [a.to_dict() for a in solution] if solution is not None else None,
            "peak_candidate_frontier_states": peak_frontier, "visited_states": len(visited),
            "node_budget": node_budget, "wall_budget_seconds": wall_seconds,
            "beam_width": beam_width if method=="structural_beam" else None}


def run(config, *, max_cases=None):
    dataset = generate_dataset(config)
    manifest_path = ROOT/"data/manifests/rewrite_protocol_v1.json"
    _write(manifest_path,dataset)
    source_files = [ROOT/"src/jepa4math/rewriting.py",Path(__file__).resolve(),ROOT/"tests/test_rewriting.py"]
    commit = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True).stdout.strip() or None
    output = {"protocol": config["protocol"], "config": config, "config_sha256": _hash(config),
              "dataset_sha256": dataset["dataset_sha256"], "dataset_path": str(manifest_path.relative_to(ROOT)),
              "source_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "git_commit": commit, "python": platform.python_version(), "platform": platform.platform(),
              "records": [], "complete": False,
              "claim_scope": "Verified exact-environment pilot; no learned rewriting model or JEPA comparison has been run."}
    env = RewriteEnv(config["max_nodes"])
    episodes = dataset["episodes"][:max_cases] if max_cases else dataset["episodes"]
    result_path = ROOT/"results/rewrite_benchmark.json"
    for index,episode in enumerate(episodes):
        initial = env.make_state(episode["current_ast"],episode["goal_ast"])
        for budget_mode in config["budget_modes"]:
            for method in config["methods"]:
                result = search(env,initial,method,node_budget=config["node_budget"] if budget_mode=="fixed_nodes" else 10**9,
                                wall_seconds=60.0 if budget_mode=="fixed_nodes" else config["wall_seconds"],
                                beam_width=config["beam_width"],max_depth=config["beam_max_depth"])
                result.update({"episode_id":episode["episode_id"], "split":episode["split"],
                               "family_id":episode["family_id"], "seed":episode["seed"], "budget_mode":budget_mode,
                               "current_nodes":node_count(initial.current), "goal_nodes":node_count(initial.goal),
                               "generator_walk_length":episode["generator_walk_length"]})
                output["records"].append(result)
                _write(result_path,output)
        print(f"Finished rewrite case {index+1}/{len(episodes)}: {episode['episode_id']}",flush=True)
    exact = {}
    for record in output["records"]:
        if record["exact_shortest_distance"] is not None:
            exact[record["episode_id"]] = record["exact_shortest_distance"]
    for record in output["records"]:
        optimum = exact.get(record["episode_id"])
        record["known_exact_shortest_distance"] = optimum
        record["optimality_gap"] = record["solution_length"]-optimum if record["solved"] and optimum is not None else None
    summaries = []
    for split in ("train","validation","development"):
        for budget_mode in config["budget_modes"]:
            for method in config["methods"]:
                rows = [r for r in output["records"] if (r["split"],r["budget_mode"],r["method"])==(split,budget_mode,method)]
                if rows:
                    summaries.append({"split":split,"budget_mode":budget_mode,"method":method,"cases":len(rows),
                                      "solved":sum(r["solved"] for r in rows), "solve_rate":sum(r["solved"] for r in rows)/len(rows),
                                      "mean_expansions":sum(r["expansions"] for r in rows)/len(rows),
                                      "mean_wall_seconds":sum(r["wall_seconds_including_verification"] for r in rows)/len(rows)})
    output["summaries"] = summaries
    output["complete"] = True
    output["cases_run"] = len(episodes)
    _write(result_path,output)
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/rewrite_protocol.json")
    parser.add_argument("--max-cases",type=int,default=None)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()),max_cases=args.max_cases)
