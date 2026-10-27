"""Bounded exact-state search with learned scores used only for ordering.

All successful paths are replayed through CubeEnv.verify_trace. Timers include
validation, enumeration, scoring, and replay; an indivisible call can overrun a
deadline, which is reported. A budget failure never proves unreachability.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


def _metrics_snapshot(scorer):
    metrics = getattr(scorer, "metrics", None)
    if callable(metrics):
        metrics = metrics()
    return dict(metrics) if isinstance(metrics, dict) else {}


class _Run:
    def __init__(self, env, initial, goal, max_nodes, max_seconds, scorer=None):
        self.started = time.perf_counter()
        if max_nodes < 0 or int(max_nodes) != max_nodes:
            raise ValueError("max_nodes must be a nonnegative integer")
        if max_seconds is not None and (max_seconds < 0 or not math.isfinite(max_seconds)):
            raise ValueError("max_seconds must be finite and nonnegative, or None")
        self.env, self.scorer = env, scorer
        # Public serialization validates supplied cube configurations.
        self.initial_key = env.serialize(initial)
        self.goal_key = env.serialize(env.goal if goal is None else goal)
        self.initial = np.frombuffer(self.initial_key, dtype=np.uint8)
        self.goal = np.frombuffer(self.goal_key, dtype=np.uint8)
        self.max_nodes, self.max_seconds = int(max_nodes), max_seconds
        self.deadline = math.inf if max_seconds is None else self.started + max_seconds
        self.expanded = self.generated = self.calls = self.scorer_calls = 0
        self.peak_frontier = self.peak_visited = 1
        self.metrics_before = _metrics_snapshot(scorer)
        self.path_storage = 0

    def budget(self):
        if time.perf_counter() >= self.deadline:
            return "time_budget"
        if self.expanded >= self.max_nodes:
            return "node_budget"
        return None

    def track(self, frontier, visited, path_storage=0):
        self.peak_frontier = max(self.peak_frontier, frontier)
        self.peak_visited = max(self.peak_visited, visited)
        self.path_storage = max(self.path_storage, path_storage)

    def finish(self, reason, path=None, optimal=False, depth=0, budget_termination=None):
        names = [] if path is None else [self.env.actions[int(a)] for a in path]
        verified = False
        if path is not None:
            verified = bool(self.env.verify_trace(self.initial, names, self.goal))
            self.calls += len(names)
            if not verified:
                raise RuntimeError("Search produced a path that failed exact replay")
        metrics_after = _metrics_snapshot(self.scorer)
        forward_count = metrics_after.get("model_forward_calls")
        if forward_count is not None:
            forward_count = int(forward_count - self.metrics_before.get("model_forward_calls", 0))
        elif self.scorer is None:
            forward_count = 0
        elapsed = time.perf_counter() - self.started
        # Coarse explicit storage model, not process RSS: byte keys, dictionary
        # and tuple references, plus path elements and current frontier records.
        memory_estimate = (self.peak_visited * (len(self.initial_key) + 128)
                           + 8 * self.path_storage + 80 * self.peak_frontier)
        return {
            "solved": verified,
            "verified": verified,
            "actions": names,
            "action_indices": [] if path is None else [int(a) for a in path],
            "solution_length": len(names) if verified else None,
            "optimal": bool(verified and optimal),
            "exact_distance": len(names) if verified and optimal else None,
            "stop_reason": reason,
            "budget_termination": budget_termination,
            "expanded_nodes": self.expanded,
            "generated_successors": self.generated,
            "environment_calls": self.calls,
            "model_forward_calls": forward_count,
            "scorer_calls": self.scorer_calls,
            "wall_seconds": elapsed,
            "time_budget_overrun": bool(self.max_seconds is not None and elapsed > self.max_seconds),
            "time_budget_overrun_seconds": 0.0 if self.max_seconds is None else max(0.0, elapsed - self.max_seconds),
            "max_nodes": self.max_nodes,
            "max_seconds": self.max_seconds,
            "depth_reached": depth,
            "peak_frontier": self.peak_frontier,
            "peak_visited": self.peak_visited,
            "memory_estimate_bytes": int(memory_estimate),
            "memory_measurement": "estimated search bookkeeping; excludes model and process RSS",
        }


def beam_search(env, initial, scorer: Callable, beam_width=256,
                max_nodes=10000, max_seconds=None, max_depth=80, goal=None,
                score_batch_size=256, frontier_observer=None):
    """Beam search with exact transitions and lower-is-better candidate scores.

    scorer(states[B,N], actions[B], successors[B,N], goal[N]) -> scores[B].
    Optional scorer.metrics['model_forward_calls'] is a cumulative counter;
    without it neural forward count is null, rather than guessed from batches.
    Nodes are parent expansions; generated successors count legal transitions
    actually enumerated after immediate-inverse pruning, including duplicates.
    Optional frontier_observer receives a complete scored frontier just before
    selection is applied. Candidate tuples contain immutable bytes/paths;
    mutable score/index/state arrays are copies. Observation time is included
    in the timer, so diagnostic observers should use node budgets, not timing
    comparisons. The default None preserves historical search behavior.
    """
    run = _Run(env, initial, goal, max_nodes, max_seconds, scorer)
    if beam_width < 1 or int(beam_width) != beam_width:
        raise ValueError("beam_width must be a positive integer")
    if max_depth < 0 or int(max_depth) != max_depth:
        raise ValueError("max_depth must be a nonnegative integer")
    if score_batch_size < 1 or int(score_batch_size) != score_batch_size:
        raise ValueError("score_batch_size must be a positive integer")
    if run.initial_key == run.goal_key:
        return run.finish("solved", (), optimal=True)
    frontier = [(run.initial_key, ())]
    visited = {run.initial_key}
    storage = 0
    for depth in range(int(max_depth)):
        candidates = []
        for key, path in frontier:
            stop = run.budget()
            if stop:
                return run.finish(stop, depth=depth)
            run.expanded += 1
            state = np.frombuffer(key, dtype=np.uint8)
            choices = [a for a in range(len(env.actions))
                       if not path or a != (path[-1] ^ 1)]
            successors = state[env.permutations[choices]]
            run.generated += len(choices)
            run.calls += len(choices)
            for action, successor in zip(choices, successors):
                next_key = successor.tobytes()
                next_path = (*path, action)
                # Goal detection precedes scoring and all duplicate pruning.
                if next_key == run.goal_key:
                    return run.finish("solved", next_path, depth=depth + 1)
                if next_key in visited:
                    continue
                visited.add(next_key)
                candidates.append((key, action, next_key, next_path))
                storage += len(next_path)
            run.track(len(frontier) + len(candidates), len(visited), storage)
        if not candidates:
            return run.finish("frontier_exhausted", depth=depth + 1)
        # Further scoring cannot help after the parent-expansion budget ends.
        stop = run.budget()
        if stop:
            return run.finish(stop, depth=depth + 1)
        if depth + 1 == max_depth:
            return run.finish("depth_budget", depth=depth + 1)
        chunks = []
        for start in range(0, len(candidates), int(score_batch_size)):
            if time.perf_counter() >= run.deadline:
                return run.finish("time_budget", depth=depth + 1)
            batch = candidates[start:start + int(score_batch_size)]
            states = np.stack([np.frombuffer(c[0], dtype=np.uint8) for c in batch])
            actions = np.asarray([c[1] for c in batch], dtype=np.int64)
            next_states = np.stack([np.frombuffer(c[2], dtype=np.uint8) for c in batch])
            scores = np.asarray(scorer(states, actions, next_states, run.goal), dtype=np.float64)
            run.scorer_calls += 1
            if scores.shape != (len(batch),) or not np.isfinite(scores).all():
                raise ValueError("scorer must return one finite scalar per successor")
            chunks.append(scores)
        if time.perf_counter() >= run.deadline:
            return run.finish("time_budget", depth=depth + 1)
        all_scores = np.concatenate(chunks)
        ordering = np.argsort(all_scores, kind="stable")[:int(beam_width)]
        if frontier_observer is not None:
            frontier_observer(depth=depth + 1, candidates=tuple(candidates),
                              scores=all_scores.copy(), selected=ordering.copy(),
                              initial=run.initial.copy(), goal=run.goal.copy())
        frontier = [(candidates[int(i)][2], candidates[int(i)][3]) for i in ordering]
    return run.finish("depth_budget", depth=int(max_depth))


def bidirectional_bfs(env, initial, max_nodes=100000, max_seconds=None,
                      max_depth=20, goal=None):
    """Exact bidirectional BFS, bounded by expansions, time, and path length.

    Complete BFS layers expand from the smaller frontier. A meeting path is
    certified shortest only after the current layer completes; if a budget
    interrupts that layer, an already found path remains verified but is not
    labelled optimal. ``max_depth`` bounds the combined search depth.
    """
    run = _Run(env, initial, goal, max_nodes, max_seconds)
    if max_depth < 0 or int(max_depth) != max_depth:
        raise ValueError("max_depth must be a nonnegative integer")
    if run.initial_key == run.goal_key:
        return run.finish("solved", (), optimal=True)
    seen = [{run.initial_key: ()}, {run.goal_key: ()}]
    frontiers = [[run.initial_key], [run.goal_key]]
    depths = [0, 0]
    storage = 0
    run.track(2, 2)
    while all(frontiers) and sum(depths) < max_depth:
        side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
        other = 1 - side
        new_frontier, best = [], None
        for key in frontiers[side]:
            stop = run.budget()
            if stop:
                return run.finish("solved" if best is not None else stop,
                                  best, optimal=False, depth=sum(depths),
                                  budget_termination=stop)
            run.expanded += 1
            path = seen[side][key]
            state = np.frombuffer(key, dtype=np.uint8)
            choices = [a for a in range(len(env.actions))
                       if not path or a != (path[-1] ^ 1)]
            successors = state[env.permutations[choices]]
            run.generated += len(choices)
            run.calls += len(choices)
            for action, successor in zip(choices, successors):
                next_key = successor.tobytes()
                if next_key in seen[side]:
                    continue
                next_path = (*path, action)
                seen[side][next_key] = next_path
                storage += len(next_path)
                new_frontier.append(next_key)
                if next_key in seen[other]:
                    forwards = next_path if side == 0 else seen[other][next_key]
                    backwards = seen[other][next_key] if side == 0 else next_path
                    joined = forwards + tuple(a ^ 1 for a in reversed(backwards))
                    if best is None or len(joined) < len(best):
                        best = joined
            run.track(len(new_frontier) + len(frontiers[other]),
                      len(seen[0]) + len(seen[1]), storage)
        depths[side] += 1
        frontiers[side] = new_frontier
        if best is not None:
            return run.finish("solved", best, optimal=True, depth=sum(depths))
    reason = "depth_budget" if sum(depths) >= max_depth else "frontier_exhausted"
    return run.finish(reason, depth=sum(depths))


# Name convenient for callers that group exact search methods under BFS.
exact_bidirectional_bfs = bidirectional_bfs
