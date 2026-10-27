import time
import unittest
from collections import deque

import numpy as np

from jepa4math.cube import CubeEnv
from jepa4math.search import beam_search, bidirectional_bfs


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.env = CubeEnv(size=2)

    def scrambled(self, actions):
        state = self.env.goal
        for action in actions:
            state = self.env.step(state, action)
        return state

    def scorer(self, states, actions, successors, goal):
        # Check candidate alignment, not merely the ability to call a scorer.
        np.testing.assert_array_equal(
            successors, np.take_along_axis(states, self.env.permutations[actions], axis=1))
        return (successors != goal).sum(axis=1)

    def test_initial_goal_and_no_expansions(self):
        result = beam_search(self.env, self.env.goal, self.scorer, max_nodes=0)
        self.assertTrue(result["verified"])
        self.assertEqual(result["exact_distance"], 0)
        self.assertEqual(result["expanded_nodes"], 0)

    def test_goal_detected_before_hostile_scoring(self):
        def forbidden(*args):
            raise AssertionError("Goal must be accepted before scoring")
        state = self.scrambled(["U"])
        result = beam_search(self.env, state, forbidden, beam_width=1, max_nodes=1)
        self.assertTrue(result["verified"])
        self.assertEqual(result["solution_length"], 1)
        self.assertEqual(result["environment_calls"], result["generated_successors"] + 1)

    def test_verified_nontrivial_beam_and_alignment(self):
        state = self.scrambled(["U", "R", "F"])
        result = beam_search(self.env, state, self.scorer, beam_width=500,
                             max_nodes=2000, max_depth=6, score_batch_size=7)
        self.assertTrue(result["solved"])
        self.assertTrue(self.env.verify_trace(state, result["actions"]))
        self.assertLessEqual(result["expanded_nodes"], 2000)
        self.assertIsNone(result["model_forward_calls"])

    def test_node_and_depth_budgets(self):
        state = self.scrambled(["U", "R", "F"])
        for search in (beam_search, bidirectional_bfs):
            extras = {"scorer": self.scorer} if search is beam_search else {}
            result = search(self.env, state, max_nodes=0, **extras)
            self.assertFalse(result["solved"])
            self.assertEqual(result["expanded_nodes"], 0)
            self.assertEqual(result["stop_reason"], "node_budget")
            result = search(self.env, state, max_nodes=100, max_depth=0, **extras)
            self.assertEqual(result["stop_reason"], "depth_budget")
            self.assertIsNone(result["exact_distance"])

    def test_time_budget_and_indivisible_call_overrun(self):
        state = self.scrambled(["U", "R", "F"])
        result = beam_search(self.env, state, self.scorer, max_seconds=0)
        self.assertEqual(result["expanded_nodes"], 0)
        self.assertEqual(result["stop_reason"], "time_budget")
        def slow(*args):
            time.sleep(0.03)
            return self.scorer(*args)
        result = beam_search(self.env, state, slow, max_seconds=0.02,
                             score_batch_size=2)
        self.assertTrue(result["time_budget_overrun"])
        self.assertEqual(result["stop_reason"], "time_budget")
        self.assertEqual(result["scorer_calls"], 1)

    def test_exact_bfs_matches_independent_shallow_enumeration(self):
        # Independent forward BFS labels every state through depth 2 exactly.
        distances = {self.env.serialize(self.env.goal): 0}
        queue = deque([self.env.goal])
        while queue:
            state = queue.popleft()
            depth = distances[self.env.serialize(state)]
            if depth == 2:
                continue
            for action in self.env.actions:
                successor = self.env.step(state, action)
                key = self.env.serialize(successor)
                if key not in distances:
                    distances[key] = depth + 1
                    queue.append(successor)
        # A deterministic subset covers all first moves and nontrivial depth 2.
        items = list(distances.items())
        selected = items[:13] + items[13::7]
        for key, optimum in selected:
            state = self.env.deserialize(key)
            result = bidirectional_bfs(self.env, state, max_nodes=1000, max_depth=2)
            self.assertTrue(result["verified"])
            self.assertEqual(result["exact_distance"], optimum)
            self.assertTrue(result["optimal"])

    def test_arbitrary_goal_and_forward_counter_delta(self):
        goal = self.scrambled(["F", "R"])
        initial = self.env.step(goal, "U")
        result = bidirectional_bfs(self.env, initial, goal=goal)
        self.assertEqual(result["exact_distance"], 1)
        self.assertTrue(self.env.verify_trace(initial, result["actions"], goal))
        class Counted:
            metrics = {"model_forward_calls": 8}
            def __call__(inner, *args):
                inner.metrics["model_forward_calls"] += 2
                return self.scorer(*args)
        counted = Counted()
        result = beam_search(self.env, self.scrambled(["U", "R"]), counted,
                             beam_width=30, max_nodes=100, max_depth=3)
        self.assertEqual(result["model_forward_calls"], result["scorer_calls"] * 2)


if __name__ == "__main__":
    unittest.main()
