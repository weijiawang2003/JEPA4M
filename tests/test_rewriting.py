import json

import pytest

from jepa4math.rewriting import (RewriteAction, RewriteEnv, RewriteState, ast_from_json,
    evaluate_ast, node_count, renamed_semantic_key, semantic_polynomial, structural_heuristic)


@pytest.mark.parametrize("expr", [
    ("Mul", "a", ("Add", "b", "c")),
    ("Mul", ("Add", "a", "b"), ("Add", "c", "d")),
    ("Add", ("Add", "x", "0"), ("Mul", "1", "y")),
    ("Add", ("Mul", "x", "y"), ("Mul", "x", "z")),
])
def test_every_rule_preserves_semantics_and_has_exact_inverse(expr):
    env = RewriteEnv(31)
    state = RewriteState(expr, expr)
    actions = env.legal_actions(state)
    assert actions
    for action in actions:
        updated = env.step(state, action)
        assert semantic_polynomial(updated.current) == semantic_polynomial(expr)
        for values in [dict(a=2,b=-3,c=4,d=7,x=11,y=-2,z=5), dict(a=0,b=1,c=-1,d=0,x=-5,y=3,z=7)]:
            assert evaluate_ast(updated.current, values) == evaluate_ast(expr, values)
        assert action.inverse() in env.legal_actions(updated)
        assert env.step(updated, action.inverse()) == state


def test_syntax_is_never_canonicalized():
    current = ("Add", "x", "y")
    goal = ("Add", "y", "x")
    env = RewriteEnv()
    state = env.make_state(current, goal)
    assert semantic_polynomial(current) == semantic_polynomial(goal)
    assert not env.is_goal(state)
    assert state.current != state.goal
    assert structural_heuristic(current, goal) > 0
    assert env.deserialize(env.serialize(state)) == state
    assert env.verify_trace(state, [RewriteAction("add_comm", 1)])


def test_factoring_is_syntactic_and_locations_are_exact():
    env = RewriteEnv()
    expr = ("Add", ("Mul", "a", "b"), ("Mul", "b", "a"))
    state = RewriteState(expr, expr)
    with pytest.raises(ValueError, match="does not apply"):
        env.step(state, RewriteAction("distribute_left", -1))
    updated = env.step(state, RewriteAction("mul_comm", 1, (1,)))
    factored = env.step(updated, RewriteAction("distribute_left", -1))
    assert factored.current == ("Mul", "a", ("Add", "b", "b"))


def test_polynomial_and_variable_renaming_audit():
    expanded = ("Add", ("Mul", "x", "y"), ("Mul", "x", "z"))
    renamed = ("Mul", "c", ("Add", "a", "b"))
    assert renamed_semantic_key(expanded) == renamed_semantic_key(renamed)
    assert renamed_semantic_key(("Add", "x", "x")) != renamed_semantic_key(("Add", "x", "y"))
    assert semantic_polynomial(("Mul", "0", expanded)) == ()
    assert semantic_polynomial(("Add", "1", "1")) == (((), 2),)
    assert semantic_polynomial(("Mul", "x", "x")) == ((("x", "x"), 1),)


def test_finite_bound_and_invalid_actions():
    env = RewriteEnv(3)
    state = env.make_state(("Add", "x", "y"), ("Add", "y", "x"))
    assert env.legal_actions(state) == (RewriteAction("add_comm", 1),)
    with pytest.raises(ValueError, match="max_nodes"):
        env.step(state, RewriteAction("add_zero_right", 1))
    for action in [RewriteAction("bad",1), RewriteAction("add_comm",-1), RewriteAction("add_comm",True),
                   RewriteAction("add_comm",1,(2,)), RewriteAction("add_comm",1,(0,0))]:
        with pytest.raises(ValueError):
            env.step(state, action)
    for bad in [["Add","x","y"], ("Div","x","y"), ("Add","x"), "2", ""]:
        with pytest.raises(ValueError):
            env.make_state(bad,"x")


def test_serialization_rejects_bad_input_and_round_trips_actions():
    env = RewriteEnv()
    action = RewriteAction("mul_assoc", -1, (0,1))
    assert RewriteAction.from_dict(json.loads(json.dumps(action.to_dict()))) == action
    for payload in [b"null", b"{}", b'["x"]', b'[["Div","x","y"],"x"]', b"\xff", "x"]:
        with pytest.raises(ValueError):
            env.deserialize(payload)
    assert ast_from_json(["Add","x","y"]) == ("Add","x","y")
    assert node_count(("Add","x","y")) == 3


def test_benchmark_search_proves_one_step_and_accepts_zero_budget_goal():
    # Import the executable protocol without running its command-line entry.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/rewrite_benchmark.py"
    spec = importlib.util.spec_from_file_location("rewrite_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    env = RewriteEnv(15)
    for method in ("exact_bfs", "structural_beam"):
        solved = env.make_state("x", "x")
        result = benchmark.search(env, solved, method, node_budget=0, wall_seconds=0, beam_width=4, max_depth=2)
        assert result["solved"] and result["solution_length"] == 0 and result["expansions"] == 0
        initial = env.make_state(("Mul", "a", ("Add", "b", "c")),
                                 ("Add", ("Mul", "a", "b"), ("Mul", "a", "c")))
        result = benchmark.search(env, initial, method, node_budget=2, wall_seconds=2, beam_width=4, max_depth=2)
        assert result["solved"] and result["solution_length"] == 1
        assert env.verify_trace(initial, [RewriteAction.from_dict(a) for a in result["path"]])
        assert result["exact_environment_calls"] == sum(result[k] for k in (
            "exact_transition_calls", "action_enumeration_calls", "goal_test_calls", "verification_transition_calls"))
        if method == "exact_bfs":
            assert result["exact_shortest_distance"] == 1


def test_learning_tokens_preserve_literal_names_and_exact_paths():
    from jepa4math.rewrite_data import AST_TOKEN_VOCAB, encode_action, tokenize_ast
    left = tokenize_ast(("Add", "x0", "x1"))
    right = tokenize_ast(("Add", "x1", "x0"))
    assert left != right
    assert left[1] == right[2] == AST_TOKEN_VOCAB.index("x0")
    assert left[2] == right[1] == AST_TOKEN_VOCAB.index("x1")
    assert encode_action(RewriteAction("add_assoc", -1, (0,1)))["action_path"] == [0,1,2]
    assert encode_action(RewriteAction("add_assoc", -1))["action_direction"] == 0


def test_learning_data_is_deterministic_reconstructable_and_family_separated():
    from jepa4math.rewrite_data import generate_rewrite_dataset, encode_transition
    config = {"families":{"train":2,"validation":2,"development":2},
              "episodes_per_family":2,"walk_steps":4}
    first = generate_rewrite_dataset(config)
    second = generate_rewrite_dataset(config)
    assert first["manifest"]["dataset_sha256"] == second["manifest"]["dataset_sha256"]
    assert first["manifest"]["forward_transition_counts"] == {"train":16,"validation":16,"development":16}
    for key,value in first["manifest"]["leakage_audit"].items():
        if key.startswith("cross_split_"):
            assert value == 0
    for record in first["transitions"]:
        env = RewriteEnv(47 if record["split"] == "development" else 31)
        current = env.make_state(record["current_ast"], record["goal_ast"])
        action = RewriteAction.from_dict(record["action"])
        assert env.step(current, action).current == record["next_ast"]
        encoded = encode_transition(record)
        assert len(encoded["state_tokens"]) == node_count(record["current_ast"])
        assert not {"episode_id","family_id","semantic_group_id","seed"}.intersection(encoded)
        assert encoded["next_cost"]-encoded["current_cost"] == (1 if record["trajectory_direction"]=="forward" else -1)
