"""Environment checks independent of any training or heuristic implementation."""

from collections import Counter

import numpy as np
import pytest

from jepa4math.cube import CubeEnv, FACES


@pytest.fixture(params=[2, 3])
def env(request):
    return CubeEnv(request.param)


def test_move_permutations_and_group_identities(env):
    reference = np.arange(env.state_size)
    for index, action in enumerate(env.actions):
        permutation = env.permutations[index]
        np.testing.assert_array_equal(np.sort(permutation), reference)
        state = env.goal
        for _ in range(4):
            state = env.step(state, action)
        np.testing.assert_array_equal(state, env.goal)
        np.testing.assert_array_equal(env.step(env.step(env.goal, action), env.inverse_action(action)), env.goal)
        assert not env.is_goal(env.step(env.goal, action))


def test_facelet_orientation_against_manual_u_and_f_moves(env):
    # Independent face adjacency oracle. Clockwise U carries F's upper row to
    # L's upper row. Clockwise F carries U's lower row to R's left column.
    size = env.size
    u = env.step(env.goal, "U").reshape(6, size, size)
    assert np.all(u[FACES.index("L"), 0] == FACES.index("F"))
    assert np.all(u[FACES.index("F"), 0] == FACES.index("R"))
    assert np.all(u[FACES.index("R"), 0] == FACES.index("B"))
    assert np.all(u[FACES.index("B"), 0] == FACES.index("L"))
    f = env.step(env.goal, "F").reshape(6, size, size)
    assert np.all(f[FACES.index("R"), :, 0] == FACES.index("U"))
    assert np.all(f[FACES.index("D"), 0] == FACES.index("R"))
    assert np.all(f[FACES.index("L"), :, -1] == FACES.index("D"))
    assert np.all(f[FACES.index("U"), -1] == FACES.index("L"))


def test_long_scramble_is_physical_and_replays(env):
    state, actions = env.scramble(1000, seed=123)
    state_again, history_again = env.scramble(1000, seed=123)
    np.testing.assert_array_equal(state, state_again)
    assert actions == history_again
    assert len(actions) == 1000
    assert all(right != env.inverse_action(left) for left, right in zip(actions, actions[1:]))
    env.validate(state)
    assert env.verify_trace(env.goal, actions, goal=state)
    assert env.verify_trace(state, env.inverse_actions(actions))
    assert not env.verify_trace(env.step(env.goal, "R"), [])
    # Six repetitions of the commutator restore all stickers, not just counts.
    assert env.verify_trace(env.goal, ("R", "U", "R'", "U'") * 6)


def test_serialization_ownership_and_successor_batch(env):
    state, _ = env.scramble(25, seed=4)
    restored = env.deserialize(env.serialize(state))
    np.testing.assert_array_equal(restored, state)
    restored[:] = 0
    assert not np.array_equal(restored, state)
    batch = np.stack([env.goal, state])
    successors = env.successor_batch(batch)
    assert successors.shape == (2, 12, env.state_size)
    for row in range(2):
        np.testing.assert_array_equal(successors[row], env.successors(batch[row]))
        for index, action in enumerate(env.actions):
            np.testing.assert_array_equal(successors[row, index], env.step(batch[row], action))
    original = env.goal
    original[:] = 0
    assert env.is_goal(env.goal)


def test_complete_shallow_bfs_counts(env):
    # Direct counting: depth 1 gives twelve quarter turns. At depth 2 the
    # 12*12 words reduce by 12 inverse cancellations, 6 duplicate half-turn
    # words, and 12 duplicates from commuting opposite faces: 114 states.
    distances = env.exact_bfs(2)
    assert Counter(distances.values()) == {0: 1, 1: 12, 2: 114}
    for key, depth in distances.items():
        state = env.deserialize(key)
        if depth:
            assert any(distances.get(successor.tobytes()) == depth - 1 for successor in env.successors(state))
    with pytest.raises(RuntimeError, match="max_states"):
        env.exact_bfs(2, max_states=20)


def test_reject_single_corner_twist_and_reflection(env):
    state = env.goal
    indices = env.corner_indices[0]
    state[indices] = np.roll(state[indices], 1)
    with pytest.raises(ValueError, match="twists"):
        env.validate(state)
    state = env.goal
    state[indices[1]], state[indices[2]] = state[indices[2]], state[indices[1]]
    with pytest.raises(ValueError, match="reflected"):
        env.validate(state)


def test_reject_nonphysical_edge_flip_and_parity():
    env = CubeEnv(3)
    state = env.goal
    edge = env.edge_indices[0]
    state[edge] = state[edge[::-1]]
    with pytest.raises(ValueError, match="flips"):
        env.validate(state)
    state = env.goal
    edge_a, edge_b = env.edge_indices[:2]
    state[edge_a], state[edge_b] = state[edge_b].copy(), state[edge_a].copy()
    with pytest.raises(ValueError, match="parity"):
        env.validate(state)
    state = env.goal
    state[4], state[13] = state[13], state[4]
    with pytest.raises(ValueError, match="Centers"):
        env.validate(state)


def test_2x2_does_not_invent_an_edge_parity_restriction():
    env = CubeEnv(2)
    state = env.goal
    a, b = env.corner_indices[:2]
    state[a], state[b] = state[b].copy(), state[a].copy()
    env.validate(state)


def test_reject_invalid_inputs(env):
    for bad in [env.goal.astype(np.int64), env.goal.tolist(), env.goal[:-1], np.zeros(env.state_size, dtype=np.uint8),
                np.full(env.state_size, 255, dtype=np.uint8)]:
        with pytest.raises(ValueError):
            env.step(bad, "U")
    for action in ["U2", "X", 12, -1, True, 0.5]:
        with pytest.raises(ValueError):
            env.step(env.goal, action)
    for data in [b"bad", env.goal.tobytes()[:-1], env.goal.tolist()]:
        with pytest.raises(ValueError):
            env.deserialize(data)
    for length in [-1, 1.5, True]:
        with pytest.raises(ValueError):
            env.scramble(length)
    with pytest.raises(ValueError):
        env.successor_batch(np.zeros((1, env.state_size), dtype=np.int64))
    assert env.verify_trace(env.goal, [])
    zero, history = env.scramble(0)
    np.testing.assert_array_equal(zero, env.goal)
    assert history == ()
