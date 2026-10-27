"""Exact, fixed-frame 2x2 and 3x3 cubes in the quarter-turn metric.

The implementation is independent of any learned model or external cube solver.
Permutations are derived with integer three-dimensional geometry: x points right,
y up, and z forwards. Clockwise is viewed from outside the turned face. The
facelet order is U, R, F, D, L, B, each in row-major order viewed from outside.
There are 12 actions; a half-turn costs TWO actions. Whole-cube rotations are
not quotiented out and the target has an exact, fixed color orientation.

Public state operations check physical cubie constraints, including corner
chirality/twist, edge flip, and (for 3x3) corner/edge permutation parity. These
are the standard complete reachability constraints for these unbandaged cubes.
The 2x2 has no edge parity constraint; all eight corners remain represented.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterable

import numpy as np


FACES = ("U", "R", "F", "D", "L", "B")
ACTIONS = tuple(a for f in ("U", "D", "L", "R", "F", "B") for a in (f, f + "'"))
_NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}
_RIGHT_UP = {
    "U": ((1, 0, 0), (0, 0, -1)),
    "R": ((0, 0, -1), (0, 1, 0)),
    "F": ((1, 0, 0), (0, 1, 0)),
    "D": ((1, 0, 0), (0, 0, 1)),
    "L": ((0, 0, 1), (0, 1, 0)),
    "B": ((-1, 0, 0), (0, 1, 0)),
}
# Every corner tuple has the same handedness. Cyclic rotations give its three
# possible orientations; reflected tuples describe a physically invalid cubie.
_CORNERS = (
    ("U", "R", "F"), ("U", "F", "L"), ("U", "L", "B"), ("U", "B", "R"),
    ("D", "F", "R"), ("D", "L", "F"), ("D", "B", "L"), ("D", "R", "B"),
)
_EDGES = (
    ("U", "R"), ("U", "F"), ("U", "L"), ("U", "B"),
    ("D", "R"), ("D", "F"), ("D", "L"), ("D", "B"),
    ("F", "R"), ("F", "L"), ("B", "L"), ("B", "R"),
)


def _quarter_rotate(vector: np.ndarray, normal: np.ndarray, direction: int) -> np.ndarray:
    """Integer Rodrigues formula for +/- 90 degrees about a unit axis."""
    return direction * np.cross(normal, vector) + normal * np.dot(normal, vector)


def _parity(permutation: np.ndarray) -> int:
    return sum(int(permutation[i] > permutation[j]) for i, j in combinations(range(len(permutation)), 2)) % 2


class CubeEnv:
    """Exact sticker environment. State arrays must have dtype ``np.uint8``.

    ``scramble`` returns a state and its reproducible generating action history.
    Its requested length is a random-walk length, never a shortest-distance label.
    Invalid states/actions raise ValueError (including in ``verify_trace``).
    """

    actions = ACTIONS
    action_names = ACTIONS
    num_actions = len(ACTIONS)

    def __init__(self, size: int = 3):
        if isinstance(size, bool) or size not in (2, 3):
            raise ValueError("Only cube sizes 2 and 3 are supported")
        self.size = int(size)
        self.state_size = 6 * self.size * self.size
        self.num_stickers = self.state_size
        self._goal = np.repeat(np.arange(6, dtype=np.uint8), self.size * self.size)
        self._goal.setflags(write=False)
        coordinates = []
        normals = []
        extent = self.size - 1
        for face in FACES:
            normal = np.asarray(_NORMALS[face])
            right, up = (np.asarray(v) for v in _RIGHT_UP[face])
            for row in range(self.size):
                for col in range(self.size):
                    coordinates.append(normal * extent + right * (2 * col - extent) + up * (extent - 2 * row))
                    normals.append(normal)
        self._coordinates = np.asarray(coordinates)
        self._normals = np.asarray(normals)
        lookup = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(coordinates, normals))}
        permutations = []
        for action in self.actions:
            axis = np.asarray(_NORMALS[action[0]])
            direction = 1 if action.endswith("'") else -1
            permutation = np.arange(self.state_size)
            for source, (position, normal) in enumerate(zip(coordinates, normals)):
                if np.dot(position, axis) == extent:
                    destination = lookup[(tuple(_quarter_rotate(position, axis, direction)),
                                          tuple(_quarter_rotate(normal, axis, direction)))]
                    permutation[destination] = source
            permutations.append(permutation)
        self.permutations = np.asarray(permutations)
        self.permutations.setflags(write=False)
        self.corner_indices = self._cubie_indices(_CORNERS, lookup)
        self._corner_identity, self._corner_orientation = self._cubie_lookup(_CORNERS)
        if self.size == 3:
            self.edge_indices = self._cubie_indices(_EDGES, lookup)
            self._edge_identity, self._edge_orientation = self._cubie_lookup(_EDGES)
            self._center_indices = np.arange(6) * 9 + 4
        self._inverse_indices = np.arange(self.num_actions) ^ 1

    def _cubie_indices(self, cubies, lookup) -> np.ndarray:
        result = []
        for faces in cubies:
            position = tuple(np.sum([_NORMALS[f] for f in faces], axis=0) * (self.size - 1))
            result.append([lookup[(position, _NORMALS[f])] for f in faces])
        array = np.asarray(result)
        array.setflags(write=False)
        return array

    @staticmethod
    def _cubie_lookup(cubies) -> tuple[np.ndarray, np.ndarray]:
        width = len(cubies[0])
        identity = np.full(6 ** width, -1, dtype=np.int16)
        orientation = np.full(6 ** width, -1, dtype=np.int16)
        for index, faces in enumerate(cubies):
            colors = np.asarray([FACES.index(face) for face in faces])
            for rotation in range(width):
                code = int(np.dot(np.roll(colors, rotation), 6 ** np.arange(width - 1, -1, -1)))
                identity[code] = index
                orientation[code] = rotation
        return identity, orientation

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @property
    def solved_state(self) -> np.ndarray:
        return self.goal

    def validate(self, state: np.ndarray) -> None:
        """Raise ValueError unless the state is physically reachable from goal."""
        if not isinstance(state, np.ndarray) or state.dtype != np.uint8 or state.shape != (self.state_size,):
            raise ValueError(f"State must be a uint8 ndarray of shape ({self.state_size},)")
        if np.any(state >= 6):
            raise ValueError("Sticker colors must be integers in [0, 5]")
        if not np.array_equal(np.bincount(state, minlength=6), np.full(6, self.size ** 2)):
            raise ValueError("Every color must occur size**2 times")
        corner_codes = state[self.corner_indices].astype(np.int16) @ np.array([36, 6, 1])
        corner_permutation = self._corner_identity[corner_codes]
        if not np.array_equal(np.sort(corner_permutation), np.arange(8)):
            raise ValueError("Invalid, duplicated, or reflected corner cubie")
        if int(self._corner_orientation[corner_codes].sum()) % 3:
            raise ValueError("Corner twists must sum to zero modulo three")
        if self.size == 3:
            if not np.array_equal(state[self._center_indices], np.arange(6)):
                raise ValueError("Centers must match the fixed goal orientation")
            edge_codes = state[self.edge_indices].astype(np.int16) @ np.array([6, 1])
            edge_permutation = self._edge_identity[edge_codes]
            if not np.array_equal(np.sort(edge_permutation), np.arange(12)):
                raise ValueError("Invalid or duplicated edge cubie")
            if int(self._edge_orientation[edge_codes].sum()) % 2:
                raise ValueError("Edge flips must sum to zero modulo two")
            if _parity(corner_permutation) != _parity(edge_permutation):
                raise ValueError("Corner and edge permutation parity must agree")

    def action_index(self, action: str | int) -> int:
        if isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_)):
            if 0 <= action < self.num_actions:
                return int(action)
        elif isinstance(action, str) and action in self.actions:
            return self.actions.index(action)
        raise ValueError(f"Invalid quarter-turn action: {action!r}")

    def inverse_action(self, action: str | int) -> str:
        return self.actions[self.action_index(action) ^ 1]

    def inverse_actions(self, actions: Iterable[str | int]) -> tuple[str, ...]:
        return tuple(self.inverse_action(action) for action in reversed(tuple(actions)))

    def legal_actions(self, state: np.ndarray) -> tuple[str, ...]:
        self.validate(state)
        return self.actions

    def step(self, state: np.ndarray, action: str | int) -> np.ndarray:
        self.validate(state)
        return state[self.permutations[self.action_index(action)]]

    def successors(self, state: np.ndarray) -> np.ndarray:
        """All exact successors in action order, with shape (12, state_size)."""
        self.validate(state)
        return state[self.permutations]

    def successor_batch(self, states: np.ndarray) -> np.ndarray:
        """All successors of a batch, shape (batch, 12, state_size)."""
        if not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != self.state_size or states.dtype != np.uint8:
            raise ValueError(f"Batch must have uint8 shape (batch, {self.state_size})")
        for state in states:
            self.validate(state)
        return states[:, self.permutations]

    def is_goal(self, state: np.ndarray) -> bool:
        self.validate(state)
        return bool(np.array_equal(state, self._goal))

    def serialize(self, state: np.ndarray) -> bytes:
        """Canonical raw sticker bytes; cube size is external metadata."""
        self.validate(state)
        return state.tobytes()

    def deserialize(self, data: bytes) -> np.ndarray:
        if not isinstance(data, bytes):
            raise ValueError("Serialized state must be bytes")
        state = np.frombuffer(data, dtype=np.uint8).copy()
        self.validate(state)
        return state

    def verify_trace(self, initial_state: np.ndarray, actions: Iterable[str | int], goal: np.ndarray | None = None) -> bool:
        """Independently replay symbolic actions, then compare exact goal stickers."""
        self.validate(initial_state)
        target = self._goal if goal is None else goal
        self.validate(target)
        state = initial_state.copy()
        for action in actions:
            state = self.step(state, action)
        return bool(np.array_equal(state, target))

    def scramble(self, length: int, seed: int | np.random.Generator = 0) -> tuple[np.ndarray, tuple[str, ...]]:
        """Seeded random walk excluding only immediate inverse moves.

        A returned history can contain detours or cycles. ``length`` records
        generator effort and is merely an upper bound on optimal QTM distance.
        """
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 0:
            raise ValueError("Scramble length must be a nonnegative integer")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        state = self.goal
        history = []
        previous = None
        for _ in range(length):
            candidates = np.arange(self.num_actions)
            if previous is not None:
                candidates = candidates[candidates != (previous ^ 1)]
            index = int(rng.choice(candidates))
            state = state[self.permutations[index]]
            history.append(self.actions[index])
            previous = index
        return state, tuple(history)

    def exact_bfs(self, max_depth: int, *, max_states: int = 1_000_000) -> dict[bytes, int]:
        """Exact shortest QTM distances for every state through ``max_depth``.

        Unlisted states have unknown distance greater than the completed radius;
        they must not receive the radius as a distance label. The fixed-frame
        2x2 graph is larger than versions that identify global cube rotations.
        Raises before returning a partial table if the memory guard is exceeded.
        """
        if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if isinstance(max_states, bool) or not isinstance(max_states, (int, np.integer)) or max_states < 1:
            raise ValueError("max_states must be a positive integer")
        root = self._goal.tobytes()
        distances = {root: 0}
        queue = deque([root])
        while queue:
            key = queue.popleft()
            depth = distances[key]
            if depth == max_depth:
                continue
            state = np.frombuffer(key, dtype=np.uint8)
            for successor in state[self.permutations]:
                new_key = successor.tobytes()
                if new_key not in distances:
                    if len(distances) >= max_states:
                        raise RuntimeError("Exact BFS max_states exceeded; no partial distance table returned")
                    distances[new_key] = depth + 1
                    queue.append(new_key)
        return distances
