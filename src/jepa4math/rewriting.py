"""Exact syntax-preserving equality rewriting over a commutative semiring.

The search/model state is an ordered AST and a goal AST. Polynomial normal
forms are audit metadata only: they NEVER replace the state. This module uses
elementary algebraic identities, not a computer algebra system's simplifier.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import permutations
import json
import re
from typing import Iterator


AST = str | tuple[str, "AST", "AST"]
Path = tuple[int, ...]
_VARIABLE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_RULES = ("add_assoc", "mul_assoc", "add_comm", "mul_comm",
          "distribute_left", "distribute_right", "add_zero_left", "add_zero_right",
          "mul_one_left", "mul_one_right")


def validate_ast(expr: AST) -> None:
    if isinstance(expr, str):
        if expr in ("0", "1") or _VARIABLE.fullmatch(expr):
            return
        raise ValueError(f"Invalid AST leaf: {expr!r}")
    if not isinstance(expr, tuple) or len(expr) != 3 or expr[0] not in ("Add", "Mul"):
        raise ValueError("AST must be a variable/0/1 or an immutable (Add|Mul, left, right) tuple")
    validate_ast(expr[1])
    validate_ast(expr[2])


@lru_cache(maxsize=100_000)
def node_count(expr: AST) -> int:
    return 1 if isinstance(expr, str) else 1 + node_count(expr[1]) + node_count(expr[2])


def ast_from_json(value) -> AST:
    if isinstance(value, str):
        expr = value
    elif isinstance(value, list) and len(value) == 3:
        expr = (value[0], ast_from_json(value[1]), ast_from_json(value[2]))
    else:
        raise ValueError("Invalid JSON AST")
    validate_ast(expr)
    return expr


def ast_paths(expr: AST, path: Path = ()) -> Iterator[tuple[Path, AST]]:
    yield path, expr
    if isinstance(expr, tuple):
        yield from ast_paths(expr[1], path + (0,))
        yield from ast_paths(expr[2], path + (1,))


def _replace(expr: AST, path: Path, replacement: AST) -> AST:
    if not path:
        return replacement
    child = path[0] + 1
    return (expr[0], _replace(expr[1], path[1:], replacement), expr[2]) if child == 1 else (expr[0], expr[1], _replace(expr[2], path[1:], replacement))


@lru_cache(maxsize=100_000)
def semantic_polynomial(expr: AST) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Audit-only exact integer polynomial: sorted (monomial, coefficient) pairs.

    A monomial is a sorted tuple of variable names, preserving multiplicity.
    This function is separate from rule application and from state encoding.
    """
    validate_ast(expr)
    if isinstance(expr, str):
        return () if expr == "0" else ((((), 1),) if expr == "1" else (((expr,), 1),))
    left, right = dict(semantic_polynomial(expr[1])), dict(semantic_polynomial(expr[2]))
    result = Counter(left)
    if expr[0] == "Add":
        result.update(right)
    else:
        result = Counter()
        for lm, lc in left.items():
            for rm, rc in right.items():
                result[tuple(sorted(lm + rm))] += lc * rc
    return tuple(sorted((monomial, coefficient) for monomial, coefficient in result.items() if coefficient))


def evaluate_ast(expr: AST, values: dict[str, int]) -> int:
    """Independent recursive integer evaluation, without polynomial conversion."""
    if isinstance(expr, str):
        return int(expr) if expr in ("0", "1") else values[expr]
    left, right = evaluate_ast(expr[1], values), evaluate_ast(expr[2], values)
    return left + right if expr[0] == "Add" else left * right


def renamed_semantic_key(expr: AST) -> str:
    """Exact variable-renaming invariant semantic grouping for <= 6 variables.

    Enumerating all name bijections avoids the common, invalid assumption that
    sorting original variable names produces a renaming-invariant family key.
    This audit can merge structurally different goals, intentionally protecting
    the split from equivalent expressions and renamed copies of a template.
    """
    polynomial = semantic_polynomial(expr)
    variables = sorted({v for monomial, _ in polynomial for v in monomial})
    if len(variables) > 6:
        raise ValueError("Exact variable-renaming audit supports at most six variables")
    candidates = []
    for assignment in permutations(range(len(variables))):
        rename = dict(zip(variables, assignment))
        candidates.append(tuple(sorted((tuple(sorted(rename[v] for v in m)), c) for m, c in polynomial)))
    canonical = min(candidates) if candidates else ()
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RewriteAction:
    rule_id: str
    direction: int
    node_path: Path = ()

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "direction": self.direction, "node_path": list(self.node_path)}

    @classmethod
    def from_dict(cls, record: dict) -> "RewriteAction":
        if set(record) != {"rule_id", "direction", "node_path"} or not isinstance(record["node_path"], list):
            raise ValueError("Invalid serialized rewrite action")
        return cls(record["rule_id"], record["direction"], tuple(record["node_path"]))

    def inverse(self) -> "RewriteAction":
        return RewriteAction(self.rule_id, self.direction if self.rule_id.endswith("_comm") else -self.direction, self.node_path)


@dataclass(frozen=True)
class RewriteState:
    current: AST
    goal: AST


def _binary(expr: AST, op: str) -> bool:
    return isinstance(expr, tuple) and expr[0] == op


def _local_rewrite(expr: AST, rule: str, direction: int) -> AST | None:
    if rule in ("add_comm", "mul_comm"):
        op = "Add" if rule == "add_comm" else "Mul"
        if direction == 1 and _binary(expr, op):
            return (op, expr[2], expr[1])
    elif rule in ("add_assoc", "mul_assoc"):
        op = "Add" if rule == "add_assoc" else "Mul"
        if _binary(expr, op):
            if direction == 1 and _binary(expr[1], op):
                return (op, expr[1][1], (op, expr[1][2], expr[2]))
            if direction == -1 and _binary(expr[2], op):
                return (op, (op, expr[1], expr[2][1]), expr[2][2])
    elif rule == "distribute_left":
        if direction == 1 and _binary(expr, "Mul") and _binary(expr[2], "Add"):
            return ("Add", ("Mul", expr[1], expr[2][1]), ("Mul", expr[1], expr[2][2]))
        if direction == -1 and _binary(expr, "Add") and _binary(expr[1], "Mul") and _binary(expr[2], "Mul") and expr[1][1] == expr[2][1]:
            return ("Mul", expr[1][1], ("Add", expr[1][2], expr[2][2]))
    elif rule == "distribute_right":
        if direction == 1 and _binary(expr, "Mul") and _binary(expr[1], "Add"):
            return ("Add", ("Mul", expr[1][1], expr[2]), ("Mul", expr[1][2], expr[2]))
        if direction == -1 and _binary(expr, "Add") and _binary(expr[1], "Mul") and _binary(expr[2], "Mul") and expr[1][2] == expr[2][2]:
            return ("Mul", ("Add", expr[1][1], expr[2][1]), expr[1][2])
    elif rule in ("add_zero_left", "add_zero_right", "mul_one_left", "mul_one_right"):
        op, identity = ("Add", "0") if rule.startswith("add_") else ("Mul", "1")
        side = 1 if rule.endswith("left") else 2
        if direction == 1:
            return (op, identity, expr) if side == 1 else (op, expr, identity)
        if _binary(expr, op) and expr[side] == identity:
            return expr[3 - side]
    return None


class RewriteEnv:
    """Bounded AST syntax graph with exact local rules and fixed target syntax.

    Direction +1 expands distribution/identities or right-associates; direction
    -1 is its inverse. Commutativity is self-inverse and uses only +1. Syntactic
    no-ops are omitted. The node bound makes every fixed-vocabulary state space
    finite. New variable names are never introduced by any action.
    """
    def __init__(self, max_nodes: int = 63, allow_identity_expansion: bool = True):
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        self.max_nodes = max_nodes
        self.allow_identity_expansion = allow_identity_expansion

    def validate(self, state: RewriteState) -> None:
        if not isinstance(state, RewriteState):
            raise ValueError("Expected RewriteState(current_ast, goal_ast)")
        for expr in (state.current, state.goal):
            validate_ast(expr)
            if node_count(expr) > self.max_nodes:
                raise ValueError("AST exceeds max_nodes")

    def make_state(self, current: AST, goal: AST) -> RewriteState:
        state = RewriteState(current, goal)
        self.validate(state)
        return state

    def legal_actions(self, state: RewriteState) -> tuple[RewriteAction, ...]:
        self.validate(state)
        count = node_count(state.current)
        actions = []
        for path, expr in ast_paths(state.current):
            for rule in _RULES:
                directions = (1,) if rule.endswith("_comm") else (1, -1)
                for direction in directions:
                    if not self.allow_identity_expansion and direction == 1 and ("zero" in rule or "one" in rule):
                        continue
                    replacement = _local_rewrite(expr, rule, direction)
                    if replacement is not None and replacement != expr and count - node_count(expr) + node_count(replacement) <= self.max_nodes:
                        actions.append(RewriteAction(rule, direction, path))
        return tuple(actions)

    def step(self, state: RewriteState, action: RewriteAction) -> RewriteState:
        self.validate(state)
        if not isinstance(action, RewriteAction) or action.rule_id not in _RULES or type(action.direction) is not int or action.direction not in (-1, 1):
            raise ValueError("Invalid rewrite action")
        if not isinstance(action.node_path, tuple) or any(type(i) is not int or i not in (0, 1) for i in action.node_path):
            raise ValueError("node_path must be a tuple of child indices 0/1")
        expr = state.current
        for index in action.node_path:
            if not isinstance(expr, tuple):
                raise ValueError("Action path descends below a leaf")
            expr = expr[index + 1]
        if not self.allow_identity_expansion and action.direction == 1 and ("zero" in action.rule_id or "one" in action.rule_id):
            raise ValueError("Identity expansion is disabled")
        replacement = _local_rewrite(expr, action.rule_id, action.direction)
        if replacement is None or replacement == expr:
            raise ValueError("Rule does not apply at this node or is a syntactic no-op")
        updated = RewriteState(_replace(state.current, action.node_path, replacement), state.goal)
        self.validate(updated)
        return updated

    def is_goal(self, state: RewriteState) -> bool:
        self.validate(state)
        return state.current == state.goal

    def serialize(self, state: RewriteState) -> bytes:
        self.validate(state)
        return json.dumps([state.current, state.goal], separators=(",", ":")).encode()

    def deserialize(self, payload: bytes) -> RewriteState:
        if not isinstance(payload, bytes):
            raise ValueError("Serialized state must be bytes")
        try:
            record = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid serialized state") from exc
        if not isinstance(record, list) or len(record) != 2:
            raise ValueError("Expected serialized [current, goal]")
        return self.make_state(ast_from_json(record[0]), ast_from_json(record[1]))

    def verify_trace(self, initial_state: RewriteState, actions, goal: AST | None = None) -> bool:
        self.validate(initial_state)
        target = initial_state.goal if goal is None else goal
        validate_ast(target)
        state = initial_state
        for action in actions:
            state = self.step(state, action)
        return state.current == target


def structural_heuristic(current: AST, goal: AST) -> float:
    """Inadmissible positional tree mismatch, retaining ordered syntax."""
    if current == goal:
        return 0.0
    if isinstance(current, str) or isinstance(goal, str):
        return float(max(node_count(current), node_count(goal)))
    return float(current[0] != goal[0]) + structural_heuristic(current[1], goal[1]) + structural_heuristic(current[2], goal[2])
