"""Reconstructable, family-separated data for learned syntax rewriting.

Model tokenization preserves literal x0..x5 names and ordered prefix syntax.
Semantic canonicalization is exclusively an independent split audit. Neither
walk length nor its reverse label is asserted to be a shortest distance.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .rewriting import (RewriteAction, RewriteEnv, RewriteState, ast_from_json,
                       ast_paths, node_count, renamed_semantic_key, semantic_polynomial)


AST_TOKEN_VOCAB = ("<PAD>", "Add", "Mul", "0", "1", "x0", "x1", "x2", "x3", "x4", "x5")
ACTION_RULE_VOCAB = ("add_assoc", "mul_assoc", "add_comm", "mul_comm", "distribute_left", "distribute_right",
                     "add_zero_left", "add_zero_right", "mul_one_left", "mul_one_right")
AST_PAD_ID = 0
PATH_END_ID = 2
PATH_PAD_ID = 3
DEFAULT_CONFIG = {
    "protocol": "rewrite_learning_v1",
    "families": {"train":128,"validation":32,"development":32},
    "split_seeds": {"train":53001,"validation":53002,"development":53003},
    "reserved_final_test_seed":53999,
    "episodes_per_family":8,
    "walk_steps":8,
    "include_reverse_transitions":True,
    "max_nodes": {"train":31,"validation":31,"development":47},
    "goal_node_counts": {"train":[9,11,13,15,17,19],"validation":[9,11,13,15,17,19],"development":[23,25,27,29,31,33]},
    "active_variables": {"train":[2,3,4],"validation":[2,3,4],"development":[4,5]},
    "goal_rewrites":2,
    "max_family_attempts":100_000,
}


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def tokenize_ast(expr) -> list[int]:
    """Ordered prefix tokens with FIXED literal variable IDs; no renaming."""
    if isinstance(expr,list):
        expr = ast_from_json(expr)
    tokens = []
    vocabulary = {token:index for index,token in enumerate(AST_TOKEN_VOCAB)}
    def visit(node):
        if isinstance(node,str):
            if node not in vocabulary or node in ("<PAD>","Add","Mul"):
                raise ValueError("Model vocabulary supports only literal variables x0..x5 and 0/1")
            tokens.append(vocabulary[node])
        else:
            if not isinstance(node,tuple) or len(node)!=3 or node[0] not in ("Add","Mul"):
                raise ValueError("Invalid AST")
            tokens.append(vocabulary[node[0]])
            visit(node[1]); visit(node[2])
    visit(expr)
    return tokens


def encode_action(action: RewriteAction | dict) -> dict:
    if isinstance(action,dict):
        action = RewriteAction.from_dict(action)
    if action.rule_id not in ACTION_RULE_VOCAB or type(action.direction) is not int or action.direction not in (-1,1):
        raise ValueError("Invalid action encoding")
    if not isinstance(action.node_path,tuple) or any(type(i) is not int or i not in (0,1) for i in action.node_path):
        raise ValueError("Invalid action path")
    return {"action_rule":ACTION_RULE_VOCAB.index(action.rule_id), "action_direction":int(action.direction==1),
            "action_path":list(action.node_path)+[PATH_END_ID]}


def encode_transition(record: dict) -> dict:
    """Only model tensors and supervised targets, with no family/seed leakage.

    next_tokens and cost values are targets. The predictor's inputs must be
    state_tokens and action features; a goal scorer additionally uses goal_tokens.
    """
    return {"state_tokens":tokenize_ast(record["current_ast"]),
            "goal_tokens":tokenize_ast(record["goal_ast"]),
            "next_tokens":tokenize_ast(record["next_ast"]),
            **encode_action(record["action"]),
            "current_cost":float(record["current_walk_upper_bound"]),
            "next_cost":float(record["next_walk_upper_bound"])}


def _rename(expr,mapping):
    return mapping.get(expr,expr) if isinstance(expr,str) else (expr[0],_rename(expr[1],mapping),_rename(expr[2],mapping))


def structural_template(expr, *, ignore_repeated_variable_identity=False):
    """Audit-only ordered template; repeated variables retain shared identity."""
    binding = {}
    def visit(node):
        if isinstance(node,str):
            if node in ("0","1"):
                return node
            if ignore_repeated_variable_identity:
                return "VAR"
            if node not in binding:
                binding[node] = "VAR"+str(len(binding))
            return binding[node]
        return (node[0],visit(node[1]),visit(node[2]))
    return visit(expr)


def _random_ast(rng,nodes,variables):
    if nodes==1:
        return rng.choice(("0","1")) if rng.random()<0.08 else rng.choice(variables)
    left_size = rng.choice(range(1,nodes-1,2))
    return (rng.choice(("Add","Mul")),_random_ast(rng,left_size,variables),_random_ast(rng,nodes-1-left_size,variables))


def _choose_action(env,state,rng,previous=None, *, structural_only=False):
    actions = [a for a in env.legal_actions(state) if previous is None or a!=previous.inverse()]
    if structural_only:
        structural = [a for a in actions if "zero" not in a.rule_id and "one" not in a.rule_id]
        actions = structural or actions
    if not actions:
        return None
    weights = []
    for action in actions:
        if "distribute" in action.rule_id:
            weight = 8 if action.direction==-1 else 4
        elif "assoc" in action.rule_id:
            weight = 3
        elif "comm" in action.rule_id:
            weight = 2
        else:
            weight = 1 if action.direction==-1 else 0.08
        weights.append(weight)
    return rng.choices(actions,weights=weights,k=1)[0]


def _record(episode,step_index,current,action,following,goal,reverse=False):
    return {"split":episode["split"],"episode_id":episode["episode_id"],"family_id":episode["family_id"],
            "semantic_group_id":episode["semantic_group_id"],"seed":episode["seed"],"step_index":step_index,
            "current_ast":following if reverse else current,"goal_ast":goal,
            "action":(action.inverse() if reverse else action).to_dict(),"next_ast":current if reverse else following,
            "current_walk_upper_bound":step_index+1 if reverse else step_index,
            "next_walk_upper_bound":step_index if reverse else step_index+1,
            "label_provenance":"Length of an exactly replayed goal-origin trajectory prefix; upper bound only, NOT shortest distance",
            "trajectory_direction":"reverse" if reverse else "forward"}


def audit_rewrite_dataset(episodes,transitions):
    def cross_split(records,key):
        groups = {}
        for record in records:
            groups.setdefault(key(record),set()).add(record["split"])
        return sum(len(splits)>1 for splits in groups.values())
    return {
        "cross_split_semantic_groups":cross_split(episodes,lambda e:e["semantic_group_id"]),
        "cross_split_goal_templates":cross_split(episodes,lambda e:_hash(structural_template(e["goal_ast"]))),
        "cross_split_goal_skeletons":cross_split(episodes,lambda e:_hash(structural_template(e["goal_ast"],ignore_repeated_variable_identity=True))),
        "cross_split_state_goal_pairs":cross_split(transitions,lambda r:_hash([r["current_ast"],r["goal_ast"]])),
        "cross_split_transition_templates":cross_split(transitions,lambda r:_hash([structural_template(r["current_ast"]),structural_template(r["goal_ast"])])),
        "cross_split_generating_histories":cross_split(episodes,lambda e:_hash(e["generating_actions"])),
        "exact_duplicate_transitions":len(transitions)-len({_hash([r["current_ast"],r["goal_ast"],r["action"],r["next_ast"]]) for r in transitions}),
        "note":"Within-family detours and reverse augmentation can intentionally repeat transitions. All repeated records remain in their family's split.",
    }


def generate_rewrite_dataset(config: dict | None = None, output_dir: str | Path | None = None) -> dict:
    """Generate semantic-family splits, exact transitions, and stable vocab.

    Returns episodes, flat transitions, and a manifest. Optional output writes
    one reconstructable ``rewrite_learning_v1.json`` in output_dir. Config keys
    override defaults; nested per-split dictionaries are merged one level.
    """
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for key,value in (config or {}).items():
        if isinstance(value,dict) and isinstance(cfg.get(key),dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    all_semantics = set()
    all_templates = set()
    skeleton_owner = {}
    episodes,transitions,families = [],[],[]
    attempts_by_split = {}
    for split in ("train","validation","development"):
        rng = random.Random(cfg["split_seeds"][split])
        env = RewriteEnv(cfg["max_nodes"][split])
        attempts = 0
        for family_index in range(cfg["families"][split]):
            while True:
                attempts += 1
                if attempts > cfg["max_family_attempts"]:
                    raise RuntimeError(f"Could not create {split} families without semantic/template leakage")
                variables = tuple("x"+str(i) for i in range(rng.choice(cfg["active_variables"][split])))
                template = _random_ast(rng,rng.choice(cfg["goal_node_counts"][split]),variables)
                polynomial = semantic_polynomial(template)
                actual_variables = {v for monomial,_ in polynomial for v in monomial}
                if len(actual_variables)<2 or len(polynomial)>128:
                    continue
                semantic_key = renamed_semantic_key(template)
                template_key = _hash(structural_template(template))
                skeleton = _hash(structural_template(template,ignore_repeated_variable_identity=True))
                if semantic_key in all_semantics or template_key in all_templates or skeleton in skeleton_owner:
                    continue
                break
            family_id = f"{split}_family_{family_index:04d}"
            all_semantics.add(semantic_key)
            all_templates.add(template_key)
            skeleton_owner[skeleton] = family_id
            family = {"family_id":family_id,"split":split,"semantic_group_id":semantic_key,
                      "template_sha256":template_key,"goal_skeleton_sha256":skeleton,"base_goal_ast":template}
            families.append(family)
            for episode_index in range(cfg["episodes_per_family"]):
                seed = cfg["split_seeds"][split]*100000+family_index*100+episode_index
                episode_rng = random.Random(seed)
                permutation = episode_rng.sample(["x"+str(i) for i in range(6)],6)
                goal = _rename(template,dict(zip(("x"+str(i) for i in range(6)),permutation)))
                goal_state = RewriteState(goal,goal)
                previous = None
                for _ in range(cfg["goal_rewrites"]):
                    candidate = _choose_action(env,goal_state,episode_rng,previous,structural_only=True)
                    if candidate is None:
                        break
                    proposed = env.step(goal_state,candidate)
                    goal_skeleton = _hash(structural_template(proposed.current,ignore_repeated_variable_identity=True))
                    if goal_skeleton in skeleton_owner and skeleton_owner[goal_skeleton]!=family_id:
                        continue
                    skeleton_owner[goal_skeleton] = family_id
                    goal_state = proposed
                    previous = candidate
                goal = goal_state.current
                assert renamed_semantic_key(goal)==semantic_key
                episode = {"episode_id":f"{family_id}_episode_{episode_index:02d}","family_id":family_id,
                           "semantic_group_id":semantic_key,"split":split,"seed":seed,"goal_ast":goal,
                           "max_nodes":env.max_nodes,"generating_actions":[],"goal_variable_permutation":permutation}
                state = RewriteState(goal,goal)
                previous = None
                explicit_detour = None
                for step_index in range(cfg["walk_steps"]):
                    if episode_index%4==0 and step_index==2:
                        candidates = [a for a in env.legal_actions(state) if a.rule_id=="add_zero_right" and a.direction==1]
                        action = episode_rng.choice(candidates) if candidates else _choose_action(env,state,episode_rng,previous)
                        explicit_detour = action
                    elif explicit_detour is not None and step_index==3:
                        action = explicit_detour.inverse()
                    else:
                        action = _choose_action(env,state,episode_rng,previous)
                    if action is None:
                        raise RuntimeError("Generator unexpectedly reached a state with no legal moves")
                    updated = env.step(state,action)
                    assert semantic_polynomial(state.current)==semantic_polynomial(updated.current)
                    transitions.append(_record(episode,step_index,state.current,action,updated.current,goal))
                    if cfg["include_reverse_transitions"]:
                        transitions.append(_record(episode,step_index,state.current,action,updated.current,goal,reverse=True))
                    episode["generating_actions"].append(action.to_dict())
                    state,previous = updated,action
                witness = [RewriteAction.from_dict(a).inverse() for a in reversed(episode["generating_actions"])]
                assert env.verify_trace(state,witness)
                episode.update({"current_ast":state.current,"walk_length":len(episode["generating_actions"]),
                                "verified_witness":[a.to_dict() for a in witness],
                                "label_provenance":"generator walk length; exact shortest distance unknown"})
                episodes.append(episode)
        attempts_by_split[split] = attempts
    audit = audit_rewrite_dataset(episodes,transitions)
    for key,value in audit.items():
        if key.startswith("cross_split_") and value:
            raise RuntimeError(f"Leakage audit failed: {key}={value}")
    record_counts = dict(Counter(r["split"] for r in transitions))
    forward_counts = dict(Counter(r["split"] for r in transitions if r["trajectory_direction"]=="forward"))
    manifest = {"protocol":cfg["protocol"],"config":cfg,"config_sha256":_hash(cfg),
                "dataset_sha256":_hash({"episodes":episodes,"transitions":transitions}),
                "family_records":families,"family_attempts":attempts_by_split,"leakage_audit":audit,
                "transition_counts":record_counts,"forward_transition_counts":forward_counts,
                "action_counts":dict(Counter(r["action"]["rule_id"]+":"+str(r["action"]["direction"]) for r in transitions)),
                "vocabulary":{"ast_tokens":AST_TOKEN_VOCAB,"action_rules":ACTION_RULE_VOCAB,
                              "direction_ids":{"inverse":0,"forward":1},"path_end_id":PATH_END_ID,"path_pad_id":PATH_PAD_ID},
                "model_input_fields":["state_tokens","goal_tokens","action_rule","action_direction","action_path"],
                "supervised_target_fields":["next_tokens","current_cost","next_cost"],
                "cost_label_meaning":"verified trajectory-prefix upper bound, not shortest distance",
                "variable_encoding":"Literal x0..x5 IDs, fixed across current/next/goal and all episodes; no alpha normalization",
                "reserved_final_test_seed":cfg["reserved_final_test_seed"],"final_test_generated":False,
                "held_out_compositions":{"enforced":False,"claim_permitted":False,"preregistered_next_protocol":"Hold out selected ordered rule pairs from EVERY training trajectory and predictive rollout before measuring composition shift."}}
    result = {"manifest":manifest,"episodes":episodes,"transitions":transitions}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True,exist_ok=True)
        destination = directory/"rewrite_learning_v1.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result,sort_keys=True,separators=(",", ":"))+"\n")
        temporary.replace(destination)
        result["output_path"] = str(destination)
    return result


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("data/manifests"))
    parser.add_argument("--config",type=Path)
    args = parser.parse_args()
    generated = generate_rewrite_dataset(json.loads(args.config.read_text()) if args.config else None,args.output_dir)
    print(json.dumps({"path":generated["output_path"],"manifest":generated["manifest"]},indent=2))
