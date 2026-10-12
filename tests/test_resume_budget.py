import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from simpletes.config import EngineConfig
from simpletes.engine.checkpoint import CheckpointManager
from simpletes.node import Node, NodeDatabase, Status
from simpletes.policies import create_selector


def completed_checkpoint(tmp_path, monkeypatch, *, selector="balance", budget=60, chains=4, k=1):
    config = EngineConfig("seed.py", "evaluator.py", "task.txt", selector=selector,
                          max_generations=budget, num_chains=chains, k_candidates=k,
                          output_path=str(tmp_path), reflection_mode=False)
    policy = create_selector(selector, num_chains=chains, max_generations=budget, k=k)
    db = NodeDatabase()
    root = Node(id="root", code="x = 0", metrics={"combined_score": 0.0},
                score=0.0, status=Status.DONE)
    db.add(root)
    state = policy.state_dict()
    state.update(root_id=root.id, initialized=True, ready_chains=[])
    for chain, quota in policy.chain_gen_budget.items():
        history = [root.id]
        for attempt in range(quota):
            node = Node(id=f"chain-{chain}-{attempt}", code=f"x = {attempt + 1}",
                        parent_ids=[history[-1]], metrics={"combined_score": float(attempt + 1)},
                        score=float(attempt + 1), status=Status.DONE)
            db.add(node)
            history.append(node.id)
        state["chains"][str(chain)] = history
        state["chain_history"][str(chain)] = history
        state["chain_prompt_count"][str(chain)] = (quota + k - 1) // k
    policy.load_state_dict(state)
    policy.reconcile_with_db(db)
    manager = CheckpointManager(config, "original-instance", str(tmp_path))
    snapshot = asyncio.run(manager.snapshot(db, db.best().id, budget + 1, budget,
                                           0, 0, 0, db.best().score, policy))
    monkeypatch.setattr("simpletes.engine.checkpoint.save_score_statistics", lambda *args: None)
    manager.write_sync(*snapshot, [])
    return config, policy, db, manager


def restore(config, tmp_path):
    policy = create_selector(config.selector, num_chains=config.num_chains,
                             max_generations=config.max_generations, k=config.k_candidates)
    db = NodeDatabase()
    manager = CheckpointManager(config, "new-instance", str(tmp_path))
    restored = manager.load(str(tmp_path), db, policy)
    return policy, db, restored, manager


@pytest.mark.parametrize("selector", ["balance", "puct", "rpucg"])
def test_resume_completed_run_adds_capacity_and_preserves_history(tmp_path, monkeypatch, selector):
    config, original, original_db, _ = completed_checkpoint(tmp_path, monkeypatch, selector=selector)
    before = original.state_dict()
    policy, db, restored, manager = restore(replace(config, max_generations=120), tmp_path)

    assert restored["instance_id"] == "original-instance"
    assert restored["generation_attempts"] == 60
    assert restored["completed_evaluations"] == 61
    assert restored["best_node_id"] == original_db.best().id
    assert restored["best_score"] == original_db.best().score
    assert {key: node.to_dict() for key, node in db.nodes.items()} == {
        key: node.to_dict() for key, node in original_db.nodes.items()}
    assert policy.max_generations == 120
    assert policy.chain_gen_budget == {i: 30 for i in range(4)}
    assert policy.prompt_budget == {i: 30 for i in range(4)}
    assert policy.chain_prompt_count == {i: 15 for i in range(4)}
    assert set(policy._ready_chains) == set(range(4))
    after = policy.state_dict()
    for key in before.keys() - {"max_generations", "chain_gen_budget", "prompt_budget", "ready_chains"}:
        assert after[key] == before[key]
    inspirations, _, chain = policy.select(db, 1, None, 1)
    assert inspirations and chain in range(4)
    assert all(node.id in original_db.nodes for node in inspirations)

    # Persist the increased limit: a subsequent resume must not add it again.
    snapshot = asyncio.run(manager.snapshot(db, restored["best_node_id"], 61, 60,
                                           0, 0, 0, restored["best_score"], policy))
    manager.write_sync(*snapshot, [])
    persisted = json.loads((Path(manager._last_checkpoint_path) / "config.json").read_text())
    assert persisted["max_generations"] == 120
    again, _, _, _ = restore(replace(config, max_generations=120), tmp_path)
    assert again.chain_gen_budget == policy.chain_gen_budget
    assert again.prompt_budget == policy.prompt_budget
    assert again.chain_prompt_count == policy.chain_prompt_count


def test_resume_without_increase_keeps_exhausted_chains(tmp_path, monkeypatch):
    config, original, _, _ = completed_checkpoint(tmp_path, monkeypatch)
    policy, _, restored, _ = restore(config, tmp_path)
    assert policy.state_dict() == original.state_dict()
    assert not policy._ready_chains
    assert restored["generation_attempts"] == config.max_generations


@pytest.mark.parametrize("budget,new_budget,chains,k,ready", [
    (0, 2, 4, 1, {0, 1}),
    (1, 2, 1, 2, {0}),
    (5, 6, 2, 2, {1}),
    (5, 9, 2, 2, {0, 1}),
])
def test_extension_reopens_partial_batches_and_zero_budget_chains(
        tmp_path, monkeypatch, budget, new_budget, chains, k, ready):
    config, original, _, _ = completed_checkpoint(tmp_path, monkeypatch, budget=budget, chains=chains, k=k)
    policy, db, restored, _ = restore(replace(config, max_generations=new_budget), tmp_path)
    assert restored["generation_attempts"] == budget
    assert set(policy._ready_chains) == ready
    assert sum(policy.chain_gen_budget.values()) == new_budget
    for chain in ready:
        assert policy.prompt_budget[chain] > original.prompt_budget[chain]
    assert policy.select(db, 1, None, 1)[0]


def test_resume_rejects_decreased_budget_without_loading_nodes(tmp_path, monkeypatch):
    config, _, _, _ = completed_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot decrease max_generations"):
        restore(replace(config, max_generations=59), tmp_path)


@pytest.mark.parametrize("changes,key", [
    ({"selector": "puct"}, "selector"),
    ({"num_chains": 2}, "num_chains"),
    ({"k_candidates": 2}, "k_candidates"),
])
def test_extension_rejects_changed_search_topology(tmp_path, monkeypatch, changes, key):
    config, _, _, _ = completed_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=f"cannot change {key}"):
        restore(replace(config, max_generations=120, **changes), tmp_path)


@pytest.mark.parametrize("selector", ["balance", "puct", "rpucg", "llm_puct", "llm_rpucg", "llm_elite"])
def test_extension_preserves_pending_batch_and_policy_state(selector):
    policy = create_selector(selector, num_chains=2, max_generations=2, k=1)
    policy.register_batch(7, 0, ["root"], 1)
    before = deepcopy(policy.state_dict())
    policy.extend_budget(4)
    after = policy.state_dict()
    for key in before.keys() - {"max_generations", "chain_gen_budget", "prompt_budget", "ready_chains"}:
        assert after[key] == before[key]
    assert after["batches"]["7"] == before["batches"]["7"]
    policy.extend_budget(4)
    assert policy.state_dict() == after


@pytest.mark.parametrize("direct_checkpoint", [False, True])
def test_cli_resume_inherits_settings_and_accepts_new_total_budget(tmp_path, monkeypatch, direct_checkpoint):
    from main import parse_run_args
    from simpletes.cli import _validate_required_args, build_parser

    config, _, _, manager = completed_checkpoint(tmp_path, monkeypatch)
    path = manager._last_checkpoint_path if direct_checkpoint else str(tmp_path)
    parser = build_parser()
    args = parse_run_args(parser, ["--resume", path, "--max-generations", "120"])
    _validate_required_args(args, parser)
    assert args.max_generations == 120
    assert args.num_chains == config.num_chains
    assert args.k_candidates == config.k_candidates
    assert args.selector == config.selector
    assert Path(args.init_program).resolve() == Path(config.init_program).resolve()
    assert Path(args.evaluator).resolve() == Path(config.evaluator_path).resolve()
    assert Path(args.instruction).resolve() == Path(config.instruction_path).resolve()
    assert args.disable_reflection
    args = parse_run_args(build_parser(), ["--resume", path])
    assert args.max_generations == 60


def test_legacy_checkpoint_restores_chain_dimensions_from_policy(tmp_path, monkeypatch):
    from main import parse_run_args
    from simpletes.cli import build_parser

    config, _, _, manager = completed_checkpoint(tmp_path, monkeypatch, chains=2, k=2)
    path = Path(manager._last_checkpoint_path) / "config.json"
    saved = json.loads(path.read_text())
    del saved["num_chains"], saved["k_candidates"]
    path.write_text(json.dumps(saved))
    args = parse_run_args(build_parser(), ["--resume", str(tmp_path), "--max-generations", "120"])
    assert (args.num_chains, args.k_candidates) == (2, 2)
    policy, _, _, _ = restore(replace(config, max_generations=120), tmp_path)
    assert policy.num_chains == 2 and policy.k == 2
    assert sum(policy.chain_gen_budget.values()) == 120


def test_checkpoint_saves_resume_settings_and_excludes_credentials(tmp_path):
    config = EngineConfig("seed.py", "evaluator.py", "task.txt", num_chains=2,
                          k_candidates=1, use_gzip=True, exploitation_ratio=0.9,
                          backpressure_multiplier=2, api_key="generator-secret",
                          llm_policy_api_key="policy-secret")
    saved = CheckpointManager(config, "instance", str(tmp_path))._config_to_dict()
    assert saved["num_chains"] == 2 and saved["k_candidates"] == 1
    assert saved["use_gzip"] and saved["exploitation_ratio"] == 0.9
    assert saved["backpressure_multiplier"] == 2
    assert "api_key" not in saved and "llm_policy_api_key" not in saved
    assert "secret" not in json.dumps(saved)
