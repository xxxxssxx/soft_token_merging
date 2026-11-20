"""Two synthetic interactions exercise the loop, history, and delayed merge logs."""

from copy import deepcopy
import json
from importlib import import_module
from contextlib import nullcontext

import pytest
import requests
from types import SimpleNamespace
import torch

from classification.jev import JevClassifier, JevConfig
from compression.operator import CompressionConfig, compress
from compression.sequence import Sequence
from experiment.recording import task_summary
from experiment.runner import collect_task
from runtime.agent import GenerationConfig, Protocol


class CharacterTokenizer:
    all_special_ids = []

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ScriptedAgent:
    model = SimpleNamespace(device="cpu")
    tokenizer = CharacterTokenizer()
    protocol = Protocol()
    generation = GenerationConfig()

    def generate(self, tape, generator):
        tape.append(self.protocol.assistant())
        kwargs = tape.uncached("cpu")
        n = len(kwargs["ids"])
        offset = tape.cache.logical_length if tape.cache else 0
        seq = Sequence.create(torch.ones(n, 2), kwargs["span"], kwargs["actions"], offset)
        _, events = compress(seq, CompressionConfig(slots=1, min_layer_savings=0), 21)
        tape.cache = SimpleNamespace(logical_length=offset + n)
        text = "```python\nprint(1)\n```"
        tape.append(text)
        return "print(1)", {
            "text": text, "stop": "action_complete", "compression_events": events,
            "input_tokens": len(tape.ids), "output_tokens": len(text),
            "forward_input_tokens": n, "token_layer_positions": n,
            "attention_pairs": 0, "wall_seconds": 0.0,
            "generated_ids": [1, 2], "cache_layer_lengths": [n],
        }


class World:
    task = SimpleNamespace(supervisor={}, instruction="find item")
    observation = '{"id":123,"unrelated":"background text"}'

    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        return self.observation

    def task_completed(self):
        return len(self.actions) == 2

    def evaluate(self):
        return SimpleNamespace(success=True, pass_count=1, num_tests=1)


def test_loop_classifies_full_history_and_records_compression_on_prior_turn(jev_api):
    snapshots = []

    classifier = JevClassifier(JevConfig(), session=jev_api.session)
    record = {"task_id": "synthetic", "seed": 1}
    world = World()
    collect_task(world, ScriptedAgent(), classifier, {"max_steps": 2}, record,
                 lambda: snapshots.append(deepcopy(record)))
    records = [json.loads(request.body) for request, _ in jev_api.requests]
    assert records[0]["state"]["history"] == []
    assert records[0]["state"]["agent_context"]["system_prompt"] == ScriptedAgent.protocol.config["system_prompt"]
    assert records[1]["state"]["history"][0]["observation"] == world.observation
    assert record["turns"][0]["compression"]["merged_tokens"] > 1
    assert record["turns"][1]["compression"] is None
    assert record["turns"][0]["fields"][0]["action"] == "KEEP"
    assert record["turns"][0]["fields"][1]["action"] == "MERGE"
    assert record["turns"][0]["fields"][0]["text"] == '{"id":123'
    for snapshot in snapshots:
        for turn in snapshot["turns"]:
            assert "field_decisions" not in turn and "observation_tokens" not in turn
            for field in turn.get("fields", []):
                assert "start" not in field and "end" not in field
            if "agent" in turn:
                assert "generated_ids" not in turn["agent"]
                assert "cache_layer_lengths" not in turn["agent"]
                assert all("soft_groups" not in event for event in turn["agent"]["compression_events"])
            if turn.get("compression") is not None:
                assert "soft_groups" not in turn["compression"]
    assert record["official_score"]["success"]
    assert any(snapshot["turns"][-1]["phase"] == "observed" for snapshot in snapshots if snapshot["turns"])
    record["task_wall_seconds"] = 0.0
    summary = task_summary(record)
    assert summary["jev"]["calls"] == 2
    assert summary["jev"]["input_tokens"] == 200
    assert summary["jev"]["output_tokens"] == 24
    assert summary["compression"]["processed_observations"] == 1
    assert summary["compression"]["unprocessed_observations"] == 1


@pytest.fixture(params=["experiment.runner", "experiment.runner_diagnose"])
def synthetic_run(monkeypatch, tmp_path, request):
    runner = import_module(request.param)
    from experiment.config import ROOT, load_config
    from transformers import AutoTokenizer

    config = load_config(ROOT / "experiment/config.json")
    config.update(output_directory=str(tmp_path / "run"), max_steps=2)
    world = World()
    monkeypatch.setattr(runner, "select_tasks", lambda config: ({"task_ids": ["synthetic"], "seed": 1}, ["synthetic"]))
    monkeypatch.setattr(runner, "prepare_environment", lambda *args: None)
    monkeypatch.setattr(runner, "check_environment_service", lambda *args: None)
    monkeypatch.setattr(runner, "load_model", lambda *args: ScriptedAgent.model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: ScriptedAgent.tokenizer)
    monkeypatch.setattr(runner, "Agent", lambda *args: ScriptedAgent())
    monkeypatch.setattr(runner, "open_environment", lambda *args: nullcontext(world))
    return runner, config, tmp_path / "run"


def test_base_run_needs_no_key_or_classifier(synthetic_run, monkeypatch):
    runner, config, output = synthetic_run
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    config["variant"] = "base"

    def forbidden(*args):
        raise AssertionError("base must not create a classifier")

    monkeypatch.setattr(runner, "JevClassifier", forbidden)
    runner.run(config)
    assert json.loads((output / "status.json").read_text())["phase"] == "finished"
    summary = json.loads((output / "summary.json").read_text())["task_results"][0]
    assert summary["jev"]["calls"] == 0
    assert summary["compression"]["merged_tokens"] == 0
    assert json.loads((output / "runtime.json").read_text())["classification"] is None


@pytest.mark.parametrize("status", [401, 422, 429, 529])
def test_api_failure_preserves_partial_trajectory(synthetic_run, jev_api, monkeypatch, status):
    runner, config, output = synthetic_run
    jev_api.status, jev_api.body = status, {"detail": "API error"}
    classifier = JevClassifier(JevConfig(), session=jev_api.session)
    closed = []
    monkeypatch.setattr(classifier, "close", lambda: closed.append(True))
    monkeypatch.setattr(runner, "JevClassifier", lambda config: classifier)
    with pytest.raises(requests.HTTPError):
        runner.run(config)
    state = json.loads((output / "status.json").read_text())
    trajectory = json.loads((output / "trajectories/test_normal-synthetic.json").read_text())
    assert state["phase"] == "error" and state["completed"] == 0
    assert trajectory["official_score"] is None
    assert trajectory["turns"][0]["phase"] == "classifying"
    assert trajectory["turns"][0]["raw_observation"] == World.observation
    assert not (output / "summary.json").exists()
    assert closed == [True]
    assert "test-jev-key" not in (output / "run.json").read_text()


def test_success_preserves_current_prompt_and_result_order(synthetic_run, jev_api, monkeypatch):
    runner, config, output = synthetic_run
    classifier = JevClassifier(JevConfig(**config["jev"]), session=jev_api.session)
    closed = []
    monkeypatch.setattr(classifier, "close", lambda: closed.append(True))
    monkeypatch.setattr(runner, "JevClassifier", lambda config: classifier)
    runner.run(config)
    run = json.loads((output / "run.json").read_text())
    trajectory = json.loads((output / "trajectories/test_normal-synthetic.json").read_text())
    assert run["classification_prompt"]["version"] == "task-relevance-v3"
    assert list(trajectory)[:6] == [
        "completed", "official_score", "instruction", "termination_reason",
        "task_wall_seconds", "finished_at",
    ]
    assert trajectory["official_score"]["success"]
    summary = json.loads((output / "summary.json").read_text())["task_results"][0]
    assert summary["jev"]["output_tokens"] == 24
    assert closed == [True]
