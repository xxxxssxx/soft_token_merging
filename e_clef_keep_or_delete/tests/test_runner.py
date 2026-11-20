"""Two synthetic interactions exercise the loop, history, and delayed deletion logs."""

from copy import deepcopy
from types import SimpleNamespace
import torch

from classification.clef import ClefClassifier, ClefConfig
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


def test_loop_classifies_full_history_and_records_compression_on_prior_turn():
    requests, snapshots = [], []

    def transport(record):
        requests.append(deepcopy(record))
        return {
            "answers": {key: {"relevant": .9 if key == "field_0" else .1,
                              "irrelevant": .1 if key == "field_0" else .9}
                        for key in record["questions"]},
            "input_tokens": 100, "wall_seconds": 0.0,
        }

    classifier = ClefClassifier(ClefConfig(), transport)
    record = {"task_id": "synthetic", "seed": 1}
    world = World()
    collect_task(world, ScriptedAgent(), classifier, {"max_steps": 2}, record,
                 lambda: snapshots.append(deepcopy(record)))
    assert requests[0]["state"]["history"] == []
    assert requests[0]["state"]["agent_context"]["system_prompt"] == ScriptedAgent.protocol.config["system_prompt"]
    assert requests[1]["state"]["history"][0]["observation"] == world.observation
    assert record["turns"][0]["compression"]["deleted_tokens"] > 1
    assert record["turns"][0]["compression"]["merged_tokens"] == 0
    assert record["turns"][0]["compression"]["soft_tokens"] == 0
    assert record["turns"][1]["compression"] is None
    assert record["turns"][0]["fields"][0]["action"] == "KEEP"
    assert record["turns"][0]["fields"][1]["action"] == "DELETE"
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
    assert summary["clef"]["calls"] == 2
    assert summary["compression"]["processed_observations"] == 1
    assert summary["compression"]["unprocessed_observations"] == 1
