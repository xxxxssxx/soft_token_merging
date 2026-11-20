from classification.clef import ClefClassifier, ClefConfig
from classification.types import ClassificationRequest, TokenAction as A
from observation.fields import split_observation
from experiment.config import ROOT, load_config


def test_joint_batches_receive_complete_history_and_observation():
    requests = []

    def transport(record):
        requests.append(record)
        return {
            "answers": {
                key: {"relevant": 0.9 if key == "field_0" else 0.1,
                      "irrelevant": 0.1 if key == "field_0" else 0.9}
                for key in record["questions"]
            },
            "input_tokens": 100, "wall_seconds": 0.01,
            "model": "test", "release_revision": "test",
        }

    classifier = ClefClassifier(ClefConfig(fields_per_request=2), transport)
    history = [{"action": "print(1)", "observation": "1", "assistant": "```python\nprint(1)\n```"}]
    raw = '{"api":"lookup", "id":12345, "description":"background"}'
    agent_prompt = "Return one Python code block. Use the available tool APIs."
    result = classifier.classify(ClassificationRequest(
        "find item", history, "lookup()", split_observation(raw), agent_prompt,
    ))
    assert [decision.action for decision in result.decisions] == [A.KEEP, A.MERGE, A.MERGE]
    assert len(requests) == 2
    for request in requests:
        assert request["state"]["history"] == history
        assert request["state"]["task"] == "find item"
        state = request["state"]
        assert state["classification_policy"] == classifier.prompt["instructions"]
        assert state["agent_context"]["system_prompt"] == agent_prompt
        assert "not to you (Clef)" in state["agent_context"]["notice"]
        assert [f["text"] for f in state["observation_fields"]] == ['"lookup"', '12345', '"background"']
        for key, question in request["questions"].items():
            index = int(key.removeprefix("field_"))
            assert question["instructions"].endswith(state["observation_fields"][index]["text"])
            assert classifier.prompt["instructions"] not in question["instructions"]
            assert agent_prompt not in question["instructions"]
            assert question["criteria"] == classifier.prompt["criteria"]
    assert result.metadata["input_tokens"] == 200
    assert result.metadata["output_tokens"] == 0
    assert result.metadata["prompt_version"] == "task-relevance-v2"


def test_config_selects_conservative_prompt_and_records_version():
    config = load_config(ROOT / "experiment/config.json")
    requests = []

    def transport(record):
        requests.append(record)
        return {
            "answers": {"field_0": {"relevant": 0.5, "irrelevant": 0.5}},
            "input_tokens": 100,
        }

    classifier = ClefClassifier(ClefConfig(**config["clef"]), transport)
    result = classifier.classify(ClassificationRequest(
        "find item", [], "lookup()", split_observation('"possibly useful"'), "Agent prompt",
    ))
    assert classifier.prompt["version"] == "task-relevance-v3"
    assert requests[0]["state"]["classification_policy"] == classifier.prompt["instructions"]
    assert requests[0]["questions"]["field_0"]["instructions"].startswith(classifier.prompt["question"])
    assert result.metadata["prompt_version"] == "task-relevance-v3"
    assert result.decisions[0].action == A.KEEP
