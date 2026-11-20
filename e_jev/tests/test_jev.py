import json

import pytest
import requests

from classification.jev import API_URL, JevClassifier, JevConfig
from classification.types import ClassificationRequest, TokenAction as A
from observation.fields import split_observation


def classification_request():
    history = [{"action": "print(1)", "observation": "1", "assistant": "```python\nprint(1)\n```"}]
    return ClassificationRequest(
        "find item", history, "lookup()",
        split_observation('{"api":"lookup", "id":12345, "description":"background"}'),
        "Return one Python code block. Use the available tool APIs.",
    )


def test_joint_batches_receive_complete_history_and_observation(jev_api):
    classifier = JevClassifier(
        JevConfig(fields_per_request=2, prompt_version="task-relevance-v2"), session=jev_api.session,
    )
    request = classification_request()
    result = classifier.classify(request)
    assert [decision.action for decision in result.decisions] == [A.KEEP, A.MERGE, A.MERGE]
    assert len(jev_api.requests) == 2
    for http_request, options in jev_api.requests:
        assert http_request.url == API_URL
        assert http_request.headers["Authorization"] == "Bearer test-jev-key"
        assert http_request.headers["TypeSafe-Danger-Zone-Long-Context"] == "true"
        assert options["timeout"] == 300
        record = json.loads(http_request.body)
        assert record["model"] == "jev-1.14.0"
        state = record["state"]
        assert state["history"] == request.history
        assert state["task"] == request.task
        assert state["classification_policy"] == classifier.prompt["instructions"]
        assert state["agent_context"]["system_prompt"] == request.agent_system_prompt
        assert "not to you (Jev)" in state["agent_context"]["notice"]
        assert [f["text"] for f in state["observation_fields"]] == ['"lookup"', '12345', '"background"']
        for key, question in record["questions"].items():
            index = int(key.removeprefix("field_"))
            assert question["instructions"].endswith(state["observation_fields"][index]["text"])
            assert classifier.prompt["instructions"] not in question["instructions"]
            assert request.agent_system_prompt not in question["instructions"]
            assert question["criteria"] == classifier.prompt["criteria"]
    assert result.metadata["input_tokens"] == 200
    assert result.metadata["output_tokens"] == 24
    assert all(call["model"] == "jev-1.14.0" for call in result.metadata["calls"])
    assert result.metadata["prompt_version"] == "task-relevance-v2"
    assert "test-jev-key" not in json.dumps(classifier.runtime_metadata)
    classifier.close()


def test_threshold_uses_probability_not_selected_choice(jev_api):
    jev_api.body = {
        "model": "jev-1.14.0", "usage": {"input_tokens": 1, "output_tokens": 2},
        "answers": {"field_0": {
            "type": "choice", "choice": "irrelevant", "confidence": 0.0,
            "probabilities": {"relevant": 0.5, "irrelevant": 0.5},
        }},
    }
    classifier = JevClassifier(JevConfig(long_context=False), session=jev_api.session)
    result = classifier.classify(ClassificationRequest("task", [], "action", split_observation("1"), "prompt"))
    assert result.decisions[0].action == A.KEEP
    assert "TypeSafe-Danger-Zone-Long-Context" not in jev_api.requests[0][0].headers


def test_missing_answers_are_an_error(jev_api):
    jev_api.body = {"answers": {}}
    classifier = JevClassifier(JevConfig(), session=jev_api.session)
    with pytest.raises(ValueError, match="requested fields"):
        classifier.classify(classification_request())


@pytest.mark.parametrize("status", [401, 422, 429, 529])
def test_api_errors_propagate_without_keep_all_fallback(jev_api, status):
    jev_api.status, jev_api.body = status, {"detail": "API error"}
    classifier = JevClassifier(JevConfig(), session=jev_api.session)
    with pytest.raises(requests.HTTPError):
        classifier.classify(classification_request())


def test_key_required_only_when_constructing_classifier(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevClassifier(JevConfig())


def test_config_selects_conservative_prompt_and_keeps_uncertainty(jev_api):
    from experiment.config import ROOT, load_config

    config = load_config(ROOT / "experiment/config.json")
    jev_api.body = {
        "model": "jev-1.14.0", "usage": {"input_tokens": 10, "output_tokens": 2},
        "answers": {"field_0": {"probabilities": {"relevant": 0.5, "irrelevant": 0.5}}},
    }
    classifier = JevClassifier(JevConfig(**config["jev"]), session=jev_api.session)
    result = classifier.classify(ClassificationRequest(
        "task", [], "action", split_observation("possibly useful"), "Agent prompt",
    ))
    assert classifier.prompt["version"] == "task-relevance-v3"
    assert result.metadata["prompt_version"] == "task-relevance-v3"
    assert result.decisions[0].action == A.KEEP
    payload = json.loads(jev_api.requests[0][0].body)
    assert payload["state"]["classification_policy"] == classifier.prompt["instructions"]
    assert payload["questions"]["field_0"]["criteria"] == classifier.prompt["criteria"]
    classifier.close()
