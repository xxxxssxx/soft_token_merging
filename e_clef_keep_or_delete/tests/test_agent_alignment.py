"""Local tokenizer only; token IDs must not change when field metadata is added."""

from pathlib import Path
import pytest

from classification.types import FieldDecision, TokenAction as A
from observation.fields import split_observation
from observation.alignment import align_observation
from runtime.agent import Protocol, Tape


def test_local_agent_tokenizer_and_tape_keep_original_ids():
    path = Path("/home/cache/astc/models/Qwen3.5-9B")
    if not (path / "tokenizer.json").exists():
        pytest.skip("local Agent tokenizer is not installed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    raw = '{"名字":"小明😀", "id":12345678901234567890, "items":[{"a":1,"b":2}]}'
    segmented = split_observation(raw)
    decisions = [FieldDecision(f.id, A.MERGE, "irrelevant") for f in segmented.fields]
    encoded = align_observation(segmented, decisions, tokenizer)
    assert encoded.ids == tokenizer.encode(raw, add_special_tokens=False)
    assert all(action == A.MERGE for action in encoded.actions)
    tape = Tape(tokenizer, Protocol(), "task")
    start = tape.observation(encoded)
    assert tape.ids[start:start + len(encoded.ids)] == encoded.ids
    assert tape.spans[start:start + len(encoded.ids)] == [0] * len(encoded.ids)
    assert tape.actions[start:start + len(encoded.ids)] == encoded.actions
    assert tape.spans[-1] == -1 and tape.actions[-1] == A.KEEP
