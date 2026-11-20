"""Small random CPU Qwen models only: no checkpoint, GPU, or AppWorld task."""

import pytest
import torch
from torch import nn

from classification.types import TokenAction as A
from compression.operator import CompressionConfig
from runtime.model import AgentModel


@pytest.fixture
def tiny_model(monkeypatch):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    # Use Transformers' CPU kernels even when CUDA extensions are installed.
    monkeypatch.setattr(qwen, "FusedRMSNormGated", None)
    monkeypatch.setattr(qwen, "causal_conv1d_fn", None)
    monkeypatch.setattr(qwen, "causal_conv1d_update", qwen.torch_causal_conv1d_update)
    monkeypatch.setattr(qwen, "chunk_gated_delta_rule", qwen.torch_chunk_gated_delta_rule)
    monkeypatch.setattr(qwen, "fused_recurrent_gated_delta_rule", qwen.torch_recurrent_gated_delta_rule)
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=8,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
                    + ["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1.0, "mrope_section": [4, 2, 2],
                         "mrope_interleaved": True},
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(10)
    backbone = qwen.Qwen3_5TextModel(config).eval()
    return AgentModel(backbone, nn.Linear(32, 64, bias=False), CompressionConfig(
        layer=6, slots=1, merge_error=100., min_layer_savings=0, min_tool_fraction=0.,
    ))


@torch.inference_mode()
def test_all_keep_matches_native_and_cached_multitoken_continuation(tiny_model):
    ids = torch.arange(1, 13)
    native = tiny_model.backbone(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    full = tiny_model(ids)
    torch.testing.assert_close(full.sequence.hidden, native, rtol=1e-4, atol=1e-5)
    prefix = tiny_model(ids[:5])
    suffix = tiny_model(ids[5:], cache=prefix.cache)
    torch.testing.assert_close(suffix.sequence.hidden, native[5:], rtol=1e-4, atol=1e-5)
    for index in (0, 1, 2, 4, 5, 6):
        torch.testing.assert_close(suffix.cache.native.recurrent_states[index],
                                   full.cache.native.recurrent_states[index], rtol=1e-4, atol=1e-5)
    assert suffix.cache.layer_lengths == [12] * 8


@torch.inference_mode()
def test_compressed_cache_survives_decode_and_next_observation(tiny_model):
    ids = torch.arange(1, 13)
    span = torch.tensor([-1, -1] + [0] * 8 + [-1, -1])
    actions = torch.tensor([A.KEEP, A.KEEP, A.MERGE, A.MERGE, A.KEEP, A.DELETE,
                            A.MERGE, A.MERGE, A.MERGE, A.MERGE, A.KEEP, A.KEEP])
    logits, first = tiny_model.next_logits(ids, span=span, actions=actions)
    assert torch.isfinite(logits).all()
    assert first.layer_lengths == [12] * 7 + [6]
    assert first.cache.layer_lengths == [12] * 7 + [6]
    assert first.cache.logical_length == 12
    assert first.events[0]["merged_tokens"] == 6
    assert first.events[0]["deleted_tokens"] == 1
    for index in (3, 7):
        assert first.cache.native.get_seq_length(index) == first.cache.layer_lengths[index]

    _, decode = tiny_model.next_logits(torch.tensor([13]), cache=first.cache)
    assert decode.cache.logical_length == 13
    assert decode.cache.layer_lengths == [13] * 7 + [7]
    assert decode.sequence.position.tolist() == [12]
    assert not decode.events
    second = tiny_model(
        torch.tensor([14, 15, 16, 17]), span=torch.tensor([1, 1, 1, -1]),
        actions=torch.tensor([A.DELETE, A.DELETE, A.DELETE, A.KEEP]), cache=decode.cache,
    )
    assert second.sequence.position.tolist() == [16]
    assert second.cache.logical_length == 17
    assert second.cache.layer_lengths == [17] * 7 + [8]
    assert second.cache.seen_spans == {0, 1}
    assert second.events[0]["deleted_tokens"] == 3
