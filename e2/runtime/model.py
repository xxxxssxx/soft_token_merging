"""Stage-boundary compression with persistent, layer-specific Qwen3.5 caches.

Adapted from ASTC runtime/model.py. Selection no longer depends on a GDN reader:
the explicit boundary calls the three-action operator with the supplied metadata.
"""

from dataclasses import dataclass
import torch
from torch import nn

from classification.types import TokenAction
from compression.operator import CompressionConfig, compress
from compression.sequence import Sequence
from runtime.gdn import gdn_forward


class HybridCache:
    def __init__(self, model):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        self.native = Qwen3_5DynamicCache(model.backbone.config)
        self.owner, self.policy = model, model.compression
        self.logical_length = 0
        self.layer_lengths = [0] * len(model.backbone.layers)
        self.seen_spans = set()
        self.valid = True

    def bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for name in ("key_cache", "value_cache", "conv_states", "recurrent_states")
            for tensor in getattr(self.native, name) if tensor is not None
        )


@dataclass
class ModelOutput:
    sequence: Sequence
    cache: HybridCache
    layer_lengths: list[int]
    events: list[dict]
    attention_pairs: int


def causal_mask(n, previous, hidden):
    if previous == 0 or n == 1:
        return None  # native SDPA supplies first-prefill causality
    queries = previous + torch.arange(n, device=hidden.device)
    keys = torch.arange(previous + n, device=hidden.device)
    allowed = keys[None, :] <= queries[:, None]
    mask = hidden.new_zeros((n, previous + n))
    return mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]


class AgentModel(nn.Module):
    def __init__(self, backbone, lm_head, compression: CompressionConfig):
        super().__init__()
        if not 0 <= compression.layer < len(backbone.layers) - 1:
            raise ValueError("compression boundary must precede at least one layer")
        self.backbone, self.lm_head, self.compression = backbone, lm_head, compression
        self.requires_grad_(False).eval()

    @property
    def device(self):
        return self.backbone.embed_tokens.weight.device

    @torch.inference_mode()
    def forward(self, ids, *, span=None, actions=None, cache=None):
        cache = HybridCache(self) if cache is None else cache
        if not cache.valid or cache.owner is not self or cache.policy != self.compression:
            raise ValueError("cache belongs to another model/policy or an interrupted forward")
        span = torch.full_like(ids, -1) if span is None else span
        actions = torch.full_like(ids, TokenAction.KEEP) if actions is None else actions
        seq = Sequence.create(
            self.backbone.embed_tokens(ids), span, actions, offset=cache.logical_length
        )
        tool_spans = {tool for tool, _, _ in seq.tool_ranges()}
        if cache.seen_spans & tool_spans:
            raise ValueError("cannot reopen a cached observation")
        cache.valid = False
        lengths, events, attention_pairs = [], [], 0
        for index, layer in enumerate(self.backbone.layers):
            n, previous = len(seq.hidden), cache.layer_lengths[index]
            lengths.append(n)
            x = seq.hidden[None]
            if layer.layer_type == "linear_attention":
                mixed = gdn_forward(layer.linear_attn, layer.input_layernorm(x), cache.native)
                hidden = x + mixed
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                positions = seq.position[None, None].expand(3, 1, -1)
                hidden = layer(
                    x,
                    position_embeddings=self.backbone.rotary_emb(x, positions),
                    position_ids=positions,
                    attention_mask=causal_mask(n, previous, x),
                    cache_position=torch.arange(previous, previous + n, device=x.device),
                    past_key_values=cache.native,
                )
                attention_pairs += previous * n + n * (n + 1) // 2
                if cache.native.get_seq_length(index) != previous + n:
                    raise RuntimeError("physical attention cache length mismatch")
            seq = seq.with_hidden(hidden[0])
            cache.layer_lengths[index] += n
            if index == self.compression.layer and tool_spans:
                # 压缩在这里发生
                seq, events = compress(seq, self.compression, len(self.backbone.layers) - index - 1)
        seq = seq.with_hidden(self.backbone.norm(seq.hidden))
        cache.logical_length += len(ids)
        cache.seen_spans.update(tool_spans)
        cache.valid = True
        return ModelOutput(seq, cache, lengths, events, attention_pairs)

    @torch.inference_mode()
    def next_logits(self, ids, **kwargs):
        output = self(ids, **kwargs)
        if int(output.sequence.source[-1]) < 0 or int(output.sequence.actions[-1]) != TokenAction.KEEP:
            raise ValueError("next-token predictor must be an original KEEP boundary")
        return self.lm_head(output.sequence.hidden[-1]), output


def load_model(path, compression, device="cuda:0", dtype="bfloat16"):
    import transformers

    if transformers.__version__ != "5.2.0":
        raise RuntimeError("Agent runtime requires Transformers 5.2.0; run Clef separately")
    native = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=getattr(torch, dtype),
        attn_implementation="sdpa", device_map={"": device},
    )
    return AgentModel(native.model.language_model, native.lm_head, compression)
