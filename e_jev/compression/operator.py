"""Execute KEEP/MERGE/DELETE without knowing how the decisions were obtained."""

from dataclasses import dataclass, fields
import math
import torch

from classification.types import TokenAction
from compression.pooling import adaptive_pool
from compression.sequence import Sequence


@dataclass(frozen=True)
class CompressionConfig:
    layer: int = 10
    slots: int = 16
    merge_error: float = 0.50
    members_per_slot: int = 2
    min_layer_savings: int = 256
    min_tool_fraction: float = 0.05

    def __post_init__(self):
        if self.layer < 0 or self.slots < 1 or self.members_per_slot < 2:
            raise ValueError("invalid compression layer or group budget")
        if not math.isfinite(self.merge_error) or self.merge_error <= 0:
            raise ValueError("merge_error must be finite and positive")
        if self.min_layer_savings < 0 or not 0 <= self.min_tool_fraction <= 1:
            raise ValueError("invalid merge commit budget")


def compress(seq: Sequence, config: CompressionConfig, remaining_layers: int):
    parts = {field.name: [] for field in fields(seq)}
    events, cursor = [], 0

    def keep(indices):
        for name in parts:
            parts[name].append(getattr(seq, name)[indices])

    for tool, start, end in seq.tool_ranges():
        keep(slice(cursor, start))
        cursor = end
        actions = seq.actions[start:end]
        candidates = ((actions == TokenAction.MERGE) & (seq.source[start:end] >= 0))
        chosen = candidates.nonzero().flatten() + start
        deleted = (actions == TokenAction.DELETE).nonzero().flatten() + start
        minimum = max(
            1,
            math.ceil(config.min_layer_savings / remaining_layers),
            math.ceil(config.min_tool_fraction * len(chosen)),
        )
        members = chosen[:0]
        centers, endpoints = seq.hidden[:0], seq.position[:0]
        groups = chosen[:0]
        worst = None
        if len(chosen) >= minimum + 1:
            indices, groups, centers, endpoints, errors = adaptive_pool(
                seq.hidden[chosen], seq.position[chosen], config.slots,
                config.merge_error, config.members_per_slot,
            )
            members = chosen[indices]
            if len(members) - len(centers) < minimum:
                members, groups = members[:0], groups[:0]
                centers, endpoints = centers[:0], endpoints[:0]
            elif len(errors):
                worst = float(errors.max())

        retained = torch.ones(end - start, device=seq.hidden.device, dtype=torch.bool)
        retained[deleted - start] = False
        retained[members - start] = False
        keep(torch.arange(start, end, device=seq.hidden.device)[retained])
        if len(centers):
            parts["hidden"].append(centers)
            parts["position"].append(endpoints)
            parts["source"].append(seq.source.new_full((len(centers),), -1))
            parts["span"].append(seq.span.new_full((len(centers),), tool))
            # Existing soft tokens never enter another merge or deletion.
            parts["actions"].append(seq.actions.new_full((len(centers),), TokenAction.KEEP))
        events.append({
            "span": tool, "tokens_before": end - start,
            "keep_requested": int((actions == TokenAction.KEEP).sum()),
            "merge_requested": len(chosen), "delete_requested": len(deleted),
            "merged_tokens": len(members), "soft_tokens": len(centers),
            "deleted_tokens": len(deleted),
            "tokens_after": end - start - len(deleted) - len(members) + len(centers),
            "minimum_merge_saving": minimum, "max_merge_error": worst,
            "deleted_sources": seq.source[deleted].tolist(),
            "soft_groups": [
                {"position": int(endpoints[group]),
                 "members": seq.source[members[groups == group]].tolist()}
                for group in range(len(centers))
            ],
        })

    keep(slice(cursor, None))
    data = {name: torch.cat(values) for name, values in parts.items()}
    order = data["position"].argsort(stable=True)
    return Sequence(**{name: value[order] for name, value in data.items()}).validate(), events
