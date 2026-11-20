"""Tokenize the original observation once, using the Agent tokenizer."""

from dataclasses import dataclass

from classification.types import FieldDecision, TokenAction
from observation.fields import SegmentedObservation


@dataclass
class TokenizedObservation:
    ids: list[int]
    offsets: list[tuple[int, int]]
    field_ids: list[list[int]]
    actions: list[TokenAction]


def align_observation(
    segmented: SegmentedObservation, decisions: list[FieldDecision], tokenizer
) -> TokenizedObservation:
    by_id = {decision.field_id: decision.action for decision in decisions}
    if len(by_id) != len(decisions) or set(by_id) != {f.id for f in segmented.fields}:
        raise ValueError("classifier must decide every field exactly once")
    encoded = tokenizer(
        segmented.source, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    actions, owners, cursor = [], [], 0
    special = set(tokenizer.all_special_ids)
    for token_id, (start, end) in zip(encoded["input_ids"], offsets, strict=True):
        while cursor < len(segmented.fields) and segmented.fields[cursor].end <= start:
            cursor += 1
        overlaps = []
        index = cursor
        while index < len(segmented.fields) and segmented.fields[index].start < end:
            if segmented.fields[index].end > start:
                overlaps.append(segmented.fields[index].id)
            index += 1
        choices = {by_id[index] for index in overlaps}
        # A shared token with conflicting field decisions stays explicit. This
        # rule also prevents partial-token deletion; it is not a content guard.
        action = choices.pop() if len(choices) == 1 else TokenAction.KEEP
        if token_id in special or end <= start:
            action = TokenAction.KEEP
        actions.append(action)
        owners.append(overlaps)
    return TokenizedObservation(encoded["input_ids"], offsets, owners, actions)
