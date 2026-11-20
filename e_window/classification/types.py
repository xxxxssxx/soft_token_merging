"""The only decision vocabulary consumed by the compression runtime."""

from dataclasses import dataclass, field
from enum import IntEnum

from observation.fields import SegmentedObservation


class TokenAction(IntEnum):
    KEEP = 0
    MERGE = 1
    DELETE = 2


@dataclass(frozen=True)
class ClassificationRequest:
    task: str
    history: list[dict]
    action: str
    observation: SegmentedObservation
    agent_system_prompt: str


@dataclass(frozen=True)
class FieldDecision:
    field_id: int
    action: TokenAction
    label: str
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    decisions: list[FieldDecision]
    metadata: dict = field(default_factory=dict)
