"""Physical order is causal; position retains the original RoPE coordinate."""

from dataclasses import dataclass, replace
import torch

from classification.types import TokenAction


@dataclass
class Sequence:
    hidden: torch.Tensor
    position: torch.Tensor
    source: torch.Tensor       # original logical index; -1 for a soft token
    span: torch.Tensor         # observation number; -1 outside observations
    actions: torch.Tensor

    @classmethod
    def create(cls, hidden, span, actions, offset=0):
        position = torch.arange(offset, offset + len(hidden), device=hidden.device)
        return cls(hidden, position, position.clone(), span, actions).validate()

    def with_hidden(self, hidden):
        return replace(self, hidden=hidden)

    def validate(self):
        n = len(self.hidden)
        for name in ("position", "source", "span", "actions"):
            value = getattr(self, name)
            if value.shape != (n,) or value.device != self.hidden.device:
                raise ValueError("inconsistent sequence metadata")
        if bool(((self.actions < 0) | (self.actions > 2)).any()):
            raise ValueError("unknown token action")
        if bool(((self.span < 0) & (self.actions != TokenAction.KEEP)).any()):
            raise ValueError("only observation tokens may be compressed")
        if bool((self.position[1:] < self.position[:-1]).any()):
            raise ValueError("logical positions must be monotone")
        original = self.source[self.source >= 0]
        if bool((original[1:] <= original[:-1]).any()):
            raise ValueError("original token order changed")
        return self

    def tool_ranges(self):
        values = self.span.tolist()
        ranges, seen, index = [], set(), 0
        while index < len(values):
            if values[index] < 0:
                index += 1
                continue
            tool, start = values[index], index
            while index < len(values) and values[index] == tool:
                index += 1
            if tool in seen or index == len(values):
                raise ValueError("observation must be contiguous and have a closing boundary")
            seen.add(tool)
            ranges.append((tool, start, index))
        return ranges
