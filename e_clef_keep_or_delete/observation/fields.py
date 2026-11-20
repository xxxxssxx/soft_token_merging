"""Adapted from /home/filter/roles/filter_utils.py.

Objects recurse to leaf values; array elements remain atomic records. Unlike the
text filter, fields here partition ALL source characters. A leaf owns the prefix
since the preceding leaf (keys, separators and opening containers); the last leaf
also owns the trailing containers/whitespace. No observation text is rewritten.
"""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ObservationField:
    id: int
    path: tuple[str | int, ...]
    start: int
    end: int
    text: str
    value_text: str  # complete leaf value, without surrounding keys/containers


@dataclass(frozen=True)
class SegmentedObservation:
    source: str
    format: str
    fields: list[ObservationField]


def _reject_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


class _JsonLeaves:
    def __init__(self, source):
        self.source = source
        self.decoder = json.JSONDecoder(
            parse_int=str, parse_float=str, parse_constant=_reject_constant
        )
        self.leaves = []

    def skip_space(self, position):
        while position < len(self.source) and self.source[position] in " \t\r\n":
            position += 1
        return position

    def parse(self, start, path=(), atomic=False):
        value, end = self.decoder.raw_decode(self.source, start)
        if atomic or not isinstance(value, (dict, list)) or not value:
            self.leaves.append((path, start, end))
            return end
        position = self.skip_space(start + 1)
        if isinstance(value, dict):
            while self.source[position] != "}":
                key, key_end = self.decoder.raw_decode(self.source, position)
                colon = self.skip_space(key_end)
                position = self.parse(self.skip_space(colon + 1), path + (key,))
                position = self.skip_space(position)
                if self.source[position] == ",":
                    position = self.skip_space(position + 1)
        else:
            index = 0
            while self.source[position] != "]":
                position = self.parse(position, path + (index,), atomic=True)
                index += 1
                position = self.skip_space(position)
                if self.source[position] == ",":
                    position = self.skip_space(position + 1)
        return end


def split_observation(source: str) -> SegmentedObservation:
    if not source:
        return SegmentedObservation(source, "text", [])
    parser = _JsonLeaves(source)
    try:
        end = parser.parse(parser.skip_space(0))
        if parser.skip_space(end) != len(source):
            raise ValueError("trailing text")
        leaves, format_name = parser.leaves, "json"
    except (ValueError, IndexError, RecursionError):
        leaves, position = [], 0
        for line_number, line in enumerate(source.splitlines(keepends=True), 1):
            start = position
            position += len(line)
            if line.strip():
                leaves.append(((line_number,), start, position))
        leaves = leaves or [((), 0, len(source))]
        format_name = "text"

    fields, start = [], 0
    for index, (path, value_start, value_end) in enumerate(leaves):
        end = value_end
        if index == len(leaves) - 1:
            end = len(source)
        fields.append(ObservationField(
            index, path, start, end, source[start:end], source[value_start:value_end]
        ))
        start = end
    return SegmentedObservation(source, format_name, fields)
