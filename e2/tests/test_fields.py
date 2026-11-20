import pytest

from classification.types import FieldDecision, TokenAction as A
from observation.alignment import align_observation
from observation.fields import split_observation


@pytest.mark.parametrize("source,format_name,paths", [
    (' {"name":"小明", "data":{"id":9007199254740993,"ok":true},"rows":[{"a":1,"b":2},{}]} \n',
     "json", [("name",), ("data", "id"), ("data", "ok"), ("rows", 0), ("rows", 1)]),
    ('{"empty":{},"values":[]}', "json", [("empty",), ("values",)]),
    ('{"duplicate":1,"duplicate":2}', "json", [("duplicate",), ("duplicate",)]),
    ('\nfirst\n\n第二行\n\n', "text", [(2,), (4,)]),
    ('{"broken":\n line', "text", [(1,), (2,)]),
    ('123456789012345678901234567890', "json", [()]),
    (' \n\t', "text", [()]),
    ('', "text", []),
])
def test_lossless_partition(source, format_name, paths):
    segmented = split_observation(source)
    assert segmented.format == format_name
    assert [field.path for field in segmented.fields] == paths
    assert "".join(field.text for field in segmented.fields) == source
    cursor = 0
    for field in segmented.fields:
        assert field.start == cursor
        assert field.text == source[field.start:field.end]
        cursor = field.end
    assert cursor == len(source)


def test_clean_json_values_preserve_exact_lexemes_and_records():
    raw = (' {"large":900719925474099312345,"precise":1.2300e-40,'
           '"nested":{"value":"逗号,和括号[]{}"},'
           '"rows":[{"dup":1,"dup":2,"password":"\\u0041\\n"},null],"empty":[]}\n')
    segmented = split_observation(raw)
    assert [f.value_text for f in segmented.fields] == [
        '900719925474099312345', '1.2300e-40', '"逗号,和括号[]{}"',
        '{"dup":1,"dup":2,"password":"\\u0041\\n"}', 'null', '[]',
    ]
    assert ''.join(f.text for f in segmented.fields) == raw


def test_clean_text_values_exclude_neighboring_blank_lines():
    segmented = split_observation('\nfirst\n\nsecond\n\n')
    assert [f.value_text for f in segmented.fields] == ['first\n', 'second\n']
    assert ''.join(f.text for f in segmented.fields) == segmented.source


class BoundaryTokenizer:
    """One real token can straddle two fields, or be a literal special token."""
    all_special_ids = [99]

    def __call__(self, text, **kwargs):
        assert text == "ab\ncd"
        return {"input_ids": [1, 2, 99], "offset_mapping": [(0, 2), (2, 4), (4, 5)]}


def test_shared_tokens_and_specials_stay_explicit():
    segmented = split_observation("ab\ncd")
    decisions = [FieldDecision(0, A.MERGE, "irrelevant"), FieldDecision(1, A.DELETE, "unused")]
    encoded = align_observation(segmented, decisions, BoundaryTokenizer())
    assert encoded.actions == [A.MERGE, A.KEEP, A.KEEP]
    assert encoded.field_ids == [[0], [0, 1], [1]]
    with pytest.raises(ValueError, match="every field"):
        align_observation(segmented, decisions[:1], BoundaryTokenizer())
