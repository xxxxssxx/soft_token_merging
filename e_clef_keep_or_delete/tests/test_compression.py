import torch

from classification.types import TokenAction as A
from compression.operator import CompressionConfig, compress
from compression.sequence import Sequence


def test_three_actions_preserve_sources_positions_and_keep_hidden():
    hidden = torch.tensor([[8., 8.], [1., 0.], [9., 9.], [1., 0.], [3., 4.], [5., 6.]])
    seq = Sequence.create(hidden, torch.tensor([0, 0, 0, 0, 0, -1]),
                          torch.tensor([A.KEEP, A.MERGE, A.DELETE, A.MERGE, A.KEEP, A.KEEP]))
    result, events = compress(seq, CompressionConfig(slots=1, min_layer_savings=0), 21)
    assert result.source.tolist() == [0, -1, 4, 5]
    assert result.position.tolist() == [0, 3, 4, 5]
    assert result.actions.tolist() == [A.KEEP] * 4
    assert torch.equal(result.hidden[[0, 2, 3]], hidden[[0, 4, 5]])
    assert torch.equal(result.hidden[1], torch.tensor([1., 0.]))
    assert events[0]["soft_groups"] == [{"position": 3, "members": [1, 3]}]
    assert events[0]["deleted_sources"] == [2]


def test_merge_budget_does_not_cancel_deletions():
    seq = Sequence.create(torch.ones(5, 2), torch.tensor([0, 0, 0, 0, -1]),
                          torch.tensor([A.MERGE, A.DELETE, A.MERGE, A.KEEP, A.KEEP]))
    result, events = compress(seq, CompressionConfig(), 21)
    assert result.source.tolist() == [0, 2, 3, 4]
    assert events[0]["merged_tokens"] == 0
    assert events[0]["deleted_tokens"] == 1


def test_rejected_merge_stays_hard():
    seq = Sequence.create(torch.eye(3), torch.tensor([0, 0, -1]),
                          torch.tensor([A.MERGE, A.MERGE, A.KEEP]))
    result, events = compress(seq, CompressionConfig(slots=1, merge_error=0.01, min_layer_savings=0), 21)
    assert torch.equal(result.source, seq.source)
    assert torch.equal(result.hidden, seq.hidden)
    assert events[0]["soft_tokens"] == 0
