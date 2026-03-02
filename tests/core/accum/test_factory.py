import pytest
import torch

import vl_saliency.core.accum.factory as m
from vl_saliency.core.accum.base import SaliencyAccumulator
from vl_saliency.core.accum.subset import HeadAccumulator, LayerAccumulator
from vl_saliency.types import HeadSelect, LayerSelect


class DummyAccum:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        return (
            isinstance(other, DummyAccum)
            and self.args == other.args
            and self.kwargs == other.kwargs
        )


@pytest.mark.parametrize(
    ["subset_select", "expected_class"],
    [
        ([0, 2], LayerAccumulator),
        (LayerSelect(layers=[0, 2]), LayerAccumulator),
        ({0: [0, 1], 1: [2]}, HeadAccumulator),
        (HeadSelect(heads={0: [0, 1], 1: [2]}), HeadAccumulator),
        (None, SaliencyAccumulator),
    ],
)
def test_build_accumulator(monkeypatch, subset_select, expected_class, build_config):
    monkeypatch.setattr(
        m, expected_class.__name__, lambda *args, **kwargs: DummyAccum(*args, **kwargs)
    )

    input_ids = torch.tensor([[1, 2, 3]])
    config = build_config(subset_select=subset_select)

    acc = m.build_accumulator(config=config, input_ids=input_ids, pixel_values=None)
    assert isinstance(acc, DummyAccum)

    expected_acc = DummyAccum(config, input_ids=input_ids, pixel_values=None)
    assert acc == expected_acc
