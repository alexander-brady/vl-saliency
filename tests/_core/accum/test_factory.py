import pytest
import torch

import vl_saliency._core.accum.factory as m
from vl_saliency._core.accum.base import SaliencyAccumulator
from vl_saliency._core.accum.subset import HeadAccumulator, LayerAccumulator
from vl_saliency.config import HeadSelect, LayerSelect


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
    ["selection", "expected_class"],
    [
        ([0, 2], LayerAccumulator),
        (LayerSelect(0, 2), LayerAccumulator),
        ({0: [0, 1], 1: [2]}, HeadAccumulator),
        (HeadSelect((0, 0), (0, 1), (1, 2)), HeadAccumulator),
        (None, SaliencyAccumulator),
    ],
)
def test_build_accumulator(monkeypatch, selection, expected_class, build_config):
    monkeypatch.setattr(
        m, expected_class.__name__, lambda *args, **kwargs: DummyAccum(*args, **kwargs)
    )

    input_ids = torch.tensor([[1, 2, 3]])
    config = build_config(selection=selection)

    acc = m.build_accumulator(config=config, input_ids=input_ids, pixel_values=None)
    assert isinstance(acc, DummyAccum)

    expected_acc = DummyAccum(config, input_ids=input_ids, pixel_values=None)
    assert acc == expected_acc
