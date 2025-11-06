import pytest
import torch

from vl_saliency.transforms.layers import (
    Aggregate,
    FirstNLayers,
    LastNLayers,
    SelectHeads,
    SelectLayers,
)

# --------------- Layer Selection ---------------


def test_select_layers(smap):
    t = SelectLayers([0, 2])
    out = smap >> t
    expected = smap.tensor()[[0, 2]]
    assert torch.equal(out.tensor(), expected)

    t = SelectLayers(1)
    out = smap >> t
    expected = smap.tensor()[[1]]
    assert torch.equal(out.tensor(), expected)


def test_select_heads(smap):
    t = SelectHeads([(0, 1), (2, 0)])
    out = smap >> t
    expected = smap.tensor()[
        [0, 2],
        [1, 0],
    ].unsqueeze(0)  # add layer dim back
    assert torch.equal(out.tensor(), expected)

    t = SelectHeads((1, 0))
    out = smap >> t
    expected = smap.tensor()[
        [1],
        [0],
    ].unsqueeze(0)  # add layer dim back
    assert torch.equal(out.tensor(), expected)


def test_select_first_layers(smap):
    t = FirstNLayers(2)
    out = smap >> t
    expected = smap.tensor()[:2]
    assert torch.equal(out.tensor(), expected)


def test_select_last_layers(smap):
    t = LastNLayers(2)
    out = smap >> t
    expected = smap.tensor()[-2:]
    assert torch.equal(out.tensor(), expected)


# --------------- Aggregate ---------------


@pytest.mark.parametrize(
    "reduce",
    [
        ("mean", torch.mean),
        ("sum", torch.sum),
        ("max", torch.amax),
        ("min", torch.amin),
        ("prod", torch.prod),
    ],
)
def test_aggregate_layers_heads(smap, reduce):
    reduce_name, reduce_fn = reduce

    # Aggregate over layers
    t = Aggregate(layer_reduce=reduce_name, head_reduce=None)
    out = smap >> t
    expected = reduce_fn(smap.tensor(), dim=0, keepdim=True)
    assert torch.equal(out.tensor(), expected)

    # Aggregate over heads
    t = Aggregate(layer_reduce=None, head_reduce=reduce_name)
    out = smap >> t
    expected = reduce_fn(smap.tensor(), dim=1, keepdim=True)
    assert torch.equal(out.tensor(), expected)

    # Aggregate over both layers and heads
    t = Aggregate(layer_reduce=reduce_name, head_reduce=reduce_name)
    out = smap >> t
    expected = reduce_fn(reduce_fn(smap.tensor(), dim=0, keepdim=True), dim=1, keepdim=True)
    assert torch.equal(out.tensor(), expected)


def test_aggregate_noop(smap):
    t = Aggregate(layer_reduce=None, head_reduce=None)
    out = smap >> t
    expected = smap.tensor()
    assert torch.equal(out.tensor(), expected)
