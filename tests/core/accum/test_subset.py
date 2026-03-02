import pytest
import torch

import vl_saliency.core.accum.base as m
from vl_saliency.core.accum.subset import HeadAccumulator, LayerAccumulator
from vl_saliency.types import HeadSelect, LayerSelect

# ------- Fixtures -------


@pytest.fixture
def input_ids():
    return torch.zeros(1, 4, dtype=torch.long)


@pytest.fixture
def qk():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 5, 8)  # B, Hq, T, D
    k = torch.randn(1, 4, 5, 8)  # B, Hkv, T, D
    return q, k


@pytest.fixture
def spy_accumulate(monkeypatch):
    calls = []

    def spy(self, q_, k_):
        calls.append(
            {
                "layer": self.current_layer,
                "q": q_.detach().clone(),
                "k": k_.detach().clone(),
            }
        )

    monkeypatch.setattr(m.SaliencyAccumulator, "accumulate_qk", spy)
    return calls


# ------- LayerAccumulator -------


@pytest.mark.parametrize("good_subset", [LayerSelect([0, 2]), [0, 2]])
def test_layer_accumulator_init(input_ids, build_config, good_subset):
    config = build_config(subset_select=good_subset)
    acc = LayerAccumulator(config, input_ids)

    assert acc.target_layers == {0, 2}
    assert acc.current_layer == 0


@pytest.mark.parametrize("bad_subset", [None, {0: [1, 2]}, HeadSelect({0: [1, 2]})])
def test_layer_accumulator_invalid_subset(input_ids, build_config, bad_subset):
    config = build_config(subset_select=bad_subset)
    with pytest.raises(ValueError):
        LayerAccumulator(config, input_ids)


def test_layer_accumulator_filters_layers(input_ids, qk, build_config, spy_accumulate):
    q, k = qk
    config = build_config(subset_select=[1, 3])
    acc = LayerAccumulator(config, input_ids)

    for _ in range(4):
        acc.accumulate_qk(q, k)

    assert [c["layer"] for c in spy_accumulate] == [1, 3]
    assert acc.current_layer == 4


def test_layer_accumulator_skips_non_target_layer(input_ids, qk, build_config, spy_accumulate):
    q, k = qk
    config = build_config(subset_select=[2])
    acc = LayerAccumulator(config, input_ids)

    acc.accumulate_qk(q, k)  # layer 0
    acc.accumulate_qk(q, k)  # layer 1

    assert spy_accumulate == []
    assert acc.current_layer == 2


# ------- HeadAccumulator -------


@pytest.mark.parametrize("good_subset", [HeadSelect({0: [1, 3]}), {0: [1, 3]}])
def test_head_accumulator_init(input_ids, build_config, good_subset):
    config = build_config(subset_select=good_subset)
    acc = HeadAccumulator(config, input_ids)

    expected = {0: torch.tensor([1, 3], device=input_ids.device)}
    assert acc.target_heads.keys() == expected.keys()
    for k in expected:
        assert torch.equal(acc.target_heads[k], expected[k])

    assert acc.current_layer == 0


@pytest.mark.parametrize("bad_subset", [[0, 1], None, LayerSelect([0, 1])])
def test_head_accumulator_invalid_subset(input_ids, build_config, bad_subset):
    config = build_config(subset_select=bad_subset)
    with pytest.raises(ValueError):
        HeadAccumulator(config, input_ids)


def test_head_accumulator_standard_heads(input_ids, qk, build_config, spy_accumulate):
    q, k = qk
    config = build_config(subset_select={0: [1, 3]})
    acc = HeadAccumulator(config, input_ids)

    acc.accumulate_qk(q, k)

    assert len(spy_accumulate) == 1
    call = spy_accumulate[0]

    expected_idx = torch.tensor([1, 3], device=q.device)
    assert torch.equal(call["q"], q.index_select(1, expected_idx))
    assert torch.equal(call["k"], k.index_select(1, expected_idx))


def test_head_accumulator_skips_missing_layer(input_ids, qk, build_config, spy_accumulate):
    q, k = qk
    config = build_config(subset_select={2: [0]})
    acc = HeadAccumulator(config, input_ids)

    acc.accumulate_qk(q, k)  # layer 0

    assert spy_accumulate == []
    assert acc.current_layer == 1


def test_head_accumulator_gqa(build_config, input_ids, spy_accumulate):
    torch.manual_seed(0)

    # Hq=4, Hkv=2 → group_size=2
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)

    config = build_config(subset_select={0: [2, 3]})
    acc = HeadAccumulator(config, input_ids)

    acc.accumulate_qk(q, k)

    assert len(spy_accumulate) == 1
    call = spy_accumulate[0]

    # q heads 2,3
    expected_q = q.index_select(1, torch.tensor([2, 3], device=q.device))

    # both map to kv head 1
    expected_k = k.index_select(
        1,
        torch.tensor([1, 1], device=k.device),
    )

    assert torch.equal(call["q"], expected_q)
    assert torch.equal(call["k"], expected_k)
