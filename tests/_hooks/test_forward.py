import pytest

import vl_saliency._hooks.forward as m
from tests.utils import DummyOutput
from vl_saliency._hooks.forward import build_saliency_forward
from vl_saliency.output import SaliencyOutput

# ------- Dummmy Helpers and Fixtures -----


def dummy_forward(*args, **kwargs):
    return DummyOutput(dummy="forward_output")


class DummyAccumulator:
    def __init__(self, *args, saliency="generated_saliency", **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.saliency = saliency


@pytest.fixture
def build_accumulator(monkeypatch):
    def _build_accumulator(saliency="generated_saliency"):
        return DummyAccumulator(saliency=saliency)

    return _build_accumulator


@pytest.fixture(autouse=True)
def patch_accumulator(monkeypatch):
    monkeypatch.setattr(m, "build_accumulator", lambda **kwargs: DummyAccumulator(**kwargs))


# ------- Tests -----


def test_build_forward_passes_trace(build_model, build_accumulator, build_config):
    model = build_model(return_dict=True)

    forward = build_saliency_forward(
        config=build_config(), attn_implementation="attn_impl", forward=dummy_forward
    )

    out = forward(
        model_self=model,
        input_ids="input_ids",
        pixel_values="pixel_values",
        saliency=build_accumulator(saliency="passed_saliency"),
        extra_arg="extra",
    )

    assert out.dummy == "forward_output"
    assert out.saliency == "passed_saliency"


@pytest.mark.parametrize(
    ["model_return_dict", "forward_return_dict", "expected_type"],
    [
        (True, None, SaliencyOutput),
        (True, False, tuple),
        (True, True, SaliencyOutput),
        (False, None, tuple),
        (False, False, tuple),
        (False, True, SaliencyOutput),
    ],
)
def test_build_forward_respects_return_dict(
    build_model,
    model_return_dict,
    forward_return_dict,
    expected_type,
    build_accumulator,
    build_config,
):
    model = build_model(return_dict=model_return_dict)

    forward = build_saliency_forward(
        config=build_config(), attn_implementation="attn_impl", forward=dummy_forward
    )

    out = forward(
        model_self=model,
        input_ids="input_ids",
        pixel_values="pixel_values",
        return_dict=forward_return_dict,
        extra_arg="extra",
    )

    assert isinstance(out, expected_type)
    if expected_type == SaliencyOutput:
        assert out.dummy == "forward_output"
        assert out.saliency == "generated_saliency"
    else:
        assert out[0] == "forward_output"
        assert out[1] == "generated_saliency"
