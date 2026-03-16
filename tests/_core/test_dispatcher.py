import sys
import types

import pytest
import torch

import vl_saliency._core.dispatcher as m

# ------ Fixtures for testing ------


@pytest.fixture(autouse=True)
def clear_cache():
    m.get_qk_accumulator.cache_clear()
    m._is_triton_available.cache_clear()


@pytest.fixture
def triton_available(monkeypatch):
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language  # type: ignore[assignment]

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)


# -------Test cases for get_saliency_qk -----


def test_torch_backend(monkeypatch):
    # Mock the compiled function to test that it's returned
    monkeypatch.setattr(m, "saliency_qk_compiled", lambda *args, **kwargs: "compiled")
    fn = m.get_qk_accumulator(
        "torch", head_reduce="mean", layer_reduce="mean", head_op=None, layer_op=None
    )
    assert fn == "compiled"


def test_triton_backend(triton_available, monkeypatch):
    monkeypatch.setattr(m, "saliency_qk_triton", lambda *args, **kwargs: "triton")
    fn = m.get_qk_accumulator(
        "triton", head_reduce="mean", layer_reduce="mean", head_op=None, layer_op=None
    )
    assert fn == "triton"


def test_torch_eager_backend(monkeypatch):
    monkeypatch.setattr(m, "saliency_qk_eager", lambda *args, **kwargs: "eager")
    fn = m.get_qk_accumulator(
        "torch_eager", head_reduce="mean", layer_reduce="mean", head_op=None, layer_op=None
    )
    assert fn == "eager"

    # Test that the eager function is returned when compilation fails
    monkeypatch.setattr(
        m, "saliency_qk_compiled", lambda *args, **kwargs: (_ for _ in ()).throw(Exception())
    )
    fn = m.get_qk_accumulator(
        "torch", head_reduce="mean", layer_reduce="mean", head_op=None, layer_op=None
    )
    assert fn == "eager"


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        m.get_qk_accumulator("<<<INVALID>>>")  # type: ignore


# ------- Test cases for assign_auto -------


def test_torch_auto_backend(monkeypatch):
    monkeypatch.setattr(m, "assign_auto", lambda device, head_reduce: "torch_eager")
    monkeypatch.setattr(m, "saliency_qk_eager", lambda *args, **kwargs: "eager")
    fn = m.get_qk_accumulator(
        "auto", head_reduce="mean", layer_reduce="mean", head_op=None, layer_op=None
    )
    assert fn == "eager"


def test_auto_selects_triton(monkeypatch):
    monkeypatch.setattr(m, "_is_triton_available", lambda: True)
    assert m.assign_auto(torch.device("cuda"), head_reduce="sum") == "triton"

    # Test that head_reduce affects selection
    assert m.assign_auto(torch.device("cuda"), head_reduce="prod") == "torch"


def test_auto_selects_torch(monkeypatch):
    monkeypatch.setattr(m, "_is_triton_available", lambda: False)
    assert m.assign_auto(torch.device("cuda"), head_reduce="sum") == "torch"
    assert m.assign_auto(torch.device("cpu"), head_reduce="sum") == "torch_eager"


# ------- Test cases for _is_triton_available -------


def test_is_triton_available_import_error(monkeypatch):
    monkeypatch.setattr("builtins.__import__", lambda *a, **k: (_ for _ in ()).throw(ImportError()))
    assert m._is_triton_available() is False


def test_is_triton_available_no_cuda(monkeypatch, triton_available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert m._is_triton_available() is False


def test_is_triton_available_no_device_capability(monkeypatch, triton_available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda: (_ for _ in ()).throw(Exception())
    )
    assert m._is_triton_available() is False


def test_is_triton_available_success(monkeypatch, triton_available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    assert m._is_triton_available() is True
