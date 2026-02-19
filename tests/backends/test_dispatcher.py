import sys
import types

import pytest
import torch

import vl_saliency.backends.dispatcher as m

# ------ Fixtures for testing ------


@pytest.fixture(autouse=True)
def clear_cache():
    m.get_saliency_qk.cache_clear()
    m._is_triton_available.cache_clear()


@pytest.fixture
def triton_available(monkeypatch):
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language  # type: ignore[assignment]

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)


# ----- Test cases for get_saliency_qk -----


def test_torch_backend(monkeypatch):
    monkeypatch.setattr(m, "saliency_qk_torch", lambda: "compiled")
    fn = m.get_saliency_qk("torch", torch.device("cuda"))
    assert fn == "compiled"


def test_triton_backend(triton_available):
    fn = m.get_saliency_qk("triton", torch.device("cuda"))
    assert fn is m.saliency_qk_triton


def test_torch_eager_backend():
    fn = m.get_saliency_qk("torch_eager", torch.device("cpu"))
    assert fn is m.saliency_qk_torch_eager


def test_auto_selects_triton(monkeypatch):
    monkeypatch.setattr(m, "_is_triton_available", lambda: True)
    fn = m.get_saliency_qk("auto", torch.device("cuda"))
    assert fn is m.saliency_qk_triton


def test_auto_selects_torch(monkeypatch):
    monkeypatch.setattr(m, "_is_triton_available", lambda: False)
    monkeypatch.setattr(m, "saliency_qk_torch", lambda: "compiled")
    fn = m.get_saliency_qk("auto", torch.device("cuda"))
    assert fn == "compiled"


def test_auto_selects_eager_on_cpu():
    fn = m.get_saliency_qk("auto", torch.device("cpu"))
    assert fn is m.saliency_qk_torch_eager


def test_invalid_backend():
    with pytest.raises(ValueError):
        m.get_saliency_qk("invalid", torch.device("cpu"))


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
