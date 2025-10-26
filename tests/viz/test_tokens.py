import html

import pytest
import torch

import vl_saliency.viz.tokens as tokens
from vl_saliency.viz.tokens import render_token_ids


class DummyTokenizer:
    def __init__(self, id2tok, all_special_ids=()):
        self.id2tok = dict(id2tok)
        self.all_special_ids = list(all_special_ids)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [self.id2tok[i] for i in ids if i not in self.all_special_ids or not skip_special_tokens]


# ------------------------- Errors --------------------------


def test_raises_value_error_on_invalid_tensor_dim(dummy_processor):
    ids = torch.randn(2, 3, 4)  # 3D tensor, invalid

    with pytest.raises(ValueError, match="generated_ids must be a 1D or 2D tensor."):
        render_token_ids(ids, dummy_processor, return_html=True)


# ------------------------- Content -------------------------


def test_returns_html_and_contains_tokens_and_titles_for_1d_and_gen_start(dummy_processor):
    # ids -> tokens (index 0 = prompt, rest generated)
    id2tok = {10: "Hello", 11: "world", 12: "!"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    ids = torch.tensor([10, 11, 12])
    out = render_token_ids(ids, dummy_processor, gen_start=1, return_html=True)

    # tokens present
    assert "Hello" in out and "world" in out and "!" in out
    # token ids present
    assert "10" in out and "11" in out and "12" in out


def test_special_tokens(dummy_processor):
    token = "<assistant>"
    id2tok = {1: token, 10: "Hello", 11: "world"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok, all_special_ids=[1])

    ids = torch.tensor([1, 10, 11])
    out = render_token_ids(ids, dummy_processor, return_html=True)

    assert html.escape(token) in out


def test_handles_2d_input_and_skip_tokens_int(dummy_processor):
    id2tok = {1: "AAAA", 2: "BBBB", 3: "CCCC"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)
    ids = torch.tensor([[1, 2, 3]])  # 2D input path

    out = render_token_ids(ids, dummy_processor, skip_tokens=2, return_html=True)
    assert "AAAA" in out and "CCCC" in out
    assert "BBBB" not in out  # skipped


def test_skip_tokens_sequence_included(dummy_processor):
    id2tok = {5: "foo", 6: "bar"}
    tok = DummyTokenizer(
        id2tok,
    )
    dummy_processor.tokenizer = tok

    ids = torch.tensor([5, 6])
    out = render_token_ids(ids, dummy_processor, skip_tokens=[6], return_html=True)

    assert "foo" in out
    assert "bar" not in out  # skipped


def test_newline_markers_insert_line_break(dummy_processor):
    id2tok = {7: "\\n", 8: "Next"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    out = render_token_ids(torch.tensor([7, 8]), dummy_processor, return_html=True)
    assert "<br>" in out
    assert "Next" in out


def test_space_marker_token_keeps_prefix_and_rest(dummy_processor):
    # leading space marker "▁" should render prefix char and remainder text
    id2tok = {1: "▁world"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    out = render_token_ids(torch.tensor([1]), dummy_processor, return_html=True)
    assert "world" in out
    assert "▁" in out  # prefix character present somewhere in HTML


def test_only_number_generated_tokens(dummy_processor):
    id2tok = {7: "Hello", 8: "world", 9: "!"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    ids = torch.tensor([7, 8, 9])
    out = render_token_ids(ids, dummy_processor, gen_start=1, only_number_generated=True, return_html=True)

    assert "Index: 2" not in out  # prompt token not numbered, thus max token index is 2
    assert "Index: 1" in out  # generated token indexed 1
    assert "Index: 0" in out  # generated token indexed 0


# ------------------------- Display Tests -------------------------


def test_print_fallback_when_return_html_false(dummy_processor, monkeypatch, capsys):
    monkeypatch.setattr(tokens, "display", None)
    monkeypatch.setattr(tokens, "HTML", None)

    id2tok = {1: "Hello"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    ret = render_token_ids(torch.tensor([1]), dummy_processor, return_html=False)
    assert ret is None

    printed = capsys.readouterr().out
    assert "Hello" in printed
    assert "<div" in printed and "</div>" in printed


def test_displays(monkeypatch, dummy_processor):
    called = {}

    class FakeHTML(str):
        def __new__(cls, value):
            called["html"] = value
            return super().__new__(cls, f"<html>{value}</html>")

    def fake_display(obj):
        called["display"] = obj

    monkeypatch.setattr(tokens, "HTML", FakeHTML)
    monkeypatch.setattr(tokens, "display", fake_display)

    id2tok = {1: "Hello"}
    dummy_processor.tokenizer = DummyTokenizer(id2tok)

    ret = render_token_ids(torch.tensor([1]), dummy_processor, return_html=False)
    assert ret is None

    assert "html" in called
    assert "display" in called
    assert isinstance(called["display"], FakeHTML)
