import torch

import vl_saliency.viz.tokens as tokens
from vl_saliency.viz.tokens import render_token_ids


class DummyTokenizer:
    def __init__(self, id2tok, all_special_ids=()):
        self.id2tok = dict(id2tok)
        self.all_special_ids = list(all_special_ids)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [self.id2tok[i] for i in ids if i not in self.all_special_ids]


class DummyProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_returns_html_and_contains_tokens_and_titles_for_1d_and_gen_start():
    # ids -> tokens (index 0 = prompt, rest generated)
    id2tok = {10: "Hello", 11: "world", 12: "!"}
    proc = DummyProcessor(DummyTokenizer(id2tok))

    ids = torch.tensor([10, 11, 12])
    out = render_token_ids(ids, proc, gen_start=1, return_html=True)

    # tokens present
    assert "Hello" in out and "world" in out and "!" in out
    # token ids present
    assert "10" in out and "11" in out and "12" in out


def test_handles_2d_input_and_skip_tokens_int():
    id2tok = {1: "AAAA", 2: "BBBB", 3: "CCCC"}
    proc = DummyProcessor(DummyTokenizer(id2tok))
    ids = torch.tensor([[1, 2, 3]])  # 2D input path

    out = render_token_ids(ids, proc, skip_tokens=2, return_html=True)
    assert "AAAA" in out and "CCCC" in out
    assert "BBBB" not in out  # skipped


def test_skip_tokens_sequence_and_special_tokens_included():
    SPECIAL = 123456789
    id2tok = {5: "foo", 6: "bar", SPECIAL: "[PAD]"}
    tok = DummyTokenizer(id2tok, all_special_ids=[SPECIAL])
    proc = DummyProcessor(tok)

    ids = torch.tensor([5, 6, SPECIAL])
    out = render_token_ids(ids, proc, skip_tokens=[6], return_html=True)

    assert "foo" in out
    assert "bar" not in out  # skipped
    assert str(SPECIAL) not in out


def test_newline_markers_insert_line_break():
    id2tok = {7: "\\n", 8: "Next"}
    proc = DummyProcessor(DummyTokenizer(id2tok))

    out = render_token_ids(torch.tensor([7, 8]), proc, return_html=True)
    assert "<br>" in out
    assert "Next" in out


def test_space_marker_token_keeps_prefix_and_rest():
    # leading space marker "▁" should render prefix char and remainder text
    id2tok = {1: "▁world"}
    proc = DummyProcessor(DummyTokenizer(id2tok))

    out = render_token_ids(torch.tensor([1]), proc, return_html=True)
    assert "world" in out
    assert "▁" in out  # prefix character present somewhere in HTML


def test_print_fallback_when_return_html_false(monkeypatch, capsys):
    monkeypatch.setattr(tokens, "display", None)
    monkeypatch.setattr(tokens, "HTML", None)

    id2tok = {1: "Hello"}
    proc = DummyProcessor(DummyTokenizer(id2tok))

    ret = render_token_ids(torch.tensor([1]), proc, return_html=False)
    assert ret is None

    printed = capsys.readouterr().out
    assert "Hello" in printed
    assert "<div" in printed and "</div>" in printed
