import html

import vl_saliency.viz.tokens as tokens
from vl_saliency.viz.tokens import render_token_ids

# ------- Content -------


def test_returns_html_and_contains_tokens_and_titles_for_1d_and_gen_start(dummy_tokenizer):
    # ids -> tokens (index 0 = prompt, rest generated)

    ids = [10, 11, 12]
    out = render_token_ids(ids, dummy_tokenizer, gen_start=1, return_html=True)

    # tokens present
    assert "token_10" in out and "token_11" in out and "token_12" in out
    # token ids present
    assert "10" in out and "11" in out and "12" in out


def test_special_tokens(dummy_tokenizer):
    token = "<assistant>"

    dummy_tokenizer.all_special_ids = [1]  # Special token IDs
    dummy_tokenizer.id2tok = {1: token}  # Map special token ID to string

    ids = [1, 10, 11]
    out = render_token_ids(ids, dummy_tokenizer, return_html=True)

    assert html.escape(token) in out


def test_skip_tokens_int(dummy_tokenizer):
    ids = [1, 2, 3]

    out = render_token_ids(ids, dummy_tokenizer, skip_tokens=2, return_html=True)
    assert "token_1" in out and "token_3" in out
    assert "token_2" not in out  # skipped


def test_skip_tokens_sequence_included(dummy_tokenizer):
    ids = [5, 6]
    out = render_token_ids(ids, dummy_tokenizer, skip_tokens=[6], return_html=True)

    assert "token_5" in out
    assert "token_6" not in out  # skipped


def test_newline_markers_insert_line_break(dummy_tokenizer):
    dummy_tokenizer.id2tok = {7: "\\n", 8: "Next"}

    out = render_token_ids([7, 8], dummy_tokenizer, return_html=True)
    assert "<br>" in out
    assert "Next" in out


def test_space_marker_token_keeps_prefix_and_rest(dummy_tokenizer):
    # leading space marker "▁" should render prefix char and remainder text
    dummy_tokenizer.id2tok = {1: "▁world"}

    out = render_token_ids([1], dummy_tokenizer, return_html=True)
    assert "world" in out
    assert "▁" in out  # prefix character present somewhere in HTML


def test_only_number_generated_tokens(dummy_tokenizer):
    dummy_tokenizer.id2tok = {7: "Hello", 8: "world", 9: "!"}

    ids = [7, 8, 9]
    out = render_token_ids(
        ids, dummy_tokenizer, gen_start=1, only_number_generated=True, return_html=True
    )

    assert "Index: 2" not in out  # prompt token not numbered, thus max token index is 2
    assert "Index: 1" in out  # generated token indexed 1
    assert "Index: 0" in out  # generated token indexed 0


# ------- Display Tests -------


def test_print_fallback_when_return_html_false(dummy_tokenizer, monkeypatch, capsys):
    monkeypatch.setattr(tokens, "display", None)
    monkeypatch.setattr(tokens, "HTML", None)

    dummy_tokenizer.id2tok = {1: "Hello"}
    ret = render_token_ids([1], dummy_tokenizer, return_html=False)
    assert ret is None

    printed = capsys.readouterr().out
    assert "Hello" in printed
    assert "<div" in printed and "</div>" in printed


def test_displays(monkeypatch, dummy_tokenizer):
    called = {}

    class FakeHTML(str):
        def __new__(cls, value):
            called["html"] = value
            return super().__new__(cls, f"<html>{value}</html>")

    def fake_display(obj):
        called["display"] = obj

    monkeypatch.setattr(tokens, "HTML", FakeHTML)
    monkeypatch.setattr(tokens, "display", fake_display)

    dummy_tokenizer.id2tok = {1: "Hello"}

    ret = render_token_ids([1], dummy_tokenizer, return_html=False)
    assert ret is None

    assert "html" in called
    assert "display" in called
    assert isinstance(called["display"], FakeHTML)
