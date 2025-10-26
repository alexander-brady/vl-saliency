import pytest


class DummyTokenizer:
    pad_token_id = 0


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


@pytest.fixture
def dummy_processor():
    return DummyProcessor()
