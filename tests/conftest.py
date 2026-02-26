from typing import Any

import pytest
from transformers import PreTrainedConfig

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.utils.patch_fns import FixedPatchLayout

# ------- Configuration -----


@pytest.fixture
def build_config():
    def _mk(**overrides) -> SaliencyConfig:
        base = dict[str, Any](
            pad_token_id=0,
            image_token_id=1,
            image_patch_fn=FixedPatchLayout(16, 16),
            layer_reduce="mean",
            layer_op=None,
            head_reduce="mean",
            head_op=None,
            backend="auto",
            attn_scale=0.25,
        )
        base.update(overrides)
        return SaliencyConfig(**base)

    return _mk


# -------Models -------


class DummyModelConfig(PreTrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyModel:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def build_model_config():
    def _mk(**overrides) -> DummyModelConfig:
        return DummyModelConfig(**overrides)

    return _mk


@pytest.fixture
def build_model(build_model_config):
    def _mk(**config_overrides) -> DummyModel:
        config = build_model_config(**config_overrides)
        return DummyModel(config)

    return _mk


# -------Saliency grid -----


class DummySaliencyGrid:
    def __init__(self, data):
        self.data = data

    def map(self, *args, **kwargs):
        return self.data


@pytest.fixture
def dummy_sal_grid():
    return DummySaliencyGrid(data="dummy")
