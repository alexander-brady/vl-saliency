import pytest

from vl_saliency.config import SaliencyConfig
from vl_saliency.utils.patch_fns import StaticPatches


@pytest.fixture
def build_config():
    def _mk(**overrides) -> SaliencyConfig:
        base = dict(
            pad_token_id=0,
            image_token_id=1,
            image_patch_fn=StaticPatches(16, 16),
            layer_reduce="mean",
            layer_op=None,
            head_reduce="mean",
            head_op=None,
            backend="auto",
        )
        base.update(overrides)
        return SaliencyConfig(**base)  # type: ignore

    return _mk
