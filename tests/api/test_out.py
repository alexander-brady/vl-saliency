import pytest

from tests.utils import DummyOutput
from vl_saliency.api.out import SaliencyOutput


def test_output_forwards_dataclass_fields(dummy_sal_grid):
    base = DummyOutput(dummy="out")
    out = SaliencyOutput(base_output=base, saliency=dummy_sal_grid)
    assert out.dummy == "out"
    assert out.saliency is dummy_sal_grid

    with pytest.raises(AttributeError):
        _ = out.nonexistent_field
