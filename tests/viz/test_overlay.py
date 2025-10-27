import matplotlib
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from vl_saliency.viz.overlay import overlay

matplotlib.use("Agg")


@pytest.fixture
def dummy_image():  # simple RGB PIL image
    return Image.new("RGB", (16, 16), color="white")


# ------------------------- Overlay -------------------------


def test_overlay_returns_figure_without_image(smap):
    fig = overlay(smap.agg(), image=None, show_colorbar=False)
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # should contain one QuadMesh (color plot)
    assert ax.images, "Expected saliency image to be plotted"


def test_overlay_with_image_and_colorbar(smap, dummy_image):
    fig = overlay(smap.agg(), image=dummy_image, show_colorbar=True)
    # should now have 2 axes: one for main plot, one for colorbar
    assert len(fig.axes) >= 1
    main_ax = fig.axes[0]
    assert main_ax.images, "Saliency overlay should be plotted"
    assert fig.axes[0].get_title() == "Saliency Map"


def test_overlay_uses_existing_ax(smap, dummy_image):
    fig, ax = plt.subplots()
    fig2 = overlay(smap.agg(), image=dummy_image, ax=ax, show_colorbar=False, title="custom")
    # Should not create a new Figure
    assert fig2 is fig
    assert ax.get_title() == "custom"


def test_overlay_custom_kwargs_passed_to_imshow(smap):
    fig = overlay(smap.agg(), show_colorbar=False, cmap="viridis", alpha=0.7)
    im = fig.axes[0].images[-1]
    # Check colormap and alpha
    assert im.get_cmap().name == "viridis"
    assert im.get_alpha() == 0.7
