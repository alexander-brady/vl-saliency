from jaxtyping import Float
from torch import Tensor


# TODO: Align with multiple images per input. Currently, all items in the batch must have one image for this to work.
class StaticPatches:
    """Returns static patch shapes for all images."""

    def __init__(self, height: int, width: int):
        self.patch_shape = (height, width)

    def __call__(self, batch_size: int, image_count: int, **kwargs) -> list[list[tuple[int, int]]]:
        # For now: assuming each batch item has one image.
        return [[self.patch_shape] for _ in range(batch_size)]


def image_thw_to_patches(
    batch_size: int, image_count: int, image_grid_thw: Float[Tensor, "B 3"], **kwargs
) -> list[list[tuple[int, int]]]:
    """Convert image grid sizes from (T, H, W) format to patch shapes."""

    patches = (image_grid_thw[:, 1:] // 2).tolist()  # Each row -> [H, W]
    if len(patches) != image_count:
        raise ValueError(
            f"Number of image grid sizes ({len(patches)}) does not match number of images ({image_count})."
        )
    return [[tuple(patch)] for patch in patches]
