# from dataclasses import dataclass

# import pytest
# import torch
# from torch import Tensor
# from transformers.utils.generic import ModelOutput

# from vl_saliency.core.grid import Index, SaliencyGrid

# # -------Fixtures -----


# @dataclass
# class DummyOutput(ModelOutput):
#     dummy: torch.Tensor


# @dataclass
# class DummyTokenLayout:
#     B: int
#     patch_shapes: list[list[tuple[int, int]]]  # [B][num_imgs] -> (H, W)
#     image_token_offsets: list[list[int]]  # [B][num_imgs] -> start offset in T_img
#     gen_mask: Tensor  # [B, T_gen] bool/int mask


# @pytest.fixture
# def layout():
#     def _mk(
#         *,
#         B: int,
#         patch_shapes: list[list[tuple[int, int]]],
#         image_token_offsets: list[list[int]],
#         gen_mask: Tensor,
#     ) -> DummyTokenLayout:
#         return DummyTokenLayout(
#             B=B,
#             patch_shapes=patch_shapes,
#             image_token_offsets=image_token_offsets,
#             gen_mask=gen_mask,
#         )

#     return _mk


# @pytest.fixture
# def tensor():
#     def _mk(*, B: int, T_gen: int, T_img: int) -> Tensor:
#         return torch.arange(B * T_gen * T_img, dtype=torch.float32).view(B, T_gen, T_img)

#     return _mk


# @pytest.fixture
# def mask():
#     def _mk(*, B: int, T_gen: list[int]) -> Tensor:
#         mask = torch.zeros(B, max(T_gen), dtype=torch.bool)
#         for i in range(B):
#             mask[i, : T_gen[i]] = True
#         return mask

#     return _mk


# # -------SaliencyGrid tests -----


# def test_saliency_grid_single_batch_single_image(layout, tensor, mask):
#     gen_mask = mask(B=1, T_gen=[6])
#     lay = layout(
#         B=1,
#         patch_shapes=[[(2, 2)]],  # 1 image with 2x2 patch
#         image_token_offsets=[[0]],  # image tokens start at offset 0
#         gen_mask=gen_mask,
#     )
#     t = tensor(B=1, T_gen=6, T_img=4)  # 1 batch, 6 gen tokens, 4 image tokens
#     grid = SaliencyGrid(t, lay)

#     assert grid.batch_size == 1
#     assert grid.num_images() == 1
#     assert grid.num_tokens() == 6

#     m = grid.map(4)
#     assert m.shape == (2, 2)
#     assert torch.equal(m, torch.tensor([[16.0, 17.0], [18.0, 19.0]]))
#     assert torch.equal(grid[4], m)

#     ims = grid.image_maps()
#     assert ims.shape == (6, 2, 2)
#     assert torch.equal(ims[4], m)

#     # Test validation
#     assert grid._validate_batch_idx(None) == 0
#     assert grid._validate_img_idx(0, None) == 0


# def test_saliency_grid_single_batch_multi_image(layout, tensor, mask):
#     gen_mask = mask(B=1, T_gen=[6])
#     lay = layout(
#         B=1,
#         patch_shapes=[[(2, 2), (1, 1)]],  # 2 images with 2x2 and 1x1 patches
#         image_token_offsets=[[0, 4]],  # image tokens start at offsets 0 and 4
#         gen_mask=gen_mask,
#     )
#     t = tensor(B=1, T_gen=6, T_img=5)  # 1 batch, 6 gen tokens, 5 image tokens
#     grid = SaliencyGrid(t, lay)

#     assert grid.batch_size == 1
#     assert grid.num_images() == 2
#     assert grid.num_tokens() == 6

#     m1 = grid.map(0, 4)
#     assert m1.shape == (2, 2)
#     assert torch.equal(m1, torch.tensor([[20.0, 21.0], [22.0, 23.0]]))
#     m2 = grid.map(1, 4)
#     assert m2.shape == (1, 1)

#     assert torch.equal(m2, torch.tensor([[24.0]]))
#     assert torch.equal(grid[0, 4], m1)
#     assert torch.equal(grid[1, 4], m2)

#     # Test validation
#     assert grid._validate_batch_idx(None) == 0
#     with pytest.raises(IndexError):
#         grid._validate_img_idx(0, None)


# def test_saliency_multiple_batches_and_images(layout, tensor, mask):
#     gen_mask = mask(B=2, T_gen=[6, 4])
#     lay = layout(
#         B=2,
#         patch_shapes=[
#             [(2, 2)],
#             [(1, 1), (1, 2)],
#         ],  # 2 batches, batch 0 has 1 image with 2x2 patch, batch 1 has 2 images with 1x1 and 1x2 patches
#         image_token_offsets=[[0], [0]],  # image tokens start at offset 0 for both
#         gen_mask=gen_mask,
#     )
#     t = tensor(B=2, T_gen=6, T_img=4)  # 2 batches, max 6 gen tokens, 4 image tokens
#     grid = SaliencyGrid(t, lay)

#     assert grid.batch_size == 2
#     assert grid.num_images(0) == 1
#     assert grid.num_images(1) == 2
#     assert grid.num_tokens(0) == 6
#     assert grid.num_tokens(1) == 4

#     m00 = grid.map(0, 0, 4)  # batch 0, image 0, token 4
#     assert m00.shape == (2, 2)
#     assert torch.equal(m00, torch.tensor([[16.0, 17.0], [18.0, 19.0]]))

#     m10 = grid.map(1, 0, 3)  # batch 1, image 0, token 3
#     assert m10.shape == (1, 1)
#     assert torch.equal(m10, torch.tensor([[36.0]]))

#     assert torch.equal(grid[0, 0, 4], m00)
#     assert torch.equal(grid[1, 0, 3], m10)

#     # Test validation
#     with pytest.raises(IndexError):
#         grid._validate_batch_idx(None)
#     with pytest.raises(IndexError):
#         grid._validate_img_idx(0, None)


# def test_saliency_grid_invalid_indices(layout, tensor, mask):
#     gen_mask = mask(B=1, T_gen=[6])
#     lay = layout(
#         B=1,
#         patch_shapes=[[(2, 2)]],
#         image_token_offsets=[[2]],
#         gen_mask=gen_mask,
#     )
#     t = tensor(B=1, T_gen=6, T_img=4)
#     grid = SaliencyGrid(t, lay)

#     with pytest.raises(IndexError):
#         grid._validate_batch_idx(1)  # batch index out of bounds
#     with pytest.raises(IndexError):
#         grid._validate_img_idx(0, 1)  # image index out of bounds
#     with pytest.raises(IndexError):
#         grid._validate_token_idx(0, None)  # token index out of bounds
#     with pytest.raises(IndexError):
#         grid._validate_token_idx(0, -1)  # negative token index
#     with pytest.raises(IndexError):
#         grid._validate_token_idx(0, 6)  # token index out of bounds


# # -------Index tests -----


# def test_index_from_indices():
#     idx = Index.from_indices(4)
#     assert idx.batch_idx is None
#     assert idx.img_idx is None
#     assert idx.token_idx == 4

#     idx = Index.from_indices((1, 4))
#     assert idx.batch_idx is None
#     assert idx.img_idx == 1
#     assert idx.token_idx == 4

#     idx = Index.from_indices((0, 1, 4))
#     assert idx.batch_idx == 0
#     assert idx.img_idx == 1
#     assert idx.token_idx == 4

#     assert idx == Index.from_indices(idx)
