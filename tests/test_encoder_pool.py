"""Pooling tests — offline, synthetic tensors (no model load). Guards the decoder-embedder path."""

import torch

from xbse.encoder import last_token_pool, mean_pool


def test_mean_pool_masks_padding():
    emb = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    mask = torch.tensor([[1, 0, 1], [1, 1, 1]])  # seq0 drops the middle token
    out = mean_pool(emb, mask)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0]))  # (1+3)/2
    assert torch.allclose(out[1], torch.tensor([5.0, 5.0]))  # (4+5+6)/3


def test_last_token_pool_right_padding():
    emb = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])  # right-padded
    out = last_token_pool(emb, mask)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0]))  # last real token of seq0
    assert torch.allclose(out[1], torch.tensor([6.0, 6.0]))


def test_last_token_pool_left_padding():
    emb = torch.tensor([[[9.0, 9.0], [1.0, 1.0], [2.0, 2.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])  # left-padded -> final column is the last token
    out = last_token_pool(emb, mask)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(out[1], torch.tensor([6.0, 6.0]))
