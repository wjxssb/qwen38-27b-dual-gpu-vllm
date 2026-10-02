# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    num_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    # Keep the original sequence ID when a padded/empty sequence contributes
    # no chunks. Counting zero chunk offsets collapses those sequence IDs and
    # sends every following kernel to the wrong cu_seqlens entry.
    return cu_seqlens.new_tensor(
        [(seq_id, chunk_id)
         for seq_id, count in enumerate(num_chunks)
         for chunk_id in range(count)]
    ).reshape(-1, 2)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
