# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.scalar_type import ScalarType, scalar_types

MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]


def query_machete_supported_quant_types(zero_points: bool) -> list[ScalarType]:
    if zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_machete_supported_act_types(zero_points: bool) -> list[ScalarType]:
    return [torch.float16, torch.bfloat16]


def query_machete_supported_group_sizes(act_type: torch.dtype) -> list[int]:
    """
    Queries the supported group sizes for Machete based on the activation type.

    Args:
        act_type: The activation data type (torch.float16, torch.bfloat16).

    Returns:
        A list of supported group sizes. The group size must
        be divisible by `TileShapeK = 128 * 8 // num_bits(act_type)`.
        -1 indicates per-channel quantization.
    """
    if act_type in [torch.float16, torch.bfloat16]:
        return [-1, 64, 128]
    else:
        return [-1, 128]


def check_machete_supports_shape(in_features: int, out_featrues: int) \
    -> tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return False, "Input features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[0]}"
    if out_featrues % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return False, "Output features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[1]}"
    return True, None
