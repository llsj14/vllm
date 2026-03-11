# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    build_flashinfer_bf16_cutlass_moe_prepare_finalize,
    swap_w13_to_w31,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"


# NOTE(zyongye): Unsupported backend means backend
# that is not conform with Modular kernel format.
# We will directly call the kernel for those backend
UNSUPPORTED_BACKEND = [
    UnquantizedMoeBackend.CPU,
    UnquantizedMoeBackend.XPU,
    UnquantizedMoeBackend.TPU,
    UnquantizedMoeBackend.OOT,
]


def select_unquantized_moe_backend(
    use_ep: bool,
    use_dp: bool,
    all2all_backend: str = "",
) -> UnquantizedMoeBackend:
    """
    Select the primary unquantized MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUs.
    # DP>1 is supported: prepare/finalize will use AllToAllv or AllGather
    # depending on the all2all_backend setting.
    #
    # Auto-enabled when all2all_backend == "flashinfer_all2allv" because
    # that flag is already an explicit opt-in to use FlashInfer kernels.
    use_flashinfer_cutlass = (
        envs.VLLM_USE_FLASHINFER_MOE_FP16
        or all2all_backend == "flashinfer_all2allv"
    )
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and use_flashinfer_cutlass
        and use_ep
        and current_platform.get_device_capability()[0] >= 9
    )
    if current_platform.is_rocm():
        if rocm_aiter_moe_enabled:
            backend = UnquantizedMoeBackend.AITER
        else:
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_cuda():
        if flashinfer_cutlass_moe_enabled:
            backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
        else:
            if use_ep:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.",
                    scope="local",
                )
            backend = UnquantizedMoeBackend.TRITON
    if current_platform.is_xpu():
        backend = UnquantizedMoeBackend.XPU
    if current_platform.is_cpu():
        backend = UnquantizedMoeBackend.CPU
    if current_platform.is_tpu():
        backend = UnquantizedMoeBackend.TPU
    if current_platform.is_out_of_tree():
        backend = UnquantizedMoeBackend.OOT

    logger.info_once(_make_log_backend(backend), scope="local")
    return backend


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend == UnquantizedMoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        # Swap halves to arrange as [w3; w1] (kernel expectation).
        # Only needed for gated activations (is_act_and_mul=True, e.g. silu/gelu).
        # Non-gated activations (relu2_no_mul, is_act_and_mul=False) store w13
        # as [E, N, K] with no gate split; swapping would corrupt the weights.
        if layer.moe_config.is_act_and_mul:
            w13_weight = swap_w13_to_w31(layer.w13_weight.data)
        else:
            w13_weight = layer.w13_weight.data

    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> tuple[mk.FusedMoEModularKernel | None, bool]:
    use_inplace = True

    if backend in UNSUPPORTED_BACKEND:
        return None, use_inplace

    if backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        use_dp = moe_config.moe_parallel_config.dp_size > 1
        if use_dp:
            prepare_finalize = build_flashinfer_bf16_cutlass_moe_prepare_finalize(
                moe_config
            )
        else:
            prepare_finalize = MoEPrepareAndFinalizeNoEP()


        kernel = mk.FusedMoEModularKernel(
            prepare_finalize,
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
        use_inplace = False
    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            AiterExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe import TritonExperts

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )
    return kernel, use_inplace
