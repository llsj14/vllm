# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave

logger = init_logger(__name__)


def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Base class for FlashInfer MoE prepare and finalize operations."""

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
        use_deepseek_fp8_block_scale: bool = False,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.use_dp = use_dp
        self.local_tokens = None
        # Toggle for DeepSeek-style FP8 block-scale path where activations are
        # not quantized here and weight block scales are consumed by the kernel.
        self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def _apply_router_weight_on_input(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Apply router weight on input if needed."""
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))


class FlashInferAllToAllMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):
    """FlashInfer implementation using AllToAll communication."""

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
        use_deepseek_fp8_block_scale: bool = False,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale)
        self.alltoall_info = None

        # Initialize all2all_manager only for DP case
        self.all2all_manager = None
        if self.use_dp:
            self.all2all_manager = get_ep_group().device_communicator.all2all_manager

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )

        if not self.use_dp:
            # Non-DP case: quantize activations unless using block-scale path
            if not self.use_deepseek_fp8_block_scale:
                a1q, a1q_scale = moe_kernel_quantize_input(
                    a1,
                    quant_config.a1_gscale,
                    quant_config.quant_dtype,
                    quant_config.per_act_token_quant,
                    quant_config.block_shape,
                    is_fp4_scale_swizzled=not self.use_dp,
                )
            else:
                a1q = a1
                a1q_scale = None
        else:
            # DP case: use FlashInfer AllToAll
            global_num_tokens_cpu = get_local_sizes()
            top_k = topk_ids.size(1)

            (self.alltoall_info, topk_ids, topk_weights, a1q, a1q_scale) = (
                flashinfer_alltoall_dispatch(
                    self.all2all_manager,
                    global_num_tokens_cpu,
                    a1,
                    quant_config.a1_gscale,
                    topk_ids,
                    topk_weights,
                    top_k,
                    num_experts,
                    quant_config,
                    use_deepseek_fp8_block_scale=self.use_deepseek_fp8_block_scale,
                )
            )

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.use_dp:
            top_k = topk_ids.size(1)
            token_count = output.shape[0]
            fused_expert_output = flashinfer_alltoall_combine(
                self.all2all_manager,
                fused_expert_output,
                top_k=top_k,
                token_count=token_count,
                alltoall_info=self.alltoall_info,
            )
        output.copy_(fused_expert_output)


class FlashInferAllGatherMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):
    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
        use_deepseek_fp8_block_scale: bool = False,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )
        is_nvfp4 = quant_config.quant_dtype == "nvfp4"
        if not self.use_dp and is_nvfp4:
            return a1, None, None, topk_ids, topk_weights

        if not self.use_deepseek_fp8_block_scale:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale if is_nvfp4 else quant_config.a1_scale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=not self.use_dp,
            )
        else:
            # Block-scale path: pass activations through, omit per-token scales
            a1q = a1
            a1q_scale = None

        if self.use_dp:
            # Build gather list conditionally - omit a1q_scale if None
            # (block-scale path)
            gather_list = [topk_weights, topk_ids, a1q]
            if a1q_scale is not None:
                gather_list.append(a1q_scale)
                gathered = get_dp_group().all_gatherv(
                    gather_list,
                    dim=0,
                    sizes=get_local_sizes(),
                )
                topk_weights, topk_ids, a1q, a1q_scale = gathered
            else:
                gathered = get_dp_group().all_gatherv(
                    gather_list,
                    dim=0,
                    sizes=get_local_sizes(),
                )
                topk_weights, topk_ids, a1q = gathered
                a1q_scale = None

        if is_nvfp4 and a1q_scale is not None:
            if a1q_scale.element_size() == 1:
                a1q_scale = a1q_scale.view(torch.uint8)
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP)

        if self.use_dp:
            fused_expert_output = get_dp_group().reduce_scatterv(
                fused_expert_output, dim=0, sizes=get_local_sizes()
            )
        output.copy_(fused_expert_output)


def flashinfer_alltoall_dispatch(
    all2all_manager: All2AllManagerBase,
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    gs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    quant_config: FusedMoEQuantConfig,
    use_deepseek_fp8_block_scale: bool = False,
):
    from flashinfer.comm.trtllm_alltoall import MnnvlMoe

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (
        "FlashInfer AllToAll workspace initialization failed. "
        "MNNVL requires SYS_PTRACE capability and NVLink fabric hardware. "
        "In a container, add --cap-add=SYS_PTRACE to your docker run command."
    )

    ep_rank = all2all_manager.rank
    ep_size = all2all_manager.world_size
    max_num_token = (
        max(global_num_tokens_cpu) if global_num_tokens_cpu is not None else x.shape[0]
    )
    orig_topk_weights_dtype = topk_weights.dtype
    alltoall_info, topk_ids, topk_weights, _ = (
        MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
            topk_ids,
            topk_weights,
            None,
            all2all_manager.prepare_workspace_tensor,
            max_num_token,
            ep_rank,
            ep_size,
            num_experts,
            num_experts,
            top_k,
        )
    )
    topk_weights = topk_weights.view(dtype=orig_topk_weights_dtype)

    if quant_config.quant_dtype is None:
        # BF16 / unquantized path: send raw activations without quantization.
        x_sf = None
        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )
    elif not use_deepseek_fp8_block_scale:
        x, x_sf = moe_kernel_quantize_input(
            x,
            gs,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            is_fp4_scale_swizzled=False,  # delay swizzle to after comm
        )
        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )

        x_sf = MnnvlMoe.mnnvl_moe_alltoallv(
            x_sf,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )
        if quant_config.quant_dtype == "nvfp4":
            x_sf = nvfp4_block_scale_interleave(x_sf)
    else:
        # DeepSeek block-scale path: pass activations through without quantization.
        x_sf = None
        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )
    return alltoall_info, topk_ids, topk_weights, x, x_sf


def flashinfer_alltoall_combine(
    all2all_manager: All2AllManagerBase,
    output: torch.Tensor,
    top_k: int,
    token_count: int,
    alltoall_info,
):
    from flashinfer.comm.trtllm_alltoall import MnnvlMoe

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (
        "FlashInfer AllToAll workspace initialization failed. "
        "MNNVL requires SYS_PTRACE capability and NVLink fabric hardware. "
        "In a container, add --cap-add=SYS_PTRACE to your docker run command."
    )
    return MnnvlMoe.mnnvl_moe_alltoallv_combine(
        output,
        alltoall_info,
        all2all_manager.workspace_tensor,
        ep_rank=all2all_manager.rank,
        ep_size=all2all_manager.world_size,
        top_k=top_k,
        token_count=token_count,
    )


@dataclass
class _NCCLDispatchState:
    """State preserved between prepare() and finalize() for NCCL AllToAll."""
    local_token_count: int
    # CPU Python lists required by dist.all_to_all_single split-size args
    send_sizes: list
    recv_sizes: list
    # [total_send] — original local token index for every dispatched (token, rank) pair,
    # sorted by dest_rank so that combined[i] corresponds to token_indices[i] after
    # the reverse all_to_all_single.
    token_indices: torch.Tensor


@lru_cache(maxsize=None)
def _is_mnnvl_available(device_idx: int) -> bool:
    """Return True iff MNNVL NVLink fabric is present on *device_idx*.

    MNNVL (CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) exists on
    GB200 NVL72 / NVSwitch systems. H100 and earlier return False.
    Cached per device index so the check runs only once.
    """
    try:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported
        return bool(is_mnnvl_fabric_supported(device_idx))
    except Exception as exc:
        logger.warning(
            "Could not query MNNVL fabric support for device %d (%s). "
            "Assuming unavailable.",
            device_idx, exc,
        )
        return False


class NCCLAllToAllMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):
    """NCCL-based MoE AllToAll for H100 / non-MNNVL hardware.

    CUDA graph compatible by registering moe_forward and moe_forward_shared as
    splitting_ops.  Both custom ops are opaque to dynamo, so making them
    splitting ops causes the entire MoE forward (routing, AllToAll, expert
    kernel, reverse AllToAll) to run outside CUDA graph subgraphs — as regular
    Python between two CUDA graph captures.

    Which op is used depends on SharedFusedMoE.use_overlapped:
      - use_overlapped=True  → moe_forward_shared (shared experts inside)
      - use_overlapped=False → moe_forward (shared experts run separately)
    FlashInfer + DP disables overlap, so moe_forward is the common path.

    This allows using variable-size AllToAll (nonzero, argsort, split-sizes)
    for all T values, which is both correct and efficient.
    """

    # Class-level flag: True once any instance exists.
    # Used by gpu_model_runner.load_model() to force PIECEWISE CUDA graph mode.
    _active: bool = False

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
        use_deepseek_fp8_block_scale: bool = False,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale)
        self._state: Optional[_NCCLDispatchState] = None
        self._expert_to_rank: Optional[torch.Tensor] = None
        self._expert_to_rank_params: tuple = ()
        NCCLAllToAllMoEPrepareAndFinalize._active = True
        self._register_alltoall_splitting_ops()

    def _register_alltoall_splitting_ops(self) -> None:
        """Add moe_forward and moe_forward_shared to splitting_ops.

        Both custom ops are opaque to dynamo.  Making them splitting ops causes
        the entire MoE forward (routing, AllToAll, expert kernel, reverse
        AllToAll) to run outside CUDA graph subgraphs as regular Python between
        two CUDA graph captures.

        Which op is used depends on SharedFusedMoE.use_overlapped:
          - use_overlapped=True  → moe_forward_shared (shared experts inside)
          - use_overlapped=False → moe_forward (shared experts run separately)
        FlashInfer + DP disables overlap, so moe_forward is the common path.
        We register both to cover all configurations.
        """
        from vllm.config import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        if cfg is None:
            return
        splitting_ops = cfg.compilation_config.splitting_ops
        if splitting_ops is None:
            return
        for op_name in (
            "vllm::moe_forward",
            "vllm::moe_forward_shared",
        ):
            if op_name not in splitting_ops:
                splitting_ops.append(op_name)
                logger.info(
                    "NCCLAllToAllMoEPrepareAndFinalize: added '%s' to "
                    "splitting_ops so AllToAll runs outside CUDA graphs. "
                    "splitting_ops now: %s",
                    op_name,
                    splitting_ops,
                )

    def _ep_pg(self):
        """Return (ep_rank, ep_size, process_group).

        Must use device_group (NCCL) for GPU tensor collectives.
        """
        ep_group = get_ep_group()
        return ep_group.rank_in_group, ep_group.world_size, ep_group.device_group

    def _get_expert_to_rank(
        self,
        expert_map: Optional[torch.Tensor],
        num_experts: int,
        ep_size: int,
        pg,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a [num_experts] int64 tensor: result[e] = rank that owns expert e.

        Cached; a cache miss triggers all-gather (only on the very first call).
        """
        cache_key = (num_experts, ep_size)
        if self._expert_to_rank is not None and self._expert_to_rank_params == cache_key:
            return self._expert_to_rank

        assert num_experts % ep_size == 0, (
            f"NCCLAllToAllMoEPrepareAndFinalize requires num_experts "
            f"({num_experts}) to be exactly divisible by ep_size ({ep_size}). "
            f"Got remainder {num_experts % ep_size}."
        )
        local_num_experts = num_experts // ep_size

        if expert_map is None:
            result = torch.arange(num_experts, device=device) // local_num_experts
        else:
            owns = (expert_map >= 0).to(torch.int32)
            all_owns = [torch.empty_like(owns) for _ in range(ep_size)]
            dist.all_gather(all_owns, owns, group=pg)

            result = torch.full((num_experts,), -1, dtype=torch.long, device=device)
            for r, owns_r in enumerate(all_owns):
                result[owns_r.bool()] = r

            unassigned = int((result < 0).sum().item())
            assert unassigned == 0, (
                f"{unassigned} / {num_experts} experts have no owning rank."
            )

        self._expert_to_rank = result
        self._expert_to_rank_params = cache_key
        return result

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )

        if not self.use_dp:
            if not self.use_deepseek_fp8_block_scale:
                is_nvfp4 = quant_config.quant_dtype == "nvfp4"
                a1q, a1q_scale = moe_kernel_quantize_input(
                    a1,
                    quant_config.a1_gscale if is_nvfp4 else quant_config.a1_scale,
                    quant_config.quant_dtype,
                    quant_config.per_act_token_quant,
                    quant_config.block_shape,
                    is_fp4_scale_swizzled=True,
                )
            else:
                a1q, a1q_scale = a1, None
            return a1q, a1q_scale, None, topk_ids, topk_weights

        T = a1.shape[0]
        H = a1.shape[-1]
        K = topk_ids.shape[1]
        ep_rank, ep_size, pg = self._ep_pg()
        device = a1.device

        expert_to_rank = self._get_expert_to_rank(expert_map, num_experts, ep_size, pg, device)
        topk_ids_long = topk_ids.long()
        expert_ranks = expert_to_rank[topk_ids_long]  # [T, K]

        token_needs_rank = torch.zeros(T, ep_size, dtype=torch.bool, device=device)
        token_needs_rank.scatter_(1, expert_ranks, True)

        token_idxs, dest_ranks = token_needs_rank.nonzero(as_tuple=True)
        sort_order = dest_ranks.argsort(stable=True)
        token_idx_sorted = token_idxs[sort_order]
        dest_ranks_sorted = dest_ranks[sort_order]

        send_sizes = torch.bincount(dest_ranks_sorted, minlength=ep_size).tolist()

        send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_t = torch.empty(ep_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)
        recv_sizes = recv_sizes_t.cpu().tolist()
        total_recv = int(recv_sizes_t.sum().item())

        send_hidden = a1[token_idx_sorted]
        recv_hidden = torch.empty(total_recv, H, dtype=a1.dtype, device=device)
        dist.all_to_all_single(
            recv_hidden, send_hidden,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=pg,
        )

        send_topk_ids_flat = topk_ids[token_idx_sorted].reshape(-1)
        recv_topk_ids_flat = torch.empty(total_recv * K, dtype=topk_ids.dtype, device=device)
        dist.all_to_all_single(
            recv_topk_ids_flat, send_topk_ids_flat,
            output_split_sizes=[c * K for c in recv_sizes],
            input_split_sizes=[c * K for c in send_sizes],
            group=pg,
        )
        recv_topk_ids_full = recv_topk_ids_flat.reshape(total_recv, K)

        send_weights_flat = topk_weights[token_idx_sorted].reshape(-1)
        recv_weights_flat = torch.empty(total_recv * K, dtype=topk_weights.dtype, device=device)
        dist.all_to_all_single(
            recv_weights_flat, send_weights_flat,
            output_split_sizes=[c * K for c in recv_sizes],
            input_split_sizes=[c * K for c in send_sizes],
            group=pg,
        )
        recv_topk_weights_full = recv_weights_flat.reshape(total_recv, K)

        if ep_size > 1 and total_recv > 0:
            is_local = expert_to_rank[recv_topk_ids_full.long()] == ep_rank
            recv_topk_weights_full = recv_topk_weights_full * is_local.to(
                recv_topk_weights_full.dtype
            )

        self._state = _NCCLDispatchState(
            local_token_count=T,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
            token_indices=token_idx_sorted,
        )

        if not self.use_deepseek_fp8_block_scale and quant_config.quant_dtype is not None:
            is_nvfp4 = quant_config.quant_dtype == "nvfp4"
            recv_hidden_q, recv_hidden_scale = moe_kernel_quantize_input(
                recv_hidden,
                quant_config.a1_gscale if is_nvfp4 else quant_config.a1_scale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=False,
            )
        else:
            recv_hidden_q = recv_hidden
            recv_hidden_scale = None

        return recv_hidden_q, recv_hidden_scale, None, recv_topk_ids_full, recv_topk_weights_full

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if not self.use_dp:
            output.copy_(fused_expert_output)
            return

        state = self._state
        assert state is not None, "finalize() called before prepare()"
        self._state = None

        _, _, pg = self._ep_pg()
        H = fused_expert_output.shape[-1]
        device = fused_expert_output.device
        total_send = sum(state.send_sizes)

        combined = torch.empty(total_send, H, dtype=fused_expert_output.dtype, device=device)
        dist.all_to_all_single(
            combined, fused_expert_output,
            output_split_sizes=state.send_sizes,
            input_split_sizes=state.recv_sizes,
            group=pg,
        )

        output.zero_()
        output.index_add_(0, state.token_indices, combined)


def create_flashinfer_prepare_finalize(
    use_dp: bool,
    use_nvfp4: bool = False,
    enable_alltoallv: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
) -> FlashInferCutlassMoEPrepareAndFinalize | MoEPrepareAndFinalizeNoEP:
    """Factory: pick the right prepare/finalize backend.

    When *enable_alltoallv* is True the factory auto-detects hardware:
      - MNNVL fabric present (GB200 NVL72 …) → FlashInferAllToAllMoEPrepareAndFinalize
      - No MNNVL (H100, A100, …)             → NCCLAllToAllMoEPrepareAndFinalize
    """

    if use_dp:
        if enable_alltoallv:
            device_idx = torch.cuda.current_device()
            if _is_mnnvl_available(device_idx):
                return FlashInferAllToAllMoEPrepareAndFinalize(use_dp)
            else:
                return NCCLAllToAllMoEPrepareAndFinalize(
                    use_dp=True,
                    use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
                )
        return FlashInferAllGatherMoEPrepareAndFinalize(
            use_dp=True,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        )
    else:
        # CUTLASS FP8 BLOCK and CUTLASS NVFP4 apply input quantization
        # in a single call with the MoE experts kernel.
        defer_input_quant = use_deepseek_fp8_block_scale or use_nvfp4
        return MoEPrepareAndFinalizeNoEP(defer_input_quant=defer_input_quant)
