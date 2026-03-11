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
    """NCCL-based MoE AllToAllv for H100 / non-MNNVL hardware.

    Dispatch unit is a (token, rank) pair: token t is sent to rank r if it has
    at least one expert on rank r.  The full top-k context (all K expert IDs
    and weights) is forwarded along with each copy so that the FlashInfer
    CUTLASS kernel keeps its compiled tactic (compiled for top_k=K, not 1).

    Communication uses torch.distributed.all_to_all_single — only standard
    NCCL is required, no MNNVL fabric needed.

    prepare():
      1. Build expert→rank map via _get_expert_to_rank()
      2. token_needs_rank[T, R]: True if token t has ≥1 expert on rank r
      3. nonzero() → (token_idx, dest_rank) pairs, stable-sorted by dest_rank
      4. all_to_all_single × 4: counts, hidden states, topk_ids[K], weights[K]
      5. Return received tensors:
           a1q[total_recv, H], topk_ids[total_recv, K], weights[total_recv, K]

    finalize():
      1. Reverse all_to_all_single: send expert partial-sums back to origin ranks
         (FlashInfer CUTLASS kernel applies router weights and zeros non-local
         expert contributions via ep_rank/ep_size internally)
      2. index_add_: output[orig_token] += partial_sum_from_rank_r
    """

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
        use_deepseek_fp8_block_scale: bool = False,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale)
        self._state: Optional[_NCCLDispatchState] = None
        # Cached expert→rank mapping built lazily on the first prepare() call.
        # Keyed by (num_experts, ep_size) so it is rebuilt if the model config
        # ever changes (e.g. different MoE layers with different expert counts).
        self._expert_to_rank: Optional[torch.Tensor] = None
        self._expert_to_rank_params: tuple = ()

    def _ep_pg(self):
        """Return (ep_rank, ep_size, process_group).

        Must use device_group (NCCL) for GPU tensor collectives.
        cpu_group uses gloo and cannot handle CUDA tensors correctly.
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

        Two paths:
        - expert_map is None  → assume uniform contiguous assignment; derive
          the mapping arithmetically with no communication.
        - expert_map provided → all-gather ownership masks across the EP group
          to build the true mapping and validate completeness.

        The result is cached; a cache miss only occurs when (num_experts,
        ep_size) changes, which should never happen during normal inference.
        """
        cache_key = (num_experts, ep_size)
        if self._expert_to_rank is not None and self._expert_to_rank_params == cache_key:
            return self._expert_to_rank

        assert num_experts % ep_size == 0, (
            f"NCCLAllToAllMoEPrepareAndFinalize requires num_experts "
            f"({num_experts}) to be exactly divisible by ep_size ({ep_size}). "
            f"Got remainder {num_experts % ep_size}. "
            f"Use expert_map for non-uniform assignments."
        )
        local_num_experts = num_experts // ep_size

        if expert_map is None:
            # Uniform contiguous assignment: expert e lives on rank e // local_num_experts
            result = torch.arange(num_experts, device=device) // local_num_experts
        else:
            # Build the inverse mapping via a single all-gather.
            # expert_map[e] >= 0  ⟺  this rank owns expert e.
            owns = (expert_map >= 0).to(torch.int32)   # [num_experts]
            all_owns = [torch.empty_like(owns) for _ in range(ep_size)]
            dist.all_gather(all_owns, owns, group=pg)

            result = torch.full(
                (num_experts,), -1, dtype=torch.long, device=device
            )
            for r, owns_r in enumerate(all_owns):
                result[owns_r.bool()] = r

            unassigned = int((result < 0).sum().item())
            assert unassigned == 0, (
                f"{unassigned} / {num_experts} experts have no owning rank "
                f"after all-gather of expert_maps. "
                f"expert_map tensors may be inconsistent across ranks."
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
            # Single-rank path: quantize and return directly
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

        ep_rank, ep_size, pg = self._ep_pg()
        local_token_count = a1.shape[0]
        top_k = topk_ids.shape[1]
        hidden_size = a1.shape[-1]
        device = a1.device

        # ---- Step 1: determine which ranks each token needs ----
        # For each token t and choice k, compute which rank handles that expert.
        # Then send the FULL token (with all top_k choices) to every rank that
        # has at least one of its experts.  This preserves the original top_k
        # so the FlashInfer CUTLASS kernel keeps its compiled tactic.
        # The expert kernel uses expert_map to zero-out non-local experts, so
        # only the local contributions are accumulated — identical to the
        # AllGather path but without broadcasting every token to every rank.
        topk_ids_long = topk_ids.long()   # [T, K]
        # Use _get_expert_to_rank to resolve the global expert→rank mapping.
        # This handles both the standard uniform-contiguous case (expert_map is
        # None) and non-standard layouts (expert_map provided), and asserts that
        # num_experts is divisible by ep_size to prevent out-of-bounds scatter.
        expert_to_rank = self._get_expert_to_rank(
            expert_map, num_experts, ep_size, pg, device
        )
        expert_ranks = expert_to_rank[topk_ids_long]  # [T, K]

        # token_needs_rank[t, r] = True if token t has ≥1 expert on rank r
        token_needs_rank = torch.zeros(
            local_token_count, ep_size, dtype=torch.bool, device=device
        )
        token_needs_rank.scatter_(1, expert_ranks, True)

        # ---- Step 2: build sorted (token_idx, dest_rank) pairs ----
        token_idxs, dest_ranks = token_needs_rank.nonzero(as_tuple=True)
        sort_order        = dest_ranks.argsort(stable=True)
        token_idx_sorted  = token_idxs[sort_order]   # [total_send]
        dest_ranks_sorted = dest_ranks[sort_order]    # [total_send]

        send_sizes = torch.bincount(dest_ranks_sorted, minlength=ep_size).tolist()
        total_send = sum(send_sizes)

        # ---- Step 3: exchange counts ----
        send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_t = torch.empty(ep_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)
        recv_sizes = recv_sizes_t.cpu().tolist()
        total_recv = int(recv_sizes_t.sum().item())

        # ---- Step 4: exchange hidden states ----
        # send_sizes / recv_sizes partition the *first* dimension (row count)
        send_hidden = a1[token_idx_sorted]   # [total_send, H]
        recv_hidden = torch.empty(total_recv, hidden_size, dtype=a1.dtype, device=device)
        dist.all_to_all_single(
            recv_hidden, send_hidden,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=pg,
        )

        # ---- Step 5: exchange full topk_ids (all K choices per token) ----
        # Flatten to 1D so split-sizes are straightforward element counts.
        # Use original topk_ids dtype (not topk_ids_long which is int64) to
        # avoid dtype mismatch in all_to_all_single.
        send_topk_ids_flat = topk_ids[token_idx_sorted].reshape(-1)  # [total_send * K]
        recv_topk_ids_flat = torch.empty(
            total_recv * top_k, dtype=topk_ids.dtype, device=device
        )
        dist.all_to_all_single(
            recv_topk_ids_flat, send_topk_ids_flat,
            output_split_sizes=[c * top_k for c in recv_sizes],
            input_split_sizes=[c * top_k for c in send_sizes],
            group=pg,
        )
        recv_topk_ids_full = recv_topk_ids_flat.reshape(total_recv, top_k)

        # ---- Step 6: exchange full topk_weights (all K weights per token) ----
        send_weights_flat = topk_weights[token_idx_sorted].reshape(-1)  # [total_send * K]
        recv_weights_flat = torch.empty(
            total_recv * top_k, dtype=topk_weights.dtype, device=device
        )
        dist.all_to_all_single(
            recv_weights_flat, send_weights_flat,
            output_split_sizes=[c * top_k for c in recv_sizes],
            input_split_sizes=[c * top_k for c in send_sizes],
            group=pg,
        )
        recv_topk_weights_full = recv_weights_flat.reshape(total_recv, top_k)

        # ---- Step 6b: mask non-local expert weights ----
        # Each rank receives ALL K global expert IDs for every dispatched token.
        # The FlashInfer CUTLASS kernel expects GLOBAL expert IDs and uses ep_rank
        # / ep_size internally to determine which experts are local.  We only need
        # to zero the weights for non-local slots so that their contribution to the
        # output sum is exactly zero (0 * any_computation = 0).
        if ep_size > 1 and total_recv > 0:
            is_local_expert = (
                expert_to_rank[recv_topk_ids_full.long()] == ep_rank
            )  # [total_recv, top_k] bool
            recv_topk_weights_full = recv_topk_weights_full * is_local_expert.to(
                recv_topk_weights_full.dtype
            )

        # Save state for finalize()
        self._state = _NCCLDispatchState(
            local_token_count=local_token_count,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
            token_indices=token_idx_sorted,
        )

        # ---- Step 7: quantize received activations if needed ----
        if (
            not self.use_deepseek_fp8_block_scale
            and quant_config.quant_dtype is not None
        ):
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
            recv_hidden_q  = recv_hidden
            recv_hidden_scale = None

        # Pass full top_k to the kernel; non-local weights are already zeroed so
        # the kernel computes only local expert contributions (partial sum).
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
        # weight_and_reduce_impl is TopKWeightAndReduceNoOP for FlashInferExperts:
        # the CUTLASS kernel already applies topk_weights and zeros out non-local
        # expert contributions (using ep_rank/ep_size), so each received token's
        # fused_expert_output row is already a partial weighted sum.
        # We do not call weight_and_reduce_impl here; index_add_ accumulates
        # partial sums from all ranks into the final per-token output.
        if not self.use_dp:
            output.copy_(fused_expert_output)
            return

        state = self._state
        assert state is not None, "finalize() called before prepare()"
        self._state = None

        _, _, pg = self._ep_pg()
        hidden_size = fused_expert_output.shape[-1]
        device      = fused_expert_output.device
        total_send  = sum(state.send_sizes)

        # ---- Reverse all_to_all: send expert partial-sums back to origin ranks ----
        # In the reverse direction send/recv sizes are SWAPPED vs the forward pass:
        #   input_split_sizes  = recv_sizes  (send back the results of tokens we received)
        #   output_split_sizes = send_sizes  (receive back the results of tokens we sent)
        combined = torch.empty(
            total_send, hidden_size,
            dtype=fused_expert_output.dtype, device=device,
        )
        dist.all_to_all_single(
            combined, fused_expert_output,
            output_split_sizes=state.send_sizes,  # receive results of tokens we sent
            input_split_sizes=state.recv_sizes,   # send results of tokens we received
            group=pg,
        )

        # ---- Accumulate partial sums: output[orig_token] += partial_sum_from_rank_r ----
        # combined[i] is the partial weighted sum from the rank that processed
        # state.token_indices[i].  For tokens dispatched to multiple ranks, multiple
        # entries accumulate into the same output slot, yielding the full MoE output:
        #   output[t] = Σ_r Σ_{k: expert_k ∈ rank_r} w_k * expert_k(token_t)
        assert output.shape[0] == state.local_token_count, (
            f"output shape {output.shape} does not match "
            f"local_token_count={state.local_token_count}"
        )
        # Accumulate partial sums: output[orig_token] += partial_sum_from_rank_r
        # output must be zeroed first since it may contain stale values.
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
