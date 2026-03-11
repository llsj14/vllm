"""
End-to-end test of NCCLAllToAllMoEPrepareAndFinalize.

Run with:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 test_nccl_dispatch.py
"""
import torch
import torch.distributed as dist
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType


def nccl_dispatch_moe(x, w1, w2, topk_ids, topk_w, num_experts, ep_rank, ep_size, pg,
                      activation_type=ActivationType.Swiglu):
    """NCCL AllToAll dispatch (our implementation)."""
    local_num = num_experts // ep_size
    top_k = topk_ids.shape[1]
    local_token_count = x.shape[0]
    hidden_size = x.shape[-1]
    device = x.device

    expert_to_rank = torch.arange(num_experts, device=device) // local_num
    topk_ids_long = topk_ids.long()
    expert_ranks = expert_to_rank[topk_ids_long]

    token_needs_rank = torch.zeros(local_token_count, ep_size, dtype=torch.bool, device=device)
    for k in range(top_k):
        token_needs_rank.scatter_(1, expert_ranks[:, k:k+1], True)

    token_idxs, dest_ranks = token_needs_rank.nonzero(as_tuple=True)
    sort_order = dest_ranks.argsort(stable=True)
    token_idx_sorted = token_idxs[sort_order]
    dest_ranks_sorted = dest_ranks[sort_order]
    send_sizes = [int((dest_ranks_sorted == r).sum().item()) for r in range(ep_size)]
    total_send = sum(send_sizes)

    send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=device)
    recv_sizes_t = torch.empty(ep_size, dtype=torch.long, device=device)
    dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)
    recv_sizes = recv_sizes_t.cpu().tolist()
    total_recv = int(recv_sizes_t.sum().item())

    send_hidden = x[token_idx_sorted]
    recv_hidden = torch.empty(total_recv, hidden_size, dtype=x.dtype, device=device)
    dist.all_to_all_single(recv_hidden, send_hidden,
                           output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=pg)

    send_topk_ids_flat = topk_ids[token_idx_sorted].reshape(-1)
    recv_topk_ids_flat = torch.empty(total_recv * top_k, dtype=topk_ids.dtype, device=device)
    dist.all_to_all_single(recv_topk_ids_flat, send_topk_ids_flat,
                           output_split_sizes=[c * top_k for c in recv_sizes],
                           input_split_sizes=[c * top_k for c in send_sizes], group=pg)
    recv_topk_ids_full = recv_topk_ids_flat.reshape(total_recv, top_k)

    send_weights_flat = topk_w[token_idx_sorted].reshape(-1)
    recv_weights_flat = torch.empty(total_recv * top_k, dtype=topk_w.dtype, device=device)
    dist.all_to_all_single(recv_weights_flat, send_weights_flat,
                           output_split_sizes=[c * top_k for c in recv_sizes],
                           input_split_sizes=[c * top_k for c in send_sizes], group=pg)
    recv_topk_weights_full = recv_weights_flat.reshape(total_recv, top_k)

    if ep_size > 1 and total_recv > 0:
        is_local = (expert_to_rank[recv_topk_ids_full.long()] == ep_rank)
        recv_topk_weights_full = recv_topk_weights_full * is_local.to(recv_topk_weights_full.dtype)

    fused_out = torch.zeros(total_recv, hidden_size, dtype=x.dtype, device=device)
    if total_recv > 0:
        cutlass_fused_moe(input=recv_hidden,
                          token_selected_experts=recv_topk_ids_full.to(torch.int),
                          token_final_scales=recv_topk_weights_full,
                          fc1_expert_weights=w1, fc2_expert_weights=w2,
                          output_dtype=x.dtype, quant_scales=None,
                          ep_size=ep_size, ep_rank=ep_rank, output=fused_out,
                          activation_type=activation_type)
    torch.cuda.synchronize()

    combined = torch.empty(total_send, hidden_size, dtype=fused_out.dtype, device=device)
    dist.all_to_all_single(combined, fused_out,
                           output_split_sizes=send_sizes, input_split_sizes=recv_sizes, group=pg)

    output_buf = torch.zeros(local_token_count, hidden_size, dtype=torch.float32, device=device)
    if total_send > 0:
        output_buf.index_add_(0, token_idx_sorted, combined.float())

    return output_buf.to(x.dtype)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    ep_size = world_size
    ep_rank = rank
    device = f"cuda:{rank}"

    pg = dist.new_group(ranks=list(range(ep_size)), backend="nccl")

    num_experts = 4
    local_num = num_experts // ep_size  # = 2
    hidden_size = 64
    intermediate_size = 128
    top_k = 2
    num_tokens = 4
    # Nemotron uses relu2_no_mul; w1 has shape [E, N, K] (no gated split, so N=intermediate_size)
    activation_type = ActivationType.Relu2

    torch.manual_seed(42)
    w1 = torch.randn(local_num, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(local_num, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)

    # Each rank has its own tokens
    torch.manual_seed(rank * 100 + 1)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # rank 0: experts [0(local), 2(non-local)], rank 1: experts [2(local), 0(non-local)]
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
    topk_ids[:, 0] = ep_rank * local_num      # local expert 0 for this rank
    topk_ids[:, 1] = ((ep_rank + 1) % ep_size) * local_num  # first expert of next rank
    topk_w = torch.tensor([[0.6, 0.4]] * num_tokens, dtype=torch.float32, device=device)

    # ---- CORRECT REFERENCE: gather all weights, run ep_size=1 ----
    # Gather weights from all ranks
    gathered_w1 = [torch.empty_like(w1) for _ in range(ep_size)]
    gathered_w2 = [torch.empty_like(w2) for _ in range(ep_size)]
    dist.all_gather(gathered_w1, w1, group=pg)
    dist.all_gather(gathered_w2, w2, group=pg)
    all_w1 = torch.cat(gathered_w1, dim=0)  # [num_experts, ...]
    all_w2 = torch.cat(gathered_w2, dim=0)

    # Reference: run with ALL experts on a single rank (ep_size=1, ep_rank=0)
    ref_out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=device)
    cutlass_fused_moe(input=x,
                      token_selected_experts=topk_ids.to(torch.int),
                      token_final_scales=topk_w,
                      fc1_expert_weights=all_w1, fc2_expert_weights=all_w2,
                      output_dtype=x.dtype, quant_scales=None,
                      ep_size=1, ep_rank=0, output=ref_out,
                      activation_type=activation_type)
    torch.cuda.synchronize()

    # ---- NCCL AllToAll dispatch ----
    nccl_out = nccl_dispatch_moe(x, w1, w2, topk_ids, topk_w, num_experts, ep_rank, ep_size, pg,
                                  activation_type=activation_type)

    # ---- Compare ----
    diff = (nccl_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    nan_count = nccl_out.isnan().sum().item()
    ref_scale = ref_out.float().abs().max().item()

    # BF16 has 7 mantissa bits → relative precision ~0.78%.
    # For a sum of 2 partial sums the worst-case accumulated rounding error is
    # ~1.6% of the peak output value.  Use 3% as a generous BF16 tolerance.
    rel_err = max_diff / max(ref_scale, 1.0)
    BF16_REL_TOL = 0.03

    # Find where max diff occurs
    max_diff_idx = diff.argmax()
    max_diff_token = int(max_diff_idx // hidden_size)
    max_diff_dim = int(max_diff_idx % hidden_size)

    print(f"[rank{rank}] NCCL vs reference (ep_size=1):")
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, nan={nan_count}")
    print(f"  ref_scale={ref_scale:.2f}, rel_err={rel_err:.4f} (BF16 tol={BF16_REL_TOL})")
    print(f"  max_diff at token={max_diff_token}, dim={max_diff_dim}: "
          f"nccl={nccl_out[max_diff_token, max_diff_dim].item():.4f}, "
          f"ref={ref_out[max_diff_token, max_diff_dim].item():.4f}")
    print(f"  nccl_out[0,:4] = {nccl_out[0,:4].tolist()}")
    print(f"  ref_out[0,:4]  = {ref_out[0,:4].tolist()}")
    if rel_err < BF16_REL_TOL and nan_count == 0:
        print(f"  [PASS] NCCL dispatch is correct (rel_err={rel_err:.4f} < {BF16_REL_TOL}).")
    else:
        print(f"  [FAIL] NCCL dispatch gives wrong results!")
        if nan_count > 0:
            print(f"  NaN count: {nan_count}")

        # Debug: print dispatch details
        expert_to_rank = torch.arange(num_experts, device=device) // local_num
        topk_ids_long = topk_ids.long()
        expert_ranks = expert_to_rank[topk_ids_long]
        print(f"  topk_ids: {topk_ids[0].tolist()}")
        print(f"  expert_ranks: {expert_ranks[0].tolist()}")
        is_local = (expert_to_rank[topk_ids_long] == ep_rank)
        print(f"  is_local: {is_local[0].tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
