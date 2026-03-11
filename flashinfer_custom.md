# FlashInfer AllToAllv BF16+DP>1 지원 구현 내역

## 배경

vLLM에서 `--all2all-backend flashinfer_all2allv` 옵션 사용 시 BF16(unquantized) 모델 + DP>1 환경에서
`NotImplementedError`가 발생하는 구조적 문제를 수정하고, 실제 AllToAllv 통신 경로를 구현했습니다.

FP8 모델에서도 동일한 구조적 문제로 `flashinfer_all2allv` 백엔드를 설정해도 AllGather+ReduceScatter를
그대로 사용하는 문제가 확인되었으나, 이번 작업에서는 **미적용**입니다.

---

## 문제 원인 분석

### vLLM MoE forward의 두 가지 dispatch 경로

vLLM MoE layer(`layer.py`)에는 DP>1 환경에서 token dispatch를 처리하는 두 가지 경로가 있습니다.

**경로 A: "naive" dispatch/combine (AllGather + ReduceScatter)**

`layer.py`의 `forward_impl()`에서 아래 조건으로 진입합니다:

```python
do_naive_dispatch_combine = dp_size > 1 and not isinstance(quant_method, FusedMoEModularMethod)
```

조건이 `True`이면 `layer.py`가 직접 `get_ep_group().dispatch()` → (expert 계산) → `get_ep_group().combine()`을
순서대로 호출합니다. dispatch/combine 구현은 `all2all_backend` 설정에 따라 선택된
`All2AllManagerBase` 서브클래스(`AgRsAll2AllManager`, `FlashInferAllToAllManager` 등)에 위임됩니다.

**경로 B: "modular kernel" (prepare/finalize 내장)**

조건이 `False`이면 `layer.py`는 dispatch를 직접 호출하지 않고,
`quant_method.apply()` → `FusedMoEModularKernel.__call__()` → `prepare_finalize.prepare()` 순서로
expert 커널에게 제어를 넘깁니다. dispatch/combine은 `prepare_finalize` 내부에서 처리됩니다.

FP8/NVFP4 등 quantized 모델은 `FusedMoEModularMethod`로 래핑되어 이 경로를 사용합니다.

---

### BF16+DP>1에서 `NotImplementedError`가 발생한 구체적 이유

#### 핵심: BF16이 AllGather 경로로 떨어졌고, FlashInfer는 AllGather를 구현하지 않았음

경로 A의 `dispatch()`/`combine()`이 실제로 하는 일은 **AllGather+ReduceScatter**입니다.
이름은 dispatch지만 `AgRsAll2AllManager`의 구현을 보면 `all_gatherv()`로 되어 있습니다.

`FlashInferAllToAllManager`는 AllGather가 아닌 **AllToAllv를 위해 설계된 클래스**입니다.
AllToAllv는 경로 B의 `prepare_finalize` 내부에서 동작하므로, 경로 A에서 사용하는
AllGather 기반 `dispatch()`/`combine()`을 구현할 이유가 없었습니다.

즉 `FlashInferAllToAllManager`에 AllToAllv가 구현 안 된 것이 아니라,
**BF16이 의도치 않게 AllGather 경로(경로 A)로 떨어졌는데,
`FlashInferAllToAllManager`는 AllGather dispatch를 구현하지 않았기 때문**입니다.

```
[BF16 + DP>1 + flashinfer_all2allv]

do_naive_dispatch_combine = True   ← BF16은 FusedMoEModularMethod가 아님
  → get_ep_group().dispatch()      ← 실제로는 AllGather 요청
    → FlashInferAllToAllManager.dispatch()
      → All2AllManagerBase.dispatch() 상속
        → raise NotImplementedError  ✗
          (AllGather 미구현. AllToAllv는 경로 B에서 처리해야 함)
```

BF16이 경로 B(AllToAllv)로 진입하지 못한 구조적 제약들:

1. **`oracle/unquantized.py`에 `and (not use_dp)` 하드코딩**
   - `flashinfer_cutlass_moe_enabled` 조건에 DP>1이면 무조건 False가 되는 조건이 있었음
   - 결과: `VLLM_USE_FLASHINFER_MOE_FP16=1`을 설정해도 DP>1이면 항상 TRITON backend로 떨어짐

2. **`UnquantizedFusedMoEMethod`에 modular dispatch 신호 없음**
   - `UnquantizedFusedMoEMethod`가 `FusedMoEModularMethod`가 아니므로
     내부적으로 `FusedMoEModularKernel`을 쓰더라도 `layer.py`가 이를 알 수 없음
   - 결과: `do_naive_dispatch_combine=True` → 경로 A 강제 진입

3. **`create_flashinfer_prepare_finalize()`의 `assert use_nvfp4`**
   - DP>1+alltoallv 경로에 진입하더라도 BF16(use_nvfp4=False)이면 assert 실패

4. **`flashinfer_alltoall_dispatch()`에 BF16 분기 없음**
   - quantize를 스킵하는 `quant_dtype is None` 분기가 없어 BF16 tensor를 처리할 수 없었음

5. **`CustomCommunicator`의 추상 메서드 미구현**
   - `CommBackend`의 `barrier()`, `bcast()` 미구현으로 workspace 초기화 자체가 `TypeError`

위 5가지가 모두 맞물려 있어 어떤 설정 조합을 줘도 BF16+DP>1은 동작할 수 없는 상태였습니다.

---

### H100에서 FlashInfer AllToAllv가 동작하지 않는 이유

FlashInfer의 `MnnvlMoe.mnnvl_moe_alltoallv()`는 내부적으로 **MNNVL(Multi-Node NVLink) fabric** 기반
shared memory (`MnnvlMemory`)를 workspace로 사용합니다. 이 기능은 GB200 NVL72처럼 NVSwitch 기반
하드웨어에서만 지원되며, H100 등 일반 NVLink 하드웨어에서는 `is_mnnvl_fabric_supported()` 가 False를
반환합니다.

초기화 시 Linux `pidfd_getfd` syscall(프로세스 간 file descriptor 공유)이 필요하고,
컨테이너 환경에서는 `SYS_PTRACE` capability가 없으면 `RuntimeError: pidfd_getfd ... Permission denied`
로 실패합니다. H100에서는 하드웨어 자체가 MNNVL을 지원하지 않으므로 이 방식 자체를 사용할 수 없습니다.

**해결 방향**: H100 등 비-MNNVL 환경에서는 표준 NCCL(`torch.distributed.all_to_all_single`)을
사용하는 새로운 AllToAll 구현(`NCCLAllToAllMoEPrepareAndFinalize`)을 도입합니다.

---

### FP8+DP>1에서 성능 개선이 없던 이유 (미수정)

FP8은 BF16과 달리 `FusedMoEModularMethod`로 래핑되어 경로 B를 타기 때문에 에러는 발생하지
않습니다. 그러나 `build_flashinfer_fp8_cutlass_moe_prepare_finalize()`에서 `enable_alltoallv`를
factory 함수에 전달하지 않아 (`create_flashinfer_prepare_finalize()` 기본값 `False`),
`flashinfer_all2allv` 백엔드를 설정해도 실제로는 `FlashInferAllGatherMoEPrepareAndFinalize`
(AllGather+ReduceScatter)가 선택됩니다.

이번 작업에서는 미적용입니다.

---

## 수정 파일 목록

| 파일 | 수정 내용 |
|---|---|
| `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` | `handles_ep_dispatch_internally` property 추가 (기본값 False) |
| `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | `handles_ep_dispatch_internally` override, `all2all_backend` 전달 |
| `vllm/model_executor/layers/fused_moe/layer.py` | `do_naive_dispatch_combine` 조건에 property 추가 |
| `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` | `(not use_dp)` 제한 제거, `all2all_backend` 기반 자동 활성화, DP용 prepare_finalize 분기; **`swap_w13_to_w31`에 `is_act_and_mul` guard 추가** |
| `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` | BF16 branch 추가; `NCCLAllToAllMoEPrepareAndFinalize` 구현; `_is_mnnvl_available()` 헬퍼; `create_flashinfer_prepare_finalize()` MNNVL/NCCL 자동 선택; **`_ep_pg()`에서 `cpu_group`→`device_group` 수정** |
| `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` | `build_flashinfer_bf16_cutlass_moe_prepare_finalize()` 추가 및 단순화 |
| `vllm/distributed/device_communicators/all2all.py` | `FlashInferAllToAllManager`에 fallback `dispatch()`/`combine()` 구현; `ensure_alltoall_workspace_initialized()`에 try/except 추가; **`torch.cuda.device_count` → `torch.cuda.device_count()` 괄호 누락 버그 수정** |
| `vllm/distributed/device_communicators/mnnvl_compat.py` | `CustomCommunicator`에 `barrier()`, `bcast()` 구현 |

---

## 상세 변경 내용

---

### 1. `fused_moe_method_base.py`

`handles_ep_dispatch_internally` property 추가.
quant method가 dispatch/combine을 내부적으로 처리하는지 `layer.py`에 알려주는 신호.

```python
@property
def handles_ep_dispatch_internally(self) -> bool:
    """True이면 layer.py의 naive dispatch/combine 경로를 건너뜀."""
    return False
```

---

### 2. `unquantized_fused_moe_method.py`

**(a)** `handles_ep_dispatch_internally` override:

```python
@property
def handles_ep_dispatch_internally(self) -> bool:
    # FlashInfer CUTLASS는 prepare_finalize 안에서 dispatch/combine 처리
    return self.unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
```

**(b)** `select_unquantized_moe_backend` 호출 시 `all2all_backend` 전달:

```python
self.unquantized_backend = select_unquantized_moe_backend(
    use_ep=self.moe.moe_parallel_config.use_ep,
    use_dp=self.moe.moe_parallel_config.dp_size > 1,
    all2all_backend=self.moe.moe_parallel_config.all2all_backend,  # 추가
)
```

---

### 3. `layer.py`

`do_naive_dispatch_combine` 조건에 `handles_ep_dispatch_internally` 추가:

```python
# 변경 전
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
)

# 변경 후
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
    and not quant_method.handles_ep_dispatch_internally  # 추가
)
```

---

### 4. `oracle/unquantized.py`

**(a)** `select_unquantized_moe_backend()` 시그니처 변경 + 자동 활성화 로직:

```python
def select_unquantized_moe_backend(
    use_ep: bool,
    use_dp: bool,
    all2all_backend: str = "",   # 추가
) -> UnquantizedMoeBackend:

    # --all2all-backend flashinfer_all2allv 설정 시 자동으로 FlashInfer CUTLASS 활성화.
    # VLLM_USE_FLASHINFER_MOE_FP16=1 없이도 동작하도록 OR 조건 추가.
    use_flashinfer_cutlass = (
        envs.VLLM_USE_FLASHINFER_MOE_FP16          # 기존 방식 (호환성 유지)
        or all2all_backend == "flashinfer_all2allv" # 신규: backend 설정만으로 충분
    )
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and use_flashinfer_cutlass
        and use_ep
        # and (not use_dp)  ← 이 제한 제거: DP>1도 FlashInfer CUTLASS 허용
        and current_platform.get_device_capability()[0] >= 9
    )
```

**(b)** `make_unquantized_moe_kernel()`: DP>1이면 alltoallv-aware prepare_finalize 사용:

```python
use_dp = moe_config.moe_parallel_config.dp_size > 1
if use_dp:
    prepare_finalize = build_flashinfer_bf16_cutlass_moe_prepare_finalize(moe_config)
else:
    prepare_finalize = MoEPrepareAndFinalizeNoEP()
```

**(c)** `convert_to_unquantized_kernel_format()`: 비게이트 활성화 모델에서 `swap_w13_to_w31` 오적용 수정:

```python
# 수정 전 (버그): 비게이트 활성화에도 무조건 swap 적용 → w13_weight 손상
elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
    w13_weight = swap_w13_to_w31(layer.w13_weight.data)

# 수정 후: is_act_and_mul=True(게이트 활성화)일 때만 swap
elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
    if layer.moe_config.is_act_and_mul:
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)
    else:
        w13_weight = layer.w13_weight.data
```

---

### 5. `flashinfer_cutlass_prepare_finalize.py`

#### 5-a) BF16 분기 추가 (`flashinfer_alltoall_dispatch`)

```python
if quant_config.quant_dtype is None:
    # BF16: quantize 없이 raw activations를 alltoallv로 전송
    x_sf = None
    x = MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info, ...)
elif not use_deepseek_fp8_block_scale:
    # FP8/NVFP4: quantize 후 전송 (기존)
    ...
else:
    # DeepSeek block-scale (기존)
    ...
```

#### 5-b) `_is_mnnvl_available()` 헬퍼 (신규)

```python
@lru_cache(maxsize=None)
def _is_mnnvl_available(device_idx: int) -> bool:
    """MNNVL NVLink fabric 지원 여부 확인 (device_idx 기준, 결과 캐시).
    GB200 NVL72 등 NVSwitch 기반에서 True, H100/A100 등에서 False 반환.
    """
    try:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported
        return bool(is_mnnvl_fabric_supported(device_idx))
    except Exception:
        return False
```

#### 5-c) `NCCLAllToAllMoEPrepareAndFinalize` (신규)

H100 등 MNNVL 미지원 하드웨어에서 표준 NCCL을 이용한 AllToAllv 구현입니다.

**dispatch 단위: (token, rank) 쌍**

각 token에 대해 "어느 rank에 expert가 있는지"를 먼저 파악하고,
해당 rank마다 **전체 top_k 정보와 함께** token을 한 번씩 전송합니다.
expert 커널은 전체 topk_ids를 받되, ep_rank/ep_size로 비로컬 expert를 자동 skip하여
로컬 기여(partial sum)만 출력합니다. finalize에서 역방향 AllToAll 후 index_add_로 누적합니다.

```
이전(버그): (token t, expert e) →  recv_topk_ids [R, 1]   top_k=1  → tactic miss
현재(수정): (token t, rank r)   →  recv_topk_ids [R, K]   top_k=K  → tactic 정상
```

**알고리즘:**

```
prepare():
  1. _get_expert_to_rank()로 expert→rank 매핑 구성 (아래 별도 설명)
  2. expert_ranks[T, K] = expert_to_rank[topk_ids]  — 각 (token, k) 쌍의 목적 rank
  3. token_needs_rank[T, R] = scatter: token t가 rank r에 ≥1개 expert를 가지면 True
  4. nonzero → (token_idx, dest_rank) 쌍 추출, dest_rank 기준 stable argsort 정렬
  5. all_to_all_single: counts 교환 → recv_sizes 결정
  6. all_to_all_single: hidden states 교환  a1[token_idx_sorted] → recv_hidden[total_recv, H]
  7. all_to_all_single: global topk_ids 교환  →  recv_topk_ids[total_recv, K]
  8. all_to_all_single: topk_weights 교환   →  recv_topk_weights[total_recv, K]
  9. 수신된 activations quantize (필요 시)
  10. return recv_hidden_q, scale, None, recv_topk_ids[total_recv, K], recv_weights[total_recv, K]

finalize():
  1. all_to_all_single: expert 출력을 origin rank로 역송신 (send/recv sizes 스왑)
  2. output.zero_() + index_add_: output[token_idx_sorted[i]] += combined[i]  (partial sum 누적, BF16)
  ※ weight_and_reduce_impl은 TopKWeightAndReduceNoOP — FlashInfer CUTLASS 커널이
     ep_rank/ep_size 기반으로 비로컬 expert를 0으로 masking하고, router weight를
     내부적으로 적용한 weighted partial sum을 직접 출력하므로 finalize에서
     별도 weight scaling 불필요. index_add_가 바로 최종 합산.
```

핵심 특성:
- MNNVL workspace 없이 순수 NCCL만 사용
- token 당 top_k=K 전체를 전달 → FlashInfer CUTLASS 컴파일된 tactic 그대로 사용
- expert 커널(ep_rank, ep_size 기반)이 비로컬 expert를 skip → 로컬 partial sum만 출력
- dispatch 상태(`_NCCLDispatchState`)를 instance에 저장하여 finalize에서 재사용
- **`_ep_pg()`는 반드시 `device_group` 사용** (아래 주의 참조)

```python
@dataclass
class _NCCLDispatchState:
    local_token_count: int
    send_sizes: list        # CPU list for all_to_all_single
    recv_sizes: list        # CPU list for all_to_all_single
    # [total_send] — original local token index for every dispatched (token, rank) pair,
    # sorted by dest_rank so that combined[i] corresponds to token_indices[i]
    # after the reverse all_to_all_single.
    token_indices: torch.Tensor

class NCCLAllToAllMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):

    def _ep_pg(self):
        ep_group = get_ep_group()
        # cpu_group은 gloo 백엔드 → GPU 텐서 통신 불가
        # device_group은 NCCL 백엔드 → GPU 텐서 통신 가능
        return ep_group.rank_in_group, ep_group.world_size, ep_group.device_group
```

**`_get_expert_to_rank()` — expert→rank 매핑 빌더 (신규)**

global expert ID를 소유 rank로 변환하는 `[num_experts]` int64 텐서를 반환합니다.
결과는 `(num_experts, ep_size)` 키로 instance에 캐시되어, 첫 번째 호출 이후 재계산 없이 재사용됩니다.

두 가지 경로:

| 경로 | 조건 | 방법 |
|---|---|---|
| arithmetic | `expert_map is None` | `torch.arange(num_experts) // (num_experts // ep_size)` — 균등 연속 할당 가정 |
| all-gather | `expert_map is not None` | 각 rank의 소유 마스크를 all-gather → 역매핑 구성, 미할당 expert assert |

```python
# arithmetic 경로 (expert_map=None, 현재 vLLM 기본)
result = torch.arange(num_experts, device=device) // local_num_experts
# → expert 0-15 → rank 0, 16-31 → rank 1, ... (ep_size=8, 128 experts 기준)

# all-gather 경로 (expert_map 제공 시)
owns = (expert_map >= 0).to(torch.int32)
all_owns = [torch.empty_like(owns) for _ in range(ep_size)]
dist.all_gather(all_owns, owns, group=pg)
result = torch.full((num_experts,), -1, dtype=torch.long, device=device)
for r, owns_r in enumerate(all_owns):
    result[owns_r.bool()] = r
```

> **참고**: `FusedMoEModularMethod.apply()`는 `FlashInferExperts.supports_expert_map()==False`이므로
> `expert_map=None`을 항상 전달합니다. 현재 실제 경로는 항상 arithmetic 경로입니다.

> **⚠ `cpu_group` vs `device_group` 함정**
>
> vLLM의 `GroupCoordinator`는 두 가지 process group을 갖습니다:
> - `cpu_group`: **gloo** 백엔드, CPU 텐서 전용 (barrier, broadcast_object_list 등)
> - `device_group`: **NCCL** 백엔드, GPU 텐서 collective 전용
>
> `dist.all_to_all_single`에 CUDA 텐서를 전달할 때 `cpu_group`(gloo)을 사용하면
> 통신이 내부적으로 잘못 처리되어 **수치 오류(모델이 "1.1.1.1..." 반복 생성)** 가
> 발생합니다. 반드시 `device_group`을 사용해야 합니다.

#### 5-d) `create_flashinfer_prepare_finalize()` 업데이트

```python
def create_flashinfer_prepare_finalize(use_dp, use_nvfp4, enable_alltoallv, ...):
    if use_dp:
        if enable_alltoallv:
            device_idx = torch.cuda.current_device()
            if _is_mnnvl_available(device_idx):
                # GB200 NVL72 등 MNNVL 하드웨어
                return FlashInferAllToAllMoEPrepareAndFinalize(use_dp)
            else:
                # H100, A100 등 비-MNNVL → NCCL 기반 구현
                return NCCLAllToAllMoEPrepareAndFinalize(use_dp, ...)
        return FlashInferAllGatherMoEPrepareAndFinalize(use_dp=True, ...)
    else:
        return MoEPrepareAndFinalizeNoEP(...)
```

---

### 6. `flashinfer_utils.py`

BF16용 빌더 함수 신규 추가 (하드웨어 감지 로직은 `create_flashinfer_prepare_finalize`에 위임):

```python
def build_flashinfer_bf16_cutlass_moe_prepare_finalize(
    moe: FusedMoEConfig | None,
) -> mk.FusedMoEPrepareAndFinalize:
    use_dp = moe.moe_parallel_config.dp_size > 1
    enable_alltoallv = (
        moe.moe_parallel_config.all2all_backend == "flashinfer_all2allv"
    )
    return create_flashinfer_prepare_finalize(
        use_dp=use_dp,
        use_nvfp4=False,
        enable_alltoallv=enable_alltoallv,
    )
```

`create_flashinfer_prepare_finalize()` 내부에서 `_is_mnnvl_available()`로 하드웨어를 감지하여
MNNVL 하드웨어면 `FlashInferAllToAllMoEPrepareAndFinalize`, 비-MNNVL이면
`NCCLAllToAllMoEPrepareAndFinalize`가 자동 선택됩니다.

---

### 7. `all2all.py` — `FlashInferAllToAllManager`

`dispatch()`/`combine()` fallback 구현 (AllGather+ReduceScatter).

FlashInfer는 AllToAllv를 경로 B(modular kernel)로 처리하도록 설계되었기 때문에,
경로 A에서 사용하는 AllGather 기반 `dispatch()`/`combine()`을 원래 구현하지 않았습니다.

하지만 FlashInfer CUTLASS 선택 조건을 만족하지 못하는 환경(예: SM90 미만 GPU)에서
TRITON backend로 fallback되면, `handles_ep_dispatch_internally=False`가 되어 경로 A로
진입하고 `FlashInferAllToAllManager.dispatch()`가 호출됩니다.

이 경우 에러 없이 동작하도록 fallback 구현 추가:

```python
def dispatch(self, hidden_states, router_logits, ...):
    """Fallback: AllGather (AgRsAll2AllManager와 동일)"""
    ...

def combine(self, hidden_states, ...):
    """Fallback: ReduceScatter (AgRsAll2AllManager와 동일)"""
    ...
```

또한 `ensure_alltoall_workspace_initialized()`에서 MNNVL 초기화 실패 시 크래시 대신
`False` 반환하도록 try/except 추가:

```python
def ensure_alltoall_workspace_initialized(self):
    if not self.initialized:
        try:
            self.initialize(...)
        except RuntimeError as e:
            hint = ""
            if "pidfd_getfd" in str(e):
                hint = " MNNVL requires SYS_PTRACE. Add --cap-add=SYS_PTRACE."
            logger.error("FlashInfer AllToAll init failed: %s%s", e, hint)
            return False
    return self.initialized
```

#### 버그 수정: `torch.cuda.device_count` 괄호 누락

`initialize()` 호출 시 `gpus_per_node` 인자에 함수 객체를 전달하는 버그 수정:

```python
# 수정 전 (버그): 함수 객체를 전달 → Mapping() 내부에서 TypeError 발생
#   → except RuntimeError에 잡히지 않아 서버 크래시
gpus_per_node=torch.cuda.device_count,

# 수정 후: 함수를 호출하여 정수값 전달
gpus_per_node=torch.cuda.device_count(),
```

`TypeError`는 `RuntimeError`의 하위 타입이 아니므로 try/except에 잡히지 않고
서버 전체가 크래시됩니다. GB200 등 MNNVL 경로에서만 해당되며, H100(NCCL 경로)은
`ensure_alltoall_workspace_initialized()`를 호출하지 않으므로 영향 없습니다.

---

### 8. `mnnvl_compat.py` — `CustomCommunicator`

FlashInfer AllToAllv 경로에서 `ensure_alltoall_workspace_initialized()` → `initialize()` →
`MnnvlConfig(comm_backend=CustomCommunicator(...))` 순서로 workspace를 초기화합니다.

`CustomCommunicator`는 `flashinfer.comm.mnnvl.CommBackend` 추상 클래스를 구현하는데,
`barrier()`와 `bcast()` 두 메서드가 누락되어 `TypeError: Can't instantiate abstract class`가
발생했습니다. `torch.distributed` API로 구현 추가:

```python
def bcast(self, data, root: int):
    container = [data]
    dist.broadcast_object_list(container, src=root, group=self._group)
    return container[0]

def barrier(self) -> None:
    dist.barrier(group=self._group)
```

---

## 최종 동작 흐름

### GB200 NVL72 (MNNVL ✓) — `BF16 + DP>1 + flashinfer_all2allv`

```
[모델 로딩]
  select_unquantized_moe_backend(): all2all_backend == "flashinfer_all2allv" → FLASHINFER_CUTLASS
  make_unquantized_moe_kernel(): use_dp=True → build_flashinfer_bf16_cutlass_moe_prepare_finalize()
  create_flashinfer_prepare_finalize(): _is_mnnvl_available()=True
    → FlashInferAllToAllMoEPrepareAndFinalize (MNNVL AllToAllv)

[매 forward step]
  layer.py: handles_ep_dispatch_internally=True → do_naive_dispatch_combine=False
  quant_method.apply() → kernel()
    prepare(): flashinfer_alltoall_dispatch() → mnnvl_moe_alltoallv()  (MNNVL 직접 전송)
    FlashInferExperts()
    finalize(): flashinfer_alltoall_combine() → mnnvl_moe_alltoallv_combine()
```

### H100 (MNNVL ✗) — `BF16 + DP>1 + flashinfer_all2allv`

```
[모델 로딩]
  select_unquantized_moe_backend(): FLASHINFER_CUTLASS
  create_flashinfer_prepare_finalize(): _is_mnnvl_available()=False
    → NCCLAllToAllMoEPrepareAndFinalize (NCCL AllToAllv)

[매 forward step]
  layer.py: handles_ep_dispatch_internally=True → do_naive_dispatch_combine=False
  quant_method.apply() → kernel()
    prepare():
      _get_expert_to_rank(): expert→rank 매핑 (캐시)
      token_needs_rank[T, R]: token t가 rank r에 expert를 가지면 True
      (token_idx, dest_rank) 쌍 추출 → dest_rank 기준 stable sort
      dist.all_to_all_single ×4 (counts, hidden, expert_ids[K], weights[K])
      → recv_hidden[total_recv, H], topk_ids[total_recv, K], weights[total_recv, K]
    FlashInferExperts() — top_k=K로 처리 (ep_rank/ep_size 기반 비로컬 expert skip)
    finalize():
      dist.all_to_all_single (expert 출력 역송신, send/recv sizes 스왑)
      index_add_: output_buf[token_idx_sorted[i]] += combined[i]  (partial sum 누적)
      output.copy_(output_buf)
```

---

## 케이스별 동작 요약

| 모델 | backend | 하드웨어 | 이전 상태 | 수정 후 |
|---|---|---|---|---|
| BF16+DP>1 | `flashinfer_all2allv` | GB200 NVL72 (MNNVL ✓) | `NotImplementedError` 크래시 | FlashInfer CUTLASS + MNNVL AllToAllv ✓ |
| BF16+DP>1 | `flashinfer_all2allv` | H100 (MNNVL ✗) | `AssertionError` 크래시 | FlashInfer CUTLASS + **NCCL AllToAllv** ✓ |
| BF16+DP>1 | 기타 | 무관 | `NotImplementedError` 크래시 | TRITON + AllGather+ReduceScatter (fallback) ✓ |
| FP8+DP>1 | `flashinfer_all2allv` | 무관 | 에러 없지만 AllGather 사용 (성능 개선 없음) | **미수정** — 동일하게 AllGather 사용 |
| NVFP4+DP>1 | `flashinfer_all2allv` | GB200 NVL72 (MNNVL ✓) | 정상 동작 | 변경 없음 |
| 모든 모델 | `flashinfer_all2allv` | GB200 NVL72 (MNNVL ✓) | MNNVL init 시 `TypeError` 서버 크래시 (`torch.cuda.device_count` 괄호 누락) | **괄호 추가** — `device_count()` 수정 ✓ |
| BF16+DP>1 | `flashinfer_all2allv` | H100 (NCCL AllToAllv) | NCCL 경로에서 experts가 여러 rank에 걸친 token의 정합성 깨짐 (비로컬 expert weight 미마스킹) | **비로컬 expert weight 마스킹** — `recv_topk_weights_full * is_local_expert` 수정 ✓ |
| **BF16+DP>1, 비게이트 활성화 (relu2_no_mul)** | `flashinfer_all2allv` | H100 (NCCL AllToAllv) | **모든 추론 결과가 쓰레기** — "1.8.8.8..." 반복 출력 | **`swap_w13_to_w31` guard 추가** — `is_act_and_mul=False`이면 swap 생략 ✓ |

---

## 주의사항

- `VLLM_USE_FLASHINFER_MOE_FP16=1` 환경 변수 없이도 `--all2all-backend flashinfer_all2allv`만으로 자동 활성화됨
- MNNVL 하드웨어 감지(`_is_mnnvl_available`)는 device_idx 기준으로 **한 번만** 실행되고 캐시됨
- H100에서 `NCCLAllToAllMoEPrepareAndFinalize`는 FlashInfer CUTLASS expert 커널을 그대로 사용하므로
  TRITON 대비 성능 이점은 유지됨. 단, MNNVL 직접 메모리 전송이 아닌 NCCL 통신을 사용
- 비게이트 활성화 모델(`is_act_and_mul=False`, relu2_no_mul 등)에서는 `swap_w13_to_w31`을 적용하면
  안 됨 — `oracle/unquantized.py`의 `convert_to_unquantized_kernel_format`에 guard 추가됨

### 비로컬 expert weight 미마스킹 → 요청별 간헐적 정합성 오류

**증상**: 일부 요청은 처음부터 끝까지 정합성이 깨진 결과가 나오고, 다른 요청은 정상 동작하는 패턴이 반복됩니다. 나쁜 요청은 첫 생성 토큰부터 잘못되어 끝까지 유지됩니다.

**원인**: MNNVL 경로(`mnnvl_moe_alltoallv_prepare_without_allgather`)는 dispatch 전에 `topk_ids`/`topk_weights`를 **로컬 expert만 남기도록 필터링**합니다. 즉, 수신 rank의 커널은 자신의 local expert ID만 봅니다.

그러나 NCCL 경로는 전체 K개 global expert ID를 수신 rank에 그대로 전달합니다. 커널이 `ep_rank/ep_size`로 비로컬 expert를 올바르게 skip하지 않을 경우, 비로컬 expert의 기여까지 partial sum에 포함되어 double-counting이 발생합니다.

```
예시 (ep_size=2, ep_rank=0, top_k=4):
  Token T: topk_ids=[e0, e1, e2, e3], weights=[w0, w1, w2, w3]
  e0, e1 → rank 0 (로컬)   /   e2, e3 → rank 1 (비로컬)

  올바른 동작:
    rank 0 partial sum = w0*e0(T) + w1*e1(T)
    rank 1 partial sum = w2*e2(T) + w3*e3(T)
    → 합산: 전체 MoE 출력 ✓

  NCCL 버그 (커널이 비로컬 skip 안 함):
    rank 0 출력 = w0*e0(T) + w1*e1(T) + w2*WRONG(T) + w3*WRONG(T)  ✗
    rank 1 출력 = w0*WRONG(T) + w1*WRONG(T) + w2*e2(T) + w3*e3(T) ✗
    → 합산: 쓰레기 값
```

**왜 간헐적인가**: top-k expert가 모두 같은 rank에 있는 token은 1개 rank에만 dispatch되어 비로컬 expert가 없으므로 정상 동작합니다. expert가 여러 rank에 걸친 token만 오류가 발생합니다. 같은 prompt로 만들어진 KV 캐시는 일관성이 있으므로 해당 요청은 처음부터 끝까지 동일한 품질을 유지합니다.

**수정**: `recv_topk_weights_full`에서 비로컬 expert의 weight를 0으로 마스킹합니다.
이 방법은 커널의 EP filtering 구현 여부와 무관하게 항상 올바른 partial sum을 보장합니다.

```python
# NCCL 수신 후, 비로컬 expert weight 마스킹 (Step 6b)
if ep_size > 1 and total_recv > 0:
    is_local_expert = (expert_to_rank[recv_topk_ids_full.long()] == ep_rank)
    recv_topk_weights_full = recv_topk_weights_full * is_local_expert.to(recv_topk_weights_full.dtype)
```

이로써 커널은 비로컬 expert에 대해 `0 * expert_e(token) = 0`을 계산하여 partial sum에 영향 없음.

---

### `finalize()` accumulation dtype — FP32 버퍼 제거 경위

초기 구현에서는 `index_add_`의 BF16 비결정성을 우려해 FP32 임시 버퍼를 사용했습니다:

```python
# 초기 구현 (제거됨)
output_buf = torch.zeros(local_token_count, hidden_size, dtype=torch.float32, device=device)
output_buf.index_add_(0, state.token_indices, combined.float())
output.copy_(output_buf)

# 현재 구현
output.zero_()
output.index_add_(0, state.token_indices, combined)  # BF16 그대로
```

**FP32 버퍼를 제거한 이유:**

1. **오차가 실제로 무시할 수준**: ep_size=8에서 토큰당 최대 8회 누적 시 BF16 비결정성으로 발생하는 오차는 ~0.01% 수준입니다. 최종 logit 차이를 뒤집을 만큼 크지 않습니다.

2. **근본 원인이 아니었음**: FP32 버퍼를 추가했을 때도 정합성 문제가 계속 발생했습니다. 실제 원인은 `swap_w13_to_w31` 오적용이었습니다.

3. **NVFP4(MNNVL 경로)와 일관성**: NVFP4의 `mnnvl_moe_alltoallv_combine`도 BF16으로 합산합니다(아래 참조). FP32 버퍼는 오히려 두 경로 간 불일치를 만드는 것이었습니다.

4. **매 forward마다 불필요한 overhead**: `[local_tokens, hidden_size]` FP32 텐서 할당 + `.float()` 변환 + `.copy_()`가 매 step 발생했습니다.

**NVFP4의 expert 출력 dtype 흐름:**

NVFP4는 weight/activation의 *저장 포맷*일 뿐, expert 커널 밖으로 나오는 결과는 BF16입니다.

```
[입력]  NVFP4 quantized activations + NVFP4 weights
         ↓
[커널 내부] Tensor Core NVFP4 GEMM
            누적(accumulation): FP32  (Tensor Core 표준)
            출력 write-back:    BF16  (dequantize)
         ↓
[커널 출력] fused_expert_output: BF16
         ↓
[mnnvl_moe_alltoallv_combine]
  output_tensor = torch.zeros(..., dtype=x.dtype)  # BF16
  moe_comm(...)  # rank 간 gather
  return torch.sum(output_tensor.reshape(token_count, top_k, H), dim=1)  # BF16 sum
```

따라서 AllToAll combine 단계에서는 NVFP4/BF16 구분 없이 항상 BF16끼리 합산합니다.
NCCL 경로의 `index_add_`도 동일하게 BF16으로 처리하는 것이 일관성 있는 구현입니다.

---

### 비게이트 활성화(relu2_no_mul)에서 `swap_w13_to_w31` 오적용 → 완전한 정합성 파괴

**증상**: FlashInfer AllToAll 커널 적용 시 모든 추론 결과가 "1.8.8.8..." 또는 "and\nand\n..." 같은
무의미한 반복 토큰으로 출력됩니다. 디폴트 Triton 백엔드에서는 동일 모델이 정상 동작합니다.

**원인**: `convert_to_unquantized_kernel_format()`이 `FLASHINFER_CUTLASS` 백엔드에 대해
`swap_w13_to_w31()`를 **무조건 호출**했습니다.

`swap_w13_to_w31()`는 게이트 활성화(silu, geglu 등 `is_act_and_mul=True`) 모델의 `w13_weight`가
`[E, 2N, K]`(gate + up 두 행렬 연결) 형태일 때, 상하 절반을 교환하여
`[gate; up]` → `[up; gate]`(= `[w3; w1]`) 순서로 재배치하기 위한 함수입니다.

비게이트 활성화(relu2_no_mul, `is_act_and_mul=False`) 모델(예: Nemotron-3-Nano-30B)은
`w13_weight`를 `[E, N, K]` — 단일 프로젝션 행렬로 초기화합니다. 이 텐서에 `swap_w13_to_w31`를
적용하면 텐서의 앞/뒤 절반이 뒤섞여 가중치가 완전히 손상됩니다.

Triton 백엔드는 이 swap을 적용하지 않아 문제가 없었고, FlashInfer 경로에서만 항상 잘못된
결과가 나오는 이유가 이것이었습니다.

```
[게이트 활성화 — is_act_and_mul=True]
  w13_weight shape: [E, 2N, K]  (gate 절반 + up 절반)
  swap_w13_to_w31 적용 → [E, 2N, K] (up 절반 + gate 절반, 커널 기대 형식) ✓

[비게이트 활성화 — is_act_and_mul=False]
  w13_weight shape: [E, N, K]   (단일 프로젝션)
  swap_w13_to_w31 적용 → 텐서 전체가 반으로 쪼개져 섞임 ✗  → 쓰레기 출력
```

**수정**: `convert_to_unquantized_kernel_format()`에 `is_act_and_mul` 조건 추가.
FP8 경로는 이미 동일한 guard를 갖고 있었으며, 이번 수정으로 BF16 경로도 동일하게 정렬되었습니다.

---

### `cpu_group` vs `device_group` 수치 오류

`NCCLAllToAllMoEPrepareAndFinalize`의 `dist.all_to_all_single` 호출에는
반드시 **`ep_group.device_group`** (NCCL)을 사용해야 합니다.

`ep_group.cpu_group`은 gloo 백엔드로 생성된 CPU 전용 process group입니다.
이 그룹으로 GPU 텐서의 `all_to_all_single`을 호출하면 크래시 없이 잘못된 통신 결과를
반환하여, 모델이 **"1.1.1.1..." 형태의 의미없는 토큰을 반복 생성**하는 수치 오류가 발생합니다.

```python
# 잘못된 코드 (gloo 그룹 → GPU 통신 불가)
dist.all_to_all_single(..., group=ep_group.cpu_group)

# 올바른 코드 (NCCL 그룹 → GPU 통신 정상)
dist.all_to_all_single(..., group=ep_group.device_group)
```

### FlashInfer CUTLASS tactic 없음 → NCCL 연쇄 crash

**증상**: 서버 재시작 직후 첫 요청에서 아래 로그와 함께 모든 rank가 죽음:

```
get tactic <flashinfer.fused_moe.core...MoERunner> 43, due to
[rank*]:[W...] ProcessGroupNCCL::HeartbeatMonitor ... Failed to recv, got 0 bytes.
```

**원인**: 기존 `prepare()`는 각 **(token, expert) 쌍**을 개별 아이템으로 dispatch했습니다.
즉 `recv_topk_ids [total_recv, 1]`을 expert 커널에 넘겨 `top_k=1`이 되었습니다.
FlashInfer CUTLASS 커널은 `top_k=6`(모델 기본값)으로 컴파일되어 있어 `top_k=1`에 대한
tactic이 없으므로 한 rank가 crash → 나머지 rank는 NCCL collective에서 대기 → 타임아웃으로
HeartbeatMonitor가 전체를 종료합니다.

**수정 (현재 구현)**: dispatch 단위를 **(token, rank) 쌍**으로 변경합니다.
각 token에 대해 하나 이상의 expert가 있는 rank마다 **전체 top_k 정보와 함께** token을
한 번씩 전송합니다. expert 커널은 원래 `top_k=K`를 그대로 받고,
ep_rank/ep_size 기반으로 비로컬 expert를 skip하여 로컬 기여(partial sum)만 출력합니다.

```
이전: (token t, expert e)  →  recv_topk_ids [R, 1]   top_k=1  → tactic miss
현재: (token t, rank r)    →  recv_topk_ids [R, K]   top_k=K  → tactic 정상
```

`finalize()`는 각 rank의 partial weighted sum을 `index_add_`로 원래 token 위치에 누적합니다.

> **주의**: 문서의 `expert_map`으로 비로컬 expert를 걸러낸다는 표현은 엄밀히는 부정확합니다.
> 실제로는 FlashInfer CUTLASS 커널이 ep_rank/ep_size 파라미터를 이용해 내부적으로
> 로컬 expert 범위 `[ep_rank * local_num_experts, (ep_rank+1) * local_num_experts)`를 계산하고,
> 이 범위 밖의 expert ID는 기여를 0으로 처리합니다. `expert_map=None`이 전달되더라도
> ep_rank가 각 rank에 올바르게 0~7로 설정되어 있어 정상 동작합니다.

### 컨테이너 권한 문제 (`pidfd_getfd` Permission denied)

MNNVL workspace 초기화 시 Linux `pidfd_getfd()` syscall이 사용됩니다. 컨테이너 환경에서
`SYS_PTRACE` capability가 없으면:

```
RuntimeError: pidfd_getfd(pidfd=..., fd=...) failed with errno 1: Operation not permitted.
```

**해결 방법**: Docker 실행 시 `--cap-add=SYS_PTRACE` 추가

```bash
docker run --cap-add=SYS_PTRACE ...
```

H100 환경에서는 이 에러와 무관하게 NCCL AllToAll로 자동 전환됩니다.

---

## 디버그 방법

별도의 디버그 로그 출력은 없습니다. 동작 확인이 필요하면 `prepare()` / `finalize()` 내에
임시로 print/logging을 추가하거나 `test_nccl_dispatch.py`를 실행하세요.

MNNVL 하드웨어 감지 실패 시(GB200 환경에서 예외 발생) 다음 경고가 출력됩니다:
```
WARNING Could not query MNNVL fabric support for device 0 (...). Assuming unavailable.
```
이 경우 자동으로 NCCL AllToAll 경로로 fallback됩니다.
