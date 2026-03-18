# FlashInfer AllToAllv BF16+DP>1 지원 구현 내역

## TL;DR

`--all2all-backend flashinfer_all2allv` 옵션을 BF16(unquantized) 모델 + DP>1 환경에서 쓰면
`NotImplementedError`로 크래시됐습니다. 구조적 원인 5가지를 모두 수정하고,
H100(비-MNNVL) 환경을 위한 NCCL 기반 AllToAllv 구현을 새로 추가했습니다.

> FP8+DP>1의 AllGather 사용 문제(에러는 없지만 성능 개선 없음)는 이번 작업 범위 밖입니다.

---

## 1. 배경 지식: vLLM MoE의 두 가지 dispatch 경로

`layer.py`의 `forward_impl()`은 DP>1일 때 아래 조건으로 분기합니다.

```python
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
)
```

```
┌─────────────────────────────────────────────────────────────────┐
│  경로 A  "naive"  (do_naive_dispatch_combine = True)            │
│                                                                  │
│  layer.py가 직접 제어:                                          │
│    get_ep_group().dispatch()  ← 실제 구현은 AllGather           │
│    → (expert 계산)                                              │
│    → get_ep_group().combine() ← 실제 구현은 ReduceScatter       │
│                                                                  │
│  사용되는 manager: AgRsAll2AllManager, FlashInferAllToAllManager │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  경로 B  "modular kernel"  (do_naive_dispatch_combine = False)  │
│                                                                  │
│  quant_method.apply()                                            │
│    → FusedMoEModularKernel.__call__()                           │
│      → prepare_finalize.prepare()  ← dispatch가 여기서 발생    │
│      → (expert 계산)                                            │
│      → prepare_finalize.finalize() ← combine이 여기서 발생     │
│                                                                  │
│  FP8, NVFP4 등 FusedMoEModularMethod로 래핑된 모델이 사용      │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 포인트**: `FlashInferAllToAllManager`는 AllToAllv를 **경로 B**의 `prepare_finalize` 안에서만
처리하도록 설계됐습니다. 경로 A의 AllGather 기반 `dispatch()`는 원래부터 구현 대상이 아니었습니다.

---

## 2. 왜 BF16+DP>1이 동작하지 않았는가

BF16 모델은 `FusedMoEModularMethod`가 아니므로 `do_naive_dispatch_combine=True` → **경로 A**로
강제 진입합니다. 경로 A는 AllGather를 요구하는데 `FlashInferAllToAllManager`는 이를 구현하지
않았으므로 `NotImplementedError`가 발생합니다.

```
[BF16 + DP>1 + flashinfer_all2allv] 기존 흐름

do_naive_dispatch_combine = True   ← BF16은 FusedMoEModularMethod가 아님
  → get_ep_group().dispatch()      ← AllGather 요청
    → FlashInferAllToAllManager.dispatch()
      → All2AllManagerBase.dispatch() 상속
        → raise NotImplementedError  ✗
```

BF16이 경로 B(AllToAllv)로 진입하지 못한 구조적 제약이 **5겹**으로 쌓여 있었습니다.

| # | 위치 | 제약 | 결과 |
|---|---|---|---|
| 1 | `oracle/unquantized.py` | `and (not use_dp)` 하드코딩 | DP>1이면 `VLLM_USE_FLASHINFER_MOE_FP16=1`을 설정해도 TRITON으로 떨어짐 |
| 2 | `unquantized_fused_moe_method.py` | `FusedMoEModularMethod` 미상속 | `layer.py`가 modular dispatch 불가를 알 수 없어 경로 A 강제 진입 |
| 3 | `flashinfer_cutlass_prepare_finalize.py` | `assert use_nvfp4` | BF16(`use_nvfp4=False`)이면 assert 실패 |
| 4 | `flashinfer_alltoall_dispatch()` | BF16 분기 없음 | quantize 스킵 로직(`quant_dtype is None`)이 없어 BF16 tensor 처리 불가 |
| 5 | `mnnvl_compat.py` | `CommBackend` 추상 메서드 미구현 | `barrier()`, `bcast()` 누락 → workspace 초기화 자체가 `TypeError` |

이 5개가 모두 맞물려 있어 어떤 설정 조합을 줘도 BF16+DP>1은 동작할 수 없는 상태였습니다.

---

## 3. H100에서 FlashInfer AllToAllv를 쓸 수 없는 이유

FlashInfer의 `MnnvlMoe.mnnvl_moe_alltoallv()`는 **MNNVL(Multi-Node NVLink) fabric** 기반
shared memory를 workspace로 사용합니다.

```
MNNVL 지원 여부:
  GB200 NVL72 (NVSwitch 기반)  →  is_mnnvl_fabric_supported() = True  ✓
  H100 / A100 (일반 NVLink)    →  is_mnnvl_fabric_supported() = False  ✗
```

컨테이너 환경에서는 초기화 시 Linux `pidfd_getfd` syscall이 필요한데,
`SYS_PTRACE` capability가 없으면 `RuntimeError: pidfd_getfd ... Permission denied`로 실패합니다.

**해결 방향**: H100 등 비-MNNVL 환경에서는 표준 NCCL(`torch.distributed.all_to_all_single`)로
동작하는 새 구현체 `NCCLAllToAllMoEPrepareAndFinalize`를 도입했습니다.

---

## 4. 수정 전략 개요

두 가지 핵심 변경으로 문제를 해결했습니다.

### 4-a. BF16이 경로 B로 진입할 수 있도록 "신호" 추가

`FusedMoEMethodBase`에 `handles_ep_dispatch_internally` property를 추가하고,
`UnquantizedFusedMoEMethod`가 FlashInfer CUTLASS 백엔드일 때 `True`를 반환하도록 override.
`layer.py`의 분기 조건에 이 property를 추가해 경로 A를 우회.

```python
# layer.py — 변경 후
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
    and not quant_method.handles_ep_dispatch_internally  # ← 추가
)
```

### 4-b. 하드웨어에 따라 AllToAll 구현체 자동 선택

```
build_flashinfer_bf16_cutlass_moe_prepare_finalize()
  → create_flashinfer_prepare_finalize(use_dp=True, enable_alltoallv=True)
    → _is_mnnvl_available(device_idx)?
        True  → FlashInferAllToAllMoEPrepareAndFinalize  (MNNVL, 기존)
        False → NCCLAllToAllMoEPrepareAndFinalize        (NCCL, 신규)
```

---

## 5. 수정 파일 목록

| 파일 | 변경 유형 | 내용 |
|---|---|---|
| `fused_moe_method_base.py` | 기능 추가 | `handles_ep_dispatch_internally` property (기본값 `False`) |
| `unquantized_fused_moe_method.py` | 기능 추가 | 위 property override, `all2all_backend` 전달 |
| `layer.py` | 조건 수정 | `do_naive_dispatch_combine`에 property 조건 추가 |
| `oracle/unquantized.py` | 제한 해제 + 버그 수정 | `(not use_dp)` 제거, `all2all_backend` 자동 활성화, DP용 prepare_finalize 분기, `swap_w13_to_w31` guard 추가 |
| `flashinfer_cutlass_prepare_finalize.py` | 기능 추가 + 버그 수정 | BF16 분기, `NCCLAllToAllMoEPrepareAndFinalize` 구현, `_is_mnnvl_available()`, MNNVL/NCCL 자동 선택, `cpu_group`→`device_group` 수정 |
| `flashinfer_utils.py` | 기능 추가 | `build_flashinfer_bf16_cutlass_moe_prepare_finalize()` |
| `all2all.py` | 버그 수정 + 안전성 | fallback `dispatch()`/`combine()` 구현, try/except 추가, `device_count` 괄호 누락 수정 |
| `mnnvl_compat.py` | 버그 수정 | `CustomCommunicator`에 `barrier()`, `bcast()` 구현 |

---

## 6. 상세 변경 내용

### 6-1. `fused_moe_method_base.py`

dispatch/combine을 내부에서 처리하는지 `layer.py`에 알려주는 신호 property 추가.

```python
@property
def handles_ep_dispatch_internally(self) -> bool:
    """True이면 layer.py의 naive dispatch/combine 경로를 건너뜀."""
    return False  # 기본값: 경로 A 사용
```

---

### 6-2. `unquantized_fused_moe_method.py`

**(a)** FlashInfer CUTLASS일 때 dispatch 내부 처리 선언:

```python
@property
def handles_ep_dispatch_internally(self) -> bool:
    return self.unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
```

**(b)** backend 선택 시 `all2all_backend` 전달 (이전에는 전달 안 함):

```python
self.unquantized_backend = select_unquantized_moe_backend(
    use_ep=self.moe.moe_parallel_config.use_ep,
    use_dp=self.moe.moe_parallel_config.dp_size > 1,
    all2all_backend=self.moe.moe_parallel_config.all2all_backend,  # 추가
)
```

---

### 6-3. `layer.py`

`do_naive_dispatch_combine` 조건 완화:

```python
# 변경 전: FusedMoEModularMethod 여부만 체크
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
)

# 변경 후: 새 property도 체크 → BF16 FlashInfer CUTLASS가 경로 A를 우회 가능
do_naive_dispatch_combine = (
    dp_size > 1
    and not isinstance(quant_method, FusedMoEModularMethod)
    and not quant_method.handles_ep_dispatch_internally
)
```

---

### 6-4. `oracle/unquantized.py`

**(a)** `select_unquantized_moe_backend()`: DP>1 제한 제거 + 자동 활성화:

```python
def select_unquantized_moe_backend(
    use_ep: bool,
    use_dp: bool,
    all2all_backend: str = "",   # 파라미터 추가
) -> UnquantizedMoeBackend:

    use_flashinfer_cutlass = (
        envs.VLLM_USE_FLASHINFER_MOE_FP16           # 기존 방식 (호환성 유지)
        or all2all_backend == "flashinfer_all2allv"  # 신규: backend 설정만으로 충분
    )
    flashinfer_cutlass_moe_enabled = (
        has_flashinfer_cutlass_fused_moe()
        and use_flashinfer_cutlass
        and use_ep
        # and (not use_dp)  ← 이 제한 제거
        and current_platform.get_device_capability()[0] >= 9
    )
```

**(b)** `make_unquantized_moe_kernel()`: DP>1이면 AllToAllv-aware prepare_finalize 사용:

```python
use_dp = moe_config.moe_parallel_config.dp_size > 1
if use_dp:
    prepare_finalize = build_flashinfer_bf16_cutlass_moe_prepare_finalize(moe_config)
else:
    prepare_finalize = MoEPrepareAndFinalizeNoEP()
```

**(c)** `convert_to_unquantized_kernel_format()`: 비게이트 활성화 모델의 `swap_w13_to_w31` 오적용 수정
(자세한 내용은 [§ 버그: swap_w13_to_w31 오적용](#bug-swap) 참조):

```python
# 수정 전 (버그): 모든 모델에 무조건 swap
elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
    w13_weight = swap_w13_to_w31(layer.w13_weight.data)

# 수정 후: 게이트 활성화(is_act_and_mul=True)일 때만 swap
elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
    if layer.moe_config.is_act_and_mul:
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)
    else:
        w13_weight = layer.w13_weight.data  # 비게이트: swap 없이 그대로 사용
```

---

### 6-5. `flashinfer_cutlass_prepare_finalize.py`

#### (a) BF16 분기 추가 (`flashinfer_alltoall_dispatch`)

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

#### (b) `_is_mnnvl_available()` — 하드웨어 감지 헬퍼 (신규)

결과를 `lru_cache`로 캐시하여 첫 번째 호출 후 재쿼리 없이 재사용.

```python
@lru_cache(maxsize=None)
def _is_mnnvl_available(device_idx: int) -> bool:
    try:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported
        return bool(is_mnnvl_fabric_supported(device_idx))
    except Exception as exc:
        logger.warning("Could not query MNNVL fabric support for device %d (%s). "
                       "Assuming unavailable.", device_idx, exc)
        return False
```

#### (c) `create_flashinfer_prepare_finalize()` — 구현체 자동 선택

```python
def create_flashinfer_prepare_finalize(use_dp, use_nvfp4, enable_alltoallv, ...):
    if use_dp:
        if enable_alltoallv:
            device_idx = torch.cuda.current_device()
            if _is_mnnvl_available(device_idx):
                return FlashInferAllToAllMoEPrepareAndFinalize(use_dp)   # GB200
            else:
                return NCCLAllToAllMoEPrepareAndFinalize(use_dp, ...)    # H100
        return FlashInferAllGatherMoEPrepareAndFinalize(use_dp=True, ...)
    else:
        return MoEPrepareAndFinalizeNoEP(...)
```

#### (d) `NCCLAllToAllMoEPrepareAndFinalize` — H100용 신규 구현

자세한 내용은 [§ 7](#section-nccl)을 참조하세요.

---

### 6-6. `flashinfer_utils.py`

BF16용 빌더 함수 신규 추가. 하드웨어 감지 로직은 `create_flashinfer_prepare_finalize`에 위임.

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

---

### 6-7. `all2all.py` — `FlashInferAllToAllManager`

**Fallback `dispatch()`/`combine()` 추가**: FlashInfer CUTLASS 조건 불충족(예: SM90 미만)으로
TRITON으로 fallback되면 `handles_ep_dispatch_internally=False`가 되어 경로 A로 진입합니다.
이때 에러 없이 동작하도록 `AgRsAll2AllManager`와 동일한 AllGather 구현을 추가.

**`ensure_alltoall_workspace_initialized()` 개선**: MNNVL 초기화 실패 시 크래시 대신 `False` 반환.

**버그 수정 — `torch.cuda.device_count` 괄호 누락**:

```python
# 수정 전 (버그): 함수 객체가 전달됨 → Mapping() 내부에서 TypeError
#   TypeError는 RuntimeError의 하위 타입이 아니므로 try/except에 안 잡혀 서버 크래시
gpus_per_node=torch.cuda.device_count,

# 수정 후: 함수를 호출하여 정수값 전달
gpus_per_node=torch.cuda.device_count(),
```

---

### 6-8. `mnnvl_compat.py` — `CustomCommunicator`

`CommBackend` 추상 클래스의 `barrier()`, `bcast()` 미구현으로 workspace 초기화 시
`TypeError: Can't instantiate abstract class`가 발생했습니다. `torch.distributed` API로 구현 추가.

```python
def bcast(self, data, root: int):
    container = [data]
    dist.broadcast_object_list(container, src=root, group=self._group)
    return container[0]

def barrier(self) -> None:
    dist.barrier(group=self._group)
```

---

## 7. NCCLAllToAllMoEPrepareAndFinalize 상세 설계 {#section-nccl}

H100 등 MNNVL 미지원 하드웨어에서 표준 NCCL로 AllToAllv를 구현합니다.

### dispatch 단위: (token, rank) 쌍

FlashInfer CUTLASS 커널은 `top_k=K`로 컴파일됩니다. 초기 설계는 (token, expert) 쌍 단위로
dispatch하여 `top_k=1`이 됐고, tactic miss → rank crash → NCCL timeout으로 연쇄 장애가 발생했습니다.

```
이전(버그): (token t, expert e) →  recv_topk_ids [R, 1]   top_k=1  → tactic miss → crash
현재(수정): (token t, rank r)   →  recv_topk_ids [R, K]   top_k=K  → tactic 정상
```

각 token을 해당 token의 expert가 있는 **rank마다 한 번씩** 전송합니다.
전송할 때 전체 top_k 정보(K개 expert ID + weight)를 함께 보냅니다.
수신 rank의 커널은 ep_rank/ep_size 기반으로 비로컬 expert를 skip하여 로컬 partial sum만 출력합니다.

### 알고리즘

**prepare():**

```
1. _get_expert_to_rank():  expert ID → 소유 rank 매핑 [num_experts] 텐서 구성 (캐시)
2. expert_ranks[T, K]   =  expert_to_rank[topk_ids]   — 각 (token, k)의 목적 rank
3. token_needs_rank[T, R]: token t가 rank r에 ≥1개 expert를 가지면 True
4. nonzero() → (token_idx, dest_rank) 쌍 추출, dest_rank 기준 stable sort
5. all_to_all_single: counts 교환 → recv_sizes 결정
6. all_to_all_single: hidden states 교환  →  recv_hidden[total_recv, H]
7. all_to_all_single: topk_ids[K] 교환   →  recv_topk_ids[total_recv, K]
8. all_to_all_single: topk_weights[K] 교환 → recv_topk_weights[total_recv, K]
9. 비로컬 expert weight 마스킹 (Step 6b, 아래 참조)
10. 필요 시 activations quantize
→ return recv_hidden_q, scale, None, recv_topk_ids, recv_topk_weights
```

**finalize():**

```
1. all_to_all_single: expert partial sum을 origin rank로 역송신 (send/recv sizes 스왑)
2. output.zero_() + index_add_: output[token_idx[i]] += combined[i]  (BF16 누적)
   ※ FlashInfer CUTLASS 커널이 router weight를 내부 적용한 weighted partial sum을 출력하므로
      finalize에서 별도 weight scaling 불필요. index_add_가 바로 최종 합산.
```

### _get_expert_to_rank() — expert→rank 매핑 빌더

global expert ID를 소유 rank로 변환하는 `[num_experts]` int64 텐서를 반환합니다.
`(num_experts, ep_size)` 키로 캐시되어 재계산 없이 재사용됩니다.

| 경로 | 조건 | 방법 |
|---|---|---|
| arithmetic | `expert_map is None` (현재 vLLM 기본) | `torch.arange(num_experts) // (num_experts // ep_size)` |
| all-gather | `expert_map is not None` | 각 rank의 소유 마스크를 all-gather 후 역매핑 구성 |

```python
# arithmetic 경로 예시 (ep_size=8, 128 experts)
result = torch.arange(128) // 16
# → expert 0-15 → rank 0, 16-31 → rank 1, ..., 112-127 → rank 7
```

### 비로컬 expert weight 마스킹 (Step 6b)

MNNVL 경로는 dispatch 전에 topk_ids/topk_weights를 **로컬 expert만 남기도록 필터링**합니다.
NCCL 경로는 전체 K개 global expert ID를 그대로 전달하므로, 커널의 EP filtering 여부와 무관하게
비로컬 expert의 weight를 명시적으로 0으로 마스킹합니다.

```python
# Step 6b: 비로컬 expert weight 마스킹
if ep_size > 1 and total_recv > 0:
    is_local_expert = (expert_to_rank[recv_topk_ids_full.long()] == ep_rank)
    recv_topk_weights_full = recv_topk_weights_full * is_local_expert.to(dtype)
```

**마스킹이 없으면 어떤 문제가 발생하는가:**

```
예시 (ep_size=2, ep_rank=0, top_k=4):
  Token T: topk_ids=[e0, e1, e2, e3], weights=[w0, w1, w2, w3]
  e0, e1 → rank 0 (로컬),  e2, e3 → rank 1 (비로컬)

  올바른 동작:
    rank 0 partial sum = w0*e0(T) + w1*e1(T)                           ✓
    rank 1 partial sum =                     w2*e2(T) + w3*e3(T)       ✓

  마스킹 없는 버그 (커널이 비로컬 skip 안 할 경우):
    rank 0 출력 = w0*e0(T) + w1*e1(T) + w2*WRONG(T) + w3*WRONG(T)    ✗
    rank 1 출력 = w0*WRONG(T) + w1*WRONG(T) + w2*e2(T) + w3*e3(T)   ✗
```

**왜 간헐적으로만 발생하는가**: top-k expert가 모두 같은 rank에 있는 token은 1개 rank에만 dispatch되어
비로컬 expert가 없으므로 정상 동작합니다. expert가 여러 rank에 걸친 token만 오류가 발생합니다.
같은 prompt는 항상 같은 패턴이므로 해당 요청은 첫 토큰부터 끝까지 일관되게 깨집니다.

### `_ep_pg()` — 반드시 `device_group` 사용

```python
def _ep_pg(self):
    ep_group = get_ep_group()
    # cpu_group: gloo 백엔드 → CPU 텐서 전용. GPU 텐서로 호출 시 크래시 없이 잘못된 결과 반환
    # device_group: NCCL 백엔드 → GPU 텐서 collective 전용
    return ep_group.rank_in_group, ep_group.world_size, ep_group.device_group
```

> `cpu_group`(gloo)으로 GPU 텐서 `all_to_all_single`을 호출하면 에러 없이 잘못된 통신 결과를
> 반환하여 모델이 **"1.1.1.1..." 형태의 토큰을 반복 생성**합니다.

---

## 8. 디버깅 과정에서 발견한 버그들

### 버그 1: `swap_w13_to_w31` 비게이트 모델 오적용 {#bug-swap}

**증상**: FlashInfer 경로에서 모든 추론 결과가 "1.8.8.8..." 또는 "and\nand\n..." 같은 무의미한
반복 토큰으로 출력됩니다. Triton 백엔드에서는 동일 모델이 정상 동작합니다.

**원인**: `swap_w13_to_w31()`는 게이트 활성화(silu, geglu 등) 모델의 `w13_weight`가
`[E, 2N, K]`(gate + up 연결) 형태일 때 상하 절반을 `[gate; up]` → `[up; gate]`로 재배치하는 함수입니다.

비게이트 활성화(relu2_no_mul, `is_act_and_mul=False`) 모델(예: Nemotron-3-Nano-30B)은
`w13_weight`가 `[E, N, K]` — 단일 프로젝션이므로, swap을 적용하면 텐서 전체가 반으로 쪼개져 손상됩니다.

```
[게이트 활성화 — is_act_and_mul=True]
  w13_weight: [E, 2N, K]  (gate 절반 + up 절반 연결)
  swap 적용 → [E, 2N, K]  (up + gate 순서로 재배치, 커널 기대 형식) ✓

[비게이트 활성화 — is_act_and_mul=False]
  w13_weight: [E, N, K]   (단일 프로젝션, gate 없음)
  swap 적용 → 텐서 전체가 반으로 쪼개져 섞임 ✗  → 쓰레기 출력
```

FP8 경로는 이미 동일한 guard를 갖고 있었으나 BF16 경로에는 없었습니다.

---

### 버그 2: `torch.cuda.device_count` 괄호 누락

**증상**: MNNVL 경로 초기화 시 서버 전체 크래시.

**원인**: `Mapping()` 생성자에 정수가 아닌 함수 객체가 전달되어 `TypeError` 발생.
`except RuntimeError`는 `TypeError`의 부모가 아니므로 예외가 잡히지 않고 서버 크래시.

H100(NCCL 경로)은 `ensure_alltoall_workspace_initialized()`를 호출하지 않으므로 영향 없습니다.

---

### 버그 3: `cpu_group` vs `device_group` 혼용

**증상**: 모델이 "1.1.1.1..." 형태의 의미없는 토큰을 반복 생성.

**원인**: CUDA 텐서로 `all_to_all_single`을 호출할 때 gloo 백엔드인 `cpu_group`을 사용하면
크래시 없이 잘못된 통신 결과를 반환합니다.

---

### 버그 4: `CustomCommunicator` 추상 메서드 미구현

**증상**: `TypeError: Can't instantiate abstract class CustomCommunicator with abstract methods barrier, bcast`

**원인**: `CommBackend` 추상 클래스의 `barrier()`, `bcast()` 미구현.

---

### 설계 결정: `finalize()`에서 FP32 버퍼 사용 안 함

초기 구현에서는 `index_add_`의 BF16 비결정성을 우려해 FP32 임시 버퍼를 사용했으나 제거했습니다.

```python
# 초기 구현 (제거됨)
output_buf = torch.zeros(local_token_count, hidden_size, dtype=torch.float32)
output_buf.index_add_(0, state.token_indices, combined.float())
output.copy_(output_buf)

# 현재 구현
output.zero_()
output.index_add_(0, state.token_indices, combined)  # BF16 그대로
```

제거 이유:
1. **오차가 무시할 수준**: ep_size=8, 최대 8회 누적 기준 BF16 비결정성 오차 ~0.01%
2. **실제 버그 원인이 아님**: FP32 버퍼 추가 후에도 정합성 문제 지속 → 실제 원인은 `swap_w13_to_w31`
3. **NVFP4(MNNVL) 경로와 일관성**: `mnnvl_moe_alltoallv_combine`도 BF16으로 합산
4. **매 forward step마다 불필요한 오버헤드**: FP32 텐서 할당 + `.float()` + `.copy_()`

---

## 9. 최종 동작 흐름

### GB200 NVL72 (MNNVL 지원) — BF16 + DP>1 + flashinfer_all2allv

```
[모델 로딩]
  select_unquantized_moe_backend(): all2all_backend == "flashinfer_all2allv" → FLASHINFER_CUTLASS
  make_unquantized_moe_kernel(): use_dp=True
    → build_flashinfer_bf16_cutlass_moe_prepare_finalize()
    → create_flashinfer_prepare_finalize(): _is_mnnvl_available()=True
    → FlashInferAllToAllMoEPrepareAndFinalize

[매 forward step]
  layer.py: handles_ep_dispatch_internally=True → do_naive_dispatch_combine=False
  quant_method.apply() → kernel()
    prepare():  mnnvl_moe_alltoallv_prepare_without_allgather() → mnnvl_moe_alltoallv()
    FlashInferExperts()
    finalize(): mnnvl_moe_alltoallv_combine()
```

### H100 (MNNVL 미지원) — BF16 + DP>1 + flashinfer_all2allv

```
[모델 로딩]
  select_unquantized_moe_backend(): FLASHINFER_CUTLASS
  create_flashinfer_prepare_finalize(): _is_mnnvl_available()=False
    → NCCLAllToAllMoEPrepareAndFinalize

[매 forward step]
  layer.py: handles_ep_dispatch_internally=True → do_naive_dispatch_combine=False
  quant_method.apply() → kernel()
    prepare():
      _get_expert_to_rank(): expert→rank 매핑 (캐시)
      token_needs_rank[T, R]: token t가 rank r에 expert를 가지면 True
      (token_idx, dest_rank) 추출 → stable sort
      dist.all_to_all_single ×4 (counts, hidden, topk_ids[K], weights[K])
      비로컬 expert weight 마스킹
    FlashInferExperts() — top_k=K 그대로 처리
    finalize():
      dist.all_to_all_single (역방향, send/recv sizes 스왑)
      index_add_: output[token_idx[i]] += combined[i]
```

---

## 10. 케이스별 동작 요약

| 모델 | backend | 하드웨어 | 이전 | 수정 후 |
|---|---|---|---|---|
| BF16+DP>1 | `flashinfer_all2allv` | GB200 (MNNVL) | `NotImplementedError` | FlashInfer CUTLASS + MNNVL AllToAllv ✓ |
| BF16+DP>1 | `flashinfer_all2allv` | H100 (NCCL) | `AssertionError` | FlashInfer CUTLASS + NCCL AllToAllv ✓ |
| BF16+DP>1 | 기타 backend | 무관 | `NotImplementedError` | TRITON + AllGather+ReduceScatter (fallback) ✓ |
| BF16+DP>1 (비게이트) | `flashinfer_all2allv` | H100 | "1.8.8.8..." 반복 출력 | `swap_w13_to_w31` guard 추가 ✓ |
| 모든 모델 | `flashinfer_all2allv` | GB200 (MNNVL) | `TypeError` 서버 크래시 | `device_count()` 괄호 수정 ✓ |
| FP8+DP>1 | `flashinfer_all2allv` | 무관 | 에러 없음, AllGather 사용 | **미수정** — 동일하게 AllGather 사용 |
| NVFP4+DP>1 | `flashinfer_all2allv` | GB200 (MNNVL) | 정상 동작 | 변경 없음 |

---

## 11. 운영 주의사항

- `VLLM_USE_FLASHINFER_MOE_FP16=1` 환경 변수 없이도 `--all2all-backend flashinfer_all2allv`만으로 자동 활성화
- MNNVL 하드웨어 감지는 device_idx 기준 **한 번만** 실행 후 캐시됨
- H100에서도 FlashInfer CUTLASS expert 커널을 그대로 사용하므로 TRITON 대비 성능 이점은 유지됨. 단, MNNVL 직접 메모리 전송이 아닌 NCCL 통신을 사용
- 비게이트 활성화 모델(`is_act_and_mul=False`, relu2_no_mul 등)에서는 `swap_w13_to_w31` 적용 금지 — guard 추가됨

**컨테이너 권한 문제**: MNNVL workspace 초기화 시 `SYS_PTRACE` capability가 없으면:
```
RuntimeError: pidfd_getfd(pidfd=..., fd=...) failed with errno 1: Operation not permitted.
```
→ Docker 실행 시 `--cap-add=SYS_PTRACE` 추가 필요. H100 환경에서는 자동으로 NCCL 경로로 전환됩니다.

---

## 12. 디버그 방법

별도 디버그 로그는 없습니다. 동작 확인이 필요하면 `prepare()` / `finalize()` 내에
임시 `print`/`logging`을 추가하거나 `test_nccl_dispatch.py`를 실행하세요.

MNNVL 하드웨어 감지 실패 시:
```
WARNING Could not query MNNVL fabric support for device 0 (...). Assuming unavailable.
```
이 경우 자동으로 NCCL AllToAll 경로로 fallback됩니다.
