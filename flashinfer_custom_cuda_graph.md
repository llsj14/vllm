# CUDA Graph 호환 MoE AllToAll 구현 기록

## 배경 및 목표

`--enforce-eager` 옵션 없이 vLLM 서버를 실행할 때 CUDA Graph 캡처 중 런타임 에러 발생 및
모델 출력 비정합 문제를 해결하고, H100 (비-MNNVL) 환경에서 MoE Expert Parallelism의
AllToAll 통신을 올바르게 구현한다.

**핵심 요구사항**: AllGather/ReduceScatter 방식이 아닌 **진짜 AllToAll**을 유지하면서
CUDA Graph도 정상 동작해야 함.

**실행 환경**
- GPU: NVIDIA H100 80GB HBM3 × 8
- NCCL: 2.27.5
- vLLM 설정: TP=1, DP=8, EP=8, `--all2all-backend flashinfer_all2allv`
- 모델: NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (H=7168, 32 MoE layers)

```bash
vllm serve /mnt/lvm/checkpoints/huggingface/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.3 \
  --all2all-backend flashinfer_all2allv
  # (--enforce-eager 없이 정상 동작)
```

---

## 최종 해결 전략 (요약)

핵심 아이디어: **`moe_forward` / `moe_forward_shared` 자체를 splitting op으로 등록**.

이 두 custom op은 이미 torch.compile에 opaque합니다. splitting op으로 만들면
MoE 전체(routing → AllToAll dispatch → expert kernel → reverse AllToAll → accumulation)가
CUDA graph subgraph **바깥**에서 eager Python으로 실행됩니다.

따라서:
- variable-size AllToAll (`nonzero`, `argsort`, `split_sizes`) 사용 가능 — 모든 T에서 정확
- static buffer 불필요 — 메모리 효율적
- zero-padding expert kernel 불필요 — 실제 dispatch된 토큰만 처리

추가로, **PIECEWISE CUDA graph의 입력 주소 불안정 문제**를 `CUDAGraphWrapper`에서
자동 입력 복사로 해결하여 정합성을 보장합니다.

---

## 핵심 제약 사항

### NCCL 2.27.5에서 `ncclAllToAll` 미존재

```
Available NCCL functions: ncclAllGather, ncclAllReduce, ncclBroadcast,
ncclGroupStart, ncclGroupEnd, ncclRecv, ncclReduce, ncclReduceScatter, ncclSend, ...
```

`ncclAllToAll`은 NCCL public API에 존재하지 않음 (`objdump` 확인). 따라서
`torch.distributed.all_to_all_single`은 내부적으로 다음 fallback 패턴을 사용:

```
ncclGroupStart
  ncclSend(rank 0 → all), ncclRecv(rank 0 ← all)
  ncclSend(rank 1 → all), ncclRecv(rank 1 ← all)
  ...
ncclGroupEnd
```

= **ep_size×2 P2P ops per AllToAll call** (ep_size=8이면 16 ops + 2 group ops = 18 ops/call)

**핵심 문제**: 이 P2P group 패턴은 CUDA Graph 안에 캡처되면 H100/NCCL-2.27.5에서 불안정.

---

## 문제 히스토리 및 해결 과정

### 문제 1: CUDA Graph 안에서 AllToAll 캡처 시 크래시

**증상**
```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```
`--enforce-eager` 제거 후 서버 시작 시 CUDA Graph 캡처 중 에러 발생.

**원인**
`prepare()`에서 variable split sizes를 가진 `dist.all_to_all_single`을 직접 호출.
`nonzero()` 등 동적 연산은 CUDA graph stream 캡처 중 호출 금지.

---

### 문제 2: FULL CUDA Graph 모드에서 빈 응답

**증상**
- `completion_tokens: 32`인데 `"content": ""` (빈 문자열)
- 비결정적: 같은 요청에서 가끔 정상, 가끔 빈 응답

**원인**
`FULL_AND_PIECEWISE` 모드에서 Decode 배치는 `CUDAGraphWrapper(FULL)` 사용:
```
1개 big CUDA graph에 전체 model forward 캡처
= 32 layers × 다수 AllToAll × 18 NCCL ops
= ~2048 NCCL P2P ops in a single CUDA graph
```
NCCL 2.27.5에서 수천 개의 P2P group ops가 하나의 CUDA graph 안에 있으면 **불안정**.

**해결 (hard guard)**: `gpu_model_runner.py`의 `load_model()` 안에
`NCCLAllToAllMoEPrepareAndFinalize._active` 플래그를 확인하여
`CUDAGraphMode.PIECEWISE`로 강제 설정 → `CUDAGraphWrapper(FULL)` 생성 완전 차단.

---

### 문제 3: `@torch._dynamo.disable`로는 CUDA graph 외부 실행 불가

**근본 원인 분석**

`@torch._dynamo.disable`을 `prepare()`에 적용하면:
- dynamo가 `prepare()`를 FX graph 안의 **opaque `call_function` 노드**로 처리
- 노드의 target = Python method object (NOT `torch._ops.OpOverload`)
- vLLM의 `should_split()` 함수는 `OpOverload`만 인식

→ opaque `prepare()` call은 **splitting op이 아님**
→ `CUDAGraphWrapper(PIECEWISE)` subgraph 안에 포함
→ `torch.cuda.graph()` context 안에서 실행 → AllToAll 캡처 → 불안정

**핵심 교훈**: `@torch._dynamo.disable`은 dynamo tracing의 graph break만 만들 뿐,
vLLM의 piecewise splitting과는 완전히 별개. CUDA graph 외부 실행을 보장하지 않음.

---

### 문제 4: Static buffer 접근법의 정합성 실패

**시도**: `ep_alltoall_dispatch` / `ep_alltoall_combine` custom op을 splitting op으로 등록하고,
`[ep_size, T_max, H]` 크기의 static buffer에 equal-split AllToAll 사용.

**문제점**:
- T=1 decode에서도 `ep_size × T_max = 4096`개 토큰을 expert kernel에 투입
- 대부분이 zero-padded → FlashInfer CUTLASS 커널에서 정합성 문제 발생
- 메모리 낭비: 32 layers × 9 buffers × ep_size(8) × T_max(512) × H(7168) × 2B ≈ 수 GB

---

### 문제 5 (최종 해결): `moe_forward`를 splitting op으로 등록 + CUDAGraphWrapper 입력 복사

**핵심 통찰**: `moe_forward` / `moe_forward_shared`는 이미 `torch.library` custom op으로
등록되어 있고 (`direct_register_custom_op`), dynamo에 opaque합니다.
이것 자체를 splitting op으로 만들면 **MoE 전체**가 CUDA graph 바깥에서 실행됩니다.

```
MoE forward (moe_forward splitting op) 안에서 일어나는 일:
  1. Routing (topk_ids → expert_ranks → token_needs_rank)
  2. Sparse packing (nonzero, argsort → token_idx_sorted, send_sizes)
  3. Size exchange (all_to_all_single for counts)
  4. AllToAll dispatch (hidden, topk_ids, weights)
  5. Non-local weight masking
  6. Expert GEMM kernel
  7. Reverse AllToAll (combine)
  8. index_add_ accumulation
```

이 모든 것이 eager Python으로 실행 → variable-size AllToAll 사용 가능 → 정합성 보장.

**그러나 새로운 문제 발생: PIECEWISE CUDA graph 입력 주소 불안정**

`moe_forward`는 매번 **새로운 텐서를 동적 생성**해서 리턴합니다. 반면 PIECEWISE CUDA graph의
각 subgraph는 캡처 시점의 입력 텐서 **메모리 주소에 고정**됩니다:

```
CUDA graph 캡처 (1회):
  subgraph_before → moe_forward (주소 A의 텐서 리턴) → subgraph_after (주소 A에서 읽기 캡처)

CUDA graph replay (매 추론):
  subgraph_before → moe_forward (주소 B의 새 텐서 리턴) → subgraph_after (여전히 주소 A에서 읽음!)
  ↑ 주소 A에는 stale 데이터 → 정합성 깨짐
```

참고: `attention`은 이 문제가 없습니다. `unified_attention_with_output`은 pre-allocated
`output` 버퍼에 쓰고 그걸 리턴하므로 주소가 항상 안정적입니다.

**해결**: `CUDAGraphWrapper`에서 replay 전에 입력 주소가 바뀌었으면 데이터를 캡처
시점의 버퍼로 자동 복사:

```python
# vllm/compilation/cuda_graph.py — CUDAGraphWrapper.__call__()

# 캡처 시: 입력 텐서 참조 저장
entry.input_tensors = [x for x in args if isinstance(x, torch.Tensor)]

# Replay 시: 주소 비교 후 불일치 시 캡처 버퍼로 복사
if entry.input_tensors is not None:
    for captured_t, runtime_arg in zip(
        entry.input_tensors,
        (x for x in args if isinstance(x, torch.Tensor)),
    ):
        if runtime_arg.data_ptr() != captured_t.data_ptr():
            captured_t.copy_(runtime_arg)

entry.cudagraph.replay()
```

- 주소가 같으면 (기존 attention 등) → 오버헤드 없음 (포인터 비교만)
- 주소가 다르면 (moe_forward 등) → `copy_()` 한번으로 정확한 데이터 전달

---

## 최종 구현 상세

### `NCCLAllToAllMoEPrepareAndFinalize` 전체 구조

static buffer 없음. 원래 working code의 variable-size AllToAll을 그대로 사용.

#### `prepare()` 흐름

```python
def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map, ...):
    # [1] Routing: 각 토큰이 어떤 rank로 보내져야 하는지 결정
    expert_to_rank = self._get_expert_to_rank(expert_map, ...)  # 캐시됨
    expert_ranks = expert_to_rank[topk_ids.long()]              # [T, K]
    token_needs_rank = torch.zeros(T, ep_size, dtype=bool)
    token_needs_rank.scatter_(1, expert_ranks, True)

    # [2] Sparse packing: nonzero + argsort → variable split sizes
    token_idxs, dest_ranks = token_needs_rank.nonzero(as_tuple=True)
    sort_order = dest_ranks.argsort(stable=True)
    token_idx_sorted = token_idxs[sort_order]
    send_sizes = torch.bincount(dest_ranks_sorted, minlength=ep_size).tolist()

    # [3] Size exchange: 각 rank가 얼마나 받을지 교환
    dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)

    # [4] AllToAll dispatch: hidden, topk_ids, weights 교환
    dist.all_to_all_single(recv_hidden, send_hidden,
        output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=pg)
    # (topk_ids, weights도 동일하게)

    # [5] Non-local weight masking: 다른 rank의 expert weight를 0으로
    is_local = expert_to_rank[recv_topk_ids.long()] == ep_rank
    recv_topk_weights *= is_local.to(recv_topk_weights.dtype)

    # [6] State 저장 (finalize에서 역방향 AllToAll에 사용)
    self._state = _NCCLDispatchState(
        local_token_count=T, send_sizes=send_sizes,
        recv_sizes=recv_sizes, token_indices=token_idx_sorted)

    return recv_hidden_q, recv_hidden_scale, None, recv_topk_ids, recv_topk_weights
```

#### `finalize()` 흐름

```python
def finalize(self, output, fused_expert_output, ...):
    state = self._state   # prepare()에서 저장한 상태

    # [1] Reverse AllToAll: expert 처리 결과를 원래 rank로 돌려보냄
    combined = torch.empty(total_send, H, ...)
    dist.all_to_all_single(combined, fused_expert_output,
        output_split_sizes=state.send_sizes,
        input_split_sizes=state.recv_sizes, group=pg)

    # [2] Accumulation: 원래 토큰 위치에 결과 누적
    output.zero_()
    output.index_add_(0, state.token_indices, combined)
```

---

## 현재 동작 방식 (최종 상태)

### 실행 경로 요약

```
[모든 T 값 — decode, prefill 모두 동일한 경로]

  ── CUDA graph subgraph A (compiled) ──
  RMSNorm, attention, residual add, ...
  RMSNorm, shared_experts(MLP)
  ── subgraph A 끝 ──

  ── moe_forward (splitting op, eager Python) ──
  routing → nonzero/argsort → AllToAll dispatch →
  expert GEMM → reverse AllToAll → index_add_
  ── moe_forward 끝 ──

  ── CUDA graph subgraph B (compiled) ──
  residual add, next layer...
  ── subgraph B 끝 ──

  (CUDAGraphWrapper가 subgraph B의 입력 주소가
   바뀌면 자동으로 copy_ 실행 → 정합성 보장)
```

### 모드 결정 흐름

```
Step 1. create_flashinfer_prepare_finalize()  [모델 레이어 생성 시]
   → NCCLAllToAllMoEPrepareAndFinalize() 생성
      → _active = True
      → _register_alltoall_splitting_ops()
         → splitting_ops에 "vllm::moe_forward", "vllm::moe_forward_shared" 추가

Step 2. gpu_model_runner.load_model()  [hard guard]
   → NCCLAllToAllMoEPrepareAndFinalize._active = True 확인
   → cudagraph_mode가 FULL 포함 시 CUDAGraphMode.PIECEWISE 강제 설정
   → CUDAGraphWrapper(FULL) 생성 안 함  ✓

Step 3. torch.compile tracing (첫 forward 시 lazy 실행)
   → FX graph에 moe_forward / moe_forward_shared OpOverload 노드 포함
   → split_graph(): moe_forward에서 split → piecewise 경계 생성
   → 각 non-splitting subgraph를 Inductor로 컴파일 + CUDAGraphWrapper(PIECEWISE)로 래핑

Step 4. CUDA graph capture (T=1,2,4,...,512 — 51개 크기)
   → 각 subgraph별로 CUDA graph 캡처 (moe_forward는 eager 실행)
   → CUDAGraphWrapper에 input_tensors 저장 (replay 시 주소 비교/복사용)

Step 5. Inference (replay)
   → subgraph A replay → moe_forward eager → subgraph B에 copy_ 후 replay
```

---

## 변경된 파일

### 1. `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`

`NCCLAllToAllMoEPrepareAndFinalize` 클래스:

| 항목 | 내용 |
|---|---|
| `_active: bool = False` | 클래스 레벨 플래그. `gpu_model_runner` hard guard에서 참조 |
| `_register_alltoall_splitting_ops()` | `__init__`에서 `vllm::moe_forward`, `vllm::moe_forward_shared`를 `splitting_ops`에 추가 |
| `_ep_pg()` | EP process group 정보 반환 |
| `_get_expert_to_rank()` | expert→rank 매핑 (all_gather, 캐시됨) |
| `prepare()` | variable-size AllToAll dispatch (nonzero, argsort, split_sizes) |
| `finalize()` | reverse AllToAll + `index_add_` accumulation |

**사용하지 않는 것들** (이전 static buffer 접근법에서 제거/미사용):
- `_graph_bufs` — static buffer 세트 (코드에서 완전 제거됨)
- `_ensure_graph_bufs_helper()` — buffer 할당 (코드에서 완전 제거됨)
- `_prepare_large_t_eager()` / `_finalize_large_t_eager()` — eager fallback (코드에서 완전 제거됨)
- `ep_alltoall_dispatch` / `ep_alltoall_combine` — `parallel_state.py`에 정의는 남아있으나
  `NCCLAllToAllMoEPrepareAndFinalize`에서는 사용하지 않음 (직접 `dist.all_to_all_single` 호출)

### 2. `vllm/compilation/cuda_graph.py`

`CUDAGraphWrapper` 수정 — PIECEWISE CUDA graph 입력 주소 불안정 해결:

| 항목 | 내용 |
|---|---|
| `CUDAGraphEntry.input_tensors` | 캡처 시 입력 텐서 참조 저장 (신규 필드) |
| 캡처 시 | `entry.input_tensors = [x for x in args if isinstance(x, torch.Tensor)]` |
| Replay 시 | 주소 비교 후 불일치 시 `captured_t.copy_(runtime_arg)` 실행 |

이 수정은 모든 splitting op에 일반적으로 적용되며, 기존 attention subgraph에는
오버헤드가 없음 (주소가 이미 안정적이므로 포인터 비교만 수행).

### 3. `vllm/v1/worker/gpu_model_runner.py`

`load_model()` 안, CUDAGraphWrapper 생성 직전에 hard guard:

```python
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    NCCLAllToAllMoEPrepareAndFinalize,
)
if NCCLAllToAllMoEPrepareAndFinalize._active:
    mode = self.compilation_config.cudagraph_mode
    if mode is not None and mode.has_full_cudagraphs():
        logger.info(
            "NCCLAllToAllMoEPrepareAndFinalize [hard guard]: "
            "blocking CUDAGraphWrapper(FULL) — downgrading "
            "cudagraph_mode %s → PIECEWISE.", mode,
        )
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

---

## 서버 기동 시 예상 로그

```
# splitting_ops 등록 확인:
INFO [flashinfer_cutlass_prepare_finalize.py]
     NCCLAllToAllMoEPrepareAndFinalize: added 'vllm::moe_forward' to
     splitting_ops so AllToAll runs outside CUDA graphs.

# hard guard 작동:
INFO [gpu_model_runner.py] NCCLAllToAllMoEPrepareAndFinalize [hard guard]:
     blocking CUDAGraphWrapper(FULL) — downgrading cudagraph_mode
     FULL_AND_PIECEWISE → PIECEWISE.

# PIECEWISE 모드로 CUDA graph 캡처:
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█| 51/51 [00:12]
```

### 재시작 전 필수 사항

```bash
# 1. torch compile 캐시 완전 삭제
rm -rf ~/.cache/vllm/torch_compile_cache/

# 2. 서버 재시작
```

---

## 핵심 설계 원칙 (결론)

| 방법 | 결과 | 이유 |
|------|------|------|
| `@torch._dynamo.disable` on `prepare()` | ❌ 실패 | opaque call_function은 splitting_op이 아님 → CUDAGraphWrapper 안에 포함 |
| Static buffer + custom AllToAll op splitting | ❌ 정합성 실패 | zero-padded 4096 토큰 expert kernel → FlashInfer CUTLASS에서 정합성 문제 |
| `moe_forward` 자체를 splitting op으로 등록 | ✅ 성공 | MoE 전체가 CUDA graph 바깥 eager 실행 → variable-size AllToAll 정확 |
| CUDAGraphWrapper 입력 주소 자동 복사 | ✅ 필수 | splitting op이 새 텐서를 리턴 → 다음 subgraph의 CUDA graph 입력 주소 불일치 해결 |
| AllGather fallback | ❌ 거부 | AllToAll 요구사항 불충족 |
| FULL CUDA graph | ❌ 불안정 | 수천 NCCL P2P ops in one graph → H100/NCCL 2.27.5에서 불안정 |

**두 가지 핵심 수정의 조합으로 해결**:
1. `moe_forward`를 splitting op으로 → AllToAll이 CUDA graph 밖에서 실행
2. `CUDAGraphWrapper`에서 입력 주소 변경 시 자동 복사 → CUDA graph replay 정합성 보장
