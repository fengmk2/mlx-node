# Performance & profiling

## Profiling

Per-generation profiling is exposed from `@mlx-node/lm`:

```typescript
import { enableProfiling, disableProfiling } from '@mlx-node/lm';

enableProfiling();
// run generations...
disableProfiling();
```

The store lives in `crates/mlx-core/src/profiling.rs` (global `PROFILING_STORE: Mutex<Vec<GenerationProfile>>`, gate via `PROFILING_ENABLED: AtomicBool`). NAPI exports: `setProfilingEnabled`, `isProfilingEnabled`, `getProfilingData`, `resetProfilingData`.

The per-generation profiler (`crates/mlx-core/src/decode_profiler.rs`) records:

- TTFT (`time_to_first_token_ms`)
- Phase breakdown: `forward`, `sample`, `eval_token`, `extract`, `async_eval`
- Memory snapshots before / after each generation

> Note: MLX lazy evaluation means `prefillMs` measures only graph construction (~1 ms). Use `timeToFirstTokenMs` as the real prefill latency indicator.

## Environment variables

### Profiling and tracing

| Var                        | Effect                            |
| -------------------------- | --------------------------------- |
| `MLX_PROFILE_DECODE=1`     | Auto-enables profiling at startup |
| `MLX_NODE_LOG`             | Tracing-level filter              |
| `MLX_INFERENCE_TRACE_FILE` | Path for inference trace dump     |
| `MLX_DEBUG_GEMMA4_DUMP`    | Diagnostic dumps for Gemma4       |

### Compile / decode control

| Var                                                  | Effect                                                   |
| ---------------------------------------------------- | -------------------------------------------------------- |
| `MLX_NO_COMPILE=1`                                   | Disable compiled C++ forward path (Qwen3.5)              |
| `MLX_EVAL_ALL_CACHES=1`                              | Revert to eval-all-caches (default is token-only)        |
| `MLX_QWEN35_NATIVE_KV_WRITE` / `MLX_NATIVE_KV_WRITE` | Toggle native KV-write optimization on Qwen3.5 attention |
| `MLX_WEIGHT_MATERIALIZE_CHUNK_MB`                    | Weight-loading chunk size                                |

### Paged-attention

| Var                                     | Effect                                     |
| --------------------------------------- | ------------------------------------------ |
| `MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL` | Override decode-time `clear_cache` cadence |
| `MLX_PAGED_PREFILL_EVAL_INTERVAL`       | Override prefill `eval` cadence            |
| `MLX_PAGED_PREFILL_CHUNK_SIZE`          | Prefill chunk size                         |
| `MLX_TEST_PAGED`                        | Test-only paged-path toggle                |

### Memory pool

| Var                   | Effect                                   |
| --------------------- | ---------------------------------------- |
| `MLX_CACHE_LIMIT_GB`  | Hard Metal pool ceiling                  |
| `MLX_GPU_HEADROOM_GB` | Headroom term in the auto-sizing formula |

## Key performance patterns

- `token.eval()` immediately after sampling — without it MLX builds an unbounded lazy graph.
- `synchronize_and_clear_cache()` every 256 steps — prevents memory accumulation during long generations.
- Dtype-aware scalar ops — any `f32` scalar in a binary op with bf16 promotes the **entire** result to f32.
- Token-only eval — caches materialize through the dependency graph; no need to eval every cache tensor explicitly.
- For bf16 / f16 data extraction: use `to_uint16_native()` instead of round-tripping through f32.

## GPU architecture detection

`mlx_gpu_architecture_gen()` (FFI in `crates/mlx-sys/src/lib.rs`) returns a generation number:

| Chip | Gen |
| ---- | --- |
| M1   | 13  |
| M2   | 14  |
| M3   | 15  |
| M4   | 16  |
| M5   | 17  |

The Qwen3.5 chunked GDN prefill kernel is gated on M5+ (`CHUNK_MIN_GPU_GEN = 17`) with a 64-token minimum sequence length — on M5, Neural Accelerators make `simdgroup_matrix` ops roughly 4× faster than the per-step kernel; on M1–M4 the per-step kernel wins.

## Quantization

| Scheme       | How it's invoked                                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4-bit affine | `mlx_quantized_matmul` (mode `affine`, configurable group size and bits)                                                                                 |
| MXFP8        | `mlx_gather_qmm` with `mode="mxfp8"` (used for MoE expert routing); returns `[quantized, scales]`                                                        |
| FP8 E4M3     | `mlx_dequantize` — dequant **before** expert stacking; no re-quantization after stacking                                                                 |
| FP8 KV cache | Paged-adapter only — `KVCacheDType::Fp8` with per-layer scale management via `KvScaleManager`. FP8 KV is intentionally rejected by the flat-path attach. |

### Recipes

`crates/mlx-core/src/convert.rs` supports:

- mlx-lm-style mixed-bit: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
- `qwen3_5` — Qwen3.5-tuned recipe
- `unsloth` — requires imatrix calibration

AWQ-style imatrix pre-scaling is supported for improved low-bit quality.

`quant_predicate` defaults: router gates → 8-bit; everything else → 4-bit.
