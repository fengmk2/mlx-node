# C++ FFI bridge

The bridge between MLX (C++) and the NAPI/Rust layer lives in `crates/mlx-sys/`. The Rust side declares the FFI surface in `lib.rs`; the C++ side implements each declaration across topical `.cpp` files compiled by the `cc` crate.

## File inventory

`crates/mlx-sys/src/`:

| File                     | Purpose                                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlx_array_ops.cpp`      | Array construction, arithmetic, indexing, dtype-safe scalar ops                                                                                       |
| `mlx_advanced_ops.cpp`   | quantized_matmul, gather_qmm, conv2d, FP8 dequant, PaddleOCR forward                                                                                  |
| `mlx_nn_ops.cpp`         | NN ops, data extraction, random, math                                                                                                                 |
| `mlx_fused_ops.cpp`      | Fused SwiGLU MLP and supporting ops                                                                                                                   |
| `mlx_misc_ops.cpp`       | Synchronization, compiled sampling helpers                                                                                                            |
| `mlx_stream.cpp`         | Stream/device management, memory limits                                                                                                               |
| `mlx_autograd.cpp`       | `value_and_grad` integration                                                                                                                          |
| `mlx_gated_delta.cpp`    | Metal GDN kernel opaque handles and shader indexing                                                                                                   |
| `mlx_qwen35.cpp`         | Compiled Qwen3.5 dense forward (uses `mlx::core::compile`)                                                                                            |
| `mlx_qwen35_moe.cpp`     | Compiled Qwen3.5 MoE forward with expert routing (uses `mlx::core::compile`)                                                                          |
| `mlx_qwen35_vlm.cpp`     | Qwen3.5 VLM prefill — runs the full LM forward over text+vision embeddings and stores caches; the compiled decode path then resumes from those caches |
| `mlx_qwen35_common.h`    | Shared compiled-forward helpers — linear_proj, attn, GDN, RoPE                                                                                        |
| `mlx_common.h`           | FFI macros, error handling, array conversion                                                                                                          |
| `mlx_common_weights.cpp` | Common weight storage for compiled forward passes                                                                                                     |
| `mlx_paged_dispatch.cpp` | C++ paged-attention kernel dispatch                                                                                                                   |
| `mlx_paged_ops.cpp`      | `PagedKVWrite` / `PagedAttention` custom MLX ops (largest file in the bridge)                                                                         |
| `mlx_paged_profile.cpp`  | Profile-run helpers for auto-sizing the block pool                                                                                                    |

`crates/mlx-sys/src/lib.rs` is the FFI declaration root (~300 `pub fn` wrappers around `unsafe extern "C-unwind"` blocks).

## Compiled forward paths

Qwen3.5 dense + MoE decode use `mlx::core::compile` to cache the forward graph: trace once, reuse via `compile_replace`. Key design points:

- Pre-allocated KV caches passed in as compile inputs
- `fast::rope` invoked with an array-valued offset
- `slice_update` invoked with an array start index
- Path only enabled when `mlx_qwen35_weight_count() > 0`

```
mlx_qwen35.cpp        dense compiled decode (mlx::core::compile)
mlx_qwen35_moe.cpp    MoE compiled decode + expert routing (mlx::core::compile)
mlx_qwen35_vlm.cpp    VLM prefill — stores caches that the compiled decode path resumes from
mlx_qwen35_common.h   shared helpers (linear_proj, attn, GDN, RoPE)
```

### Pitfalls

- `mlx::core::array` has **no default constructor** — initialize via `mlx_array_from_scalar(...)` or other helpers.
- `int32` is not in scope inside inner namespaces — use `mlx::core::int32`.
- Adding a **new** `.cpp` file requires `rm -rf target/release/build/mlx-sys-*` once; the `cc` crate caches its source-file list across builds and won't pick up new files otherwise.

### Env vars

| Var                     | Effect                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| `MLX_NO_COMPILE=1`      | Disables the compiled forward path; falls back to per-step Rust forward |
| `MLX_EVAL_ALL_CACHES=1` | Reverts to eval-all-caches strategy (vs. the default token-only eval)   |

## Process-wide globals

Compiled paths use process-wide globals in `crates/mlx-core/src/models/qwen3_5/model.rs`:

- `DENSE_COMPILED_MUTEX: std::sync::Mutex<()>` — serializes dense compiled-path access
- `COMPILED_WEIGHTS_RWLOCK: std::sync::RwLock<()>` — read locks during compiled forward, write locks during weight load

The paged-cache code path bypasses both locks entirely (see [paged-cache.md](paged-cache.md) for the compile-lockout contract).

## Metal shaders

`crates/mlx-paged-attn/metal/`:

| File                              | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `attention/paged_attention.metal` | Paged-attention attention kernel      |
| `cache/reshape_and_cache.metal`   | KV cache reshape operations           |
| `cache/copy_blocks.metal`         | Block copy for paged cache management |
| `float8.metal`                    | FP8 type conversions and helpers      |
| `utils.metal`                     | Common Metal utilities                |

`crates/mlx-sys/build.rs` compiles `.metal` sources into `paged_attn.metallib` and copies both `paged_attn.metallib` and `mlx.metallib` into `target/<profile>/` and `target/<profile>/deps/` so integration tests discover them.
