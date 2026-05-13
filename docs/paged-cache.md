# Block-paged KV cache (vLLM-aligned)

A vLLM-style block-paged KV cache lives alongside the legacy flat `Vec<KVCache>` path. Multiple in-flight requests share refcounted KV blocks for any prompt prefix they have in common (system prompt, shared few-shot preamble, repeated tool-result frames, etc.).

Routing is per-model via the `use_block_paged_cache: Option<bool>` config field. Only the full-attention layers of supported models go through the paged adapter — sliding-window, convolutional, and recurrent (GDN) layers stay on their dedicated cache types regardless of the flag.

## Foundation types

| Type                  | Location                                                    | Role                                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BlockAllocator`      | `crates/mlx-paged-attn/src/block_allocator.rs`              | Logical lifecycle — per-block refcounts, LRU eviction, prefix-hash table for cross-request reuse                                                                                                    |
| `LayerKVPool`         | `crates/mlx-paged-attn/src/layer_kv_pool.rs`                | Physical storage — per-layer Metal K and V `Buffer` pairs sized to `paged_cache_memory_mb`                                                                                                          |
| `PagedKVCacheAdapter` | `crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs` | Session-friendly wrapper. Per-request lifecycle: `reset_for_new_request` → `find_cached_prefix` → `allocate_suffix_blocks` → `record_tokens` → `register_full_blocks_for_reuse` → `release_request` |

`BlockAllocator` and `LayerKVPool` are intentionally split so the legacy `CacheEngineManager` path (used by `use_paged_attention`, a different flag — see below) is unaffected. `paged_cache_memory_mb` defaults to 2048 when `None`.

## Per-model support matrix

| Model             | Default | Status                                                                                                                                                                                                                                          |
| ----------------- | :-----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen3**         | **on**  | Greedy + prefix-reuse byte-equal vs. flat path on Qwen3-0.6B BF16. Opt out via `use_block_paged_cache: Some(false)`.                                                                                                                            |
| **LFM2.5**        | **on**  | Same parity result on LFM2.5-1.2B. Hybrid arch — only `full_attention` layers go through the adapter; conv layers stay on `Lfm2LayerCache::Conv`.                                                                                               |
| **Gemma4**        | **on**  | Same parity result on Gemma-4-E2B-IT. Sliding layers stay on `RotatingKVCache`; global layers go through the adapter; KV-shared layers consume the anchor via `SharedOnGlobal` / `SharedOnSliding`.                                             |
| **Qwen3.5 Dense** | **off** | Single-turn greedy parity verified on Qwen3.5-0.8B BF16. Default-flip pending a perf decision against the compiled C++ flat path. GDN linear-attention layers stay on flat `ArraysCache` (no cross-request reuse — vLLM `MambaManager` stance). |
| **Qwen3.5 MoE**   | **off** | Forward dispatch wired and parity-test scaffold present, but no local MoE checkpoint to verify against yet.                                                                                                                                     |

For Qwen3.5 (dense + MoE) the per-dispatch-site **compile lockout** is critical: every chat-entry site (`chat_sync_core`, `chat_tokens_delta_sync`, `chat_stream_sync_inner`, `chat_stream_tokens_delta_sync_inner`) early-returns into the paged variant **before** acquiring `DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK`, so flat-path turns and paged-path turns can interleave without corrupting compiled state. VLM checkpoints are permitted under a text-only contract — image-bearing turns fail loudly when `paged_adapter.is_some()`.

## Relationship to `use_paged_attention`

`use_block_paged_cache` is independent of `use_paged_attention`. The latter drives the legacy `PagedKVCache` + `ContinuousBatchingScheduler` path used by the production server. Both can be on or off independently.

## Parity gate

Before `use_block_paged_cache` defaults can flip from `Some(false)` to enabled, four parity tests must pass on real weights:

| Test                                                        | Gate env var                    |
| ----------------------------------------------------------- | ------------------------------- |
| `crates/mlx-core/tests/qwen3_paged_vs_flat_parity.rs`       | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/lfm2_paged_vs_flat_parity.rs`        | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/gemma4_paged_vs_flat_parity.rs`      | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/qwen3_5_paged_vs_flat_parity.rs`     | `MLX_TEST_MODEL_PATH`           |
| `crates/mlx-core/tests/qwen3_5_moe_paged_vs_flat_parity.rs` | `MLX_TEST_MODEL_PATH`           |
| `__test__/models/qwen3-paged-parity.test.ts`                | `QWEN3_PAGED_PARITY_MODEL_PATH` |

All Rust tests are `#[ignore]` and skip cleanly without the env var; the TS test uses `it.runIf`. Example invocation:

```bash
MLX_TEST_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
  cargo test -p mlx-core --test qwen3_paged_vs_flat_parity \
  -- --ignored --nocapture

QWEN3_PAGED_PARITY_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
  yarn vite run test __test__/models/qwen3-paged-parity.test.ts
```

Pass criteria: byte-equal `text` and `numTokens` between flat and paged on every prompt; byte-equal across a two-turn dialog (validates `find_cached_prefix` + `finalize_turn_keep_live` cross-turn semantics).

The TS test deliberately uses `chatSessionStart` rather than `generate` — `generate_sync` always uses fresh flat caches and never consults `paged_adapter`, so it would silently mask divergence.
