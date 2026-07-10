# DSV4-Flash Indexer and Hybrid KV Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the missing DSV4 ratio=4 top-512 indexer and store ratio=4 compressed attention KV in NUMA-local CPU memory while gathering only selected rows back to the layer GPU.

**Architecture:** Keep the 128-token window, ratio=128 compressed KV, and ratio=4 index cache on GPU. Store ratio=4 attention compressed KV as BF16 rows in CPU memory; after GPU index scoring returns top-512 row ids, gather those rows into a bounded pinned double buffer and copy only the gathered 512KiB per ratio=4 layer to GPU.

**Tech Stack:** C++17, CUDA SM75, BF16/FP32 fastllm `Data`, AVX2-capable dual-socket NUMA host, existing DSV4 compressor and sparse-attention kernels, CMake `UNIT_TEST`.

## Global Constraints

- Target model parameters and compression ratios must come from the loaded DSV4 config; tests also cover reduced synthetic dimensions.
- Official main-model context limit is 1,048,576 tokens; do not claim support beyond it.
- First correctness milestone is 256K, followed by the official 1M limit.
- Do not pin the full cold cache; pin only bounded staging buffers.
- Ratio=4 attention compressed KV is the only cache offloaded in this plan.
- `windowKV`, ratio=128 `compressedKV`, ratio=4 index cache, and compressor tail remain on the layer compute GPU.
- The CPU cold store must be NUMA-local to the GPU that owns the corresponding attention layer.
- Stable top-k tie-break is `(score descending, block index ascending)`.
- CPU fallback remains available for tiny reference tests but is not an accepted long-context runtime path.
- All memory-accounting output must distinguish logical bytes, allocated bytes, CPU bytes and GPU bytes.
- Commit messages follow the repository `git-commit-zh-split` skill and use Chinese.

---

## File Structure

| File | Responsibility |
|---|---|
| `include/models/deepseekv4.h` | Extend per-request/per-layer cache state with index cache and cold-store metadata. |
| `include/models/deepseekv4_kv_offload.h` | Define memory estimator, cold-store and gather interfaces. |
| `src/models/deepseekv4_kv_offload.cpp` | Implement CPU/NUMA cold rows, sorted gather and bounded staging lifetime. |
| `src/models/deepseekv4.cpp` | Execute indexer, append caches and route gathered rows to sparse attention. |
| `include/devices/cuda/fastllm-cuda.cuh` | Declare DSV4 index-score/top-k CUDA entry points. |
| `src/devices/cuda/models/deepseekv4-kernels.cu` | Implement index score, stable top-k and any GPU-side cache conversion. |
| `test/ops/regressionOps.cpp` | Small deterministic indexer and cold-gather correctness regressions. |
| `test/benchmark/dsv4_kv_report.py` | Parse memory/timing JSONL and verify expected cache slopes. |
| `tools/fastllm_pytools/util.py` | Expose explicit DSV4 KV residency configuration. |
| `docs/deepseek.md` | Document DSV4 cache modes, capacity formula and tested commands. |
| `CMakeLists.txt` | Register the new CPU implementation in all relevant build variants. |

### Task 1: Make DSV4 cache capacity calculable and testable

**Files:**
- Create: `include/models/deepseekv4_kv_offload.h`
- Create: `src/models/deepseekv4_kv_offload.cpp`
- Modify: `CMakeLists.txt`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: context tokens, batch size, head dimension, index dimension, window size and compression ratios.
- Produces: `DeepSeekV4CacheBytes EstimateDeepSeekV4CacheBytes(const DeepSeekV4CacheShape &shape)`.

- [ ] **Step 1: Add a failing capacity regression**

```cpp
#include "models/deepseekv4_kv_offload.h"

void RunDeepSeekV4CacheEstimateRegression() {
    fastllm::DeepSeekV4CacheShape shape;
    shape.tokens = 1000000;
    shape.batch = 1;
    shape.headDim = 512;
    shape.indexDim = 128;
    shape.windowSize = 128;
    shape.compressRatios.assign({0, 0});
    for (int i = 0; i < 20; i++) {
        shape.compressRatios.push_back(4);
        shape.compressRatios.push_back(128);
    }
    shape.compressRatios.push_back(4);

    auto bytes = fastllm::EstimateDeepSeekV4CacheBytes(shape);
    Expect(bytes.ratio4ColdBytes == 5376000000ULL, "ratio4 cold-byte estimate mismatch");
    Expect(bytes.ratio128GpuBytes == 159989760ULL, "ratio128 byte estimate mismatch");
    Expect(bytes.indexGpuBytes == 1344000000ULL, "index byte estimate mismatch");
    Expect(bytes.totalGrowingBytes == 6879989760ULL, "total cache-byte estimate mismatch");
}
```

- [ ] **Step 2: Build to verify failure**

```bash
cmake -S . -B build-kv-test -DUNIT_TEST=ON -DUSE_CUDA=ON -DCUDA_ARCH=75
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
```

Expected: compile failure because `deepseekv4_kv_offload.h` does not exist.

- [ ] **Step 3: Define exact capacity types**

```cpp
#pragma once

#include "fastllm.h"

#include <cstdint>
#include <vector>

namespace fastllm {
    struct DeepSeekV4CacheShape {
        uint64_t tokens = 0;
        uint64_t batch = 1;
        uint64_t headDim = 512;
        uint64_t indexDim = 128;
        uint64_t windowSize = 128;
        std::vector<int> compressRatios;
    };

    struct DeepSeekV4CacheBytes {
        uint64_t windowGpuBytes = 0;
        uint64_t ratio4ColdBytes = 0;
        uint64_t ratio128GpuBytes = 0;
        uint64_t indexGpuBytes = 0;
        uint64_t totalGrowingBytes = 0;
    };

    DeepSeekV4CacheBytes EstimateDeepSeekV4CacheBytes(
        const DeepSeekV4CacheShape &shape);
}
```

- [ ] **Step 4: Implement overflow-safe calculation**

Use integer floor division for compressed block counts, BF16 two-byte compressed/index rows and FLOAT32 four-byte `windowKV`. Before every multiplication, reject overflow using `UINT64_MAX / operand`.

The calculation must be:

```cpp
blocks = tokens / ratio;
compressed = batch * blocks * headDim * 2;
index = ratio == 4 ? batch * blocks * indexDim * 2 : 0;
window = batch * min(tokens, windowSize) * headDim * 4;
```

- [ ] **Step 5: Register and run tests**

Add `src/models/deepseekv4_kv_offload.cpp` to `FASTLLM_CXX_SOURCES`, then run:

```bash
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
./build-kv-test/regressionOps
```

Expected output includes `dsv4 cache estimate regression: PASS`.

- [ ] **Step 6: Commit capacity accounting**

```bash
git add CMakeLists.txt include/models/deepseekv4_kv_offload.h src/models/deepseekv4_kv_offload.cpp test/ops/regressionOps.cpp
git commit -m "补充DSV4缓存容量模型"
```

### Task 2: Implement a small CPU reference for the official indexer

**Files:**
- Modify: `include/models/deepseekv4_kv_offload.h`
- Modify: `src/models/deepseekv4_kv_offload.cpp`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: query `[batch, heads, indexDim]`, shared index rows `[batch, blocks, indexDim]`, per-head weights `[batch, heads]`, and top-k.
- Produces: stable INT32 indices `[batch, min(topk, blocks)]`.

```cpp
void DeepSeekV4IndexerReference(
    const std::vector<float> &query,
    const std::vector<float> &indexRows,
    const std::vector<float> &headWeights,
    int batch, int heads, int blocks, int indexDim, int topk,
    std::vector<int32_t> &indices,
    std::vector<float> *scores = nullptr);
```

- [ ] **Step 1: Add a failing deterministic top-k test**

Use `batch=1`, `heads=2`, `blocks=5`, `indexDim=4`, `topk=3`. Choose non-tied values and assert exact indices `{3, 1, 4}`. Add a second case with equal scores and assert lower block id wins.

- [ ] **Step 2: Run and verify missing implementation**

```bash
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
```

Expected: link failure for `DeepSeekV4IndexerReference`.

- [ ] **Step 3: Implement official scoring semantics**

For every batch and block:

```cpp
float score = 0.0f;
for (int h = 0; h < heads; h++) {
    float dot = Dot(query[b, h, :], indexRows[b, block, :]);
    score += std::max(dot, 0.0f) * headWeights[b, h];
}
```

Sort candidates by:

```cpp
if (lhs.score != rhs.score) return lhs.score > rhs.score;
return lhs.index < rhs.index;
```

- [ ] **Step 4: Run the regression**

```bash
./build-kv-test/regressionOps
```

Expected output includes `dsv4 indexer reference regression: PASS`.

- [ ] **Step 5: Commit the reference**

```bash
git add include/models/deepseekv4_kv_offload.h src/models/deepseekv4_kv_offload.cpp test/ops/regressionOps.cpp
git commit -m "补充DSV4索引器参考实现"
```

### Task 3: Add CUDA index scoring and stable top-k

**Files:**
- Modify: `include/devices/cuda/fastllm-cuda.cuh`
- Modify: `src/devices/cuda/models/deepseekv4-kernels.cu`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: CUDA query, index rows, head weights and top-k.
- Produces: INT32 top-k block ids and optional FLOAT32 selected scores on the same CUDA device.

```cpp
bool FastllmCudaDeepSeekV4IndexerTopK(
    const Data &query,
    const Data &indexRows,
    const Data &headWeights,
    int topk,
    Data &topkIndices,
    Data *topkScores);
```

- [ ] **Step 1: Add CUDA-vs-reference regressions**

Generate deterministic non-tied tensors for:

```text
(batch, heads, blocks, dim, topk)
=(1, 4, 17, 16, 5)
(2, 8, 257, 32, 32)
(1, 64, 1024, 128, 512)
```

Require exact index equality and selected-score `max_abs <= 1e-3`.

- [ ] **Step 2: Run to verify missing CUDA API**

```bash
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
```

Expected: compile or link failure for `FastllmCudaDeepSeekV4IndexerTopK`.

- [ ] **Step 3: Implement score generation**

Use one logical output per `(batch, block)`. Accumulate over heads and index dimension in FP32. Support BF16 index rows and BF16/FP16/FP32 query. Reject mismatched dimensions and devices before launch.

- [ ] **Step 4: Implement deterministic top-k**

For the first correct version, use segmented radix sort or an existing repository top-k primitive if it supports stable tie handling. If no suitable primitive exists, implement blockwise candidate reduction followed by a final sort over at most `ceil(blocks / tile) * topk` candidates.

The final comparator must encode score descending and block id ascending. Do not use atomic insertion whose result depends on scheduling.

- [ ] **Step 5: Run correctness and sanitizer**

```bash
./build-kv-test/regressionOps
compute-sanitizer --tool memcheck ./build-kv-test/regressionOps
```

Expected: all indexer CUDA cases pass and memcheck reports zero errors.

- [ ] **Step 6: Record a 250K-block benchmark**

Add a benchmark mode that runs `(heads=64, blocks=250000, dim=128, topk=512)`. Report score kernel time, top-k time and total bytes read. This models a ratio=4 layer at 1M context.

- [ ] **Step 7: Commit CUDA indexer**

```bash
git add include/devices/cuda/fastllm-cuda.cuh src/devices/cuda/models/deepseekv4-kernels.cu test/ops/regressionOps.cpp
git commit -m "实现DSV4索引器CUDA内核"
```

### Task 4: Build and use the ratio=4 index cache in DSV4 forward

**Files:**
- Modify: `include/models/deepseekv4.h`
- Modify: `src/models/deepseekv4.cpp`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Extends `DeepSeekV4DecodeLayerCache` with `indexCompressedKV`, `indexCompressedBlocks`, and bounded index-compressor raw state.
- Produces top-512 ids for ratio=4 layers; ratio=128 and ratio=0 layers return sequential/empty ids as before.

- [ ] **Step 1: Add cache-lifecycle tests**

For a synthetic ratio=4 layer, append token chunks of lengths `3`, `1`, `7`, and `5`. After each append assert:

```text
indexCompressedBlocks == floor(totalLen / 4)
attentionCompressedBlocks == floor(totalLen / 4)
```

Copy the request state, restore it, append one token, and assert both original and restored states produce the same top-k ids.

- [ ] **Step 2: Run to expose missing cache fields**

```bash
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
```

Expected: compile failure for missing index-cache fields.

- [ ] **Step 3: Add index-cache state**

Add these fields to both decode and history layer cache structs:

```cpp
Data indexCompressedKV;
int indexCompressedBlocks = 0;
Data indexCompressorKVRaw;
Data indexCompressorScoreRaw;
int indexCompressorRawTokenBase = 0;
```

Update copy constructors, assignment, history snapshot and restore logic. Keep index cache on the layer GPU when CUDA is preferred.

- [ ] **Step 4: Reuse compressor mechanics with official indexer dimensions**

For ratio=4 layers only:

1. compute `indexer.compressor.wkv` and `wgate` outputs with width `2 * index_head_dim`;
2. apply the same overlapping ratio=4 pooling rule;
3. normalize with `indexer.compressor.norm.weight`;
4. apply indexer RoPE to the final `qk_rope_head_dim=64` dimensions;
5. apply the reference Hadamard rotation/FP4 simulation expected by the checkpoint;
6. append BF16 rows to `indexCompressedKV`.

- [ ] **Step 5: Compute official index weights and query**

Use existing `qr` as the input to `indexer.wq_b`, reshape to `[batch, seq, index_n_heads, index_head_dim]`, apply indexer RoPE and rotation. Compute:

```text
weights = indexer.weights_proj(attnInput)
weights *= index_head_dim^-0.5 * index_n_heads^-0.5
```

For decode `seq=1`, call `FastllmCudaDeepSeekV4IndexerTopK` and produce at most 512 block ids.

- [ ] **Step 6: Gate the new path explicitly during bring-up**

Use `FASTLLM_DSV4_INDEXER=1` for initial A/B testing. Once reference parity and long-context tests pass, make indexer enabled by default for ratio=4 layers and retain `FASTLLM_DSV4_INDEXER=0` only as a debug fallback.

- [ ] **Step 7: Run cache lifecycle and CUDA parity tests**

```bash
FASTLLM_DSV4_INDEXER=1 ./build-kv-test/regressionOps
```

Expected output includes `dsv4 index-cache lifecycle regression: PASS` and all previous tests pass.

- [ ] **Step 8: Commit model integration**

```bash
git add include/models/deepseekv4.h src/models/deepseekv4.cpp test/ops/regressionOps.cpp
git commit -m "接入DSV4压缩注意力索引器"
```

### Task 5: Implement a bounded NUMA-local cold KV store

**Files:**
- Modify: `include/models/deepseekv4_kv_offload.h`
- Modify: `src/models/deepseekv4_kv_offload.cpp`
- Modify: `include/models/deepseekv4.h`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: BF16 compressed rows and INT32 selected row ids.
- Produces: a contiguous BF16 gathered tensor preserving input id order.

```cpp
class DeepSeekV4ColdKVStore {
public:
    DeepSeekV4ColdKVStore(int rowDim, int numaNode, uint64_t reserveRows);
    void Append(const Data &bf16Rows);
    void Gather(const std::vector<int32_t> &rowIds, Data &gatheredPinned);
    uint64_t Rows() const;
    uint64_t LogicalBytes() const;
    int NumaNode() const;
};
```

- [ ] **Step 1: Add CPU gather tests**

Create 1024 BF16 rows of dimension 16 where value `(row, col)` encodes `row * 32 + col`. Gather ids `{999, 0, 17, 17, 511}` and assert all five rows, including the duplicate, preserve order exactly.

Also assert out-of-range ids throw an error naming the invalid id and current row count.

- [ ] **Step 2: Run to verify missing cold store**

```bash
cmake --build build-kv-test -j "$(nproc)" --target regressionOps
```

Expected: compile failure for `DeepSeekV4ColdKVStore`.

- [ ] **Step 3: Implement append storage without full-cache pinning**

Allocate cold rows as ordinary page-aligned host memory. On Linux with NUMA enabled, bind allocation/first-touch to `numaNode`; otherwise use the normal CPU allocator and report `numaNode=-1`.

Grow in chunks of at least 4096 rows. Do not reallocate/copy the entire store for every compressed block.

- [ ] **Step 4: Implement sorted/coalesced gather**

Build `(rowId, originalPosition)` pairs, sort by row id, copy consecutive row runs with `memcpy`, then scatter run results to the original output order. Allocate only the gathered output as pinned memory through the existing CUDA host allocation helpers.

- [ ] **Step 5: Add a staging-byte cap**

Default maximum pinned staging is:

```text
max_concurrency * ratio4_layers_per_active_gpu * 2 buffers * topk * row_bytes
```

Reject configurations exceeding `FASTLLM_DSV4_KV_PINNED_LIMIT_MB`, default 512MB, with a message containing requested and allowed bytes.

- [ ] **Step 6: Run CPU tests**

```bash
./build-kv-test/regressionOps
```

Expected output includes `dsv4 cold-kv gather regression: PASS`.

- [ ] **Step 7: Commit cold storage**

```bash
git add include/models/deepseekv4_kv_offload.h src/models/deepseekv4_kv_offload.cpp include/models/deepseekv4.h test/ops/regressionOps.cpp
git commit -m "实现DSV4冷KV主存存储"
```

### Task 6: Gather top-512 rows and run sparse attention

**Files:**
- Modify: `src/models/deepseekv4.cpp`
- Modify: `src/devices/cuda/models/deepseekv4-kernels.cu`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: GPU top-k ids and CPU cold store.
- Produces: GPU BF16 `[batch, selectedCount, headDim]` gathered cache accepted by the existing cached sparse attention kernel.

- [ ] **Step 1: Add full-GPU vs hybrid-row parity test**

Build 4096 compressed rows, select 512 deterministic ids, and compare:

1. GPU gather from a fully GPU-resident cache;
2. CPU cold-store gather plus pinned H2D.

Require byte-identical gathered BF16 rows before attention and `max_abs <= 1e-3` after attention.

- [ ] **Step 2: Run to establish failure**

```bash
FASTLLM_DSV4_INDEXER=1 ./build-kv-test/regressionOps
```

Expected: hybrid path unavailable or current code copies all compressed rows.

- [ ] **Step 3: Add explicit residency mode**

Parse `FASTLLM_DSV4_KV_RESIDENCY` with exact accepted values:

```text
gpu
hybrid
```

Unknown values are fatal. `gpu` retains the full ratio=4 cache on GPU. `hybrid` stores it only in `DeepSeekV4ColdKVStore` after creation.

- [ ] **Step 4: Implement top-k id transfer and gather**

For each ratio=4 decode layer:

1. run GPU indexer top-k;
2. copy at most 512 INT32 ids to a small host buffer;
3. call `DeepSeekV4ColdKVStore::Gather`;
4. copy the contiguous pinned tensor to the layer GPU;
5. call cached sparse attention with `compressedCount=selectedCount`.

The gathered rows already carry compressor normalization and positional rotation; sparse attention must not re-rotate the keys.

- [ ] **Step 5: Preserve prefix snapshot and restore**

History snapshots store cold rows in CPU memory and index rows according to their existing GPU/CPU snapshot policy. Restoring a prefix must not upload the entire cold store. It may upload only index cache, ratio=128 cache and window cache.

- [ ] **Step 6: Add memory/timing JSONL**

Under `FASTLLM_DSV4_KV_STATS=1`, emit:

```json
{"layer":2,"tokens":262144,"ratio":4,"cold_bytes":1409286144,"index_gpu_bytes":16777216,"gather_bytes":524288,"index_us":0.0,"cpu_gather_us":0.0,"h2d_us":0.0,"attention_us":0.0}
```

Populate real timings and byte counts. Aggregate at most once per 43 layer calls by default.

- [ ] **Step 7: Run parity and sanitizer**

```bash
FASTLLM_DSV4_INDEXER=1 FASTLLM_DSV4_KV_RESIDENCY=hybrid ./build-kv-test/regressionOps
compute-sanitizer --tool memcheck ./build-kv-test/regressionOps
```

Expected: GPU/hybrid parity passes and memcheck reports zero errors.

- [ ] **Step 8: Commit hybrid integration**

```bash
git add src/models/deepseekv4.cpp src/devices/cuda/models/deepseekv4-kernels.cu test/ops/regressionOps.cpp
git commit -m "接入DSV4冷热KV混合缓存"
```

### Task 7: Expose configuration and validate cache slopes

**Files:**
- Modify: `tools/fastllm_pytools/util.py`
- Create: `test/benchmark/dsv4_kv_report.py`
- Modify: `docs/deepseek.md`

**Interfaces:**
- CLI: `--dsv4_kv_residency gpu|hybrid` maps to the environment/runtime setting.
- Parser validates measured cold-cache slope against the model-derived expectation.

- [ ] **Step 1: Add parser unit tests**

```python
def expected_ratio4_bytes_per_token(ratios, head_dim=512, dtype_bytes=2):
    return sum(head_dim * dtype_bytes // ratio for ratio in ratios if ratio == 4)

def assert_slope(samples, expected, tolerance=0.02):
    first, last = samples[0], samples[-1]
    measured = (last["cold_bytes"] - first["cold_bytes"]) / (last["tokens"] - first["tokens"])
    if abs(measured - expected) > expected * tolerance:
        raise AssertionError(f"cold KV slope {measured} differs from {expected}")
```

Test expected slope `5376` for the official 21 ratio=4 layers and both passing/failing measured samples.

- [ ] **Step 2: Run parser test before implementation**

```bash
python3 -m unittest test.benchmark.dsv4_kv_report
```

Expected: failure until helper functions and JSONL loading are implemented.

- [ ] **Step 3: Add CLI validation**

Expose only `gpu` and `hybrid`. Default remains `gpu` until Task 8's 256K correctness gate passes; after that, DSV4 on models whose predicted free GPU memory is insufficient may auto-suggest hybrid but must not silently switch modes.

- [ ] **Step 4: Document capacity and invocation**

Document:

- the 5,376-byte/token CPU cold slope;
- the 1,344-byte/token GPU index slope;
- why generic `--cpu_kvcache` is not the DSV4 hybrid mode;
- tested 256K and 1M commands;
- batch multiplication of cache usage.

- [ ] **Step 5: Run parser tests**

```bash
python3 -m unittest test.benchmark.dsv4_kv_report
```

Expected: `OK`.

- [ ] **Step 6: Commit configuration and docs**

```bash
git add tools/fastllm_pytools/util.py test/benchmark/dsv4_kv_report.py docs/deepseek.md
git commit -m "补充DSV4混合缓存配置"
```

### Task 8: Validate 256K, then the official 1M limit

**Files:**
- Create on server: `artifacts/dsv4_kv_validation/<timestamp>/`
- Modify after evidence exists: `docs/superpowers/specs/2026-07-10-dsv4-flash-ep-kv-design.md`

**Interfaces:**
- Consumes: full DSV4-Flash weights, completed Tasks 1-7, server topology probe.
- Produces: reproducible correctness, memory and performance evidence.

- [ ] **Step 1: Establish a 4K full-GPU reference**

Run a fixed prompt with `FASTLLM_DSV4_KV_RESIDENCY=gpu`, save the first 16 generated-position logits, tokens, timings and memory telemetry.

- [ ] **Step 2: Compare 4K hybrid output**

Run the same input with `FASTLLM_DSV4_KV_RESIDENCY=hybrid`. Require:

```text
token ids identical
logit max_abs <= 5e-2
logit cosine >= 0.999
```

- [ ] **Step 3: Validate synthetic/repeated 128K**

Use deterministic token input or a checked-in long-text generator. Generate 100 decode tokens. Verify:

- CPU cold bytes slope within 2% of 5,376 bytes/token/sequence;
- no ratio=4 full compressed cache remains on GPU;
- no monotonic staging allocation growth after warmup;
- every ratio=4 layer selects at most 512 rows.

- [ ] **Step 4: Pass the 256K gate**

Run 256K prefill/cache construction and generate 100 tokens. Save p50/p95 for index, CPU gather, H2D and attention. This is the first milestone that permits changing the recommended mode to hybrid.

- [ ] **Step 5: Attempt the official 1M limit**

Only after 256K passes, run 1,048,576 context and generate at least 16 tokens. Record peak per-GPU memory, process RSS, cold bytes, index bytes, gather bandwidth and per-token latency.

- [ ] **Step 6: Update the design evidence table**

Add exact server, commit, command and result rows to the design document. Mark any failed gate with the observed failure, not an optimistic projection.

- [ ] **Step 7: Commit measured evidence**

```bash
git add docs/superpowers/specs/2026-07-10-dsv4-flash-ep-kv-design.md
git commit -m "记录DSV4长上下文验证结果"
```

## Final Verification

Run from a clean build on the target server:

```bash
cmake -S . -B build-kv-final -DUNIT_TEST=ON -DUSE_CUDA=ON -DCUDA_ARCH=75
cmake --build build-kv-final -j "$(nproc)" --target regressionOps
./build-kv-final/regressionOps
compute-sanitizer --tool memcheck ./build-kv-final/regressionOps
python3 -m unittest test.benchmark.dsv4_kv_report
git status --short
```

Required evidence:

- build exits `0`;
- capacity, reference indexer, CUDA indexer, cache lifecycle, cold gather and hybrid parity tests pass;
- compute-sanitizer reports `ERROR SUMMARY: 0 errors`;
- parser reports `OK`;
- 256K end-to-end gate passes before any 1M success claim;
- checked-in evidence names the exact commit and server probe directory;
- working tree is clean.

## Execution Order

Tasks 1-6 are strictly sequential. Task 7 can begin after Task 1, but its final documentation must reflect Task 6 behavior. Task 8 begins only after Tasks 1-7 pass on synthetic tests. Expert-parallel work may proceed independently until full-model Task 8, where cache ownership must follow the final layer/device placement.
