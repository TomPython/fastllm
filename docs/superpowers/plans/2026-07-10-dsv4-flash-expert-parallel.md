# DSV4-Flash Expert Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn fastllm's hidden MultiCUDA MergeMOE prototype into a correct, observable, topology-aware 8-GPU output-reduce expert-parallel backend for DeepSeek-V4-Flash on RTX 2080 Ti.

**Architecture:** Replicate the per-layer activation and router metadata to all eight GPUs, mask routes on-device so each GPU runs only its owned experts, then sum `[batch, hidden]` partial outputs through NCCL AllReduce. Keep the first implementation deterministic and round-robin for correctness; add weighted placement only after real routing histograms exist.

**Tech Stack:** C++17, CUDA SM75, NCCL, fastllm MultiCUDA worker pool, CMake `UNIT_TEST`, `test/ops/regressionOps.cpp`, Linux shell probes.

## Global Constraints

- Target hardware is 8 × RTX 2080 Ti 22GB with no NVLink, 2 × E5-2680 v4, and 64GB DDR4-2133.
- Base revision is `d176be5ab71f65c1c47aa0cadff1a759b9f47ace` or a descendant containing the same DSV4 and MultiCUDA paths.
- First protocol is output AllReduce; do not add token all-to-all in this plan.
- Routed experts must never be replicated; each routed expert has exactly one GPU owner.
- Shared expert remains outside routed MergeMOE by running DSV4 with `--cuda_se true`.
- Do not introduce a mandatory dependency beyond the existing CUDA and NCCL build.
- CPU/D2H route filtering is forbidden in the steady-state EP path.
- Every CUDA fast path must have a deterministic CPU or single-CUDA reference test.
- Commit messages follow the repository `git-commit-zh-split` skill and use Chinese.

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/dsv4_ep_probe.sh` | Collect immutable server topology, P2P, NCCL, memory and build facts. |
| `include/devices/multicuda/moe_expert_placement.h` | Pure C++ expert-owner interfaces with no CUDA dependency. |
| `src/devices/multicuda/moe_expert_placement.cpp` | Deterministic and weighted owner-map algorithms. |
| `include/devices/multicuda/fastllm-multicuda.cuh` | Declare GPU route-mask and EP instrumentation entry points. |
| `src/devices/multicuda/fastllm-multicuda.cu` | Implement route masking and retain existing NCCL wrappers. |
| `src/devices/multicuda/multicudadevice.cpp` | Place routed experts, schedule per-GPU local MergeMOE and reduce outputs. |
| `test/ops/regressionOps.cpp` | Owner-map, route-mask and MultiCUDA MergeMOE correctness regressions. |
| `test/benchmark/dsv4_ep_report.py` | Parse JSONL runtime samples and produce comparable summaries. |
| `docs/mixforward.md` | Document supported EP environment/configuration and limitations. |
| `CMakeLists.txt` | Register the placement implementation and tests in CUDA builds. |

### Task 1: Capture the target server baseline

**Files:**
- Create: `tools/dsv4_ep_probe.sh`
- Create on server at runtime: `artifacts/dsv4_ep_probe/<timestamp>/`

**Interfaces:**
- Consumes: Linux shell, `nvidia-smi`, `lscpu`, optional `numactl`, optional `p2pBandwidthLatencyTest`, optional `all_reduce_perf`.
- Produces: a directory containing raw command output and `summary.txt`; later performance claims must cite this directory.

- [ ] **Step 1: Add the probe script**

```bash
#!/usr/bin/env bash
set -euo pipefail

out="${1:-artifacts/dsv4_ep_probe/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"

nvidia-smi -L > "$out/nvidia-smi-L.txt"
nvidia-smi topo -m > "$out/nvidia-smi-topo.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv > "$out/gpus.csv"
lscpu -e=CPU,NODE,SOCKET,CORE > "$out/lscpu-e.txt"
lscpu > "$out/lscpu.txt"
command -v numactl >/dev/null && numactl --hardware > "$out/numactl.txt" || true

if command -v p2pBandwidthLatencyTest >/dev/null; then
    p2pBandwidthLatencyTest > "$out/p2p.txt"
fi

if command -v all_reduce_perf >/dev/null; then
    all_reduce_perf -b 8K -e 1M -f 2 -g 8 > "$out/nccl-allreduce.txt"
fi

{
    echo "commit=$(git rev-parse HEAD)"
    echo "kernel=$(uname -r)"
    echo "cuda=$(nvcc --version | tail -n 1 || true)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
} > "$out/summary.txt"

echo "$out"
```

- [ ] **Step 2: Run shell syntax validation**

Run:

```bash
bash -n tools/dsv4_ep_probe.sh
```

Expected: exit code `0`, no output.

- [ ] **Step 3: Run the probe on the RTX 2080 Ti server**

Run:

```bash
bash tools/dsv4_ep_probe.sh
```

Expected: prints one artifact directory; that directory contains at least `nvidia-smi-L.txt`, `nvidia-smi-topo.txt`, `gpus.csv`, `lscpu-e.txt`, and `summary.txt`.

- [ ] **Step 4: Record the topology decision**

Append a small checked-in table to `docs/superpowers/specs/2026-07-10-dsv4-flash-ep-kv-design.md` using the actual `nvidia-smi topo -m` result. Each GPU row must record CUDA id, PCI bus id, closest NUMA node, and whether P2P is available to every other GPU.

- [ ] **Step 5: Commit the probe**

```bash
git add tools/dsv4_ep_probe.sh docs/superpowers/specs/2026-07-10-dsv4-flash-ep-kv-design.md
git commit -m "补充DSV4服务器拓扑探针"
```

### Task 2: Extract deterministic expert placement

**Files:**
- Create: `include/devices/multicuda/moe_expert_placement.h`
- Create: `src/devices/multicuda/moe_expert_placement.cpp`
- Modify: `CMakeLists.txt`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: expert count, CUDA device ids, optional per-expert byte size and observed route count.
- Produces:
  - `MoeExpertPlacement BuildRoundRobinMoeExpertPlacement(int expertCount, const std::vector<int>& devices)`
  - `MoeExpertPlacement BuildWeightedMoeExpertPlacement(const std::vector<uint64_t>& expertBytes, const std::vector<uint64_t>& routeCounts, const std::vector<int>& devices)`

- [ ] **Step 1: Add failing pure-C++ tests**

Add this test helper and call it from `main()` before device-dependent tests:

```cpp
#include "devices/multicuda/moe_expert_placement.h"

void RunMoePlacementRegression() {
    using fastllm::BuildRoundRobinMoeExpertPlacement;
    using fastllm::BuildWeightedMoeExpertPlacement;

    auto rr = BuildRoundRobinMoeExpertPlacement(10, {0, 2, 5});
    Expect(rr.ownerByExpert == std::vector<int>({0, 2, 5, 0, 2, 5, 0, 2, 5, 0}),
           "round-robin owner map mismatch");
    Expect(rr.expertsByDevice.at(0).size() == 4, "device 0 expert count mismatch");
    Expect(rr.expertsByDevice.at(2).size() == 3, "device 2 expert count mismatch");
    Expect(rr.expertsByDevice.at(5).size() == 3, "device 5 expert count mismatch");

    auto weighted = BuildWeightedMoeExpertPlacement(
        {10, 10, 10, 10}, {100, 1, 100, 1}, {0, 1});
    Expect(weighted.ownerByExpert[0] != weighted.ownerByExpert[2],
           "hot experts should be split across devices");
    Expect(weighted.totalBytesByDevice.at(0) == 20 && weighted.totalBytesByDevice.at(1) == 20,
           "weighted placement must preserve byte balance");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cmake -S . -B build-ep-test -DUNIT_TEST=ON -DUSE_CUDA=ON -DCUDA_ARCH=75
cmake --build build-ep-test -j "$(nproc)" --target regressionOps
```

Expected: compile failure because `moe_expert_placement.h` does not exist.

- [ ] **Step 3: Add the public placement types**

```cpp
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace fastllm {
    struct MoeExpertPlacement {
        std::vector<int> ownerByExpert;
        std::map<int, std::vector<int>> expertsByDevice;
        std::map<int, uint64_t> totalBytesByDevice;
        std::map<int, uint64_t> weightedRoutesByDevice;
    };

    MoeExpertPlacement BuildRoundRobinMoeExpertPlacement(
        int expertCount, const std::vector<int> &devices);

    MoeExpertPlacement BuildWeightedMoeExpertPlacement(
        const std::vector<uint64_t> &expertBytes,
        const std::vector<uint64_t> &routeCounts,
        const std::vector<int> &devices);
}
```

- [ ] **Step 4: Implement deterministic placement**

Implement round-robin exactly as `devices[expert % devices.size()]`. Implement weighted placement by sorting experts by `(routeCounts desc, expertBytes desc, expert id asc)` and assigning each expert to the device minimizing this tuple:

```cpp
std::tuple<uint64_t, uint64_t, int>{
    placement.weightedRoutesByDevice[device],
    placement.totalBytesByDevice[device],
    device
}
```

Reject empty devices and mismatched byte/route vector sizes with `ErrorInFastLLM`.

- [ ] **Step 5: Register the source in CUDA builds**

Add `src/devices/multicuda/moe_expert_placement.cpp` beside `multicudadevice.cpp` in both CUDA and ROCm MultiCUDA source lists in `CMakeLists.txt`.

- [ ] **Step 6: Run the regression**

Run:

```bash
cmake --build build-ep-test -j "$(nproc)" --target regressionOps
./build-ep-test/regressionOps
```

Expected output includes `moe placement regression: PASS`; existing device tests either pass or explicitly skip.

- [ ] **Step 7: Commit placement separately**

```bash
git add CMakeLists.txt include/devices/multicuda/moe_expert_placement.h src/devices/multicuda/moe_expert_placement.cpp test/ops/regressionOps.cpp
git commit -m "抽离多卡专家放置策略"
```

### Task 3: Replace D2H filtering with a CUDA route mask

**Files:**
- Modify: `include/devices/multicuda/fastllm-multicuda.cuh`
- Modify: `src/devices/multicuda/fastllm-multicuda.cu`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: INT32 `[rows, topk]` indices, FLOAT32 scores, an INT32 owner table, and current owner slot.
- Produces: same-shape local indices/scores where non-local routes are `-1` and `0.0f`.

```cpp
void FastllmCudaMaskMoeRoutesForOwner(
    const int32_t *indices,
    const float *scores,
    int32_t *localIndices,
    float *localScores,
    int routeCount,
    const int32_t *ownerByExpert,
    int expertCount,
    int ownerSlot);
```

- [ ] **Step 1: Add a failing CUDA regression**

Use indices `{0,1,7,3,4,7}` with owner table `{0,1,0,1,0,1,0,1}`. For owner slot `0`, expect `{0,-1,-1,-1,4,-1}` and scores `{1,0,0,0,5,0}`. Add the symmetric owner slot `1` assertion.

- [ ] **Step 2: Build to verify the missing symbol**

```bash
cmake --build build-ep-test -j "$(nproc)" --target regressionOps
```

Expected: compile or link failure for `FastllmCudaMaskMoeRoutesForOwner`.

- [ ] **Step 3: Implement the mask kernel**

```cpp
__global__ void FastllmMaskMoeRoutesKernel(
    const int32_t *indices, const float *scores,
    int32_t *localIndices, float *localScores,
    int routeCount, const int32_t *ownerByExpert,
    int expertCount, int ownerSlot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routeCount) return;
    int expert = indices[i];
    bool local = expert >= 0 && expert < expertCount && ownerByExpert[expert] == ownerSlot;
    localIndices[i] = local ? expert : -1;
    localScores[i] = local ? scores[i] : 0.0f;
}
```

Launch with 256 threads and `(routeCount + 255) / 256` blocks. Check `cudaGetLastError()` and use the existing fastllm CUDA error reporting convention.

- [ ] **Step 4: Run the regression**

```bash
./build-ep-test/regressionOps
```

Expected output includes `cuda moe route-mask regression: PASS`.

- [ ] **Step 5: Run compute-sanitizer**

```bash
compute-sanitizer --tool memcheck ./build-ep-test/regressionOps
```

Expected: `ERROR SUMMARY: 0 errors`.

- [ ] **Step 6: Commit the kernel**

```bash
git add include/devices/multicuda/fastllm-multicuda.cuh src/devices/multicuda/fastllm-multicuda.cu test/ops/regressionOps.cpp
git commit -m "改用CUDA筛选本地专家路由"
```

### Task 4: Integrate placement and route masking into MultiCuda MergeMOE

**Files:**
- Modify: `src/devices/multicuda/multicudadevice.cpp`
- Modify: `test/ops/regressionOps.cpp`

**Interfaces:**
- Consumes: existing `FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL` switch and `MoeExpertPlacement`.
- Produces: MultiCUDA MergeMOE outputs matching the single-device reference for batch 1, 8 and 16.

- [ ] **Step 1: Add parameterized EP regressions**

Add `RunMultiCudaMergeMoeExpertParallelRegression(int batch, int topk)` and run it for `(1,6)`, `(8,6)`, `(16,6)` when at least two CUDA devices exist. Create 16 small FP16 experts, use a deterministic route pattern, compute the reference on `cuda:0`, then compute on `multicuda:0,1` with EP enabled.

Use a scoped environment helper that restores the previous value:

```cpp
class ScopedEnvFlag {
public:
    ScopedEnvFlag(const char *name, const char *value) : name(name) {
        const char *old = std::getenv(name);
        hadPrevious = old != nullptr;
        if (hadPrevious) previous = old;
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
    ~ScopedEnvFlag() {
#ifdef _WIN32
        _putenv_s(name.c_str(), hadPrevious ? previous.c_str() : "");
#else
        if (!hadPrevious) unsetenv(name.c_str());
        else setenv(name.c_str(), previous.c_str(), 1);
#endif
    }
private:
    std::string name;
    std::string previous;
    bool hadPrevious = false;
};
```

- [ ] **Step 2: Run the regression to expose current failures**

```bash
FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL=1 ./build-ep-test/regressionOps
```

Expected before integration: batch>1 mismatch or evidence that the old D2H branch was used.

- [ ] **Step 3: Build one placement per MergeMOE weight set**

Replace direct `expert % devices.size()` decisions with `BuildRoundRobinMoeExpertPlacement`. Cache the placement by a stable key derived from the weight vector address plus expert count; invalidate it when the device list changes.

Do not use raw weight pointer values as a persistent cross-process identifier. The cache is process-local only.

- [ ] **Step 4: Allocate per-device owner tables and local routing tensors**

For each GPU worker:

1. allocate/upload `ownerByExpert` once;
2. allocate `localIndex` and `localScore` at the same logical shape as the router tensors;
3. call `FastllmCudaMaskMoeRoutesForOwner`;
4. call local `DoCudaMergeMOE` with only local expert weights;
5. initialize the local output to zero even when no local route is active.

- [ ] **Step 5: Remove steady-state route D2H copies**

Delete the expert-parallel branch copies that move the full index and score arrays to host for filtering. Keep a debug-only assertion under `FASTLLM_MULTICUDA_MOE_EP_VERIFY_ROUTE_MASK=1` that copies both GPU and CPU masks and compares them.

- [ ] **Step 6: Preserve NCCL fallback behavior**

If NCCL cannot initialize, fail EP explicitly with:

```text
MultiCuda MergeMOE expert parallel requires NCCL; disable FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL or fix NCCL initialization.
```

Do not silently fall back to CPU summation because that changes the performance contract and can exceed host memory.

- [ ] **Step 7: Run regressions and sanitizer**

```bash
FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL=1 ./build-ep-test/regressionOps
compute-sanitizer --tool memcheck ./build-ep-test/regressionOps
```

Expected: all three EP cases print `PASS`; memcheck reports zero errors.

- [ ] **Step 8: Commit integration**

```bash
git add src/devices/multicuda/multicudadevice.cpp test/ops/regressionOps.cpp
git commit -m "完善多卡MergeMOE专家并行"
```

### Task 5: Add EP observability and reproducible benchmarks

**Files:**
- Modify: `src/devices/multicuda/multicudadevice.cpp`
- Modify: `test/ops/cppOps.cpp`
- Create: `test/benchmark/dsv4_ep_report.py`

**Interfaces:**
- Environment: `FASTLLM_MULTICUDA_MOE_EP_STATS=1` emits one JSON object per reporting window to stderr.
- JSON fields: `layer`, `batch`, `topk`, `owner_routes`, `local_us`, `allreduce_us`, `output_bytes`, `devices`.

- [ ] **Step 1: Add a parser test before instrumentation**

Create a parser whose pure function is:

```python
def summarize(samples: list[dict]) -> dict:
    return {
        "samples": len(samples),
        "local_p50_us": percentile([x["local_us"] for x in samples], 50),
        "local_p95_us": percentile([x["local_us"] for x in samples], 95),
        "allreduce_p50_us": percentile([x["allreduce_us"] for x in samples], 50),
        "allreduce_p95_us": percentile([x["allreduce_us"] for x in samples], 95),
        "routes_by_device": merge_counts(x["owner_routes"] for x in samples),
    }
```

Add an inline `unittest` with three samples and exact percentile expectations.

- [ ] **Step 2: Run the parser test and verify failure**

```bash
python3 -m unittest test.benchmark.dsv4_ep_report
```

Expected: failure until `percentile`, `merge_counts`, and `summarize` are implemented.

- [ ] **Step 3: Add runtime timing**

Use CUDA events around local MergeMOE and NCCL AllReduce. Accumulate 43 calls by default, configurable with `FASTLLM_MULTICUDA_MOE_EP_STATS_EVERY`. Emit valid single-line JSON only when the stats flag is enabled.

- [ ] **Step 4: Extend `optest` with DSV4 dimensions**

Add an EP benchmark case using defaults:

```text
hidden=4096 inter=2048 experts=256 topk=6 batch=1
```

Support `batch=1,8,16`. Use compact test weights when the full 256-expert allocation would exceed the current test GPU, and print the exact allocated expert count in benchmark metadata.

- [ ] **Step 5: Run parser and microbenchmarks**

```bash
python3 -m unittest test.benchmark.dsv4_ep_report
./optest --op mergemoe_ep --device multicuda:0,1 --param batch=1 --iters 100 --warmup 20
./optest --op mergemoe_ep --device multicuda:0,1 --param batch=8 --iters 100 --warmup 20
./optest --op mergemoe_ep --device multicuda:0,1 --param batch=16 --iters 100 --warmup 20
```

Expected: parser tests pass; benchmark output reports finite positive `avg_ms` and no mismatch.

- [ ] **Step 6: Commit observability separately**

```bash
git add src/devices/multicuda/multicudadevice.cpp test/ops/cppOps.cpp test/benchmark/dsv4_ep_report.py
git commit -m "补充专家并行性能观测"
```

### Task 6: Validate DSV4 shared expert and full-model placement

**Files:**
- Modify: `src/models/deepseekv4.cpp`
- Modify: `tools/fastllm_pytools/util.py`
- Modify: `docs/mixforward.md`

**Interfaces:**
- Consumes: `--cuda_se true`, `--device`, `--moe_device`, EP environment flag.
- Produces: one startup placement report and deterministic rejection of unsupported configurations.

- [ ] **Step 1: Add configuration validation**

When EP is enabled for DSV4, require:

```text
cuda shared expert enabled
moe device starts with multicuda
at least two unique CUDA ids
NCCL available
```

Reject invalid combinations before model warmup, naming the failing condition.

- [ ] **Step 2: Add a placement report**

Under `FASTLLM_DSV4_PLACEMENT_REPORT=1`, print per device:

```json
{"device":0,"routed_expert_bytes":0,"dense_weight_bytes":0,"shared_expert_bytes":0,"estimated_free_bytes":0}
```

Populate real values from Data byte counts after placement. Never label estimates as actual CUDA free memory; include both fields when available.

- [ ] **Step 3: Add fixed-prompt parity script**

Document and run the same prompt with EP disabled and enabled, saving logits for the first 16 generated positions. Calculate:

```python
max_abs = float(np.max(np.abs(reference - candidate)))
cosine = float(np.dot(reference.ravel(), candidate.ravel()) /
               (np.linalg.norm(reference) * np.linalg.norm(candidate)))
assert max_abs <= 5e-2
assert cosine >= 0.999
```

- [ ] **Step 4: Run 8-card server validation**

Run batch/concurrency 1, 8 and 16 with 4K context first. Record:

- total tok/s;
- per-request tok/s;
- p50/p95 inter-token latency;
- each GPU peak memory;
- local expert time;
- AllReduce time;
- route counts by GPU.

Expected: no GPU is missing routed weights; EP output passes parity; concurrency 8 throughput exceeds concurrency 1 normalized throughput without unbounded p95 latency.

- [ ] **Step 5: Document supported invocation**

Add a tested command to `docs/mixforward.md` with the actual model path and GPU list used on the server. Do not document untested environment combinations.

- [ ] **Step 6: Commit DSV4 integration docs and validation**

```bash
git add src/models/deepseekv4.cpp tools/fastllm_pytools/util.py docs/mixforward.md
git commit -m "接入DSV4路由专家并行"
```

## Final Verification

Run all of the following on the target server from a clean build directory:

```bash
cmake -S . -B build-ep-final -DUNIT_TEST=ON -DUSE_CUDA=ON -DCUDA_ARCH=75
cmake --build build-ep-final -j "$(nproc)" --target regressionOps optest
./build-ep-final/regressionOps
compute-sanitizer --tool memcheck ./build-ep-final/regressionOps
python3 -m unittest test.benchmark.dsv4_ep_report
git status --short
```

Required evidence:

- build exits `0`;
- every EP regression reports `PASS` or an explicit hardware-based `SKIP`;
- compute-sanitizer reports `ERROR SUMMARY: 0 errors`;
- Python parser tests report `OK`;
- working tree is clean;
- a checked-in benchmark note includes the server probe artifact path and full-model parity metrics.

## Execution Order

Tasks 1 through 4 are strictly sequential. Task 5 can begin after Task 3 but must be rebased onto Task 4 before final benchmarks. Task 6 requires Tasks 1 through 5 and the complete model on the remote server.
