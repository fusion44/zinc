# Host-memory offload for MoE expert tensors — research branch

Date: 2026-05-07
Status: Design (Approach A — prove the physics)
Branch: `research/host-mem-offload`

## Goal

Answer one question: can ZINC load Qwen 3.6 35B-A3B UD Q4_K_XL (~21 GB) on a 16 GB AMD RDNA4 card by placing the sparse MoE expert tensors in system RAM and having the GPU read them over PCIe, while still producing coherent output at usable throughput?

This is a throwaway research branch. Its only deliverable is the answer to that question, with measured numbers. It is not a feature, not a knob, not a productized capability. If the result is positive, a follow-up project will design the real feature.

## Target hardware

- GPU: AMD Radeon RX 9070 XT (RDNA4, Navi 48), 16 GB GDDR6
- PCIe link: Gen 5 x16 (32 GT/s), confirmed via `lspci -vv -d 1002:`
- Host: Linux, RADV/Mesa Vulkan stack, resizable BAR assumed enabled
- Model file: `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (already downloaded)

Sustained PCIe bandwidth at Gen 5 x16: ~56 GB/s. Per-token sparse-expert read volume: ~1.1 GB. Theoretical decode ceiling: ~50 tok/s. Realistic estimate at ~60% of ceiling: ~30 tok/s. Reference for comparison: same model fully resident on a 32 GB RDNA4 runs at ~99 tok/s decode.

## Out of scope

- Fit-check changes in `--check`
- New CLI flags, environment variables, or config knobs
- Catalog updates, new GPU profile (`amd-rdna4-16gb` is left for the follow-up feature)
- RDNA3 path, Metal path, NVIDIA path
- KV cache offload, weight offload beyond MoE experts, tiered residency
- OOM fallback, retry logic, automatic policy
- Documentation outside this spec
- New automated tests
- Per-expert hot/cold tracking (ruled out by GGUF's fused expert tensor layout anyway)

## Architecture

One architectural change point — the tensor-upload loop in `src/model/loader.zig:462-488` — plus a small new helper in `src/vulkan/buffer.zig`. Today the loop unconditionally calls `Buffer.initDeviceLocal` for every GGUF tensor. We add a name-based classifier and a second allocation path that targets host-visible memory.

Downstream code is unaffected:

- The `LoadedTensor` struct is unchanged. A `Buffer` does not expose its memory type to consumers.
- Compute graph construction (`src/model/architecture.zig`), descriptor binding, and the dmmv shaders stay as-is. SSBO bindings work uniformly across memory types in Vulkan.
- The router, attention, RoPE, RMSNorm, embeddings, LM head, and KV cache all remain device-local and run at native speed.

The change reverts cleanly: delete the helper, delete the classifier, restore the original allocation call.

## Components

### `src/vulkan/buffer.zig` — new helper

```zig
pub fn initHostVisibleStorage(instance: *const Instance, size: vk.c.VkDeviceSize) !Buffer
```

Allocates a Vulkan buffer with:

- usage: `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` (no `TRANSFER_DST_BIT` — there is no device-local destination to copy to)
- memory properties: `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`

Returns the buffer with `mapped` set, ready for direct memcpy from the GGUF mmap. Mirrors the structure of the existing `initStaging` helper but with storage-buffer usage and no implicit small-allocation assumption.

### `src/model/loader.zig` — classifier

Private function:

```zig
fn shouldOffloadToHost(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "ffn_gate_exps.weight") or
           std.mem.endsWith(u8, name, "ffn_up_exps.weight") or
           std.mem.endsWith(u8, name, "ffn_down_exps.weight") or
           std.mem.endsWith(u8, name, "ffn_down_exps_scale.weight");
}
```

The four suffixes cover the fused expert tensors emitted by GGUF for both Q4_K_XL (Qwen 3.5/3.6) and Q4_K_M (Gemma 4 26B-A4B-style with separate per-row scales). Dense models contain no tensors matching these suffixes, so the function is a no-op for them.

### `src/model/loader.zig` — branch in upload loop

In the existing loop over `gf.tensors.items`, branch on `shouldOffloadToHost(tensor_info.name)`:

- **Host-visible path**: allocate via `initHostVisibleStorage`, memcpy `mmap_data[data_offset..][0..tensor_size]` directly into `gpu_buf.mapped`, skip the staging buffer and `vkCmdCopyBuffer` entirely. Add `tensor_size` to a new `total_host_visible` accumulator.
- **Device-local path**: existing logic, unchanged.

Replace the existing single log line with one that reports both totals:

```
Loaded N tensors | <X> MB device-local VRAM | <Y> MB host-visible (system RAM)
```

Net diff: ~50 lines.

## Data flow

### Load time, per offloaded tensor

```
GGUF mmap region
   memcpy
HOST_VISIBLE buffer (system RAM, mapped)
   bound to descriptor set the same as any SSBO
```

No staging buffer. No GPU-side copy. The Vulkan driver creates a buffer object whose memory comes from a HOST_VISIBLE memory type. On RADV with rebar, this is plain pageable system RAM that the GPU can read directly via PCIe BAR.

### Decode time, per token, per MoE layer

```
1. Router runs on GPU, picks K experts (small, VRAM-resident)
2. dmmv dispatched for ffn_gate / ffn_up / ffn_down expert tensors:
     - shader binds ffn_*_exps.weight as SSBO  (host-visible buffer)
     - shader thread reads slot [expert_id, row, col]
     - GPU memory controller routes the read to PCIe instead of GDDR6
     - bytes flow GPU <- PCIe <- system RAM
3. Result accumulates in VRAM-resident scratch
4. Continues to next layer
```

The shader is unchanged because Vulkan abstracts the memory type away — an SSBO read is an SSBO read. The cost shows up purely as latency on those reads. Other ops (attention, norms, RoPE, LM head, KV cache) run at native speed against GDDR6.

## Error handling

Minimal, by design.

- **Host-visible allocation failure** (likely cause: insufficient free system RAM — the model needs ~18 GB resident in host memory). Propagate the error, abort load. No retry.
- **No HOST_VISIBLE | HOST_COHERENT memory type for storage usage**. Theoretically possible, will not happen on RADV/RDNA4. `findMemoryType` returns null and the load aborts with the existing `error.NoSuitableMemoryType`.
- **Resizable BAR disabled or small BAR**. We cannot reliably detect this from inside Vulkan. Symptom: post-load log shows the expected ~18 GB host-visible allocation, but decode tok/s collapses to <1. The two log lines (host-visible bytes, device-local bytes) plus the throughput number make the diagnosis obvious to a reader.
- **Incoherent output**. Treated as the strongest possible signal — it would mean GPU reads of host-visible SSBOs are returning corrupt data, which is a RADV bug, not something we can work around at this layer. Eyeball check, no automated assertion.

## Testing & success criteria

### Test 1 — preflight

```bash
./zig-out/bin/zinc --check -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

`--check`'s VRAM-fit warning is expected. Do not modify `--check` to suppress it.

### Test 2 — sniff check (correctness)

```bash
./zig-out/bin/zinc -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --chat
```

Pass: response contains "Paris" or otherwise coherent sentence completion.
Fail: gibberish, repeating tokens, NaN-looking output. Indicates host-visible SSBO reads are broken on RADV — design assumption invalidated.

### Test 3 — measurement

```bash
./zig-out/bin/zinc -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt "Write a short paragraph about the Apollo program." --chat
```

Capture:

- decode tok/s (ZINC already logs this)
- peak RSS via `/usr/bin/time -v` (expect ~18-19 GB to confirm host-visible allocation actually went to system RAM)
- the new "device-local / host-visible" log line

### Test 4 — existing suite

`zig build test` must still pass. The change only affects MoE loading; dense-model tests (Qwen 3 8B) do not exercise the offload branch.

### Pass/fail thresholds

| Outcome | Decode tok/s | Output | Verdict |
|---|---:|---|---|
| Clear pass | ≥ 10 | coherent | physics works, promote to feature design |
| Marginal | 1–10 | coherent | physics works but driver/rebar issue worth investigating before promoting |
| Driver fail | < 1 | coherent | rebar likely off or RADV slow path; needs diagnosis, not a feature |
| Correctness fail | any | gibberish | RADV SSBO host-visible read path broken; design invalidated |

## Risks

1. **Resizable BAR not enabled on the test machine.** Will land in the "marginal" or "driver fail" buckets and require BIOS-level investigation before any conclusion is meaningful. Verify `dmesg | grep -i "bar"` and motherboard settings before drawing perf conclusions.
2. **RADV SSBO read from HOST_VISIBLE memory may take a slow path** even with rebar enabled — for example by going through GTT mapping rather than direct BAR access. Symptoms identical to risk 1. Mitigation: read RADV memory type properties from `vkGetPhysicalDeviceMemoryProperties` and confirm a memory type with `HOST_VISIBLE | HOST_COHERENT` and a heap large enough for ~18 GB exists, distinct from the rebar VRAM type.
3. **System RAM pressure.** Allocating ~18 GB of pinned-ish host-visible memory plus the OS, the dev shell, and ZINC's other working set may push a 32 GB host into swap. Document required free RAM in test instructions; recommend ≥32 GB host RAM and closing the browser.
4. **GGUF tensor name suffix coverage.** If a Q4_K_XL Qwen 3.6 tensor uses a name we don't match (rare but possible — quantizers vary), it falls through to device-local and we OOM. Mitigation: dump tensor names with `zinc --check -m ... --json` and confirm the four suffixes cover all `ffn_*_exps*` tensors before declaring any test result.

## Promotion criteria

Promote to a real feature design (separate spec) only if:

- Test 2 produces coherent output, and
- Test 3 produces ≥ 10 tok/s decode

Anything else means we either need to fix the driver path first (BAR, rebar, RADV memory type selection) or that the approach itself does not deliver enough throughput to be useful, and a different mechanism (KV offload, smaller quant, dynamic context) should be explored instead.

## Results (2026-05-07)

**Verdict: clear pass — promote to feature design.**

### Hardware confirmed

- GPU: AMD Radeon RX 9070 XT (RDNA4, Navi 48), reported as `RADV GFX1201`
- VRAM: 16304 MB, 576 GB/s, 64 CUs, wave64, coopmat=yes
- PCIe link: Gen 5 x16 (32 GT/s)
- Resizable BAR: enabled (`Region 0: Memory at f800000000 (64-bit, prefetchable) [size=16G]`)
- System RAM: 62 GB

### Allocation split (load log)

```
info(loader): Loaded 733 tensors | 2554 MB device-local VRAM | 18760 MB host-visible (system RAM)
```

The classifier matched four expert tensor families per layer × 40 layers as expected. Peak host RSS settled around 20.6 GiB total with a light desktop running (Niri compositor, btop). Of that, ~18.7 GiB is the pinned host-visible Vulkan allocation.

### Test 2 — sniff (correctness)

```
Prompt:  "The capital of France is"
Output:  "The capital of France is **Paris**."
```

Coherent. No gibberish, no NaN, no token loops. Confirms RADV serves storage-buffer reads from `HOST_VISIBLE | HOST_COHERENT` memory correctly.

### Test 3 — throughput

| Scenario | Decode tok/s | Notes |
|---|---:|---|
| Short prompt, short generation (sniff) | 59.8 | Cache-best — top-8 of 256 experts hot in GPU cache, no PCIe traffic per token |
| Interactive chat, short context | 13–18 | Sustained, more diverse expert routing |
| Interactive chat, ~10 K cumulative ctx | 12.3 | Slight drop from KV-read scaling with context length |

The 13–18 tok/s steady-state band matches the pre-test prediction (~15-30 tok/s realistic for Gen 5 x16 with sparse MoE). The 60 tok/s sniff number reflects a best-case where the same handful of experts repeats across all 9 output tokens — not representative of real workloads.

**Prefill** is noticeably slower than decode and is the dominant wait time on long-paste inputs. A 6700-token paste produced a multi-second wait before first output token. This is consistent with two known facts:

1. ZINC's batched prefill path is gated off for `n_experts > 0 || ssm_d_inner > 0`, which is exactly Qwen 3.5/3.6 35B-A3B. So prefill on this model is on the per-token slow path even *without* offload.
2. Per-token prefill streaming experts over PCIe compounds the issue.

`nvtop` would help diagnose where time is going (compute vs. memory engine), but the user did not run it during this test.

### Two follow-up patches that landed during integration testing

The plan anticipated three commits (`shouldOffloadToHost`, `initHostVisibleStorage`, loader branch). Two more were needed once the model actually loaded:

- `646ff70` — `forward.zig`'s `tensorBytes` was summing all GGUF bytes against the VRAM budget, leaving negative KV budget and aborting engine init with `ContextLengthDoesNotFit`. Fixed by filtering through `loader.shouldOffloadToHost` (made `pub` for the cross-module call).
- `8f66ad7` — `model_manager.zig` had the same bug in `autoContextTokensForDeviceBudget`, capping interactive sessions at the 4096-token fallback default and producing a misleading `VRAM 21.52 / 15.92 GiB` UI line. Same fix.

After both, the chat UI reports `VRAM 13.51 / 15.92 GiB · ctx reserved 10.94 GiB (71680 tok cap)` — accurate accounting and a usable context budget.

### Caveats and stale numbers

- The `forward.zig` log line `Modeled decode bandwidth: 1016.8 GB/s effective, 576 GB/s theoretical (176.5% utilization, ~17010.5 MB/token)` is wrong: the bandwidth model assumes all weights live in GDDR6, so it overcounts effective bandwidth when most reads come from system RAM. Diagnostic only — does not affect inference correctness.
- `forward_metal.zig` and `model_manager_metal.zig` have analogous `tensorBytes` helpers. Metal uses unified memory so the bug doesn't manifest, but the code is now technically inconsistent with this branch's mental model. Cleanup, not blocking.
- The `LoadedTensor.gpu_buffer` doc comment in `loader.zig` and the `Buffer.upload` doc comment in `vulkan/buffer.zig` still describe the pre-offload world. Cosmetic, low priority.
- The `dmmv_q4k_o_proj_merge.spv` shader was missing on the test machine — pre-existing build issue, unrelated to this branch.

### Recommendation

Promote to a real feature design. Useful next steps:

1. Decide the activation rule — auto-detect via VRAM-fit on `--check`, or an opt-in flag, or both.
2. Generalize the classifier — fall back to `try device-local, on OOM retry host-visible` for non-MoE architectures (Phase B in the original brainstorm).
3. Add the `amd-rdna4-16gb` GPU profile to the catalog so `model list` correctly shows the 35B-A3B model as runnable on these cards.
4. Fix the bandwidth model in `forward.zig` to account for tiered memory residency.
5. Investigate prefill — the per-token MoE+SSM path was already a known bottleneck on full-VRAM RDNA4; it becomes the dominant pain point with offload. Reactivating batched prefill for this architecture (the cycle-50 work) would help both cases.
