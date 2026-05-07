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
