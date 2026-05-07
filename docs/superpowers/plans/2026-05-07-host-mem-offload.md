# Host-memory offload (research branch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the ZINC tensor loader so MoE expert tensors are allocated in `HOST_VISIBLE | HOST_COHERENT` Vulkan memory instead of `DEVICE_LOCAL`, allowing Qwen 3.6 35B-A3B Q4_K_XL (~21 GB) to load on a 16 GB AMD RDNA4 card by streaming sparse-expert reads over PCIe Gen 5.

**Architecture:** Add a name-based classifier (`shouldOffloadToHost`) and a host-visible storage-buffer helper (`Buffer.initHostVisibleStorage`). Branch the existing tensor-upload loop in `loader.zig` so matching tensors take the host-visible path (direct memcpy from GGUF mmap, no staging round-trip) while everything else keeps the existing device-local path.

**Tech Stack:** Zig 0.15.2, Vulkan 1.3 (RADV/Mesa), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-07-host-mem-offload-research-design.md`

**Branch:** `research/host-mem-offload` (already created off main).

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/vulkan/buffer.zig` | Modify (add helper) | New `initHostVisibleStorage` constructor for host-visible storage SSBOs |
| `src/model/loader.zig` | Modify (classifier + branch) | Detect MoE expert tensors and route them to host memory; report both totals |

No new files. No shader changes. No CLI/config/catalog changes.

---

## Task 1: Add `shouldOffloadToHost` classifier with unit test

**Files:**
- Modify: `src/model/loader.zig` (add fn near top of file, add test at bottom alongside existing `parseArchitecture` test)

- [ ] **Step 1: Write the failing test**

Append this `test` block to `src/model/loader.zig` immediately after the existing `test "parseArchitecture"` block (around line 505):

```zig
test "shouldOffloadToHost matches MoE expert tensor suffixes" {
    // Fused per-layer MoE expert tensors — should offload.
    try std.testing.expect(shouldOffloadToHost("blk.0.ffn_gate_exps.weight"));
    try std.testing.expect(shouldOffloadToHost("blk.47.ffn_up_exps.weight"));
    try std.testing.expect(shouldOffloadToHost("blk.7.ffn_down_exps.weight"));
    // Q4_K_M variants emit a separate per-row scale tensor.
    try std.testing.expect(shouldOffloadToHost("blk.3.ffn_down_exps_scale.weight"));

    // Non-expert tensors — must stay device-local.
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_q.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_k.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_v.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_output.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_gate.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_up.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_down.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_gate_inp.weight"));
    try std.testing.expect(!shouldOffloadToHost("output.weight"));
    try std.testing.expect(!shouldOffloadToHost("token_embd.weight"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
zig build test 2>&1 | tail -30
```

Expected: compile error referencing `shouldOffloadToHost` (function not defined).

- [ ] **Step 3: Implement the classifier**

Add this private function to `src/model/loader.zig`. Place it near the top of the file (after the `ModelInspection` struct, before the first public `pub fn`). The exact location does not matter for correctness; pick anywhere in module scope.

```zig
/// Return true if this GGUF tensor name designates a fused MoE expert weight tensor
/// that should be allocated in host-visible (system RAM) memory rather than VRAM.
/// Matches the four suffixes emitted by GGUF for sparse-MoE architectures:
/// `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`, and
/// `ffn_down_exps_scale.weight` (Q4_K_M variants only).
/// Dense tensors and non-expert MoE tensors (router gate, attention, embeddings, etc.)
/// stay device-local.
fn shouldOffloadToHost(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "ffn_gate_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_up_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_down_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_down_exps_scale.weight");
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
zig build test 2>&1 | tail -30
```

Expected: all tests pass, including the new `shouldOffloadToHost` test.

- [ ] **Step 5: Commit**

```bash
git add src/model/loader.zig
git commit -m "loader: add shouldOffloadToHost classifier for MoE expert tensors

Pure function, name-suffix match on the four fused expert tensor names
emitted by GGUF. Used in a follow-up commit to route those tensors to
host-visible memory.
"
```

---

## Task 2: Add `Buffer.initHostVisibleStorage` helper

**Files:**
- Modify: `src/vulkan/buffer.zig` (add new constructor between `initStaging` and `upload`)

- [ ] **Step 1: Add the helper**

In `src/vulkan/buffer.zig`, immediately after the existing `initStaging` function (around line 135) and before the existing `upload` function, add:

```zig
    /// Create a host-visible storage buffer that the GPU reads in place via PCIe.
    /// @param instance Active Vulkan instance and logical device.
    /// @param size Buffer size in bytes.
    /// @returns A storage buffer ready for direct CPU writes through `mapped`.
    /// @note Memory is allocated as `HOST_VISIBLE | HOST_COHERENT`, so the GPU reads
    /// it over PCIe BAR rather than from device-local VRAM. Used to offload large
    /// rarely-touched tensors (MoE expert weights) when device memory is insufficient.
    /// Coherent memory means no explicit invalidate is needed; we only write at load time.
    pub fn initHostVisibleStorage(instance: *const Instance, size: vk.c.VkDeviceSize) !Buffer {
        var buf = try init(
            instance,
            size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );

        var ptr: ?*anyopaque = null;
        const result = vk.c.vkMapMemory(instance.device, buf.memory, 0, size, 0, &ptr);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkMapMemory failed: {d}", .{result});
            buf.deinit();
            return error.MapMemoryFailed;
        }
        buf.mapped = @ptrCast(ptr);

        return buf;
    }
```

Note: this helper has no unit test. The buffer module's existing constructors (`initDeviceLocal`, `initStaging`) likewise have no isolated tests because they require a live Vulkan device. Correctness is verified via the integration tests in Task 4.

- [ ] **Step 2: Verify it compiles**

Run:
```bash
zig build 2>&1 | tail -10
```

Expected: clean build, no errors. The function should be unused at this point (it gets called in Task 3).

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run:
```bash
zig build test 2>&1 | tail -20
```

Expected: all existing tests pass, including the new classifier test from Task 1.

- [ ] **Step 4: Commit**

```bash
git add src/vulkan/buffer.zig
git commit -m "vulkan/buffer: add initHostVisibleStorage helper

Allocates a HOST_VISIBLE | HOST_COHERENT storage buffer the GPU can read
in place over PCIe. Mirrors initStaging structurally but uses
STORAGE_BUFFER usage (no TRANSFER_DST) since there is no device-local
destination to copy to.
"
```

---

## Task 3: Branch the upload loop on the classifier

**Files:**
- Modify: `src/model/loader.zig:461-493` (the tensor-upload loop and final log line)

- [ ] **Step 1: Replace the upload loop**

In `src/model/loader.zig`, find this block (around line 461 onward, immediately after the `errdefer` that cleans up `loaded_tensors`):

```zig
    var total_vram: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;

        // Create device-local buffer
        var gpu_buf = try Buffer.initDeviceLocal(
            instance,
            tensor_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        );
        errdefer gpu_buf.deinit();

        // Stage and copy data to GPU
        const src_data = mmap_data[data_offset..][0..@intCast(tensor_size)];
        var staging = try Buffer.initStaging(instance, tensor_size);
        defer staging.deinit();

        staging.upload(src_data);
        try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &gpu_buf, tensor_size);

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });

        total_vram += tensor_size;
    }

    log.info("Loaded {d} tensors | {d} MB VRAM", .{
        loaded_tensors.items.len,
        total_vram / (1024 * 1024),
    });
```

Replace it with:

```zig
    var total_vram: u64 = 0;
    var total_host_visible: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;
        const src_data = mmap_data[data_offset..][0..@intCast(tensor_size)];
        const offload = shouldOffloadToHost(tensor_info.name);

        var gpu_buf = blk: {
            if (offload) {
                break :blk try Buffer.initHostVisibleStorage(instance, tensor_size);
            } else {
                break :blk try Buffer.initDeviceLocal(
                    instance,
                    tensor_size,
                    vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                );
            }
        };
        errdefer gpu_buf.deinit();

        if (offload) {
            // Host-visible: GPU reads directly over PCIe; memcpy from mmap, no staging.
            gpu_buf.upload(src_data);
            total_host_visible += tensor_size;
        } else {
            // Device-local: stage in host memory, then GPU-side copy into VRAM.
            var staging = try Buffer.initStaging(instance, tensor_size);
            defer staging.deinit();
            staging.upload(src_data);
            try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &gpu_buf, tensor_size);
            total_vram += tensor_size;
        }

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });
    }

    log.info("Loaded {d} tensors | {d} MB device-local VRAM | {d} MB host-visible (system RAM)", .{
        loaded_tensors.items.len,
        total_vram / (1024 * 1024),
        total_host_visible / (1024 * 1024),
    });
```

- [ ] **Step 2: Verify it compiles**

Run:
```bash
zig build 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Run existing test suite to confirm no regressions**

Run:
```bash
zig build test 2>&1 | tail -20
```

Expected: all tests pass. The change only affects loading paths exercised at runtime; the unit tests do not touch Vulkan.

- [ ] **Step 4: Sanity-build a release binary**

Run:
```bash
zig build -Doptimize=ReleaseFast 2>&1 | tail -5
```

Expected: clean release build. (The integration test in Task 4 needs `ReleaseFast` for meaningful tok/s numbers.)

- [ ] **Step 5: Commit**

```bash
git add src/model/loader.zig
git commit -m "loader: route MoE expert tensors to host-visible memory

When shouldOffloadToHost matches a tensor name, allocate via
Buffer.initHostVisibleStorage and memcpy directly from the GGUF mmap.
The GPU reads the tensor in place over PCIe BAR — no staging buffer,
no GPU-side copy. Track host-visible bytes separately and report both
totals in the post-load summary line.

Research-branch change to validate whether sparse MoE inference can
work on a VRAM-constrained card. See
docs/superpowers/specs/2026-05-07-host-mem-offload-research-design.md.
"
```

---

## Task 4: Run the integration tests on the remote 16 GB box

These tests cannot run on the dev machine — they require the RX 9070 XT and the Qwen 3.6 35B-A3B GGUF. Push the branch and SSH to the test box.

**Files:**
- No code changes. Verification step.

- [ ] **Step 1: Push the branch and pull on the remote**

On the dev machine:
```bash
git push -u origin research/host-mem-offload
```

On the remote (`obox`, model in `~/models/`):
```bash
cd ~/dev/zinc  # adjust path if different
git fetch
git switch research/host-mem-offload
zig build -Doptimize=ReleaseFast
```

Expected: clean build on the remote.

- [ ] **Step 2: Verify resizable BAR is enabled (sanity check before any perf measurement)**

On the remote (the GPU is at PCI bus address `03:00.0` per the spec's hardware section):
```bash
dmesg | grep -i "BAR" | grep -iE "amdgpu|0000:03:00" | head -5
sudo lspci -vvs 03:00.0 | grep -E "Region [0-9]:" | head -5
```

Expected: a `Region 0` line sized to match the GPU's 16 GB VRAM, e.g. `Region 0: Memory at ... [size=16G]`. A `[size=256M]` line means rebar is **off** and you must enable it in BIOS (look for "Above 4G Decoding" and "Re-Size BAR Support") before continuing. Without rebar, Step 5's perf result is meaningless — the GPU will still produce coherent output but will go through a slow GTT path.

- [ ] **Step 3: Preflight `--check`**

On the remote:
```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --check -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

Expected: GPU detected as RX 9070 XT (RDNA4), shaders found, and a VRAM-fit warning ("model exceeds available VRAM" or similar). The warning is expected — do not change `--check`.

- [ ] **Step 4: Sniff test for output coherence**

On the remote:
```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt "The capital of France is" --chat 2>&1 | tee /tmp/zinc-sniff.log
```

Pass: the response contains "Paris" or otherwise completes the sentence coherently.
Fail: gibberish, repeating tokens, or NaN-looking output → host-visible SSBO reads are returning corrupt data on RADV. Stop and capture the log; this invalidates the design assumption.

Also confirm the new log line appears, e.g.:
```
loader: Loaded 723 tensors | 3xxx MB device-local VRAM | 18xxx MB host-visible (system RAM)
```

The host-visible total should be ~18 GB and the device-local total should be ~3 GB.

- [ ] **Step 5: Throughput measurement**

On the remote:
```bash
export RADV_PERFTEST=coop_matrix
/usr/bin/time -v ./zig-out/bin/zinc \
    -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt "Write a short paragraph about the Apollo program." --chat \
    2>&1 | tee /tmp/zinc-throughput.log
```

Capture from the log:
- Decode tok/s (ZINC logs this at the end of the run)
- "Maximum resident set size" from `time -v` (expect ~18-19 GB to confirm system RAM is actually used)
- The "device-local / host-visible" log line

- [ ] **Step 6: Existing test suite still green**

Optional but recommended, on the remote:
```bash
zig build test 2>&1 | tail -20
```

Expected: same pass/fail set as `main`. The change should not break any existing tests.

- [ ] **Step 7: Record the verdict**

Append a short results block to the spec file (`docs/superpowers/specs/2026-05-07-host-mem-offload-research-design.md`) under a new `## Results (YYYY-MM-DD)` section, with these fields:

```markdown
## Results (2026-05-07)

- Build: <clean / errors>
- Sniff test (Test 4): <coherent / gibberish>; first 200 chars of output: <quote>
- Decode tok/s: <N>
- Peak RSS: <N GB>
- Device-local VRAM: <N MB>
- Host-visible (system RAM): <N MB>
- Verdict: <clear pass / marginal / driver fail / correctness fail>
- Notes: <anything notable — rebar status, dmesg surprises, etc.>
```

Cross-reference the verdict against the pass/fail thresholds in the spec.

- [ ] **Step 8: Commit results**

```bash
git add docs/superpowers/specs/2026-05-07-host-mem-offload-research-design.md
git commit -m "spec: record host-mem-offload research results"
```

---

## Self-review notes

- **Spec coverage:** every numbered section of the spec maps to a task. Architecture/Components → Tasks 1-3. Data flow → implicit in Task 3 (the loop is the data flow). Error handling → implicit (no new code paths beyond standard Vulkan error propagation, which the existing `try` already handles). Testing & success criteria → Task 4.
- **No placeholders:** every code step is complete. No TBDs, TODOs, or `similar to Task N` stubs.
- **Type consistency:** `shouldOffloadToHost(name: []const u8) bool` is referenced identically in Tasks 1 and 3. `Buffer.initHostVisibleStorage(instance: *const Instance, size: vk.c.VkDeviceSize) !Buffer` matches its definition in Task 2 and call site in Task 3. `total_host_visible` is the same name in declaration and final log.
- **Reversibility:** the entire change is contained to two files. `git revert` of all four commits, or `git switch main && git branch -D research/host-mem-offload`, restores the prior behavior.
