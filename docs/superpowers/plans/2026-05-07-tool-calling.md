# OpenAI-compatible tool calling (ChatML) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI-compatible `tools` / `tool_choice` / `tool_calls` semantics to ZINC's `/v1/chat/completions` endpoint for ChatML-family models (Qwen3, Qwen3.5, Qwen3.6) so agentic clients (opencode, openai-python) can use ZINC as a drop-in backend. Other template kinds (Llama3, Gemma, gpt-oss) silently ignore `tools` and behave exactly as today.

**Architecture:** A new `src/server/tool_format.zig` module exposes a `ToolFormat` vtable interface with two concrete implementations: `ChatMLToolFormat` (Qwen3 verbatim tool prompt + parser + streaming state machine) and `NoopToolFormat` (silent fallback). Routes.zig and tokenizer.zig get small integration changes — tool concerns are dispatched through `tool_format.forTemplate(template_kind)` with no per-kind branching at call sites.

**Tech Stack:** Zig 0.15.2, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-07-tool-calling-design.md`

**Branch:** This plan should run on a fresh branch (`feature/tool-calling`) off `main`. The spec on the host-mem-offload branch was for design, not implementation. The user is expected to set up the branch before starting.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/server/tool_format.zig` | Create | Tool definition/call types, vtable interface, ChatMLToolFormat, NoopToolFormat, StreamingDetector, factory |
| `src/server/routes.zig` | Modify | Extend ChatRequestBody with tools/tool_choice; wire tool_format into chat template call; integrate StreamingDetector with SSE writer; parse final assistant output for tool_calls |
| `src/model/tokenizer.zig` | Modify | Extend ChatTemplateOptions with tools+tool_format; ChatML branch renders tool definitions and tool result messages via the interface |

No other files affected. Metal path (`forward_metal.zig`, `model_manager_metal.zig`) is unchanged — tool calling lives entirely in the server/template layer.

---

## Phase 1 — `tool_format.zig` module skeleton

### Task 1: Create the module with types and NoopToolFormat

**Files:**
- Create: `src/server/tool_format.zig`

This task creates the new module with all its public types, the `ToolFormat` vtable, and a working `NoopToolFormat` that compiles, exports a `forTemplate` factory always returning `NoopToolFormat`, and has its own unit tests. ChatMLToolFormat comes in later tasks; for now `forTemplate(.chatml)` also returns Noop. This keeps every commit a passing build.

- [ ] **Step 1: Create the file with types, vtable, and Noop impl**

Create `/home/f44/dev/stuff/zinc/src/server/tool_format.zig`:

```zig
//! Pluggable tool-calling format dispatch for chat completions.
//! @section Tool Calling
//! ChatMLToolFormat handles Qwen3-family models. NoopToolFormat is the
//! silent fallback for any other template kind.
const std = @import("std");
const TemplateKind = @import("../model/tokenizer.zig").TemplateKind;

const log = std.log.scoped(.tool_format);

/// One tool definition extracted from the request's `tools` array.
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    /// Raw JSON object representing the parameters schema. Not validated.
    parameters_json: []const u8,
};

/// One parsed tool call extracted from assistant output.
pub const ToolCall = struct {
    /// Generated as "call_<n>" by the parser. OpenAI requires non-empty id.
    id: []const u8,
    name: []const u8,
    /// Raw JSON object string for the tool's arguments.
    arguments_json: []const u8,
};

pub const ParsedAssistantOutput = struct {
    /// Anything outside `<tool_call>...</tool_call>` blocks.
    text_content: []const u8,
    /// Empty slice if no tool calls were detected.
    tool_calls: []const ToolCall,
};

pub const FeedResult = enum {
    /// Bytes pass through to the SSE content delta.
    emit_as_content,
    /// Bytes are buffered internally; do not emit.
    hold,
    /// A complete tool_call was just parsed; pull it via takePendingToolCall.
    tool_call_complete,
};

pub const StreamingDetector = struct {
    state: State = .normal_text,
    hold_buf: std.ArrayList(u8) = .{},
    pending_calls: std.ArrayList(ToolCall) = .{},
    next_id: u32 = 0,
    allocator: std.mem.Allocator,

    const State = enum { normal_text, buffer_partial_tag, inside_tool_call };

    pub fn deinit(self: *StreamingDetector) void {
        self.hold_buf.deinit(self.allocator);
        self.pending_calls.deinit(self.allocator);
    }

    pub fn feed(self: *StreamingDetector, chunk: []const u8) !FeedResult {
        _ = self;
        _ = chunk;
        return .emit_as_content; // placeholder; real impl in Task 7+
    }

    pub fn takePendingToolCall(self: *StreamingDetector) ?ToolCall {
        if (self.pending_calls.items.len == 0) return null;
        return self.pending_calls.orderedRemove(0);
    }

    pub fn finalize(self: *StreamingDetector) []const u8 {
        return self.hold_buf.items;
    }
};

pub const ToolFormat = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        renderToolDefinitions: *const fn (
            ctx: *anyopaque,
            tools: []const ToolDefinition,
            buf: *std.ArrayList(u8),
            allocator: std.mem.Allocator,
        ) anyerror!void,

        renderToolResultMessage: *const fn (
            ctx: *anyopaque,
            tool_call_id: []const u8,
            content: []const u8,
            buf: *std.ArrayList(u8),
            allocator: std.mem.Allocator,
        ) anyerror!void,

        parseAssistantToolCalls: *const fn (
            ctx: *anyopaque,
            model_output: []const u8,
            allocator: std.mem.Allocator,
        ) anyerror!ParsedAssistantOutput,

        newStreamingDetector: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
        ) anyerror!*StreamingDetector,
    };

    pub fn renderToolDefinitions(
        self: ToolFormat,
        tools: []const ToolDefinition,
        buf: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
    ) anyerror!void {
        return self.vtable.renderToolDefinitions(self.ptr, tools, buf, allocator);
    }

    pub fn renderToolResultMessage(
        self: ToolFormat,
        tool_call_id: []const u8,
        content: []const u8,
        buf: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
    ) anyerror!void {
        return self.vtable.renderToolResultMessage(self.ptr, tool_call_id, content, buf, allocator);
    }

    pub fn parseAssistantToolCalls(
        self: ToolFormat,
        model_output: []const u8,
        allocator: std.mem.Allocator,
    ) anyerror!ParsedAssistantOutput {
        return self.vtable.parseAssistantToolCalls(self.ptr, model_output, allocator);
    }

    pub fn newStreamingDetector(
        self: ToolFormat,
        allocator: std.mem.Allocator,
    ) anyerror!*StreamingDetector {
        return self.vtable.newStreamingDetector(self.ptr, allocator);
    }
};

// ============================================================
// NoopToolFormat — silent fallback for non-ChatML templates.
// ============================================================

pub const NoopToolFormat = struct {
    fn render_defs(_: *anyopaque, _: []const ToolDefinition, _: *std.ArrayList(u8), _: std.mem.Allocator) !void {}

    fn render_result(_: *anyopaque, _: []const u8, content: []const u8, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        try buf.appendSlice(allocator, content);
    }

    fn parse_calls(_: *anyopaque, model_output: []const u8, allocator: std.mem.Allocator) !ParsedAssistantOutput {
        _ = allocator;
        return .{ .text_content = model_output, .tool_calls = &.{} };
    }

    fn new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
        const d = try allocator.create(StreamingDetector);
        d.* = .{ .allocator = allocator };
        return d;
    }

    const vtable = ToolFormat.VTable{
        .renderToolDefinitions = render_defs,
        .renderToolResultMessage = render_result,
        .parseAssistantToolCalls = parse_calls,
        .newStreamingDetector = new_detector,
    };

    var instance: u8 = 0; // dummy ctx pointer; Noop has no state
};

pub fn noopToolFormat() ToolFormat {
    return .{
        .ptr = @ptrCast(&NoopToolFormat.instance),
        .vtable = &NoopToolFormat.vtable,
    };
}

// ============================================================
// Factory: pick the right ToolFormat for a template kind.
// ============================================================

pub fn forTemplate(template_kind: TemplateKind) ToolFormat {
    _ = template_kind; // until Task 11 wires ChatMLToolFormat in
    return noopToolFormat();
}

// ============================================================
// Tests
// ============================================================

test "NoopToolFormat.renderToolDefinitions is a no-op" {
    const tf = noopToolFormat();
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{ .name = "foo", .description = "bar", .parameters_json = "{}" },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}

test "NoopToolFormat.parseAssistantToolCalls returns text_content unchanged" {
    const tf = noopToolFormat();
    const result = try tf.parseAssistantToolCalls("hello world", std.testing.allocator);
    try std.testing.expectEqualStrings("hello world", result.text_content);
    try std.testing.expectEqual(@as(usize, 0), result.tool_calls.len);
}

test "NoopToolFormat.renderToolResultMessage appends raw content" {
    const tf = noopToolFormat();
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    try tf.renderToolResultMessage("call_0", "result text", &buf, std.testing.allocator);
    try std.testing.expectEqualStrings("result text", buf.items);
}

test "forTemplate returns a usable ToolFormat for every kind" {
    inline for (.{ .chatml, .llama3, .gemma, .unknown }) |kind| {
        const tf = forTemplate(kind);
        // Smoke test: the returned ToolFormat's vtable methods are callable.
        const result = try tf.parseAssistantToolCalls("x", std.testing.allocator);
        _ = result;
    }
}
```

- [ ] **Step 2: Add the module to the build**

In `/home/f44/dev/stuff/zinc/src/server/routes.zig`, near the top with the other server-module imports, add a line that imports `tool_format` so the file gets compiled as part of the test build. The exact placement is alongside the existing `model_manager_mod` and `http` imports — look for the block of `const X = @import(...)` lines near the top of `routes.zig`:

```zig
const tool_format = @import("tool_format.zig");
```

This is intentionally an unused import for now — Zig is fine with unused imports at module top level. Later tasks will use the symbols. (If lint complains later, the next task that uses `tool_format.X` removes the lint by virtue of using it.)

- [ ] **Step 3: Verify compile and tests pass**

```bash
zig build 2>&1 | tail -5
zig build test 2>&1 | grep -E "tests? passed|FAIL" | head -5
```

Expected: clean build; test count is 4 higher than before this task. The 4 new tests are the ones in `tool_format.zig`.

- [ ] **Step 4: Commit**

```bash
git add src/server/tool_format.zig src/server/routes.zig
git commit -m "tool_format: module skeleton with NoopToolFormat

Adds ToolDefinition/ToolCall/ParsedAssistantOutput/FeedResult types,
the ToolFormat vtable interface, a stub StreamingDetector, and a
working NoopToolFormat that silently ignores tools. forTemplate is
wired but returns Noop for every template kind — ChatMLToolFormat
gets wired in a later task. Module is imported by routes.zig so the
build picks it up.
"
```

---

## Phase 2 — ChatMLToolFormat parser (TDD)

### Task 2: parseAssistantToolCalls — single tool call

**Files:**
- Modify: `src/server/tool_format.zig`

This task adds the `ChatMLToolFormat` struct (still incomplete) with just enough of `parseAssistantToolCalls` to handle a single `<tool_call>...</tool_call>` block. Subsequent tasks add multi-call support, malformed-JSON fallback, and the renderers.

- [ ] **Step 1: Write the failing test**

Append to the bottom of `src/server/tool_format.zig`:

```zig
test "ChatMLToolFormat.parseAssistantToolCalls extracts a single tool call" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me check that.
        \\<tool_call>
        \\{"name": "Bash", "arguments": {"command": "ls /"}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer std.testing.allocator.free(parsed.tool_calls);

    try std.testing.expectEqualStrings("Let me check that.\n", parsed.text_content);
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expectEqualStrings("call_0", parsed.tool_calls[0].id);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expectEqualStrings(
        \\{"command": "ls /"}
    , parsed.tool_calls[0].arguments_json);
}
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: compile error referencing `chatmlToolFormat` (function not defined yet).

- [ ] **Step 3: Implement ChatMLToolFormat with single-call parser**

In `src/server/tool_format.zig`, **above** the `// Tests` comment block, add:

```zig
// ============================================================
// ChatMLToolFormat — Qwen3 / Qwen3.5 / Qwen3.6 style.
// ============================================================

const tool_call_open = "<tool_call>";
const tool_call_close = "</tool_call>";

pub const ChatMLToolFormat = struct {
    fn parse_calls(
        _: *anyopaque,
        model_output: []const u8,
        allocator: std.mem.Allocator,
    ) !ParsedAssistantOutput {
        var calls: std.ArrayList(ToolCall) = .{};
        errdefer calls.deinit(allocator);

        var text_content: []const u8 = model_output;
        var search_from: usize = 0;
        var next_id: u32 = 0;

        while (std.mem.indexOfPos(u8, model_output, search_from, tool_call_open)) |open_at| {
            const after_open = open_at + tool_call_open.len;
            const close_at = std.mem.indexOfPos(u8, model_output, after_open, tool_call_close) orelse break;

            const inner = std.mem.trim(u8, model_output[after_open..close_at], " \t\r\n");
            const Parsed = struct { name: []const u8 = "", arguments: std.json.Value = .null };
            var parsed = std.json.parseFromSlice(Parsed, allocator, inner, .{ .ignore_unknown_fields = true }) catch {
                // Malformed JSON: skip this block; later task handles graceful fallback.
                search_from = close_at + tool_call_close.len;
                continue;
            };
            defer parsed.deinit();

            const id = try std.fmt.allocPrint(allocator, "call_{d}", .{next_id});
            next_id += 1;

            // Re-serialize arguments as JSON string. (For now, simple object handling.)
            var args_buf: std.ArrayList(u8) = .{};
            defer args_buf.deinit(allocator);
            try std.json.Stringify.value(parsed.value.arguments, .{}, args_buf.writer(allocator));

            try calls.append(allocator, .{
                .id = id,
                .name = try allocator.dupe(u8, parsed.value.name),
                .arguments_json = try args_buf.toOwnedSlice(allocator),
            });

            // text_content is everything before the first tool_call open tag.
            if (calls.items.len == 1) {
                text_content = model_output[0..open_at];
            }
            search_from = close_at + tool_call_close.len;
        }

        return .{
            .text_content = text_content,
            .tool_calls = try calls.toOwnedSlice(allocator),
        };
    }

    fn render_defs(_: *anyopaque, _: []const ToolDefinition, _: *std.ArrayList(u8), _: std.mem.Allocator) !void {
        // Implemented in Task 5.
        return;
    }

    fn render_result(_: *anyopaque, _: []const u8, _: []const u8, _: *std.ArrayList(u8), _: std.mem.Allocator) !void {
        // Implemented in Task 6.
        return;
    }

    fn new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
        const d = try allocator.create(StreamingDetector);
        d.* = .{ .allocator = allocator };
        return d;
    }

    const vtable = ToolFormat.VTable{
        .renderToolDefinitions = render_defs,
        .renderToolResultMessage = render_result,
        .parseAssistantToolCalls = parse_calls,
        .newStreamingDetector = new_detector,
    };

    var instance: u8 = 0;
};

pub fn chatmlToolFormat() ToolFormat {
    return .{
        .ptr = @ptrCast(&ChatMLToolFormat.instance),
        .vtable = &ChatMLToolFormat.vtable,
    };
}
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
zig build test 2>&1 | tail -20
```

Expected: pass count is 1 higher than after Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: ChatMLToolFormat skeleton + single-call parser

Implements parseAssistantToolCalls for the simple case of one
<tool_call>...</tool_call> block in the assistant output. Generates
ids as call_<n>, re-serializes the arguments JSON object, and returns
text_content as everything before the first open tag. Multi-call,
malformed-JSON fallback, renderers, and streaming come in subsequent
tasks.
"
```

---

### Task 3: parseAssistantToolCalls — multiple parallel calls

**Files:**
- Modify: `src/server/tool_format.zig`

The single-call parser already handles the loop structure. This task verifies it correctly handles multiple calls and adds the explicit test.

- [ ] **Step 1: Add the failing test**

Append to the test section of `src/server/tool_format.zig`:

```zig
test "ChatMLToolFormat.parseAssistantToolCalls extracts multiple parallel calls" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me run two commands.
        \\<tool_call>
        \\{"name": "Bash", "arguments": {"command": "ls /"}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "Read", "arguments": {"path": "/etc/hostname"}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer {
        for (parsed.tool_calls) |c| {
            std.testing.allocator.free(c.id);
            std.testing.allocator.free(c.name);
            std.testing.allocator.free(c.arguments_json);
        }
        std.testing.allocator.free(parsed.tool_calls);
    }

    try std.testing.expectEqual(@as(usize, 2), parsed.tool_calls.len);
    try std.testing.expectEqualStrings("call_0", parsed.tool_calls[0].id);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expectEqualStrings("call_1", parsed.tool_calls[1].id);
    try std.testing.expectEqualStrings("Read", parsed.tool_calls[1].name);
    try std.testing.expectEqualStrings("Let me run two commands.\n", parsed.text_content);
}
```

Also update the single-call test (added in Task 2) to free the now-allocated `id`, `name`, and `arguments_json`. Replace its `defer` with the same multi-field free pattern:

```zig
defer {
    for (parsed.tool_calls) |c| {
        std.testing.allocator.free(c.id);
        std.testing.allocator.free(c.name);
        std.testing.allocator.free(c.arguments_json);
    }
    std.testing.allocator.free(parsed.tool_calls);
}
```

- [ ] **Step 2: Run the test, confirm it passes immediately**

```bash
zig build test 2>&1 | grep -E "tests? passed|FAIL" | head -3
```

Expected: pass — the parser already loops over all `<tool_call>` blocks.

If the test fails, fix the parser. The most likely issue is `text_content` being overwritten on later calls; the implementation only sets it on the first match (`if (calls.items.len == 1)`).

- [ ] **Step 3: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: test parallel tool call extraction"
```

---

### Task 4: parseAssistantToolCalls — malformed JSON fallback

**Files:**
- Modify: `src/server/tool_format.zig`

Per the spec error-handling section: malformed JSON inside `<tool_call>` should be dropped from `tool_calls` and the entire `<tool_call>...</tool_call>` block (including tags) included in `text_content`. The current parser silently skips malformed blocks — we need to keep them in `text_content` instead.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "ChatMLToolFormat.parseAssistantToolCalls falls back to text on malformed JSON" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me try.
        \\<tool_call>
        \\{not valid json}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "OK", "arguments": {}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer {
        for (parsed.tool_calls) |c| {
            std.testing.allocator.free(c.id);
            std.testing.allocator.free(c.name);
            std.testing.allocator.free(c.arguments_json);
        }
        std.testing.allocator.free(parsed.tool_calls);
        std.testing.allocator.free(parsed.text_content);
    }

    // The valid call still extracted; id is "call_0" since malformed didn't get an id.
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expectEqualStrings("call_0", parsed.tool_calls[0].id);
    try std.testing.expectEqualStrings("OK", parsed.tool_calls[0].name);
    // The malformed block (with tags) appears in text_content.
    try std.testing.expect(std.mem.indexOf(u8, parsed.text_content, "<tool_call>\n{not valid json}\n</tool_call>") != null);
}
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: failure on the `text_content` assertion — currently the malformed block is silently dropped.

- [ ] **Step 3: Update the parser to preserve malformed blocks in text_content**

In `src/server/tool_format.zig`, replace the body of `ChatMLToolFormat.parse_calls` with:

```zig
fn parse_calls(
    _: *anyopaque,
    model_output: []const u8,
    allocator: std.mem.Allocator,
) !ParsedAssistantOutput {
    var calls: std.ArrayList(ToolCall) = .{};
    errdefer calls.deinit(allocator);
    var text_buf: std.ArrayList(u8) = .{};
    errdefer text_buf.deinit(allocator);

    var search_from: usize = 0;
    var next_id: u32 = 0;

    while (std.mem.indexOfPos(u8, model_output, search_from, tool_call_open)) |open_at| {
        // Append text between search_from and open_at to text_buf.
        try text_buf.appendSlice(allocator, model_output[search_from..open_at]);

        const after_open = open_at + tool_call_open.len;
        const close_at = std.mem.indexOfPos(u8, model_output, after_open, tool_call_close) orelse {
            // Unclosed tool_call: leave open tag and rest of input in text_buf.
            try text_buf.appendSlice(allocator, model_output[open_at..]);
            search_from = model_output.len;
            break;
        };

        const inner = std.mem.trim(u8, model_output[after_open..close_at], " \t\r\n");
        const Parsed = struct { name: []const u8 = "", arguments: std.json.Value = .null };
        var parsed = std.json.parseFromSlice(Parsed, allocator, inner, .{ .ignore_unknown_fields = true }) catch {
            // Malformed: include the full block (with tags) in text_buf.
            log.warn("malformed JSON inside <tool_call>; falling back to text", .{});
            try text_buf.appendSlice(allocator, model_output[open_at .. close_at + tool_call_close.len]);
            search_from = close_at + tool_call_close.len;
            continue;
        };
        defer parsed.deinit();

        const id = try std.fmt.allocPrint(allocator, "call_{d}", .{next_id});
        next_id += 1;

        var args_buf: std.ArrayList(u8) = .{};
        defer args_buf.deinit(allocator);
        try std.json.Stringify.value(parsed.value.arguments, .{}, args_buf.writer(allocator));

        try calls.append(allocator, .{
            .id = id,
            .name = try allocator.dupe(u8, parsed.value.name),
            .arguments_json = try args_buf.toOwnedSlice(allocator),
        });

        search_from = close_at + tool_call_close.len;
    }

    // Append any tail after the last tool_call.
    if (search_from < model_output.len) {
        try text_buf.appendSlice(allocator, model_output[search_from..]);
    }

    return .{
        .text_content = try text_buf.toOwnedSlice(allocator),
        .tool_calls = try calls.toOwnedSlice(allocator),
    };
}
```

Note the change: `text_content` is now an owned allocation that the caller must free, in addition to freeing `tool_calls`. Update the previous tests' defers accordingly:

In the single-call test (Task 2), add at the end of the defer:
```zig
std.testing.allocator.free(parsed.text_content);
```

In the multi-call test (Task 3), add the same line.

- [ ] **Step 4: Run all tests, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: all tests pass. The `text_content` is now allocated, not a slice into `model_output` — the test defers that were updated above prevent leaks.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: malformed tool_call falls back to text_content

Per the spec's error-handling section, broken JSON inside <tool_call>
should not abort the response. Drop the call from tool_calls and
include the full block (with tags) in text_content. Also handles
unclosed tool_call tags by including the open tag + tail in
text_content. text_content is now an owned allocation.
"
```

---

## Phase 3 — ChatMLToolFormat renderers

### Task 5: renderToolDefinitions

**Files:**
- Modify: `src/server/tool_format.zig`

This emits Qwen3's verbatim tool prompt fragment. The exact text is hard-coded — pulled from Qwen3.5's published `tokenizer.chat_template` from `unsloth/Qwen3.5-35B-A3B-GGUF`. Verify the literal string against that source before merging.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "ChatMLToolFormat.renderToolDefinitions emits Qwen3 verbatim tool prompt" {
    const tf = chatmlToolFormat();
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{
            .name = "Bash",
            .description = "Execute a bash command.",
            .parameters_json =
            \\{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
            ,
        },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);

    // Critical fragments — exact verbatim text from Qwen3.5's chat_template.
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "# Tools") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "</tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"name\": \"Bash\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"description\": \"Execute a bash command.\"") != null);
    // Each tool's JSON object is one line inside <tools>...</tools>.
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tool_call>") != null);
}
```

- [ ] **Step 2: Run, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — the buf is empty since `render_defs` is a stub.

- [ ] **Step 3: Implement renderToolDefinitions**

In `src/server/tool_format.zig`, replace the stub `ChatMLToolFormat.render_defs` with:

```zig
fn render_defs(
    _: *anyopaque,
    tools: []const ToolDefinition,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    if (tools.len == 0) return;

    // Verbatim from Qwen3.5's published chat_template. If you change a single
    // character of the static text below, verify against unsloth/Qwen3.5-35B-A3B-GGUF
    // tokenizer.chat_template — model accuracy depends on it matching training.
    try buf.appendSlice(allocator,
        \\
        \\# Tools
        \\
        \\You may call one or more functions to assist with the user query.
        \\
        \\You are provided with function signatures within <tools></tools> XML tags:
        \\<tools>
        \\
    );

    for (tools) |tool| {
        // One JSON object per line, OpenAI-style {type: "function", function: {...}}.
        try buf.appendSlice(allocator, "{\"type\": \"function\", \"function\": {\"name\": \"");
        try buf.appendSlice(allocator, tool.name);
        try buf.appendSlice(allocator, "\", \"description\": \"");
        try buf.appendSlice(allocator, tool.description);
        try buf.appendSlice(allocator, "\", \"parameters\": ");
        try buf.appendSlice(allocator, tool.parameters_json);
        try buf.appendSlice(allocator, "}}\n");
    }

    try buf.appendSlice(allocator,
        \\</tools>
        \\
        \\For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        \\<tool_call>
        \\{"name": <function-name>, "arguments": <args-json-object>}
        \\</tool_call>
    );
}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: renderToolDefinitions for ChatML

Verbatim Qwen3.5 tool-prompt format: # Tools heading, <tools>...</tools>
block with one JSON object per tool, then the <tool_call> instruction
example. Empty tools array is a no-op.
"
```

---

### Task 6: renderToolResultMessage

**Files:**
- Modify: `src/server/tool_format.zig`

Tool result messages render as `<tool_response>...</tool_response>` blocks. The renderer just produces the block; the tokenizer caller (Task 14) wraps multiple consecutive tool messages inside one user turn.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "ChatMLToolFormat.renderToolResultMessage wraps content in tool_response tags" {
    const tf = chatmlToolFormat();
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    try tf.renderToolResultMessage("call_0", "exit code 0\nfile: hello.txt\n", &buf, std.testing.allocator);
    try std.testing.expectEqualStrings(
        \\<tool_response>
        \\exit code 0
        \\file: hello.txt
        \\</tool_response>
        \\
    , buf.items);
}
```

- [ ] **Step 2: Run, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — buf is empty.

- [ ] **Step 3: Implement**

In `src/server/tool_format.zig`, replace the stub `ChatMLToolFormat.render_result` with:

```zig
fn render_result(
    _: *anyopaque,
    tool_call_id: []const u8,
    content: []const u8,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    _ = tool_call_id; // Qwen3 doesn't render the call id; the order matches the assistant's tool_calls.
    try buf.appendSlice(allocator, "<tool_response>\n");
    try buf.appendSlice(allocator, content);
    if (content.len == 0 or content[content.len - 1] != '\n') {
        try buf.appendSlice(allocator, "\n");
    }
    try buf.appendSlice(allocator, "</tool_response>\n");
}
```

- [ ] **Step 4: Run, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: renderToolResultMessage for ChatML

Wraps the result content in <tool_response>...</tool_response>.
The tool_call_id is not part of the render — Qwen3 matches results
to calls by position, not by id.
"
```

---

## Phase 4 — StreamingDetector

### Task 7: StreamingDetector — emit normal text

**Files:**
- Modify: `src/server/tool_format.zig`

Build the streaming state machine incrementally. First: feed plain text → emit_as_content.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "StreamingDetector.feed emits normal text as content" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }
    const result = try detector.feed("Hello, world!");
    try std.testing.expectEqual(FeedResult.emit_as_content, result);
}
```

- [ ] **Step 2: Run, confirm it passes**

The current stub returns `.emit_as_content` already. This test is here to lock in that behavior before subsequent tasks add real state.

```bash
zig build test 2>&1 | tail -20
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: regression test for StreamingDetector content passthrough"
```

---

### Task 8: StreamingDetector — full tool call across one chunk

**Files:**
- Modify: `src/server/tool_format.zig`

Implement enough of `StreamingDetector.feed` to detect a complete `<tool_call>...</tool_call>` block in a single chunk and produce `tool_call_complete`.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "StreamingDetector.feed returns tool_call_complete for one full call" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    const chunk = "Sure!\n<tool_call>\n{\"name\":\"X\",\"arguments\":{}}\n</tool_call>";
    // The detector must emit "Sure!\n" first as content, then signal completion.
    // Convention: feed returns the *most recent* state transition. For a chunk
    // that contains both content and a complete tool call, the caller should
    // call feed again with an empty slice to drain remaining state. We model
    // this by having feed return tool_call_complete only when the call is
    // fully parsed; the caller then pulls calls via takePendingToolCall.

    const result = try detector.feed(chunk);
    try std.testing.expectEqual(FeedResult.tool_call_complete, result);
    const call = detector.takePendingToolCall() orelse return error.NoCall;
    defer {
        std.testing.allocator.free(call.id);
        std.testing.allocator.free(call.name);
        std.testing.allocator.free(call.arguments_json);
    }
    try std.testing.expectEqualStrings("X", call.name);
    try std.testing.expectEqualStrings("call_0", call.id);
}
```

- [ ] **Step 2: Run, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — `feed` always returns `.emit_as_content`.

- [ ] **Step 3: Implement single-chunk parsing**

In `src/server/tool_format.zig`, replace `StreamingDetector.feed` and update the struct to track the parser:

```zig
pub fn feed(self: *StreamingDetector, chunk: []const u8) !FeedResult {
    // Append the chunk to the hold buffer; we always parse from there.
    try self.hold_buf.appendSlice(self.allocator, chunk);

    // Look for a complete <tool_call>...</tool_call> in the buffer.
    if (std.mem.indexOf(u8, self.hold_buf.items, tool_call_open)) |open_at| {
        if (std.mem.indexOfPos(u8, self.hold_buf.items, open_at + tool_call_open.len, tool_call_close)) |close_at| {
            // Parse the call.
            const inner_start = open_at + tool_call_open.len;
            const inner = std.mem.trim(u8, self.hold_buf.items[inner_start..close_at], " \t\r\n");
            const Parsed = struct { name: []const u8 = "", arguments: std.json.Value = .null };
            var parsed = std.json.parseFromSlice(Parsed, self.allocator, inner, .{ .ignore_unknown_fields = true }) catch {
                // Malformed: skip past, leave bytes in hold_buf as content next time.
                return .emit_as_content;
            };
            defer parsed.deinit();

            const id = try std.fmt.allocPrint(self.allocator, "call_{d}", .{self.next_id});
            self.next_id += 1;

            var args_buf: std.ArrayList(u8) = .{};
            defer args_buf.deinit(self.allocator);
            try std.json.Stringify.value(parsed.value.arguments, .{}, args_buf.writer(self.allocator));

            try self.pending_calls.append(self.allocator, .{
                .id = id,
                .name = try self.allocator.dupe(u8, parsed.value.name),
                .arguments_json = try args_buf.toOwnedSlice(self.allocator),
            });

            // Trim consumed bytes (everything up to and including </tool_call>) out of hold_buf.
            const after_close = close_at + tool_call_close.len;
            std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
            self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);

            return .tool_call_complete;
        }
    }

    return .emit_as_content;
}
```

Note: the buffered text *before* the open tag (`Sure!\n` in the test) is currently lost on `tool_call_complete`. That's a deficiency this task knowingly leaves — Task 9 fixes it by having `feed` flush prior content before signaling tool_call_complete. The current test only checks that the tool call is captured.

- [ ] **Step 4: Run, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: the new test passes; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: StreamingDetector parses single-chunk tool calls"
```

---

### Task 9: StreamingDetector — flush content before tool_call

**Files:**
- Modify: `src/server/tool_format.zig`

When a chunk contains both prefix content (`"Sure!\n"`) and a complete tool call, we need to emit the content first, then signal tool_call_complete. The cleanest way: split the chunk processing so feed only consumes one transition at a time, and the caller invokes feed in a loop.

Refactor the contract: `feed(chunk)` consumes any leading bytes that don't start a tool_call, and either returns `.emit_as_content` (with the content available via `takeContentDelta`) or `.tool_call_complete` (no leading content; pull the call via `takePendingToolCall`). When both apply to the same buffer, the first call returns `.emit_as_content`, then the caller passes an empty chunk and gets `.tool_call_complete`.

- [ ] **Step 1: Add the failing test**

Replace the previous Task 8 test (which only checked the tool call was captured) with a more rigorous version. Since the previous test still passes today, the cleanest approach is to add a new test that asserts content is emitted first.

Append:

```zig
test "StreamingDetector.feed emits prefix content before tool_call_complete" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    const chunk = "Sure!\n<tool_call>\n{\"name\":\"X\",\"arguments\":{}}\n</tool_call>";

    // First call: the detector should return emit_as_content with "Sure!\n"
    // available via takeContentDelta.
    const r1 = try detector.feed(chunk);
    try std.testing.expectEqual(FeedResult.emit_as_content, r1);
    const content = detector.takeContentDelta();
    try std.testing.expectEqualStrings("Sure!\n", content);

    // Second call (empty): drain the parsed tool call.
    const r2 = try detector.feed("");
    try std.testing.expectEqual(FeedResult.tool_call_complete, r2);
    const call = detector.takePendingToolCall() orelse return error.NoCall;
    defer {
        std.testing.allocator.free(call.id);
        std.testing.allocator.free(call.name);
        std.testing.allocator.free(call.arguments_json);
    }
    try std.testing.expectEqualStrings("X", call.name);
}
```

- [ ] **Step 2: Run, confirm it fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — `takeContentDelta` does not exist.

- [ ] **Step 3: Refactor StreamingDetector**

In `src/server/tool_format.zig`, replace the entire `StreamingDetector` struct definition with this version. The hold buffer is split into `content_pending` (bytes ready for emit) and the underlying `hold_buf` (bytes still being inspected for partial tags):

```zig
pub const StreamingDetector = struct {
    /// Bytes pending emission as content delta.
    content_pending: std.ArrayList(u8) = .{},
    /// Bytes being inspected for an incomplete <tool_call> open tag.
    hold_buf: std.ArrayList(u8) = .{},
    /// Tool calls fully parsed and waiting for the caller to drain.
    pending_calls: std.ArrayList(ToolCall) = .{},
    next_id: u32 = 0,
    allocator: std.mem.Allocator,

    /// Maximum bytes to hold while waiting for an open tag to disambiguate.
    /// "<tool_call>" is 11 bytes; we use 24 for safety margin.
    const max_hold = 24;

    pub fn deinit(self: *StreamingDetector) void {
        self.content_pending.deinit(self.allocator);
        self.hold_buf.deinit(self.allocator);
        for (self.pending_calls.items) |c| {
            self.allocator.free(c.id);
            self.allocator.free(c.name);
            self.allocator.free(c.arguments_json);
        }
        self.pending_calls.deinit(self.allocator);
    }

    pub fn feed(self: *StreamingDetector, chunk: []const u8) !FeedResult {
        try self.hold_buf.appendSlice(self.allocator, chunk);

        // 1. Look for a complete <tool_call>...</tool_call> in the buffer.
        if (std.mem.indexOf(u8, self.hold_buf.items, tool_call_open)) |open_at| {
            // Anything before the open tag is content. Move it to content_pending.
            if (open_at > 0) {
                try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..open_at]);
                std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[open_at..]);
                self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - open_at);
            }

            // Try to find the close tag.
            const close_idx = std.mem.indexOfPos(u8, self.hold_buf.items, tool_call_open.len, tool_call_close);
            if (close_idx) |close_at| {
                // Parse the inner JSON.
                const inner = std.mem.trim(u8, self.hold_buf.items[tool_call_open.len..close_at], " \t\r\n");
                const Parsed = struct { name: []const u8 = "", arguments: std.json.Value = .null };
                var parsed = std.json.parseFromSlice(Parsed, self.allocator, inner, .{ .ignore_unknown_fields = true }) catch {
                    // Malformed: emit the entire block as content.
                    log.warn("malformed JSON inside <tool_call>; emitting as content", .{});
                    const after_close = close_at + tool_call_close.len;
                    try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..after_close]);
                    std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                    self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);
                    if (self.content_pending.items.len > 0) return .emit_as_content;
                    return .hold;
                };
                defer parsed.deinit();

                const id = try std.fmt.allocPrint(self.allocator, "call_{d}", .{self.next_id});
                self.next_id += 1;

                var args_buf: std.ArrayList(u8) = .{};
                defer args_buf.deinit(self.allocator);
                try std.json.Stringify.value(parsed.value.arguments, .{}, args_buf.writer(self.allocator));

                try self.pending_calls.append(self.allocator, .{
                    .id = id,
                    .name = try self.allocator.dupe(u8, parsed.value.name),
                    .arguments_json = try args_buf.toOwnedSlice(self.allocator),
                });

                // Strip the consumed block from hold_buf.
                const after_close = close_at + tool_call_close.len;
                std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);

                // If we still have pending content from before the tag, emit it first.
                if (self.content_pending.items.len > 0) return .emit_as_content;
                return .tool_call_complete;
            }
            // Open tag found but no close yet — wait for more chunks.
            if (self.content_pending.items.len > 0) return .emit_as_content;
            return .hold;
        }

        // 2. No open tag in buffer. Check whether the tail could be the start of one.
        const tail_start = if (self.hold_buf.items.len > max_hold) self.hold_buf.items.len - max_hold else 0;
        if (std.mem.indexOfScalarPos(u8, self.hold_buf.items, tail_start, '<')) |lt_at| {
            // The tail starting at lt_at *might* be a partial open tag. Hold it.
            // Bytes before lt_at are safe to emit.
            if (lt_at > 0) {
                try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..lt_at]);
                std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[lt_at..]);
                self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - lt_at);
            }
            // Check if the tail can no longer match (mismatched bytes).
            const tail = self.hold_buf.items;
            const ok_so_far = std.mem.startsWith(u8, tool_call_open, tail) or std.mem.startsWith(u8, tail, tool_call_open);
            if (!ok_so_far or tail.len > tool_call_open.len) {
                // Tail can't be the start of <tool_call>; flush it as content.
                try self.content_pending.appendSlice(self.allocator, tail);
                self.hold_buf.clearRetainingCapacity();
            }
            if (self.content_pending.items.len > 0) return .emit_as_content;
            return .hold;
        }

        // 3. No '<' anywhere — it's all content.
        try self.content_pending.appendSlice(self.allocator, self.hold_buf.items);
        self.hold_buf.clearRetainingCapacity();
        if (self.content_pending.items.len > 0) return .emit_as_content;
        return .hold;
    }

    /// Drain pending content bytes (zero copy: you must consume before next feed).
    pub fn takeContentDelta(self: *StreamingDetector) []const u8 {
        const out = self.content_pending.items;
        // Reset without freeing; next emission reuses the buffer.
        self.content_pending = .{};
        return out;
    }

    pub fn takePendingToolCall(self: *StreamingDetector) ?ToolCall {
        if (self.pending_calls.items.len == 0) return null;
        return self.pending_calls.orderedRemove(0);
    }

    /// Called at end of stream. Returns any held bytes (treated as content).
    pub fn finalize(self: *StreamingDetector) []const u8 {
        // If there's an unclosed <tool_call> at end-of-stream, the held bytes
        // include the open tag — emit as plain content per the spec.
        const pending = self.content_pending.items;
        const held = self.hold_buf.items;
        if (held.len == 0) return pending;
        // Concatenate pending + held into pending and return.
        self.content_pending.appendSlice(self.allocator, held) catch return pending;
        self.hold_buf.clearRetainingCapacity();
        return self.content_pending.items;
    }
};
```

Note: `takeContentDelta` returns a slice that becomes invalid as soon as the next `feed` is called (since we reset `content_pending`). Callers must consume the slice (e.g., write it to the SSE output stream) immediately.

- [ ] **Step 4: Run all tests, confirm pass**

```bash
zig build test 2>&1 | tail -25
```

Expected: all pass. The Task 8 test still passes (one feed call returns tool_call_complete because the chunk has no leading content for a single-call test... wait — Task 8's chunk *did* have leading content `"Sure!\n"`. With the new behavior that test fails.)

If the Task 8 test fails: replace its assertions with two-step pattern matching the Task 9 test. The simpler alternative is to delete the Task 8 test since Task 9's is strictly more thorough.

For this plan, **delete the Task 8 test** (the one that asserted `.tool_call_complete` for a chunk with leading content). Locate the test block named `"StreamingDetector.feed returns tool_call_complete for one full call"` and remove it entirely. Re-run tests; all should pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: StreamingDetector flushes content before tool calls

Refactor: separate content_pending (ready to emit) from hold_buf
(bytes still being inspected for tags). feed now returns
emit_as_content first when there's leading content, then
tool_call_complete on the next call. Adds takeContentDelta accessor.
Handles partial-tag held bytes up to max_hold (24).
"
```

---

### Task 10: StreamingDetector — false-positive partial tag

**Files:**
- Modify: `src/server/tool_format.zig`

A chunk like `"Hello <th"` contains `<` but is not a tool_call. The detector should hold it briefly, and on the next chunk that disambiguates it (e.g. `"ought>"`), flush all held bytes as content.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "StreamingDetector flushes false-positive partial tag as content" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    // First chunk: ends with "<th" — could potentially be the start of <tool_call>.
    // The detector should hold it (or flush "Hello " and hold "<th").
    _ = try detector.feed("Hello <th");

    // Second chunk: "ink>" makes the prefix "<think>" which is *not* a tool_call.
    // The detector should flush all held bytes as content.
    const r2 = try detector.feed("ink>");
    try std.testing.expectEqual(FeedResult.emit_as_content, r2);

    // Aggregate everything emitted so far.
    var all: std.ArrayList(u8) = .{};
    defer all.deinit(std.testing.allocator);
    try all.appendSlice(std.testing.allocator, detector.takeContentDelta());

    try std.testing.expectEqualStrings("Hello <think>", all.items);
}
```

- [ ] **Step 2: Run, confirm pass or fail**

```bash
zig build test 2>&1 | tail -20
```

Expected: the test should pass with the implementation from Task 9. The relevant branch is the "tail can no longer match" check in `feed` — when `tail.len > tool_call_open.len` or the tail isn't a prefix of `<tool_call>`, we flush.

If it fails, inspect the held buffer state. The most likely issue: `<th` is a 3-byte prefix of `<tool_call>` (since both start with `<t`), so the detector keeps holding. After `<think>`, the `<th` portion has diverged at index 2 (`i` vs `o`), so we should flush — verify the prefix check in the implementation.

- [ ] **Step 3: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: regression test for false-positive partial tag"
```

---

### Task 11: Wire ChatMLToolFormat into forTemplate

**Files:**
- Modify: `src/server/tool_format.zig`

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "forTemplate returns ChatMLToolFormat for chatml kind" {
    const tf = forTemplate(.chatml);
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{ .name = "f", .description = "d", .parameters_json = "{}" },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
    // ChatML emits a non-empty buffer; Noop emits empty.
    try std.testing.expect(buf.items.len > 0);
}

test "forTemplate returns NoopToolFormat for non-chatml kinds" {
    inline for (.{ .llama3, .gemma, .unknown }) |kind| {
        const tf = forTemplate(kind);
        var buf: std.ArrayList(u8) = .{};
        defer buf.deinit(std.testing.allocator);
        const tools = [_]ToolDefinition{
            .{ .name = "f", .description = "d", .parameters_json = "{}" },
        };
        try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 0), buf.items.len);
    }
}
```

- [ ] **Step 2: Run, confirm chatml test fails**

```bash
zig build test 2>&1 | tail -20
```

Expected: chatml test fails (current `forTemplate` returns Noop for everything).

- [ ] **Step 3: Update forTemplate**

In `src/server/tool_format.zig`, replace the body of `forTemplate`:

```zig
pub fn forTemplate(template_kind: TemplateKind) ToolFormat {
    return switch (template_kind) {
        .chatml => chatmlToolFormat(),
        else => noopToolFormat(),
    };
}
```

- [ ] **Step 4: Update the Task 1 smoke test to free the chatml allocation**

The smoke test added in Task 1 (`forTemplate returns a usable ToolFormat for every kind`) calls `parseAssistantToolCalls("x", ...)` for every kind. After this task, chatml's parse returns owned `text_content` and an empty owned `tool_calls` slice — both leak unless freed. Replace the test body with:

```zig
test "forTemplate returns a usable ToolFormat for every kind" {
    inline for (.{ .chatml, .llama3, .gemma, .unknown }) |kind| {
        const tf = forTemplate(kind);
        const result = try tf.parseAssistantToolCalls("x", std.testing.allocator);
        // ChatMLToolFormat allocates text_content and tool_calls; Noop doesn't.
        // Both shapes are safe to free with allocator.free if owned by us.
        if (kind == .chatml) {
            std.testing.allocator.free(result.text_content);
            std.testing.allocator.free(result.tool_calls);
        }
    }
}
```

- [ ] **Step 5: Run, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: all tests pass with no memory-leak errors from the testing allocator.

- [ ] **Step 6: Commit**

```bash
git add src/server/tool_format.zig
git commit -m "tool_format: forTemplate returns ChatMLToolFormat for chatml

Also updates the per-kind smoke test to free the ChatML allocations
that were leaking once chatml started returning owned slices.
"
```

---

## Phase 5 — tokenizer.zig integration

### Task 12: Extend ChatTemplateOptions with tools and tool_format

**Files:**
- Modify: `src/model/tokenizer.zig`

This task only adds the fields. Subsequent tasks add the rendering logic that consumes them.

- [ ] **Step 1: Locate ChatTemplateOptions**

Find the existing `ChatTemplateOptions` struct in `src/model/tokenizer.zig`. Use grep:

```bash
grep -n "ChatTemplateOptions" src/model/tokenizer.zig
```

Expected output includes a definition like `pub const ChatTemplateOptions = struct { ... }`. Note the line number.

- [ ] **Step 2: Extend the struct**

Open `src/model/tokenizer.zig` and add two fields to `ChatTemplateOptions`:

```zig
const tool_format_mod = @import("../server/tool_format.zig");

pub const ChatTemplateOptions = struct {
    // ... existing fields (add_generation_prompt, enable_thinking, skip_thinking_template, etc.) ...

    /// Tool definitions to render into the system message. Empty slice = no tools.
    tools: []const tool_format_mod.ToolDefinition = &.{},
    /// The format renderer to use for tool definitions and tool result messages.
    /// If null, no tool rendering happens (silent fallback).
    tool_format: ?tool_format_mod.ToolFormat = null,
};
```

The `tool_format_mod` import goes near the top of the file with the other `@import` lines.

- [ ] **Step 3: Verify compile**

```bash
zig build 2>&1 | tail -5
```

Expected: clean compile. Existing callers of `applyChatTemplateWithOptions(..., .{ ... })` continue to work — the new fields have defaults.

- [ ] **Step 4: Run tests, confirm no regression**

```bash
zig build test 2>&1 | grep -E "tests? passed|FAIL" | head -3
```

Expected: same pass count as after Task 11.

- [ ] **Step 5: Commit**

```bash
git add src/model/tokenizer.zig
git commit -m "tokenizer: add tools and tool_format to ChatTemplateOptions

Default values mean every existing call site continues to work
unchanged. Tool rendering is wired in subsequent tasks.
"
```

---

### Task 13: Render tool definitions in ChatML system message

**Files:**
- Modify: `src/model/tokenizer.zig`

When `tools.len > 0` and a `tool_format` is provided, the chatml branch should call `renderToolDefinitions` after emitting the system message body but before the closing `<|im_end|>`.

- [ ] **Step 1: Add the failing test**

Append to the test section near the bottom of `src/model/tokenizer.zig`:

```zig
test "applyChatTemplateWithOptions chatml renders tool definitions in system message" {
    const tool_format_local = @import("../server/tool_format.zig");
    var tok = makeTestTokenizer("<|im_start|>");
    const roles = [_][]const u8{ "system", "user" };
    const contents = [_][]const u8{ "You are helpful.", "Hello" };
    const tools = [_]tool_format_local.ToolDefinition{
        .{ .name = "Bash", .description = "Run bash.", .parameters_json = "{}" },
    };
    var buf: [4096]u8 = undefined;
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{
        .tools = &tools,
        .tool_format = tool_format_local.forTemplate(.chatml),
        .add_generation_prompt = true,
    }, &buf);

    // System message contains the original content and the tool block.
    try std.testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "# Tools") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\": \"Bash\"") != null);
    // Tool block sits inside the system turn (before <|im_end|> that ends it).
    const sys_end = std.mem.indexOf(u8, result, "<|im_end|>") orelse return error.NoSysEnd;
    try std.testing.expect(std.mem.indexOf(u8, result[0..sys_end], "# Tools") != null);
}
```

(`makeTestTokenizer` is the existing helper in `main.zig`'s test section. If it's not pub, copy its body inline or move it — easier to just construct the tokenizer struct fields directly here. If it doesn't compile, replace `makeTestTokenizer("<|im_start|>")` with manual struct construction following the pattern at `src/main.zig:2205-2214`.)

- [ ] **Step 2: Run, confirm fail**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — current chatml branch doesn't emit the tool block.

- [ ] **Step 3: Update the chatml branch**

In `src/model/tokenizer.zig`, locate `applyChatTemplateWithOptions` (around line 901). Inside the `.chatml =>` branch, replace the existing `for (0..n) |i|` loop body with one that renders tools after the system message. The exact edit:

```zig
.chatml => {
    var tools_rendered = false;
    for (0..n) |i| {
        const written = std.fmt.bufPrint(buf[pos..], "<|im_start|>{s}\n{s}", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
        pos += written.len;

        // Inject tool definitions into the first system message we encounter.
        if (!tools_rendered and options.tools.len > 0 and options.tool_format != null and
            (std.mem.eql(u8, roles[i], "system") or std.mem.eql(u8, roles[i], "developer")))
        {
            // Render into a temporary heap buffer because the tool format API
            // wants ArrayList(u8). Copy result into our fixed buf.
            var tool_buf: std.ArrayList(u8) = .{};
            defer tool_buf.deinit(std.heap.page_allocator);
            try options.tool_format.?.renderToolDefinitions(options.tools, &tool_buf, std.heap.page_allocator);
            if (pos + tool_buf.items.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[pos .. pos + tool_buf.items.len], tool_buf.items);
            pos += tool_buf.items.len;
            tools_rendered = true;
        }

        const close = std.fmt.bufPrint(buf[pos..], "<|im_end|>\n", .{}) catch return error.BufferTooSmall;
        pos += close.len;
    }
    // If no system message was provided but tools are, emit a synthetic system turn.
    if (!tools_rendered and options.tools.len > 0 and options.tool_format != null) {
        const open = std.fmt.bufPrint(buf[pos..], "<|im_start|>system", .{}) catch return error.BufferTooSmall;
        pos += open.len;
        var tool_buf: std.ArrayList(u8) = .{};
        defer tool_buf.deinit(std.heap.page_allocator);
        try options.tool_format.?.renderToolDefinitions(options.tools, &tool_buf, std.heap.page_allocator);
        if (pos + tool_buf.items.len > buf.len) return error.BufferTooSmall;
        @memcpy(buf[pos .. pos + tool_buf.items.len], tool_buf.items);
        pos += tool_buf.items.len;
        const close = std.fmt.bufPrint(buf[pos..], "<|im_end|>\n", .{}) catch return error.BufferTooSmall;
        pos += close.len;
    }

    if (options.add_generation_prompt) {
        // ... existing add_generation_prompt logic stays unchanged ...
    }
},
```

Note: this uses `std.heap.page_allocator` for the temporary tool buffer because `applyChatTemplateWithOptions` doesn't currently take an allocator parameter. If you'd prefer a stack-based fallback, allocate a `[2048]u8` and use a `std.heap.FixedBufferAllocator` — but page_allocator is simpler and the rendered tool prompt is bounded.

The "existing add_generation_prompt logic stays unchanged" placeholder refers to lines 912-920 in the unchanged file. Preserve that block exactly.

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -25
```

Expected: the new chatml-tools test passes; existing chatml tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/tokenizer.zig
git commit -m "tokenizer: render tool definitions in ChatML system message

When ChatTemplateOptions has tools + tool_format, inject the tool
prompt into the first system/developer message. If none exists, emit
a synthetic system turn just for tools.
"
```

---

### Task 14: Render tool result messages

**Files:**
- Modify: `src/model/tokenizer.zig`

`role: "tool"` messages should be rendered via `tool_format.renderToolResultMessage` rather than the standard ChatML `<|im_start|>tool` header. Per the spec, consecutive tool messages collapse into a single user turn aggregating their `<tool_response>` blocks.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "applyChatTemplateWithOptions chatml renders consecutive tool messages as one user turn" {
    const tool_format_local = @import("../server/tool_format.zig");
    var tok = makeTestTokenizer("<|im_start|>");
    const roles = [_][]const u8{ "user", "assistant", "tool", "tool", "assistant" };
    const contents = [_][]const u8{
        "go",
        "<tool_call>{\"name\":\"X\",\"arguments\":{}}</tool_call>",
        "result_a",
        "result_b",
        "done",
    };
    var buf: [4096]u8 = undefined;
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{
        .tool_format = tool_format_local.forTemplate(.chatml),
    }, &buf);

    // Two tool messages should appear inside ONE user turn.
    try std.testing.expect(std.mem.indexOf(u8, result, "<tool_response>\nresult_a\n</tool_response>\n<tool_response>\nresult_b\n</tool_response>") != null);
    // Exactly one <|im_start|>user opening that contains both tool responses.
    var count: usize = 0;
    var idx: usize = 0;
    while (std.mem.indexOfPos(u8, result, idx, "<|im_start|>user")) |found| {
        count += 1;
        idx = found + 1;
    }
    // First user turn = "go", second user turn = aggregated tool responses.
    try std.testing.expectEqual(@as(usize, 2), count);
}
```

- [ ] **Step 2: Run, confirm fail**

```bash
zig build test 2>&1 | tail -25
```

Expected: fail — current branch emits `<|im_start|>tool\nresult_a<|im_end|>` instead of aggregated user turn with tool_response wrappers.

- [ ] **Step 3: Update the chatml branch**

In `src/model/tokenizer.zig`, in the chatml branch's message loop, replace the per-message body with the version below. The change adds tool-message handling that:
- starts a new `<|im_start|>user` turn if the previous message wasn't also a tool
- calls `renderToolResultMessage` for the body
- closes the turn only when the next message is *not* also a tool

```zig
.chatml => {
    var tools_rendered = false;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const is_tool = std.mem.eql(u8, roles[i], "tool");

        if (is_tool and options.tool_format != null) {
            // Check if previous was also tool — if so, we're inside an open user turn.
            const prev_was_tool = i > 0 and std.mem.eql(u8, roles[i - 1], "tool");
            if (!prev_was_tool) {
                const open = std.fmt.bufPrint(buf[pos..], "<|im_start|>user\n", .{}) catch return error.BufferTooSmall;
                pos += open.len;
            }

            // Render the tool_response.
            var trbuf: std.ArrayList(u8) = .{};
            defer trbuf.deinit(std.heap.page_allocator);
            try options.tool_format.?.renderToolResultMessage("", contents[i], &trbuf, std.heap.page_allocator);
            if (pos + trbuf.items.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[pos .. pos + trbuf.items.len], trbuf.items);
            pos += trbuf.items.len;

            // Close the user turn only if next message is not also tool.
            const next_is_tool = (i + 1 < n) and std.mem.eql(u8, roles[i + 1], "tool");
            if (!next_is_tool) {
                const close = std.fmt.bufPrint(buf[pos..], "<|im_end|>\n", .{}) catch return error.BufferTooSmall;
                pos += close.len;
            }
            continue;
        }

        // Non-tool message: existing behavior.
        const written = std.fmt.bufPrint(buf[pos..], "<|im_start|>{s}\n{s}", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
        pos += written.len;

        if (!tools_rendered and options.tools.len > 0 and options.tool_format != null and
            (std.mem.eql(u8, roles[i], "system") or std.mem.eql(u8, roles[i], "developer")))
        {
            var tool_buf: std.ArrayList(u8) = .{};
            defer tool_buf.deinit(std.heap.page_allocator);
            try options.tool_format.?.renderToolDefinitions(options.tools, &tool_buf, std.heap.page_allocator);
            if (pos + tool_buf.items.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[pos .. pos + tool_buf.items.len], tool_buf.items);
            pos += tool_buf.items.len;
            tools_rendered = true;
        }

        const close = std.fmt.bufPrint(buf[pos..], "<|im_end|>\n", .{}) catch return error.BufferTooSmall;
        pos += close.len;
    }
    // ... synthetic system turn for tools-without-system block stays unchanged from Task 13 ...
    // ... add_generation_prompt block stays unchanged ...
},
```

The block after the loop (synthetic system turn + add_generation_prompt) is the same as Task 13 left it.

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -25
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/tokenizer.zig
git commit -m "tokenizer: render ChatML tool result messages aggregated in user turn

Consecutive role:tool messages collapse into one <|im_start|>user
turn containing multiple <tool_response> blocks, matching Qwen3's
training distribution.
"
```

---

## Phase 6 — routes.zig integration

### Task 15: Add tools and tool_choice to ChatRequestBody

**Files:**
- Modify: `src/server/routes.zig`

- [ ] **Step 1: Add the failing test**

In the test section of `src/server/routes.zig` (near the existing chat parse tests around line 3100+), append:

```zig
test "parseChatRequest accepts tools array" {
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"Bash","description":"d","parameters":{"type":"object"}}}]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.tools.len);
    try std.testing.expectEqualStrings("Bash", parsed.tools[0].name);
    try std.testing.expectEqualStrings("d", parsed.tools[0].description);
    try std.testing.expect(std.mem.indexOf(u8, parsed.tools[0].parameters_json, "\"type\":\"object\"") != null);
}

test "parseChatRequest accepts tool_choice none and auto" {
    const body_none =
        \\{"messages":[{"role":"user","content":"hi"}],"tool_choice":"none"}
    ;
    var parsed_none = try parseChatRequest(std.testing.allocator, body_none);
    defer parsed_none.deinit();
    try std.testing.expectEqual(ToolChoice.none, parsed_none.tool_choice);

    const body_auto =
        \\{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}
    ;
    var parsed_auto = try parseChatRequest(std.testing.allocator, body_auto);
    defer parsed_auto.deinit();
    try std.testing.expectEqual(ToolChoice.auto, parsed_auto.tool_choice);
}
```

- [ ] **Step 2: Run, confirm fail**

```bash
zig build test 2>&1 | tail -20
```

Expected: compile errors referencing `parsed.tools` and `ToolChoice`.

- [ ] **Step 3: Implement the fields**

In `src/server/routes.zig`, near `ChatMessage` (around line 780), add:

```zig
const tool_format_mod = @import("tool_format.zig");

const RequestToolFunction = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    parameters: std.json.Value = .null,
};

const RequestTool = struct {
    type: []const u8 = "function",
    function: RequestToolFunction = .{},
};

pub const ToolChoice = enum { auto, none };

fn parseToolChoice(value: ?std.json.Value) ToolChoice {
    const v = value orelse return .auto;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "none")) return .none;
            return .auto;
        },
        else => return .auto, // function-named choice treated as auto for now
    }
}
```

Extend `ChatRequestBody` (around line 785) with two new fields:

```zig
const ChatRequestBody = struct {
    model: []const u8 = "",
    session_id: []const u8 = "",
    messages: []const ChatMessage = &.{},
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    enable_thinking: ?bool = null,
    tools: []const RequestTool = &.{},
    tool_choice: ?std.json.Value = null,
};
```

Extend `ParsedChatRequest` (around line 796) with the converted tools and choice:

```zig
pub const ParsedChatRequest = struct {
    parsed: std.json.Parsed(ChatRequestBody),
    roles: []const []const u8,
    contents: []const []const u8,
    session_id: []const u8,
    max_tokens: u32,
    stream: bool,
    temperature: f32,
    top_p: f32,
    enable_thinking: ?bool,
    tools: []const tool_format_mod.ToolDefinition,
    tool_choice: ToolChoice,
    /// Owns parameters_json strings (re-serialized from std.json.Value).
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *ParsedChatRequest) void {
        self.arena.deinit();
        self.parsed.deinit();
    }
};
```

Note: the existing `ParsedChatRequest.deinit` only calls `parsed.deinit()`. If the existing code has a different name for the deinit method, preserve that and adapt.

In `parseChatRequest` (around line 951), after the existing parsing, add the tool conversion. Replace the final `return .{ ... }` with:

```zig
    // ... existing role/content building ...

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    const tools_in = parsed.value.tools;
    var tools_out = try arena_alloc.alloc(tool_format_mod.ToolDefinition, tools_in.len);
    for (tools_in, 0..) |t, ti| {
        // Re-serialize parameters as JSON so we can pass it as []const u8 downstream.
        var params_buf: std.ArrayList(u8) = .{};
        defer params_buf.deinit(arena_alloc);
        try std.json.Stringify.value(t.function.parameters, .{}, params_buf.writer(arena_alloc));
        tools_out[ti] = .{
            .name = t.function.name,
            .description = t.function.description,
            .parameters_json = try params_buf.toOwnedSlice(arena_alloc),
        };
    }

    return .{
        .parsed = parsed,
        .roles = roles[0..count],
        .contents = contents[0..count],
        .session_id = parsed.value.session_id,
        .max_tokens = parsed.value.max_tokens,
        .stream = parsed.value.stream,
        .temperature = parsed.value.temperature,
        .top_p = parsed.value.top_p,
        .enable_thinking = parsed.value.enable_thinking,
        .tools = tools_out,
        .tool_choice = parseToolChoice(parsed.value.tool_choice),
        .arena = arena,
    };
}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -25
```

Expected: new tests pass; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/routes.zig
git commit -m "routes: parse tools and tool_choice in chat requests

Adds RequestTool/RequestToolFunction structs, ToolChoice enum
('auto' | 'none' — named-function variant treated as auto), and
re-serializes the parameters JSON object into a []const u8 stored
in a per-request arena. ParsedChatRequest gains tools and
tool_choice fields.
"
```

---

### Task 16: Wire tool_format into applyChatTemplateWithOptions call sites

**Files:**
- Modify: `src/server/routes.zig`

The chat template is invoked from `handleChatCompletions` (around line 1593+). Pass tools and tool_format through `ChatTemplateOptions`.

- [ ] **Step 1: Locate the existing applyChatTemplate calls**

```bash
grep -n "applyChatTemplate" src/server/routes.zig
```

Note the call site(s) — typically two: one for the buffered prompt buffer setup and one for streaming.

- [ ] **Step 2: Update each call**

For each `applyChatTemplateWithOptions` call in `routes.zig`, add the new options:

```zig
const tf = tool_format_mod.forTemplate(tokenizer.detectTemplateKind());
const tools_for_template = if (parsed.tool_choice == .none) &.{} else parsed.tools;

return tokenizer.applyChatTemplateWithOptions(roles, contents, .{
    .enable_thinking = enable_thinking,
    .skip_thinking_template = skip_thinking_template,
    .tools = tools_for_template,
    .tool_format = tf,
}, buf);
```

Apply to both call sites that pass options. Existing fields like `enable_thinking` and `skip_thinking_template` are preserved; only the two new fields are added.

- [ ] **Step 3: Run tests, confirm no regression**

```bash
zig build test 2>&1 | grep -E "tests? passed|FAIL" | head -3
```

Expected: same pass count.

- [ ] **Step 4: Commit**

```bash
git add src/server/routes.zig
git commit -m "routes: pass tool_format and tools to applyChatTemplate

Selects the right ToolFormat per detected template kind. Honors
tool_choice 'none' by passing an empty tools slice — the tool prompt
is not injected even if tools were defined in the request.
"
```

---

### Task 17: Render assistant tool_calls history into prompt

**Files:**
- Modify: `src/server/routes.zig`

When the client replays the conversation including a previous assistant turn that had `tool_calls`, ZINC must inject those calls back into the prompt as `<tool_call>...</tool_call>` blocks. The current `parseChatRequest` discards this — assistant messages with empty content (and no `tool_calls`) are skipped.

This task accepts `tool_calls` on assistant messages and reconstructs them into `<tool_call>` text in the prompt history.

- [ ] **Step 1: Add the failing test**

Append:

```zig
test "parseChatRequest replays assistant tool_calls history into content" {
    const body =
        \\{"messages":[
        \\  {"role":"user","content":"q"},
        \\  {"role":"assistant","content":null,"tool_calls":[
        \\    {"id":"call_a","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"ls\"}"}}
        \\  ]},
        \\  {"role":"tool","tool_call_id":"call_a","content":"file1\nfile2\n"},
        \\  {"role":"user","content":"summarize"}
        \\]}
    ;
    var parsed = try parseChatRequest(std.testing.allocator, body);
    defer parsed.deinit();

    // Find the assistant entry's content.
    var assistant_idx: ?usize = null;
    for (parsed.roles, 0..) |r, i| {
        if (std.mem.eql(u8, r, "assistant")) {
            assistant_idx = i;
            break;
        }
    }
    const idx = assistant_idx orelse return error.NoAssistant;
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[idx], "<tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[idx], "\"name\": \"Bash\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.contents[idx], "ls") != null);
}
```

- [ ] **Step 2: Run, confirm fail**

```bash
zig build test 2>&1 | tail -20
```

Expected: fail — the assistant message has empty content, so it's currently skipped in `parseChatRequest`.

- [ ] **Step 3: Extend ChatMessage with tool_calls**

In `src/server/routes.zig`, extend `ChatMessage` (already extended for content variants) with a `tool_calls` field:

```zig
const HistoricalToolCall = struct {
    id: []const u8 = "",
    type: []const u8 = "function",
    function: HistoricalToolCallFunction = .{},
};

const HistoricalToolCallFunction = struct {
    name: []const u8 = "",
    arguments: []const u8 = "", // serialized JSON string
};

const ChatMessage = struct {
    role: []const u8 = "",
    content: Content = .{},
    tool_calls: []const HistoricalToolCall = &.{},
    tool_call_id: []const u8 = "",
};
```

In `parseChatRequest`, after the existing role/content loop, add a step to reconstruct `tool_calls` into content text. Modify the loop that builds `roles[]`/`contents[]`:

```zig
for (messages) |message| {
    if (message.role.len == 0) continue;

    // If assistant has tool_calls but empty content, render them as <tool_call> blocks.
    if (std.mem.eql(u8, message.role, "assistant") and message.content.text.len == 0 and message.tool_calls.len > 0) {
        var rendered: std.ArrayList(u8) = .{};
        for (message.tool_calls) |tc| {
            try rendered.appendSlice(arena_alloc, "<tool_call>\n{\"name\": \"");
            try rendered.appendSlice(arena_alloc, tc.function.name);
            try rendered.appendSlice(arena_alloc, "\", \"arguments\": ");
            try rendered.appendSlice(arena_alloc, tc.function.arguments);
            try rendered.appendSlice(arena_alloc, "}\n</tool_call>\n");
        }
        roles[count] = "assistant";
        contents[count] = try rendered.toOwnedSlice(arena_alloc);
        count += 1;
        continue;
    }

    if (message.content.text.len == 0) continue;

    roles[count] = message.role;
    contents[count] = if (std.mem.eql(u8, message.role, "assistant"))
        sanitizeAssistantHistoryContent(message.content.text)
    else
        message.content.text;
    count += 1;
}
```

Note: `arena_alloc` was added in Task 15. Make sure the arena is set up before this loop. If the arena is currently created later in `parseChatRequest`, move its initialization earlier so it's available here.

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -25
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/routes.zig
git commit -m "routes: replay assistant tool_calls history as <tool_call> text

Multi-turn agentic flows require the assistant's prior tool calls to
appear in the prompt so the model maintains context. Render them back
into the assistant message's content text using the ChatML
<tool_call>{...}</tool_call> format.
"
```

---

### Task 18: Non-streaming response — parse tool_calls from final output

**Files:**
- Modify: `src/server/routes.zig`

For non-streaming responses, after the model finishes generating, call `parseAssistantToolCalls` and emit the structured `tool_calls` field instead of (or alongside) `content`.

- [ ] **Step 1: Locate the response builder**

```bash
grep -n "finish_reason\|\"choices\"" src/server/routes.zig | head -10
```

Note the line where the final non-streaming response is constructed.

- [ ] **Step 2: Add the failing test**

This test exercises end-to-end non-streaming response building. Since constructing a full inference engine is heavy, the test focuses on the response-shape function. If your codebase already exposes a helper that builds the response JSON from `(content, tool_calls, finish_reason)`, test that. Otherwise, this becomes an integration test only and can be deferred to manual verification at Task 20.

If a helper exists, write a unit test:

```zig
test "buildChatResponseJson includes tool_calls when present" {
    const tools = [_]tool_format_mod.ToolCall{
        .{ .id = "call_0", .name = "Bash", .arguments_json = "{\"cmd\":\"ls\"}" },
    };
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(std.testing.allocator);
    try buildChatResponseJson(.{
        .content = "",
        .tool_calls = &tools,
        .finish_reason = "tool_calls",
    }, &buf, std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"tool_calls\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"call_0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"finish_reason\":\"tool_calls\"") != null);
}
```

If `buildChatResponseJson` doesn't exist, skip the unit test and rely on Task 20's E2E verification — note that explicitly in the commit.

- [ ] **Step 3: Update the response builder**

Locate the JSON serialization for the chat completion response (search for `\"choices\"` in `routes.zig`). After the model output is collected:

```zig
// After model output is fully buffered in `output_text`:
const tool_format_for_resp = tool_format_mod.forTemplate(tokenizer.detectTemplateKind());
const parsed_output = try tool_format_for_resp.parseAssistantToolCalls(output_text, allocator);
defer {
    for (parsed_output.tool_calls) |c| {
        allocator.free(c.id);
        allocator.free(c.name);
        allocator.free(c.arguments_json);
    }
    allocator.free(parsed_output.tool_calls);
    allocator.free(parsed_output.text_content);
}

const finish_reason: []const u8 = if (parsed_output.tool_calls.len > 0) "tool_calls" else "stop";
// ... emit JSON with parsed_output.text_content as content and parsed_output.tool_calls serialized
//     as the OpenAI tool_calls array. If text_content is empty AND tool_calls exist, OpenAI
//     uses null for content; emit `\"content\":null` in that case.
```

The exact serialization depends on the existing JSON-builder pattern in `routes.zig`. The minimum required to be opencode-compatible:

```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "...",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "<text_content or null>",
      "tool_calls": [
        {
          "id": "call_0",
          "type": "function",
          "function": {
            "name": "...",
            "arguments": "<args_json string>"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Note: `tool_calls.function.arguments` is a JSON-encoded **string** in the OpenAI response, not a nested object. So when serializing, encode the `arguments_json` value as a JSON string (escape quotes). If your JSON writer doesn't have a string-of-JSON convenience method, manually call `std.json.Stringify.value(parsed.value.arguments_json, ...)` to produce a properly-escaped string.

- [ ] **Step 4: Run tests, confirm pass**

```bash
zig build test 2>&1 | tail -20
```

Expected: same pass count.

- [ ] **Step 5: Commit**

```bash
git add src/server/routes.zig
git commit -m "routes: emit tool_calls in non-streaming response

When the model output contains <tool_call> blocks, parse them via
ChatMLToolFormat.parseAssistantToolCalls and serialize as OpenAI-shape
tool_calls array. content is null when no text precedes the calls.
finish_reason becomes 'tool_calls'.
"
```

---

### Task 19: Streaming SSE — integrate StreamingDetector

**Files:**
- Modify: `src/server/routes.zig`

For streaming, the SSE writer wraps each generated text chunk in `detector.feed()`. The result determines whether to emit a content delta, a tool_call delta, or hold.

- [ ] **Step 1: Locate the streaming SSE writer**

```bash
grep -n "data: \\|sendStream\\|streamDelta" src/server/routes.zig | head -10
```

Find the function that writes SSE chunks to the connection.

- [ ] **Step 2: Add an integration test (manual checklist)**

Streaming integration is hard to unit-test without a full inference loop. We rely on Task 20's E2E. Document the expected behavior here as a manual checklist:

- [ ] When the model emits `Sure! I'll check.\n<tool_call>{...}</tool_call>`, the SSE stream should:
  - Send `data: {"choices":[{"delta":{"content":"Sure! I'll check.\n"}}]}` (one or more deltas).
  - Send `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"X","arguments":"<args>"}}]}}]}` (single delta).
  - Send `data: {"choices":[{"finish_reason":"tool_calls"}]}` followed by `data: [DONE]`.

- [ ] **Step 3: Implement**

In the streaming path, replace the direct `write content delta` logic with a detector-driven loop:

```zig
// At the start of the streaming response, create a detector for this request.
const tf = tool_format_mod.forTemplate(tokenizer.detectTemplateKind());
const detector = try tf.newStreamingDetector(allocator);
defer {
    detector.deinit();
    allocator.destroy(detector);
}
var any_tool_call_emitted = false;

// For each generated text chunk produced by the inference engine:
const fr = try detector.feed(chunk);
switch (fr) {
    .emit_as_content => {
        const c = detector.takeContentDelta();
        if (c.len > 0) {
            // existing SSE-write-content-delta code, with `c` as the content
            try writeContentDelta(conn, c);
        }
    },
    .hold => {
        // do nothing
    },
    .tool_call_complete => {
        // First flush any leftover content (rare but possible).
        const c = detector.takeContentDelta();
        if (c.len > 0) try writeContentDelta(conn, c);

        while (detector.takePendingToolCall()) |tc| {
            defer {
                allocator.free(tc.id);
                allocator.free(tc.name);
                allocator.free(tc.arguments_json);
            }
            try writeToolCallDelta(conn, tc, any_tool_call_emitted);
            any_tool_call_emitted = true;
        }
    },
}

// At end of generation:
const tail = detector.finalize();
if (tail.len > 0) try writeContentDelta(conn, tail);
const finish: []const u8 = if (any_tool_call_emitted) "tool_calls" else if (max_tokens_hit) "length" else "stop";
try writeFinishReason(conn, finish);
```

`writeContentDelta`, `writeToolCallDelta`, and `writeFinishReason` are helper names — adapt to the existing SSE-writer conventions in `routes.zig`. The tool_call delta JSON shape:

```json
{"choices":[{"delta":{"tool_calls":[{"index":N,"id":"call_X","type":"function","function":{"name":"...","arguments":"..."}}]}}]}
```

Where `N` is the running tool-call index (increments with each call), and `arguments` is a JSON-encoded string.

- [ ] **Step 4: Run all tests, confirm no regression**

```bash
zig build test 2>&1 | grep -E "tests? passed|FAIL" | head -3
```

Expected: same pass count. Streaming behavior is verified end-to-end in Task 20.

- [ ] **Step 5: Commit**

```bash
git add src/server/routes.zig
git commit -m "routes: route streaming output through StreamingDetector

Hybrid streaming: content streams as it arrives; <tool_call> blocks
are buffered and emitted as one tool_calls delta on close. Sets
finish_reason to 'tool_calls' when any tool call was emitted.
"
```

---

## Phase 7 — End-to-end verification

### Task 20: Manual integration test on the remote box

**Files:**
- No code changes. Manual verification.

This task drives opencode against the freshly-built ZINC server and confirms the round-trip works. Cannot be automated from this dev box.

- [ ] **Step 1: Build and deploy**

```bash
just sync-obox
ssh obox 'cd zinc && nix develop -c zig build -Doptimize=ReleaseFast'
```

- [ ] **Step 2: Start the server**

```bash
ssh obox 'cd zinc && nix develop -c bash -lc "RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -p 8080"'
```

In another terminal, verify:

```bash
curl -s http://obox:8080/v1/models | jq
```

Expected: model list returned cleanly.

- [ ] **Step 3: Sanity test — chat with a tool offered**

```bash
curl -s http://obox:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role":"user","content":"List the files in /tmp"}],
    "tools": [{
      "type":"function",
      "function":{
        "name":"Bash",
        "description":"Run a bash command and return its output.",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
      }
    }]
  }' | jq
```

Pass criteria: response contains a `tool_calls` array with one entry whose `function.name` is "Bash" and `function.arguments` parses to `{"command": "ls /tmp"}` (or close — model may use `ls -la`). `finish_reason` is `"tool_calls"`.

- [ ] **Step 4: Multi-turn — return a tool result and verify continuation**

```bash
curl -s http://obox:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role":"user","content":"List the files in /tmp"},
      {"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"ls /tmp\"}"}}]},
      {"role":"tool","tool_call_id":"call_0","content":"file1.txt\nfile2.txt\n"}
    ],
    "tools": [{"type":"function","function":{"name":"Bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
  }' | jq
```

Pass criteria: response contains `content` with a natural-language summary mentioning `file1.txt` and `file2.txt`. `finish_reason` is `"stop"`.

- [ ] **Step 5: opencode end-to-end**

Run a real opencode session against ZINC and verify a multi-step agentic task completes (e.g. "list files in /tmp, then read the first one, then summarize"). All tool calls should round-trip; no connection drops; final answer should reflect the actual file content.

- [ ] **Step 6: Streaming test**

```bash
curl -N -s http://obox:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role":"user","content":"What time is it? Use the bash tool with date."}],
    "tools": [{"type":"function","function":{"name":"Bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}],
    "stream": true
  }'
```

Pass criteria: SSE events arrive incrementally; any pre-tool-call narration ("Let me check...") appears as `delta.content` chunks; the tool call appears as a single `delta.tool_calls` chunk; final event is `[DONE]`.

- [ ] **Step 7: Record results in the spec**

Append a `## Results` section to `docs/superpowers/specs/2026-05-07-tool-calling-design.md` capturing:

- Date of verification
- Model used (Qwen3.6 35B-A3B Q4_K_XL with host-mem offload)
- Single-call test outcome
- Multi-turn test outcome
- opencode E2E outcome
- Streaming test outcome
- Any unexpected behavior worth following up on
- Verdict: clear pass / needs fixes / promote-to-feature

- [ ] **Step 8: Commit results**

```bash
git add docs/superpowers/specs/2026-05-07-tool-calling-design.md
git commit -m "spec: record tool-calling integration test results"
```

---

## Self-review notes

**Spec coverage check:**

- "tools and tool_choice accepted in the request" → Task 15
- "Tool definitions injected into the prompt for ChatML-family models using Qwen3's exact format" → Task 5
- "Multiple tool calls in one assistant turn (parallel)" → Task 3
- "Assistant output parsed for `<tool_call>` blocks, returned as OpenAI-format tool_calls" → Task 4 + Task 18
- "Tool result messages rendered back into the prompt as `<tool_response>`" → Task 6 + Task 14
- "Hybrid streaming: regular content streams as today, tool calls buffered atomically" → Tasks 7-10 + Task 19
- "finish_reason: tool_calls" → Tasks 18 + 19
- "Llama3 / Gemma fall through to NoopToolFormat" → Tasks 1 + 11

**Out-of-scope check:** none of the tasks implement non-ChatML formats, Jinja interpretation, tool repair/retry, or constrained decoding for forced tool_choice. ✓

**Type consistency:** `ToolDefinition`, `ToolCall`, `ParsedAssistantOutput`, `FeedResult`, `StreamingDetector`, `ToolFormat` defined in Task 1 are used identically in subsequent tasks. `ToolChoice` enum and `ParsedChatRequest` extensions defined in Task 15 are used in Tasks 16-19.

**Reversibility:** every commit is independent enough that `git revert` of any single one keeps the build green (with the next task's tests perhaps becoming pending — the test file is the only place a revert can break things, and the test file revert always restores the prior pass count).

**Known limitations to flag in PR description:**

- `tool_call_id` is rendered into `<tool_response>` but not validated against any prior assistant `tool_calls`. A client sending a result for a non-existent call_id will get rendered in the prompt as-is.
- Forced tool_choice (`{type: "function", function: {name: "X"}}`) is parsed but treated as `"auto"`. Honoring it requires constrained decoding.
- The Qwen3.5 verbatim tool-prompt string in `tool_format.zig` is hard-coded; if Qwen3.7 or future models change conventions, the renderer will need a per-version override.
