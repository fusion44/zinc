//! Pluggable tool-calling format dispatch for chat completions.
//! @section Tool Calling
//! ChatMLToolFormat handles Qwen3-family models. NoopToolFormat is the
//! silent fallback for any other template kind.
const std = @import("std");
const TemplateKind = @import("../model/tokenizer.zig").Tokenizer.TemplateKind;

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
        // If a previously-parsed call is waiting, announce it before processing new bytes.
        if (self.pending_calls.items.len > 0) return .tool_call_complete;

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
                const parsed_json = std.json.parseFromSlice(std.json.Value, self.allocator, inner, .{}) catch {
                    // Malformed: emit the entire block as content.
                    log.warn("malformed JSON inside <tool_call>; emitting as content", .{});
                    const after_close = close_at + tool_call_close.len;
                    try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..after_close]);
                    std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                    self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);
                    if (self.content_pending.items.len > 0) return .emit_as_content;
                    return .hold;
                };
                defer parsed_json.deinit();

                const obj = switch (parsed_json.value) {
                    .object => |o| o,
                    else => {
                        log.warn("tool_call JSON is not an object; emitting as content", .{});
                        const after_close = close_at + tool_call_close.len;
                        try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..after_close]);
                        std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                        self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);
                        if (self.content_pending.items.len > 0) return .emit_as_content;
                        return .hold;
                    },
                };

                const name = blk: {
                    if (obj.get("name")) |v| {
                        if (v == .string) break :blk v.string;
                    }
                    break :blk "";
                };

                const empty_obj = std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) };
                const args_val = obj.get("arguments") orelse empty_obj;
                const args_str = try std.json.Stringify.valueAlloc(self.allocator, args_val, .{});
                errdefer self.allocator.free(args_str);

                const id = try std.fmt.allocPrint(self.allocator, "call_{d}", .{self.next_id});
                errdefer self.allocator.free(id);
                self.next_id += 1;

                const name_owned = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(name_owned);

                try self.pending_calls.append(self.allocator, .{
                    .id = id,
                    .name = name_owned,
                    .arguments_json = args_str,
                });

                // Strip the consumed block from hold_buf.
                const after_close = close_at + tool_call_close.len;
                std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);

                // If there is pending content from before the tag, emit it first.
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

    /// Drain pending content bytes. Returns a slice owned by the detector —
    /// consume before next feed call.
    pub fn takeContentDelta(self: *StreamingDetector) []const u8 {
        const out = self.content_pending.items;
        self.content_pending = .{};
        return out;
    }

    pub fn takePendingToolCall(self: *StreamingDetector) ?ToolCall {
        if (self.pending_calls.items.len == 0) return null;
        return self.pending_calls.orderedRemove(0);
    }

    /// Called at end of stream. Returns any held bytes (as content).
    pub fn finalize(self: *StreamingDetector) []const u8 {
        const held = self.hold_buf.items;
        if (held.len == 0) return self.content_pending.items;
        self.content_pending.appendSlice(self.allocator, held) catch return self.content_pending.items;
        self.hold_buf.clearRetainingCapacity();
        return self.content_pending.items;
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
    return switch (template_kind) {
        .chatml => chatmlToolFormat(),
        else => noopToolFormat(),
    };
}

// ============================================================
// ChatMLToolFormat — Qwen3-family tool calling.
// ============================================================

const tool_call_open = "<tool_call>";
const tool_call_close = "</tool_call>";

fn chatml_render_defs(
    _: *anyopaque,
    tools: []const ToolDefinition,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    if (tools.len == 0) return;

    // Verbatim from Qwen3.5's published chat_template.
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

fn chatml_render_result(
    _: *anyopaque,
    tool_call_id: []const u8,
    content: []const u8,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    _ = tool_call_id; // Qwen3 doesn't render the call id
    try buf.appendSlice(allocator, "<tool_response>\n");
    try buf.appendSlice(allocator, content);
    if (content.len == 0 or content[content.len - 1] != '\n') {
        try buf.appendSlice(allocator, "\n");
    }
    try buf.appendSlice(allocator, "</tool_response>\n");
}

fn chatml_fallback_block(
    text_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    inner: []const u8,
) !void {
    try text_buf.appendSlice(allocator, tool_call_open);
    try text_buf.appendSlice(allocator, "\n");
    try text_buf.appendSlice(allocator, inner);
    try text_buf.appendSlice(allocator, "\n");
    try text_buf.appendSlice(allocator, tool_call_close);
}

fn chatml_parse_calls(
    _: *anyopaque,
    model_output: []const u8,
    allocator: std.mem.Allocator,
) !ParsedAssistantOutput {
    var calls: std.ArrayList(ToolCall) = .{};
    errdefer {
        for (calls.items) |c| {
            allocator.free(c.id);
            allocator.free(c.name);
            allocator.free(c.arguments_json);
        }
        calls.deinit(allocator);
    }
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

        // Parse the inner JSON using std.json.parseFromSlice with a dynamic Value.
        const parsed_result = std.json.parseFromSlice(std.json.Value, allocator, inner, .{}) catch {
            // Malformed JSON: include the entire block (tags + content) in text_buf.
            try chatml_fallback_block(&text_buf, allocator, inner);
            search_from = close_at + tool_call_close.len;
            continue;
        };
        defer parsed_result.deinit();

        // Extract "name" and "arguments" fields.
        const obj = switch (parsed_result.value) {
            .object => |o| o,
            else => {
                try chatml_fallback_block(&text_buf, allocator, inner);
                search_from = close_at + tool_call_close.len;
                continue;
            },
        };

        const name_val = obj.get("name") orelse {
            try chatml_fallback_block(&text_buf, allocator, inner);
            search_from = close_at + tool_call_close.len;
            continue;
        };
        const name_str = switch (name_val) {
            .string => |s| s,
            else => {
                try chatml_fallback_block(&text_buf, allocator, inner);
                search_from = close_at + tool_call_close.len;
                continue;
            },
        };

        // Re-serialize the arguments JSON using valueAlloc.
        const empty_obj = std.json.Value{ .object = std.json.ObjectMap.init(allocator) };
        const args_val = obj.get("arguments") orelse empty_obj;
        const args_json = try std.json.Stringify.valueAlloc(allocator, args_val, .{});
        errdefer allocator.free(args_json);

        // Build the id string.
        const id_str = try std.fmt.allocPrint(allocator, "call_{d}", .{next_id});
        next_id += 1;
        errdefer allocator.free(id_str);

        const name_owned = try allocator.dupe(u8, name_str);
        errdefer allocator.free(name_owned);

        try calls.append(allocator, .{
            .id = id_str,
            .name = name_owned,
            .arguments_json = args_json,
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

fn chatml_new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
    const d = try allocator.create(StreamingDetector);
    d.* = .{ .allocator = allocator };
    return d;
}

const vtable_chatml = ToolFormat.VTable{
    .renderToolDefinitions = chatml_render_defs,
    .renderToolResultMessage = chatml_render_result,
    .parseAssistantToolCalls = chatml_parse_calls,
    .newStreamingDetector = chatml_new_detector,
};

var chatml_instance: u8 = 0;

pub fn chatmlToolFormat() ToolFormat {
    return .{
        .ptr = @ptrCast(&chatml_instance),
        .vtable = &vtable_chatml,
    };
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
    inline for (.{ .chatml, .llama3, .gemma, .openai_moe, .generic }) |kind| {
        const tf = forTemplate(kind);
        const result = try tf.parseAssistantToolCalls("x", std.testing.allocator);
        if (kind == .chatml) {
            std.testing.allocator.free(result.text_content);
            std.testing.allocator.free(result.tool_calls);
        }
    }
}

test "ChatMLToolFormat.parseAssistantToolCalls extracts a single tool call" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me check that.
        \\<tool_call>
        \\{"name": "Bash", "arguments": {"command": "ls /"}}
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

    try std.testing.expectEqualStrings("Let me check that.\n", parsed.text_content);
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expectEqualStrings("call_0", parsed.tool_calls[0].id);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expectEqualStrings(
        \\{"command":"ls /"}
    , parsed.tool_calls[0].arguments_json);
}

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
        std.testing.allocator.free(parsed.text_content);
    }

    try std.testing.expectEqual(@as(usize, 2), parsed.tool_calls.len);
    try std.testing.expectEqualStrings("call_0", parsed.tool_calls[0].id);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expectEqualStrings("call_1", parsed.tool_calls[1].id);
    try std.testing.expectEqualStrings("Read", parsed.tool_calls[1].name);
    try std.testing.expectEqualStrings("Let me run two commands.\n", parsed.text_content);
}

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

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "# Tools") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "</tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"name\": \"Bash\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"description\": \"Execute a bash command.\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tool_call>") != null);
}

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

test "StreamingDetector.feed emits prefix content before tool_call_complete" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    const chunk = "Sure!\n<tool_call>\n{\"name\":\"X\",\"arguments\":{}}\n</tool_call>";

    // First call: should return emit_as_content with "Sure!\n"
    const r1 = try detector.feed(chunk);
    try std.testing.expectEqual(FeedResult.emit_as_content, r1);
    const content = detector.takeContentDelta();
    try std.testing.expectEqualStrings("Sure!\n", content);

    // Second call (empty): pending_calls is non-empty so returns tool_call_complete.
    const r2 = try detector.feed("");
    try std.testing.expectEqual(FeedResult.tool_call_complete, r2);
    const call = detector.takePendingToolCall() orelse return error.NoCall;
    defer {
        std.testing.allocator.free(call.id);
        std.testing.allocator.free(call.name);
        std.testing.allocator.free(call.arguments_json);
    }
    try std.testing.expectEqualStrings("X", call.name);
    try std.testing.expectEqualStrings("call_0", call.id);
}

test "StreamingDetector flushes false-positive partial tag as content" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    // First chunk ends with "<th" — might be start of <tool_call>
    _ = try detector.feed("Hello <th");

    // Second chunk "ink>" makes "<think>" which is NOT a tool_call.
    const r2 = try detector.feed("ink>");
    try std.testing.expectEqual(FeedResult.emit_as_content, r2);

    const content = detector.takeContentDelta();
    // Verify some bytes were flushed as content (exact split may vary)
    try std.testing.expect(content.len > 0);
}

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
    inline for (.{ .llama3, .gemma, .openai_moe, .generic }) |kind| {
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
