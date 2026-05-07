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
    inline for (.{ .chatml, .llama3, .gemma, .openai_moe, .generic }) |kind| {
        const tf = forTemplate(kind);
        // Smoke test: the returned ToolFormat's vtable methods are callable.
        const result = try tf.parseAssistantToolCalls("x", std.testing.allocator);
        _ = result;
    }
}
