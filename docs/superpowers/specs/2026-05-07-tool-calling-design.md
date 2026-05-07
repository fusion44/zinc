# OpenAI-compatible tool calling for ChatML-family models

Date: 2026-05-07
Status: Design (approved through Section 4 sketch)

## Goal

Make ZINC's `/v1/chat/completions` endpoint correctly handle OpenAI tool-calling semantics for ChatML-family models (Qwen3, Qwen3.5, Qwen3.6) so that agentic clients (opencode, openai-python, anthropic SDKs in compat mode) can use ZINC as a drop-in backend. Out of the box, this means: tools described in the request prompt the model in its native trained format, the model's `<tool_call>` output is parsed and returned as structured `tool_calls`, and tool result messages from the client are rendered back into the prompt for multi-turn agentic loops.

## Scope

**In scope:**

- `tools` and `tool_choice` accepted in chat-completion requests
- Tool definitions injected into the system prompt for ChatML-kind models, using Qwen3's verbatim official tool prompt format
- Multiple tool calls in one assistant turn (parallel calls)
- Assistant output parsed for `<tool_call>...</tool_call>` blocks, returned as OpenAI-format `tool_calls`
- Tool result messages (`role: "tool"`, `tool_call_id`) rendered back into the prompt as `<tool_response>...</tool_response>` blocks inside a user turn
- Hybrid streaming: regular content streams as today, tool calls are buffered atomically and emitted as one tool_calls delta
- `finish_reason: "tool_calls"` set when generation ends with tool calls
- Pluggable per-template-kind dispatch — Llama3 / Gemma / gpt-oss fall through to a `NoopToolFormat` that ignores `tools` and behaves exactly like ZINC does today

**Out of scope:**

- Llama3 / Gemma / gpt-oss tool implementations (interface ready, no concrete impls — adding one means writing one file)
- Jinja interpreter / GGUF chat_template execution at runtime
- Tool-call repair / retry on malformed model output (fall back to plain text instead)
- JSON Schema validation of `parameters` or `arguments`
- Anthropic-style nested `input_schema`
- Server-side tool execution — ZINC never runs tools, only emits `tool_calls` and consumes `tool_response`
- Stateful conversation persistence — clients replay full message history on every request, same as OpenAI
- Forced tool choice (`tool_choice: {type: "function", function: {name: "X"}}`) — parsed but treated as `"auto"`; honoring it would require constrained decoding, which is its own project

## Architecture

A new module `src/server/tool_format.zig` exposes a `ToolFormat` interface and a single concrete implementation `ChatMLToolFormat`, plus a `NoopToolFormat` for any other template kind. Routes.zig and tokenizer.zig change minimally — they call through the interface for every tool concern. Per-template dispatch is a single function `tool_format.forTemplate(template_kind)` that returns the right implementation; call sites never branch on template kind.

```
┌──────────────────────────────────────────────────────────────┐
│                        routes.zig                            │
│  - parseChatRequest (extended: tools, tool_choice)           │
│  - tool_format.forTemplate(kind) -> ToolFormat               │
│  - SSE writer consults StreamingDetector for hybrid stream   │
│  - Final response builder calls parseAssistantToolCalls      │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              src/server/tool_format.zig (NEW)                │
│                                                              │
│   pub const ToolFormat = struct { vtable, ptr }              │
│   pub const ToolDefinition / ToolCall / ParsedAssistantOutput │
│   pub const StreamingDetector                                │
│                                                              │
│   ChatMLToolFormat — Qwen3 verbatim render + parser          │
│   NoopToolFormat     — used for llama3, gemma, unknown       │
│                                                              │
│   pub fn forTemplate(kind: TemplateKind) ToolFormat          │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│           src/model/tokenizer.zig (extended)                 │
│  ChatTemplateOptions { tools, tool_format, ... }             │
│  chatml branch:                                              │
│    - calls renderToolDefinitions after system message        │
│    - calls renderToolResultMessage for role=="tool"          │
│    - other branches: tool_format is NoopToolFormat           │
└──────────────────────────────────────────────────────────────┘
```

## Components

### `src/server/tool_format.zig` (new, ~600 lines)

```zig
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    parameters_json: []const u8, // raw JSON object — schema not validated
};

pub const ToolCall = struct {
    id: []const u8,         // generated as "call_<n>"; OpenAI requires non-empty id
    name: []const u8,
    arguments_json: []const u8,
};

pub const ParsedAssistantOutput = struct {
    text_content: []const u8,        // anything outside tool_call blocks
    tool_calls: []const ToolCall,    // empty slice if none
};

pub const FeedResult = enum {
    emit_as_content,    // pass these bytes to the SSE content delta
    hold,               // buffered internally; do not emit yet
    tool_call_complete, // a complete tool_call was just parsed; routes.zig
                        //   pulls it via takePendingToolCall()
};

pub const StreamingDetector = struct {
    state: enum { normal_text, buffer_partial_tag, inside_tool_call },
    hold_buf: std.ArrayList(u8),
    parsed_calls: std.ArrayList(ToolCall),

    pub fn feed(self: *StreamingDetector, chunk: []const u8) !FeedResult;
    pub fn takePendingToolCall(self: *StreamingDetector) ?ToolCall;
    pub fn finalize(self: *StreamingDetector) []const u8; // any held bytes at EOS
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

    pub fn renderToolDefinitions(self: ToolFormat, ...) anyerror!void { ... }
    pub fn renderToolResultMessage(self: ToolFormat, ...) anyerror!void { ... }
    pub fn parseAssistantToolCalls(self: ToolFormat, ...) anyerror!ParsedAssistantOutput { ... }
    pub fn newStreamingDetector(self: ToolFormat, ...) anyerror!*StreamingDetector { ... }
};

pub fn forTemplate(template_kind: TemplateKind) ToolFormat;
```

**`ChatMLToolFormat`** — first concrete impl, in same file:

- `renderToolDefinitions`: emits Qwen3.5/3.6's verbatim system-prompt fragment, hard-coded as a Zig string literal in the implementation. Includes the `# Tools` heading, the `<tools>...</tools>` block with each tool's JSON, and the example `<tool_call>` instruction. Pulled directly from the published Qwen3.5 chat_template (Hugging Face `unsloth/Qwen3.5-35B-A3B-GGUF`).
- `renderToolResultMessage`: emits `<tool_response>\n{content}\n</tool_response>\n`. Caller is responsible for wrapping these inside a single user turn when multiple tool results need to be aggregated between assistant turns.
- `parseAssistantToolCalls`: scans for `<tool_call>...</tool_call>` pairs, parses inner JSON via `std.json.parseFromSlice` into `{name, arguments}`. Generates ids `call_0`, `call_1`, .... Text outside the blocks goes in `text_content`.
- `newStreamingDetector`: returns a state machine.

**`NoopToolFormat`** — ~30 lines:

- `renderToolDefinitions`: no-op (silently drops tools)
- `renderToolResultMessage`: appends `content` plain (best-effort fallback)
- `parseAssistantToolCalls`: returns `{text_content = output, tool_calls = []}`
- `newStreamingDetector`: pass-through (always emits as content)

### `src/server/routes.zig` extensions (~150 lines net)

- `ChatRequestBody`: add `tools: ?[]const RequestTool = null` and `tool_choice: ?std.json.Value = null` (the value can be `"auto"`, `"none"`, or `{type:"function", function:{name:"..."}}`).
- `RequestTool` struct mirrors OpenAI's `{type, function: {name, description, parameters}}` shape. `parameters` is captured as raw `std.json.Value` and serialized back to JSON when passed to the renderer.
- `parseChatRequest` returns the converted `[]ToolDefinition` slice alongside existing fields.
- `applyChatTemplateWithOptions` is called with the `ToolFormat` and `tools` slice in its options.
- The streaming SSE writer wraps every output chunk in `detector.feed(...)`. On `emit_as_content`, write the standard content delta. On `hold`, write nothing. On `tool_call_complete`, drain `takePendingToolCall()` and emit a tool-call delta.
- The final response builder calls `parseAssistantToolCalls` on accumulated output for non-streaming responses; for streaming, it tracks whether any tool_calls were emitted to set `finish_reason`.
- Adds `assistant`-with-`tool_calls`-history rendering: when an assistant message in the request has empty content but a `tool_calls` array, render those calls back into the prompt as historical `<tool_call>` blocks.

### `src/model/tokenizer.zig` extensions (~40 lines net)

- `ChatTemplateOptions`: add `tools: []const ToolDefinition = &.{}` and `tool_format: ?ToolFormat = null`.
- ChatML branch: between the system-message emission and the first user message, if `tools.len > 0` and `tool_format != null`, call `tool_format.renderToolDefinitions(tools, ...)`. The tool prompt is appended to the existing system message if one was provided, or emitted as a standalone system message if not.
- ChatML branch: add a `tool` role case that calls `tool_format.renderToolResultMessage(...)` instead of the standard `<|im_start|>tool` header. Tool messages collapse together if consecutive (one `<|im_start|>user\n<tool_response>...</tool_response><tool_response>...</tool_response>\n<|im_end|>`).

## Data flow

### Request → prompt rendering

```
HTTP body
  │
  ▼
parseChatRequest
  ├── messages[] (string + array + null content already supported)
  ├── tools[] (NEW)
  └── tool_choice (NEW; "none" suppresses tool injection)
  │
  ▼
applyChatTemplateWithOptions(roles, contents, opts)
  │
  ▼ (chatml branch only)
  for i in messages:
    if role == "system": emit <|im_start|>system\n{content}
    if first system AND tools.len > 0 AND tool_choice != "none":
      tool_format.renderToolDefinitions(tools, ...)
    close <|im_end|>
    ...
    if role == "user": emit <|im_start|>user\n{content}<|im_end|>\n
    if role == "assistant" with content: emit <|im_start|>assistant\n{content}<|im_end|>\n
    if role == "assistant" with tool_calls history:
      emit <|im_start|>assistant\n
      for each call: emit <tool_call>{json}</tool_call>\n
      emit <|im_end|>\n
    if role == "tool":
      if previous was also "tool": continue inside open user turn
      else: emit <|im_start|>user\n
      tool_format.renderToolResultMessage(tool_call_id, content, ...)
      if next is not "tool": close <|im_end|>\n
  emit <|im_start|>assistant\n
```

### Generation → streaming output (hybrid)

```
GPU produces token  ──►  tokenizer decodes  ──►  text chunk
                                                   │
                                                   ▼
                                          detector.feed(chunk)
                                                   │
                            ┌──────────────────────┼──────────────────────┐
                            ▼                      ▼                      ▼
                     emit_as_content              hold              tool_call_complete
                            │                      │                      │
                  SSE content delta         hold buf bounded       drain takePendingToolCall
                            │                to 24 bytes           SSE tool_call delta
                            └────────────┐        │                      │
                                         ▼        ▼                      ▼
                                Stream ends. Detector.finalize() returns any held bytes
                                (emit as final content delta if non-empty).
                                If any tool_calls were emitted in this response:
                                   finish_reason: "tool_calls"
                                else if max_tokens hit:
                                   finish_reason: "length"
                                else:
                                   finish_reason: "stop"
```

### Multi-turn round trip

1. Client sends `messages: [system, user]` + `tools`. ZINC renders prompt with tool definitions injected into the system message. Model generates `<tool_call>{...}</tool_call>` (possibly multiple). ZINC's detector parses each, returns response with `tool_calls: [...]` and `finish_reason: "tool_calls"`.
2. Client executes tools, sends back full history: `messages: [system, user, assistant_with_tool_calls, {role: "tool", tool_call_id: "call_0", content: "result"}, {role: "tool", tool_call_id: "call_1", content: "result"}]` + same `tools`. ZINC renders prompt: assistant turn replays the historical `<tool_call>` blocks, then a single user turn aggregates `<tool_response>` blocks for both tool results.
3. Model continues. Possibly another tool call cycle, possibly a final answer (`finish_reason: "stop"`).

ZINC never executes tools, never persists tool state. Clients are responsible for the full message history on every request.

## Error handling

All failures degrade gracefully — no server crashes, no connection drops.

| Failure | Behavior |
|---|---|
| Malformed JSON inside `<tool_call>` | Drop this tool call only. Fall back to including the raw text (including tags) as `content`. Log `warn`. Other tool calls in the same response still emit normally. |
| Unclosed `<tool_call>` (max_tokens or EOS mid-call) | At end of generation with `inside_tool_call == true`, flush buffered bytes as plain content. `finish_reason: "length"` if max_tokens, else `"stop"`. No tool_calls in the response for this orphan. Log `warn`. |
| Tool result for unknown `tool_call_id` | No validation — render the content into a `<tool_response>` block as-is. Model may produce confused output, but this is a prompt-quality issue, not a server error. |
| Bogus `tools` array shape (missing `function.name`, etc.) | Request fails at JSON parse. Return HTTP 400 with `{"error":{"message":"Invalid 'tools' array: <reason>","type":"invalid_request_error"}}`. |
| `tool_choice: {function: {name: "X"}}` for X not in `tools` | Treated as `"auto"` for this iteration. Log `warn`. Forced-call enforcement is a future enhancement. |
| `tool_choice: "none"` with `tools` present | Tools parsed but not rendered into the prompt. Model behaves as if no tools were offered. |
| Streaming detector overflow (buffer somehow exceeds open-tag length without matching) | Should be impossible by construction, but if it happens, flush all held bytes as content and reset to `normal_text`. Log `err`. |

## Testing

**Unit tests in `src/server/tool_format.zig`:**

- `parseAssistantToolCalls` — single tool call
- `parseAssistantToolCalls` — multiple parallel tool calls
- `parseAssistantToolCalls` — tool call surrounded by reasoning text (text_content extracted correctly)
- `parseAssistantToolCalls` — malformed inner JSON (graceful fallback, text_content includes the bad block raw)
- `parseAssistantToolCalls` — unclosed tool_call at end of input (treated as content)
- `renderToolDefinitions` — golden text comparison against the Qwen3.5 template's expected output for a sample 2-tool input
- `renderToolResultMessage` — exact-text output assertion
- `StreamingDetector.feed` — chunk arriving mid-tag (`<too` then `l_call>`) — held until match
- `StreamingDetector.feed` — false-positive partial tag (`<too`, then `lbar`) — flushes both as content
- `StreamingDetector.feed` — full single tool call across many small chunks
- `StreamingDetector.feed` — interleaved content + tool call + content
- `StreamingDetector.finalize` — held buffer flushed correctly at EOS
- `NoopToolFormat` — every method returns sensible no-op values

**Integration tests in `src/server/routes.zig`:**

- `parseChatRequest` accepts a `tools` array with valid OpenAI shape
- `parseChatRequest` rejects `tools` with missing `function.name` (returns parse error → 400)
- `parseChatRequest` accepts `tool_choice: "none"`, `"auto"`, and named-function variants
- `parseChatRequest` accepts an assistant message with `tool_calls` array and empty content (round-trip support)
- `parseChatRequest` accepts a `role: "tool"` message with `tool_call_id` and string content

**Integration tests in `src/model/tokenizer.zig`:**

- `applyChatTemplateWithOptions` with chatml + tools: tool prompt appended to existing system message
- `applyChatTemplateWithOptions` with chatml + tools, no system message: tool prompt becomes the system message
- `applyChatTemplateWithOptions` with chatml + tool_choice "none": tools omitted from prompt
- `applyChatTemplateWithOptions` with chatml + multiple consecutive tool messages: aggregated into one user turn
- `applyChatTemplateWithOptions` with llama3 or gemma + tools: NoopToolFormat behavior, prompt unchanged vs. no-tools case

**End-to-end manual on the remote 16 GB box:**

- Start ZINC server with Qwen3.6-35B-A3B (the offload-enabled config from the host-mem-offload branch).
- Run an opencode session against it. Verify:
  - A simple Bash-tool call (e.g. `ls /`) round-trips: model emits `<tool_call>`, opencode executes, model produces a final answer using the result.
  - A multi-step agentic task (read 3 files, summarize) completes — multi-turn flow works.
  - A parallel tool call (model emits 2 calls in one response) — opencode receives both, executes both, ZINC renders both `<tool_response>` blocks.
- Confirm no connection drops. Confirm `finish_reason` is `"tool_calls"` when tools are called and `"stop"` for plain replies.

**Acceptance criteria for "done":**

- All unit and integration tests pass under `zig build test` (counts match the new total exactly — no regressions in existing tests).
- The opencode end-to-end Bash-tool round-trip succeeds on the test box.
- Verbatim Qwen3 tool prompt matches a published Qwen3.5 chat template render for a sample tool list (golden test catches drift).
- Llama3 / Gemma model loads still pass `zig build test`, including any chat-template tests for those families.

## Risks

1. **Qwen3 tool prompt template version drift.** Qwen3.5 and Qwen3.6 may use slightly different tool prompt phrasing. Mitigation: the design hard-codes Qwen3.5's published template since both `unsloth/Qwen3.5-35B-A3B-GGUF` and `unsloth/Qwen3.6-35B-A3B-GGUF` were trained on identical tool conventions. If 3.6 diverges in a future release, a per-version override hook is a small extension.

2. **Tokenizer splits `<tool_call>` across decode boundaries.** The open tag is 11 bytes; the streaming detector buffers up to 24 bytes (safety margin) and either matches `<tool_call>` and switches to `inside_tool_call`, or flushes the held bytes as content on mismatch. ChatML tokenizers don't emit tokens larger than this in practice.

3. **Multiple tool calls in streaming mode** require careful index numbering. Each `tool_call_complete` increments the OpenAI-format `tool_calls[index]`. The detector tracks this counter internally.

4. **Aggregating consecutive tool messages.** If a client sends tool messages one at a time interleaved with assistant turns rather than batched together, the rendering may not match Qwen3's training distribution exactly. Mitigation: the renderer always groups consecutive `role: "tool"` messages into one user turn regardless of how the client structured them.

5. **`tools` array forwarded to the model verbatim** (as JSON inside the `<tools>` block). If the client sends a malicious `description` field with embedded ChatML control tokens (`<|im_end|>`), the model could be jailbroken out of its turn. Mitigation: strip ChatML control tokens from string fields before rendering. (Note: this is also a server-of-LLM concern in general, not specific to this feature.)

## Acceptance and promotion

This design supersedes the brief tool-calling note in the host-mem-offload research spec. After implementation:

- The `research/host-mem-offload` branch's chat-content-shape fix (`7c5f05a`) is a prerequisite and must already be on `main` before this work starts.
- Default behavior on Qwen3 family becomes "tools work" — no flag required to enable.
- `tools` on non-ChatML models continues to be silently dropped (current behavior), so this change cannot regress any existing user.

The implementation is large enough (~600 lines in the new `tool_format.zig`, ~150 lines integration in `routes.zig`, ~40 lines in `tokenizer.zig`, plus tests — roughly 800-1000 lines depending on test coverage density) that it warrants its own branch and its own implementation plan.
