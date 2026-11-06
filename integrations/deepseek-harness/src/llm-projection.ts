// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Projection between DeepSeek Harness model calls and the OpenAI Chat
 * Completions shape that NeMo Relay's built-in `OpenAIChatCodec` understands.
 *
 * DeepSeek Harness calls are provider-neutral (`GenerateOptions` plus
 * `StreamChunk` deltas). Relay request middleware, sanitizers, and exporters
 * work on provider payloads normalized by a codec, so the integration sends a
 * JSON projection in OpenAI Chat form.
 *
 * DeepSeek Harness freezes agent requests and requires that the session log
 * reconstructs exactly what the model saw, so request edits made by Relay
 * middleware are refused instead of being applied or silently dropped. The
 * response side is fully interceptable: the host records whatever chunks the
 * listener yields.
 */
import type {
  ContentBlock,
  FinishReason,
  GenerateOptions,
  JsonValue,
  RequestMessage,
  StreamChunk,
  TokenUsage,
  ToolSchema,
} from './dsh-types.js';

type JsonRecord = { [key: string]: JsonValue };

/** Chunk field that pairs a projected chunk with the host chunk it came from. */
export const CHUNK_SEQUENCE_KEY = 'deepseek_harness_seq';

/** Thrown when Relay middleware changed a request in a way the host cannot apply. */
export class UnsupportedRewriteError extends Error {
  constructor(detail: string) {
    super(
      `NeMo Relay middleware changed the model request (${detail}), but DeepSeek Harness sends requests exactly as logged; the call was blocked`,
    );
    this.name = 'UnsupportedRewriteError';
  }
}

/* -------------------------------------------------------------------------- */
/* Requests                                                                    */
/* -------------------------------------------------------------------------- */

/** OpenAI Chat projection of one host request. */
export type RequestProjection = {
  content: JsonRecord;
};

/**
 * Project `GenerateOptions` into an OpenAI Chat Completions request body.
 *
 * Consecutive user messages become one user message whose content parts keep
 * their order. Relay records an agent's later model calls from the last user
 * message on (`retain_current_request_turn` in `crates/runtime/src/api/llm.rs`;
 * only the first call of a fresh Agent scope keeps the full history), and
 * DeepSeek Harness sends the user's prompt followed by separate runtime-context
 * and skill-reminder user messages. Without the merge, every recorded call after
 * the first would start at the last reminder and lose the prompt. The merge
 * changes no content and no order, only where one user message ends.
 */
export function projectRequest(options: GenerateOptions): RequestProjection {
  const messages: JsonRecord[] = [];
  if (typeof options.system === 'string' && options.system.length > 0) {
    messages.push({ role: 'system', content: options.system });
  }
  for (const message of options.messages) {
    const projected = projectMessage(message);
    const previous = messages.at(-1);
    if (
      projected.role === 'user' &&
      previous?.role === 'user' &&
      Array.isArray(previous.content) &&
      Array.isArray(projected.content)
    ) {
      previous.content = [...previous.content, ...projected.content];
    } else {
      messages.push(projected);
    }
  }

  const content: JsonRecord = { model: options.model, messages, stream: true };
  if (options.tools !== undefined && options.tools.length > 0) {
    content.tools = options.tools.map(projectTool);
  }
  if (options.temperature !== undefined) {
    content.temperature = options.temperature;
  }
  if (options.maxTokens !== undefined) {
    content.max_tokens = options.maxTokens;
  }
  if (options.stop !== undefined) {
    content.stop = [...options.stop];
  }
  if (options.reasoningEffort !== undefined) {
    content.reasoning_effort = options.reasoningEffort;
  }
  return { content };
}

function projectTool(tool: ToolSchema): JsonValue {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: toJson(tool.parameters),
    },
  };
}

function projectMessage(message: RequestMessage): JsonRecord {
  switch (message.role) {
    case 'assistant': {
      const text = message.content
        .filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
        .map((block) => block.text)
        .join('');
      const toolCalls = message.content
        .filter((block): block is Extract<ContentBlock, { type: 'tool-call' }> => block.type === 'tool-call')
        .map((block) => ({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: block.arguments },
        }));
      const projected: JsonRecord = { role: 'assistant', content: text.length > 0 ? text : null };
      if (toolCalls.length > 0) {
        projected.tool_calls = toolCalls;
      }
      return projected;
    }
    case 'tool':
      return { role: 'tool', tool_call_id: message.toolCallId, content: projectParts(message.content) };
    default:
      return { role: message.role, content: projectParts(message.content) };
  }
}

function projectParts(blocks: readonly ContentBlock[]): JsonValue[] {
  return blocks.map((block): JsonValue => {
    switch (block.type) {
      case 'text':
        return { type: 'text', text: block.text };
      case 'image':
        return { type: 'image_url', image_url: { url: `dsh-attachment:${block.attachment.attachmentId}` } };
      case 'file':
        return { type: 'text', text: `[file: ${block.attachment.name}]` };
      case 'tool-addition':
      case 'tool-removal':
        return { type: 'text', text: `[${block.type}: ${block.toolName}]` };
      default:
        return { type: 'text', text: `[${block.type}]` };
    }
  });
}

/** Request fields that determine what the model sees; edits to them cannot be dropped silently. */
const MODEL_VISIBLE_FIELDS = new Set([
  'model',
  'messages',
  'stream',
  'tools',
  'temperature',
  'max_tokens',
  'stop',
  'reasoning_effort',
]);

/**
 * Classify request edits made by Relay middleware.
 *
 * DeepSeek Harness has already logged the request and routed it to its
 * provider adapter, so it sends requests exactly as logged. An edit to a
 * model-visible field (for example prompt redaction or routing) therefore
 * blocks the call with {@link UnsupportedRewriteError} rather than being
 * ignored. Other body fields, such as advisory adaptive hints, have no host
 * representation; they are returned as dropped so the caller can record them.
 */
export function classifyRequestEdits(projection: RequestProjection, received: unknown): { dropped: string[] } {
  if (jsonEqual(received, projection.content)) {
    return { dropped: [] };
  }
  const next = asRecord(received);
  if (next === undefined) {
    throw new UnsupportedRewriteError('request body');
  }
  const changed = [...new Set([...Object.keys(projection.content), ...Object.keys(next)])]
    .filter((key) => !jsonEqual(projection.content[key], next[key]))
    .sort();
  const blocked = changed.filter((key) => MODEL_VISIBLE_FIELDS.has(key));
  if (blocked.length > 0) {
    throw new UnsupportedRewriteError(blocked.join(', '));
  }
  return { dropped: changed };
}

/* -------------------------------------------------------------------------- */
/* Responses                                                                   */
/* -------------------------------------------------------------------------- */

/** Project one host stream chunk into an OpenAI Chat completion chunk, or `undefined` for host framing. */
export function projectChunk(chunk: StreamChunk, seq: number): JsonRecord | undefined {
  const base = (choices: JsonValue[]): JsonRecord => ({
    object: 'chat.completion.chunk',
    [CHUNK_SEQUENCE_KEY]: seq,
    choices,
  });
  switch (chunk.type) {
    case 'text-delta':
      return base([{ index: 0, delta: { content: chunk.text } }]);
    case 'reasoning-delta':
      return base([{ index: 0, delta: { reasoning_content: chunk.text } }]);
    case 'tool-call-delta': {
      const fn: JsonRecord = { arguments: chunk.argumentsDelta };
      if (chunk.name !== undefined) {
        fn.name = chunk.name;
      }
      return base([
        { index: 0, delta: { tool_calls: [{ index: chunk.index, id: chunk.id, type: 'function', function: fn }] } },
      ]);
    }
    case 'usage':
      return { ...base([]), usage: projectUsage(chunk.usage) };
    case 'finish': {
      const projected = base([{ index: 0, delta: {}, finish_reason: finishReason(chunk.reason) }]);
      if (chunk.reason.kind === 'error' || chunk.reason.kind === 'aborted') {
        projected.error = { message: chunk.reason.failure.message, code: chunk.reason.failure.code };
      }
      return projected;
    }
    default:
      return undefined;
  }
}

export function projectUsage(usage: TokenUsage): JsonRecord {
  const cached = usage.cacheReadTokens ?? 0;
  const written = usage.cacheWriteTokens ?? 0;
  const prompt = usage.inputTokens + cached + written;
  const projected: JsonRecord = {
    prompt_tokens: prompt,
    completion_tokens: usage.outputTokens,
    total_tokens: usage.totalTokens ?? prompt + usage.outputTokens,
    prompt_tokens_details: { cached_tokens: cached },
  };
  if (usage.reasoningTokens !== undefined && usage.reasoningTokens > 0) {
    projected.completion_tokens_details = { reasoning_tokens: usage.reasoningTokens };
  }
  return projected;
}

function hostUsage(value: JsonValue | undefined): TokenUsage | undefined {
  const usage = asRecord(value);
  if (usage === undefined) {
    return undefined;
  }
  const prompt = numberOr(usage.prompt_tokens, 0);
  const cached = numberOr(asRecord(usage.prompt_tokens_details)?.cached_tokens, 0);
  const output = numberOr(usage.completion_tokens, 0);
  const result: TokenUsage = {
    inputTokens: Math.max(0, prompt - cached),
    outputTokens: output,
    totalTokens: numberOr(usage.total_tokens, prompt + output),
  };
  if (cached > 0) {
    result.cacheReadTokens = cached;
  }
  const reasoning = numberOr(asRecord(usage.completion_tokens_details)?.reasoning_tokens, 0);
  if (reasoning > 0) {
    result.reasoningTokens = reasoning;
  }
  return result;
}

export function finishReason(reason: FinishReason): string {
  switch (reason.kind) {
    case 'stop':
      return 'stop';
    case 'tool-calls':
      return 'tool_calls';
    case 'max-tokens':
      return 'length';
    case 'aborted':
      return 'cancelled';
    case 'error':
      return 'error';
  }
}

function hostFinishReason(value: JsonValue | undefined, error: JsonRecord | undefined): FinishReason {
  if (error !== undefined || value === 'error') {
    return {
      kind: 'error',
      failure: { message: stringOr(error?.message, 'model call failed'), code: stringOr(error?.code, 'NEMO_RELAY') },
    };
  }
  switch (value) {
    case 'tool_calls':
      return { kind: 'tool-calls' };
    case 'length':
      return { kind: 'max-tokens' };
    case 'cancelled':
      return { kind: 'aborted', failure: { message: 'cancelled', code: 'ABORTED' } };
    default:
      return { kind: 'stop' };
  }
}

/** Collects intercepted completion chunks into one OpenAI Chat completion for the LLM end event. */
export class CompletionAggregator {
  private text = '';
  private reasoning = '';
  private readonly toolCalls = new Map<number, { id: string; name: string; arguments: string }>();
  private usage: JsonValue | undefined;
  private finish: string | undefined;
  private error: JsonRecord | undefined;

  constructor(
    private readonly model: string,
    private readonly id: string,
  ) {}

  add(chunk: unknown): void {
    const record = asRecord(chunk as JsonValue);
    if (record === undefined) {
      return;
    }
    if (record.usage !== undefined && record.usage !== null) {
      this.usage = record.usage;
    }
    const error = asRecord(record.error);
    if (error !== undefined) {
      this.error = error;
    }
    const choices = Array.isArray(record.choices) ? record.choices : [];
    for (const choice of choices) {
      const choiceRecord = asRecord(choice);
      const delta = asRecord(choiceRecord?.delta);
      if (typeof delta?.content === 'string') {
        this.text += delta.content;
      }
      if (typeof delta?.reasoning_content === 'string') {
        this.reasoning += delta.reasoning_content;
      }
      for (const call of Array.isArray(delta?.tool_calls) ? delta.tool_calls : []) {
        const callRecord = asRecord(call);
        const index = numberOr(callRecord?.index, 0);
        const fn = asRecord(callRecord?.function);
        const entry = this.toolCalls.get(index) ?? { id: '', name: '', arguments: '' };
        if (typeof callRecord?.id === 'string' && callRecord.id.length > 0) {
          entry.id = callRecord.id;
        }
        if (typeof fn?.name === 'string' && fn.name.length > 0) {
          entry.name = fn.name;
        }
        if (typeof fn?.arguments === 'string') {
          entry.arguments += fn.arguments;
        }
        this.toolCalls.set(index, entry);
      }
      if (typeof choiceRecord?.finish_reason === 'string') {
        this.finish = choiceRecord.finish_reason;
      }
    }
  }

  response(): JsonRecord {
    const message: JsonRecord = { role: 'assistant', content: this.text.length > 0 ? this.text : null };
    if (this.reasoning.length > 0) {
      message.reasoning_content = this.reasoning;
    }
    if (this.toolCalls.size > 0) {
      message.tool_calls = [...this.toolCalls.entries()]
        .sort(([left], [right]) => left - right)
        .map(([, call]) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: call.arguments },
        }));
    }
    const response: JsonRecord = {
      id: this.id,
      object: 'chat.completion',
      model: this.model,
      choices: [{ index: 0, message, finish_reason: this.finish ?? (this.toolCalls.size > 0 ? 'tool_calls' : 'stop') }],
    };
    if (this.usage !== undefined) {
      response.usage = this.usage;
    }
    if (this.error !== undefined) {
      response.error = this.error;
    }
    return response;
  }
}

type DeltaChunk = Extract<StreamChunk, { type: 'text-delta' | 'reasoning-delta' | 'tool-call-delta' }>;

function isDelta(chunk: StreamChunk): chunk is DeltaChunk {
  return chunk.type === 'text-delta' || chunk.type === 'reasoning-delta' || chunk.type === 'tool-call-delta';
}

function deltaText(chunk: DeltaChunk): string {
  return chunk.type === 'tool-call-delta' ? chunk.argumentsDelta : chunk.text;
}

/**
 * Rebuilds host stream chunks from intercepted completion chunks.
 *
 * Host chunks wait in arrival order until Relay returns their projection, so
 * stream-level intercepts can rewrite or drop deltas, and chunks produced
 * entirely by an execution intercept (for example a cached response) become a
 * well-formed host block sequence. A block whose deltas changed ends with the
 * text that was actually delivered.
 */
export class ChunkBridge {
  private readonly pending: Array<{ chunk: StreamChunk; projection: JsonRecord | undefined; seq: number | undefined }> =
    [];
  private readonly delivered = new Map<number, string>();
  private readonly changed = new Set<number>();
  private readonly synthesized = new Map<
    string,
    { index: number; type: 'text' | 'reasoning' | 'tool-call'; id: string; name: string; text: string }
  >();
  private nextSeq = 0;
  private maxHostIndex = -1;
  private nextSyntheticIndex = 0;
  terminalDelivered = false;

  /** Queue one host chunk; returns its Relay projection, if any. */
  accept(chunk: StreamChunk): JsonRecord | undefined {
    if ('index' in chunk) {
      this.maxHostIndex = Math.max(this.maxHostIndex, chunk.index);
    }
    const seq = this.nextSeq++;
    const projection = projectChunk(chunk, seq);
    this.pending.push({ chunk, projection, seq: projection === undefined ? undefined : seq });
    return projection;
  }

  /** Host chunks released by one chunk returned from the Relay stream. */
  release(received: unknown): StreamChunk[] {
    const record = asRecord(received as JsonValue);
    const seq = record?.[CHUNK_SEQUENCE_KEY];
    if (record === undefined || typeof seq !== 'number' || !this.pending.some((entry) => entry.seq === seq)) {
      return this.mark(this.synthesize(record));
    }
    const released: StreamChunk[] = [];
    while (this.pending.length > 0) {
      const entry = this.pending.shift();
      if (entry === undefined) {
        break;
      }
      if (entry.seq === seq) {
        released.push(
          this.deliver(jsonEqual(entry.projection, record) ? entry.chunk : this.rewrite(entry.chunk, record)),
        );
        break;
      }
      // Earlier framing chunks pass through; earlier projected chunks were dropped by middleware.
      this.skip(entry, released, false);
    }
    return this.mark(released);
  }

  /**
   * Host chunks still queued when the Relay stream ends. After a normal end, projected
   * chunks Relay never returned were dropped by middleware; after a failure, they are
   * released unchanged because Relay never processed them.
   */
  drain(mode: 'completed' | 'failed'): StreamChunk[] {
    const released: StreamChunk[] = [];
    for (const entry of this.pending.splice(0)) {
      this.skip(entry, released, mode === 'failed');
    }
    if (this.synthesized.size > 0 && !this.terminalDelivered) {
      released.push(...this.closeSynthesized(), { type: 'finish', reason: { kind: 'stop' } });
    }
    return this.mark(released);
  }

  private skip(entry: { chunk: StreamChunk; seq: number | undefined }, out: StreamChunk[], keep: boolean): void {
    if (entry.seq === undefined || keep || !isDelta(entry.chunk)) {
      // Framing, usage, and finish chunks keep the host stream protocol intact.
      out.push(this.deliver(entry.chunk));
      return;
    }
    this.changed.add(entry.chunk.index);
  }

  private mark(chunks: StreamChunk[]): StreamChunk[] {
    if (chunks.some((chunk) => chunk.type === 'finish')) {
      this.terminalDelivered = true;
    }
    return chunks;
  }

  /** Track delivered delta text and fix block ends whose deltas changed. */
  private deliver(chunk: StreamChunk): StreamChunk {
    if (isDelta(chunk)) {
      this.delivered.set(chunk.index, (this.delivered.get(chunk.index) ?? '') + deltaText(chunk));
      return chunk;
    }
    if (chunk.type !== 'block-end') {
      return chunk;
    }
    const text = this.delivered.get(chunk.index) ?? '';
    this.delivered.delete(chunk.index);
    if (!this.changed.delete(chunk.index)) {
      return chunk;
    }
    if (chunk.block.type === 'text' || chunk.block.type === 'reasoning') {
      return { ...chunk, block: { ...chunk.block, text } };
    }
    if (chunk.block.type === 'tool-call') {
      return { ...chunk, block: { ...chunk.block, arguments: text } };
    }
    return chunk;
  }

  /** Apply intercepted delta content to a host chunk. */
  private rewrite(chunk: StreamChunk, record: JsonRecord): StreamChunk {
    const delta = asRecord(asRecord(Array.isArray(record.choices) ? record.choices[0] : undefined)?.delta);
    if (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta') {
      const key = chunk.type === 'text-delta' ? 'content' : 'reasoning_content';
      this.changed.add(chunk.index);
      return { ...chunk, text: typeof delta?.[key] === 'string' ? (delta[key] as string) : '' };
    }
    if (chunk.type === 'tool-call-delta') {
      const call = asRecord(Array.isArray(delta?.tool_calls) ? delta.tool_calls[0] : undefined);
      const args = asRecord(call?.function)?.arguments;
      this.changed.add(chunk.index);
      return { ...chunk, argumentsDelta: typeof args === 'string' ? args : '' };
    }
    if (chunk.type === 'usage') {
      return { type: 'usage', usage: hostUsage(record.usage) ?? chunk.usage };
    }
    return chunk;
  }

  /** Turn a completion chunk that has no host origin into host block chunks. */
  private synthesize(record: JsonRecord | undefined): StreamChunk[] {
    if (record === undefined) {
      return [];
    }
    const out: StreamChunk[] = [];
    const choice = asRecord(Array.isArray(record.choices) ? record.choices[0] : undefined);
    const delta = asRecord(choice?.delta);
    if (typeof delta?.reasoning_content === 'string' && delta.reasoning_content.length > 0) {
      out.push(...this.syntheticDelta('reasoning', 'reasoning', delta.reasoning_content));
    }
    if (typeof delta?.content === 'string' && delta.content.length > 0) {
      out.push(...this.syntheticDelta('text', 'text', delta.content));
    }
    for (const call of Array.isArray(delta?.tool_calls) ? delta.tool_calls : []) {
      const callRecord = asRecord(call);
      const fn = asRecord(callRecord?.function);
      const block = this.syntheticBlock(`tool-call:${numberOr(callRecord?.index, 0)}`, 'tool-call', out);
      if (typeof callRecord?.id === 'string' && callRecord.id.length > 0) {
        block.id = callRecord.id;
      }
      if (typeof fn?.name === 'string' && fn.name.length > 0) {
        block.name = fn.name;
      }
      const args = typeof fn?.arguments === 'string' ? fn.arguments : '';
      block.text += args;
      const chunk: StreamChunk = { type: 'tool-call-delta', index: block.index, id: block.id, argumentsDelta: args };
      if (block.name.length > 0) {
        chunk.name = block.name;
      }
      out.push(chunk);
    }
    const usage = hostUsage(record.usage);
    if (usage !== undefined) {
      out.push({ type: 'usage', usage });
    }
    if (typeof choice?.finish_reason === 'string') {
      out.push(...this.closeSynthesized(), {
        type: 'finish',
        reason: hostFinishReason(choice.finish_reason, asRecord(record.error)),
      });
    }
    return out;
  }

  private syntheticDelta(key: string, type: 'text' | 'reasoning', text: string): StreamChunk[] {
    const out: StreamChunk[] = [];
    const block = this.syntheticBlock(key, type, out);
    block.text += text;
    out.push(
      type === 'text'
        ? { type: 'text-delta', index: block.index, text }
        : { type: 'reasoning-delta', index: block.index, text },
    );
    return out;
  }

  private syntheticBlock(
    key: string,
    type: 'text' | 'reasoning' | 'tool-call',
    out: StreamChunk[],
  ): { index: number; type: 'text' | 'reasoning' | 'tool-call'; id: string; name: string; text: string } {
    let block = this.synthesized.get(key);
    if (block === undefined) {
      const index = Math.max(this.maxHostIndex + 1, this.nextSyntheticIndex);
      this.nextSyntheticIndex = index + 1;
      block = { index, type, id: `relay-call-${index}`, name: '', text: '' };
      this.synthesized.set(key, block);
      out.push({ type: 'block-start', index, blockType: type });
    }
    return block;
  }

  private closeSynthesized(): StreamChunk[] {
    const out: StreamChunk[] = [];
    for (const block of this.synthesized.values()) {
      const content: ContentBlock =
        block.type === 'tool-call'
          ? { type: 'tool-call', id: block.id, name: block.name, arguments: block.text }
          : { type: block.type, text: block.text };
      out.push({ type: 'block-end', index: block.index, block: content });
    }
    this.synthesized.clear();
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/* JSON helpers                                                                */
/* -------------------------------------------------------------------------- */

/** Detach a host value into plain JSON (drops functions, undefined, and cycles' non-JSON parts). */
export function toJson(value: unknown): JsonValue {
  if (value === undefined) {
    return null;
  }
  try {
    return JSON.parse(JSON.stringify(value)) as JsonValue;
  } catch {
    return null;
  }
}

/** Structural JSON equality, treating a missing key and an `undefined` value alike. */
export function jsonEqual(left: unknown, right: unknown): boolean {
  if (left === right) {
    return true;
  }
  if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object') {
    return left === right || (left === undefined && right === null) || (left === null && right === undefined);
  }
  if (Array.isArray(left) !== Array.isArray(right)) {
    return false;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((entry, index) => jsonEqual(entry, right[index]));
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const keys = new Set([...Object.keys(leftRecord), ...Object.keys(rightRecord)]);
  for (const key of keys) {
    if (!jsonEqual(leftRecord[key], rightRecord[key])) {
      return false;
    }
  }
  return true;
}

export function asRecord(value: JsonValue | undefined | unknown): JsonRecord | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as JsonRecord) : undefined;
}

function numberOr(value: JsonValue | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function stringOr(value: JsonValue | undefined, fallback: string): string {
  return typeof value === 'string' && value.length > 0 ? value : fallback;
}
