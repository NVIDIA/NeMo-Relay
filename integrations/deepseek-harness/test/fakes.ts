// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Recording fakes for the NeMo Relay runtime and plugin host, plus helpers for
 * DeepSeek Harness host values.
 */
import type { ConfigDiagnostic, NemoRelayModules } from '../src/modules.js';
import type { Agent, HarnessToolResult, Session, StreamChunk, ToolExecution } from '../src/dsh-types.js';

type Json = unknown;

export type FakeBehavior = {
  /** Validation diagnostics, fixed or computed from the validated document. */
  diagnostics?: ConfigDiagnostic[] | ((config: any) => ConfigDiagnostic[]);
  initializeError?: string;
  llmReject?: string;
  llmEdit?: (request: Record<string, any>) => Record<string, any>;
  /** Map each chunk returned to the host; `null` drops it. */
  chunkMap?: (chunk: Record<string, any>) => Record<string, any> | null;
  /** Execution intercept answer that bypasses the host stream. */
  llmCached?: Array<Record<string, any>>;
  toolReject?: string;
  toolArgsEdit?: (args: Json) => Json;
  toolResultEdit?: (result: { result: Json; annotation?: Json }) => { result: Json; annotation?: Json };
  toolCached?: { result: Json; annotation?: Json };
};

/**
 * Where a scope, mark, or call landed: the active scope stack, and its parent as
 * Relay resolves it (an explicit parent, else the top of that stack, else the
 * stack's implicit root, recorded as `null`).
 */
export type FakePlacement = { stack: unknown; effectiveParent: unknown };

export type FakeRecord = {
  pushed: Array<
    {
      name: string;
      type: unknown;
      parent: unknown;
      metadata: any;
      input: any;
      timestamp: unknown;
      handle: FakeHandle;
    } & FakePlacement
  >;
  popped: Array<{ handle: FakeHandle; output: any; metadata: any }>;
  marks: Array<{ name: string; parent: unknown; data: any; metadata: any } & FakePlacement>;
  llm: Array<
    {
      name: string;
      request: any;
      parent: unknown;
      attributes: unknown;
      metadata: any;
      model: unknown;
      codec: boolean;
    } & FakePlacement
  >;
  llmEnd: any[];
  tools: Array<{ name: string; args: any; parent: unknown; metadata: any; toolCallId: unknown } & FakePlacement>;
  /** Scope stacks created through `createScopeStack`, in creation order. */
  stacks: unknown[];
  conditional: Array<{ name: string; args: any }>;
  /** Registered tool conditional-execution guardrails, run by managed tool calls like Relay does. */
  guardrails: Map<string, { priority: number; callback: (toolName: string, args: any) => unknown }>;
  sanitizers: Map<string, string>;
  subscribers: Map<string, (event: unknown) => void>;
  validated: unknown[];
  initialized: unknown[];
  closed: number;
  flushed: number;
};

export type FakeHandle = { uuid: string; name: string };

export function createFakeModules(behavior: FakeBehavior = {}): { modules: NemoRelayModules; record: FakeRecord } {
  let nextId = 0;
  const record: FakeRecord = {
    pushed: [],
    popped: [],
    marks: [],
    llm: [],
    llmEnd: [],
    tools: [],
    stacks: [],
    conditional: [],
    guardrails: new Map(),
    sanitizers: new Map(),
    subscribers: new Map(),
    validated: [],
    initialized: [],
    closed: 0,
    flushed: 0,
  };
  const register = (kind: string) => (name: string) => {
    record.sanitizers.set(name, kind);
  };
  const deregister = (name: string) => record.sanitizers.delete(name);
  /** Relay's conditional-execution chain: higher priority first; a string rejects the call. */
  const runGuardrails = async (toolName: string, args: Json) => {
    if (behavior.toolReject !== undefined) {
      throw new Error(`guardrail rejected: ${behavior.toolReject}`);
    }
    const chain = [...record.guardrails.values()].sort((left, right) => right.priority - left.priority);
    for (const guardrail of chain) {
      const reason = await guardrail.callback(toolName, args);
      if (typeof reason === 'string') {
        throw new Error(`guardrail rejected: ${reason}`);
      }
    }
  };

  // Open scopes per stack; `withScopeStack` activates a stack for the synchronous part of its callback.
  const defaultStack = { id: 'default-stack' };
  const open = new Map<unknown, FakeHandle[]>([[defaultStack, []]]);
  let active: unknown = defaultStack;
  const place = (parent: unknown): FakePlacement => ({
    stack: active,
    effectiveParent: parent ?? open.get(active)?.at(-1) ?? null,
  });

  const nf = {
    ScopeType: { Custom: 9 },
    createScopeStack: () => {
      const stack = { id: `stack-${nextId++}` };
      open.set(stack, []);
      record.stacks.push(stack);
      return stack;
    },
    withScopeStack: (stack: unknown, callback: () => unknown) => {
      const previous = active;
      active = stack;
      try {
        return callback();
      } finally {
        active = previous;
      }
    },
    pushScope: (
      name: string,
      type: unknown,
      parent: unknown,
      _attributes: unknown,
      _data: unknown,
      metadata: any,
      input: any,
      timestamp: unknown,
    ) => {
      const handle = { uuid: `scope-${nextId++}`, name };
      record.pushed.push({ name, type, parent, metadata, input, timestamp, handle, ...place(parent) });
      open.get(active)?.push(handle);
      return handle;
    },
    popScope: (handle: FakeHandle, output: any, _timestamp: unknown, metadata: any) => {
      record.popped.push({ handle, output, metadata });
      const scopes = open.get(active) ?? [];
      const index = scopes.indexOf(handle);
      if (index >= 0) {
        scopes.splice(index, 1);
      }
    },
    event: (name: string, parent: unknown, data: any, metadata: any) => {
      record.marks.push({ name, parent, data, metadata, ...place(parent) });
    },
    llmStreamCallExecute: async (
      name: string,
      request: any,
      func: (request: Json, signal: AbortSignal) => AsyncIterable<Json>,
      collector: ((chunk: Json) => void) | null | undefined,
      finalizer: (() => Json) | null | undefined,
      parent: unknown,
      attributes: unknown,
      _data: unknown,
      metadata: any,
      model: unknown,
      codecDecode?: unknown,
    ) => {
      record.llm.push({
        name,
        request,
        parent,
        attributes,
        metadata,
        model,
        codec: typeof codecDecode === 'function',
        ...place(parent),
      });
      if (behavior.llmReject !== undefined) {
        throw new Error(`guardrail rejected: ${behavior.llmReject}`);
      }
      const controller = new AbortController();
      let source: AsyncIterator<Json>;
      if (behavior.llmCached !== undefined) {
        source = (async function* () {
          yield* behavior.llmCached ?? [];
        })();
      } else {
        const received = behavior.llmEdit === undefined ? request : behavior.llmEdit(structuredClone(request));
        try {
          source = (await func(received, controller.signal))[Symbol.asyncIterator]();
        } catch (error) {
          throw new Error(`internal error: ${(error as Error).message}`);
        }
      }
      let ended = false;
      return {
        async next(): Promise<Json | null> {
          for (;;) {
            let step: IteratorResult<Json>;
            try {
              step = await source.next();
            } catch (error) {
              throw new Error(`internal error: ${(error as Error).message}`);
            }
            if (step.done === true) {
              if (!ended) {
                ended = true;
                record.llmEnd.push(finalizer?.());
              }
              return null;
            }
            const chunk =
              behavior.chunkMap === undefined
                ? step.value
                : behavior.chunkMap(structuredClone(step.value) as Record<string, any>);
            if (chunk === null) {
              continue;
            }
            collector?.(chunk);
            return chunk;
          }
        },
        async close(): Promise<void> {
          controller.abort();
          await source.return?.();
        },
      };
    },
    toolCallExecuteAsync: async (
      name: string,
      args: Json,
      func: (args: Json, signal: AbortSignal) => Promise<{ result: Json; annotation?: Json }>,
      parent: unknown,
      _attributes: unknown,
      _data: unknown,
      metadata: any,
      toolCallId: unknown,
    ) => {
      record.tools.push({ name, args, parent, metadata, toolCallId, ...place(parent) });
      await runGuardrails(name, args);
      if (behavior.toolCached !== undefined) {
        return behavior.toolCached;
      }
      const received = behavior.toolArgsEdit === undefined ? args : behavior.toolArgsEdit(structuredClone(args));
      let result: { result: Json; annotation?: Json };
      try {
        result = await func(received, new AbortController().signal);
      } catch (error) {
        throw new Error(`internal error: ${(error as Error).message}`);
      }
      return behavior.toolResultEdit === undefined ? result : behavior.toolResultEdit(structuredClone(result));
    },
    toolConditionalExecution: async (name: string, args: Json) => {
      record.conditional.push({ name, args });
      await runGuardrails(name, args);
    },
    registerToolConditionalExecutionGuardrail: (
      name: string,
      priority: number,
      callback: (toolName: string, args: Json) => unknown,
    ) => {
      if (record.guardrails.has(name)) {
        throw new Error(`guardrail "${name}" is already registered`);
      }
      record.guardrails.set(name, { priority, callback });
    },
    deregisterToolConditionalExecutionGuardrail: (name: string) => record.guardrails.delete(name),
    registerSubscriber: (name: string, callback: (event: unknown) => void) => {
      record.subscribers.set(name, callback);
    },
    deregisterSubscriber: (name: string) => record.subscribers.delete(name),
    flushSubscribers: async () => {
      record.flushed += 1;
    },
    registerLlmSanitizeRequestGuardrail: register('llm-request'),
    deregisterLlmSanitizeRequestGuardrail: deregister,
    registerLlmSanitizeResponseGuardrail: register('llm-response'),
    deregisterLlmSanitizeResponseGuardrail: deregister,
    registerToolSanitizeRequestGuardrail: register('tool-request'),
    deregisterToolSanitizeRequestGuardrail: deregister,
    registerToolSanitizeResponseGuardrail: register('tool-response'),
    deregisterToolSanitizeResponseGuardrail: deregister,
    registerMarkSanitizeGuardrail: register('mark'),
    deregisterMarkSanitizeGuardrail: deregister,
    OpenAIChatCodec: class {
      decode(request: Json): Json {
        return request;
      }
      encode(annotated: Json): Json {
        return annotated;
      }
      decodeResponse(response: Json): Json {
        return response;
      }
    },
  };

  const pluginHost = {
    validate: (config: unknown) => {
      record.validated.push(config);
      const diagnostics =
        typeof behavior.diagnostics === 'function' ? behavior.diagnostics(config) : (behavior.diagnostics ?? []);
      return {
        config: { diagnostics },
        dynamic_plugins: [],
        config_paths: ['/fake/plugins.toml'],
        resolved_config: config,
      };
    },
    initialize: async (config: unknown) => {
      record.initialized.push(config);
      if (behavior.initializeError !== undefined) {
        throw new Error(behavior.initializeError);
      }
      return {
        report: {
          config: { diagnostics: [] },
          dynamic_plugins: [],
          config_paths: ['/fake/plugins.toml'],
          resolved_config: config,
        },
        isActive: true,
        close: async () => {
          record.closed += 1;
        },
        [Symbol.asyncDispose]: async () => undefined,
      };
    },
  };

  return { modules: { nf, pluginHost } as unknown as NemoRelayModules, record };
}

export function fakeSession(id = 'session-1', header: Partial<Session['header']> = {}): Session {
  return { id, header: { id, createdAt: 0, ...header } };
}

export function fakeAgent(session: Session = fakeSession()): Agent {
  return { id: session.id, session };
}

export function event(type: string, data: unknown, time = 1_000, seq = 0) {
  return { type, data, time, seq };
}

export function toolExec(
  agent: Agent | undefined,
  name = 'bash',
  args: unknown = { command: 'ls' },
  callId = 'call-1',
): ToolExecution {
  return {
    callId,
    rootCallId: callId,
    name,
    arguments: args,
    ...(agent === undefined ? {} : { agent }),
    signal: new AbortController().signal,
  };
}

export function okResult(text = 'done', value: unknown = { ok: true }): HarnessToolResult {
  return { isError: false, value: value as never, content: [{ type: 'text', text }] };
}

/** A host model stream with one text block, usage, and a stop finish. */
export function textStream(text = 'Hello there'): StreamChunk[] {
  const half = Math.ceil(text.length / 2);
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: text.slice(0, half) },
    { type: 'text-delta', index: 0, text: text.slice(half) },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'usage', usage: { inputTokens: 10, outputTokens: 4, cacheReadTokens: 6 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ];
}

export function streamOf(chunks: StreamChunk[]): () => AsyncIterable<StreamChunk> {
  return () =>
    (async function* () {
      yield* chunks;
    })();
}

export async function collect(iterable: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const out: StreamChunk[] = [];
  for await (const chunk of iterable) {
    out.push(chunk);
  }
  return out;
}
