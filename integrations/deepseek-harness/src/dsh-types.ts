// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Package-owned declarations of the DeepSeek Harness plugin contracts this
 * integration uses.
 *
 * DeepSeek Harness publishes these shapes through its runtime type registry
 * rather than as an npm type package, so the integration mirrors only the
 * fields it reads. Every DeepSeek Harness value crosses into NeMo Relay as a
 * JSON projection; host objects (agents, sessions, signals) stay on this side.
 */

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

/* -------------------------------------------------------------------------- */
/* Content and messages                                                        */
/* -------------------------------------------------------------------------- */

export type TextBlock = { type: 'text'; text: string };
export type ReasoningBlock = { type: 'reasoning'; text: string };
export type ToolCallBlock = { type: 'tool-call'; id: string; name: string; arguments: string };
export type ImageBlock = {
  type: 'image';
  attachment: { attachmentId: string; mediaType: string; bytes: number; width: number; height: number; name?: string };
  offloaded?: true;
};
export type FileBlock = { type: 'file'; attachment: { attachmentId: string; name: string; bytes: number } };
export type ToolAdditionBlock = { type: 'tool-addition'; toolName: string };
export type ToolRemovalBlock = { type: 'tool-removal'; toolName: string };

export type ContentBlock =
  | TextBlock
  | ReasoningBlock
  | ToolCallBlock
  | ImageBlock
  | FileBlock
  | ToolAdditionBlock
  | ToolRemovalBlock;

export type ContentBlockType = ContentBlock['type'];

export type MessageSource = { kind: string; [key: string]: unknown };

type MessageBase = {
  readonly id: string;
  readonly content: readonly ContentBlock[];
  readonly source: MessageSource;
};

export type SystemMessage = MessageBase & { readonly role: 'system' };
export type DeveloperMessage = MessageBase & { readonly role: 'developer' };
export type UserMessage = MessageBase & { readonly role: 'user' };
export type AssistantMessage = MessageBase & { readonly role: 'assistant' };
export type ToolResultMessage = MessageBase & {
  readonly role: 'tool';
  readonly toolCallId: string;
  readonly isError?: boolean;
};
/** A user turn input that has not been committed as a message yet. */
export type RequestUserInput = { readonly role: 'user'; readonly content: readonly ContentBlock[] };

export type RequestMessage =
  | SystemMessage
  | DeveloperMessage
  | UserMessage
  | AssistantMessage
  | ToolResultMessage
  | RequestUserInput;

export type ToolSchema = {
  deferLoading?: true;
  name: string;
  description: string;
  parameters: Record<string, unknown>;
};

/* -------------------------------------------------------------------------- */
/* Model calls: the `llm/stream` waterfall                                     */
/* -------------------------------------------------------------------------- */

/** `GenerateOptions`: the complete provider-neutral request of one streaming model call. */
export type GenerateOptions = {
  provider: string;
  model: string;
  reasoningEffort?: string;
  messages: RequestMessage[];
  system?: string;
  tools?: ToolSchema[];
  toolHistory?: unknown;
  temperature?: number;
  maxTokens?: number;
  stop?: string[];
  signal?: AbortSignal;
  sessionId?: string;
  purpose?: 'compaction' | 'session-title';
};

export type TokenUsage = {
  inputTokens: number;
  outputTokens: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
};

export type LlmFailure = {
  readonly message: string;
  readonly code: string;
  readonly status?: number;
  readonly providerRetryAfterMs?: number;
  readonly requestId?: string;
};

export type FinishReason =
  | { kind: 'stop' }
  | { kind: 'tool-calls' }
  | { kind: 'max-tokens' }
  | { kind: 'aborted'; failure: LlmFailure }
  | { kind: 'error'; failure: LlmFailure };

export type StreamChunk =
  | { type: 'block-start'; index: number; blockType: ContentBlockType }
  | { type: 'text-delta'; index: number; text: string }
  | { type: 'reasoning-delta'; index: number; text: string }
  | { type: 'tool-call-delta'; index: number; id: string; name?: string; argumentsDelta: string }
  | { type: 'block-end'; index: number; block: ContentBlock }
  | { type: 'usage'; usage: TokenUsage }
  | { type: 'finish'; reason: FinishReason; replayState?: unknown };

export type LlmStreamListener = (
  options: GenerateOptions,
  next: () => AsyncIterable<StreamChunk>,
) => AsyncIterable<StreamChunk>;

/* -------------------------------------------------------------------------- */
/* Tool calls: `tools/pre-execute` and `tools/execute`                          */
/* -------------------------------------------------------------------------- */

export type ToolErrorInfo = { name: string; code: string; reason?: string };

export type ToolExecution = {
  readonly callId: string;
  readonly rootCallId: string;
  readonly name: string;
  readonly schema?: ToolSchema;
  readonly arguments: unknown;
  readonly agent?: Agent;
  readonly parent?: unknown;
  readonly signal: AbortSignal;
};

export type ToolExecutionSuccess = {
  readonly isError: false;
  readonly value: JsonValue;
  readonly content: ContentBlock[];
  readonly meta?: JsonValue;
  readonly additionalContexts?: unknown[];
  readonly concludesTurn?: true;
};

export type ToolExecutionFailure = {
  readonly isError: true;
  readonly error: { message: string; info?: ToolErrorInfo };
  readonly content: ContentBlock[];
  readonly meta?: JsonValue;
  readonly additionalContexts?: unknown[];
};

export type HarnessToolResult = ToolExecutionSuccess | ToolExecutionFailure;

export type PostToolDecision =
  | { kind: 'accept'; content?: ContentBlock[]; value?: never; additionalContexts?: unknown[] }
  | { kind: 'accept'; value: JsonValue; content?: never; additionalContexts?: unknown[] }
  | { kind: 'block'; feedback: ContentBlock[]; additionalContexts?: unknown[] };

export type PreToolDecision =
  | { kind: 'allow' }
  | { kind: 'deny'; reason: string; info?: ToolErrorInfo }
  | { kind: 'cancel' }
  | { kind: 'ask'; reason?: string; displayReason?: { readonly en: string; readonly [locale: string]: string } };

/* -------------------------------------------------------------------------- */
/* Agents, sessions, and plugin context                                        */
/* -------------------------------------------------------------------------- */

export type SessionHeader = {
  readonly id: string;
  readonly createdAt: number;
  readonly cwd?: string;
  readonly parentSession?: string;
  readonly origin?: 'subagent';
  readonly delegationDepth?: number;
  readonly agentPreset?: string;
};

export type SessionEvent = {
  readonly type: string;
  readonly seq: number;
  readonly time: number;
  readonly data?: unknown;
};

export type Session = {
  readonly id: string;
  readonly header: SessionHeader;
};

export type Agent = {
  readonly id: string;
  readonly session: Session;
};

export type SessionStartSource = 'startup' | 'resume' | 'clear' | 'compact';

/** `agent/pre-step` waterfall payload: the step an agent is about to run. */
export type PreStepPayload = {
  readonly agent: Agent;
  readonly messages: readonly unknown[];
  readonly turn: number;
  readonly step: number;
  readonly signal: AbortSignal;
  readonly modelSelection?: unknown;
};

/**
 * `agent/pre-step` decision. This plugin never builds one; it only passes the
 * decision of the remaining listeners through, so its fields stay opaque.
 */
export type PreStepDecision = { readonly kind: string; readonly [field: string]: unknown };

export type ListenerOptions = { global?: boolean; prepend?: boolean };

/** The subset of the DeepSeek Harness Cordis context this plugin uses. */
export type HarnessContext = {
  on(name: string, listener: (...args: any[]) => unknown, options?: ListenerOptions): () => void;
  effect(execute: () => () => void | Promise<void>, label?: string): () => void;
  inject(names: string[], callback: (ctx: HarnessContext & Record<string, any>) => void): void;
  get(name: string): unknown;
  /** Cordis logger service. Its methods read `this`; call them as methods, never detached. */
  logger?: { info?(message: string): void; warn?(message: string): void };
  /** The plugin's fiber; `entry` is its Loader row, which the config editor edits. */
  fiber?: { entry?: unknown };
};

/** Session projection unit served to Web clients (`ctx.sessionProjections`). */
export type ProjectionDefinition = {
  key: string;
  stateVersion: number;
  stateSchema: { parse(value: unknown): unknown };
  init(header: SessionHeader): unknown;
  apply(state: unknown, event: SessionEvent): unknown;
  wire?: { viewSchema: { parse(value: unknown): unknown }; view(state: unknown): unknown };
};

/** Human slash command (`ctx.commands`). */
export type CommandDefinition = {
  definitionId?: string;
  name: string;
  description: string;
  input?: { hint: string };
  /** `false` keeps the raw input out of the session's `command/run` event. */
  recordInput?: boolean;
  handler(invocation: { agent: Agent; rawInput: string }):
    | Promise<{ kind: 'success' | 'error'; text?: string }>
    | {
        kind: 'success' | 'error';
        text?: string;
      };
};
