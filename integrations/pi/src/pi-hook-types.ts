// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Structural mirror of the subset of pi's extension API this integration uses.
 *
 * Mirrored from pi `v0.84.0` (`a5f43bf8a`),
 * `packages/coding-agent/src/core/extensions/types.ts`. Declaring the shapes
 * locally -- the same approach `integrations/openclaw/src/openclaw-hook-types.ts`
 * takes for its host agent -- keeps this directory buildable without depending
 * on the pi package, which matters because pi ships breaking changes through
 * *minor* releases and has no major-release channel.
 *
 * Re-verify these signatures against the pinned pi version before relying on
 * them; a silent shape change would show up as missing spans, not a type error.
 */

/** Fired when an agent loop starts. Carries no run identifier. */
export type AgentStartEvent = { type: 'agent_start' };

/**
 * Fired when an agent loop ends.
 *
 * Note the absence of `willRetry`: the *public session* `agent_end` carries it,
 * the extension-facing one does not, and `auto_retry_start`/`_end` never reach
 * extensions at all. Detecting a retry here is therefore impossible; close the
 * logical run on `agent_settled` instead.
 */
export type AgentEndEvent = { type: 'agent_end'; messages: unknown[] };

/** Fired once per logical agent run, from a `finally`. */
export type AgentSettledEvent = { type: 'agent_settled' };

/** Fired at the start of each turn. `turnIndex` resets to 0 on run re-entry. */
export type TurnStartEvent = { type: 'turn_start'; turnIndex: number; timestamp: number };

export type TurnEndEvent = {
  type: 'turn_end';
  turnIndex: number;
  message: unknown;
  toolResults: unknown[];
};

export type SessionStartEvent = {
  type: 'session_start';
  reason: 'startup' | 'reload' | 'new' | 'resume' | 'fork';
  previousSessionFile?: string;
};

/**
 * Fired when the current session is torn down.
 *
 * `reason` matters: only `quit` and the session-replacement reasons mean the
 * session is actually over. `reload` tears down and rebuilds the extension
 * runtime while the session itself continues, so treating it as an end splits
 * one logical session into two traces.
 */
export type SessionShutdownEvent = {
  type: 'session_shutdown';
  reason: 'quit' | 'reload' | 'new' | 'resume' | 'fork';
  /** Destination session file when shutting down due to session replacement. */
  targetSessionFile?: string;
};

/**
 * Fired when a tool starts executing.
 *
 * Fires *before* argument validation and before the `tool_call` hook, and also
 * for calls that never execute, so a handle map keyed on this must tolerate a
 * miss. `args` are the pre-clone originals.
 */
export type ToolExecutionStartEvent = {
  type: 'tool_execution_start';
  toolCallId: string;
  toolName: string;
  args: unknown;
};

export type ToolExecutionEndEvent = {
  type: 'tool_execution_end';
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError: boolean;
};

/**
 * Fired before a tool executes; the only pi hook that can block.
 *
 * `input` is mutable -- mutating it in place patches the arguments, later
 * `tool_call` handlers see earlier mutations, and no re-validation happens
 * afterwards.
 */
export type ToolCallEvent = {
  type: 'tool_call';
  toolCallId: string;
  toolName: string;
  input: Record<string, unknown>;
};

/** Returning `{block: true}` short-circuits the remaining `tool_call` handlers. */
export type ToolCallEventResult = {
  block?: boolean;
  reason?: string;
};

/** Minimal view of pi's extension context. */
export type ExtensionContext = {
  cwd: string;
  mode: string;
  hasUI: boolean;
  sessionManager: { getSessionId(): string };
};

export type ExtensionHandler<TEvent, TResult = void> = (
  event: TEvent,
  ctx: ExtensionContext,
) => TResult | undefined | Promise<TResult | undefined>;

/** Minimal view of pi's `ExtensionAPI`, limited to what this extension registers. */
export type ExtensionAPI = {
  on(event: 'session_start', handler: ExtensionHandler<SessionStartEvent>): void;
  on(event: 'session_shutdown', handler: ExtensionHandler<SessionShutdownEvent>): void;
  on(event: 'agent_start', handler: ExtensionHandler<AgentStartEvent>): void;
  on(event: 'agent_end', handler: ExtensionHandler<AgentEndEvent>): void;
  on(event: 'agent_settled', handler: ExtensionHandler<AgentSettledEvent>): void;
  on(event: 'turn_start', handler: ExtensionHandler<TurnStartEvent>): void;
  on(event: 'turn_end', handler: ExtensionHandler<TurnEndEvent>): void;
  on(event: 'tool_execution_start', handler: ExtensionHandler<ToolExecutionStartEvent>): void;
  on(event: 'tool_execution_end', handler: ExtensionHandler<ToolExecutionEndEvent>): void;
  on(event: 'tool_call', handler: ExtensionHandler<ToolCallEvent, ToolCallEventResult>): void;
};
