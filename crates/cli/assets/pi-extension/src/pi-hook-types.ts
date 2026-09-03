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
 * Unlike OpenClaw, pi *does* publish these declarations -- `ToolCallEvent`,
 * `ToolCallEventResult`, `ExtensionAPI` and `ProviderConfig` all come out of
 * `@earendil-works/pi-coding-agent`'s package root. Importing them is declined on
 * cost, not availability. That package is the entire agent -- TUI, provider
 * stack, a wasm image codec -- and it ships an `npm-shrinkwrap.json` pinning
 * ~140 further packages, every one of which would land in this repository's
 * lockfile and in the Node license inventory the license-diff job walks, to type
 * a file that erases to nothing at runtime. The shapes below are also
 * deliberately *wider* than pi's: `toolName: string` with
 * `input: Record<string, unknown>` accepts every member of pi's per-tool
 * `ToolCallEvent` union, which is what lets `src/argument-transform.ts` write
 * gateway-supplied keys without narrowing per tool.
 *
 * The version and commit above are the contract that keeps this in step. On a pi
 * bump, re-read that file and re-run `just test-pi`. Re-verify these signatures
 * before relying on them; a silent shape change would show up as missing spans,
 * not a type error.
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
 * Fired *before* context compaction, and cancellable.
 *
 * `willRetry` is the one place pi tells an extension that a re-entry is coming:
 * the extension-facing `agent_end` carries no such marker. `preparation` also
 * carries the pre-compaction token count and whether the cut lands mid-turn.
 *
 * A handler must return `undefined` here. Returning an object is how pi's API
 * spells "cancel this compaction, or replace its result", so an accidental
 * return value from an observability hook would change pi's behavior.
 */
export type SessionBeforeCompactEvent = {
  type: 'session_before_compact';
  reason: 'manual' | 'threshold' | 'overflow';
  willRetry: boolean;
  preparation?: { tokensBefore?: number; isSplitTurn?: boolean };
};

/** Fired after context compaction has actually happened. Not cancellable. */
export type SessionCompactEvent = {
  type: 'session_compact';
  reason: 'manual' | 'threshold' | 'overflow';
  willRetry: boolean;
  fromExtension: boolean;
  compactionEntry?: { tokensBefore?: number };
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

/**
 * Returning `{block: true}` short-circuits the remaining `tool_call` handlers.
 *
 * Nothing else does. A truthy result without `block` is *retained* but does not
 * stop iteration, so a handler that runs after this one still sees -- and can
 * still mutate -- the same `input` object, with no re-validation before it
 * executes. Loading first protects against being pre-empted; it does not make the
 * verdict final.
 */
export type ToolCallEventResult = {
  block?: boolean;
  reason?: string;
};

/**
 * Fired when the user runs a shell command inline with the `!` or `!!` prefix.
 *
 * This path never reaches the tool registry, so `tool_call` does not see it and
 * none of the tool gating applies. `excludeFromContext` is the `!!` form, which
 * keeps the command and its output out of the model's context -- including the
 * output of a refusal.
 *
 * It fires in pi's interactive TUI and in RPC mode's `bash` command; there is no
 * bang prefix in headless `-p` mode, which has no input loop to type it into.
 */
export type UserBashEvent = {
  type: 'user_bash';
  command: string;
  excludeFromContext: boolean;
  cwd: string;
};

/**
 * Returning `result` makes pi skip execution and record that result as if the
 * command had run; returning `operations` replaces the execution backend but
 * keeps the original command. There is no block-and-reason form, so a refusal
 * is a synthetic failed result -- see `src/user-bash.ts`.
 *
 * Unlike `tool_call`, `emitUserBash` wraps handlers in try/catch, so an
 * exception here fails **open**: pi logs it and runs the command. And the first
 * handler to return anything at all wins, so an earlier-loading extension can
 * preempt this one -- `pi -e` loads first, `pi install` loads last.
 */
export type UserBashEventResult = {
  result?: {
    output: string;
    exitCode: number | undefined;
    cancelled: boolean;
    truncated: boolean;
  };
};

/** A model, narrowed to the fields this extension reads. */
export type PiModel = {
  id: string;
  api: string;
  provider: string;
  baseUrl: string;
};

/** Fired when a model is selected, including the initial selection. */
export type ModelSelectEvent = {
  type: 'model_select';
  model: PiModel;
  previousModel?: PiModel;
  source?: string;
};

/** Minimal view of pi's extension context. */
export type ExtensionContext = {
  cwd: string;
  mode: string;
  hasUI: boolean;
  sessionManager: { getSessionId(): string };
  /** The active model. Undefined before one is resolved. */
  model?: PiModel;
  /**
   * The catalog, used to see a provider's *other* models before redirecting it.
   *
   * `registerProvider(name, {baseUrl})` rewrites every model of the provider, so
   * a decision taken from the active model alone silently moves its siblings.
   * Optional because a caller may not supply one; without it the check degrades
   * to per-model, which is unsound for a provider that mixes API families.
   */
  modelRegistry?: { getAll(): PiModel[] };
};

export type ExtensionHandler<TEvent, TResult = void> = (
  event: TEvent,
  ctx: ExtensionContext,
) => TResult | undefined | Promise<TResult | undefined>;

/** Minimal view of pi's `ExtensionAPI`, limited to what this extension registers. */
export type ExtensionAPI = {
  on(event: 'session_start', handler: ExtensionHandler<SessionStartEvent>): void;
  on(event: 'session_shutdown', handler: ExtensionHandler<SessionShutdownEvent>): void;
  on(event: 'session_before_compact', handler: ExtensionHandler<SessionBeforeCompactEvent>): void;
  on(event: 'session_compact', handler: ExtensionHandler<SessionCompactEvent>): void;
  on(event: 'agent_start', handler: ExtensionHandler<AgentStartEvent>): void;
  on(event: 'agent_end', handler: ExtensionHandler<AgentEndEvent>): void;
  on(event: 'agent_settled', handler: ExtensionHandler<AgentSettledEvent>): void;
  on(event: 'turn_start', handler: ExtensionHandler<TurnStartEvent>): void;
  on(event: 'turn_end', handler: ExtensionHandler<TurnEndEvent>): void;
  on(event: 'tool_execution_start', handler: ExtensionHandler<ToolExecutionStartEvent>): void;
  on(event: 'tool_execution_end', handler: ExtensionHandler<ToolExecutionEndEvent>): void;
  on(event: 'tool_call', handler: ExtensionHandler<ToolCallEvent, ToolCallEventResult>): void;
  on(event: 'user_bash', handler: ExtensionHandler<UserBashEvent, UserBashEventResult>): void;
  on(event: 'model_select', handler: ExtensionHandler<ModelSelectEvent>): void;

  /**
   * Register or override a model provider.
   *
   * With `baseUrl` and no `models`, pi rewrites the URL of every existing model
   * for that provider and preserves their API, headers and costs
   * (`core/provider-composer.ts:215`). During initial extension load the call is
   * queued and applied once the runner binds its context, so calling it from a
   * factory is safe.
   */
  registerProvider(
    name: string,
    config: { baseUrl?: string; headers?: Record<string, string> },
  ): void;
};
