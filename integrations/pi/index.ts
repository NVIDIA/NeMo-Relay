// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NeMo Relay extension for the pi coding agent.
 *
 * pi has no native hook-configuration file and its external stream is
 * observation-only, so hook calls must originate inside an extension. This one
 * is a thin HTTP client to the NeMo Relay CLI gateway: it forwards pi's
 * lifecycle to `/hooks/pi`, and gates tool calls on the gateway's verdict.
 *
 * **Governance.** `tool_call` is the only pi hook that can block, and for
 * model-invoked tools it is the only pre-execution decision point that sees
 * arguments -- `--tools` / `--exclude-tools` / `--no-tools` and the runtime
 * `setActiveTools` are applied at tool-registry construction, never per call.
 * A gateway guardrail rejection arrives as HTTP 403 and is translated into
 * `{block, reason}`; pi hands that reason to the model verbatim, so the model
 * reads the guardrail's own words.
 *
 * **Lifecycle mapping.** Two pi shapes make a naive mapping wrong:
 *
 * 1. *Agent-run re-entry.* One prompt can re-enter the agent run several times
 *    (provider retry, post-compaction, queued follow-up), and `turnIndex` resets
 *    to 0 each time, so turn indices collide within one prompt. The extension
 *    `agent_end` payload carries no `willRetry` marker, so a retry cannot be
 *    detected there. `agent_settled` is the only event that fires exactly once
 *    per logical run. Both an attempt counter and a session-monotonic turn
 *    sequence are therefore sent as metadata, because the gateway's own model is
 *    flat (session -> turn -> tool) and cannot express the nesting.
 * 2. *Concurrent tools.* pi preflights sibling calls sequentially then executes
 *    them concurrently, so `tool_execution_end` arrives out of submission order.
 *    All per-call state is keyed by `toolCallId`, which is the only correlator
 *    pi provides.
 *
 * Load it with `pi -e <path-to-this-file>`, or let `nemo-relay launch pi` do it.
 *
 * Environment (set by the launcher, overridable by hand):
 * - `NEMO_RELAY_PI_GATEWAY_URL`  gateway base URL (default `http://127.0.0.1:4040`)
 * - `NEMO_RELAY_PI_TIMEOUT_MS`   per-request timeout (default 5000)
 * - `NEMO_RELAY_PI_FAIL`         `closed` to block when the gateway is unreachable
 */
import {
  type GatewayConfig,
  configFromEnv,
  postAndForget,
  postHook,
  resolveFault,
} from './src/gateway-client.ts';
import type {
  AgentEndEvent,
  AgentSettledEvent,
  AgentStartEvent,
  ExtensionAPI,
  ExtensionContext,
  SessionShutdownEvent,
  SessionStartEvent,
  ToolCallEvent,
  ToolCallEventResult,
  ToolExecutionEndEvent,
  ToolExecutionStartEvent,
  TurnEndEvent,
  TurnStartEvent,
} from './src/pi-hook-types.ts';

export default function nemoRelayExtension(pi: ExtensionAPI): void {
  let config: GatewayConfig | null = null;

  /** Attempt counter within one logical agent run; reset on `agent_settled`. */
  let attemptIndex = 0;
  /** Session-monotonic turn counter; pi's own `turnIndex` resets on re-entry. */
  let turnSeq = 0;
  /** Tool names by call id, so the end payload can name the tool pi started. */
  const toolNames = new Map<string, string>();

  /**
   * Serializes every post to the gateway, in hook order.
   *
   * Firing posts concurrently reorders them: the gateway derives session and
   * turn boundaries from arrival order, so a late `agent_start` can land after
   * a `turn_start`, and a `session_shutdown` that overtakes an in-flight post
   * closes the session and lets the straggler open a second one. Both were
   * observed in an acceptance trace before this queue existed.
   *
   * The chain absorbs failures so one bad post cannot stall the rest, and
   * observability hooks still do not block pi -- they are enqueued, not
   * awaited. The gating hook does await, which means it also waits for
   * anything queued ahead of it; that ordering guarantee is worth the latency,
   * because a tool span opened under the wrong turn is simply wrong.
   */
  let chain: Promise<unknown> = Promise.resolve();

  // Declared as a function rather than a generic arrow: `<T>(...) => ...` in a
  // .ts file is ambiguous with JSX, and pi's jiti loader resolves it that way
  // and fails to load the extension -- silently, because pi collects extension
  // load errors rather than aborting.
  function enqueue<T>(job: () => Promise<T>): Promise<T> {
    const result = chain.then(job, job);
    chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /**
   * Resolve configuration lazily.
   *
   * pi's extension docs are explicit that a factory may run in invocations that
   * never start a session, so factories must not open resources. Reading the
   * environment is deferred to the first hook instead.
   */
  const ensureConfig = (ctx: ExtensionContext): GatewayConfig => {
    config ??= configFromEnv(safeSessionId(ctx));
    return config;
  };

  /** Queue an observability-only hook without charging pi's critical path. */
  const emit = (ctx: ExtensionContext, payload: Record<string, unknown>): void => {
    const active = ensureConfig(ctx);
    void enqueue(() => postAndForget(active, payload));
  };

  // ---------------------------------------------------------------------------
  // Session lifecycle
  //
  // pi's session_start/session_shutdown are the session boundary, NOT
  // agent_start/agent_end -- one pi session re-enters the agent run many times,
  // so treating agent_start as a session start would open a session per retry.
  // ---------------------------------------------------------------------------

  pi.on('session_start', async (event: SessionStartEvent, ctx: ExtensionContext) => {
    emit(ctx, { hook_event_name: 'session_start', reason: event.reason, cwd: ctx.cwd });
  });

  pi.on('session_shutdown', async (_event: SessionShutdownEvent, ctx: ExtensionContext) => {
    emit(ctx, { hook_event_name: 'session_shutdown' });
    // Drain before the process exits, or trailing spans are lost. Because the
    // queue is serial, this also guarantees session_shutdown is the last post
    // to reach the gateway rather than merely one of the last.
    await chain;
  });

  // ---------------------------------------------------------------------------
  // Agent-run lifecycle
  // ---------------------------------------------------------------------------

  pi.on('agent_start', async (_event: AgentStartEvent, ctx: ExtensionContext) => {
    emit(ctx, { hook_event_name: 'agent_start', attempt_index: attemptIndex });
    attemptIndex += 1;
  });

  pi.on('agent_end', async (event: AgentEndEvent, ctx: ExtensionContext) => {
    emit(ctx, {
      hook_event_name: 'agent_end',
      // Deliberately not a run boundary: pi may re-enter after this.
      attempt_index: Math.max(0, attemptIndex - 1),
      message_count: Array.isArray(event.messages) ? event.messages.length : 0,
    });
  });

  pi.on('agent_settled', async (_event: AgentSettledEvent, ctx: ExtensionContext) => {
    emit(ctx, { hook_event_name: 'agent_settled', attempts: attemptIndex });
    attemptIndex = 0;
  });

  pi.on('turn_start', async (event: TurnStartEvent, ctx: ExtensionContext) => {
    emit(ctx, {
      hook_event_name: 'turn_start',
      // pi's turn_index resets to 0 on re-entry; turn_seq does not, so a
      // consumer can still order turns across the whole session.
      turn_index: event.turnIndex,
      turn_seq: turnSeq,
      attempt_index: Math.max(0, attemptIndex - 1),
    });
    turnSeq += 1;
  });

  pi.on('turn_end', async (event: TurnEndEvent, ctx: ExtensionContext) => {
    emit(ctx, { hook_event_name: 'turn_end', turn_index: event.turnIndex });
  });

  // ---------------------------------------------------------------------------
  // Tool lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Fires before validation and before `tool_call`, and also for calls that
   * never execute. Recorded only so `tool_execution_end` can name its tool; it
   * is deliberately not forwarded as a tool start, because doing so would open
   * gateway spans for calls pi then discards.
   */
  pi.on('tool_execution_start', async (event: ToolExecutionStartEvent, _ctx: ExtensionContext) => {
    toolNames.set(event.toolCallId, event.toolName);
  });

  /**
   * The governance seam, and the only hook that blocks.
   *
   * Trap: `emitToolCall` returns on the first `{block: true}`, so an
   * earlier-loading extension can block before this handler runs. The call is
   * still blocked, but nothing is evaluated and the gateway never sees it.
   */
  pi.on(
    'tool_call',
    async (
      event: ToolCallEvent,
      ctx: ExtensionContext,
    ): Promise<ToolCallEventResult | undefined> => {
      const active = ensureConfig(ctx);
      // Enqueued rather than posted directly, so any observability hook fired
      // earlier in the same turn reaches the gateway first and the tool span
      // opens under the right turn.
      const outcome = await enqueue(() =>
        postHook(active, {
          hook_event_name: 'tool_call',
          tool_call_id: event.toolCallId,
          tool_name: event.toolName,
          input: event.input,
        }),
      );

      const decision =
        outcome.kind === 'fault' ? resolveFault(active, outcome.detail, event.toolName) : outcome;

      // `undefined` is the only correct allow value: a truthy result without
      // `block` is inert but overwrites earlier handlers' results.
      if (decision.kind !== 'block') return undefined;
      return { block: true, reason: decision.reason };
    },
  );

  /**
   * The tool end boundary for every outcome.
   *
   * `tool_result` does not fire for blocked calls -- they take pi's
   * `kind: "immediate"` path and never reach `afterToolCall` -- but
   * `tool_execution_end` always fires, with `isError: true`. So this is the only
   * hook that closes both allowed and blocked calls.
   */
  pi.on('tool_execution_end', async (event: ToolExecutionEndEvent, ctx: ExtensionContext) => {
    const toolName = event.toolName || toolNames.get(event.toolCallId) || 'unknown';
    toolNames.delete(event.toolCallId);
    emit(ctx, {
      hook_event_name: 'tool_execution_end',
      tool_call_id: event.toolCallId,
      tool_name: toolName,
      result: summarize(event.result, event.isError),
      status: event.isError ? 'error' : 'ok',
    });
  });
}

/** pi's session id, with a fallback so a missing manager cannot break loading. */
function safeSessionId(ctx: ExtensionContext): string {
  try {
    return ctx.sessionManager?.getSessionId?.() ?? 'unknown-session';
  } catch {
    return 'unknown-session';
  }
}

const MAX_CONTENT_CHARS = 2000;

/** Keep forwarded tool results small and JSON-safe. */
function summarize(result: unknown, isError: boolean): unknown {
  if (result === null || result === undefined) {
    return { content: isError ? 'Tool failed with no result.' : 'Tool completed with no result.' };
  }
  if (typeof result === 'string') return { content: truncate(result) };
  if (typeof result === 'object') {
    const record = result as Record<string, unknown>;
    const content = record.content ?? record.output ?? record.text;
    return {
      content:
        typeof content === 'string'
          ? truncate(content)
          : `Tool ${isError ? 'failed' : 'completed'}.`,
      result_keys: Object.keys(record).slice(0, 20),
    };
  }
  return { content: truncate(String(result)) };
}

function truncate(value: string): string {
  return value.length <= MAX_CONTENT_CHARS
    ? value
    : `${value.slice(0, MAX_CONTENT_CHARS)}... [truncated ${value.length - MAX_CONTENT_CHARS} chars]`;
}
