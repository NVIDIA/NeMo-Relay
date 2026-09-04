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
 * **Governance, on two paths.** For model-invoked tools, `tool_call` is the only
 * pre-execution decision point that sees arguments -- `--tools` /
 * `--exclude-tools` / `--no-tools` and the runtime `setActiveTools` are applied
 * at tool-registry construction, never per call. A gateway guardrail rejection
 * arrives as HTTP 403 and is translated into `{block, reason}`; pi hands that
 * reason to the model verbatim, so the model reads the guardrail's own words.
 *
 * The user's own bang-prefixed shell (`!cmd`) never reaches the tool registry,
 * so `tool_call` cannot see it. It is gated separately through `user_bash`,
 * under its own tool name and with a refusal shaped as a failed command rather
 * than as a blocked tool call, because pi gives that hook no reason contract.
 * See `src/user-bash.ts`.
 *
 * **Lifecycle mapping.** Two pi shapes make a naive mapping wrong:
 *
 * 1. *Agent-run re-entry.* One prompt can re-enter the agent run several times
 *    (provider retry, post-compaction, queued follow-up), and `turnIndex` resets
 *    to 0 each time, so turn indices collide within one prompt. The extension
 *    `agent_end` payload carries no `willRetry` marker, so a retry cannot be
 *    detected there -- `session_before_compact` is the one hook that announces
 *    one in advance, and only for the compaction case. `agent_settled` is the
 *    only event that fires exactly once per logical run. Every attributable
 *    hook therefore carries an attempt counter and a session-monotonic turn
 *    sequence, because the gateway's own model is flat (session -> turn ->
 *    tool) and cannot express the nesting. They travel as payload keys and the
 *    gateway promotes them into event metadata, which is what makes them
 *    survive on tool spans.
 * 2. *Concurrent tools.* pi preflights sibling calls sequentially then executes
 *    them concurrently, so `tool_execution_end` arrives out of submission order.
 *    All per-call state is keyed by `toolCallId`, which is the only correlator
 *    pi provides.
 *
 * Load it with `pi -e <path-to-this-file>`, or let `nemo-relay run --agent pi` do it.
 *
 * **Model redirection.** pi resolves a base URL per model from a generated
 * catalog, so the extension points the active model's provider at the gateway
 * itself. Under a launched session it also names the endpoint the gateway
 * should forward to, so a provider the gateway was never configured for is
 * still reachable. See `src/provider-redirect.ts`.
 *
 * Environment (set by the launcher, overridable by hand):
 * - `NEMO_RELAY_PI_GATEWAY_URL`  gateway base URL (default `http://127.0.0.1:4040`)
 * - `NEMO_RELAY_PI_TIMEOUT_MS`   per-request timeout (default 5000)
 * - `NEMO_RELAY_PI_FAIL`         `closed` to block when the gateway is unreachable
 * - `NEMO_RELAY_PI_REDIRECT`     `force` to skip the upstream check, `off` to disable
 * - `NEMO_RELAY_PI_{OPENAI,ANTHROPIC}_UPSTREAM`  what the gateway forwards to
 */
import {
  type GatewayConfig,
  configFromEnv,
  postAndForget,
  postHook,
  resolveFault,
} from './src/gateway-client.ts';
import {
  applyTransform,
  decideTransform,
  refusalReason,
} from './src/argument-transform.ts';
import {
  type RedirectConfig,
  decideRedirect,
  isNotable,
  redirectConfigFromEnv,
} from './src/provider-redirect.ts';
import {
  USER_BASH_END_HOOK,
  USER_BASH_HOOK,
  USER_BASH_TOOL_NAME,
  refusalResult,
  transformRefusalReason,
} from './src/user-bash.ts';
import type {
  AgentEndEvent,
  AgentSettledEvent,
  AgentStartEvent,
  ExtensionAPI,
  ExtensionContext,
  ModelSelectEvent,
  SessionBeforeCompactEvent,
  SessionCompactEvent,
  SessionShutdownEvent,
  SessionStartEvent,
  ToolCallEvent,
  ToolCallEventResult,
  ToolExecutionEndEvent,
  ToolExecutionStartEvent,
  TurnEndEvent,
  PiModel,
  TurnStartEvent,
  UserBashEvent,
  UserBashEventResult,
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
   * Inline shell commands have no pi-supplied identifier.
   *
   * `user_bash` carries only the command text, so the gate synthesizes a call
   * id to correlate its own start and end. Per runtime rather than globally
   * unique, which is all the gateway's tool map needs: it is keyed by call id
   * within one session.
   */
  let userBashSeq = 0;

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
  // and fails to load the extension. pi 0.84 prints that failure on stderr and
  // exits before the session starts, so it is reported rather than silent --
  // but it is reported by pi, not by anything here, and nothing Relay writes
  // will appear at all.
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

  /**
   * Forward a hook and wait for the gateway to have processed it.
   *
   * Used only for the two turn boundaries. Everything else the extension sends
   * is observability that can settle late, but a turn boundary *defines the
   * parent* of whatever comes next -- and model traffic does not travel through
   * this queue at all. pi sends model requests to the gateway directly over
   * HTTP, so a queued `turn_end` races them: an acceptance trace showed the
   * next turn's LLM span opening under the previous turn, because pi's request
   * beat our post. Awaiting these two costs two local round trips per turn and
   * removes the race, on the same reasoning that already makes `tool_call`
   * await -- a span opened under the wrong turn is simply wrong.
   */
  const emitOrdered = async (
    ctx: ExtensionContext,
    payload: Record<string, unknown>,
  ): Promise<void> => {
    const active = ensureConfig(ctx);
    await enqueue(() => postAndForget(active, payload));
  };

  /**
   * The attempt and turn a hook belongs to.
   *
   * Both counters hold the *next* value -- they are incremented as soon as the
   * event that opens their span is forwarded -- so the live attempt and turn
   * are one behind. The gateway promotes these two keys out of the payload and
   * into event metadata, which is the only reason they survive on tool events:
   * tool spans are built from the extracted call id, name, arguments, result
   * and metadata, and the raw payload is discarded.
   *
   * Sent on every hook that can be attributed. Without it, `tool_call` and
   * `tool_execution_end` are recoverable to an attempt only by reading
   * surrounding events in arrival order, which stops working the moment two
   * attempts are in flight.
   */
  const attribution = (): { attempt_index: number; turn_seq: number } => ({
    attempt_index: Math.max(0, attemptIndex - 1),
    turn_seq: Math.max(0, turnSeq - 1),
  });

  /** Providers already pointed at the gateway, so the redirect is idempotent. */
  const redirectedProviders = new Set<string>();
  /** Session id the current registrations carry, so a replacement can invalidate them. */
  let registeredSessionKey: string | null = null;
  let redirect: RedirectConfig | null = null;
  /**
   * Each provider's catalog as it was *before* we touched it.
   *
   * Snapshotted on first sight and never refreshed, because that is the whole
   * point: once `registerProvider` has run, every model of that provider reports
   * the gateway's URL, so re-reading the registry would compare the gateway
   * against itself and call any provider safe.
   */
  const pristineCatalog = new Map<string, PiModel[]>();

  const siblingsOf = (ctx: ExtensionContext, provider: string): PiModel[] => {
    const cached = pristineCatalog.get(provider);
    if (cached) return cached;
    const all = ctx.modelRegistry?.getAll?.() ?? [];
    const models = all.filter((candidate) => candidate.provider === provider);
    pristineCatalog.set(provider, models);
    return models;
  };

  /**
   * Point the active model's provider at the gateway, when that is safe.
   *
   * Runs on `session_start` and again on every `model_select`, because the
   * decision is per-model: switching from a provider the gateway fronts to one
   * it does not must not leave the new provider redirected. Each outcome is
   * reported to the gateway as a mark, so a trace with no LLM spans says why in
   * the trace itself rather than only in the user's terminal.
   */
  const applyRedirect = (ctx: ExtensionContext, source: string): void => {
    const active = ensureConfig(ctx);
    redirect ??= redirectConfigFromEnv(active.url);

    // The join key is baked into the registration, so it cannot follow a session
    // replacement on its own. pi `v0.84.0` tears the extension runtime down and
    // builds a fresh one for `/new`, `/resume` and `/fork`, so this branch does not
    // fire there -- it is kept for a host that reuses one runtime across session
    // ids, which would otherwise leave every redirected provider stamping the old
    // one onto its requests. Re-register when it moves.
    const sessionKey = safeSessionId(ctx);
    if (registeredSessionKey !== null && registeredSessionKey !== sessionKey) {
      redirectedProviders.clear();
    }
    registeredSessionKey = sessionKey;
    const decision = decideRedirect(
      ctx.model,
      redirect,
      redirectedProviders,
      ctx.model ? siblingsOf(ctx, ctx.model.provider) : [],
    );
    if (decision.kind === 'redirect') {
      // `baseUrl` rewrites the URL of every existing model for this provider and
      // keeps their API and costs. `headers` carries two things the gateway needs
      // from a redirected call: the session join key, and -- when the launcher
      // started this gateway -- its transparent-proxy credential.
      //
      // Both go here rather than in a `before_provider_headers` handler because
      // that hook is **global and carries no request identity**: its event has
      // only the headers, and its context is freshly built, so `ctx.model` is
      // whatever is selected *now*, not what this request is for. Scoping on it
      // gets both directions wrong -- omitting the key from a redirected call
      // whose model was captured before a switch, and leaking an internal session
      // id, or the credential, to a third-party provider we deliberately did not
      // redirect. Attaching them to the registration makes the scope structural:
      // only providers we actually pointed at the gateway ever send either.
      //
      // Without the credential the redirect succeeds and every model call then
      // comes back 401 -- a gateway started by `nemo-relay run` authenticates its
      // own client before any intercept can rewrite the route.
      pi.registerProvider(decision.provider, {
        baseUrl: redirect.gatewayUrl,
        headers: {
          'x-nemo-relay-session-id': sessionKey,
          ...(redirect.proxyToken ? { 'x-nemo-relay-proxy-token': redirect.proxyToken } : {}),
          // Only when the gateway does not already front this endpoint. It rides on
          // the registration for the same reason the credential does -- the scope is
          // structural, so a provider we left alone never names an upstream -- and
          // the gateway ignores it unless the proxy credential above is present and
          // matches, which is what stops any other local process steering it.
          ...(decision.namedUpstream
            ? { 'x-nemo-relay-upstream-base-url': decision.namedUpstream }
            : {}),
        },
      });
      redirectedProviders.add(decision.provider);
    }
    // A transient skip -- no model resolved yet, or a provider already pointed
    // at the gateway -- explains nothing, and a mark per session_start for it
    // is noise in every trace.
    if (!isNotable(decision)) return;
    emit(ctx, {
      hook_event_name: 'model_redirect',
      source,
      outcome: decision.kind,
      // The stable identifier, alongside the prose. Docs tell operators to interpret
      // these codes, so a consumer must not have to pattern-match a human sentence.
      ...(decision.kind === 'skip' ? { code: decision.code } : {}),
      reason: decision.reason,
      ...(decision.provider ? { provider: decision.provider } : {}),
      ...(decision.api ? { model_api: decision.api } : {}),
      ...(decision.kind === 'redirect' ? { upstream: decision.upstream } : {}),
      ...(ctx.model ? { model_id: ctx.model.id } : {}),
      ...attribution(),
    });
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
    applyRedirect(ctx, 'session_start');
  });

  /**
   * Re-evaluate on every model switch.
   *
   * pi fires this for the initial selection too, so a session that resolves its
   * model after `session_start` is still covered.
   */
  pi.on('model_select', async (_event: ModelSelectEvent, ctx: ExtensionContext) => {
    applyRedirect(ctx, 'model_select');
  });

  /**
   * Only some shutdown reasons actually end the session.
   *
   * `/reload` tears down and rebuilds the extension runtime while the session
   * itself continues, with the same session id. Forwarding a session end there
   * closes the gateway's session scope, and the `session_start` that follows
   * opens a second one -- silently splitting one logical session into two
   * disconnected traces. `quit` and the three session-replacement reasons
   * (`new`, `resume`, `fork`) do end the session and are forwarded.
   *
   * Known limitation: `attemptIndex` and `turnSeq` live in this factory's
   * closure, and pi re-runs the factory on reload with `moduleCache: false`, so
   * they restart at 0 mid-session. Rebuilding them would mean replaying the
   * session, which is out of scope here; `turn_seq` is therefore monotonic
   * within a runtime, not strictly within a session.
   */
  pi.on('session_shutdown', async (event: SessionShutdownEvent, ctx: ExtensionContext) => {
    if (event.reason === 'reload') {
      // Still drain: posts already queued belong to the continuing session.
      await chain;
      return;
    }
    emit(ctx, {
      hook_event_name: 'session_shutdown',
      reason: event.reason,
      ...(event.targetSessionFile ? { target_session_file: event.targetSessionFile } : {}),
    });
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
    emit(ctx, {
      hook_event_name: 'agent_settled',
      // Two different facts, both wanted: `attempts` is how many attempts the
      // run took, `attempt_index` is the last of them. Sending only the count
      // made this the one hook a consumer filtering on `attempt_index` missed.
      attempts: attemptIndex,
      ...attribution(),
    });
    attemptIndex = 0;
  });

  pi.on('turn_start', async (event: TurnStartEvent, ctx: ExtensionContext) => {
    const seq = turnSeq;
    turnSeq += 1;
    await emitOrdered(ctx, {
      hook_event_name: 'turn_start',
      // pi's turn_index resets to 0 on re-entry; turn_seq does not, so a
      // consumer can still order turns across the whole session.
      turn_index: event.turnIndex,
      turn_seq: seq,
      attempt_index: Math.max(0, attemptIndex - 1),
    });
  });

  pi.on('turn_end', async (event: TurnEndEvent, ctx: ExtensionContext) => {
    await emitOrdered(ctx, {
      hook_event_name: 'turn_end',
      // pi carries turn_index on the close but not turn_seq, so the close could
      // not be matched to its own open across a re-entry, where turn_index 0
      // appears once per attempt.
      turn_index: event.turnIndex,
      ...attribution(),
    });
  });

  // ---------------------------------------------------------------------------
  // Compaction
  //
  // Only `session_compact` is a compaction *event* to the gateway; the runtime
  // treats one as proof the context was rebuilt and marks the agent fresh so
  // the next model call records full context rather than a delta. That effect
  // is latent until pi's model traffic is routed through the gateway, but the
  // boundary is recorded now either way.
  // ---------------------------------------------------------------------------

  /**
   * Announced, not yet done, and cancellable by any extension loading after
   * this one -- so it is forwarded as a mark. Its `willRetry` is the only
   * advance notice pi gives an extension that the agent run is about to
   * re-enter; `agent_end` carries no such marker.
   *
   * Returns nothing on purpose: a returned object is how pi's API spells
   * "cancel this compaction, or replace its result".
   */
  pi.on('session_before_compact', async (event: SessionBeforeCompactEvent, ctx) => {
    emit(ctx, {
      hook_event_name: 'session_before_compact',
      reason: event.reason,
      will_retry: event.willRetry,
      tokens_before: event.preparation?.tokensBefore,
      is_split_turn: event.preparation?.isSplitTurn,
      ...attribution(),
    });
  });

  pi.on('session_compact', async (event: SessionCompactEvent, ctx) => {
    emit(ctx, {
      hook_event_name: 'session_compact',
      reason: event.reason,
      will_retry: event.willRetry,
      from_extension: event.fromExtension,
      tokens_before: event.compactionEntry?.tokensBefore,
      ...attribution(),
    });
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
          ...attribution(),
        }),
      );

      const decision =
        outcome.kind === 'fault' ? resolveFault(active, outcome, event.toolName) : outcome;

      if (decision.kind === 'block') return { block: true, reason: decision.reason };

      // Allowed. A request intercept may still have rewritten the arguments, in which case they
      // arrive in the response body and pi expects them applied to `event.input` in place.
      if (decision.kind === 'allow' && decision.body) {
        const transform = decideTransform(decision.body, event.toolCallId, event.input);
        if (transform.kind === 'refuse') {
          // Blocking, not falling back: running the original arguments would discard a policy
          // decision. Distinct from a guardrail rejection, and the reason says so.
          return { block: true, reason: refusalReason(event.toolName, transform.reason) };
        }
        if (transform.kind === 'apply') {
          applyTransform(event.input, transform.input);
          emit(ctx, {
            hook_event_name: 'tool_arguments_transformed',
            tool_call_id: event.toolCallId,
            tool_name: event.toolName,
            ...attribution(),
          });
        }
      }

      // `undefined` is the only correct allow value: a truthy result without
      // `block` is inert but overwrites earlier handlers' results.
      return undefined;
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
      ...attribution(),
    });
  });

  // ---------------------------------------------------------------------------
  // Inline shell
  // ---------------------------------------------------------------------------

  /** Close the gate span, whatever the outcome, so the trace stays balanced. */
  const endUserBash = (
    ctx: ExtensionContext,
    callId: string,
    // `policy-allowed` rather than `ok`: pi reports no completion for inline shell,
    // so the gate knows what it decided and never what happened. `fault-allowed` is
    // separate on purpose -- a command the gateway never ruled on is not a command policy
    // permitted, and recording both as `policy-allowed` made an unenforced session read
    // exactly like an enforced one.
    status: 'policy-allowed' | 'fault-allowed' | 'error',
    content: string,
  ): void => {
    emit(ctx, {
      hook_event_name: USER_BASH_END_HOOK,
      tool_call_id: callId,
      tool_name: USER_BASH_TOOL_NAME,
      result: { content },
      status,
      ...attribution(),
    });
  };

  /**
   * The second governance seam: pi's bang-prefixed inline shell.
   *
   * `!cmd` and `!!cmd` bypass the tool registry entirely -- they reach
   * `emitUserBash`, not `tool_call` -- so nothing about the tool gate covers
   * them. This closes that gap using the gateway machinery that already exists:
   * the command is posted as a tool start named `user_bash`, so the same
   * conditional-execution guardrail chain decides it and the same 403 contract
   * carries the reason back.
   *
   * Three pi facts shape every line of this handler.
   *
   * 1. **There is no block-and-reason contract.** A refusal is a synthetic
   *    failed `BashResult` and pi records it exactly as if the command had run.
   *    See `src/user-bash.ts` for the shape and why those field values.
   * 2. **`emitUserBash` catches**, so throwing fails *open* and the command
   *    runs unchecked. Nothing here is allowed to escape; the catch resolves an
   *    internal failure through the same policy as an unreachable gateway.
   * 3. **The first handler to return anything wins.** Loaded with `-e` this one
   *    is first; installed with `pi install` it loads last and an earlier
   *    extension can preempt it, in which case the gateway never sees the
   *    command at all.
   *
   * What the gate does *not* do is observe the command. pi has no completion
   * hook for inline shell -- `user_bash` is the only one -- so on an allow the
   * span closes immediately and measures the policy round trip, not the
   * command. Taking execution over to fix that would mean supplying pi with
   * custom `operations`, which drops the user's configured shell path and pi's
   * process-tree cancellation; the sidecar does not change how pi runs things.
   */
  pi.on(
    'user_bash',
    async (
      event: UserBashEvent,
      ctx: ExtensionContext,
    ): Promise<UserBashEventResult | undefined> => {
      const callId = `user-bash-${userBashSeq++}`;
      try {
        const active = ensureConfig(ctx);
        // Enqueued like `tool_call`, so a turn boundary posted earlier in the
        // same turn reaches the gateway first.
        const outcome = await enqueue(() =>
          postHook(active, {
            hook_event_name: USER_BASH_HOOK,
            tool_call_id: callId,
            tool_name: USER_BASH_TOOL_NAME,
            input: {
              command: event.command,
              cwd: event.cwd,
              // `!!` keeps the command and its output out of the model's
              // context, including the output of a refusal. A policy may
              // reasonably care.
              exclude_from_context: event.excludeFromContext,
            },
            ...attribution(),
          }),
        );

        const faulted = outcome.kind === 'fault';
        const decision = faulted ? resolveFault(active, outcome, USER_BASH_TOOL_NAME) : outcome;

        if (decision.kind === 'block') {
          endUserBash(ctx, callId, 'error', decision.reason);
          return { result: refusalResult(decision.reason) };
        }

        // An intercept rewrote the command. pi's result type can replace the
        // result or the execution backend but never the command itself, so the
        // rewrite cannot be honored -- and running the original would discard
        // the policy decision. Refuse, and say which of the two it is.
        if (decision.kind === 'allow' && decision.body?.tool_call?.input !== undefined) {
          const reason = transformRefusalReason();
          endUserBash(ctx, callId, 'error', reason);
          return { result: refusalResult(reason) };
        }

        // Deliberately not `ok`: pi has not run the command yet, and has no completion
        // hook to tell us how it went. It can still fail, and a later `user_bash`
        // handler can replace execution entirely -- so claiming success here would put
        // a false outcome in the trace. What we know is what we decided.
        endUserBash(
          ctx,
          callId,
          faulted ? 'fault-allowed' : 'policy-allowed',
          faulted
            ? 'Allowed without a policy decision: the gateway did not answer and the fail-open policy applies.'
            : 'Allowed by policy; pi has not reported the outcome.',
        );
        // `undefined` is the only correct allow value: any object at all is a
        // result or an operations override, and either would stop pi running
        // the command as the user typed it.
        return undefined;
      } catch (error) {
        // Reached only if something inside this handler failed -- `postHook`
        // resolves its own transport errors. Failing open here would be silent,
        // so it is resolved as a fault under the configured policy instead.
        //
        // The policy is re-read rather than defaulted: if `ensureConfig` is
        // what failed then `config` is still null, and hard-coding fail-open
        // there would override an explicit `NEMO_RELAY_PI_FAIL=closed` -- the
        // one case where an operator has asked for exactly this to block.
        const detail = error instanceof Error ? error.message : String(error);
        const fault = resolveFault(
          config ?? configFromEnv(safeSessionId(ctx)),
          // `handler`, not a transport failure: the gateway was never asked. Saying it could
          // not be reached would send the reader to a socket that is fine.
          { kind: 'fault', origin: 'handler', detail: `the inline-shell gate failed: ${detail}` },
          USER_BASH_TOOL_NAME,
        );
        if (fault.kind === 'block') return { result: refusalResult(fault.reason) };
        return undefined;
      }
    },
  );
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
    const text = toolResultText(content);
    return {
      content: text === null ? `Tool ${isError ? 'failed' : 'completed'}.` : text,
      result_keys: Object.keys(record).slice(0, 20),
    };
  }
  return { content: primitiveSummary(result, isError) };
}

function primitiveSummary(result: unknown, isError: boolean): string {
  switch (typeof result) {
    case 'boolean':
    case 'number':
    case 'bigint':
    case 'symbol':
      return truncate(String(result));
    default:
      return `Tool ${isError ? 'failed' : 'completed'} with an unsupported result type.`;
  }
}

/** Extract and bound Pi's ordered text blocks without forwarding image or unknown parts. */
function toolResultText(content: unknown): string | null {
  if (typeof content === 'string') return truncate(content);
  if (!Array.isArray(content)) return null;

  let text = '';
  let omittedChars = 0;
  let foundText = false;

  const append = (value: string): void => {
    const kept = sliceAtCodePointBoundary(
      value,
      Math.max(0, MAX_CONTENT_CHARS - text.length),
    );
    text += kept;
    omittedChars += value.length - kept.length;
  };

  for (const part of content) {
    if (typeof part !== 'object' || part === null || Array.isArray(part)) continue;
    const block = part as Record<string, unknown>;
    if (block.type !== 'text' || typeof block.text !== 'string') continue;
    if (foundText) append('\n');
    append(block.text);
    foundText = true;
  }
  if (!foundText) return null;
  return omittedChars === 0 ? text : `${text}... [truncated ${omittedChars} chars]`;
}

/** Return at most limit UTF-16 units without splitting a surrogate pair. */
function sliceAtCodePointBoundary(value: string, limit: number): string {
  let end = Math.min(value.length, limit);
  if (
    end > 0 &&
    end < value.length &&
    value.charCodeAt(end - 1) >= 0xd800 &&
    value.charCodeAt(end - 1) <= 0xdbff &&
    value.charCodeAt(end) >= 0xdc00 &&
    value.charCodeAt(end) <= 0xdfff
  ) {
    end -= 1;
  }
  return value.slice(0, end);
}

function truncate(value: string): string {
  if (value.length <= MAX_CONTENT_CHARS) return value;
  const kept = sliceAtCodePointBoundary(value, MAX_CONTENT_CHARS);
  return `${kept}... [truncated ${value.length - kept.length} chars]`;
}
