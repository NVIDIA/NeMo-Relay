// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Runtime coordinator for the DeepSeek Harness integration.
 *
 * DeepSeek Harness exposes its real model stream (`llm/stream`) and real tool
 * body (`tools/execute`) to plugins, so this integration places NeMo Relay's
 * managed execution wrappers at those boundaries: guardrails, intercepts,
 * sanitizers, codecs, and lifecycle events run exactly as they do for any
 * managed Relay call. Session events supply the turn structure.
 *
 * Scope model (the CLI gateway's model for hosts without a reliable session
 * end): each DeepSeek Harness session owns one scope stack; each turn is a
 * top-level `Custom` scope with `nemo_relay_scope_role: "turn"`, so the
 * observability plugin writes one ATIF trajectory per turn.
 */
import { randomUUID } from 'node:crypto';

import type { NemoRelayDeepSeekHarnessConfig, NemoRelayPluginHostConfig } from './config.js';
import type {
  Agent,
  GenerateOptions,
  HarnessToolResult,
  JsonValue,
  PostToolDecision,
  PreToolDecision,
  Session,
  SessionEvent,
  SessionHeader,
  StreamChunk,
  ToolExecution,
} from './dsh-types.js';
import {
  createHealthSnapshot,
  emptyCounters,
  type NemoRelayHealthSnapshot,
  type RuntimeCounters,
  type RuntimeStatus,
} from './health.js';
import {
  ChunkBridge,
  CompletionAggregator,
  asRecord,
  classifyRequestEdits,
  jsonEqual,
  projectRequest,
  toJson,
  UnsupportedRewriteError,
} from './llm-projection.js';
import {
  bindingPrecondition,
  defaultNemoRelayModuleLoader,
  describeBindingFailure,
  type ConfigDiagnostic,
  type LlmRequestShape,
  type NemoRelayModuleLoader,
  type NemoRelayModules,
  type PluginConfig,
  type PluginHostActivation,
  type RelayJson,
  type ScopeHandle,
  type ScopeStack,
} from './modules.js';
import type { PhoenixExporter } from './phoenix.js';
import { atifComponent, planPluginDocument, secretRedactionComponent, type PluginDocumentPlan } from './plugin-doc.js';
import {
  diffToolResult,
  resultAnnotation,
  wrapToolResult,
  type ToolResultEdit,
  type ToolResultWrapper,
} from './tool-projection.js';

export const AGENT_NAME = 'deepseek-harness';
export const TURN_SCOPE_NAME = 'deepseek-harness-turn';
export const MARK_PREFIX = 'deepseek-harness.';
/** How long a model or tool call waits for a runtime that is still starting before it runs without Relay. */
export const DEFAULT_START_WAIT_MS = 3_000;
/** Session structure kept while the runtime starts; later entries are dropped. */
export const START_BUFFER_LIMIT = 1_000;
const LLM_STREAMING = 2;
const SANITIZER_PREFIX = 'nemo-relay-deepseek-harness.capture.';
const GUARDRAIL_REJECTED = /^guardrail rejected:\s*/i;
const GUARDRAIL_CODE = 'NEMO_RELAY_GUARDRAIL_REJECTED';

export type RuntimeLogger = {
  info(message: string): void;
  warn(message: string): void;
};

export type RuntimeOptions = {
  config: NemoRelayDeepSeekHarnessConfig;
  moduleLoader?: NemoRelayModuleLoader;
  logger?: RuntimeLogger;
  /** Resolve a live DeepSeek Harness session for its header (parent, delegation depth). */
  sessionLookup?: (sessionId: string) => Session | undefined;
  /** Receives every Relay event this runtime observes (used by the conversation view). */
  onEvent?: (event: unknown) => void;
  /**
   * Tool conditional-execution guardrail owned by this integration. Relay runs it
   * for every managed tool call; while Relay is unavailable, `tools/pre-execute`
   * evaluates it directly so the user's rules still hold.
   */
  guardrail?: RuntimeGuardrail;
  /** How long a call waits for a start in flight; {@link DEFAULT_START_WAIT_MS} by default. */
  startWaitMs?: number;
  /** DeepSeek Harness's default model when the plugin host activates (the ATIF trajectory model). */
  defaultModel?: () => string | undefined;
};

export type RuntimeGuardrail = {
  name: string;
  priority: number;
  /** A rejection reason to block the call, or `null` to allow it. Must not throw. */
  evaluate(toolName: string, args: unknown): string | null;
};

type TurnState = {
  index: number;
  /** What started the turn: `user`, or the DeepSeek Harness message source of a notice (such as `subagent-settled`). */
  source: string;
  /** Whether a message of this turn has named its source yet. */
  sourceSeen: boolean;
  startMicros: number;
  prompt: string[];
  handle?: ScopeHandle | undefined;
  lastText?: string | undefined;
};

/** Message sources DeepSeek Harness adds as context to a turn; they never start one. */
const CONTEXT_SOURCES: ReadonlySet<string> = new Set(['runtime-context', 'skill-catalog']);

type SessionState = {
  id: string;
  /** The session's turns and the calls inside them. */
  stack: ScopeStack;
  /** Calls that belong to the session but not to a turn (session titles); created on first use. */
  sideStack?: ScopeStack | undefined;
  header?: SessionHeader | undefined;
  turn?: TurnState | undefined;
  step?: { turn: number; step: number } | undefined;
};

/**
 * Model-call purposes that run beside a turn rather than inside it. A `null`
 * parent means "the top of the stack", which on the session stack is the open
 * turn, so these calls run on the session's side stack instead.
 */
const OUT_OF_TURN_PURPOSES: ReadonlySet<string> = new Set(['session-title']);

type LlmTarget = {
  stack: ScopeStack;
  parent: ScopeHandle | null;
  metadata: Record<string, JsonValue>;
  session?: SessionState;
};

/** Host notifications that arrived while the runtime was starting, replayed in order once it runs. */
type BufferedEntry =
  | { kind: 'session-event'; session: Session; event: SessionEvent }
  | { kind: 'step'; agent: Agent; turn: number; step: number }
  | { kind: 'agent-disposed'; agent: Agent };

/** Owns one Relay runtime instance across DeepSeek Harness plugin activation and disposal. */
export class NemoRelayDeepSeekHarnessRuntime {
  private readonly config: NemoRelayDeepSeekHarnessConfig;
  private readonly moduleLoader: NemoRelayModuleLoader;
  private readonly logger: RuntimeLogger;
  private readonly sessionLookup: ((sessionId: string) => Session | undefined) | undefined;
  private readonly onEvent: ((event: unknown) => void) | undefined;
  private readonly guardrail: RuntimeGuardrail | undefined;
  private readonly defaultModel: (() => string | undefined) | undefined;
  private modules: NemoRelayModules | undefined;
  private activation: PluginHostActivation | undefined;
  private statusValue: RuntimeStatus = { state: 'not_initialized' };
  private initializedPluginHost = false;
  private pluginHostOutputsHealthy = true;
  private resolvedConfig: unknown;
  private configPaths: string[] = [];
  private readonly counters: RuntimeCounters = emptyCounters();
  private readonly sessions = new Map<string, SessionState>();
  private harnessStack: ScopeStack | undefined;
  private readonly toolEdits = new WeakMap<object, ToolResultEdit>();
  private readonly disposers: Array<() => void> = [];
  private codec:
    | {
        decode(request: LlmRequestShape): RelayJson;
        encode(annotated: RelayJson, original: LlmRequestShape): LlmRequestShape;
        decodeResponse(response: RelayJson): RelayJson;
      }
    | undefined;
  private startPromise: Promise<void> | undefined;
  /** True while `start()` is in flight: host notifications are buffered and calls wait for it. */
  private starting = false;
  private buffered: BufferedEntry[] = [];
  private droppedWhileStarting = 0;
  private readonly startWaitMs: number;
  private startWaitExpired = false;
  private phoenixExport:
    | { subscriber: InstanceType<NemoRelayModules['nf']['OpenTelemetrySubscriber']>; name: string; key: string }
    | undefined;
  private plan: PluginDocumentPlan | undefined;
  private readonly loggedDiagnostics = new Set<string>();
  private readonly subscriberName = `nemo-relay-deepseek-harness.${randomUUID()}`;

  constructor(options: RuntimeOptions) {
    this.config = options.config;
    this.moduleLoader = options.moduleLoader ?? defaultNemoRelayModuleLoader;
    this.logger = guardLogger(options.logger);
    this.sessionLookup = options.sessionLookup;
    this.onEvent = options.onEvent;
    this.guardrail = options.guardrail;
    this.startWaitMs = Math.max(0, options.startWaitMs ?? DEFAULT_START_WAIT_MS);
    this.defaultModel = options.defaultModel;
  }

  /** Current coarse runtime status. */
  status(): RuntimeStatus {
    return this.statusValue;
  }

  /** Operator-facing health payload for `/relay` and the status file. */
  health(): NemoRelayHealthSnapshot {
    return createHealthSnapshot({
      status: this.statusValue,
      initializedPluginHost: this.initializedPluginHost,
      pluginHostOutputsHealthy: this.pluginHostOutputsHealthy,
      config: this.config,
      resolvedConfig: this.resolvedConfig,
      configPaths: this.configPaths,
      counters: { ...this.counters },
      activeSessions: this.sessions.size,
    });
  }

  /** The activated plugin-host document and the defaults it received, once the runtime started. */
  pluginPlan(): PluginDocumentPlan | undefined {
    return this.plan;
  }

  /** The fully merged plugin-host configuration Relay reported, when known. */
  resolvedPluginConfig(): unknown {
    return this.resolvedConfig;
  }

  /** The loaded `nemo-relay-node` version, when known. */
  relayVersion(): string | undefined {
    return this.modules?.version;
  }

  /**
   * Export to Phoenix through Relay's own OpenTelemetry exporter with the
   * OpenInference projection. A direct subscriber leaves the plugin-host
   * activation untouched: no exporter, redaction policy, or in-flight ATIF
   * trajectory is torn down to connect or disconnect Phoenix.
   */
  phoenixExporter(): PhoenixExporter {
    return {
      ready: () => this.isActive(),
      connect: (target) => this.connectPhoenix(target),
      disconnect: (flush) => this.disconnectPhoenix(flush),
      connected: () => this.phoenixExport !== undefined,
    };
  }

  private connectPhoenix(target: { endpoint: string; project: string }): void {
    const modules = this.modules;
    if (modules === undefined || !this.isActive()) {
      throw new Error('NeMo Relay is not running');
    }
    const key = `${target.endpoint}\n${target.project}`;
    if (this.phoenixExport?.key === key) {
      return;
    }
    void this.disconnectPhoenix(false);
    const subscriber = new modules.nf.OpenTelemetrySubscriber({
      type: 'openinference',
      endpoint: target.endpoint,
      transport: 'http_binary',
      serviceName: AGENT_NAME,
      resourceAttributes: { 'openinference.project.name': target.project },
      timeoutMillis: 3000,
    });
    const name = `${this.subscriberName}.phoenix`;
    subscriber.register(name);
    this.phoenixExport = { subscriber, name, key };
  }

  private async disconnectPhoenix(flush: boolean): Promise<void> {
    const current = this.phoenixExport;
    if (current === undefined) {
      return;
    }
    this.phoenixExport = undefined;
    if (flush) {
      // Deliver queued events to the exporter before it stops receiving them.
      await this.modules?.nf.flushSubscribers().catch(() => undefined);
    }
    try {
      current.subscriber.deregister(current.name);
    } catch {
      // Already gone.
    }
    if (flush) {
      try {
        current.subscriber.forceFlush();
      } catch (error) {
        this.logger.warn(`NeMo Relay could not deliver the last spans to Phoenix: ${toMessage(error)}`);
      }
    }
    try {
      current.subscriber.shutdown();
    } catch {
      // Shutting down a stopped exporter changes nothing.
    }
  }

  /** Whether host hooks currently route through Relay. */
  isActive(): boolean {
    return this.modules !== undefined && (this.statusValue.state === 'ready' || this.statusValue.state === 'degraded');
  }

  /**
   * Load the binding, initialize the Relay plugin host, and register capture sanitizers. Idempotent.
   *
   * DeepSeek Harness can begin a turn before this settles (a fresh process submits its task right
   * after boot). Session structure that arrives meanwhile is buffered and replayed in order before
   * the returned promise resolves, and model and tool calls wait for it (see `settleStart`), so the
   * first turn of a process keeps its prompt and its calls run through Relay.
   */
  start(): Promise<void> {
    if (this.startPromise === undefined) {
      this.starting = true;
      this.startPromise = this.startOnce();
    }
    return this.startPromise;
  }

  private async startOnce(): Promise<void> {
    try {
      await this.activate();
    } catch (error) {
      // Calls pass through rather than run on a half-initialized runtime.
      this.statusValue = { state: 'disabled', reason: `start failed: ${toMessage(error)}` };
      this.logger.warn(`NeMo Relay failed to start: ${toMessage(error)}`);
    } finally {
      this.starting = false;
      this.replayBuffered();
    }
  }

  /** Apply the notifications buffered during start, or drop them when Relay did not become active. */
  private replayBuffered(): void {
    const entries = this.buffered;
    this.buffered = [];
    if (this.droppedWhileStarting > 0) {
      this.logger.warn(
        `NeMo Relay dropped ${this.droppedWhileStarting} session notification(s) that arrived while it started`,
      );
      this.droppedWhileStarting = 0;
    }
    if (!this.isActive()) {
      return;
    }
    for (const entry of entries) {
      switch (entry.kind) {
        case 'session-event':
          this.onSessionEvent(entry.session, entry.event);
          break;
        case 'step':
          this.noteStep(entry.agent, entry.turn, entry.step);
          break;
        case 'agent-disposed':
          this.onAgentDisposed(entry.agent);
          break;
      }
    }
  }

  /** Keep a host notification until the start in flight settles; true when it was buffered. */
  private bufferWhileStarting(entry: BufferedEntry): boolean {
    if (!this.starting) {
      return false;
    }
    if (this.buffered.length < START_BUFFER_LIMIT) {
      this.buffered.push(entry);
    } else {
      this.droppedWhileStarting += 1;
    }
    return true;
  }

  /**
   * Wait for a start in flight, at most `startWaitMs` and never past `signal`, so a call made
   * while the binding loads still runs through Relay. After the wait the call fails open: it runs
   * without Relay when Relay is not active yet.
   */
  private async settleStart(signal?: AbortSignal): Promise<void> {
    const pending = this.startPromise;
    if (!this.starting || pending === undefined) {
      return;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    let onAbort: (() => void) | undefined;
    const waited = await Promise.race([
      pending.then(
        () => 'started' as const,
        () => 'started' as const,
      ),
      new Promise<'timeout'>((resolve) => {
        timer = setTimeout(() => resolve('timeout'), this.startWaitMs);
      }),
      new Promise<'aborted'>((resolve) => {
        if (signal === undefined) {
          return;
        }
        if (signal.aborted) {
          resolve('aborted');
          return;
        }
        onAbort = () => resolve('aborted');
        signal.addEventListener('abort', onAbort, { once: true });
      }),
    ]);
    clearTimeout(timer);
    if (onAbort !== undefined) {
      signal?.removeEventListener('abort', onAbort);
    }
    if (waited === 'timeout' && !this.startWaitExpired) {
      this.startWaitExpired = true;
      this.logger.warn(
        `NeMo Relay is still starting after ${this.startWaitMs} ms; calls run without Relay until it is ready`,
      );
    }
  }

  private async activate(): Promise<void> {
    if (!this.config.enabled) {
      this.statusValue = { state: 'disabled', reason: 'disabled by configuration' };
      return;
    }
    let modules: NemoRelayModules;
    try {
      // A host that cannot run the binding is refused before the native addon loads.
      const precondition = this.moduleLoader === defaultNemoRelayModuleLoader ? bindingPrecondition() : undefined;
      if (precondition !== undefined) {
        throw new Error(precondition);
      }
      modules = await this.moduleLoader();
    } catch (error) {
      this.statusValue = { state: 'disabled', reason: describeBindingFailure(error) };
      this.logger.warn(`NeMo Relay disabled: ${this.statusValue.reason}`);
      return;
    }
    if (this.statusValue.state === 'stopping' || this.statusValue.state === 'stopped') {
      return;
    }
    this.modules = modules;
    let degraded: string | undefined;

    try {
      const plan = this.planDocument(modules);
      if (plan.errors.length > 0) {
        degraded = `plugin configuration has ${plan.errors.length} error(s): ${plan.errors.join('; ')}`;
      } else {
        this.activation = await modules.pluginHost.initialize(plan.document as unknown as PluginConfig);
        this.initializedPluginHost = true;
        this.recordReport(
          this.activation.report.config.diagnostics,
          this.activation.report.config_paths,
          this.activation.report.resolved_config,
        );
      }
    } catch (error) {
      degraded = `plugin host failed to initialize: ${toMessage(error)}`;
    }
    if (degraded !== undefined) {
      this.pluginHostOutputsHealthy = false;
      this.logger.warn(`NeMo Relay plugin host degraded: ${degraded}`);
    }

    // Each registration is independent: a failing one must not leave the rules guardrail unregistered.
    const registrations: Array<[string, () => void]> = [
      ['codec', () => (this.codec = new modules.nf.OpenAIChatCodec())],
      ['capture sanitizers', () => this.registerCaptureSanitizers(modules)],
      ['rules guardrail', () => this.registerGuardrail(modules)],
      ['view subscriber', () => this.registerSubscriber(modules)],
    ];
    for (const [label, register] of registrations) {
      try {
        register();
      } catch (error) {
        degraded ??= `runtime registration failed (${label}): ${toMessage(error)}`;
        this.logger.warn(`NeMo Relay ${label} registration failed: ${toMessage(error)}`);
      }
    }
    this.statusValue = degraded === undefined ? { state: 'ready' } : { state: 'degraded', reason: degraded };
  }

  /** Close open turns, flush subscribers, and close the plugin host activation. */
  async stop(): Promise<void> {
    if (this.statusValue.state === 'stopped') {
      return;
    }
    await this.startPromise?.catch(() => undefined);
    this.statusValue = { state: 'stopping' };
    const modules = this.modules;
    if (modules !== undefined) {
      for (const session of this.sessions.values()) {
        this.closeTurn(session, { reason: 'shutdown' }, 'UNSET');
      }
      try {
        await modules.nf.flushSubscribers();
      } catch (error) {
        this.logger.warn(`NeMo Relay flush failed: ${toMessage(error)}`);
      }
    }
    // The Phoenix manager flushes and disconnects first; this only releases an exporter left behind.
    await this.disconnectPhoenix(false);
    this.sessions.clear();
    for (const dispose of this.disposers.splice(0).reverse()) {
      try {
        dispose();
      } catch {
        // A registration that is already gone needs no cleanup.
      }
    }
    if (this.activation !== undefined) {
      try {
        await this.activation.close();
        this.activation = undefined;
      } catch (error) {
        // Keep the activation so a later stop can retry its teardown.
        this.statusValue = { state: 'degraded', reason: `plugin host close failed: ${toMessage(error)}` };
        return;
      }
    }
    this.statusValue = { state: 'stopped' };
  }

  /* ------------------------------------------------------------------------ */
  /* Session structure                                                         */
  /* ------------------------------------------------------------------------ */

  /** Apply one committed DeepSeek Harness session event (buffered while the runtime starts). */
  onSessionEvent(session: Session, event: SessionEvent): void {
    if (this.bufferWhileStarting({ kind: 'session-event', session, event }) || !this.isActive()) {
      return;
    }
    try {
      this.dispatchSessionEvent(session, event);
    } catch (error) {
      this.counters.middlewareErrors += 1;
      this.logger.warn(`NeMo Relay could not map ${event.type}: ${toMessage(error)}`);
    }
  }

  private dispatchSessionEvent(session: Session, event: SessionEvent): void {
    const data = asRecord(event.data) ?? {};
    switch (event.type) {
      case 'turn/start': {
        const state = this.session(session.id, session);
        this.closeTurn(state, { reason: 'superseded' }, 'UNSET');
        const index = typeof data.turn === 'number' ? data.turn : 0;
        state.turn = {
          index,
          source: 'user',
          sourceSeen: false,
          startMicros: Math.round(event.time * 1000),
          prompt: [],
        };
        return;
      }
      case 'user/message': {
        const state = this.sessions.get(session.id);
        const turn = state?.turn;
        const source = asRecord(data.source)?.kind;
        if (state === undefined || turn === undefined || typeof source !== 'string' || CONTEXT_SOURCES.has(source)) {
          return;
        }
        const text = messageText(data.content);
        if (turn.handle !== undefined) {
          if (source === 'user') {
            this.mark(state, 'user_message', this.config.capture.includePrompts ? { text } : null);
          }
          return;
        }
        // The turn's first message names what started it: the user, or a notice such as a settled subagent.
        if (!turn.sourceSeen) {
          turn.source = source;
          turn.sourceSeen = true;
          turn.prompt.push(text);
        } else if (source === 'user') {
          turn.prompt.push(text);
        }
        return;
      }
      case 'turn/end': {
        const state = this.sessions.get(session.id);
        if (state === undefined) {
          return;
        }
        const reason = asRecord(data.reason);
        const kind = typeof reason?.kind === 'string' ? reason.kind : 'completed';
        const failure = asRecord(reason?.error);
        const error = typeof failure?.message === 'string' ? failure.message : undefined;
        this.closeTurn(state, error === undefined ? { reason: kind } : { reason: kind, error }, statusForTurnEnd(kind));
        return;
      }
      case 'compaction/end': {
        const state = this.sessions.get(session.id);
        if (state !== undefined) {
          // Reserved mark name: resets LLM history freshness for observability consumers.
          this.emitMark(state, 'compaction', toJson(event.data ?? null));
        }
        return;
      }
      case 'subagent/catalog': {
        const state = this.session(session.id, session);
        this.mark(state, 'subagent', {
          child_session_id: typeof data.childId === 'string' ? data.childId : null,
          label: typeof data.label === 'string' ? data.label : null,
          mode: typeof data.mode === 'string' ? data.mode : null,
        });
        return;
      }
      case 'approval/asked':
      case 'approval/decided': {
        const state = this.sessions.get(session.id);
        if (state !== undefined) {
          this.mark(
            state,
            event.type === 'approval/asked' ? 'approval_asked' : 'approval_decided',
            toJson(event.data ?? null),
          );
        }
        return;
      }
      default:
        return;
    }
  }

  /** Record the turn and step an agent is about to run (buffered while the runtime starts). */
  noteStep(agent: Agent | undefined, turn: unknown, step: unknown): void {
    if (typeof agent?.id !== 'string' || !isStepNumber(turn) || !isStepNumber(step)) {
      return;
    }
    if (this.bufferWhileStarting({ kind: 'step', agent, turn, step }) || !this.isActive()) {
      return;
    }
    this.session(agent.id, agent.session).step = { turn, step };
  }

  /** Close the turn of a disposed agent and release its session state. */
  onAgentDisposed(agent: Agent): void {
    if (this.bufferWhileStarting({ kind: 'agent-disposed', agent })) {
      return;
    }
    const state = this.sessions.get(agent.id);
    if (state === undefined) {
      return;
    }
    this.closeTurn(state, { reason: 'disposed' }, 'UNSET');
    this.sessions.delete(agent.id);
  }

  private session(id: string, source?: Session): SessionState {
    let state = this.sessions.get(id);
    if (state === undefined) {
      state = { id, stack: this.nf().createScopeStack() };
      this.sessions.set(id, state);
    }
    const header = source?.header ?? (state.header === undefined ? this.sessionLookup?.(id)?.header : undefined);
    if (header !== undefined) {
      state.header = header;
    }
    return state;
  }

  private sessionMetadata(state: SessionState): Record<string, JsonValue> {
    const metadata: Record<string, JsonValue> = { session_id: state.id, 'gen_ai.agent.name': AGENT_NAME };
    if (state.header?.parentSession !== undefined) {
      metadata.parent_session_id = state.header.parentSession;
    }
    if (state.header?.delegationDepth !== undefined && state.header.delegationDepth > 0) {
      metadata.delegation_depth = state.header.delegationDepth;
    }
    return metadata;
  }

  private turnMetadata(state: SessionState, turn: TurnState): Record<string, JsonValue> {
    return {
      ...this.sessionMetadata(state),
      nemo_relay_scope_role: 'turn',
      turn_index: turn.index,
      turn_source: turn.source,
    };
  }

  /**
   * Open the current turn scope on first use, so the scope carries the user prompt that
   * DeepSeek Harness commits after `turn/start`. A turn already running when Relay
   * started is adopted from the step the agent reported.
   */
  private ensureTurn(state: SessionState): ScopeHandle | null {
    if (state.turn === undefined) {
      if (state.step === undefined) {
        return null;
      }
      state.turn = {
        index: state.step.turn,
        source: 'resumed',
        sourceSeen: true,
        startMicros: Date.now() * 1000,
        prompt: [],
      };
    }
    const turn = state.turn;
    if (turn.handle === undefined) {
      const nf = this.nf();
      const prompt = turn.prompt.join('\n\n');
      const input = this.config.capture.includePrompts && prompt.length > 0 ? { prompt } : null;
      turn.handle = nf.withScopeStack(state.stack, () =>
        nf.pushScope(
          TURN_SCOPE_NAME,
          scopeTypeCustom(nf),
          null,
          0,
          null,
          this.turnMetadata(state, turn),
          input,
          turn.startMicros,
        ),
      ) as ScopeHandle;
      this.counters.turnsOpened += 1;
    }
    return turn.handle ?? null;
  }

  private closeTurn(
    state: SessionState,
    output: { reason: string; error?: string },
    status: 'OK' | 'ERROR' | 'UNSET',
  ): void {
    const turn = state.turn;
    state.turn = undefined;
    // A later call outside a turn (for example a manual compaction) must not adopt this turn's step.
    state.step = undefined;
    if (turn?.handle === undefined) {
      return;
    }
    const handle = turn.handle;
    const metadata: Record<string, JsonValue> = {
      ...this.turnMetadata(state, turn),
      turn_end_reason: output.reason,
      // Relay's OpenTelemetry exporters set the span status from this field.
      'otel.status_code': status,
    };
    if (status === 'ERROR') {
      metadata['otel.status_description'] = output.error ?? output.reason;
      metadata['error.type'] = output.reason;
    }
    const result: Record<string, JsonValue> = { reason: output.reason };
    if (output.error !== undefined) {
      result.error = output.error;
    }
    if (this.config.capture.includeResponses && turn.lastText !== undefined) {
      result.final_text = turn.lastText;
    }
    try {
      const nf = this.nf();
      nf.withScopeStack(state.stack, () => nf.popScope(handle, result, null, metadata));
      this.counters.turnsClosed += 1;
    } catch (error) {
      this.counters.middlewareErrors += 1;
      this.logger.warn(`NeMo Relay could not close turn ${turn.index} of ${state.id}: ${toMessage(error)}`);
    }
  }

  private mark(
    state: SessionState,
    name: string,
    data: JsonValue,
    at?: { stack: ScopeStack; parent: ScopeHandle | null },
  ): void {
    this.emitMark(state, `${MARK_PREFIX}${name}`, data, at);
  }

  /** Emit a mark in the session's open turn, or at `at` (the stack and parent of the call it describes). */
  private emitMark(
    state: SessionState,
    name: string,
    data: JsonValue,
    at?: { stack: ScopeStack; parent: ScopeHandle | null },
  ): void {
    const nf = this.nf();
    const stack = at?.stack ?? state.stack;
    const parent = at === undefined ? (state.turn?.handle ?? null) : at.parent;
    nf.withScopeStack(stack, () => nf.event(name, parent, data, this.sessionMetadata(state), null));
    this.counters.marksEmitted += 1;
  }

  /* ------------------------------------------------------------------------ */
  /* Model calls                                                               */
  /* ------------------------------------------------------------------------ */

  /** `llm/stream` listener body: route the call through a managed Relay LLM stream. */
  streamLlm(options: GenerateOptions, next: () => AsyncIterable<StreamChunk>): AsyncIterable<StreamChunk> {
    if (this.starting) {
      return this.streamAfterStart(options, next);
    }
    if (!this.isActive()) {
      return next();
    }
    return this.managedStream(options, next);
  }

  private async *streamAfterStart(
    options: GenerateOptions,
    next: () => AsyncIterable<StreamChunk>,
  ): AsyncGenerator<StreamChunk> {
    await this.settleStart(options.signal);
    yield* this.isActive() ? this.managedStream(options, next) : next();
  }

  private llmTarget(options: GenerateOptions): LlmTarget {
    const purpose = options.purpose;
    if (options.sessionId === undefined) {
      this.harnessStack ??= this.nf().createScopeStack();
      return {
        stack: this.harnessStack,
        parent: null,
        metadata: { 'gen_ai.agent.name': AGENT_NAME, request_purpose: purpose ?? 'harness' },
      };
    }
    const state = this.session(options.sessionId);
    if (purpose !== undefined && OUT_OF_TURN_PURPOSES.has(purpose)) {
      // Title generation runs beside the turn: keep it out of the turn's trajectory, usage, and
      // history freshness, but in the session (its metadata routes it to the session's traces).
      state.sideStack ??= this.nf().createScopeStack();
      return {
        stack: state.sideStack,
        parent: null,
        metadata: { ...this.sessionMetadata(state), request_purpose: purpose },
        session: state,
      };
    }
    const parent = this.ensureTurn(state);
    const metadata: Record<string, JsonValue> = { ...this.sessionMetadata(state) };
    if (state.turn !== undefined && parent !== null) {
      metadata.turn_index = state.turn.index;
      if (state.step?.turn === state.turn.index) {
        metadata.step_index = state.step.step;
      }
    }
    if (purpose !== undefined) {
      metadata.request_purpose = purpose;
    }
    return { stack: state.stack, parent, metadata, session: state };
  }

  private async *managedStream(
    options: GenerateOptions,
    next: () => AsyncIterable<StreamChunk>,
  ): AsyncGenerator<StreamChunk> {
    const nf = this.nf();
    let target: LlmTarget;
    let projection: ReturnType<typeof projectRequest>;
    try {
      target = this.llmTarget(options);
      projection = projectRequest(options);
    } catch (error) {
      this.counters.middlewareErrors += 1;
      this.logger.warn(`NeMo Relay could not prepare a model call; continuing without Relay: ${toMessage(error)}`);
      yield* next();
      return;
    }
    const bridge = new ChunkBridge();
    const aggregator = new CompletionAggregator(options.model, `deepseek-harness-${randomUUID()}`);
    const flags: { funcCalled: boolean; rewrite?: UnsupportedRewriteError; hostError?: unknown; dropped: string[] } = {
      funcCalled: false,
      dropped: [],
    };
    const codec = this.codec;

    const produce = (received: unknown, signal: AbortSignal): AsyncIterable<JsonValue> => {
      flags.funcCalled = true;
      try {
        flags.dropped = classifyRequestEdits(projection, asRecord(received)?.content).dropped;
      } catch (error) {
        if (error instanceof UnsupportedRewriteError) {
          flags.rewrite = error;
        }
        throw error;
      }
      return hostStream(next, bridge, signal, flags);
    };

    this.counters.llmCalls += 1;
    let relayStream: Awaited<ReturnType<typeof nf.llmStreamCallExecute>>;
    try {
      relayStream = (await nf.withScopeStack(target.stack, () =>
        nf.llmStreamCallExecute(
          options.provider,
          { headers: {}, content: projection.content },
          produce,
          (chunk: unknown) => aggregator.add(chunk),
          () => aggregator.response(),
          target.parent,
          LLM_STREAMING,
          null,
          target.metadata,
          options.model,
          codec === undefined ? undefined : (request: LlmRequestShape) => codec.decode(request),
          codec === undefined
            ? undefined
            : (arg: { annotated: RelayJson; original: LlmRequestShape }) => codec.encode(arg.annotated, arg.original),
          codec === undefined ? undefined : (response: RelayJson) => codec.decodeResponse(response),
        ),
      )) as Awaited<ReturnType<typeof nf.llmStreamCallExecute>>;
    } catch (error) {
      yield* this.failedBeforeStream(error, flags, next);
      return;
    }

    try {
      for (;;) {
        const chunk = await relayStream.next();
        if (chunk === null) {
          break;
        }
        yield* bridge.release(chunk);
      }
      yield* bridge.drain('completed');
      this.afterModelCall(target, aggregator, flags.dropped);
    } catch (error) {
      yield* bridge.drain('failed');
      if (flags.hostError !== undefined) {
        throw flags.hostError;
      }
      if (!bridge.terminalDelivered) {
        if (!flags.funcCalled || flags.rewrite !== undefined) {
          yield* this.failedBeforeStream(error, flags, next);
          return;
        }
        this.counters.middlewareErrors += 1;
        yield failureFinish('NEMO_RELAY_STREAM_ERROR', `NeMo Relay stream failed: ${toMessage(error)}`);
      }
    } finally {
      // Stops the producer when the host stops reading early and waits for its cleanup.
      await relayStream.close().catch(() => undefined);
    }
  }

  /** Map a managed-call failure that happened before any host chunk was produced. */
  private async *failedBeforeStream(
    error: unknown,
    flags: { funcCalled: boolean; rewrite?: UnsupportedRewriteError },
    next: () => AsyncIterable<StreamChunk>,
  ): AsyncGenerator<StreamChunk> {
    const message = toMessage(error);
    if (GUARDRAIL_REJECTED.test(message)) {
      this.counters.guardrailRejections += 1;
      yield failureFinish(GUARDRAIL_CODE, `Blocked by NeMo Relay policy: ${message.replace(GUARDRAIL_REJECTED, '')}`);
      return;
    }
    if (flags.rewrite !== undefined) {
      this.counters.guardrailRejections += 1;
      yield failureFinish('NEMO_RELAY_REQUEST_REWRITE', flags.rewrite.message);
      return;
    }
    this.counters.middlewareErrors += 1;
    this.logger.warn(`NeMo Relay middleware failed before a model call; continuing without Relay: ${message}`);
    if (!flags.funcCalled) {
      yield* next();
    } else {
      yield failureFinish('NEMO_RELAY_STREAM_ERROR', `NeMo Relay middleware failed: ${message}`);
    }
  }

  private afterModelCall(target: LlmTarget, aggregator: CompletionAggregator, dropped: string[]): void {
    const response = aggregator.response();
    const choice = asRecord(Array.isArray(response.choices) ? response.choices[0] : undefined);
    const text = asRecord(choice?.message)?.content;
    if (typeof text === 'string' && text.length > 0 && target.session?.turn !== undefined && target.parent !== null) {
      target.session.turn.lastText = text;
    }
    if (dropped.length > 0 && target.session !== undefined) {
      this.mark(target.session, 'request_edit_dropped', { fields: dropped }, target);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* Tool calls                                                                */
  /* ------------------------------------------------------------------------ */

  private toolTarget(exec: ToolExecution): {
    stack: ScopeStack;
    parent: ScopeHandle | null;
    metadata: Record<string, JsonValue>;
  } {
    if (exec.agent === undefined) {
      this.harnessStack ??= this.nf().createScopeStack();
      return {
        stack: this.harnessStack,
        parent: null,
        metadata: { 'gen_ai.agent.name': AGENT_NAME, tool_call_id: exec.callId },
      };
    }
    const state = this.session(exec.agent.id, exec.agent.session);
    const parent = this.ensureTurn(state);
    const metadata: Record<string, JsonValue> = { ...this.sessionMetadata(state), tool_call_id: exec.callId };
    if (state.turn !== undefined) {
      metadata.turn_index = state.turn.index;
      if (state.step?.turn === state.turn.index) {
        metadata.step_index = state.step.step;
      }
    }
    if (exec.parent !== undefined && exec.rootCallId !== exec.callId) {
      metadata.parent_tool_call_id = exec.rootCallId;
    }
    return { stack: state.stack, parent, metadata };
  }

  /**
   * `tools/pre-execute` listener body. Managed execution runs Relay's guardrails itself;
   * this check only runs when DeepSeek Harness would otherwise ask the user to approve a
   * call that Relay policy is going to block.
   */
  async preExecute(exec: ToolExecution, next: () => Promise<PreToolDecision>): Promise<PreToolDecision> {
    const decision = await next();
    if (this.starting) {
      await this.settleStart(exec.signal);
    }
    if (!this.isActive()) {
      return decision.kind === 'allow' || decision.kind === 'ask' ? this.directRuleCheck(exec, decision) : decision;
    }
    if (decision.kind !== 'ask') {
      return decision;
    }
    const nf = this.nf();
    try {
      const target = this.toolTarget(exec);
      await nf.withScopeStack(target.stack, () => nf.toolConditionalExecution(exec.name, toJson(exec.arguments ?? {})));
      return decision;
    } catch (error) {
      const message = toMessage(error);
      if (!GUARDRAIL_REJECTED.test(message)) {
        this.counters.middlewareErrors += 1;
        return decision;
      }
      this.counters.guardrailRejections += 1;
      return guardrailDenial(message.replace(GUARDRAIL_REJECTED, ''));
    }
  }

  /**
   * While Relay is not active (the binding is still loading or failed to load),
   * managed calls do not run, so the integration's own rules are checked here.
   * A plugin disabled by configuration checks nothing.
   */
  private directRuleCheck(exec: ToolExecution, decision: PreToolDecision): PreToolDecision {
    if (!this.config.enabled || this.guardrail === undefined) {
      return decision;
    }
    let reason: string | null;
    try {
      reason = this.guardrail.evaluate(exec.name, toJson(exec.arguments ?? {}));
    } catch {
      reason = null;
    }
    if (reason === null) {
      return decision;
    }
    this.counters.guardrailRejections += 1;
    return guardrailDenial(reason);
  }

  /** `tools/execute` listener body: run the host tool body inside a managed Relay tool call. */
  async executeTool(exec: ToolExecution, next: () => Promise<HarnessToolResult>): Promise<HarnessToolResult> {
    if (this.starting) {
      await this.settleStart(exec.signal);
    }
    if (!this.isActive()) {
      return next();
    }
    const nf = this.nf();
    let target: { stack: ScopeStack; parent: ScopeHandle | null; metadata: Record<string, JsonValue> };
    try {
      target = this.toolTarget(exec);
    } catch (error) {
      this.counters.middlewareErrors += 1;
      this.logger.warn(`NeMo Relay could not attribute tool ${exec.name}: ${toMessage(error)}`);
      return next();
    }
    const args = toJson(exec.arguments ?? {});
    const token = randomUUID();
    const state: {
      host?: HarnessToolResult;
      wrapper?: ToolResultWrapper;
      rewrite: boolean;
      bodyStarted: boolean;
      bodyError?: unknown;
    } = { rewrite: false, bodyStarted: false };
    this.counters.toolCalls += 1;
    try {
      const returned = (await nf.withScopeStack(target.stack, () =>
        nf.toolCallExecuteAsync(
          exec.name,
          args,
          async (received: unknown) => {
            if (!jsonEqual(received, args)) {
              // DeepSeek Harness freezes tool arguments and logs them before execution.
              state.rewrite = true;
              throw new Error('NeMo Relay middleware rewrote the tool arguments');
            }
            state.bodyStarted = true;
            try {
              state.host = await next();
            } catch (error) {
              state.bodyError = error;
              throw error;
            }
            state.wrapper = wrapToolResult(state.host);
            return { result: state.wrapper, annotation: resultAnnotation(token) };
          },
          target.parent,
          0,
          null,
          target.metadata,
          exec.callId,
        ),
      )) as { result: unknown; annotation?: unknown };
      const edit = diffToolResult(state.wrapper, returned.result, token, returned.annotation);
      if (edit.kind === 'replace') {
        return edit.result;
      }
      if (edit.kind !== 'none') {
        this.toolEdits.set(exec, edit);
      }
      return state.host ?? toolFailure('NEMO_RELAY_TOOL_ERROR', 'NeMo Relay returned no tool result');
    } catch (error) {
      if (state.host !== undefined) {
        // Relay failed after the tool body ran; the host result stands.
        this.counters.middlewareErrors += 1;
        this.logger.warn(`NeMo Relay tool lifecycle failed after ${exec.name} ran: ${toMessage(error)}`);
        return state.host;
      }
      if (state.bodyStarted) {
        // Never run a tool body twice: surface the host failure as DeepSeek Harness would.
        throw state.bodyError ?? error;
      }
      const message = toMessage(error);
      if (GUARDRAIL_REJECTED.test(message)) {
        this.counters.guardrailRejections += 1;
        return toolFailure(GUARDRAIL_CODE, `Blocked by NeMo Relay policy: ${message.replace(GUARDRAIL_REJECTED, '')}`);
      }
      if (state.rewrite) {
        this.counters.guardrailRejections += 1;
        return toolFailure(
          'NEMO_RELAY_ARGUMENT_REWRITE',
          'NeMo Relay middleware rewrote the tool arguments, but DeepSeek Harness runs tools exactly as the model requested them; the call was blocked',
        );
      }
      this.counters.middlewareErrors += 1;
      this.logger.warn(`NeMo Relay middleware failed before ${exec.name}; continuing without Relay: ${message}`);
      return next();
    }
  }

  /** `tools/post-execute` listener body: apply result changes made by Relay middleware. */
  async postExecute(
    exec: ToolExecution,
    result: HarnessToolResult,
    next: () => Promise<PostToolDecision>,
  ): Promise<PostToolDecision> {
    const decision = await next();
    const edit = this.toolEdits.get(exec);
    this.toolEdits.delete(exec);
    if (edit === undefined || edit.kind === 'none' || edit.kind === 'replace' || decision.kind !== 'accept') {
      return decision;
    }
    const extra = decision.additionalContexts === undefined ? {} : { additionalContexts: decision.additionalContexts };
    if (edit.kind === 'value' && !result.isError) {
      return { kind: 'accept', value: edit.value, ...extra };
    }
    const text = edit.kind === 'content' ? edit.text : JSON.stringify(edit.value);
    return { kind: 'accept', content: [{ type: 'text', text }], ...extra };
  }

  /* ------------------------------------------------------------------------ */
  /* Registrations                                                             */
  /* ------------------------------------------------------------------------ */

  /**
   * Decide the document to activate: the user's `plugins` document plus the
   * integration's defaults, validated against the discovered `plugins.toml`
   * layers. A default that the layered configuration rejects is dropped, so a
   * default can never stop a valid user configuration from activating.
   */
  private planDocument(modules: NemoRelayModules): { document: NemoRelayPluginHostConfig; errors: string[] } {
    const user = this.config.plugins;
    const userReport = modules.pluginHost.validate(user as unknown as PluginConfig);
    const userErrors = errorDiagnostics(userReport.config.diagnostics);
    if (userErrors.length > 0) {
      this.recordReport(userReport.config.diagnostics, userReport.config_paths, userReport.resolved_config);
      const reason = 'the plugin configuration has errors';
      this.plan = { document: user, redaction: { applied: false, reason }, atif: { applied: false, reason } };
      return { document: user, errors: userErrors.map((diagnostic) => diagnostic.message) };
    }
    let plan = planPluginDocument(this.config, userReport.resolved_config, { modelName: this.defaultModelName() });
    let report = userReport;
    for (let attempt = 0; plan.document !== user && attempt < 2; attempt += 1) {
      const candidate = modules.pluginHost.validate(plan.document as unknown as PluginConfig);
      const errors = errorDiagnostics(candidate.config.diagnostics);
      if (errors.length === 0) {
        report = candidate;
        break;
      }
      // Drop the defaults the errors name; on the second failure, drop them all.
      const reason = `rejected by the layered plugin configuration: ${errors.map((error) => error.message).join('; ')}`;
      this.logger.warn(`NeMo Relay did not apply a default: ${reason}`);
      const named = new Set(errors.map((error) => error.component));
      const keepRedaction = attempt === 0 && plan.redaction.applied && !named.has('pii_redaction');
      const keepAtif = attempt === 0 && plan.atif.applied && !named.has('observability');
      plan = {
        document: {
          ...user,
          components: [
            ...user.components,
            ...(keepRedaction ? [secretRedactionComponent()] : []),
            ...(keepAtif && plan.atif.directory !== undefined
              ? [atifComponent(plan.atif.directory, plan.atif.modelName)]
              : []),
          ],
        },
        redaction: keepRedaction || !plan.redaction.applied ? plan.redaction : { applied: false, reason },
        atif: keepAtif || !plan.atif.applied ? plan.atif : { applied: false, reason },
      };
      if (!keepRedaction && !keepAtif) {
        plan = { ...plan, document: user };
      }
    }
    this.plan = plan;
    this.recordReport(report.config.diagnostics, report.config_paths, report.resolved_config);
    return { document: plan.document, errors: [] };
  }

  /** The host's default model, or `undefined` when the host cannot tell (never throws). */
  private defaultModelName(): string | undefined {
    try {
      const model = this.defaultModel?.();
      return typeof model === 'string' && model.length > 0 ? model : undefined;
    } catch {
      return undefined;
    }
  }

  private recordReport(diagnostics: ConfigDiagnostic[], configPaths: string[], resolvedConfig: unknown): void {
    this.configPaths = [...configPaths];
    this.resolvedConfig = resolvedConfig;
    for (const diagnostic of diagnostics) {
      const prefix = diagnostic.component ? `${diagnostic.component}: ` : '';
      const message = `NeMo Relay ${prefix}${diagnostic.code}: ${diagnostic.message}`;
      // Validation and activation report the same static diagnostics; log each once.
      if (this.loggedDiagnostics.has(message)) {
        continue;
      }
      this.loggedDiagnostics.add(message);
      if (diagnostic.level === 'error') {
        this.logger.warn(message);
      } else {
        this.logger.info(message);
      }
    }
  }

  /** Implement capture settings as observability-only sanitizers; managed calls still see raw payloads. */
  private registerCaptureSanitizers(modules: NemoRelayModules): void {
    const nf = modules.nf;
    const capture = this.config.capture;
    const register = (kind: string, add: (name: string) => void, remove: (name: string) => boolean): void => {
      const name = `${SANITIZER_PREFIX}${kind}.${this.subscriberName}`;
      add(name);
      this.disposers.push(() => void remove(name));
    };
    if (!capture.includePrompts) {
      register(
        'llm-request',
        (name) => nf.registerLlmSanitizeRequestGuardrail(name, Number.MAX_SAFE_INTEGER, () => null),
        (name) => nf.deregisterLlmSanitizeRequestGuardrail(name),
      );
    }
    if (!capture.includeResponses) {
      register(
        'llm-response',
        (name) => nf.registerLlmSanitizeResponseGuardrail(name, Number.MAX_SAFE_INTEGER, () => null),
        (name) => nf.deregisterLlmSanitizeResponseGuardrail(name),
      );
      register(
        'llm-chunk',
        (name) =>
          nf.registerMarkSanitizeGuardrail(name, Number.MIN_SAFE_INTEGER, (event: unknown, fields) =>
            asRecord(event as JsonValue)?.name === 'llm.chunk' ? { ...fields, data: null } : fields,
          ),
        (name) => nf.deregisterMarkSanitizeGuardrail(name),
      );
    }
    if (capture.stripToolArgs) {
      register(
        'tool-request',
        (name) =>
          nf.registerToolSanitizeRequestGuardrail(
            name,
            Number.MIN_SAFE_INTEGER,
            (_toolName: string, value: unknown) => {
              const record = asRecord(value as JsonValue);
              return { stripped: true, ...(record === undefined ? {} : { argKeys: Object.keys(record) }) };
            },
          ),
        (name) => nf.deregisterToolSanitizeRequestGuardrail(name),
      );
    }
    if (capture.stripToolResults) {
      register(
        'tool-response',
        (name) =>
          nf.registerToolSanitizeResponseGuardrail(
            name,
            Number.MIN_SAFE_INTEGER,
            (toolName: string, value: unknown) => {
              const wrapper = asRecord(value as JsonValue);
              return {
                content: `Tool ${toolName} ${wrapper?.is_error === true ? 'failed' : 'completed'}.`,
                stripped: true,
              };
            },
          ),
        (name) => nf.deregisterToolSanitizeResponseGuardrail(name),
      );
    }
  }

  /** Register the integration's rules as a Relay tool conditional-execution guardrail. */
  private registerGuardrail(modules: NemoRelayModules): void {
    const guardrail = this.guardrail;
    if (guardrail === undefined) {
      return;
    }
    const nf = modules.nf;
    try {
      // A previous plugin generation that did not shut down cleanly must not keep the name.
      nf.deregisterToolConditionalExecutionGuardrail(guardrail.name);
    } catch {
      // Not registered.
    }
    nf.registerToolConditionalExecutionGuardrail(
      guardrail.name,
      guardrail.priority,
      (toolName: string, args: unknown) => guardrail.evaluate(toolName, args),
    );
    this.disposers.push(() => void nf.deregisterToolConditionalExecutionGuardrail(guardrail.name));
  }

  private registerSubscriber(modules: NemoRelayModules): void {
    const onEvent = this.onEvent;
    if (onEvent === undefined) {
      return;
    }
    const name = this.subscriberName;
    modules.nf.registerSubscriber(name, (event: unknown) => {
      try {
        onEvent(event);
      } catch (error) {
        this.logger.warn(`NeMo Relay view subscriber failed: ${toMessage(error)}`);
      }
    });
    this.disposers.push(() => void modules.nf.deregisterSubscriber(name));
  }

  private nf(): NemoRelayModules['nf'] {
    if (this.modules === undefined) {
      throw new Error('NeMo Relay runtime is not started');
    }
    return this.modules.nf;
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Pull host chunks for the managed stream. Host chunks are queued in the bridge before
 * their projections reach Relay, so the consumer can release them in order.
 */
async function* hostStream(
  next: () => AsyncIterable<StreamChunk>,
  bridge: ChunkBridge,
  signal: AbortSignal,
  flags: { hostError?: unknown },
): AsyncGenerator<JsonValue> {
  let iterator: AsyncIterator<StreamChunk>;
  try {
    iterator = next()[Symbol.asyncIterator]();
  } catch (error) {
    flags.hostError = error;
    throw error;
  }
  const aborted = new Promise<'aborted'>((resolve) => {
    if (signal.aborted) {
      resolve('aborted');
    } else {
      signal.addEventListener('abort', () => resolve('aborted'), { once: true });
    }
  });
  let failure: { message: string; code: string } | undefined;
  try {
    for (;;) {
      const step = await Promise.race([iterator.next(), aborted]);
      if (step === 'aborted') {
        void iterator.return?.()?.catch(() => undefined);
        return;
      }
      if (step.done === true) {
        break;
      }
      const chunk = step.value;
      if (chunk.type === 'finish' && chunk.reason.kind === 'error') {
        failure = { message: chunk.reason.failure.message, code: chunk.reason.failure.code };
      }
      const projection = bridge.accept(chunk);
      if (projection !== undefined) {
        yield projection;
      }
    }
  } catch (error) {
    flags.hostError = error;
    throw error;
  }
  if (failure !== undefined) {
    // The finish chunk is already queued for the host; failing here records the span as an error.
    throw new Error(`${failure.code}: ${failure.message}`);
  }
}

function guardrailDenial(reason: string): PreToolDecision {
  return {
    kind: 'deny',
    reason: `Blocked by NeMo Relay policy: ${reason}`,
    info: { name: 'NemoRelayGuardrail', code: GUARDRAIL_CODE, reason },
  };
}

function failureFinish(code: string, message: string): StreamChunk {
  return { type: 'finish', reason: { kind: 'error', failure: { message, code } } };
}

function toolFailure(code: string, message: string): HarnessToolResult {
  return {
    isError: true,
    error: { message, info: { name: 'NemoRelay', code, reason: message } },
    content: [{ type: 'text', text: `Error: ${message}` }],
  };
}

function errorDiagnostics(diagnostics: ConfigDiagnostic[]): ConfigDiagnostic[] {
  return diagnostics.filter((diagnostic) => diagnostic.level === 'error');
}

function isStepNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/** OpenTelemetry status of a turn that DeepSeek Harness ended: completed turns succeeded. */
function statusForTurnEnd(kind: string): 'OK' | 'ERROR' | 'UNSET' {
  switch (kind) {
    case 'completed':
      return 'OK';
    case 'error':
      return 'ERROR';
    default:
      // Cancelled (`aborted`), refused by a step listener (`blocked`), or a kind this version does not know.
      return 'UNSET';
  }
}

function scopeTypeCustom(nf: NemoRelayModules['nf']): Parameters<NemoRelayModules['nf']['pushScope']>[1] {
  return (nf.ScopeType?.Custom ?? 9) as Parameters<NemoRelayModules['nf']['pushScope']>[1];
}

function messageText(content: unknown): string {
  if (!Array.isArray(content)) {
    return '';
  }
  return content
    .map((block) => {
      const record = asRecord(block as JsonValue);
      if (record?.type === 'text' && typeof record.text === 'string') {
        return record.text;
      }
      return typeof record?.type === 'string' ? `[${record.type}]` : '';
    })
    .filter((text) => text.length > 0)
    .join('\n');
}

export function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Logging is best effort: a failing logger must never break the host's model or tool calls. */
function guardLogger(logger: RuntimeLogger | undefined): RuntimeLogger {
  const call = (level: keyof RuntimeLogger, message: string): void => {
    try {
      logger?.[level](message);
    } catch {
      // Ignore logger failures.
    }
  };
  return { info: (message) => call('info', message), warn: (message) => call('warn', message) };
}
