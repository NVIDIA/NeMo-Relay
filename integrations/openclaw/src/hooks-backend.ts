// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main OpenClaw hook replay dispatcher.
 *
 * OpenClaw hook callbacks arrive here as lifecycle, LLM, model-timing, tool, and
 * subagent events. This class routes each event to focused replay modules and
 * owns fail-open behavior so observability never breaks the agent runtime.
 */
import type { NemoRelayHookBackendConfig } from './config.js';
import { emitMark, toJsonRecord } from './hook-replay/marks.js';
import { llmKey, nowMicros } from './hook-replay/correlation.js';
import {
  emitUnpairedModelCallTimingMarks,
  recordBeforeMessageWrite,
  recordLlmInput,
  recordLlmOutput,
  recordModelCallEnded,
  recordModelCallStarted,
  replayAgentEndMessages,
  replayPendingLlmOutputsForSession,
} from './hook-replay/llm.js';
import {
  guardBeforeToolCall,
  replayAfterToolCall,
  startEagerSkillToolCall,
  type EagerSkillToolCall,
} from './hook-replay/tool.js';
import { detectSkillLoads } from './hook-replay/skill-load.js';
import {
  createHookReplayState,
  drainSession,
  closeSessionRoot,
  deleteSession,
  ensureSession,
  materializeSessionRoot,
  queueCapturedEmit,
  resolveSessionOwnerKey,
  type HookReplayBackendState,
  type SessionLookupInput,
  type SessionState,
} from './hook-replay/session.js';
import type { NemoRelayRuntimeModule } from './modules.js';
import type {
  PluginHookAfterToolCallEvent,
  PluginHookAgentContext,
  PluginHookAgentEndEvent,
  PluginHookBeforeAgentFinalizeEvent,
  PluginHookBeforeMessageWriteContext,
  PluginHookBeforeMessageWriteEvent,
  PluginHookBeforeToolCallEvent,
  PluginHookGatewayContext,
  PluginHookGatewayStartEvent,
  PluginHookLlmInputEvent,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
  PluginHookModelCallStartedEvent,
  PluginHookSessionContext,
  PluginHookSessionEndEvent,
  PluginHookSessionStartEvent,
  PluginHookSubagentContext,
  PluginHookSubagentEndedEvent,
  PluginHookSubagentSpawnedEvent,
  PluginHookToolContext,
} from './openclaw-hook-types.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';
import type { JsonObject as JsonRecord } from 'nemo-relay-node/typed';

export type HookReplayBackendOptions = {
  nf: NemoRelayRuntimeModule;
  config: NemoRelayHookBackendConfig;
  logger: PluginLogger;
  agentVersion: string;
};

type PendingSubagentLineage = {
  childSessionKey: string;
  requesterSessionKey: string;
  runId?: string;
  agentId?: string;
  observedAtMs: number;
};

/** Replays OpenClaw public hook events into NeMo Relay scopes, spans, and marks. */
export class HookReplayBackend {
  private readonly nf: NemoRelayRuntimeModule;
  private readonly config: NemoRelayHookBackendConfig;
  private readonly logger: PluginLogger;
  private readonly agentVersion: string;
  private readonly stateValue = createHookReplayState();
  private readonly warningCounts = new Map<string, number>();
  private readonly pendingSubagentLineageByChildSessionKey = new Map<string, PendingSubagentLineage>();
  private readonly pendingSubagentChildKeyByRunId = new Map<string, string>();
  private readonly eagerSkillLoadsByToolCall = new Map<string, EagerSkillToolCall>();

  constructor(options: HookReplayBackendOptions) {
    this.nf = options.nf;
    this.config = options.config;
    this.logger = options.logger;
    this.agentVersion = options.agentVersion;
  }

  /** Return mutable replay state for tests and health snapshots. */
  state(): HookReplayBackendState {
    return this.stateValue;
  }

  /** Keep gateway_start registered even though session roots are created lazily. */
  onGatewayStart(_event: PluginHookGatewayStartEvent, _ctx: PluginHookGatewayContext): void {
    // Gateway events have no session root in the hook backend. Keep this hook
    // registered so later telemetry lifecycle can attach without changing the shell.
  }

  /** Open or alias an explicit OpenClaw session root. */
  onSessionStart(event: PluginHookSessionStartEvent, ctx: PluginHookSessionContext): void {
    const observedAtMicros = nowMicros();
    const session = this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      agentId: ctx.agentId,
      source: 'session_start',
      resumedFrom: event.resumedFrom,
      timestamp: observedAtMicros,
    });

    this.promoteDeferredSubagentSessionsForRequester(session?.sessionKey ?? event.sessionKey ?? ctx.sessionKey);

    // ensureSession opens the root scope and emits openclaw.session_start for both explicit and lazy sessions.
  }

  /** Close one explicit OpenClaw session and export its ATIF artifact. */
  async onSessionEnd(event: PluginHookSessionEndEvent, ctx: PluginHookSessionContext): Promise<void> {
    const session = this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      agentId: ctx.agentId,
      source: 'lazy_session',
    });

    if (!session) {
      return;
    }

    await this.closeSession(
      session,
      sessionEndSummary(event),
      toJsonRecord({
        source: 'openclaw.session_end',
        hook_event_name: 'session_end',
        sessionId: session.sessionId,
        sessionKey: event.sessionKey ?? ctx.sessionKey,
        agentId: ctx.agentId,
      }),
    );
  }

  /** Buffer an LLM request snapshot until a matching response or trajectory replay arrives. */
  onLlmInput(event: PluginHookLlmInputEvent, ctx: PluginHookAgentContext): void {
    recordLlmInput(this.sessionManager(), event, ctx);
  }

  /** Replay an LLM output immediately or keep it briefly for a late input snapshot. */
  onLlmOutput(event: PluginHookLlmOutputEvent, ctx: PluginHookAgentContext): void {
    recordLlmOutput(this.sessionManager(), event, ctx);
  }

  /** Record provider-call start timing when OpenClaw exposes a call id. */
  onModelCallStarted(event: PluginHookModelCallStartedEvent, ctx: PluginHookAgentContext): void {
    recordModelCallStarted(this.sessionManager(), event, ctx);
  }

  /** Record provider-call completion timing for later LLM-span correlation. */
  onModelCallEnded(event: PluginHookModelCallEndedEvent, ctx: PluginHookAgentContext): void {
    recordModelCallEnded(this.sessionManager(), event, ctx);
  }

  /** Replay a finished OpenClaw tool call as a NeMo Relay tool span or blocked mark. */
  onAfterToolCall(event: PluginHookAfterToolCallEvent, ctx: PluginHookToolContext): void {
    replayAfterToolCall(this.sessionManager(), event, ctx, this.consumeEagerSkillToolCall(event, ctx));
  }

  /** Run conditional-execution guardrails before OpenClaw invokes a tool. */
  async onBeforeToolCall(event: PluginHookBeforeToolCallEvent, ctx: PluginHookToolContext): Promise<void> {
    const session = await guardBeforeToolCall(this.sessionManager(), event, ctx);
    const toolCallId = event.toolCallId ?? ctx.toolCallId;
    if (!session || !toolCallId) {
      return;
    }
    const detections = detectSkillLoads(event.toolName, event.params);
    if (detections.length === 0) {
      return;
    }

    const observedAtMicros = nowMicros();
    this.pruneEagerSkillLoads(Math.trunc(observedAtMicros / 1000));
    const eagerToolCall = startEagerSkillToolCall(
      this.sessionManager(),
      event,
      ctx,
      session,
      detections,
      observedAtMicros,
    );
    if (eagerToolCall) {
      this.eagerSkillLoadsByToolCall.set(this.skillToolCallKey(session.ownerKey, toolCallId), eagerToolCall);
    }
  }

  /** Capture assistant message writes that may contain the clearest provider output. */
  onBeforeMessageWrite(event: PluginHookBeforeMessageWriteEvent, ctx: PluginHookBeforeMessageWriteContext): void {
    recordBeforeMessageWrite(this.sessionManager(), event, ctx);
  }

  /** Finalize one agent run, replaying message-write trajectory when needed. */
  onAgentEnd(event: PluginHookAgentEndEvent, ctx: PluginHookAgentContext): void {
    const observedAtMicros = nowMicros();
    const session = this.ensureSession({
      sessionId: ctx.sessionId,
      sessionKey: ctx.sessionKey,
      runId: event.runId ?? ctx.runId,
      agentId: ctx.agentId,
      source: 'lazy_session',
      timestamp: observedAtMicros,
    });

    if (!session) {
      return;
    }

    const finalOutput = replayAgentEndMessages(this.sessionManager(), event, ctx, session);
    if (finalOutput && (!session.finalOutput || 'content' in finalOutput)) {
      session.finalOutput = finalOutput;
    }

    this.emitSessionMark(
      'openclaw.agent_end',
      session,
      toJsonRecord({
        runId: event.runId ?? ctx.runId,
        success: event.success,
        error: event.error,
        durationMs: event.durationMs,
        messageCount: event.messages.length,
      }),
      observedAtMicros,
    );
  }

  /** Remember the last assistant text before OpenClaw finalizes the response. */
  onBeforeAgentFinalize(event: PluginHookBeforeAgentFinalizeEvent, ctx: PluginHookAgentContext): void {
    const observedAtMicros = nowMicros();
    const session = this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      runId: event.runId ?? ctx.runId,
      agentId: ctx.agentId,
      source: 'lazy_session',
      timestamp: observedAtMicros,
    });

    if (!session) {
      return;
    }

    if (typeof event.lastAssistantMessage === 'string' && event.lastAssistantMessage.length > 0) {
      session.finalOutput = toJsonRecord({
        content: event.lastAssistantMessage,
        source: 'openclaw.before_agent_finalize',
        runId: event.runId ?? ctx.runId,
      });
    }

    this.emitSessionMark(
      'openclaw.before_agent_finalize',
      session,
      toJsonRecord({
        runId: event.runId ?? ctx.runId,
        turnId: event.turnId,
        provider: event.provider,
        model: event.model,
        cwd: event.cwd,
        transcriptPath: event.transcriptPath,
        stopHookActive: event.stopHookActive,
        messageCount: event.messages?.length,
      }),
      observedAtMicros,
    );
  }

  /** Attach subagent spawn metadata to the requester session when possible. */
  onSubagentSpawned(event: PluginHookSubagentSpawnedEvent, ctx: PluginHookSubagentContext): void {
    const observedAtMicros = nowMicros();
    this.trackPendingSubagentLineage(event, ctx, Math.trunc(observedAtMicros / 1000));
    const requesterSession = this.ensureRequesterSessionAnchor(ctx.requesterSessionKey, observedAtMicros);
    const childSessionKey = ctx.childSessionKey ?? event.childSessionKey;
    const session =
      requesterSession ??
      this.ensureSession({
        childSessionKey,
        runId: ctx.runId ?? event.runId,
        agentId: event.agentId,
        source: 'lazy_session',
        timestamp: observedAtMicros,
      });

    if (!session) {
      return;
    }

    this.emitSessionMark(
      'openclaw.subagent_spawned',
      session,
      toJsonRecord({
        runId: event.runId,
        childSessionKey: event.childSessionKey,
        agentId: event.agentId,
        label: event.label,
        mode: event.mode,
        threadRequested: event.threadRequested,
      }),
      observedAtMicros,
    );

    if (!ctx.requesterSessionKey || requesterSession?.rootHandle) {
      this.promoteDeferredSubagentSession(event.childSessionKey);
    }
  }

  /** Attach subagent completion metadata to the requester or child session. */
  onSubagentEnded(event: PluginHookSubagentEndedEvent, ctx: PluginHookSubagentContext): void {
    const observedAtMicros = nowMicros();
    const session =
      this.ensureSession({
        requesterSessionKey: ctx.requesterSessionKey,
        source: 'lazy_session',
        timestamp: observedAtMicros,
      }) ??
      this.ensureSession({
        childSessionKey: ctx.childSessionKey ?? event.targetSessionKey,
        runId: ctx.runId ?? event.runId,
        source: 'lazy_session',
        timestamp: observedAtMicros,
      });

    if (!session) {
      return;
    }

    this.materializeDeferredSessionRoot(session);

    this.emitSessionMark(
      'openclaw.subagent_ended',
      session,
      toJsonRecord({
        runId: event.runId ?? ctx.runId,
        targetSessionKey: event.targetSessionKey,
        targetKind: event.targetKind,
        reason: event.reason,
        outcome: event.outcome,
        error: event.error,
        endedAt: event.endedAt,
        sendFarewell: event.sendFarewell,
        accountId: event.accountId,
      }),
      observedAtMicros,
    );
  }

  /** Drain all active sessions when the OpenClaw gateway is stopping. */
  async drainForGatewayStop(reason?: string): Promise<void> {
    await this.closeAllSessions({ reason: reason ?? 'gateway_stop' });
  }

  /** Close one session selected by a runtime lifecycle cleanup hook. */
  async cleanupSession(input: SessionLookupInput & { reason: string }): Promise<void> {
    const ownerKey = resolveSessionOwnerKey(this.stateValue, input);
    if (!ownerKey) {
      return;
    }

    const session = this.stateValue.sessions.get(ownerKey);
    if (!session) {
      return;
    }

    await this.closeSession(session, { reason: input.reason });
  }

  /** Stop the backend and close every active session. */
  async stop(reason: string): Promise<void> {
    await this.closeAllSessions({ reason });
  }

  /** Run replay code with bounded warning logs and no exception escape. */
  safeReplay(label: string, session: SessionState | undefined, emit: () => void): void {
    try {
      emit();
    } catch (error) {
      this.stateValue.counters.replayErrors += 1;
      this.logBoundedWarn(
        `safe-replay:${label}`,
        `nemo-relay replay failed: label=${label} session=${session?.sessionId ?? 'unknown'} error=${toMessage(error)}`,
      );
    }
  }

  /** Async variant of safeReplay for hooks that need export or cleanup awaits. */
  async safeReplayAsync(label: string, session: SessionState | undefined, emit: () => Promise<void>): Promise<void> {
    try {
      await emit();
    } catch (error) {
      this.stateValue.counters.replayErrors += 1;
      this.logBoundedWarn(
        `safe-replay:${label}`,
        `nemo-relay async replay failed: label=${label} session=${session?.sessionId ?? 'unknown'} error=${toMessage(error)}`,
      );
    }
  }

  /** Emit spans/marks under the stored session scope stack and ATIF capture window. */
  emitCapturedUnderSession(label: string, session: SessionState, emit: () => void): void {
    if (queueCapturedEmit(session, label, emit)) {
      return;
    }

    this.safeReplay(label, session, () => {
      const previousStack = this.nf.currentScopeStack();
      try {
        this.nf.setThreadScopeStack(session.stack);
        emit();
      } finally {
        this.nf.setThreadScopeStack(previousStack);
      }
    });
  }

  /** Force any pending LLM outputs for a session to replay before closure. */
  replayPendingLlmOutputsForSession(session: SessionState, options: { allowPlaceholderRequest: boolean }): void {
    replayPendingLlmOutputsForSession(this.sessionManager(), session, options);
  }

  /** Emit model-call timing diagnostics that could not be paired with an LLM span. */
  emitUnpairedModelCallTimingMarks(session: SessionState): void {
    emitUnpairedModelCallTimingMarks(this.sessionManager(), session);
  }

  /** Create or resolve a session through the shared session manager facade. */
  private ensureSession(input: Parameters<typeof ensureSession>[1]): SessionState | undefined {
    return ensureSession(this.sessionManager(), input);
  }

  /** Drain, close, export, and delete one session. */
  private async closeSession(session: SessionState, summary: JsonRecord, metadata?: JsonRecord): Promise<void> {
    this.materializeDeferredSessionRoot(session);
    for (const [key, record] of this.eagerSkillLoadsByToolCall) {
      if (key.startsWith(`${session.ownerKey}\u0000`)) {
        this.closeEagerSkillToolCall(session, record, 'session_closed', nowMicros());
        this.eagerSkillLoadsByToolCall.delete(key);
      }
    }
    drainSession(this.sessionManager(), session);
    closeSessionRoot(this.sessionManager(), session, summary, session.finalOutput ?? summary, metadata);
    this.flushSubscriberDelivery('session_close');
    this.forgetPendingSubagentLineage(session);
    deleteSession(this.stateValue, session);
  }

  /** Consume the tool span already started by tool-call middleware. */
  private consumeEagerSkillToolCall(
    event: PluginHookAfterToolCallEvent,
    ctx: PluginHookToolContext,
  ): EagerSkillToolCall | undefined {
    const toolCallId = event.toolCallId ?? ctx.toolCallId;
    const ownerKey = resolveSessionOwnerKey(this.stateValue, {
      sessionId: ctx.sessionId,
      sessionKey: ctx.sessionKey,
      runId: event.runId ?? ctx.runId,
    });
    if (!toolCallId || !ownerKey) {
      return undefined;
    }
    const record = this.eagerSkillLoadsByToolCall.get(this.skillToolCallKey(ownerKey, toolCallId));
    this.eagerSkillLoadsByToolCall.delete(this.skillToolCallKey(ownerKey, toolCallId));
    return record;
  }

  private skillToolCallKey(ownerKey: string, toolCallId: string): string {
    return `${ownerKey}\u0000${toolCallId}`;
  }

  private pruneEagerSkillLoads(nowMs: number): void {
    for (const [key, record] of this.eagerSkillLoadsByToolCall) {
      if (nowMs - record.observedAtMs > 5 * 60 * 1000) {
        const [ownerKey] = key.split('\u0000', 1);
        const session = ownerKey ? this.stateValue.sessions.get(ownerKey) : undefined;
        if (session) {
          this.closeEagerSkillToolCall(session, record, 'after_tool_call_timeout', nowMs * 1000);
        }
        this.eagerSkillLoadsByToolCall.delete(key);
      }
    }
  }

  private closeEagerSkillToolCall(
    session: SessionState,
    record: EagerSkillToolCall,
    reason: string,
    timestamp: number,
  ): void {
    this.emitCapturedUnderSession('eager_skill_tool_cleanup', session, () => {
      this.nf.toolCallEnd(
        record.handle,
        toJsonRecord({ content: 'Skill-read tool ended without an after_tool_call event.', reason }),
        null,
        toJsonRecord({ source: 'openclaw.skill_load_cleanup', reason }),
        timestamp,
      );
      this.stateValue.counters.toolSpansReplayed += 1;
    });
  }

  /** Emit a session-level OpenClaw lifecycle mark. */
  private emitSessionMark(name: string, session: SessionState, data: JsonRecord, timestampMicros?: number): void {
    this.emitCapturedUnderSession(name, session, () => {
      const params: Parameters<typeof emitMark>[0] = {
        nf: this.nf,
        state: this.stateValue,
        session,
        name,
        data,
        metadata: toJsonRecord({
          source: name,
          hook_event_name: name.startsWith('openclaw.') ? name.slice('openclaw.'.length) : undefined,
          sessionId: session.sessionId,
          sessionKey: session.sessionKey,
          agentId: session.agentId,
          runId: typeof data.runId === 'string' ? data.runId : undefined,
        }),
      };

      if (timestampMicros !== undefined) {
        params.timestamp = timestampMicros;
      }

      emitMark(params);
    });
  }

  /** Close every active session with the same lifecycle summary. */
  private async closeAllSessions(summary: JsonRecord): Promise<void> {
    for (const session of [...this.stateValue.sessions.values()]) {
      await this.closeSession(session, summary);
    }
  }

  /** Wait for native subscriber/exporter delivery after a replay closure boundary. */
  private flushSubscriberDelivery(label: string): void {
    try {
      this.nf.flushSubscribers?.();
    } catch (error) {
      this.logBoundedWarn(
        `flush-subscribers:${label}`,
        `nemo-relay subscriber flush failed: label=${label} error=${toMessage(error)}`,
      );
    }
  }

  /** Build the narrow manager interface consumed by focused replay modules. */
  private sessionManager() {
    return {
      nf: this.nf,
      config: this.config,
      logger: this.logger,
      state: this.stateValue,
      agentVersion: this.agentVersion,
      emitCapturedUnderSession: (label: string, session: SessionState, emit: () => void) =>
        this.emitCapturedUnderSession(label, session, emit),
      replayPendingLlmOutputsForSession: (session: SessionState, options: { allowPlaceholderRequest: boolean }) =>
        this.replayPendingLlmOutputsForSession(session, options),
      emitUnpairedModelCallTimingMarks: (session: SessionState) => this.emitUnpairedModelCallTimingMarks(session),
      logBoundedWarn: (key: string, message: string) => this.logBoundedWarn(key, message),
      resolveSessionRootContext: (input: Parameters<typeof ensureSession>[1]) => this.resolveSessionRootContext(input),
    };
  }

  /** Prefer nested child scopes only when the hook surface provides real subagent lineage. */
  private resolveSessionRootContext(input: Parameters<typeof ensureSession>[1]): Partial<Parameters<typeof ensureSession>[1]> | undefined {
    this.pruneExpiredPendingSubagentLineage(microsToMs(input.timestamp) ?? Date.now());
    const lineage = this.resolvePendingSubagentLineage(input);
    if (lineage) {
      const parentSession = this.resolveTrackedSession({ requesterSessionKey: lineage.requesterSessionKey });
      return {
        childSessionKey: input.childSessionKey ?? lineage.childSessionKey,
        runId: input.runId ?? lineage.runId,
        agentId: input.agentId ?? lineage.agentId,
        scopeRole: 'subagent',
        parentHandle: parentSession?.rootHandle,
        deferRootOpen: input.deferRootOpen ?? (parentSession?.rootHandle ? false : true),
      };
    }

    if (this.isDocumentedSubagentSessionKey(input.sessionKey ?? input.childSessionKey)) {
      return {
        scopeRole: 'subagent',
        deferRootOpen: input.deferRootOpen ?? true,
      };
    }

    return undefined;
  }

  /** Track stable parent/child lineage from subagent hooks until child session hooks can use it. */
  private trackPendingSubagentLineage(
    event: PluginHookSubagentSpawnedEvent,
    ctx: PluginHookSubagentContext,
    observedAtMs: number,
  ): void {
    this.pruneExpiredPendingSubagentLineage(observedAtMs);
    const requesterSessionKey = ctx.requesterSessionKey?.trim();
    const childSessionKey = (ctx.childSessionKey ?? event.childSessionKey)?.trim();
    if (!requesterSessionKey || !childSessionKey) {
      return;
    }

    this.pendingSubagentLineageByChildSessionKey.set(childSessionKey, {
      childSessionKey,
      requesterSessionKey,
      runId: ctx.runId ?? event.runId,
      agentId: event.agentId,
      observedAtMs,
    });
    if (ctx.runId ?? event.runId) {
      this.pendingSubagentChildKeyByRunId.set(ctx.runId ?? event.runId, childSessionKey);
    }
  }

  /** Resolve the requester session if it exists, or seed a deferred lazy root placeholder for later promotion. */
  private ensureRequesterSessionAnchor(requesterSessionKey: string | undefined, timestampMicros?: number): SessionState | undefined {
    const trimmedRequesterSessionKey = requesterSessionKey?.trim();
    if (!trimmedRequesterSessionKey) {
      return undefined;
    }

    return (
      this.resolveTrackedSession({ requesterSessionKey: trimmedRequesterSessionKey }) ??
      this.ensureSession({
        sessionKey: trimmedRequesterSessionKey,
        requesterSessionKey: trimmedRequesterSessionKey,
        source: 'lazy_session',
        timestamp: timestampMicros,
        deferRootOpen: true,
      })
    );
  }

  /** Open a deferred child session root once the requester scope is known. */
  private promoteDeferredSubagentSession(childSessionKey: string): void {
    const session = this.resolveTrackedSession({ sessionKey: childSessionKey, childSessionKey });
    if (!session) {
      return;
    }

    this.materializeDeferredSessionRoot(session);
  }

  /** Promote any deferred child sessions waiting on a requester root once it exists. */
  private promoteDeferredSubagentSessionsForRequester(requesterSessionKey: string | undefined): void {
    const trimmedRequesterSessionKey = requesterSessionKey?.trim();
    if (!trimmedRequesterSessionKey) {
      return;
    }

    const requesterSession = this.resolveTrackedSession({ requesterSessionKey: trimmedRequesterSessionKey });
    if (!requesterSession?.rootHandle) {
      return;
    }

    for (const lineage of this.pendingSubagentLineageByChildSessionKey.values()) {
      if (lineage.requesterSessionKey === trimmedRequesterSessionKey) {
        this.promoteDeferredSubagentSession(lineage.childSessionKey);
      }
    }
  }

  /** Materialize one deferred session root with nested lineage when available. */
  private materializeDeferredSessionRoot(session: SessionState): void {
    if (session.rootHandle) {
      return;
    }

    this.pruneExpiredPendingSubagentLineage(Date.now());
    const lineage = this.resolvePendingSubagentLineage({
      sessionId: session.sessionId,
      sessionKey: session.sessionKey,
    });
    const parentSession =
      lineage === undefined
        ? undefined
        : this.ensureRequesterSessionAnchor(lineage.requesterSessionKey, session.pendingRootTimestampMicros);
    if (parentSession && !parentSession.rootHandle) {
      materializeSessionRoot(this.sessionManager(), parentSession, {
        sessionId: parentSession.sessionId,
        sessionKey: parentSession.sessionKey ?? lineage?.requesterSessionKey,
        source: parentSession.source,
        resumedFrom: parentSession.resumedFrom,
        timestamp: parentSession.pendingRootTimestampMicros,
      });
    }
    const parentHandle = parentSession?.rootHandle;

    materializeSessionRoot(this.sessionManager(), session, {
      sessionId: session.sessionId,
      sessionKey: session.sessionKey,
      runId: lineage?.runId,
      agentId: session.agentId ?? lineage?.agentId,
      source: session.source,
      resumedFrom: session.resumedFrom,
      scopeRole: session.scopeRole,
      parentHandle,
    });
  }

  /** Resolve stable subagent lineage from child session key first, then run id as a fallback. */
  private resolvePendingSubagentLineage(input: SessionLookupInput): PendingSubagentLineage | undefined {
    const childSessionKey = [input.sessionKey, input.childSessionKey]
      .find((value): value is string => this.isDocumentedSubagentSessionKey(value))
      ?.trim();
    if (childSessionKey) {
      return this.pendingSubagentLineageByChildSessionKey.get(childSessionKey);
    }

    const runChildSessionKey =
      typeof input.runId === 'string' && input.runId.length > 0 ? this.pendingSubagentChildKeyByRunId.get(input.runId) : undefined;
    return runChildSessionKey === undefined ? undefined : this.pendingSubagentLineageByChildSessionKey.get(runChildSessionKey);
  }

  /** Drop stale pending lineage entries so abandoned subagent spawns do not accumulate indefinitely. */
  private pruneExpiredPendingSubagentLineage(nowMs: number): void {
    const ttlMs = this.config.correlation.recordTtlMs;
    for (const [childSessionKey, lineage] of this.pendingSubagentLineageByChildSessionKey) {
      if (nowMs - lineage.observedAtMs > ttlMs) {
        this.pendingSubagentLineageByChildSessionKey.delete(childSessionKey);
      }
    }

    for (const [runId, childSessionKey] of this.pendingSubagentChildKeyByRunId) {
      if (!this.pendingSubagentLineageByChildSessionKey.has(childSessionKey)) {
        this.pendingSubagentChildKeyByRunId.delete(runId);
      }
    }
  }

  /** Resolve one session through the same alias map used by the replay state. */
  private resolveTrackedSession(input: SessionLookupInput): SessionState | undefined {
    const ownerKey = resolveSessionOwnerKey(this.stateValue, input);
    return ownerKey === undefined ? undefined : this.stateValue.sessions.get(ownerKey);
  }

  /** Free lineage bookkeeping once the child session is closed. */
  private forgetPendingSubagentLineage(session: SessionState): void {
    const childSessionKey = session.sessionKey;
    if (childSessionKey) {
      this.pendingSubagentLineageByChildSessionKey.delete(childSessionKey);
    }

    for (const [runId, trackedChildSessionKey] of this.pendingSubagentChildKeyByRunId) {
      if (trackedChildSessionKey === childSessionKey) {
        this.pendingSubagentChildKeyByRunId.delete(runId);
      }
    }
  }

  /** Match the documented native subagent session key shape without depending on private OpenClaw internals. */
  private isDocumentedSubagentSessionKey(value?: string): value is string {
    return typeof value === 'string' && /^agent:[^:]+:subagent:/.test(value);
  }

  /** Log one warning per key to avoid noisy repeated hook failures. */
  private logBoundedWarn(key: string, message: string): void {
    const count = this.warningCounts.get(key) ?? 0;
    this.warningCounts.set(key, count + 1);
    if (count === 0) {
      this.logger.warn?.(message);
    }
  }
}

export { llmKey };

/** Expose owner-key resolution for tests without exporting the full session module. */
export function resolveBackendSessionOwnerKey(
  state: HookReplayBackendState,
  input: Parameters<typeof resolveSessionOwnerKey>[1],
): string | undefined {
  return resolveSessionOwnerKey(state, input);
}

function microsToMs(timestampMicros: number | undefined): number | undefined {
  return timestampMicros === undefined ? undefined : Math.trunc(timestampMicros / 1000);
}

/** Build the lifecycle summary stored as the session_end mark payload. */
function sessionEndSummary(event: PluginHookSessionEndEvent): JsonRecord {
  return toJsonRecord({
    sessionId: event.sessionId,
    sessionKey: event.sessionKey,
    messageCount: event.messageCount,
    durationMs: event.durationMs,
    reason: event.reason,
    sessionFile: event.sessionFile,
    transcriptArchived: event.transcriptArchived,
    nextSessionId: event.nextSessionId,
    nextSessionKey: event.nextSessionKey,
  });
}

/** Convert thrown values into stable log strings. */
function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
