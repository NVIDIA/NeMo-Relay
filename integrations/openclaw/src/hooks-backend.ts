// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig } from "./config.js";
import { exportAtifJson, withAtifCapture } from "./atif-capture.js";
import { emitMark, toJsonRecord } from "./hook-replay/marks.js";
import { llmKey } from "./hook-replay/correlation.js";
import {
  emitUnpairedModelCallTimingMarks,
  recordLlmInput,
  recordLlmOutput,
  recordModelCallEnded,
  recordModelCallStarted,
  replayPendingLlmOutputsForSession,
} from "./hook-replay/llm.js";
import { replayAfterToolCall } from "./hook-replay/tool.js";
import {
  createHookReplayState,
  drainSession,
  closeSessionRoot,
  deleteSession,
  ensureSession,
  resolveSessionKey,
  type HookReplayBackendState,
  type SessionState,
} from "./hook-replay/session.js";
import type { NemoFlowRuntimeModule } from "./modules.js";
import type {
  PluginHookAfterToolCallEvent,
  PluginHookAgentContext,
  PluginHookAgentEndEvent,
  PluginHookBeforeAgentFinalizeEvent,
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
} from "./openclaw-hook-types.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";
import type { JsonObject as JsonRecord } from "nemo-flow-node/typed";

export type HookReplayBackendOptions = {
  nf: NemoFlowRuntimeModule;
  config: NemoFlowHookBackendConfig;
  logger: PluginLogger;
  agentVersion: string;
  resolvedAtifOutputDir: string;
  markOutputDegraded: (output: "atif" | "otel" | "openInference") => void;
};

export class HookReplayBackend {
  private readonly nf: NemoFlowRuntimeModule;
  private readonly config: NemoFlowHookBackendConfig;
  private readonly logger: PluginLogger;
  private readonly agentVersion: string;
  private readonly resolvedAtifOutputDir: string;
  private readonly markOutputDegradedValue: (output: "atif" | "otel" | "openInference") => void;
  private readonly stateValue = createHookReplayState();
  private readonly warningCounts = new Map<string, number>();

  constructor(options: HookReplayBackendOptions) {
    this.nf = options.nf;
    this.config = options.config;
    this.logger = options.logger;
    this.agentVersion = options.agentVersion;
    this.resolvedAtifOutputDir = options.resolvedAtifOutputDir;
    this.markOutputDegradedValue = options.markOutputDegraded;
  }

  state(): HookReplayBackendState {
    return this.stateValue;
  }

  onGatewayStart(_event: PluginHookGatewayStartEvent, _ctx: PluginHookGatewayContext): void {
    // Gateway events have no session root in the hook backend. Keep this hook
    // registered so later telemetry lifecycle can attach without changing the shell.
  }

  onSessionStart(event: PluginHookSessionStartEvent, ctx: PluginHookSessionContext): void {
    this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      agentId: ctx.agentId,
      source: "session_start",
      resumedFrom: event.resumedFrom,
    });

    // ensureSession opens the root scope and emits openclaw.session_start for both explicit and lazy sessions.
  }

  async onSessionEnd(event: PluginHookSessionEndEvent, ctx: PluginHookSessionContext): Promise<void> {
    const session = this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      agentId: ctx.agentId,
      source: "lazy_session",
    });

    if (!session) {
      return;
    }

    await this.closeSession(session, sessionEndSummary(event));
  }

  onLlmInput(event: PluginHookLlmInputEvent, ctx: PluginHookAgentContext): void {
    recordLlmInput(this.sessionManager(), event, ctx);
  }

  onLlmOutput(event: PluginHookLlmOutputEvent, ctx: PluginHookAgentContext): void {
    recordLlmOutput(this.sessionManager(), event, ctx);
  }

  onModelCallStarted(event: PluginHookModelCallStartedEvent, ctx: PluginHookAgentContext): void {
    recordModelCallStarted(this.sessionManager(), event, ctx);
  }

  onModelCallEnded(event: PluginHookModelCallEndedEvent, ctx: PluginHookAgentContext): void {
    recordModelCallEnded(this.sessionManager(), event, ctx);
  }

  onAfterToolCall(event: PluginHookAfterToolCallEvent, ctx: PluginHookToolContext): void {
    replayAfterToolCall(this.sessionManager(), event, ctx);
  }

  onAgentEnd(event: PluginHookAgentEndEvent, ctx: PluginHookAgentContext): void {
    const session = this.ensureSession({
      sessionId: ctx.sessionId,
      sessionKey: ctx.sessionKey,
      runId: event.runId ?? ctx.runId,
      agentId: ctx.agentId,
      source: "lazy_session",
    });

    if (!session) {
      return;
    }

    this.emitSessionMark(
      "openclaw.agent_end",
      session,
      toJsonRecord({
        runId: event.runId ?? ctx.runId,
        success: event.success,
        error: event.error,
        durationMs: event.durationMs,
        messageCount: event.messages.length,
      }),
    );
  }

  onBeforeAgentFinalize(event: PluginHookBeforeAgentFinalizeEvent, ctx: PluginHookAgentContext): void {
    const session = this.ensureSession({
      sessionId: event.sessionId,
      sessionKey: event.sessionKey ?? ctx.sessionKey,
      runId: event.runId ?? ctx.runId,
      agentId: ctx.agentId,
      source: "lazy_session",
    });

    if (!session) {
      return;
    }

    this.emitSessionMark(
      "openclaw.before_agent_finalize",
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
    );
  }

  onSubagentSpawned(event: PluginHookSubagentSpawnedEvent, ctx: PluginHookSubagentContext): void {
    const session =
      this.ensureSession({
        requesterSessionKey: ctx.requesterSessionKey,
        source: "lazy_session",
      }) ??
      this.ensureSession({
        childSessionKey: ctx.childSessionKey ?? event.childSessionKey,
        runId: ctx.runId ?? event.runId,
        agentId: event.agentId,
        source: "lazy_session",
      });

    if (!session) {
      return;
    }

    this.emitSessionMark(
      "openclaw.subagent_spawned",
      session,
      toJsonRecord({
        runId: event.runId,
        childSessionKey: event.childSessionKey,
        agentId: event.agentId,
        label: event.label,
        mode: event.mode,
        threadRequested: event.threadRequested,
      }),
    );
  }

  onSubagentEnded(event: PluginHookSubagentEndedEvent, ctx: PluginHookSubagentContext): void {
    const session =
      this.ensureSession({
        requesterSessionKey: ctx.requesterSessionKey,
        source: "lazy_session",
      }) ??
      this.ensureSession({
        childSessionKey: ctx.childSessionKey ?? event.targetSessionKey,
        runId: ctx.runId ?? event.runId,
        source: "lazy_session",
      });

    if (!session) {
      return;
    }

    this.emitSessionMark(
      "openclaw.subagent_ended",
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
    );
  }

  async drainForGatewayStop(reason?: string): Promise<void> {
    await this.closeAllSessions({ reason: reason ?? "gateway_stop" });
  }

  async stop(reason: string): Promise<void> {
    await this.closeAllSessions({ reason });
  }

  safeReplay(label: string, session: SessionState | undefined, emit: () => void): void {
    try {
      emit();
    } catch (error) {
      this.stateValue.counters.replayErrors += 1;
      this.logBoundedWarn(
        `safe-replay:${label}`,
        `nemo-flow replay failed: label=${label} session=${session?.sessionId ?? "unknown"} error=${toMessage(error)}`,
      );
    }
  }

  async safeReplayAsync(
    label: string,
    session: SessionState | undefined,
    emit: () => Promise<void>,
  ): Promise<void> {
    try {
      await emit();
    } catch (error) {
      this.stateValue.counters.replayErrors += 1;
      this.logBoundedWarn(
        `safe-replay:${label}`,
        `nemo-flow async replay failed: label=${label} session=${session?.sessionId ?? "unknown"} error=${toMessage(error)}`,
      );
    }
  }

  emitCapturedUnderSession(label: string, session: SessionState, emit: () => void): void {
    this.safeReplay(label, session, () => {
      const previousStack = this.nf.currentScopeStack();
      try {
        this.nf.setThreadScopeStack(session.stack);
        this.withAtifCapture(session, emit);
      } finally {
        this.nf.setThreadScopeStack(previousStack);
      }
    });
  }

  replayPendingLlmOutputsForSession(session: SessionState, options: { allowPlaceholderRequest: boolean }): void {
    replayPendingLlmOutputsForSession(this.sessionManager(), session, options);
  }

  emitUnpairedModelCallTimingMarks(session: SessionState): void {
    emitUnpairedModelCallTimingMarks(this.sessionManager(), session);
  }

  private ensureSession(input: Parameters<typeof ensureSession>[1]): SessionState | undefined {
    return ensureSession(this.sessionManager(), input);
  }

  private async closeSession(session: SessionState, summary: JsonRecord): Promise<void> {
    drainSession(this.sessionManager(), session);
    closeSessionRoot(this.sessionManager(), session, summary);
    await exportAtifJson(this.sessionManager(), session);
    deleteSession(this.stateValue, session);
  }

  private emitSessionMark(name: string, session: SessionState, data: JsonRecord): void {
    this.emitCapturedUnderSession(name, session, () => {
      emitMark({
        nf: this.nf,
        state: this.stateValue,
        session,
        name,
        data,
      });
    });
  }

  private async closeAllSessions(summary: JsonRecord): Promise<void> {
    for (const session of [...this.stateValue.sessions.values()]) {
      await this.closeSession(session, summary);
    }
  }

  private withAtifCapture(_session: SessionState, emit: () => void): void {
    withAtifCapture(this.sessionManager(), _session, emit);
  }

  private sessionManager() {
    return {
      nf: this.nf,
      config: this.config,
      logger: this.logger,
      state: this.stateValue,
      agentVersion: this.agentVersion,
      resolvedAtifOutputDir: this.resolvedAtifOutputDir,
      emitCapturedUnderSession: (label: string, session: SessionState, emit: () => void) =>
        this.emitCapturedUnderSession(label, session, emit),
      replayPendingLlmOutputsForSession: (session: SessionState, options: { allowPlaceholderRequest: boolean }) =>
        this.replayPendingLlmOutputsForSession(session, options),
      emitUnpairedModelCallTimingMarks: (session: SessionState) => this.emitUnpairedModelCallTimingMarks(session),
      markOutputDegraded: (output: "atif" | "otel" | "openInference") => this.markOutputDegradedValue(output),
      logBoundedWarn: (key: string, message: string) => this.logBoundedWarn(key, message),
    };
  }

  private logBoundedWarn(key: string, message: string): void {
    const count = this.warningCounts.get(key) ?? 0;
    this.warningCounts.set(key, count + 1);
    if (count === 0) {
      this.logger.warn?.(message);
    }
  }
}

export { llmKey };

export function resolveBackendSessionKey(
  state: HookReplayBackendState,
  input: Parameters<typeof resolveSessionKey>[1],
): string | undefined {
  return resolveSessionKey(state, input);
}

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

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
