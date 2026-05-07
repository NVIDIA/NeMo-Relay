// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as fs from "node:fs/promises";
import * as path from "node:path";

import type { SessionManager, SessionState } from "./hook-replay/session.js";

export function createAtifExporter(manager: SessionManager, session: SessionState): void {
  if (!manager.config.atif.enabled || session.atif) {
    return;
  }

  try {
    session.atif = {
      exporter: new manager.nf.AtifExporter(
        session.sessionId,
        manager.config.atif.agentName,
        manager.agentVersion,
        null,
      ),
      registrationName: `openclaw.nemo-flow.atif.${makeSafeSessionId(session.sessionId)}`,
      capturing: false,
    };
  } catch (error) {
    manager.markOutputDegraded("atif");
    manager.state.counters.replayErrors += 1;
    manager.logBoundedWarn(
      `atif_constructor_failed:${session.sessionId}`,
      `nemo-flow failed to construct ATIF exporter for session ${session.sessionId}: ${toMessage(error)}`,
    );
  }
}

export function withAtifCapture(
  manager: SessionManager,
  session: SessionState,
  emit: () => void,
): void {
  const { state, logBoundedWarn, markOutputDegraded } = manager;
  if (!session.atif || session.atif.disabled) {
    emit();
    return;
  }
  if (session.atif.capturing) {
    emit();
    return;
  }

  session.atif.capturing = true;
  let registered = false;
  try {
    session.atif.exporter.register(session.atif.registrationName);
    registered = true;
    session.atif.registeredOnce = true;
    emit();
  } catch (error) {
    if (!registered) {
      session.atif.disabled = true;
      markOutputDegraded("atif");
      state.counters.replayErrors += 1;
      logBoundedWarn(
        `atif_register_failed:${session.atif.registrationName}`,
        `nemo-flow failed to register ATIF capture ${session.atif.registrationName}; disabling ATIF for session ${session.sessionId}: ${toMessage(error)}`,
      );
      emit();
      return;
    }
    throw error;
  } finally {
    if (registered) {
      try {
        const removed = session.atif.exporter.deregister(session.atif.registrationName);
        if (!removed) {
          session.atif.leakedRegistration = true;
          session.atif.disabled = true;
          markOutputDegraded("atif");
          state.counters.replayErrors += 1;
          logBoundedWarn(
            `atif_deregister_missing:${session.atif.registrationName}`,
            `nemo-flow ATIF capture ${session.atif.registrationName} was already deregistered; disabling ATIF for session ${session.sessionId} to avoid duplicate global subscribers`,
          );
        }
      } catch (error) {
        session.atif.leakedRegistration = true;
        session.atif.disabled = true;
        markOutputDegraded("atif");
        state.counters.replayErrors += 1;
        logBoundedWarn(
          `atif_deregister_failed:${session.atif.registrationName}`,
          `nemo-flow failed to deregister ATIF capture ${session.atif.registrationName}; disabling ATIF for session ${session.sessionId} to avoid duplicate global subscribers: ${toMessage(error)}`,
        );
      }
    }
    session.atif.capturing = false;
  }
}

export async function exportAtifJson(manager: SessionManager, session: SessionState): Promise<void> {
  if (!session.atif) {
    return;
  }
  if (session.atif.disabled && !session.atif.registeredOnce) {
    clearAtifExporter(manager, session, session.atif);
    delete session.atif;
    return;
  }

  const atif = session.atif;
  try {
    await fs.mkdir(manager.resolvedAtifOutputDir, { recursive: true });
    const targetPath = path.join(manager.resolvedAtifOutputDir, `${makeSafeSessionId(session.sessionId)}.json`);
    await fs.writeFile(targetPath, atif.exporter.exportJson(), "utf8");
    manager.state.counters.atifFilesWritten += 1;
  } catch (error) {
    manager.markOutputDegraded("atif");
    manager.state.counters.replayErrors += 1;
    manager.logBoundedWarn(
      `atif_export_failed:${session.sessionId}`,
      `nemo-flow failed to export ATIF for session ${session.sessionId}: ${toMessage(error)}`,
    );
  } finally {
    clearAtifExporter(manager, session, atif);
    delete session.atif;
  }
}

export function makeSafeSessionId(sessionId: string): string {
  const encoded = Buffer.from(sessionId, "utf8").toString("base64url");
  return encoded.length > 0 ? encoded : "empty-session-id";
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clearAtifExporter(
  manager: SessionManager,
  session: SessionState,
  atif: NonNullable<SessionState["atif"]>,
): void {
  try {
    atif.exporter.clear();
  } catch (error) {
    manager.markOutputDegraded("atif");
    manager.state.counters.replayErrors += 1;
    manager.logBoundedWarn(
      `atif_clear_failed:${session.sessionId}`,
      `nemo-flow failed to clear ATIF exporter for session ${session.sessionId}: ${toMessage(error)}`,
    );
  }
}
