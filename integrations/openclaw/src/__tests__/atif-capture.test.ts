// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";
import { describe, it } from "node:test";

import {
  createAtifExporter,
  exportAtifJson,
  makeSafeSessionId,
  withAtifCapture,
} from "../atif-capture.js";
import { parseConfig } from "../config.js";
import { createHookReplayState, type SessionManager, type SessionState } from "../hook-replay/session.js";
import type { NemoFlowRuntimeModule } from "../modules.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

describe("ATIF capture", () => {
  it("registers and deregisters around synchronous emit", () => {
    const manager = createManager();
    const session = createSession(manager, "session-1");

    createAtifExporter(manager, session);
    withAtifCapture(manager, session, () => {
      manager.nf.event("test", session.rootHandle, { ok: true });
    });

    const exporter = session.atif?.exporter as FakeAtifExporter | undefined;
    assert.ok(exporter);
    assert.deepEqual(exporter.actions, [
      "register:openclaw.nemo-flow.atif.c2Vzc2lvbi0x",
      "deregister:openclaw.nemo-flow.atif.c2Vzc2lvbi0x",
    ]);
    assert.equal(manager.state.counters.replayErrors, 0);
  });

  it("disables future capture when deregister fails", () => {
    const manager = createManager({ deregisterReturn: false });
    const session = createSession(manager, "session-1");

    createAtifExporter(manager, session);
    withAtifCapture(manager, session, () => {});

    assert.equal(session.atif?.disabled, true);
    assert.equal(session.atif?.leakedRegistration, true);
    assert.deepEqual(manager.degradedOutputs, ["atif"]);
    assert.equal(manager.state.counters.replayErrors, 1);
  });

  it("emits outside ATIF capture and skips export when register fails", async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-atif-"));
    try {
      const manager = createManager({ outputDir, registerThrows: true });
      const session = createSession(manager, "session-1");
      let emitted = false;

      createAtifExporter(manager, session);
      withAtifCapture(manager, session, () => {
        emitted = true;
      });
      await exportAtifJson(manager, session);

      assert.equal(emitted, true);
      assert.equal(session.atif, undefined);
      assert.deepEqual(await fs.readdir(outputDir), []);
      assert.deepEqual(manager.degradedOutputs, ["atif"]);
      assert.equal(manager.state.counters.replayErrors, 1);
    } finally {
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  });

  it("exports ATIF JSON after a captured window even when deregister fails", async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-atif-"));
    try {
      const manager = createManager({ outputDir, deregisterReturn: false });
      const session = createSession(manager, "session-1");

      createAtifExporter(manager, session);
      withAtifCapture(manager, session, () => {});
      await exportAtifJson(manager, session);

      const targetPath = path.join(outputDir, `${makeSafeSessionId("session-1")}.json`);
      assert.equal(await fs.readFile(targetPath, "utf8"), "{\"ok\":true}");
      assert.equal(manager.state.counters.atifFilesWritten, 1);
      assert.equal(manager.state.counters.replayErrors, 1);
      assert.deepEqual(manager.degradedOutputs, ["atif"]);
      assert.equal(session.atif, undefined);
    } finally {
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  });

  it("does not block session replay when exporter construction fails", () => {
    const manager = createManager({ constructorThrows: true });
    const session = createSession(manager, "session-1");

    createAtifExporter(manager, session);

    assert.equal(session.atif, undefined);
    assert.deepEqual(manager.degradedOutputs, ["atif"]);
    assert.equal(manager.state.counters.replayErrors, 1);
  });

  it("exports ATIF JSON to a safe session filename and clears exporter", async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-atif-"));
    try {
      const manager = createManager({ outputDir });
      const session = createSession(manager, "../session:1");

      createAtifExporter(manager, session);
      await exportAtifJson(manager, session);

      const targetPath = path.join(outputDir, `${makeSafeSessionId("../session:1")}.json`);
      assert.equal(await fs.readFile(targetPath, "utf8"), "{\"ok\":true}");
      assert.equal(manager.state.counters.atifFilesWritten, 1);
      assert.equal(session.atif, undefined);
    } finally {
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  });

  it("clears exporter and marks ATIF degraded when export fails", async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-atif-"));
    const outputFile = path.join(outputDir, "not-a-directory");
    await fs.writeFile(outputFile, "block mkdir", "utf8");
    try {
      const manager = createManager({ outputDir: outputFile });
      const session = createSession(manager, "session-1");

      createAtifExporter(manager, session);
      const exporter = session.atif?.exporter as FakeAtifExporter | undefined;
      await exportAtifJson(manager, session);

      assert.ok(exporter);
      assert.equal(exporter.cleared, true);
      assert.deepEqual(manager.degradedOutputs, ["atif"]);
      assert.equal(manager.state.counters.replayErrors, 1);
      assert.equal(session.atif, undefined);
    } finally {
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  });

  it("does not throw or retain ATIF state when exporter clear fails", async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-atif-"));
    try {
      const manager = createManager({ outputDir, clearThrows: true });
      const session = createSession(manager, "session-1");

      createAtifExporter(manager, session);
      await assert.doesNotReject(() => exportAtifJson(manager, session));

      const targetPath = path.join(outputDir, `${makeSafeSessionId("session-1")}.json`);
      assert.equal(await fs.readFile(targetPath, "utf8"), "{\"ok\":true}");
      assert.equal(manager.state.counters.atifFilesWritten, 1);
      assert.equal(manager.state.counters.replayErrors, 1);
      assert.deepEqual(manager.degradedOutputs, ["atif"]);
      assert.equal(session.atif, undefined);
    } finally {
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  });
});

type TestManager = SessionManager & {
  degradedOutputs: Array<"atif" | "otel" | "openInference">;
};

function createManager(params: {
  outputDir?: string;
  constructorThrows?: boolean;
  registerThrows?: boolean;
  deregisterReturn?: boolean;
  clearThrows?: boolean;
} = {}): TestManager {
  const degradedOutputs: TestManager["degradedOutputs"] = [];
  const nf = createNemoFlowRuntime(params);
  const manager: TestManager = {
    nf,
    config: parseConfig({ atif: { enabled: true } }),
    logger: createLogger(),
    state: createHookReplayState(),
    agentVersion: "test-version",
    resolvedAtifOutputDir: params.outputDir ?? "/tmp/nemo-flow-atif",
    degradedOutputs,
    emitCapturedUnderSession: (_label, _session, emit) => emit(),
    replayPendingLlmOutputsForSession: () => {},
    emitUnpairedModelCallTimingMarks: () => {},
    markOutputDegraded: (output) => degradedOutputs.push(output),
    logBoundedWarn: () => {},
  };
  return manager;
}

function createSession(manager: TestManager, sessionId: string): SessionState {
  const session: SessionState = {
    sessionId,
    source: "session_start",
    stack: manager.nf.createScopeStack(),
    rootHandle: { id: "root" } as unknown as ReturnType<NemoFlowRuntimeModule["pushScope"]>,
  };
  manager.state.sessions.set(sessionId, session);
  return session;
}

function createLogger(): PluginLogger {
  return {
    info: () => {},
    warn: () => {},
    error: () => {},
  };
}

function createNemoFlowRuntime(params: {
  constructorThrows?: boolean;
  registerThrows?: boolean;
  deregisterReturn?: boolean;
  clearThrows?: boolean;
}): NemoFlowRuntimeModule {
  const AtifExporter = params.constructorThrows
    ? FailingAtifExporter
    : class extends FakeAtifExporter {
        constructor(sessionId: string, agentName: string, agentVersion: string, modelName?: string | null) {
          super(sessionId, agentName, agentVersion, modelName, params.deregisterReturn ?? true);
          this.registerThrows = params.registerThrows ?? false;
          this.clearThrows = params.clearThrows ?? false;
        }
      };

  return {
    ScopeType: { Agent: 0 } as NemoFlowRuntimeModule["ScopeType"],
    createScopeStack: () => ({}) as unknown as ReturnType<NemoFlowRuntimeModule["createScopeStack"]>,
    currentScopeStack: () => ({}) as unknown as ReturnType<NemoFlowRuntimeModule["currentScopeStack"]>,
    setThreadScopeStack: () => {},
    pushScope: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["pushScope"]>),
    popScope: () => {},
    event: () => {},
    llmCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["llmCall"]>),
    llmCallEnd: () => {},
    toolCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["toolCall"]>),
    toolCallEnd: () => {},
    AtifExporter,
    OpenTelemetrySubscriber: FakeSubscriber,
    OpenInferenceSubscriber: FakeSubscriber,
  };
}

class FakeAtifExporter {
  readonly actions: string[] = [];
  cleared = false;
  protected registerThrows = false;
  protected clearThrows = false;

  constructor(
    readonly sessionId: string,
    readonly agentName: string,
    readonly agentVersion: string,
    readonly modelName: string | null | undefined,
    private readonly deregisterReturn: boolean,
  ) {}

  register(name: string): void {
    if (this.registerThrows) {
      throw new Error("register failed");
    }
    this.actions.push(`register:${name}`);
  }

  deregister(name: string): boolean {
    this.actions.push(`deregister:${name}`);
    return this.deregisterReturn;
  }

  exportJson(): string {
    return "{\"ok\":true}";
  }

  clear(): void {
    if (this.clearThrows) {
      throw new Error("clear failed");
    }
    this.cleared = true;
  }
}

class FailingAtifExporter {
  constructor() {
    throw new Error("constructor failed");
  }

  register(): void {}
  deregister(): boolean {
    return true;
  }
  exportJson(): string {
    return "{}";
  }
  clear(): void {}
}

class FakeSubscriber {
  register(): void {}
  deregister(): boolean {
    return true;
  }
  forceFlush(): void {}
  shutdown(): void {}
}
