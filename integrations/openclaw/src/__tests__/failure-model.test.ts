// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseConfig } from "../config.js";
import { HookReplayBackend } from "../hooks-backend.js";
import type { NemoFlowRuntimeModule } from "../modules.js";
import type { PluginLoggerLike } from "../types.js";

describe("Replay failure model", () => {
  it("grace timer replay failure is caught and counted", async () => {
    const logger = createLogger();
    const backend = new HookReplayBackend({
      nf: createThrowingLlmRuntime(),
      config: parseConfig({
        atif: { enabled: false },
        correlation: { llmOutputGraceMs: 1 },
      }),
      logger,
      agentVersion: "test-version",
      resolvedAtifOutputDir: "/tmp/openclaw-state/plugins/nemo-flow/atif",
      markOutputDegraded: () => {},
    });

    backend.onLlmOutput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        assistantTexts: ["hi"],
      },
      { runId: "run-1", sessionId: "session-1" },
    );

    await delay(10);

    assert.equal(backend.state().counters.replayErrors, 1);
    assert.equal(logger.messages.warn.length, 1);
    assert.match(logger.messages.warn[0] ?? "", /llm_output/);
  });
});

type TestLogger = PluginLoggerLike & {
  messages: {
    warn: string[];
  };
};

function createLogger(): TestLogger {
  const messages: TestLogger["messages"] = { warn: [] };
  return {
    messages,
    info: () => {},
    warn: (message) => messages.warn.push(message),
  };
}

function createThrowingLlmRuntime(): NemoFlowRuntimeModule {
  let nextScopeId = 0;
  const previousStack = { id: "previous" };
  return {
    ScopeType: { Agent: 0 },
    createScopeStack: () => ({ id: `stack-${nextScopeId++}` }),
    currentScopeStack: () => previousStack,
    setThreadScopeStack: () => {},
    pushScope: () => ({ id: `scope-${nextScopeId++}` }),
    popScope: () => {},
    event: () => {},
    llmCall: () => {
      throw new Error("llmCall failed");
    },
    llmCallEnd: () => {},
    toolCall: () => ({}),
    toolCallEnd: () => {},
    AtifExporter: FakeAtifExporter,
    OpenTelemetrySubscriber: FakeSubscriber,
    OpenInferenceSubscriber: FakeSubscriber,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class FakeAtifExporter {
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
