// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseConfig } from "../config.js";
import { HookReplayBackend } from "../hooks-backend.js";
import type { NemoFlowRuntimeModule } from "../modules.js";
import type { PluginLoggerLike } from "../types.js";

describe("Tool replay", () => {
  it("replays after_tool_call with stripped payloads by default", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onAfterToolCall(
      {
        toolName: "read_file",
        params: { path: "/secret", token: "value" },
        toolCallId: "tool-call-1",
        runId: "run-1",
        result: { text: "secret" },
        durationMs: 7,
      },
      { runId: "run-1", sessionId: "session-1", toolCallId: "tool-call-1" },
    );

    assert.equal(nf.calls.toolCall.length, 1);
    assert.equal(nf.calls.toolCallEnd.length, 1);
    assert.equal(backend.state().counters.toolSpansReplayed, 1);
    assert.deepEqual(nf.calls.toolCall[0]?.args, {
      stripped: true,
      argKeys: ["path", "token"],
    });
    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, {
      stripped: true,
      hasError: false,
    });
  });

  it("captures full tool payloads only when trusted config opts in", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, {
      capture: {
        stripToolArgs: false,
        stripToolResults: false,
      },
    });

    backend.onAfterToolCall(
      {
        toolName: "read_file",
        params: { path: "/workspace/file.txt" },
        toolCallId: "tool-call-1",
        runId: "run-1",
        result: { text: "ok" },
        durationMs: 7,
      },
      { runId: "run-1", sessionId: "session-1", toolCallId: "tool-call-1" },
    );

    assert.deepEqual(nf.calls.toolCall[0]?.args, { path: "/workspace/file.txt" });
    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, { result: { text: "ok" } });
  });

  it("passes non-null tool end payload when result and error are missing", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, {
      capture: {
        stripToolResults: false,
      },
    });

    backend.onAfterToolCall(
      {
        toolName: "noop",
        params: {},
        toolCallId: "tool-call-1",
        runId: "run-1",
      },
      { runId: "run-1", sessionId: "session-1", toolCallId: "tool-call-1" },
    );

    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, { result: null });
    assert.deepEqual(nf.calls.toolCallEnd[0]?.data, { result: null });
  });

  it("emits blocked tool mark instead of successful tool span", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onAfterToolCall(
      {
        toolName: "dangerous_tool",
        params: {},
        toolCallId: "tool-call-1",
        runId: "run-1",
        result: { details: { status: "blocked", deniedReason: "policy" } },
        durationMs: 3,
      },
      { runId: "run-1", sessionId: "session-1", toolCallId: "tool-call-1" },
    );

    assert.equal(nf.calls.toolCall.length, 0);
    assert.ok(nf.calls.event.some((event) => event.name === "openclaw.tool_blocked"));
  });
});

type TestNemoFlowRuntime = NemoFlowRuntimeModule & {
  calls: {
    pushScope: Array<{ name: string; scopeType: number; data: unknown }>;
    popScope: Array<{ handle: unknown; output: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown }>;
    setThreadScopeStack: unknown[];
    llmCall: Array<{ name: string; request: unknown }>;
    llmCallEnd: Array<{ handle: unknown; response: unknown }>;
    toolCall: Array<{ name: string; args: unknown }>;
    toolCallEnd: Array<{ handle: unknown; result: unknown; data: unknown }>;
  };
};

function createBackend(
  nf: TestNemoFlowRuntime,
  overrides: {
    capture?: Partial<ReturnType<typeof parseConfig>["capture"]>;
  } = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: parseConfig({
      atif: { enabled: false },
      capture: overrides.capture,
    }),
    logger: createLogger(),
    agentVersion: "test-version",
    resolvedAtifOutputDir: "/tmp/openclaw-state/plugins/nemo-flow/atif",
    markOutputDegraded: () => {},
  });
}

function createLogger(): PluginLoggerLike {
  return {
    info: () => {},
    warn: () => {},
  };
}

function createNemoFlowRuntime(): TestNemoFlowRuntime {
  let nextScopeId = 0;
  const previousStack = { id: "previous" };
  const calls: TestNemoFlowRuntime["calls"] = {
    pushScope: [],
    popScope: [],
    event: [],
    setThreadScopeStack: [],
    llmCall: [],
    llmCallEnd: [],
    toolCall: [],
    toolCallEnd: [],
  };

  return {
    ScopeType: { Agent: 0 },
    calls,
    createScopeStack: () => ({ id: `stack-${nextScopeId++}` }),
    currentScopeStack: () => previousStack,
    setThreadScopeStack: (stack) => calls.setThreadScopeStack.push(stack),
    pushScope: (name, scopeType, _handle, _attributes, data) => {
      const handle = { id: `scope-${nextScopeId++}` };
      calls.pushScope.push({ name, scopeType, data });
      return handle;
    },
    popScope: (handle, output) => calls.popScope.push({ handle, output }),
    event: (name, handle, data) => calls.event.push({ name, handle, data }),
    llmCall: (name, request) => {
      const handle = { id: `llm-${nextScopeId++}` };
      calls.llmCall.push({ name, request });
      return handle;
    },
    llmCallEnd: (handle, response) => calls.llmCallEnd.push({ handle, response }),
    toolCall: (name, args) => {
      const handle = { id: `tool-${nextScopeId++}` };
      calls.toolCall.push({ name, args });
      return handle;
    },
    toolCallEnd: (handle, result, data) => calls.toolCallEnd.push({ handle, result, data }),
    AtifExporter: FakeAtifExporter,
    OpenTelemetrySubscriber: FakeSubscriber,
    OpenInferenceSubscriber: FakeSubscriber,
  };
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
