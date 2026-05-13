// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HookReplayBackend tests covering session lifecycle, aliases, marks, and cleanup.
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseConfig } from "../config.js";
import { errorToJson, toJsonRecord } from "../hook-replay/marks.js";
import { HookReplayBackend } from "../hooks-backend.js";
import type { NemoFlowRuntimeModule } from "../modules.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

describe("HookReplayBackend", () => {
  it("opens a session root and records aliases on session_start", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart(
      { sessionId: "session-1", sessionKey: "session-key-1", resumedFrom: "previous-session" },
      { sessionId: "session-1", sessionKey: "session-key-1", agentId: "agent-1" },
    );

    const session = backend.state().sessions.get("session-1");
    assert.ok(session);
    assert.equal(session.sessionId, "session-1");
    assert.equal(session.sessionKey, "session-key-1");
    assert.equal(session.agentId, "agent-1");
    assert.equal(session.resumedFrom, "previous-session");
    assert.equal(backend.state().sessionAliases.get("session-key-1"), "session-1");
    assert.equal(nf.calls.pushScope.length, 1);
    assert.deepEqual(nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);
  });

  it("emits session_start when a session is created lazily from llm_input", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "lazy-session",
        provider: "openai",
        model: "gpt",
        prompt: "hello",
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: "run-1", sessionId: "lazy-session" },
    );

    assert.deepEqual(nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);
    assert.deepEqual(nf.calls.event[0]?.data, {
      sessionId: "lazy-session",
      source: "lazy_session",
      runId: "run-1",
    });
  });

  it("keeps concurrent sessions isolated by scope handle and alias", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: "a", sessionKey: "ka" }, { sessionId: "a", sessionKey: "ka" });
    backend.onSessionStart({ sessionId: "b", sessionKey: "kb" }, { sessionId: "b", sessionKey: "kb" });

    const first = backend.state().sessions.get("a");
    const second = backend.state().sessions.get("b");
    assert.ok(first?.rootHandle);
    assert.ok(second?.rootHandle);
    assert.notEqual(first.rootHandle, second.rootHandle);
    assert.equal(backend.state().sessionAliases.get("ka"), "a");
    assert.equal(backend.state().sessionAliases.get("kb"), "b");
  });

  it("drains before close, emits unpaired timing mark, and evicts session records", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: "session-1" }, { sessionId: "session-1" });
    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt",
        prompt: "hello",
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: "run-1", sessionId: "session-1" },
    );
    backend.onLlmOutput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt",
        assistantTexts: ["hi"],
      },
      { runId: "run-1", sessionId: "session-1" },
    );
    backend.onModelCallEnded(
      {
        runId: "run-1",
        callId: "call-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt",
        durationMs: 42,
        outcome: "completed",
      },
      { runId: "run-1", sessionId: "session-1" },
    );

    await backend.onSessionEnd(
      { sessionId: "session-1", messageCount: 3, reason: "idle" },
      { sessionId: "session-1" },
    );

    assert.equal(backend.state().sessions.size, 0);
    assert.equal(backend.state().sessionAliases.size, 0);
    assert.equal(backend.state().llmInputs.size, 0);
    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(backend.state().modelCallsByCallId.size, 0);
    assert.equal(backend.state().modelTimingsByLlmKey.size, 0);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      [
        "openclaw.session_start",
        "openclaw.model_call_timing_unpaired",
        "openclaw.session_end",
      ],
    );
    assert.equal(nf.calls.popScope.length, 1);
  });

  it("emits blocked tool marks from after_tool_call only", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: "session-1", sessionKey: "sk" }, { sessionId: "session-1", sessionKey: "sk" });
    backend.onAfterToolCall(
      {
        toolName: "dangerous_tool",
        params: {},
        toolCallId: "tool-call-1",
        result: { details: { status: "blocked", deniedReason: "policy" } },
        durationMs: 5,
      },
      { sessionKey: "sk", runId: "run-1", toolName: "dangerous_tool", toolCallId: "tool-call-1" },
    );

    assert.deepEqual(nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.tool_blocked",
    ]);
    assert.deepEqual(nf.calls.event[1]?.data, {
      toolName: "dangerous_tool",
      toolCallId: "tool-call-1",
      runId: "run-1",
      blocked: true,
      deniedReason: "policy",
      durationMs: 5,
    });
  });

  it("safe replay restores the previous scope stack and fails open", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: "session-1" }, { sessionId: "session-1" });
    const session = backend.state().sessions.get("session-1");
    assert.ok(session);

    assert.doesNotThrow(() => {
      backend.emitCapturedUnderSession("test_throw", session, () => {
        throw new Error("boom");
      });
    });

    assert.equal(backend.state().counters.replayErrors, 1);
    assert.equal(nf.calls.setThreadScopeStack.at(-1), nf.previousStack);
  });

  it("bounds repeated replay warnings by label", () => {
    const nf = createNemoFlowRuntime();
    const logger = createLogger();
    const backend = createBackend(nf, logger);

    backend.safeReplay("same_failure", undefined, () => {
      throw new Error("first");
    });
    backend.safeReplay("same_failure", undefined, () => {
      throw new Error("second");
    });

    assert.equal(logger.messages.warn.length, 1);
    assert.match(logger.messages.warn[0] ?? "", /same_failure/);
    assert.equal(backend.state().counters.replayErrors, 2);
  });

  it("returns undefined from before_agent_finalize", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    const result = backend.onBeforeAgentFinalize(
      {
        runId: "run-1",
        sessionId: "session-1",
        stopHookActive: false,
      },
      { runId: "run-1", sessionId: "session-1" },
    );

    assert.equal(result, undefined);
    assert.deepEqual(nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.before_agent_finalize",
    ]);
  });

  it("keeps gateway stop reason out of the root session output when a final answer is known", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onAgentEnd(
      {
        runId: "run-1",
        messages: [
          { role: "user", content: "hello" },
          { role: "assistant", provider: "openai", model: "gpt", content: "Final answer." },
        ],
        success: true,
      },
      { runId: "run-1", sessionId: "session-1" },
    );
    await backend.drainForGatewayStop("gateway stopping");

    assert.deepEqual(nf.calls.popScope[0]?.output, {
      content: "Final answer.",
      source: "openclaw.agent_end",
      runId: "run-1",
      success: true,
    });
    assert.deepEqual(nf.calls.event.at(-1)?.data, { reason: "gateway stopping" });
  });

  it("records subagent marks under the requester alias without merging child session identity", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart(
      { sessionId: "parent-session", sessionKey: "parent-key" },
      { sessionId: "parent-session", sessionKey: "parent-key" },
    );
    backend.onSubagentSpawned(
      {
        childSessionKey: "child-key",
        agentId: "child-agent",
        mode: "run",
        threadRequested: false,
        runId: "child-run",
      },
      { requesterSessionKey: "parent-key", childSessionKey: "child-key", runId: "child-run" },
    );

    assert.equal(backend.state().sessionAliases.get("child-key"), undefined);
    assert.deepEqual(nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.subagent_spawned",
    ]);
  });

  it("uses child session key as a lazy-session fallback without aliasing it away", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onSubagentSpawned(
      {
        childSessionKey: "child-key",
        agentId: "child-agent",
        mode: "run",
        threadRequested: false,
        runId: "child-run",
      },
      { childSessionKey: "child-key", runId: "child-run" },
    );

    assert.ok(backend.state().sessions.get("child-key"));
    assert.equal(backend.state().sessionAliases.get("child-run"), "child-key");
    assert.equal(backend.state().sessionAliases.get("child-key"), undefined);
  });

  it("normalizes circular replay payloads before NAPI boundaries", () => {
    const payload: Record<string, unknown> = { ok: true };
    payload.self = payload;

    assert.deepEqual(toJsonRecord(payload), {
      ok: true,
      self: { ok: true, self: "[Circular]" },
    });
    assert.deepEqual(toJsonRecord({
      finite: 42,
      nan: Number.NaN,
      positiveInfinity: Number.POSITIVE_INFINITY,
      negativeInfinity: Number.NEGATIVE_INFINITY,
    }), {
      finite: 42,
      nan: null,
      positiveInfinity: null,
      negativeInfinity: null,
    });
    assert.deepEqual(errorToJson(new Error("boom")).message, "boom");
  });

  it("normalizes prototype keys without mutating output prototypes", () => {
    const payload: Record<string, unknown> = {};
    Object.defineProperty(payload, "__proto__", {
      enumerable: true,
      value: { polluted: true },
    });

    const normalized = toJsonRecord(payload);

    assert.equal(Object.getPrototypeOf(normalized), Object.prototype);
    assert.deepEqual(normalized["__proto__"], { polluted: true });
    assert.equal(({} as Record<string, unknown>).polluted, undefined);
  });
});

type TestNemoFlowRuntime = NemoFlowRuntimeModule & {
  previousStack: { id: "previous" };
  calls: {
    pushScope: Array<{ name: string; scopeType: number; data: unknown }>;
    popScope: Array<{ handle: unknown; output: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown }>;
    setThreadScopeStack: unknown[];
  };
};

type TestLogger = PluginLogger & {
  messages: {
    warn: string[];
  };
};

function createBackend(
  nf: TestNemoFlowRuntime,
  logger = createLogger(),
  options: {
    config?: ReturnType<typeof parseConfig>;
  } = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: options.config ?? parseConfig({}),
    logger,
    agentVersion: "test-version",
  });
}

function createLogger(): TestLogger {
  const messages: TestLogger["messages"] = { warn: [] };
  return {
    messages,
    info: () => {},
    warn: (message) => messages.warn.push(message),
    error: () => {},
  };
}

function createNemoFlowRuntime(): TestNemoFlowRuntime {
  let nextScopeId = 0;
  const previousStack = { id: "previous" as const };
  const calls: TestNemoFlowRuntime["calls"] = {
    pushScope: [],
    popScope: [],
    event: [],
    setThreadScopeStack: [],
  };

  return {
    ScopeType: { Agent: 0 } as NemoFlowRuntimeModule["ScopeType"],
    previousStack,
    calls,
    createScopeStack: () => ({ id: `stack-${nextScopeId++}` }) as unknown as ReturnType<NemoFlowRuntimeModule["createScopeStack"]>,
    currentScopeStack: () => previousStack as unknown as ReturnType<NemoFlowRuntimeModule["currentScopeStack"]>,
    setThreadScopeStack: (stack) => calls.setThreadScopeStack.push(stack),
    pushScope: (name, scopeType, _handle, _attributes, data) => {
      const handle = { id: `scope-${nextScopeId++}` };
      calls.pushScope.push({ name, scopeType, data });
      return handle as unknown as ReturnType<NemoFlowRuntimeModule["pushScope"]>;
    },
    popScope: (handle, output) => calls.popScope.push({ handle, output }),
    event: (name, handle, data) => calls.event.push({ name, handle, data }),
    llmCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["llmCall"]>),
    llmCallEnd: () => {},
    toolCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["toolCall"]>),
    toolCallEnd: () => {},
  };
}
