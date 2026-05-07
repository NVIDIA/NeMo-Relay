// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseConfig } from "../config.js";
import { HookReplayBackend } from "../hooks-backend.js";
import type { NemoFlowRuntimeModule } from "../modules.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

describe("LLM replay", () => {
  it("replays llm output with buffered input under the session root", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        systemPrompt: "be concise",
        prompt: "hello",
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: "run-1", sessionId: "session-1", agentId: "agent-1" },
    );
    backend.onLlmOutput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        assistantTexts: ["hi"],
        resolvedRef: "provider/model",
        harnessId: "harness-1",
        usage: { input: 2, output: 3 },
      },
      { runId: "run-1", sessionId: "session-1", agentId: "agent-1" },
    );

    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal(backend.state().counters.llmSpansReplayed, 1);
    assert.equal(backend.state().llmInputs.size, 0);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [{ role: "user", content: "hello" }]);
    assert.equal(request.content.systemPrompt, "be concise");
    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.content, "hi");
    assert.deepEqual(response.token_usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });
  });

  it("replays pending output when matching input arrives and cancels pending queue", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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
    assert.equal(backend.state().llmOutputsPendingInput.size, 1);

    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        prompt: "hello",
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: "run-1", sessionId: "session-1" },
    );

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, false);
    assert.equal(request.content.prompt, "hello");
  });

  it("replays placeholder request when output grace timer expires", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 1 });

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

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, true);
    assert.equal(request.content.prompt, "");
  });

  it("does not keep the process alive while waiting for llm output grace", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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

    const pending = [...backend.state().llmOutputsPendingInput.values()][0]?.[0];
    assert.ok(pending?.timer);
    if (isRefableTimer(pending.timer)) {
      assert.equal(pending.timer.hasRef(), false);
    }
    await backend.onSessionEnd(
      { sessionId: "session-1", messageCount: 1, reason: "idle" },
      { sessionId: "session-1" },
    );
    assert.equal(pending.timer, undefined);
  });

  it("drains pending llm output with placeholder request on session end", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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

    await backend.onSessionEnd(
      { sessionId: "session-1", messageCount: 1, reason: "idle" },
      { sessionId: "session-1" },
    );

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal(nf.calls.popScope.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, true);
    assert.equal(request.content.prompt, "");
  });

  it("attaches model timing only when timing is unambiguous", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted("call-1"), { runId: "run-1", sessionId: "session-1" });
    backend.onModelCallEnded(modelEnded("call-1", 42), { runId: "run-1", sessionId: "session-1" });
    backend.onLlmInput(llmInput(), { runId: "run-1", sessionId: "session-1" });
    backend.onLlmOutput(llmOutput(), { runId: "run-1", sessionId: "session-1" });

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.openclaw.duration_ms, 42);
    assert.equal(response.openclaw.outcome, "completed");
    assert.equal(backend.state().counters.llmSpansReplayed, 1);
  });

  it("emits ambiguity mark and does not attach ambiguous timing", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted("call-1"), { runId: "run-1", sessionId: "session-1" });
    backend.onModelCallEnded(modelEnded("call-1", 42), { runId: "run-1", sessionId: "session-1" });
    backend.onModelCallStarted(modelStarted("call-2"), { runId: "run-1", sessionId: "session-1" });
    backend.onModelCallEnded(modelEnded("call-2", 55), { runId: "run-1", sessionId: "session-1" });
    backend.onLlmInput(llmInput(), { runId: "run-1", sessionId: "session-1" });
    backend.onLlmOutput(llmOutput(), { runId: "run-1", sessionId: "session-1" });

    assert.ok(nf.calls.event.some((event) => event.name === "openclaw.model_call_timing_ambiguous"));
    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal("duration_ms" in response.openclaw, false);
  });

  it("emits unpaired mark for model_call_started without matching end on session drain", async () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted("call-1"), { runId: "run-1", sessionId: "session-1" });

    await backend.onSessionEnd(
      { sessionId: "session-1", messageCount: 1, reason: "idle" },
      { sessionId: "session-1" },
    );

    const unpaired = nf.calls.event.find((event) => event.name === "openclaw.model_call_timing_unpaired");
    assert.ok(unpaired);
    assert.deepEqual(unpaired.data, {
      runId: "run-1",
      callId: "call-1",
      provider: "openai",
      model: "gpt-4",
    });
  });

  it("strips prompt fields when prompt capture is disabled", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(
      nf,
      {},
      {
        includePrompts: false,
      },
    );

    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        systemPrompt: "classified system",
        prompt: "classified prompt",
        historyMessages: [{ role: "user", content: "classified history" }],
        imagesCount: 1,
      },
      { runId: "run-1", sessionId: "session-1" },
    );
    backend.onLlmOutput(llmOutput(), { runId: "run-1", sessionId: "session-1" });

    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal("prompt" in request.content, false);
    assert.equal("systemPrompt" in request.content, false);
    assert.deepEqual(request.content.messages, []);
    assert.equal(request.content.imagesCount, 1);
  });

  it("strips response content when response capture is disabled", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(
      nf,
      {},
      {
        includeResponses: false,
      },
    );

    backend.onLlmInput(llmInput(), { runId: "run-1", sessionId: "session-1" });
    backend.onLlmOutput(
      {
        ...llmOutput(),
        assistantTexts: ["classified response"],
      },
      { runId: "run-1", sessionId: "session-1" },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal("content" in response, false);
    assert.equal(response.assistant_texts_count, 1);
  });

  it("does not duplicate current prompt when history already ends with the same user message", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: "run-1",
        sessionId: "session-1",
        provider: "openai",
        model: "gpt-4",
        prompt: "hello",
        historyMessages: [{ role: "user", content: "hello" }],
        imagesCount: 0,
      },
      { runId: "run-1", sessionId: "session-1" },
    );
    backend.onLlmOutput(llmOutput(), { runId: "run-1", sessionId: "session-1" });

    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [{ role: "user", content: "hello" }]);
  });

  it("evicts stale expanded correlation records by TTL", () => {
    const nf = createNemoFlowRuntime();
    const backend = createBackend(nf, { recordTtlMs: 1 });
    const stalePendingOutput = {
      sessionKey: "session-1",
      sessionId: "session-1",
      runId: "old-run",
      provider: "openai",
      model: "gpt-4",
      event: llmOutput(),
      ctx: { runId: "old-run", sessionId: "session-1" },
      observedAtMs: 0,
      timer: setTimeout(() => {}, 10_000),
    };

    backend.state().llmInputs.set("stale-input", [
      {
        sessionKey: "session-1",
        sessionId: "session-1",
        runId: "old-run",
        provider: "openai",
        model: "gpt-4",
        prompt: "old",
        historyMessages: [],
        imagesCount: 0,
        observedAtMs: 0,
      },
    ]);
    backend.state().llmOutputsPendingInput.set("stale-output", [stalePendingOutput]);
    backend.state().modelTimingsByLlmKey.set("stale-timing", [
      {
        sessionKey: "session-1",
        sessionId: "session-1",
        runId: "old-run",
        callId: "old-call",
        provider: "openai",
        model: "gpt-4",
        consumed: false,
        observedAtMs: 0,
      },
    ]);

    backend.onLlmInput(llmInput(), { runId: "run-1", sessionId: "session-1" });

    assert.equal(backend.state().llmInputs.has("stale-input"), false);
    assert.equal(backend.state().llmOutputsPendingInput.has("stale-output"), false);
    assert.equal(stalePendingOutput.timer, undefined);
    assert.equal(backend.state().modelTimingsByLlmKey.has("stale-timing"), false);
  });
});

type ReplayRequest = {
  content: {
    messages?: unknown[];
    prompt?: string;
    systemPrompt?: string;
    imagesCount?: number;
    placeholderRequest?: boolean;
  };
};

type ReplayResponse = {
  content?: string;
  assistant_texts_count?: number;
  token_usage?: Record<string, number>;
  openclaw: Record<string, unknown>;
};

type TestNemoFlowRuntime = NemoFlowRuntimeModule & {
  calls: {
    pushScope: Array<{ name: string; scopeType: number; data: unknown }>;
    popScope: Array<{ handle: unknown; output: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown }>;
    setThreadScopeStack: unknown[];
    llmCall: Array<{ name: string; request: unknown; modelName: string | null | undefined }>;
    llmCallEnd: Array<{ handle: unknown; response: unknown }>;
    toolCall: Array<{ name: string; args: unknown }>;
    toolCallEnd: Array<{ handle: unknown; result: unknown; data: unknown }>;
  };
};

function createBackend(
  nf: TestNemoFlowRuntime,
  correlation: Partial<ReturnType<typeof parseConfig>["correlation"]> = {},
  capture: Partial<ReturnType<typeof parseConfig>["capture"]> = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: parseConfig({
      atif: { enabled: false },
      correlation,
      capture,
    }),
    logger: createLogger(),
    agentVersion: "test-version",
    resolvedAtifOutputDir: "/tmp/openclaw-state/plugins/nemo-flow/atif",
    markOutputDegraded: () => {},
  });
}

function createLogger(): PluginLogger {
  return {
    info: () => {},
    warn: () => {},
    error: () => {},
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
    ScopeType: { Agent: 0 } as NemoFlowRuntimeModule["ScopeType"],
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
    llmCall: (name, request, _handle, _attributes, _data, _metadata, modelName) => {
      const handle = { id: `llm-${nextScopeId++}` };
      calls.llmCall.push({ name, request, modelName });
      return handle as unknown as ReturnType<NemoFlowRuntimeModule["llmCall"]>;
    },
    llmCallEnd: (handle, response) => calls.llmCallEnd.push({ handle, response }),
    toolCall: (name, args) => {
      const handle = { id: `tool-${nextScopeId++}` };
      calls.toolCall.push({ name, args });
      return handle as unknown as ReturnType<NemoFlowRuntimeModule["toolCall"]>;
    },
    toolCallEnd: (handle, result, data) => calls.toolCallEnd.push({ handle, result, data }),
    AtifExporter: FakeAtifExporter,
    OpenTelemetrySubscriber: FakeSubscriber,
    OpenInferenceSubscriber: FakeSubscriber,
  };
}

function llmInput() {
  return {
    runId: "run-1",
    sessionId: "session-1",
    provider: "openai",
    model: "gpt-4",
    prompt: "hello",
    historyMessages: [],
    imagesCount: 0,
  };
}

function llmOutput() {
  return {
    runId: "run-1",
    sessionId: "session-1",
    provider: "openai",
    model: "gpt-4",
    assistantTexts: ["hi"],
  };
}

function modelStarted(callId: string) {
  return {
    runId: "run-1",
    callId,
    sessionId: "session-1",
    provider: "openai",
    model: "gpt-4",
  };
}

function modelEnded(callId: string, durationMs: number) {
  return {
    ...modelStarted(callId),
    durationMs,
    outcome: "completed" as const,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRefableTimer(timer: unknown): timer is { hasRef: () => boolean } {
  return (
    typeof timer === "object" &&
    timer !== null &&
    "hasRef" in timer &&
    typeof (timer as { hasRef?: unknown }).hasRef === "function"
  );
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
