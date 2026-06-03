// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool replay tests for stripped payloads, trusted payload capture, and blocked tools.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import { HookReplayBackend } from '../src/hooks-backend.js';
import type { NemoRelayRuntimeModule } from '../src/modules.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';

describe('Tool replay', () => {
  it('replays after_tool_call with stripped payloads by default', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onAfterToolCall(
      {
        toolName: 'read_file',
        params: { path: '/secret', token: 'value' },
        toolCallId: 'tool-call-1',
        runId: 'run-1',
        result: { text: 'secret' },
        durationMs: 7,
      },
      {
        runId: 'run-1',
        sessionId: 'session-1',
        sessionKey: 'session-key-1',
        agentId: 'agent-1',
        toolCallId: 'tool-call-1',
      },
    );

    assert.equal(nf.calls.toolCall.length, 1);
    assert.equal(nf.calls.toolCallEnd.length, 1);
    assert.equal(backend.state().counters.toolSpansReplayed, 1);
    assert.deepEqual(nf.calls.toolCall[0]?.args, {
      stripped: true,
      argKeys: ['path', 'token'],
    });
    assert.equal(nf.calls.toolCall[0]?.data, null);
    assert.deepEqual(nf.calls.toolCall[0]?.metadata, {
      source: 'openclaw.after_tool_call',
      runId: 'run-1',
      sessionId: 'session-1',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
      toolCallId: 'tool-call-1',
      durationMs: 7,
    });
    assertNoOverclaimedHookMetadata(nf.calls.toolCall[0]?.metadata);
    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, {
      content: 'Tool read_file completed.',
      openclaw: {
        toolName: 'read_file',
        toolCallId: 'tool-call-1',
        durationMs: 7,
        hasError: false,
        stripped: true,
        resultKeys: ['text'],
      },
    });
    assert.equal(nf.calls.toolCallEnd[0]?.data, null);
    assert.deepEqual(nf.calls.toolCallEnd[0]?.metadata, {
      source: 'openclaw.after_tool_call',
      runId: 'run-1',
      sessionId: 'session-1',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
      toolCallId: 'tool-call-1',
      durationMs: 7,
    });
    assertNoOverclaimedHookMetadata(nf.calls.toolCallEnd[0]?.metadata);
  });

  it('captures full tool payloads only when trusted config opts in', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, {
      capture: {
        stripToolArgs: false,
        stripToolResults: false,
      },
    });

    backend.onAfterToolCall(
      {
        toolName: 'read_file',
        params: { path: '/workspace/file.txt' },
        toolCallId: 'tool-call-1',
        runId: 'run-1',
        result: { text: 'ok' },
        durationMs: 7,
      },
      { runId: 'run-1', sessionId: 'session-1', toolCallId: 'tool-call-1' },
    );

    assert.deepEqual(nf.calls.toolCall[0]?.args, { path: '/workspace/file.txt' });
    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, {
      content: 'Tool read_file completed.',
      openclaw: {
        toolName: 'read_file',
        toolCallId: 'tool-call-1',
        durationMs: 7,
        hasError: false,
        stripped: false,
        resultKeys: ['text'],
      },
      result: { text: 'ok' },
    });
    assert.equal(nf.calls.toolCallEnd[0]?.data, null);
  });

  it('passes non-null tool end payload when result and error are missing', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, {
      capture: {
        stripToolResults: false,
      },
    });

    backend.onAfterToolCall(
      {
        toolName: 'noop',
        params: {},
        toolCallId: 'tool-call-1',
        runId: 'run-1',
      },
      { runId: 'run-1', sessionId: 'session-1', toolCallId: 'tool-call-1' },
    );

    assert.deepEqual(nf.calls.toolCallEnd[0]?.result, {
      content: 'Tool noop completed.',
      openclaw: {
        toolName: 'noop',
        toolCallId: 'tool-call-1',
        hasError: false,
        stripped: false,
      },
      result: null,
    });
    assert.equal(nf.calls.toolCallEnd[0]?.data, null);
  });

  it('emits blocked tool mark instead of successful tool span', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onAfterToolCall(
      {
        toolName: 'dangerous_tool',
        params: {},
        toolCallId: 'tool-call-1',
        runId: 'run-1',
        result: { details: { status: 'blocked', deniedReason: 'policy' } },
        durationMs: 3,
      },
      {
        runId: 'run-1',
        sessionId: 'session-1',
        sessionKey: 'session-key-1',
        agentId: 'agent-1',
        toolCallId: 'tool-call-1',
      },
    );

    assert.equal(nf.calls.toolCall.length, 0);
    assert.ok(nf.calls.event.some((event) => event.name === 'openclaw.tool_blocked'));
    assert.deepEqual(nf.calls.event.find((event) => event.name === 'openclaw.tool_blocked')?.metadata, {
      source: 'openclaw.after_tool_call',
      hook_event_name: 'after_tool_call',
      sessionId: 'session-1',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
      runId: 'run-1',
      toolCallId: 'tool-call-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event.find((event) => event.name === 'openclaw.tool_blocked')?.metadata);
  });

  it('runs tool guardrails even when no session key is available', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    await backend.onBeforeToolCall({ toolName: 'shell', params: { command: 'pwd' } }, {});

    assert.deepEqual(nf.calls.toolConditionalExecution, [{ name: 'shell', args: { command: 'pwd' } }]);
    assert.equal(nf.calls.setThreadScopeStack.length, 0);
  });
});

type TestNemoRelayRuntime = NemoRelayRuntimeModule & {
  calls: {
    pushScope: Array<{ name: string; scopeType: number; data: unknown }>;
    popScope: Array<{ handle: unknown; output: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown; metadata: unknown }>;
    setThreadScopeStack: unknown[];
    llmCall: Array<{ name: string; request: unknown }>;
    llmCallEnd: Array<{ handle: unknown; response: unknown }>;
    toolCall: Array<{ name: string; args: unknown; data: unknown; metadata: unknown }>;
    toolCallEnd: Array<{ handle: unknown; result: unknown; data: unknown; metadata: unknown }>;
    toolConditionalExecution: Array<{ name: string; args: unknown }>;
  };
};

function createBackend(
  nf: TestNemoRelayRuntime,
  overrides: {
    capture?: Partial<ReturnType<typeof parseConfig>['capture']>;
  } = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: parseConfig({
      capture: overrides.capture,
    }),
    logger: createLogger(),
    agentVersion: 'test-version',
  });
}

function createLogger(): PluginLogger {
  return {
    info: () => {},
    warn: () => {},
    error: () => {},
  };
}

function createNemoRelayRuntime(): TestNemoRelayRuntime {
  let nextScopeId = 0;
  const previousStack = { id: 'previous' };
  const calls: TestNemoRelayRuntime['calls'] = {
    pushScope: [],
    popScope: [],
    event: [],
    setThreadScopeStack: [],
    llmCall: [],
    llmCallEnd: [],
    toolCall: [],
    toolCallEnd: [],
    toolConditionalExecution: [],
  };

  return {
    ScopeType: { Agent: 0 } as NemoRelayRuntimeModule['ScopeType'],
    calls,
    createScopeStack: () =>
      ({ id: `stack-${nextScopeId++}` }) as unknown as ReturnType<NemoRelayRuntimeModule['createScopeStack']>,
    currentScopeStack: () => previousStack as unknown as ReturnType<NemoRelayRuntimeModule['currentScopeStack']>,
    setThreadScopeStack: (stack) => calls.setThreadScopeStack.push(stack),
    pushScope: (name, scopeType, _handle, _attributes, data) => {
      const handle = { id: `scope-${nextScopeId++}` };
      calls.pushScope.push({ name, scopeType, data });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['pushScope']>;
    },
    popScope: (handle, output) => calls.popScope.push({ handle, output }),
    event: (name, handle, data, metadata) => calls.event.push({ name, handle, data, metadata }),
    llmCall: (name, request) => {
      const handle = { id: `llm-${nextScopeId++}` };
      calls.llmCall.push({ name, request });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['llmCall']>;
    },
    llmCallEnd: (handle, response) => calls.llmCallEnd.push({ handle, response }),
    toolCall: (name, args, _handle, _attributes, data, metadata) => {
      const handle = { id: `tool-${nextScopeId++}` };
      calls.toolCall.push({ name, args, data, metadata });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['toolCall']>;
    },
    toolCallEnd: (handle, result, data, metadata) => calls.toolCallEnd.push({ handle, result, data, metadata }),
    toolConditionalExecution: async (name, args) => {
      calls.toolConditionalExecution.push({ name, args });
    },
  };
}

function assertNoOverclaimedHookMetadata(metadata: unknown): void {
  assert.ok(metadata && typeof metadata === 'object');
  const record = metadata as Record<string, unknown>;
  assert.equal('agent_kind' in record, false);
  assert.equal('provider_payload_exact' in record, false);
  assert.equal('fidelity_source' in record, false);
  assert.equal('gateway_path' in record, false);
  assert.equal('gateway_route' in record, false);
  assert.equal('correlation' in record, false);
}
