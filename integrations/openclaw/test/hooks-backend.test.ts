// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HookReplayBackend tests covering session lifecycle, aliases, marks, and cleanup.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import { errorToJson, toJsonRecord } from '../src/hook-replay/marks.js';
import { HookReplayBackend } from '../src/hooks-backend.js';
import type { NemoRelayRuntimeModule } from '../src/modules.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';

describe('HookReplayBackend', () => {
  it('opens a session root and records aliases on session_start', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart(
      { sessionId: 'session-1', sessionKey: 'session-key-1', resumedFrom: 'previous-session' },
      { sessionId: 'session-1', sessionKey: 'session-key-1', agentId: 'agent-1' },
    );

    const session = backend.state().sessions.get('session-1');
    assert.ok(session);
    assert.equal(session.sessionId, 'session-1');
    assert.equal(session.sessionKey, 'session-key-1');
    assert.equal(session.agentId, 'agent-1');
    assert.equal(session.resumedFrom, 'previous-session');
    assert.equal(backend.state().sessionAliases.get('session-key-1'), 'session-1');
    assert.equal(nf.calls.pushScope.length, 1);
    assert.deepEqual(nf.calls.pushScope[0]?.metadata, {
      source: 'openclaw.session_start',
      hook_event_name: 'session_start',
      sessionId: 'session-1',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
    });
    assert.deepEqual(nf.calls.pushScope[0]?.input, {
      sessionId: 'session-1',
      source: 'session_start',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
      resumedFrom: 'previous-session',
    });
    assertNoOverclaimedHookMetadata(nf.calls.pushScope[0]?.metadata);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start'],
    );
    assert.deepEqual(nf.calls.event[0]?.metadata, {
      source: 'openclaw.session_start',
      hook_event_name: 'session_start',
      sessionId: 'session-1',
      sessionKey: 'session-key-1',
      agentId: 'agent-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[0]?.metadata);
  });

  it('emits session_start when a session is created lazily from llm_input', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'lazy-session',
        provider: 'openai',
        model: 'gpt',
        prompt: 'hello',
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: 'run-1', sessionId: 'lazy-session' },
    );

    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start'],
    );
    assert.deepEqual(nf.calls.event[0]?.data, {
      sessionId: 'lazy-session',
      source: 'lazy_session',
      runId: 'run-1',
    });
    assert.deepEqual(nf.calls.pushScope[0]?.metadata, {
      source: 'openclaw.lazy_session',
      sessionId: 'lazy-session',
      runId: 'run-1',
    });
    assert.deepEqual(nf.calls.pushScope[0]?.input, {
      sessionId: 'lazy-session',
      source: 'lazy_session',
      runId: 'run-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.pushScope[0]?.metadata);
    assert.deepEqual(nf.calls.event[0]?.metadata, {
      source: 'openclaw.lazy_session',
      sessionId: 'lazy-session',
      runId: 'run-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[0]?.metadata);
  });

  it('keeps concurrent sessions isolated by scope handle and alias', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: 'a', sessionKey: 'ka' }, { sessionId: 'a', sessionKey: 'ka' });
    backend.onSessionStart({ sessionId: 'b', sessionKey: 'kb' }, { sessionId: 'b', sessionKey: 'kb' });

    const first = backend.state().sessions.get('a');
    const second = backend.state().sessions.get('b');
    assert.ok(first?.rootHandle);
    assert.ok(second?.rootHandle);
    assert.notEqual(first.rootHandle, second.rootHandle);
    assert.equal(backend.state().sessionAliases.get('ka'), 'a');
    assert.equal(backend.state().sessionAliases.get('kb'), 'b');
  });

  it('drains before close, emits unpaired timing mark, and evicts session records', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: 'session-1' }, { sessionId: 'session-1' });
    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt',
        prompt: 'hello',
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmOutput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt',
        assistantTexts: ['hi'],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onModelCallEnded(
      {
        runId: 'run-1',
        callId: 'call-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt',
        durationMs: 42,
        outcome: 'completed',
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    await backend.onSessionEnd({ sessionId: 'session-1', messageCount: 3, reason: 'idle' }, { sessionId: 'session-1' });

    assert.equal(backend.state().sessions.size, 0);
    assert.equal(backend.state().sessionAliases.size, 0);
    assert.equal(backend.state().llmInputs.size, 0);
    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(backend.state().modelCallsByCallId.size, 0);
    assert.equal(backend.state().modelTimingsByLlmKey.size, 0);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.model_call_timing_unpaired', 'openclaw.session_end'],
    );
    assert.deepEqual(nf.calls.event.at(-1)?.metadata, {
      source: 'openclaw.session_end',
      hook_event_name: 'session_end',
      sessionId: 'session-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event.at(-1)?.metadata);
    assert.equal(nf.calls.popScope.length, 1);
  });

  it('emits blocked tool marks from after_tool_call only', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: 'session-1', sessionKey: 'sk' }, { sessionId: 'session-1', sessionKey: 'sk' });
    backend.onAfterToolCall(
      {
        toolName: 'dangerous_tool',
        params: {},
        toolCallId: 'tool-call-1',
        result: { details: { status: 'blocked', deniedReason: 'policy' } },
        durationMs: 5,
      },
      { sessionKey: 'sk', runId: 'run-1', toolName: 'dangerous_tool', toolCallId: 'tool-call-1' },
    );

    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.tool_blocked'],
    );
    assert.deepEqual(nf.calls.event[1]?.data, {
      toolName: 'dangerous_tool',
      toolCallId: 'tool-call-1',
      runId: 'run-1',
      blocked: true,
      deniedReason: 'policy',
      durationMs: 5,
    });
    assert.deepEqual(nf.calls.event[1]?.metadata, {
      source: 'openclaw.after_tool_call',
      hook_event_name: 'after_tool_call',
      sessionId: 'session-1',
      sessionKey: 'sk',
      runId: 'run-1',
      toolCallId: 'tool-call-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[1]?.metadata);
  });

  it('backdates a lazy session root to tool start when after_tool_call is the first hook', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000], () => {
      backend.onAfterToolCall(
        {
          toolName: 'read_file',
          params: { path: '/workspace/file.txt' },
          toolCallId: 'tool-call-1',
          runId: 'run-1',
          result: { text: 'ok' },
          durationMs: 7,
        },
        {
          runId: 'run-1',
          sessionId: 'session-1',
          sessionKey: 'session-key-1',
          toolCallId: 'tool-call-1',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 1);
    assert.equal(nf.calls.pushScope[0]?.timestamp, 993_000);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start'],
    );
    assert.equal(nf.calls.event[0]?.timestamp, 993_000);
  });

  it('safe replay restores the previous scope stack and fails open', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart({ sessionId: 'session-1' }, { sessionId: 'session-1' });
    const session = backend.state().sessions.get('session-1');
    assert.ok(session);

    assert.doesNotThrow(() => {
      backend.emitCapturedUnderSession('test_throw', session, () => {
        throw new Error('boom');
      });
    });

    assert.equal(backend.state().counters.replayErrors, 1);
    assert.equal(nf.calls.setThreadScopeStack.at(-1), nf.previousStack);
  });

  it('bounds repeated replay warnings by label', () => {
    const nf = createNemoRelayRuntime();
    const logger = createLogger();
    const backend = createBackend(nf, logger);

    backend.safeReplay('same_failure', undefined, () => {
      throw new Error('first');
    });
    backend.safeReplay('same_failure', undefined, () => {
      throw new Error('second');
    });

    assert.equal(logger.messages.warn.length, 1);
    assert.match(logger.messages.warn[0] ?? '', /same_failure/);
    assert.equal(backend.state().counters.replayErrors, 2);
  });

  it('returns undefined from before_agent_finalize', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    const result = backend.onBeforeAgentFinalize(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        stopHookActive: false,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(result, undefined);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.before_agent_finalize'],
    );
    assert.deepEqual(nf.calls.event[1]?.metadata, {
      source: 'openclaw.before_agent_finalize',
      hook_event_name: 'before_agent_finalize',
      sessionId: 'session-1',
      runId: 'run-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[1]?.metadata);
    assert.equal(nf.calls.event[1]?.handle, nf.calls.event[0]?.handle);
  });

  it('keeps gateway stop reason out of the root session output when a final answer is known', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', provider: 'openai', model: 'gpt', content: 'Final answer.' },
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    await backend.drainForGatewayStop('gateway stopping');

    assert.deepEqual(nf.calls.popScope[0]?.output, {
      content: 'Final answer.',
      source: 'openclaw.agent_end',
      runId: 'run-1',
      success: true,
    });
    assert.deepEqual(nf.calls.event[1]?.metadata, {
      source: 'openclaw.agent_end',
      hook_event_name: 'agent_end',
      sessionId: 'session-1',
      runId: 'run-1',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[1]?.metadata);
    assert.equal(nf.calls.event[1]?.handle, nf.calls.event[0]?.handle);
    assert.deepEqual(nf.calls.event.at(-1)?.data, { reason: 'gateway stopping' });
  });

  it('records subagent marks under the requester alias without merging child session identity', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart(
      { sessionId: 'parent-session', sessionKey: 'parent-key' },
      { sessionId: 'parent-session', sessionKey: 'parent-key' },
    );
    backend.onSubagentSpawned(
      {
        childSessionKey: 'child-key',
        agentId: 'child-agent',
        mode: 'run',
        threadRequested: false,
        runId: 'child-run',
      },
      { requesterSessionKey: 'parent-key', childSessionKey: 'child-key', runId: 'child-run' },
    );

    assert.equal(backend.state().sessionAliases.get('child-key'), undefined);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.subagent_spawned'],
    );
    assert.deepEqual(nf.calls.event[1]?.metadata, {
      source: 'openclaw.subagent_spawned',
      hook_event_name: 'subagent_spawned',
      sessionId: 'parent-session',
      sessionKey: 'parent-key',
      runId: 'child-run',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[1]?.metadata);
    assert.equal(nf.calls.event[1]?.handle, nf.calls.event[0]?.handle);
  });

  it('records subagent end marks under the requester alias without merging child session identity', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onSessionStart(
      { sessionId: 'parent-session', sessionKey: 'parent-key' },
      { sessionId: 'parent-session', sessionKey: 'parent-key' },
    );
    backend.onSubagentEnded(
      {
        targetSessionKey: 'child-key',
        targetKind: 'subagent',
        reason: 'completed',
        outcome: 'ok',
        runId: 'child-run',
      },
      { requesterSessionKey: 'parent-key', childSessionKey: 'child-key', runId: 'child-run' },
    );

    assert.equal(backend.state().sessionAliases.get('child-key'), undefined);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.subagent_ended'],
    );
    assert.deepEqual(nf.calls.event[1]?.metadata, {
      source: 'openclaw.subagent_ended',
      hook_event_name: 'subagent_ended',
      sessionId: 'parent-session',
      sessionKey: 'parent-key',
      runId: 'child-run',
    });
    assertNoOverclaimedHookMetadata(nf.calls.event[1]?.metadata);
    assert.equal(nf.calls.event[1]?.handle, nf.calls.event[0]?.handle);
  });

  it('nests child session activity under the requester when subagent_spawned arrives first', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000, 2000, 3000, 4000, 5000], () => {
      backend.onSessionStart(
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
      );
      backend.onSubagentSpawned(
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
          mode: 'run',
          threadRequested: false,
          runId: 'child-run',
        },
        {
          requesterSessionKey: 'parent-key',
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
      backend.onSessionStart(
        { sessionId: 'child-session', sessionKey: 'agent:child-agent:subagent:child-key' },
        {
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
        },
      );
      backend.onAgentEnd(
        {
          runId: 'child-run',
          messages: [{ role: 'assistant', provider: 'openai', model: 'gpt', content: 'Child answer.' }],
          success: true,
        },
        {
          runId: 'child-run',
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 2);
    assert.equal(nf.calls.pushScope[1]?.parentHandle, nf.calls.pushScope[0]?.handle);
    assert.equal(nf.calls.pushScope[1]?.timestamp, 3000 * 1000);
    assert.deepEqual(nf.calls.pushScope[1]?.metadata, {
      source: 'openclaw.session_start',
      hook_event_name: 'session_start',
      sessionId: 'child-session',
      sessionKey: 'agent:child-agent:subagent:child-key',
      agentId: 'child-agent',
      runId: 'child-run',
      nemo_relay_scope_role: 'subagent',
    });

    const spawnMark = nf.calls.event.find((event) => event.name === 'openclaw.subagent_spawned');
    const childAgentEnd = nf.calls.event.find((event) => event.name === 'openclaw.agent_end');
    assert.equal(spawnMark?.handle, nf.calls.pushScope[0]?.handle);
    assert.equal(childAgentEnd?.handle, nf.calls.pushScope[1]?.handle);
    assert.equal(spawnMark?.timestamp, 2000 * 1000);
    assert.equal(childAgentEnd?.timestamp, 4000 * 1000);
  });

  it('keeps session_start-first child activity on the existing child scope after subagent_spawned', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000, 2000, 3000, 4000], () => {
      backend.onSessionStart(
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
      );
      backend.onSessionStart(
        { sessionId: 'child-session', sessionKey: 'agent:child-agent:subagent:child-key' },
        {
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
        },
      );
      backend.onSubagentSpawned(
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
          mode: 'run',
          threadRequested: false,
          runId: 'child-run',
        },
        {
          requesterSessionKey: 'parent-key',
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
      backend.onAgentEnd(
        {
          runId: 'child-run',
          messages: [{ role: 'assistant', provider: 'openai', model: 'gpt', content: 'Child answer.' }],
          success: true,
        },
        {
          runId: 'child-run',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 2);
    assert.equal(nf.calls.pushScope[1]?.parentHandle, nf.calls.pushScope[0]?.handle);
    assert.equal(nf.calls.pushScope[1]?.timestamp, 2000 * 1000);
    assert.deepEqual(nf.calls.pushScope[1]?.metadata, {
      source: 'openclaw.session_start',
      hook_event_name: 'session_start',
      sessionId: 'child-session',
      sessionKey: 'agent:child-agent:subagent:child-key',
      agentId: 'child-agent',
      runId: 'child-run',
      nemo_relay_scope_role: 'subagent',
    });

    const childAgentEndEvents = nf.calls.event.filter(
      (event) => event.name === 'openclaw.agent_end' && event.handle === nf.calls.pushScope[1]?.handle,
    );
    assert.equal(childAgentEndEvents.length, 1);
    assert.equal(childAgentEndEvents[0]?.timestamp, 4000 * 1000);
  });

  it('expires stale pending subagent lineage before a late child session_start can reuse it', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, createLogger(), {
      config: parseConfig({ correlation: { recordTtlMs: 1 } }),
    });

    withMockedNow([1000, 1000, 1002], () => {
      backend.onSessionStart(
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
      );
      backend.onSubagentSpawned(
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
          mode: 'run',
          threadRequested: false,
          runId: 'child-run',
        },
        {
          requesterSessionKey: 'parent-key',
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
      backend.onSessionStart(
        { sessionId: 'child-session', sessionKey: 'agent:child-agent:subagent:child-key' },
        {
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 1);
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.subagent_spawned'],
    );
  });

  it('reconciles a deferred child session with later session_start before lineage promotion', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000, 2000, 3000, 4000], () => {
      backend.onSessionStart(
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
      );
      backend.onAgentEnd(
        {
          runId: 'child-run',
          messages: [{ role: 'assistant', provider: 'openai', model: 'gpt', content: 'Child answer.' }],
          success: true,
        },
        {
          runId: 'child-run',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
      backend.onSessionStart(
        { sessionId: 'child-session', sessionKey: 'agent:child-agent:subagent:child-key' },
        {
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
      backend.onSubagentSpawned(
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
          mode: 'run',
          threadRequested: false,
          runId: 'child-run',
        },
        {
          requesterSessionKey: 'parent-key',
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
      backend.onAgentEnd(
        {
          runId: 'child-run',
          messages: [{ role: 'assistant', provider: 'openai', model: 'gpt', content: 'Follow-up answer.' }],
          success: true,
        },
        {
          runId: 'child-run',
          agentId: 'child-agent',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 2);
    assert.equal(nf.calls.pushScope[1]?.parentHandle, nf.calls.pushScope[0]?.handle);
    assert.equal(nf.calls.pushScope[1]?.timestamp, 3000 * 1000);

    const childSessionStartIndex = nf.calls.event.findIndex((event) => event.name === 'openclaw.session_start' && event.handle === nf.calls.pushScope[1]?.handle);
    const childAgentEndIndex = nf.calls.event.findIndex((event) => event.name === 'openclaw.agent_end' && event.handle === nf.calls.pushScope[1]?.handle);
    const spawnMarkIndex = nf.calls.event.findIndex((event) => event.name === 'openclaw.subagent_spawned');
    const childAgentEndEvents = nf.calls.event.filter(
      (event) => event.name === 'openclaw.agent_end' && event.handle === nf.calls.pushScope[1]?.handle,
    );
    assert.ok(childSessionStartIndex >= 0);
    assert.ok(childAgentEndIndex > childSessionStartIndex);
    assert.ok(spawnMarkIndex >= 0);
    assert.equal(childAgentEndEvents.length, 2);
    assert.equal(nf.calls.event[spawnMarkIndex]?.handle, nf.calls.pushScope[0]?.handle);
    assert.equal(nf.calls.event[childSessionStartIndex]?.timestamp, 3000 * 1000);
    assert.equal(nf.calls.event[childAgentEndIndex]?.timestamp, 2000 * 1000);
    assert.equal(nf.calls.event[spawnMarkIndex]?.timestamp, 4000 * 1000);
  });

  it('materializes a child root with the child session key when only child lineage is available', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000], () => {
      backend.onSubagentEnded(
        {
          targetSessionKey: 'agent:child-agent:subagent:child-key',
          targetKind: 'subagent',
          reason: 'completed',
          runId: 'child-run',
        },
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
    });

    assert.equal(nf.calls.pushScope.length, 1);
    assert.equal(nf.calls.pushScope[0]?.parentHandle, null);
    assert.equal(nf.calls.pushScope[0]?.timestamp, 1000 * 1000);
    assert.deepEqual(nf.calls.pushScope[0]?.metadata, {
      source: 'openclaw.lazy_session',
      sessionId: 'agent:child-agent:subagent:child-key',
      sessionKey: 'agent:child-agent:subagent:child-key',
      runId: 'child-run',
      nemo_relay_scope_role: 'subagent',
    });
    assert.deepEqual(
      nf.calls.event.map((event) => event.name),
      ['openclaw.session_start', 'openclaw.subagent_ended'],
    );
    assert.equal(nf.calls.event[0]?.timestamp, 1000 * 1000);
    assert.equal(nf.calls.event[1]?.timestamp, 1000 * 1000);
    assert.equal(nf.calls.event[1]?.handle, nf.calls.pushScope[0]?.handle);
  });

  it('keeps deferred child model timing correlated when explicit session_start arrives later', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    withMockedNow([1000, 2000, 3000, 4000, 5000, 6000], () => {
      backend.onSessionStart(
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
        { sessionId: 'parent-session', sessionKey: 'parent-key' },
      );
      backend.onModelCallEnded(
        {
          runId: 'child-run',
          callId: 'call-1',
          sessionKey: 'agent:child-agent:subagent:child-key',
          provider: 'openai',
          model: 'gpt',
          durationMs: 12,
          outcome: 'completed',
        },
        {
          runId: 'child-run',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
      backend.onSessionStart(
        { sessionId: 'child-session', sessionKey: 'agent:child-agent:subagent:child-key' },
        {
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
      backend.onSubagentSpawned(
        {
          childSessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
          mode: 'run',
          threadRequested: false,
          runId: 'child-run',
        },
        {
          requesterSessionKey: 'parent-key',
          childSessionKey: 'agent:child-agent:subagent:child-key',
          runId: 'child-run',
        },
      );
      backend.onLlmInput(
        {
          runId: 'child-run',
          sessionId: 'child-session',
          provider: 'openai',
          model: 'gpt',
          prompt: 'hello',
          historyMessages: [],
          imagesCount: 0,
        },
        {
          runId: 'child-run',
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
      backend.onLlmOutput(
        {
          runId: 'child-run',
          sessionId: 'child-session',
          provider: 'openai',
          model: 'gpt',
          assistantTexts: ['child answer'],
        },
        {
          runId: 'child-run',
          sessionId: 'child-session',
          sessionKey: 'agent:child-agent:subagent:child-key',
          agentId: 'child-agent',
        },
      );
    });

    assert.equal(nf.calls.llmCall.length, 1);
    assert.deepEqual(nf.calls.llmCall[0]?.metadata, {
      source: 'openclaw.llm_output',
      runId: 'child-run',
      sessionId: 'child-session',
      sessionKey: 'agent:child-agent:subagent:child-key',
      agentId: 'child-agent',
      provider: 'openai',
      model: 'gpt',
      callId: 'call-1',
    });

    await backend.onSessionEnd(
      {
        sessionId: 'child-session',
        sessionKey: 'agent:child-agent:subagent:child-key',
        messageCount: 1,
        reason: 'idle',
      },
      {
        sessionId: 'child-session',
        sessionKey: 'agent:child-agent:subagent:child-key',
        agentId: 'child-agent',
      },
    );

    assert.equal(backend.state().modelCallsByCallId.size, 0);
    assert.equal(backend.state().modelTimingsByLlmKey.size, 0);
    assert.equal(
      nf.calls.event.some((event) => event.name === 'openclaw.model_call_timing_unpaired'),
      false,
    );
  });

  it('normalizes circular replay payloads before NAPI boundaries', () => {
    const payload: Record<string, unknown> = { ok: true };
    payload.self = payload;

    assert.deepEqual(toJsonRecord(payload), {
      ok: true,
      self: { ok: true, self: '[Circular]' },
    });
    assert.deepEqual(
      toJsonRecord({
        finite: 42,
        nan: Number.NaN,
        positiveInfinity: Number.POSITIVE_INFINITY,
        negativeInfinity: Number.NEGATIVE_INFINITY,
      }),
      {
        finite: 42,
        nan: null,
        positiveInfinity: null,
        negativeInfinity: null,
      },
    );
    assert.deepEqual(errorToJson(new Error('boom')).message, 'boom');
  });

  it('normalizes prototype keys without mutating output prototypes', () => {
    const payload: Record<string, unknown> = {};
    Object.defineProperty(payload, '__proto__', {
      enumerable: true,
      value: { polluted: true },
    });

    const normalized = toJsonRecord(payload);

    assert.equal(Object.getPrototypeOf(normalized), Object.prototype);
    assert.deepEqual(normalized['__proto__'], { polluted: true });
    assert.equal(({} as Record<string, unknown>).polluted, undefined);
  });
});

type TestNemoRelayRuntime = NemoRelayRuntimeModule & {
  previousStack: { id: 'previous' };
  calls: {
    pushScope: Array<{
      name: string;
      scopeType: number;
      handle: unknown;
      parentHandle: unknown;
      data: unknown;
      metadata: unknown;
      input: unknown;
      timestamp: unknown;
    }>;
    popScope: Array<{ handle: unknown; output: unknown; timestamp: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown; metadata: unknown; timestamp: unknown }>;
    llmCall: Array<{
      name: string;
      handle: unknown;
      request: unknown;
      parentHandle: unknown;
      data: unknown;
      metadata: unknown;
      modelName: unknown;
      timestamp: unknown;
    }>;
    llmCallEnd: Array<{ handle: unknown; response: unknown; data: unknown; metadata: unknown; timestamp: unknown }>;
    setThreadScopeStack: unknown[];
    toolConditionalExecution: Array<{ name: string; args: unknown }>;
  };
};

type TestLogger = PluginLogger & {
  messages: {
    warn: string[];
  };
};

function createBackend(
  nf: TestNemoRelayRuntime,
  logger = createLogger(),
  options: {
    config?: ReturnType<typeof parseConfig>;
  } = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: options.config ?? parseConfig({}),
    logger,
    agentVersion: 'test-version',
  });
}

function createLogger(): TestLogger {
  const messages: TestLogger['messages'] = { warn: [] };
  return {
    messages,
    info: () => {},
    warn: (message) => messages.warn.push(message),
    error: () => {},
  };
}

function createNemoRelayRuntime(): TestNemoRelayRuntime {
  let nextScopeId = 0;
  const previousStack = { id: 'previous' as const };
  const calls: TestNemoRelayRuntime['calls'] = {
    pushScope: [],
    popScope: [],
    event: [],
    llmCall: [],
    llmCallEnd: [],
    setThreadScopeStack: [],
    toolConditionalExecution: [],
  };

  return {
    ScopeType: { Agent: 0 } as NemoRelayRuntimeModule['ScopeType'],
    previousStack,
    calls,
    createScopeStack: () =>
      ({ id: `stack-${nextScopeId++}` }) as unknown as ReturnType<NemoRelayRuntimeModule['createScopeStack']>,
    currentScopeStack: () => previousStack as unknown as ReturnType<NemoRelayRuntimeModule['currentScopeStack']>,
    setThreadScopeStack: (stack) => calls.setThreadScopeStack.push(stack),
    pushScope: (...args: Parameters<NemoRelayRuntimeModule['pushScope']>) => {
      const [name, scopeType, parentHandle, , data, metadata, input, timestamp] = args;
      const handle = { id: `scope-${nextScopeId++}` };
      calls.pushScope.push({ name, scopeType, handle, parentHandle, data, metadata, input, timestamp });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['pushScope']>;
    },
    popScope: (...args: Parameters<NemoRelayRuntimeModule['popScope']>) => {
      const [handle, output, timestamp] = args;
      calls.popScope.push({ handle, output, timestamp });
    },
    event: (...args: Parameters<NemoRelayRuntimeModule['event']>) => {
      const [name, handle, data, metadata, timestamp] = args;
      calls.event.push({ name, handle, data, metadata, timestamp });
    },
    llmCall: (...args: Parameters<NemoRelayRuntimeModule['llmCall']>) => {
      const [name, request, parentHandle, , data, metadata, modelName, timestamp] = args;
      const handle = { id: `llm-${nextScopeId++}` };
      calls.llmCall.push({ name, handle, request, parentHandle, data, metadata, modelName, timestamp });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['llmCall']>;
    },
    llmCallEnd: (...args: Parameters<NemoRelayRuntimeModule['llmCallEnd']>) => {
      const [handle, response, data, metadata, timestamp] = args;
      calls.llmCallEnd.push({ handle, response, data, metadata, timestamp });
    },
    toolCall: () => ({}) as unknown as ReturnType<NemoRelayRuntimeModule['toolCall']>,
    toolCallEnd: () => {},
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

function withMockedNow<T>(values: number[], fn: () => T): T {
  const originalNow = Date.now;
  let index = 0;
  Date.now = () => values[Math.min(index++, values.length - 1)] ?? originalNow();
  try {
    return fn();
  } finally {
    Date.now = originalNow;
  }
}
