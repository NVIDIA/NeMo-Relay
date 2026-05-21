// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Failure-model tests that ensure hook replay fails open and records diagnostics.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import { HookReplayBackend } from '../src/hooks-backend.js';
import type { NemoRelayRuntimeModule } from '../src/modules.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';

describe('Replay failure model', () => {
  it('grace timer replay failure is caught and counted', async () => {
    const logger = createLogger();
    const backend = new HookReplayBackend({
      nf: createThrowingLlmRuntime(),
      config: parseConfig({
        correlation: { llmOutputGraceMs: 1 },
      }),
      logger,
      agentVersion: 'test-version',
    });

    backend.onLlmOutput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        assistantTexts: ['hi'],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    await waitFor(() => backend.state().counters.replayErrors === 1 && logger.messages.warn.length >= 1);

    assert.equal(backend.state().counters.replayErrors, 1);
    assert.equal(logger.messages.warn.length, 1);
    assert.match(logger.messages.warn[0] ?? '', /llm_output/);
  });
});

type TestLogger = PluginLogger & {
  messages: {
    warn: string[];
  };
};

function createLogger(): TestLogger {
  const messages: TestLogger['messages'] = { warn: [] };
  return {
    messages,
    info: () => {},
    warn: (message) => messages.warn.push(message),
    error: () => {},
  };
}

function createThrowingLlmRuntime(): NemoRelayRuntimeModule {
  let nextScopeId = 0;
  const previousStack = { id: 'previous' };
  return {
    ScopeType: { Agent: 0 } as NemoRelayRuntimeModule['ScopeType'],
    createScopeStack: () =>
      ({ id: `stack-${nextScopeId++}` }) as unknown as ReturnType<NemoRelayRuntimeModule['createScopeStack']>,
    currentScopeStack: () => previousStack as unknown as ReturnType<NemoRelayRuntimeModule['currentScopeStack']>,
    setThreadScopeStack: () => {},
    pushScope: () => ({ id: `scope-${nextScopeId++}` }) as unknown as ReturnType<NemoRelayRuntimeModule['pushScope']>,
    popScope: () => {},
    event: () => {},
    llmCall: () => {
      throw new Error('llmCall failed');
    },
    llmCallEnd: () => {},
    toolCall: () => ({}) as unknown as ReturnType<NemoRelayRuntimeModule['toolCall']>,
    toolCallEnd: () => {},
    toolConditionalExecution: async () => {},
  };
}

async function waitFor(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > timeoutMs) {
      throw new Error('timed out waiting for replay failure state');
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
