// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LLM replay tests for hook correlation, token accounting, capture policy, and diagnostics.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import { HookReplayBackend } from '../src/hooks-backend.js';
import type { NemoRelayRuntimeModule } from '../src/modules.js';
import type { PluginLogger } from 'openclaw/plugin-sdk/plugin-entry';

describe('LLM replay', () => {
  it('replays llm output with buffered input under the session root', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        systemPrompt: 'be concise',
        prompt: 'hello',
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: 'run-1', sessionId: 'session-1', agentId: 'agent-1' },
    );
    backend.onLlmOutput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        assistantTexts: ['hi'],
        resolvedRef: 'provider/model',
        harnessId: 'harness-1',
        usage: { input: 2, output: 3 },
      },
      { runId: 'run-1', sessionId: 'session-1', agentId: 'agent-1' },
    );

    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal(backend.state().counters.llmSpansReplayed, 1);
    assert.equal(backend.state().llmInputs.size, 0);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [{ role: 'user', content: 'hello' }]);
    assert.equal(request.content.systemPrompt, 'be concise');
    assert.equal(nf.calls.llmCall[0]?.data, null);
    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.content, 'hi');
    assert.equal(nf.calls.llmCallEnd[0]?.data, null);
    assert.deepEqual(response.usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });
    assert.equal('token_usage' in response, false);
  });

  it('preserves tool calls from llm output lastAssistant for ATIF and OpenInference', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      {
        ...llmOutput(),
        assistantTexts: [],
        lastAssistant: {
          role: 'assistant',
          tool_calls: [
            {
              id: 'call-search-1',
              type: 'function',
              function: {
                name: 'web_search',
                arguments: { query: 'private search' },
              },
            },
          ],
        },
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.content, 'tool calls: web_search');
    assert.deepEqual(response.tool_calls, [
      {
        id: 'call-search-1',
        type: 'function',
        function: {
          name: 'web_search',
          arguments: { stripped: true },
        },
      },
    ]);
    assert.deepEqual((response.openclaw as ResponseOpenClaw).assistant_tool_call_names, ['web_search']);
  });

  it('normalizes Anthropic messages content blocks from message-write replay', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);
    const assistantMessage = {
      role: 'assistant',
      provider: 'anthropic',
      model: 'claude-sonnet-4',
      content: [
        { type: 'text', text: 'I will search for it.' },
        {
          type: 'tool_use',
          id: 'toolu-search-1',
          name: 'web_search',
          input: { query: 'release notes' },
        },
      ],
      usage: {
        input_tokens: 11,
        output_tokens: 7,
        cache_read_input_tokens: 3,
        cache_creation_input_tokens: 5,
        cost: { total: 0.0042 },
      },
    };

    backend.onLlmInput(
      {
        ...llmInput(),
        provider: 'anthropic',
        model: 'claude-sonnet-4',
        prompt: 'Find the release notes.',
        historyMessages: [{ role: 'user', content: [{ type: 'text', text: 'Find the release notes.' }] }],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onModelCallEnded(
      {
        ...modelEnded('call-1', 42),
        provider: 'anthropic',
        model: 'claude-sonnet-4',
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onBeforeMessageWrite({ message: assistantMessage }, { sessionKey: 'session-1' });
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: [{ type: 'text', text: 'Find the release notes.' }] },
          assistantMessage,
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [
      { role: 'user', content: [{ type: 'text', text: 'Find the release notes.' }] },
    ]);
    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.content, 'I will search for it.');
    assert.deepEqual(response.tool_calls, [
      {
        type: 'tool_use',
        id: 'toolu-search-1',
        name: 'web_search',
        input: { stripped: true },
      },
    ]);
    assert.deepEqual(response.usage, {
      prompt_tokens: 11,
      completion_tokens: 7,
      cached_tokens: 3,
      cache_read_tokens: 3,
      cache_write_tokens: 5,
      total_tokens: 18,
      cost_usd: 0.0042,
    });
    assert.deepEqual((response.openclaw as ResponseOpenClaw).assistant_tool_call_names, ['web_search']);
  });

  it('normalizes OpenAI responses usage from message-write replay', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);
    const assistantMessage = {
      role: 'assistant',
      provider: 'openai',
      model: 'gpt-4.1',
      content: [{ type: 'output_text', text: 'Done.' }],
      usage: {
        input_tokens: 75,
        output_tokens: 20,
        total_tokens: 95,
        input_tokens_details: { cached_tokens: 10 },
      },
    };

    backend.onLlmInput(
      {
        ...llmInput(),
        model: 'gpt-4.1',
        prompt: 'Use the responses endpoint.',
        historyMessages: [{ role: 'user', content: 'Use the responses endpoint.' }],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onModelCallEnded({ ...modelEnded('call-1', 42), model: 'gpt-4.1' }, { runId: 'run-1', sessionId: 'session-1' });
    backend.onBeforeMessageWrite({ message: assistantMessage }, { sessionKey: 'session-1' });
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [{ role: 'user', content: 'Use the responses endpoint.' }, assistantMessage],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.content, 'Done.');
    assert.deepEqual(response.usage, {
      prompt_tokens: 75,
      completion_tokens: 20,
      cached_tokens: 10,
      cache_read_tokens: 10,
      total_tokens: 95,
    });
  });

  it('uses the observed input time as the fallback llm span start time', () => {
    const now = Date.now;
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    try {
      Date.now = () => 1_000;
      backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
      Date.now = () => 1_250;
      backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });
    } finally {
      Date.now = now;
    }

    assert.equal(nf.calls.llmCall[0]?.timestamp, 1_000_000);
    assert.equal(nf.calls.llmCallEnd[0]?.timestamp, 1_250_000);
  });

  it('folds cache read and write tokens into prompt token totals', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      {
        ...llmOutput(),
        usage: {
          input: 1,
          output: 1_454,
          cacheRead: 4_869,
          cacheWrite: 544,
          total: 6_868,
        },
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.deepEqual(response.usage, {
      prompt_tokens: 5_414,
      completion_tokens: 1_454,
      cached_tokens: 4_869,
      cache_read_tokens: 4_869,
      cache_write_tokens: 544,
      total_tokens: 6_868,
    });
  });

  it('does not derive impossible prompt tokens from inconsistent usage totals', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      {
        ...llmOutput(),
        usage: {
          input: 3,
          output: 10,
          total: 5,
        },
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.deepEqual(response.usage, {
      prompt_tokens: 3,
      completion_tokens: 10,
    });
  });

  it('replays pending output when matching input arrives and cancels pending queue', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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
    assert.equal(backend.state().llmOutputsPendingInput.size, 1);

    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        prompt: 'hello',
        historyMessages: [],
        imagesCount: 0,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, false);
    assert.equal(request.content.prompt, 'hello');
  });

  it('replays placeholder request when output grace timer expires', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 1 });

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

    await delay(10);

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, true);
    assert.equal(request.content.prompt, '');
  });

  it('does not keep the process alive while waiting for llm output grace', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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

    const pending = [...backend.state().llmOutputsPendingInput.values()][0]?.[0];
    assert.ok(pending?.timer);
    if (isRefableTimer(pending.timer)) {
      assert.equal(pending.timer.hasRef(), false);
    }
    await backend.onSessionEnd({ sessionId: 'session-1', messageCount: 1, reason: 'idle' }, { sessionId: 'session-1' });
    assert.equal(pending.timer, undefined);
  });

  it('drains pending llm output with placeholder request on session end', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { llmOutputGraceMs: 10_000 });

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

    await backend.onSessionEnd({ sessionId: 'session-1', messageCount: 1, reason: 'idle' }, { sessionId: 'session-1' });

    assert.equal(backend.state().llmOutputsPendingInput.size, 0);
    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal(nf.calls.popScope.length, 1);
    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal(request.content.placeholderRequest, true);
    assert.equal(request.content.prompt, '');
  });

  it('attaches model timing only when timing is unambiguous', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted('call-1'), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal(response.openclaw.duration_ms, 42);
    assert.equal(response.openclaw.outcome, 'completed');
    assert.equal(backend.state().counters.llmSpansReplayed, 1);
  });

  it('emits ambiguity mark and does not attach ambiguous timing', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted('call-1'), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallStarted(modelStarted('call-2'), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-2', 55), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });

    assert.ok(nf.calls.event.some((event) => event.name === 'openclaw.model_call_timing_ambiguous'));
    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal('duration_ms' in response.openclaw, false);
  });

  it('replays recorded assistant messages as ordered llm spans with usage and timing', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);
    const firstAssistant = {
      role: 'assistant',
      provider: 'openai',
      model: 'gpt-4',
      content: [
        { type: 'thinking', thinking: 'private reasoning', thinkingSignature: 'opaque-signature' },
        { type: 'toolCall', name: 'web_search', arguments: { query: 'answer' } },
      ],
      usage: { input: 10, output: 5, totalTokens: 15 },
      stopReason: 'tool_use',
    };
    const finalAssistant = {
      role: 'assistant',
      provider: 'openai',
      model: 'gpt-4',
      content: [{ type: 'text', text: 'Final answer.' }],
      usage: { input: 20, output: 7, totalTokens: 27 },
      stopReason: 'stop',
    };

    const historyMessages: unknown[] = [];
    backend.onLlmInput(
      { ...llmInput(), prompt: 'Find the answer.', historyMessages },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    historyMessages.push(firstAssistant);
    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-2', 55), { runId: 'run-1', sessionId: 'session-1' });
    backend.onBeforeMessageWrite({ message: firstAssistant }, { sessionKey: 'session-1' });
    backend.onBeforeMessageWrite(
      { message: { role: 'toolResult', content: 'tool result' } },
      { sessionKey: 'session-1' },
    );
    backend.onBeforeMessageWrite({ message: finalAssistant }, { sessionKey: 'session-1' });
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: 'Find the answer.' },
          firstAssistant,
          { role: 'tool', content: 'tool result' },
          finalAssistant,
        ],
        success: true,
        durationMs: 100,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(nf.calls.llmCall.length, 2);
    assert.equal(nf.calls.llmCallEnd.length, 2);
    assert.equal(
      nf.calls.event.some((event) => event.name === 'openclaw.model_call_timing_ambiguous'),
      false,
    );
    const firstResponse = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    const firstRequest = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(firstRequest.content.messages, [{ role: 'user', content: 'Find the answer.' }]);
    assert.equal(firstResponse.content, 'tool calls: web_search');
    assert.equal((firstResponse.openclaw as ResponseOpenClaw).assistant_tool_call_names?.[0], 'web_search');
    assert.equal(firstResponse.openclaw.duration_ms, 42);
    assert.deepEqual(firstResponse.usage, {
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    const secondResponse = nf.calls.llmCallEnd[1]?.response as ReplayResponse;
    const secondRequest = nf.calls.llmCall[1]?.request as ReplayRequest;
    assert.deepEqual(secondRequest.content.messages?.[1], {
      role: 'assistant',
      provider: 'openai',
      model: 'gpt-4',
      content: [
        { type: 'thinking', stripped: true },
        { type: 'toolCall', name: 'web_search', arguments: { stripped: true } },
      ],
      usage: { input: 10, output: 5, totalTokens: 15 },
      stopReason: 'tool_use',
    });
    assert.deepEqual(secondRequest.content.messages?.at(-1), { role: 'toolResult', content: { stripped: true } });
    assert.equal(secondResponse.content, 'Final answer.');
    assert.equal(secondResponse.openclaw.duration_ms, 55);
    assert.deepEqual(secondResponse.usage, {
      prompt_tokens: 20,
      completion_tokens: 7,
      total_tokens: 27,
    });
  });

  it('uses model_call timestamps for recorded assistant message spans', () => {
    const now = Date.now;
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    try {
      Date.now = () => 1_000;
      backend.onModelCallStarted(modelStarted('call-1'), { runId: 'run-1', sessionId: 'session-1' });
      Date.now = () => 1_250;
      backend.onModelCallEnded(modelEnded('call-1', 250), { runId: 'run-1', sessionId: 'session-1' });
      Date.now = () => 1_260;
      backend.onBeforeMessageWrite(
        { message: { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' } },
        { sessionKey: 'session-1' },
      );
      Date.now = () => 2_000;
      backend.onAgentEnd(
        {
          runId: 'run-1',
          messages: [
            { role: 'user', content: 'hello' },
            { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' },
          ],
          success: true,
        },
        { runId: 'run-1', sessionId: 'session-1' },
      );
    } finally {
      Date.now = now;
    }

    assert.equal(nf.calls.llmCall[0]?.timestamp, 1_000_000);
    assert.equal(nf.calls.llmCallEnd[0]?.timestamp, 1_250_000);
    assert.equal(nf.calls.pushScope[0]?.timestamp, 1_000_000);
  });

  it('suppresses collapsed llm_output after recorded assistant message replay', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onBeforeMessageWrite(
      { message: { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' } },
      { sessionKey: 'session-1' },
    );
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' },
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });

    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal(backend.state().llmInputs.size, 0);
  });

  it('replays multiple llm_output hooks from the same run', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput({ ...llmInput(), prompt: 'first' }, { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      { ...llmOutput(), assistantTexts: ['first answer'] },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmInput({ ...llmInput(), prompt: 'second' }, { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      { ...llmOutput(), assistantTexts: ['second answer'] },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(nf.calls.llmCall.length, 2);
    assert.equal(nf.calls.llmCallEnd.length, 2);
    assert.equal((nf.calls.llmCallEnd[0]?.response as ReplayResponse).content, 'first answer');
    assert.equal((nf.calls.llmCallEnd[1]?.response as ReplayResponse).content, 'second answer');
  });

  it('does not reconstruct agent_end transcripts without reliable message-write state', () => {
    const transcriptOnlyNf = createNemoRelayRuntime();
    const transcriptOnlyBackend = createBackend(transcriptOnlyNf);

    transcriptOnlyBackend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: 'current question' },
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'current answer' },
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(transcriptOnlyNf.calls.llmCall.length, 0);
    assert.equal(transcriptOnlyNf.calls.llmCallEnd.length, 0);

    const compactedNf = createNemoRelayRuntime();
    const compactedBackend = createBackend(compactedNf);

    compactedBackend.onLlmInput(
      {
        ...llmInput(),
        prompt: 'current question',
        historyMessages: [
          { role: 'user', content: 'previous question 1' },
          { role: 'assistant', content: 'previous answer 1' },
        ],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    compactedBackend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [{ role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'previous answer' }],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(compactedNf.calls.llmCall.length, 0);
    assert.equal(compactedNf.calls.llmCallEnd.length, 0);
  });

  it('replays compacted message-write turns from the latest llm input snapshot', () => {
    const now = Date.now;
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    try {
      Date.now = () => 1_000;
      backend.onLlmInput(
        { ...llmInput(), runId: 'old-run', prompt: 'old question' },
        { runId: 'old-run', sessionId: 'session-1' },
      );
      Date.now = () => 2_000;
      backend.onLlmInput(
        { ...llmInput(), runId: 'run-1', prompt: 'current question' },
        { runId: 'run-1', sessionId: 'session-1' },
      );
    } finally {
      Date.now = now;
    }

    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onBeforeMessageWrite(
      { message: { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'current answer' } },
      { sessionKey: 'session-1' },
    );
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'compacted previous answer' },
          { role: 'user', content: 'current question' },
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'current answer' },
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [{ role: 'user', content: 'current question' }]);
    assert.equal((nf.calls.llmCallEnd[0]?.response as ReplayResponse).content, 'current answer');
  });

  it('does not duplicate trajectory replay across llm_output, message-write, and late hooks', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onModelCallEnded(modelEnded('call-1', 42), { runId: 'run-1', sessionId: 'session-1' });
    backend.onBeforeMessageWrite(
      { message: { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'Final answer.' } },
      { sessionKey: 'session-1' },
    );
    backend.onAgentEnd(
      {
        runId: 'run-1',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' },
          { role: 'tool', content: 'tool result' },
          { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'Final answer.' },
        ],
        success: true,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmInput({ ...llmInput(), prompt: 'late duplicate' }, { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      { ...llmOutput(), assistantTexts: ['late duplicate'] },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    assert.equal(nf.calls.llmCall.length, 1);
    assert.equal(nf.calls.llmCallEnd.length, 1);
    assert.equal((nf.calls.llmCallEnd[0]?.response as ReplayResponse).content, 'hi');
  });

  it('bounds replayed run markers for long-lived sessions', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { maxRecordsPerKey: 2 });

    for (const runId of ['run-1', 'run-2', 'run-3']) {
      backend.onLlmInput({ ...llmInput(), runId }, { runId, sessionId: 'session-1' });
      backend.onLlmOutput({ ...llmOutput(), runId }, { runId, sessionId: 'session-1' });
      backend.onAgentEnd(
        {
          runId,
          messages: [
            { role: 'user', content: 'hello' },
            { role: 'assistant', provider: 'openai', model: 'gpt-4', content: 'hi' },
          ],
          success: true,
        },
        { runId, sessionId: 'session-1' },
      );
    }

    const session = backend.state().sessions.get('session-1');
    assert.deepEqual([...(session?.trajectoryReplayedRuns ?? [])], ['run-2', 'run-3']);
  });

  it('bounds run bookkeeping for long-lived sessions without agent_end', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { maxRecordsPerKey: 2 });

    for (const runId of ['run-1', 'run-2', 'run-3']) {
      backend.onLlmInput(
        {
          ...llmInput(),
          runId,
          prompt: `prompt for ${runId}`,
        },
        { runId, sessionId: 'session-1' },
      );
      backend.onLlmOutput({ ...llmOutput(), runId }, { runId, sessionId: 'session-1' });
    }

    const session = backend.state().sessions.get('session-1');
    assert.deepEqual([...(session?.agentRunInputSnapshots?.keys() ?? [])], ['run-2', 'run-3']);
    assert.deepEqual([...(session?.hookLlmOutputReplayCounts?.keys() ?? [])], ['run-2', 'run-3']);
  });

  it('emits unpaired mark for model_call_started without matching end on session drain', async () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onModelCallStarted(modelStarted('call-1'), { runId: 'run-1', sessionId: 'session-1' });

    await backend.onSessionEnd({ sessionId: 'session-1', messageCount: 1, reason: 'idle' }, { sessionId: 'session-1' });

    const unpaired = nf.calls.event.find((event) => event.name === 'openclaw.model_call_timing_unpaired');
    assert.ok(unpaired);
    assert.deepEqual(unpaired.data, {
      runId: 'run-1',
      callId: 'call-1',
      provider: 'openai',
      model: 'gpt-4',
    });
  });

  it('strips prompt fields when prompt capture is disabled', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(
      nf,
      {},
      {
        includePrompts: false,
      },
    );

    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        systemPrompt: 'classified system',
        prompt: 'classified prompt',
        historyMessages: [{ role: 'user', content: 'classified history' }],
        imagesCount: 1,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });

    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.equal('prompt' in request.content, false);
    assert.equal('systemPrompt' in request.content, false);
    assert.deepEqual(request.content.messages, []);
    assert.equal(request.content.imagesCount, 1);
  });

  it('strips response content when response capture is disabled', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(
      nf,
      {},
      {
        includeResponses: false,
      },
    );

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });
    backend.onLlmOutput(
      {
        ...llmOutput(),
        assistantTexts: ['classified response'],
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );

    const response = nf.calls.llmCallEnd[0]?.response as ReplayResponse;
    assert.equal('content' in response, false);
    assert.equal(response.assistant_texts_count, 1);
  });

  it('does not duplicate current prompt when history already ends with the same user message', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf);

    backend.onLlmInput(
      {
        runId: 'run-1',
        sessionId: 'session-1',
        provider: 'openai',
        model: 'gpt-4',
        prompt: 'hello',
        historyMessages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
        imagesCount: 0,
      },
      { runId: 'run-1', sessionId: 'session-1' },
    );
    backend.onLlmOutput(llmOutput(), { runId: 'run-1', sessionId: 'session-1' });

    const request = nf.calls.llmCall[0]?.request as ReplayRequest;
    assert.deepEqual(request.content.messages, [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }]);
  });

  it('evicts stale expanded correlation records by TTL', () => {
    const nf = createNemoRelayRuntime();
    const backend = createBackend(nf, { recordTtlMs: 1 });
    const stalePendingOutput = {
      sessionKey: 'session-1',
      sessionId: 'session-1',
      runId: 'old-run',
      provider: 'openai',
      model: 'gpt-4',
      event: llmOutput(),
      ctx: { runId: 'old-run', sessionId: 'session-1' },
      observedAtMs: 0,
      timer: setTimeout(() => {}, 10_000),
    };

    backend.state().llmInputs.set('stale-input', [
      {
        sessionKey: 'session-1',
        sessionId: 'session-1',
        runId: 'old-run',
        provider: 'openai',
        model: 'gpt-4',
        prompt: 'old',
        historyMessages: [],
        imagesCount: 0,
        observedAtMs: 0,
      },
    ]);
    backend.state().llmOutputsPendingInput.set('stale-output', [stalePendingOutput]);
    backend.state().modelTimingsByLlmKey.set('stale-timing', [
      {
        sessionKey: 'session-1',
        sessionId: 'session-1',
        runId: 'old-run',
        callId: 'old-call',
        provider: 'openai',
        model: 'gpt-4',
        consumed: false,
        observedAtMs: 0,
      },
    ]);

    backend.onLlmInput(llmInput(), { runId: 'run-1', sessionId: 'session-1' });

    assert.equal(backend.state().llmInputs.has('stale-input'), false);
    assert.equal(backend.state().llmOutputsPendingInput.has('stale-output'), false);
    assert.equal(stalePendingOutput.timer, undefined);
    assert.equal(backend.state().modelTimingsByLlmKey.has('stale-timing'), false);
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
  tool_calls?: unknown[];
  assistant_texts_count?: number;
  usage?: Record<string, number>;
  openclaw: Record<string, unknown>;
};

type ResponseOpenClaw = {
  assistant_tool_call_names?: string[];
  duration_ms?: number;
  [key: string]: unknown;
};

type TestNemoRelayRuntime = NemoRelayRuntimeModule & {
  calls: {
    pushScope: Array<{ name: string; scopeType: number; data: unknown; timestamp: number | null | undefined }>;
    popScope: Array<{ handle: unknown; output: unknown }>;
    event: Array<{ name: string; handle: unknown; data: unknown }>;
    setThreadScopeStack: unknown[];
    llmCall: Array<{
      name: string;
      request: unknown;
      data: unknown;
      modelName: string | null | undefined;
      timestamp: number | null | undefined;
    }>;
    llmCallEnd: Array<{ handle: unknown; response: unknown; data: unknown; timestamp: number | null | undefined }>;
    toolCall: Array<{ name: string; args: unknown }>;
    toolCallEnd: Array<{ handle: unknown; result: unknown; data: unknown }>;
    toolConditionalExecution: Array<{ name: string; args: unknown }>;
  };
};

function createBackend(
  nf: TestNemoRelayRuntime,
  correlation: Partial<ReturnType<typeof parseConfig>['correlation']> = {},
  capture: Partial<ReturnType<typeof parseConfig>['capture']> = {},
): HookReplayBackend {
  return new HookReplayBackend({
    nf,
    config: parseConfig({
      correlation,
      capture,
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
    pushScope: (name, scopeType, _handle, _attributes, data, _links, _metadata, timestamp) => {
      const handle = { id: `scope-${nextScopeId++}` };
      calls.pushScope.push({ name, scopeType, data, timestamp });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['pushScope']>;
    },
    popScope: (handle, output) => calls.popScope.push({ handle, output }),
    event: (name, handle, data) => calls.event.push({ name, handle, data }),
    llmCall: (name, request, _handle, _attributes, data, _metadata, modelName, timestamp) => {
      const handle = { id: `llm-${nextScopeId++}` };
      calls.llmCall.push({ name, request, data, modelName, timestamp });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['llmCall']>;
    },
    llmCallEnd: (handle, response, data, _metadata, timestamp) =>
      calls.llmCallEnd.push({ handle, response, data, timestamp }),
    toolCall: (name, args) => {
      const handle = { id: `tool-${nextScopeId++}` };
      calls.toolCall.push({ name, args });
      return handle as unknown as ReturnType<NemoRelayRuntimeModule['toolCall']>;
    },
    toolCallEnd: (handle, result, data) => calls.toolCallEnd.push({ handle, result, data }),
    toolConditionalExecution: async (name, args) => {
      calls.toolConditionalExecution.push({ name, args });
    },
  };
}

function llmInput() {
  return {
    runId: 'run-1',
    sessionId: 'session-1',
    provider: 'openai',
    model: 'gpt-4',
    prompt: 'hello',
    historyMessages: [],
    imagesCount: 0,
  };
}

function llmOutput() {
  return {
    runId: 'run-1',
    sessionId: 'session-1',
    provider: 'openai',
    model: 'gpt-4',
    assistantTexts: ['hi'],
  };
}

function modelStarted(callId: string) {
  return {
    runId: 'run-1',
    callId,
    sessionId: 'session-1',
    provider: 'openai',
    model: 'gpt-4',
  };
}

function modelEnded(callId: string, durationMs: number) {
  return {
    ...modelStarted(callId),
    durationMs,
    outcome: 'completed' as const,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRefableTimer(timer: unknown): timer is { hasRef: () => boolean } {
  return (
    typeof timer === 'object' &&
    timer !== null &&
    'hasRef' in timer &&
    typeof (timer as { hasRef?: unknown }).hasRef === 'function'
  );
}
