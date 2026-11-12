// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Runtime tests: plugin-host lifecycle, turn scopes, and managed model and tool
 * calls against a recording fake of the NeMo Relay binding.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import {
  bindingPrecondition,
  defaultNemoRelayModuleLoader,
  describeBindingFailure,
  RUNTIME_OWNER_ENV,
  RuntimeOwnerConflictError,
  runtimeOwnerConflict,
  type NemoRelayModules,
} from '../src/modules.js';
import { GuardrailPolicy, RULES_GUARDRAIL_NAME, RULES_GUARDRAIL_PRIORITY } from '../src/policy.js';
import {
  NemoRelayDeepSeekHarnessRuntime,
  START_BUFFER_LIMIT,
  TURN_SCOPE_NAME,
  type RuntimeGuardrail,
} from '../src/runtime.js';
import type { GenerateOptions, StreamChunk } from '../src/dsh-types.js';
import {
  collect,
  createFakeModules,
  event,
  fakeAgent,
  fakeSession,
  okResult,
  streamOf,
  textStream,
  toolExec,
  type FakeBehavior,
} from './fakes.js';

async function startRuntime(behavior: FakeBehavior = {}, rawConfig: unknown = {}) {
  const { modules, record } = createFakeModules(behavior);
  const warnings: string[] = [];
  const runtime = new NemoRelayDeepSeekHarnessRuntime({
    config: parseConfig(rawConfig, { DSH_HOME: '/tmp/dsh-home' }),
    moduleLoader: async () => modules,
    logger: { info: () => undefined, warn: (message) => warnings.push(message) },
  });
  await runtime.start();
  return { runtime, record, warnings };
}

function agentRequest(sessionId: string, overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return {
    provider: 'deepseek',
    model: 'deepseek-chat',
    system: 'You are helpful.',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Fix the bug' }] }],
    sessionId,
    ...overrides,
  };
}

/** A module loader that finishes only when the test releases it, like a native binding still loading. */
function lateLoader(modules: NemoRelayModules) {
  let release: () => void = () => undefined;
  let fail: (error: Error) => void = () => undefined;
  const gate = new Promise<void>((resolve, reject) => {
    release = resolve;
    fail = reject;
  });
  return {
    loader: async () => {
      await gate;
      return modules;
    },
    release: () => release(),
    fail: (error: Error) => fail(error),
  };
}

const tick = (ms = 5) => new Promise((resolve) => setTimeout(resolve, ms));

describe('startup', () => {
  it('maps a turn that began while Relay was still starting, and runs its first calls through Relay', async () => {
    const { modules, record } = createFakeModules();
    const late = lateLoader(modules);
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: late.loader,
    });
    const started = runtime.start();
    const session = fakeSession('s1');
    const agent = fakeAgent(session);
    // A fresh DeepSeek Harness process submits its task before the binding has loaded.
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, 2_000));
    runtime.onSessionEvent(
      session,
      event('user/message', { content: [{ type: 'text', text: 'Fix the bug' }], source: { kind: 'user' } }),
    );
    runtime.noteStep(agent, 1, 1);
    const chunks = collect(runtime.streamLlm(agentRequest('s1'), streamOf(textStream('Done'))));
    const tool = runtime.executeTool(toolExec(agent, 'bash', { command: 'ls' }), async () => okResult('a.txt'));
    await tick();
    late.release();
    await started;
    assert.deepEqual(await chunks, textStream('Done'));
    assert.equal((await tool).isError, false);
    runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }, 3_000));

    assert.equal(record.pushed.length, 1, 'the turn that began during start is recorded');
    const turn = record.pushed[0];
    assert.deepEqual(turn?.input, { prompt: 'Fix the bug' });
    assert.equal(turn?.metadata.turn_source, 'user');
    assert.equal(turn?.timestamp, 2_000_000, 'the turn starts when DeepSeek Harness started it');
    assert.equal(record.llm.length, 1, 'the first model call ran through Relay');
    assert.equal(record.llm[0]?.parent, turn?.handle);
    assert.equal(record.llm[0]?.metadata.step_index, 1);
    assert.equal(record.tools[0]?.parent, turn?.handle);
    assert.deepEqual(
      record.popped.map((entry) => entry.output.reason),
      ['completed'],
    );
  });

  it('fails open when Relay is not ready within the start wait, then replays the turn once it is', async () => {
    const { modules, record } = createFakeModules();
    const late = lateLoader(modules);
    const warnings: string[] = [];
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: late.loader,
      logger: { info: () => undefined, warn: (message) => warnings.push(message) },
      startWaitMs: 10,
    });
    const started = runtime.start();
    const session = fakeSession('s1');
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, 2_000));
    runtime.onSessionEvent(
      session,
      event('user/message', { content: [{ type: 'text', text: 'Fix the bug' }], source: { kind: 'user' } }),
    );
    let hostCalls = 0;
    const host = () => {
      hostCalls += 1;
      return streamOf(textStream())();
    };
    assert.deepEqual(await collect(runtime.streamLlm(agentRequest('s1'), host)), textStream());
    assert.equal(hostCalls, 1);
    assert.equal(record.llm.length, 0, 'the waiting call ran without Relay');
    assert.ok(warnings.some((message) => message.includes('still starting after 10 ms')));

    // An aborted call does not wait for the start at all.
    const controller = new AbortController();
    controller.abort();
    const slow = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: lateLoader(modules).loader,
      startWaitMs: 60_000,
    });
    void slow.start();
    const before = Date.now();
    await collect(slow.streamLlm(agentRequest('s2', { signal: controller.signal }), host));
    assert.ok(Date.now() - before < 1_000);

    late.release();
    await started;
    await collect(runtime.streamLlm(agentRequest('s1'), streamOf(textStream())));
    assert.equal(record.pushed.length, 1, 'the buffered turn is replayed when Relay is ready');
    assert.deepEqual(record.pushed[0]?.input, { prompt: 'Fix the bug' });
    assert.equal(record.llm[0]?.parent, record.pushed[0]?.handle);
  });

  it('drops the buffered session structure when Relay does not start', async () => {
    const { modules, record } = createFakeModules();
    const late = lateLoader(modules);
    const runtime = new NemoRelayDeepSeekHarnessRuntime({ config: parseConfig({}), moduleLoader: late.loader });
    const started = runtime.start();
    const session = fakeSession('s1');
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    runtime.noteStep(fakeAgent(session), 1, 1);
    const chunks = collect(runtime.streamLlm(agentRequest('s1'), streamOf(textStream())));
    late.fail(new Error('Cannot find native binding'));
    await started;
    assert.deepEqual(await chunks, textStream(), 'the waiting call runs without Relay');
    assert.equal(runtime.status().state, 'disabled');
    assert.deepEqual([record.pushed.length, record.llm.length, record.marks.length], [0, 0, 0]);
  });

  it(`keeps at most ${START_BUFFER_LIMIT} notifications while starting`, async () => {
    const { modules, record } = createFakeModules();
    const late = lateLoader(modules);
    const warnings: string[] = [];
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}),
      moduleLoader: late.loader,
      logger: { info: () => undefined, warn: (message) => warnings.push(message) },
    });
    const started = runtime.start();
    const session = fakeSession('s1');
    for (let index = 0; index < START_BUFFER_LIMIT + 3; index += 1) {
      runtime.onSessionEvent(session, event('subagent/catalog', { childId: `child-${index}`, mode: 'continuable' }));
    }
    late.release();
    await started;
    assert.equal(record.marks.length, START_BUFFER_LIMIT);
    assert.equal(record.marks[0]?.data.child_session_id, 'child-0', 'the earliest notifications are kept, in order');
    assert.ok(warnings.some((message) => message.includes('dropped 3 session notification(s)')));
  });
});

describe('plugin host lifecycle', () => {
  it('validates, initializes, and closes the plugin host', async () => {
    const userObservability = { kind: 'observability', enabled: true, config: {} };
    const { runtime, record } = await startRuntime({}, { plugins: { components: [userObservability] } });
    assert.equal(runtime.status().state, 'ready');
    // The user's document alone, then the document with the defaults the layers leave open.
    assert.equal(record.validated.length, 2);
    assert.equal(record.initialized.length, 1);
    const initialized = record.initialized[0] as { components: Array<{ kind: string }> };
    assert.deepEqual(
      initialized.components.map((component) => component.kind),
      ['observability', 'pii_redaction'],
    );
    assert.equal(runtime.pluginPlan()?.atif.applied, false);
    assert.deepEqual(runtime.health().configPaths, ['/fake/plugins.toml']);
    await runtime.stop();
    assert.equal(record.closed, 1);
    assert.equal(runtime.status().state, 'stopped');
  });

  it('drops a default the layered configuration rejects instead of failing the user configuration', async () => {
    const { runtime, record, warnings } = await startRuntime({
      // A plugins.toml layer whose redaction policy cannot merge with the default profiles.
      diagnostics: (config: { components: Array<{ kind: string }> }) =>
        config.components.some((component) => component.kind === 'pii_redaction')
          ? [
              {
                level: 'error',
                code: 'pii_redaction.unsupported_value',
                component: 'pii_redaction',
                message: "legacy field 'mode' cannot be combined with profiles",
              },
            ]
          : [],
    });
    assert.equal(runtime.status().state, 'ready');
    const initialized = record.initialized[0] as { components: Array<{ kind: string }> };
    assert.deepEqual(
      initialized.components.map((component) => component.kind),
      ['observability'],
    );
    const plan = runtime.pluginPlan();
    assert.equal(plan?.atif.applied, true);
    assert.equal(plan?.redaction.applied, false);
    assert.match(plan?.redaction.reason ?? '', /cannot be combined with profiles/);
    assert.ok(warnings.some((message) => message.includes('did not apply a default')));
  });

  it('skips initialization and degrades when validation reports errors, but keeps middleware active', async () => {
    const { runtime, record, warnings } = await startRuntime({
      diagnostics: [{ level: 'error', code: 'plugin.unknown_component', message: 'unknown component kind' }],
    });
    assert.equal(record.initialized.length, 0);
    assert.equal(runtime.status().state, 'degraded');
    assert.equal(runtime.isActive(), true);
    assert.ok(warnings.some((message) => message.includes('plugin.unknown_component')));
  });

  it('is disabled with an actionable reason, and passes calls through, when the binding cannot load', async () => {
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}),
      moduleLoader: async () => {
        throw new Error('Cannot find native binding. npm has a bug related to optional dependencies');
      },
    });
    await runtime.start();
    assert.equal(runtime.isActive(), false);
    const status = runtime.status();
    assert.equal(status.state, 'disabled');
    assert.match(status.state === 'disabled' ? (status.reason ?? '') : '', /native binding for .* is not installed/);
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(textStream())));
    assert.deepEqual(chunks, textStream());
  });

  it('explains binding failures by cause: Node.js version, platform, missing package', () => {
    const supported = { node: '24.21.0', platform: 'linux', arch: 'x64' };
    assert.equal(bindingPrecondition(supported), undefined);
    assert.equal(
      bindingPrecondition({ ...supported, node: '22.12.0' }),
      'NeMo Relay needs Node.js 24.13.1 or newer, but this DeepSeek Harness runs Node.js 22.12.0. Update DeepSeek Harness, then restart it.',
    );
    assert.equal(bindingPrecondition({ ...supported, node: '24.13.1' }), undefined);
    assert.match(bindingPrecondition({ ...supported, platform: 'darwin' }) ?? '', /no native binding for darwin-x64/);
    assert.equal(
      describeBindingFailure(new Error('Cannot find native binding. npm has a bug\nmore detail'), supported),
      'The NeMo Relay native binding for linux-x64 is not installed (npm package nemo-relay-node-linux-x64-gnu or nemo-relay-node-linux-x64-musl). Reinstall the plugin so npm installs it, then restart DeepSeek Harness. Cause: Cannot find native binding. npm has a bug',
    );
    assert.equal(
      describeBindingFailure(new Error('dlopen failed: symbol not found'), {
        ...supported,
        platform: 'darwin',
        arch: 'arm64',
      }),
      'nemo-relay-node could not load on darwin-arm64 with Node.js 24.21.0: dlopen failed: symbol not found',
    );
  });

  it('detects a NeMo Relay runtime of another binding or major version in this process', () => {
    const pid = 4242;
    assert.equal(runtimeOwnerConflict(undefined, pid, '1.0.0'), undefined);
    assert.equal(runtimeOwnerConflict('', pid, '1.0.0'), undefined);
    // Another process's token in the inherited environment: Relay reclaims it.
    assert.equal(runtimeOwnerConflict('pid=7;binding=node;version=0', pid, '1.0.0'), undefined);
    // This plugin's own binding, already loaded, as after a plugin restart.
    assert.equal(runtimeOwnerConflict(`pid=${pid};binding=node;version=1`, pid, '1.0.0'), undefined);
    assert.equal(runtimeOwnerConflict(`pid=${pid};binding=node;version=1`, pid, '1.4.2+0a1b2c3d'), undefined);
    // Relay replaces a malformed token.
    assert.equal(runtimeOwnerConflict(`pid=${pid};binding=node`, pid, '1.0.0'), undefined);
    assert.match(
      runtimeOwnerConflict(`pid=${pid};binding=node;version=0`, pid, '1.0.0') ?? '',
      /node binding, version 0\).*version 1 binding now would stop DeepSeek Harness.*Restart DeepSeek Harness/,
    );
    assert.match(runtimeOwnerConflict(`pid=${pid};binding=python;version=1`, pid, '1.0.0') ?? '', /python binding/);
    // Without its own version, only another binding kind is a known conflict.
    assert.equal(runtimeOwnerConflict(`pid=${pid};binding=node;version=0`, pid, undefined), undefined);
    assert.match(runtimeOwnerConflict(`pid=${pid};binding=python;version=1`, pid, undefined) ?? '', /python binding/);
  });

  it('stays off, without loading the binding, when another major version owns the process', async () => {
    const previous = process.env[RUNTIME_OWNER_ENV];
    // The token a NeMo Relay 0.x binding publishes. Loading the 1.x binding beside it aborts the process,
    // so this test can only pass if the loader refuses before the native library loads.
    process.env[RUNTIME_OWNER_ENV] = `pid=${process.pid};binding=node;version=0`;
    try {
      await assert.rejects(defaultNemoRelayModuleLoader(), RuntimeOwnerConflictError);
      const runtime = new NemoRelayDeepSeekHarnessRuntime({ config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }) });
      await runtime.start();
      const status = runtime.status();
      assert.equal(status.state, 'disabled');
      assert.match(
        status.state === 'disabled' ? (status.reason ?? '') : '',
        /already runs in this DeepSeek Harness process.*Restart DeepSeek Harness/,
      );
      const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(textStream())));
      assert.deepEqual(chunks, textStream());
    } finally {
      if (previous === undefined) {
        delete process.env[RUNTIME_OWNER_ENV];
      } else {
        process.env[RUNTIME_OWNER_ENV] = previous;
      }
    }
  });

  it('does nothing when disabled', async () => {
    const { runtime, record } = await startRuntime({}, { enabled: false });
    assert.equal(runtime.status().state, 'disabled');
    assert.equal(record.validated.length, 0);
  });
});

describe('turn scopes', () => {
  it('opens one top-level turn scope per DeepSeek Harness turn with the user prompt', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession('session-1', { parentSession: 'parent-1', delegationDepth: 1 });
    runtime.onSessionEvent(session, event('turn/start', { turn: 3 }, 2_000));
    runtime.onSessionEvent(
      session,
      event('user/message', { content: [{ type: 'text', text: 'context' }], source: { kind: 'runtime-context' } }),
    );
    runtime.onSessionEvent(
      session,
      event('user/message', { content: [{ type: 'text', text: 'Fix the bug' }], source: { kind: 'user' } }),
    );
    assert.equal(record.pushed.length, 0, 'the turn opens lazily so it can carry the prompt');

    await collect(runtime.streamLlm(agentRequest('session-1'), streamOf(textStream('All fixed'))));
    runtime.onSessionEvent(session, event('turn/end', { turn: 3, reason: { kind: 'completed' } }, 3_000));

    assert.equal(record.pushed.length, 1);
    const turn = record.pushed[0];
    assert.equal(turn?.name, TURN_SCOPE_NAME);
    assert.equal(turn?.type, 9);
    assert.equal(turn?.parent, null);
    assert.deepEqual(turn?.input, { prompt: 'Fix the bug' });
    assert.equal(turn?.timestamp, 2_000_000);
    assert.deepEqual(turn?.metadata, {
      session_id: 'session-1',
      'gen_ai.agent.name': 'deepseek-harness',
      parent_session_id: 'parent-1',
      delegation_depth: 1,
      nemo_relay_scope_role: 'turn',
      turn_index: 3,
      turn_source: 'user',
    });
    assert.equal(record.llm[0]?.parent, turn?.handle);
    assert.equal(record.popped.length, 1);
    assert.deepEqual(record.popped[0]?.output, { reason: 'completed', final_text: 'All fixed' });
    assert.equal(record.popped[0]?.metadata['otel.status_code'], 'OK');
  });

  it('records the OpenTelemetry status of each turn: completed OK, failed ERROR, cancelled or refused UNSET', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession();
    for (const [turn, reason] of [
      [1, { kind: 'completed' }],
      [2, { kind: 'error', error: { message: 'provider down', code: 'SERVER' } }],
      [3, { kind: 'aborted' }],
      [4, { kind: 'blocked' }],
    ] as const) {
      runtime.onSessionEvent(session, event('turn/start', { turn }));
      await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream())));
      runtime.onSessionEvent(session, event('turn/end', { turn, reason }));
    }
    assert.deepEqual(
      record.popped.map((entry) => [entry.metadata.turn_end_reason, entry.metadata['otel.status_code']]),
      [
        ['completed', 'OK'],
        ['error', 'ERROR'],
        ['aborted', 'UNSET'],
        ['blocked', 'UNSET'],
      ],
    );
    assert.equal(record.popped[1]?.metadata['otel.status_description'], 'provider down');
  });

  it('records what started a turn: the user, or a DeepSeek Harness notice such as a settled subagent', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession();
    const message = (kind: string, text: string) =>
      event('user/message', { content: [{ type: 'text', text }], source: { kind } });
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    runtime.onSessionEvent(session, message('user', 'Start a subagent'));
    runtime.onSessionEvent(session, message('runtime-context', 'Current runtime context.'));
    runtime.onSessionEvent(session, message('skill-catalog', '<system-reminder>skills</system-reminder>'));
    await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream())));
    runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }));
    runtime.onSessionEvent(session, event('turn/start', { turn: 2 }));
    runtime.onSessionEvent(session, message('subagent-settled', 'Background subagent child-1 finished.'));
    await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream())));
    runtime.onSessionEvent(session, event('turn/end', { turn: 2, reason: { kind: 'completed' } }));

    assert.deepEqual(
      record.pushed.map((entry) => [entry.metadata.turn_source, entry.input]),
      [
        ['user', { prompt: 'Start a subagent' }],
        ['subagent-settled', { prompt: 'Background subagent child-1 finished.' }],
      ],
    );
  });

  it("names DeepSeek Harness's default model as the model of the default ATIF trajectories", async () => {
    const { modules, record } = createFakeModules();
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: async () => modules,
      defaultModel: () => 'deepseek-chat',
    });
    await runtime.start();
    const atif = (record.initialized[0] as { components: Array<{ kind: string; config: any }> }).components.find(
      (component) => component.kind === 'observability',
    )?.config.atif;
    assert.deepEqual([atif?.agent_name, atif?.model_name], ['deepseek-harness', 'deepseek-chat']);

    // A host that cannot tell leaves Relay's own default in place.
    const unknown = createFakeModules();
    await new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: async () => unknown.modules,
      defaultModel: () => {
        throw new Error('no agentDefaultModel service');
      },
    }).start();
    const fallback = (unknown.record.initialized[0] as { components: Array<{ kind: string; config: any }> }).components;
    assert.equal(fallback.find((component) => component.kind === 'observability')?.config.atif.model_name, undefined);
  });

  it('closes open turns on shutdown so exporters can finish their trajectories', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession();
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream())));
    await runtime.stop();
    assert.equal(record.popped.length, 1);
    assert.equal(record.popped[0]?.output.reason, 'shutdown');
    assert.ok(record.flushed >= 1);
  });

  it('runs title generation beside the open turn, on a side stack of the session', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession('session-1');
    const title = () =>
      collect(runtime.streamLlm(agentRequest(session.id, { purpose: 'session-title' }), streamOf(textStream('Fix'))));
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream('All fixed'))));
    await title();
    await title();
    runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }));

    const turn = record.pushed[0];
    const [agentCall, firstTitle, secondTitle] = record.llm;
    assert.equal(agentCall?.effectiveParent, turn?.handle);
    // A null parent means "the top of the stack": on the session stack that would be the open turn.
    assert.equal(firstTitle?.effectiveParent, null, 'the title call is not parented to the turn');
    assert.notEqual(firstTitle?.stack, turn?.stack);
    assert.equal(secondTitle?.stack, firstTitle?.stack, 'one side stack per session');
    assert.deepEqual(firstTitle?.metadata, {
      session_id: 'session-1',
      'gen_ai.agent.name': 'deepseek-harness',
      request_purpose: 'session-title',
    });
    assert.equal(record.popped[0]?.output.final_text, 'All fixed', 'the title is not the turn answer');

    // The side stack is released with the session.
    runtime.onAgentDisposed(fakeAgent(session));
    await title();
    assert.notEqual(record.llm[3]?.stack, firstTitle?.stack);
  });
});

describe('managed model calls', () => {
  it('passes host chunks through unchanged and aggregates the response for Relay', async () => {
    const { runtime, record } = await startRuntime();
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(textStream('Hello there'))));
    assert.deepEqual(chunks, textStream('Hello there'));
    const call = record.llm[0];
    assert.equal(call?.name, 'deepseek');
    assert.equal(call?.model, 'deepseek-chat');
    assert.equal(call?.attributes, 2, 'streaming attribute');
    assert.equal(call?.codec, true);
    assert.deepEqual(call?.request.content.messages, [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: [{ type: 'text', text: 'Fix the bug' }] },
    ]);
    const response = record.llmEnd[0];
    assert.equal(response.choices[0].message.content, 'Hello there');
    assert.deepEqual(response.usage, {
      prompt_tokens: 16,
      completion_tokens: 4,
      total_tokens: 20,
      prompt_tokens_details: { cached_tokens: 6 },
    });
  });

  it('turns a guardrail rejection into a terminal error the host records', async () => {
    const { runtime } = await startRuntime({ llmReject: 'No production credentials in prompts.' });
    let hostCalled = false;
    const chunks = await collect(
      runtime.streamLlm(agentRequest('s'), () => {
        hostCalled = true;
        return streamOf(textStream())();
      }),
    );
    assert.equal(hostCalled, false);
    assert.deepEqual(chunks, [
      {
        type: 'finish',
        reason: {
          kind: 'error',
          failure: {
            code: 'NEMO_RELAY_GUARDRAIL_REJECTED',
            message: 'Blocked by NeMo Relay policy: No production credentials in prompts.',
          },
        },
      },
    ]);
    assert.equal(runtime.health().counters.guardrailRejections, 1);
  });

  it('blocks model-visible request edits instead of dropping them', async () => {
    const { runtime } = await startRuntime({
      llmEdit: (request) => ({ ...request, content: { ...request.content, messages: [] } }),
    });
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(textStream())));
    assert.equal(chunks.length, 1);
    const finish = chunks[0] as Extract<StreamChunk, { type: 'finish' }>;
    assert.equal(finish.reason.kind, 'error');
    assert.equal(finish.reason.kind === 'error' ? finish.reason.failure.code : '', 'NEMO_RELAY_REQUEST_REWRITE');
  });

  it('records advisory request fields the host cannot send and still runs the call', async () => {
    const { runtime, record } = await startRuntime({
      llmEdit: (request) => ({
        ...request,
        content: { ...request.content, nvext: { agent_hints: { latency: 'low' } } },
      }),
    });
    const session = fakeSession('s');
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(textStream())));
    assert.deepEqual(chunks, textStream());
    assert.deepEqual(record.marks.at(-1)?.data, { fields: ['nvext'] });
    assert.equal(record.marks.at(-1)?.name, 'deepseek-harness.request_edit_dropped');
  });

  it('applies stream intercept rewrites and keeps the block text consistent', async () => {
    const { runtime } = await startRuntime({
      chunkMap: (chunk) => {
        const delta = chunk.choices?.[0]?.delta;
        if (typeof delta?.content === 'string') {
          delta.content = delta.content.replace('secret', '[redacted]');
        }
        return chunk;
      },
    });
    const host: StreamChunk[] = [
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'the secret ' },
      { type: 'text-delta', index: 0, text: 'is out' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'the secret is out' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ];
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf(host)));
    assert.deepEqual(chunks, [
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'the [redacted] ' },
      { type: 'text-delta', index: 0, text: 'is out' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'the [redacted] is out' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);
  });

  it('turns an execution-intercept answer into a well-formed host stream without calling the provider', async () => {
    const { runtime } = await startRuntime({
      llmCached: [
        { object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content: 'cached ' } }] },
        {
          object: 'chat.completion.chunk',
          choices: [{ index: 0, delta: { content: 'answer' }, finish_reason: 'stop' }],
        },
      ],
    });
    let hostCalled = false;
    const chunks = await collect(
      runtime.streamLlm(agentRequest('s'), () => {
        hostCalled = true;
        return streamOf(textStream())();
      }),
    );
    assert.equal(hostCalled, false);
    assert.deepEqual(chunks, [
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'cached ' },
      { type: 'text-delta', index: 0, text: 'answer' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'cached answer' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);
  });

  it('delivers a host failure finish and records the span as failed', async () => {
    const { runtime } = await startRuntime();
    const failure: StreamChunk = {
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'rate limited', code: 'RATE_LIMIT' } },
    };
    const chunks = await collect(runtime.streamLlm(agentRequest('s'), streamOf([failure])));
    assert.deepEqual(chunks, [failure]);
  });

  it('rethrows host stream errors after releasing queued chunks', async () => {
    const { runtime } = await startRuntime();
    const host = () =>
      (async function* () {
        yield { type: 'block-start', index: 0, blockType: 'text' } as StreamChunk;
        throw new Error('consumer failure');
      })();
    await assert.rejects(collect(runtime.streamLlm(agentRequest('s'), host)), /consumer failure/);
  });
});

describe('managed tool calls', () => {
  it('runs the host tool body inside the managed call and returns the host result object', async () => {
    const { runtime, record } = await startRuntime();
    const session = fakeSession();
    const agent = fakeAgent(session);
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    runtime.noteStep(agent, 1, 2);
    const host = okResult('file list');
    const result = await runtime.executeTool(toolExec(agent, 'bash', { command: 'ls' }, 'call-7'), async () => host);
    assert.equal(result, host);
    const call = record.tools[0];
    assert.equal(call?.name, 'bash');
    assert.deepEqual(call?.args, { command: 'ls' });
    assert.equal(call?.toolCallId, 'call-7');
    assert.equal(call?.parent, record.pushed[0]?.handle);
    assert.equal(call?.metadata.step_index, 2);
  });

  it('returns a failed result when a guardrail rejects the call, without running the body', async () => {
    const { runtime } = await startRuntime({ toolReject: 'Ask before deleting files.' });
    let ran = false;
    const result = await runtime.executeTool(toolExec(fakeAgent()), async () => {
      ran = true;
      return okResult();
    });
    assert.equal(ran, false);
    assert.equal(result.isError, true);
    assert.equal(result.isError ? result.error.info?.code : '', 'NEMO_RELAY_GUARDRAIL_REJECTED');
    assert.match(result.content[0]?.type === 'text' ? result.content[0].text : '', /Ask before deleting files\./);
  });

  it('blocks argument rewrites because the host runs tools exactly as requested', async () => {
    const { runtime } = await startRuntime({ toolArgsEdit: () => ({ command: 'ls -la' }) });
    let ran = false;
    const result = await runtime.executeTool(toolExec(fakeAgent()), async () => {
      ran = true;
      return okResult();
    });
    assert.equal(ran, false);
    assert.equal(result.isError ? result.error.info?.code : '', 'NEMO_RELAY_ARGUMENT_REWRITE');
  });

  it('applies response-intercept edits through tools/post-execute', async () => {
    const { runtime } = await startRuntime({
      toolResultEdit: (returned) => {
        const wrapper = returned.result as { content: string };
        return { ...returned, result: { ...wrapper, content: wrapper.content.replace('hunter2', '[redacted]') } };
      },
    });
    const exec = toolExec(fakeAgent());
    const host = {
      isError: true as const,
      error: { message: 'bad password hunter2' },
      content: [{ type: 'text' as const, text: 'bad password hunter2' }],
    };
    const result = await runtime.executeTool(exec, async () => host);
    assert.equal(result, host);
    const decision = await runtime.postExecute(exec, host, async () => ({ kind: 'accept' }));
    assert.deepEqual(decision, { kind: 'accept', content: [{ type: 'text', text: 'bad password [redacted]' }] });
  });

  it('propagates a tool body failure without running the body again', async () => {
    const { runtime, record } = await startRuntime();
    let runs = 0;
    await assert.rejects(
      runtime.executeTool(toolExec(fakeAgent()), async () => {
        runs += 1;
        throw new Error('disk full');
      }),
      /disk full/,
    );
    assert.equal(runs, 1);
    assert.equal(record.tools.length, 1);
  });

  it('applies value edits from response intercepts to successful results', async () => {
    const { runtime } = await startRuntime({
      toolResultEdit: (returned) => ({
        ...returned,
        result: { ...(returned.result as Record<string, unknown>), result: { ok: false, redacted: true } },
      }),
    });
    const exec = toolExec(fakeAgent());
    const host = okResult('done', { ok: true });
    assert.equal(await runtime.executeTool(exec, async () => host), host);
    const decision = await runtime.postExecute(exec, host, async () => ({ kind: 'accept' }));
    assert.deepEqual(decision, { kind: 'accept', value: { ok: false, redacted: true } });
  });

  it('uses a cached result from an execution intercept without running the body', async () => {
    const { runtime } = await startRuntime({
      toolCached: {
        result: { schema: 'nemo.relay.tool_result@1', content: 'cached listing', is_error: false, result: ['a', 'b'] },
      },
    });
    let ran = false;
    const result = await runtime.executeTool(toolExec(fakeAgent()), async () => {
      ran = true;
      return okResult();
    });
    assert.equal(ran, false);
    assert.deepEqual(result, {
      isError: false,
      value: ['a', 'b'],
      content: [{ type: 'text', text: 'cached listing' }],
    });
  });

  it('denies an approval prompt for a call Relay policy rejects', async () => {
    const { runtime, record } = await startRuntime({ toolReject: 'No pushes to main.' });
    const exec = toolExec(fakeAgent(), 'bash', { command: 'git push origin main' });
    const decision = await runtime.preExecute(exec, async () => ({ kind: 'ask', reason: 'needs approval' }));
    assert.deepEqual(decision, {
      kind: 'deny',
      reason: 'Blocked by NeMo Relay policy: No pushes to main.',
      info: { name: 'NemoRelayGuardrail', code: 'NEMO_RELAY_GUARDRAIL_REJECTED', reason: 'No pushes to main.' },
    });
    const allowed = await runtime.preExecute(exec, async () => ({ kind: 'allow' }));
    assert.deepEqual(allowed, { kind: 'allow' });
    assert.equal(record.conditional.length, 1, 'allowed calls are checked once, by the managed call');
  });
});

describe('rules guardrail', () => {
  const rulesGuardrail = (rules: GuardrailPolicy): RuntimeGuardrail => ({
    name: RULES_GUARDRAIL_NAME,
    priority: RULES_GUARDRAIL_PRIORITY,
    evaluate: (toolName, args) => rules.evaluate(toolName, args),
  });
  const policyWith = (rules: unknown[]) =>
    new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: { kind: 'file', location: 'memory', read: () => rules, write: async () => undefined },
    });

  it('blocks a managed tool call through Relay with the rule id in the reason', async () => {
    const { modules, record } = createFakeModules();
    const rules = policyWith([{ id: 'no-e2e', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK', reason: 'E2E block.' }]);
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}, { DSH_HOME: '/tmp/dsh-home' }),
      moduleLoader: async () => modules,
      guardrail: rulesGuardrail(rules),
    });
    await runtime.start();
    assert.equal(record.guardrails.get(RULES_GUARDRAIL_NAME)?.priority, 1000);

    let ran = false;
    const result = await runtime.executeTool(
      toolExec(fakeAgent(), 'bash', { command: 'echo NEMO_RELAY_E2E_BLOCK' }),
      async () => {
        ran = true;
        return okResult();
      },
    );
    assert.equal(ran, false);
    assert.equal(result.isError ? result.error.info?.code : '', 'NEMO_RELAY_GUARDRAIL_REJECTED');
    assert.equal(
      result.content[0]?.type === 'text' ? result.content[0].text : '',
      'Error: Blocked by NeMo Relay policy: E2E block. (NeMo Relay rule "no-e2e")',
    );
    const allowed = await runtime.executeTool(
      toolExec(fakeAgent(), 'bash', { command: 'echo ok' }, 'call-2'),
      async () => okResult('ok'),
    );
    assert.equal(allowed.isError, false);
    assert.equal(rules.snapshot().rejections, 1);

    await runtime.stop();
    assert.equal(record.guardrails.has(RULES_GUARDRAIL_NAME), false);
  });

  it('still enforces the rules while the Relay binding is unavailable, unless the plugin is disabled', async () => {
    const rules = policyWith([{ id: 'no-e2e', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK', reason: 'E2E block.' }]);
    const exec = toolExec(fakeAgent(), 'bash', { command: 'echo NEMO_RELAY_E2E_BLOCK' });
    for (const [enabled, expected] of [
      [true, 'deny'],
      [false, 'allow'],
    ] as const) {
      const runtime = new NemoRelayDeepSeekHarnessRuntime({
        config: parseConfig({ enabled }),
        moduleLoader: async () => {
          throw new Error('missing native binding');
        },
        guardrail: rulesGuardrail(rules),
      });
      await runtime.start();
      assert.equal(runtime.isActive(), false);
      const decision = await runtime.preExecute(exec, async () => ({ kind: 'allow' }));
      assert.equal(decision.kind, expected);
      if (decision.kind === 'deny') {
        assert.equal(decision.reason, 'Blocked by NeMo Relay policy: E2E block. (NeMo Relay rule "no-e2e")');
      }
    }
  });
});

describe('capture settings', () => {
  it('registers observability-only sanitizers for stripped payloads and removes them on stop', async () => {
    const { runtime, record } = await startRuntime(
      {},
      { capture: { includePrompts: false, stripToolArgs: true, stripToolResults: true } },
    );
    assert.deepEqual([...record.sanitizers.values()].sort(), ['llm-request', 'tool-request', 'tool-response']);
    await runtime.stop();
    assert.equal(record.sanitizers.size, 0);
  });

  it('omits the prompt from the turn input when prompts are not captured', async () => {
    const { runtime, record } = await startRuntime({}, { capture: { includePrompts: false } });
    const session = fakeSession();
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }));
    runtime.onSessionEvent(
      session,
      event('user/message', { content: [{ type: 'text', text: 'private' }], source: { kind: 'user' } }),
    );
    await collect(runtime.streamLlm(agentRequest(session.id), streamOf(textStream())));
    assert.equal(record.pushed[0]?.input, null);
  });
});
