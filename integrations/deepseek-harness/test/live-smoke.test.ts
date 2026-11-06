// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Opt-in live smoke test against the real NeMo Relay Node binding and plugin host.
 */
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { it } from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

import { Worker } from 'node:worker_threads';

import { parseConfig } from '../src/config.js';
import type { GenerateOptions, HarnessToolResult, RequestMessage, StreamChunk } from '../src/dsh-types.js';
import { defaultNemoRelayModuleLoader } from '../src/modules.js';
import { PhoenixManager } from '../src/phoenix.js';
import { atifSessionDirectory } from '../src/plugin-doc.js';
import {
  DEFAULT_RULES,
  FileRulesStore,
  GuardrailPolicy,
  RULES_GUARDRAIL_NAME,
  RULES_GUARDRAIL_PRIORITY,
} from '../src/policy.js';
import { NemoRelayDeepSeekHarnessRuntime } from '../src/runtime.js';
import { SessionTraceWriter, VIEW_REQUEST_SUMMARY_KEY } from '../src/view.js';
import { collect, event, fakeAgent, fakeSession, okResult, streamOf, toolExec } from './fakes.js';

const liveSmokeEnabled = process.env.NEMO_RELAY_DEEPSEEK_HARNESS_LIVE_SMOKE === '1';

const TOOL_CALL_STREAM: StreamChunk[] = [
  { type: 'block-start', index: 0, blockType: 'reasoning' },
  { type: 'reasoning-delta', index: 0, text: 'I should list the files.' },
  { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'I should list the files.' } },
  { type: 'block-start', index: 1, blockType: 'tool-call' },
  { type: 'tool-call-delta', index: 1, id: 'call-1', name: 'bash', argumentsDelta: '{"command":' },
  { type: 'tool-call-delta', index: 1, id: 'call-1', argumentsDelta: '"ls"}' },
  {
    type: 'block-end',
    index: 1,
    block: { type: 'tool-call', id: 'call-1', name: 'bash', arguments: '{"command":"ls"}' },
  },
  { type: 'usage', usage: { inputTokens: 120, outputTokens: 18, cacheReadTokens: 64 } },
  { type: 'finish', reason: { kind: 'tool-calls' } },
];

const ANSWER_STREAM: StreamChunk[] = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: 'The directory has ' },
  { type: 'text-delta', index: 0, text: 'two files.' },
  { type: 'block-end', index: 0, block: { type: 'text', text: 'The directory has two files.' } },
  { type: 'usage', usage: { inputTokens: 160, outputTokens: 9 } },
  { type: 'finish', reason: { kind: 'stop' } },
];

const TITLE_STREAM: StreamChunk[] = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: 'Directory listing' },
  { type: 'block-end', index: 0, block: { type: 'text', text: 'Directory listing' } },
  { type: 'usage', usage: { inputTokens: 40, outputTokens: 3 } },
  { type: 'finish', reason: { kind: 'stop' } },
];

function request(sessionId: string, text: string, history: RequestMessage[] = []): GenerateOptions {
  return {
    provider: 'deepseek',
    model: 'deepseek-chat',
    system: 'You are a coding agent.',
    // DeepSeek Harness's layout: the prompt, then runtime context and skill reminders as separate user messages.
    messages: [
      { role: 'user', content: [{ type: 'text', text }] },
      { id: 'ctx', role: 'user', source: { kind: 'runtime-context' }, content: [{ type: 'text', text: '<ctx/>' }] },
      {
        id: 'skills',
        role: 'user',
        source: { kind: 'skills' },
        content: [{ type: 'text', text: '<system-reminder>skills</system-reminder>' }],
      },
      ...history,
    ],
    tools: [
      {
        name: 'bash',
        description: 'Run a shell command',
        parameters: { type: 'object', properties: { command: { type: 'string' } } },
      },
    ],
    sessionId,
  };
}

/** The history DeepSeek Harness sends after the model called `bash` once. */
const TOOL_HISTORY: RequestMessage[] = [
  {
    id: 'a1',
    role: 'assistant',
    source: { kind: 'model' },
    content: [{ type: 'tool-call', id: 'call-1', name: 'bash', arguments: '{"command":"ls"}' }],
  },
  {
    id: 't1',
    role: 'tool',
    toolCallId: 'call-1',
    source: { kind: 'tool' },
    content: [{ type: 'text', text: 'a.txt' }],
  },
];

async function isolatedDirectories(): Promise<{ root: string; output: string }> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-live-'));
  // Keep ambient user plugin files out of the test; CI has no system plugin file.
  process.env.XDG_CONFIG_HOME = path.join(root, 'xdg');
  const output = path.join(root, 'out');
  await fs.mkdir(output, { recursive: true });
  // Never activate a discovered plugins.toml layer (such as an operator's system file that exports
  // telemetry): validation reads the layers without activating anything, so check it first.
  const { validate } = await import('nemo-relay-node/plugin');
  const layers = validate({ version: 1, components: [] }).config_paths;
  assert.deepEqual(
    layers,
    [],
    `the live smoke test refuses to run over plugins.toml layers ${layers.join(', ')}; set NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG=1 with a binding built with --features __skip-implicit-config`,
  );
  return { root, output };
}

it(
  'runs a DeepSeek Harness turn through managed Relay calls and exports ATOF and ATIF',
  { skip: !liveSmokeEnabled },
  async () => {
    const { root, output } = await isolatedDirectories();
    const config = parseConfig({
      dataDir: path.join(root, 'data'),
      plugins: {
        version: 1,
        components: [
          {
            kind: 'observability',
            enabled: true,
            config: {
              version: 4,
              atof: {
                enabled: true,
                sinks: [{ type: 'file', output_directory: output, filename: 'events.jsonl', mode: 'append' }],
              },
              atif: {
                enabled: true,
                agent_name: 'deepseek-harness',
                output_directory: output,
                filename_template: 'trajectory-{session_id}.json',
              },
            },
          },
        ],
      },
    });
    const writer = new SessionTraceWriter(path.join(config.dataDir, 'view'));
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config,
      moduleLoader: defaultNemoRelayModuleLoader,
      onEvent: (relayEvent) => writer.accept(relayEvent),
    });
    await runtime.start();
    assert.deepEqual(runtime.status(), { state: 'ready' });
    assert.equal(runtime.health().initializedPluginHost, true);

    const session = fakeSession('live-session-1');
    const agent = fakeAgent(session);
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, Date.now(), 1));
    runtime.onSessionEvent(
      session,
      event(
        'user/message',
        {
          id: 'm1',
          role: 'user',
          content: [{ type: 'text', text: 'What is in this directory?' }],
          source: { kind: 'user' },
        },
        Date.now(),
        2,
      ),
    );
    runtime.noteStep(agent, 1, 1);
    assert.deepEqual(
      await collect(runtime.streamLlm(request(session.id, 'What is in this directory?'), streamOf(TOOL_CALL_STREAM))),
      TOOL_CALL_STREAM,
    );
    // DeepSeek Harness titles the session while the turn runs.
    const title: GenerateOptions = {
      provider: 'deepseek',
      model: 'deepseek-chat',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Generate the session title' }] }],
      sessionId: session.id,
      purpose: 'session-title',
    };
    assert.deepEqual(await collect(runtime.streamLlm(title, streamOf(TITLE_STREAM))), TITLE_STREAM);

    const host = okResult('a.txt\nb.txt', { stdout: 'a.txt\nb.txt', exitCode: 0 });
    assert.equal(
      await runtime.executeTool(toolExec(agent, 'bash', { command: 'ls' }, 'call-1'), async () => host),
      host,
    );

    runtime.noteStep(agent, 1, 2);
    assert.deepEqual(
      await collect(
        runtime.streamLlm(request(session.id, 'What is in this directory?', TOOL_HISTORY), streamOf(ANSWER_STREAM)),
      ),
      ANSWER_STREAM,
    );
    runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }, Date.now(), 9));
    await runtime.stop();
    await writer.flush();
    assert.deepEqual(runtime.status(), { state: 'stopped' });

    const events = (await fs.readFile(path.join(output, 'events.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line) as Record<string, any>);
    const starts = events.filter((entry) => entry.kind === 'scope' && entry.scope_category === 'start');
    const turn = starts.find((entry) => entry.metadata?.nemo_relay_scope_role === 'turn');
    assert.equal(turn?.name, 'deepseek-harness-turn');
    assert.equal(turn?.metadata?.session_id, session.id);
    const isTitle = (entry: Record<string, any>) => entry.metadata?.request_purpose === 'session-title';
    const llmStarts = starts.filter((entry) => entry.category === 'llm' && !isTitle(entry));
    assert.equal(llmStarts.length, 2);
    assert.ok(llmStarts.every((entry) => entry.parent_uuid === turn?.uuid));
    // The title call belongs to the session, not to the turn it ran beside.
    const titleStart = starts.find((entry) => entry.category === 'llm' && isTitle(entry));
    assert.ok(titleStart, 'the title call is recorded');
    assert.notEqual(titleStart.parent_uuid, turn?.uuid, 'the title call is not parented to the turn scope');
    assert.equal(titleStart.metadata?.session_id, session.id);
    // Relay records a later call from its last user message on; the merged user message keeps the prompt.
    const secondRequest = JSON.stringify(llmStarts[1]?.data?.content?.messages ?? null);
    assert.ok(
      secondRequest.includes('What is in this directory?'),
      `the second managed call's recorded request contains the prompt: ${secondRequest.slice(0, 400)}`,
    );
    const tool = starts.find((entry) => entry.category === 'tool');
    assert.equal(tool?.name, 'bash');
    assert.equal(tool?.parent_uuid, turn?.uuid);
    const llmEnd = events.find(
      (entry) => entry.kind === 'scope' && entry.scope_category === 'end' && entry.category === 'llm',
    );
    assert.equal(llmEnd?.data?.choices?.[0]?.message?.tool_calls?.[0]?.function?.arguments, '{"command":"ls"}');

    const trajectories = (await fs.readdir(output)).filter((file) => file.startsWith('trajectory-'));
    assert.equal(trajectories.length, 1, 'one ATIF trajectory per turn');
    const trajectory = JSON.parse(await fs.readFile(path.join(output, trajectories[0] ?? ''), 'utf8')) as Record<
      string,
      any
    >;
    assert.equal(trajectory.agent?.name, 'deepseek-harness');
    assert.ok(Array.isArray(trajectory.steps) && trajectory.steps.length >= 2);
    assert.ok(
      !JSON.stringify(trajectory.steps).includes('Directory listing'),
      'the turn trajectory leaves out the title call',
    );

    const viewTrace = await fs.readFile(writer.pathFor(session.id), 'utf8');
    assert.ok(!viewTrace.includes('"llm.chunk"'), 'the view trace skips per-chunk receipt marks');
    const viewEvents = viewTrace
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line) as Record<string, any>);

    // Relay's exporter keeps each complete request; the view trace keeps what the view shows and the sizes.
    const exported = llmStarts[1]?.data?.content;
    assert.ok(Array.isArray(exported?.tools) && exported.tools.length > 0, 'the ATOF exporter records the tool list');
    assert.ok(exported.messages.length > 2, 'the ATOF exporter records the message history');
    const compact = viewEvents.find((entry) => entry.uuid === llmStarts[1]?.uuid);
    assert.ok(compact, 'the view trace has the same call');
    assert.equal('tools' in compact.data.content, false);
    assert.equal(compact.category_profile?.annotated_request, undefined);
    assert.deepEqual(compact.metadata?.[VIEW_REQUEST_SUMMARY_KEY], {
      request_messages: exported.messages.length,
      request_tools: exported.tools.length,
    });
    assert.ok(compact.data.content.messages.length < exported.messages.length);
    assert.ok(JSON.stringify(compact.data.content.messages).includes('What is in this directory?'));

    // The conversation view must understand the trace this binding produced.
    const view = await loadClientModel();
    const model = view.buildModel(viewEvents);
    assert.equal(model.turns.length, 1);
    const summary = view.turnSummary(model.turns[0]);
    assert.deepEqual([summary.llm, summary.tools, summary.errors, summary.blocked], [2, 1, 0, 0]);
  },
);

/** Evaluate the Client view through a stub module loader and return its model helpers. */
async function loadClientModel(): Promise<{ buildModel(events: unknown[]): any; turnSummary(turn: unknown): any }> {
  const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
  const source = await fs.readFile(path.join(packageRoot, 'client', 'client.js'), 'utf8');
  let definition: { factory(require: (name: string) => unknown): { __test: any } } | undefined;
  const loader = { load: (value: typeof definition) => void (definition = value) };
  vm.runInNewContext(source, { window: { __ModuleLoader__: loader } });
  assert.ok(definition, 'client.js registers a module');
  // Only the model helpers run here, so React needs no working renderer.
  const react = { createElement: () => null, memo: (component: unknown) => component, Fragment: 'fragment' };
  return definition.factory(() => react).__test;
}

it(
  'enforces Relay guardrails and refuses request rewrites with the real binding',
  { skip: !liveSmokeEnabled },
  async () => {
    await isolatedDirectories();
    const nf = await import('nemo-relay-node');
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({}),
      moduleLoader: defaultNemoRelayModuleLoader,
    });
    await runtime.start();
    const session = fakeSession('live-session-2');
    const agent = fakeAgent(session);
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, Date.now(), 1));

    nf.registerToolConditionalExecutionGuardrail('live.no-force-push', 100, (toolName: string, args: unknown) =>
      toolName === 'bash' && JSON.stringify(args).includes('--force') ? 'Force pushes are not allowed.' : null,
    );
    nf.registerLlmRequestIntercept('live.drop-history', 100, false, ({ request, annotated }) => {
      const edited = structuredClone(annotated) as Record<string, any> | null;
      if (edited !== null && Array.isArray(edited.messages)) {
        edited.messages = edited.messages.slice(-1);
      }
      return { request, annotated: edited };
    });
    try {
      let ran = false;
      const blocked = await runtime.executeTool(toolExec(agent, 'bash', { command: 'git push --force' }), async () => {
        ran = true;
        return okResult();
      });
      assert.equal(ran, false);
      assert.equal(blocked.isError ? blocked.error.info?.code : '', 'NEMO_RELAY_GUARDRAIL_REJECTED');
      assert.match(
        blocked.content[0]?.type === 'text' ? blocked.content[0].text : '',
        /Force pushes are not allowed\./,
      );

      const allowed = await runtime.executeTool(
        toolExec(agent, 'bash', { command: 'git status' }, 'call-2'),
        async () => okResult('clean'),
      );
      assert.equal(allowed.isError, false);

      let providerCalled = false;
      const chunks = await collect(
        runtime.streamLlm(request(session.id, 'hello'), () => {
          providerCalled = true;
          return streamOf(ANSWER_STREAM)();
        }),
      );
      assert.equal(providerCalled, false);
      const finish = chunks.at(-1);
      assert.equal(
        finish?.type === 'finish' && finish.reason.kind === 'error' ? finish.reason.failure.code : '',
        'NEMO_RELAY_REQUEST_REWRITE',
      );
    } finally {
      nf.deregisterToolConditionalExecutionGuardrail('live.no-force-push');
      nf.deregisterLlmRequestIntercept('live.drop-history');
      await runtime.stop();
    }
  },
);

it(
  'activates the default plugin document, scrubs credentials from every event, and writes per-session ATIF',
  { skip: !liveSmokeEnabled },
  async () => {
    const { root } = await isolatedDirectories();
    const dataDir = path.join(root, 'data');
    const writer = new SessionTraceWriter(path.join(dataDir, 'view'));
    const events: Array<Record<string, any>> = [];
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({ dataDir }),
      moduleLoader: defaultNemoRelayModuleLoader,
      onEvent: (relayEvent) => {
        events.push(relayEvent as Record<string, any>);
        writer.accept(relayEvent);
      },
      defaultModel: () => 'deepseek-chat',
    });
    await runtime.start();
    // The one-click defaults initialize cleanly: Relay's pii_redaction and observability components.
    assert.deepEqual(runtime.status(), { state: 'ready' });
    assert.equal(runtime.health().initializedPluginHost, true);
    assert.equal(runtime.pluginPlan()?.redaction.applied, true);
    assert.equal(runtime.pluginPlan()?.atif.applied, true);

    const secret = 'sk-test1234567890abcdef1234';
    const session = fakeSession('live-session-3');
    const agent = fakeAgent(session);
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, Date.now(), 1));
    runtime.onSessionEvent(
      session,
      event('user/message', {
        content: [{ type: 'text', text: `Use ${secret} for the API` }],
        source: { kind: 'user' },
      }),
    );
    runtime.noteStep(agent, 1, 1);
    await collect(runtime.streamLlm(request(session.id, `Use ${secret} for the API`), streamOf(ANSWER_STREAM)));
    const host = okResult(`OPENAI_API_KEY=${secret}`, { stdout: `OPENAI_API_KEY=${secret}` });
    const result = await runtime.executeTool(
      toolExec(agent, 'bash', { command: `printenv OPENAI_API_KEY # ${secret}` }, 'call-secret'),
      async () => host,
    );
    assert.equal(result, host, 'the host and the model still receive the real result');
    runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }, Date.now(), 9));
    await runtime.stop();
    await writer.flush();

    const published = JSON.stringify(events);
    assert.ok(
      events.some((entry) => entry.category === 'tool'),
      'the tool call was observed',
    );
    assert.ok(!published.includes(secret), 'no published event carries the credential');
    assert.ok(published.includes('[REDACTED]'));
    const viewTrace = await fs.readFile(writer.pathFor(session.id), 'utf8');
    assert.ok(!viewTrace.includes(secret), 'the ATOF view trace carries no credential');

    // One trajectory per turn, in the session's own directory.
    const sessionDir = atifSessionDirectory(path.join(dataDir, 'atif'), session.id);
    const files = await fs.readdir(sessionDir);
    assert.equal(files.length, 1);
    assert.match(files[0] ?? '', /^trajectory-.+\.json$/);
    const trajectory = await fs.readFile(path.join(sessionDir, files[0] ?? ''), 'utf8');
    assert.ok(!trajectory.includes(secret), 'the ATIF trajectory carries no credential');
    assert.equal(JSON.parse(trajectory).extra?.nemo_relay?.session_id, session.id);
    // The provider request, tool list included, stays in the ATOF trace: no step carries a copy.
    assert.ok(JSON.parse(trajectory).steps.length > 0);
    assert.ok(!trajectory.includes('"llm_request"'), 'no step copies the provider request');
    // The default trajectories describe DeepSeek Harness and its default model, not Relay's placeholders.
    assert.deepEqual(
      [JSON.parse(trajectory).agent?.name, JSON.parse(trajectory).agent?.model_name],
      ['deepseek-harness', 'deepseek-chat'],
    );
    // A completed turn ends with an OK status for Relay's OpenTelemetry exporters.
    const turnEnd = events.find(
      (entry) => entry.scope_category === 'end' && entry.metadata?.nemo_relay_scope_role === 'turn',
    );
    assert.equal(turnEnd?.metadata?.['otel.status_code'], 'OK');
  },
);

it(
  'blocks managed tool calls with NeMo Relay rules, including a rule saved a moment earlier',
  { skip: !liveSmokeEnabled },
  async () => {
    const { root } = await isolatedDirectories();
    const dataDir = path.join(root, 'data');
    const policy = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: new FileRulesStore(path.join(dataDir, 'rules.json')),
    });
    const events: Array<Record<string, any>> = [];
    const runtime = new NemoRelayDeepSeekHarnessRuntime({
      config: parseConfig({ dataDir }),
      moduleLoader: defaultNemoRelayModuleLoader,
      onEvent: (relayEvent) => void events.push(relayEvent as Record<string, any>),
      guardrail: {
        name: RULES_GUARDRAIL_NAME,
        priority: RULES_GUARDRAIL_PRIORITY,
        evaluate: (toolName, args) => policy.evaluate(toolName, args),
      },
    });
    await runtime.start();
    const session = fakeSession('live-session-4');
    const agent = fakeAgent(session);
    runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, Date.now(), 1));
    const text = (result: HarnessToolResult) => (result.content[0]?.type === 'text' ? result.content[0].text : '');
    // No tool body below executes anything; the commands are only text the rules inspect.
    let ran = 0;
    const body = async () => {
      ran += 1;
      return okResult('ran');
    };
    try {
      const rm = ['rm', '-rf'].join(' ');
      const shipped = await runtime.executeTool(toolExec(agent, 'bash', { command: `echo ${rm} ~` }, 'call-a'), body);
      assert.equal(
        text(shipped),
        `Error: Blocked by NeMo Relay policy: ${DEFAULT_RULES[0]?.reason} (NeMo Relay rule "no-recursive-delete-of-root-or-home")`,
      );

      const before = policy.snapshot();
      const saved = await policy.save(before.revision, [
        ...before.rules,
        { id: 'e2e-block', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK', reason: 'Blocked for the live smoke test.' },
      ]);
      assert.equal(saved.ok, true);
      const blocked = await runtime.executeTool(
        toolExec(agent, 'bash', { command: 'echo NEMO_RELAY_E2E_BLOCK' }, 'call-b'),
        body,
      );
      assert.equal(
        text(blocked),
        'Error: Blocked by NeMo Relay policy: Blocked for the live smoke test. (NeMo Relay rule "e2e-block")',
      );
      // An approval prompt for a blocked call is denied before the user is asked.
      const denied = await runtime.preExecute(
        toolExec(agent, 'bash', { command: 'echo NEMO_RELAY_E2E_BLOCK' }, 'call-c'),
        async () => ({ kind: 'ask' }),
      );
      assert.equal(denied.kind, 'deny');
      const allowed = await runtime.executeTool(toolExec(agent, 'bash', { command: 'echo ok' }, 'call-d'), body);
      assert.equal(allowed.isError, false);
      assert.equal(ran, 1, 'only the allowed call ran its body');
    } finally {
      runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }, Date.now(), 9));
      await runtime.stop();
    }

    // Relay records every rejected managed call as a mark named after the tool, carrying the rule's reason.
    const rejections = events.filter((entry) => entry.kind === 'mark' && entry.data?.rejected === true);
    assert.deepEqual(
      rejections.map((entry) => [entry.name, entry.data.rejection_reason, entry.metadata?.session_id]),
      [
        ['bash', `${DEFAULT_RULES[0]?.reason} (NeMo Relay rule "no-recursive-delete-of-root-or-home")`, session.id],
        ['bash', 'Blocked for the live smoke test. (NeMo Relay rule "e2e-block")', session.id],
      ],
    );
    assert.ok(
      events.some((entry) => entry.kind === 'scope' && JSON.stringify(entry).includes(RULES_GUARDRAIL_NAME)),
      'the rules guardrail evaluations are observable',
    );
    assert.equal(policy.snapshot().rejections, 3);
  },
);

it(
  'exports OpenInference spans to a Phoenix through Relay with the project and session',
  { skip: !liveSmokeEnabled },
  async () => {
    const { root } = await isolatedDirectories();
    const project = 'nemo-relay-live-smoke';
    const session = fakeSession('live-session-5');
    const receiver = await startFakePhoenix([project, session.id]);
    const dataDir = path.join(root, 'data');
    const config = parseConfig({ dataDir, phoenix: { url: receiver.url, project, launch: { enabled: false } } });
    const runtime = new NemoRelayDeepSeekHarnessRuntime({ config, moduleLoader: defaultNemoRelayModuleLoader });
    await runtime.start();
    const phoenix = new PhoenixManager({ config: config.phoenix, dataDir, exporter: runtime.phoenixExporter() });
    try {
      const started = await phoenix.start();
      assert.deepEqual(
        [started.ok, started.phoenix.state, started.phoenix.mode, started.phoenix.version, started.phoenix.sessionUrl],
        [true, 'running', 'external', '21.0.0', `${receiver.url}/redirects/sessions/{sessionId}`],
      );
      const agent = fakeAgent(session);
      runtime.onSessionEvent(session, event('turn/start', { turn: 1 }, Date.now(), 1));
      runtime.noteStep(agent, 1, 1);
      await collect(runtime.streamLlm(request(session.id, 'What is in this directory?'), streamOf(ANSWER_STREAM)));
      runtime.onSessionEvent(session, event('turn/end', { turn: 1, reason: { kind: 'completed' } }, Date.now(), 9));
      // Stop flushes the spans and disconnects; a Phoenix someone else runs keeps running.
      const stopped = await phoenix.stop();
      assert.equal(stopped.phoenix.state, 'stopped');
      const traces = await receiver.waitFor((request) => request.url === '/v1/traces');
      assert.equal(traces.type, 'application/x-protobuf');
      assert.deepEqual(
        traces.contains,
        { [project]: true, [session.id]: true },
        'the project and session reach Phoenix',
      );
    } finally {
      await phoenix.dispose();
      await runtime.stop();
      await receiver.close();
    }
  },
);

type ReceivedRequest = { url: string; method: string; type: string; contains: Record<string, boolean> };

/**
 * A stand-in for Phoenix in a worker thread: it answers the health check and
 * records OTLP requests. Relay's exporter flushes synchronously on the main
 * thread, so the receiver must run on another one.
 */
async function startFakePhoenix(needles: string[]): Promise<{
  url: string;
  waitFor(predicate: (request: ReceivedRequest) => boolean): Promise<ReceivedRequest>;
  close(): Promise<void>;
}> {
  const worker = new Worker(
    `
    const http = require('node:http');
    const { parentPort, workerData } = require('node:worker_threads');
    const server = http.createServer((req, res) => {
      const chunks = [];
      req.on('data', (chunk) => chunks.push(chunk));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        const contains = Object.fromEntries(workerData.needles.map((needle) => [needle, body.includes(needle)]));
        parentPort.postMessage({ url: req.url, method: req.method, type: req.headers['content-type'] ?? '', contains });
        if (req.url === '/arize_phoenix_version') {
          res.writeHead(200, { 'content-type': 'text/plain' });
          res.end('21.0.0');
        } else {
          res.writeHead(200, { 'content-type': 'application/x-protobuf' });
          res.end();
        }
      });
    });
    server.listen(0, '127.0.0.1', () => parentPort.postMessage({ port: server.address().port }));
    `,
    { eval: true, workerData: { needles } },
  );
  const received: ReceivedRequest[] = [];
  const port = await new Promise<number>((resolve, reject) => {
    worker.once('error', reject);
    worker.on('message', (message: { port?: number } & ReceivedRequest) =>
      message.port === undefined ? void received.push(message) : resolve(message.port),
    );
  });
  return {
    url: `http://127.0.0.1:${port}`,
    async waitFor(predicate) {
      const deadline = Date.now() + 5_000;
      for (;;) {
        const match = received.find(predicate);
        if (match !== undefined) {
          return match;
        }
        assert.ok(Date.now() < deadline, `the receiver got no matching request: ${JSON.stringify(received)}`);
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
    },
    close: async () => void (await worker.terminate()),
  };
}
