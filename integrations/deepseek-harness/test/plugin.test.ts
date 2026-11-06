// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin entry, configuration, projection, and view-trace tests.
 */
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, stat } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { apply, VIEW_PROJECTION_KEY } from '../index.js';
import type { StreamChunk } from '../src/dsh-types.js';
import {
  ChunkBridge,
  classifyRequestEdits,
  projectRequest,
  projectUsage,
  UnsupportedRewriteError,
} from '../src/llm-projection.js';
import type { NemoRelayModules } from '../src/modules.js';
import { compactForView, SessionTraceWriter, VIEW_REQUEST_SUMMARY_KEY, VIEW_TOOL_RESULT_CHARS } from '../src/view.js';
import { createFakeModules } from './fakes.js';

describe('plugin entry', () => {
  it('registers global listeners, the /relay command, and the view projection', async () => {
    const { modules, record } = createFakeModules();
    const listeners = new Map<string, { options: unknown }>();
    const disposers: Array<() => unknown> = [];
    let command: any;
    let projection: any;
    const scope = {
      commands: { register: (definition: unknown) => ((command = definition), () => undefined) },
      sessionProjections: { register: (definition: unknown) => void (projection = definition) },
      effect: (execute: () => () => unknown) => disposers.push(execute()),
    };
    const ctx = {
      on: (name: string, _listener: unknown, options?: unknown) => (listeners.set(name, { options }), () => undefined),
      effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
      inject: (_names: string[], callback: (child: any) => void) => callback(scope),
      get: () => undefined,
      logger: { info: () => undefined, warn: () => undefined },
    };
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-plugin-'));

    apply(ctx, { dataDir }, { moduleLoader: async () => modules });
    await new Promise((resolve) => setTimeout(resolve, 10));

    assert.deepEqual([...listeners.keys()].sort(), [
      'agent/disposed',
      'agent/pre-step',
      'llm/stream',
      'session/event',
      'tools/execute',
      'tools/post-execute',
      'tools/pre-execute',
    ]);
    for (const { options } of listeners.values()) {
      assert.deepEqual(options, { global: true });
    }
    assert.equal(command.name, 'relay');
    assert.equal(command.recordInput, false, 'rule payloads stay out of the session log');
    const reply = await command.handler({
      agent: { id: 's1', session: { id: 's1', header: { id: 's1' } } },
      rawInput: '',
    });
    assert.equal(reply.kind, 'success');
    assert.match(reply.text, /^NeMo Relay: ready/);
    assert.match(reply.text, /s1\.atof\.jsonl/);
    assert.match(reply.text, /Rules: 1 of 1 on, fail-open, shipped defaults/);

    assert.equal(projection.key, VIEW_PROJECTION_KEY);
    const state = projection.apply(projection.init({ id: 's/1' }), { type: 'turn/start', seq: 4, time: 0 });
    const view = projection.wire.view(state);
    assert.deepEqual(view, {
      v: 4,
      sessionId: 's/1',
      seq: 4,
      file: path.join(dataDir, 'view', 's_1.atof.jsonl'),
      // This process's own status file.
      status: view.status,
      // Relay's own filename sanitization of `{metadata.session_id}`.
      atifDir: path.join(dataDir, 'atif', 's-1'),
    });
    assert.equal(path.dirname(view.status), path.join(dataDir, 'status'));
    assert.match(path.basename(view.status), new RegExp(`^${process.pid}-[0-9a-f]{4}\\.json$`));

    const running = JSON.parse(await readFile(view.status, 'utf8'));
    assert.equal(running.v, 4);
    assert.equal(running.state, 'ready');
    assert.equal(running.pid, process.pid);
    assert.equal(running.statusFile, view.status);
    // This checkout resolves schemastery, so the plugin exports its schema; the test passes a raw row.
    assert.deepEqual(running.config, { schema: true, live: false });
    assert.equal(running.guardrails.editable, true);
    assert.equal(running.guardrails.source, 'default');
    assert.deepEqual(running.capture, {
      includePrompts: true,
      includeResponses: true,
      stripToolArgs: false,
      stripToolResults: false,
    });
    assert.deepEqual(running.exporters, [{ kind: 'atif', target: path.join(dataDir, 'atif') }]);
    assert.equal(running.redaction.secrets, true);
    assert.equal(running.phoenix.state, 'stopped');

    for (const dispose of disposers.reverse()) {
      await dispose();
    }
    assert.equal(record.closed, 1);
    await assert.rejects(stat(view.status), { code: 'ENOENT' }, 'a stopped process removes its status file');
    // The command still answers from memory.
    const stopped = JSON.parse((await command.handler({ agent: undefined, rawInput: 'json' })).text);
    assert.equal(stopped.state, 'stopped');
    assert.equal(stopped.guardrails.editable, false);
  });

  it('keeps the status of each DeepSeek Harness process sharing a data directory apart', async () => {
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-multi-'));
    const start = async () => {
      const { modules } = createFakeModules();
      const disposers: Array<() => unknown> = [];
      let projection: any;
      const scope = {
        commands: { register: () => () => undefined },
        sessionProjections: { register: (definition: unknown) => void (projection = definition) },
        effect: (execute: () => () => unknown) => disposers.push(execute()),
      };
      const ctx = {
        on: () => () => undefined,
        effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
        inject: (_names: string[], callback: (child: any) => void) => callback(scope),
        get: () => undefined,
        logger: { info: () => undefined, warn: () => undefined },
      };
      apply(ctx, { dataDir }, { moduleLoader: async () => modules });
      await new Promise((resolve) => setTimeout(resolve, 10));
      const status: string = projection.wire.view(projection.init({ id: 's' })).status;
      return {
        status,
        stop: async () => {
          for (const dispose of disposers.reverse()) {
            await dispose();
          }
        },
      };
    };
    // The desktop app keeps running while a CLI run in the same DSH_HOME starts and exits.
    const desktop = await start();
    const cli = await start();
    assert.notEqual(cli.status, desktop.status);
    await cli.stop();
    await assert.rejects(stat(cli.status), { code: 'ENOENT' });
    assert.equal(
      JSON.parse(await readFile(desktop.status, 'utf8')).state,
      'ready',
      'the running process still reads ready',
    );
    await desktop.stop();
  });

  it('applies a rule saved with /relay to the very next managed tool call, without a restart', async () => {
    const { modules } = createFakeModules();
    const host = await applyWithHost({ info: () => undefined, warn: () => undefined }, modules);
    const agent = { id: 's1', session: { id: 's1', header: { id: 's1', createdAt: 0 } } };
    const run = async (rawInput: string) => JSON.parse((await host.command.handler({ agent, rawInput })).text);
    const execute = (command: string) =>
      host.listeners.get('tools/execute')?.(
        {
          callId: 'c',
          rootCallId: 'c',
          name: 'bash',
          arguments: { command },
          agent,
          signal: new AbortController().signal,
        },
        async () => ({ isError: false, value: null, content: [{ type: 'text', text: 'ran' }] }),
      );

    assert.equal((await execute('echo NEMO_RELAY_E2E_BLOCK')).isError, false);
    const before = await run('json');
    const reply = await host.command.handler({
      agent,
      rawInput: `rules save ${JSON.stringify({
        base: before.guardrails.revision,
        rules: [
          ...before.guardrails.rules,
          { id: 'e2e-block', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK', reason: 'Blocked for the e2e check.' },
        ],
        rid: 'view-save-1',
      })}`,
    });
    // The transcript shows a sentence; the view reads the structured result from the status file,
    // which the command rewrote before it replied.
    const status = JSON.parse(await readFile(before.statusFile, 'utf8'));
    const result = status.actions.find((entry: { rid?: string }) => entry.rid === 'view-save-1');
    assert.deepEqual(reply, {
      kind: 'success',
      text: `Saved 2 NeMo Relay rules (revision ${status.guardrails.revision}).`,
    });
    assert.deepEqual([result.action, result.ok, result.revision], ['rules.save', true, status.guardrails.revision]);
    assert.equal(status.guardrails.source, 'file');
    const blocked = await execute('echo NEMO_RELAY_E2E_BLOCK');
    assert.equal(
      blocked.content[0].text,
      'Error: Blocked by NeMo Relay policy: Blocked for the e2e check. (NeMo Relay rule "e2e-block")',
    );
    assert.equal((await run('json')).guardrails.rejections, 1);
    await host.dispose();
    assert.deepEqual(host.unhandled, []);
  });

  it('changes the fail mode with /relay while running, and a new plugin generation keeps it', async () => {
    const quiet = { info: () => undefined, warn: () => undefined };
    const agent = { id: 's1', session: { id: 's1', header: { id: 's1', createdAt: 0 } } };
    const first = await applyWithHost(quiet, createFakeModules().modules);
    const run = async (host: typeof first, rawInput: string) =>
      JSON.parse((await host.command.handler({ agent, rawInput })).text);
    const changed = await first.command.handler({ agent, rawInput: 'rules failmode closed' });
    assert.deepEqual(changed, { kind: 'success', text: 'Fail mode is now closed.' });
    const status = JSON.parse(await readFile((await run(first, 'json')).statusFile, 'utf8'));
    assert.equal(status.guardrails.failMode, 'closed', 'the status file reports it');
    // Without a config editor the fail mode is saved beside the rules.
    assert.equal(JSON.parse(await readFile(path.join(first.dataDir, 'rules.json'), 'utf8')).failMode, 'closed');
    await first.dispose();

    const second = await applyWithHost(quiet, createFakeModules().modules, first.dataDir);
    assert.equal((await run(second, 'json')).guardrails.failMode, 'closed');
    await second.dispose();
    assert.deepEqual([...first.unhandled, ...second.unhandled], []);
  });

  it('applies configured rules without a config editor, and rules saved to rules.json once it has one', async () => {
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-stores-'));
    const noCurl = { id: 'no-curl', tool: 'bash', match: 'curl', reason: 'No curl.' };
    const ENTRY = { id: 'nemo-relay' };
    /** Start the plugin on `row`, with DeepSeek Harness's config editor when `editor` is given. */
    const start = async (row: Record<string, unknown>, editor?: Record<string, unknown>) => {
      const disposers: Array<() => unknown> = [];
      const warnings: string[] = [];
      let command: any;
      const scope = {
        commands: { register: (definition: unknown) => ((command = definition), () => undefined) },
        sessionProjections: { register: () => undefined },
        effect: (execute: () => () => unknown) => disposers.push(execute()),
      };
      const ctx = {
        on: () => () => undefined,
        effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
        inject: (_names: string[], callback: (child: any) => void) => callback(scope),
        get: (name: string) => (name === 'configEditor' ? editor : undefined),
        logger: { info: () => undefined, warn: (message: string) => void warnings.push(message) },
        fiber: { entry: ENTRY },
      };
      apply(
        ctx,
        { dataDir, phoenix: { enabled: false }, ...row },
        { moduleLoader: async () => createFakeModules().modules },
      );
      await new Promise((resolve) => setTimeout(resolve, 20));
      const run = (rawInput: string) => command.handler({ agent: undefined, rawInput });
      const guardrails = async () => {
        const status = JSON.parse((await run('json')).text).guardrails;
        return [status.source, status.failMode, status.rules.map((rule: { id: string }) => rule.id)];
      };
      const stop = async () => {
        for (const dispose of disposers.reverse()) {
          await dispose();
        }
      };
      return { run, guardrails, warnings, stop };
    };
    const configured = { guardrails: { failMode: 'closed', rules: [noCurl] } };

    // No config editor, or one that cannot address this row: the configured rules apply, with one warning.
    for (const editor of [undefined, { edit: async () => undefined, entries: () => [] }]) {
      const host = await start(configured, editor);
      assert.deepEqual(await host.guardrails(), ['config', 'closed', ['no-curl']]);
      assert.equal(host.warnings.filter((message) => /saves rule edits in .*rules\.json/.test(message)).length, 1);
      await host.stop();
    }

    // A save without a config editor goes to rules.json ...
    const headless = await start(configured);
    const base = JSON.parse((await headless.run('json')).text).guardrails;
    const added = { id: 'no-wget', tool: 'bash', match: 'wget', reason: 'No wget.' };
    const saved = await headless.run(
      `rules save ${JSON.stringify({ base: base.revision, rules: [...base.rules, added] })}`,
    );
    assert.equal(saved.kind, 'success');
    assert.deepEqual(await headless.guardrails(), ['file', 'closed', ['no-curl', 'no-wget']]);
    await headless.stop();

    // ... and a later start with a config editor, on a row without rules, still applies them.
    let row: Record<string, unknown> = {};
    const edits: unknown[] = [];
    const editor = {
      edit: async (entry: unknown, change: (current: Record<string, unknown>, inherited: object) => typeof row) => {
        edits.push(entry);
        row = change(row, {});
      },
    };
    const editable = await start({}, editor);
    assert.deepEqual(await editable.guardrails(), ['file', 'open', ['no-curl', 'no-wget']]);
    assert.deepEqual(editable.warnings, []);
    // The next save moves them into the row.
    const current = JSON.parse((await editable.run('json')).text).guardrails;
    const moved = await editable.run(
      `rules save ${JSON.stringify({ base: current.revision, rules: current.rules.slice(0, 1) })}`,
    );
    assert.equal(moved.kind, 'success');
    assert.deepEqual(edits, [ENTRY]);
    assert.deepEqual(
      (row.guardrails as { rules: Array<{ id: string }> }).rules.map((rule) => rule.id),
      ['no-curl'],
    );
    assert.deepEqual(await editable.guardrails(), ['config', 'open', ['no-curl']]);
    await assert.rejects(stat(path.join(dataDir, 'rules.json')), { code: 'ENOENT' });
    await editable.stop();
  });

  it('offers the agent the nemo_relay tool and skill over the operations /relay uses, asking before loosening', async () => {
    const host = await applyWithAgentSurface({});
    assert.deepEqual(host.injected.sort(), ['commands', 'sessionProjections', 'skills', 'tools']);
    const tool = host.tools[0];
    assert.equal(tool?.name, 'nemo_relay');
    const exec = { agent: { id: 's1', session: { id: 's1', header: { id: 's1', createdAt: 0 } } } };
    const json = async () => JSON.parse((await host.command.handler({ agent: exec.agent, rawInput: 'json' })).text);

    // The agent adds a rule: the user's /relay sees it, and Relay enforces it on the next call.
    const added = await tool.execute(
      { action: 'rules.add', id: 'no-force-push', tool: 'bash', match: 'push\\s+--force', reason: 'No force pushes.' },
      exec,
    );
    assert.equal(added.action, 'rules.add');
    assert.ok((await json()).guardrails.rules.some((rule: { id: string }) => rule.id === 'no-force-push'));
    const blocked = await host.listeners.get('tools/execute')?.(
      { callId: 'c', rootCallId: 'c', name: 'bash', arguments: { command: 'git push --force' }, ...exec },
      async () => ({ isError: false, value: null, content: [] }),
    );
    assert.match(blocked.content[0].text, /No force pushes\. \(NeMo Relay rule "no-force-push"\)/);
    await tool.execute({ action: 'guardrails.failMode', mode: 'closed' }, exec);
    assert.equal((await json()).guardrails.failMode, 'closed', 'the host fail-mode operation is wired');

    // Loosening changes ask the user; a downstream deny still wins; other calls pass through.
    const preExecute = (name: string, args: unknown, downstream: unknown) =>
      host.listeners.get('tools/pre-execute')?.(
        { callId: 'p', rootCallId: 'p', name, arguments: args, ...exec, signal: new AbortController().signal },
        async () => downstream,
      );
    const remove = { action: 'rules.remove', id: 'no-force-push' };
    const asked = await preExecute('nemo_relay', remove, { kind: 'allow' });
    assert.equal(asked.kind, 'ask');
    assert.equal(typeof asked.displayReason.en, 'string');
    assert.equal(typeof asked.displayReason.zh, 'string');
    assert.deepEqual(await preExecute('nemo_relay', remove, { kind: 'deny', reason: 'no' }), {
      kind: 'deny',
      reason: 'no',
    });
    assert.deepEqual(await preExecute('nemo_relay', { action: 'rules.add' }, { kind: 'allow' }), { kind: 'allow' });
    assert.deepEqual(await preExecute('bash', { command: 'ls' }, { kind: 'allow' }), { kind: 'allow' });

    // The skill describes the live state.
    const provider = host.skillProviders[0]?.();
    assert.deepEqual(
      (await provider.list()).map((skill: { name: string }) => skill.name),
      ['nemo-relay'],
    );
    assert.match((await provider.get({ name: 'nemo-relay' })).content, /no-force-push/);

    await host.dispose();
    assert.deepEqual([host.unregistered.sort(), host.unhandled], [['skill', 'tool'], []]);
  });

  it('keeps the agent surface off with agent.enabled: false', async () => {
    const host = await applyWithAgentSurface({ agent: { enabled: false } });
    assert.deepEqual(host.injected.sort(), ['commands', 'sessionProjections']);
    const decision = await host.listeners.get('tools/pre-execute')?.(
      {
        callId: 'p',
        rootCallId: 'p',
        name: 'nemo_relay',
        arguments: { action: 'rules.reset' },
        signal: new AbortController().signal,
      },
      async () => ({ kind: 'allow' }),
    );
    assert.deepEqual(decision, { kind: 'allow' });
    await host.dispose();
  });

  it('records the step from agent/pre-step and passes the decision through untouched', async () => {
    const { modules, record } = createFakeModules();
    const listeners = new Map<string, (...args: any[]) => any>();
    const disposers: Array<() => unknown> = [];
    const ctx = {
      on: (name: string, listener: (...args: any[]) => any) => (listeners.set(name, listener), () => undefined),
      effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
      inject: () => undefined,
      get: () => undefined,
      logger: { info: () => undefined, warn: () => undefined },
    };
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-prestep-'));
    apply(ctx, { dataDir }, { moduleLoader: async () => modules });
    await new Promise((resolve) => setTimeout(resolve, 10));
    const preStep = listeners.get('agent/pre-step');
    const execute = listeners.get('tools/execute');
    assert.ok(preStep && execute);

    const session = { id: 's1', header: { id: 's1', createdAt: 0 } };
    const agent = { id: 's1', session };
    listeners.get('session/event')?.(session, { type: 'turn/start', seq: 1, time: 1, data: { turn: 2 } });
    const decision = Object.freeze({ kind: 'enter', messages: [], startsRequestSeries: true });
    const signal = new AbortController().signal;
    assert.equal(await preStep({ agent, messages: [], turn: 2, step: 5, signal }, async () => decision), decision);
    await execute(
      { callId: 'c1', rootCallId: 'c1', name: 'bash', arguments: { command: 'ls' }, agent, signal },
      async () => ({ isError: false, value: null, content: [] }),
    );
    assert.equal(record.tools[0]?.metadata.turn_index, 2);
    assert.equal(record.tools[0]?.metadata.step_index, 5);

    // A malformed payload is ignored; a rejection from a later listener is not this plugin's to swallow.
    const reject = { kind: 'reject' };
    assert.equal(await preStep({ agent: undefined, turn: 'x', step: null }, async () => reject), reject);
    await assert.rejects(
      preStep({ agent, messages: [], turn: 2, step: 6, signal }, async () => {
        throw new Error('later listener failed');
      }),
      /later listener failed/,
    );
    for (const dispose of disposers.reverse()) {
      await dispose();
    }
  });

  it('logs plugin-host diagnostics through a Cordis logger whose methods read `this`', async () => {
    // Same shape as Cordis LoggerService: the service is callable and its methods call `this()`.
    const messages: string[] = [];
    const service: any = function () {
      return {
        info: (message: string) => messages.push(`info: ${message}`),
        warn: (message: string) => messages.push(`warn: ${message}`),
      };
    };
    for (const level of ['info', 'warn']) {
      service[level] = function (this: any, ...args: unknown[]) {
        return this()[level](...args);
      };
    }
    const { modules } = createFakeModules({
      diagnostics: [
        { level: 'warning', code: 'plugin.configuration_inherited', message: 'inherited /etc/nemo-relay/plugins.toml' },
        { level: 'error', code: 'plugin.invalid', message: 'bad component' },
      ],
    });
    const host = await applyWithHost(service, modules);

    assert.deepEqual(host.unhandled, []);
    assert.ok(messages.some((message) => message.startsWith('info: ') && message.includes('configuration_inherited')));
    assert.ok(messages.some((message) => message.startsWith('warn: ') && message.includes('plugin.invalid')));
    await host.dispose();
  });

  it('keeps the host alive when logging fails', async () => {
    const throwing = {
      info: () => {
        throw new Error('logger unavailable');
      },
      warn: () => {
        throw new Error('logger unavailable');
      },
    };
    const { modules } = createFakeModules({
      diagnostics: [{ level: 'error', code: 'plugin.invalid', message: 'bad component' }],
      initializeError: 'never reached',
    });
    const host = await applyWithHost(throwing, modules);

    assert.deepEqual(host.unhandled, []);
    const reply = await host.command.handler({ agent: { id: 's', session: { id: 's', header: { id: 's' } } } });
    assert.match(reply.text, /^NeMo Relay: degraded/);
    await host.dispose();
    assert.deepEqual(host.unhandled, []);
  });

  it('stays out of the host when its configuration is invalid', async () => {
    const warnings: string[] = [];
    const listeners: string[] = [];
    const ctx = {
      on: (name: string) => (listeners.push(name), () => undefined),
      effect: () => () => undefined,
      inject: () => undefined,
      get: () => undefined,
      logger: { warn: (message: string) => void warnings.push(message) },
    };
    assert.doesNotThrow(() => apply(ctx, { backend: 'hooks' }));
    assert.deepEqual(listeners, []);
    assert.match(warnings[0] ?? '', /invalid plugin configuration: config\.backend is not supported/);
  });

  it('loads the native binding only through the dynamic module loader', async () => {
    const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
    const files = [
      path.join(root, 'index.js'),
      ...(await readdir(path.join(root, 'src'))).map((file) => path.join(root, 'src', file)),
    ];
    for (const file of files.filter((candidate) => candidate.endsWith('.js'))) {
      const source = await readFile(file, 'utf8');
      assert.doesNotMatch(
        source,
        /^\s*import\s[^(]*['"]nemo-relay-node/m,
        `${path.basename(file)} statically imports the binding`,
      );
    }
  });
});

describe('model call projection', () => {
  it('projects a provider-neutral history into OpenAI Chat messages', () => {
    const projection = projectRequest({
      provider: 'deepseek',
      model: 'deepseek-chat',
      system: 'Be brief.',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'List files' }] },
        {
          id: 'a1',
          role: 'assistant',
          source: { kind: 'model' },
          content: [
            { type: 'reasoning', text: 'thinking' },
            { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls"}' },
          ],
        },
        {
          id: 't1',
          role: 'tool',
          toolCallId: 'c1',
          source: { kind: 'tool' },
          content: [{ type: 'text', text: 'a.txt' }],
        },
      ],
      maxTokens: 512,
      reasoningEffort: 'high',
    });
    assert.deepEqual(projection.content, {
      model: 'deepseek-chat',
      stream: true,
      max_tokens: 512,
      reasoning_effort: 'high',
      messages: [
        { role: 'system', content: 'Be brief.' },
        { role: 'user', content: [{ type: 'text', text: 'List files' }] },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'c1', type: 'function', function: { name: 'bash', arguments: '{"command":"ls"}' } }],
        },
        { role: 'tool', tool_call_id: 'c1', content: [{ type: 'text', text: 'a.txt' }] },
      ],
    });
  });

  it('merges consecutive user messages, so the prompt survives Relay keeping only the current user turn', () => {
    const text = (value: string) => ({ type: 'text' as const, text: value });
    const user = (id: string, kind: string, ...parts: string[]) => ({
      id,
      role: 'user' as const,
      source: { kind },
      content: parts.map(text),
    });
    const projection = projectRequest({
      provider: 'deepseek',
      model: 'deepseek-chat',
      system: 'Be brief.',
      // DeepSeek Harness's request layout: the prompt, then runtime context and skill reminders as more user messages.
      messages: [
        user('u1', 'user', 'Fix the bug'),
        user('u2', 'runtime-context', '<runtime-context>'),
        user('u3', 'skills', '<system-reminder>', 'skills list'),
        {
          id: 'a1',
          role: 'assistant',
          source: { kind: 'model' },
          content: [{ type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls"}' }],
        },
        { id: 't1', role: 'tool', toolCallId: 'c1', source: { kind: 'tool' }, content: [text('a.txt')] },
      ],
    });
    const messages = projection.content.messages as Array<{ role: string; content: unknown }>;
    assert.deepEqual(
      messages.map((message) => message.role),
      ['system', 'user', 'assistant', 'tool'],
    );
    assert.deepEqual(messages[1]?.content, [
      text('Fix the bug'),
      text('<runtime-context>'),
      text('<system-reminder>'),
      text('skills list'),
    ]);
    // Relay records a later call from the last user message on (`retain_current_request_turn`).
    const lastUser = messages.map((message) => message.role).lastIndexOf('user');
    assert.ok(JSON.stringify(messages.slice(lastUser)).includes('Fix the bug'));

    // Edits are still classified against exactly what Relay received.
    assert.deepEqual(classifyRequestEdits(projection, structuredClone(projection.content)), { dropped: [] });
    const edited = structuredClone(projection.content) as { messages: Array<{ content: Array<{ text: string }> }> };
    const merged = edited.messages[1]?.content[2];
    assert.ok(merged);
    merged.text = '<system-reminder> (redacted)';
    assert.throws(() => classifyRequestEdits(projection, edited), UnsupportedRewriteError);
  });

  it('separates model-visible request edits from advisory fields', () => {
    const projection = projectRequest({ provider: 'p', model: 'm', messages: [] });
    assert.deepEqual(classifyRequestEdits(projection, structuredClone(projection.content)), { dropped: [] });
    assert.deepEqual(classifyRequestEdits(projection, { ...projection.content, nvext: { agent_hints: {} } }), {
      dropped: ['nvext'],
    });
    assert.throws(
      () => classifyRequestEdits(projection, { ...projection.content, model: 'other' }),
      UnsupportedRewriteError,
    );
  });

  it('counts cached and cache-write tokens as prompt tokens', () => {
    assert.deepEqual(
      projectUsage({ inputTokens: 10, outputTokens: 5, cacheReadTokens: 30, cacheWriteTokens: 2, reasoningTokens: 3 }),
      {
        prompt_tokens: 42,
        completion_tokens: 5,
        total_tokens: 47,
        prompt_tokens_details: { cached_tokens: 30 },
        completion_tokens_details: { reasoning_tokens: 3 },
      },
    );
  });

  it('keeps tool-call blocks consistent when stream intercepts drop or rewrite deltas', () => {
    const bridge = new ChunkBridge();
    const host: StreamChunk[] = [
      { type: 'block-start', index: 0, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 0, id: 'c1', name: 'bash', argumentsDelta: '{"command":' },
      { type: 'tool-call-delta', index: 0, id: 'c1', argumentsDelta: '"rm -rf /tmp/x"}' },
      {
        type: 'block-end',
        index: 0,
        block: { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"rm -rf /tmp/x"}' },
      },
      { type: 'finish', reason: { kind: 'tool-calls' } },
    ];
    const projections = host.map((chunk) => bridge.accept(chunk));
    const second = structuredClone(projections[2]) as Record<string, any>;
    second.choices[0].delta.tool_calls[0].function.arguments = '"ls /tmp/x"}';
    const released = [
      ...bridge.release(projections[1]),
      ...bridge.release(second),
      ...bridge.release(projections[4]),
      ...bridge.drain('completed'),
    ];
    assert.deepEqual(released.at(-2), {
      type: 'block-end',
      index: 0,
      block: { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls /tmp/x"}' },
    });
    assert.equal(bridge.terminalDelivered, true);
  });
});

describe('view trace writer', () => {
  it('routes events by session id and propagation root and skips chunk receipts', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-view-'));
    const writer = new SessionTraceWriter(directory);
    writer.accept({
      kind: 'scope',
      name: 'turn',
      propagation_root_uuid: 'root-1',
      metadata: { session_id: 'session/1' },
    });
    writer.accept({ kind: 'scope', name: 'guardrail', propagation_root_uuid: 'root-1', metadata: null });
    writer.accept({ kind: 'mark', name: 'llm.chunk', propagation_root_uuid: 'root-1', metadata: null });
    writer.accept({ kind: 'mark', name: 'unrelated', propagation_root_uuid: 'root-2', metadata: null });
    await writer.flush();
    const file = writer.pathFor('session/1');
    assert.equal(path.basename(file), 'session_1.atof.jsonl');
    const names = (await readFile(file, 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line).name);
    assert.deepEqual(names, ['turn', 'guardrail']);
    assert.equal((await stat(file)).mode & 0o777, 0o600);
  });

  /** An LLM start event shaped like Relay's: the request in `data.content` and again in the codec annotation. */
  function llmStart(messages: unknown[], toolCount = 3): Record<string, unknown> {
    const tools = Array.from({ length: toolCount }, (_, index) => ({
      type: 'function',
      function: { name: `tool_${index}`, description: 'x'.repeat(2000), parameters: { type: 'object' } },
    }));
    return {
      kind: 'scope',
      category: 'llm',
      scope_category: 'start',
      name: 'deepseek',
      uuid: 'llm-1',
      metadata: { session_id: 's', step_index: 2 },
      category_profile: { model_name: 'deepseek-chat', annotated_request: { messages, tools, model: 'deepseek-chat' } },
      data: { headers: {}, content: { model: 'deepseek-chat', stream: true, messages, tools } },
    };
  }

  it('keeps what the view shows of an LLM request and records the sizes of what it drops', () => {
    const big = 'r'.repeat(VIEW_TOOL_RESULT_CHARS + 500);
    const messages = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Fix the bug' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1' }] },
      { role: 'tool', tool_call_id: 'c1', content: 'first result' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c2' }, { id: 'c3' }] },
      { role: 'tool', tool_call_id: 'c2', content: big },
      { role: 'tool', tool_call_id: 'c3', content: [{ type: 'text', text: big }, { type: 'image_url' }] },
    ];
    const event = llmStart(messages);
    const before = JSON.stringify(event);
    const compact = compactForView(event) as any;

    // The latest user message, then the tool results this call reacts to, in order.
    assert.deepEqual(
      compact.data.content.messages.map((message: any) => [message.role, message.tool_call_id]),
      [
        ['user', undefined],
        ['tool', 'c2'],
        ['tool', 'c3'],
      ],
    );
    assert.equal(compact.data.content.messages[0].content, 'Fix the bug');
    assert.equal(compact.data.content.messages[1].content.length, VIEW_TOOL_RESULT_CHARS + 1);
    assert.equal(compact.data.content.messages[2].content[0].text.length, VIEW_TOOL_RESULT_CHARS + 1);
    assert.deepEqual(compact.data.content.messages[2].content[1], { type: 'image_url' });
    // Request parameters stay; the tool list and the duplicate annotation go, with their sizes recorded.
    assert.equal(compact.data.content.model, 'deepseek-chat');
    assert.equal(compact.data.content.stream, true);
    assert.equal('tools' in compact.data.content, false);
    assert.deepEqual(compact.category_profile, { model_name: 'deepseek-chat' });
    assert.deepEqual(compact.metadata, {
      session_id: 's',
      step_index: 2,
      [VIEW_REQUEST_SUMMARY_KEY]: { request_messages: 7, request_tools: 3 },
    });
    assert.equal(compact.uuid, 'llm-1');
    // The subscriber's event is not modified.
    assert.equal(JSON.stringify(event), before);
  });

  it('keeps a user message that follows the last assistant message, and leaves other events alone', () => {
    const steering = compactForView(
      llmStart([
        { role: 'user', content: 'Start' },
        { role: 'assistant', content: 'Working' },
        { role: 'tool', tool_call_id: 'c1', content: 'done' },
        { role: 'user', content: 'Also check the tests' },
      ]),
    ) as any;
    assert.deepEqual(
      steering.data.content.messages.map((message: any) => message.content),
      ['done', 'Also check the tests'],
    );
    const firstCall = compactForView(
      llmStart([
        { role: 'system', content: 's' },
        { role: 'user', content: 'Hi' },
      ]),
    ) as any;
    assert.deepEqual(firstCall.data.content.messages, [{ role: 'user', content: 'Hi' }]);

    for (const untouched of [
      { kind: 'scope', category: 'llm', scope_category: 'end', data: { choices: [] } },
      { kind: 'scope', category: 'tool', scope_category: 'start', data: { command: 'ls' } },
      { kind: 'scope', category: 'llm', scope_category: 'start', data: null },
      { kind: 'scope', category: 'llm', scope_category: 'start', data: { content: { prompt: 'no messages' } } },
      'not an event',
      null,
    ]) {
      assert.equal(compactForView(untouched), untouched);
    }
  });

  it('writes compacted LLM requests, so a long turn does not repeat its history in every call', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-view-'));
    const writer = new SessionTraceWriter(directory);
    const history: unknown[] = [{ role: 'user', content: 'Go' }];
    for (let step = 0; step < 40; step += 1) {
      writer.accept(llmStart([...history], 50));
      history.push({ role: 'assistant', content: null, tool_calls: [{ id: `c${step}` }] });
      history.push({ role: 'tool', tool_call_id: `c${step}`, content: 'x'.repeat(3000) });
    }
    await writer.flush();
    const lines = (await readFile(writer.pathFor('s'), 'utf8')).trim().split('\n');
    assert.equal(lines.length, 40);
    // Uncompacted, the last call alone repeats 50 tool schemas twice and 79 messages twice (over 400 KB).
    assert.ok(Math.max(...lines.map((line) => line.length)) < 8000, 'each call stays small');
    assert.deepEqual(JSON.parse(lines[39] ?? '{}').metadata[VIEW_REQUEST_SUMMARY_KEY], {
      request_messages: 79,
      request_tools: 50,
    });
  });
});

/**
 * Apply the plugin to a host that offers the `tools` and `skills` services, with
 * `defineTool` replaced by an identity function so tests can call the tool.
 */
async function applyWithAgentSurface(rawConfig: Record<string, unknown>) {
  const { modules } = createFakeModules();
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => void unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  const disposers: Array<() => unknown> = [];
  const listeners = new Map<string, (...args: any[]) => any>();
  const injected: string[] = [];
  const tools: any[] = [];
  const skillProviders: Array<() => any> = [];
  const unregistered: string[] = [];
  let command: any;
  const scope = {
    commands: { register: (definition: unknown) => ((command = definition), () => undefined) },
    sessionProjections: { register: () => undefined },
    tools: {
      register: (definition: unknown) => (tools.push(definition), () => void unregistered.push('tool')),
    },
    skills: {
      registerProvider: (create: () => unknown) => (skillProviders.push(create), () => void unregistered.push('skill')),
    },
    effect: (execute: () => () => unknown) => disposers.push(execute()),
  };
  const ctx = {
    on: (name: string, listener: (...args: any[]) => any) => (listeners.set(name, listener), () => undefined),
    effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
    inject: (names: string[], callback: (child: any) => void) => (injected.push(...names), callback(scope)),
    get: () => undefined,
    logger: { info: () => undefined, warn: () => undefined },
  };
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-agent-'));
  apply(
    ctx,
    { dataDir, ...rawConfig },
    {
      moduleLoader: async () => modules,
      importTools: async () => ({ defineTool: (definition: unknown) => definition }),
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  return {
    command,
    listeners,
    injected,
    tools,
    skillProviders,
    unregistered,
    unhandled,
    async dispose() {
      for (const dispose of disposers.reverse()) {
        await dispose();
      }
      await new Promise((resolve) => setTimeout(resolve, 10));
      process.off('unhandledRejection', onUnhandled);
    },
  };
}

/**
 * Apply the plugin to a minimal host and record unhandled rejections while it runs.
 * DeepSeek Harness exits its Host on an unhandled rejection, so none may escape.
 */
async function applyWithHost(logger: unknown, modules: NemoRelayModules, existingDataDir?: string) {
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => void unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  const disposers: Array<() => unknown> = [];
  const listeners = new Map<string, (...args: any[]) => any>();
  let command: any;
  const scope = {
    commands: { register: (definition: unknown) => ((command = definition), () => undefined) },
    sessionProjections: { register: () => undefined },
    effect: (execute: () => () => unknown) => disposers.push(execute()),
  };
  const ctx = {
    on: (name: string, listener: (...args: any[]) => any) => (listeners.set(name, listener), () => undefined),
    effect: (execute: () => () => unknown) => (disposers.push(execute()), () => undefined),
    inject: (_names: string[], callback: (child: any) => void) => callback(scope),
    get: () => undefined,
    logger,
  } as unknown as Parameters<typeof apply>[0];
  const dataDir = existingDataDir ?? (await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-host-')));
  apply(ctx, { dataDir }, { moduleLoader: async () => modules });
  await new Promise((resolve) => setTimeout(resolve, 20));
  return {
    unhandled,
    command,
    listeners,
    dataDir,
    async dispose() {
      for (const dispose of disposers.reverse()) {
        await dispose();
      }
      await new Promise((resolve) => setTimeout(resolve, 10));
      process.off('unhandledRejection', onUnhandled);
    },
  };
}
