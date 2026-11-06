// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `/relay` command contract: subcommand parsing, plain-language replies
 * (JSON only for `/relay json`), and each action's structured result in the
 * status snapshot, published before the reply.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { runRelayCommand, type PhoenixControl, type RelayCommandContext } from '../src/commands.js';
import { parseConfig } from '../src/config.js';
import { createHealthSnapshot } from '../src/health.js';
import { GuardrailPolicy, type RulesStore } from '../src/policy.js';
import {
  ACTION_LOG_SIZE,
  ActionLog,
  buildStatusSnapshot,
  type PhoenixSnapshot,
  type RelayActionRecord,
  type StatusSnapshot,
} from '../src/status.js';

const RULE_ID = 'no-recursive-delete-of-root-or-home';

function memoryStore(): RulesStore {
  let rules: readonly unknown[] | undefined;
  let failMode: 'open' | 'closed' | undefined;
  return {
    kind: 'config',
    location: 'plugin configuration (guardrails.rules)',
    read: () => rules,
    write: async (next) => void (rules = next === undefined ? undefined : [...next]),
    readFailMode: () => failMode,
    writeFailMode: async (next) => void (failMode = next),
  };
}

function setup(phoenix?: PhoenixControl) {
  const config = parseConfig({ dataDir: '/data/relay' });
  const policy = new GuardrailPolicy({ enabled: true, failMode: 'open', store: memoryStore() });
  policy.editable = true;
  const actions = new ActionLog();
  /** Every status the command published, as the status file would hold it. */
  const published: StatusSnapshot[] = [];
  const offline: PhoenixSnapshot = {
    enabled: true,
    state: 'stopped',
    connected: false,
    url: config.phoenix.url,
    project: 'p',
  };
  const snapshot = () =>
    buildStatusSnapshot({
      health: createHealthSnapshot({
        status: { state: 'ready' },
        initializedPluginHost: true,
        pluginHostOutputsHealthy: true,
        config,
      }),
      config,
      plan: undefined,
      resolved: { components: [] },
      relayVersion: '1.0.0',
      guardrails: policy.snapshot(),
      phoenix: phoenix?.snapshot() ?? offline,
      viewDirectory: '/data/relay/view',
      actions: actions.list(),
    });
  const context: RelayCommandContext = {
    snapshot,
    publish: () => void published.push(structuredClone(snapshot())),
    recordAction: (entry) => actions.record(entry),
    policy,
    phoenix,
    sessionId: 'session/1',
    sessionTrace: '/data/relay/view/session_1.atof.jsonl',
  };
  const run = (input: string) => runRelayCommand(input, context);
  /** Run an action and return its reply and the result the status file held when the reply came. */
  const act = async (input: string) => {
    const reply = await run(input);
    assert.equal(parseJson(reply.text), undefined, `"${input}" replies in words, not JSON: ${reply.text}`);
    const record = published.at(-1)?.actions.at(-1);
    assert.ok(record, `"${input}" published its result before replying`);
    return { reply, record };
  };
  return { run, act, policy, published, actions, context };
}

function parseJson(text: string): unknown {
  try {
    const value = JSON.parse(text) as unknown;
    return value !== null && typeof value === 'object' ? value : undefined;
  } catch {
    return undefined;
  }
}

/** The record without its timestamp, which every record carries. */
function stable(record: RelayActionRecord): Omit<RelayActionRecord, 'at'> {
  assert.match(record.at, /^\d{4}-\d{2}-\d{2}T/);
  const { at: _at, ...rest } = record;
  return rest;
}

describe('/relay status and json', () => {
  it('replies with text for status and the v4 snapshot for json, publishing first', async () => {
    const { run, published } = setup();
    for (const input of ['', 'status', '  STATUS  ']) {
      const reply = await run(input);
      assert.equal(reply.kind, 'success');
      assert.match(reply.text, /^NeMo Relay: ready — nemo-relay-node 1\.0\.0/);
      assert.match(reply.text, /Rules: 1 of 1 on, fail-open, shipped defaults/);
      assert.match(reply.text, /Session trace \(ATOF\): \/data\/relay\/view\/session_1\.atof\.jsonl/);
    }
    const json = await run('json');
    assert.equal(json.kind, 'success');
    assert.ok(!json.text.includes('\n'), 'one line');
    const snapshot = JSON.parse(json.text) as StatusSnapshot;
    assert.equal(snapshot.v, 4);
    assert.equal(snapshot.id, 'nemo-relay');
    assert.deepEqual(snapshot.actions, []);
    assert.deepEqual(Object.keys(snapshot.guardrails).sort(), [
      'editable',
      'enabled',
      'failMode',
      'location',
      'rejections',
      'revision',
      'rules',
      'source',
    ]);
    assert.equal(published.length, 4);
  });

  it('answers unknown subcommands with usage instead of guessing', async () => {
    const reply = await setup().run('frobnicate now');
    assert.equal(reply.kind, 'error');
    assert.match(reply.text, /Unknown \/relay subcommand "frobnicate"/);
    assert.match(reply.text, /\/relay rules test/);
  });
});

describe('/relay rules', () => {
  it('lists the rules with the revision an edit must name', async () => {
    const { run, policy } = setup();
    const reply = await run('rules');
    assert.ok(reply.text.includes(`1. ${RULE_ID} [on] tool /^(?:bash|pwsh)$/i`), reply.text);
    assert.ok(reply.text.includes(`/relay rules save {"base":"${policy.snapshot().revision}"`));
  });

  it('saves and resets in words, and leaves the result for the request id in the status file', async () => {
    const { act, policy } = setup();
    const base = policy.snapshot().revision;
    const rule = { id: 'no-e2e', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK', reason: 'E2E.' };

    const saved = await act(`rules save ${JSON.stringify({ base, rules: [rule], rid: 'view-1' })}`);
    const revision = policy.snapshot().revision;
    assert.deepEqual(saved.reply, { kind: 'success', text: `Saved 1 NeMo Relay rule (revision ${revision}).` });
    assert.deepEqual(stable(saved.record), { rid: 'view-1', action: 'rules.save', ok: true, revision });

    const two = await act(
      `rules save ${JSON.stringify({ base: revision, rules: [rule, { ...rule, id: 'other' }], rid: 'view-2' })}`,
    );
    assert.equal(two.reply.text, `Saved 2 NeMo Relay rules (revision ${policy.snapshot().revision}).`);

    const reset = await act(`rules reset ${JSON.stringify({ base: policy.snapshot().revision, rid: 'view-3' })}`);
    assert.deepEqual(reset.reply, {
      kind: 'success',
      text: `Restored the default NeMo Relay rules (revision ${base}).`,
    });
    assert.deepEqual(stable(reset.record), { rid: 'view-3', action: 'rules.reset', ok: true, revision: base });
  });

  it('refuses stale, malformed, and invalid edits as errors, with the details in the status file', async () => {
    const { act, policy } = setup();
    const base = policy.snapshot().revision;
    await act(`rules save ${JSON.stringify({ base, rules: [] })}`);

    const stale = await act(`rules save ${JSON.stringify({ base, rules: [], rid: 's1' })}`);
    assert.deepEqual(stale.reply, { kind: 'error', text: 'Rules changed elsewhere; reload and try again.' });
    assert.deepEqual([stale.record.rid, stale.record.ok, stale.record.code], ['s1', false, 'stale']);

    for (const payload of ['', '{not json', '[1,2]', '"text"']) {
      const malformed = await act(`rules save ${payload}`);
      assert.equal(malformed.reply.kind, 'error', payload);
      assert.deepEqual(
        [malformed.record.action, malformed.record.ok, malformed.record.code],
        ['rules.save', false, 'invalid'],
      );
      assert.equal(malformed.record.rid, undefined);
    }
    const badRule = await act(
      `rules save ${JSON.stringify({ base: policy.snapshot().revision, rules: [{ id: 'x', match: '(' }], rid: 'bad' })}`,
    );
    assert.equal(badRule.reply.kind, 'error');
    assert.match(badRule.reply.text, /^The rules were not saved\.\nRule 1 \(x\): The pattern is not a valid/);
    assert.equal(badRule.record.errors?.[0]?.field, 'match', 'the view can mark the field');

    for (const rid of [7, '', 'x'.repeat(65), 'has space']) {
      const badId = await act(`rules test ${JSON.stringify({ tool: 'bash', arguments: {}, rid })}`);
      assert.deepEqual([badId.reply.kind, badId.record.code, badId.record.rid], ['error', 'invalid', undefined]);
      assert.match(badId.reply.text, /request id \(rid\)/);
    }
    assert.equal((await act('rules reset')).record.code, 'invalid');
    assert.equal((await act('rules rename {}')).record.code, 'invalid');
  });

  it('dry-runs a call in words without changing the rules', async () => {
    const { act, policy } = setup();
    const before = policy.snapshot();
    const rm = ['rm', '-rf'].join(' ');
    const blocked = await act(
      `rules test ${JSON.stringify({ tool: 'bash', arguments: { command: `${rm} ~` }, rid: 't1' })}`,
    );
    assert.deepEqual(blocked.reply, { kind: 'success', text: `NeMo Relay would block this call: rule "${RULE_ID}".` });
    assert.deepEqual(stable(blocked.record), {
      rid: 't1',
      action: 'rules.test',
      ok: true,
      verdict: 'block',
      ruleId: RULE_ID,
      reason: `${before.rules[0]?.reason} (NeMo Relay rule "${RULE_ID}")`,
    });
    const allowed = await act(
      `rules test ${JSON.stringify({ tool: 'bash', arguments: { command: 'ls' }, rid: 't2' })}`,
    );
    assert.deepEqual(allowed.reply, { kind: 'success', text: 'NeMo Relay would allow this call.' });
    assert.deepEqual(stable(allowed.record), { rid: 't2', action: 'rules.test', ok: true, verdict: 'allow' });
    assert.equal((await act('rules test {"arguments":{}}')).record.code, 'invalid');
    assert.deepEqual(policy.snapshot(), before, 'no rejection is counted and the revision is unchanged');
  });

  it('records a dry run that fails as an error for its request id', async () => {
    const actions = new ActionLog();
    const policy = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: memoryStore(),
      tester: () => {
        throw new Error('the tester broke');
      },
    });
    const reply = await runRelayCommand('rules test {"tool":"bash","arguments":{"command":"ls"},"rid":"t9"}', {
      snapshot: () => {
        throw new Error('unused');
      },
      publish: () => undefined,
      recordAction: (entry) => actions.record(entry),
      policy,
      phoenix: undefined,
      sessionId: undefined,
      sessionTrace: undefined,
    });
    assert.deepEqual(reply, { kind: 'error', text: 'NeMo Relay could not check this call: the tester broke' });
    assert.deepEqual(actions.list().map(stable), [
      {
        rid: 't9',
        action: 'rules.test',
        ok: false,
        code: 'error',
        error: 'NeMo Relay could not check this call: the tester broke',
      },
    ]);
  });

  it('changes the fail mode from a word or a JSON request with an id', async () => {
    const { act, run, policy } = setup();
    const closed = await act('rules failmode CLOSED');
    assert.deepEqual(closed.reply, { kind: 'success', text: 'Fail mode is now closed.' });
    assert.deepEqual(stable(closed.record), { action: 'rules.failmode', ok: true, failMode: 'closed' });
    assert.equal(policy.failMode, 'closed');
    assert.match((await run('rules')).text, /fail-closed[\s\S]*change it with \/relay rules failmode open/);
    assert.equal(JSON.parse((await run('json')).text).guardrails.failMode, 'closed', 'the status reports it');

    const open = await act(`rules failmode ${JSON.stringify({ mode: 'open', rid: 'f1' })}`);
    assert.deepEqual(open.reply, { kind: 'success', text: 'Fail mode is now open.' });
    assert.deepEqual(stable(open.record), { rid: 'f1', action: 'rules.failmode', ok: true, failMode: 'open' });

    // The word form takes the request id in a trailing object, as the usage text says.
    const worded = await act('rules failmode closed {"rid":"f3"}');
    assert.deepEqual(worded.reply, { kind: 'success', text: 'Fail mode is now closed.' });
    assert.deepEqual(stable(worded.record), { rid: 'f3', action: 'rules.failmode', ok: true, failMode: 'closed' });
    assert.deepEqual(stable((await act('rules failmode open {"rid":"f4"}')).record), {
      rid: 'f4',
      action: 'rules.failmode',
      ok: true,
      failMode: 'open',
    });
    assert.equal((await act('rules failmode closed {"rid":"has space"}')).record.code, 'invalid');
    assert.equal((await act('rules failmode closed now')).record.code, 'invalid');

    for (const input of ['rules failmode', 'rules failmode sometimes', 'rules failmode {"rid":"f2"}']) {
      const refused = await act(input);
      assert.deepEqual(
        [refused.reply.kind, refused.record.ok, refused.record.code],
        ['error', false, 'invalid'],
        input,
      );
    }
    assert.equal(policy.failMode, 'open');
  });

  it(`keeps the last ${ACTION_LOG_SIZE} results, oldest first`, async () => {
    const { act, actions } = setup();
    for (let index = 0; index < ACTION_LOG_SIZE + 4; index += 1) {
      await act(`rules test ${JSON.stringify({ tool: 'bash', arguments: { command: 'ls' }, rid: `r${index}` })}`);
    }
    const rids = actions.list().map((record) => record.rid);
    assert.equal(rids.length, ACTION_LOG_SIZE);
    assert.deepEqual([rids[0], rids.at(-1)], ['r4', `r${ACTION_LOG_SIZE + 3}`]);
  });
});

describe('/relay phoenix', () => {
  it('runs Phoenix actions, replies in words, and records the Phoenix state', async () => {
    const calls: string[] = [];
    let state: PhoenixSnapshot = {
      enabled: true,
      state: 'stopped',
      connected: false,
      url: 'http://127.0.0.1:16006',
      project: 'p',
    };
    const control: PhoenixControl = {
      snapshot: () => state,
      start: async () => {
        calls.push('start');
        state = {
          ...state,
          state: 'running',
          connected: true,
          mode: 'owned',
          sessionUrl: 'http://127.0.0.1:16006/redirects/sessions/{sessionId}',
        };
        return { ok: true, phoenix: state };
      },
      stop: async () => {
        calls.push('stop');
        state = { enabled: true, state: 'stopped', connected: false, url: state.url, project: 'p' };
        return { ok: true, phoenix: state };
      },
      connect: async () => (
        calls.push('connect'),
        { ok: false, error: 'No Phoenix answers at http://127.0.0.1:16006', phoenix: state }
      ),
      disconnect: async () => (calls.push('disconnect'), { ok: true, phoenix: state }),
    };
    const { act, run } = setup(control);
    assert.deepEqual(JSON.parse((await run('json')).text).exporters, []);

    const started = await act('phoenix start {"rid":"p1"}');
    assert.deepEqual(started.reply, {
      kind: 'success',
      text: 'Phoenix is running at http://127.0.0.1:16006 and receiving traces.',
    });
    assert.deepEqual(
      [started.record.rid, started.record.action, started.record.phoenix?.state],
      ['p1', 'phoenix.start', 'running'],
    );
    // The connected Phoenix is an exporter of its own, outside the plugin-host document.
    const status = JSON.parse((await run('json')).text);
    assert.deepEqual(status.exporters, [{ kind: 'openinference', target: 'http://127.0.0.1:16006/v1/traces' }]);
    assert.equal(status.outputs.openInference, 'enabled');

    const refused = await act('phoenix connect');
    assert.deepEqual(refused.reply, { kind: 'error', text: 'Phoenix: No Phoenix answers at http://127.0.0.1:16006' });
    assert.deepEqual(
      [refused.record.ok, refused.record.error],
      [false, 'No Phoenix answers at http://127.0.0.1:16006'],
    );

    assert.equal((await act('phoenix stop')).reply.text, 'Stopped Phoenix at http://127.0.0.1:16006.');
    assert.equal(
      (await act('phoenix disconnect')).reply.text,
      'Stopped sending traces to Phoenix at http://127.0.0.1:16006.',
    );
    assert.deepEqual(calls, ['start', 'connect', 'stop', 'disconnect']);

    const text = await run('phoenix');
    assert.match(text.text, /Phoenix: not connected/);
    const unknown = await act('phoenix dance');
    assert.deepEqual([unknown.reply.kind, unknown.record.code], ['error', 'invalid']);
    assert.equal((await act('phoenix start not-json')).record.code, 'invalid');
  });

  it('says that a Phoenix someone else runs keeps running after stop', async () => {
    const external: PhoenixSnapshot = {
      enabled: true,
      state: 'running',
      connected: true,
      mode: 'external',
      url: 'http://127.0.0.1:6006',
      project: 'p',
    };
    const control: PhoenixControl = {
      snapshot: () => external,
      start: async () => ({ ok: true, phoenix: external }),
      stop: async () => ({ ok: true, phoenix: { ...external, state: 'stopped', connected: false } }),
      connect: async () => ({ ok: true, phoenix: external }),
      disconnect: async () => ({ ok: true, phoenix: external }),
    };
    const { act } = setup(control);
    assert.equal(
      (await act('phoenix stop')).reply.text,
      'Stopped sending traces to Phoenix at http://127.0.0.1:6006; it keeps running.',
    );
  });

  it('reports Phoenix as unavailable when the configuration turns it off', async () => {
    const reply = await setup().act('phoenix start {"rid":"p9"}');
    assert.equal(reply.reply.kind, 'error');
    assert.deepEqual([reply.record.rid, reply.record.code], ['p9', 'unavailable']);
  });
});

describe('command robustness', () => {
  it('never rejects, even when building the status fails', async () => {
    const { policy } = setup();
    const reply = await runRelayCommand('status', {
      snapshot: () => {
        throw new Error('snapshot failed');
      },
      publish: () => undefined,
      policy,
      phoenix: undefined,
      sessionId: undefined,
      sessionTrace: undefined,
    });
    assert.deepEqual(reply, { kind: 'error', text: 'NeMo Relay could not run the command: snapshot failed' });
  });
});
