// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `nemo_relay` agent tool: each action reaches the same rules and Phoenix
 * operations as `/relay`, loosening and Phoenix actions are gated behind the
 * user's approval, and registration never fails DeepSeek Harness.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  AGENT_ACTIONS,
  AGENT_TOOL_NAME,
  agentToolPreExecute,
  classifyAgentAction,
  createAgentTool,
  defineToolLocally,
  registerAgentTool,
  type AgentToolArgs,
  type AgentToolDefinitionOptions,
  type AgentToolHost,
  type AgentToolOperations,
  type AgentToolResult,
  type FailModeChangeResult,
} from '../src/agent-tool.js';
import type { PhoenixControl } from '../src/commands.js';
import { parseConfig, type FailMode } from '../src/config.js';
import type { PreToolDecision } from '../src/dsh-types.js';
import { createHealthSnapshot } from '../src/health.js';
import { DEFAULT_RULES, GuardrailPolicy, type RulesStore } from '../src/policy.js';
import { buildStatusSnapshot, type PhoenixSnapshot } from '../src/status.js';

const DEFAULT_ID = 'no-recursive-delete-of-root-or-home';
/** Built at runtime so no command line or source text carries the literal. */
const RECURSIVE_DELETE = ['rm', '-rf'].join(' ');
/** Whether this checkout can resolve DeepSeek Harness's tools package (it normally cannot). */
const DSH_TOOLS: string = '@deepseek-ai/dsh-tools';
const dshToolsInstalled = await import(DSH_TOOLS).then(
  () => true,
  () => false,
);

function memoryStore(): RulesStore {
  let rules: readonly unknown[] | undefined;
  return {
    kind: 'config',
    location: 'plugin configuration (guardrails.rules)',
    read: () => rules,
    write: async (next) => void (rules = next === undefined ? undefined : [...next]),
  };
}

/** `defineTool` stand-in: the tool definition is the options object itself. */
const passThrough = (options: AgentToolDefinitionOptions): AgentToolDefinitionOptions => options;

type HarnessOptions = {
  phoenix?: PhoenixControl;
  setFailMode?: (mode: FailMode) => Promise<FailModeChangeResult>;
};

function harness(options: HarnessOptions = {}) {
  const config = parseConfig({ dataDir: '/data/relay' });
  const policy = new GuardrailPolicy({ enabled: true, failMode: 'open', store: memoryStore() });
  let publications = 0;
  const sessions: Array<string | undefined> = [];
  const offline: PhoenixSnapshot = {
    enabled: true,
    state: 'stopped',
    connected: false,
    url: 'http://127.0.0.1:16006',
    project: 'p',
  };
  const operations: AgentToolOperations = {
    context: (sessionId) => {
      sessions.push(sessionId);
      return {
        snapshot: () =>
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
            phoenix: options.phoenix?.snapshot() ?? offline,
            viewDirectory: '/data/relay/view',
          }),
        publish: () => void (publications += 1),
        policy,
        phoenix: options.phoenix,
        sessionId,
        sessionTrace: sessionId === undefined ? undefined : `/data/relay/view/${sessionId}.atof.jsonl`,
      };
    },
    ...(options.setFailMode === undefined ? {} : { setFailMode: options.setFailMode }),
  };
  const tool = createAgentTool(passThrough, operations) as AgentToolDefinitionOptions;
  /** Run one call the way DeepSeek Harness does, from an agent of `sessionId` (or from no agent). */
  const run = (args: Record<string, unknown>, sessionId: string | null = 'session/1'): Promise<AgentToolResult> =>
    tool.execute(args as AgentToolArgs, {
      callId: 'call-1',
      signal: new AbortController().signal,
      ...(sessionId === null ? {} : { agent: { id: 'agent-1', session: { id: sessionId } } }),
    });
  return { tool, run, policy, sessions, publications: () => publications };
}

/** A Phoenix control that records calls and starts in `state`. */
function fakePhoenix(initial: Partial<PhoenixSnapshot> = {}) {
  const calls: string[] = [];
  let state: PhoenixSnapshot = {
    enabled: true,
    state: 'stopped',
    connected: false,
    url: 'http://127.0.0.1:16006',
    project: 'deepseek-harness',
    ...initial,
  };
  const running = (): PhoenixSnapshot => ({
    ...state,
    state: 'running',
    connected: true,
    mode: 'owned',
    sessionUrl: `${state.url}/redirects/sessions/{sessionId}`,
  });
  const control: PhoenixControl = {
    snapshot: () => state,
    start: async () => (calls.push('start'), (state = running()), { ok: true, phoenix: state }),
    connect: async () => (
      calls.push('connect'),
      { ok: false, error: `No Phoenix answers at ${state.url}`, phoenix: state }
    ),
    disconnect: async () => (
      calls.push('disconnect'),
      (state = { ...state, connected: false }),
      { ok: true, phoenix: state }
    ),
    stop: async () => (
      calls.push('stop'),
      (state = { ...state, state: 'stopped', connected: false }),
      { ok: true, phoenix: state }
    ),
  };
  return { control, calls };
}

describe('nemo_relay definition', () => {
  it('is a compact, always-loaded tool whose schema offers every action', () => {
    const { tool } = harness();
    assert.equal(tool.name, AGENT_TOOL_NAME);
    assert.equal('deferLoading' in tool, false, 'deferred tools stay inactive in DeepSeek Harness until re-added');
    const action = tool.parameters.action as { required?: boolean; enum?: unknown[] };
    assert.equal(action.required, true);
    assert.deepEqual(action.enum, [...AGENT_ACTIONS]);
    assert.match(tool.description, /nemo-relay skill/, 'the description points to the skill for details');
    const modelFacing = JSON.stringify({ name: tool.name, description: tool.description, parameters: tool.parameters });
    assert.ok(modelFacing.length < 1800, `the model-facing schema stays compact (${modelFacing.length} bytes)`);
  });

  it('presents reads as read cards and lets only reads run concurrently', () => {
    const { tool } = harness();
    const reads = ['status', 'rules.list', 'rules.test', 'phoenix.status', 'phoenix.link'];
    for (const action of AGENT_ACTIONS) {
      const args = { action, id: 'no-e2e', tool: 'bash', mode: 'closed' } as AgentToolArgs;
      assert.equal(tool.isConcurrencySafe(args), reads.includes(action), action);
      const card = tool.presentCall(args);
      assert.equal(card.kind, reads.includes(action) ? 'read' : 'other', action);
      assert.equal(card.rawInput, args);
    }
    assert.equal(tool.presentCall({ action: 'rules.disable', id: 'no-e2e' }).title, 'Turn off NeMo Relay rule: no-e2e');
    assert.equal(tool.presentCall({ action: 'phoenix.start' }).title, 'Start Phoenix');
  });

  it('renders concise text followed by one line of JSON details', () => {
    const { tool } = harness();
    const args = { action: 'rules.list' } as AgentToolArgs;
    assert.deepEqual(
      tool.output.render(args, { action: 'rules.list', text: 'Rules: 1 of 1 on', data: { revision: 'ab' } }),
      [{ type: 'text', text: 'Rules: 1 of 1 on\n{"revision":"ab"}' }],
    );
    assert.deepEqual(tool.output.render(args, { action: 'status', text: 'NeMo Relay: ready' }), [
      { type: 'text', text: 'NeMo Relay: ready' },
    ]);
  });
});

describe('nemo_relay status and rules', () => {
  it('status is the /relay status of the calling session, published first', async () => {
    const { run, sessions, publications } = harness();
    const result = await run({ action: 'status' });
    assert.match(result.text, /^NeMo Relay: ready — nemo-relay-node 1\.0\.0/);
    assert.match(result.text, /Rules: 1 of 1 on, fail-open, shipped defaults/);
    assert.match(result.text, /Session trace \(ATOF\): \/data\/relay\/view\/session\/1\.atof\.jsonl/);
    assert.deepEqual(sessions, ['session/1']);
    assert.equal(publications(), 1);
  });

  it('rules.list returns the summary, the rules, and the revision an edit is based on', async () => {
    const { run, policy } = harness();
    const result = await run({ action: 'rules.list' });
    assert.match(result.text, /^Rules: 1 of 1 on, fail-open, shipped defaults \(revision [0-9a-f]{8}\)/);
    const data = result.data as { revision: string; failMode: string; rules: Array<{ id: string }> };
    assert.equal(data.revision, policy.snapshot().revision);
    assert.equal(data.failMode, 'open');
    assert.deepEqual(
      data.rules.map((rule) => rule.id),
      [DEFAULT_ID],
    );
  });

  it('rules.test reports the verdict of a dry run without counting it', async () => {
    const { run, policy } = harness();
    const before = policy.snapshot();
    const blocked = await run({ action: 'rules.test', tool: 'bash', arguments: { command: `${RECURSIVE_DELETE} ~` } });
    assert.deepEqual(blocked.data, {
      verdict: 'block',
      ruleId: DEFAULT_ID,
      reason: `${before.rules[0]?.reason} (NeMo Relay rule "${DEFAULT_ID}")`,
    });
    assert.match(blocked.text, /^Blocked by rule "no-recursive-delete-of-root-or-home"/);
    const allowed = await run({ action: 'rules.test', tool: 'bash', arguments: { command: 'ls -la' } });
    assert.deepEqual(allowed.data, { verdict: 'allow' });
    await assert.rejects(run({ action: 'rules.test', arguments: {} }), /rules\.test needs "tool"/);
    assert.deepEqual(policy.snapshot(), before, 'no rejection is counted and the rules are unchanged');
  });

  it('rules.add appends an enabled rule that the next call is checked against', async () => {
    const { run, policy } = harness();
    const result = await run({
      action: 'rules.add',
      id: 'no-e2e',
      tool: 'bash',
      match: 'NEMO_RELAY_E2E_BLOCK',
      reason: 'E2E marker.',
    });
    assert.match(result.text, /^Added rule "no-e2e"\. Rules: 2 of 2 on/);
    const after = policy.snapshot();
    assert.deepEqual(
      after.rules.map((rule) => [rule.id, rule.enabled]),
      [
        [DEFAULT_ID, true],
        ['no-e2e', true],
      ],
    );
    assert.deepEqual(result.data, { revision: after.revision, rule: after.rules[1] });
    assert.equal(
      policy.evaluate('bash', { command: "echo 'NEMO_RELAY_E2E_BLOCK'" }),
      'E2E marker. (NeMo Relay rule "no-e2e")',
    );
    assert.equal(policy.evaluate('read', { path: 'NEMO_RELAY_E2E_BLOCK' }), null, 'the tool selector still applies');
  });

  it('rules.add stores the normalized id and saves nothing for a duplicate or invalid rule', async () => {
    const { run, policy } = harness();
    const added = await run({ action: 'rules.add', id: 'No E2E', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK' });
    assert.equal((added.data as { rule: { id: string } }).rule.id, 'No-E2E');
    const revision = policy.snapshot().revision;
    await assert.rejects(
      run({ action: 'rules.add', id: 'no-e2e', tool: 'bash', match: 'x' }),
      /"No-E2E" already exists/,
    );
    await assert.rejects(
      run({ action: 'rules.add', id: 'bad', tool: 'bash', match: '(' }),
      /not a valid regular expression/,
    );
    await assert.rejects(run({ action: 'rules.add', id: 'slow', match: '(a+)+$' }), /repeats a repeating group/);
    await assert.rejects(run({ action: 'rules.add', id: 'everything' }), /would block every tool call/);
    await assert.rejects(run({ action: 'rules.add', id: 'every', tool: '//' }), /would block every tool call/);
    await assert.rejects(run({ action: 'rules.add', id: 'any', tool: '/./' }), /would block every tool call/);
    await assert.rejects(run({ action: 'rules.add', tool: 'bash', match: 'x' }), /rules\.add needs "id"/);
    assert.equal(policy.snapshot().revision, revision);
  });

  it('rules.add and rules.update refuse a pattern that is slow on ordinary text', async () => {
    const { run, policy } = harness();
    // Passes the nested-repetition check and compiles, yet takes exponential time on any ordinary text.
    const slow = String.raw`^(?:[\s\S]|[^\u0000])*\u0000$`;
    await assert.rejects(
      run({ action: 'rules.add', id: 'audit', tool: '*', match: slow }),
      /Rule 2 \(audit\): The pattern is too slow on ordinary text; simplify it/,
    );
    await run({ action: 'rules.add', id: 'audit', tool: 'bash', match: 'audit' });
    await assert.rejects(
      run({ action: 'rules.update', id: 'audit', match: slow }),
      /Rule 2 \(audit\): The pattern is too slow on ordinary text; simplify it/,
    );
    await assert.rejects(
      run({ action: 'rules.update', id: 'audit', tool: '/^(a|aa)+$/' }),
      /Rule 2 \(audit\): The tool pattern is too slow on ordinary text; simplify it/,
    );
    assert.deepEqual(
      policy.snapshot().rules.map((rule) => [rule.id, rule.tool, rule.match]),
      [
        [DEFAULT_ID, DEFAULT_RULES[0]?.tool, DEFAULT_RULES[0]?.match],
        ['audit', 'bash', 'audit'],
      ],
    );
  });

  it('rules.add reports the id that was stored for an id without a usable character', async () => {
    const { run, policy } = harness();
    const first = await run({ action: 'rules.add', id: '!!!', tool: 'bash', match: 'x' });
    assert.match(first.text, /^Added rule "rule-2"\./);
    assert.equal((first.data as { rule: { id: string; match: string } }).rule.id, 'rule-2');
    await run({ action: 'rules.add', id: 'rule-1', tool: 'bash', match: 'y' });
    const third = await run({ action: 'rules.add', id: '???', tool: 'bash', match: 'z' });
    assert.match(third.text, /^Added rule "rule-4"\./);
    assert.deepEqual((third.data as { rule: { id: string; match: string } }).rule, policy.snapshot().rules[3]);
    assert.deepEqual(
      policy
        .snapshot()
        .rules.map((rule) => [rule.id, rule.match])
        .slice(1),
      [
        ['rule-2', 'x'],
        ['rule-1', 'y'],
        ['rule-4', 'z'],
      ],
    );
  });

  it('rules.update changes only the fields it is given', async () => {
    const { run, policy } = harness();
    const before = policy.snapshot().rules[0];
    const result = await run({ action: 'rules.update', id: DEFAULT_ID.toUpperCase(), reason: 'Ask the user first.' });
    const after = policy.snapshot().rules[0];
    assert.deepEqual(after, { ...before, reason: 'Ask the user first.' }, 'ids match regardless of case');
    assert.deepEqual((result.data as { rule: unknown }).rule, after);
    await assert.rejects(run({ action: 'rules.update', id: DEFAULT_ID }), /needs at least one of/);
    await assert.rejects(
      run({ action: 'rules.update', id: 'missing', match: 'x' }),
      /No rule is named "missing"\. Rule ids: no-recursive-delete-of-root-or-home\./,
    );
  });

  it('rules.disable and rules.enable switch enforcement, and a no-op saves nothing', async () => {
    const { run, policy } = harness();
    const call = { command: `${RECURSIVE_DELETE} ~` };
    await run({ action: 'rules.disable', id: DEFAULT_ID });
    assert.equal(policy.evaluate('bash', call), null);
    const enabled = await run({ action: 'rules.enable', id: DEFAULT_ID });
    assert.match(enabled.text, /^Turned rule "no-recursive-delete-of-root-or-home" on\./);
    assert.match(String(policy.evaluate('bash', call)), /NeMo Relay rule "no-recursive-delete-of-root-or-home"/);
    const revision = policy.snapshot().revision;
    const again = await run({ action: 'rules.enable', id: DEFAULT_ID });
    assert.match(again.text, /is already on/);
    assert.equal(policy.snapshot().revision, revision);
  });

  it('rules.remove deletes a rule and rules.reset restores the shipped rules', async () => {
    const { run, policy } = harness();
    const shipped = policy.snapshot().revision;
    const removed = await run({ action: 'rules.remove', id: DEFAULT_ID });
    assert.deepEqual(policy.snapshot().rules, []);
    assert.equal((removed.data as { removed: { id: string } }).removed.id, DEFAULT_ID);
    assert.equal(policy.evaluate('bash', { command: `${RECURSIVE_DELETE} ~` }), null);
    const reset = await run({ action: 'rules.reset' });
    assert.equal(policy.snapshot().revision, shipped);
    assert.deepEqual(
      (reset.data as { rules: Array<{ id: string }> }).rules.map((rule) => rule.id),
      DEFAULT_RULES.map((rule) => rule.id),
    );
  });

  it('of two concurrent edits one is saved and the other reports the conflict', async () => {
    const { run, policy } = harness();
    const outcomes = await Promise.allSettled([
      run({ action: 'rules.add', id: 'first', tool: 'bash', match: 'one' }),
      run({ action: 'rules.add', id: 'second', tool: 'bash', match: 'two' }),
    ]);
    assert.deepEqual(
      outcomes.map((outcome) => outcome.status),
      ['fulfilled', 'rejected'],
    );
    assert.match(String((outcomes[1] as PromiseRejectedResult).reason), /changed while this edit was being saved/);
    assert.deepEqual(
      policy.snapshot().rules.map((rule) => rule.id),
      [DEFAULT_ID, 'first'],
    );
  });
});

describe('nemo_relay fail mode', () => {
  it('changes the fail mode through the host operation', async () => {
    const requested: FailMode[] = [];
    const { run } = harness({
      setFailMode: async (mode) => (requested.push(mode), { ok: true, failMode: mode }),
    });
    const result = await run({ action: 'guardrails.failMode', mode: 'closed' });
    assert.deepEqual(requested, ['closed']);
    assert.match(result.text, /now fail closed: NeMo Relay blocks a call/);
    assert.deepEqual(result.data, { failMode: 'closed' });
    await assert.rejects(run({ action: 'guardrails.failMode', mode: 'sideways' }), /needs "mode": "open" or "closed"/);
  });

  it('reports a refused change, and says to ask the user when the host cannot change it', async () => {
    const refused = harness({
      setFailMode: async () => ({ ok: false, code: 'unavailable', error: 'The configuration is read-only.' }),
    });
    await assert.rejects(
      refused.run({ action: 'guardrails.failMode', mode: 'closed' }),
      /The configuration is read-only\./,
    );
    const { run } = harness();
    await assert.rejects(
      run({ action: 'guardrails.failMode', mode: 'closed' }),
      /Ask the user to set guardrails\.failMode/,
    );
    const unchanged = await run({ action: 'guardrails.failMode', mode: 'open' });
    assert.match(unchanged.text, /already fail open/, 'the current mode needs no host operation');
  });
});

describe('nemo_relay Phoenix', () => {
  it('start, disconnect, and stop call PhoenixControl and return the Phoenix state', async () => {
    const { control, calls } = fakePhoenix();
    const { run } = harness({ phoenix: control });
    const started = await run({ action: 'phoenix.start' });
    assert.match(started.text, /^Phoenix: http:\/\/127\.0\.0\.1:16006, started by NeMo Relay/);
    assert.match(started.text, /This session: http:\/\/127\.0\.0\.1:16006\/redirects\/sessions\/session%2F1$/);
    const data = started.data as Record<string, unknown>;
    assert.equal(data.sessionLink, 'http://127.0.0.1:16006/redirects/sessions/session%2F1');
    assert.equal('sessionUrl' in data, false, 'the link replaces the template');
    await run({ action: 'phoenix.disconnect' });
    await run({ action: 'phoenix.stop' });
    assert.deepEqual(calls, ['start', 'disconnect', 'stop']);
  });

  it('a failed action is an error carrying Phoenix’s reason', async () => {
    const { control, calls } = fakePhoenix();
    const { run } = harness({ phoenix: control });
    await assert.rejects(run({ action: 'phoenix.connect' }), /No Phoenix answers at http:\/\/127\.0\.0\.1:16006/);
    assert.deepEqual(calls, ['connect']);
  });

  it('phoenix.status and phoenix.link describe this session in Phoenix', async () => {
    const { control } = fakePhoenix();
    const { run } = harness({ phoenix: control });
    await assert.rejects(run({ action: 'phoenix.link' }), /not receiving this session's traces/);
    await run({ action: 'phoenix.start' });
    const status = await run({ action: 'phoenix.status' });
    assert.match(status.text, /This session: http:\/\/127\.0\.0\.1:16006\/redirects\/sessions\/session%2F1/);
    assert.equal((status.data as { state: string }).state, 'running');
    const link = await run({ action: 'phoenix.link' }, 'abc 123');
    assert.deepEqual(link.data, { url: 'http://127.0.0.1:16006/redirects/sessions/abc%20123' });
    await assert.rejects(run({ action: 'phoenix.link' }, null), /does not belong to a session/);
  });

  it('every Phoenix change is unavailable when the configuration turns Phoenix off', async () => {
    const { run } = harness();
    for (const action of ['phoenix.start', 'phoenix.connect', 'phoenix.disconnect', 'phoenix.stop']) {
      await assert.rejects(run({ action }), /phoenix\.enabled: false/, action);
    }
  });
});

describe('classifyAgentAction', () => {
  it('runs reads and tightening directly and asks before loosening or any Phoenix change', () => {
    const table: Array<[Record<string, unknown>, boolean]> = [
      [{ action: 'status' }, false],
      [{ action: 'rules.list' }, false],
      [{ action: 'rules.test', tool: 'bash', arguments: { command: 'ls' } }, false],
      [{ action: 'rules.add', id: 'no-force-push', tool: 'bash', match: 'push' }, false],
      [{ action: 'rules.enable', id: 'no-force-push' }, false],
      [{ action: 'guardrails.failMode', mode: 'closed' }, false],
      [{ action: 'phoenix.status' }, false],
      [{ action: 'phoenix.link' }, false],
      [{ action: 'rules.update', id: 'no-force-push', match: 'x' }, true],
      [{ action: 'rules.disable', id: 'no-force-push' }, true],
      [{ action: 'rules.remove', id: 'no-force-push' }, true],
      [{ action: 'rules.reset' }, true],
      [{ action: 'guardrails.failMode', mode: 'open' }, true],
      [{ action: 'phoenix.start' }, true],
      [{ action: 'phoenix.connect' }, true],
      // Stopping the traces reduces what the user sees of the agent.
      [{ action: 'phoenix.disconnect' }, true],
      [{ action: 'phoenix.stop' }, true],
    ];
    for (const [args, needsApproval] of table) {
      const verdict = classifyAgentAction(args);
      assert.equal(verdict.needsApproval, needsApproval, JSON.stringify(args));
      assert.ok(verdict.reason.length > 0);
      assert.equal(verdict.displayReason !== undefined, needsApproval, 'a prompt text exactly when asking');
      if (verdict.displayReason !== undefined) {
        assert.ok(verdict.displayReason.en.length > 0 && verdict.displayReason.zh.length > 0);
      }
    }
    const covered = new Set(table.map(([args]) => args.action));
    assert.deepEqual(
      AGENT_ACTIONS.filter((action) => !covered.has(action)),
      [],
      'the table covers every action',
    );
  });

  it('asks for malformed or unrecognized arguments', () => {
    const malformed: unknown[] = [
      undefined,
      null,
      'status',
      ['status'],
      {},
      { action: 7 },
      { action: 'rules.purge' },
      { action: 'Status' },
      { action: 'guardrails.failMode' },
      { action: 'guardrails.failMode', mode: 'CLOSED' },
    ];
    for (const args of malformed) {
      assert.equal(classifyAgentAction(args).needsApproval, true, JSON.stringify(args));
    }
  });

  it('names the rule and the changed fields in the prompt, without control characters', () => {
    const update = classifyAgentAction({ action: 'rules.update', id: 'no-force-push', match: 'x', reason: 'y' });
    assert.match(update.displayReason?.en ?? '', /rule "no-force-push" \(match, reason\)/);
    assert.match(update.displayReason?.zh ?? '', /规则“no-force-push”/);
    const hostile = classifyAgentAction({ action: 'rules.remove', id: `x"\n\u0007${'y'.repeat(100)}` });
    assert.match(hostile.displayReason?.en ?? '', /rule "xy{63}"\. /);
    assert.match(classifyAgentAction({ action: 'rules.disable' }).displayReason?.en ?? '', /rule \(no id given\)/);
  });
});

describe('agentToolPreExecute', () => {
  const allow = async (): Promise<PreToolDecision> => ({ kind: 'allow' });
  const exec = (name: string, args: unknown) => ({ name, arguments: args });

  it('asks before a gated nemo_relay call and lets other calls through', async () => {
    assert.deepEqual(
      await agentToolPreExecute(exec(AGENT_TOOL_NAME, { action: 'rules.add', id: 'x', match: 'y' }), allow),
      {
        kind: 'allow',
      },
    );
    const decision = await agentToolPreExecute(
      exec(AGENT_TOOL_NAME, { action: 'rules.remove', id: 'no-force-push' }),
      allow,
    );
    assert.equal(decision.kind, 'ask');
    assert.deepEqual(
      decision.kind === 'ask' ? decision.displayReason : undefined,
      classifyAgentAction({ action: 'rules.remove', id: 'no-force-push' }).displayReason,
    );
    assert.deepEqual(await agentToolPreExecute(exec('bash', { action: 'rules.remove' }), allow), { kind: 'allow' });
  });

  it('keeps a later listener’s deny, cancel, or ask', async () => {
    const decisions: PreToolDecision[] = [
      { kind: 'deny', reason: 'blocked elsewhere' },
      { kind: 'cancel' },
      { kind: 'ask', reason: 'another plugin asks' },
    ];
    for (const downstream of decisions) {
      const call = exec(AGENT_TOOL_NAME, { action: 'phoenix.start' });
      assert.deepEqual(await agentToolPreExecute(call, async () => downstream), downstream);
      assert.deepEqual(await agentToolPreExecute(exec('bash', {}), async () => downstream), downstream);
    }
  });
});

describe('registerAgentTool', () => {
  type Registered = { definition: unknown; options: unknown[] };

  function fakeTools(behavior: { throwOnRegister?: string } = {}) {
    const registered: Registered[] = [];
    let unregistered = 0;
    const host: AgentToolHost = {
      tools: {
        register: (definition: unknown, ...options: unknown[]) => {
          if (behavior.throwOnRegister !== undefined) {
            throw new Error(behavior.throwOnRegister);
          }
          registered.push({ definition, options });
          return () => void (unregistered += 1);
        },
      },
    };
    return { host, registered, unregistered: () => unregistered };
  }

  const operations: AgentToolOperations = {
    context: () => {
      throw new Error('not called while registering');
    },
  };
  const dshTools = async () => ({ defineTool: (options: AgentToolDefinitionOptions) => ({ defined: options.name }) });

  it('registers the tool built by the imported defineTool and unregisters it once', async () => {
    const { host, registered, unregistered } = fakeTools();
    const dispose = registerAgentTool(host, { operations, importTools: dshTools });
    assert.equal(await dispose.ready, true);
    assert.deepEqual(registered, [{ definition: { defined: AGENT_TOOL_NAME }, options: [] }]);
    dispose();
    dispose();
    assert.equal(unregistered(), 1);
  });

  it('forwards an activation condition to the registry', async () => {
    const { host, registered } = fakeTools();
    const when = () => true;
    assert.equal(await registerAgentTool(host, { operations, importTools: dshTools, when }).ready, true);
    assert.deepEqual(registered[0]?.options, [{ when }]);
  });

  it('registers nothing when disposed before DeepSeek Harness tools load', async () => {
    const { host, registered } = fakeTools();
    const dispose = registerAgentTool(host, { operations, importTools: dshTools });
    dispose();
    assert.equal(await dispose.ready, false);
    assert.deepEqual(registered, []);
  });

  it('is a no-op, never a failure, without a usable tool registry', async () => {
    const warnings: string[] = [];
    const warn = (message: string) => void warnings.push(message);
    assert.equal(await registerAgentTool({}, { operations, warn, importTools: dshTools }).ready, false);
    const throwingHost = {
      get tools(): never {
        throw new Error('tools is not injected');
      },
    };
    assert.equal(await registerAgentTool(throwingHost, { operations, warn, importTools: dshTools }).ready, false);
    const refusing = fakeTools({ throwOnRegister: 'tool "nemo_relay" is already registered' });
    assert.equal(await registerAgentTool(refusing.host, { operations, warn, importTools: dshTools }).ready, false);
    assert.ok(warnings.some((message) => message.includes('already registered')));
    const throwingWarn = () => {
      throw new Error('logger failed');
    };
    assert.equal(await registerAgentTool(refusing.host, { operations, warn: throwingWarn }).ready, false);
  });

  it('builds the definition itself when an installed bundle cannot import @deepseek-ai/dsh-tools', async () => {
    const failing = async () => {
      throw new Error("Cannot find package '@deepseek-ai/dsh-tools'");
    };
    for (const importTools of [failing, async () => ({}), undefined]) {
      if (importTools === undefined && dshToolsInstalled) {
        continue;
      }
      const warnings: string[] = [];
      const { host, registered } = fakeTools();
      const dispose = registerAgentTool(host, {
        operations,
        warn: (message) => void warnings.push(message),
        ...(importTools === undefined ? {} : { importTools }),
      });
      assert.equal(await dispose.ready, true);
      const definition = registered[0]?.definition as Record<string, any>;
      assert.equal(definition.name, AGENT_TOOL_NAME);
      assert.deepEqual(warnings, []);
      dispose();
    }
  });
});

describe('defineToolLocally', () => {
  /** What `ToolsService.register` checks (dsh-tools lib/index.js `register`), plus JSON Schema sanity. */
  function assertRegistryShape(definition: Record<string, any>) {
    assert.equal(typeof definition.name, 'string');
    assert.equal(typeof definition.description, 'string');
    assert.equal(typeof definition.execute, 'function');
    assert.equal(typeof definition.output?.render, 'function');
    const noFlags = (node: unknown): void => {
      if (node !== null && typeof node === 'object') {
        assert.notEqual((node as Record<string, unknown>).required, true, 'no per-property required flags remain');
        assert.notEqual((node as Record<string, unknown>).type, 'json', 'no author-only json type remains');
        Object.values(node).forEach(noFlags);
      }
    };
    noFlags(definition.parameters);
    noFlags(definition.output.schema);
  }

  function localTool(calls: AgentToolArgs[] = []) {
    const operations: AgentToolOperations = {
      context: () => {
        throw new Error('not used');
      },
    };
    let options: AgentToolDefinitionOptions | undefined;
    createAgentTool((value) => void (options = value), operations);
    assert.ok(options);
    const definition = defineToolLocally({
      ...options,
      execute: async (args) => (calls.push(args), { action: args.action, text: 'done' }),
    }) as Record<string, any>;
    return definition;
  }

  it('compiles the property-map specs to the JSON Schema the registry expects', () => {
    const definition = localTool();
    assertRegistryShape(definition);
    assert.deepEqual(definition.parameters.required, ['action']);
    assert.equal(definition.parameters.type, 'object');
    assert.deepEqual(definition.parameters.properties.action, { type: 'string', enum: [...AGENT_ACTIONS] });
    assert.equal(definition.parameters.properties.arguments.additionalProperties, true);
    assert.deepEqual(definition.output.schema, {
      type: 'object',
      additionalProperties: false,
      properties: { action: { type: 'string' }, text: { type: 'string' }, data: {} },
      required: ['action', 'text'],
    });
  });

  it('checks the arguments before running, like defineTool', async () => {
    const calls: AgentToolArgs[] = [];
    const definition = localTool(calls);
    for (const [args, problem] of [
      [{}, /arguments\.action is required/],
      [{ action: 'rules.explode' }, /arguments\.action must be one of/],
      [{ action: 'rules.add', id: 7 }, /arguments\.id must be a string/],
      [{ action: 'guardrails.failMode', mode: 'sometimes' }, /arguments\.mode must be one of "open", "closed"/],
      [{ action: 'rules.test', arguments: 'ls' }, /arguments\.arguments must be an object/],
      ['status', /arguments must be an object/],
    ] as const) {
      await assert.rejects(definition.execute(args, {}), problem, JSON.stringify(args));
      assert.equal(definition.presentCall(args), undefined, 'no call card for invalid arguments');
    }
    assert.deepEqual(calls, []);
    const ok = { action: 'rules.test', tool: 'bash', arguments: { command: 'ls' } };
    assert.deepEqual(await definition.execute(ok, {}), { action: 'rules.test', text: 'done' });
    assert.deepEqual(calls, [ok]);
    assert.equal(definition.presentCall(ok).kind, 'read');
    assert.equal(definition.isConcurrencySafe({ action: 'status' }), true);
    assert.equal(definition.isConcurrencySafe({ action: 'rules.add' }), false);
    assert.deepEqual(definition.output.render(ok, { action: 'rules.test', text: 'Allowed.' }), [
      { type: 'text', text: 'Allowed.' },
    ]);
  });
});
