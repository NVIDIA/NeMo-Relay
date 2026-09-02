// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Drives the extension's lifecycle handlers and asserts what reaches the gateway.
 *
 * `gateway-client.test.mjs` covers the wire contract in isolation; nothing
 * exercised the handlers themselves, so the identity fields the gateway cannot
 * infer -- `attempt_index` and `turn_seq` -- were implemented and demonstrated
 * once in a live trace but never pinned. These tests pin them, plus the
 * shutdown-reason behavior.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { listen, load as loadExtension, named, stubGateway } from './harness.mjs';

const extension = (await import('../index.ts')).default;

// Headless: pi's print mode has no UI, and none of these hooks depend on one.
const load = () => loadExtension(extension, { ctx: { mode: 'print', hasUI: false } });

describe('lifecycle identity the gateway cannot infer', () => {
  let ctx;
  let url;

  before(async () => {
    ctx = stubGateway();
    url = await listen(ctx.server);
    process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  it('attributes each turn to its attempt across an agent-run re-entry', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    // Attempt 0, two turns.
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('turn_start', { turnIndex: 1, timestamp: 2 });
    await fire('turn_end', { turnIndex: 1 });
    await fire('agent_end', { messages: [] });
    // Re-entry: pi resets turnIndex to 0.
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 3 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('agent_end', { messages: [] });
    await fire('agent_settled');
    await fire('session_shutdown', { reason: 'quit' });

    const starts = named(ctx.posts, 'turn_start');
    assert.equal(starts.length, 3);

    // pi's turn_index collides across the re-entry...
    assert.deepEqual(
      starts.map((p) => p.turn_index),
      [0, 1, 0],
    );
    // ...while turn_seq stays monotonic and attempt_index attributes each turn.
    assert.deepEqual(
      starts.map((p) => p.turn_seq),
      [0, 1, 2],
    );
    assert.deepEqual(
      starts.map((p) => p.attempt_index),
      [0, 0, 1],
    );

    assert.deepEqual(
      named(ctx.posts, 'agent_start').map((p) => p.attempt_index),
      [0, 1],
    );
    assert.equal(named(ctx.posts, 'agent_settled')[0].attempts, 2);
  });

  it('resets the attempt counter on agent_settled so a second prompt starts at 0', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('agent_end', { messages: [] });
    await fire('agent_settled');
    await fire('agent_start');
    await fire('session_shutdown', { reason: 'quit' });

    assert.deepEqual(
      named(ctx.posts, 'agent_start').map((p) => p.attempt_index),
      [0, 0],
    );
  });

  it('posts every hook in order, never concurrently reordered', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('agent_end', { messages: [] });
    await fire('agent_settled');
    await fire('session_shutdown', { reason: 'quit' });

    assert.deepEqual(
      ctx.posts.map((p) => p.hook_event_name),
      [
        'session_start',
        'agent_start',
        'turn_start',
        'turn_end',
        'agent_end',
        'agent_settled',
        'session_shutdown',
      ],
    );
  });

  it('carries the session id on every post', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('session_shutdown', { reason: 'quit' });
    assert.ok(ctx.posts.length > 0);
    for (const post of ctx.posts) {
      assert.equal(post.session_id, 'sess-under-test');
    }
  });
});

describe('session_shutdown reason', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  it('does not end the session on /reload, which would split one session into two traces', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('session_shutdown', { reason: 'reload' });
    assert.deepEqual(named(ctx.posts, 'session_shutdown'), []);
    // The queue is still drained, so nothing already posted is lost.
    assert.equal(named(ctx.posts, 'session_start').length, 1);
  });

  for (const reason of ['quit', 'new', 'resume', 'fork']) {
    it(`ends the session on ${reason}, and forwards the reason`, async () => {
      const fire = load();
      await fire('session_start', { reason: 'startup' });
      await fire('session_shutdown', { reason });
      const ends = named(ctx.posts, 'session_shutdown');
      assert.equal(ends.length, 1);
      assert.equal(ends[0].reason, reason);
    });
  }

  it('forwards the replacement target when pi supplies one', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('session_shutdown', { reason: 'fork', targetSessionFile: '/s/next.jsonl' });
    assert.equal(named(ctx.posts, 'session_shutdown')[0].target_session_file, '/s/next.jsonl');
  });
});

describe('attribution on every hook that has one', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  // Tool hooks used to carry no attribution at all, which made "which attempt ran this tool"
  // answerable only by reading surrounding events in arrival order -- and that stops working the
  // moment two attempts are in flight.
  it('attributes tool hooks to the attempt and turn that ran them', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('agent_end', { messages: [] });
    // Second attempt: pi's turn_index is 0 again, so only turn_seq separates the two turns.
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 2 });
    await fire('tool_execution_start', { toolCallId: 'c1', toolName: 'read', args: {} });
    await fire('tool_call', { toolCallId: 'c1', toolName: 'read', input: { path: 'a.txt' } });
    await fire('tool_execution_end', {
      toolCallId: 'c1',
      toolName: 'read',
      result: 'ok',
      isError: false,
    });
    await fire('turn_end', { turnIndex: 0 });
    await fire('session_shutdown', { reason: 'quit' });

    for (const name of ['tool_call', 'tool_execution_end']) {
      const post = named(ctx.posts, name)[0];
      assert.equal(post.attempt_index, 1, `${name} attempt_index`);
      assert.equal(post.turn_seq, 1, `${name} turn_seq`);
    }
  });

  // pi carries turn_index on the close but not turn_seq, so without this the close could not be
  // matched to its own open across a re-entry, where turn_index 0 appears once per attempt.
  it('gives turn_end the same turn_seq its turn_start announced', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('turn_start', { turnIndex: 1, timestamp: 2 });
    await fire('turn_end', { turnIndex: 1 });
    await fire('session_shutdown', { reason: 'quit' });

    assert.deepEqual(
      named(ctx.posts, 'turn_end').map((p) => [p.turn_index, p.turn_seq, p.attempt_index]),
      [
        [0, 0, 0],
        [1, 1, 0],
      ],
    );
  });

  // `attempts` is how many attempts the run took; `attempt_index` is the last of them. Sending
  // only the count made this the one hook a consumer filtering on attempt_index missed.
  it('sends agent_settled the attempt count and the last attempt index', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('agent_end', { messages: [] });
    await fire('agent_start');
    await fire('agent_end', { messages: [] });
    await fire('agent_settled');
    await fire('session_shutdown', { reason: 'quit' });

    const settled = named(ctx.posts, 'agent_settled')[0];
    assert.equal(settled.attempts, 2);
    assert.equal(settled.attempt_index, 1);
  });
});

describe('compaction', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  it('forwards both the announcement and the completion, with the token count', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('session_before_compact', {
      reason: 'threshold',
      willRetry: false,
      preparation: { tokensBefore: 120_000, isSplitTurn: true },
    });
    await fire('session_compact', {
      reason: 'threshold',
      willRetry: false,
      fromExtension: false,
      compactionEntry: { tokensBefore: 120_000 },
    });
    await fire('session_shutdown', { reason: 'quit' });

    const announced = named(ctx.posts, 'session_before_compact')[0];
    assert.equal(announced.reason, 'threshold');
    assert.equal(announced.will_retry, false);
    assert.equal(announced.tokens_before, 120_000);
    assert.equal(announced.is_split_turn, true);
    assert.equal(announced.attempt_index, 0);
    assert.equal(announced.turn_seq, 0);

    const completed = named(ctx.posts, 'session_compact')[0];
    assert.equal(completed.reason, 'threshold');
    assert.equal(completed.from_extension, false);
    assert.equal(completed.tokens_before, 120_000);
  });

  // pi spells "cancel this compaction, or replace its result" as a returned object, so an
  // observability handler that returned anything would change pi's behavior.
  it('never returns a value that could cancel or replace pi compaction', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    assert.equal(
      await fire('session_before_compact', { reason: 'manual', willRetry: false }),
      undefined,
    );
    assert.equal(
      await fire('session_compact', { reason: 'manual', willRetry: false, fromExtension: false }),
      undefined,
    );
    await fire('session_shutdown', { reason: 'quit' });
  });

  // `willRetry` on the announcement is the only advance notice pi gives an extension that the
  // agent run is about to re-enter; `agent_end` carries no such marker.
  it('carries the retry marker that agent_end lacks', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('session_before_compact', { reason: 'overflow', willRetry: true });
    await fire('session_shutdown', { reason: 'quit' });
    assert.equal(named(ctx.posts, 'session_before_compact')[0].will_retry, true);
  });
});

describe('concurrent tools in one turn', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  // pi preflights sibling calls sequentially and then executes them concurrently, so
  // `tool_execution_end` arrives in an order that has nothing to do with submission.
  // `toolCallId` is the only correlator pi gives, and every piece of per-call state is
  // keyed by it -- a regression that keyed on anything else would pass every other test
  // here, because no other test uses more than one call id.
  it('keeps two calls distinct when their ends arrive out of submission order', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });

    await fire('tool_execution_start', { toolCallId: 'a', toolName: 'read', args: {} });
    await fire('tool_execution_start', { toolCallId: 'b', toolName: 'bash', args: {} });
    // Both gates in flight at once, which is what the serial queue has to survive.
    await Promise.all([
      fire('tool_call', { toolCallId: 'a', toolName: 'read', input: { path: 'a.txt' } }),
      fire('tool_call', { toolCallId: 'b', toolName: 'bash', input: { command: 'ls' } }),
    ]);
    // Closing in the reverse order, which is the case that motivates the id keying.
    await fire('tool_execution_end', { toolCallId: 'b', toolName: '', result: 'ok', isError: false });
    await fire('tool_execution_end', { toolCallId: 'a', toolName: '', result: 'ok', isError: false });
    await fire('turn_end', { turnIndex: 0 });
    await fire('session_shutdown', { reason: 'quit' });

    const ends = named(ctx.posts, 'tool_execution_end');
    assert.equal(ends.length, 2);
    // `toolName` was empty on both ends, so each had to be recovered from the start that
    // named it -- and recovered from the *right* one.
    const byId = Object.fromEntries(ends.map((post) => [post.tool_call_id, post.tool_name]));
    assert.deepEqual(byId, { a: 'read', b: 'bash' });
    assert.ok(
      ends.every((post) => post.turn_seq === 0),
      'both calls ran in the same turn, so both must carry that turn',
    );
  });

  it('serializes posts even when gating hooks run concurrently', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await Promise.all([
      fire('tool_call', { toolCallId: 'a', toolName: 'read', input: {} }),
      fire('tool_call', { toolCallId: 'b', toolName: 'read', input: {} }),
    ]);
    await fire('session_shutdown', { reason: 'quit' });

    // The gateway derives boundaries from arrival order, so two concurrent gates must
    // still reach it after the turn that owns them.
    const order = ctx.posts.map((post) => post.hook_event_name);
    assert.ok(
      order.indexOf('turn_start') < order.indexOf('tool_call'),
      `a tool span must not open before its turn: ${order.join(', ')}`,
    );
    assert.equal(named(ctx.posts, 'tool_call').length, 2);
  });
});

describe('unpaired tool boundaries', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  // Deliberate asymmetry, and one a future contributor would otherwise "fix":
  // `tool_execution_start` fires before validation and for calls pi then discards, so
  // forwarding it as a tool start would open gateway spans for calls that never ran.
  it('never forwards tool_execution_start, which fires for calls that never execute', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('tool_execution_start', { toolCallId: 'ghost', toolName: 'read', args: {} });
    await fire('session_shutdown', { reason: 'quit' });

    assert.equal(named(ctx.posts, 'tool_execution_start').length, 0);
    assert.equal(named(ctx.posts, 'tool_call').length, 0);
  });

  // The reason this hook closes the span rather than `tool_result`: a blocked call takes
  // pi's immediate path and never reaches `afterToolCall`, so `tool_result` never fires --
  // but `tool_execution_end` always does, with `isError: true`.
  it('closes a blocked call, which never reaches tool_result', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('tool_execution_start', { toolCallId: 'blocked', toolName: 'read', args: {} });
    await fire('tool_execution_end', {
      toolCallId: 'blocked',
      toolName: '',
      result: 'Tool failed.',
      isError: true,
    });
    await fire('session_shutdown', { reason: 'quit' });

    const [end] = named(ctx.posts, 'tool_execution_end');
    assert.ok(end, 'a blocked call must still close');
    assert.equal(end.status, 'error');
    assert.equal(end.tool_name, 'read', 'the name comes from the start, which did fire');
  });

  it('falls back to a placeholder when no start ever named the tool', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('tool_execution_end', { toolCallId: 'orphan', toolName: '', result: null, isError: false });
    await fire('session_shutdown', { reason: 'quit' });

    // Dropping the post would lose the span entirely; the gateway synthesizes the missing
    // start, and a named-but-unknown tool is more useful than nothing.
    const [end] = named(ctx.posts, 'tool_execution_end');
    assert.equal(end.tool_name, 'unknown');
  });
});

describe('compaction-driven re-entry', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  // The individual fields are pinned elsewhere; this pins the *sequence*, which is the
  // one path where `willRetry` is genuine advance notice that the run is about to
  // re-enter, and where pi's turn_index restarts at 0 for the second time.
  it('keeps counting turns across a compaction, where pi restarts its own index', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('session_before_compact', {
      reason: 'overflow',
      willRetry: true,
      preparation: { tokensBefore: 120_000, isSplitTurn: false },
    });
    await fire('agent_end', { messages: [] });
    await fire('session_compact', { reason: 'overflow', willRetry: true, fromExtension: false });
    await fire('agent_start');
    await fire('turn_start', { turnIndex: 0, timestamp: 2 });
    await fire('turn_end', { turnIndex: 0 });
    await fire('agent_end', { messages: [] });
    await fire('agent_settled');
    await fire('session_shutdown', { reason: 'quit' });

    const starts = named(ctx.posts, 'turn_start');
    // pi says turn 0 both times; only turn_seq can tell them apart.
    assert.deepEqual(starts.map((post) => post.turn_index), [0, 0]);
    assert.deepEqual(starts.map((post) => post.turn_seq), [0, 1]);
    assert.deepEqual(starts.map((post) => post.attempt_index), [0, 1]);

    const [announced] = named(ctx.posts, 'session_before_compact');
    assert.equal(announced.will_retry, true);
    assert.equal(announced.tokens_before, 120_000);
    assert.equal(announced.attempt_index, 0, 'announced during the first attempt');

    const [settled] = named(ctx.posts, 'agent_settled');
    assert.equal(settled.attempts, 2, 'the compaction re-entry is a second attempt');
  });
});

describe('what an interrupted session loses', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  // pi registers no SIGINT handler in any mode -- all three build `["SIGTERM"]` plus
  // SIGHUP off-Windows -- and raw mode is set only by the TUI, so under `-p`,
  // `--mode json` and `--mode rpc` a user's Ctrl+C is a real SIGINT that terminates
  // with teardown never running. `session_shutdown` never fires, the drain never
  // happens, and whatever is still queued is lost.
  //
  // This pins the *bound* on that loss: every awaited hook has already reached the
  // gateway by the time pi continues, so an interrupt can only drop observability
  // marks queued since the last awaited one. Both gates and both turn boundaries
  // await; nothing else does.
  it('has already delivered every awaited hook, so only queued marks can be lost', async () => {
    const fire = load();
    await fire('session_start', { reason: 'startup' });
    await fire('agent_start');

    // Awaited: a turn boundary defines what later spans parent to.
    await fire('turn_start', { turnIndex: 0, timestamp: 1 });
    assert.equal(
      named(ctx.posts, 'turn_start').length,
      1,
      'turn_start must be delivered before pi continues, not queued',
    );

    // Awaited: the gate cannot decide without a verdict.
    await fire('tool_call', { toolCallId: 'c1', toolName: 'read', input: { path: 'a.txt' } });
    assert.equal(named(ctx.posts, 'tool_call').length, 1, 'the gate is a round trip, by design');

    await fire('turn_end', { turnIndex: 0 });
    assert.equal(named(ctx.posts, 'turn_end').length, 1);

    // The interrupt: no session_shutdown, so no drain. What is already delivered stays
    // delivered -- the gateway holds it -- so the trace is left open, not lost.
    const atInterrupt = ctx.posts.map((post) => post.hook_event_name);
    assert.ok(atInterrupt.includes('session_start'));
    assert.ok(atInterrupt.includes('turn_start'));
    assert.ok(atInterrupt.includes('tool_call'));
    assert.ok(atInterrupt.includes('turn_end'));
    assert.ok(
      !atInterrupt.includes('session_shutdown'),
      'the whole point: an interrupted session never closes',
    );
  });
});

describe('the session join key on redirected providers', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
    process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = 'https://api.openai.com/v1';
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
    delete process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
    delete process.env.NEMO_RELAY_PROXY_CREDENTIAL;
  });

  const model = {
    id: 'gpt-test',
    api: 'openai-completions',
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
  };

  /** Drive the extension while recording what it registers, and with a settable session id. */
  function loadRecording(sessionId) {
    const registrations = [];
    const handlers = new Map();
    let current = sessionId;
    extension({
      on(name, handler) {
        if (!handlers.has(name)) handlers.set(name, []);
        handlers.get(name).push(handler);
      },
      registerProvider: (name, config) => registrations.push({ name, config }),
    });
    const context = {
      cwd: '/work',
      mode: 'interactive',
      hasUI: true,
      sessionManager: { getSessionId: () => current },
      model,
      modelRegistry: { getAll: () => [model] },
    };
    const fire = async (name, event = {}) => {
      for (const handler of handlers.get(name) ?? []) {
        await handler({ type: name, ...event }, context);
      }
    };
    return { fire, registrations, setSession: (id) => (current = id) };
  }

  // The header rides on the registration, not on a per-request hook: pi's
  // `before_provider_headers` carries no request identity, and its context reports
  // the *currently selected* model -- so scoping on it would both omit the key from
  // a redirected call whose model was captured before a switch, and leak an internal
  // session id to a provider we deliberately did not redirect.
  it('rides on the provider registration, so only redirected providers send it', async () => {
    const { fire, registrations } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });

    assert.equal(registrations.length, 1, 'the matching provider should be redirected');
    assert.equal(registrations[0].name, 'openai');
    assert.equal(registrations[0].config.headers['x-nemo-relay-session-id'], 'sess-one');
  });

  // The key is baked in at registration, so it cannot follow a replacement on its
  // own. pi `v0.84.0` rebuilds the runtime for `/new`, `/resume` and `/fork`, so
  // this pins the defence rather than a path pi takes today.
  it('is refreshed when the session is replaced under the same runtime', async () => {
    const { fire, registrations, setSession } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });

    setSession('sess-two');
    await fire('session_start', { reason: 'resume' });

    assert.equal(registrations.length, 2, 'a replacement must re-register, not reuse');
    assert.equal(registrations[1].config.headers['x-nemo-relay-session-id'], 'sess-two');
  });

  // A gateway started by `nemo-relay run` authenticates its own client before any
  // intercept can rewrite the route, so a redirected call without this credential is
  // rejected 401 -- the redirect succeeds and every model call then fails, which is
  // the one outcome redirection exists to avoid.
  it('carries the launcher proxy credential on a redirected provider', async () => {
    process.env.NEMO_RELAY_PROXY_CREDENTIAL = 'nrp_testtoken';
    const { fire, registrations } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });

    assert.equal(registrations.length, 1);
    assert.equal(registrations[0].config.headers['x-nemo-relay-proxy-token'], 'nrp_testtoken');
  });

  // A standalone `nemo-relay --bind` daemon requires no credential, so the launcher
  // sets nothing. Sending the header empty would be a credential claim we cannot back.
  it('omits the proxy credential header when the launcher set none', async () => {
    const { fire, registrations } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });

    assert.equal(registrations.length, 1);
    assert.ok(
      !('x-nemo-relay-proxy-token' in registrations[0].config.headers),
      'an absent credential must not become an empty header',
    );
  });

  // Same structural scope as the session id, and it matters more here: the credential
  // authenticates *this* invocation, so a provider the gateway does not front must
  // never see it. `registerProvider` runs only on a redirect, which is what enforces it.
  // No credential, so naming the upstream is not available and a mismatch is still a
  // refusal -- which is the state that leaves a provider un-redirected, and so the
  // state in which nothing may be attached to it.
  it('never reaches a provider that was not redirected', async () => {
    process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = 'https://elsewhere.example/v1';
    try {
      const { fire, registrations } = loadRecording('sess-one');
      await fire('session_start', { reason: 'startup' });
      assert.equal(registrations.length, 0, 'a mismatched upstream must not register');
    } finally {
      process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = 'https://api.openai.com/v1';
    }
  });

  // Asserts the extension's half of the contract only: that a mismatch now registers,
  // and that the registration carries the endpoint. Whether the gateway then forwards
  // there, opens an LLM span and applies model-call policy is asserted on the Rust
  // side, in `crates/cli/tests/coverage/shared/server_tests.rs`.
  it('names the endpoint when the gateway does not front it', async () => {
    process.env.NEMO_RELAY_PROXY_CREDENTIAL = 'nrp_testtoken';
    process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = 'https://elsewhere.example/v1';
    try {
      const { fire, registrations } = loadRecording('sess-one');
      await fire('session_start', { reason: 'startup' });

      assert.equal(registrations.length, 1, 'a named upstream makes the redirect safe');
      assert.equal(
        registrations[0].config.headers['x-nemo-relay-upstream-base-url'],
        'https://api.openai.com/v1',
        'the header must name the endpoint the model would otherwise have called',
      );
    } finally {
      process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = 'https://api.openai.com/v1';
    }
  });

  // Naming an upstream the gateway already forwards to would be a header that changes
  // nothing, on every request, for the common case.
  it('names no endpoint when the gateway already fronts the model', async () => {
    process.env.NEMO_RELAY_PROXY_CREDENTIAL = 'nrp_testtoken';
    const { fire, registrations } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });

    assert.equal(registrations.length, 1);
    assert.ok(
      !('x-nemo-relay-upstream-base-url' in registrations[0].config.headers),
      'a static match must not name an upstream',
    );
  });

  it('does not re-register when the session id has not moved', async () => {
    const { fire, registrations } = loadRecording('sess-one');
    await fire('session_start', { reason: 'startup' });
    await fire('model_select', { model });
    assert.equal(registrations.length, 1, 'the redirect must stay idempotent');
  });
});
