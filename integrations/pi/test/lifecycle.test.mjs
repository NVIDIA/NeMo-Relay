// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Drives the extension's lifecycle handlers and asserts what reaches the gateway.
 *
 * `gateway-client.test.mjs` covers the wire contract in isolation; nothing
 * exercised the handlers themselves, so the identity fields the gateway cannot
 * infer -- `attempt_index` and `turn_seq` -- were implemented and demonstrated
 * once in a live trace but never pinned. These tests pin them, plus the
 * shutdown-reason behaviour.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { after, before, beforeEach, describe, it } from 'node:test';

const extension = (await import('../index.ts')).default;

/** Collects every payload posted to /hooks/pi. */
function stubGateway() {
  const posts = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => {
      body += c;
    });
    req.on('end', () => {
      posts.push(JSON.parse(body || '{}'));
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end('{}');
    });
  });
  return { server, posts };
}

/** Registers the extension and returns a driver that fires hooks in order. */
function load() {
  const handlers = new Map();
  const pi = {
    on(name, handler) {
      if (!handlers.has(name)) handlers.set(name, []);
      handlers.get(name).push(handler);
    },
  };
  extension(pi);
  const ctx = {
    cwd: '/work',
    mode: 'print',
    hasUI: false,
    sessionManager: { getSessionId: () => 'sess-under-test' },
  };
  return async (name, event = {}) => {
    let result;
    for (const handler of handlers.get(name) ?? []) {
      result = await handler({ type: name, ...event }, ctx);
    }
    return result;
  };
}

const named = (posts, name) => posts.filter((p) => p.hook_event_name === name);

describe('lifecycle identity the gateway cannot infer', () => {
  let ctx;
  let url;

  before(async () => {
    ctx = stubGateway();
    await new Promise((r) => ctx.server.listen(0, '127.0.0.1', r));
    url = `http://127.0.0.1:${ctx.server.address().port}`;
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
    await new Promise((r) => ctx.server.listen(0, '127.0.0.1', r));
    process.env.NEMO_RELAY_PI_GATEWAY_URL = `http://127.0.0.1:${ctx.server.address().port}`;
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
    await new Promise((r) => ctx.server.listen(0, '127.0.0.1', r));
    process.env.NEMO_RELAY_PI_GATEWAY_URL = `http://127.0.0.1:${ctx.server.address().port}`;
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
    await new Promise((r) => ctx.server.listen(0, '127.0.0.1', r));
    process.env.NEMO_RELAY_PI_GATEWAY_URL = `http://127.0.0.1:${ctx.server.address().port}`;
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
  // observability handler that returned anything would change pi's behaviour.
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
