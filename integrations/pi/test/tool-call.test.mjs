// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Drives the `tool_call` gate end to end.
 *
 * Every component this handler composes was already pinned -- the 403 shape in
 * `gateway-client.test.mjs`, the transform decision matrix in
 * `argument-transform.test.mjs`, the gateway's half in Rust -- and the handler
 * that wires them together had no test at all. That is an easy gap to miss in
 * review precisely because the coverage either side of it looks complete, and
 * it is the primary governance seam: for a model-invoked tool, this is the only
 * pre-execution decision point that sees arguments.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { drain, listen, load, named, rejection, stubGateway } from './harness.mjs';

const extension = (await import('../index.ts')).default;

const call = (input = { path: '/work/README.md' }) => ({
  toolCallId: 'c1',
  toolName: 'read',
  input,
});

describe('the tool_call gate', () => {
  let gateway;
  let url;

  before(async () => {
    gateway = stubGateway('tool_call');
    url = await listen(gateway.server);
    process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
  });

  after(() => {
    gateway.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
    delete process.env.NEMO_RELAY_PI_FAIL;
  });

  beforeEach(() => {
    gateway.reset();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
    delete process.env.NEMO_RELAY_PI_FAIL;
  });

  it('blocks a guardrail rejection and hands pi the reason verbatim', async () => {
    const fire = load(extension);
    gateway.replyWith(rejection('read .env is blocked; use .env.example'));

    const result = await fire('tool_call', call({ path: '/work/.env' }));

    // pi passes this string to the model with no framing at all, so anything
    // added here is read by the model as part of the policy's own words.
    assert.deepEqual(result, {
      block: true,
      reason: 'read .env is blocked; use .env.example',
    });
  });

  it('allows by returning undefined, never a truthy object', async () => {
    const fire = load(extension);
    const result = await fire('tool_call', call());
    // A truthy result without `block` decides nothing, but it stops every
    // later extension's `tool_call` handler from running.
    assert.equal(result, undefined);
  });

  it('applies a rewrite to the object pi will execute, and records that it did', async () => {
    const fire = load(extension);
    gateway.replyWith({
      status: 200,
      payload: { tool_call: { tool_call_id: 'c1', input: { path: '/work/.env.example' } } },
    });

    const event = call({ path: '/work/.env' });
    const result = await fire('tool_call', event);

    assert.equal(result, undefined, 'a rewrite is an allow, not a block');
    // In place, not replaced: pi hands the same object to the tool and to every
    // later handler, so a new reference would simply be dropped.
    assert.deepEqual(event.input, { path: '/work/.env.example' });

    await drain(fire);
    const [recorded] = named(gateway.posts, 'tool_arguments_transformed');
    assert.ok(recorded, 'the trace must record that the arguments were not the ones proposed');
    assert.equal(recorded.tool_call_id, 'c1');
    assert.equal(recorded.tool_name, 'read');
  });

  it('blocks a rewrite it cannot apply safely rather than running the original', async () => {
    const fire = load(extension);
    gateway.replyWith({
      status: 200,
      payload: {
        tool_call: { tool_call_id: 'c1', input: { path: '/work/.env', sudo: true } },
      },
    });

    const event = call({ path: '/work/.env' });
    const result = await fire('tool_call', event);

    assert.equal(result?.block, true);
    assert.match(result.reason, /added sudo/);
    // Falling back to the original arguments would silently discard a policy
    // decision, which is the failure the transform exists to prevent.
    assert.match(result.reason, /not a judgment about your request/);
    assert.deepEqual(event.input, { path: '/work/.env' }, 'a refused rewrite must not be applied');
  });

  it('does not post a transform mark when nothing rewrote the arguments', async () => {
    const fire = load(extension);
    await fire('tool_call', call());
    await drain(fire);
    assert.equal(named(gateway.posts, 'tool_arguments_transformed').length, 0);
  });

  it('fails open by default when the gateway is unreachable', async () => {
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:1';
    const fire = load(extension);
    // A dead sidecar must not brick the agent, matching how the shipped
    // hooks.json files use --fail-open everywhere except pre-tool events.
    assert.equal(await fire('tool_call', call()), undefined);
  });

  it('fails closed on demand, and says the block is infrastructure and not policy', async () => {
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:1';
    process.env.NEMO_RELAY_PI_FAIL = 'closed';
    const fire = load(extension);

    const result = await fire('tool_call', call());

    assert.equal(result?.block, true);
    // Telling the model a policy considered and refused its call, when nothing
    // did, gives it a false premise to reason from.
    assert.match(result.reason, /infrastructure fault, not a judgment/);
  });

  it('treats a 403 without the guardrail marker as a fault, not a policy decision', async () => {
    const fire = load(extension);
    gateway.replyWith({ status: 403, payload: { error: { message: 'nope' } } });
    // An authorization failure is not a verdict about the request, so under the
    // default fail-open policy the call proceeds rather than being reported to
    // the model as refused.
    assert.equal(await fire('tool_call', call()), undefined);
  });

  it('resolves a slow gateway through the failure policy rather than hanging pi', async () => {
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = '50';
    try {
      const fire = load(extension);
      gateway.replyWith({ status: 200, payload: {}, delayMs: 400 });
      // pi awaits this handler on its critical path, so a gateway that never
      // answers must become a decision, not a stall.
      assert.equal(await fire('tool_call', call()), undefined);
    } finally {
      delete process.env.NEMO_RELAY_PI_TIMEOUT_MS;
    }
  });
});

describe('an extension ahead of the gate', () => {
  let gateway;
  let url;

  before(async () => {
    gateway = stubGateway('tool_call');
    url = await listen(gateway.server);
    process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
  });

  after(() => {
    gateway.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => gateway.reset());

  // The documented blind spot, pinned as behavior rather than prose: pi stops
  // at the first handler that returns anything, so on the `pi install` path --
  // which loads last -- the call is blocked and the gateway never learns it
  // happened. `-e` inverts the order, which is why the launcher uses it.
  it('preempts the tool gate, and the gateway never sees the call', async () => {
    const fire = load(extension, {
      before: { tool_call: async () => ({ block: true, reason: 'blocked by another extension' }) },
    });

    const result = await fire('tool_call', call());

    assert.deepEqual(result, { block: true, reason: 'blocked by another extension' });
    await drain(fire);
    assert.equal(
      named(gateway.posts, 'tool_call').length,
      0,
      'the gate never ran, so no policy was consulted and nothing was recorded',
    );
  });

  // The two hooks do NOT resolve competing handlers the same way, and an earlier
  // version of this harness papered over the difference. `emitToolCall` runs every
  // handler and short-circuits only on a block, so a non-blocking result from an
  // extension ahead of us is inert -- our gate still runs, and the gateway still
  // sees the call. Only `emitUserBash` is first-truthy-wins.
  it('does not preempt the tool gate with a non-blocking result', async () => {
    const fire = load(extension, {
      // Truthy, but no `block`: pi keeps going.
      before: { tool_call: async () => ({ reason: 'just an opinion' }) },
    });

    const result = await fire('tool_call', call());

    // What pi hands back is the earlier extension's object, not ours: our allow
    // is `undefined`, and `undefined` does not overwrite a previous truthy
    // result. It is inert because it carries no `block` -- which is the whole
    // reason this extension returns `undefined` rather than `{}` on an allow.
    assert.deepEqual(result, { reason: 'just an opinion' });
    assert.notEqual(result.block, true, 'inert: nothing was blocked');

    await drain(fire);
    assert.equal(
      named(gateway.posts, 'tool_call').length,
      1,
      'and our gate still ran, so the gateway saw and decided the call',
    );
  });

  it('preempts the inline-shell gate the same way', async () => {
    const fire = load(extension, {
      before: {
        user_bash: async () => ({
          result: { output: 'handled elsewhere', exitCode: 0, cancelled: false, truncated: false },
        }),
      },
    });

    const result = await fire('user_bash', {
      command: 'git status',
      excludeFromContext: false,
      cwd: '/work',
    });

    assert.equal(result.result.exitCode, 0);
    await drain(fire);
    assert.equal(named(gateway.posts, 'user_bash').length, 0);
  });
});
