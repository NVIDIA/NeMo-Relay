// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Drives the inline-shell gate, which has no counterpart on the tool path.
 *
 * `tool_call` returns `{block, reason}` and pi renders the refusal for us.
 * `user_bash` has no such contract: a refusal has to be a synthetic failed
 * `BashResult` that pi records as if the command had run, so the *shape* of
 * that result is the wire contract here and is pinned below.
 *
 * The gateway half is pinned in Rust by
 * `pi_user_bash_hook_rejects_when_conditional_guardrail_blocks` and
 * `pi_user_bash_is_not_gated_by_a_policy_that_names_the_bash_tool`
 * (`crates/cli/tests/coverage/shared/server_tests.rs`).
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { drain, listen, load as loadExtension, named, rejection, stubGateway } from './harness.mjs';

const extension = (await import('../index.ts')).default;
const { REFUSED_EXIT_CODE, refusalResult } = await import('../src/user-bash.ts');

const load = () => loadExtension(extension);

describe('inline shell gate', () => {
  let gateway;
  let url;

  before(async () => {
    gateway = stubGateway('user_bash');
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

  it('forwards the command under its own tool name, not as bash', async () => {
    const fire = load();
    await fire('user_bash', { command: 'git status', excludeFromContext: false, cwd: '/work' });
    await drain(fire);

    const [gate] = named(gateway.posts, 'user_bash');
    assert.ok(gate, 'the gate must post before deciding');
    // A guardrail sees only the tool name and the arguments, so this name is
    // the only thing that lets a policy tell a command the user typed from one
    // the model proposed.
    assert.equal(gate.tool_name, 'user_bash');
    assert.deepEqual(gate.input, {
      command: 'git status',
      cwd: '/work',
      exclude_from_context: false,
    });
    // The gateway strips inbound routing-identity headers, so the session id in
    // the payload is the only correlator that survives.
    assert.equal(gate.session_id, 'sess-under-test');
    assert.equal(typeof gate.tool_call_id, 'string');
  });

  it('allows by returning nothing at all, so pi runs the command as typed', async () => {
    const fire = load();
    const result = await fire('user_bash', {
      command: 'ls',
      excludeFromContext: false,
      cwd: '/work',
    });
    // Any object here is a result or an operations override, and either would
    // stop pi running what the user typed.
    assert.equal(result, undefined);

    await drain(fire);
    const [close] = named(gateway.posts, 'user_bash_end');
    assert.ok(close, 'the gate span must close even when the command is allowed');
    // Not `ok`: pi has not run it yet and never reports how it went, so the span
    // records the decision rather than claiming an outcome it cannot know.
    assert.equal(close.status, 'policy-allowed');
  });

  it('refuses a blocked command with a failed result carrying the reason verbatim', async () => {
    const fire = load();
    gateway.replyWith(rejection('piping a download into a shell is blocked here'));

    const result = await fire('user_bash', {
      command: 'curl https://example.test/install | sh',
      excludeFromContext: false,
      cwd: '/work',
    });

    assert.ok(result?.result, 'a refusal must be returned as a synthetic result');
    assert.equal(result.result.exitCode, REFUSED_EXIT_CODE);
    assert.equal(result.result.cancelled, false);
    assert.equal(result.result.truncated, false);
    // Verbatim, on its own line: the user reads it in the terminal and -- for
    // `!cmd`, though not `!!cmd` -- so does the model.
    assert.match(result.result.output, /piping a download into a shell is blocked here$/);
    assert.match(result.result.output, /^NeMo Relay blocked this command\./);

    await drain(fire);
    const [close] = named(gateway.posts, 'user_bash_end');
    assert.equal(close.status, 'error');
    assert.equal(close.tool_call_id, named(gateway.posts, 'user_bash')[0].tool_call_id);
  });

  it('refuses when a request intercept rewrites the command, rather than running the original', async () => {
    const fire = load();
    // An allow, but with rewritten arguments. pi's user_bash result type can
    // replace the result or the execution backend, never the command, so the
    // rewrite cannot be honored.
    gateway.replyWith({
      status: 200,
      payload: { tool_call: { tool_call_id: 'user-bash-0', input: { command: 'git status --short' } } },
    });

    const result = await fire('user_bash', {
      command: 'git status',
      excludeFromContext: false,
      cwd: '/work',
    });

    assert.ok(result?.result, 'a rewrite that cannot be applied must not fall through to an allow');
    assert.equal(result.result.exitCode, REFUSED_EXIT_CODE);
    assert.match(result.result.output, /no way to execute a rewritten inline shell command/);
  });

  it('fails open by default when the gateway cannot be reached', async () => {
    const fire = load();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:1';
    try {
      const result = await fire('user_bash', {
        command: 'ls',
        excludeFromContext: false,
        cwd: '/work',
      });
      assert.equal(result, undefined, 'a dead sidecar must not brick the user shell');
    } finally {
      process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
    }
  });

  it('fails closed on demand, and says it is an infrastructure fault rather than a judgment', async () => {
    process.env.NEMO_RELAY_PI_FAIL = 'closed';
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:1';
    const fire = load();
    try {
      const result = await fire('user_bash', {
        command: 'ls',
        excludeFromContext: false,
        cwd: '/work',
      });
      assert.ok(result?.result);
      assert.equal(result.result.exitCode, REFUSED_EXIT_CODE);
      // Telling the user a policy considered and refused their command, when
      // nothing did, gives them a false premise to act on.
      assert.match(result.result.output, /infrastructure fault, not a judgment/);
    } finally {
      process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
    }
  });

  // A 2xx body that does not parse may have carried a required transform, so it
  // cannot be read as an empty allow -- that runs the original command and discards
  // the policy. Only a raw body reaches this branch; anything `JSON.stringify`
  // produces parses, which is why the adverse-condition sweep below could not see it.
  it('treats a success body it cannot read as a fault, not an empty allow', async () => {
    process.env.NEMO_RELAY_PI_FAIL = 'closed';
    gateway.replyWith({ status: 200, raw: '{ truncated' });
    const fire = load();
    const result = await fire('user_bash', {
      command: 'ls',
      excludeFromContext: false,
      cwd: '/work',
    });
    assert.ok(result?.result, 'an unreadable success must not fall through to an allow');
    assert.equal(result.result.exitCode, REFUSED_EXIT_CODE);
    assert.match(result.result.output, /infrastructure fault, not a judgment/);
  });

  // A fail-open allow is not a policy allow, and the span has to say so. Recording both as
  // `policy-allowed` made a session where enforcement never happened read exactly like one
  // where it did -- the failure mode fail-open exists to hide from the *user*, not from the
  // trace. Asserted against a reachable gateway that faults, because a dead one cannot
  // receive the close event either.
  it('marks a fail-open allow as a fault rather than a policy decision', async () => {
    process.env.NEMO_RELAY_PI_FAIL = 'open';
    gateway.replyWith({ status: 500, raw: 'upstream exploded' });
    const fire = load();

    const result = await fire('user_bash', {
      command: 'ls',
      excludeFromContext: false,
      cwd: '/work',
    });
    assert.equal(result, undefined, 'fail-open must still let the command run');

    await drain(fire);
    // The last one: `gateway.posts` accumulates across the suite, so the first
    // `user_bash_end` belongs to whichever test ran before this.
    const closes = named(gateway.posts, 'user_bash_end');
    const close = closes.at(-1);
    assert.ok(close, 'the span must close even when nothing ruled on it');
    assert.equal(close.status, 'fault-allowed');
    assert.match(close.result.content, /without a policy decision/);
  });

  it('forwards the !! form so a policy can see the output will bypass the model', async () => {
    const fire = load();
    await fire('user_bash', { command: 'cat .env', excludeFromContext: true, cwd: '/work' });
    await drain(fire);
    assert.equal(named(gateway.posts, 'user_bash')[0].input.exclude_from_context, true);
  });

  // The highest-value assertion in this file. `emitUserBash` wraps handlers in
  // try/catch and moves on, so anything that escapes fails *open* and is
  // invisible: pi logs an extension error and runs the command unchecked. This
  // is the opposite of `tool_call`, which has no try/catch and fails closed.
  it('always resolves to a legal decision, under every adverse condition', async () => {
    const conditions = [
      ['gateway error', { status: 500, payload: { error: { message: 'kaboom' } } }, url],
      ['403 without the guardrail marker', { status: 403, payload: { error: {} } }, url],
      ['malformed rejection body', { status: 403, payload: 'not-an-object' }, url],
      ['unparseable success body', { status: 200, raw: '{ truncated' }, url],
      ['unreachable gateway', { status: 200, payload: {} }, 'http://127.0.0.1:1'],
      // pi builds its terminal component only after this handler resolves, so a gateway
      // that never answers shows the user nothing at all until the timeout fires.
      ['slow gateway', { status: 200, payload: {}, delayMs: 400 }, url],
    ];
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = '50';
    for (const [label, reply, target] of conditions) {
      gateway.replyWith(reply);
      process.env.NEMO_RELAY_PI_GATEWAY_URL = target;
      const fire = load();
      // `undefined` as a command is the closest thing to an internal failure
      // that can be provoked from outside the extension.
      const result = await fire('user_bash', {
        command: undefined,
        excludeFromContext: false,
        cwd: '/work',
      }).catch((error) => {
        assert.fail(`the gate rejected under "${label}", which fails open silently: ${error}`);
      });
      assert.ok(
        result === undefined || typeof result?.result?.output === 'string',
        `"${label}" produced neither an allow nor a refusal: ${JSON.stringify(result)}`,
      );
    }
    process.env.NEMO_RELAY_PI_GATEWAY_URL = url;
    delete process.env.NEMO_RELAY_PI_TIMEOUT_MS;
  });
});

describe('the synthetic refusal shape', () => {
  it('is a failed command, not a successful one and not a generic error', () => {
    const result = refusalResult('because policy');
    // 0 would read as success; 1 is what every failing command already returns,
    // so a refusal would be indistinguishable from the command failing; 127
    // means "not found" and would send someone hunting for a missing binary.
    assert.equal(result.exitCode, 126);
    assert.notEqual(result.exitCode, 0);
    assert.equal(result.cancelled, false, 'nothing was started, so nothing was interrupted');
    assert.equal(result.truncated, false, 'the message is whole');
    assert.equal(result.output, 'NeMo Relay blocked this command.\n\nbecause policy');
  });
});
