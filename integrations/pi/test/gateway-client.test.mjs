// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests the extension's half of the `/hooks/pi` wire contract.
 *
 * The gateway's half is pinned in Rust by
 * `pi_tool_call_hook_rejects_when_conditional_guardrail_blocks` and
 * `pi_tool_call_hook_allows_when_no_guardrail_objects`
 * (`crates/cli/tests/coverage/shared/server_tests.rs`). These tests run against
 * a local HTTP server that reproduces the exact shapes those tests assert, so
 * the two halves are checked against the same contract from both sides.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { after, before, describe, it } from 'node:test';

const { postHook, resolveFault, configFromEnv } = await import('../src/gateway-client.ts');

/** Start a server that replies with `handler(requestBody)`. */
function serve(handler) {
  const received = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      received.push({ url: req.url, headers: req.headers, body: JSON.parse(body || '{}') });
      const { status, payload, raw, delayMs, bodyDelayMs } = handler(received.at(-1));
      const send = () => {
        res.writeHead(status, { 'content-type': 'application/json' });
        // `raw` lets a case emit a body JSON.parse cannot read.
        const body = raw ?? JSON.stringify(payload ?? {});
        if (bodyDelayMs) setTimeout(() => res.end(body), bodyDelayMs);
        else res.end(body);
      };
      if (delayMs) setTimeout(send, delayMs);
      else send();
    });
  });
  return { server, received };
}

async function listen(server) {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${server.address().port}`;
}

const baseConfig = (url, overrides = {}) => ({
  url,
  timeoutMs: 2000,
  onFault: 'open',
  sessionId: 'test-session',
  ...overrides,
});

describe('gateway client wire contract', () => {
  let ctx;
  let url;

  before(async () => {
    ctx = serve((request) => {
      const name = request.body.hook_event_name;
      if (name === 'slow') return { status: 200, payload: {}, delayMs: 500 };
      if (name === 'slow-body') return { status: 200, payload: {}, bodyDelayMs: 500 };
      if (name === 'bad-json') return { status: 200, raw: '{ truncated' };
      if (name === 'array-body') return { status: 200, payload: [] };
      if (name === 'string-body') return { status: 200, payload: 'ok' };
      if (name === 'boom') return { status: 500, payload: { error: { message: 'kaboom' } } };
      if (name === 'naked-403') return { status: 403, payload: { error: { message: 'nope' } } };
      if (request.body.tool_name === 'read' && request.body.input?.path?.endsWith('.env')) {
        // Byte-for-byte the shape CliError::into_response produces.
        return {
          status: 403,
          payload: {
            error: {
              message: 'guardrail rejected: read .env is blocked; use .env.example',
              type: 'nemo_relay_guardrail_rejected',
              reason: 'read .env is blocked; use .env.example',
            },
          },
        };
      }
      return { status: 200, payload: {} };
    });
    url = await listen(ctx.server);
  });

  after(() => ctx.server.close());

  // A 2xx body may carry a required argument transform, so an unreadable one is not
  // an empty allow -- treating it as one runs the original arguments and silently
  // discards a policy decision, which is the failure a refused transform blocks for.
  it('treats an unreadable or non-object 2xx body as a fault, not a bare allow', async () => {
    for (const name of ['bad-json', 'array-body', 'string-body']) {
      const outcome = await postHook(baseConfig(url), { hook_event_name: name });
      assert.equal(outcome.kind, 'fault', `${name} must not be a plain allow`);
      assert.match(outcome.detail, /not a JSON object/);
      assert.equal(outcome.origin, 'response', `${name} answered; it was not unreachable`);
    }
  });

  it('treats 2xx as allow', async () => {
    const outcome = await postHook(baseConfig(url), {
      hook_event_name: 'tool_call',
      tool_name: 'read',
      input: { path: 'README.md' },
    });
    // An allow may carry a body (a rewritten payload from a request intercept), so assert the
    // verdict rather than the exact object.
    assert.equal(outcome.kind, 'allow');
    assert.equal(outcome.reason, undefined);
  });

  it('turns a guardrail 403 into a block carrying the reason verbatim', async () => {
    const outcome = await postHook(baseConfig(url), {
      hook_event_name: 'tool_call',
      tool_name: 'read',
      input: { path: '/work/.env' },
    });
    assert.equal(outcome.kind, 'block');
    // Verbatim matters: pi passes this straight to the model, so any added
    // framing would become part of what the model reads.
    assert.equal(outcome.reason, 'read .env is blocked; use .env.example');
    assert.ok(!outcome.reason.startsWith('guardrail rejected:'), 'runtime framing must be stripped');
  });

  it('does not present a 403 without the guardrail marker as a policy decision', async () => {
    // An authorization failure is not a judgment about the request; reporting
    // it as one would tell the model a policy considered and refused its call.
    const outcome = await postHook(baseConfig(url), { hook_event_name: 'naked-403' });
    assert.equal(outcome.kind, 'fault');
    assert.equal(outcome.origin, 'response', 'a refusal is an answer');
  });

  it('reports a non-403 error status as a fault, not a block', async () => {
    const outcome = await postHook(baseConfig(url), { hook_event_name: 'boom' });
    assert.equal(outcome.kind, 'fault');
    assert.match(outcome.detail, /HTTP 500/);
    assert.equal(outcome.origin, 'response');
  });

  it('times out rather than hanging pi\'s critical path', async () => {
    const outcome = await postHook(baseConfig(url, { timeoutMs: 50 }), {
      hook_event_name: 'slow',
    });
    assert.equal(outcome.kind, 'fault');
    assert.match(outcome.detail, /did not respond within 50ms/);
    assert.equal(outcome.origin, 'timeout', 'slow is not the same as absent');
  });

  it('keeps the timeout active while reading the response body', async () => {
    const outcome = await postHook(baseConfig(url, { timeoutMs: 50 }), {
      hook_event_name: 'slow-body',
    });
    assert.equal(outcome.kind, 'fault');
    assert.match(outcome.detail, /did not respond within 50ms/);
    assert.equal(outcome.origin, 'timeout');
  });

  it('reports an unreachable gateway as a fault', async () => {
    // Port 1 is reserved and never listening.
    const outcome = await postHook(baseConfig('http://127.0.0.1:1'), {
      hook_event_name: 'tool_call',
    });
    assert.equal(outcome.kind, 'fault');
    assert.equal(outcome.origin, 'transport');
  });

  it('sends the session id in both the header and the payload', async () => {
    await postHook(baseConfig(url, { sessionId: 'sess-42' }), { hook_event_name: 'session_start' });
    const last = ctx.received.at(-1);
    assert.equal(last.url, '/hooks/pi');
    assert.equal(last.headers['x-nemo-relay-session-id'], 'sess-42');
    // The gateway strips inbound routing-identity headers, so the payload copy
    // is what actually survives.
    assert.equal(last.body.session_id, 'sess-42');
  });
});

describe('failure policy', () => {
  it('fails open by default so a dead sidecar does not brick the agent', () => {
    const outcome = resolveFault(
      { url: '', timeoutMs: 1, onFault: 'open', sessionId: 's' },
      { kind: 'fault', origin: 'transport', detail: 'connection refused' },
      'read',
    );
    assert.deepEqual(outcome, { kind: 'allow' });
  });

  it('fails closed on request, and says the block is infrastructure not policy', () => {
    const outcome = resolveFault(
      { url: '', timeoutMs: 1, onFault: 'closed', sessionId: 's' },
      { kind: 'fault', origin: 'transport', detail: 'connection refused' },
      'read',
    );
    assert.equal(outcome.kind, 'block');
    assert.match(outcome.reason, /infrastructure fault, not a judgment/);
    assert.match(outcome.reason, /could not be reached/);
    assert.match(outcome.reason, /connection refused/);
  });

  // A gateway that replied 413 was reached. Sending the reader to debug connectivity is
  // the wrong search, and this sentence is what the model reads too, so it has to name
  // the failure that actually happened.
  it('says the gateway answered when it answered, rather than that it was unreachable', () => {
    const outcome = resolveFault(
      { url: '', timeoutMs: 1, onFault: 'closed', sessionId: 's' },
      { kind: 'fault', origin: 'response', detail: 'gateway returned HTTP 413' },
      'write',
    );
    assert.equal(outcome.kind, 'block');
    assert.match(outcome.reason, /answered this write call without a usable decision/);
    assert.doesNotMatch(outcome.reason, /could not be reached/);
    // The tail is unchanged: nothing judged the request either way.
    assert.match(outcome.reason, /infrastructure fault, not a judgment/);
    assert.match(outcome.reason, /HTTP 413/);
  });

  // A timeout is not an unreachable gateway, and a handler failure is not a transport
  // result. All four block identically; the sentence is the only thing telling the reader
  // which of four places to look.
  it('gives each fault origin its own opening', () => {
    const config = { url: '', timeoutMs: 1, onFault: 'closed', sessionId: 's' };
    const opening = (origin) =>
      resolveFault(config, { kind: 'fault', origin, detail: 'd' }, 'read').reason;

    assert.match(opening('transport'), /could not be reached/);
    assert.match(opening('timeout'), /did not answer in time/);
    assert.match(opening('response'), /without a usable decision/);
    assert.match(opening('handler'), /gate failed before it could authorize/);

    const openings = ['transport', 'timeout', 'response', 'handler'].map(opening);
    assert.equal(new Set(openings).size, 4, 'each origin must read differently');
    for (const reason of openings) {
      assert.match(reason, /infrastructure fault, not a judgment/);
    }
  });
});

describe('configFromEnv', () => {
  const saved = { ...process.env };
  after(() => {
    process.env = saved;
  });

  it('defaults to the gateway default bind and fails open', () => {
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
    delete process.env.NEMO_RELAY_PI_TIMEOUT_MS;
    delete process.env.NEMO_RELAY_PI_FAIL;
    const config = configFromEnv('s1');
    assert.equal(config.url, 'http://127.0.0.1:4040');
    assert.equal(config.timeoutMs, 5000);
    assert.equal(config.onFault, 'open');
  });

  it('strips a trailing slash so the path join cannot double up', () => {
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:9999/';
    assert.equal(configFromEnv('s1').url, 'http://127.0.0.1:9999');
  });

  it('ignores a non-numeric or non-positive timeout', () => {
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = 'soon';
    assert.equal(configFromEnv('s1').timeoutMs, 5000);
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = '0';
    assert.equal(configFromEnv('s1').timeoutMs, 5000);
  });

  // Node holds a timer delay in a 32-bit int, so an unclamped value above 2^31-1 wraps to
  // ~1 ms rather than to a long wait -- and a 1 ms timeout faults every gated call, which
  // under the default fail-open policy stops enforcement without a word. A value too large
  // to honor has to behave like "effectively never", not like "instantly".
  it('clamps a timeout too large for a timer instead of letting it wrap to ~1ms', () => {
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = String(2 ** 31);
    assert.equal(configFromEnv('s1').timeoutMs, 2_147_483_647);
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = String(Number.MAX_SAFE_INTEGER);
    assert.equal(configFromEnv('s1').timeoutMs, 2_147_483_647);
  });

  it('opts into fail-closed only on the exact value', () => {
    process.env.NEMO_RELAY_PI_FAIL = 'closed';
    assert.equal(configFromEnv('s1').onFault, 'closed');
    process.env.NEMO_RELAY_PI_FAIL = 'CLOSED';
    assert.equal(configFromEnv('s1').onFault, 'open');
  });
});
