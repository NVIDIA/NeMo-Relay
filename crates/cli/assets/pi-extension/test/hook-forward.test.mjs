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
 * Run: npm test --workspace=nemo-relay-pi
 */
import './harness.mjs';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { after, before, describe, it, mock } from 'node:test';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';

const { clientId, postHook } = await import('../src/standalone/transport.ts');
const { resolveFault } = await import('../src/gateway.ts');
const { faultPolicyFromEnv, gatewayConfigFromEnv } = await import('../src/standalone/config.ts');

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
  ...gatewayConfigFromEnv(),
  url,
  timeoutMs: 2000,
  ...overrides,
});

describe('hook forward wire contract', () => {
  it('classifies oversized stdout and stderr as response faults', async () => {
    for (const stream of ['stdout', 'stderr']) {
      const config = baseConfig('http://127.0.0.1:4040');
      const outcome = await postHook(config, {
        hook_event_name: 'fixture_output_overflow',
        stream,
      });
      assert.equal(outcome.kind, 'fault');
      assert.equal(outcome.origin, 'response', stream);
      assert.equal(outcome.detail, 'hook forward output exceeded the output limit');
      assert.equal(resolveFault('open', outcome, 'read').kind, 'allow');
      const blocked = resolveFault('closed', outcome, 'read');
      assert.equal(blocked.kind, 'block');
      assert.match(blocked.reason, /gateway answered/);
    }
  });

  it('prioritizes an output overflow over process-kill timeout fields', async () => {
    const previousExecFile = childProcess.execFile;
    const decorated = mock.method(childProcess, 'execFile', (command, args, options, callback) =>
      previousExecFile(command, args, options, (error, stdout, stderr) => {
        if (error?.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') {
          error.killed = true;
          error.signal = 'SIGKILL';
        }
        callback(error, stdout, stderr);
      }),
    );
    syncBuiltinESMExports();
    try {
      const outcome = await postHook(baseConfig('http://127.0.0.1:4040'), {
        hook_event_name: 'fixture_output_overflow',
        stream: 'stdout',
      });
      assert.equal(outcome.kind, 'fault');
      assert.equal(outcome.origin, 'response');
      assert.equal(outcome.detail, 'hook forward output exceeded the output limit');
    } finally {
      decorated.mock.restore();
      syncBuiltinESMExports();
    }
  });

  it('reports a missing Relay executable through the existing fault policy', async () => {
    const config = baseConfig('http://127.0.0.1:4040', {
      relayCommand: 'missing-nemo-relay-executable-for-test',
    });
    const outcome = await postHook(config, { hook_event_name: 'tool_call' });
    assert.equal(outcome.kind, 'fault');
    assert.equal(outcome.origin, 'transport');
    assert.equal(resolveFault('open', outcome, 'read').kind, 'allow');
    assert.equal(resolveFault('closed', outcome, 'read').kind, 'block');
  });

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

  it("times out rather than hanging pi's critical path", async () => {
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

  it('posts to the pi hook route', async () => {
    await postHook(baseConfig(url), { hook_event_name: 'session_start', session_id: 'sess-42' });
    const last = ctx.received.at(-1);
    assert.equal(last.url, '/hooks/pi');
    assert.equal(last.body.session_id, 'sess-42');
  });

  it('passes --transparent-run to the forwarder only when the launcher set a proxy credential', async () => {
    const seen = [];
    const previousExecFile = childProcess.execFile;
    const decorated = mock.method(childProcess, 'execFile', (command, args, options, callback) => {
      seen.push(args);
      return previousExecFile(command, args, options, callback);
    });
    syncBuiltinESMExports();
    try {
      delete process.env.NEMO_RELAY_PROXY_CREDENTIAL;
      await postHook(baseConfig(url), { hook_event_name: 'tool_call' });
      process.env.NEMO_RELAY_PROXY_CREDENTIAL = 'nrp_testtoken';
      await postHook(baseConfig(url), { hook_event_name: 'tool_call' });
    } finally {
      decorated.mock.restore();
      syncBuiltinESMExports();
      delete process.env.NEMO_RELAY_PROXY_CREDENTIAL;
    }
    assert.equal(seen.length, 2);
    assert.equal(seen[0].includes('--transparent-run'), false);
    assert.equal(seen[1].includes('--transparent-run'), true);
  });

  it('reports a payload that cannot be serialized as a fault instead of throwing', async () => {
    const outcome = await postHook(baseConfig(url), {
      hook_event_name: 'tool_call',
      input: {
        toJSON() {
          throw null;
        },
      },
    });
    assert.equal(outcome.kind, 'fault');
    assert.equal(outcome.origin, 'handler');
    assert.match(outcome.detail, /JSON-safe/);
  });
});

describe('failure policy', () => {
  it('fails open by default so a dead sidecar does not brick the agent', () => {
    const outcome = resolveFault('open', { kind: 'fault', origin: 'transport', detail: 'connection refused' }, 'read');
    assert.deepEqual(outcome, { kind: 'allow', body: {} });
  });

  it('fails closed on request, and says the block is infrastructure not policy', () => {
    const outcome = resolveFault(
      'closed',
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
      'closed',
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

  // A timeout is not an unreachable gateway, and unreadable configuration is not a
  // transport result. All four block identically; the sentence is the only thing telling
  // the reader which of four places to look.
  it('gives each fault origin its own opening', () => {
    const opening = (origin) => resolveFault('closed', { kind: 'fault', origin, detail: 'd' }, 'read').reason;

    assert.match(opening('transport'), /could not be reached/);
    assert.match(opening('timeout'), /did not answer in time/);
    assert.match(opening('response'), /without a usable decision/);
    assert.match(opening('configuration'), /could not load its configuration/);

    const openings = ['transport', 'timeout', 'response', 'configuration'].map(opening);
    assert.equal(new Set(openings).size, 4, 'each origin must read differently');
    for (const reason of openings) {
      assert.match(reason, /infrastructure fault, not a judgment/);
    }
  });
});

describe('configuration from the environment', () => {
  it('preserves the authenticated owner when pi reloads its extension modules', async () => {
    const reloaded = await import('../src/standalone/transport.ts?reload-owner-test');
    assert.equal(reloaded.clientId, clientId);
  });

  const saved = { ...process.env };
  after(() => {
    process.env = saved;
  });

  it('defaults to the gateway default bind and fails open', () => {
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
    delete process.env.NEMO_RELAY_PI_TIMEOUT_MS;
    delete process.env.NEMO_RELAY_PI_FAIL;
    const config = gatewayConfigFromEnv();
    assert.equal(config.url, 'http://127.0.0.1:4040');
    assert.equal(config.timeoutMs, 5000);
    assert.equal(faultPolicyFromEnv(), 'open');
  });

  it('strips a trailing slash so the path join cannot double up', () => {
    process.env.NEMO_RELAY_PI_GATEWAY_URL = 'http://127.0.0.1:9999/';
    assert.equal(gatewayConfigFromEnv().url, 'http://127.0.0.1:9999');
  });

  it('ignores a non-numeric, non-positive, or fractional timeout', () => {
    for (const raw of ['soon', '0', '-1', '5000.5']) {
      process.env.NEMO_RELAY_PI_TIMEOUT_MS = raw;
      assert.equal(gatewayConfigFromEnv().timeoutMs, 5000, raw);
    }
  });

  // Node holds a timer delay in a 32-bit int, so an unclamped value above 2^31-1 wraps to
  // ~1 ms rather than to a long wait -- and a 1 ms timeout faults every gated call, which
  // under the default fail-open policy stops enforcement without a word. A value too large
  // to honor has to behave like "effectively never", not like "instantly".
  it('clamps a timeout too large for a timer instead of letting it wrap to ~1ms', () => {
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = String(2 ** 31);
    assert.equal(gatewayConfigFromEnv().timeoutMs, 2_147_483_647);
    process.env.NEMO_RELAY_PI_TIMEOUT_MS = String(Number.MAX_SAFE_INTEGER);
    assert.equal(gatewayConfigFromEnv().timeoutMs, 2_147_483_647);
  });

  it('opts into fail-closed only on the exact value', () => {
    process.env.NEMO_RELAY_PI_FAIL = 'closed';
    assert.equal(faultPolicyFromEnv(), 'closed');
    process.env.NEMO_RELAY_PI_FAIL = 'CLOSED';
    assert.equal(faultPolicyFromEnv(), 'open');
  });
});
