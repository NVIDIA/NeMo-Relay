// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared test harness: a stub gateway, and a driver that fires pi's hooks.
 *
 * ⚠️ **The driver returns the *first* non-`undefined` result, not the last.**
 * That is pi's rule, in both `emitToolCall` and `emitUserBash`, and it is the
 * difference between "our gate decided" and "an extension ahead of us decided
 * and we never ran". A last-wins driver silently inverts the trap this
 * extension documents in two places, and would let a regression that breaks
 * preemption behaviour pass.
 */
import { createServer } from 'node:http';

/**
 * A gateway whose reply to the *gated* hook is set per test.
 *
 * Every other post is answered 200 `{}`: they are observability, and answering
 * them specially would mean a test asserting on a block could not tell which
 * post the block came from.
 *
 * @param gatedHook the `hook_event_name` whose reply `replyWith` controls
 */
export function stubGateway(gatedHook) {
  const posts = [];
  let reply = { status: 200, payload: {} };
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      const parsed = JSON.parse(body || '{}');
      posts.push(parsed);
      const gated = gatedHook !== undefined && parsed.hook_event_name === gatedHook;
      const { status, payload, delayMs } = gated ? reply : { status: 200, payload: {} };
      const send = () => {
        res.writeHead(status, { 'content-type': 'application/json' });
        res.end(JSON.stringify(payload ?? {}));
      };
      if (delayMs) setTimeout(send, delayMs);
      else send();
    });
  });
  return {
    server,
    posts,
    replyWith(next) {
      reply = next;
    },
    reset() {
      posts.length = 0;
      reply = { status: 200, payload: {} };
    },
  };
}

/** Start a stub server on a free port and return its base URL. */
export async function listen(server) {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${server.address().port}`;
}

/**
 * Register the extension and return a driver that fires hooks the way pi does.
 *
 * `before` registers handlers *ahead* of the extension, which is the
 * `pi install` load order: package-sourced extensions load last, so anything
 * already installed answers first. Loading with `-e` inverts it, and is what
 * `nemo-relay run --agent pi` uses precisely so the gate runs first.
 *
 * @param extension the extension factory under test
 * @param options.before `{ [hookName]: handler }` registered before the extension
 * @param options.ctx overrides merged into the extension context
 */
export function load(extension, options = {}) {
  const handlers = new Map();
  const register = (name, handler) => {
    if (!handlers.has(name)) handlers.set(name, []);
    handlers.get(name).push(handler);
  };
  for (const [name, handler] of Object.entries(options.before ?? {})) {
    register(name, handler);
  }
  extension({
    on: register,
    registerProvider() {},
  });
  const ctx = {
    cwd: '/work',
    mode: 'interactive',
    hasUI: true,
    sessionManager: { getSessionId: () => 'sess-under-test' },
    ...options.ctx,
  };
  return async (name, event = {}) => {
    for (const handler of handlers.get(name) ?? []) {
      const result = await handler({ type: name, ...event }, ctx);
      // First truthy result wins and stops iteration -- pi's rule. Note that
      // `{}` is truthy: an allow must be `undefined`, or it silently preempts
      // every extension behind it while deciding nothing.
      if (result !== undefined) return result;
    }
    return undefined;
  };
}

/**
 * Drain the extension's serial post queue.
 *
 * Gating hooks await their own verdict, but everything else is enqueued and not
 * awaited. `session_shutdown` awaits the chain -- that is how the extension
 * guarantees nothing is lost on exit -- so it doubles as the drain.
 */
export const drain = (fire) => fire('session_shutdown', { reason: 'quit' });

/** Every post with a given hook event name, in arrival order. */
export const named = (posts, name) => posts.filter((post) => post.hook_event_name === name);

/** The 403 body `CliError::into_response` produces, byte for byte. */
export const rejection = (reason) => ({
  status: 403,
  payload: {
    error: {
      message: `guardrail rejected: ${reason}`,
      type: 'nemo_relay_guardrail_rejected',
      reason,
    },
  },
});
