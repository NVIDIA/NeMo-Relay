// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared test harness: a stub gateway, and a driver that fires pi's hooks.
 *
 * **The two gated hooks resolve competing handlers by different rules, and
 * the driver has to model each one.** A driver that picks either rule for both
 * misrepresents one of them, and preemption is exactly what these tests exist
 * to pin.
 *
 * | Hook | pi's rule | Catches? |
 * |---|---|---|
 * | `tool_call` | Runs **every** handler, keeps the **last** truthy result, and returns early **only** on `{block: true}` | No — an exception propagates |
 * | `user_bash` | Returns the **first** truthy result and stops | Yes — a throw fails *open* |
 *
 * So an earlier extension preempts the tool gate only by *blocking*; a
 * non-blocking result from one does not stop us, it is simply overwritten by
 * whichever handler answers last. On the inline-shell path any truthy result at
 * all preempts, which is why `{}` is dangerous there.
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
 *
 * A reply is `{status, payload, delayMs}`, or `{status, raw, delayMs}` to send a
 * body verbatim -- which is the only way to reach the unreadable-success path.
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
      const { status, payload, raw, delayMs } = gated ? reply : { status: 200, payload: {} };
      const send = () => {
        res.writeHead(status, { 'content-type': 'application/json' });
        // `raw` lets a case put a body on the wire that `JSON.parse` cannot read.
        // The client treats an unreadable success as a fault rather than an empty
        // allow, and nothing `JSON.stringify` produces can reach that branch.
        res.end(raw ?? JSON.stringify(payload ?? {}));
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
    let last;
    for (const handler of handlers.get(name) ?? []) {
      const result = await handler({ type: name, ...event }, ctx);
      if (!result) continue;
      if (name === 'user_bash') {
        // First truthy result wins and stops iteration. `{}` is truthy, so an
        // allow must be `undefined` or it silently preempts every extension
        // behind it while deciding nothing.
        return result;
      }
      if (name === 'tool_call') {
        // Every handler runs; only a block short-circuits. A non-blocking
        // truthy result does not preempt -- it is overwritten by whoever
        // answers last, which is why returning one is inert rather than fatal.
        last = result;
        if (result.block) return result;
        continue;
      }
      last = result;
    }
    return last;
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
