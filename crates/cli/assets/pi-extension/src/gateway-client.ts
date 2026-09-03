// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP client for the NeMo Relay CLI gateway's `/hooks/pi` endpoint.
 *
 * The wire contract, verified against `crates/cli`:
 *
 * - **Allow** is any 2xx. The body is `{}` unless a request intercept rewrote
 *   the arguments, in which case it carries
 *   `{"tool_call": {"tool_call_id": "...", "input": {...}}}` and the caller is
 *   expected to execute those arguments instead. See `argument-transform.ts`.
 * - **Block** is `403` with
 *   `{"error": {"type": "nemo_relay_guardrail_rejected", "reason": "<why>"}}`.
 *   The rejection comes from the tool conditional-execution guardrail chain that
 *   the gateway runs in `start_tool`, and `error.reason` is the guardrail's own
 *   words. pi passes that string to the model verbatim as an error tool result,
 *   so it must be forwarded unchanged.
 * - Anything else is a transport or gateway fault, and is resolved by the
 *   configured failure policy rather than being reported as a policy decision.
 *
 * pi awaits extension handlers on its critical path, so every call here is on
 * the critical path of the tool it gates. Observability-only hooks are therefore
 * sent without awaiting, and only the gating hook blocks.
 */

/** Outcome of posting one hook to the gateway. */
export type HookOutcome =
  /** Allowed. `body` carries a rewritten payload when a request intercept produced one. */
  | { kind: 'allow'; body?: { tool_call?: { tool_call_id?: unknown; input?: unknown } } }
  | { kind: 'block'; reason: string }
  /**
   * Neither a verdict nor a usable success.
   *
   * `origin` says where it went wrong, because all four send whoever reads the
   * block somewhere different and they all block identically under
   * `NEMO_RELAY_PI_FAIL=closed`. A single "could not be reached" was wrong for
   * three of them.
   */
  | { kind: 'fault'; detail: string; origin: FaultOrigin };

/**
 * Where a fault happened, which is what a reader has to know to act on it.
 *
 * - `transport` -- nothing answered. The gateway is down, or the URL is wrong.
 * - `timeout` -- nothing answered *in time*. The gateway may be up and slow, and
 *   posts are serialized, so a gate also waits out everything queued ahead of it.
 * - `response` -- the gateway answered, and the answer was not a decision: a
 *   rejected payload, an unreadable body, a 403 with no guardrail marker.
 * - `handler` -- the gateway was never asked. This extension threw.
 */
export type FaultOrigin = 'transport' | 'timeout' | 'response' | 'handler';

/** The fault arm of {@link HookOutcome}, named so a caller can build one. */
export type HookFault = Extract<HookOutcome, { kind: 'fault' }>;

export type GatewayConfig = {
  /** Base URL of the gateway, e.g. `http://127.0.0.1:4040`. */
  url: string;
  /** Per-request timeout in milliseconds. */
  timeoutMs: number;
  /**
   * What to do when the gateway cannot be reached or errors.
   *
   * `open` lets the tool run; `closed` blocks it. Defaults to `open` because a
   * dead sidecar should not brick the user's agent, matching how the shipped
   * `hooks.json` files use `--fail-open` for everything except the pre-tool and
   * permission events.
   */
  onFault: 'open' | 'closed';
  /** Session identifier sent with every payload. */
  sessionId: string;
};

const GUARDRAIL_REJECTION_TYPE = 'nemo_relay_guardrail_rejected';

/** Build the routing-identity headers the gateway expects. */
function headers(config: GatewayConfig): Record<string, string> {
  return {
    'content-type': 'application/json',
    // On the hook route the gateway *reads* this header to key the session
    // (`adapters::pi_session_id`), and without it a second live agent drops each
    // call into a throwaway root. It is stripped and re-derived only on the
    // provider-passthrough route, which is a different request entirely -- which
    // is why the provider registration attaches its own copy. It must also appear
    // in the payload, because a payload session id is what a direct `curl` of the
    // hook route has.
    'x-nemo-relay-session-id': config.sessionId,
  };
}

/**
 * Post one hook and wait for the verdict.
 *
 * Used only for the gating hook (`tool_call`). Everything else should use
 * {@link postAndForget} so pi's critical path is not charged an extra round
 * trip for an observability-only event.
 */
export async function postHook(
  config: GatewayConfig,
  payload: Record<string, unknown>,
): Promise<HookOutcome> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetch(`${config.url}/hooks/pi`, {
      method: 'POST',
      headers: headers(config),
      body: JSON.stringify({ session_id: config.sessionId, ...payload }),
      signal: controller.signal,
    });

    if (response.ok) return await allowedOutcome(response);
    if (response.status === 403) return await forbiddenOutcome(response);
    return { kind: 'fault', origin: 'response', detail: `gateway returned HTTP ${response.status}` };
  } catch (error) {
    const timedOut = error instanceof Error && error.name === 'AbortError';
    let detail = `gateway request failed: ${String(error)}`;
    if (error instanceof Error) detail = `gateway request failed: ${error.message}`;
    if (timedOut) detail = `gateway did not respond within ${config.timeoutMs}ms`;
    // A timeout is not an unreachable gateway. It may be up and slow, and because posts are
    // serialized a gate also waits out everything queued ahead of it -- so the remedy is the
    // timeout value or the gateway's speed, not the socket.
    return { kind: 'fault', origin: timedOut ? 'timeout' : 'transport', detail };
  } finally {
    clearTimeout(timer);
  }
}

async function allowedOutcome(response: Response): Promise<HookOutcome> {
  const body = await safeJson(response);
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    return {
      kind: 'fault',
      origin: 'response',
      detail: 'gateway returned a success body that is not a JSON object',
    };
  }
  return { kind: 'allow', body };
}

async function forbiddenOutcome(response: Response): Promise<HookOutcome> {
  const body = await safeJson(response);
  const error = body?.error;
  if (error?.type === GUARDRAIL_REJECTION_TYPE && typeof error.reason === 'string') {
    return { kind: 'block', reason: error.reason };
  }
  return { kind: 'fault', origin: 'response', detail: 'gateway returned 403 without a guardrail reason' };
}

/**
 * Post one hook without waiting for it.
 *
 * Returns a promise that never rejects, so a failed observability post cannot
 * surface as an unhandled rejection inside pi's TUI. Callers should collect
 * these and await them at session shutdown so nothing is lost on exit.
 */
export function postAndForget(
  config: GatewayConfig,
  payload: Record<string, unknown>,
): Promise<void> {
  return postHook(config, payload).then(
    () => undefined,
    () => undefined,
  );
}

/** Resolve a fault into an allow/block decision using the configured policy. */
export function resolveFault(
  config: GatewayConfig,
  fault: HookFault,
  toolName: string,
): HookOutcome {
  if (config.onFault === 'open') return { kind: 'allow' };
  // One opening per origin, one tail. The tail is the part a model has to act on and it is
  // the same for all four: nothing judged the request, so the request is not what to change.
  // The openings differ because they are debugged in four different places, and "could not
  // be reached" said of a gateway that replied 413 sends the reader to the one thing that is
  // working. This string reaches the model verbatim, so it is also what the user reads.
  const openings: Record<FaultOrigin, string> = {
    transport: `The NeMo Relay policy gateway could not be reached to authorize this ${toolName} call`,
    timeout: `The NeMo Relay policy gateway did not answer in time to authorize this ${toolName} call`,
    response: `The NeMo Relay policy gateway answered this ${toolName} call without a usable decision`,
    handler: `The NeMo Relay policy gate failed before it could authorize this ${toolName} call`,
  };
  const opening = openings[fault.origin];
  return {
    kind: 'block',
    reason:
      `${opening}, so it was blocked rather than allowed through unchecked. This is an ` +
      `infrastructure fault, not a judgment about the request. Details: ${fault.detail}`,
  };
}

async function safeJson(response: Response): Promise<{
  error?: Record<string, unknown>;
  tool_call?: { tool_call_id?: unknown; input?: unknown };
} | null> {
  try {
    return (await response.json()) as {
      error?: Record<string, unknown>;
      tool_call?: { tool_call_id?: unknown; input?: unknown };
    };
  } catch {
    return null;
  }
}

/** The largest delay Node can hold in a timer without wrapping. */
const MAX_TIMEOUT_MS = 2_147_483_647;

/** Read gateway configuration from the environment the CLI's launcher sets. */
export function configFromEnv(sessionId: string): GatewayConfig {
  const url = (process.env.NEMO_RELAY_PI_GATEWAY_URL ?? 'http://127.0.0.1:4040').replace(/\/+$/, '');
  const timeoutRaw = Number(process.env.NEMO_RELAY_PI_TIMEOUT_MS);
  // Clamped, not just validated. Node stores a timer delay in a 32-bit int, so anything
  // above 2^31-1 wraps to ~1 ms rather than to a long wait -- and a 1 ms timeout makes every
  // gated call fault, which under the default fail-open policy silently stops enforcing.
  // A value too large to honor should behave like "effectively never", not like "instantly".
  const timeoutMs =
    Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? Math.min(timeoutRaw, MAX_TIMEOUT_MS) : 5000;
  return {
    url,
    timeoutMs,
    onFault: process.env.NEMO_RELAY_PI_FAIL === 'closed' ? 'closed' : 'open',
    sessionId,
  };
}
