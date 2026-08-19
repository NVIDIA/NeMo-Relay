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
   * `reached` is false when nothing came back -- the connection failed, or nothing
   * answered in time -- and true when the gateway did answer and the answer was not
   * a decision, such as a rejected payload or a body that will not parse. Both block
   * under `NEMO_RELAY_PI_FAIL=closed`, but they send whoever reads the block to two
   * different places, so the reason has to say which one happened.
   */
  | { kind: 'fault'; detail: string; reached: boolean };

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
    // The gateway strips inbound routing-identity headers as an anti-spoofing
    // measure and re-derives its own, so this is the only session signal that
    // survives -- it must also appear in the payload.
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

    if (response.ok) {
      // An allow body is `{}` unless a request intercept rewrote the arguments -- so a body we
      // cannot read is NOT a plain allow. It may have carried a transform, and treating it as an
      // empty allow would run the original arguments while silently discarding a policy decision:
      // exactly the failure a refused transform blocks the call to prevent. An unreadable success
      // is an infrastructure fault, resolved by `NEMO_RELAY_PI_FAIL` like any other.
      const body = await safeJson(response);
      if (body === null || typeof body !== 'object' || Array.isArray(body)) {
        return {
          kind: 'fault',
          reached: true,
          detail: 'gateway returned a success body that is not a JSON object',
        };
      }
      return { kind: 'allow', body };
    }

    if (response.status === 403) {
      const body = await safeJson(response);
      const error = body?.error;
      if (error?.type === GUARDRAIL_REJECTION_TYPE && typeof error.reason === 'string') {
        return { kind: 'block', reason: error.reason };
      }
      // A 403 without the guardrail marker is an authorization fault, not a
      // policy decision; do not present it to the model as one.
      return { kind: 'fault', reached: true, detail: `gateway returned 403 without a guardrail reason` };
    }

    return { kind: 'fault', reached: true, detail: `gateway returned HTTP ${response.status}` };
  } catch (error) {
    const detail =
      error instanceof Error && error.name === 'AbortError'
        ? `gateway did not respond within ${config.timeoutMs}ms`
        : `gateway request failed: ${error instanceof Error ? error.message : String(error)}`;
    // Nothing usable came back: a transport failure, or a timeout that may have arrived
    // and never answered. Either way there is no response to have misread.
    return { kind: 'fault', reached: false, detail };
  } finally {
    clearTimeout(timer);
  }
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
  // Two openings, one tail. The tail is the part a model has to act on and it is the
  // same either way: nothing judged the request, so the request is not what to change.
  // The opening differs because "could not be reached", said of a gateway that replied
  // 413, sends the reader to debug connectivity -- the one thing that is working. This
  // string reaches the model verbatim, so it is also what the user reads.
  const opening = fault.reached
    ? `The NeMo Relay policy gateway answered this ${toolName} call without a usable decision`
    : `The NeMo Relay policy gateway could not be reached to authorize this ${toolName} call`;
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

/** Read gateway configuration from the environment the CLI's launcher sets. */
export function configFromEnv(sessionId: string): GatewayConfig {
  const url = (process.env.NEMO_RELAY_PI_GATEWAY_URL ?? 'http://127.0.0.1:4040').replace(/\/+$/, '');
  const timeoutRaw = Number(process.env.NEMO_RELAY_PI_TIMEOUT_MS);
  return {
    url,
    timeoutMs: Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? timeoutRaw : 5000,
    onFault: process.env.NEMO_RELAY_PI_FAIL === 'closed' ? 'closed' : 'open',
    sessionId,
  };
}
