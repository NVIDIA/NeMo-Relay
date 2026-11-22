// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
// pi's /reload replaces modules while retaining the session. Keep its owner identity
// on the process global so reloading cannot strand the authenticated session.
const CLIENT_ID = Symbol.for('nemo-relay.pi.hook-client.v1');
const clientState = globalThis as typeof globalThis & { [CLIENT_ID]?: string };
const fallbackClientId = clientState[CLIENT_ID] ??= randomUUID();

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
 * Hooks are delivered through `nemo-relay hook-forward pi`, which authenticates
 * the gateway and attaches the client's signed identity before sending the payload.
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
  /** Relay executable used for authenticated hook forwarding. */
  relayCommand?: string;
  /** Set by the launcher when the Relay executable supports framed forwarding. */
  streamHooks?: boolean;
  /** Stable identity for this extension client, signed by the Relay subprocess. */
  clientId?: string;
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
// A bounded 1 MiB gateway body can expand sixfold inside the JSON envelope.
const MAX_FORWARD_OUTPUT_BYTES = 8 * 1024 * 1024;
const STREAM_IDLE_MS = 5_000;

class HookForwardStream {
  readonly child: ChildProcessWithoutNullStreams;
  closed = false;
  private pending: {
    resolve: (value: string) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | undefined;
  private chunks: Buffer[] = [];
  private size = 0;
  private stderr = '';
  private idle?: ReturnType<typeof setTimeout>;

  constructor(command: string, args: string[]) {
    this.child = spawn(command, [...args, '--pi-extension-stream'], { windowsHide: true });
    this.child.on('error', (error) => this.close(error));
    this.child.on('close', () => this.close(new Error(this.stderr.trim() || 'hook forward exited')));
    this.child.stdin.on('error', (error) => this.close(error));
    this.child.stderr.on('data', (chunk: Buffer) => {
      // Drain all diagnostics, retaining only the tail for transport failures.
      this.stderr = this.stderr.slice(-4_096) + chunk.subarray(-4_096).toString('utf8');
    });
    this.child.stdout.on('data', (chunk: Buffer) => this.read(chunk));
  }

  send(input: string, timeoutMs: number): Promise<string> {
    if (this.pending) return Promise.reject(new Error('a Pi hook is already in flight'));
    clearTimeout(this.idle);
    this.stderr = '';
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.close(Object.assign(new Error('hook forward timed out'), { killed: true, signal: 'SIGKILL' }));
      }, timeoutMs);
      this.pending = { resolve, reject, timer };
      this.child.stdin.write(input + '\n');
    });
  }

  private read(chunk: Buffer): void {
    if (this.closed) return;
    const pending = this.pending;
    const end = chunk.indexOf(10);
    if (!pending || (end !== -1 && end !== chunk.length - 1)) {
      this.close(new Error('hook forward returned an unexpected response frame'));
      return;
    }
    const data = end === -1 ? chunk : chunk.subarray(0, end);
    this.size += data.length;
    if (this.size > MAX_FORWARD_OUTPUT_BYTES) {
      this.close(Object.assign(new Error('hook forward output exceeded the output limit'), {
        code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER',
      }));
      return;
    }
    this.chunks.push(data);
    if (end === -1) return;
    const raw = Buffer.concat(this.chunks, this.size).toString('utf8');
    this.chunks = [];
    this.size = 0;
    this.pending = undefined;
    clearTimeout(pending.timer);
    // Retire idle helpers even when a host omits its session_shutdown hook.
    this.idle = setTimeout(() => this.close(), STREAM_IDLE_MS);
    pending.resolve(raw);
  }

  close(error = new Error('hook forward closed')): void {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.idle);
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(error);
      this.pending = undefined;
    }
    this.chunks = [];
    this.child.kill('SIGKILL');
  }
}

const streams = new WeakMap<GatewayConfig, HookForwardStream>();

export function closeHookStream(config: GatewayConfig): void {
  streams.get(config)?.close();
  streams.delete(config);
}

function forwardHook(
  config: GatewayConfig,
  args: string[],
  input: string,
): Promise<string> {
  const command = config.relayCommand ?? process.env.NEMO_RELAY_PI_COMMAND ?? 'nemo-relay';
  if (config.streamHooks) {
    let stream = streams.get(config);
    if (!stream || stream.closed) {
      stream = new HookForwardStream(command, args);
      streams.set(config, stream);
    }
    return stream.send(input, config.timeoutMs);
  }
  return new Promise((resolve, reject) => {
    const child = execFile(command, args, {
      timeout: config.timeoutMs,
      maxBuffer: MAX_FORWARD_OUTPUT_BYTES,
      killSignal: 'SIGKILL',
      windowsHide: true,
      encoding: 'utf8',
    }, (error, stdout, stderr) => {
      if (error) reject(Object.assign(error, { detail: stderr.trim() || error.message }));
      else resolve(stdout);
    });
    child.stdin?.on('error', () => undefined);
    child.stdin?.end(input);
  });
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
  const args = [
    'hook-forward', 'pi',
    '--gateway-url', config.url,
    '--pi-extension-client', config.clientId ?? fallbackClientId,
    '--fail-closed',
  ];
  if (process.env.NEMO_RELAY_PROXY_CREDENTIAL) args.push('--transparent-run');
  try {
    const raw = await forwardHook(config, args, JSON.stringify({ session_id: config.sessionId, ...payload }));
    let envelope: { status?: unknown; body?: unknown; transportError?: unknown };
    try { envelope = JSON.parse(raw); }
    catch { return { kind: 'fault', origin: 'response', detail: 'hook forward returned an unreadable response' }; }
    if (config.streamHooks && typeof envelope?.transportError === 'string') {
      return { kind: 'fault', origin: 'transport', detail: `gateway request failed: ${envelope.transportError}` };
    }
    if (!envelope || typeof envelope.status !== 'number' || !Number.isInteger(envelope.status)
      || envelope.status < 200 || envelope.status > 599 || typeof envelope.body !== 'string') {
      return { kind: 'fault', origin: 'response', detail: 'hook forward returned an invalid response' };
    }
    const body = [204, 205, 304].includes(envelope.status) ? null : envelope.body;
    const response = new Response(body, { status: envelope.status });
    if (response.ok) return await allowedOutcome(response);
    if (response.status === 403) return await forbiddenOutcome(response);
    return { kind: 'fault', origin: 'response', detail: `gateway returned HTTP ${response.status}` };
  } catch (error) {
    const failure = error as { detail?: string; killed?: boolean; signal?: string; code?: unknown };
    if (failure.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') {
      return { kind: 'fault', origin: 'response', detail: 'hook forward output exceeded the output limit' };
    }
    const detail = failure.detail ?? (error instanceof Error ? error.message : String(error));
    const timedOut = failure.killed === true && failure.signal === 'SIGKILL';
    return {
      kind: 'fault', origin: timedOut ? 'timeout' : 'transport',
      detail: timedOut ? `gateway did not respond within ${config.timeoutMs}ms` : `gateway request failed: ${detail}`,
    };
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
    relayCommand: process.env.NEMO_RELAY_PI_COMMAND ?? 'nemo-relay',
    streamHooks: process.env.NEMO_RELAY_PI_HOOK_STREAM === '1',
    clientId: fallbackClientId,
    timeoutMs,
    onFault: process.env.NEMO_RELAY_PI_FAIL === 'closed' ? 'closed' : 'open',
    sessionId,
  };
}
