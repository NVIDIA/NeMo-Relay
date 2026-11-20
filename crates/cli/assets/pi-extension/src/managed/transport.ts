// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type HookOutcome, errorDetail, isRecord, outcomeFromResponse } from '../gateway.ts';

export const CLIENT_TOKEN_HEADER = 'x-nemo-relay-client-token';
export const UPSTREAM_BASE_URL_HEADER = 'x-nemo-relay-upstream-base-url';

const HOOK_TIMEOUT_MS = 30_000;
const ROUTE_READY_BUDGET_MS = 180_000;
const MIN_RETRY_AFTER_MS = 100;
const MAX_RETRY_AFTER_MS = 5_000;
const ROUTE_NOT_READY = 'route is not ready';
const MAX_HOOK_PAYLOAD_BYTES = 20 * 1024 * 1024;
const MAX_HOOK_RESPONSE_BYTES = 1024 * 1024;

export type DaemonEndpoint = {
  daemonAddress: string;
  credential: string;
};

class OversizedResponse extends Error {
  constructor() {
    super('daemon hook response exceeds the managed response limit');
    this.name = 'OversizedResponse';
  }
}

export async function postDaemonHook(
  endpoint: DaemonEndpoint,
  payload: Record<string, unknown>,
  routeReadyBudgetMs: number = ROUTE_READY_BUDGET_MS,
): Promise<HookOutcome> {
  let encoded: string;
  try {
    encoded = JSON.stringify(payload);
  } catch (error) {
    return { kind: 'fault', origin: 'handler', detail: `hook payload is not JSON-safe: ${errorDetail(error)}` };
  }
  if (Buffer.byteLength(encoded) > MAX_HOOK_PAYLOAD_BYTES) {
    return {
      kind: 'fault',
      origin: 'transport',
      detail: 'hook payload exceeds the managed payload limit',
    };
  }
  const deadline = Date.now() + routeReadyBudgetMs;
  for (;;) {
    const reply = await attempt(endpoint, encoded);
    if ('kind' in reply) return reply;
    if (reply.status === 503 && isRouteNotReady(reply.body) && Date.now() + reply.retryAfterMs <= deadline) {
      await new Promise((resolve) => setTimeout(resolve, reply.retryAfterMs));
      continue;
    }
    return outcomeFromResponse(reply.status, reply.body);
  }
}

type DaemonReply = { status: number; body: unknown; retryAfterMs: number };

async function attempt(endpoint: DaemonEndpoint, encoded: string): Promise<DaemonReply | HookOutcome> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
  try {
    const response = await fetch(`${endpoint.daemonAddress}/hooks/pi`, {
      method: 'POST',
      redirect: 'error',
      headers: {
        'content-type': 'application/json',
        [CLIENT_TOKEN_HEADER]: endpoint.credential,
      },
      body: encoded,
      signal: controller.signal,
    });
    return {
      status: response.status,
      body: decodeBody(await readBounded(response)),
      retryAfterMs: retryAfterMs(response.headers.get('retry-after')),
    };
  } catch (error) {
    if (error instanceof OversizedResponse) {
      return { kind: 'fault', origin: 'response', detail: error.message };
    }
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        kind: 'fault',
        origin: 'timeout',
        detail: `daemon did not answer within ${HOOK_TIMEOUT_MS}ms`,
      };
    }
    return {
      kind: 'fault',
      origin: 'transport',
      detail: `daemon hook request failed: ${errorDetail(error)}`,
    };
  } finally {
    clearTimeout(timer);
  }
}

function isRouteNotReady(body: unknown): boolean {
  return isRecord(body) && isRecord(body.error) && body.error.message === ROUTE_NOT_READY;
}

function retryAfterMs(header: string | null): number {
  const seconds = Number(header);
  if (header === null || header === '' || !Number.isInteger(seconds) || seconds < 0) return 1_000;
  return Math.min(Math.max(seconds * 1_000, MIN_RETRY_AFTER_MS), MAX_RETRY_AFTER_MS);
}

async function readBounded(response: Response): Promise<Uint8Array> {
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    length += part.value.byteLength;
    if (length > MAX_HOOK_RESPONSE_BYTES) {
      await reader.cancel();
      throw new OversizedResponse();
    }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function decodeBody(bytes: Uint8Array): unknown {
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    return null;
  }
}
