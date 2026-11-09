// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';

import { type HookOutcome, errorDetail, isRecord, outcomeFromResponse } from '../gateway.ts';
import type { GatewayConfig } from './config.ts';

const CLIENT_ID = Symbol.for('nemo-relay.pi.hook-client.v1');
const clientState = globalThis as typeof globalThis & { [CLIENT_ID]?: string };

export const clientId: string = (clientState[CLIENT_ID] ??= randomUUID());

const MAX_OUTPUT_BYTES = 8 * 1024 * 1024;
const BODYLESS_STATUSES = new Set([204, 205, 304]);

type ExecFailure = Error & {
  code?: unknown;
  killed?: boolean;
  signal?: string;
  stderr?: string;
};

export async function postHook(config: GatewayConfig, payload: Record<string, unknown>): Promise<HookOutcome> {
  const args = ['hook-forward', 'pi', '--gateway-url', config.url, '--pi-extension-client', clientId, '--fail-closed'];
  if (config.transparentRun) args.push('--transparent-run');

  let input: string;
  try {
    input = JSON.stringify(payload);
  } catch (error) {
    return { kind: 'fault', origin: 'handler', detail: `hook payload is not JSON-safe: ${errorDetail(error)}` };
  }
  let raw: string;
  try {
    raw = await forward(config, args, input);
  } catch (error) {
    return executionFault(error, config.timeoutMs);
  }

  let envelope: unknown;
  try {
    envelope = JSON.parse(raw);
  } catch {
    return {
      kind: 'fault',
      origin: 'response',
      detail: 'hook forward returned an unreadable response',
    };
  }
  if (
    !isRecord(envelope) ||
    typeof envelope.status !== 'number' ||
    !Number.isInteger(envelope.status) ||
    envelope.status < 200 ||
    envelope.status > 599 ||
    typeof envelope.body !== 'string'
  ) {
    return {
      kind: 'fault',
      origin: 'response',
      detail: 'hook forward returned an invalid response',
    };
  }
  return outcomeFromResponse(envelope.status, decodeBody(envelope.status, envelope.body));
}

function forward(config: GatewayConfig, args: string[], input: string): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const child = execFile(
      config.relayCommand,
      args,
      {
        timeout: config.timeoutMs,
        maxBuffer: MAX_OUTPUT_BYTES,
        killSignal: 'SIGKILL',
        windowsHide: true,
        encoding: 'utf8',
      },
      (error, stdout, stderr) => {
        if (error) reject(Object.assign(error, { stderr }));
        else resolve(stdout);
      },
    );
    child.stdin?.on('error', () => undefined);
    child.stdin?.end(input);
  });
}

function decodeBody(status: number, body: string): unknown {
  if (BODYLESS_STATUSES.has(status)) return null;
  try {
    return JSON.parse(body);
  } catch {
    return null;
  }
}

function executionFault(error: unknown, timeoutMs: number): HookOutcome {
  if (!(error instanceof Error)) {
    return { kind: 'fault', origin: 'transport', detail: `gateway request failed: ${errorDetail(error)}` };
  }
  const failure: ExecFailure = error;
  if (failure.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') {
    return {
      kind: 'fault',
      origin: 'response',
      detail: 'hook forward output exceeded the output limit',
    };
  }
  if (failure.killed === true && failure.signal === 'SIGKILL') {
    return {
      kind: 'fault',
      origin: 'timeout',
      detail: `gateway did not respond within ${timeoutMs}ms`,
    };
  }
  const detail = failure.stderr?.trim() || errorDetail(error);
  return { kind: 'fault', origin: 'transport', detail: `gateway request failed: ${detail}` };
}
