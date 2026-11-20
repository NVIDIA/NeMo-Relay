// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type TransformEnvelope = {
  tool_call_id?: unknown;
  input?: unknown;
};

export type AllowBody = {
  tool_call?: TransformEnvelope;
};

export type FaultOrigin = 'transport' | 'timeout' | 'response' | 'configuration' | 'handler';

export type HookOutcome =
  | { kind: 'allow'; body: AllowBody }
  | { kind: 'block'; reason: string }
  | { kind: 'fault'; origin: FaultOrigin; detail: string };

export type HookFault = Extract<HookOutcome, { kind: 'fault' }>;

export type Verdict = Extract<HookOutcome, { kind: 'allow' | 'block' }>;

export type FaultPolicy = 'open' | 'closed';

export const GUARDRAIL_REJECTION_TYPE = 'nemo_relay_guardrail_rejected';

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function errorDetail(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function outcomeFromResponse(status: number, body: unknown): HookOutcome {
  if (status >= 200 && status <= 299) {
    if (!isRecord(body)) {
      return {
        kind: 'fault',
        origin: 'response',
        detail: 'gateway returned a success body that is not a JSON object',
      };
    }
    return { kind: 'allow', body };
  }
  if (status === 403 && isRecord(body)) {
    const error = body.error;
    if (isRecord(error) && error.type === GUARDRAIL_REJECTION_TYPE && typeof error.reason === 'string') {
      return { kind: 'block', reason: error.reason };
    }
    return {
      kind: 'fault',
      origin: 'response',
      detail: 'gateway returned 403 without a guardrail reason',
    };
  }
  return { kind: 'fault', origin: 'response', detail: `gateway returned HTTP ${status}` };
}

export function resolveFault(policy: FaultPolicy, fault: HookFault, toolName: string): Verdict {
  if (policy === 'open') return { kind: 'allow', body: {} };
  const openings: Record<FaultOrigin, string> = {
    transport: `The NeMo Relay policy gateway could not be reached to authorize this ${toolName} call`,
    timeout: `The NeMo Relay policy gateway did not answer in time to authorize this ${toolName} call`,
    response: `The NeMo Relay policy gateway answered this ${toolName} call without a usable decision`,
    configuration: `The NeMo Relay policy gate could not load its configuration to authorize this ${toolName} call`,
    handler: `The NeMo Relay policy gate failed before asking the gateway about this ${toolName} call`,
  };
  return {
    kind: 'block',
    reason:
      `${openings[fault.origin]}, so it was blocked rather than allowed through unchecked. This is an ` +
      `infrastructure fault, not a judgment about the request. Details: ${fault.detail}`,
  };
}
