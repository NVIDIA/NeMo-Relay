// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FaultPolicy } from '../gateway.ts';
import type { RedirectConfig } from '../provider-redirect.ts';

export type GatewayConfig = {
  url: string;
  relayCommand: string;
  timeoutMs: number;
  transparentRun: boolean;
};

const DEFAULT_GATEWAY_URL = 'http://127.0.0.1:4040';
const DEFAULT_TIMEOUT_MS = 5000;
const MAX_TIMEOUT_MS = 2_147_483_647;

export function gatewayConfigFromEnv(): GatewayConfig {
  const timeoutRaw = Number(process.env.NEMO_RELAY_PI_TIMEOUT_MS);
  return {
    url: (process.env.NEMO_RELAY_PI_GATEWAY_URL ?? DEFAULT_GATEWAY_URL).replace(/\/+$/, ''),
    relayCommand: process.env.NEMO_RELAY_PI_COMMAND ?? 'nemo-relay',
    timeoutMs:
      Number.isInteger(timeoutRaw) && timeoutRaw > 0 ? Math.min(timeoutRaw, MAX_TIMEOUT_MS) : DEFAULT_TIMEOUT_MS,
    transparentRun: proxyCredentialFromEnv() !== undefined,
  };
}

export function faultPolicyFromEnv(): FaultPolicy {
  return process.env.NEMO_RELAY_PI_FAIL === 'closed' ? 'closed' : 'open';
}

export function proxyCredentialFromEnv(): string | undefined {
  const value = process.env.NEMO_RELAY_PROXY_CREDENTIAL;
  return value === undefined || value === '' ? undefined : value;
}

export function redirectConfigFromEnv(canNameUpstream: boolean, relayBaseUrl: string): RedirectConfig {
  const raw = process.env.NEMO_RELAY_PI_REDIRECT;
  const openaiUpstream = process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM;
  const anthropicUpstream = process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM;
  return {
    mode: raw === 'off' || raw === 'force' ? raw : 'match',
    canNameUpstream,
    relayBaseUrl,
    ...(openaiUpstream ? { openaiUpstream } : {}),
    ...(anthropicUpstream ? { anthropicUpstream } : {}),
  };
}
