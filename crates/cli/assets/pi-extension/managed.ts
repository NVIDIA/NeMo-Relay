// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from 'node:child_process';

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

import { type Wiring, createExtension } from './src/extension.ts';
import { errorDetail } from './src/gateway.ts';
import {
  CLIENT_TOKEN_ENV,
  type DeploymentConfig,
  readCredential,
  readDeploymentConfig,
} from './src/managed/deployment.ts';
import { type McpLease, sharedLease } from './src/managed/mcp-lease.ts';
import { CLIENT_TOKEN_HEADER, UPSTREAM_BASE_URL_HEADER, postDaemonHook } from './src/managed/transport.ts';

export type ManagedRuntime = {
  config: DeploymentConfig;
  credential: string;
  lease: McpLease;
};

export function managedWiring(runtime: ManagedRuntime): Wiring {
  const endpoint = { daemonAddress: runtime.config.daemonAddress, credential: runtime.credential };
  return {
    redirect: { mode: 'match', canNameUpstream: true, relayBaseUrl: runtime.config.daemonAddress },
    ready: () => runtime.lease.ensureReady(),
    async post(payload) {
      try {
        await runtime.lease.ensureReady();
      } catch (error) {
        return {
          kind: 'fault',
          origin: 'transport',
          detail: `managed MCP route is unavailable: ${errorDetail(error)}`,
        };
      }
      return postDaemonHook(endpoint, payload);
    },
    async close(payload) {
      try {
        return await postDaemonHook(endpoint, payload, 0);
      } finally {
        await runtime.lease.release();
      }
    },
    providerRegistration(decision) {
      return {
        baseUrl: runtime.config.daemonAddress,
        headers: {
          [CLIENT_TOKEN_HEADER]: runtime.credential,
          [UPSTREAM_BASE_URL_HEADER]: decision.upstream,
        },
      };
    },
  };
}

function initializeRuntime(): ManagedRuntime {
  const config = readDeploymentConfig(new URL('./managed-config.json', import.meta.url));
  const credential = readCredential(process.env[CLIENT_TOKEN_ENV]);
  return { config, credential, lease: sharedLease(config, credential, spawn) };
}

export default function managedNemoRelayPi(pi: ExtensionAPI): void {
  createExtension(pi, {
    onFault: 'closed',
    bind: async () => managedWiring(initializeRuntime()),
  });
}
