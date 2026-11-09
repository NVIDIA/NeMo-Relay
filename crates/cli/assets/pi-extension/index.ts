// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

import { createExtension } from './src/extension.ts';
import {
  faultPolicyFromEnv,
  gatewayConfigFromEnv,
  proxyCredentialFromEnv,
  redirectConfigFromEnv,
} from './src/standalone/config.ts';
import { postHook } from './src/standalone/transport.ts';

export default function nemoRelayExtension(pi: ExtensionAPI): void {
  createExtension(pi, {
    onFault: faultPolicyFromEnv(),
    bind: async () => {
      const gateway = gatewayConfigFromEnv();
      const proxyCredential = proxyCredentialFromEnv();
      return {
        redirect: redirectConfigFromEnv(proxyCredential !== undefined, gateway.url),
        post: (payload) => postHook(gateway, payload),
        close: (payload) => postHook(gateway, payload),
        providerRegistration: (decision, sessionId) => ({
          baseUrl: gateway.url,
          headers: {
            'x-nemo-relay-session-id': sessionId,
            ...(proxyCredential !== undefined ? { 'x-nemo-relay-proxy-token': proxyCredential } : {}),
            ...(decision.namedUpstream !== undefined
              ? { 'x-nemo-relay-upstream-base-url': decision.namedUpstream }
              : {}),
          },
        }),
      };
    },
  });
}
