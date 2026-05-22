/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Test helper for querying the OpenClaw gateway status endpoint in live smoke runs.
 */
import assert from 'node:assert/strict';

import type { OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';

import type { NemoRelayHealthSnapshot } from '../src/health.js';

export type TestGatewayMethodHandler = Parameters<OpenClawPluginApi['registerGatewayMethod']>[1];

export async function callGatewayStatus(
  handler: TestGatewayMethodHandler | undefined,
): Promise<NemoRelayHealthSnapshot> {
  assert.ok(handler);
  let status: NemoRelayHealthSnapshot | undefined;

  await handler({
    req: {} as never,
    params: {},
    client: null,
    isWebchatConnect: () => false,
    respond: (ok, payload, error) => {
      assert.equal(ok, true);
      assert.equal(error, undefined);
      status = payload as NemoRelayHealthSnapshot;
    },
    context: {} as never,
  });

  assert.ok(status);
  return status;
}
