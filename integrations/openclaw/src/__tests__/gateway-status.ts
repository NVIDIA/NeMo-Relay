/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import assert from "node:assert/strict";

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";

import type { NemoFlowHealthSnapshot } from "../health.js";

export type TestGatewayMethodHandler = Parameters<OpenClawPluginApi["registerGatewayMethod"]>[1];

export async function callGatewayStatus(
  handler: TestGatewayMethodHandler | undefined,
): Promise<NemoFlowHealthSnapshot> {
  assert.ok(handler);
  let status: NemoFlowHealthSnapshot | undefined;

  await handler({
    req: {} as never,
    params: {},
    client: null,
    isWebchatConnect: () => false,
    respond: (ok, payload, error) => {
      assert.equal(ok, true);
      assert.equal(error, undefined);
      status = payload as NemoFlowHealthSnapshot;
    },
    context: {} as never,
  });

  assert.ok(status);
  return status;
}
