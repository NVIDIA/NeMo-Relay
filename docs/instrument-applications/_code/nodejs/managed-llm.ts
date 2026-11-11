// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LlmRequest, llmCallExecuteAsync, type LlmRequestShape } from 'nemo-relay-node';

/** Execute the local provider callback through Relay. */
export async function runExample() {
  const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });

  const response = await llmCallExecuteAsync(
    'demo-provider',
    request,
    async (providerRequest: LlmRequestShape) => ({ text: 'hi', request: providerRequest.content }),
    null,
    null,
    null,
    null,
    'demo-model',
  );
  return response;
}
