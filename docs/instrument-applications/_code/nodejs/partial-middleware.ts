// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LlmRequest,
  llmConditionalExecution,
  llmRequestIntercepts,
  toolConditionalExecution,
  toolRequestIntercepts,
} from 'nemo-relay-node';

/** Apply selected middleware without invoking a callback. */
export async function runExample() {
  const toolArgs = await toolRequestIntercepts('search', { query: 'weather' });
  await toolConditionalExecution('search', toolArgs);

  const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });
  const outcome = await llmRequestIntercepts('demo-provider', request);
  await llmConditionalExecution(outcome.request);
  return { toolArguments: toolArgs, request: outcome.request };
}
