// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toolCall, toolCallEnd } from 'nemo-relay-node';

/** Complete the manual tool lifecycle and preserve operation errors. */
export function runExample(query = 'weather') {
  function search(query: string): { hits: number } {
    if (!query) throw new Error('query is required');
    return { hits: 2 };
  }

  const handle = toolCall('search', { query }, null, null, { attempt: 1 }, null, null);
  let result: { hits: number };
  try {
    result = search(query);
  } catch (executionError) {
    const message = executionError instanceof Error ? executionError.message : String(executionError);
    try {
      toolCallEnd(handle, { result: { error: message } }, null, null);
    } catch (completionError) {
      console.error('Relay could not record tool completion:', completionError);
    }
    throw executionError;
  }
  toolCallEnd(handle, { result }, null, null);
  return result;
}
