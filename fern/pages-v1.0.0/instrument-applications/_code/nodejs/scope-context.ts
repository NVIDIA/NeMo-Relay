// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScopeType, createScopeStack, event, setThreadScopeStack, withScope } from 'nemo-relay-node';

/** Use an explicit scope stack across execution boundaries. */
export async function runExample() {
  const workerStack = createScopeStack();
  setThreadScopeStack(workerStack);

  await withScope('request', ScopeType.Agent, async (handle) => {
    event('started', handle, { ok: true }, null);
  });
}
