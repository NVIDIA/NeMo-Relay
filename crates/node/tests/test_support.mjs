// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { flushSubscribers } = require('../index.js');

export async function waitForSubscriberCallbacks(predicate, timeoutMs = 15000) {
  await flushSubscribers();
  // flushSubscribers() waits for Relay's Rust subscriber dispatcher, but JS
  // subscriber callbacks are queued onto Node's event loop through N-API
  // ThreadsafeFunction. Yield event-loop turns until the observed JS-side
  // callback state is ready, with a timeout to avoid hanging the test forever.
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    await flushSubscribers();
    if (Date.now() >= deadline) {
      throw new Error('timed out waiting for subscriber callbacks');
    }
    await new Promise((resolve) => setImmediate(resolve));
  }
  await flushSubscribers();
}
