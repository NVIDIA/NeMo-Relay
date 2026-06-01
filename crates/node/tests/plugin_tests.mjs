// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';

import * as plugin from '../plugin.js';

test('layer forwards documents to core and returns merged JSON', () => {
  // Smoke test only: merge semantics are covered by the core crate. This
  // verifies the wrapper forwards both documents and returns merged JSON.
  assert.deepEqual(plugin.layer({ a: 1 }, { b: 2 }), { a: 1, b: 2 });
});
