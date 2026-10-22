// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');

test('exact validation rejects host diagnostics instead of discarding them', () => {
  for (const diagnostics of [null, true, {}, { enabled: true }, { enabled: 'invalid' }]) {
    assert.throws(
      () => plugin.validateExact({ version: 1, components: [], diagnostics }),
      /diagnostics.*layered validation/,
    );
  }
});

test('exact validation accepts static plugin configuration', () => {
  assert.deepEqual(plugin.validateExact(plugin.defaultConfig()).dynamic_plugins, []);
});
