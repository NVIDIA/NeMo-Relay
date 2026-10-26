// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

test('the public diagnostics reader example runs in its own service process', () => {
  const example = fileURLToPath(new URL('../../../examples/diagnostics/node.cjs', import.meta.url));
  const result = spawnSync(process.execPath, [example], {
    env: { ...process.env, NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG: '1' },
    encoding: 'utf8',
    timeout: 60000,
  });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, '');
});
