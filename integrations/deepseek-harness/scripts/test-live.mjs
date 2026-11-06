/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// The live smoke activates the plugin host; it must never pick up a discovered plugins.toml layer.
// Bindings built with `--features __skip-implicit-config` honor the variable; the test itself
// refuses to run when validation still reports a layer.
const result = spawnSync(process.execPath, ['--test', '.test-dist/test/live-smoke.test.js'], {
  cwd: packageRoot,
  env: {
    ...process.env,
    NEMO_RELAY_DEEPSEEK_HARNESS_LIVE_SMOKE: '1',
    NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG: process.env.NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG ?? '1',
  },
  stdio: 'inherit',
});

process.exit(result.status ?? 1);
