// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const packageDirectory = fileURLToPath(new URL('..', import.meta.url));
const loggingEnvironmentNames = ['NEMO_RELAY_LOG', 'NEMO_RELAY_LOG_STDERR_FORMAT', 'NEMO_RELAY_LOG_CONFIG_PATH'];

function requireBinding(loggingEnvironment) {
  const environment = { ...process.env };
  for (const name of loggingEnvironmentNames) {
    delete environment[name];
  }
  Object.assign(environment, loggingEnvironment);
  return spawnSync(process.execPath, ['-e', "require('./index.js')"], {
    cwd: packageDirectory,
    encoding: 'utf8',
    env: environment,
  });
}

describe('operational logging', () => {
  it('initializes from the logging environment', () => {
    const result = requireBinding({
      NEMO_RELAY_LOG: 'info',
      NEMO_RELAY_LOG_STDERR_FORMAT: 'jsonl',
    });

    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stderr, /"event":"logging_initialized"/);
  });

  it('rejects an invalid logging environment', () => {
    const result = requireBinding({ NEMO_RELAY_LOG: '' });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /NEMO_RELAY_LOG must not be empty/);
  });
});
