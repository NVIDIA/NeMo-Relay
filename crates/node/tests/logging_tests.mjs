// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDirectory = fileURLToPath(new URL('..', import.meta.url));
const loggingEnvironmentNames = ['NEMO_RELAY_LOG', 'NEMO_RELAY_LOG_STDERR', 'NEMO_RELAY_LOG_STDERR_FORMAT', 'NEMO_RELAY_LOG_CONFIG_PATH'];

function requireBinding(loggingEnvironment, source = "require('./index.js')") {
  const environment = { ...process.env };
  for (const name of loggingEnvironmentNames) {
    delete environment[name];
  }
  Object.assign(environment, loggingEnvironment);
  return spawnSync(process.execPath, ['-e', source], {
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

  it('disables stderr logging from the environment', () => {
    const result = requireBinding({
      NEMO_RELAY_LOG: 'info',
      NEMO_RELAY_LOG_STDERR: 'false',
    });

    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stderr, '');
  });

  it('rejects an invalid logging environment', () => {
    const result = requireBinding({ NEMO_RELAY_LOG: '' });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /NEMO_RELAY_LOG must not be empty/);
  });

  it('flushes file sinks during environment cleanup', () => {
    const directory = mkdtempSync(join(tmpdir(), 'nemo-relay-node-logging-'));
    try {
      const configPath = join(directory, 'logging.toml');
      const logPath = join(directory, 'operational.jsonl');
      writeFileSync(
        configPath,
        `[logging]
level = "info"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = ${JSON.stringify(logPath)}
level = "info"
format = "jsonl"
queue_capacity = 16
`,
      );

      const result = requireBinding({ NEMO_RELAY_LOG_CONFIG_PATH: configPath });

      assert.equal(result.status, 0, result.stderr);
      assert.match(readFileSync(logPath, 'utf8'), /"event":"logging_shutdown_started"/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('keeps logging active while another Node environment remains', () => {
    const directory = mkdtempSync(join(tmpdir(), 'nemo-relay-node-worker-logging-'));
    try {
      const configPath = join(directory, 'logging.toml');
      const logPath = join(directory, 'operational.jsonl');
      writeFileSync(
        configPath,
        `[logging]
level = "info"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = ${JSON.stringify(logPath)}
level = "info"
format = "jsonl"
queue_capacity = 16
`,
      );
      const workerSource = `require(${JSON.stringify(join(packageDirectory, 'index.js'))})`;
      const source = `
const { Worker } = require('node:worker_threads');
const relay = require('./index.js');
const worker = new Worker(${JSON.stringify(workerSource)}, {
  eval: true,
});
worker.once('error', (error) => {
  console.error(error);
  process.exitCode = 1;
});
worker.once('exit', (code) => {
  if (code !== 0) process.exitCode = code;
  relay.deregisterPlugin('adaptive');
});
`;

      const result = requireBinding({ NEMO_RELAY_LOG_CONFIG_PATH: configPath }, source);

      assert.equal(result.status, 0, result.stderr);
      const output = readFileSync(logPath, 'utf8');
      assert.match(output, /"event":"plugin_deregistered"/);
      assert.match(output, /"event":"logging_shutdown_started"/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
