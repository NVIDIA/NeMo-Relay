#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';

const ignoredDirectories = new Set(['.git', '.venv', 'node_modules', 'target', 'tmp']);

function command(name, args, cwd) {
  const result = spawnSync(name, args, { cwd, stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`${name} ${args.join(' ')} failed with exit code ${result.status}`);
  }
}

function argumentsFrom(args) {
  let version;
  let output;
  let platform;
  for (let index = 0; index < args.length; index += 2) {
    const value = args[index + 1];
    if (args[index] === '--version') {
      version = value;
    } else if (args[index] === '--out') {
      output = value;
    } else if (args[index] === '--platform') {
      platform = value;
    } else {
      throw new Error(`Unexpected argument: ${args[index]}`);
    }
  }
  if (!version || !output || !platform) {
    throw new Error('Usage: package_node_musllinux.mjs --version VERSION --platform PLATFORM --out DIRECTORY');
  }
  return { output: resolve(output), platform, version };
}

function main() {
  const { output, platform, version } = argumentsFrom(process.argv.slice(2));
  const repository = process.cwd();
  const temporaryDirectory = mkdtempSync(join(tmpdir(), 'nemo-relay-node-musllinux-'));
  const sourceDirectory = join(temporaryDirectory, 'source');

  try {
    mkdirSync(output, { recursive: true });
    cpSync(repository, sourceDirectory, {
      filter: (source) => !ignoredDirectories.has(basename(source)),
      recursive: true,
    });
    command('npm', ['ci', '--workspace=nemo-relay-node', '--ignore-scripts'], sourceDirectory);
    command('npm', ['run', '--workspace=nemo-relay-node', 'build'], sourceDirectory);
    command(
      'python3',
      [
        'scripts/package-node-bin.py',
        '--node-dir',
        'crates/node',
        '--platform',
        platform,
        '--version',
        version,
        '--output-dir',
        output,
      ],
      sourceDirectory,
    );

    const packages = readdirSync(output).filter((entry) => entry.endsWith('.tgz'));
    if (packages.length !== 1) {
      throw new Error(`Expected one npm package artifact in ${output}, found ${packages.length}`);
    }
  } finally {
    rmSync(temporaryDirectory, { force: true, recursive: true });
  }
}

main();
