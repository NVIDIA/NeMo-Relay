// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const cliPackagePath = require.resolve('@napi-rs/cli/package.json');
const napiCli = path.join(path.dirname(cliPackagePath), require(cliPackagePath).bin.napi);
const tsc = require.resolve('typescript/bin/tsc');

function run(args) {
  const result = spawnSync(process.execPath, args, { stdio: 'inherit' });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

run([napiCli, 'build', '--platform', '--js', 'binding.js', '--dts', 'binding.d.ts', ...process.argv.slice(2)]);
run([tsc, '-p', 'tsconfig.json']);
