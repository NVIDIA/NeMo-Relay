// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const crateDir = path.resolve(scriptDir, '..');
const pkgDir = path.join(crateDir, 'pkg');

if (fs.existsSync(pkgDir)) {
  fs.rmSync(pkgDir, { recursive: true });
}

const wasmPackArgs = ['build'];
if (process.env.NEMO_RELAY_WASM_RELEASE) {
  wasmPackArgs.push('--release');
}
wasmPackArgs.push('--target', 'nodejs', '--out-dir', 'pkg');

function run(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: 'inherit', cwd: crateDir, shell: true });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run('wasm-pack', wasmPackArgs);
run('node', [path.join(scriptDir, 'prepare_pkg.mjs')]);
