// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const napiCli = fileURLToPath(
  new URL('../node_modules/@napi-rs/cli/scripts/index.js', import.meta.url),
);

const build = spawnSync(
  process.execPath,
  [napiCli, 'build', '--platform', '--dts', 'index.d.ts', ...process.argv.slice(2)],
  { stdio: 'inherit' },
);

if (build.error) {
  throw build.error;
}

if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

const appendTypes = spawnSync(
  process.execPath,
  ['scripts/append_root_types.mjs', 'index.d.ts', 'root-types.d.ts'],
  { stdio: 'inherit' },
);

if (appendTypes.error) {
  throw appendTypes.error;
}

process.exit(appendTypes.status ?? 1);
