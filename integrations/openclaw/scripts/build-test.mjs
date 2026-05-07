/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawnSync } from "node:child_process";
import { rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tsc = path.join(packageRoot, "node_modules", "typescript", "bin", "tsc");

rmSync(path.join(packageRoot, ".test-dist"), { recursive: true, force: true });

const result = spawnSync(process.execPath, [tsc, "-p", "tsconfig.test.json"], {
  cwd: packageRoot,
  stdio: "inherit",
});

process.exit(result.status ?? 1);
