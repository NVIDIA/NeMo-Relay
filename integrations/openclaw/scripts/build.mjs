/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Production build helper for the OpenClaw integration package.
 *
 * The script removes stale output and invokes the workspace TypeScript compiler
 * directly so npm lifecycle behavior stays predictable in CI and local builds.
 */
import { spawnSync } from "node:child_process";
import { rmSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const tsc = require.resolve("typescript/bin/tsc");

rmSync(path.join(packageRoot, "dist"), { recursive: true, force: true });

const result = spawnSync(process.execPath, [tsc, "-p", "tsconfig.build.json"], {
  cwd: packageRoot,
  stdio: "inherit",
});

process.exit(result.status ?? 1);
