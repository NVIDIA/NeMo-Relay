/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const result = spawnSync(process.execPath, ["--test", ".test-dist/src/__tests__/live-smoke.test.js"], {
  cwd: packageRoot,
  env: {
    ...process.env,
    NEMO_FLOW_OPENCLAW_LIVE_SMOKE: "1",
  },
  stdio: "inherit",
});

process.exit(result.status ?? 1);
