// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import * as plugin from '../plugin.js';

async function withProjectPluginsToml({ atifEnabled }, callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-relay-node-plugin-'));
  const project = path.join(root, 'project');
  const configDir = path.join(project, '.nemo-relay');
  const oldCwd = process.cwd();
  const oldXdg = process.env.XDG_CONFIG_HOME;
  const oldHome = process.env.HOME;

  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(
    path.join(configDir, 'plugins.toml'),
    `
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = ${atifEnabled}
output_directory = ${JSON.stringify(path.join(root, 'atif'))}
filename_template = "missing-session-id.json"
`,
  );
  process.chdir(project);
  process.env.XDG_CONFIG_HOME = path.join(root, 'xdg');
  process.env.HOME = path.join(root, 'home');

  try {
    await callback({ root, project });
  } finally {
    try {
      plugin.clear();
    } catch (error) {
      console.warn('plugin.clear() failed during cleanup:', error);
    }
    process.chdir(oldCwd);
    if (oldXdg === undefined) {
      delete process.env.XDG_CONFIG_HOME;
    } else {
      process.env.XDG_CONFIG_HOME = oldXdg;
    }
    if (oldHome === undefined) {
      delete process.env.HOME;
    } else {
      process.env.HOME = oldHome;
    }
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test('initialize layers code config over project plugins.toml', async () => {
  await withProjectPluginsToml({ atifEnabled: false }, async () => {
    await assert.rejects(
      () =>
        plugin.initialize({
          components: [
            {
              kind: 'observability',
              config: {
                atif: {
                  enabled: true,
                },
              },
            },
          ],
        }),
      /filename_template/,
    );
  });
});

test('initialize with no arguments uses project plugins.toml', async () => {
  await withProjectPluginsToml({ atifEnabled: true }, async () => {
    await assert.rejects(() => plugin.initialize(), /filename_template/);
  });
});
