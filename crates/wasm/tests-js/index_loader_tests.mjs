// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { pkgDir, testsJsDir, wasm } from './test_support.mjs';

test('WebAssembly generated package exposes the expected package metadata', () => {
  const packageJson = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'));
  assert.equal(packageJson.name, 'nemo-relay-wasm');
  assert.equal(packageJson.types, 'nemo_relay_wasm.d.ts');
  assert.equal(packageJson.exports['.'].types, './nemo_relay_wasm.d.ts');
  assert.equal(packageJson.exports['./typed'].default, './typed.js');
  assert.equal(packageJson.exports['./plugin'].default, './plugin.js');
  assert.equal(packageJson.exports['./adaptive'].default, './adaptive.js');
  assert.equal(packageJson.exports['./pii_redaction'].default, './pii_redaction.js');
  assert.equal(packageJson.exports['./typed.js'].default, './typed.js');
  assert.equal(typeof wasm.ScopeType.Agent, 'number');
  assert.equal(wasm.ScopeType.Agent, 0);
});

test('WebAssembly generated package includes the expected wrapper files', () => {
  const expectedFiles = [
    'index.js',
    'nemo_relay_wasm.d.ts',
    'typed.js',
    'typed.d.ts',
    'plugin.js',
    'plugin.d.ts',
    'adaptive.js',
    'adaptive.d.ts',
    'pii_redaction.js',
    'pii_redaction.d.ts',
  ];

  for (const fileName of expectedFiles) {
    assert.equal(fs.existsSync(path.join(pkgDir, fileName)), true, `expected ${fileName} in pkg/`);
  }
});

test('WebAssembly package keeps the generated root declaration as the source of truth for exports metadata', () => {
  const packageJson = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'));
  assert.equal(packageJson.types, 'nemo_relay_wasm.d.ts');
  assert.equal(packageJson.exports['.'].types, './nemo_relay_wasm.d.ts');
});

test('WebAssembly package root declaration contains the documented public types and exports', () => {
  const indexJs = fs.readFileSync(path.join(pkgDir, 'index.js'), 'utf8');
  const wasmDts = fs.readFileSync(path.join(pkgDir, 'nemo_relay_wasm.d.ts'), 'utf8');

  for (const typeName of ['Json', 'JsonObject', 'OpenTelemetryConfig', 'OpenInferenceConfig']) {
    assert.match(wasmDts, new RegExp(String.raw`export (type|interface) ${typeName}\b`));
  }

  assert.match(indexJs, /nemo_relay_wasm\.js/);
  for (const name of Object.keys(wasm)) {
    assert.match(wasmDts, new RegExp(String.raw`\b${name}\b`), `expected ${name} in nemo_relay_wasm.d.ts`);
  }
});

test('WebAssembly JS wrapper covers TextEncoder fallback for unicode strings', () => {
  const child = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
        import assert from 'node:assert/strict';
        import { createRequire } from 'node:module';

        delete TextEncoder.prototype.encodeInto;

        const require = createRequire(import.meta.url);
        const wasm = require('../pkg');
        const stack = new wasm.ScopeStack();
        wasm.setThreadScopeStack(stack);

        const scope = wasm.pushScope('ascii-é', 1, null, 0, null, null);
        assert.equal(scope.name, 'ascii-é');
        wasm.event('ascii-é-mark', scope, { ok: true }, null);
      `,
    ],
    {
      cwd: testsJsDir,
      encoding: 'utf8',
      env: process.env,
    },
  );

  assert.equal(child.status, 0, child.stderr || child.stdout);
});
