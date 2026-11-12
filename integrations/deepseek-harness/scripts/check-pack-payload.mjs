/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Package payload validation for the DeepSeek Harness integration.
 *
 * This script guards the npm package boundary: the generated dist files, the
 * Client view, and every file DeepSeek Harness reads from the bundle manifest
 * must be packed, while tests, maps, and test build output must stay out.
 */
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const npmExecPath = process.env.npm_execpath;

/** Fail the pack check with a precise validation message. */
function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

/** Run a child command from the package root and surface stderr on failure. */
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: packageRoot,
    encoding: 'utf8',
    ...options,
  });
  if (result.status !== 0) {
    if (result.error) {
      process.stderr.write(`${result.error.message}\n`);
    }
    process.stderr.write(result.stderr ?? '');
    throw new Error(`${command} ${args.join(' ')} failed`);
  }
  return result;
}

/** Run npm through the active npm CLI when available, preserving workspace behavior. */
function runNpm(args, options = {}) {
  if (npmExecPath) {
    return run(process.execPath, [npmExecPath, ...args], options);
  }
  return run(npm, args, {
    shell: process.platform === 'win32',
    ...options,
  });
}

/** Normalize npm pack paths to POSIX style without a leading ./ prefix. */
function normalizePackagePath(value) {
  return value.replace(/^\.\//, '').replaceAll('\\', '/');
}

/** Recursively list files below a package-local directory. */
function walkFiles(root, prefix = '') {
  const absoluteRoot = path.join(packageRoot, root, prefix);
  const output = [];
  for (const entry of readdirSync(absoluteRoot)) {
    const relative = path.posix.join(prefix, entry);
    const absolute = path.join(packageRoot, root, relative);
    if (statSync(absolute).isDirectory()) {
      output.push(...walkFiles(root, relative));
    } else {
      output.push(path.posix.join(root, relative));
    }
  }
  return output.sort();
}

runNpm(['run', 'build'], { stdio: 'inherit' });

const pack = runNpm(['pack', '--dry-run', '--json', '--ignore-scripts']);
const packInfo = JSON.parse(pack.stdout)[0];
assert(packInfo, 'npm pack did not return package metadata');

const packedFiles = new Set(packInfo.files.map((file) => normalizePackagePath(file.path)));
const packageJson = JSON.parse(readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));

assert(
  packageJson.repository?.url === 'https://github.com/NVIDIA/NeMo-Relay',
  'package repository.url must match the GitHub Actions provenance source repository',
);
assert(
  packageJson.repository?.directory === 'integrations/deepseek-harness',
  'package repository.directory must identify the DeepSeek Harness workspace',
);

const requiredFiles = [
  'package.json',
  'README.md',
  'dist/index.js',
  'dist/index.d.ts',
  'client/client.js',
  'cordis.patch.yml',
  'icon.svg',
  'locale/en.json',
  'locale/zh.json',
  // The nemo-relay skill reads this file at runtime and falls back silently without it.
  'skills/nemo-relay.md',
];

for (const file of requiredFiles) {
  assert(packedFiles.has(file), `packed package is missing ${file}`);
}

const bundlePatch = packageJson.dsh?.bundle?.patch;
assert(typeof bundlePatch === 'string', 'package dsh.bundle.patch must name the Loader patch');
assert(packedFiles.has(normalizePackagePath(bundlePatch)), `dsh.bundle.patch ${bundlePatch} is not packed`);

for (const [subpath, target] of Object.entries(packageJson.exports ?? {})) {
  const file = typeof target === 'string' ? target : target?.default;
  if (typeof file === 'string' && !file.includes('*')) {
    assert(packedFiles.has(normalizePackagePath(file)), `export ${subpath} target ${file} is not packed`);
  }
}

for (const file of packedFiles) {
  assert(!file.startsWith('test/'), `packed package includes test artifact ${file}`);
  assert(!file.startsWith('.test-dist/'), `packed package includes test output ${file}`);
  assert(!file.startsWith('src/') && file !== 'index.ts', `packed package includes TypeScript source ${file}`);
  assert(!file.endsWith('.map'), `packed package includes source/declaration map ${file}`);
}

const builtDistFiles = new Set(walkFiles('dist'));
for (const file of builtDistFiles) {
  assert(packedFiles.has(file), `built dist file ${file} is not packed`);
}

for (const file of packedFiles) {
  if (file.startsWith('dist/')) {
    assert(builtDistFiles.has(file), `packed dist file ${file} was not produced by the fresh build`);
  }
}

console.log(`pack payload ok: ${packedFiles.size} files`);
