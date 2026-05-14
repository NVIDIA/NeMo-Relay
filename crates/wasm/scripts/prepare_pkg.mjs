// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const crateDir = path.resolve(scriptDir, '..');
const esmWrapperDir = path.join(crateDir, 'wrappers', 'esm');
const nodeJsWrapperDir = path.join(crateDir, 'wrappers', 'nodejs');
const pkgDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(crateDir, 'pkg');

const rootJsFiles = ['index.js'];
const jsWrapperFiles = ['typed.js', 'plugin.js', 'adaptive.js', 'observability.js'];
const typeWrapperFiles = ['typed.d.ts', 'plugin.d.ts', 'adaptive.d.ts', 'observability.d.ts'];
const wrapperFiles = [...rootJsFiles, ...jsWrapperFiles, ...typeWrapperFiles];
const packageMetadata = {
  description: 'WebAssembly bindings for the NeMo Flow agent runtime.',
  keywords: ['agents', 'ai', 'llm', 'middleware', 'nemo-flow', 'observability', 'runtime', 'tools', 'wasm'],
  homepage: 'https://github.com/NVIDIA/NeMo-Flow#readme',
  bugs: {
    url: 'https://github.com/NVIDIA/NeMo-Flow/issues',
  },
  repository: {
    type: 'git',
    url: 'git+https://github.com/NVIDIA/NeMo-Flow.git',
    directory: 'crates/wasm',
  },
  author: 'NVIDIA Corporation & Affiliates',
  license: 'Apache-2.0',
};

// Wrapper sources refer to the crate-local dev layout; published files need to
// point at the package-local wasm entrypoint inside pkg/.
const replacements = [
  ['./pkg/index.js', './index.js'],
  ['./pkg/nemo_flow_wasm.js', './nemo_flow_wasm.js'],
  ['./pkg/nemo_flow_wasm', './nemo_flow_wasm'],
];

function copyWithReplacements(sourcePath, destinationPath) {
  let content = fs.readFileSync(sourcePath, 'utf8');

  for (const [from, to] of replacements) {
    content = content.replaceAll(from, to);
  }

  fs.writeFileSync(destinationPath, content);
}

function copyTypeWrapperFile(fileName) {
  const sourcePath = path.join(esmWrapperDir, fileName);
  const destinationPath = path.join(pkgDir, fileName);
  copyWithReplacements(sourcePath, destinationPath);
}

function writeJsWrapperFiles(manifest) {
  // wasm-pack emits ESM packages for module targets and CommonJS packages for
  // --target nodejs, so the helper wrappers must match the generated package.
  const sourceDir = manifest.type === 'module' ? esmWrapperDir : nodeJsWrapperDir;

  for (const fileName of [...rootJsFiles, ...jsWrapperFiles]) {
    copyWithReplacements(path.join(sourceDir, fileName), path.join(pkgDir, fileName));
  }
}

function updatePackageManifest(manifest) {
  const manifestPath = path.join(pkgDir, 'package.json');
  const existingFiles = Array.isArray(manifest.files) ? manifest.files : [];
  const rootTypes = 'nemo_flow_wasm.d.ts';

  Object.assign(manifest, packageMetadata);

  // wasm-pack generates a restrictive files allowlist, so the copied helper
  // wrappers must be added here or npm pack/publish will drop them.
  manifest.files = Array.from(new Set([...existingFiles, ...wrapperFiles]));

  manifest.main = 'index.js';
  manifest.types = rootTypes;
  manifest.sideEffects = Array.from(
    new Set([...(Array.isArray(manifest.sideEffects) ? manifest.sideEffects : []), './index.js']),
  );
  const rootJs = manifest.main;
  manifest.exports = {
    '.': {
      types: `./${rootTypes}`,
      default: `./${rootJs}`,
    },
    './typed': {
      types: './typed.d.ts',
      default: './typed.js',
    },
    './plugin': {
      types: './plugin.d.ts',
      default: './plugin.js',
    },
    './adaptive': {
      types: './adaptive.d.ts',
      default: './adaptive.js',
    },
    './observability': {
      types: './observability.d.ts',
      default: './observability.js',
    },
    './typed.js': {
      types: './typed.d.ts',
      default: './typed.js',
    },
    './plugin.js': {
      types: './plugin.d.ts',
      default: './plugin.js',
    },
    './adaptive.js': {
      types: './adaptive.d.ts',
      default: './adaptive.js',
    },
    './observability.js': {
      types: './observability.d.ts',
      default: './observability.js',
    },
  };

  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

const manifestPath = path.join(pkgDir, 'package.json');

if (!fs.existsSync(manifestPath)) {
  throw new Error(`expected wasm-pack output at ${pkgDir}`);
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

writeJsWrapperFiles(manifest);

for (const fileName of typeWrapperFiles) {
  copyTypeWrapperFile(fileName);
}

updatePackageManifest(manifest);
