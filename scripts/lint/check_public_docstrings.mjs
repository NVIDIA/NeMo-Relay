// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const nodeEntries = [
  {
    file: 'crates/node/plugin.js',
    functions: [
      { name: 'defaultConfig', params: [] },
      { name: 'ComponentSpec', params: ['kind', 'config', 'options'] },
      { name: 'validate', params: ['config'] },
      { name: 'initialize', params: ['config'] },
      { name: 'clear', params: [] },
      { name: 'report', params: [] },
      { name: 'listKinds', params: [] },
      { name: 'register', params: ['pluginKind', 'plugin'] },
      { name: 'deregister', params: ['pluginKind'] },
    ],
  },
  {
    file: 'crates/node/adaptive.js',
    functions: [
      { name: 'defaultConfig', params: [] },
      { name: 'inMemoryBackend', params: [] },
      { name: 'redisBackend', params: ['url', 'keyPrefix'] },
      { name: 'telemetryConfig', params: ['config'] },
      { name: 'adaptiveHintsConfig', params: ['config'] },
      { name: 'toolParallelismConfig', params: ['config'] },
      { name: 'acgConfig', params: ['config'] },
      { name: 'ComponentSpec', params: ['config', 'options'] },
    ],
  },
  {
    file: 'crates/node/typed.js',
    functions: [
      { name: 'encodeWithCodec', params: ['codec', 'payload'] },
      { name: 'typedToolExecute', params: ['name', 'args', 'func', 'argsCodec', 'resultCodec', 'options'] },
      { name: 'typedLlmExecute', params: ['name', 'request', 'func', 'responseJsonCodec', 'options'] },
      {
        name: 'typedLlmStreamExecute',
        params: ['name', 'request', 'func', 'collector', 'finalizer', 'chunkJsonCodec', 'responseJsonCodec', 'options'],
      },
    ],
  },
  {
    file: 'crates/node/plugin.d.ts',
    functions: [
      { name: 'defaultConfig', params: [] },
      { name: 'ComponentSpec', params: ['kind', 'config', 'options'] },
      { name: 'validate', params: ['config'] },
      { name: 'initialize', params: ['config'] },
      { name: 'clear', params: [] },
      { name: 'report', params: [] },
      { name: 'listKinds', params: [] },
      { name: 'register', params: ['pluginKind', 'plugin'] },
      { name: 'deregister', params: ['pluginKind'] },
    ],
  },
  {
    file: 'crates/node/adaptive.d.ts',
    functions: [
      { name: 'defaultConfig', params: [] },
      { name: 'inMemoryBackend', params: [] },
      { name: 'redisBackend', params: ['url', 'keyPrefix'] },
      { name: 'telemetryConfig', params: ['config'] },
      { name: 'adaptiveHintsConfig', params: ['config'] },
      { name: 'toolParallelismConfig', params: ['config'] },
      { name: 'acgConfig', params: ['config'] },
      { name: 'ComponentSpec', params: ['config', 'options'] },
    ],
  },
  {
    file: 'crates/node/typed.d.ts',
    functions: [
      { name: '__testEncodeWithCodec', params: ['codec', 'payload'] },
      { name: 'typedToolExecute', params: ['name', 'args', 'func', 'argsCodec', 'resultCodec', 'options'] },
      { name: 'typedLlmExecute', params: ['name', 'request', 'func', 'responseCodec', 'options'] },
      {
        name: 'typedLlmStreamExecute',
        params: ['name', 'request', 'func', 'collector', 'finalizer', 'chunkCodec', 'responseCodec', 'options'],
      },
    ],
  },
];
function escapeRegExp(value) {
  return value.replaceAll(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
}

function readDocblock(lines, declarationLine) {
  let lineIndex = declarationLine - 2;

  while (lineIndex >= 0 && lines[lineIndex].trim() === '') {
    lineIndex -= 1;
  }

  if (lineIndex < 0 || !lines[lineIndex].trim().endsWith('*/')) {
    return null;
  }

  const end = lineIndex;
  while (lineIndex >= 0 && !lines[lineIndex].trim().startsWith('/**')) {
    lineIndex -= 1;
  }

  if (lineIndex < 0) {
    return null;
  }

  return lines.slice(lineIndex, end + 1).join('\n');
}

function stripDocLine(line) {
  return line.replace(/^\s*\/?\*+\s?/, '').replace(/\*\/$/, '').trim();
}

function assertDocblock(docblock, filePath, name, params) {
  const lines = docblock
    .split('\n')
    .map(stripDocLine)
    .filter((line) => line.length > 0);
  const proseLines = lines.filter((line) => !line.startsWith('@'));

  if (proseLines.length < 2) {
    throw new Error(`${filePath}: \`${name}\` must include a one-line summary and a detailed description.`);
  }

  for (const param of params) {
    const paramPattern = new RegExp(
      String.raw`@param(?:\s+\{[^\n]+\})?\s+\[?${escapeRegExp(param)}(?:[^\]\s]*)\]?\s+-\s+\S`,
    );

    if (!paramPattern.test(docblock)) {
      throw new Error(`${filePath}: \`${name}\` is missing a documented \`@param\` entry for \`${param}\`.`);
    }
  }

  if (!/@returns?(?:\s+\{[^}]+\})?\s+\S/.test(docblock)) {
    throw new Error(`${filePath}: \`${name}\` must include a \`@returns\` description.`);
  }

  if (!/@(?:remarks|throws)\b/.test(docblock)) {
    throw new Error(`${filePath}: \`${name}\` must document non-obvious behavior or exceptions with \`@remarks\` or \`@throws\`.`);
  }
}

function findDeclarationLine(lines, name) {
  const suffix = String.raw`(?:<[^>]+>)?\s*\(`;
  const patterns = [
    new RegExp(String.raw`^export\s+async\s+function\s+${escapeRegExp(name)}${suffix}`),
    new RegExp(String.raw`^export\s+function\s+${escapeRegExp(name)}${suffix}`),
    new RegExp(String.raw`^export\s+declare\s+function\s+${escapeRegExp(name)}${suffix}`),
    new RegExp(String.raw`^async\s+function\s+${escapeRegExp(name)}${suffix}`),
    new RegExp(String.raw`^function\s+${escapeRegExp(name)}${suffix}`),
  ];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (patterns.some((pattern) => pattern.test(line))) {
      return index + 1;
    }
  }

  throw new Error(`${name} declaration not found.`);
}

export function checkPublicDocstrings(entries) {
  for (const entry of entries) {
    const absolutePath = path.join(repoRoot, entry.file);
    const content = fs.readFileSync(absolutePath, 'utf8');
    const lines = content.split('\n');

    for (const docTarget of entry.functions) {
      const declarationLine = findDeclarationLine(lines, docTarget.name);
      const docblock = readDocblock(lines, declarationLine);

      if (!docblock) {
        throw new Error(`${entry.file}: \`${docTarget.name}\` is missing a docblock.`);
      }

      assertDocblock(docblock, entry.file, docTarget.name, docTarget.params);
    }
  }
}

export function getDocstringEntries(target) {
  if (target === 'node' || target === 'all') {
    return nodeEntries;
  }

  throw new Error(`unknown docstring target: ${target}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const target = process.argv[2] || 'all';
  checkPublicDocstrings(getDocstringEntries(target));
  console.log(`docstring check passed for ${target}`);
}
