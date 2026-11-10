// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync as readNodeFileSync } from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const realLib = require('../index.js');
const indexFilename = new URL('../index.js', import.meta.url);
const indexDir = path.dirname(indexFilename.pathname);
const indexSource = readNodeFileSync(indexFilename, 'utf8');

const PACKAGE = 'nemo-relay-node';
const BINARY = 'nemo-relay';
const PACKAGE_VERSION = JSON.parse(readNodeFileSync(new URL('../package.json', import.meta.url), 'utf8')).version;

function loadIndexForTest({
  platform,
  arch,
  env = {},
  config = undefined,
  providedModules = {},
  lddFile = undefined,
  lddCommand = undefined,
  processReport = undefined,
  packageVersion = PACKAGE_VERSION,
}) {
  const module = { exports: {} };
  const calls = [];

  function fakeRequire(specifier) {
    calls.push(specifier);
    if (specifier === 'fs') {
      return {
        readFileSync(target) {
          if (target === '/usr/bin/ldd' && lddFile !== undefined) {
            return lddFile;
          }
          throw Object.assign(new Error(`ENOENT: ${target}`), { code: 'ENOENT' });
        },
      };
    }
    if (specifier === 'child_process') {
      return {
        execSync() {
          if (lddCommand instanceof Error) {
            throw lddCommand;
          }
          if (lddCommand === undefined) {
            throw new Error('ldd unavailable');
          }
          return lddCommand;
        },
      };
    }
    if (specifier.endsWith('/package.json')) {
      const owner = specifier.slice(0, -'/package.json'.length);
      if (Object.hasOwn(providedModules, owner) && !(providedModules[owner] instanceof Error)) {
        return { version: packageVersion };
      }
      throw new Error(`Cannot find module '${specifier}'`);
    }
    if (Object.hasOwn(providedModules, specifier)) {
      const value = providedModules[specifier];
      if (value instanceof Error) {
        throw value;
      }
      return value;
    }
    throw Object.assign(new Error(`Cannot find module '${specifier}'`), {
      code: 'MODULE_NOT_FOUND',
    });
  }

  const fakeProcess = {
    platform,
    arch,
    env,
    config,
    report:
      processReport === undefined
        ? undefined
        : {
            getReport() {
              return processReport;
            },
          },
  };

  vm.runInNewContext(
    indexSource,
    {
      module,
      exports: module.exports,
      require: fakeRequire,
      __dirname: indexDir,
      process: fakeProcess,
    },
    { filename: indexFilename.pathname },
  );

  return { exports: module.exports, calls };
}

function causeMessages(error) {
  const messages = [];
  for (let current = error; current; current = current.cause) {
    messages.push(String(current.message ?? current));
  }
  return messages;
}

function assertLoadFailure(run, ...patterns) {
  let error;
  try {
    run();
  } catch (thrown) {
    error = thrown;
  }
  assert.ok(error, 'expected the loader to throw');
  assert.match(error.message, /Cannot find native binding/);
  const messages = causeMessages(error);
  for (const pattern of patterns) {
    assert.ok(
      messages.some((message) => pattern.test(message)),
      `expected a cause matching ${pattern}, got: ${JSON.stringify(messages)}`,
    );
  }
}

const TUPLES = [
  ['android', 'arm64', 'android-arm64'],
  ['android', 'arm', 'android-arm-eabi'],
  ['win32', 'x64', 'win32-x64-msvc'],
  ['win32', 'x64', 'win32-x64-gnu', { config: { variables: { shlib_suffix: 'dll.a' } } }],
  ['win32', 'ia32', 'win32-ia32-msvc'],
  ['win32', 'arm64', 'win32-arm64-msvc'],
  ['darwin', 'x64', 'darwin-x64'],
  ['darwin', 'arm64', 'darwin-arm64'],
  ['freebsd', 'x64', 'freebsd-x64'],
  ['freebsd', 'arm64', 'freebsd-arm64'],
  ['freebsd', 'ppc64', 'freebsd-ppc64'],
  ['linux', 'x64', 'linux-x64-gnu', { lddFile: 'glibc' }],
  ['linux', 'x64', 'linux-x64-musl', { lddFile: 'musl libc' }],
  ['linux', 'arm64', 'linux-arm64-gnu', { lddFile: 'glibc' }],
  ['linux', 'arm64', 'linux-arm64-musl', { lddFile: 'musl libc' }],
  ['linux', 'arm', 'linux-arm-gnueabihf', { lddFile: 'glibc' }],
  ['linux', 'arm', 'linux-arm-musleabihf', { lddFile: 'musl libc' }],
  ['linux', 'loong64', 'linux-loong64-gnu', { lddFile: 'glibc' }],
  ['linux', 'loong64', 'linux-loong64-musl', { lddFile: 'musl libc' }],
  ['linux', 'riscv64', 'linux-riscv64-gnu', { lddFile: 'glibc' }],
  ['linux', 'riscv64', 'linux-riscv64-musl', { lddFile: 'musl libc' }],
  ['linux', 'ppc64', 'linux-ppc64-gnu', { lddFile: 'glibc' }],
  ['linux', 's390x', 'linux-s390x-gnu', { lddFile: 'glibc' }],
  ['openharmony', 'arm64', 'openharmony-arm64'],
  ['openharmony', 'x64', 'openharmony-x64'],
  ['openharmony', 'arm', 'openharmony-arm'],
];

describe('index.js loader', () => {
  const binding = realLib;

  it('loads the local binary for every supported platform', () => {
    for (const [platform, arch, tuple, extra = {}] of TUPLES) {
      const specifier = `./${BINARY}.${tuple}.node`;
      const { exports, calls } = loadIndexForTest({
        platform,
        arch,
        ...extra,
        providedModules: { [specifier]: binding },
      });
      assert.equal(exports.toolCall, binding.toolCall, `${tuple} local binary`);
      assert.equal(exports.__napiBindingTarget, 'native');
      assert.ok(calls.includes(specifier), `${tuple} should require ${specifier}`);
    }
  });

  it('falls back to the platform package for every supported platform', () => {
    for (const [platform, arch, tuple, extra = {}] of TUPLES) {
      const specifier = `${PACKAGE}-${tuple}`;
      const { exports, calls } = loadIndexForTest({
        platform,
        arch,
        ...extra,
        providedModules: { [specifier]: binding },
      });
      assert.equal(exports.toolCall, binding.toolCall, `${tuple} platform package`);
      assert.ok(calls.includes(`./${BINARY}.${tuple}.node`), `${tuple} tries the local binary first`);
      assert.ok(calls.includes(specifier), `${tuple} should require ${specifier}`);
    }
  });

  it('prefers the darwin universal binary and package over arch-specific ones', () => {
    const local = loadIndexForTest({
      platform: 'darwin',
      arch: 'arm64',
      providedModules: {
        [`./${BINARY}.darwin-universal.node`]: binding,
        [`./${BINARY}.darwin-arm64.node`]: new Error('arch-specific binary must not load'),
      },
    });
    assert.equal(local.exports.toolCall, binding.toolCall);

    const pkg = loadIndexForTest({
      platform: 'darwin',
      arch: 'x64',
      providedModules: {
        [`${PACKAGE}-darwin-universal`]: binding,
        [`${PACKAGE}-darwin-x64`]: new Error('arch-specific package must not load'),
      },
    });
    assert.equal(pkg.exports.toolCall, binding.toolCall);
    assert.ok(pkg.calls.includes(`./${BINARY}.darwin-universal.node`));
  });

  it('detects musl through the ldd file, then process.report, then ldd --version', () => {
    const viaReport = loadIndexForTest({
      platform: 'linux',
      arch: 'x64',
      processReport: {
        header: {},
        sharedObjects: ['/lib/ld-musl-x86_64.so.1'],
      },
      providedModules: { [`${PACKAGE}-linux-x64-musl`]: binding },
    });
    assert.equal(viaReport.exports.toolCall, binding.toolCall);

    const glibcReport = loadIndexForTest({
      platform: 'linux',
      arch: 'x64',
      processReport: { header: { glibcVersionRuntime: '2.31' } },
      providedModules: { [`${PACKAGE}-linux-x64-gnu`]: binding },
    });
    assert.equal(glibcReport.exports.toolCall, binding.toolCall);

    const viaCommand = loadIndexForTest({
      platform: 'linux',
      arch: 'arm64',
      lddCommand: 'musl libc (aarch64)',
      providedModules: { [`${PACKAGE}-linux-arm64-musl`]: binding },
    });
    assert.equal(viaCommand.exports.toolCall, binding.toolCall);
    assert.ok(viaCommand.calls.includes('child_process'));

    const commandFailure = loadIndexForTest({
      platform: 'linux',
      arch: 'arm',
      lddCommand: new Error('which ldd failed'),
      providedModules: { [`${PACKAGE}-linux-arm-gnueabihf`]: binding },
    });
    assert.equal(commandFailure.exports.toolCall, binding.toolCall);
  });

  it('honors NAPI_RS_NATIVE_LIBRARY_PATH', () => {
    const { exports, calls } = loadIndexForTest({
      platform: 'linux',
      arch: 'x64',
      env: { NAPI_RS_NATIVE_LIBRARY_PATH: '/opt/override.node' },
      providedModules: { '/opt/override.node': binding },
    });
    assert.equal(exports.toolCall, binding.toolCall);
    assert.ok(calls.includes('/opt/override.node'));
    assert.ok(!calls.includes(`./${BINARY}.linux-x64-gnu.node`));
  });

  it('enforces the platform package version when asked', () => {
    assertLoadFailure(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'x64',
          lddFile: 'glibc',
          env: { NAPI_RS_ENFORCE_VERSION_CHECK: '1' },
          packageVersion: '0.0.1',
          providedModules: { [`${PACKAGE}-linux-x64-gnu`]: binding },
        }),
      new RegExp(`version mismatch, expected ${RegExp.escape(PACKAGE_VERSION)} but got 0\\.0\\.1`),
    );
  });

  it('reports unsupported platforms and architectures', () => {
    const cases = [
      ['android', 'x64', /Unsupported architecture on Android/],
      ['win32', 'arm', /Unsupported architecture on Windows/],
      ['darwin', 'ia32', /Unsupported architecture on macOS/],
      ['freebsd', 'arm', /Unsupported architecture on FreeBSD/],
      ['linux', 'mips', /Unsupported architecture on Linux/],
      ['openharmony', 'ia32', /Unsupported architecture on OpenHarmony/],
      ['aix', 'x64', /Unsupported OS/],
    ];
    for (const [platform, arch, pattern] of cases) {
      assertLoadFailure(() => loadIndexForTest({ platform, arch, lddFile: 'glibc' }), pattern);
    }
  });

  it('chains every load error into the thrown error', () => {
    assertLoadFailure(
      () =>
        loadIndexForTest({
          platform: 'freebsd',
          arch: 'x64',
          providedModules: { [`${PACKAGE}-freebsd-x64`]: new Error('package missing') },
        }),
      /package missing/,
      new RegExp(`Cannot find module '\\./${BINARY}\\.freebsd-x64\\.node'`),
    );
  });
});
