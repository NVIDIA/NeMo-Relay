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

function loadIndexForTest({
  platform,
  arch,
  existingFiles = [],
  providedModules = {},
  childProcessThrows = false,
  lddPath = '/usr/bin/ldd',
  lddContent = 'glibc',
  processReport = {
    header: {
      glibcVersionRuntime: '2.31',
    },
  },
}) {
  const module = {
    exports: {},
  };
  const existing = new Set(existingFiles);
  const calls = [];

  function fakeRequire(specifier) {
    calls.push(specifier);

    if (specifier === 'fs') {
      return {
        existsSync(target) {
          return existing.has(path.basename(target));
        },
        readFileSync(target) {
          if (target === lddPath) {
            return lddContent;
          }
          throw new Error(`unexpected readFileSync target: ${target}`);
        },
      };
    }

    if (specifier === 'path') {
      return {
        join: path.join,
      };
    }

    if (specifier === 'child_process') {
      return {
        execSync() {
          if (childProcessThrows) {
            throw new Error('which ldd failed');
          }
          return Buffer.from(lddPath);
        },
      };
    }

    if (Object.hasOwn(providedModules, specifier)) {
      const value = providedModules[specifier];
      if (value instanceof Error) {
        throw value;
      }
      return value;
    }

    throw new Error(`missing test module: ${specifier}`);
  }

  const fakeProcess = {
    platform,
    arch,
    report:
      processReport === null
        ? null
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
    {
      filename: indexFilename.pathname,
    },
  );

  return {
    exports: module.exports,
    calls,
  };
}

describe('index.js loader', () => {
  const binding = realLib;

  const localCases = [
    ['android', 'arm64', 'nemo-relay.android-arm64.node', './nemo-relay.android-arm64.node'],
    ['android', 'arm', 'nemo-relay.android-arm-eabi.node', './nemo-relay.android-arm-eabi.node'],
    ['win32', 'x64', 'nemo-relay.win32-x64-msvc.node', './nemo-relay.win32-x64-msvc.node'],
    ['win32', 'ia32', 'nemo-relay.win32-ia32-msvc.node', './nemo-relay.win32-ia32-msvc.node'],
    ['win32', 'arm64', 'nemo-relay.win32-arm64-msvc.node', './nemo-relay.win32-arm64-msvc.node'],
    ['freebsd', 'x64', 'nemo-relay.freebsd-x64.node', './nemo-relay.freebsd-x64.node'],
    ['linux', 'x64', 'nemo-relay.linux-x64-gnu.node', './nemo-relay.linux-x64-gnu.node'],
    ['linux', 'arm64', 'nemo-relay.linux-arm64-gnu.node', './nemo-relay.linux-arm64-gnu.node'],
    ['linux', 'arm', 'nemo-relay.linux-arm-gnueabihf.node', './nemo-relay.linux-arm-gnueabihf.node'],
    ['linux', 'riscv64', 'nemo-relay.linux-riscv64-gnu.node', './nemo-relay.linux-riscv64-gnu.node'],
    ['linux', 's390x', 'nemo-relay.linux-s390x-gnu.node', './nemo-relay.linux-s390x-gnu.node'],
  ];

  it('loads local binary branches for supported platforms', () => {
    for (const [platformName, archName, fileName, specifier] of localCases) {
      const { exports, calls } = loadIndexForTest({
        platform: platformName,
        arch: archName,
        existingFiles: [fileName],
        providedModules: {
          [specifier]: binding,
        },
      });

      assert.equal(exports.toolCall, binding.toolCall);
      assert.ok(calls.includes(specifier), `${platformName}/${archName} should require ${specifier}`);
    }
  });

  it('loads package branches for supported platforms', () => {
    const packageCases = [
      ['android', 'arm64', 'nemo-relay-node-android-arm64'],
      ['android', 'arm', 'nemo-relay-node-android-arm-eabi'],
      ['win32', 'x64', 'nemo-relay-node-win32-x64-msvc'],
      ['win32', 'ia32', 'nemo-relay-node-win32-ia32-msvc'],
      ['win32', 'arm64', 'nemo-relay-node-win32-arm64-msvc'],
      ['darwin', 'x64', 'nemo-relay-node-darwin-universal'],
      ['freebsd', 'x64', 'nemo-relay-node-freebsd-x64'],
      ['linux', 'x64', 'nemo-relay-node-linux-x64-gnu'],
      ['linux', 'arm64', 'nemo-relay-node-linux-arm64-gnu'],
      ['linux', 'arm', 'nemo-relay-node-linux-arm-gnueabihf'],
      ['linux', 'riscv64', 'nemo-relay-node-linux-riscv64-gnu'],
      ['linux', 's390x', 'nemo-relay-node-linux-s390x-gnu'],
    ];

    for (const [platformName, archName, specifier] of packageCases) {
      const { exports, calls } = loadIndexForTest({
        platform: platformName,
        arch: archName,
        providedModules: {
          [specifier]: binding,
        },
      });

      assert.equal(exports.toolCall, binding.toolCall);
      assert.ok(calls.includes(specifier), `${platformName}/${archName} should require ${specifier}`);
    }
  });

  it('covers linux musl branches using both process.report and ldd fallback', () => {
    const viaReport = loadIndexForTest({
      platform: 'linux',
      arch: 'x64',
      processReport: {
        header: {
          glibcVersionRuntime: null,
        },
      },
      providedModules: {
        'nemo-relay-node-linux-x64-musl': binding,
      },
    });
    assert.equal(viaReport.exports.toolCall, binding.toolCall);
    assert.ok(viaReport.calls.includes('nemo-relay-node-linux-x64-musl'));

    const viaLdd = loadIndexForTest({
      platform: 'linux',
      arch: 'arm64',
      processReport: null,
      lddContent: 'musl libc',
      providedModules: {
        'nemo-relay-node-linux-arm64-musl': binding,
      },
    });
    assert.equal(viaLdd.exports.toolCall, binding.toolCall);
    assert.ok(viaLdd.calls.includes('child_process'));
    assert.ok(viaLdd.calls.includes('nemo-relay-node-linux-arm64-musl'));

    const viaLddFailure = loadIndexForTest({
      platform: 'linux',
      arch: 'arm',
      processReport: null,
      childProcessThrows: true,
      providedModules: {
        'nemo-relay-node-linux-arm-musleabihf': binding,
      },
    });
    assert.equal(viaLddFailure.exports.toolCall, binding.toolCall);
    assert.ok(viaLddFailure.calls.includes('nemo-relay-node-linux-arm-musleabihf'));
  });

  it('falls back from darwin universal to arch-specific binaries', () => {
    const universalLocal = loadIndexForTest({
      platform: 'darwin',
      arch: 'arm64',
      existingFiles: ['nemo-relay.darwin-universal.node'],
      providedModules: {
        './nemo-relay.darwin-universal.node': binding,
      },
    });
    assert.equal(universalLocal.exports.toolCall, binding.toolCall);
    assert.ok(universalLocal.calls.includes('./nemo-relay.darwin-universal.node'));

    const x64 = loadIndexForTest({
      platform: 'darwin',
      arch: 'x64',
      existingFiles: ['nemo-relay.darwin-x64.node'],
      providedModules: {
        'nemo-relay-node-darwin-universal': new Error('universal missing'),
        './nemo-relay.darwin-x64.node': binding,
      },
    });
    assert.equal(x64.exports.toolCall, binding.toolCall);
    assert.ok(x64.calls.includes('nemo-relay-node-darwin-universal'));
    assert.ok(x64.calls.includes('./nemo-relay.darwin-x64.node'));

    const x64PackageFallback = loadIndexForTest({
      platform: 'darwin',
      arch: 'x64',
      providedModules: {
        'nemo-relay-node-darwin-universal': new Error('universal missing'),
        'nemo-relay-node-darwin-x64': binding,
      },
    });
    assert.equal(x64PackageFallback.exports.toolCall, binding.toolCall);
    assert.ok(x64PackageFallback.calls.includes('nemo-relay-node-darwin-x64'));

    const arm64 = loadIndexForTest({
      platform: 'darwin',
      arch: 'arm64',
      providedModules: {
        'nemo-relay-node-darwin-universal': new Error('universal missing'),
        'nemo-relay-node-darwin-arm64': binding,
      },
    });
    assert.equal(arm64.exports.toolCall, binding.toolCall);
    assert.ok(arm64.calls.includes('nemo-relay-node-darwin-arm64'));
  });

  it('throws unsupported platform and architecture errors', () => {
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'android',
          arch: 'x64',
        }),
      /Unsupported architecture on Android/,
    );
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'win32',
          arch: 'arm',
        }),
      /Unsupported architecture on Windows/,
    );
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'darwin',
          arch: 'ia32',
        }),
      /Unsupported architecture on macOS/,
    );
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'freebsd',
          arch: 'arm64',
        }),
      /Unsupported architecture on FreeBSD/,
    );
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'ppc64',
        }),
      /Unsupported architecture on Linux/,
    );
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'aix',
          arch: 'x64',
        }),
      /Unsupported OS/,
    );
  });

  it('throws the captured load error when binding resolution fails', () => {
    const failure = new Error('package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'freebsd',
          arch: 'x64',
          providedModules: {
            'nemo-relay-node-freebsd-x64': failure,
          },
        }),
      /package missing/,
    );
  });

  it('covers remaining linux loader error branches', () => {
    const androidArm64Failure = new Error('android arm64 package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'android',
          arch: 'arm64',
          providedModules: {
            'nemo-relay-node-android-arm64': androidArm64Failure,
          },
        }),
      /android arm64 package missing/,
    );

    const androidArmFailure = new Error('android arm package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'android',
          arch: 'arm',
          providedModules: {
            'nemo-relay-node-android-arm-eabi': androidArmFailure,
          },
        }),
      /android arm package missing/,
    );

    const win32X64Failure = new Error('win32 x64 package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'win32',
          arch: 'x64',
          providedModules: {
            'nemo-relay-node-win32-x64-msvc': win32X64Failure,
          },
        }),
      /win32 x64 package missing/,
    );

    const win32Ia32Failure = new Error('win32 ia32 package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'win32',
          arch: 'ia32',
          providedModules: {
            'nemo-relay-node-win32-ia32-msvc': win32Ia32Failure,
          },
        }),
      /win32 ia32 package missing/,
    );

    const win32Arm64Failure = new Error('win32 arm64 package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'win32',
          arch: 'arm64',
          providedModules: {
            'nemo-relay-node-win32-arm64-msvc': win32Arm64Failure,
          },
        }),
      /win32 arm64 package missing/,
    );

    const darwinX64LocalFailure = new Error('darwin x64 local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'darwin',
          arch: 'x64',
          existingFiles: ['nemo-relay.darwin-x64.node'],
          providedModules: {
            'nemo-relay-node-darwin-universal': new Error('universal missing'),
            './nemo-relay.darwin-x64.node': darwinX64LocalFailure,
          },
        }),
      /darwin x64 local missing/,
    );

    const darwinArm64LocalFailure = new Error('darwin arm64 local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'darwin',
          arch: 'arm64',
          existingFiles: ['nemo-relay.darwin-arm64.node'],
          providedModules: {
            'nemo-relay-node-darwin-universal': new Error('universal missing'),
            './nemo-relay.darwin-arm64.node': darwinArm64LocalFailure,
          },
        }),
      /darwin arm64 local missing/,
    );

    const x64MuslLocalFailure = new Error('x64 musl local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'x64',
          processReport: {
            header: {
              glibcVersionRuntime: null,
            },
          },
          existingFiles: ['nemo-relay.linux-x64-musl.node'],
          providedModules: {
            './nemo-relay.linux-x64-musl.node': x64MuslLocalFailure,
          },
        }),
      /x64 musl local missing/,
    );

    const x64GnuLocalFailure = new Error('x64 gnu local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'x64',
          existingFiles: ['nemo-relay.linux-x64-gnu.node'],
          providedModules: {
            './nemo-relay.linux-x64-gnu.node': x64GnuLocalFailure,
          },
        }),
      /x64 gnu local missing/,
    );

    const arm64MuslLocalFailure = new Error('arm64 musl local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'arm64',
          processReport: {
            header: {
              glibcVersionRuntime: null,
            },
          },
          existingFiles: ['nemo-relay.linux-arm64-musl.node'],
          providedModules: {
            './nemo-relay.linux-arm64-musl.node': arm64MuslLocalFailure,
          },
        }),
      /arm64 musl local missing/,
    );

    const arm64GnuFailure = new Error('arm64 gnu package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'arm64',
          providedModules: {
            'nemo-relay-node-linux-arm64-gnu': arm64GnuFailure,
          },
        }),
      /arm64 gnu package missing/,
    );

    const armMuslLocalFailure = new Error('arm musl local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'arm',
          processReport: {
            header: {
              glibcVersionRuntime: null,
            },
          },
          existingFiles: ['nemo-relay.linux-arm-musleabihf.node'],
          providedModules: {
            './nemo-relay.linux-arm-musleabihf.node': armMuslLocalFailure,
          },
        }),
      /arm musl local missing/,
    );

    const armGnuFailure = new Error('arm gnu package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'arm',
          providedModules: {
            'nemo-relay-node-linux-arm-gnueabihf': armGnuFailure,
          },
        }),
      /arm gnu package missing/,
    );

    const riscvMuslLocalFailure = new Error('riscv musl local missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'riscv64',
          processReport: {
            header: {
              glibcVersionRuntime: null,
            },
          },
          existingFiles: ['nemo-relay.linux-riscv64-musl.node'],
          providedModules: {
            './nemo-relay.linux-riscv64-musl.node': riscvMuslLocalFailure,
          },
        }),
      /riscv musl local missing/,
    );

    const riscvMuslFailure = new Error('riscv musl package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'riscv64',
          processReport: {
            header: {
              glibcVersionRuntime: null,
            },
          },
          providedModules: {
            'nemo-relay-node-linux-riscv64-musl': riscvMuslFailure,
          },
        }),
      /riscv musl package missing/,
    );

    const riscvGnuFailure = new Error('riscv gnu package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 'riscv64',
          providedModules: {
            'nemo-relay-node-linux-riscv64-gnu': riscvGnuFailure,
          },
        }),
      /riscv gnu package missing/,
    );

    const s390xFailure = new Error('s390x package missing');
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'linux',
          arch: 's390x',
          providedModules: {
            'nemo-relay-node-linux-s390x-gnu': s390xFailure,
          },
        }),
      /s390x package missing/,
    );
  });

  it('throws a generic error when resolution returns no binding and no load error', () => {
    assert.throws(
      () =>
        loadIndexForTest({
          platform: 'freebsd',
          arch: 'x64',
          providedModules: {
            'nemo-relay-node-freebsd-x64': null,
          },
        }),
      /Failed to load native binding/,
    );
  });
});
