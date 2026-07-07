// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { after, before, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const plugin = require('../plugin.js');

const nodeDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(nodeDir, '../..');
const fixtureTarget = path.join(repoRoot, 'target', 'node-dynamic-plugin-fixtures');
const tempRoot = mkdtempSync(path.join(tmpdir(), 'nemo-relay-node-dynamic-'));
const relayVersion = JSON.parse(readFileSync(path.join(nodeDir, 'package.json'), 'utf8')).version;

let nativeManifestRef;
let workerManifestRef;

function tomlString(value) {
  return JSON.stringify(value);
}

function nativeLibraryName() {
  if (process.platform === 'win32') {
    return 'nemo_relay_plugin_fixture.dll';
  }
  if (process.platform === 'darwin') {
    return 'libnemo_relay_plugin_fixture.dylib';
  }
  return 'libnemo_relay_plugin_fixture.so';
}

function workerBinaryName() {
  return process.platform === 'win32' ? 'nemo-relay-worker-plugin-fixture.exe' : 'nemo-relay-worker-plugin-fixture';
}

function buildFixture(manifestPath) {
  execFileSync(
    process.env.CARGO || 'cargo',
    ['build', '--quiet', '--manifest-path', manifestPath, '--target-dir', fixtureTarget],
    { stdio: 'inherit' },
  );
}

function buildNativeFixture(sourceManifestPath) {
  const sourceDirectory = path.dirname(sourceManifestPath);
  const fixtureDirectory = path.join(tempRoot, 'native-source');
  const fixtureSourceDirectory = path.join(fixtureDirectory, 'src');
  mkdirSync(fixtureSourceDirectory, { recursive: true });
  const pluginCrate = path.join(repoRoot, 'crates', 'plugin');
  const manifest = readFileSync(sourceManifestPath, 'utf8').replace(
    'nemo-relay-plugin = { path = "../../../../plugin" }',
    `nemo-relay-plugin = { path = ${tomlString(pluginCrate)} }`,
  );
  const fixtureManifest = path.join(fixtureDirectory, 'Cargo.toml');
  writeFileSync(fixtureManifest, manifest);
  writeFileSync(path.join(fixtureSourceDirectory, 'lib.rs'), readFileSync(path.join(sourceDirectory, 'src', 'lib.rs')));
  buildFixture(fixtureManifest);
}

function writeNativeManifest(libraryPath) {
  const directory = path.join(tempRoot, 'native');
  mkdirSync(directory, { recursive: true });
  const manifestRef = path.join(directory, 'relay-plugin.toml');
  writeFileSync(
    manifestRef,
    `manifest_version = 1

[plugin]
id = "fixture_native"
kind = "rust_dynamic"

[compat]
relay = ${tomlString(`=${relayVersion}`)}
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = ${tomlString(libraryPath)}
symbol = "nemo_relay_fixture_native_plugin"
`,
  );
  return manifestRef;
}

function writeWorkerManifest(entrypoint, name = 'worker') {
  const directory = path.join(tempRoot, name);
  mkdirSync(directory, { recursive: true });
  const manifestRef = path.join(directory, 'relay-plugin.toml');
  writeFileSync(
    manifestRef,
    `manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "worker"

[compat]
relay = ${tomlString(`=${relayVersion}`)}
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "rust"
entrypoint = ${tomlString(entrypoint)}
`,
  );
  return manifestRef;
}

function writeWorkerWrapper(entrypoint, pidFile, name) {
  const wrapper = path.join(tempRoot, `${name}.sh`);
  writeFileSync(wrapper, `#!/bin/sh\nprintf '%s' "$$" > ${tomlString(pidFile)}\nexec ${tomlString(entrypoint)}\n`, {
    mode: 0o755,
  });
  return wrapper;
}

function activationSpec(pluginId, kind, manifestRef, config = {}) {
  return {
    pluginId,
    kind,
    manifestRef,
    config,
  };
}

function nativeRequest(model = 'fixture-model') {
  return {
    headers: {},
    content: {
      model,
      messages: [],
    },
  };
}

async function executeTool(name) {
  return lib.toolCallExecute(
    name,
    { original: true },
    (args) => ({ ...args, downstream: true }),
    null,
    null,
    null,
    null,
  );
}

async function executeLlm(name) {
  return lib.llmCallExecute(
    name,
    nativeRequest(),
    (request) => ({
      downstream: true,
      requestContent: request.content,
    }),
    null,
    null,
    null,
    null,
    null,
  );
}

before(() => {
  const nativeFixture = path.join(repoRoot, 'crates', 'core', 'tests', 'fixtures', 'native_plugin', 'Cargo.toml');
  const workerFixture = path.join(repoRoot, 'crates', 'core', 'tests', 'fixtures', 'worker_plugin', 'Cargo.toml');
  buildNativeFixture(nativeFixture);
  buildFixture(workerFixture);
  nativeManifestRef = writeNativeManifest(path.join(fixtureTarget, 'debug', nativeLibraryName()));
  workerManifestRef = writeWorkerManifest(path.join(fixtureTarget, 'debug', workerBinaryName()));
});

after(() => {
  rmSync(tempRoot, { recursive: true, force: true });
});

describe('dynamic plugin host', () => {
  it('rejects empty specs without taking over static initialization', async () => {
    await assert.rejects(
      () => plugin.activateDynamicPlugins({ version: 1, components: [] }, []),
      /at least one dynamic plugin/i,
    );

    assert.deepEqual(await plugin.initialize({ version: 1, components: [] }), { diagnostics: [] });
    plugin.clear();
  });

  it('activates static base components with dynamic plugins', async () => {
    const staticKind = 'node.fixture.static-base';
    plugin.register(staticKind, {
      register(_config, context) {
        context.registerToolRequestIntercept('mark-static-base', 0, false, (_name, args) => ({
          ...args,
          staticBase: true,
        }));
      },
    });
    let activation;
    try {
      activation = await plugin.activateDynamicPlugins(
        {
          version: 1,
          components: [{ kind: staticKind, enabled: true, config: {} }],
        },
        [activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef)],
      );
      const result = await executeTool('node_static_and_dynamic_tool');
      assert.equal(result.staticBase, true);
      assert.equal(result.native_plugin_tool_execution, true);
    } finally {
      await activation?.close();
      plugin.deregister(staticKind);
    }
  });

  it('owns native managed callbacks until idempotent close', async () => {
    const activation = await plugin.activateDynamicPlugins({ version: 1, components: [] }, [
      activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef),
    ]);
    try {
      assert.deepEqual(activation.report.diagnostics, []);
      assert.equal(activation.active, true);
      assert.throws(() => plugin.clear(), /active dynamic plugin host/i);

      const toolResult = await executeTool('node_native_dynamic_tool');
      assert.equal(toolResult.downstream, true);
      assert.equal(toolResult.native_plugin_tool_execution_request, true);
      assert.equal(toolResult.native_plugin_tool_execution, true);

      const llmResult = await executeLlm('node_native_dynamic_llm');
      assert.equal(llmResult.downstream, true);
      assert.equal(llmResult.requestContent.native_plugin_llm_execution_request, true);
      assert.equal(llmResult.native_plugin_llm_execution, true);

      await Promise.all([activation.close(), activation.close()]);
      assert.equal(activation.active, false);
      await activation.close();

      const toolAfterClose = await executeTool('node_native_closed_tool');
      assert.deepEqual(toolAfterClose, { original: true, downstream: true });
      const llmAfterClose = await executeLlm('node_native_closed_llm');
      assert.equal(llmAfterClose.downstream, true);
      assert.equal(llmAfterClose.requestContent.native_plugin_llm_execution_request, undefined);
      assert.equal(llmAfterClose.native_plugin_llm_execution, undefined);
    } finally {
      await activation.close();
    }
  });

  it('owns worker managed callbacks until close', async () => {
    const activation = await plugin.activateDynamicPlugins({ version: 1, components: [] }, [
      activationSpec('fixture_worker', 'worker', workerManifestRef),
    ]);
    try {
      assert.deepEqual(activation.report.diagnostics, []);
      const toolResult = await executeTool('node_worker_dynamic_tool');
      assert.equal(toolResult.worker_plugin_tool_execution_request, true);
      assert.equal(toolResult.worker_plugin_tool_execution, true);

      const llmResult = await executeLlm('node_worker_dynamic_llm');
      assert.equal(llmResult.requestContent.worker_plugin_llm_execution_request, true);
      assert.equal(llmResult.worker_plugin_llm_execution, true);
    } finally {
      await activation.close();
    }

    const toolAfterClose = await executeTool('node_worker_closed_tool');
    assert.deepEqual(toolAfterClose, { original: true, downstream: true });
    const llmAfterClose = await executeLlm('node_worker_closed_llm');
    assert.equal(llmAfterClose.requestContent.worker_plugin_llm_execution_request, undefined);
    assert.equal(llmAfterClose.worker_plugin_llm_execution, undefined);
  });

  it(
    'keeps every concurrent close pending until the shared teardown completes',
    { skip: process.platform === 'win32' },
    async () => {
      const workerBinary = path.join(fixtureTarget, 'debug', workerBinaryName());
      const pidFile = path.join(tempRoot, 'concurrent-close-worker.pid');
      const wrapper = writeWorkerWrapper(workerBinary, pidFile, 'concurrent-close-worker');
      const manifestRef = writeWorkerManifest(wrapper, 'concurrent-close-worker');
      const activation = await plugin.activateDynamicPlugins({ version: 1, components: [] }, [
        activationSpec('fixture_worker', 'worker', manifestRef),
      ]);
      const workerPid = Number(readFileSync(pidFile, 'utf8'));
      process.kill(workerPid, 'SIGSTOP');

      const firstClose = activation.close();
      const secondClose = activation.close();
      let earlyResult;
      try {
        earlyResult = await Promise.race([
          firstClose.then(() => 'first'),
          secondClose.then(() => 'second'),
          new Promise((resolve) => setTimeout(() => resolve('pending'), 200)),
        ]);
      } finally {
        try {
          process.kill(workerPid, 'SIGCONT');
        } catch (error) {
          if (error.code !== 'ESRCH') {
            throw error;
          }
        }
        await Promise.all([firstClose, secondClose]);
      }

      assert.equal(earlyResult, 'pending');
      assert.equal(activation.active, false);
      await activation.close();
    },
  );

  it('preserves manifest and validation diagnostics in rejected promises', async () => {
    const missingManifest = path.join(tempRoot, 'missing', 'relay-plugin.toml');
    await assert.rejects(
      () =>
        plugin.activateDynamicPlugins({ version: 1, components: [] }, [
          activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef),
          activationSpec('missing_native', 'rust_dynamic', missingManifest),
        ]),
      (error) => {
        assert.match(error.message, /native plugin load failed/i);
        assert.match(error.message, /relay-plugin\.toml/);
        assert.match(error.message, /does not exist/i);
        return true;
      },
    );

    await assert.rejects(
      () =>
        plugin.activateDynamicPlugins({ version: 1, components: [] }, [
          activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef, { reject: true }),
        ]),
      /fixture rejection requested/i,
    );

    const recovered = await plugin.activateDynamicPlugins({ version: 1, components: [] }, [
      activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef),
    ]);
    await recovered.close();
  });

  it('defensively clears a native activation during garbage collection', () => {
    const pluginModule = path.join(nodeDir, 'plugin.js');
    const script = `
      import { createRequire } from 'node:module';
      const require = createRequire(${JSON.stringify(path.join(nodeDir, 'package.json'))});
      const plugin = require(${JSON.stringify(pluginModule)});
      const config = { version: 1, components: [] };
      const specs = [${JSON.stringify(activationSpec('fixture_native', 'rust_dynamic', nativeManifestRef))}];
      let activation = await plugin.activateDynamicPlugins(config, specs);
      const weak = new WeakRef(activation);
      activation = null;
      let collected = false;
      for (let index = 0; index < 100; index += 1) {
        global.gc();
        await new Promise((resolve) => setImmediate(resolve));
        if (weak.deref() === undefined) {
          collected = true;
          break;
        }
        await new Promise((resolve) => setImmediate(resolve));
      }
      if (!collected) {
        throw new Error('dynamic activation was not garbage collected');
      }
      let replacement;
      let lastError;
      for (let index = 0; index < 100; index += 1) {
        global.gc();
        await new Promise((resolve) => setImmediate(resolve));
        try {
          replacement = await plugin.activateDynamicPlugins(config, specs);
          break;
        } catch (error) {
          lastError = error;
        }
      }
      if (!replacement) {
        throw lastError ?? new Error('dynamic activation finalizer did not release ownership');
      }
      await replacement.close();
    `;
    execFileSync(process.execPath, ['--expose-gc', '--input-type=module', '--eval', script], {
      stdio: 'inherit',
      timeout: 30_000,
    });
  });

  it(
    'never waits for worker teardown on the JavaScript thread during garbage collection',
    { skip: process.platform === 'win32' },
    () => {
      const workerBinary = path.join(fixtureTarget, 'debug', workerBinaryName());
      const pidFile = path.join(tempRoot, 'finalizer-worker.pid');
      const wrapper = writeWorkerWrapper(workerBinary, pidFile, 'finalizer-worker');
      const manifestRef = writeWorkerManifest(wrapper, 'finalizer-worker');
      const pluginModule = path.join(nodeDir, 'plugin.js');
      const script = `
        import { spawn } from 'node:child_process';
        import { readFileSync } from 'node:fs';
        import { performance } from 'node:perf_hooks';
        import { createRequire } from 'node:module';
        const require = createRequire(${JSON.stringify(path.join(nodeDir, 'package.json'))});
        const plugin = require(${JSON.stringify(pluginModule)});
        const config = { version: 1, components: [] };
        const specs = [${JSON.stringify(activationSpec('fixture_worker', 'worker', manifestRef))}];
        let activation = await plugin.activateDynamicPlugins(config, specs);
        const weak = new WeakRef(activation);
        activation = null;
        await new Promise((resolve) => setImmediate(resolve));

        const workerPid = Number(readFileSync(${JSON.stringify(pidFile)}, 'utf8'));
        process.kill(workerPid, 'SIGSTOP');
        const resumer = spawn(
          '/bin/sh',
          ['-c', 'sleep 0.8; kill -CONT "$1"', 'resume-worker', String(workerPid)],
          { detached: true, stdio: 'ignore' },
        );
        resumer.unref();

        const startedAt = performance.now();
        global.gc();
        const elapsed = performance.now() - startedAt;
        if (weak.deref() !== undefined) {
          throw new Error('dynamic activation was not garbage collected');
        }
        if (elapsed >= 400) {
          throw new Error(\`dynamic activation finalizer blocked the JavaScript thread for \${elapsed}ms\`);
        }

        let replacement;
        let lastError;
        for (let index = 0; index < 500; index += 1) {
          await new Promise((resolve) => setTimeout(resolve, 10));
          try {
            replacement = await plugin.activateDynamicPlugins(config, specs);
            break;
          } catch (error) {
            lastError = error;
          }
        }
        if (!replacement) {
          throw lastError ?? new Error('dynamic activation finalizer did not release ownership');
        }
        await replacement.close();
      `;
      execFileSync(process.execPath, ['--expose-gc', '--input-type=module', '--eval', script], {
        stdio: 'inherit',
        timeout: 30_000,
      });
    },
  );
});
