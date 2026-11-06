// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Phoenix manager: adopt, launch, port fallback, stop (owned versus external),
 * failures as state, the remembered connection, processes sharing a data
 * directory, and overlapping operations, all through fake network and process
 * seams. Nothing here starts a process, signals one, or opens a socket.
 */
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtemp, readFile, stat, writeFile, mkdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { describe, it } from 'node:test';

import type { PhoenixConfig } from '../src/config.js';
import { PhoenixManager, resolveLauncher, type PhoenixDeps, type PhoenixExporter } from '../src/phoenix.js';

class FakeChild extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  constructor(readonly pid: number) {
    super();
  }
}

type Spawned = { command: string; args: string[]; options: any; child: FakeChild };

type MachineOptions = {
  up?: Record<string, string>;
  busy?: number[];
  uv?: boolean;
  spawnError?: Error;
  /** Phoenix ignores SIGTERM, so only SIGKILL ends it. */
  ignoreTerm?: boolean;
  /** How long Phoenix takes to exit after SIGTERM. */
  termDelayMs?: number;
  /** How long after the process is gone its exit event arrives. */
  exitDelayMs?: number;
  /** How long a health check takes to answer. */
  fetchDelayMs?: number;
};

/**
 * A fake machine: which URLs answer as Phoenix, which ports are taken, which
 * Phoenix processes live, and which DeepSeek Harness processes (`hosts`) live.
 */
function fakeMachine(options: MachineOptions = {}) {
  const up = new Map(Object.entries(options.up ?? {}));
  const busy = new Set(options.busy ?? []);
  const alive = new Map<number, FakeChild>();
  const hosts = new Set<number>();
  /** Command lines of pids that another program took over. */
  const commands = new Map<number, string>();
  const spawned: Spawned[] = [];
  const signals: Array<[number, string | number]> = [];
  let nextPid = 4100;
  let bootOnSpawn = true;
  const urlOf = (child: FakeChild) =>
    `http://127.0.0.1:${spawned.find((entry) => entry.child === child)?.options.env?.PHOENIX_PORT}`;
  const deps: Partial<PhoenixDeps> = {
    fetch: async (url) => {
      if (options.fetchDelayMs !== undefined) {
        await new Promise((resolve) => setTimeout(resolve, options.fetchDelayMs));
      }
      const base = url.replace(/\/arize_phoenix_version$/, '');
      const version = up.get(base);
      if (version === undefined) {
        throw Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' });
      }
      return { ok: true, text: async () => `${version}\n` };
    },
    portBusy: async (port) => busy.has(port),
    freePort: async () => 40_001,
    spawn: (command, args, spawnOptions) => {
      if (options.spawnError !== undefined) {
        throw options.spawnError;
      }
      const child = new FakeChild(nextPid++);
      alive.set(child.pid, child);
      spawned.push({ command, args, options: spawnOptions, child });
      if (bootOnSpawn) {
        // Phoenix answers on its port shortly after launch.
        setTimeout(() => alive.has(child.pid) && up.set(urlOf(child), '21.0.0'), 5);
      }
      return child;
    },
    kill: (pid, signal) => {
      signals.push([pid, signal]);
      if (hosts.has(pid) && signal === 0) {
        return;
      }
      const child = alive.get(Math.abs(pid));
      if (child === undefined) {
        throw Object.assign(new Error('kill ESRCH'), { code: 'ESRCH' });
      }
      if (signal === 0 || (signal === 'SIGTERM' && options.ignoreTerm === true)) {
        return;
      }
      const die = () => {
        if (!alive.delete(child.pid)) {
          return;
        }
        up.delete(urlOf(child));
        const exit = () => child.emit('exit', null, signal);
        if (options.exitDelayMs === undefined) {
          setImmediate(exit);
        } else {
          setTimeout(exit, options.exitDelayMs);
        }
      };
      if (signal === 'SIGTERM' && options.termDelayMs !== undefined) {
        setTimeout(die, options.termDelayMs);
      } else {
        die();
      }
    },
    processCommand: (pid) =>
      commands.get(pid) ??
      (alive.has(pid) ? '/fake/bin/uv tool run --from arize-phoenix==21.0.0 phoenix serve' : undefined),
    findExecutable: (name) => (name === 'uv' && options.uv !== false ? '/fake/bin/uv' : undefined),
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, Math.min(ms, 5))),
    env: {
      PATH: '/usr/bin',
      HOME: '/home/u',
      OPENAI_API_KEY: 'sk-never-passed',
      DEEPSEEK_API_KEY: 'never-passed',
      HTTPS_PROXY: 'http://proxy:8080',
    },
    platform: 'darwin',
  };
  return {
    deps,
    up,
    busy,
    alive,
    hosts,
    commands,
    spawned,
    signals,
    /** Signals that were sent, without the existence probes (`kill(pid, 0)`). */
    sent: () => signals.filter(([, signal]) => signal !== 0),
    holdBoot: () => void (bootOnSpawn = false),
  };
}

function fakeExporter(ready = true, disconnectMs?: number) {
  const calls: string[] = [];
  let connected = false;
  const exporter: PhoenixExporter & { calls: string[] } = {
    calls,
    ready: () => ready,
    connect: (target) => {
      calls.push(`connect ${target.endpoint} ${target.project}`);
      connected = true;
    },
    disconnect: async (flush) => {
      calls.push(`disconnect flush=${flush}`);
      if (disconnectMs !== undefined) {
        await new Promise((resolve) => setTimeout(resolve, disconnectMs));
      }
      connected = false;
    },
    connected: () => connected,
  };
  return exporter;
}

function phoenixConfig(
  overrides: Partial<PhoenixConfig> = {},
  launch: Partial<PhoenixConfig['launch']> = {},
): PhoenixConfig {
  return {
    enabled: true,
    url: 'http://127.0.0.1:16006',
    project: 'dsh-tests',
    autoConnect: true,
    ...overrides,
    launch: { enabled: true, version: '21.0.0', python: '3.13', startupTimeoutMs: 10_000, ...launch },
  };
}

async function manager(
  machine: ReturnType<typeof fakeMachine>,
  options: {
    config?: PhoenixConfig;
    exporter?: ReturnType<typeof fakeExporter>;
    dataDir?: string;
    replyBudgetMs?: number;
    stopGraceMs?: number;
    /** The DeepSeek Harness process this manager runs in; it lives until removed from `machine.hosts`. */
    pid?: number;
  } = {},
) {
  const dataDir = options.dataDir ?? (await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-phoenix-')));
  const exporter = options.exporter ?? fakeExporter();
  const changes: string[] = [];
  const pid = options.pid ?? 500;
  machine.hosts.add(pid);
  const phoenix = new PhoenixManager({
    config: options.config ?? phoenixConfig(),
    dataDir,
    exporter,
    deps: { ...machine.deps, pid },
    onChange: () => changes.push(phoenix.snapshot().state),
    replyBudgetMs: options.replyBudgetMs ?? 2_000,
    monitorMs: 10,
    ...(options.stopGraceMs === undefined ? {} : { stopGraceMs: options.stopGraceMs }),
  });
  return { phoenix, exporter, dataDir, changes };
}

async function savedConnection(dataDir: string) {
  return JSON.parse(await readFile(path.join(dataDir, 'phoenix', 'state.json'), 'utf8'));
}

async function until(predicate: () => boolean, label: string) {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, `timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

describe('Phoenix adoption', () => {
  it('uses a Phoenix that already answers, never launching or stopping it', async () => {
    const machine = fakeMachine({ up: { 'http://127.0.0.1:16006': '20.26.1' } });
    const { phoenix, exporter, dataDir } = await manager(machine);
    const started = await phoenix.start();
    assert.equal(started.ok, true);
    assert.deepEqual(started.phoenix, {
      enabled: true,
      state: 'running',
      connected: true,
      url: 'http://127.0.0.1:16006',
      project: 'dsh-tests',
      mode: 'external',
      version: '20.26.1',
      sessionUrl: 'http://127.0.0.1:16006/redirects/sessions/{sessionId}',
    });
    assert.deepEqual(exporter.calls, ['connect http://127.0.0.1:16006/v1/traces dsh-tests']);
    assert.deepEqual(await savedConnection(dataDir), {
      connected: true,
      mode: 'external',
      url: 'http://127.0.0.1:16006',
    });

    const stopped = await phoenix.stop();
    assert.equal(stopped.phoenix.state, 'stopped');
    assert.equal(machine.spawned.length, 0);
    assert.deepEqual(machine.signals, [], 'an external Phoenix is only disconnected');
    assert.deepEqual(exporter.calls.at(-1), 'disconnect flush=true');
    assert.equal((await savedConnection(dataDir)).connected, false);
  });

  it('connects only to a Phoenix that answers', async () => {
    const machine = fakeMachine();
    const { phoenix } = await manager(machine);
    const missing = await phoenix.connect();
    assert.deepEqual(
      [missing.ok, missing.error, missing.phoenix.state],
      [false, 'No Phoenix answers at http://127.0.0.1:16006', 'stopped'],
    );
    assert.equal(machine.spawned.length, 0, 'connect never launches');
    machine.up.set('http://127.0.0.1:16006', '21.0.0');
    assert.equal((await phoenix.connect()).phoenix.state, 'running');
  });
});

describe('Phoenix launch', () => {
  it('launches an owned Phoenix with a minimal environment on the next free port', async () => {
    const machine = fakeMachine({ busy: [16006] });
    const { phoenix, dataDir, exporter } = await manager(machine);
    const started = await phoenix.start();
    assert.equal(started.ok, true);
    assert.equal(started.phoenix.state, 'running');
    assert.equal(started.phoenix.mode, 'owned');
    assert.equal(started.phoenix.url, 'http://127.0.0.1:16007', 'the configured port is held by something else');

    const [launch] = machine.spawned;
    assert.equal(launch?.command, '/fake/bin/uv');
    assert.deepEqual(launch?.args, [
      'tool',
      'run',
      '--python',
      '3.13',
      '--from',
      'arize-phoenix==21.0.0',
      'phoenix',
      'serve',
    ]);
    assert.equal(launch?.options.cwd, path.join(dataDir, 'phoenix'));
    assert.equal(launch?.options.detached, true);
    const env = launch?.options.env ?? {};
    assert.equal(env.PHOENIX_HOST, '127.0.0.1');
    assert.equal(env.PHOENIX_PORT, '16007');
    assert.equal(env.PHOENIX_GRPC_PORT, '40001');
    assert.equal(env.PHOENIX_WORKING_DIR, path.join(dataDir, 'phoenix'));
    assert.equal(env.PHOENIX_TELEMETRY_ENABLED, 'false');
    assert.equal(env.PHOENIX_ALLOW_EXTERNAL_RESOURCES, 'false');
    assert.equal(env.PHOENIX_ENABLE_MCP_SERVER, 'false');
    assert.equal(env.PHOENIX_DISABLE_AGENT_ASSISTANT, 'true');
    assert.equal(env.HTTPS_PROXY, 'http://proxy:8080');
    assert.equal(env.OPENAI_API_KEY, undefined, 'credentials never reach Phoenix');
    assert.equal(env.DEEPSEEK_API_KEY, undefined);
    assert.ok(String(env.PATH).startsWith('/fake/bin'), 'the launcher directory is on PATH');
    assert.deepEqual(exporter.calls, ['connect http://127.0.0.1:16007/v1/traces dsh-tests']);
    assert.ok((await stat(path.join(dataDir, 'phoenix', 'phoenix.pid.json'))).isFile());

    const pid = launch?.child.pid ?? 0;
    const stopped = await phoenix.stop();
    assert.equal(stopped.phoenix.state, 'stopped');
    assert.deepEqual(machine.signals[0], [-pid, 'SIGTERM'], 'the owned process group is stopped');
    assert.equal(machine.alive.size, 0);
    await assert.rejects(stat(path.join(dataDir, 'phoenix', 'phoenix.pid.json')), { code: 'ENOENT' });
    assert.deepEqual(await savedConnection(dataDir), {
      connected: false,
      mode: 'owned',
      url: 'http://127.0.0.1:16007',
    });
  });

  it('replies while a slow first launch is still starting, then reports it running', async () => {
    const machine = fakeMachine();
    machine.holdBoot();
    const { phoenix } = await manager(machine, { replyBudgetMs: 20 });
    const reply = await phoenix.start();
    assert.deepEqual([reply.ok, reply.phoenix.state, reply.phoenix.phase], [true, 'starting', 'launching']);
    machine.up.set('http://127.0.0.1:16006', '21.0.0');
    await until(() => phoenix.snapshot().state === 'running', 'the launch to finish');
    await phoenix.dispose();
  });

  it('turns launch failures into state with a clear reason', async () => {
    const denied = fakeMachine({ spawnError: Object.assign(new Error('spawn EPERM'), { code: 'EPERM' }) });
    const { phoenix } = await manager(denied);
    const result = await phoenix.start();
    assert.deepEqual([result.ok, result.phoenix.state], [false, 'error']);
    assert.match(result.error ?? '', /Phoenix could not be launched \(\/fake\/bin\/uv\): spawn EPERM/);
    // Adopting a Phoenix someone else runs still works.
    denied.up.set('http://127.0.0.1:16006', '21.0.0');
    assert.equal((await phoenix.connect()).phoenix.state, 'running');

    const noUv = fakeMachine({ uv: false });
    const withoutUv = await manager(noUv);
    assert.match(withoutUv.phoenix.snapshot().launchHint ?? '', /needs uv/);
    const failed = await withoutUv.phoenix.start();
    assert.match(failed.error ?? '', /needs uv/);

    const remote = await manager(fakeMachine(), { config: phoenixConfig({ url: 'https://phoenix.example.com' }) });
    assert.match((await remote.phoenix.start()).error ?? '', /only launches Phoenix on this computer/);

    const offline = await manager(fakeMachine(), { exporter: fakeExporter(false) });
    assert.match((await offline.phoenix.start()).error ?? '', /NeMo Relay is not running/);
  });

  it('reports a launcher that exits, with the last line it printed', async () => {
    const machine = fakeMachine();
    machine.holdBoot();
    const { phoenix } = await manager(machine, { replyBudgetMs: 20 });
    await phoenix.start();
    const child = machine.spawned[0]?.child;
    child?.stderr.write('error: No solution found when resolving tool dependencies\n');
    await new Promise((resolve) => setTimeout(resolve, 5));
    child?.emit('exit', 2, null);
    await until(() => phoenix.snapshot().state === 'error', 'the exit to be reported');
    assert.equal(
      phoenix.snapshot().error,
      'Phoenix exited (code 2): error: No solution found when resolving tool dependencies',
    );
  });
});

describe('remembered connection', () => {
  it('reconnects an external Phoenix at startup, waiting for it while it is down', async () => {
    const machine = fakeMachine();
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-phoenix-'));
    await mkdir(path.join(dataDir, 'phoenix'), { recursive: true });
    await writeFile(
      path.join(dataDir, 'phoenix', 'state.json'),
      JSON.stringify({ connected: true, mode: 'external', url: 'http://127.0.0.1:16006' }),
    );
    const { phoenix } = await manager(machine, { dataDir });
    await phoenix.autoConnect();
    assert.equal(phoenix.snapshot().state, 'waiting');
    // Start does not launch a second Phoenix on the port of the one it waits for.
    assert.equal((await phoenix.start()).phoenix.state, 'waiting');
    assert.equal(machine.spawned.length, 0);
    machine.up.set('http://127.0.0.1:16006', '21.0.0');
    await until(() => phoenix.snapshot().state === 'running', 'the reconnection');
    assert.equal(phoenix.snapshot().mode, 'external');
    await phoenix.dispose();
  });

  it('exports to a remembered external Phoenix at once, before its health check answers', async () => {
    for (const answers of [true, false]) {
      const machine = fakeMachine(answers ? { up: { 'http://127.0.0.1:16006': '21.0.0' } } : {});
      let release: () => void = () => undefined;
      const gate = new Promise<void>((resolve) => (release = resolve));
      const slow = {
        ...machine,
        deps: {
          ...machine.deps,
          fetch: async (url: string, init: { signal: AbortSignal }) => {
            await gate;
            return machine.deps.fetch!(url, init);
          },
        },
      };
      const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-phoenix-'));
      await mkdir(path.join(dataDir, 'phoenix'), { recursive: true });
      await writeFile(
        path.join(dataDir, 'phoenix', 'state.json'),
        JSON.stringify({ connected: true, mode: 'external', url: 'http://127.0.0.1:16006' }),
      );
      const { phoenix, exporter } = await manager(slow, { dataDir });
      const reconnecting = phoenix.autoConnect();
      await new Promise((resolve) => setImmediate(resolve));
      // The first turn after a restart opens right after the plugin starts: spans must already flow.
      assert.deepEqual(exporter.calls, ['connect http://127.0.0.1:16006/v1/traces dsh-tests']);
      assert.equal(phoenix.snapshot().connected, false, 'not reported as connected before the check');
      release();
      await reconnecting;
      assert.equal(phoenix.snapshot().state, answers ? 'running' : 'waiting');
      assert.equal(exporter.connected(), answers, answers ? 'kept' : 'taken back when Phoenix does not answer');
      await phoenix.dispose();
    }
  });

  it('relaunches an owned Phoenix at startup, and only when autoConnect is on', async () => {
    const dataDir = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-phoenix-'));
    await mkdir(path.join(dataDir, 'phoenix'), { recursive: true });
    await writeFile(
      path.join(dataDir, 'phoenix', 'state.json'),
      JSON.stringify({ connected: true, mode: 'owned', url: 'http://127.0.0.1:16006' }),
    );

    const off = fakeMachine();
    await (await manager(off, { dataDir, config: phoenixConfig({ autoConnect: false }) })).phoenix.autoConnect();
    assert.equal(off.spawned.length, 0);

    const on = fakeMachine();
    const { phoenix } = await manager(on, { dataDir });
    await phoenix.autoConnect();
    assert.equal(phoenix.snapshot().state, 'running');
    assert.equal(on.spawned.length, 1);
    // Plugin disposal stops what it started but keeps the connection for the next start.
    await phoenix.dispose();
    assert.equal(on.alive.size, 0);
    assert.equal((await savedConnection(dataDir)).connected, true);
  });
});

describe('processes sharing a data directory', () => {
  const pidFile = (dataDir: string) => path.join(dataDir, 'phoenix', 'phoenix.pid.json');
  const owner = async (dataDir: string) => JSON.parse(await readFile(pidFile(dataDir), 'utf8')).owner;

  it('adopts a Phoenix whose DeepSeek Harness process is gone instead of launching another, and stops it', async () => {
    const machine = fakeMachine();
    const first = await manager(machine, { pid: 501 });
    await first.phoenix.start();
    assert.equal(await owner(first.dataDir), 501);
    // The Host crashed and left Phoenix running; a new process starts over the same data directory.
    machine.hosts.delete(501);
    const second = await manager(machine, { dataDir: first.dataDir, pid: 502 });
    const adopted = await second.phoenix.start();
    assert.deepEqual([adopted.phoenix.state, adopted.phoenix.mode], ['running', 'owned']);
    assert.equal(machine.spawned.length, 1);
    assert.equal(await owner(first.dataDir), 502, 'the orphan has a new owner');
    await second.phoenix.stop();
    assert.equal(machine.alive.size, 0);
  });

  it('uses, and never stops, a Phoenix that another live DeepSeek Harness process launched', async () => {
    const machine = fakeMachine();
    const app = await manager(machine, { pid: 501 });
    await app.phoenix.start();
    // A CLI run on the same data directory reconnects the way the user last connected, then exits.
    const cli = await manager(machine, { dataDir: app.dataDir, pid: 502 });
    await cli.phoenix.autoConnect();
    const borrowed = cli.phoenix.snapshot();
    assert.deepEqual([borrowed.state, borrowed.mode, borrowed.connected], ['running', 'external', true]);
    assert.equal(machine.spawned.length, 1);
    await cli.phoenix.dispose();
    // Another one stops and starts it explicitly.
    const other = await manager(machine, { dataDir: app.dataDir, pid: 503 });
    assert.equal((await other.phoenix.start()).phoenix.mode, 'external');
    assert.equal((await other.phoenix.stop()).phoenix.state, 'stopped');
    assert.equal((await other.phoenix.start()).phoenix.mode, 'external');
    await other.phoenix.dispose();
    await new Promise((resolve) => setTimeout(resolve, 20));

    assert.deepEqual(machine.sent(), [], 'no signal reaches the Phoenix of the app');
    const kept = app.phoenix.snapshot();
    assert.deepEqual([kept.state, kept.mode, kept.connected], ['running', 'owned', true]);
    assert.equal(await owner(app.dataDir), 501);
    // The next start of the app still launches its Phoenix again.
    assert.deepEqual(await savedConnection(app.dataDir), {
      connected: true,
      mode: 'owned',
      url: 'http://127.0.0.1:16006',
    });
    await app.phoenix.dispose();
    assert.equal(machine.alive.size, 0);
  });

  it('never adopts a pid that now belongs to another program', async () => {
    const machine = fakeMachine();
    const first = await manager(machine, { pid: 501 });
    await first.phoenix.start();
    const pid = machine.spawned[0]?.child.pid ?? 0;
    // The Host and its Phoenix are gone, the pid went to another program, and a Phoenix someone
    // else started answers on the same port.
    machine.hosts.delete(501);
    machine.commands.set(pid, '/usr/bin/some-other-program');
    const second = await manager(machine, { dataDir: first.dataDir, pid: 502 });
    const started = await second.phoenix.start();
    assert.deepEqual([started.phoenix.state, started.phoenix.mode], ['running', 'external']);
    await assert.rejects(stat(pidFile(first.dataDir)), { code: 'ENOENT' }, 'the stale pid file is removed');
    await second.phoenix.stop();
    await second.phoenix.dispose();
    assert.deepEqual(machine.sent(), [], 'the unrelated process is never signalled');
    assert.ok(machine.alive.has(pid));
  });

  it('never signals an adopted pid that another program took over later', async () => {
    const machine = fakeMachine();
    const first = await manager(machine, { pid: 501 });
    await first.phoenix.start();
    const pid = machine.spawned[0]?.child.pid ?? 0;
    // A pid file from before owners were recorded names an orphan too.
    await writeFile(pidFile(first.dataDir), JSON.stringify({ pid, port: 16006 }));
    const second = await manager(machine, { dataDir: first.dataDir, pid: 502 });
    assert.equal((await second.phoenix.start()).phoenix.mode, 'owned');
    // Disconnected, the adopted Phoenix is kept for a later stop; nothing watches it meanwhile.
    const kept = await second.phoenix.disconnect();
    assert.deepEqual([kept.phoenix.state, kept.phoenix.mode], ['stopped', 'owned']);
    machine.up.clear();
    machine.commands.set(pid, '/usr/bin/postgres -D /var/db');
    assert.equal((await second.phoenix.stop()).phoenix.state, 'stopped');
    await second.phoenix.dispose();
    assert.deepEqual(machine.sent(), [], 'the program that now has the pid is never signalled');
    assert.ok(machine.alive.has(pid));
  });
});

describe('overlapping operations', () => {
  const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  it('ends a stop that needed SIGKILL as stopped, not as a failure', async () => {
    const machine = fakeMachine({ ignoreTerm: true, exitDelayMs: 20 });
    const { phoenix } = await manager(machine, { stopGraceMs: 20 });
    await phoenix.start();
    const pid = machine.spawned[0]?.child.pid ?? 0;
    const stopped = await phoenix.stop();
    assert.deepEqual(machine.sent(), [
      [-pid, 'SIGTERM'],
      [-pid, 'SIGKILL'],
    ]);
    assert.equal(stopped.phoenix.state, 'stopped');
    // The exit event of the killed process arrives after the stop replied.
    await wait(50);
    assert.deepEqual([phoenix.snapshot().state, phoenix.snapshot().error], ['stopped', undefined]);
  });

  it('runs a start called during a stop after it, on a fresh Phoenix', async () => {
    const machine = fakeMachine({ termDelayMs: 30, fetchDelayMs: 2 });
    const exporter = fakeExporter(true, 3);
    const { phoenix, dataDir } = await manager(machine, { exporter });
    await phoenix.start();
    const stopping = phoenix.stop();
    // The stop has disconnected and waits for the process to exit; the old Phoenix still answers.
    await wait(8);
    const [stopped, started] = await Promise.all([stopping, phoenix.start()]);
    await wait(40);
    assert.equal(stopped.phoenix.state, 'stopped');
    assert.deepEqual([started.ok, started.phoenix.state, started.phoenix.mode], [true, 'running', 'owned']);
    assert.equal(machine.spawned.length, 2, 'the Phoenix being stopped is not adopted again');
    const settled = phoenix.snapshot();
    assert.deepEqual(
      [settled.state, settled.connected, exporter.connected(), [...machine.alive.keys()]],
      ['running', true, true, [machine.spawned[1]?.child.pid]],
    );
    assert.equal((await savedConnection(dataDir)).connected, true);
    await phoenix.dispose();
    assert.equal(machine.alive.size, 0);
  });

  it('refuses a start called during or after disposal, and leaves nothing behind', async () => {
    // 10 ms: the old Phoenix still answers. 40 ms: it is gone, and disposal has not finished.
    for (const lateMs of [10, 40]) {
      const machine = fakeMachine({ termDelayMs: 30, fetchDelayMs: 2 });
      const exporter = fakeExporter(true, 3);
      const { phoenix, dataDir } = await manager(machine, { exporter, replyBudgetMs: 500 });
      await phoenix.start();
      const disposing = phoenix.dispose();
      await wait(lateMs);
      const [late] = await Promise.all([phoenix.start(), disposing]);
      await wait(60);
      assert.deepEqual([late.ok, late.error], [false, 'NeMo Relay is shutting down'], `${lateMs} ms`);
      assert.deepEqual(
        [phoenix.snapshot().state, exporter.connected(), machine.spawned.length, machine.alive.size],
        ['stopped', false, 1, 0],
        `${lateMs} ms`,
      );
      await assert.rejects(stat(path.join(dataDir, 'phoenix', 'phoenix.pid.json')), { code: 'ENOENT' });
      for (const operation of ['connect', 'stop', 'disconnect'] as const) {
        const refused = await phoenix[operation]();
        assert.deepEqual([refused.ok, refused.error], [false, 'NeMo Relay is shutting down'], operation);
      }
      assert.equal((await savedConnection(dataDir)).connected, true, 'disposal keeps the remembered connection');
    }
  });

  it('lets a stop end a launch in progress at once, and does not run a start that a stop overtook', async () => {
    const machine = fakeMachine();
    machine.holdBoot();
    const { phoenix } = await manager(machine, { replyBudgetMs: 5_000 });
    const starting = phoenix.start();
    await until(() => machine.spawned.length === 1, 'the launch');
    const began = Date.now();
    // Called while the first start still waits for Phoenix: a second start, then a stop.
    const again = phoenix.start();
    const stopped = await phoenix.stop();
    assert.ok(Date.now() - began < 1_000, `the stop took ${Date.now() - began} ms`);
    assert.equal(stopped.phoenix.state, 'stopped');
    assert.equal((await starting).phoenix.state, 'starting');
    assert.deepEqual([(await again).ok, (await again).error], [false, 'Phoenix is being stopped']);
    assert.deepEqual([machine.spawned.length, machine.alive.size], [1, 0]);
    await wait(20);
    assert.equal(phoenix.snapshot().state, 'stopped', 'the ended launch is not reported as a failure');
  });
});

describe('launcher resolution', () => {
  it('prefers an explicit command, then uv, then a phoenix command', () => {
    const launch = { enabled: true, version: '21.0.0', python: '3.13', startupTimeoutMs: 10_000 };
    assert.deepEqual(
      resolveLauncher({ ...launch, command: ['phoenix', 'serve'] }, () => '/bin/uv'),
      {
        kind: 'custom',
        command: ['phoenix', 'serve'],
      },
    );
    assert.equal(resolveLauncher(launch, (name) => (name === 'phoenix' ? '/bin/phoenix' : undefined)).kind, 'phoenix');
    assert.equal(resolveLauncher(launch, () => undefined).kind, 'none');
  });
});
