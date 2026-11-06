// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * A local Arize Phoenix for NeMo Relay traces.
 *
 * `start` uses a Phoenix that already answers at `phoenix.url` (and never stops
 * it), or launches one with `uv tool run … phoenix serve` bound to 127.0.0.1,
 * with its own working directory under `<dataDir>/phoenix`, telemetry and
 * external resources off, and an environment without credentials. Traces reach
 * Phoenix through Relay's own OpenInference exporter (`PhoenixExporter`). The
 * connection is remembered in `<dataDir>/phoenix/state.json`, so the next
 * plugin start reconnects when `phoenix.autoConnect` is on.
 *
 * DeepSeek Harness processes can share the data directory. The pid file names
 * the process that launched Phoenix: every other live process uses that
 * Phoenix as one someone else runs and never stops it, and a Phoenix whose
 * launching process is gone is adopted by the next process that finds it.
 *
 * Every failure becomes state that the status file and the NeMo Relay view
 * show; no method rejects, and nothing here throws into the Host.
 */
import { execFileSync, spawn as nodeSpawn, type SpawnOptions } from 'node:child_process';
import {
  createWriteStream,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
  type WriteStream,
} from 'node:fs';
import net from 'node:net';
import { homedir } from 'node:os';
import path from 'node:path';

import type { PhoenixActionResult, PhoenixControl } from './commands.js';
import { isLoopbackUrl, type PhoenixConfig, type PhoenixLaunchConfig } from './config.js';
import type { PhoenixSnapshot, PhoenixState } from './status.js';

const HOST = '127.0.0.1';
const PORT_SCAN = 20;
const HEALTH_TIMEOUT_MS = 1500;
const POLL_MS = 750;
const STOP_GRACE_MS = 4000;
const SHUTTING_DOWN = 'NeMo Relay is shutting down';
const LOG_ROTATE_BYTES = 5 * 1024 * 1024;
const TAIL_CHARS = 4000;
/**
 * Environment a launched Phoenix inherits: paths, locale, proxies, and the uv settings it needs to download and
 * run. Model-provider API keys and other application secrets are never passed.
 */
const PASS_ENV = [
  'HOME',
  'USER',
  'LOGNAME',
  'SHELL',
  'PATH',
  'TMPDIR',
  'TMP',
  'TEMP',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'TZ',
  'SYSTEMROOT',
  'WINDIR',
  'COMSPEC',
  'APPDATA',
  'LOCALAPPDATA',
  'USERPROFILE',
  'PROGRAMDATA',
  'XDG_CACHE_HOME',
  'XDG_DATA_HOME',
  'XDG_CONFIG_HOME',
  'UV_CACHE_DIR',
  'UV_PYTHON_INSTALL_DIR',
  'UV_PYTHON_DOWNLOADS',
  'UV_INDEX_URL',
  'UV_DEFAULT_INDEX',
  'UV_EXTRA_INDEX_URL',
  'UV_NATIVE_TLS',
  'UV_OFFLINE',
  'PIP_INDEX_URL',
  'HTTP_PROXY',
  'HTTPS_PROXY',
  'ALL_PROXY',
  'NO_PROXY',
  'http_proxy',
  'https_proxy',
  'all_proxy',
  'no_proxy',
  'SSL_CERT_FILE',
  'SSL_CERT_DIR',
  'REQUESTS_CA_BUNDLE',
];

export type PhoenixMode = 'owned' | 'external';

/** Relay's export to Phoenix, provided by the runtime. */
export interface PhoenixExporter {
  /** Whether Relay runs and can export at all. */
  ready(): boolean;
  /** Start exporting OpenInference spans; throws when Relay cannot export. */
  connect(target: { endpoint: string; project: string }): void;
  /** Stop exporting; `flush` delivers queued spans first and blocks while it does. */
  disconnect(flush: boolean): Promise<void>;
  connected(): boolean;
}

/** The part of a child process this module uses. */
export interface ChildLike {
  readonly pid?: number | undefined;
  readonly stdout?: NodeJS.ReadableStream | null;
  readonly stderr?: NodeJS.ReadableStream | null;
  once(event: 'exit', listener: (code: number | null, signal: NodeJS.Signals | null) => void): unknown;
  once(event: 'error', listener: (error: NodeJS.ErrnoException) => void): unknown;
}

export type PhoenixDeps = {
  fetch(url: string, init: { signal: AbortSignal }): Promise<{ ok: boolean; text(): Promise<string> }>;
  spawn(command: string, args: string[], options: SpawnOptions): ChildLike;
  /** True when something accepts connections on 127.0.0.1:port. */
  portBusy(port: number): Promise<boolean>;
  /** An unused loopback port chosen by the OS. */
  freePort(): Promise<number>;
  kill(pid: number, signal: NodeJS.Signals | 0): void;
  /** The command line of a process, when it can be read. */
  processCommand(pid: number): string | undefined;
  findExecutable(name: string): string | undefined;
  sleep(ms: number): Promise<void>;
  /** This DeepSeek Harness process, recorded as the owner of a Phoenix it launches. */
  pid: number;
  env: NodeJS.ProcessEnv;
  platform: NodeJS.Platform;
};

export type PhoenixManagerOptions = {
  config: PhoenixConfig;
  dataDir: string;
  exporter: PhoenixExporter;
  log?: (message: string) => void;
  /** Called after every state change, for status publication. */
  onChange?: () => void;
  deps?: Partial<PhoenixDeps>;
  /** How long `start` waits before replying with a Phoenix that is still starting. */
  replyBudgetMs?: number;
  /** Health-check interval while running or waiting. */
  monitorMs?: number;
  /** How long a stop waits for Phoenix to exit before it kills it. */
  stopGraceMs?: number;
};

type Launcher = { kind: 'custom' | 'uv' | 'phoenix'; command: string[] } | { kind: 'none'; hint: string };

type State = {
  state: PhoenixState;
  url: string;
  mode?: PhoenixMode | undefined;
  version?: string | undefined;
  error?: string | undefined;
  phase?: 'checking' | 'launching' | 'downloading' | 'serving' | undefined;
};

type SavedConnection = { connected: boolean; mode: PhoenixMode; url: string };

/** How to start Phoenix: an explicit command, `uv tool run`, or a `phoenix` command on PATH. */
export function resolveLauncher(launch: PhoenixLaunchConfig, find: (name: string) => string | undefined): Launcher {
  if (launch.command !== undefined) {
    return { kind: 'custom', command: launch.command };
  }
  const uv = find('uv');
  if (uv !== undefined) {
    return {
      kind: 'uv',
      command: [
        uv,
        'tool',
        'run',
        '--python',
        launch.python,
        '--from',
        `arize-phoenix==${launch.version}`,
        'phoenix',
        'serve',
      ],
    };
  }
  const phoenix = find('phoenix');
  if (phoenix !== undefined) {
    return { kind: 'phoenix', command: [phoenix, 'serve'] };
  }
  return {
    kind: 'none',
    hint: 'Launching Phoenix needs uv (https://docs.astral.sh/uv/) or a phoenix command; install uv, or run Phoenix yourself and connect to it.',
  };
}

/** Starts, adopts, monitors, and stops Phoenix; implements the `/relay phoenix` operations. */
export class PhoenixManager implements PhoenixControl {
  private readonly config: PhoenixConfig;
  private readonly exporter: PhoenixExporter;
  private readonly deps: PhoenixDeps;
  private readonly log: (message: string) => void;
  private readonly onChange: () => void;
  private readonly replyBudgetMs: number;
  private readonly monitorMs: number;
  private readonly stopGraceMs: number;
  private readonly dir: string;
  private readonly statePath: string;
  private readonly pidPath: string;
  readonly logPath: string;
  private current: State;
  private launchHint: string | undefined;
  private child: ChildLike | undefined;
  private executable = 'the Phoenix launcher';
  private pid: number | undefined;
  /** True while the Phoenix in use belongs to another live DeepSeek Harness process: used, never stopped. */
  private borrowed = false;
  /** Launched processes this manager signalled: their exit is expected, not a failure. */
  private readonly dismissed = new WeakSet<ChildLike>();
  private logStream: WriteStream | undefined;
  private tail = '';
  private operation: Promise<void> | undefined;
  /** The public operations run one at a time, in call order. */
  private queue: Promise<unknown> = Promise.resolve();
  /** Stops, disconnections, and disposals that were called and have not finished. */
  private halts = 0;
  private epoch = 0;
  private stopping = false;
  private disposed = false;
  private monitor: ReturnType<typeof setInterval> | undefined;
  private polling = false;
  private misses = 0;
  private exitHook: (() => void) | undefined;

  constructor(options: PhoenixManagerOptions) {
    this.config = options.config;
    this.exporter = options.exporter;
    this.deps = { ...defaultDeps(), ...options.deps };
    this.log = options.log ?? (() => undefined);
    this.onChange = options.onChange ?? (() => undefined);
    this.replyBudgetMs = options.replyBudgetMs ?? 20_000;
    this.monitorMs = options.monitorMs ?? 15_000;
    this.stopGraceMs = options.stopGraceMs ?? STOP_GRACE_MS;
    this.dir = path.join(options.dataDir, 'phoenix');
    this.statePath = path.join(this.dir, 'state.json');
    this.pidPath = path.join(this.dir, 'phoenix.pid.json');
    this.logPath = path.join(this.dir, 'phoenix.log');
    this.current = { state: 'stopped', url: this.config.url };
    this.refreshLaunchHint();
  }

  /** The `phoenix` section of the status snapshot. */
  snapshot(): PhoenixSnapshot {
    const { state, url, mode, version, error, phase } = this.current;
    const connected = state === 'running' && this.exporter.connected();
    return {
      enabled: true,
      state,
      connected,
      url,
      project: this.config.project,
      ...(mode === undefined ? {} : { mode }),
      ...(version === undefined ? {} : { version }),
      ...(error === undefined ? {} : { error }),
      ...(connected ? { sessionUrl: `${url}/redirects/sessions/{sessionId}` } : {}),
      ...(phase === undefined || state !== 'starting' ? {} : { phase }),
      ...(this.launchHint === undefined || state === 'running' ? {} : { launchHint: this.launchHint }),
      ...(mode === 'owned' ? { log: this.logPath } : {}),
    };
  }

  /** Use a Phoenix that answers at the configured URL, or launch one; then export to it. */
  start(): Promise<PhoenixActionResult> {
    return this.enqueue(() => this.startNow());
  }

  /** Export to a Phoenix that already answers, without launching one. */
  connect(): Promise<PhoenixActionResult> {
    return this.enqueue(async () => {
      if (this.disposed) {
        return this.result(false, SHUTTING_DOWN);
      }
      if (this.current.state === 'running' && this.exporter.connected()) {
        return this.result(true);
      }
      await this.operation?.catch(() => undefined);
      try {
        const adopted = await this.adoptRunning(this.epoch);
        return adopted ? this.result(true) : this.result(false, `No Phoenix answers at ${this.config.url}`);
      } catch (error) {
        return this.result(false, message(error));
      }
    });
  }

  /** Stop a Phoenix this plugin started; only disconnect from one someone else runs. */
  stop(): Promise<PhoenixActionResult> {
    return this.enqueueHalt(async () => {
      if (this.disposed) {
        return this.result(false, SHUTTING_DOWN);
      }
      await this.halt({ forget: true, terminate: true });
      return this.result(true);
    });
  }

  /** Stop exporting and forget the connection; any Phoenix keeps running. */
  disconnect(): Promise<PhoenixActionResult> {
    return this.enqueueHalt(async () => {
      if (this.disposed) {
        return this.result(false, SHUTTING_DOWN);
      }
      await this.halt({ forget: true, terminate: false });
      return this.result(true);
    });
  }

  /** At plugin start, reconnect the way the user last connected. Never rejects. */
  autoConnect(): Promise<void> {
    return this.enqueue(async () => {
      try {
        const saved = this.readSaved();
        if (!this.config.autoConnect || saved === undefined || !saved.connected || this.disposed) {
          return;
        }
        if (saved.mode === 'owned') {
          await this.startNow();
          return;
        }
        // Only the configured address: a changed `phoenix.url` must not reconnect to an old one.
        if (saved.url !== this.config.url) {
          return;
        }
        const epoch = this.epoch;
        // Export before the health check answers: Relay drops a span whose start it did not see, so the
        // first turn after a restart would otherwise reach Phoenix without its turn span. The check
        // below confirms the connection or takes it back.
        const early = this.connectEarly(saved.url);
        const version = await this.version(saved.url);
        if (this.disposed || epoch !== this.epoch || this.current.state !== 'stopped') {
          return;
        }
        if (version === undefined) {
          if (early) {
            await this.exporter.disconnect(false).catch(() => undefined);
          }
          this.set({ state: 'waiting', url: saved.url, mode: 'external', version: undefined, error: undefined });
        } else {
          this.attach(saved.url, 'external', version);
        }
      } catch (error) {
        this.set({ state: 'error', error: message(error) });
      }
    });
  }

  /** Plugin disposal: stop exporting and stop what this plugin started, but remember the connection. */
  dispose(): Promise<void> {
    return this.enqueueHalt(async () => {
      if (this.disposed) {
        return;
      }
      await this.halt({ forget: false, terminate: true });
      this.disposed = true;
      this.syncMonitor();
    });
  }

  /* ------------------------------------------------------------------------ */

  /**
   * Run the public operations one at a time, in call order, so none sees
   * another half done: a start never adopts a Phoenix that a stop is still
   * terminating, and nothing starts once disposal began.
   */
  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.queue.then(operation);
    this.queue = result.catch(() => undefined);
    return result;
  }

  /** Queue a stop, disconnection, or disposal; a launch in progress ends now, so it runs promptly. */
  private enqueueHalt<T>(operation: () => Promise<T>): Promise<T> {
    this.halts += 1;
    this.interrupt();
    return this.enqueue(async () => {
      try {
        return await operation();
      } finally {
        this.halts -= 1;
      }
    });
  }

  /** Invalidate what is in flight and end a launch in progress, so its health polling ends promptly. */
  private interrupt(): void {
    if (this.disposed) {
      return;
    }
    this.epoch += 1;
    if (this.child !== undefined && this.current.state === 'starting') {
      this.signalTree(this.pid, 'SIGTERM');
    }
  }

  private async startNow(): Promise<PhoenixActionResult> {
    if (this.disposed) {
      return this.result(false, SHUTTING_DOWN);
    }
    if (this.halts > 0) {
      // A stop was called after this start: launching now would only be undone.
      return this.result(false, 'Phoenix is being stopped');
    }
    if (this.current.state === 'running' && this.exporter.connected()) {
      return this.result(true);
    }
    const operation = (this.operation ??= this.track(this.startOnce(this.epoch)));
    // A first launch can download for minutes: reply with "starting" after a while; the status file follows.
    let timer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([operation, new Promise<void>((resolve) => (timer = setTimeout(resolve, this.replyBudgetMs)))]);
    clearTimeout(timer);
    return this.result(this.current.state !== 'error', this.current.error);
  }

  private async startOnce(epoch: number): Promise<void> {
    const cancelled = () => this.disposed || this.epoch !== epoch;
    this.set({ state: 'starting', phase: 'checking', error: undefined, version: undefined });
    try {
      if (!this.exporter.ready()) {
        throw new Error('NeMo Relay is not running, so it cannot send traces to Phoenix.');
      }
      if (await this.adoptRunning(epoch)) {
        return;
      }
      if (cancelled()) {
        return;
      }
      const configured = this.config.url;
      // The user's own Phoenix from the last connection: wait for it rather than start a second one on its port.
      const saved = this.readSaved();
      if (saved?.connected === true && saved.mode === 'external' && saved.url === configured) {
        this.set({ state: 'waiting', url: configured, mode: 'external', phase: undefined });
        return;
      }
      if (!this.config.launch.enabled) {
        throw new Error(
          `No Phoenix answers at ${configured}, and launching one is turned off (phoenix.launch.enabled).`,
        );
      }
      if (!isLoopbackUrl(configured)) {
        throw new Error(
          `No Phoenix answers at ${configured}. NeMo Relay only launches Phoenix on this computer; use a 127.0.0.1 phoenix.url to launch one.`,
        );
      }
      this.refreshLaunchHint();
      const launcher = resolveLauncher(this.config.launch, this.deps.findExecutable);
      if (launcher.kind === 'none') {
        throw new Error(launcher.hint);
      }
      const port = await this.findPort(portOf(configured));
      const grpcPort = await this.deps.freePort();
      if (cancelled()) {
        return;
      }
      this.spawnPhoenix(launcher.command, port, grpcPort);
      await this.waitHealthy(`http://${HOST}:${port}`, epoch);
    } catch (error) {
      if (cancelled()) {
        return;
      }
      await this.terminateOwned();
      this.set({ state: 'error', phase: undefined, error: message(error) });
    }
  }

  /**
   * Attach to a Phoenix that already runs: the one the pid file names, else
   * one at the configured URL. The pid file's Phoenix becomes this manager's
   * own unless another live DeepSeek Harness process launched it.
   */
  private async adoptRunning(epoch: number): Promise<boolean> {
    const previous = this.readPid();
    if (previous !== undefined) {
      const url = `http://${HOST}:${previous.port}`;
      const version = await this.version(url);
      if (this.epoch !== epoch) {
        return false;
      }
      if (version !== undefined) {
        this.borrowed = previous.foreign;
        if (!previous.foreign) {
          this.pid = previous.pid;
          this.installExitHook();
          this.writePidFile(previous.pid, previous.port);
        }
        this.attach(url, previous.foreign ? 'external' : 'owned', version);
        return true;
      }
    }
    const version = await this.version(this.config.url);
    if (this.epoch !== epoch || version === undefined) {
      return false;
    }
    this.borrowed = false;
    this.attach(this.config.url, 'external', version);
    return true;
  }

  /** Start exporting to a remembered Phoenix before it is known to answer; true when exporting. */
  private connectEarly(url: string): boolean {
    if (!this.exporter.ready()) {
      return false;
    }
    try {
      this.exporter.connect({ endpoint: `${url}/v1/traces`, project: this.config.project });
      return true;
    } catch {
      return false;
    }
  }

  /** Export to a Phoenix that answers at `url`. */
  private attach(url: string, mode: PhoenixMode, version: string): void {
    try {
      this.exporter.connect({ endpoint: `${url}/v1/traces`, project: this.config.project });
    } catch (error) {
      throw new Error(`Phoenix answers at ${url}, but NeMo Relay cannot export to it: ${message(error)}`);
    }
    this.misses = 0;
    this.set({ state: 'running', url, mode, version, error: undefined, phase: undefined });
    // A Phoenix another process launched stays remembered as launched, so a start that finds it gone launches one.
    this.save({ connected: true, mode: this.borrowed ? 'owned' : mode, url });
  }

  private async halt(options: { forget: boolean; terminate: boolean }): Promise<void> {
    // A launch in progress ends now; a running Phoenix first receives the last spans and is stopped below.
    this.interrupt();
    this.stopping = true;
    try {
      await this.operation?.catch(() => undefined);
      await this.exporter.disconnect(this.current.state === 'running').catch((error: unknown) => {
        this.log(`NeMo Relay could not flush traces to Phoenix: ${message(error)}`);
      });
      const owned = this.current.mode === 'owned' || this.child !== undefined;
      if (options.terminate && owned) {
        await this.terminateOwned();
      }
      if (options.forget) {
        const saved = this.readSaved();
        this.save({
          connected: false,
          mode: saved?.mode ?? this.current.mode ?? 'external',
          url: saved?.url ?? this.current.url,
        });
      }
      const keepOwned = !options.terminate && this.pid !== undefined;
      this.borrowed = false;
      this.set({
        state: 'stopped',
        phase: undefined,
        error: undefined,
        version: undefined,
        mode: keepOwned ? 'owned' : undefined,
        url: keepOwned ? this.current.url : this.config.url,
      });
    } catch (error) {
      this.log(`NeMo Relay could not stop Phoenix cleanly: ${message(error)}`);
    } finally {
      this.stopping = false;
    }
  }

  private spawnPhoenix(command: string[], port: number, grpcPort: number): void {
    mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    try {
      if (statSync(this.logPath).size > LOG_ROTATE_BYTES) {
        renameSync(this.logPath, `${this.logPath}.1`);
      }
    } catch {
      // No log yet.
    }
    const logStream = createWriteStream(this.logPath, { flags: 'a', mode: 0o600 });
    logStream.on('error', () => undefined);
    logStream.write(`\n=== ${new Date().toISOString()} starting Phoenix on ${HOST}:${port} (${command.join(' ')})\n`);
    this.logStream = logStream;
    this.tail = '';

    const env: NodeJS.ProcessEnv = {};
    for (const key of PASS_ENV) {
      const value = this.deps.env[key];
      if (typeof value === 'string') {
        env[key] = value;
      }
    }
    const [executable = '', ...args] = command;
    this.executable = executable;
    const launcherDir = path.isAbsolute(executable) ? path.dirname(executable) : '';
    if (
      launcherDir !== '' &&
      !String(env.PATH ?? '')
        .split(path.delimiter)
        .includes(launcherDir)
    ) {
      env.PATH = [launcherDir, env.PATH].filter(Boolean).join(path.delimiter);
    }
    Object.assign(env, {
      PHOENIX_HOST: HOST,
      PHOENIX_PORT: String(port),
      PHOENIX_GRPC_PORT: String(grpcPort),
      PHOENIX_WORKING_DIR: this.dir,
      PHOENIX_TELEMETRY_ENABLED: 'false',
      PHOENIX_ALLOW_EXTERNAL_RESOURCES: 'false',
      PHOENIX_DISABLE_MODEL_COST_AUTO_UPDATE: 'true',
      PHOENIX_ENABLE_MCP_SERVER: 'false',
      PHOENIX_DISABLE_AGENT_ASSISTANT: 'true',
      PYTHONUNBUFFERED: '1',
    });

    let child: ChildLike;
    try {
      child = this.deps.spawn(executable, args, {
        cwd: this.dir,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: this.deps.platform !== 'win32',
        windowsHide: true,
      });
    } catch (error) {
      logStream.end();
      this.logStream = undefined;
      throw new Error(`Phoenix could not be launched (${executable}): ${message(error)}`);
    }
    this.child = child;
    const onOutput = (chunk: Buffer | string) => {
      const text = chunk.toString();
      logStream.write(text);
      this.tail = (this.tail + text).slice(-TAIL_CHARS);
      if (this.current.state === 'starting') {
        if (/Uvicorn running|Application startup complete|To view the Phoenix app/i.test(text)) {
          this.set({ phase: 'serving' });
        } else if (
          /Downloading|Downloaded|Resolved|Prepared|Installed/.test(text) &&
          this.current.phase !== 'serving'
        ) {
          this.set({ phase: 'downloading' });
        }
      }
    };
    child.stdout?.on('data', onOutput);
    child.stderr?.on('data', onOutput);
    child.stdout?.on('error', () => undefined);
    child.stderr?.on('error', () => undefined);
    child.once('error', (error) => this.onChildExit(child, null, null, error));
    child.once('exit', (code, signal) => this.onChildExit(child, code, signal));
    this.pid = typeof child.pid === 'number' ? child.pid : undefined;
    this.borrowed = false;
    this.installExitHook();
    if (this.pid !== undefined) {
      this.writePidFile(this.pid, port);
    }
    this.set({ state: 'starting', phase: 'launching', url: `http://${HOST}:${port}`, mode: 'owned' });
  }

  private onChildExit(
    child: ChildLike,
    code: number | null,
    signal: NodeJS.Signals | null,
    error?: NodeJS.ErrnoException,
  ): void {
    if (this.child !== child) {
      return;
    }
    this.child = undefined;
    if (this.pid === child.pid) {
      this.forgetProcess();
    }
    if (this.logStream !== undefined) {
      this.logStream.write(`\n=== Phoenix exited (${error === undefined ? (signal ?? code) : error.message})\n`);
      this.logStream.end();
      this.logStream = undefined;
    }
    if (this.disposed || this.stopping || this.dismissed.has(child)) {
      return;
    }
    const command = this.executable;
    const reason =
      error === undefined
        ? `Phoenix exited (${signal ?? `code ${code}`})`
        : error.code === 'ENOENT'
          ? `Could not run ${command}: not found`
          : error.code === 'EACCES' || error.code === 'EPERM'
            ? `Not allowed to run ${command}: ${error.message}`
            : error.message;
    const lastLine = this.lastLine();
    void this.exporter.disconnect(false).catch(() => undefined);
    this.set({ state: 'error', phase: undefined, error: `${reason}${lastLine === '' ? '' : `: ${lastLine}`}` });
  }

  private async waitHealthy(url: string, epoch: number): Promise<void> {
    const deadline = Date.now() + this.config.launch.startupTimeoutMs;
    while (Date.now() < deadline) {
      if (this.disposed || this.epoch !== epoch || this.current.state !== 'starting') {
        return;
      }
      const version = await this.version(url);
      if (version !== undefined && this.epoch === epoch && this.current.state === 'starting') {
        this.attach(url, 'owned', version);
        return;
      }
      await this.deps.sleep(POLL_MS);
    }
    throw new Error(
      `Phoenix did not start within ${Math.round(this.config.launch.startupTimeoutMs / 1000)} s. See ${this.logPath}`,
    );
  }

  private async findPort(base: number): Promise<number> {
    for (let candidate = base; candidate < base + PORT_SCAN && candidate <= 65_535; candidate += 1) {
      if (!(await this.deps.portBusy(candidate))) {
        return candidate;
      }
    }
    throw new Error(`Ports ${base}–${base + PORT_SCAN - 1} are all in use; set phoenix.url to a free port.`);
  }

  /** The Phoenix version served at `url`, or `undefined` when nothing (or not Phoenix) answers. */
  private async version(url: string): Promise<string | undefined> {
    try {
      const response = await this.deps.fetch(`${url}/arize_phoenix_version`, {
        signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
      });
      if (!response.ok) {
        return undefined;
      }
      const text = (await response.text()).trim();
      return /^[0-9]+\.[0-9]+/.test(text) && text.length <= 40 ? text : undefined;
    } catch {
      return undefined;
    }
  }

  /** Health checks: notice a Phoenix that stopped answering; reconnect a waited-for one. */
  private async check(): Promise<void> {
    if (this.polling || this.disposed || this.operation !== undefined) {
      return;
    }
    this.polling = true;
    try {
      const { state, url, mode } = this.current;
      const epoch = this.epoch;
      const version = await this.version(url);
      if (this.epoch !== epoch || this.current.state !== state || this.current.url !== url) {
        return;
      }
      if (state === 'waiting') {
        if (version !== undefined) {
          this.attach(url, 'external', version);
        }
        return;
      }
      if (state !== 'running') {
        return;
      }
      if (version !== undefined) {
        this.misses = 0;
        return;
      }
      this.misses += 1;
      if (this.misses < 2 || this.child !== undefined) {
        // A launched child reports its own exit.
        return;
      }
      await this.exporter.disconnect(false).catch(() => undefined);
      if (mode === 'external') {
        this.set({ state: 'waiting', version: undefined });
      } else {
        this.forgetProcess();
        this.set({ state: 'error', version: undefined, error: `Phoenix at ${url} stopped responding` });
      }
    } catch (error) {
      this.set({ state: 'error', error: message(error) });
    } finally {
      this.polling = false;
    }
  }

  private syncMonitor(): void {
    const wanted = !this.disposed && (this.current.state === 'running' || this.current.state === 'waiting');
    if (wanted && this.monitor === undefined) {
      this.monitor = setInterval(() => void this.check(), this.monitorMs);
      this.monitor.unref?.();
    } else if (!wanted && this.monitor !== undefined) {
      clearInterval(this.monitor);
      this.monitor = undefined;
    }
  }

  private set(patch: Partial<State>): void {
    const next = { ...this.current, ...patch };
    for (const key of Object.keys(next) as Array<keyof State>) {
      if (next[key] === undefined) {
        delete next[key];
      }
    }
    this.current = next as State;
    this.syncMonitor();
    try {
      this.onChange();
    } catch {
      // Status publication is best effort.
    }
  }

  private track(operation: Promise<void>): Promise<void> {
    const tracked = operation.catch((error: unknown) => this.set({ state: 'error', error: message(error) }));
    const settled = tracked.finally(() => {
      if (this.operation === settled) {
        this.operation = undefined;
      }
    });
    return settled;
  }

  private result(ok: boolean, error?: string): PhoenixActionResult {
    return { ok, ...(ok || error === undefined ? {} : { error }), phoenix: this.snapshot() };
  }

  private refreshLaunchHint(): void {
    try {
      const launcher = resolveLauncher(this.config.launch, this.deps.findExecutable);
      this.launchHint = !this.config.launch.enabled
        ? 'launching is turned off (phoenix.launch.enabled)'
        : launcher.kind === 'none'
          ? launcher.hint
          : undefined;
    } catch {
      this.launchHint = undefined;
    }
  }

  private readSaved(): SavedConnection | undefined {
    try {
      const value = JSON.parse(readFileSync(this.statePath, 'utf8')) as Partial<SavedConnection>;
      if (
        typeof value.connected === 'boolean' &&
        (value.mode === 'owned' || value.mode === 'external') &&
        typeof value.url === 'string'
      ) {
        return { connected: value.connected, mode: value.mode, url: value.url };
      }
    } catch {
      // Never connected, or unreadable.
    }
    return undefined;
  }

  private save(connection: SavedConnection): void {
    try {
      mkdirSync(this.dir, { recursive: true, mode: 0o700 });
      writeFileSync(this.statePath, `${JSON.stringify(connection)}\n`, { mode: 0o600 });
    } catch (error) {
      this.log(`NeMo Relay could not remember the Phoenix connection: ${message(error)}`);
    }
  }

  /**
   * The Phoenix the pid file names, when that process is alive and still a
   * Phoenix: a reused pid is never adopted. `foreign` is true when another
   * live DeepSeek Harness process launched it; a pid file without an owner, or
   * whose owner is gone, names an orphan.
   */
  private readPid(): { pid: number; port: number; foreign: boolean } | undefined {
    let value: { pid?: unknown; port?: unknown; owner?: unknown };
    try {
      value = JSON.parse(readFileSync(this.pidPath, 'utf8')) as typeof value;
    } catch {
      return undefined;
    }
    const { pid, port, owner } = value;
    if (
      typeof pid !== 'number' ||
      !Number.isInteger(pid) ||
      pid <= 0 ||
      typeof port !== 'number' ||
      !Number.isInteger(port) ||
      !this.alive(pid) ||
      !this.isPhoenix(pid)
    ) {
      this.removePidFile();
      return undefined;
    }
    const foreign =
      typeof owner === 'number' && Number.isInteger(owner) && owner > 0 && owner !== this.deps.pid && this.alive(owner);
    return { pid, port, foreign };
  }

  /** Record a Phoenix this process launched or adopted, with this process as its owner. */
  private writePidFile(pid: number, port: number): void {
    try {
      writeFileSync(
        this.pidPath,
        `${JSON.stringify({ pid, port, owner: this.deps.pid, startedAt: new Date().toISOString() })}\n`,
        { mode: 0o600 },
      );
    } catch {
      // Adopting after a crash is a convenience.
    }
  }

  private isPhoenix(pid: number): boolean {
    return /phoenix/i.test(this.deps.processCommand(pid) ?? '');
  }

  private async terminateOwned(): Promise<void> {
    const pid = this.pid;
    if (pid === undefined) {
      return;
    }
    if (this.signalTree(pid, 'SIGTERM')) {
      const deadline = Date.now() + this.stopGraceMs;
      while (Date.now() < deadline && this.alive(pid)) {
        await this.deps.sleep(150);
      }
      if (this.alive(pid)) {
        this.signalTree(pid, 'SIGKILL');
      }
    }
    this.forgetProcess();
  }

  /**
   * Signal the process group of this manager's Phoenix, falling back to the
   * process itself. Every signal is sent from here. A launched process is
   * known by its handle; an adopted one only by its pid, which the system may
   * have given to another program since, so its command line is read again
   * first.
   *
   * @returns False when the pid no longer runs Phoenix and nothing was sent.
   */
  private signalTree(pid: number | undefined, signal: NodeJS.Signals): boolean {
    if (pid === undefined) {
      return false;
    }
    if (this.child !== undefined && this.child.pid === pid) {
      this.dismissed.add(this.child);
    } else if (!this.isPhoenix(pid)) {
      return false;
    }
    if (this.deps.platform !== 'win32') {
      try {
        this.deps.kill(-pid, signal);
        return true;
      } catch {
        // Not a group leader: signal the process.
      }
    }
    try {
      this.deps.kill(pid, signal);
    } catch {
      // Already gone.
    }
    return true;
  }

  private alive(pid: number): boolean {
    try {
      this.deps.kill(pid, 0);
      return true;
    } catch (error) {
      return (error as NodeJS.ErrnoException | null)?.code === 'EPERM';
    }
  }

  private installExitHook(): void {
    if (this.exitHook !== undefined) {
      return;
    }
    // Never leave a Phoenix this plugin started behind when the Host exits.
    this.exitHook = () => void this.signalTree(this.pid, 'SIGTERM');
    process.once('exit', this.exitHook);
  }

  /** The process is gone or no longer ours: drop its pid, exit hook, and pid file. */
  private forgetProcess(): void {
    this.pid = undefined;
    if (this.exitHook !== undefined) {
      process.removeListener('exit', this.exitHook);
      this.exitHook = undefined;
    }
    this.removePidFile();
  }

  private removePidFile(): void {
    try {
      rmSync(this.pidPath, { force: true });
    } catch {
      // Best effort.
    }
  }

  private lastLine(): string {
    const lines = this.tail
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith('INFO:'));
    return (lines.at(-1) ?? '').slice(0, 300);
  }
}

function portOf(url: string): number {
  const parsed = new URL(url);
  const port = Number(parsed.port);
  return Number.isInteger(port) && port > 0 ? port : parsed.protocol === 'https:' ? 443 : 80;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function defaultDeps(): PhoenixDeps {
  const env = process.env;
  return {
    fetch: (url, init) => fetch(url, init),
    spawn: (command, args, options) => nodeSpawn(command, args, options),
    portBusy: (port) =>
      new Promise((resolve) => {
        const socket = net.connect({ host: HOST, port });
        const done = (busy: boolean) => {
          socket.removeAllListeners();
          socket.on('error', () => undefined);
          socket.destroy();
          resolve(busy);
        };
        socket.setTimeout(600, () => done(true));
        socket.once('connect', () => done(true));
        socket.once('error', () => done(false));
      }),
    freePort: () =>
      new Promise((resolve, reject) => {
        const server = net.createServer();
        server.once('error', reject);
        server.listen(0, HOST, () => {
          const address = server.address();
          server.close(() => resolve(typeof address === 'object' && address !== null ? address.port : 0));
        });
      }),
    kill: (pid, signal) => void process.kill(pid, signal),
    processCommand: (pid) => {
      if (process.platform === 'win32') {
        return undefined;
      }
      try {
        return execFileSync('ps', ['-o', 'command=', '-p', String(pid)], { encoding: 'utf8', timeout: 2000 }).trim();
      } catch {
        return undefined;
      }
    },
    findExecutable: (name) => findExecutable(name, env),
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    pid: process.pid,
    env,
    platform: process.platform,
  };
}

/** Find an executable on PATH or in the usual per-user install directories (desktop apps get a short PATH). */
function findExecutable(name: string, env: NodeJS.ProcessEnv): string | undefined {
  const home = homedir();
  const directories = [
    ...String(env.PATH ?? '').split(path.delimiter),
    path.join(home, '.local', 'bin'),
    path.join(home, '.cargo', 'bin'),
    '/opt/homebrew/bin',
    '/usr/local/bin',
  ];
  const names = process.platform === 'win32' ? [`${name}.exe`, `${name}.cmd`, name] : [name];
  for (const directory of directories) {
    if (directory.length === 0) {
      continue;
    }
    for (const file of names) {
      const candidate = path.join(directory, file);
      try {
        if (statSync(candidate).isFile()) {
          return candidate;
        }
      } catch {
        // Not here.
      }
    }
  }
  return undefined;
}
