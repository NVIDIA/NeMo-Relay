// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChildProcess, SpawnOptions } from 'node:child_process';
import { createInterface } from 'node:readline';

import { isRecord } from '../gateway.ts';
import { CLIENT_TOKEN_ENV, type DeploymentConfig } from './deployment.ts';

export type SpawnProcess = (command: string, args: readonly string[], options: SpawnOptions) => ChildProcess;

export type McpLease = {
  kind: typeof SHARED_LEASE_KIND;
  daemonAddress: string;
  dispatcherCommand: string;
  credential: string;
  ensureReady: () => Promise<void>;
  release: () => Promise<void>;
};

const MCP_SINGLETON: unique symbol = Symbol.for('nemo-relay.managed-pi.mcp.v1');
const SHARED_LEASE_KIND = 'nemo-relay-managed-pi-mcp-lease-v1';
const MCP_INITIALIZE_ID = 'nemo-relay-managed-pi-ready-v1';
const MCP_PROTOCOL_VERSION = '2025-11-25';
export type LeaseTimeouts = {
  readyMs: number;
  releaseMs: number;
  killMs: number;
};

const DEFAULT_TIMEOUTS: LeaseTimeouts = { readyMs: 180_000, releaseMs: 5_000, killMs: 1_000 };

export function sharedLease(config: DeploymentConfig, credential: string, spawnProcess: SpawnProcess): McpLease {
  const registry = globalThis as typeof globalThis & { [MCP_SINGLETON]?: unknown };
  const existing = registry[MCP_SINGLETON];
  if (existing !== undefined) {
    if (!isSharedLease(existing)) {
      throw new Error('the managed pi process-global MCP slot is already occupied');
    }
    if (
      existing.daemonAddress !== config.daemonAddress ||
      existing.dispatcherCommand !== config.dispatcherCommand ||
      existing.credential !== credential
    ) {
      throw new Error('managed pi configuration changed while the process-global MCP lease was active');
    }
    return existing;
  }
  const created = createSharedLease(
    config,
    credential,
    () => {
      if (registry[MCP_SINGLETON] === created) delete registry[MCP_SINGLETON];
    },
    spawnProcess,
  );
  registry[MCP_SINGLETON] = created;
  return created;
}

function isSharedLease(value: unknown): value is McpLease {
  return (
    isRecord(value) &&
    value.kind === SHARED_LEASE_KIND &&
    typeof value.daemonAddress === 'string' &&
    typeof value.dispatcherCommand === 'string' &&
    typeof value.credential === 'string' &&
    typeof value.ensureReady === 'function' &&
    typeof value.release === 'function'
  );
}

export function createSharedLease(
  config: DeploymentConfig,
  credential: string,
  removeFromRegistry: () => void,
  spawnProcess: SpawnProcess,
  timeouts: LeaseTimeouts = DEFAULT_TIMEOUTS,
): McpLease {
  let child: ChildProcess | undefined;
  let initialized = false;
  let starting: Promise<void> | undefined;
  let releasing: Promise<void> | undefined;
  let released = false;

  async function launch(): Promise<void> {
    const launched = spawnProcess(
      config.dispatcherCommand,
      ['daemon', 'mcp', '--daemon-address', config.daemonAddress],
      {
        shell: false,
        windowsHide: true,
        stdio: ['pipe', 'pipe', 'inherit'],
        env: { ...process.env, [CLIENT_TOKEN_ENV]: credential },
      },
    );
    child = launched;
    launched.on('error', () => undefined);
    launched.stdin?.on('error', () => undefined);
    launched.once('exit', () => {
      if (child === launched) {
        child = undefined;
        initialized = false;
        starting = undefined;
      }
    });
    try {
      await initializeMcp(launched, timeouts.readyMs);
      if (launched.exitCode !== null || launched.signalCode !== null) {
        throw new Error('managed pi MCP exited during initialization');
      }
      initialized = true;
    } catch (error) {
      if (child === launched) child = undefined;
      initialized = false;
      await terminate(launched, timeouts);
      throw error;
    }
  }

  async function releaseActive(): Promise<void> {
    released = true;
    if (starting) await starting.catch(() => undefined);
    const active = child;
    initialized = false;
    child = undefined;
    if (active) await terminate(active, timeouts);
    removeFromRegistry();
  }

  return {
    kind: SHARED_LEASE_KIND,
    daemonAddress: config.daemonAddress,
    dispatcherCommand: config.dispatcherCommand,
    credential,
    ensureReady(): Promise<void> {
      if (released) return Promise.reject(new Error('managed pi MCP lease was released'));
      if (initialized && child?.exitCode === null && child.signalCode === null) {
        return Promise.resolve();
      }
      if (starting) return starting;
      starting = launch().then(
        () => {
          starting = undefined;
        },
        (error: unknown) => {
          starting = undefined;
          throw error;
        },
      );
      return starting;
    },
    release(): Promise<void> {
      releasing ??= releaseActive();
      return releasing;
    },
  };
}

async function terminate(child: ChildProcess, timeouts: LeaseTimeouts): Promise<void> {
  child.stdin?.end();
  if (await waitForExit(child, timeouts.releaseMs)) return;
  child.kill();
  if (await waitForExit(child, timeouts.killMs)) return;
  child.kill('SIGKILL');
  await waitForExit(child, timeouts.killMs);
}

function initializeMcp(child: ChildProcess, readyMs: number): Promise<void> {
  const stdin = child.stdin;
  const stdout = child.stdout;
  if (!stdin || !stdout) {
    return Promise.reject(new Error('managed pi MCP did not expose stdio pipes'));
  }
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: stdout, crlfDelay: Infinity });
    const timer = setTimeout(() => finish(new Error('managed pi MCP initialize response timed out')), readyMs);
    let settled = false;

    const onError = (error: Error): void => finish(error);
    const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
      finish(new Error(`managed pi MCP exited before initialization (${code ?? signal ?? 'unknown'})`));
    };
    const onLine = (line: string): void => {
      let message: unknown;
      try {
        message = JSON.parse(line);
      } catch {
        return;
      }
      if (!isRecord(message) || message.id !== MCP_INITIALIZE_ID) return;
      if (isRecord(message.error)) {
        finish(new Error('managed pi MCP rejected the initialize request'));
        return;
      }
      const result = message.result;
      if (
        message.jsonrpc !== '2.0' ||
        !isRecord(result) ||
        typeof result.protocolVersion !== 'string' ||
        !isRecord(result.serverInfo) ||
        result.serverInfo.name !== 'nemo-relay'
      ) {
        finish(new Error('managed pi MCP returned an invalid initialize response'));
        return;
      }
      stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`, (error) =>
        finish(error ?? undefined),
      );
    };

    function finish(error?: Error): void {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.off('error', onError);
      child.off('exit', onExit);
      lines.off('line', onLine);
      lines.close();
      if (error) reject(error);
      else resolve();
    }

    child.once('error', onError);
    child.once('exit', onExit);
    lines.on('line', onLine);
    stdin.write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: MCP_INITIALIZE_ID,
        method: 'initialize',
        params: {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: { name: 'nemo-relay-managed-pi', version: '1.0.0' },
        },
      })}\n`,
      (error) => {
        if (error) finish(error);
      },
    );
  });
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => finish(false), timeoutMs);
    const onExit = (): void => finish(true);
    const finish = (exited: boolean): void => {
      clearTimeout(timer);
      child.off('exit', onExit);
      resolve(exited);
    };
    child.once('exit', onExit);
  });
}
