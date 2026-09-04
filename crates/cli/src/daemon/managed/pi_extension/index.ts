// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Immutable managed NeMo Relay integration for Pi.
 *
 * Deployment-specific values live in managed-config.json. Per-user state is
 * read only at runtime. The process-global MCP lease deliberately outlives Pi
 * extension reloads and new/resume/fork session replacement.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { isAbsolute } from 'node:path';
import { createInterface } from 'node:readline';

const CLIENT_TOKEN_ENV = 'NEMO_RELAY_CLIENT_TOKEN';
const CLIENT_TOKEN_HEADER = 'x-nemo-relay-client-token';
const UPSTREAM_BASE_URL_HEADER = 'x-nemo-relay-upstream-base-url';
const CONFIG_SCHEMA = 'nemo-relay-managed-pi-v1';
const CONFIG_PLACEHOLDER_PREFIX = '__NEMO_RELAY_';
const MCP_SINGLETON = Symbol.for('nemo-relay.managed-pi.mcp.v1');
const MCP_INITIALIZE_ID = 'nemo-relay-managed-pi-ready-v1';
const MCP_PROTOCOL_VERSION = '2025-11-25';
// The broker may legally keep a new MCP waiting through a 120-second worker drain before it can
// establish the next route generation. Leave reconciliation margin rather than killing a healthy
// lifecycle client during that window.
const MCP_READY_TIMEOUT_MS = 180_000;
const MCP_RELEASE_TIMEOUT_MS = 5_000;
const HOOK_TIMEOUT_MS = 30_000;
const MAX_HOOK_PAYLOAD_BYTES = 20 * 1024 * 1024;
const MAX_HOOK_RESPONSE_BYTES = 1024 * 1024;
const MAX_RESULT_CHARS = 2_000;
const SHARED_LEASE_KIND = 'nemo-relay-managed-pi-mcp-lease-v1';
const SERVICEABLE_APIS = new Set(['openai-completions', 'openai-responses', 'anthropic-messages']);

type DeploymentConfig = {
  schema: typeof CONFIG_SCHEMA;
  daemonAddress: string;
  dispatcherCommand: string;
};

type McpLease = {
  kind: typeof SHARED_LEASE_KIND;
  daemonAddress: string;
  dispatcherCommand: string;
  credential: string;
  ensureReady(): Promise<void>;
  release(): Promise<void>;
};

type HookOutcome =
  | { kind: 'allow'; body: Record<string, unknown> }
  | { kind: 'block'; reason: string }
  | { kind: 'fault'; reason: string };

type PiModel = {
  id: string;
  api: string;
  provider: string;
  baseUrl: string;
};

type ProviderRedirectDecision =
  | { kind: 'redirect'; reason: string; upstream: string }
  | { kind: 'skip'; code: string; reason: string };

type ExtensionContext = {
  cwd: string;
  model?: PiModel;
  modelRegistry?: { getAll?(): PiModel[] };
  sessionManager?: { getSessionId?(): string };
};

type SessionStartEvent = {
  type: 'session_start';
  reason: 'startup' | 'reload' | 'new' | 'resume' | 'fork';
  previousSessionFile?: string;
};

type SessionShutdownEvent = {
  type: 'session_shutdown';
  reason: 'quit' | 'reload' | 'new' | 'resume' | 'fork';
  targetSessionFile?: string;
};

type AgentEndEvent = { type: 'agent_end'; messages?: unknown[] };
type TurnEvent = { type: 'turn_start' | 'turn_end'; turnIndex: number };
type CompactEvent = {
  type: 'session_before_compact' | 'session_compact';
  reason: string;
  willRetry: boolean;
  fromExtension?: boolean;
  preparation?: { tokensBefore?: number; isSplitTurn?: boolean };
  compactionEntry?: { tokensBefore?: number };
};
type ToolExecutionStartEvent = {
  type: 'tool_execution_start';
  toolCallId: string;
  toolName: string;
};
type ToolExecutionEndEvent = {
  type: 'tool_execution_end';
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError: boolean;
};
type ToolCallEvent = {
  type: 'tool_call';
  toolCallId: string;
  toolName: string;
  input: Record<string, unknown>;
};
type ToolCallResult = { block: true; reason: string };
type UserBashEvent = {
  type: 'user_bash';
  command: string;
  cwd: string;
  excludeFromContext: boolean;
};
type UserBashResult = {
  result: {
    output: string;
    exitCode: 126;
    cancelled: boolean;
    truncated: boolean;
  };
};

type ExtensionHandler<TEvent, TResult = void> = (
  event: TEvent,
  context: ExtensionContext,
) => TResult | undefined | Promise<TResult | undefined>;

type ExtensionAPI = {
  on(event: 'session_start', handler: ExtensionHandler<SessionStartEvent>): void;
  on(event: 'session_shutdown', handler: ExtensionHandler<SessionShutdownEvent>): void;
  on(event: 'agent_start', handler: ExtensionHandler<{ type: 'agent_start' }>): void;
  on(event: 'agent_end', handler: ExtensionHandler<AgentEndEvent>): void;
  on(event: 'agent_settled', handler: ExtensionHandler<{ type: 'agent_settled' }>): void;
  on(event: 'turn_start', handler: ExtensionHandler<TurnEvent>): void;
  on(event: 'turn_end', handler: ExtensionHandler<TurnEvent>): void;
  on(event: 'session_before_compact', handler: ExtensionHandler<CompactEvent>): void;
  on(event: 'session_compact', handler: ExtensionHandler<CompactEvent>): void;
  on(event: 'tool_execution_start', handler: ExtensionHandler<ToolExecutionStartEvent>): void;
  on(event: 'tool_execution_end', handler: ExtensionHandler<ToolExecutionEndEvent>): void;
  on(event: 'tool_call', handler: ExtensionHandler<ToolCallEvent, ToolCallResult>): void;
  on(event: 'user_bash', handler: ExtensionHandler<UserBashEvent, UserBashResult>): void;
  on(event: 'model_select', handler: ExtensionHandler<{ type: 'model_select'; model: PiModel }>): void;
  registerProvider(name: string, config: { baseUrl: string; headers: Record<string, string> }): void;
};

type Runtime = {
  config: DeploymentConfig;
  credential: string;
  lease: McpLease;
};

export default function managedNemoRelayPi(pi: ExtensionAPI): void {
  let runtimePromise: Promise<Runtime> | undefined;
  const redirectedProviders = new Set<string>();
  let hookQueue: Promise<unknown> = Promise.resolve();
  let attemptIndex = 0;
  let turnSequence = 0;
  let userBashSequence = 0;
  const toolNames = new Map<string, string>();

  async function runtime(): Promise<Runtime> {
    runtimePromise ??= initializeRuntime();
    const active = await runtimePromise;
    // Re-establish the route if a previously ready child exited. All callers
    // share the same in-flight restart promise inside the global lease.
    await active.lease.ensureReady();
    return active;
  }

  function enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = hookQueue.then(operation, operation);
    hookQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  function attribution(): { attempt_index: number; turn_seq: number } {
    return {
      attempt_index: Math.max(0, attemptIndex - 1),
      turn_seq: Math.max(0, turnSequence - 1),
    };
  }

  function payload(
    context: ExtensionContext,
    hookEventName: string,
    fields: Record<string, unknown> = {},
  ): Record<string, unknown> {
    return {
      session_id: sessionId(context),
      hook_event_name: hookEventName,
      ...fields,
    };
  }

  async function sendObservation(body: Record<string, unknown>): Promise<void> {
    try {
      const active = await runtime();
      const outcome = await postHook(active, body);
      if (outcome.kind !== 'allow') {
        console.error(`NeMo Relay managed hook was not accepted: ${outcome.reason}`);
      }
    } catch (error) {
      console.error(`NeMo Relay managed hook failed: ${safeError(error)}`);
    }
  }

  function observe(body: Record<string, unknown>): void {
    void enqueue(() => sendObservation(body));
  }

  async function observeOrdered(body: Record<string, unknown>): Promise<void> {
    await enqueue(() => sendObservation(body));
  }

  function registerManagedProvider(
    active: Runtime,
    model: PiModel | undefined,
    context: ExtensionContext,
    source: 'session_start' | 'model_select',
  ): void {
    if (!model || redirectedProviders.has(model.provider)) return;
    const catalog = context.modelRegistry?.getAll?.();
    const decision = decideManagedProviderRedirect(model, catalog);
    if (decision.kind === 'skip') {
      observe(
        payload(context, 'model_redirect', {
          source,
          outcome: 'skip',
          code: decision.code,
          reason: decision.reason,
          provider: model.provider,
          model_api: model.api,
          model_id: model.id,
          ...attribution(),
        }),
      );
      return;
    }
    const providerConfig = {
      baseUrl: active.config.daemonAddress,
      headers: {
        [CLIENT_TOKEN_HEADER]: active.credential,
        [UPSTREAM_BASE_URL_HEADER]: decision.upstream,
      },
    };
    // Pi resolves the concrete provider endpoint before Relay redirects the provider. Preserve
    // that endpoint on the provider-wide registration so the authenticated daemon route reaches
    // the same destination Pi would have called directly, including custom providers.
    pi.registerProvider(model.provider, providerConfig);
    redirectedProviders.add(model.provider);
    observe(
      payload(context, 'model_redirect', {
        source,
        outcome: 'redirect',
        reason: decision.reason,
        provider: model.provider,
        model_api: model.api,
        model_id: model.id,
        ...attribution(),
      }),
    );
  }

  function refuseUserBash(context: ExtensionContext, callId: string, reason: string): UserBashResult {
    observe(
      payload(context, 'user_bash_end', {
        tool_call_id: callId,
        tool_name: 'user_bash',
        status: 'error',
        result: { content: reason },
        ...attribution(),
      }),
    );
    return refusedBash(reason);
  }

  pi.on('session_start', async (event, context) => {
    const active = await runtime();
    await observeOrdered(
      payload(context, 'session_start', {
        reason: event.reason,
        cwd: context.cwd,
        ...(event.previousSessionFile ? { previous_session_file: event.previousSessionFile } : {}),
      }),
    );
    // Registration happens after the MCP initialize response. A provider call
    // therefore cannot reach the daemon before this process owns a broker route.
    registerManagedProvider(active, context.model, context, 'session_start');
  });

  pi.on('model_select', async (event, context) => {
    const active = await runtime();
    registerManagedProvider(active, event.model, context, 'model_select');
  });

  pi.on('session_shutdown', async (event, context) => {
    if (event.reason === 'reload') {
      await hookQueue;
      return;
    }
    await observeOrdered(
      payload(context, 'session_shutdown', {
        reason: event.reason,
        ...(event.targetSessionFile ? { target_session_file: event.targetSessionFile } : {}),
      }),
    );
    await hookQueue;
    // New/resume/fork replace the Pi session inside the same process. Keeping
    // the global lease avoids a zero-reference drain and worker restart.
    if (event.reason === 'quit') {
      const active = await runtime();
      await active.lease.release();
    }
  });

  pi.on('agent_start', async (_event, context) => {
    observe(payload(context, 'agent_start', { attempt_index: attemptIndex }));
    attemptIndex += 1;
  });

  pi.on('agent_end', async (event, context) => {
    observe(
      payload(context, 'agent_end', {
        attempt_index: Math.max(0, attemptIndex - 1),
        message_count: event.messages?.length ?? 0,
      }),
    );
  });

  pi.on('agent_settled', async (_event, context) => {
    observe(
      payload(context, 'agent_settled', {
        attempts: attemptIndex,
        ...attribution(),
      }),
    );
    attemptIndex = 0;
  });

  pi.on('turn_start', async (event, context) => {
    const sequence = turnSequence;
    turnSequence += 1;
    await observeOrdered(
      payload(context, 'turn_start', {
        turn_index: event.turnIndex,
        turn_seq: sequence,
        attempt_index: Math.max(0, attemptIndex - 1),
      }),
    );
  });

  pi.on('turn_end', async (event, context) => {
    await observeOrdered(
      payload(context, 'turn_end', {
        turn_index: event.turnIndex,
        ...attribution(),
      }),
    );
  });

  pi.on('session_before_compact', async (event, context) => {
    observe(
      payload(context, 'session_before_compact', {
        reason: event.reason,
        will_retry: event.willRetry,
        tokens_before: event.preparation?.tokensBefore,
        is_split_turn: event.preparation?.isSplitTurn,
        ...attribution(),
      }),
    );
  });

  pi.on('session_compact', async (event, context) => {
    observe(
      payload(context, 'session_compact', {
        reason: event.reason,
        will_retry: event.willRetry,
        from_extension: event.fromExtension,
        tokens_before: event.compactionEntry?.tokensBefore,
        ...attribution(),
      }),
    );
  });

  pi.on('tool_execution_start', async (event) => {
    toolNames.set(event.toolCallId, event.toolName);
  });

  pi.on('tool_call', async (event, context) => {
    let outcome: HookOutcome;
    try {
      outcome = await enqueue(async () => {
        const active = await runtime();
        return postHook(
          active,
          payload(context, 'tool_call', {
            tool_call_id: event.toolCallId,
            tool_name: event.toolName,
            input: event.input,
            ...attribution(),
          }),
        );
      });
    } catch (error) {
      return blockedInfrastructure(event.toolName, safeError(error));
    }

    if (outcome.kind === 'fault') {
      return blockedInfrastructure(event.toolName, outcome.reason);
    }
    if (outcome.kind === 'block') {
      return { block: true, reason: outcome.reason };
    }
    const transformed = decideManagedToolTransform(outcome.body, event.toolCallId, event.input);
    if (transformed.kind === 'invalid') {
      return {
        block: true,
        reason: `NeMo Relay returned an invalid argument rewrite: ${transformed.reason}`,
      };
    }
    if (transformed.kind === 'replace') {
      // Pi executes the same object after this hook and does not revalidate it. The recursive
      // shape check below proves every assignment preserves the already-validated structure.
      Object.assign(event.input, transformed.input);
      observe(
        payload(context, 'tool_arguments_transformed', {
          tool_call_id: event.toolCallId,
          tool_name: event.toolName,
          ...attribution(),
        }),
      );
    }
    return undefined;
  });

  pi.on('tool_execution_end', async (event, context) => {
    const toolName = event.toolName || toolNames.get(event.toolCallId) || 'unknown';
    toolNames.delete(event.toolCallId);
    observe(
      payload(context, 'tool_execution_end', {
        tool_call_id: event.toolCallId,
        tool_name: toolName,
        result: summarizeManagedToolResult(event.result, event.isError),
        status: event.isError ? 'error' : 'ok',
        ...attribution(),
      }),
    );
  });

  pi.on('user_bash', async (event, context) => {
    const callId = `user-bash-${userBashSequence++}`;
    let outcome: HookOutcome;
    try {
      outcome = await enqueue(async () => {
        const active = await runtime();
        return postHook(
          active,
          payload(context, 'user_bash', {
            tool_call_id: callId,
            tool_name: 'user_bash',
            input: {
              command: event.command,
              cwd: event.cwd,
              exclude_from_context: event.excludeFromContext,
            },
            ...attribution(),
          }),
        );
      });
    } catch (error) {
      return refuseUserBash(context, callId, blockedInfrastructure('user_bash', safeError(error)).reason);
    }
    if (outcome.kind === 'fault') {
      return refuseUserBash(context, callId, blockedInfrastructure('user_bash', outcome.reason).reason);
    }
    if (outcome.kind === 'block') {
      return refuseUserBash(context, callId, outcome.reason);
    }
    const original = {
      command: event.command,
      cwd: event.cwd,
      exclude_from_context: event.excludeFromContext,
    };
    const transformed = decideManagedToolTransform(outcome.body, callId, original);
    if (transformed.kind === 'invalid') {
      return refuseUserBash(context, callId, `NeMo Relay returned an invalid argument rewrite: ${transformed.reason}`);
    }
    if (transformed.kind === 'replace') {
      return refuseUserBash(
        context,
        callId,
        'NeMo Relay rewrote this inline command, but Pi cannot safely apply inline-shell rewrites.',
      );
    }
    observe(
      payload(context, 'user_bash_end', {
        tool_call_id: callId,
        tool_name: 'user_bash',
        status: 'policy-allowed',
        result: { content: 'Allowed by policy; Pi does not expose the command outcome.' },
        ...attribution(),
      }),
    );
    return undefined;
  });
}

async function initializeRuntime(): Promise<Runtime> {
  const config = readDeploymentConfig();
  const credential = readCredential();
  const lease = sharedLease(config, credential);
  return { config, credential, lease };
}

function readDeploymentConfig(): DeploymentConfig {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(new URL('./managed-config.json', import.meta.url), 'utf8'));
  } catch (error) {
    throw new Error(`managed Pi configuration is unreadable: ${safeError(error)}`);
  }
  if (!isRecord(parsed) || parsed.schema !== CONFIG_SCHEMA) {
    throw new Error(`managed Pi configuration must use schema ${CONFIG_SCHEMA}`);
  }
  const daemonAddress = requiredRenderedString(parsed.daemonAddress, 'daemonAddress');
  const dispatcherCommand = requiredRenderedString(parsed.dispatcherCommand, 'dispatcherCommand');
  if (!isAbsolute(dispatcherCommand)) {
    throw new Error('managed Pi dispatcherCommand must be an absolute administrator-owned path');
  }
  let daemon: URL;
  try {
    daemon = new URL(daemonAddress);
  } catch {
    throw new Error('managed Pi daemonAddress must be an absolute HTTP(S) URL');
  }
  if (
    !['http:', 'https:'].includes(daemon.protocol) ||
    daemon.username !== '' ||
    daemon.password !== '' ||
    daemon.search !== '' ||
    daemon.hash !== '' ||
    !['', '/'].includes(daemon.pathname)
  ) {
    throw new Error('managed Pi daemonAddress must be a root HTTP(S) URL without credentials, query, or fragment');
  }
  return {
    schema: CONFIG_SCHEMA,
    daemonAddress: daemon.href.replace(/\/$/, ''),
    dispatcherCommand,
  };
}

function requiredRenderedString(value: unknown, field: string): string {
  if (
    typeof value !== 'string' ||
    value.length === 0 ||
    value.includes('\0') ||
    value.startsWith(CONFIG_PLACEHOLDER_PREFIX)
  ) {
    throw new Error(`managed Pi ${field} was not rendered by the administrator`);
  }
  return value;
}

function readCredential(): string {
  const value = process.env[CLIENT_TOKEN_ENV];
  if (
    value === undefined ||
    !/^[A-Za-z0-9_-]{43}$/.test(value) ||
    Buffer.from(value, 'base64url').length !== 32 ||
    Buffer.from(value, 'base64url').toString('base64url') !== value
  ) {
    throw new Error(`${CLIENT_TOKEN_ENV} must be an unpadded base64url credential encoding exactly 32 bytes`);
  }
  return value;
}

function sharedLease(config: DeploymentConfig, credential: string): McpLease {
  const registry = globalThis as unknown as Record<symbol, unknown>;
  const existing = registry[MCP_SINGLETON];
  if (existing !== undefined) {
    if (!isSharedLease(existing)) {
      throw new Error('the managed Pi process-global MCP slot is already occupied');
    }
    if (
      existing.daemonAddress !== config.daemonAddress ||
      existing.dispatcherCommand !== config.dispatcherCommand ||
      existing.credential !== credential
    ) {
      throw new Error('managed Pi configuration changed while the process-global MCP lease was active');
    }
    return existing;
  }

  const created = createSharedLease(config, credential, () => {
    if (registry[MCP_SINGLETON] === created) delete registry[MCP_SINGLETON];
  });
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

function createSharedLease(config: DeploymentConfig, credential: string, removeFromRegistry: () => void): McpLease {
  let child: ChildProcess | undefined;
  let initialized = false;
  let starting: Promise<void> | undefined;
  let releasing: Promise<void> | undefined;
  let released = false;

  const lease: McpLease = {
    kind: SHARED_LEASE_KIND,
    daemonAddress: config.daemonAddress,
    dispatcherCommand: config.dispatcherCommand,
    credential,
    ensureReady(): Promise<void> {
      if (released) return Promise.reject(new Error('managed Pi MCP lease was released'));
      if (initialized && child && child.exitCode === null && child.signalCode === null) {
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

  async function launch(): Promise<void> {
    const launched = spawn(config.dispatcherCommand, ['daemon', 'mcp', '--daemon-address', config.daemonAddress], {
      shell: false,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, [CLIENT_TOKEN_ENV]: credential },
    });
    child = launched;
    launched.stdin?.on('error', () => undefined);
    launched.once('exit', () => {
      if (child === launched) {
        child = undefined;
        initialized = false;
        starting = undefined;
      }
    });
    try {
      await initializeMcp(launched);
      if (launched.exitCode !== null || launched.signalCode !== null) {
        throw new Error('managed Pi MCP exited during initialization');
      }
      initialized = true;
    } catch (error) {
      if (child === launched) child = undefined;
      initialized = false;
      launched.kill();
      throw error;
    }
  }

  async function releaseActive(): Promise<void> {
    released = true;
    if (starting) await starting.catch(() => undefined);
    const active = child;
    initialized = false;
    child = undefined;
    if (active) {
      active.stdin?.end();
      if (!(await waitForExit(active, MCP_RELEASE_TIMEOUT_MS))) {
        active.kill();
        await waitForExit(active, 1_000);
      }
    }
    removeFromRegistry();
  }

  return lease;
}

function initializeMcp(child: ChildProcess): Promise<void> {
  const stdin = child.stdin;
  const stdout = child.stdout;
  if (!stdin || !stdout) {
    return Promise.reject(new Error('managed Pi MCP did not expose stdio pipes'));
  }
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: stdout, crlfDelay: Infinity });
    const timer = setTimeout(
      () => finish(new Error('managed Pi MCP initialize response timed out')),
      MCP_READY_TIMEOUT_MS,
    );
    let settled = false;

    const onError = (error: Error): void => finish(error);
    const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
      finish(new Error(`managed Pi MCP exited before initialization (${code ?? signal ?? 'unknown'})`));
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
        finish(new Error('managed Pi MCP rejected the initialize request'));
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
        finish(new Error('managed Pi MCP returned an invalid initialize response'));
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

async function postHook(runtime: Runtime, payload: Record<string, unknown>): Promise<HookOutcome> {
  let encoded: string;
  try {
    encoded = JSON.stringify(payload);
  } catch (error) {
    return { kind: 'fault', reason: `hook payload is not JSON-safe: ${safeError(error)}` };
  }
  if (Buffer.byteLength(encoded) > MAX_HOOK_PAYLOAD_BYTES) {
    return { kind: 'fault', reason: 'hook payload exceeds the managed payload limit' };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
  try {
    const response = await fetch(`${runtime.config.daemonAddress}/hooks/pi`, {
      method: 'POST',
      redirect: 'error',
      headers: {
        'content-type': 'application/json',
        [CLIENT_TOKEN_HEADER]: runtime.credential,
      },
      body: encoded,
      signal: controller.signal,
    });
    const decoded = await boundedJson(response);
    if (response.ok) {
      if (!isRecord(decoded)) {
        return { kind: 'fault', reason: 'daemon returned a non-object success body' };
      }
      return { kind: 'allow', body: decoded };
    }
    if (response.status === 403 && isRecord(decoded)) {
      const detail = decoded.error;
      if (isRecord(detail) && detail.type === 'nemo_relay_guardrail_rejected' && typeof detail.reason === 'string') {
        return { kind: 'block', reason: detail.reason };
      }
    }
    return { kind: 'fault', reason: `daemon returned HTTP ${response.status}` };
  } catch (error) {
    const reason =
      error instanceof Error && error.name === 'AbortError'
        ? `daemon did not answer within ${HOOK_TIMEOUT_MS}ms`
        : `daemon hook request failed: ${safeError(error)}`;
    return { kind: 'fault', reason };
  } finally {
    clearTimeout(timer);
  }
}

async function boundedJson(response: Response): Promise<unknown> {
  if (!response.body) return null;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    length += part.value.byteLength;
    if (length > MAX_HOOK_RESPONSE_BYTES) {
      await reader.cancel();
      throw new Error('daemon hook response exceeds the managed response limit');
    }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    return null;
  }
}

export function decideManagedProviderRedirect(
  model: PiModel | undefined,
  catalog: readonly PiModel[] | undefined,
): ProviderRedirectDecision {
  if (!model) return { kind: 'skip', code: 'no-model', reason: 'no model is selected' };
  if (!SERVICEABLE_APIS.has(model.api)) {
    return {
      kind: 'skip',
      code: 'unserviceable-api',
      reason: `the managed daemon serves no route for the ${model.api} API`,
    };
  }
  if (!catalog) {
    return {
      kind: 'skip',
      code: 'model-registry-unavailable',
      reason: 'Pi did not expose the provider catalog required for a safe provider-wide redirect',
    };
  }
  const siblings = catalog.filter((candidate) => candidate.provider === model.provider);
  if (siblings.length === 0) {
    return {
      kind: 'skip',
      code: 'provider-catalog-missing',
      reason: `Pi's model registry contains no models for provider ${model.provider}`,
    };
  }
  const unsupported = siblings.find((candidate) => !SERVICEABLE_APIS.has(candidate.api));
  if (unsupported) {
    return {
      kind: 'skip',
      code: 'provider-mixed-apis',
      reason:
        `redirecting ${model.provider} would also move its unsupported ` + `${unsupported.api} model ${unsupported.id}`,
    };
  }
  const upstream = normalizeBaseUrl(model.baseUrl);
  const mismatched = siblings.find((candidate) => normalizeBaseUrl(candidate.baseUrl) !== upstream);
  if (mismatched) {
    return {
      kind: 'skip',
      code: 'provider-mixed-endpoints',
      reason:
        `redirecting ${model.provider} would also move ${mismatched.id}, which targets ` +
        `${mismatched.baseUrl} rather than ${model.baseUrl}`,
    };
  }
  return {
    kind: 'redirect',
    upstream: model.baseUrl,
    reason: 'provider uses only daemon-supported APIs and every model shares its endpoint',
  };
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  try {
    const url = new URL(trimmed);
    const path = url.pathname.replace(/\/+$/, '');
    return `${url.protocol}//${url.host.toLowerCase()}${path}`;
  } catch {
    return trimmed.toLowerCase();
  }
}

export function decideManagedToolTransform(
  body: Record<string, unknown>,
  callId: string,
  current: Record<string, unknown>,
): { kind: 'none' } | { kind: 'replace'; input: Record<string, unknown> } | { kind: 'invalid'; reason: string } {
  const toolCall = body.tool_call;
  if (toolCall === undefined) return { kind: 'none' };
  if (!isRecord(toolCall)) return { kind: 'invalid', reason: 'tool_call is not an object' };
  if (toolCall.input === undefined) return { kind: 'none' };
  if (typeof toolCall.tool_call_id !== 'string' || toolCall.tool_call_id !== callId) {
    return { kind: 'invalid', reason: 'tool_call_id does not match the active call' };
  }
  if (!isRecord(toolCall.input)) {
    return { kind: 'invalid', reason: 'tool_call.input is not an object' };
  }
  const violation = shapeViolation(current, toolCall.input);
  if (violation) return { kind: 'invalid', reason: violation };
  return { kind: 'replace', input: toolCall.input };
}

function jsonType(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function shapeViolation(current: unknown, next: unknown, path = 'input'): string | null {
  const currentType = jsonType(current);
  const nextType = jsonType(next);
  if (currentType !== nextType) {
    return `${path} changed type from ${currentType} to ${nextType}`;
  }
  if (currentType === 'object') {
    const currentRecord = current as Record<string, unknown>;
    const nextRecord = next as Record<string, unknown>;
    const currentKeys = Object.keys(currentRecord).sort((left, right) => left.localeCompare(right));
    const nextKeys = Object.keys(nextRecord).sort((left, right) => left.localeCompare(right));
    const added = nextKeys.filter((key) => !currentKeys.includes(key));
    const removed = currentKeys.filter((key) => !nextKeys.includes(key));
    if (added.length > 0) return `${path} added ${added.join(', ')}`;
    if (removed.length > 0) return `${path} removed ${removed.join(', ')}`;
    for (const key of currentKeys) {
      const violation = shapeViolation(currentRecord[key], nextRecord[key], `${path}.${key}`);
      if (violation) return violation;
    }
  }
  if (currentType === 'array') {
    const currentItems = current as unknown[];
    const nextItems = next as unknown[];
    if (currentItems.length !== nextItems.length) {
      return `${path} changed length from ${currentItems.length} to ${nextItems.length}`;
    }
    for (const [index, item] of currentItems.entries()) {
      const violation = shapeViolation(item, nextItems[index], `${path}[${index}]`);
      if (violation) return violation;
    }
  }
  return null;
}

function blockedInfrastructure(toolName: string, detail: string): ToolCallResult {
  return {
    block: true,
    reason:
      `The managed NeMo Relay service could not authorize this ${toolName} call, so it was ` +
      `blocked rather than allowed through unchecked. Details: ${detail}`,
  };
}

function refusedBash(reason: string): UserBashResult {
  return {
    result: {
      output: `NeMo Relay blocked this inline shell command: ${reason}`,
      exitCode: 126,
      cancelled: false,
      truncated: false,
    },
  };
}

function sessionId(context: ExtensionContext): string {
  try {
    return context.sessionManager?.getSessionId?.() ?? 'unknown-session';
  } catch {
    return 'unknown-session';
  }
}

export function summarizeManagedToolResult(result: unknown, isError: boolean): Record<string, unknown> {
  if (result === null || result === undefined) {
    return { content: isError ? 'Tool failed with no result.' : 'Tool completed with no result.' };
  }
  if (typeof result === 'string') return { content: truncate(result) };
  if (isRecord(result)) {
    const content = result.content ?? result.output ?? result.text;
    const text = toolResultText(content);
    return {
      content: text === null ? `Tool ${isError ? 'failed' : 'completed'}.` : text,
      result_keys: Object.keys(result).slice(0, 20),
    };
  }
  return { content: primitiveSummary(result, isError) };
}

function primitiveSummary(result: unknown, isError: boolean): string {
  switch (typeof result) {
    case 'boolean':
    case 'number':
    case 'bigint':
    case 'symbol':
      return truncate(String(result));
    default:
      return `Tool ${isError ? 'failed' : 'completed'} with an unsupported result type.`;
  }
}

function toolResultText(content: unknown): string | null {
  if (typeof content === 'string') return truncate(content);
  if (!Array.isArray(content)) return null;

  let text = '';
  let omittedChars = 0;
  let foundText = false;
  const append = (value: string): void => {
    const kept = sliceAtCodePointBoundary(value, Math.max(0, MAX_RESULT_CHARS - text.length));
    text += kept;
    omittedChars += value.length - kept.length;
  };
  for (const part of content) {
    if (!isRecord(part) || part.type !== 'text' || typeof part.text !== 'string') continue;
    if (foundText) append('\n');
    append(part.text);
    foundText = true;
  }
  if (!foundText) return null;
  return omittedChars === 0 ? text : `${text}... [truncated ${omittedChars} chars]`;
}

function sliceAtCodePointBoundary(value: string, limit: number): string {
  let end = Math.min(value.length, limit);
  if (
    end > 0 &&
    end < value.length &&
    value.charCodeAt(end - 1) >= 0xd800 &&
    value.charCodeAt(end - 1) <= 0xdbff &&
    value.charCodeAt(end) >= 0xdc00 &&
    value.charCodeAt(end) <= 0xdfff
  ) {
    end -= 1;
  }
  return value.slice(0, end);
}

function truncate(value: string): string {
  if (value.length <= MAX_RESULT_CHARS) return value;
  const kept = sliceAtCodePointBoundary(value, MAX_RESULT_CHARS);
  return `${kept}... [truncated ${value.length - kept.length} chars]`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function safeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
