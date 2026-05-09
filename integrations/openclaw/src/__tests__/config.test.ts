// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { describe, it } from "node:test";

import {
  NEMO_FLOW_OPENCLAW_JSON_SCHEMA,
  nemoFlowConfigSchema,
  parseConfig,
} from "../config.js";
import type { NemoFlowModuleLoader, NemoFlowModules, NemoFlowRuntimeModule } from "../modules.js";
import { registerNemoFlowPlugin } from "../runtime-state.js";
import type { OpenClawPluginApi, PluginLogger } from "openclaw/plugin-sdk/plugin-entry";
import { callGatewayStatus, type TestGatewayMethodHandler } from "./gateway-status.js";

describe("nemo-flow OpenClaw plugin shell", () => {
  it("applies hook-backend config defaults", () => {
    const config = parseConfig(undefined);

    assert.equal(config.enabled, true);
    assert.equal(config.backend, "hooks");
    assert.deepEqual(config.nemoFlow.pluginConfig, { version: 1, components: [] });
    assert.equal(config.atif.enabled, true);
    assert.equal(config.atif.agentName, "openclaw");
    assert.equal(config.telemetry.otel.enabled, false);
    assert.equal(config.telemetry.otel.transport, "http_binary");
    assert.equal(config.telemetry.otel.instrumentationScope, "nemo-flow-otel");
    assert.equal(config.telemetry.openInference.enabled, false);
    assert.equal(
      config.telemetry.openInference.instrumentationScope,
      "nemo-flow-openinference",
    );
    assert.deepEqual(config.capture, {
      includePrompts: true,
      includeResponses: true,
      stripToolArgs: true,
      stripToolResults: true,
    });
    assert.deepEqual(config.correlation, {
      llmOutputGraceMs: 250,
      recordTtlMs: 600_000,
      maxRecordsPerKey: 32,
    });
  });

  it("rejects unsupported backends", () => {
    assert.throws(
      () => parseConfig({ backend: "managed_execution" }),
      /unsupported nemo-flow backend: managed_execution/,
    );
  });

  it("rejects invalid correlation and timeout values", () => {
    assert.throws(
      () => parseConfig({ correlation: { llmOutputGraceMs: -1 } }),
      /correlation\.llmOutputGraceMs must be a non-negative integer/,
    );
    assert.throws(
      () => parseConfig({ correlation: { recordTtlMs: 1.5 } }),
      /correlation\.recordTtlMs must be a non-negative integer/,
    );
    assert.throws(
      () => parseConfig({ correlation: { maxRecordsPerKey: 0 } }),
      /correlation\.maxRecordsPerKey must be a positive integer/,
    );
    assert.throws(
      () => parseConfig({ telemetry: { otel: { timeoutMillis: 2.5 } } }),
      /telemetry\.otel\.timeoutMillis must be a non-negative integer/,
    );
  });

  it("wraps manifest JSON Schema in OpenClawPluginConfigSchema", () => {
    assert.equal(typeof nemoFlowConfigSchema.safeParse, "function");
    assert.deepEqual(nemoFlowConfigSchema.jsonSchema, NEMO_FLOW_OPENCLAW_JSON_SCHEMA);
    assert.equal(nemoFlowConfigSchema.safeParse?.({ backend: "hooks" }).success, true);
    assert.equal(nemoFlowConfigSchema.safeParse?.({ backend: "bad" }).success, false);
  });

  it("returns without side effects outside full registration mode", () => {
    const api = createApi({ registrationMode: "discovery" });

    registerPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.equal(api.calls.hooks.length, 0);
  });

  it("returns without side effects when disabled", () => {
    const api = createApi({ pluginConfig: { enabled: false } });

    registerPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.equal(api.calls.hooks.length, 0);
    assert.deepEqual(api.messages.info, ["nemo-flow observability disabled by plugin config"]);
  });

  it("returns without side effects when config parsing fails during registration", () => {
    const api = createApi({ pluginConfig: { backend: "managed_execution" } });

    registerPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.equal(api.calls.hooks.length, 0);
    assert.match(
      api.messages.warn[0] ?? "",
      /nemo-flow observability disabled because plugin config is invalid/,
    );
  });

  it("registers service, lifecycle, and health surfaces in full mode", () => {
    const api = createApi();

    registerPlugin(api, async () => createModules());

    assert.deepEqual(
      api.calls.services.map((service) => service.id),
      ["nemo-flow-observability"],
    );
    assert.deepEqual(
      api.calls.lifecycle.map((lifecycle) => lifecycle.id),
      ["nemo-flow-observability-cleanup"],
    );
    assert.deepEqual(
      api.calls.gatewayMethods.map((method) => method.method),
      ["nemoFlow.status"],
    );
    assert.deepEqual(
      api.calls.hooks.map((hook) => hook.hookName),
      [
        "gateway_start",
        "gateway_stop",
        "session_start",
        "session_end",
        "llm_input",
        "llm_output",
        "model_call_started",
        "model_call_ended",
        "after_tool_call",
        "before_message_write",
        "agent_end",
        "before_agent_finalize",
        "subagent_spawned",
        "subagent_ended",
      ],
    );
  });

  it("uses config parsed during registration when service starts", async () => {
    const api = createApi({ pluginConfig: { atif: { enabled: false }, correlation: { maxRecordsPerKey: 1 } } });

    registerPlugin(api, async () => createModules());
    api.pluginConfig = { backend: "managed_execution" };

    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await assert.doesNotReject(async () => {
        await service.start({
          stateDir: "/tmp/openclaw-state",
          config: {} as never,
          logger: api.logger,
        });
      });
    } finally {
      await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
    }
  });

  it("continues hook-backed telemetry when plugin host validation fails", async () => {
    const modules = createModules({
      validateDiagnostics: [{ level: "error", code: "bad_config", message: "invalid" }],
    });
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

      const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
      assert.ok(sessionStart);
      await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });

      const status = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
      assert.deepEqual(modules.nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);
      assert.equal(status.status.state, "degraded");
      assert.equal(status.initializedPluginHost, false);
    } finally {
      await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
    }
  });

  it("routes gateway_stop through runtime stop", async () => {
    const modules = createModules();
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

    const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
    const gatewayStop = api.calls.hooks.find((hook) => hook.hookName === "gateway_stop");
    assert.ok(sessionStart);
    assert.ok(gatewayStop);
    await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });
    await gatewayStop.handler({ reason: "test_stop" }, {});

    const status = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(status.status.state, "stopped");
    assert.equal(status.counters.marksEmitted, 2);
    assert.deepEqual(modules.nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.session_end",
    ]);
  });

  it("keeps the runtime running for scoped lifecycle cleanup", async () => {
    const modules = createModules();
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    const lifecycle = api.calls.lifecycle[0];
    assert.ok(service);
    assert.ok(lifecycle?.cleanup);
    await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

    const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
    assert.ok(sessionStart);
    await sessionStart.handler({ sessionId: "session-1", sessionKey: "agent:main:session-1" }, {
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
    });

    await lifecycle.cleanup({ reason: "restart", sessionKey: "agent:main:session-1" });

    const statusAfterScopedCleanup = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(statusAfterScopedCleanup.status.state, "ready");
    assert.equal(statusAfterScopedCleanup.counters.marksEmitted, 2);

    await sessionStart.handler({ sessionId: "session-2" }, { sessionId: "session-2" });

    const statusAfterNextHook = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(statusAfterNextHook.status.state, "ready");
    assert.equal(statusAfterNextHook.counters.marksEmitted, 3);
    assert.deepEqual(modules.nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.session_end",
      "openclaw.session_start",
    ]);

    await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
  });

  it("restarts hook replay after unscoped runtime restart cleanup", async () => {
    const modules = createModules();
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    const lifecycle = api.calls.lifecycle[0];
    assert.ok(service);
    assert.ok(lifecycle?.cleanup);
    await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

    await lifecycle.cleanup({ reason: "restart" });
    const statusAfterRestart = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(statusAfterRestart.status.state, "not_initialized");
    assert.equal(statusAfterRestart.status.reason, "restart");

    const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
    assert.ok(sessionStart);
    await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });

    const statusAfterNextHook = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(statusAfterNextHook.status.state, "ready");
    assert.equal(statusAfterNextHook.counters.marksEmitted, 1);
    assert.deepEqual(modules.nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);

    await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
  });

  it("starts hook replay from the OpenClaw runtime when service start has not run", async () => {
    const modules = createModules();
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerPlugin(api, async () => modules);
    const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
    assert.ok(sessionStart);

    await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });

    const statusAfterHook = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
    assert.equal(statusAfterHook.status.state, "ready");
    assert.equal(statusAfterHook.counters.marksEmitted, 1);
    assert.deepEqual(modules.nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);

    const service = api.calls.services[0];
    assert.ok(service);
    await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
  });

  it("registers and shuts down telemetry subscribers in order", async () => {
    const modules = createModules();
    const api = createApi({
      pluginConfig: {
        telemetry: {
          otel: { enabled: true, endpoint: "http://otel.example" },
          openInference: { enabled: true, endpoint: "http://phoenix.example" },
        },
      },
    });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

    assert.deepEqual(
      modules.nf.calls.subscribers.map((subscriber) => [subscriber.kind, subscriber.name]),
      [
        ["otel", "openclaw.nemo-flow.otel"],
        ["openInference", "openclaw.nemo-flow.openinference"],
      ],
    );
    assert.equal(modules.nf.calls.subscribers[0]?.config.endpoint, "http://otel.example");
    assert.equal(modules.nf.calls.subscribers[1]?.config.endpoint, "http://phoenix.example");

    await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

    for (const subscriber of modules.nf.calls.subscribers) {
      assert.deepEqual(subscriber.actions, [
        `register:${subscriber.name}`,
        `deregister:${subscriber.name}`,
        "forceFlush",
        "shutdown",
      ]);
    }
  });

  it("marks subscriber registration failure degraded and keeps other outputs", async () => {
    const modules = createModules({ subscriberFailures: { otelRegister: true } });
    const api = createApi({
      pluginConfig: {
        atif: { enabled: false },
        telemetry: {
          otel: { enabled: true },
          openInference: { enabled: true },
        },
      },
    });

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });

      const status = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
      assert.equal(status.status.state, "degraded");
      assert.equal(status.outputs.otel, "degraded");
      assert.equal(status.outputs.openInference, "enabled");
      assert.deepEqual(
        modules.nf.calls.subscribers.map((subscriber) => [subscriber.kind, subscriber.actions]),
        [
          ["otel", ["shutdown"]],
          ["openInference", ["register:openclaw.nemo-flow.openinference"]],
        ],
      );
    } finally {
      await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
    }
  });

  it("marks subscriber shutdown failure degraded in runtime health", async () => {
    const modules = createModules({ subscriberFailures: { otelForceFlush: true } });
    const api = createApi({
      pluginConfig: {
        atif: { enabled: false },
        telemetry: {
          otel: { enabled: true },
        },
      },
    });
    let serviceStarted = false;

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
      serviceStarted = true;

      await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
      serviceStarted = false;

      const status = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
      assert.equal(status.status.state, "stopped");
      assert.equal(status.outputs.otel, "degraded");
    } finally {
      if (serviceStarted) {
        await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
      }
    }
  });

  it("removes beforeExit listener during normal stop", async () => {
    const modules = createModules();
    const api = createApi();
    const before = process.listenerCount("beforeExit");

    registerPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
    assert.equal(process.listenerCount("beforeExit"), before + 1);

    await service.stop?.({ stateDir: "/tmp/openclaw-state", config: {} as never, logger: api.logger });
    assert.equal(process.listenerCount("beforeExit"), before);
  });

  it("does not statically import nemo-flow-node or OpenClaw private src paths", () => {
    const files = readBuiltJavaScriptFiles(new URL("../../", import.meta.url));

    assert.doesNotMatch(files, /from ["']nemo-flow-node/);
    assert.doesNotMatch(files, /from ["']nemo-flow-node\/plugin/);
    assert.doesNotMatch(files, /openclaw\/src\//);
  });
});

function readBuiltJavaScriptFiles(directory: URL): string {
  const chunks: string[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const child = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, directory);
    if (entry.isDirectory()) {
      chunks.push(readBuiltJavaScriptFiles(child));
    } else if (entry.isFile() && entry.name.endsWith(".js")) {
      chunks.push(readFileSync(child, "utf8"));
    }
  }
  return chunks.join("\n");
}

type HookHandler = (event: unknown, ctx: unknown) => void | Promise<void>;

type TestApi = {
  id: string;
  version?: string;
  registrationMode: OpenClawPluginApi["registrationMode"];
  pluginConfig?: Record<string, unknown>;
  logger: PluginLogger;
  runtime: OpenClawPluginApi["runtime"];
  resolvePath: OpenClawPluginApi["resolvePath"];
  registerService: (service: Parameters<OpenClawPluginApi["registerService"]>[0]) => void;
  registerRuntimeLifecycle: (lifecycle: Parameters<OpenClawPluginApi["registerRuntimeLifecycle"]>[0]) => void;
  on: (hookName: string, handler: HookHandler) => void;
  registerGatewayMethod: (
    method: string,
    handler: TestGatewayMethodHandler,
    opts?: { scope?: string },
  ) => void;
  calls: {
    services: Parameters<OpenClawPluginApi["registerService"]>[0][];
    lifecycle: Parameters<OpenClawPluginApi["registerRuntimeLifecycle"]>[0][];
    gatewayMethods: Array<{
      method: string;
      handler: TestGatewayMethodHandler;
    }>;
    hooks: Array<{ hookName: string; handler: HookHandler }>;
  };
  messages: {
    info: string[];
    warn: string[];
  };
};

function createApi(params: {
  registrationMode?: OpenClawPluginApi["registrationMode"];
  pluginConfig?: Record<string, unknown>;
} = {}): TestApi {
  const messages: TestApi["messages"] = { info: [], warn: [] };
  const calls: TestApi["calls"] = {
    services: [],
    lifecycle: [],
    gatewayMethods: [],
    hooks: [],
  };
  const logger: PluginLogger = {
    info: (message) => messages.info.push(message),
    warn: (message) => messages.warn.push(message),
    error: () => {},
  };

  const api: TestApi = {
    id: "nemo-flow",
    version: "1.2.3",
    registrationMode: params.registrationMode ?? "full",
    logger,
    runtime: {
      state: {
        resolveStateDir: () => "/tmp/openclaw-state",
      },
    } as unknown as OpenClawPluginApi["runtime"],
    resolvePath: (input) => input,
    registerService: (service) => calls.services.push(service),
    registerRuntimeLifecycle: (lifecycle) => calls.lifecycle.push(lifecycle),
    on: (hookName: string, handler: HookHandler) => calls.hooks.push({ hookName, handler }),
    registerGatewayMethod: (method, handler) => calls.gatewayMethods.push({ method, handler }),
    calls,
    messages,
  };

  if (params.pluginConfig !== undefined) {
    api.pluginConfig = params.pluginConfig;
  }

  return api;
}

function registerPlugin(api: TestApi, moduleLoader?: NemoFlowModuleLoader): void {
  registerNemoFlowPlugin(api as unknown as OpenClawPluginApi, moduleLoader);
}

type TestNemoFlowRuntime = NemoFlowModules["nf"] & {
  calls: {
    event: Array<{ name: string; handle: unknown; data: unknown }>;
    subscribers: Array<{
      kind: "otel" | "openInference";
      name?: string;
      config: Record<string, unknown>;
      actions: string[];
    }>;
  };
};

type TestModules = NemoFlowModules & {
  nf: TestNemoFlowRuntime;
};

function createModules(params: {
  validateDiagnostics?: Array<{ level: "warning" | "error"; code: string; message: string }>;
  subscriberFailures?: SubscriberFailures;
} = {}): TestModules {
  const nf = createNemoFlowRuntime(params.subscriberFailures);
  return {
    nf,
    pluginHost: {
      defaultConfig: () => ({ version: 1, components: [] }),
      validate: () => ({ diagnostics: params.validateDiagnostics ?? [] }),
      initialize: async () => ({ diagnostics: [] }),
      clear: () => {},
    },
  };
}

type SubscriberFailures = {
  otelRegister?: boolean;
  openInferenceRegister?: boolean;
  otelForceFlush?: boolean;
  openInferenceForceFlush?: boolean;
  otelShutdown?: boolean;
  openInferenceShutdown?: boolean;
};

function createNemoFlowRuntime(params: SubscriberFailures = {}): TestNemoFlowRuntime {
  const calls: TestNemoFlowRuntime["calls"] = {
    event: [],
    subscribers: [],
  };
  const createSubscriber = (
    kind: "otel" | "openInference",
    failures: {
      register: boolean;
      forceFlush: boolean;
      shutdown: boolean;
    },
  ) =>
    class {
      private readonly record: TestNemoFlowRuntime["calls"]["subscribers"][number];

      constructor(config?: Record<string, unknown>) {
        this.record = { kind, config: config ?? {}, actions: [] };
        calls.subscribers.push(this.record);
      }

      register(name: string): void {
        this.record.name = name;
        if (failures.register) {
          throw new Error(`${kind} register failed`);
        }
        this.record.actions.push(`register:${name}`);
      }

      deregister(name: string): boolean {
        this.record.actions.push(`deregister:${name}`);
        return true;
      }

      forceFlush(): void {
        this.record.actions.push("forceFlush");
        if (failures.forceFlush) {
          throw new Error(`${kind} forceFlush failed`);
        }
      }

      shutdown(): void {
        this.record.actions.push("shutdown");
        if (failures.shutdown) {
          throw new Error(`${kind} shutdown failed`);
        }
      }
    };

  return {
    ScopeType: { Agent: 0 } as NemoFlowRuntimeModule["ScopeType"],
    calls,
    createScopeStack: () => ({ type: "stack" }) as unknown as ReturnType<NemoFlowRuntimeModule["createScopeStack"]>,
    currentScopeStack: () => ({ type: "previous-stack" }) as unknown as ReturnType<NemoFlowRuntimeModule["currentScopeStack"]>,
    setThreadScopeStack: () => {},
    pushScope: () => ({ type: "scope" } as unknown as ReturnType<NemoFlowRuntimeModule["pushScope"]>),
    popScope: () => {},
    event: (name, handle, data) => calls.event.push({ name, handle, data }),
    llmCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["llmCall"]>),
    llmCallEnd: () => {},
    toolCall: () => ({} as unknown as ReturnType<NemoFlowRuntimeModule["toolCall"]>),
    toolCallEnd: () => {},
    AtifExporter: FakeAtifExporter,
    OpenTelemetrySubscriber: createSubscriber("otel", {
      register: params.otelRegister ?? false,
      forceFlush: params.otelForceFlush ?? false,
      shutdown: params.otelShutdown ?? false,
    }),
    OpenInferenceSubscriber: createSubscriber("openInference", {
      register: params.openInferenceRegister ?? false,
      forceFlush: params.openInferenceForceFlush ?? false,
      shutdown: params.openInferenceShutdown ?? false,
    }),
  };
}

class FakeAtifExporter {
  register(): void {}
  deregister(): boolean {
    return true;
  }
  exportJson(): string {
    return "{}";
  }
  clear(): void {}
}
