// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import {
  NEMO_FLOW_OPENCLAW_JSON_SCHEMA,
  nemoFlowConfigSchema,
  parseConfig,
} from "../config.js";
import type { NemoFlowModules } from "../modules.js";
import type { NemoFlowHealthSnapshot } from "../health.js";
import { registerNemoFlowPlugin } from "../runtime-state.js";
import type { OpenClawHookHandlerLike, OpenClawPluginApiLike, PluginLoggerLike } from "../types.js";

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

    registerNemoFlowPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.equal(api.calls.hooks.length, 0);
  });

  it("returns without side effects when disabled", () => {
    const api = createApi({ pluginConfig: { enabled: false } });

    registerNemoFlowPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.equal(api.calls.hooks.length, 0);
    assert.deepEqual(api.messages.info, ["nemo-flow observability disabled by plugin config"]);
  });

  it("returns without side effects when config parsing fails during registration", () => {
    const api = createApi({ pluginConfig: { backend: "managed_execution" } });

    registerNemoFlowPlugin(api);

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

    registerNemoFlowPlugin(api, async () => createModules());

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
        "agent_end",
        "before_agent_finalize",
        "subagent_spawned",
        "subagent_ended",
      ],
    );
  });

  it("uses config parsed during registration when service starts", async () => {
    const api = createApi({ pluginConfig: { atif: { enabled: false }, correlation: { maxRecordsPerKey: 1 } } });

    registerNemoFlowPlugin(api, async () => createModules());
    api.pluginConfig = { backend: "managed_execution" };

    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await assert.doesNotReject(async () => {
        await service.start({
          stateDir: "/tmp/openclaw-state",
          logger: api.logger,
        });
      });
    } finally {
      await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
    }
  });

  it("continues hook-backed telemetry when plugin host validation fails", async () => {
    const modules = createModules({
      validateDiagnostics: [{ level: "error", code: "bad_config", message: "invalid" }],
    });
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });

      const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
      assert.ok(sessionStart);
      await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });

      const status = api.calls.gatewayMethods[0]?.handler();
      assert.ok(status);
      assert.deepEqual(modules.nf.calls.event.map((event) => event.name), ["openclaw.session_start"]);
      assert.equal(status.status.state, "degraded");
      assert.equal(status.initializedPluginHost, false);
    } finally {
      await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
    }
  });

  it("routes gateway_stop through runtime stop", async () => {
    const modules = createModules();
    const api = createApi({ pluginConfig: { atif: { enabled: false } } });

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });

    const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
    const gatewayStop = api.calls.hooks.find((hook) => hook.hookName === "gateway_stop");
    assert.ok(sessionStart);
    assert.ok(gatewayStop);
    await sessionStart.handler({ sessionId: "session-1" }, { sessionId: "session-1" });
    await gatewayStop.handler({ reason: "test_stop" }, {});

    const status = api.calls.gatewayMethods[0]?.handler();
    assert.ok(status);
    assert.equal(status.status.state, "stopped");
    assert.equal(status.counters.marksEmitted, 2);
    assert.deepEqual(modules.nf.calls.event.map((event) => event.name), [
      "openclaw.session_start",
      "openclaw.session_end",
    ]);
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

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });

    assert.deepEqual(
      modules.nf.calls.subscribers.map((subscriber) => [subscriber.kind, subscriber.name]),
      [
        ["otel", "openclaw.nemo-flow.otel"],
        ["openInference", "openclaw.nemo-flow.openinference"],
      ],
    );
    assert.equal(modules.nf.calls.subscribers[0]?.config.endpoint, "http://otel.example");
    assert.equal(modules.nf.calls.subscribers[1]?.config.endpoint, "http://phoenix.example");

    await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });

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

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });

      const status = api.calls.gatewayMethods[0]?.handler();
      assert.ok(status);
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
      await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
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

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    try {
      await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });
      serviceStarted = true;

      await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
      serviceStarted = false;

      const status = api.calls.gatewayMethods[0]?.handler();
      assert.ok(status);
      assert.equal(status.status.state, "stopped");
      assert.equal(status.outputs.otel, "degraded");
    } finally {
      if (serviceStarted) {
        await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
      }
    }
  });

  it("removes beforeExit listener during normal stop", async () => {
    const modules = createModules();
    const api = createApi();
    const before = process.listenerCount("beforeExit");

    registerNemoFlowPlugin(api, async () => modules);
    const service = api.calls.services[0];
    assert.ok(service);
    await service.start({ stateDir: "/tmp/openclaw-state", logger: api.logger });
    assert.equal(process.listenerCount("beforeExit"), before + 1);

    await service.stop?.({ stateDir: "/tmp/openclaw-state", logger: api.logger });
    assert.equal(process.listenerCount("beforeExit"), before);
  });

  it("does not statically import nemo-flow-node or OpenClaw private src paths", () => {
    const files = [
      readFileSync(new URL("../modules.js", import.meta.url), "utf8"),
      readFileSync(new URL("../../index.js", import.meta.url), "utf8"),
    ].join("\n");

    assert.doesNotMatch(files, /from ["']nemo-flow-node/);
    assert.doesNotMatch(files, /from ["']nemo-flow-node\/plugin/);
    assert.doesNotMatch(files, /openclaw\/src\//);
  });
});

type TestApi = OpenClawPluginApiLike & {
  calls: {
    services: Parameters<OpenClawPluginApiLike["registerService"]>[0][];
    lifecycle: Parameters<OpenClawPluginApiLike["registerRuntimeLifecycle"]>[0][];
    gatewayMethods: Array<{
      method: string;
      handler: () => NemoFlowHealthSnapshot;
    }>;
    hooks: Array<{ hookName: string; handler: OpenClawHookHandlerLike }>;
  };
  messages: {
    info: string[];
    warn: string[];
  };
};

function createApi(params: {
  registrationMode?: string;
  pluginConfig?: Record<string, unknown>;
} = {}): TestApi {
  const messages: TestApi["messages"] = { info: [], warn: [] };
  const calls: TestApi["calls"] = {
    services: [],
    lifecycle: [],
    gatewayMethods: [],
    hooks: [],
  };
  const logger: PluginLoggerLike = {
    info: (message) => messages.info.push(message),
    warn: (message) => messages.warn.push(message),
  };

  const api: TestApi = {
    id: "nemo-flow",
    version: "1.2.3",
    registrationMode: params.registrationMode ?? "full",
    logger,
    resolvePath: (input) => input,
    registerService: (service) => calls.services.push(service),
    registerRuntimeLifecycle: (lifecycle) => calls.lifecycle.push(lifecycle),
    on: (hookName, handler) => calls.hooks.push({ hookName, handler }),
    registerGatewayMethod: (method, handler) =>
      calls.gatewayMethods.push({
        method,
        handler: handler as TestApi["calls"]["gatewayMethods"][number]["handler"],
      }),
    calls,
    messages,
  };

  if (params.pluginConfig !== undefined) {
    api.pluginConfig = params.pluginConfig;
  }

  return api;
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
    ScopeType: { Agent: 0 },
    calls,
    createScopeStack: () => ({ type: "stack" }),
    currentScopeStack: () => ({ type: "previous-stack" }),
    setThreadScopeStack: () => {},
    pushScope: () => ({ type: "scope" }),
    popScope: () => {},
    event: (name, handle, data) => calls.event.push({ name, handle, data }),
    llmCall: () => ({}),
    llmCallEnd: () => {},
    toolCall: () => ({}),
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
