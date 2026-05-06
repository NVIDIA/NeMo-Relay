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
import { registerNemoFlowPlugin } from "../runtime-state.js";
import type { OpenClawPluginApiLike, PluginLoggerLike } from "../types.js";

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
  });

  it("returns without side effects when disabled", () => {
    const api = createApi({ pluginConfig: { enabled: false } });

    registerNemoFlowPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
    assert.deepEqual(api.messages.info, ["nemo-flow observability disabled by plugin config"]);
  });

  it("returns without side effects when config parsing fails during registration", () => {
    const api = createApi({ pluginConfig: { backend: "managed_execution" } });

    registerNemoFlowPlugin(api);

    assert.equal(api.calls.services.length, 0);
    assert.equal(api.calls.lifecycle.length, 0);
    assert.equal(api.calls.gatewayMethods.length, 0);
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
    gatewayMethods: Array<{ method: string }>;
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
    registerGatewayMethod: (method) => calls.gatewayMethods.push({ method }),
    calls,
    messages,
  };

  if (params.pluginConfig !== undefined) {
    api.pluginConfig = params.pluginConfig;
  }

  return api;
}

function createModules(): NemoFlowModules {
  return {
    nf: {},
    pluginHost: {
      defaultConfig: () => ({ version: 1, components: [] }),
      validate: () => ({ diagnostics: [] }),
      initialize: async () => ({ diagnostics: [] }),
      clear: () => {},
    },
  };
}
