// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OpenClawPluginConfigSchema } from "openclaw/plugin-sdk/plugin-entry";

import manifest from "../openclaw.plugin.json" with { type: "json" };

export type BackendKind = "hooks";

export type TelemetrySinkConfig = {
  enabled: boolean;
  transport?: string;
  endpoint?: string;
  headers?: Record<string, string>;
  resourceAttributes?: Record<string, string>;
  serviceName: string;
  serviceNamespace: string;
  serviceVersion?: string;
  instrumentationScope?: string;
  timeoutMillis: number;
};

export type CaptureConfig = {
  includePrompts: boolean;
  includeResponses: boolean;
  stripToolArgs: boolean;
  stripToolResults: boolean;
};

export type CorrelationConfig = {
  llmOutputGraceMs: number;
  recordTtlMs: number;
  maxRecordsPerKey: number;
};

export type AtifConfig = {
  enabled: boolean;
  outputDir?: string;
  agentName: string;
  agentVersion?: string;
};

export type NemoFlowPluginHostConfig = {
  version: number;
  components: unknown[];
  [key: string]: unknown;
};

export type NemoFlowHookBackendConfig = {
  enabled: boolean;
  backend: BackendKind;
  nemoFlow: {
    pluginConfig: NemoFlowPluginHostConfig;
  };
  atif: AtifConfig;
  telemetry: {
    openInference: TelemetrySinkConfig;
    otel: TelemetrySinkConfig;
  };
  capture: CaptureConfig;
  correlation: CorrelationConfig;
};

const DEFAULT_PLUGIN_HOST_CONFIG: NemoFlowPluginHostConfig = {
  version: 1,
  components: [],
};

export const NEMO_FLOW_OPENCLAW_JSON_SCHEMA = manifest.configSchema;

export const DEFAULT_CONFIG: NemoFlowHookBackendConfig = {
  enabled: true,
  backend: "hooks",
  nemoFlow: {
    pluginConfig: DEFAULT_PLUGIN_HOST_CONFIG,
  },
  atif: {
    enabled: true,
    agentName: "openclaw",
  },
  telemetry: {
    openInference: defaultTelemetrySinkConfig("nemo-flow-openinference"),
    otel: defaultTelemetrySinkConfig("nemo-flow-otel"),
  },
  capture: {
    includePrompts: true,
    includeResponses: true,
    stripToolArgs: true,
    stripToolResults: true,
  },
  correlation: {
    llmOutputGraceMs: 250,
    recordTtlMs: 600_000,
    maxRecordsPerKey: 32,
  },
};

export const nemoFlowConfigSchema = {
  safeParse(value: unknown) {
    try {
      return { success: true, data: parseConfig(value) };
    } catch (error) {
      return {
        success: false,
        error: {
          issues: [
            {
              path: [],
              message: error instanceof Error ? error.message : String(error),
            },
          ],
        },
      };
    }
  },
  jsonSchema: NEMO_FLOW_OPENCLAW_JSON_SCHEMA,
} satisfies OpenClawPluginConfigSchema;

export function parseConfig(value: unknown): NemoFlowHookBackendConfig {
  const raw = asRecord(value, "config", true);
  const backend = optionalString(raw.backend, "backend") ?? DEFAULT_CONFIG.backend;

  if (backend !== "hooks") {
    throw new Error(`unsupported nemo-flow backend: ${backend}`);
  }

  const atif = asRecord(raw.atif, "atif", true);
  const telemetry = asRecord(raw.telemetry, "telemetry", true);
  const otel = asRecord(telemetry.otel, "telemetry.otel", true);
  const openInference = asRecord(telemetry.openInference, "telemetry.openInference", true);
  const capture = asRecord(raw.capture, "capture", true);
  const correlation = asRecord(raw.correlation, "correlation", true);
  const nemoFlow = asRecord(raw.nemoFlow, "nemoFlow", true);

  return {
    enabled: optionalBoolean(raw.enabled, "enabled") ?? DEFAULT_CONFIG.enabled,
    backend,
    nemoFlow: {
      pluginConfig: parsePluginHostConfig(nemoFlow.pluginConfig),
    },
    atif: {
      enabled: optionalBoolean(atif.enabled, "atif.enabled") ?? DEFAULT_CONFIG.atif.enabled,
      agentName: optionalString(atif.agentName, "atif.agentName") ?? DEFAULT_CONFIG.atif.agentName,
      ...definedStringProperty("outputDir", optionalString(atif.outputDir, "atif.outputDir")),
      ...definedStringProperty(
        "agentVersion",
        optionalString(atif.agentVersion, "atif.agentVersion"),
      ),
    },
    telemetry: {
      openInference: parseTelemetrySinkConfig(
        openInference,
        DEFAULT_CONFIG.telemetry.openInference,
        "telemetry.openInference",
      ),
      otel: parseTelemetrySinkConfig(otel, DEFAULT_CONFIG.telemetry.otel, "telemetry.otel"),
    },
    capture: {
      includePrompts:
        optionalBoolean(capture.includePrompts, "capture.includePrompts") ??
        DEFAULT_CONFIG.capture.includePrompts,
      includeResponses:
        optionalBoolean(capture.includeResponses, "capture.includeResponses") ??
        DEFAULT_CONFIG.capture.includeResponses,
      stripToolArgs:
        optionalBoolean(capture.stripToolArgs, "capture.stripToolArgs") ??
        DEFAULT_CONFIG.capture.stripToolArgs,
      stripToolResults:
        optionalBoolean(capture.stripToolResults, "capture.stripToolResults") ??
        DEFAULT_CONFIG.capture.stripToolResults,
    },
    correlation: {
      llmOutputGraceMs:
        optionalNonNegativeInteger(correlation.llmOutputGraceMs, "correlation.llmOutputGraceMs") ??
        DEFAULT_CONFIG.correlation.llmOutputGraceMs,
      recordTtlMs:
        optionalNonNegativeInteger(correlation.recordTtlMs, "correlation.recordTtlMs") ??
        DEFAULT_CONFIG.correlation.recordTtlMs,
      maxRecordsPerKey:
        optionalPositiveInteger(correlation.maxRecordsPerKey, "correlation.maxRecordsPerKey") ??
        DEFAULT_CONFIG.correlation.maxRecordsPerKey,
    },
  };
}

function parsePluginHostConfig(value: unknown): NemoFlowPluginHostConfig {
  if (value === undefined) {
    return clonePluginHostConfig(DEFAULT_PLUGIN_HOST_CONFIG);
  }
  const record = asRecord(value, "nemoFlow.pluginConfig", false);
  const version = optionalNumber(record.version, "nemoFlow.pluginConfig.version") ?? 1;
  const components = record.components === undefined ? [] : record.components;

  if (!Array.isArray(components)) {
    throw new Error("nemoFlow.pluginConfig.components must be an array");
  }

  return {
    ...record,
    version,
    components: [...components],
  };
}

function parseTelemetrySinkConfig(
  raw: Record<string, unknown>,
  defaults: TelemetrySinkConfig,
  path: string,
): TelemetrySinkConfig {
  return {
    enabled: optionalBoolean(raw.enabled, `${path}.enabled`) ?? defaults.enabled,
    serviceName: optionalString(raw.serviceName, `${path}.serviceName`) ?? defaults.serviceName,
    serviceNamespace:
      optionalString(raw.serviceNamespace, `${path}.serviceNamespace`) ?? defaults.serviceNamespace,
    timeoutMillis:
      optionalNonNegativeInteger(raw.timeoutMillis, `${path}.timeoutMillis`) ?? defaults.timeoutMillis,
    ...definedStringProperty(
      "transport",
      optionalString(raw.transport, `${path}.transport`) ?? defaults.transport,
    ),
    ...definedStringProperty(
      "serviceVersion",
      optionalString(raw.serviceVersion, `${path}.serviceVersion`) ?? defaults.serviceVersion,
    ),
    ...definedStringProperty(
      "instrumentationScope",
      optionalString(raw.instrumentationScope, `${path}.instrumentationScope`) ??
        defaults.instrumentationScope,
    ),
    ...definedStringProperty("endpoint", optionalString(raw.endpoint, `${path}.endpoint`)),
    ...definedRecordProperty("headers", optionalStringRecord(raw.headers, `${path}.headers`)),
    ...definedRecordProperty(
      "resourceAttributes",
      optionalStringRecord(raw.resourceAttributes, `${path}.resourceAttributes`),
    ),
  };
}

function defaultTelemetrySinkConfig(instrumentationScope: string): TelemetrySinkConfig {
  return {
    enabled: false,
    transport: "http_binary",
    serviceName: "openclaw-nemo-flow",
    serviceNamespace: "nemo-flow",
    serviceVersion: "unknown",
    instrumentationScope,
    timeoutMillis: 3000,
  };
}

function clonePluginHostConfig(config: NemoFlowPluginHostConfig): NemoFlowPluginHostConfig {
  return {
    ...config,
    components: [...config.components],
  };
}

function asRecord(value: unknown, path: string, optional: boolean): Record<string, unknown> {
  if (value === undefined && optional) {
    return {};
  }
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  throw new Error(`${path} must be an object`);
}

function optionalBoolean(value: unknown, path: string): boolean | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error(`${path} must be a boolean`);
  }
  return value;
}

function optionalNumber(value: unknown, path: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} must be a finite number`);
  }
  return value;
}

function optionalNonNegativeInteger(value: unknown, path: string): number | undefined {
  const parsed = optionalNumber(value, path);
  if (parsed === undefined) {
    return undefined;
  }
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${path} must be a non-negative integer`);
  }
  return parsed;
}

function optionalPositiveInteger(value: unknown, path: string): number | undefined {
  const parsed = optionalNumber(value, path);
  if (parsed === undefined) {
    return undefined;
  }
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${path} must be a positive integer`);
  }
  return parsed;
}

function optionalString(value: unknown, path: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`${path} must be a string`);
  }
  return value;
}

function optionalStringRecord(
  value: unknown,
  path: string,
): Record<string, string> | undefined {
  if (value === undefined) {
    return undefined;
  }
  const record = asRecord(value, path, false);
  const out: Record<string, string> = {};

  for (const [key, item] of Object.entries(record)) {
    if (typeof item !== "string") {
      throw new Error(`${path}.${key} must be a string`);
    }
    out[key] = item;
  }

  return out;
}

function definedStringProperty<K extends string>(
  key: K,
  value: string | undefined,
): Partial<Record<K, string>> {
  return value === undefined ? {} : { [key]: value } as Record<K, string>;
}

function definedRecordProperty<K extends string>(
  key: K,
  value: Record<string, string> | undefined,
): Partial<Record<K, Record<string, string>>> {
  return value === undefined ? {} : { [key]: value } as Record<K, Record<string, string>>;
}
