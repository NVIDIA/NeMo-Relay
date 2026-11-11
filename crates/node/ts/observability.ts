// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json } from '../binding.js';
import type { ConfigPolicy } from './plugin.js';

export type { ConfigDiagnostic, ConfigReport } from '../binding.js';
export type { ConfigPolicy } from './plugin.js';

export interface AtofConfig {
  enabled?: boolean;
  sinks?: AtofSinkConfig[];
}

export type AtofSinkConfig = AtofFileSinkConfig | AtofStreamSinkConfig;

export interface AtofFileSinkConfig {
  type: 'file';
  output_directory?: string;
  filename?: string;
  mode?: 'append' | 'overwrite' | (string & {});
}

export interface AtofStreamSinkConfig {
  type: 'stream';
  url: string;
  transport?: 'http_post' | 'websocket' | 'ndjson' | (string & {});
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  header_file?: Record<string, string>;
  timeout_millis?: number;
  field_name_policy?: 'preserve' | 'replace_dots' | (string & {});
  name?: string;
}

/** @deprecated Use AtofStreamSinkConfig. */
export type AtofEndpointConfig = AtofStreamSinkConfig;

export interface S3StorageConfig {
  type: 's3';
  bucket: string;
  key_prefix?: string;
  access_key_id?: string;
  secret_access_key_var?: string;
  session_token_var?: string;
  region?: string;
  endpoint_url?: string;
  allow_http?: boolean;
}

export interface HttpStorageConfig {
  type: 'http';
  endpoint: string;
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  header_file?: Record<string, string>;
  timeout_millis?: number;
}

export interface AtifConfig {
  enabled?: boolean;
  agent_name?: string;
  agent_version?: string;
  model_name?: string;
  tool_definitions?: Record<string, Json>[];
  extra?: Record<string, Json>;
  output_directory?: string;
  filename_template?: string;
  storage?: S3StorageConfig | HttpStorageConfig | Array<S3StorageConfig | HttpStorageConfig>;
  /** Keeps raw events in `extra.observed_events`, default `true`. */
  include_observed_events?: boolean;
  /** Keeps full provider requests in trajectory steps, default `true`. */
  include_llm_requests?: boolean;
}

export interface OpenTelemetryTlsConfig {
  /** PEM CA bundle replacing system CAs. */
  certificate?: string;
  /** PEM client certificate chain, used with `private_key`. */
  client_certificate?: string;
  /** PEM private key, used with `client_certificate`. */
  private_key?: string;
}

export interface OpenTelemetryEndpointConfig {
  type: 'full' | 'gen_ai' | 'openinference';
  endpoint: string;
  tls?: OpenTelemetryTlsConfig;
  mark_projection?: 'inherit' | 'event' | 'tool';
  mark_exclude_names?: string[];
  attribute_mappings?: Array<{ key: string; alias: string }>;
  promote_metadata_prefixes?: string[];
  promote_resource_metadata_prefixes?: string[];
  transport?: 'http_binary' | 'grpc';
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  header_file?: Record<string, string>;
  resource_attributes?: Record<string, string>;
  service_name?: string;
  service_namespace?: string;
  service_version?: string;
  instrumentation_scope?: string;
  timeout_millis?: number;
  max_queue_size?: number;
  max_export_batch_size?: number;
  scheduled_delay_millis?: number;
  completed_span_context_ttl_millis?: number;
}

export interface OpenTelemetrySignalEndpointConfig {
  endpoint: string;
  tls?: OpenTelemetryTlsConfig;
  transport?: 'http_binary' | 'grpc';
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  header_file?: Record<string, string>;
  resource_attributes?: Record<string, string>;
  service_name?: string;
  service_namespace?: string;
  service_version?: string;
  instrumentation_scope?: string;
  timeout_millis?: number;
  /** Root-scope metadata prefixes copied to resource attributes. */
  promote_resource_metadata_prefixes?: string[];
}

export interface OpenTelemetryLogSectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetrySignalEndpointConfig[];
  minimum_severity?: 'trace' | 'debug' | 'info' | 'warn' | 'warning' | 'error';
  max_queue_size?: number;
  max_export_batch_size?: number;
  scheduled_delay_millis?: number;
  completed_span_context_ttl_millis?: number;
}

export interface OpenTelemetryMetricSectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetrySignalEndpointConfig[];
  export_interval_millis?: number;
  temporality?: 'cumulative' | 'delta' | 'low_memory';
  max_instruments?: number;
  cardinality_limit?: number;
}

export interface OpenTelemetryFileSinkConfig {
  type?: 'full' | 'gen_ai' | 'openinference';
  output_directory: string;
  filename?: string;
  /** `json_lines` for one OTLP/JSON record per line, `proto` for length-delimited records. */
  format?: 'json_lines' | 'proto';
  mode?: 'append' | 'overwrite';
  mark_projection?: 'inherit' | 'event' | 'tool';
  mark_exclude_names?: string[];
  attribute_mappings?: Array<{ key: string; alias: string }>;
  promote_metadata_prefixes?: string[];
  promote_resource_metadata_prefixes?: string[];
  resource_attributes?: Record<string, string>;
  service_name?: string;
  service_namespace?: string;
  service_version?: string;
  instrumentation_scope?: string;
  max_queue_size?: number;
  max_export_batch_size?: number;
  scheduled_delay_millis?: number;
  completed_span_context_ttl_millis?: number;
}

export interface OpenTelemetrySectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetryEndpointConfig[];
  file_sinks?: OpenTelemetryFileSinkConfig[];
  logs?: OpenTelemetryLogSectionConfig;
  metrics?: OpenTelemetryMetricSectionConfig;
}

export interface Config {
  version?: number;
  atof?: AtofConfig;
  atif?: AtifConfig;
  opentelemetry?: OpenTelemetrySectionConfig;
  policy?: ConfigPolicy;
  /** Keeps full sanitized request data on LLM start events. */
  enable_full_payloads?: boolean;
  /** Non-loopback hosts allowed to receive unencrypted HTTP or WebSocket exports, headers included. */
  insecure_http_hosts?: string[];
}

export interface ComponentSpec {
  kind: 'observability';
  enabled?: boolean;
  config: Config;
}

/** Plugin kind of the observability component. */
export const OBSERVABILITY_PLUGIN_KIND = 'observability';

const DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS = 60_000;

/** Creates a default observability config. */
export function defaultConfig(): Config {
  return {
    version: 4,
  };
}

/** Creates Agent Trajectory Observability Format (ATOF) settings with defaults applied. */
export function atofConfig(config: AtofConfig = {}): AtofConfig {
  return {
    enabled: false,
    ...config,
  };
}

/** Creates Agent Trajectory Interchange Format (ATIF) settings with defaults applied. */
export function atifConfig(config: AtifConfig = {}): AtifConfig {
  return {
    enabled: false,
    agent_name: 'NeMo Relay',
    model_name: 'unknown',
    filename_template: 'nemo-relay-atif-{session_id}.json',
    include_observed_events: true,
    include_llm_requests: true,
    ...config,
  };
}

/** Creates one OpenTelemetry trace endpoint with defaults applied. */
export function openTelemetryEndpoint(config: OpenTelemetryEndpointConfig): OpenTelemetryEndpointConfig {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('OpenTelemetry endpoint config is required');
  }
  if (!['full', 'gen_ai', 'openinference'].includes(config.type)) {
    throw new TypeError('OpenTelemetry endpoint type must be "full", "gen_ai", or "openinference"');
  }
  if (typeof config.endpoint !== 'string' || config.endpoint.trim() === '') {
    throw new TypeError('OpenTelemetry endpoint must be a nonblank string');
  }
  return {
    transport: 'http_binary',
    service_name: 'unknown_service',
    instrumentation_scope: 'opentelemetry',
    timeout_millis: 3000,
    completed_span_context_ttl_millis: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS,
    headers: {},
    header_env: {},
    resource_attributes: {},
    promote_metadata_prefixes: [],
    promote_resource_metadata_prefixes: [],
    ...config,
  };
}

/** Creates one file sink for OTLP spans with defaults applied. */
export function openTelemetryFileSink(config: OpenTelemetryFileSinkConfig): OpenTelemetryFileSinkConfig {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('OpenTelemetry file sink config is required');
  }
  if (typeof config.output_directory !== 'string' || config.output_directory.trim() === '') {
    throw new TypeError('OpenTelemetry file sink output_directory must be a nonblank string');
  }
  if (config.format !== undefined && !['json_lines', 'proto'].includes(config.format)) {
    throw new TypeError('OpenTelemetry file sink format must be "json_lines" or "proto"');
  }
  if (config.mode !== undefined && !['append', 'overwrite'].includes(config.mode)) {
    throw new TypeError('OpenTelemetry file sink mode must be "append" or "overwrite"');
  }
  return {
    type: 'full',
    format: 'json_lines',
    mode: 'overwrite',
    service_name: 'unknown_service',
    instrumentation_scope: 'opentelemetry',
    completed_span_context_ttl_millis: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS,
    resource_attributes: {},
    promote_metadata_prefixes: [],
    promote_resource_metadata_prefixes: [],
    ...config,
  };
}

/** Creates one OpenTelemetry log or metric endpoint with defaults applied. */
export function openTelemetrySignalEndpoint(
  config: OpenTelemetrySignalEndpointConfig,
): OpenTelemetrySignalEndpointConfig {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('OpenTelemetry signal endpoint config is required');
  }
  if (typeof config.endpoint !== 'string' || config.endpoint.trim() === '') {
    throw new TypeError('OpenTelemetry signal endpoint must be a nonblank string');
  }
  return {
    transport: 'http_binary',
    headers: {},
    header_env: {},
    resource_attributes: {},
    service_name: 'unknown_service',
    instrumentation_scope: 'opentelemetry',
    timeout_millis: 3000,
    ...config,
  };
}

/** Creates OTLP log settings with defaults applied. */
export function openTelemetryLogConfig(config: OpenTelemetryLogSectionConfig = {}): OpenTelemetryLogSectionConfig {
  return {
    enabled: false,
    minimum_severity: 'info',
    max_queue_size: 2048,
    max_export_batch_size: 512,
    scheduled_delay_millis: 1000,
    completed_span_context_ttl_millis: DEFAULT_COMPLETED_SPAN_CONTEXT_TTL_MILLIS,
    ...config,
  };
}

/** Creates OTLP metric settings with defaults applied. */
export function openTelemetryMetricConfig(
  config: OpenTelemetryMetricSectionConfig = {},
): OpenTelemetryMetricSectionConfig {
  return {
    enabled: false,
    export_interval_millis: 60000,
    temporality: 'cumulative',
    max_instruments: 256,
    cardinality_limit: 2000,
    ...config,
  };
}

/** Creates OpenTelemetry settings with defaults applied. */
export function openTelemetryConfig(config: OpenTelemetrySectionConfig = {}): OpenTelemetrySectionConfig {
  return {
    enabled: false,
    endpoints: [],
    file_sinks: [],
    ...config,
  };
}

/** Wraps observability config as a plugin component. */
export function ComponentSpec(config: Config, { enabled = true }: { enabled?: boolean } = {}): ComponentSpec {
  return {
    kind: OBSERVABILITY_PLUGIN_KIND,
    enabled,
    config,
  };
}
