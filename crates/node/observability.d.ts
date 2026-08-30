// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json } from './index';
import type { ConfigPolicy, ConfigDiagnostic, ConfigReport } from './plugin';

export { ConfigPolicy, ConfigDiagnostic, ConfigReport };

export interface AtofConfig {
  enabled?: boolean;
  sinks?: AtofSinkConfig[];
}

export type AtofSinkConfig = AtofFileSinkConfig | AtofStreamSinkConfig;

export interface AtofFileSinkConfig {
  type: 'file';
  output_directory?: string;
  filename?: string;
  mode?: 'append' | 'overwrite' | string;
}

export interface AtofStreamSinkConfig {
  type: 'stream';
  url: string;
  transport?: 'http_post' | 'websocket' | 'ndjson' | string;
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  timeout_millis?: number;
  field_name_policy?: 'preserve' | 'replace_dots' | string;
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
}

export interface OpenTelemetryEndpointConfig {
  type: 'full' | 'gen_ai' | 'openinference';
  endpoint: string;
  mark_projection?: 'inherit' | 'event' | 'tool';
  mark_exclude_names?: string[];
  attribute_mappings?: Array<{ key: string; alias: string }>;
  promote_metadata_prefixes?: string[];
  promote_resource_metadata_prefixes?: string[];
  transport?: 'http_binary' | 'grpc';
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
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
  transport?: 'http_binary' | 'grpc';
  headers?: Record<string, string>;
  header_env?: Record<string, string>;
  resource_attributes?: Record<string, string>;
  service_name?: string;
  service_namespace?: string;
  service_version?: string;
  instrumentation_scope?: string;
  timeout_millis?: number;
}

export interface OpenTelemetryLogSectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetrySignalEndpointConfig[];
  minimum_severity?: 'trace' | 'debug' | 'info' | 'warn' | 'warning' | 'error';
  max_queue_size?: number;
  max_export_batch_size?: number;
  scheduled_delay_millis?: number;
}

export interface OpenTelemetryMetricSectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetrySignalEndpointConfig[];
  export_interval_millis?: number;
  temporality?: 'cumulative' | 'delta' | 'low_memory';
  max_instruments?: number;
  cardinality_limit?: number;
}

export interface OpenTelemetrySectionConfig {
  enabled?: boolean;
  endpoints?: OpenTelemetryEndpointConfig[];
  logs?: OpenTelemetryLogSectionConfig;
  metrics?: OpenTelemetryMetricSectionConfig;
}

export interface Config {
  version?: number;
  atof?: AtofConfig;
  atif?: AtifConfig;
  opentelemetry?: OpenTelemetrySectionConfig;
  policy?: ConfigPolicy;
  /** Retain complete sanitized request data on every LLM start event. */
  enable_full_payloads?: boolean;
}

export interface ComponentSpec {
  kind: 'observability';
  enabled?: boolean;
  config: Config;
}

/** Top-level plugin kind used by the built-in observability component. */
export declare const OBSERVABILITY_PLUGIN_KIND: 'observability';
/** Create a default observability component config. */
export declare function defaultConfig(): Config;
/** Create filesystem-backed Agent Trajectory Observability Format (ATOF) JSONL settings with defaults applied. */
export declare function atofConfig(config?: AtofConfig): AtofConfig;
/** Create per-agent Agent Trajectory Interchange Format (ATIF) trajectory settings with defaults applied. */
export declare function atifConfig(config?: AtifConfig): AtifConfig;
/** Create one typed OpenTelemetry endpoint. */
export declare function openTelemetryEndpoint(config: OpenTelemetryEndpointConfig): OpenTelemetryEndpointConfig;
/** Create one signal-specific OpenTelemetry endpoint for logs or metrics. */
export declare function openTelemetrySignalEndpoint(
  config: OpenTelemetrySignalEndpointConfig,
): OpenTelemetrySignalEndpointConfig;
/** Create OTLP log pipeline settings with defaults applied. */
export declare function openTelemetryLogConfig(config?: OpenTelemetryLogSectionConfig): OpenTelemetryLogSectionConfig;
/** Create OTLP metric pipeline settings with defaults applied. */
export declare function openTelemetryMetricConfig(
  config?: OpenTelemetryMetricSectionConfig,
): OpenTelemetryMetricSectionConfig;
/** Create multi-endpoint OpenTelemetry settings. */
export declare function openTelemetryConfig(config?: OpenTelemetrySectionConfig): OpenTelemetrySectionConfig;
/** Wrap observability config as a top-level plugin component. */
export declare function ComponentSpec(
  config: Config,
  options?: {
    enabled?: boolean;
  },
): ComponentSpec;
