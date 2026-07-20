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

export interface OtlpConfig {
  enabled?: boolean;
  mark_projection?: 'inherit' | 'event' | 'tool';
  mark_exclude_names?: string[];
  attribute_mappings?: OtlpAttributeMapping[];
  /** OpenTelemetry-only semantic projection. OpenInference supports `generic`. */
  semantic_convention?: 'generic' | 'gen_ai';
  /** OpenTelemetry-only capture of sanitized GenAI content. */
  capture_content?: boolean;
  transport?: 'http_binary' | 'grpc' | string;
  endpoint?: string;
  headers?: Record<string, string>;
  resource_attributes?: Record<string, string>;
  service_name?: string;
  service_namespace?: string;
  service_version?: string;
  instrumentation_scope?: string;
  timeout_millis?: number;
}

/** Copy a projected OTLP attribute to an additional attribute name. */
export interface OtlpAttributeMapping {
  key: string;
  alias: string;
}

export interface Config {
  version?: number;
  atof?: AtofConfig;
  atif?: AtifConfig;
  opentelemetry?: OtlpConfig;
  openinference?: OtlpConfig;
  policy?: ConfigPolicy;
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
/** Create OTLP exporter settings for OpenTelemetry or OpenInference. */
export declare function otlpConfig(config?: OtlpConfig): OtlpConfig;
/** Wrap observability config as a top-level plugin component. */
export declare function ComponentSpec(
  config: Config,
  options?: {
    enabled?: boolean;
  },
): ComponentSpec;
