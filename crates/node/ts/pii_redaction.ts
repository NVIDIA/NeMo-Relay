// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type { ConfigDiagnostic, ConfigReport } from '../binding.js';

export interface ConfigPolicy {
  unknown_component?: 'ignore' | 'warn' | 'error' | (string & {});
  unknown_field?: 'ignore' | 'warn' | 'error' | (string & {});
  unsupported_value?: 'ignore' | 'warn' | 'error' | (string & {});
}

export interface BuiltinConfig {
  preset?: 'trajectory_context';
  action?: 'remove' | 'redact' | 'regex_replace' | 'hash' | 'mask' | (string & {});
  target_paths?: string[];
  target_path_globs?: string[];
  pattern?: string;
  detector?: string;
  replacement?: string;
  mask_char?: string;
  unmasked_prefix?: number;
  unmasked_suffix?: number;
  custom_mark_payload_policy?: 'preserve' | 'redact_all_leaves';
  metric_string_attribute_allowlist?: Record<string, string[]>;
}

export interface LocalModelConfig {
  backend?: string;
  model_id?: string;
  detector_profile?: string;
  allow_network?: boolean;
  max_latency_ms?: number;
}

export interface Config {
  version?: number;
  mode?: 'builtin' | 'local_model' | (string & {});
  input?: boolean;
  output?: boolean;
  tool_input?: boolean;
  tool_output?: boolean;
  mark?: boolean;
  priority?: number;
  codec?:
    | 'openai_chat'
    | 'openai_responses'
    | 'anthropic_messages'
    | 'oci_genai'
    | 'gemini_generate_content'
    | (string & {});
  builtin?: BuiltinConfig;
  local?: LocalModelConfig;
  policy?: ConfigPolicy;
}

export interface ComponentSpec {
  kind: 'pii_redaction';
  enabled?: boolean;
  config: Config;
}

/** Plugin kind of the PII redaction component. */
export const PII_REDACTION_PLUGIN_KIND = 'pii_redaction';

/** Creates a default PII redaction config. */
export function defaultConfig(): Config {
  return {
    version: 1,
    mode: 'builtin',
    input: true,
    output: true,
    tool_input: true,
    tool_output: true,
    mark: true,
    priority: 100,
  };
}

/** Creates built-in backend settings with defaults applied. */
export function builtinConfig(config: BuiltinConfig = {}): BuiltinConfig {
  return {
    ...(config.preset === 'trajectory_context' ? {} : { action: 'remove' }),
    ...config,
  };
}

/** Creates local-model backend settings with defaults applied. */
export function localModelConfig(config: LocalModelConfig = {}): LocalModelConfig {
  return {
    ...config,
  };
}

/** Wraps PII redaction config as a plugin component. */
export function ComponentSpec(config: Config, { enabled = true }: { enabled?: boolean } = {}): ComponentSpec {
  return {
    kind: PII_REDACTION_PLUGIN_KIND,
    enabled,
    config,
  };
}
