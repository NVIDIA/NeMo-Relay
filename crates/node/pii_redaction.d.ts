// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ConfigDiagnostic, ConfigReport } from './plugin.js';

export { ConfigDiagnostic, ConfigReport };

export interface ConfigPolicy {
  unknown_component?: 'ignore' | 'warn' | 'error' | string;
  unknown_field?: 'ignore' | 'warn' | 'error' | string;
  unsupported_value?: 'ignore' | 'warn' | 'error' | string;
}

export interface BuiltinConfig {
  action?: 'remove' | 'redact' | 'regex_replace' | 'hash' | 'mask' | string;
  target_paths?: string[];
  pattern?: string;
  detector?: string;
  replacement?: string;
  mask_char?: string;
  unmasked_prefix?: number;
  unmasked_suffix?: number;
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
  mode?: 'builtin' | 'local_model' | string;
  input?: boolean;
  output?: boolean;
  tool_input?: boolean;
  tool_output?: boolean;
  mark?: boolean;
  priority?: number;
  codec?: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | string;
  builtin?: BuiltinConfig;
  local?: LocalModelConfig;
  policy?: ConfigPolicy;
}

export interface ComponentSpec {
  kind: 'pii_redaction';
  enabled?: boolean;
  config: Config;
}

/** Top-level plugin kind used by the built-in PII redaction component. */
export declare const PII_REDACTION_PLUGIN_KIND: 'pii_redaction';
/** Create a default PII redaction component config. */
export declare function defaultConfig(): Config;
/** Create deterministic built-in redaction backend settings with defaults applied. */
export declare function builtinConfig(config?: BuiltinConfig): BuiltinConfig;
/** Create future local-model backend settings with defaults applied. */
export declare function localModelConfig(config?: LocalModelConfig): LocalModelConfig;
/** Wrap PII redaction config as a top-level plugin component. */
export declare function ComponentSpec(
  config: Config,
  options?: {
    enabled?: boolean;
  },
): import('./plugin.js').ComponentSpec;
