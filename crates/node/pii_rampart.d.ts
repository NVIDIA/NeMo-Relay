// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ConfigPolicy {
  unknown_component?: 'ignore' | 'warn' | 'error' | string;
  unknown_field?: 'ignore' | 'warn' | 'error' | string;
  unsupported_value?: 'ignore' | 'warn' | 'error' | string;
}

export interface Config {
  version?: number;
  model_path: string;
  input?: boolean;
  output?: boolean;
  mark?: boolean;
  tool_input?: boolean;
  tool_output?: boolean;
  priority?: number;
  codec?: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | string;
  target_paths?: string[];
  target_path_patterns?: string[];
  min_score?: number;
  excluded_labels?: string[];
  replacement?: string;
  max_windows_per_payload?: number;
  inference_batch_size?: number;
  policy?: ConfigPolicy;
}

type NonEmptyStringArray = [string, ...string[]];

export type ConfigWithSelectors = Omit<
  Partial<Config>,
  'model_path' | 'target_paths' | 'target_path_patterns'
> &
  (
    | { target_paths: NonEmptyStringArray; target_path_patterns?: string[] }
    | { target_paths?: string[]; target_path_patterns: NonEmptyStringArray }
  );

export declare const RAMPART_PII_PLUGIN_KIND: 'pii_rampart';
export declare const RAMPART_MODEL_ID: 'nationaldesignstudio/rampart';
export declare const RAMPART_MODEL_REVISION: 'b1993e4e68b082835b80ffc65acc03325ea2e501';
export declare function defaultConfig(modelPath: string, config: ConfigWithSelectors): Config;
export declare function ComponentSpec(
  config: Config,
  options?: { enabled?: boolean },
): import('./plugin.js').ComponentSpec;
export declare function validateConfig(config: Config): import('./plugin.js').ConfigReport;
