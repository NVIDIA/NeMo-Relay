// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OptimizerRuntime as Runtime } from './pkg/nemo_flow_wasm';
import type { JsonObject, JsonValue, LlmRequestShape } from './typed';

export type UnsupportedBehavior = 'ignore' | 'warn' | 'error';

export interface ConfigPolicy {
  unknown_component?: UnsupportedBehavior;
  unknown_field?: UnsupportedBehavior;
  unsupported_value?: UnsupportedBehavior;
}

export interface BackendSpec {
  kind: string;
  config?: JsonObject;
}

export interface StateConfig {
  backend: BackendSpec;
}

export interface ComponentSpec {
  kind: string;
  enabled?: boolean;
  config?: JsonObject;
}

export interface Config {
  version?: number;
  agent_id?: string;
  state?: StateConfig;
  components?: ComponentSpec[];
  policy?: ConfigPolicy;
}

export interface ConfigDiagnostic {
  level: 'warning' | 'error';
  code: string;
  component?: string;
  field?: string;
  message: string;
}

export interface ConfigReport {
  diagnostics: ConfigDiagnostic[];
}

export interface TelemetryComponentConfig {
  subscriber_name?: string;
  learners?: string[];
}

export interface DynamoHintsComponentConfig {
  priority?: number;
  break_chain?: boolean;
  inject_header?: boolean;
  inject_body_path?: string;
}

export interface ToolParallelismComponentConfig {
  priority?: number;
  mode?: 'observe_only' | 'inject_hints' | 'schedule' | string;
}

export interface PluginDiagnostic {
  level: 'warning' | 'error';
  code: string;
  component?: string;
  field?: string;
  message: string;
}

export interface PluginContext {
  registerSubscriber(name: string, callback: (event: JsonValue) => void): void;
  registerLlmRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (
      name: string,
      request: LlmRequestShape,
      annotated: JsonValue | null,
    ) => {
      request: LlmRequestShape;
      annotated: JsonValue | null;
    },
  ): void;
  registerLlmExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      next: (request: LlmRequestShape) => JsonValue | Promise<JsonValue>,
    ) => JsonValue | Promise<JsonValue>,
  ): void;
  registerLlmStreamExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      request: LlmRequestShape,
      next: (request: LlmRequestShape) => AsyncIterable<JsonValue> | Promise<AsyncIterable<JsonValue>>,
    ) => AsyncIterable<JsonValue> | Promise<AsyncIterable<JsonValue>>,
  ): void;
  registerToolRequestIntercept(
    name: string,
    priority: number,
    breakChain: boolean,
    callback: (name: string, args: JsonValue) => JsonValue,
  ): void;
  registerToolExecutionIntercept(
    name: string,
    priority: number,
    callback: (
      args: JsonValue,
      next: (args: JsonValue) => JsonValue | Promise<JsonValue>,
    ) => JsonValue | Promise<JsonValue>,
  ): void;
}

export interface PluginHandler {
  validate?(instanceId: string, pluginConfig: JsonObject): PluginDiagnostic[];
  register(instanceId: string, pluginConfig: JsonObject, context: PluginContext): void;
}

export declare function validateConfig(config: Config): ConfigReport;
export declare function registerPlugin(pluginKind: string, handler: PluginHandler): void;
export declare function deregisterPlugin(pluginKind: string): boolean;
export { Runtime };
export declare function defaultConfig(): Config;
export declare function inMemoryBackend(): BackendSpec;
export declare function redisBackend(url: string, keyPrefix?: string): BackendSpec;
export declare function telemetryComponent(config?: TelemetryComponentConfig): ComponentSpec;
export declare function dynamoHintsComponent(config?: DynamoHintsComponentConfig): ComponentSpec;
export declare function toolParallelismComponent(config?: ToolParallelismComponentConfig): ComponentSpec;
export declare function externalComponent(
  pluginKind: string,
  instanceId: string,
  pluginConfig?: JsonObject,
): ComponentSpec;
