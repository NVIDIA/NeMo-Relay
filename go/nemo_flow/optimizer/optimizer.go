// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package optimizer

import nemo_flow "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"

type UnsupportedBehavior = nemo_flow.UnsupportedBehavior

const (
	UnsupportedBehaviorIgnore = nemo_flow.UnsupportedBehaviorIgnore
	UnsupportedBehaviorWarn   = nemo_flow.UnsupportedBehaviorWarn
	UnsupportedBehaviorError  = nemo_flow.UnsupportedBehaviorError
)

type DiagnosticLevel = nemo_flow.OptimizerDiagnosticLevel

const (
	DiagnosticLevelWarning = nemo_flow.OptimizerDiagnosticLevelWarning
	DiagnosticLevelError   = nemo_flow.OptimizerDiagnosticLevelError
)

type Config = nemo_flow.OptimizerConfig
type StateConfig = nemo_flow.OptimizerStateConfig
type BackendSpec = nemo_flow.OptimizerBackendSpec
type ComponentSpec = nemo_flow.OptimizerComponentSpec
type ConfigPolicy = nemo_flow.OptimizerConfigPolicy
type ConfigReport = nemo_flow.OptimizerConfigReport
type ConfigDiagnostic = nemo_flow.OptimizerConfigDiagnostic

type TelemetryComponentConfig = nemo_flow.TelemetryComponentConfig
type DynamoHintsComponentConfig = nemo_flow.DynamoHintsComponentConfig
type ToolParallelismComponentConfig = nemo_flow.ToolParallelismComponentConfig
type ExternalComponentConfig = nemo_flow.ExternalComponentConfig

type Runtime = nemo_flow.OptimizerRuntime
type PluginContext = nemo_flow.OptimizerPluginContext
type PluginHandler = nemo_flow.OptimizerPluginHandler
type PluginHandlerFuncs = nemo_flow.OptimizerPluginHandlerFuncs

func NewConfig() Config {
	return nemo_flow.NewOptimizerConfig()
}

func NewInMemoryBackend() BackendSpec {
	return nemo_flow.NewInMemoryOptimizerBackend()
}

func NewRedisBackend(url, keyPrefix string) BackendSpec {
	return nemo_flow.NewRedisOptimizerBackend(url, keyPrefix)
}

func NewTelemetryComponentConfig() TelemetryComponentConfig {
	return nemo_flow.NewTelemetryComponentConfig()
}

func NewDynamoHintsComponentConfig() DynamoHintsComponentConfig {
	return nemo_flow.NewDynamoHintsComponentConfig()
}

func NewToolParallelismComponentConfig() ToolParallelismComponentConfig {
	return nemo_flow.NewToolParallelismComponentConfig()
}

func TelemetryComponent(config TelemetryComponentConfig) ComponentSpec {
	return nemo_flow.TelemetryComponent(config)
}

func DynamoHintsComponent(config DynamoHintsComponentConfig) ComponentSpec {
	return nemo_flow.DynamoHintsComponent(config)
}

func ToolParallelismComponent(config ToolParallelismComponentConfig) ComponentSpec {
	return nemo_flow.ToolParallelismComponent(config)
}

func ExternalComponent(config ExternalComponentConfig) ComponentSpec {
	return nemo_flow.ExternalComponent(config)
}

func ValidateConfig(config Config) (ConfigReport, error) {
	return nemo_flow.ValidateOptimizerConfig(config)
}

func NewRuntime(config Config) (*Runtime, error) {
	return nemo_flow.NewOptimizerRuntime(config)
}

func RegisterPlugin(pluginKind string, handler PluginHandler) error {
	return nemo_flow.RegisterOptimizerPlugin(pluginKind, handler)
}

func DeregisterPlugin(pluginKind string) error {
	return nemo_flow.DeregisterOptimizerPlugin(pluginKind)
}
