// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package optimizer

import natnexus "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"

type UnsupportedBehavior = natnexus.UnsupportedBehavior

const (
	UnsupportedBehaviorIgnore = natnexus.UnsupportedBehaviorIgnore
	UnsupportedBehaviorWarn   = natnexus.UnsupportedBehaviorWarn
	UnsupportedBehaviorError  = natnexus.UnsupportedBehaviorError
)

type DiagnosticLevel = natnexus.OptimizerDiagnosticLevel

const (
	DiagnosticLevelWarning = natnexus.OptimizerDiagnosticLevelWarning
	DiagnosticLevelError   = natnexus.OptimizerDiagnosticLevelError
)

type Config = natnexus.OptimizerConfig
type StateConfig = natnexus.OptimizerStateConfig
type BackendSpec = natnexus.OptimizerBackendSpec
type ComponentSpec = natnexus.OptimizerComponentSpec
type ConfigPolicy = natnexus.OptimizerConfigPolicy
type ConfigReport = natnexus.OptimizerConfigReport
type ConfigDiagnostic = natnexus.OptimizerConfigDiagnostic

type TelemetryComponentConfig = natnexus.TelemetryComponentConfig
type DynamoHintsComponentConfig = natnexus.DynamoHintsComponentConfig
type ToolParallelismComponentConfig = natnexus.ToolParallelismComponentConfig
type ExternalComponentConfig = natnexus.ExternalComponentConfig

type Runtime = natnexus.OptimizerRuntime
type PluginContext = natnexus.OptimizerPluginContext
type PluginHandler = natnexus.OptimizerPluginHandler
type PluginHandlerFuncs = natnexus.OptimizerPluginHandlerFuncs

func NewConfig() Config {
	return natnexus.NewOptimizerConfig()
}

func NewInMemoryBackend() BackendSpec {
	return natnexus.NewInMemoryOptimizerBackend()
}

func NewRedisBackend(url, keyPrefix string) BackendSpec {
	return natnexus.NewRedisOptimizerBackend(url, keyPrefix)
}

func NewTelemetryComponentConfig() TelemetryComponentConfig {
	return natnexus.NewTelemetryComponentConfig()
}

func NewDynamoHintsComponentConfig() DynamoHintsComponentConfig {
	return natnexus.NewDynamoHintsComponentConfig()
}

func NewToolParallelismComponentConfig() ToolParallelismComponentConfig {
	return natnexus.NewToolParallelismComponentConfig()
}

func TelemetryComponent(config TelemetryComponentConfig) ComponentSpec {
	return natnexus.TelemetryComponent(config)
}

func DynamoHintsComponent(config DynamoHintsComponentConfig) ComponentSpec {
	return natnexus.DynamoHintsComponent(config)
}

func ToolParallelismComponent(config ToolParallelismComponentConfig) ComponentSpec {
	return natnexus.ToolParallelismComponent(config)
}

func ExternalComponent(config ExternalComponentConfig) ComponentSpec {
	return natnexus.ExternalComponent(config)
}

func ValidateConfig(config Config) (ConfigReport, error) {
	return natnexus.ValidateOptimizerConfig(config)
}

func NewRuntime(config Config) (*Runtime, error) {
	return natnexus.NewOptimizerRuntime(config)
}

func RegisterPlugin(pluginKind string, handler PluginHandler) error {
	return natnexus.RegisterOptimizerPlugin(pluginKind, handler)
}

func DeregisterPlugin(pluginKind string) error {
	return natnexus.DeregisterOptimizerPlugin(pluginKind)
}
