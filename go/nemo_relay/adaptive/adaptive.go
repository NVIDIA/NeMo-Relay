// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package adaptive

import nemo_relay "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"

// UnsupportedBehavior controls how adaptive config validation handles unsupported input.
type UnsupportedBehavior = nemo_relay.UnsupportedBehavior

const (
	UnsupportedBehaviorIgnore = nemo_relay.UnsupportedBehaviorIgnore
	UnsupportedBehaviorWarn   = nemo_relay.UnsupportedBehaviorWarn
	UnsupportedBehaviorError  = nemo_relay.UnsupportedBehaviorError
)

// DiagnosticLevel is the severity of one adaptive validation diagnostic.
type DiagnosticLevel = nemo_relay.DiagnosticLevel

const (
	DiagnosticLevelWarning = nemo_relay.DiagnosticLevelWarning
	DiagnosticLevelError   = nemo_relay.DiagnosticLevelError
)

// Config is the canonical adaptive config document.
type Config = nemo_relay.AdaptiveConfig

// ComponentSpec wraps adaptive config as a top-level adaptive component.
type ComponentSpec = nemo_relay.AdaptiveComponentSpec

// Runtime owns adaptive runtime registrations outside the plugin system.
type Runtime = nemo_relay.AdaptiveRuntime

// StateConfig selects the adaptive state backend.
type StateConfig = nemo_relay.AdaptiveStateConfig

// BackendSpec selects the adaptive state backend kind and backend-specific config.
type BackendSpec = nemo_relay.AdaptiveBackendSpec

// TelemetryConfig configures built-in adaptive telemetry.
type TelemetryConfig = nemo_relay.TelemetryConfig

// AdaptiveHintsConfig configures built-in adaptive hint injection.
type AdaptiveHintsConfig = nemo_relay.AdaptiveHintsConfig

// ToolParallelismConfig configures built-in adaptive tool scheduling.
type ToolParallelismConfig = nemo_relay.ToolParallelismConfig

// AcgStabilityThresholds configures ACG prompt-stability classification.
type AcgStabilityThresholds = nemo_relay.AcgStabilityThresholds

// AcgConfig configures the adaptive cache governor.
type AcgConfig = nemo_relay.AcgConfig

// CacheWriteTokenCounts is prompt-cache write tokens retained for one cache duration.
type CacheWriteTokenCounts = nemo_relay.CacheWriteTokenCounts

// CacheUsage is normalized LLM token usage for cache telemetry.
type CacheUsage = nemo_relay.CacheUsage

// AgentIdentity identifies the agent associated with cache telemetry.
type AgentIdentity = nemo_relay.AgentIdentity

// CacheRequestFactsInput is the typed input for building cache request facts.
type CacheRequestFactsInput = nemo_relay.CacheRequestFactsInput

// CacheRequestFacts describes request-time facts used to classify cache misses.
type CacheRequestFacts = nemo_relay.CacheRequestFacts

// CacheTelemetryEventInput is the typed input for building cache telemetry events.
type CacheTelemetryEventInput = nemo_relay.CacheTelemetryEventInput

// CacheTelemetryEvent is the normalized adaptive cache telemetry event.
type CacheTelemetryEvent = nemo_relay.CacheTelemetryEvent

// PluginKind is the top-level plugin kind used by the adaptive component.
const PluginKind = nemo_relay.AdaptivePluginKind

// NewConfig returns a default adaptive config with version 1.
func NewConfig() Config {
	return nemo_relay.NewAdaptiveConfig()
}

// NewInMemoryBackend returns an in-memory adaptive backend spec.
func NewInMemoryBackend() BackendSpec {
	return nemo_relay.NewInMemoryAdaptiveBackend()
}

// NewRedisBackend returns a Redis adaptive backend spec.
func NewRedisBackend(url, keyPrefix string) BackendSpec {
	return nemo_relay.NewRedisAdaptiveBackend(url, keyPrefix)
}

// NewTelemetryConfig returns default adaptive telemetry settings.
func NewTelemetryConfig() TelemetryConfig {
	return nemo_relay.NewTelemetryConfig()
}

// NewAdaptiveHintsConfig returns default adaptive hints injection settings.
func NewAdaptiveHintsConfig() AdaptiveHintsConfig {
	return nemo_relay.NewAdaptiveHintsConfig()
}

// NewToolParallelismConfig returns default adaptive tool scheduling settings.
func NewToolParallelismConfig() ToolParallelismConfig {
	return nemo_relay.NewToolParallelismConfig()
}

// NewAcgStabilityThresholds returns default ACG stability thresholds.
func NewAcgStabilityThresholds() AcgStabilityThresholds {
	return nemo_relay.NewAcgStabilityThresholds()
}

// NewAcgConfig returns default adaptive cache governor settings.
func NewAcgConfig() AcgConfig {
	return nemo_relay.NewAcgConfig()
}

// NewComponentSpec wraps adaptive config as an enabled top-level adaptive component.
func NewComponentSpec(config Config) ComponentSpec {
	return nemo_relay.NewAdaptiveComponentSpec(config)
}

// Component converts adaptive config directly into the shared plugin shape.
func Component(config Config) nemo_relay.PluginComponentSpec {
	return nemo_relay.AdaptiveComponent(config)
}

// ValidateConfig validates an adaptive runtime config without constructing a runtime.
func ValidateConfig(config Config) (nemo_relay.ConfigReport, error) {
	return nemo_relay.ValidateAdaptiveConfig(config)
}

// NewRuntime creates an owned adaptive runtime from config.
func NewRuntime(config Config) (*Runtime, error) {
	return nemo_relay.NewAdaptiveRuntime(config)
}

// BuildCacheTelemetryEvent builds one cache telemetry event from normalized usage.
func BuildCacheTelemetryEvent(input CacheTelemetryEventInput) (*CacheTelemetryEvent, error) {
	return nemo_relay.BuildCacheTelemetryEvent(input)
}

// SetLatencySensitivity sets manual latency sensitivity on the current scope.
func SetLatencySensitivity(value uint32) error {
	return nemo_relay.SetLatencySensitivity(value)
}

// CacheMissReasonKind mirrors the native cache telemetry contract.
type CacheMissReasonKind = nemo_relay.CacheMissReasonKind

// CacheMissReason mirrors the native cache telemetry contract.
type CacheMissReason = nemo_relay.CacheMissReason

// CacheMissEvidence mirrors the native cache telemetry contract.
type CacheMissEvidence = nemo_relay.CacheMissEvidence

// CacheMissDiagnosis mirrors the native cache telemetry contract.
type CacheMissDiagnosis = nemo_relay.CacheMissDiagnosis

// CacheMissPrefixMismatchEvidence mirrors the native cache telemetry contract.
type CacheMissPrefixMismatchEvidence = nemo_relay.CacheMissPrefixMismatchEvidence

// CacheMissBelowMinimumThresholdEvidence mirrors the native cache telemetry contract.
type CacheMissBelowMinimumThresholdEvidence = nemo_relay.CacheMissBelowMinimumThresholdEvidence

// CacheMissRetentionExpiredEvidence mirrors the native cache telemetry contract.
type CacheMissRetentionExpiredEvidence = nemo_relay.CacheMissRetentionExpiredEvidence

// CacheMissUnknownEvidence mirrors the native cache telemetry contract.
type CacheMissUnknownEvidence = nemo_relay.CacheMissUnknownEvidence

const (
	CacheMissReasonPrefixMismatch        = nemo_relay.CacheMissReasonPrefixMismatch
	CacheMissReasonBelowMinimumThreshold = nemo_relay.CacheMissReasonBelowMinimumThreshold
	CacheMissReasonRetentionExpired      = nemo_relay.CacheMissReasonRetentionExpired
	CacheMissReasonRoutingMismatch       = nemo_relay.CacheMissReasonRoutingMismatch
	CacheMissReasonEvicted               = nemo_relay.CacheMissReasonEvicted
	CacheMissReasonUnsupportedFeature    = nemo_relay.CacheMissReasonUnsupportedFeature
	CacheMissReasonColdStart             = nemo_relay.CacheMissReasonColdStart
	CacheMissReasonUnknown               = nemo_relay.CacheMissReasonUnknown
	CacheMissReasonOther                 = nemo_relay.CacheMissReasonOther
)
