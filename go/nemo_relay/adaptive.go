// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// AdaptivePluginKind is the top-level plugin kind used by the adaptive component.
const AdaptivePluginKind = "adaptive"

// AdaptiveConfig is the canonical Go shape for the adaptive plugin config document.
type AdaptiveConfig struct {
	Version         uint32                 `json:"version,omitempty"`
	AgentID         string                 `json:"agent_id,omitempty"`
	State           *AdaptiveStateConfig   `json:"state,omitempty"`
	Telemetry       *TelemetryConfig       `json:"telemetry,omitempty"`
	AdaptiveHints   *AdaptiveHintsConfig   `json:"adaptive_hints,omitempty"`
	ToolParallelism *ToolParallelismConfig `json:"tool_parallelism,omitempty"`
	Acg             *AcgConfig             `json:"acg,omitempty"`
	ResponseCache   *ResponseCacheConfig   `json:"response_cache,omitempty"`
	Policy          *ConfigPolicy          `json:"policy,omitempty"`
}

// AdaptiveStateConfig selects the adaptive state backend.
type AdaptiveStateConfig struct {
	Backend AdaptiveBackendSpec `json:"backend"`
}

// AdaptiveBackendSpec selects the backend kind and backend-specific config.
type AdaptiveBackendSpec struct {
	Kind   string         `json:"kind"`
	Config map[string]any `json:"config,omitempty"`
}

// TelemetryConfig configures the built-in adaptive telemetry subscriber and learners.
type TelemetryConfig struct {
	SubscriberName string   `json:"subscriber_name,omitempty"`
	Learners       []string `json:"learners,omitempty"`
}

// AdaptiveHintsConfig configures built-in LLM request hint injection.
type AdaptiveHintsConfig struct {
	Priority       int32  `json:"priority,omitempty"`
	BreakChain     bool   `json:"break_chain,omitempty"`
	InjectHeader   bool   `json:"inject_header,omitempty"`
	InjectBodyPath string `json:"inject_body_path,omitempty"`
}

// ToolParallelismConfig configures built-in adaptive tool scheduling.
type ToolParallelismConfig struct {
	Priority int32  `json:"priority,omitempty"`
	Mode     string `json:"mode,omitempty"`
}

// AcgStabilityThresholds configures prompt stability classification thresholds.
type AcgStabilityThresholds struct {
	StableThreshold                  float64 `json:"stable_threshold,omitempty"`
	SemiStableThreshold              float64 `json:"semi_stable_threshold,omitempty"`
	MinObservationsForFullConfidence uint32  `json:"min_observations_for_full_confidence,omitempty"`
}

// AcgConfig configures the adaptive cache governor.
type AcgConfig struct {
	Provider            string                  `json:"provider,omitempty"`
	ObservationWindow   uint32                  `json:"observation_window,omitempty"`
	Priority            int32                   `json:"priority,omitempty"`
	StabilityThresholds *AcgStabilityThresholds `json:"stability_thresholds,omitempty"`
}

// ResponseCacheConfig configures the opt-in LLM response cache: a section
// of the adaptive config (a sibling to acg/adaptive_hints/tool_parallelism), not a
// standalone plugin kind. The Rust core validates and installs it from the adaptive
// runtime; this struct only has to carry the section through to the FFI validator.
type ResponseCacheConfig struct {
	// TTLSeconds is how long a stored answer stays reusable, in seconds (> 0).
	// Always serialized: with omitempty an explicit 0 would vanish and Rust's
	// default (3600) would silently replace it instead of failing validation.
	TTLSeconds uint64 `json:"ttl_seconds"`
	// Namespace is folded into every key to separate environments/tenants.
	Namespace string `json:"namespace,omitempty"`
	// Priority is the execution-intercept priority; lower runs first/outermost.
	// Always serialized so an explicit 0 (outermost) stays expressible.
	Priority int32 `json:"priority"`
	// BypassRate is the probability in [0.0, 1.0] of skipping the cache and running live.
	BypassRate float64 `json:"bypass_rate,omitempty"`
	// CacheNondeterministic caches temperature>0 requests too (default true).
	//
	// Deliberately NOT omitempty: the Rust default is true, so omitting a false
	// value would let the core re-apply true and silently re-enable caching of
	// nondeterministic requests. Always serializing the field keeps an explicit
	// false expressible — unlike the other zero-default fields here.
	CacheNondeterministic bool `json:"cache_nondeterministic"`
	// KeyStrategy is the key strategy. Only "exact_request" is supported.
	KeyStrategy string `json:"key_strategy,omitempty"`
	// HeaderAllowlist lists request headers folded into the key; never auth headers.
	HeaderAllowlist []string `json:"header_allowlist,omitempty"`
	// SkipKeys lists extra top-level request-body keys to drop from the key.
	SkipKeys []string `json:"skip_keys,omitempty"`
	// Backend selects the cache's own storage backend (distinct from the adaptive
	// state backend). Defaults to in-memory when nil.
	Backend *ResponseCacheBackendConfig `json:"backend,omitempty"`
}

// ResponseCacheBackendConfig selects the response-cache backend kind and options.
type ResponseCacheBackendConfig struct {
	// Kind is "in_memory" or "redis" (redis needs the redis-backend build feature).
	Kind string `json:"kind"`
	// Config holds backend-specific options (in_memory: max_bytes;
	// redis: url/key_prefix).
	Config map[string]any `json:"config,omitempty"`
}

// AdaptiveComponentSpec wraps one adaptive config as a top-level plugin component.
type AdaptiveComponentSpec struct {
	Enabled bool           `json:"enabled,omitempty"`
	Config  AdaptiveConfig `json:"config"`
}

// NewAdaptiveConfig returns a default adaptive config with version 1.
func NewAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{Version: 1}
}

// NewInMemoryAdaptiveBackend returns an in-memory adaptive backend spec.
func NewInMemoryAdaptiveBackend() AdaptiveBackendSpec {
	return AdaptiveBackendSpec{
		Kind:   "in_memory",
		Config: map[string]any{},
	}
}

// NewRedisAdaptiveBackend returns a Redis adaptive backend spec.
func NewRedisAdaptiveBackend(url, keyPrefix string) AdaptiveBackendSpec {
	return AdaptiveBackendSpec{
		Kind: "redis",
		Config: map[string]any{
			"url":        url,
			"key_prefix": keyPrefix,
		},
	}
}

// NewTelemetryConfig returns default built-in adaptive telemetry settings.
func NewTelemetryConfig() TelemetryConfig {
	return TelemetryConfig{}
}

// NewAdaptiveHintsConfig returns default adaptive hints injection settings.
func NewAdaptiveHintsConfig() AdaptiveHintsConfig {
	return AdaptiveHintsConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}
}

// NewToolParallelismConfig returns default adaptive tool scheduling settings.
func NewToolParallelismConfig() ToolParallelismConfig {
	return ToolParallelismConfig{
		Priority: 100,
		Mode:     "observe_only",
	}
}

// NewAcgStabilityThresholds returns default ACG stability thresholds.
func NewAcgStabilityThresholds() AcgStabilityThresholds {
	return AcgStabilityThresholds{
		StableThreshold:                  0.95,
		SemiStableThreshold:              0.50,
		MinObservationsForFullConfidence: 20,
	}
}

// NewAcgConfig returns default adaptive cache governor settings.
func NewAcgConfig() AcgConfig {
	thresholds := NewAcgStabilityThresholds()
	return AcgConfig{
		Provider:            "passthrough",
		ObservationWindow:   100,
		Priority:            50,
		StabilityThresholds: &thresholds,
	}
}

// NewResponseCacheConfig returns default response cache settings, mirroring
// the Rust ResponseCacheConfig defaults (caching on, exact-request keying). Backend
// is left nil so the core applies its in-memory default; set it for redis or to tune
// the in-memory budget.
func NewResponseCacheConfig() ResponseCacheConfig {
	return ResponseCacheConfig{
		TTLSeconds:            3600,
		Priority:              50,
		CacheNondeterministic: true,
		KeyStrategy:           "exact_request",
	}
}

// NewInMemoryResponseCacheBackend returns an in-memory response-cache backend spec.
func NewInMemoryResponseCacheBackend() ResponseCacheBackendConfig {
	return ResponseCacheBackendConfig{
		Kind:   "in_memory",
		Config: map[string]any{},
	}
}

// NewRedisResponseCacheBackend returns a Redis response-cache backend spec.
func NewRedisResponseCacheBackend(url, keyPrefix string) ResponseCacheBackendConfig {
	return ResponseCacheBackendConfig{
		Kind: "redis",
		Config: map[string]any{
			"url":        url,
			"key_prefix": keyPrefix,
		},
	}
}

// NewAdaptiveComponentSpec wraps adaptive config as an enabled top-level component.
func NewAdaptiveComponentSpec(config AdaptiveConfig) AdaptiveComponentSpec {
	return AdaptiveComponentSpec{
		Enabled: true,
		Config:  config,
	}
}

// PluginComponent converts the adaptive component wrapper into the shared plugin shape.
func (spec AdaptiveComponentSpec) PluginComponent() PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    AdaptivePluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
}

// AdaptiveComponent converts adaptive config directly into a shared plugin component.
//
// Prefer NewAdaptiveComponentSpec(config).PluginComponent() when the adaptive
// component wrapper itself is part of the public surface you want to expose.
func AdaptiveComponent(config AdaptiveConfig) PluginComponentSpec {
	return NewAdaptiveComponentSpec(config).PluginComponent()
}

func mustConfigMap(value any) map[string]any {
	payload, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		panic(err)
	}
	if out == nil {
		return map[string]any{}
	}
	return out
}
