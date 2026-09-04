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

// ResponseCacheKeyStrategy selects how LLM response-cache keys are derived.
type ResponseCacheKeyStrategy string

const (
	// ResponseCacheKeyStrategyExactRequest keys on the normalized request exactly.
	ResponseCacheKeyStrategyExactRequest ResponseCacheKeyStrategy = "exact_request"
	// ResponseCacheKeyStrategyLogical ignores tool-description wording while preserving interfaces.
	ResponseCacheKeyStrategyLogical ResponseCacheKeyStrategy = "logical"
)

// ResponseCacheConfig configures the opt-in LLM response and tool-result cache: a section
// of the adaptive config (a sibling to acg/adaptive_hints/tool_parallelism), not a
// standalone plugin kind. The Rust core validates and installs it from the adaptive
// runtime; this struct only has to carry the section through to the FFI validator.
type ResponseCacheConfig struct {
	// TTLSeconds is how long a stored answer stays reusable, in seconds (> 0).
	// Nil delegates to Rust's default (3600); a pointer to 0 is rejected.
	TTLSeconds *uint64 `json:"ttl_seconds,omitempty"`
	// Namespace is a required, non-empty cache trust domain folded into every
	// key. One configured cache must not span mutually untrusted tenants or
	// upstreams; the empty constructor value is rejected at validation.
	Namespace string `json:"namespace,omitempty"`
	// Priority is the execution-intercept priority; lower runs first/outermost.
	// Nil delegates to Rust's default (50); a pointer to 0 selects outermost.
	Priority *int32 `json:"priority,omitempty"`
	// BypassRate is the probability in [0.0, 1.0] of skipping the cache and running live.
	BypassRate float64 `json:"bypass_rate,omitempty"`
	// CacheNondeterministic lets requests that are not explicitly deterministic
	// use the cache (default false).
	CacheNondeterministic bool `json:"cache_nondeterministic"`
	// KeyStrategy is the typed LLM response-cache key derivation strategy.
	KeyStrategy ResponseCacheKeyStrategy `json:"key_strategy,omitempty"`
	// HeaderAllowlist lists request headers folded into the key; never auth headers.
	HeaderAllowlist []string `json:"header_allowlist,omitempty"`
	// Backend selects the cache's own storage backend (distinct from the adaptive
	// state backend). Defaults to in-memory when nil.
	Backend *ResponseCacheBackendConfig `json:"backend,omitempty"`
	// Tools configures the optional tool-result cache.
	Tools *ResponseCacheToolsConfig `json:"tools,omitempty"`
}

// ResponseCacheToolsConfig configures caching for read-only, stable tools.
type ResponseCacheToolsConfig struct {
	Enabled bool `json:"enabled,omitempty"`
	// Priority is the execution-intercept priority. Nil delegates to Rust's
	// default (100); a pointer to 0 selects outermost.
	Priority *int32 `json:"priority,omitempty"`
	// CacheErrors lets error-shaped tool results be cached (default false).
	CacheErrors bool                              `json:"cache_errors"`
	Default     *ResponseCacheToolClass           `json:"default,omitempty"`
	Classes     map[string]ResponseCacheToolClass `json:"classes,omitempty"`
	// Overrides keys are exact tool names or prefix*, *suffix, and *contains* patterns.
	Overrides map[string]ResponseCacheToolOverride `json:"overrides,omitempty"`
}

// ResponseCacheToolClass defines a shared tool-cache policy.
type ResponseCacheToolClass struct {
	Cacheable   bool     `json:"cacheable,omitempty"`
	TTLSeconds  *uint64  `json:"ttl_seconds,omitempty"`
	BypassRate  *float64 `json:"bypass_rate,omitempty"`
	ToolVersion *string  `json:"tool_version,omitempty"`
	ArgSkip     []string `json:"arg_skip,omitempty"`
	// Members contains exact tool names or prefix*, *suffix, and *contains* patterns.
	Members []string `json:"members,omitempty"`
}

// ResponseCacheToolOverride refines a resolved tool-cache policy.
type ResponseCacheToolOverride struct {
	Cacheable   *bool     `json:"cacheable,omitempty"`
	TTLSeconds  *uint64   `json:"ttl_seconds,omitempty"`
	BypassRate  *float64  `json:"bypass_rate,omitempty"`
	ToolVersion *string   `json:"tool_version,omitempty"`
	ArgSkip     *[]string `json:"arg_skip,omitempty"`
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
// the Rust ResponseCacheConfig defaults (exact-request keying with nondeterministic
// caching disabled). Set Namespace to one cache trust domain before validation.
// Backend is left nil so the core applies its in-memory default; set it for redis
// or to tune the in-memory budget.
func NewResponseCacheConfig() ResponseCacheConfig {
	ttlSeconds := uint64(3600)
	priority := int32(50)
	return ResponseCacheConfig{
		TTLSeconds:            &ttlSeconds,
		Priority:              &priority,
		CacheNondeterministic: false,
		KeyStrategy:           ResponseCacheKeyStrategyExactRequest,
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

// NewResponseCacheToolsConfig returns a disabled tool-result cache config.
func NewResponseCacheToolsConfig() ResponseCacheToolsConfig {
	priority := int32(100)
	return ResponseCacheToolsConfig{
		CacheErrors: false,
		Priority:    &priority,
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
