// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiAdaptiveRuntime FfiAdaptiveRuntime;
typedef struct FfiScopeHandle FfiScopeHandle;

extern int32_t nemo_relay_adaptive_validate_config(const char* config_json, char** out_json);
extern int32_t nemo_relay_adaptive_runtime_create(const char* config_json, FfiAdaptiveRuntime** out);
extern int32_t nemo_relay_adaptive_runtime_register(FfiAdaptiveRuntime* runtime);
extern int32_t nemo_relay_adaptive_runtime_deregister(FfiAdaptiveRuntime* runtime);
extern int32_t nemo_relay_adaptive_runtime_shutdown(FfiAdaptiveRuntime* runtime);
extern int32_t nemo_relay_adaptive_runtime_wait_for_idle(FfiAdaptiveRuntime* runtime);
extern int32_t nemo_relay_adaptive_runtime_report_json(FfiAdaptiveRuntime* runtime, char** out_json);
extern int32_t nemo_relay_adaptive_runtime_bind_scope(FfiAdaptiveRuntime* runtime, const FfiScopeHandle* scope);
extern int32_t nemo_relay_adaptive_runtime_build_cache_request_facts(FfiAdaptiveRuntime* runtime, const char* options_json, char** out_json);
extern int32_t nemo_relay_adaptive_build_cache_telemetry_event(const char* options_json, char** out_json);
extern int32_t nemo_relay_adaptive_set_latency_sensitivity(uint32_t value);
extern void nemo_relay_adaptive_runtime_free(FfiAdaptiveRuntime* ptr);
extern void nemo_relay_string_free(char* ptr);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// AdaptiveRuntime owns adaptive runtime registrations outside the generic plugin system.
type AdaptiveRuntime struct {
	ptr *C.FfiAdaptiveRuntime
}

// CacheUsage is normalized LLM token usage used to build adaptive cache telemetry.
type CacheUsage struct {
	PromptTokens     *uint64         `json:"prompt_tokens,omitempty"`
	CompletionTokens *uint64         `json:"completion_tokens,omitempty"`
	TotalTokens      *uint64         `json:"total_tokens,omitempty"`
	CacheReadTokens  *uint64         `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens *uint64         `json:"cache_write_tokens,omitempty"`
	Cost             json.RawMessage `json:"cost,omitempty"`
}

// AgentIdentity identifies the agent associated with cache telemetry.
type AgentIdentity struct {
	AgentID         string `json:"agent_id"`
	TemplateVersion string `json:"template_version"`
	ToolsetHash     string `json:"toolset_hash"`
	ModelFamily     string `json:"model_family"`
	TenantScope     string `json:"tenant_scope"`
}

// CacheRequestFactsInput is the typed input for building cache request facts.
type CacheRequestFactsInput struct {
	Provider         string          `json:"provider"`
	RequestID        string          `json:"request_id"`
	AnnotatedRequest json.RawMessage `json:"annotated_request"`
	AgentID          string          `json:"agent_id"`
	Timestamp        string          `json:"timestamp,omitempty"`
}

// CacheRequestFacts describes request-time facts used to classify cache misses.
type CacheRequestFacts struct {
	Provider                   string   `json:"provider"`
	StablePrefixLength         uint64   `json:"stable_prefix_length"`
	StablePrefixTokens         *uint32  `json:"stable_prefix_tokens,omitempty"`
	RequiredMinTokens          *uint32  `json:"required_min_tokens,omitempty"`
	FirstMismatchSpanID        *string  `json:"first_mismatch_span_id,omitempty"`
	FirstMismatchSequenceIndex *uint32  `json:"first_mismatch_sequence_index,omitempty"`
	ExpectedHashPrefix         *string  `json:"expected_hash_prefix,omitempty"`
	ActualHashPrefix           *string  `json:"actual_hash_prefix,omitempty"`
	RetentionWindowSecs        *float64 `json:"retention_window_secs,omitempty"`
	ObservedGapSecs            *float64 `json:"observed_gap_secs,omitempty"`
	MissingFacts               []string `json:"missing_facts,omitempty"`
}

// CacheTelemetryEventInput is the typed input for building cache telemetry events.
type CacheTelemetryEventInput struct {
	Provider        string             `json:"provider"`
	RequestID       string             `json:"request_id"`
	Usage           *CacheUsage        `json:"usage,omitempty"`
	RequestFacts    *CacheRequestFacts `json:"request_facts,omitempty"`
	AgentID         string             `json:"agent_id"`
	TemplateVersion string             `json:"template_version"`
	ToolsetHash     string             `json:"toolset_hash"`
	ModelFamily     string             `json:"model_family"`
	TenantScope     string             `json:"tenant_scope"`
	Timestamp       string             `json:"timestamp,omitempty"`
}

// CacheTelemetryEvent is the normalized adaptive cache telemetry event.
type CacheTelemetryEvent struct {
	RequestID           string         `json:"request_id"`
	AgentIdentity       AgentIdentity  `json:"agent_identity"`
	CacheReadTokens     uint64         `json:"cache_read_tokens"`
	CacheCreationTokens uint64         `json:"cache_creation_tokens"`
	TotalPromptTokens   uint64         `json:"total_prompt_tokens"`
	HitRate             float64        `json:"hit_rate"`
	MissReason          map[string]any `json:"miss_reason,omitempty"`
	MissDiagnosis       map[string]any `json:"miss_diagnosis,omitempty"`
	Provider            string         `json:"provider"`
	Timestamp           string         `json:"timestamp"`
}

func adaptiveConfigCString(config AdaptiveConfig) (*C.char, error) {
	payload, err := jsonMarshal(config)
	if err != nil {
		return nil, err
	}
	return C.CString(string(payload)), nil
}

func adaptiveOptionsCString(value any) (*C.char, error) {
	payload, err := jsonMarshal(value)
	if err != nil {
		return nil, err
	}
	return C.CString(string(payload)), nil
}

func adaptiveJSONString(status C.int32_t, out *C.char) (string, error) {
	return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
		C.nemo_relay_string_free(out)
	})
}

// ValidateAdaptiveConfig validates an adaptive runtime config without constructing a runtime.
func ValidateAdaptiveConfig(config AdaptiveConfig) (ConfigReport, error) {
	cConfig, err := adaptiveConfigCString(config)
	if err != nil {
		return ConfigReport{}, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var out *C.char
	raw, err := adaptiveJSONString(C.nemo_relay_adaptive_validate_config(cConfig, &out), out)
	if err != nil {
		return ConfigReport{}, err
	}
	var report ConfigReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// NewAdaptiveRuntime creates an owned adaptive runtime from config.
func NewAdaptiveRuntime(config AdaptiveConfig) (*AdaptiveRuntime, error) {
	cConfig, err := adaptiveConfigCString(config)
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var out *C.FfiAdaptiveRuntime
	if err := checkStatus(C.nemo_relay_adaptive_runtime_create(cConfig, &out)); err != nil {
		return nil, err
	}
	r := &AdaptiveRuntime{ptr: out}
	runtime.SetFinalizer(r, func(r *AdaptiveRuntime) {
		if r.ptr != nil {
			C.nemo_relay_adaptive_runtime_free(r.ptr)
			r.ptr = nil
		}
	})
	return r, nil
}

// Register activates all configured adaptive runtime features.
func (r *AdaptiveRuntime) Register() error {
	if r == nil || r.ptr == nil {
		return errNilAdaptiveRuntime()
	}
	return checkStatus(C.nemo_relay_adaptive_runtime_register(r.ptr))
}

// Deregister removes all adaptive runtime registrations owned by this runtime.
func (r *AdaptiveRuntime) Deregister() error {
	if r == nil || r.ptr == nil {
		return errNilAdaptiveRuntime()
	}
	return checkStatus(C.nemo_relay_adaptive_runtime_deregister(r.ptr))
}

// Shutdown deregisters the adaptive runtime and consumes its Rust runtime state.
func (r *AdaptiveRuntime) Shutdown() error {
	if r == nil || r.ptr == nil {
		return errNilAdaptiveRuntime()
	}
	err := checkStatus(C.nemo_relay_adaptive_runtime_shutdown(r.ptr))
	C.nemo_relay_adaptive_runtime_free(r.ptr)
	r.ptr = nil
	runtime.SetFinalizer(r, nil)
	return err
}

// WaitForIdle blocks until adaptive telemetry has processed pending events.
func (r *AdaptiveRuntime) WaitForIdle() error {
	if r == nil || r.ptr == nil {
		return errNilAdaptiveRuntime()
	}
	return checkStatus(C.nemo_relay_adaptive_runtime_wait_for_idle(r.ptr))
}

// Report returns the runtime validation report captured during construction.
func (r *AdaptiveRuntime) Report() (ConfigReport, error) {
	if r == nil || r.ptr == nil {
		return ConfigReport{}, errNilAdaptiveRuntime()
	}
	var out *C.char
	raw, err := adaptiveJSONString(C.nemo_relay_adaptive_runtime_report_json(r.ptr, &out), out)
	if err != nil {
		return ConfigReport{}, err
	}
	var report ConfigReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// BindScope binds the runtime's ACG request rewrite to an active scope.
func (r *AdaptiveRuntime) BindScope(scope *ScopeHandle) error {
	if r == nil || r.ptr == nil {
		return errNilAdaptiveRuntime()
	}
	if scope == nil || scope.ptr == nil {
		return errNilScopeHandle()
	}
	return checkStatus(C.nemo_relay_adaptive_runtime_bind_scope(
		r.ptr,
		(*C.FfiScopeHandle)(unsafe.Pointer(scope.ptr)),
	))
}

// BuildCacheRequestFacts derives cache request facts from an annotated request.
func (r *AdaptiveRuntime) BuildCacheRequestFacts(input CacheRequestFactsInput) (*CacheRequestFacts, error) {
	if r == nil || r.ptr == nil {
		return nil, errNilAdaptiveRuntime()
	}
	cOptions, err := adaptiveOptionsCString(input)
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cOptions))

	var out *C.char
	raw, err := adaptiveJSONString(C.nemo_relay_adaptive_runtime_build_cache_request_facts(r.ptr, cOptions, &out), out)
	if err != nil {
		return nil, err
	}
	if raw == "null" {
		return nil, nil
	}
	var facts CacheRequestFacts
	if err := jsonUnmarshal([]byte(raw), &facts); err != nil {
		return nil, err
	}
	return &facts, nil
}

// BuildCacheTelemetryEvent builds one cache telemetry event from normalized usage.
func BuildCacheTelemetryEvent(input CacheTelemetryEventInput) (*CacheTelemetryEvent, error) {
	cOptions, err := adaptiveOptionsCString(input)
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cOptions))

	var out *C.char
	raw, err := adaptiveJSONString(C.nemo_relay_adaptive_build_cache_telemetry_event(cOptions, &out), out)
	if err != nil {
		return nil, err
	}
	if raw == "null" {
		return nil, nil
	}
	var event CacheTelemetryEvent
	if err := jsonUnmarshal([]byte(raw), &event); err != nil {
		return nil, err
	}
	return &event, nil
}

// SetLatencySensitivity sets manual latency sensitivity on the current scope.
func SetLatencySensitivity(value uint32) error {
	return checkStatus(C.nemo_relay_adaptive_set_latency_sensitivity(C.uint32_t(value)))
}

func errNilAdaptiveRuntime() error {
	return errors.New("adaptive runtime is nil or shut down")
}

func errNilScopeHandle() error {
	return errors.New("scope handle is nil")
}
