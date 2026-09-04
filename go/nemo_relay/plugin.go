// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiPluginContext FfiPluginContext;
typedef struct FfiPluginHostActivation FfiPluginHostActivation;
typedef struct FfiLlmSanitizeRequestCodec FfiLlmSanitizeRequestCodec;
typedef struct FfiLlmSanitizeResponseCodec FfiLlmSanitizeResponseCodec;
typedef struct NemoRelayLlmSanitizeRequestContext { uint32_t codec_kind; const char* codec_id; const FfiLlmSanitizeRequestCodec* codec; } NemoRelayLlmSanitizeRequestContext;
typedef struct NemoRelayLlmSanitizeResponseContext { uint32_t codec_kind; const char* codec_id; const FfiLlmSanitizeResponseCodec* codec; } NemoRelayLlmSanitizeResponseContext;

typedef void (*NemoRelayFreeFn)(void* user_data);
typedef char* (*NemoRelayPluginValidateCb)(void* user_data, const char* plugin_config_json);
typedef int32_t (*NemoRelayPluginRegisterCb)(void* user_data, const char* plugin_config_json, FfiPluginContext* ctx);
typedef void (*NemoRelayEventSubscriberFn)(void* user_data, const void* event);
typedef char* (*NemoRelayEventMetadataInjectorFn)(void* user_data, const void* event);
typedef char* (*NemoRelayEventSanitizeFn)(void* user_data, const void* event, const char* fields_json);
typedef char* (*NemoRelayToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef void* (*NemoRelayLlmSanitizeRequestCb)(void* user_data, const void* request, NemoRelayLlmSanitizeRequestContext context);
typedef char* (*NemoRelayLlmSanitizeResponseCb)(void* user_data, const char* response_json, NemoRelayLlmSanitizeResponseContext context);
typedef char* (*NemoRelayLlmConditionalCb)(void* user_data, const void* request);
typedef int32_t (*NemoRelayLlmRequestInterceptCb)(void* user_data, const char* name, const void* request, const char* annotated_json, char** out_outcome_json);
typedef char* (*NemoRelayLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NemoRelayLlmExecInterceptCb)(void* user_data, const char* native_json, NemoRelayLlmExecNextFn next_fn, void* next_ctx);
typedef char* (*NemoRelayToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NemoRelayToolExecInterceptCb)(void* user_data, const char* args_json, NemoRelayToolExecNextFn next_fn, void* next_ctx);

extern int32_t nemo_relay_plugin_initialize(const char* config_json, const char* additional_plugins_toml, FfiPluginHostActivation** out_activation, char** out_report_json);
extern int32_t nemo_relay_plugin_host_activation_report_json(FfiPluginHostActivation* activation, char** out_report_json);
extern int32_t nemo_relay_plugin_host_activation_is_active(FfiPluginHostActivation* activation, _Bool* out_active);
extern int32_t nemo_relay_plugin_validate(const char* config_json, const char* additional_plugins_toml, char** out_report_json);
extern int32_t nemo_relay_plugin_validate_exact(const char* config_json, char** out_report_json);
extern int32_t nemo_relay_plugin_host_activation_close(FfiPluginHostActivation* activation);
extern void nemo_relay_plugin_host_activation_free(FfiPluginHostActivation** activation);
extern int32_t nemo_relay_list_plugin_kinds_json(char** out_json);
extern int32_t nemo_relay_register_plugin(const char* plugin_kind, NemoRelayPluginValidateCb validate_cb, NemoRelayPluginRegisterCb register_cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_plugin(const char* plugin_kind);
extern void nemo_relay_string_free(char* ptr);

extern int32_t nemo_relay_plugin_context_register_subscriber(FfiPluginContext* ctx, const char* name, NemoRelayEventSubscriberFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_event_metadata_injector(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayEventMetadataInjectorFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_mark_sanitize_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_scope_sanitize_start_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_scope_sanitize_end_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolConditionalFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmSanitizeRequestCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmSanitizeResponseCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmConditionalCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoRelayLlmRequestInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_stream_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);

extern char* goPluginValidateTrampoline(void*, const char*);
extern int32_t goPluginRegisterTrampoline(void*, const char*, FfiPluginContext*);
extern void goEventSubscriberTrampoline(void*, const void*);
extern char* goEventMetadataInjectorTrampoline(void*, const void*);
extern char* goEventSanitizeTrampoline(void*, const void*, const char*);
extern void goFreeTrampoline(void*);
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern void* goLlmRequestTrampoline(void*, const void*, NemoRelayLlmSanitizeRequestContext);
extern char* goLlmResponseTrampoline(void*, const char*, NemoRelayLlmSanitizeResponseContext);
extern char* goLlmConditionalTrampoline(void*, const void*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NemoRelayLlmExecNextFn, void*);
extern int32_t goLlmRequestInterceptTrampoline(void*, const char*, const void*, const char*, char**);
extern char* goToolExecInterceptTrampoline(void*, const char*, NemoRelayToolExecNextFn, void*);
*/
import "C"

import (
	"errors"
	"log"
	"runtime"
	"sync"
	"unsafe"
)

const errPluginContextClosed = "plugin context is closed"

func checkedJSONString(status int32, raw func() string, free func()) (string, error) {
	if err := checkStatus(C.int32_t(status)); err != nil {
		return "", err
	}
	defer free()
	return raw(), nil
}

var (
	initializePluginHostJSON = func(configJSON string, additionalPluginsTOML *string) (unsafe.Pointer, string, error) {
		cConfig := C.CString(configJSON)
		defer C.free(unsafe.Pointer(cConfig))
		var cAdditional *C.char
		if additionalPluginsTOML != nil {
			cAdditional = C.CString(*additionalPluginsTOML)
			defer C.free(unsafe.Pointer(cAdditional))
		}
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		var activation *C.FfiPluginHostActivation
		var report *C.char
		status := C.nemo_relay_plugin_initialize(cConfig, cAdditional, &activation, &report)
		if err := checkStatus(status); err != nil {
			cleanupPartialPluginHostActivation(activation, report)
			return nil, "", err
		}
		if activation == nil || report == nil {
			cleanupPartialPluginHostActivation(activation, report)
			return nil, "", errors.New("plugin host activation returned incomplete outputs")
		}
		defer C.nemo_relay_string_free(report)
		return unsafe.Pointer(activation), C.GoString(report), nil
	}
	validatePluginHostJSON = func(configJSON string, additionalPluginsTOML *string) (string, error) {
		config := C.CString(configJSON)
		defer C.free(unsafe.Pointer(config))
		var additional *C.char
		if additionalPluginsTOML != nil {
			additional = C.CString(*additionalPluginsTOML)
			defer C.free(unsafe.Pointer(additional))
		}
		var report *C.char
		status := C.nemo_relay_plugin_validate(config, additional, &report)
		return checkedJSONString(int32(status), func() string { return C.GoString(report) }, func() {
			C.nemo_relay_string_free(report)
		})
	}
	validateExactPluginHostJSON = func(configJSON string) (string, error) {
		config := C.CString(configJSON)
		defer C.free(unsafe.Pointer(config))
		var report *C.char
		status := C.nemo_relay_plugin_validate_exact(config, &report)
		return checkedJSONString(int32(status), func() string { return C.GoString(report) }, func() {
			C.nemo_relay_string_free(report)
		})
	}
	pluginHostActivationReportJSON = func(ptr unsafe.Pointer) (string, error) {
		var report *C.char
		status := C.nemo_relay_plugin_host_activation_report_json((*C.FfiPluginHostActivation)(ptr), &report)
		return checkedJSONString(int32(status), func() string { return C.GoString(report) }, func() {
			C.nemo_relay_string_free(report)
		})
	}
	pluginHostActivationIsActive = func(ptr unsafe.Pointer) (bool, error) {
		var active C.bool
		status := C.nemo_relay_plugin_host_activation_is_active((*C.FfiPluginHostActivation)(ptr), &active)
		if err := checkStatus(status); err != nil {
			return false, err
		}
		return bool(active), nil
	}
	clearPluginHostActivation = func(ptr unsafe.Pointer) error {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		status := C.nemo_relay_plugin_host_activation_close((*C.FfiPluginHostActivation)(ptr))
		return checkStatus(status)
	}
	freePluginHostActivation = func(ptr unsafe.Pointer) {
		activation := (*C.FfiPluginHostActivation)(ptr)
		C.nemo_relay_plugin_host_activation_free(&activation)
	}
	listPluginKindsJSON = func() (string, error) {
		var out *C.char
		status := C.nemo_relay_list_plugin_kinds_json(&out)
		return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
			C.nemo_relay_string_free(out)
		})
	}
)

func cleanupPartialPluginHostActivation(activation *C.FfiPluginHostActivation, report *C.char) {
	if report != nil {
		C.nemo_relay_string_free(report)
	}
	if activation != nil {
		C.nemo_relay_plugin_host_activation_close(activation)
		C.nemo_relay_plugin_host_activation_free(&activation)
	}
}

// DiagnosticLevel is the severity level for one plugin diagnostic.
type DiagnosticLevel string

const (
	DiagnosticLevelWarning DiagnosticLevel = "warning"
	DiagnosticLevelError   DiagnosticLevel = "error"
)

// UnsupportedBehavior controls how the plugin system handles unsupported config.
type UnsupportedBehavior string

const (
	UnsupportedBehaviorIgnore UnsupportedBehavior = "ignore"
	UnsupportedBehaviorWarn   UnsupportedBehavior = "warn"
	UnsupportedBehaviorError  UnsupportedBehavior = "error"
)

// ConfigPolicy controls how the plugin system handles unknown or unsupported config.
type ConfigPolicy struct {
	UnknownComponent UnsupportedBehavior `json:"unknown_component,omitempty"`
	UnknownField     UnsupportedBehavior `json:"unknown_field,omitempty"`
	UnsupportedValue UnsupportedBehavior `json:"unsupported_value,omitempty"`
}

// ConfigDiagnostic is one validation or compatibility diagnostic.
type ConfigDiagnostic struct {
	Level     DiagnosticLevel `json:"level"`
	Code      string          `json:"code"`
	Component *string         `json:"component,omitempty"`
	Field     *string         `json:"field,omitempty"`
	Message   string          `json:"message"`
}

// ConfigReport is the validation or activation report for a plugin config.
type ConfigReport struct {
	Diagnostics        []ConfigDiagnostic  `json:"diagnostics,omitempty"`
	RuntimeDiagnostics []RuntimeDiagnostic `json:"runtime_diagnostics,omitempty"`
}

// RuntimeDiagnostic is one bounded aggregate of a runtime plugin failure.
type RuntimeDiagnostic struct {
	Code      string  `json:"code"`
	Component string  `json:"component"`
	Field     *string `json:"field,omitempty"`
	Message   string  `json:"message"`
	SessionID *string `json:"session_id,omitempty"`
	Count     uint64  `json:"count"`
}

// PluginComponentSpec is one top-level plugin component.
type PluginComponentSpec struct {
	Kind    string         `json:"kind"`
	Enabled bool           `json:"enabled,omitempty"`
	Config  map[string]any `json:"config,omitempty"`
}

// PluginConfig is the canonical plugin configuration document.
type PluginConfig struct {
	Version    uint32                `json:"version,omitempty"`
	Components []PluginComponentSpec `json:"components,omitempty"`
	Policy     *ConfigPolicy         `json:"policy,omitempty"`
}

// DynamicPluginKind identifies the runtime lane used by a dynamic plugin.
type DynamicPluginKind string

const (
	// DynamicPluginKindRustDynamic loads an ABI-compatible native shared library.
	DynamicPluginKindRustDynamic DynamicPluginKind = "rust_dynamic"
	// DynamicPluginKindWorker starts an isolated worker plugin runtime.
	DynamicPluginKindWorker DynamicPluginKind = "worker"
)

// PluginHostActivation owns the runtime registrations, native libraries, and
// workers created by Initialize. Copies share one activation
// lifetime and may be closed safely from any copy. Failed teardown leaves the
// activation active so a later Close call can retry.
//
// Experimental: this API needs a production consumer before its lifecycle
// contract is considered stable.
type PluginHostActivation struct {
	state *pluginHostActivationState
}

// PluginHostReport contains static and dynamic validation results for one host.
type PluginHostReport struct {
	Config         ConfigReport                    `json:"config"`
	DynamicPlugins []DynamicPluginValidationReport `json:"dynamic_plugins"`
}

// DynamicPluginCheckState is the result of one validation check.
type DynamicPluginCheckState string

const (
	// DynamicPluginCheckStateUnknown indicates that a check was not applicable.
	DynamicPluginCheckStateUnknown DynamicPluginCheckState = "unknown"
	// DynamicPluginCheckStateValid indicates that a check succeeded.
	DynamicPluginCheckStateValid DynamicPluginCheckState = "valid"
	// DynamicPluginCheckStateInvalid indicates that a check failed.
	DynamicPluginCheckStateInvalid DynamicPluginCheckState = "invalid"
)

// DynamicPluginValidationStatus is the canonical status attached to one plugin report.
type DynamicPluginValidationStatus struct {
	Manifest        DynamicPluginCheckState `json:"manifest"`
	Compatibility   DynamicPluginCheckState `json:"compatibility"`
	Integrity       DynamicPluginCheckState `json:"integrity"`
	Environment     DynamicPluginCheckState `json:"environment"`
	Authenticity    DynamicPluginCheckState `json:"authenticity"`
	PolicySatisfied DynamicPluginCheckState `json:"policy_satisfied"`
	CheckedAt       *string                 `json:"checked_at,omitempty"`
	Message         *string                 `json:"message,omitempty"`
}

// DynamicPluginFailure is an actionable dynamic-plugin validation failure.
type DynamicPluginFailure struct {
	Phase   string `json:"phase"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// DynamicPluginValidationReport is the typed core result for one declaration.
type DynamicPluginValidationReport struct {
	PluginID    string                        `json:"plugin_id"`
	ManifestRef string                        `json:"manifest_ref"`
	Kind        DynamicPluginKind             `json:"kind"`
	Status      DynamicPluginValidationStatus `json:"status"`
	Failure     *DynamicPluginFailure         `json:"failure,omitempty"`
	Selected    bool                          `json:"selected"`
}

type pluginHostActivationState struct {
	mu       sync.Mutex
	ptr      unsafe.Pointer
	closed   bool
	closeErr error
}

// PluginContext is the component-scoped registration context passed to plugins.
type PluginContext struct {
	ptr *C.FfiPluginContext
}

// Plugin is the plugin callback contract.
//
// Validate receives one component-local config object and returns diagnostics.
// Register installs middleware and subscribers for one component instance.
type Plugin interface {
	Validate(pluginConfig map[string]any) ([]ConfigDiagnostic, error)
	Register(pluginConfig map[string]any, ctx *PluginContext) error
}

// PluginFuncs adapts plain functions to the Plugin interface.
type PluginFuncs struct {
	ValidateFunc func(pluginConfig map[string]any) ([]ConfigDiagnostic, error)
	RegisterFunc func(pluginConfig map[string]any, ctx *PluginContext) error
}

// Validate delegates to ValidateFunc when provided.
func (h PluginFuncs) Validate(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
	if h.ValidateFunc == nil {
		return nil, nil
	}
	return h.ValidateFunc(pluginConfig)
}

// Register delegates to RegisterFunc when provided.
func (h PluginFuncs) Register(pluginConfig map[string]any, ctx *PluginContext) error {
	if h.RegisterFunc == nil {
		return nil
	}
	return h.RegisterFunc(pluginConfig, ctx)
}

// NewPluginConfig returns a default plugin config with version 1.
func NewPluginConfig() PluginConfig {
	return PluginConfig{
		Version:    1,
		Components: []PluginComponentSpec{},
	}
}

// NewPluginComponent returns an enabled top-level component with empty config.
func NewPluginComponent(kind string) PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    kind,
		Enabled: true,
		Config:  map[string]any{},
	}
}

// marshalPluginHostActivationConfig serializes the activation-only wire shape.
// It keeps presence handling private so the established public Go config
// structs and their encoding method sets remain unchanged. Component enabled
// values are explicit on this wire because Relay defaults an omitted value to
// true, while the Go field's zero value is false.
func marshalPluginHostActivationConfig(config PluginConfig) ([]byte, error) {
	type componentJSON struct {
		Kind    string         `json:"kind"`
		Enabled bool           `json:"enabled"`
		Config  map[string]any `json:"config,omitempty"`
	}
	type configJSON struct {
		Version    uint32          `json:"version,omitempty"`
		Components []componentJSON `json:"components,omitempty"`
		Policy     *ConfigPolicy   `json:"policy,omitempty"`
	}

	components := make([]componentJSON, len(config.Components))
	for i, component := range config.Components {
		components[i] = componentJSON{
			Kind:    component.Kind,
			Enabled: component.Enabled,
			Config:  component.Config,
		}
	}
	return jsonMarshal(configJSON{
		Version:    config.Version,
		Components: components,
		Policy:     config.Policy,
	})
}

// Initialize activates the core-owned static and dynamic plugin host.
// Programmatic config is lowest precedence. An optional explicit file replaces
// user-file discovery, and the system file overlays either source.
func Initialize(config PluginConfig, additionalPluginsTOML *string) (*PluginHostActivation, PluginHostReport, error) {
	configPayload, err := marshalPluginHostActivationConfig(config)
	if err != nil {
		return nil, PluginHostReport{}, err
	}
	ptr, rawReport, err := initializePluginHostJSON(string(configPayload), additionalPluginsTOML)
	if err != nil {
		return nil, PluginHostReport{}, err
	}
	activation := newPluginHostActivation(ptr)
	var report PluginHostReport
	if err := jsonUnmarshal([]byte(rawReport), &report); err != nil {
		_ = activation.Close()
		return nil, PluginHostReport{}, err
	}
	return activation, report, nil
}

// Validate validates the same layered plugin-host configuration as Initialize
// without loading code or acquiring the process-wide host lease.
func Validate(config PluginConfig, additionalPluginsTOML *string) (PluginHostReport, error) {
	payload, err := marshalPluginHostActivationConfig(config)
	if err != nil {
		return PluginHostReport{}, err
	}
	raw, err := validatePluginHostJSON(string(payload), additionalPluginsTOML)
	if err != nil {
		return PluginHostReport{}, err
	}
	var report PluginHostReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return PluginHostReport{}, err
	}
	return report, nil
}

// ValidateExact validates only the supplied static plugin configuration.
// Unlike Validate, it does not discover or merge plugins.toml files.
func ValidateExact(config PluginConfig) (PluginHostReport, error) {
	payload, err := marshalPluginHostActivationConfig(config)
	if err != nil {
		return PluginHostReport{}, err
	}
	raw, err := validateExactPluginHostJSON(string(payload))
	if err != nil {
		return PluginHostReport{}, err
	}
	var report PluginHostReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return PluginHostReport{}, err
	}
	return report, nil
}

// Report returns the immutable core-owned activation report.
func (activation *PluginHostActivation) Report() (PluginHostReport, error) {
	if activation == nil || activation.state == nil {
		return PluginHostReport{}, errors.New("plugin host activation is nil")
	}
	activation.state.mu.Lock()
	defer activation.state.mu.Unlock()
	if activation.state.ptr == nil || activation.state.closed {
		return PluginHostReport{}, errors.New("plugin host activation is closed")
	}
	raw, err := pluginHostActivationReportJSON(activation.state.ptr)
	if err != nil {
		return PluginHostReport{}, err
	}
	var report PluginHostReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return PluginHostReport{}, err
	}
	return report, nil
}

// IsActive reports whether the activation still owns the process-wide plugin host.
func (activation *PluginHostActivation) IsActive() bool {
	if activation == nil || activation.state == nil {
		return false
	}
	activation.state.mu.Lock()
	defer activation.state.mu.Unlock()
	if activation.state.ptr == nil || activation.state.closed {
		return false
	}
	active, err := pluginHostActivationIsActive(activation.state.ptr)
	return err == nil && active
}

func newPluginHostActivation(ptr unsafe.Pointer) *PluginHostActivation {
	state := &pluginHostActivationState{ptr: ptr}
	runtime.SetFinalizer(state, finalizePluginHostActivation)
	return &PluginHostActivation{state: state}
}

var reportPluginHostActivationCleanupError = func(err error) {
	log.Printf("nemo_relay: plugin host activation cleanup failed during finalization: %v", err)
}

func finalizePluginHostActivation(state *pluginHostActivationState) {
	go func() {
		if err := state.close(); err != nil {
			reportPluginHostActivationCleanupError(err)
		}
	}()
}

// Close removes callbacks and subscribers before unloading plugin libraries
// and workers. It is safe to call Close repeatedly or on a nil activation.
func (activation *PluginHostActivation) Close() error {
	if activation == nil || activation.state == nil {
		return nil
	}
	return activation.state.close()
}

func (state *pluginHostActivationState) close() error {
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.closed {
		return state.closeErr
	}

	ptr := state.ptr
	if ptr == nil {
		state.closed = true
		return nil
	}
	state.closeErr = clearPluginHostActivation(ptr)
	if state.closeErr != nil {
		active, err := pluginHostActivationIsActive(ptr)
		if err != nil || active {
			return state.closeErr
		}
	}

	state.closed = true
	state.ptr = nil
	runtime.SetFinalizer(state, nil)
	freePluginHostActivation(ptr)
	return state.closeErr
}

// ListPluginKinds lists plugin kinds registered with the registry.
func ListPluginKinds() ([]string, error) {
	raw, err := listPluginKindsJSON()
	if err != nil {
		return nil, err
	}
	var kinds []string
	if err := jsonUnmarshal([]byte(raw), &kinds); err != nil {
		return nil, err
	}
	return kinds, nil
}

// RegisterPlugin registers a plugin kind for later validation and initialization.
//
// Registering the same kind twice returns an error.
func RegisterPlugin(pluginKind string, plugin Plugin) error {
	cPluginKind := C.CString(pluginKind)
	defer C.free(unsafe.Pointer(cPluginKind))
	userData := registerClosure(plugin)
	status := C.nemo_relay_register_plugin(
		cPluginKind,
		(C.NemoRelayPluginValidateCb)(C.goPluginValidateTrampoline),
		(C.NemoRelayPluginRegisterCb)(C.goPluginRegisterTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	)
	return checkStatus(status)
}

// DeregisterPlugin removes a previously registered plugin kind.
//
// This affects future validation and initialization only. Active runtime
// registrations remain until the owning PluginHostActivation closes.
func DeregisterPlugin(pluginKind string) error {
	cPluginKind := C.CString(pluginKind)
	defer C.free(unsafe.Pointer(cPluginKind))
	return checkStatus(C.nemo_relay_deregister_plugin(cPluginKind))
}

// RegisterSubscriber registers an infallible event subscriber for this
// component. The callback receives an owned [Event] snapshot that is safe to
// retain after the callback returns.
func (ctx *PluginContext) RegisterSubscriber(name string, fn EventSubscriberFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_subscriber(
		ctx.ptr,
		cName,
		(C.NemoRelayEventSubscriberFn)(C.goEventSubscriberTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterEventMetadataInjector registers an event metadata injector for this
// component. Relay qualifies its name and removes it when plugin configuration
// is cleared or registration rolls back.
func (ctx *PluginContext) RegisterEventMetadataInjector(name string, priority int32, fn EventMetadataInjectorFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	if fn == nil {
		return errEventMetadataInjectorCallbackNil
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_event_metadata_injector(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayEventMetadataInjectorFn)(C.goEventMetadataInjectorTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

func (ctx *PluginContext) registerEventSanitizer(name string, priority int32, fn EventSanitizeFunc, surface int) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	callback := C.NemoRelayEventSanitizeFn(C.goEventSanitizeTrampoline)
	free := C.NemoRelayFreeFn(C.goFreeTrampoline)
	var status C.int32_t
	switch surface {
	case 0:
		status = C.nemo_relay_plugin_context_register_mark_sanitize_guardrail(ctx.ptr, cName, C.int32_t(priority), callback, userData, free)
	case 1:
		status = C.nemo_relay_plugin_context_register_scope_sanitize_start_guardrail(ctx.ptr, cName, C.int32_t(priority), callback, userData, free)
	default:
		status = C.nemo_relay_plugin_context_register_scope_sanitize_end_guardrail(ctx.ptr, cName, C.int32_t(priority), callback, userData, free)
	}
	return checkStatus(status)
}

// RegisterMarkSanitizeGuardrail registers a mark event sanitizer for this component.
func (ctx *PluginContext) RegisterMarkSanitizeGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return ctx.registerEventSanitizer(name, priority, fn, 0)
}

// RegisterScopeSanitizeStartGuardrail registers a scope-start sanitizer for this component.
func (ctx *PluginContext) RegisterScopeSanitizeStartGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return ctx.registerEventSanitizer(name, priority, fn, 1)
}

// RegisterScopeSanitizeEndGuardrail registers a scope-end sanitizer for this component.
func (ctx *PluginContext) RegisterScopeSanitizeEndGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return ctx.registerEventSanitizer(name, priority, fn, 2)
}

// RegisterToolSanitizeRequestGuardrail registers a tool sanitize-request guardrail for this component.
func (ctx *PluginContext) RegisterToolSanitizeRequestGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayToolSanitizeFn)(C.goToolSanitizeTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolSanitizeResponseGuardrail registers a tool sanitize-response guardrail for this component.
func (ctx *PluginContext) RegisterToolSanitizeResponseGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayToolSanitizeFn)(C.goToolSanitizeTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolConditionalExecutionGuardrail registers a tool conditional-execution guardrail for this component.
func (ctx *PluginContext) RegisterToolConditionalExecutionGuardrail(name string, priority int32, fn ToolConditionalFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayToolConditionalFn)(C.goToolConditionalTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmSanitizeRequestGuardrail registers an LLM sanitize-request guardrail for this component.
func (ctx *PluginContext) RegisterLlmSanitizeRequestGuardrail(name string, priority int32, fn LLMRequestFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayLlmSanitizeRequestCb)(C.goLlmRequestTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmSanitizeResponseGuardrail registers an LLM sanitize-response guardrail for this component.
func (ctx *PluginContext) RegisterLlmSanitizeResponseGuardrail(name string, priority int32, fn LLMResponseFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayLlmSanitizeResponseCb)(C.goLlmResponseTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmConditionalExecutionGuardrail registers an LLM conditional-execution guardrail for this component.
func (ctx *PluginContext) RegisterLlmConditionalExecutionGuardrail(name string, priority int32, fn LLMConditionalFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayLlmConditionalCb)(C.goLlmConditionalTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmRequestIntercept registers an LLM request intercept for this component.
//
// Lower priorities run first. When breakChain is true, later request
// intercepts in the chain are skipped after this callback runs.
func (ctx *PluginContext) RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NemoRelayLlmRequestInterceptCb)(C.goLlmRequestInterceptTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolRequestIntercept registers a tool request intercept for this component.
//
// Lower priorities run first. When breakChain is true, later request
// intercepts in the chain are skipped after this callback runs.
func (ctx *PluginContext) RegisterToolRequestIntercept(name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_tool_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NemoRelayToolSanitizeFn)(C.goToolSanitizeTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmExecutionIntercept registers an LLM execution intercept for this component.
func (ctx *PluginContext) RegisterLlmExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmStreamExecutionIntercept registers a streaming LLM execution intercept for this component.
func (ctx *PluginContext) RegisterLlmStreamExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_llm_stream_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolExecutionIntercept registers a tool execution intercept for this component.
func (ctx *PluginContext) RegisterToolExecutionIntercept(name string, priority int32, fn ToolExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New(errPluginContextClosed)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_relay_plugin_context_register_tool_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoRelayToolExecInterceptCb)(C.goToolExecInterceptTrampoline),
		userData,
		(C.NemoRelayFreeFn)(C.goFreeTrampoline),
	))
}

func pluginConfigCString(config PluginConfig) (*C.char, error) {
	payload, err := jsonMarshal(config)
	if err != nil {
		return nil, err
	}
	return C.CString(string(payload)), nil
}
