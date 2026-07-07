// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiPluginContext FfiPluginContext;

typedef void (*NemoRelayFreeFn)(void* user_data);
typedef char* (*NemoRelayPluginValidateCb)(void* user_data, const char* plugin_config_json);
typedef int32_t (*NemoRelayPluginRegisterCb)(void* user_data, const char* plugin_config_json, FfiPluginContext* ctx);
typedef void (*NemoRelayEventSubscriberFn)(void* user_data, const void* event);
typedef char* (*NemoRelayToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef void* (*NemoRelayLlmRequestCb)(void* user_data, const void* request);
typedef char* (*NemoRelayLlmResponseFn)(void* user_data, const char* response_json);
typedef char* (*NemoRelayLlmConditionalCb)(void* user_data, const void* request);
typedef int32_t (*NemoRelayLlmRequestInterceptCb)(void* user_data, const char* name, const void* request, const char* annotated_json, char** out_outcome_json);
typedef char* (*NemoRelayLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NemoRelayLlmExecInterceptCb)(void* user_data, const char* native_json, NemoRelayLlmExecNextFn next_fn, void* next_ctx);
typedef char* (*NemoRelayToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NemoRelayToolExecInterceptCb)(void* user_data, const char* args_json, NemoRelayToolExecNextFn next_fn, void* next_ctx);

extern int32_t nemo_relay_validate_plugin_config(const char* config_json, char** out_json);
extern int32_t nemo_relay_initialize_plugins(const char* config_json, char** out_json);
extern int32_t nemo_relay_clear_plugin_configuration(void);
extern int32_t nemo_relay_active_plugin_report_json(char** out_json);
extern int32_t nemo_relay_list_plugin_kinds_json(char** out_json);
extern int32_t nemo_relay_register_plugin(const char* plugin_kind, NemoRelayPluginValidateCb validate_cb, NemoRelayPluginRegisterCb register_cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_plugin(const char* plugin_kind);
extern void nemo_relay_string_free(char* ptr);

extern int32_t nemo_relay_plugin_context_register_subscriber(FfiPluginContext* ctx, const char* name, NemoRelayEventSubscriberFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolConditionalFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmRequestCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmResponseFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmConditionalCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoRelayLlmRequestInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_llm_stream_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_plugin_context_register_tool_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoRelayToolExecInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);

extern char* goPluginValidateTrampoline(void*, const char*);
extern int32_t goPluginRegisterTrampoline(void*, const char*, FfiPluginContext*);
extern void goEventSubscriberTrampoline(void*, const void*);
extern void goFreeTrampoline(void*);
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern void* goLlmRequestTrampoline(void*, const void*);
extern char* goLlmResponseTrampoline(void*, const char*);
extern char* goLlmConditionalTrampoline(void*, const void*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NemoRelayLlmExecNextFn, void*);
extern int32_t goLlmRequestInterceptTrampoline(void*, const char*, const void*, const char*, char**);
extern char* goToolExecInterceptTrampoline(void*, const char*, NemoRelayToolExecNextFn, void*);
*/
import "C"

import (
	"errors"
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
	validatePluginConfigJSON = func(config PluginConfig) (string, error) {
		cConfig, err := pluginConfigCString(config)
		if err != nil {
			return "", err
		}
		defer C.free(unsafe.Pointer(cConfig))

		var out *C.char
		status := C.nemo_relay_validate_plugin_config(cConfig, &out)
		return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
			C.nemo_relay_string_free(out)
		})
	}
	initializePluginsJSON = func(config PluginConfig) (string, error) {
		cConfig, err := pluginConfigCString(config)
		if err != nil {
			return "", err
		}
		defer C.free(unsafe.Pointer(cConfig))

		var out *C.char
		status := C.nemo_relay_initialize_plugins(cConfig, &out)
		return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
			C.nemo_relay_string_free(out)
		})
	}
	activePluginReportJSON = func() (string, error) {
		var out *C.char
		status := C.nemo_relay_active_plugin_report_json(&out)
		return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
			C.nemo_relay_string_free(out)
		})
	}
	listPluginKindsJSON = func() (string, error) {
		var out *C.char
		status := C.nemo_relay_list_plugin_kinds_json(&out)
		return checkedJSONString(int32(status), func() string { return C.GoString(out) }, func() {
			C.nemo_relay_string_free(out)
		})
	}
)

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
	Diagnostics []ConfigDiagnostic `json:"diagnostics,omitempty"`
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

// ValidatePluginConfig validates a plugin config without changing runtime state.
//
// It returns the validation report or an error if the config could not be
// serialized for the FFI boundary.
func ValidatePluginConfig(config PluginConfig) (ConfigReport, error) {
	raw, err := validatePluginConfigJSON(config)
	if err != nil {
		return ConfigReport{}, err
	}
	var report ConfigReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// InitializePlugins validates and activates a plugin config.
//
// The returned report describes the successfully activated configuration.
// Initialization replaces the current active config and rolls back partial
// registration on failure.
func InitializePlugins(config PluginConfig) (ConfigReport, error) {
	raw, err := initializePluginsJSON(config)
	if err != nil {
		return ConfigReport{}, err
	}
	var report ConfigReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// ClearPluginConfiguration removes all active plugin component registrations.
//
// Registered plugin kinds remain available for future validation or
// initialization.
func ClearPluginConfiguration() error {
	return checkStatus(C.nemo_relay_clear_plugin_configuration())
}

// ActivePluginReport returns the last successfully activated plugin report.
//
// A nil report means no plugin configuration is currently active.
func ActivePluginReport() (*ConfigReport, error) {
	raw, err := activePluginReportJSON()
	if err != nil {
		return nil, err
	}
	if raw == "null" {
		return nil, nil
	}
	var report ConfigReport
	if err := jsonUnmarshal([]byte(raw), &report); err != nil {
		return nil, err
	}
	return &report, nil
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
// registrations remain until cleared or replaced.
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
		(C.NemoRelayLlmRequestCb)(C.goLlmRequestTrampoline),
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
		(C.NemoRelayLlmResponseFn)(C.goLlmResponseTrampoline),
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
