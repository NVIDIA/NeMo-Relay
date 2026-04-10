// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiPluginContext FfiPluginContext;

typedef void (*NemoFlowFreeFn)(void* user_data);
typedef char* (*NemoFlowPluginValidateCb)(void* user_data, const char* plugin_config_json);
typedef int32_t (*NemoFlowPluginRegisterCb)(void* user_data, const char* plugin_config_json, FfiPluginContext* ctx);
typedef void (*NemoFlowEventSubscriberFn)(void* user_data, const void* event);
typedef char* (*NemoFlowToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef int32_t (*NemoFlowLlmRequestInterceptCb)(void* user_data, const char* name, const void* request, const char* annotated_json, void** out_request, char** out_annotated_json);
typedef char* (*NemoFlowLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NemoFlowLlmExecInterceptCb)(void* user_data, const char* native_json, NemoFlowLlmExecNextFn next_fn, void* next_ctx);
typedef char* (*NemoFlowToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NemoFlowToolExecInterceptCb)(void* user_data, const char* args_json, NemoFlowToolExecNextFn next_fn, void* next_ctx);

extern int32_t nemo_flow_validate_plugin_config(const char* config_json, char** out_json);
extern int32_t nemo_flow_initialize_plugins(const char* config_json, char** out_json);
extern int32_t nemo_flow_clear_plugin_configuration(void);
extern int32_t nemo_flow_active_plugin_report_json(char** out_json);
extern int32_t nemo_flow_list_plugin_kinds_json(char** out_json);
extern int32_t nemo_flow_register_plugin(const char* plugin_kind, NemoFlowPluginValidateCb validate_cb, NemoFlowPluginRegisterCb register_cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_deregister_plugin(const char* plugin_kind);
extern void nemo_flow_string_free(char* ptr);

extern int32_t nemo_flow_plugin_context_register_subscriber(FfiPluginContext* ctx, const char* name, NemoFlowEventSubscriberFn cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_plugin_context_register_llm_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoFlowLlmRequestInterceptCb cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_plugin_context_register_tool_request_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NemoFlowToolSanitizeFn cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_plugin_context_register_llm_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoFlowLlmExecInterceptCb cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_plugin_context_register_llm_stream_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoFlowLlmExecInterceptCb cb, void* user_data, NemoFlowFreeFn free_fn);
extern int32_t nemo_flow_plugin_context_register_tool_execution_intercept(FfiPluginContext* ctx, const char* name, int32_t priority, NemoFlowToolExecInterceptCb cb, void* user_data, NemoFlowFreeFn free_fn);

extern char* goPluginValidateTrampoline(void*, const char*);
extern int32_t goPluginRegisterTrampoline(void*, const char*, FfiPluginContext*);
extern void goEventSubscriberTrampoline(void*, const void*);
extern void goFreeTrampoline(void*);
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NemoFlowLlmExecNextFn, void*);
extern int32_t goLlmRequestInterceptTrampoline(void*, const char*, const void*, const char*, void**, char**);
extern char* goToolExecInterceptTrampoline(void*, const char*, NemoFlowToolExecNextFn, void*);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"unsafe"
)

// DiagnosticLevel is the severity level for one plugin-host diagnostic.
type DiagnosticLevel string

const (
	DiagnosticLevelWarning DiagnosticLevel = "warning"
	DiagnosticLevelError   DiagnosticLevel = "error"
)

// UnsupportedBehavior controls how the plugin host handles unsupported config.
type UnsupportedBehavior string

const (
	UnsupportedBehaviorIgnore UnsupportedBehavior = "ignore"
	UnsupportedBehaviorWarn   UnsupportedBehavior = "warn"
	UnsupportedBehaviorError  UnsupportedBehavior = "error"
)

// ConfigPolicy controls how the plugin host handles unknown or unsupported config.
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

// PluginComponentSpec is one top-level hosted plugin component.
type PluginComponentSpec struct {
	Kind    string         `json:"kind"`
	Enabled bool           `json:"enabled,omitempty"`
	Config  map[string]any `json:"config,omitempty"`
}

// PluginConfig is the canonical plugin-host configuration document.
type PluginConfig struct {
	Version    uint32                `json:"version,omitempty"`
	Components []PluginComponentSpec `json:"components,omitempty"`
	Policy     *ConfigPolicy         `json:"policy,omitempty"`
}

// PluginContext is the component-scoped registration context passed to hosted plugins.
type PluginContext struct {
	ptr *C.FfiPluginContext
}

// Plugin is the hosted plugin callback contract.
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

// NewPluginConfig returns a default plugin-host config with version 1.
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

// ValidatePluginConfig validates a plugin-host config without changing runtime state.
//
// It returns the validation report or an error if the config could not be
// serialized for the FFI boundary.
func ValidatePluginConfig(config PluginConfig) (ConfigReport, error) {
	cConfig, err := pluginConfigCString(config)
	if err != nil {
		return ConfigReport{}, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var out *C.char
	status := C.nemo_flow_validate_plugin_config(cConfig, &out)
	if err := checkStatus(status); err != nil {
		return ConfigReport{}, err
	}
	defer C.nemo_flow_string_free(out)

	var report ConfigReport
	if err := json.Unmarshal([]byte(C.GoString(out)), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// InitializePlugins validates and activates a plugin-host config.
//
// The returned report describes the successfully activated configuration.
// Initialization replaces the current active config and rolls back partial
// registration on failure.
func InitializePlugins(config PluginConfig) (ConfigReport, error) {
	cConfig, err := pluginConfigCString(config)
	if err != nil {
		return ConfigReport{}, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var out *C.char
	status := C.nemo_flow_initialize_plugins(cConfig, &out)
	if err := checkStatus(status); err != nil {
		return ConfigReport{}, err
	}
	defer C.nemo_flow_string_free(out)

	var report ConfigReport
	if err := json.Unmarshal([]byte(C.GoString(out)), &report); err != nil {
		return ConfigReport{}, err
	}
	return report, nil
}

// ClearPluginConfiguration removes all active plugin component registrations.
//
// Registered plugin kinds remain available for future validation or
// initialization.
func ClearPluginConfiguration() error {
	return checkStatus(C.nemo_flow_clear_plugin_configuration())
}

// ActivePluginReport returns the last successfully activated plugin report.
//
// A nil report means no plugin configuration is currently active.
func ActivePluginReport() (*ConfigReport, error) {
	var out *C.char
	status := C.nemo_flow_active_plugin_report_json(&out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nemo_flow_string_free(out)

	if C.GoString(out) == "null" {
		return nil, nil
	}
	var report ConfigReport
	if err := json.Unmarshal([]byte(C.GoString(out)), &report); err != nil {
		return nil, err
	}
	return &report, nil
}

// ListPluginKinds lists plugin kinds registered with the registry.
func ListPluginKinds() ([]string, error) {
	var out *C.char
	status := C.nemo_flow_list_plugin_kinds_json(&out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nemo_flow_string_free(out)

	var kinds []string
	if err := json.Unmarshal([]byte(C.GoString(out)), &kinds); err != nil {
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
	status := C.nemo_flow_register_plugin(
		cPluginKind,
		(C.NemoFlowPluginValidateCb)(C.goPluginValidateTrampoline),
		(C.NemoFlowPluginRegisterCb)(C.goPluginRegisterTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
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
	return checkStatus(C.nemo_flow_deregister_plugin(cPluginKind))
}

// RegisterSubscriber registers an infallible event subscriber for this component.
func (ctx *PluginContext) RegisterSubscriber(name string, fn EventSubscriberFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_subscriber(
		ctx.ptr,
		cName,
		(C.NemoFlowEventSubscriberFn)(C.goEventSubscriberTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmRequestIntercept registers an LLM request intercept for this component.
//
// Lower priorities run first. When breakChain is true, later request
// intercepts in the chain are skipped after this callback runs.
func (ctx *PluginContext) RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_llm_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NemoFlowLlmRequestInterceptCb)(C.goLlmRequestInterceptTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolRequestIntercept registers a tool request intercept for this component.
//
// Lower priorities run first. When breakChain is true, later request
// intercepts in the chain are skipped after this callback runs.
func (ctx *PluginContext) RegisterToolRequestIntercept(name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_tool_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NemoFlowToolSanitizeFn)(C.goToolSanitizeTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmExecutionIntercept registers an LLM execution intercept for this component.
func (ctx *PluginContext) RegisterLlmExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_llm_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoFlowLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmStreamExecutionIntercept registers a streaming LLM execution intercept for this component.
func (ctx *PluginContext) RegisterLlmStreamExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_llm_stream_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoFlowLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolExecutionIntercept registers a tool execution intercept for this component.
func (ctx *PluginContext) RegisterToolExecutionIntercept(name string, priority int32, fn ToolExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nemo_flow_plugin_context_register_tool_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NemoFlowToolExecInterceptCb)(C.goToolExecInterceptTrampoline),
		userData,
		(C.NemoFlowFreeFn)(C.goFreeTrampoline),
	))
}

func pluginConfigCString(config PluginConfig) (*C.char, error) {
	payload, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}
	return C.CString(string(payload)), nil
}
