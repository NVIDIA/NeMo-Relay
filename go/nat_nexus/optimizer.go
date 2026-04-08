// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiOptimizerRuntime FfiOptimizerRuntime;
typedef struct FfiOptimizerPluginContext FfiOptimizerPluginContext;

extern int32_t nat_nexus_validate_optimizer_config(const char* config_json, char** out_json);
extern int32_t nat_nexus_optimizer_runtime_create(const char* config_json, FfiOptimizerRuntime** out);
extern int32_t nat_nexus_optimizer_runtime_register(FfiOptimizerRuntime* runtime);
extern int32_t nat_nexus_optimizer_runtime_deregister(FfiOptimizerRuntime* runtime);
extern int32_t nat_nexus_optimizer_runtime_shutdown(FfiOptimizerRuntime* runtime);
extern int32_t nat_nexus_optimizer_runtime_report_json(const FfiOptimizerRuntime* runtime, char** out_json);
extern void nat_nexus_optimizer_runtime_free(FfiOptimizerRuntime* ptr);
extern void nat_nexus_string_free(char* ptr);

typedef void (*NatNexusFreeFn)(void* user_data);
typedef char* (*NatNexusOptimizerPluginValidateCb)(void* user_data, const char* instance_id, const char* plugin_config_json);
typedef int32_t (*NatNexusOptimizerPluginRegisterCb)(void* user_data, const char* instance_id, const char* plugin_config_json, FfiOptimizerPluginContext* ctx);
typedef void (*NatNexusEventSubscriberFn)(void* user_data, const void* event);
typedef char* (*NatNexusToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef int32_t (*NatNexusLlmRequestInterceptCb)(void* user_data, const char* name, const void* request, const char* annotated_json, void** out_request, char** out_annotated_json);
typedef char* (*NatNexusLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NatNexusLlmExecInterceptCb)(void* user_data, const char* native_json, NatNexusLlmExecNextFn next_fn, void* next_ctx);
typedef char* (*NatNexusToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NatNexusToolExecInterceptCb)(void* user_data, const char* args_json, NatNexusToolExecNextFn next_fn, void* next_ctx);

extern int32_t nat_nexus_optimizer_register_plugin(const char* plugin_kind, NatNexusOptimizerPluginValidateCb validate_cb, NatNexusOptimizerPluginRegisterCb register_cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_deregister_plugin(const char* plugin_kind);
extern int32_t nat_nexus_optimizer_plugin_context_register_subscriber(FfiOptimizerPluginContext* ctx, const char* name, NatNexusEventSubscriberFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_plugin_context_register_llm_request_intercept(FfiOptimizerPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NatNexusLlmRequestInterceptCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_plugin_context_register_tool_request_intercept(FfiOptimizerPluginContext* ctx, const char* name, int32_t priority, _Bool break_chain, NatNexusToolSanitizeFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_plugin_context_register_llm_execution_intercept(FfiOptimizerPluginContext* ctx, const char* name, int32_t priority, NatNexusLlmExecInterceptCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_plugin_context_register_llm_stream_execution_intercept(FfiOptimizerPluginContext* ctx, const char* name, int32_t priority, NatNexusLlmExecInterceptCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_optimizer_plugin_context_register_tool_execution_intercept(FfiOptimizerPluginContext* ctx, const char* name, int32_t priority, NatNexusToolExecInterceptCb cb, void* user_data, NatNexusFreeFn free_fn);

extern char* goOptimizerPluginValidateTrampoline(void*, const char*, const char*);
extern int32_t goOptimizerPluginRegisterTrampoline(void*, const char*, const char*, FfiOptimizerPluginContext*);
extern void goEventSubscriberTrampoline(void*, const void*);
extern void goFreeTrampoline(void*);
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NatNexusLlmExecNextFn, void*);
extern int32_t goLlmRequestInterceptTrampoline(void*, const char*, const void*, const char*, void**, char**);
extern char* goToolExecInterceptTrampoline(void*, const char*, NatNexusToolExecNextFn, void*);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"unsafe"
)

// UnsupportedBehavior controls how optimizer config validation handles unknown or unsupported input.
type UnsupportedBehavior string

const (
	UnsupportedBehaviorIgnore UnsupportedBehavior = "ignore"
	UnsupportedBehaviorWarn   UnsupportedBehavior = "warn"
	UnsupportedBehaviorError  UnsupportedBehavior = "error"
)

// OptimizerDiagnosticLevel is the severity level returned by config validation.
type OptimizerDiagnosticLevel string

const (
	OptimizerDiagnosticLevelWarning OptimizerDiagnosticLevel = "warning"
	OptimizerDiagnosticLevelError   OptimizerDiagnosticLevel = "error"
)

// OptimizerConfig is the canonical Go shape for the optimizer runtime config document.
type OptimizerConfig struct {
	Version    uint32                   `json:"version,omitempty"`
	AgentID    string                   `json:"agent_id,omitempty"`
	State      *OptimizerStateConfig    `json:"state,omitempty"`
	Components []OptimizerComponentSpec `json:"components,omitempty"`
	Policy     *OptimizerConfigPolicy   `json:"policy,omitempty"`
}

// OptimizerStateConfig configures shared optimizer state used by stateful components.
type OptimizerStateConfig struct {
	Backend OptimizerBackendSpec `json:"backend"`
}

// OptimizerBackendSpec dynamically selects the runtime backend.
type OptimizerBackendSpec struct {
	Kind   string         `json:"kind"`
	Config map[string]any `json:"config,omitempty"`
}

// OptimizerComponentSpec dynamically selects a built-in optimizer component.
type OptimizerComponentSpec struct {
	Kind    string         `json:"kind"`
	Enabled bool           `json:"enabled,omitempty"`
	Config  map[string]any `json:"config,omitempty"`
}

// OptimizerConfigPolicy controls how compatibility diagnostics are handled.
type OptimizerConfigPolicy struct {
	UnknownComponent UnsupportedBehavior `json:"unknown_component,omitempty"`
	UnknownField     UnsupportedBehavior `json:"unknown_field,omitempty"`
	UnsupportedValue UnsupportedBehavior `json:"unsupported_value,omitempty"`
}

// OptimizerConfigReport is returned by config validation and runtime creation.
type OptimizerConfigReport struct {
	Diagnostics []OptimizerConfigDiagnostic `json:"diagnostics,omitempty"`
}

// OptimizerConfigDiagnostic describes one config warning or error.
type OptimizerConfigDiagnostic struct {
	Level     OptimizerDiagnosticLevel `json:"level"`
	Code      string                   `json:"code"`
	Component *string                  `json:"component,omitempty"`
	Field     *string                  `json:"field,omitempty"`
	Message   string                   `json:"message"`
}

// TelemetryComponentConfig is the typed helper config for the telemetry component.
type TelemetryComponentConfig struct {
	SubscriberName string   `json:"subscriber_name,omitempty"`
	Learners       []string `json:"learners,omitempty"`
}

// DynamoHintsComponentConfig is the typed helper config for the dynamo hints component.
type DynamoHintsComponentConfig struct {
	Priority       int32  `json:"priority"`
	BreakChain     bool   `json:"break_chain,omitempty"`
	InjectHeader   bool   `json:"inject_header"`
	InjectBodyPath string `json:"inject_body_path"`
}

// ToolParallelismComponentConfig is the typed helper config for the tool parallelism component.
type ToolParallelismComponentConfig struct {
	Priority int32  `json:"priority"`
	Mode     string `json:"mode"`
}

// ExternalComponentConfig is the typed helper config for hosted external components.
type ExternalComponentConfig struct {
	PluginKind   string         `json:"plugin_kind"`
	InstanceID   string         `json:"instance_id"`
	PluginConfig map[string]any `json:"plugin_config,omitempty"`
}

// OptimizerRuntime is the Go wrapper for the native optimizer runtime.
type OptimizerRuntime struct {
	ptr *C.FfiOptimizerRuntime
}

// OptimizerPluginContext is only valid during hosted plugin registration.
type OptimizerPluginContext struct {
	ptr *C.FfiOptimizerPluginContext
}

// OptimizerPluginHandler handles hosted optimizer plugin validation and registration.
type OptimizerPluginHandler interface {
	Validate(instanceID string, pluginConfig map[string]any) ([]OptimizerConfigDiagnostic, error)
	Register(instanceID string, pluginConfig map[string]any, ctx *OptimizerPluginContext) error
}

// OptimizerPluginHandlerFuncs is a convenience implementation for function-based plugins.
type OptimizerPluginHandlerFuncs struct {
	ValidateFunc func(instanceID string, pluginConfig map[string]any) ([]OptimizerConfigDiagnostic, error)
	RegisterFunc func(instanceID string, pluginConfig map[string]any, ctx *OptimizerPluginContext) error
}

func (h OptimizerPluginHandlerFuncs) Validate(instanceID string, pluginConfig map[string]any) ([]OptimizerConfigDiagnostic, error) {
	if h.ValidateFunc == nil {
		return nil, nil
	}
	return h.ValidateFunc(instanceID, pluginConfig)
}

func (h OptimizerPluginHandlerFuncs) Register(instanceID string, pluginConfig map[string]any, ctx *OptimizerPluginContext) error {
	if h.RegisterFunc == nil {
		return nil
	}
	return h.RegisterFunc(instanceID, pluginConfig, ctx)
}

// NewOptimizerConfig returns a config with the current document version set.
func NewOptimizerConfig() OptimizerConfig {
	return OptimizerConfig{
		Version:    1,
		Components: []OptimizerComponentSpec{},
	}
}

// NewInMemoryOptimizerBackend returns the built-in in-memory backend spec.
func NewInMemoryOptimizerBackend() OptimizerBackendSpec {
	return OptimizerBackendSpec{
		Kind:   "in_memory",
		Config: map[string]any{},
	}
}

// NewRedisOptimizerBackend returns the Redis backend spec.
func NewRedisOptimizerBackend(url, keyPrefix string) OptimizerBackendSpec {
	return OptimizerBackendSpec{
		Kind: "redis",
		Config: map[string]any{
			"url":        url,
			"key_prefix": keyPrefix,
		},
	}
}

// NewTelemetryComponentConfig returns the default telemetry component config.
func NewTelemetryComponentConfig() TelemetryComponentConfig {
	return TelemetryComponentConfig{}
}

// NewDynamoHintsComponentConfig returns the default dynamo hints config.
func NewDynamoHintsComponentConfig() DynamoHintsComponentConfig {
	return DynamoHintsComponentConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}
}

// NewToolParallelismComponentConfig returns the default tool parallelism config.
func NewToolParallelismComponentConfig() ToolParallelismComponentConfig {
	return ToolParallelismComponentConfig{
		Priority: 100,
		Mode:     "observe_only",
	}
}

// TelemetryComponent converts the typed telemetry config to the canonical dynamic component spec.
func TelemetryComponent(config TelemetryComponentConfig) OptimizerComponentSpec {
	return OptimizerComponentSpec{
		Kind:    "telemetry",
		Enabled: true,
		Config:  mustConfigMap(config),
	}
}

// DynamoHintsComponent converts the typed dynamo config to the canonical dynamic component spec.
func DynamoHintsComponent(config DynamoHintsComponentConfig) OptimizerComponentSpec {
	return OptimizerComponentSpec{
		Kind:    "dynamo_hints",
		Enabled: true,
		Config:  mustConfigMap(config),
	}
}

// ToolParallelismComponent converts the typed tool parallelism config to the canonical dynamic component spec.
func ToolParallelismComponent(config ToolParallelismComponentConfig) OptimizerComponentSpec {
	return OptimizerComponentSpec{
		Kind:    "tool_parallelism",
		Enabled: true,
		Config:  mustConfigMap(config),
	}
}

// ExternalComponent converts hosted external component config to the canonical component spec.
func ExternalComponent(config ExternalComponentConfig) OptimizerComponentSpec {
	return OptimizerComponentSpec{
		Kind:    "external_component",
		Enabled: true,
		Config:  mustConfigMap(config),
	}
}

// ValidateOptimizerConfig validates the config and returns the full diagnostics report.
func ValidateOptimizerConfig(config OptimizerConfig) (OptimizerConfigReport, error) {
	cConfig, err := optimizerConfigCString(config)
	if err != nil {
		return OptimizerConfigReport{}, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var out *C.char
	status := C.nat_nexus_validate_optimizer_config(cConfig, &out)
	if err := checkStatus(status); err != nil {
		return OptimizerConfigReport{}, err
	}
	defer C.nat_nexus_string_free(out)

	var report OptimizerConfigReport
	if err := json.Unmarshal([]byte(C.GoString(out)), &report); err != nil {
		return OptimizerConfigReport{}, err
	}
	return report, nil
}

// NewOptimizerRuntime constructs an optimizer runtime from config.
func NewOptimizerRuntime(config OptimizerConfig) (*OptimizerRuntime, error) {
	cConfig, err := optimizerConfigCString(config)
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cConfig))

	var ptr *C.FfiOptimizerRuntime
	status := C.nat_nexus_optimizer_runtime_create(cConfig, &ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &OptimizerRuntime{ptr: ptr}, nil
}

// Register registers the optimizer runtime globally.
func (r *OptimizerRuntime) Register() error {
	if r == nil || r.ptr == nil {
		return errors.New("optimizer runtime is closed")
	}
	return checkStatus(C.nat_nexus_optimizer_runtime_register(r.ptr))
}

// Deregister deregisters the optimizer runtime.
func (r *OptimizerRuntime) Deregister() error {
	if r == nil || r.ptr == nil {
		return errors.New("optimizer runtime is closed")
	}
	return checkStatus(C.nat_nexus_optimizer_runtime_deregister(r.ptr))
}

// Shutdown gracefully shuts down the runtime and consumes the native handle.
func (r *OptimizerRuntime) Shutdown() error {
	if r == nil || r.ptr == nil {
		return errors.New("optimizer runtime is closed")
	}
	ptr := r.ptr
	r.ptr = nil
	return checkStatus(C.nat_nexus_optimizer_runtime_shutdown(ptr))
}

// Report returns the runtime creation diagnostics report.
func (r *OptimizerRuntime) Report() (OptimizerConfigReport, error) {
	if r == nil || r.ptr == nil {
		return OptimizerConfigReport{}, errors.New("optimizer runtime is closed")
	}

	var out *C.char
	status := C.nat_nexus_optimizer_runtime_report_json(r.ptr, &out)
	if err := checkStatus(status); err != nil {
		return OptimizerConfigReport{}, err
	}
	defer C.nat_nexus_string_free(out)

	var report OptimizerConfigReport
	if err := json.Unmarshal([]byte(C.GoString(out)), &report); err != nil {
		return OptimizerConfigReport{}, err
	}
	return report, nil
}

// Close frees the native handle without waiting for graceful shutdown.
func (r *OptimizerRuntime) Close() {
	if r == nil || r.ptr == nil {
		return
	}
	C.nat_nexus_optimizer_runtime_free(r.ptr)
	r.ptr = nil
}

// RegisterOptimizerPlugin registers a hosted optimizer plugin backed by Go callbacks.
func RegisterOptimizerPlugin(pluginKind string, handler OptimizerPluginHandler) error {
	cPluginKind := C.CString(pluginKind)
	defer C.free(unsafe.Pointer(cPluginKind))
	userData := registerClosure(handler)
	status := C.nat_nexus_optimizer_register_plugin(
		cPluginKind,
		(C.NatNexusOptimizerPluginValidateCb)(C.goOptimizerPluginValidateTrampoline),
		(C.NatNexusOptimizerPluginRegisterCb)(C.goOptimizerPluginRegisterTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	)
	return checkStatus(status)
}

// DeregisterOptimizerPlugin removes a hosted optimizer plugin by kind.
func DeregisterOptimizerPlugin(pluginKind string) error {
	cPluginKind := C.CString(pluginKind)
	defer C.free(unsafe.Pointer(cPluginKind))
	return checkStatus(C.nat_nexus_optimizer_deregister_plugin(cPluginKind))
}

// RegisterSubscriber adds a subscriber to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterSubscriber(name string, fn EventSubscriberFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_subscriber(
		ctx.ptr,
		cName,
		(C.NatNexusEventSubscriberFn)(C.goEventSubscriberTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmRequestIntercept adds an LLM request intercept to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_llm_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NatNexusLlmRequestInterceptCb)(C.goLlmRequestInterceptTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolRequestIntercept adds a tool request intercept to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterToolRequestIntercept(name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_tool_request_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		C._Bool(breakChain),
		(C.NatNexusToolSanitizeFn)(C.goToolSanitizeTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmExecutionIntercept adds an LLM execution intercept to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterLlmExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_llm_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NatNexusLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterLlmStreamExecutionIntercept adds an LLM stream execution intercept to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterLlmStreamExecutionIntercept(name string, priority int32, fn LLMExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_llm_stream_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NatNexusLlmExecInterceptCb)(C.goLlmExecInterceptTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

// RegisterToolExecutionIntercept adds a tool execution intercept to the hosted plugin context.
func (ctx *OptimizerPluginContext) RegisterToolExecutionIntercept(name string, priority int32, fn ToolExecutionInterceptFunc) error {
	if ctx == nil || ctx.ptr == nil {
		return errors.New("optimizer plugin context is closed")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	userData := registerClosure(fn)
	return checkStatus(C.nat_nexus_optimizer_plugin_context_register_tool_execution_intercept(
		ctx.ptr,
		cName,
		C.int32_t(priority),
		(C.NatNexusToolExecInterceptCb)(C.goToolExecInterceptTrampoline),
		userData,
		(C.NatNexusFreeFn)(C.goFreeTrampoline),
	))
}

func optimizerConfigCString(config OptimizerConfig) (*C.char, error) {
	payload, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}
	return C.CString(string(payload)), nil
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
