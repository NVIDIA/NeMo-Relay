// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nat_nexus provides Go bindings for the Nexus agent runtime via CGo.
//
// Nexus is a multi-language agent runtime framework that provides execution
// scope management, lifecycle events, and middleware (guardrails and intercepts)
// for tool and LLM calls. The core runtime is written in Rust; this package
// wraps the C FFI layer produced by the nat-nexus-ffi crate.
//
// The package exposes a hierarchical scope stack, tool and LLM call lifecycle
// management, priority-ordered guardrails for request/response sanitization and
// conditional gating, priority-ordered intercepts for request/response
// transformation and execution replacement, and an observer-pattern event
// subscription system.
//
// Sub-packages scope, tools, llm, guardrails, intercepts, and subscribers
// re-export the most common functions under shorter names for convenience.
//
// Build prerequisites: the nat-nexus-ffi shared library must be built first
// (cargo build --release -p nat-nexus-ffi) and discoverable via CGO_LDFLAGS.
package nat_nexus

/*
#cgo LDFLAGS: -lnvidia_nat_nexus_ffi
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiScopeStack FfiScopeStack;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;
typedef struct FfiStream FfiStream;

typedef void (*NatNexusFreeFn)(void* user_data);

// Core API
extern int32_t nat_nexus_get_handle(FfiScopeHandle** out);
extern int32_t nat_nexus_push_scope(const char* name, int32_t scope_type, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, FfiScopeHandle** out);
extern int32_t nat_nexus_pop_scope(const FfiScopeHandle* handle);
extern int32_t nat_nexus_event(const char* name, const FfiScopeHandle* parent, const char* data_json, const char* metadata_json);

// Tool lifecycle
extern int32_t nat_nexus_tool_call(const char* name, const char* args_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* tool_call_id, FfiToolHandle** out);
extern int32_t nat_nexus_tool_call_end(const FfiToolHandle* handle, const char* result_json, const char* data_json, const char* metadata_json);

// Tool call execute (with C function pointer callbacks)
typedef char* (*NatNexusToolExecFn)(void* user_data, const char* args_json);
extern int32_t nat_nexus_tool_call_execute(
	const char* name, const char* args_json,
	NatNexusToolExecFn func_cb, void* func_user_data, NatNexusFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	char** out);

// LLM lifecycle
typedef void (*NatNexusCollectorCb)(const char* chunk_json);
typedef struct Option_NatNexusCollectorCb { NatNexusCollectorCb cb; } Option_NatNexusCollectorCb;
typedef char* (*NatNexusFinalizerCb)();
typedef struct Option_NatNexusFinalizerCb { NatNexusFinalizerCb cb; } Option_NatNexusFinalizerCb;

static inline Option_NatNexusCollectorCb makeOptCollectorCb(NatNexusCollectorCb cb) {
	Option_NatNexusCollectorCb opt = { cb };
	return opt;
}
static inline Option_NatNexusFinalizerCb makeOptFinalizerCb(NatNexusFinalizerCb cb) {
	Option_NatNexusFinalizerCb opt = { cb };
	return opt;
}

extern int32_t nat_nexus_llm_call(const char* name, const char* native_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* model_name, FfiLLMHandle** out);
extern int32_t nat_nexus_llm_call_end(const FfiLLMHandle* handle, const char* response_json, const char* data_json, const char* metadata_json);

// LLM call execute
typedef char* (*NatNexusLlmExecFn)(void* user_data, const char* native_json);
extern int32_t nat_nexus_llm_call_execute(
	const char* name, const char* native_json,
	NatNexusLlmExecFn func_cb, void* func_user_data, NatNexusFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	char** out);

// LLM stream execute
extern int32_t nat_nexus_llm_stream_call_execute(
	const char* name, const char* native_json,
	NatNexusLlmExecFn func_cb, void* func_user_data, NatNexusFreeFn func_free,
	Option_NatNexusCollectorCb collector, Option_NatNexusFinalizerCb finalizer,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	FfiStream** out);

// Tool guardrails
typedef char* (*NatNexusToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nat_nexus_register_tool_sanitize_request_guardrail(const char* name, int32_t priority, NatNexusToolSanitizeFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_tool_sanitize_request_guardrail(const char* name);
extern int32_t nat_nexus_register_tool_sanitize_response_guardrail(const char* name, int32_t priority, NatNexusToolSanitizeFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_tool_sanitize_response_guardrail(const char* name);

typedef char* (*NatNexusToolConditionalFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nat_nexus_register_tool_conditional_execution_guardrail(const char* name, int32_t priority, NatNexusToolConditionalFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_tool_conditional_execution_guardrail(const char* name);

// Tool intercepts
extern int32_t nat_nexus_register_tool_request_intercept(const char* name, int32_t priority, _Bool break_chain, NatNexusToolSanitizeFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_tool_request_intercept(const char* name);
extern int32_t nat_nexus_register_tool_response_intercept(const char* name, int32_t priority, _Bool break_chain, NatNexusToolSanitizeFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_tool_response_intercept(const char* name);

// Middleware chain intercept callback types (must be declared before use in externs)
typedef char* (*NatNexusToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NatNexusToolExecInterceptCb)(void* user_data, const char* args_json, NatNexusToolExecNextFn next_fn, void* next_ctx);
extern int32_t nat_nexus_register_tool_execution_intercept(const char* name, int32_t priority, NatNexusToolExecInterceptCb exec_cb, void* exec_user_data, NatNexusFreeFn exec_free);
extern int32_t nat_nexus_deregister_tool_execution_intercept(const char* name);

// LLM guardrails
typedef FfiLLMRequest* (*NatNexusLlmRequestCb)(void* user_data, const FfiLLMRequest* request);
extern int32_t nat_nexus_register_llm_sanitize_request_guardrail(const char* name, int32_t priority, NatNexusLlmRequestCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_llm_sanitize_request_guardrail(const char* name);

typedef char* (*NatNexusLlmResponseFn)(void* user_data, const char* response_json);
extern int32_t nat_nexus_register_llm_sanitize_response_guardrail(const char* name, int32_t priority, NatNexusLlmResponseFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_llm_sanitize_response_guardrail(const char* name);

typedef char* (*NatNexusLlmConditionalCb)(void* user_data, const FfiLLMRequest* request);
extern int32_t nat_nexus_register_llm_conditional_execution_guardrail(const char* name, int32_t priority, NatNexusLlmConditionalCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_llm_conditional_execution_guardrail(const char* name);

// LLM intercepts
extern int32_t nat_nexus_register_llm_request_intercept(const char* name, int32_t priority, _Bool break_chain, NatNexusLlmRequestCb cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_llm_request_intercept(const char* name);
typedef char* (*NatNexusLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NatNexusLlmExecInterceptCb)(void* user_data, const char* native_json, NatNexusLlmExecNextFn next_fn, void* next_ctx);

extern int32_t nat_nexus_register_llm_execution_intercept(const char* name, int32_t priority, NatNexusLlmExecInterceptCb exec_cb, void* exec_user_data, NatNexusFreeFn exec_free);
extern int32_t nat_nexus_deregister_llm_execution_intercept(const char* name);
extern int32_t nat_nexus_register_llm_stream_execution_intercept(const char* name, int32_t priority, NatNexusLlmExecInterceptCb exec_cb, void* exec_user_data, NatNexusFreeFn exec_free);
extern int32_t nat_nexus_deregister_llm_stream_execution_intercept(const char* name);

// Subscribers
typedef void (*NatNexusEventSubscriberFn)(void* user_data, const FfiEvent* event);
extern int32_t nat_nexus_register_subscriber(const char* name, NatNexusEventSubscriberFn cb, void* user_data, NatNexusFreeFn free_fn);
extern int32_t nat_nexus_deregister_subscriber(const char* name);

// Standalone middleware chains
extern int32_t nat_nexus_tool_request_intercepts(const char* name, const char* args_json, char** out);
extern int32_t nat_nexus_tool_conditional_execution(const char* name, const char* args_json);
extern int32_t nat_nexus_tool_response_intercepts(const char* name, const char* result_json, char** out);
extern int32_t nat_nexus_llm_request_intercepts(const char* request_json, char** out);
extern int32_t nat_nexus_llm_conditional_execution(const char* request_json);
// Error
extern const char* nat_nexus_last_error();

// String free
extern void nat_nexus_string_free(char* ptr);

// Scope stack isolation
extern int32_t nat_nexus_scope_stack_create(FfiScopeStack** out);
extern int32_t nat_nexus_scope_stack_set_thread(const FfiScopeStack* stack);
extern _Bool nat_nexus_scope_stack_active(void);
extern void nat_nexus_scope_stack_free(FfiScopeStack* ptr);

// ATIF exporter
extern int32_t nat_nexus_atif_exporter_create(const char*, const char*, const char*, const char*, void**);
extern int32_t nat_nexus_atif_exporter_register(const void*, const char*);
extern int32_t nat_nexus_atif_exporter_deregister(const char*);
extern int32_t nat_nexus_atif_exporter_export(const void*, const char*, char**);
extern int32_t nat_nexus_atif_exporter_clear(const void*);
extern void nat_nexus_atif_exporter_free(void*);

// Go trampoline forward declarations (defined via //export in callbacks.go)
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern char* goToolExecTrampoline(void*, const char*);
extern void goEventSubscriberTrampoline(void*, const FfiEvent*);
extern void goFreeTrampoline(void*);
extern FfiLLMRequest* goLlmRequestTrampoline(void*, const FfiLLMRequest*);
extern char* goLlmResponseTrampoline(void*, const char*);
extern char* goLlmConditionalTrampoline(void*, const FfiLLMRequest*);
extern char* goLlmExecTrampoline(void*, const char*);
extern char* goToolExecInterceptTrampoline(void*, const char*, NatNexusToolExecNextFn, void*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NatNexusLlmExecNextFn, void*);
extern void goCollectorTrampoline(const char*);
extern char* goFinalizerTrampoline();
*/
import "C"

import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

func lastError() error {
	msg := C.nat_nexus_last_error()
	if msg == nil {
		return errors.New("unknown nat_nexus error")
	}
	return errors.New(C.GoString(msg))
}

func checkStatus(status C.int32_t) error {
	if status == 0 {
		return nil
	}
	return lastError()
}

// ---------------------------------------------------------------------------
// Scope options (functional options pattern)
// ---------------------------------------------------------------------------

type scopeOptions struct {
	parent     *C.FfiScopeHandle
	attributes uint32
	data       *C.char
	metadata   *C.char
}

// ScopeOption is a functional option that configures optional parameters for
// [PushScope]. Options are applied in the order they are passed. Available
// options include [WithParent], [WithScopeAttributes], [WithData], and [WithMetadata].
type ScopeOption func(*scopeOptions)

// WithParent sets the parent scope handle for the new scope. If parent is nil,
// the scope is created under the current top of the scope stack. Use this to
// build non-linear scope hierarchies (e.g., forking parallel branches).
func WithParent(parent *ScopeHandle) ScopeOption {
	return func(o *scopeOptions) {
		if parent != nil {
			o.parent = parent.ptr
		}
	}
}

// WithScopeAttributes sets scope attribute bitflags. Attribute constants such
// as [ScopeAttrParallel] and [ScopeAttrRelocatable] can be combined with
// bitwise OR.
func WithScopeAttributes(attrs uint32) ScopeOption {
	return func(o *scopeOptions) {
		o.attributes = attrs
	}
}

// WithData attaches an arbitrary JSON application data payload to the new scope.
// Data is typically used for domain-specific information (e.g., user inputs,
// configuration) as opposed to the operational metadata payload.
func WithData(data json.RawMessage) ScopeOption {
	return func(o *scopeOptions) {
		o.data = C.CString(string(data))
	}
}

// WithMetadata attaches an arbitrary JSON metadata payload to the new scope.
// Metadata is typically used for operational context (e.g., trace IDs, session
// info) as opposed to the primary data payload.
func WithMetadata(metadata json.RawMessage) ScopeOption {
	return func(o *scopeOptions) {
		o.metadata = C.CString(string(metadata))
	}
}

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty (i.e., no scope has been
// pushed). The returned [ScopeHandle] is reference-counted and safe to hold
// beyond the lifetime of the scope itself.
func GetHandle() (*ScopeHandle, error) {
	var out *C.FfiScopeHandle
	status := C.nat_nexus_get_handle(&out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newScopeHandle(out), nil
}

// PushScope creates a new scope and pushes it onto the hierarchical scope
// stack. The scope is assigned a unique UUID and emits a Start event to all
// registered subscribers. Use [PopScope] to end the scope. Optional parameters
// can be set via [WithParent], [WithScopeAttributes], [WithData], and [WithMetadata].
//
// The name should be a human-readable identifier for the scope (e.g.,
// "my-agent", "search-tool"). The scopeType categorizes the scope for
// observability; see [ScopeType] constants for valid values.
func PushScope(name string, scopeType ScopeType, opts ...ScopeOption) (*ScopeHandle, error) {
	o := &scopeOptions{}
	for _, opt := range opts {
		opt(o)
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if o.data != nil {
		defer C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		defer C.free(unsafe.Pointer(o.metadata))
	}

	var out *C.FfiScopeHandle
	status := C.nat_nexus_push_scope(cName, C.int32_t(scopeType), o.parent, C.uint32_t(o.attributes), o.data, o.metadata, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newScopeHandle(out), nil
}

// PopScope removes the given scope from the scope stack and emits an End event
// to all registered subscribers. The handle must have been returned by a
// previous call to [PushScope]. Popping scopes out of stack order returns an
// error.
func PopScope(handle *ScopeHandle) error {
	return checkStatus(C.nat_nexus_pop_scope(handle.ptr))
}

// ---------------------------------------------------------------------------
// Event options
// ---------------------------------------------------------------------------

type eventOptions struct {
	parent   *C.FfiScopeHandle
	data     *C.char
	metadata *C.char
}

// EventOption is a functional option that configures optional parameters for
// [EmitEvent]. Available options include [WithEventParent], [WithEventData],
// and [WithEventMetadata].
type EventOption func(*eventOptions)

// WithEventParent sets the parent scope handle for the event. If not provided,
// the event is associated with the scope currently at the top of the stack.
func WithEventParent(parent *ScopeHandle) EventOption {
	return func(o *eventOptions) {
		if parent != nil {
			o.parent = parent.ptr
		}
	}
}

// WithEventData attaches an arbitrary JSON data payload to the event. This data
// is delivered to all registered subscribers and can be used for structured
// logging, tracing, or custom instrumentation.
func WithEventData(data json.RawMessage) EventOption {
	return func(o *eventOptions) {
		o.data = C.CString(string(data))
	}
}

// WithEventMetadata attaches an arbitrary JSON metadata payload to the event.
// Metadata is typically used for operational context (e.g., trace IDs, timing
// hints) as opposed to the primary data payload.
func WithEventMetadata(metadata json.RawMessage) EventOption {
	return func(o *eventOptions) {
		o.metadata = C.CString(string(metadata))
	}
}

// EmitEvent emits an instantaneous Mark event within the current scope. Mark
// events represent point-in-time occurrences (e.g., checkpoints, milestones)
// and are delivered to all registered subscribers. Optional data and metadata
// payloads can be attached via [WithEventData] and [WithEventMetadata].
func EmitEvent(name string, opts ...EventOption) error {
	o := &eventOptions{}
	for _, opt := range opts {
		opt(o)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if o.data != nil {
		defer C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		defer C.free(unsafe.Pointer(o.metadata))
	}

	return checkStatus(C.nat_nexus_event(cName, o.parent, o.data, o.metadata))
}

// ---------------------------------------------------------------------------
// Tool lifecycle options
// ---------------------------------------------------------------------------

type toolCallOptions struct {
	parent     *C.FfiScopeHandle
	attributes uint32
	data       *C.char
	metadata   *C.char
	toolCallID *C.char
}

// ToolCallOption is a functional option that configures optional parameters for
// tool call functions ([ToolCall], [ToolCallEnd], [ToolCallExecute]). Available
// options include [WithToolParent], [WithToolAttributes], [WithToolData], and
// [WithToolMetadata].
type ToolCallOption func(*toolCallOptions)

// WithToolParent sets the parent scope handle for a tool call. If not provided,
// the tool call is associated with the scope currently at the top of the stack.
func WithToolParent(parent *ScopeHandle) ToolCallOption {
	return func(o *toolCallOptions) {
		if parent != nil {
			o.parent = parent.ptr
		}
	}
}

// WithToolAttributes sets attribute bitflags for a tool call. See [ToolAttrLocal]
// for available flags. Multiple flags can be combined with bitwise OR.
func WithToolAttributes(attrs uint32) ToolCallOption {
	return func(o *toolCallOptions) {
		o.attributes = attrs
	}
}

// WithToolData attaches an arbitrary JSON data payload to the tool call events.
// This data is delivered to subscribers in the Start and End events.
func WithToolData(data json.RawMessage) ToolCallOption {
	return func(o *toolCallOptions) {
		o.data = C.CString(string(data))
	}
}

// WithToolMetadata attaches an arbitrary JSON metadata payload to the tool call
// events. Metadata is typically used for operational context (e.g., trace IDs).
func WithToolMetadata(metadata json.RawMessage) ToolCallOption {
	return func(o *toolCallOptions) {
		o.metadata = C.CString(string(metadata))
	}
}

// WithToolCallID sets an optional tool call ID for the tool call. This ID is
// typically assigned by the LLM to correlate the tool invocation with the
// original tool_call request in the conversation. Pass an empty string or omit
// this option to leave the tool call ID unset.
func WithToolCallID(id string) ToolCallOption {
	return func(o *toolCallOptions) {
		o.toolCallID = C.CString(id)
	}
}

func freeToolOpts(o *toolCallOptions) {
	if o.data != nil {
		C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		C.free(unsafe.Pointer(o.metadata))
	}
	if o.toolCallID != nil {
		C.free(unsafe.Pointer(o.toolCallID))
	}
}

// ToolCall starts a tool call lifecycle and returns a [ToolHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [ToolCallEnd] when the tool completes. For a higher-level API that
// manages the full lifecycle automatically, use [ToolCallExecute] instead.
//
// The name identifies the tool being invoked, and args contains the tool
// arguments as JSON. Optional parameters can be set via [ToolCallOption] values.
func ToolCall(name string, args json.RawMessage, opts ...ToolCallOption) (*ToolHandle, error) {
	o := &toolCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeToolOpts(o)

	cName := C.CString(name)
	cArgs := C.CString(string(args))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cArgs))

	var out *C.FfiToolHandle
	status := C.nat_nexus_tool_call(cName, cArgs, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.toolCallID, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newToolHandle(out), nil
}

// ToolCallEnd completes a tool call that was previously started with [ToolCall].
// It emits an End event to all subscribers with the provided result JSON. The
// handle must have been returned by a prior [ToolCall] invocation.
func ToolCallEnd(handle *ToolHandle, result json.RawMessage, opts ...ToolCallOption) error {
	o := &toolCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeToolOpts(o)

	cResult := C.CString(string(result))
	defer C.free(unsafe.Pointer(cResult))

	return checkStatus(C.nat_nexus_tool_call_end(handle.ptr, cResult, o.data, o.metadata))
}

// ToolCallExecute runs a complete tool call lifecycle through the full
// middleware pipeline: conditional-execution guardrails (on raw args),
// request intercepts, sanitize-request guardrails, execution intercepts,
// the provided fn, response intercepts, sanitize-response guardrails.
// On rejection, only a standalone Mark event is emitted (no Start/End pair)
// and GuardrailRejected is returned. This is the recommended high-level API
// for tool invocations.
func ToolCallExecute(name string, args json.RawMessage, fn ToolExecutionFunc, opts ...ToolCallOption) (json.RawMessage, error) {
	o := &toolCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeToolOpts(o)

	id := registerClosure(fn)

	cName := C.CString(name)
	cArgs := C.CString(string(args))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cArgs))

	var out *C.char
	status := C.nat_nexus_tool_call_execute(
		cName, cArgs,
		C.NatNexusToolExecFn(C.goToolExecTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nat_nexus_string_free(out)
	return result, nil
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

type llmCallOptions struct {
	parent     *C.FfiScopeHandle
	attributes uint32
	data       *C.char
	metadata   *C.char
	modelName  *C.char
}

// LLMCallOption is a functional option that configures optional parameters for
// LLM call functions ([LlmCall], [LlmCallEnd], [LlmCallExecute],
// [LlmStreamCallExecute], [LlmConditionalExecution]). Available options include
// [WithLLMParent], [WithLLMAttributes], [WithLLMData], [WithLLMMetadata], and
// [WithLLMModelName].
type LLMCallOption func(*llmCallOptions)

// WithLLMParent sets the parent scope handle for an LLM call. If not provided,
// the LLM call is associated with the scope currently at the top of the stack.
func WithLLMParent(parent *ScopeHandle) LLMCallOption {
	return func(o *llmCallOptions) {
		if parent != nil {
			o.parent = parent.ptr
		}
	}
}

// WithLLMAttributes sets attribute bitflags for an LLM call. See
// [LLMAttrStateless] and [LLMAttrStreaming] for available flags. Multiple flags
// can be combined with bitwise OR.
func WithLLMAttributes(attrs uint32) LLMCallOption {
	return func(o *llmCallOptions) {
		o.attributes = attrs
	}
}

// WithLLMData attaches an arbitrary JSON data payload to the LLM call events.
// This data is delivered to subscribers in the Start and End events.
func WithLLMData(data json.RawMessage) LLMCallOption {
	return func(o *llmCallOptions) {
		o.data = C.CString(string(data))
	}
}

// WithLLMMetadata attaches an arbitrary JSON metadata payload to the LLM call
// events. Metadata is typically used for operational context (e.g., trace IDs).
func WithLLMMetadata(metadata json.RawMessage) LLMCallOption {
	return func(o *llmCallOptions) {
		o.metadata = C.CString(string(metadata))
	}
}

// WithLLMModelName sets an optional model name for the LLM call. This is used
// to record which specific model (e.g., "gpt-4", "claude-3-opus") was invoked,
// separate from the logical LLM provider name. Pass an empty string or omit
// this option to leave the model name unset.
func WithLLMModelName(name string) LLMCallOption {
	return func(o *llmCallOptions) {
		o.modelName = C.CString(name)
	}
}

func freeLLMOpts(o *llmCallOptions) {
	if o.data != nil {
		C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		C.free(unsafe.Pointer(o.metadata))
	}
	if o.modelName != nil {
		C.free(unsafe.Pointer(o.modelName))
	}
}

// LlmCall starts an LLM call lifecycle and returns an [LLMHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [LlmCallEnd] when the LLM responds. For a higher-level API that manages
// the full lifecycle automatically, use [LlmCallExecute] or
// [LlmStreamCallExecute] instead.
//
// The name identifies the LLM provider/model, and request is an LLMRequest-shaped
// value ({headers, content}) that will be serialized to JSON. Optional parameters
// can be set via [LLMCallOption] values.
func LlmCall(name string, request interface{}, opts ...LLMCallOption) (*LLMHandle, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	cName := C.CString(name)
	cRequest := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.FfiLLMHandle
	status := C.nat_nexus_llm_call(cName, cRequest, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.modelName, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newLLMHandle(out), nil
}

// LlmCallEnd completes an LLM call that was previously started with [LlmCall].
// It emits an End event to all subscribers with the provided response JSON. The
// handle must have been returned by a prior [LlmCall] invocation.
func LlmCallEnd(handle *LLMHandle, response json.RawMessage, opts ...LLMCallOption) error {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	cResponse := C.CString(string(response))
	defer C.free(unsafe.Pointer(cResponse))

	return checkStatus(C.nat_nexus_llm_call_end(handle.ptr, cResponse, o.data, o.metadata))
}

// LlmCallExecute runs a complete LLM call lifecycle through the full
// middleware pipeline: conditional-execution guardrails (on raw request),
// request intercepts, sanitize-request guardrails, execution intercepts,
// the provided fn, response intercepts, sanitize-response guardrails.
// On rejection, only a standalone Mark event is emitted (no Start/End pair)
// and GuardrailRejected is returned. This is the recommended high-level API
// for non-streaming LLM invocations.
func LlmCallExecute(name string, request interface{}, fn LLMExecutionFunc, opts ...LLMCallOption) (json.RawMessage, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	id := registerClosure(fn)

	cName := C.CString(name)
	cRequest := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.char
	status := C.nat_nexus_llm_call_execute(
		cName, cRequest,
		C.NatNexusLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nat_nexus_string_free(out)
	return result, nil
}

// LlmStreamCallExecute runs a streaming LLM call lifecycle. Like
// [LlmCallExecute], conditional-execution guardrails run first on the raw
// request. If accepted, it runs the remaining middleware pipeline and returns
// an [LlmStream] that yields individual SSE (Server-Sent Event) chunks.
// Stream response intercepts are applied to each chunk as it is consumed.
// The caller must call [LlmStream.Next] repeatedly until [io.EOF] is
// returned, then call [LlmStream.Close].
//
// The optional collector callback is invoked with each intercepted chunk string,
// allowing the caller to accumulate chunks for aggregation. The optional
// finalizer callback is invoked once when the stream is exhausted and must
// return a JSON string representing the aggregated response. Pass nil for
// either to use the default no-op behavior.
func LlmStreamCallExecute(name string, request interface{}, fn LLMExecutionFunc, collector CollectorFunc, finalizer FinalizerFunc, opts ...LLMCallOption) (*LlmStream, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	id := registerClosure(fn)

	cName := C.CString(name)
	cRequest := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cRequest))

	// Set the active collector/finalizer for the duration of the blocking FFI call.
	// The C collector/finalizer callbacks are plain function pointers (no user_data),
	// so we route through global state protected by collectorMu.
	collectorMu.Lock()
	activeCollector = collector
	activeFinalizer = finalizer
	collectorMu.Unlock()

	defer func() {
		collectorMu.Lock()
		activeCollector = nil
		activeFinalizer = nil
		collectorMu.Unlock()
	}()

	var cCollector C.Option_NatNexusCollectorCb
	if collector != nil {
		cCollector = C.makeOptCollectorCb(C.NatNexusCollectorCb(C.goCollectorTrampoline))
	} else {
		cCollector = C.makeOptCollectorCb(nil)
	}

	var cFinalizer C.Option_NatNexusFinalizerCb
	if finalizer != nil {
		cFinalizer = C.makeOptFinalizerCb(C.NatNexusFinalizerCb(C.goFinalizerTrampoline))
	} else {
		cFinalizer = C.makeOptFinalizerCb(nil)
	}

	var out *C.FfiStream
	status := C.nat_nexus_llm_stream_call_execute(
		cName, cRequest,
		C.NatNexusLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
		cCollector,
		cFinalizer,
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newLlmStream(out), nil
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (Tool)
// ---------------------------------------------------------------------------

// RegisterToolSanitizeRequestGuardrail registers a guardrail that sanitizes
// tool request arguments before they are passed to the tool. The callback
// receives the tool name and arguments JSON and must return the (possibly
// modified) arguments. Guardrails are invoked in priority order (lower values
// run first). The name must be unique among tool sanitize-request guardrails;
// registering a duplicate name returns an AlreadyExists error.
func RegisterToolSanitizeRequestGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeRequestGuardrail removes a previously registered tool
// sanitize-request guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_sanitize_request_guardrail(cName))
}

// RegisterToolSanitizeResponseGuardrail registers a guardrail that sanitizes
// tool response data before it is returned to the caller. The callback receives
// the tool name and response JSON and must return the (possibly modified)
// response. Guardrails are invoked in priority order (lower values run first).
func RegisterToolSanitizeResponseGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeResponseGuardrail removes a previously registered tool
// sanitize-response guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_sanitize_response_guardrail(cName))
}

// RegisterToolConditionalExecutionGuardrail registers a guardrail that
// conditionally gates tool execution. The callback receives the tool name and
// arguments, and returns nil to allow execution or a non-nil pointer to an
// error message string to reject it (resulting in a GuardrailRejected error).
// Multiple conditional guardrails run in priority order; the first rejection
// short-circuits the chain.
func RegisterToolConditionalExecutionGuardrail(name string, priority int32, fn ToolConditionalFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusToolConditionalFn(C.goToolConditionalTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolConditionalExecutionGuardrail removes a previously registered
// tool conditional-execution guardrail by name. Returns a NotFound error if no
// guardrail with the given name is registered.
func DeregisterToolConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_conditional_execution_guardrail(cName))
}

// RegisterToolRequestIntercept registers an intercept that transforms tool
// request arguments before they reach the tool. Intercepts run in priority
// order (lower values first). When breakChain is true, no lower-priority
// intercepts in the chain are invoked after this one, allowing early
// short-circuiting of the pipeline.
func RegisterToolRequestIntercept(name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NatNexusToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolRequestIntercept removes a previously registered tool request
// intercept by name.
func DeregisterToolRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_request_intercept(cName))
}

// RegisterToolResponseIntercept registers an intercept that transforms tool
// response data after the tool returns. Intercepts run in priority order (lower
// values first). When breakChain is true, no lower-priority intercepts in the
// chain are invoked after this one, allowing early short-circuiting of the
// pipeline.
func RegisterToolResponseIntercept(name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NatNexusToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolResponseIntercept removes a previously registered tool response
// intercept by name.
func DeregisterToolResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_response_intercept(cName))
}

// RegisterToolExecutionIntercept registers an execution intercept following
// the middleware chain pattern. execFn is called with the args and a `next`
// function. Call `next` to invoke the next intercept or original
// implementation; skip calling `next` to short-circuit the chain.
func RegisterToolExecutionIntercept(name string, priority int32, execFn ToolExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_tool_execution_intercept(
		cName, C.int32_t(priority),
		C.NatNexusToolExecInterceptCb(C.goToolExecInterceptTrampoline),
		execID,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolExecutionIntercept removes a previously registered tool
// execution intercept by name.
func DeregisterToolExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_tool_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (LLM)
// ---------------------------------------------------------------------------

// RegisterLlmSanitizeRequestGuardrail registers a guardrail that sanitizes LLM
// request data before the call is made. The callback receives the request
// headers and content JSON and must return the (possibly modified) versions.
// Guardrails are invoked in priority order (lower values run first).
func RegisterLlmSanitizeRequestGuardrail(name string, priority int32, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusLlmRequestCb(C.goLlmRequestTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeRequestGuardrail removes a previously registered LLM
// sanitize-request guardrail by name.
func DeregisterLlmSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_sanitize_request_guardrail(cName))
}

// RegisterLlmSanitizeResponseGuardrail registers a guardrail that sanitizes
// LLM response data before it is returned to the caller. The callback receives
// the response as plain JSON and must return the (possibly modified) response
// JSON. Guardrails are invoked in priority order (lower values run first).
func RegisterLlmSanitizeResponseGuardrail(name string, priority int32, fn LLMResponseFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusLlmResponseFn(C.goLlmResponseTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeResponseGuardrail removes a previously registered LLM
// sanitize-response guardrail by name.
func DeregisterLlmSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_sanitize_response_guardrail(cName))
}

// RegisterLlmConditionalExecutionGuardrail registers a guardrail that
// conditionally gates LLM execution. The callback receives the LLM request
// parameters and returns nil to allow execution or a non-nil pointer to an
// error message string to reject it (resulting in a GuardrailRejected error).
// Multiple conditional guardrails run in priority order; the first rejection
// short-circuits the chain.
func RegisterLlmConditionalExecutionGuardrail(name string, priority int32, fn LLMConditionalFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NatNexusLlmConditionalCb(C.goLlmConditionalTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmConditionalExecutionGuardrail removes a previously registered
// LLM conditional-execution guardrail by name.
func DeregisterLlmConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_conditional_execution_guardrail(cName))
}

// RegisterLlmRequestIntercept registers an intercept that transforms the LLM
// request (headers and content) before the call is made. Intercepts run in
// priority order (lower values first). When breakChain is true, no
// lower-priority intercepts in the chain are invoked after this one.
func RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NatNexusLlmRequestCb(C.goLlmRequestTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmRequestIntercept removes a previously registered LLM request
// intercept by name.
func DeregisterLlmRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_request_intercept(cName))
}

// RegisterLlmExecutionIntercept registers an execution intercept following
// the middleware chain pattern. execFn is called with the request parameters
// and a `next` function. Call `next` to invoke the next intercept or original
// implementation; skip calling `next` to short-circuit the chain.
func RegisterLlmExecutionIntercept(name string, priority int32, execFn LLMExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_execution_intercept(
		cName, C.int32_t(priority),
		C.NatNexusLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmExecutionIntercept removes a previously registered LLM
// execution intercept by name.
func DeregisterLlmExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_execution_intercept(cName))
}

// RegisterLlmStreamExecutionIntercept registers an execution intercept for
// streaming LLM calls following the middleware chain pattern. execFn is called
// with the request parameters and a `next` function. Call `next` to invoke the
// next intercept or original implementation; skip calling `next` to
// short-circuit.
func RegisterLlmStreamExecutionIntercept(name string, priority int32, execFn LLMExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_llm_stream_execution_intercept(
		cName, C.int32_t(priority),
		C.NatNexusLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamExecutionIntercept removes a previously registered LLM
// stream execution intercept by name.
func DeregisterLlmStreamExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_llm_stream_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Subscriber registration
// ---------------------------------------------------------------------------

// RegisterSubscriber registers a named event subscriber that will be called for
// every lifecycle event (Start, End, Mark) emitted by the runtime. Subscribers
// are identified by a unique name; registering a duplicate returns an
// AlreadyExists error. The callback receives an [Event] pointer that is only
// valid for the duration of the call; callers must not retain it.
func RegisterSubscriber(name string, fn EventSubscriberFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_register_subscriber(
		cName,
		C.NatNexusEventSubscriberFn(C.goEventSubscriberTrampoline),
		id,
		C.NatNexusFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterSubscriber removes a named event subscriber. Returns a NotFound
// error if no subscriber with the given name is registered.
func DeregisterSubscriber(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nat_nexus_deregister_subscriber(cName))
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

// ScopeStack represents an isolated scope stack for per-request/per-goroutine isolation.
// Each ScopeStack has its own root scope and is independent of other scope stacks.
type ScopeStack struct {
	ptr *C.FfiScopeStack
}

// NewScopeStack creates a new isolated scope stack.
// The caller must call Close() when done.
func NewScopeStack() (*ScopeStack, error) {
	var ptr *C.FfiScopeStack
	status := C.nat_nexus_scope_stack_create(&ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &ScopeStack{ptr: ptr}, nil
}

// Close frees the scope stack. After calling Close, the ScopeStack must not be used.
func (s *ScopeStack) Close() {
	if s.ptr != nil {
		C.nat_nexus_scope_stack_free(s.ptr)
		s.ptr = nil
	}
}

// Run binds this scope stack to the current OS thread and executes fn.
// The calling goroutine is locked to the OS thread for the duration of fn.
// All Nexus scope operations within fn will use this scope stack.
//
// This is the canonical way to propagate a scope stack to a worker goroutine.
func (s *ScopeStack) Run(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.nat_nexus_scope_stack_set_thread(s.ptr)
	fn()
}

// ScopeStackActive returns true if the current OS thread has an explicitly-bound
// scope stack (set via ScopeStack.Run or directly via set_thread), or false if
// only the auto-created default is present.
//
// This function must be called from a goroutine locked to an OS thread
// (e.g. inside ScopeStack.Run) for the result to be meaningful.
func ScopeStackActive() bool {
	return bool(C.nat_nexus_scope_stack_active())
}

// ---------------------------------------------------------------------------
// ATIF Exporter
// ---------------------------------------------------------------------------

// AtifExporter collects lifecycle events and exports them as ATIF trajectories.
type AtifExporter struct {
	ptr unsafe.Pointer
}

// NewAtifExporter creates a new ATIF exporter.
// modelName can be empty string for no model name.
func NewAtifExporter(sessionID, agentName, agentVersion, modelName string) (*AtifExporter, error) {
	cSessionID := C.CString(sessionID)
	defer C.free(unsafe.Pointer(cSessionID))
	cAgentName := C.CString(agentName)
	defer C.free(unsafe.Pointer(cAgentName))
	cAgentVersion := C.CString(agentVersion)
	defer C.free(unsafe.Pointer(cAgentVersion))

	var cModelName *C.char
	if modelName != "" {
		cModelName = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModelName))
	}

	var ptr unsafe.Pointer
	status := C.nat_nexus_atif_exporter_create(cSessionID, cAgentName, cAgentVersion, cModelName, &ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &AtifExporter{ptr: ptr}, nil
}

// Register registers the exporter as an event subscriber with the given name.
func (e *AtifExporter) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nat_nexus_atif_exporter_register(e.ptr, cName)
	return checkStatus(status)
}

// Deregister removes the exporter subscriber by name.
func (e *AtifExporter) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nat_nexus_atif_exporter_deregister(cName)
	return checkStatus(status)
}

// ExportJSON exports collected events as an ATIF trajectory JSON string.
// rootUUID filters to a specific root scope. Pass empty string for no filter.
func (e *AtifExporter) ExportJSON(rootUUID string) (json.RawMessage, error) {
	var cRootUUID *C.char
	if rootUUID != "" {
		cRootUUID = C.CString(rootUUID)
		defer C.free(unsafe.Pointer(cRootUUID))
	}

	var cOut *C.char
	status := C.nat_nexus_atif_exporter_export(e.ptr, cRootUUID, &cOut)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nat_nexus_string_free(cOut)
	return json.RawMessage(C.GoString(cOut)), nil
}

// Clear removes all collected events.
func (e *AtifExporter) Clear() {
	C.nat_nexus_atif_exporter_clear(e.ptr)
}

// Close frees the exporter handle.
func (e *AtifExporter) Close() {
	if e.ptr != nil {
		C.nat_nexus_atif_exporter_free(e.ptr)
		e.ptr = nil
	}
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

// ToolRequestIntercepts runs the registered tool request intercept chain on the
// given arguments and returns the transformed arguments.
func ToolRequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cArgs := C.CString(string(args))
	defer C.free(unsafe.Pointer(cArgs))

	var out *C.char
	status := C.nat_nexus_tool_request_intercepts(cName, cArgs, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nat_nexus_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// ToolConditionalExecution runs the registered tool conditional execution
// guardrail chain. Returns nil if all guardrails pass, or an error with the
// rejection reason if blocked.
func ToolConditionalExecution(name string, args json.RawMessage) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cArgs := C.CString(string(args))
	defer C.free(unsafe.Pointer(cArgs))

	status := C.nat_nexus_tool_conditional_execution(cName, cArgs)
	return checkStatus(status)
}

// ToolResponseIntercepts runs the registered tool response intercept chain on
// the given result and returns the transformed result.
func ToolResponseIntercepts(name string, result json.RawMessage) (json.RawMessage, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cResult := C.CString(string(result))
	defer C.free(unsafe.Pointer(cResult))

	var out *C.char
	status := C.nat_nexus_tool_response_intercepts(cName, cResult, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nat_nexus_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// LlmRequestIntercepts runs the registered LLM request intercept chain on the
// given request (serialized as JSON) and returns the transformed request JSON.
func LlmRequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.char
	status := C.nat_nexus_llm_request_intercepts(cRequest, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nat_nexus_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all guardrails pass, or an error with the
// rejection reason if blocked. The request should be in LLMRequest JSON format
// ({"headers": {...}, "content": {...}}).
func LlmConditionalExecution(request json.RawMessage) error {
	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	status := C.nat_nexus_llm_conditional_execution(cRequest)
	return checkStatus(status)
}
