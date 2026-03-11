// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nvmagic provides Go bindings for the NVMagic agent runtime via CGo.
//
// NVMagic is a multi-language agent runtime framework that provides execution
// scope management, lifecycle events, and middleware (guardrails and intercepts)
// for tool and LLM calls. The core runtime is written in Rust; this package
// wraps the C FFI layer produced by the nvmagic-ffi crate.
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
// Build prerequisites: the nvmagic-ffi shared library must be built first
// (cargo build --release -p nvmagic-ffi) and discoverable via CGO_LDFLAGS.
package nvmagic

/*
#cgo LDFLAGS: -lnvmagic_ffi
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiScopeStack FfiScopeStack;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiLLMResponse FfiLLMResponse;
typedef struct FfiEvent FfiEvent;
typedef struct FfiStream FfiStream;

typedef void (*NvMagicFreeFn)(void* user_data);

// Core API
extern int32_t nvmagic_get_handle(FfiScopeHandle** out);
extern int32_t nvmagic_push_scope(const char* name, int32_t scope_type, const FfiScopeHandle* parent, uint32_t attributes, FfiScopeHandle** out);
extern int32_t nvmagic_pop_scope(const FfiScopeHandle* handle);
extern int32_t nvmagic_event(const char* name, const FfiScopeHandle* parent, const char* data_json, const char* metadata_json);

// Tool lifecycle
extern int32_t nvmagic_tool_call(const char* name, const char* args_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* tool_call_id, FfiToolHandle** out);
extern int32_t nvmagic_tool_call_end(const FfiToolHandle* handle, const char* result_json, const char* data_json, const char* metadata_json);

// Tool call execute (with C function pointer callbacks)
typedef char* (*NvMagicToolExecFn)(void* user_data, const char* args_json);
extern int32_t nvmagic_tool_call_execute(
	const char* name, const char* args_json,
	NvMagicToolExecFn func_cb, void* func_user_data, NvMagicFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	char** out);

// LLM lifecycle (converter callbacks are nullable NvMagicJsonCb function pointers)
// Option<fn_ptr> in Rust is ABI-compatible with a nullable function pointer.
// We define the Option struct explicitly so CGo can pass it by value correctly.
typedef char* (*NvMagicJsonCb)(void* user_data, const char* json);
typedef struct Option_NvMagicJsonCb { NvMagicJsonCb cb; } Option_NvMagicJsonCb;
typedef void (*NvMagicCollectorCb)(const char* chunk_json);
typedef struct Option_NvMagicCollectorCb { NvMagicCollectorCb cb; } Option_NvMagicCollectorCb;
typedef char* (*NvMagicFinalizerCb)();
typedef struct Option_NvMagicFinalizerCb { NvMagicFinalizerCb cb; } Option_NvMagicFinalizerCb;

static inline Option_NvMagicJsonCb makeOptJsonCb(NvMagicJsonCb cb) {
	Option_NvMagicJsonCb opt = { cb };
	return opt;
}
static inline Option_NvMagicCollectorCb makeOptCollectorCb(NvMagicCollectorCb cb) {
	Option_NvMagicCollectorCb opt = { cb };
	return opt;
}
static inline Option_NvMagicFinalizerCb makeOptFinalizerCb(NvMagicFinalizerCb cb) {
	Option_NvMagicFinalizerCb opt = { cb };
	return opt;
}

extern int32_t nvmagic_llm_call(const char* name, const char* native_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* model_name, Option_NvMagicJsonCb to_request_cb, void* to_request_ud, NvMagicFreeFn to_request_free, FfiLLMHandle** out);
extern int32_t nvmagic_llm_call_end(const FfiLLMHandle* handle, const char* response_json, const char* data_json, const char* metadata_json, Option_NvMagicJsonCb to_response_cb, void* to_response_ud, NvMagicFreeFn to_response_free);

// LLM call execute
typedef char* (*NvMagicLlmExecFn)(void* user_data, const char* native_json);
extern int32_t nvmagic_llm_call_execute(
	const char* name, const char* native_json,
	NvMagicLlmExecFn func_cb, void* func_user_data, NvMagicFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	Option_NvMagicJsonCb to_request_cb, void* to_request_ud, NvMagicFreeFn to_request_free,
	Option_NvMagicJsonCb to_response_cb, void* to_response_ud, NvMagicFreeFn to_response_free,
	char** out);

// LLM stream execute
extern int32_t nvmagic_llm_stream_call_execute(
	const char* name, const char* native_json,
	NvMagicLlmExecFn func_cb, void* func_user_data, NvMagicFreeFn func_free,
	Option_NvMagicCollectorCb collector, Option_NvMagicFinalizerCb finalizer,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	Option_NvMagicJsonCb to_request_cb, void* to_request_ud, NvMagicFreeFn to_request_free,
	Option_NvMagicJsonCb to_response_cb, void* to_response_ud, NvMagicFreeFn to_response_free,
	FfiStream** out);

// Tool guardrails
typedef char* (*NvMagicToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nvmagic_register_tool_sanitize_request_guardrail(const char* name, int32_t priority, NvMagicToolSanitizeFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_tool_sanitize_request_guardrail(const char* name);
extern int32_t nvmagic_register_tool_sanitize_response_guardrail(const char* name, int32_t priority, NvMagicToolSanitizeFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_tool_sanitize_response_guardrail(const char* name);

typedef char* (*NvMagicToolConditionalFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nvmagic_register_tool_conditional_execution_guardrail(const char* name, int32_t priority, NvMagicToolConditionalFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_tool_conditional_execution_guardrail(const char* name);

// Tool intercepts
extern int32_t nvmagic_register_tool_request_intercept(const char* name, int32_t priority, _Bool break_chain, NvMagicToolSanitizeFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_tool_request_intercept(const char* name);
extern int32_t nvmagic_register_tool_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvMagicToolSanitizeFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_tool_response_intercept(const char* name);

typedef _Bool (*NvMagicToolExecConditionalFn)(void* user_data, const char* name, const char* args_json);
// Middleware chain intercept callback types (must be declared before use in externs)
typedef char* (*NvMagicToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NvMagicToolExecInterceptCb)(void* user_data, const char* args_json, NvMagicToolExecNextFn next_fn, void* next_ctx);
extern int32_t nvmagic_register_tool_execution_intercept(const char* name, int32_t priority, NvMagicToolExecConditionalFn cond_cb, void* cond_user_data, NvMagicFreeFn cond_free, NvMagicToolExecInterceptCb exec_cb, void* exec_user_data, NvMagicFreeFn exec_free);
extern int32_t nvmagic_deregister_tool_execution_intercept(const char* name);

// LLM guardrails
typedef char* (*NvMagicLlmRequestFn)(void* user_data, const char* native_json);
extern int32_t nvmagic_register_llm_sanitize_request_guardrail(const char* name, int32_t priority, NvMagicLlmRequestFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_sanitize_request_guardrail(const char* name);

typedef char* (*NvMagicLlmResponseFn)(void* user_data, const char* response_json);
extern int32_t nvmagic_register_llm_sanitize_response_guardrail(const char* name, int32_t priority, NvMagicLlmResponseFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_sanitize_response_guardrail(const char* name);

typedef char* (*NvMagicJsonFn)(void* user_data, const char* json);

typedef char* (*NvMagicLlmConditionalFn)(void* user_data, const char* native_json);
extern int32_t nvmagic_register_llm_conditional_execution_guardrail(const char* name, int32_t priority, NvMagicLlmConditionalFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_conditional_execution_guardrail(const char* name);

// LLM intercepts
extern int32_t nvmagic_register_llm_request_intercept(const char* name, int32_t priority, _Bool break_chain, NvMagicJsonFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_request_intercept(const char* name);
extern int32_t nvmagic_register_llm_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvMagicLlmResponseFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_response_intercept(const char* name);

typedef char* (*NvMagicSseInterceptFn)(void* user_data, const char* chunk_json);
extern int32_t nvmagic_register_llm_stream_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvMagicSseInterceptFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_llm_stream_response_intercept(const char* name);

typedef _Bool (*NvMagicLlmExecConditionalFn)(void* user_data, const char* native_json);
typedef char* (*NvMagicLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NvMagicLlmExecInterceptCb)(void* user_data, const char* native_json, NvMagicLlmExecNextFn next_fn, void* next_ctx);

extern int32_t nvmagic_register_llm_execution_intercept(const char* name, int32_t priority, NvMagicLlmExecConditionalFn cond_cb, void* cond_user_data, NvMagicFreeFn cond_free, NvMagicLlmExecInterceptCb exec_cb, void* exec_user_data, NvMagicFreeFn exec_free);
extern int32_t nvmagic_deregister_llm_execution_intercept(const char* name);
extern int32_t nvmagic_register_llm_stream_execution_intercept(const char* name, int32_t priority, NvMagicLlmExecConditionalFn cond_cb, void* cond_user_data, NvMagicFreeFn cond_free, NvMagicLlmExecInterceptCb exec_cb, void* exec_user_data, NvMagicFreeFn exec_free);
extern int32_t nvmagic_deregister_llm_stream_execution_intercept(const char* name);

// Subscribers
typedef void (*NvMagicEventSubscriberFn)(void* user_data, const FfiEvent* event);
extern int32_t nvmagic_register_subscriber(const char* name, NvMagicEventSubscriberFn cb, void* user_data, NvMagicFreeFn free_fn);
extern int32_t nvmagic_deregister_subscriber(const char* name);

// Standalone middleware chains
extern int32_t nvmagic_tool_request_intercepts(const char* name, const char* args_json, char** out);
extern int32_t nvmagic_tool_conditional_execution(const char* name, const char* args_json);
extern int32_t nvmagic_tool_response_intercepts(const char* name, const char* result_json, char** out);
extern int32_t nvmagic_llm_request_intercepts(const char* request_json, char** out);
extern int32_t nvmagic_llm_conditional_execution(const char* request_json, Option_NvMagicJsonCb to_request_cb, void* to_request_ud, NvMagicFreeFn to_request_free);
extern int32_t nvmagic_llm_response_intercepts(const char* response_json, char** out);

// Error
extern const char* nvmagic_last_error();

// String free
extern void nvmagic_string_free(char* ptr);

// Scope stack isolation
extern int32_t nvmagic_scope_stack_create(FfiScopeStack** out);
extern int32_t nvmagic_scope_stack_set_thread(const FfiScopeStack* stack);
extern void nvmagic_scope_stack_free(FfiScopeStack* ptr);

// ATIF exporter
extern int32_t nvmagic_atif_exporter_create(const char*, const char*, const char*, const char*, void**);
extern int32_t nvmagic_atif_exporter_register(const void*, const char*);
extern int32_t nvmagic_atif_exporter_deregister(const char*);
extern int32_t nvmagic_atif_exporter_export(const void*, const char*, char**);
extern int32_t nvmagic_atif_exporter_clear(const void*);
extern void nvmagic_atif_exporter_free(void*);

// Go trampoline forward declarations (defined via //export in callbacks.go)
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern _Bool goToolExecConditionalTrampoline(void*, const char*, const char*);
extern char* goToolExecTrampoline(void*, const char*);
extern char* goJSONTrampoline(void*, const char*);
extern void goEventSubscriberTrampoline(void*, const FfiEvent*);
extern void goFreeTrampoline(void*);
extern char* goLlmRequestTrampoline(void*, const char*);
extern char* goLlmResponseTrampoline(void*, const char*);
extern char* goLlmConditionalTrampoline(void*, const char*);
extern _Bool goLlmExecConditionalTrampoline(void*, const char*);
extern char* goLlmExecTrampoline(void*, const char*);
extern char* goToolExecInterceptTrampoline(void*, const char*, NvMagicToolExecNextFn, void*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NvMagicLlmExecNextFn, void*);
extern char* goChunkInterceptTrampoline(void*, const char*);
extern char* goToRequestTrampoline(void*, const char*);
extern char* goToResponseTrampoline(void*, const char*);
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
	msg := C.nvmagic_last_error()
	if msg == nil {
		return errors.New("unknown nvmagic error")
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
}

// ScopeOption is a functional option that configures optional parameters for
// [PushScope]. Options are applied in the order they are passed. Available
// options include [WithParent] and [WithScopeAttributes].
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

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty (i.e., no scope has been
// pushed). The returned [ScopeHandle] is reference-counted and safe to hold
// beyond the lifetime of the scope itself.
func GetHandle() (*ScopeHandle, error) {
	var out *C.FfiScopeHandle
	status := C.nvmagic_get_handle(&out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newScopeHandle(out), nil
}

// PushScope creates a new scope and pushes it onto the hierarchical scope
// stack. The scope is assigned a unique UUID and emits a Start event to all
// registered subscribers. Use [PopScope] to end the scope. Optional parameters
// can be set via [WithParent] and [WithScopeAttributes].
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

	var out *C.FfiScopeHandle
	status := C.nvmagic_push_scope(cName, C.int32_t(scopeType), o.parent, C.uint32_t(o.attributes), &out)
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
	return checkStatus(C.nvmagic_pop_scope(handle.ptr))
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

	return checkStatus(C.nvmagic_event(cName, o.parent, o.data, o.metadata))
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
	status := C.nvmagic_tool_call(cName, cArgs, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.toolCallID, &out)
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

	return checkStatus(C.nvmagic_tool_call_end(handle.ptr, cResult, o.data, o.metadata))
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
	status := C.nvmagic_tool_call_execute(
		cName, cArgs,
		C.NvMagicToolExecFn(C.goToolExecTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nvmagic_string_free(out)
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

	// Per-call converter callbacks (nullable).
	toRequest    LLMToRequestFunc
	toRequestID  unsafe.Pointer // closure registry ID for toRequest
	toResponse   LLMToResponseFunc
	toResponseID unsafe.Pointer // closure registry ID for toResponse
}

// LLMCallOption is a functional option that configures optional parameters for
// LLM call functions ([LlmCall], [LlmCallEnd], [LlmCallExecute],
// [LlmStreamCallExecute]). Available options include [WithLLMParent],
// [WithLLMAttributes], [WithLLMData], and [WithLLMMetadata].
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

// WithLLMToRequest sets a per-call converter function that transforms native
// LLM JSON into an LLMRequest JSON representation. This converter is applied
// during call start ([LlmCall], [LlmCallExecute], [LlmStreamCallExecute]) and
// during conditional execution ([LlmConditionalExecution]) to convert the
// provider-specific request format into the canonical LLMRequest used by
// guardrails and intercepts. Pass nil or omit to use the default identity
// conversion.
func WithLLMToRequest(fn LLMToRequestFunc) LLMCallOption {
	return func(o *llmCallOptions) {
		o.toRequest = fn
		if fn != nil {
			o.toRequestID = registerClosure(fn)
		}
	}
}

// WithLLMToResponse sets a per-call converter function that transforms native
// LLM JSON into an LLMResponse JSON representation. This converter is applied
// during call end ([LlmCallEnd], [LlmCallExecute], [LlmStreamCallExecute])
// to convert the provider-specific response format into the canonical
// LLMResponse used by guardrails and intercepts. Pass nil or omit to use the
// default identity conversion.
func WithLLMToResponse(fn LLMToResponseFunc) LLMCallOption {
	return func(o *llmCallOptions) {
		o.toResponse = fn
		if fn != nil {
			o.toResponseID = registerClosure(fn)
		}
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
	// Note: converter callback closure IDs (toRequestID, toResponseID) are
	// NOT freed here. They are passed to C along with goFreeTrampoline, and
	// C calls the free function when it is done with the callback.
}

// toRequestCParams returns the C callback triple for the to_request converter.
// Returns a zero Option_NvMagicJsonCb (None) when no converter is set.
func (o *llmCallOptions) toRequestCParams() (C.Option_NvMagicJsonCb, unsafe.Pointer, C.NvMagicFreeFn) {
	if o.toRequest == nil {
		return C.makeOptJsonCb(nil), nil, nil
	}
	return C.makeOptJsonCb(C.NvMagicJsonCb(C.goToRequestTrampoline)), o.toRequestID, C.NvMagicFreeFn(C.goFreeTrampoline)
}

// toResponseCParams returns the C callback triple for the to_response converter.
// Returns a zero Option_NvMagicJsonCb (None) when no converter is set.
func (o *llmCallOptions) toResponseCParams() (C.Option_NvMagicJsonCb, unsafe.Pointer, C.NvMagicFreeFn) {
	if o.toResponse == nil {
		return C.makeOptJsonCb(nil), nil, nil
	}
	return C.makeOptJsonCb(C.NvMagicJsonCb(C.goToResponseTrampoline)), o.toResponseID, C.NvMagicFreeFn(C.goFreeTrampoline)
}

// LlmCall starts an LLM call lifecycle and returns an [LLMHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [LlmCallEnd] when the LLM responds. For a higher-level API that manages
// the full lifecycle automatically, use [LlmCallExecute] or
// [LlmStreamCallExecute] instead.
//
// The name identifies the LLM provider/model, and native is any JSON-serializable
// value representing the native LLM request payload. Optional parameters can be
// set via [LLMCallOption] values.
func LlmCall(name string, native interface{}, opts ...LLMCallOption) (*LLMHandle, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	nativeJSON, err := json.Marshal(native)
	if err != nil {
		return nil, err
	}

	cName := C.CString(name)
	cNative := C.CString(string(nativeJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cNative))

	toReqCb, toReqUD, toReqFree := o.toRequestCParams()

	var out *C.FfiLLMHandle
	status := C.nvmagic_llm_call(cName, cNative, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.modelName, toReqCb, toReqUD, toReqFree, &out)
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

	toRespCb, toRespUD, toRespFree := o.toResponseCParams()

	return checkStatus(C.nvmagic_llm_call_end(handle.ptr, cResponse, o.data, o.metadata, toRespCb, toRespUD, toRespFree))
}

// LlmCallExecute runs a complete LLM call lifecycle through the full
// middleware pipeline: conditional-execution guardrails (on raw request),
// request intercepts, sanitize-request guardrails, execution intercepts,
// the provided fn, response intercepts, sanitize-response guardrails.
// On rejection, only a standalone Mark event is emitted (no Start/End pair)
// and GuardrailRejected is returned. This is the recommended high-level API
// for non-streaming LLM invocations.
func LlmCallExecute(name string, native interface{}, fn LLMExecutionFunc, opts ...LLMCallOption) (json.RawMessage, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	nativeJSON, err := json.Marshal(native)
	if err != nil {
		return nil, err
	}

	id := registerClosure(fn)

	cName := C.CString(name)
	cNative := C.CString(string(nativeJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cNative))

	toReqCb, toReqUD, toReqFree := o.toRequestCParams()
	toRespCb, toRespUD, toRespFree := o.toResponseCParams()

	var out *C.char
	status := C.nvmagic_llm_call_execute(
		cName, cNative,
		C.NvMagicLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		toReqCb, toReqUD, toReqFree,
		toRespCb, toRespUD, toRespFree,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nvmagic_string_free(out)
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
func LlmStreamCallExecute(name string, native interface{}, fn LLMExecutionFunc, collector CollectorFunc, finalizer FinalizerFunc, opts ...LLMCallOption) (*LlmStream, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	nativeJSON, err := json.Marshal(native)
	if err != nil {
		return nil, err
	}

	id := registerClosure(fn)

	cName := C.CString(name)
	cNative := C.CString(string(nativeJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cNative))

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

	var cCollector C.Option_NvMagicCollectorCb
	if collector != nil {
		cCollector = C.makeOptCollectorCb(C.NvMagicCollectorCb(C.goCollectorTrampoline))
	} else {
		cCollector = C.makeOptCollectorCb(nil)
	}

	var cFinalizer C.Option_NvMagicFinalizerCb
	if finalizer != nil {
		cFinalizer = C.makeOptFinalizerCb(C.NvMagicFinalizerCb(C.goFinalizerTrampoline))
	} else {
		cFinalizer = C.makeOptFinalizerCb(nil)
	}

	toReqCb, toReqUD, toReqFree := o.toRequestCParams()
	toRespCb, toRespUD, toRespFree := o.toResponseCParams()

	var out *C.FfiStream
	status := C.nvmagic_llm_stream_call_execute(
		cName, cNative,
		C.NvMagicLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		cCollector,
		cFinalizer,
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		toReqCb, toReqUD, toReqFree,
		toRespCb, toRespUD, toRespFree,
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
	return checkStatus(C.nvmagic_register_tool_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeRequestGuardrail removes a previously registered tool
// sanitize-request guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_sanitize_request_guardrail(cName))
}

// RegisterToolSanitizeResponseGuardrail registers a guardrail that sanitizes
// tool response data before it is returned to the caller. The callback receives
// the tool name and response JSON and must return the (possibly modified)
// response. Guardrails are invoked in priority order (lower values run first).
func RegisterToolSanitizeResponseGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_tool_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeResponseGuardrail removes a previously registered tool
// sanitize-response guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nvmagic_register_tool_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicToolConditionalFn(C.goToolConditionalTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolConditionalExecutionGuardrail removes a previously registered
// tool conditional-execution guardrail by name. Returns a NotFound error if no
// guardrail with the given name is registered.
func DeregisterToolConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_conditional_execution_guardrail(cName))
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
	return checkStatus(C.nvmagic_register_tool_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvMagicToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolRequestIntercept removes a previously registered tool request
// intercept by name.
func DeregisterToolRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_request_intercept(cName))
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
	return checkStatus(C.nvmagic_register_tool_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvMagicToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolResponseIntercept removes a previously registered tool response
// intercept by name.
func DeregisterToolResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_response_intercept(cName))
}

// RegisterToolExecutionIntercept registers an execution intercept following
// the middleware chain pattern. The condFn callback is evaluated first; if it
// returns true, execFn is called with the args and a `next` function. Call
// `next` to invoke the next intercept or original implementation; skip calling
// `next` to short-circuit the chain.
func RegisterToolExecutionIntercept(name string, priority int32, condFn ToolExecConditionalFunc, execFn ToolExecutionInterceptFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_tool_execution_intercept(
		cName, C.int32_t(priority),
		C.NvMagicToolExecConditionalFn(C.goToolExecConditionalTrampoline),
		condID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		C.NvMagicToolExecInterceptCb(C.goToolExecInterceptTrampoline),
		execID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolExecutionIntercept removes a previously registered tool
// execution intercept by name.
func DeregisterToolExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_tool_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (LLM)
// ---------------------------------------------------------------------------

// RegisterLlmSanitizeRequestGuardrail registers a guardrail that sanitizes LLM
// request JSON before the call is made. The callback receives the native request
// JSON and must return the (possibly modified) JSON. Guardrails are invoked in
// priority order (lower values run first).
func RegisterLlmSanitizeRequestGuardrail(name string, priority int32, fn JSONFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicLlmRequestFn(C.goLlmRequestTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeRequestGuardrail removes a previously registered LLM
// sanitize-request guardrail by name.
func DeregisterLlmSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_sanitize_request_guardrail(cName))
}

// RegisterLlmSanitizeResponseGuardrail registers a guardrail that sanitizes
// LLM response data before it is returned to the caller. The callback receives
// the serialized LLMResponse JSON (containing a "data" field) and must return
// the (possibly modified) response JSON. Guardrails are invoked in priority
// order (lower values run first).
func RegisterLlmSanitizeResponseGuardrail(name string, priority int32, fn LLMResponseFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicLlmResponseFn(C.goLlmResponseTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeResponseGuardrail removes a previously registered LLM
// sanitize-response guardrail by name.
func DeregisterLlmSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nvmagic_register_llm_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NvMagicLlmConditionalFn(C.goLlmConditionalTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmConditionalExecutionGuardrail removes a previously registered
// LLM conditional-execution guardrail by name.
func DeregisterLlmConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_conditional_execution_guardrail(cName))
}

// RegisterLlmRequestIntercept registers an intercept that transforms the LLM
// request JSON before the call is made. Intercepts run in priority order (lower
// values first). When breakChain is true, no lower-priority intercepts in the
// chain are invoked after this one.
func RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn JSONFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvMagicJsonFn(C.goJSONTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmRequestIntercept removes a previously registered LLM request
// intercept by name.
func DeregisterLlmRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_request_intercept(cName))
}

// RegisterLlmResponseIntercept registers an intercept that transforms the LLM
// response after the LLM returns. The callback receives the serialized
// LLMResponse JSON (containing a "data" field) and must return the (possibly
// modified) response JSON. Intercepts run in priority order (lower values
// first). When breakChain is true, no lower-priority intercepts in the chain
// are invoked after this one.
func RegisterLlmResponseIntercept(name string, priority int32, breakChain bool, fn LLMResponseFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvMagicLlmResponseFn(C.goLlmResponseTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmResponseIntercept removes a previously registered LLM response
// intercept by name.
func DeregisterLlmResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_response_intercept(cName))
}

// RegisterLlmStreamResponseIntercept registers an intercept that transforms
// individual chunks during a streaming LLM response. The callback receives
// each chunk as JSON and must return the (possibly modified) chunk JSON.
// Intercepts run in priority order (lower values first). When breakChain is
// true, no lower-priority intercepts are invoked after this one.
func RegisterLlmStreamResponseIntercept(name string, priority int32, breakChain bool, fn ChunkInterceptFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_stream_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvMagicSseInterceptFn(C.goChunkInterceptTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamResponseIntercept removes a previously registered LLM
// stream response intercept by name.
func DeregisterLlmStreamResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_stream_response_intercept(cName))
}

// RegisterLlmExecutionIntercept registers an execution intercept following
// the middleware chain pattern. The condFn callback is evaluated first; if it
// returns true, execFn is called with the request parameters and a `next`
// function. Call `next` to invoke the next intercept or original
// implementation; skip calling `next` to short-circuit the chain.
func RegisterLlmExecutionIntercept(name string, priority int32, condFn LLMExecConditionalFunc, execFn LLMExecutionInterceptFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_execution_intercept(
		cName, C.int32_t(priority),
		C.NvMagicLlmExecConditionalFn(C.goLlmExecConditionalTrampoline),
		condID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		C.NvMagicLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmExecutionIntercept removes a previously registered LLM
// execution intercept by name.
func DeregisterLlmExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_execution_intercept(cName))
}

// RegisterLlmStreamExecutionIntercept registers an execution intercept for
// streaming LLM calls following the middleware chain pattern. The condFn
// callback is evaluated first; if it returns true, execFn is called with the
// request parameters and a `next` function. Call `next` to invoke the next
// intercept or original implementation; skip calling `next` to short-circuit.
func RegisterLlmStreamExecutionIntercept(name string, priority int32, condFn LLMExecConditionalFunc, execFn LLMExecutionInterceptFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_register_llm_stream_execution_intercept(
		cName, C.int32_t(priority),
		C.NvMagicLlmExecConditionalFn(C.goLlmExecConditionalTrampoline),
		condID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
		C.NvMagicLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamExecutionIntercept removes a previously registered LLM
// stream execution intercept by name.
func DeregisterLlmStreamExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_llm_stream_execution_intercept(cName))
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
	return checkStatus(C.nvmagic_register_subscriber(
		cName,
		C.NvMagicEventSubscriberFn(C.goEventSubscriberTrampoline),
		id,
		C.NvMagicFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterSubscriber removes a named event subscriber. Returns a NotFound
// error if no subscriber with the given name is registered.
func DeregisterSubscriber(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvmagic_deregister_subscriber(cName))
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
	status := C.nvmagic_scope_stack_create(&ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &ScopeStack{ptr: ptr}, nil
}

// Close frees the scope stack. After calling Close, the ScopeStack must not be used.
func (s *ScopeStack) Close() {
	if s.ptr != nil {
		C.nvmagic_scope_stack_free(s.ptr)
		s.ptr = nil
	}
}

// Run binds this scope stack to the current OS thread and executes fn.
// The calling goroutine is locked to the OS thread for the duration of fn.
// All NVMagic scope operations within fn will use this scope stack.
func (s *ScopeStack) Run(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.nvmagic_scope_stack_set_thread(s.ptr)
	fn()
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
	status := C.nvmagic_atif_exporter_create(cSessionID, cAgentName, cAgentVersion, cModelName, &ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &AtifExporter{ptr: ptr}, nil
}

// Register registers the exporter as an event subscriber with the given name.
func (e *AtifExporter) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nvmagic_atif_exporter_register(e.ptr, cName)
	return checkStatus(status)
}

// Deregister removes the exporter subscriber by name.
func (e *AtifExporter) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nvmagic_atif_exporter_deregister(cName)
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
	status := C.nvmagic_atif_exporter_export(e.ptr, cRootUUID, &cOut)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nvmagic_string_free(cOut)
	return json.RawMessage(C.GoString(cOut)), nil
}

// Clear removes all collected events.
func (e *AtifExporter) Clear() {
	C.nvmagic_atif_exporter_clear(e.ptr)
}

// Close frees the exporter handle.
func (e *AtifExporter) Close() {
	if e.ptr != nil {
		C.nvmagic_atif_exporter_free(e.ptr)
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
	status := C.nvmagic_tool_request_intercepts(cName, cArgs, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nvmagic_string_free(out)
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

	status := C.nvmagic_tool_conditional_execution(cName, cArgs)
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
	status := C.nvmagic_tool_response_intercepts(cName, cResult, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nvmagic_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// LlmRequestIntercepts runs the registered LLM request intercept chain on the
// given request (serialized as JSON) and returns the transformed request JSON.
func LlmRequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.char
	status := C.nvmagic_llm_request_intercepts(cRequest, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nvmagic_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all guardrails pass, or an error with the
// rejection reason if blocked. Optional [LLMCallOption] values can supply a
// [WithLLMToRequest] converter for the guardrail to operate on.
func LlmConditionalExecution(request json.RawMessage, opts ...LLMCallOption) error {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	toReqCb, toReqUD, toReqFree := o.toRequestCParams()

	status := C.nvmagic_llm_conditional_execution(cRequest, toReqCb, toReqUD, toReqFree)
	return checkStatus(status)
}

// LlmResponseIntercepts runs the registered LLM response intercept chain on
// the given response and returns the transformed response. The response JSON
// should be a serialized LLMResponse (containing a "data" field).
func LlmResponseIntercepts(response json.RawMessage) (json.RawMessage, error) {
	cResponse := C.CString(string(response))
	defer C.free(unsafe.Pointer(cResponse))

	var out *C.char
	status := C.nvmagic_llm_response_intercepts(cResponse, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nvmagic_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}
