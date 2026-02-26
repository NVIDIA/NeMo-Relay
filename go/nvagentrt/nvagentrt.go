// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nvagentrt provides Go bindings for the NVAgentRT agent runtime via CGo.
//
// NVAgentRT is a multi-language agent runtime framework that provides execution
// scope management, lifecycle events, and middleware (guardrails and intercepts)
// for tool and LLM calls. The core runtime is written in Rust; this package
// wraps the C FFI layer produced by the nvagentrt-ffi crate.
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
// Build prerequisites: the nvagentrt-ffi shared library must be built first
// (cargo build --release -p nvagentrt-ffi) and discoverable via CGO_LDFLAGS.
package nvagentrt

/*
#cgo LDFLAGS: -lnvagentrt_ffi
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

typedef void (*NvAgentRtFreeFn)(void* user_data);

// Core API
extern int32_t nvagentrt_get_handle(FfiScopeHandle** out);
extern int32_t nvagentrt_push_scope(const char* name, int32_t scope_type, const FfiScopeHandle* parent, uint32_t attributes, FfiScopeHandle** out);
extern int32_t nvagentrt_pop_scope(const FfiScopeHandle* handle);
extern int32_t nvagentrt_event(const char* name, const FfiScopeHandle* parent, const char* data_json, const char* metadata_json);

// Tool lifecycle
extern int32_t nvagentrt_tool_call(const char* name, const char* args_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, FfiToolHandle** out);
extern int32_t nvagentrt_tool_call_end(const FfiToolHandle* handle, const char* result_json, const char* data_json, const char* metadata_json);

// Tool call execute (with C function pointer callbacks)
typedef char* (*NvAgentRtToolExecFn)(void* user_data, const char* args_json);
extern int32_t nvagentrt_tool_call_execute(
	const char* name, const char* args_json,
	NvAgentRtToolExecFn func_cb, void* func_user_data, NvAgentRtFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	char** out);

// LLM lifecycle
extern int32_t nvagentrt_llm_call(const char* name, const FfiLLMRequest* request, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, FfiLLMHandle** out);
extern int32_t nvagentrt_llm_call_end(const FfiLLMHandle* handle, const char* response_json, const char* data_json, const char* metadata_json);

// LLM call execute
typedef char* (*NvAgentRtLlmExecFn)(void* user_data, const FfiLLMRequest* request);
extern int32_t nvagentrt_llm_call_execute(
	const char* name, const FfiLLMRequest* request,
	NvAgentRtLlmExecFn func_cb, void* func_user_data, NvAgentRtFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	char** out);

// LLM stream execute
extern int32_t nvagentrt_llm_stream_call_execute(
	const char* name, const FfiLLMRequest* request,
	NvAgentRtLlmExecFn func_cb, void* func_user_data, NvAgentRtFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	FfiStream** out);

// Tool guardrails
typedef char* (*NvAgentRtToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nvagentrt_register_tool_sanitize_request_guardrail(const char* name, int32_t priority, NvAgentRtToolSanitizeFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_tool_sanitize_request_guardrail(const char* name);
extern int32_t nvagentrt_register_tool_sanitize_response_guardrail(const char* name, int32_t priority, NvAgentRtToolSanitizeFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_tool_sanitize_response_guardrail(const char* name);

typedef char* (*NvAgentRtToolConditionalFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nvagentrt_register_tool_conditional_execution_guardrail(const char* name, int32_t priority, NvAgentRtToolConditionalFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_tool_conditional_execution_guardrail(const char* name);

// Tool intercepts
extern int32_t nvagentrt_register_tool_request_intercept(const char* name, int32_t priority, _Bool break_chain, NvAgentRtToolSanitizeFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_tool_request_intercept(const char* name);
extern int32_t nvagentrt_register_tool_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvAgentRtToolSanitizeFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_tool_response_intercept(const char* name);

typedef _Bool (*NvAgentRtToolExecConditionalFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nvagentrt_register_tool_execution_intercept(const char* name, int32_t priority, NvAgentRtToolExecConditionalFn cond_cb, void* cond_user_data, NvAgentRtFreeFn cond_free, NvAgentRtToolExecFn exec_cb, void* exec_user_data, NvAgentRtFreeFn exec_free);
extern int32_t nvagentrt_deregister_tool_execution_intercept(const char* name);

// LLM guardrails
typedef FfiLLMRequest* (*NvAgentRtLlmRequestFn)(void* user_data, const FfiLLMRequest* request);
extern int32_t nvagentrt_register_llm_sanitize_request_guardrail(const char* name, int32_t priority, NvAgentRtLlmRequestFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_sanitize_request_guardrail(const char* name);

typedef char* (*NvAgentRtJsonFn)(void* user_data, const char* json);
extern int32_t nvagentrt_register_llm_sanitize_response_guardrail(const char* name, int32_t priority, NvAgentRtJsonFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_sanitize_response_guardrail(const char* name);

typedef char* (*NvAgentRtLlmConditionalFn)(void* user_data, const FfiLLMRequest* request);
extern int32_t nvagentrt_register_llm_conditional_execution_guardrail(const char* name, int32_t priority, NvAgentRtLlmConditionalFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_conditional_execution_guardrail(const char* name);

// LLM intercepts
extern int32_t nvagentrt_register_llm_request_intercept(const char* name, int32_t priority, _Bool break_chain, NvAgentRtLlmRequestFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_request_intercept(const char* name);
extern int32_t nvagentrt_register_llm_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvAgentRtJsonFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_response_intercept(const char* name);

typedef char* (*NvAgentRtSseInterceptFn)(void* user_data, const char* sse_json);
extern int32_t nvagentrt_register_llm_stream_response_intercept(const char* name, int32_t priority, _Bool break_chain, NvAgentRtSseInterceptFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_llm_stream_response_intercept(const char* name);

typedef _Bool (*NvAgentRtLlmExecConditionalFn)(void* user_data, const FfiLLMRequest* request);
extern int32_t nvagentrt_register_llm_execution_intercept(const char* name, int32_t priority, NvAgentRtLlmExecConditionalFn cond_cb, void* cond_user_data, NvAgentRtFreeFn cond_free, NvAgentRtLlmExecFn exec_cb, void* exec_user_data, NvAgentRtFreeFn exec_free);
extern int32_t nvagentrt_deregister_llm_execution_intercept(const char* name);
extern int32_t nvagentrt_register_llm_stream_execution_intercept(const char* name, int32_t priority, NvAgentRtLlmExecConditionalFn cond_cb, void* cond_user_data, NvAgentRtFreeFn cond_free, NvAgentRtLlmExecFn exec_cb, void* exec_user_data, NvAgentRtFreeFn exec_free);
extern int32_t nvagentrt_deregister_llm_stream_execution_intercept(const char* name);

// Subscribers
typedef void (*NvAgentRtEventSubscriberFn)(void* user_data, const FfiEvent* event);
extern int32_t nvagentrt_register_subscriber(const char* name, NvAgentRtEventSubscriberFn cb, void* user_data, NvAgentRtFreeFn free_fn);
extern int32_t nvagentrt_deregister_subscriber(const char* name);

// Error
extern const char* nvagentrt_last_error();

// String free
extern void nvagentrt_string_free(char* ptr);

// Scope stack isolation
extern int32_t nvagentrt_scope_stack_create(FfiScopeStack** out);
extern int32_t nvagentrt_scope_stack_set_thread(const FfiScopeStack* stack);
extern void nvagentrt_scope_stack_free(FfiScopeStack* ptr);

// Go trampoline forward declarations (defined via //export in callbacks.go)
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern _Bool goToolExecConditionalTrampoline(void*, const char*, const char*);
extern char* goToolExecTrampoline(void*, const char*);
extern char* goJSONTrampoline(void*, const char*);
extern void goEventSubscriberTrampoline(void*, const FfiEvent*);
extern void goFreeTrampoline(void*);
extern FfiLLMRequest* goLlmRequestTrampoline(void*, const FfiLLMRequest*);
extern char* goLlmConditionalTrampoline(void*, const FfiLLMRequest*);
extern _Bool goLlmExecConditionalTrampoline(void*, const FfiLLMRequest*);
extern char* goLlmExecTrampoline(void*, const FfiLLMRequest*);
extern char* goSseInterceptTrampoline(void*, const char*);
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
	msg := C.nvagentrt_last_error()
	if msg == nil {
		return errors.New("unknown nvagentrt error")
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
	status := C.nvagentrt_get_handle(&out)
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
	status := C.nvagentrt_push_scope(cName, C.int32_t(scopeType), o.parent, C.uint32_t(o.attributes), &out)
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
	return checkStatus(C.nvagentrt_pop_scope(handle.ptr))
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

	return checkStatus(C.nvagentrt_event(cName, o.parent, o.data, o.metadata))
}

// ---------------------------------------------------------------------------
// Tool lifecycle options
// ---------------------------------------------------------------------------

type toolCallOptions struct {
	parent     *C.FfiScopeHandle
	attributes uint32
	data       *C.char
	metadata   *C.char
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

func freeToolOpts(o *toolCallOptions) {
	if o.data != nil {
		C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		C.free(unsafe.Pointer(o.metadata))
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
	status := C.nvagentrt_tool_call(cName, cArgs, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, &out)
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

	return checkStatus(C.nvagentrt_tool_call_end(handle.ptr, cResult, o.data, o.metadata))
}

// ToolCallExecute runs a complete tool call lifecycle: it emits a Start event,
// runs the full middleware pipeline (request intercepts, sanitize-request
// guardrails, conditional-execution guardrails, execution intercepts, the
// provided fn, response intercepts, sanitize-response guardrails), emits an
// End event, and returns the final result JSON. This is the recommended
// high-level API for tool invocations.
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
	status := C.nvagentrt_tool_call_execute(
		cName, cArgs,
		C.NvAgentRtToolExecFn(C.goToolExecTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nvagentrt_string_free(out)
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

func freeLLMOpts(o *llmCallOptions) {
	if o.data != nil {
		C.free(unsafe.Pointer(o.data))
	}
	if o.metadata != nil {
		C.free(unsafe.Pointer(o.metadata))
	}
}

// LlmCall starts an LLM call lifecycle and returns an [LLMHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [LlmCallEnd] when the LLM responds. For a higher-level API that manages
// the full lifecycle automatically, use [LlmCallExecute] or
// [LlmStreamCallExecute] instead.
//
// The name identifies the LLM provider/model, and request contains the HTTP
// request parameters. Optional parameters can be set via [LLMCallOption] values.
func LlmCall(name string, request *LLMRequest, opts ...LLMCallOption) (*LLMHandle, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var out *C.FfiLLMHandle
	status := C.nvagentrt_llm_call(cName, request.ptr, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, &out)
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

	return checkStatus(C.nvagentrt_llm_call_end(handle.ptr, cResponse, o.data, o.metadata))
}

// LlmCallExecute runs a complete LLM call lifecycle: it emits a Start event,
// runs the full middleware pipeline (request intercepts, sanitize-request
// guardrails, conditional-execution guardrails, execution intercepts, the
// provided fn, response intercepts, sanitize-response guardrails), emits an
// End event, and returns the final response JSON. This is the recommended
// high-level API for non-streaming LLM invocations.
func LlmCallExecute(name string, request *LLMRequest, fn LLMExecutionFunc, opts ...LLMCallOption) (json.RawMessage, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	id := registerClosure(fn)

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var out *C.char
	status := C.nvagentrt_llm_call_execute(
		cName, request.ptr,
		C.NvAgentRtLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		&out,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nvagentrt_string_free(out)
	return result, nil
}

// LlmStreamCallExecute runs a streaming LLM call lifecycle. Like
// [LlmCallExecute], it runs the full middleware pipeline, but instead of
// returning a single response JSON, it returns an [LlmStream] that yields
// individual SSE (Server-Sent Event) chunks. Stream response intercepts are
// applied to each chunk as it is consumed. The caller must call [LlmStream.Next]
// repeatedly until [io.EOF] is returned, then call [LlmStream.Close].
func LlmStreamCallExecute(name string, request *LLMRequest, fn LLMExecutionFunc, opts ...LLMCallOption) (*LlmStream, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	id := registerClosure(fn)

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var out *C.FfiStream
	status := C.nvagentrt_llm_stream_call_execute(
		cName, request.ptr,
		C.NvAgentRtLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
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
	return checkStatus(C.nvagentrt_register_tool_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeRequestGuardrail removes a previously registered tool
// sanitize-request guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_sanitize_request_guardrail(cName))
}

// RegisterToolSanitizeResponseGuardrail registers a guardrail that sanitizes
// tool response data before it is returned to the caller. The callback receives
// the tool name and response JSON and must return the (possibly modified)
// response. Guardrails are invoked in priority order (lower values run first).
func RegisterToolSanitizeResponseGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_tool_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeResponseGuardrail removes a previously registered tool
// sanitize-response guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nvagentrt_register_tool_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtToolConditionalFn(C.goToolConditionalTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolConditionalExecutionGuardrail removes a previously registered
// tool conditional-execution guardrail by name. Returns a NotFound error if no
// guardrail with the given name is registered.
func DeregisterToolConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_conditional_execution_guardrail(cName))
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
	return checkStatus(C.nvagentrt_register_tool_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvAgentRtToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolRequestIntercept removes a previously registered tool request
// intercept by name.
func DeregisterToolRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_request_intercept(cName))
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
	return checkStatus(C.nvagentrt_register_tool_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvAgentRtToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolResponseIntercept removes a previously registered tool response
// intercept by name.
func DeregisterToolResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_response_intercept(cName))
}

// RegisterToolExecutionIntercept registers an intercept that can replace tool
// execution entirely. The condFn callback is evaluated first; if it returns
// true, execFn is called instead of the original tool implementation.
func RegisterToolExecutionIntercept(name string, priority int32, condFn ToolExecConditionalFunc, execFn ToolExecutionFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_tool_execution_intercept(
		cName, C.int32_t(priority),
		C.NvAgentRtToolExecConditionalFn(C.goToolExecConditionalTrampoline),
		condID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		C.NvAgentRtToolExecFn(C.goToolExecTrampoline),
		execID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolExecutionIntercept removes a previously registered tool
// execution intercept by name.
func DeregisterToolExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_tool_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (LLM)
// ---------------------------------------------------------------------------

// RegisterLlmSanitizeRequestGuardrail registers a guardrail that sanitizes LLM
// request parameters (HTTP method, URL, headers, and body) before the call is
// made. The callback receives these fields and must return the (possibly
// modified) versions. Guardrails are invoked in priority order (lower values
// run first).
func RegisterLlmSanitizeRequestGuardrail(name string, priority int32, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtLlmRequestFn(C.goLlmRequestTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeRequestGuardrail removes a previously registered LLM
// sanitize-request guardrail by name.
func DeregisterLlmSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_sanitize_request_guardrail(cName))
}

// RegisterLlmSanitizeResponseGuardrail registers a guardrail that sanitizes
// LLM response JSON before it is returned to the caller. The callback receives
// the response JSON and must return the (possibly modified) JSON. Guardrails are
// invoked in priority order (lower values run first).
func RegisterLlmSanitizeResponseGuardrail(name string, priority int32, fn JSONFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtJsonFn(C.goJSONTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeResponseGuardrail removes a previously registered LLM
// sanitize-response guardrail by name.
func DeregisterLlmSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nvagentrt_register_llm_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NvAgentRtLlmConditionalFn(C.goLlmConditionalTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmConditionalExecutionGuardrail removes a previously registered
// LLM conditional-execution guardrail by name.
func DeregisterLlmConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_conditional_execution_guardrail(cName))
}

// RegisterLlmRequestIntercept registers an intercept that transforms LLM
// request parameters (HTTP method, URL, headers, body) before the call is made.
// Intercepts run in priority order (lower values first). When breakChain is
// true, no lower-priority intercepts in the chain are invoked after this one.
func RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvAgentRtLlmRequestFn(C.goLlmRequestTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmRequestIntercept removes a previously registered LLM request
// intercept by name.
func DeregisterLlmRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_request_intercept(cName))
}

// RegisterLlmResponseIntercept registers an intercept that transforms LLM
// response JSON after the LLM returns. Intercepts run in priority order (lower
// values first). When breakChain is true, no lower-priority intercepts in the
// chain are invoked after this one.
func RegisterLlmResponseIntercept(name string, priority int32, breakChain bool, fn JSONFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvAgentRtJsonFn(C.goJSONTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmResponseIntercept removes a previously registered LLM response
// intercept by name.
func DeregisterLlmResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_response_intercept(cName))
}

// RegisterLlmStreamResponseIntercept registers an intercept that transforms
// individual Server-Sent Event (SSE) chunks during a streaming LLM response.
// The callback receives each SSE event as JSON and must return the (possibly
// modified) event. Intercepts run in priority order (lower values first). When
// breakChain is true, no lower-priority intercepts are invoked after this one.
func RegisterLlmStreamResponseIntercept(name string, priority int32, breakChain bool, fn SseInterceptFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_stream_response_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NvAgentRtSseInterceptFn(C.goSseInterceptTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamResponseIntercept removes a previously registered LLM
// stream response intercept by name.
func DeregisterLlmStreamResponseIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_stream_response_intercept(cName))
}

// RegisterLlmExecutionIntercept registers an intercept that can replace LLM
// execution entirely. The condFn callback is evaluated first; if it returns
// true, execFn is called instead of the original LLM implementation.
func RegisterLlmExecutionIntercept(name string, priority int32, condFn LLMExecConditionalFunc, execFn LLMExecutionFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_execution_intercept(
		cName, C.int32_t(priority),
		C.NvAgentRtLlmExecConditionalFn(C.goLlmExecConditionalTrampoline),
		condID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		C.NvAgentRtLlmExecFn(C.goLlmExecTrampoline),
		execID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmExecutionIntercept removes a previously registered LLM
// execution intercept by name.
func DeregisterLlmExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_execution_intercept(cName))
}

// RegisterLlmStreamExecutionIntercept registers an intercept that can replace
// streaming LLM execution entirely. The condFn callback is evaluated first;
// if it returns true, execFn is called instead of the original implementation.
func RegisterLlmStreamExecutionIntercept(name string, priority int32, condFn LLMExecConditionalFunc, execFn LLMExecutionFunc) error {
	condID := registerClosure(condFn)
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_register_llm_stream_execution_intercept(
		cName, C.int32_t(priority),
		C.NvAgentRtLlmExecConditionalFn(C.goLlmExecConditionalTrampoline),
		condID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
		C.NvAgentRtLlmExecFn(C.goLlmExecTrampoline),
		execID,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamExecutionIntercept removes a previously registered LLM
// stream execution intercept by name.
func DeregisterLlmStreamExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_llm_stream_execution_intercept(cName))
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
	return checkStatus(C.nvagentrt_register_subscriber(
		cName,
		C.NvAgentRtEventSubscriberFn(C.goEventSubscriberTrampoline),
		id,
		C.NvAgentRtFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterSubscriber removes a named event subscriber. Returns a NotFound
// error if no subscriber with the given name is registered.
func DeregisterSubscriber(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nvagentrt_deregister_subscriber(cName))
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
	status := C.nvagentrt_scope_stack_create(&ptr)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &ScopeStack{ptr: ptr}, nil
}

// Close frees the scope stack. After calling Close, the ScopeStack must not be used.
func (s *ScopeStack) Close() {
	if s.ptr != nil {
		C.nvagentrt_scope_stack_free(s.ptr)
		s.ptr = nil
	}
}

// Run binds this scope stack to the current OS thread and executes fn.
// The calling goroutine is locked to the OS thread for the duration of fn.
// All NVAgentRT scope operations within fn will use this scope stack.
func (s *ScopeStack) Run(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.nvagentrt_scope_stack_set_thread(s.ptr)
	fn()
}
