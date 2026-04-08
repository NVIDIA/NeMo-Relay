// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// callbacks.go defines the Go callback type aliases used by the Nexus
// middleware and subscriber systems, and the CGo trampoline functions that
// bridge Go closures to C function pointers.
//
// The trampoline mechanism works as follows: when a Go closure is registered
// (e.g., via [RegisterToolSanitizeRequestGuardrail]), it is stored in a
// global closure registry keyed by a monotonically increasing integer ID.
// That ID is passed as the void* user_data parameter to the C FFI. When the
// C side invokes the callback, the corresponding //export trampoline function
// is called, which looks up the closure by ID and invokes it with Go-native
// arguments. The goFreeTrampoline is called by the C side when the callback
// is deregistered, removing the closure from the registry.

package nat_nexus

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;

typedef void (*NatNexusFreeFn)(void* user_data);
typedef char* (*NatNexusToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NatNexusToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NatNexusToolExecFn)(void* user_data, const char* args_json);
typedef FfiLLMRequest* (*NatNexusLlmRequestCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NatNexusLlmConditionalCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NatNexusLlmExecFn)(void* user_data, const char* native_json);
typedef char* (*NatNexusLlmResponseFn)(void* user_data, const char* response_json);
typedef void (*NatNexusEventSubscriberFn)(void* user_data, const FfiEvent* event);

// Middleware chain next function types
typedef char* (*NatNexusToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NatNexusToolExecInterceptCb)(void* user_data, const char* args_json, NatNexusToolExecNextFn next_fn, void* next_ctx);
typedef char* (*NatNexusLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NatNexusLlmExecInterceptCb)(void* user_data, const char* native_json, NatNexusLlmExecNextFn next_fn, void* next_ctx);

// Helper to call the tool exec next function pointer from Go
static inline char* callToolExecNext(NatNexusToolExecNextFn next_fn, const char* args_json, void* next_ctx) {
	return next_fn(args_json, next_ctx);
}

// Helper to call the LLM exec next function pointer from Go
static inline char* callLlmExecNext(NatNexusLlmExecNextFn next_fn, const char* native_json, void* next_ctx) {
	return next_fn(native_json, next_ctx);
}

// LLMRequest accessors (also declared in types.go, needed here for trampolines)
extern FfiLLMRequest* nat_nexus_llm_request_new(const char* headers_json, const char* content_json);
extern char* nat_nexus_llm_request_headers(const FfiLLMRequest* ptr);
extern char* nat_nexus_llm_request_content(const FfiLLMRequest* ptr);
extern void nat_nexus_string_free(char* ptr);
extern void nat_nexus_set_last_error_message(const char* msg);
*/
import "C"

import (
	"encoding/json"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Global closure registry: maps integer IDs to Go closures.
// The ID is passed as void* user_data to C callbacks.
// ---------------------------------------------------------------------------

var (
	closureRegistryMu sync.Mutex
	closureRegistry   = make(map[uintptr]interface{})
	closureNextID     atomic.Uint64
)

func setLastErrorMessage(msg string) {
	cMsg := C.CString(msg)
	defer C.free(unsafe.Pointer(cMsg))
	C.nat_nexus_set_last_error_message(cMsg)
}

// registerClosure stores fn in the global registry and returns an
// unsafe.Pointer that encodes the registry key. The returned pointer is
// suitable for passing as void* user_data to C callbacks.
func registerClosure(fn interface{}) unsafe.Pointer {
	id := uintptr(closureNextID.Add(1))
	closureRegistryMu.Lock()
	closureRegistry[id] = fn
	closureRegistryMu.Unlock()

	// Allocate the callback token in C-owned memory so we don't pass a Go
	// pointer through C and can release it explicitly on deregistration.
	p := (*uintptr)(C.malloc(C.size_t(unsafe.Sizeof(uintptr(0)))))
	if p == nil {
		panic("nat_nexus: failed to allocate callback token")
	}
	*p = id
	return unsafe.Pointer(p)
}

func closureID(userData unsafe.Pointer) uintptr {
	return *(*uintptr)(userData)
}

func lookupClosure(userData unsafe.Pointer) interface{} {
	id := closureID(userData)
	closureRegistryMu.Lock()
	fn := closureRegistry[id]
	closureRegistryMu.Unlock()
	return fn
}

func unregisterClosure(userData unsafe.Pointer) {
	id := closureID(userData)
	closureRegistryMu.Lock()
	delete(closureRegistry, id)
	closureRegistryMu.Unlock()
	C.free(userData)
}

// ---------------------------------------------------------------------------
// Go callback type definitions
// ---------------------------------------------------------------------------

// ToolSanitizeFunc is a callback that receives a tool name and its arguments
// as JSON, and returns the (possibly modified) arguments. It is used by both
// sanitize guardrails and request intercepts for tools.
type ToolSanitizeFunc func(name string, args json.RawMessage) json.RawMessage

// ToolConditionalFunc is a callback that decides whether a tool call should
// proceed. It returns nil to allow execution, or a non-nil pointer to an error
// message string to reject the call.
type ToolConditionalFunc func(name string, args json.RawMessage) *string

// ToolExecutionFunc is a callback that executes a tool call, receiving the
// arguments as JSON and returning the result JSON or an error.
type ToolExecutionFunc func(args json.RawMessage) (json.RawMessage, error)

// ToolExecutionInterceptFunc is a callback for tool execution intercepts
// following the middleware chain pattern. It receives the tool arguments and
// a `next` function. Call `next` to invoke the next intercept in the chain
// (or the original tool implementation if this is the innermost intercept).
// Skip calling `next` to short-circuit the chain entirely.
type ToolExecutionInterceptFunc func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error)

// LLMResponseFunc is a callback that transforms an LLM response. It receives
// the response as plain JSON and must return the (possibly modified) response
// JSON.
type LLMResponseFunc func(responseJSON json.RawMessage) json.RawMessage

// LLMRequestFunc is a callback that transforms an LLM request. It receives
// the headers JSON and content JSON from the FfiLLMRequest, and returns the
// (possibly modified) versions of each. The Go binding uses JSON
// serialization rather than opaque C pointers for ergonomics.
type LLMRequestFunc func(headers, content json.RawMessage) (headers2, content2 json.RawMessage)

// LLMConditionalFunc is a callback that decides whether an LLM call should
// proceed. It returns nil to allow execution, or a non-nil pointer to an error
// message string to reject the call.
type LLMConditionalFunc func(headers, content json.RawMessage) *string

// LLMExecutionFunc is a callback that executes an LLM call, receiving the
// serialized LLMRequest as JSON and returning the response JSON or an error.
type LLMExecutionFunc func(requestJSON json.RawMessage) (json.RawMessage, error)

// LLMExecutionInterceptFunc is a callback for LLM execution intercepts
// following the middleware chain pattern. It receives the serialized LLMRequest
// as JSON and a `next` function. Call `next` to invoke the next intercept in
// the chain (or the original LLM implementation if this is the innermost
// intercept). Skip calling `next` to short-circuit the chain entirely.
type LLMExecutionInterceptFunc func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error)

// CollectorFunc is a callback invoked with each intercepted chunk during a
// streaming LLM response. It is used to accumulate chunks on the Go side for
// aggregation. The chunk JSON is only valid for the duration of the call.
type CollectorFunc func(chunkJSON json.RawMessage)

// FinalizerFunc is a callback invoked exactly once when a streaming LLM
// response is exhausted. It takes no arguments and must return a JSON string
// representing the aggregated response.
type FinalizerFunc func() string

// EventSubscriberFunc is a callback invoked for each lifecycle event emitted
// by the runtime. The concrete value is one of the event variant types that
// implement [Event] and is only valid for the duration of the callback.
type EventSubscriberFunc func(event Event)

// ---------------------------------------------------------------------------
// CGo trampoline functions (//export)
// These are called from C with the closure ID as user_data.
// ---------------------------------------------------------------------------

//export goToolSanitizeTrampoline
func goToolSanitizeTrampoline(userData unsafe.Pointer, name *C.char, argsJSON *C.char) *C.char {
	fn := lookupClosure(userData).(ToolSanitizeFunc)
	goName := C.GoString(name)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	result := fn(goName, goArgs)
	return C.CString(string(result))
}

//export goToolConditionalTrampoline
func goToolConditionalTrampoline(userData unsafe.Pointer, name *C.char, argsJSON *C.char) *C.char {
	fn := lookupClosure(userData).(ToolConditionalFunc)
	goName := C.GoString(name)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	result := fn(goName, goArgs)
	if result == nil {
		return nil
	}
	return C.CString(*result)
}

//export goToolExecTrampoline
func goToolExecTrampoline(userData unsafe.Pointer, argsJSON *C.char) *C.char {
	fn := lookupClosure(userData).(ToolExecutionFunc)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	result, err := fn(goArgs)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goEventSubscriberTrampoline
func goEventSubscriberTrampoline(userData unsafe.Pointer, event *C.FfiEvent) {
	fn := lookupClosure(userData).(EventSubscriberFunc)
	goEvent := newEvent(event)
	fn(goEvent)
}

//export goFreeTrampoline
func goFreeTrampoline(userData unsafe.Pointer) {
	unregisterClosure(userData)
}

//export goLlmRequestTrampoline
func goLlmRequestTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.FfiLLMRequest {
	fn := lookupClosure(userData).(LLMRequestFunc)

	// Extract headers and content from the incoming FfiLLMRequest
	cHeaders := C.nat_nexus_llm_request_headers(request)
	cContent := C.nat_nexus_llm_request_content(request)
	goHeaders := json.RawMessage(C.GoString(cHeaders))
	goContent := json.RawMessage(C.GoString(cContent))
	C.nat_nexus_string_free(cHeaders)
	C.nat_nexus_string_free(cContent)

	// Call the Go callback
	newHeaders, newContent := fn(goHeaders, goContent)

	// Create a new FfiLLMRequest from the result
	cNewHeaders := C.CString(string(newHeaders))
	cNewContent := C.CString(string(newContent))
	defer C.free(unsafe.Pointer(cNewHeaders))
	defer C.free(unsafe.Pointer(cNewContent))
	return C.nat_nexus_llm_request_new(cNewHeaders, cNewContent)
}

//export goLlmResponseTrampoline
func goLlmResponseTrampoline(userData unsafe.Pointer, responseJSON *C.char) *C.char {
	fn := lookupClosure(userData).(LLMResponseFunc)
	goJSON := json.RawMessage(C.GoString(responseJSON))
	result := fn(goJSON)
	return C.CString(string(result))
}

//export goLlmConditionalTrampoline
func goLlmConditionalTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.char {
	fn := lookupClosure(userData).(LLMConditionalFunc)

	// Extract headers and content from the incoming FfiLLMRequest
	cHeaders := C.nat_nexus_llm_request_headers(request)
	cContent := C.nat_nexus_llm_request_content(request)
	goHeaders := json.RawMessage(C.GoString(cHeaders))
	goContent := json.RawMessage(C.GoString(cContent))
	C.nat_nexus_string_free(cHeaders)
	C.nat_nexus_string_free(cContent)

	result := fn(goHeaders, goContent)
	if result == nil {
		return nil
	}
	return C.CString(*result)
}

//export goLlmExecTrampoline
func goLlmExecTrampoline(userData unsafe.Pointer, nativeJSON *C.char) *C.char {
	fn := lookupClosure(userData).(LLMExecutionFunc)
	goJSON := json.RawMessage(C.GoString(nativeJSON))

	result, err := fn(goJSON)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goToolExecInterceptTrampoline
func goToolExecInterceptTrampoline(userData unsafe.Pointer, argsJSON *C.char, nextFn C.NatNexusToolExecNextFn, nextCtx unsafe.Pointer) *C.char {
	fn := lookupClosure(userData).(ToolExecutionInterceptFunc)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	goNext := func(args json.RawMessage) (json.RawMessage, error) {
		cArgs := C.CString(string(args))
		defer C.free(unsafe.Pointer(cArgs))
		result := C.callToolExecNext(nextFn, cArgs, nextCtx)
		if result == nil {
			return nil, lastError()
		}
		defer C.nat_nexus_string_free(result)
		return json.RawMessage(C.GoString(result)), nil
	}
	result, err := fn(goArgs, goNext)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goLlmExecInterceptTrampoline
func goLlmExecInterceptTrampoline(userData unsafe.Pointer, nativeJSON *C.char, nextFn C.NatNexusLlmExecNextFn, nextCtx unsafe.Pointer) *C.char {
	fn := lookupClosure(userData).(LLMExecutionInterceptFunc)
	goJSON := json.RawMessage(C.GoString(nativeJSON))

	goNext := func(reqJSON json.RawMessage) (json.RawMessage, error) {
		cJSON := C.CString(string(reqJSON))
		defer C.free(unsafe.Pointer(cJSON))

		result := C.callLlmExecNext(nextFn, cJSON, nextCtx)
		if result == nil {
			return nil, lastError()
		}
		defer C.nat_nexus_string_free(result)
		return json.RawMessage(C.GoString(result)), nil
	}

	result, err := fn(goJSON, goNext)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}
