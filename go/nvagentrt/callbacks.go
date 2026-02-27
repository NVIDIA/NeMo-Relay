// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// callbacks.go defines the Go callback type aliases used by the NVAgentRT
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

package nvagentrt

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;

typedef void (*NvAgentRtFreeFn)(void* user_data);
typedef char* (*NvAgentRtToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NvAgentRtToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef _Bool (*NvAgentRtToolExecConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NvAgentRtToolExecFn)(void* user_data, const char* args_json);
typedef char* (*NvAgentRtJsonFn)(void* user_data, const char* json);
typedef FfiLLMRequest* (*NvAgentRtLlmRequestFn)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NvAgentRtLlmConditionalFn)(void* user_data, const FfiLLMRequest* request);
typedef _Bool (*NvAgentRtLlmExecConditionalFn)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NvAgentRtLlmExecFn)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NvAgentRtSseInterceptFn)(void* user_data, const char* sse_json);
typedef void (*NvAgentRtEventSubscriberFn)(void* user_data, const FfiEvent* event);

// LLMRequest accessors (also declared in types.go, needed here for trampolines)
extern FfiLLMRequest* nvagentrt_llm_request_new(const char* method, const char* url, const char* headers_json, const char* body_json);
extern char* nvagentrt_llm_request_method(const FfiLLMRequest* ptr);
extern char* nvagentrt_llm_request_url(const FfiLLMRequest* ptr);
extern char* nvagentrt_llm_request_headers(const FfiLLMRequest* ptr);
extern char* nvagentrt_llm_request_body(const FfiLLMRequest* ptr);
extern void nvagentrt_string_free(char* ptr);
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

// collectorMu protects the active collector/finalizer callbacks during a
// streaming LLM call. Because the FFI collector/finalizer callbacks do not
// carry user_data, we use thread-local-like global state set immediately
// before the blocking FFI call.
var (
	collectorMu     sync.Mutex
	activeCollector CollectorFunc
	activeFinalizer FinalizerFunc
)

// registerClosure stores fn in the global registry and returns an
// unsafe.Pointer that encodes the registry key. The returned pointer is
// suitable for passing as void* user_data to C callbacks.
func registerClosure(fn interface{}) unsafe.Pointer {
	id := uintptr(closureNextID.Add(1))
	closureRegistryMu.Lock()
	closureRegistry[id] = fn
	closureRegistryMu.Unlock()
	// Heap-allocate the ID so we have a real pointer for C's void*.
	p := new(uintptr)
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
}

// ---------------------------------------------------------------------------
// Go callback type definitions
// ---------------------------------------------------------------------------

// ToolSanitizeFunc is a callback that receives a tool name and its arguments
// as JSON, and returns the (possibly modified) arguments. It is used by both
// sanitize guardrails and request/response intercepts for tools.
type ToolSanitizeFunc func(name string, args json.RawMessage) json.RawMessage

// ToolConditionalFunc is a callback that decides whether a tool call should
// proceed. It returns nil to allow execution, or a non-nil pointer to an error
// message string to reject the call.
type ToolConditionalFunc func(name string, args json.RawMessage) *string

// ToolExecConditionalFunc is a callback that returns true if an execution
// intercept should handle the tool call, or false to pass through to the
// next intercept or the original implementation.
type ToolExecConditionalFunc func(name string, args json.RawMessage) bool

// ToolExecutionFunc is a callback that executes a tool call, receiving the
// arguments as JSON and returning the result JSON or an error.
type ToolExecutionFunc func(args json.RawMessage) (json.RawMessage, error)

// JSONFunc is a generic callback that transforms a JSON value and returns
// the modified JSON. It is used for LLM response sanitization and intercepts.
type JSONFunc func(value json.RawMessage) json.RawMessage

// LLMRequestFunc is a callback that transforms an LLM request. It receives
// the HTTP method, URL, headers JSON, and body JSON, and returns the
// (possibly modified) versions of each. The Go binding uses JSON
// serialization rather than opaque C pointers for ergonomics.
type LLMRequestFunc func(method, url string, headers, body json.RawMessage) (method2, url2 string, headers2, body2 json.RawMessage)

// LLMConditionalFunc is a callback that decides whether an LLM call should
// proceed. It returns nil to allow execution, or a non-nil pointer to an error
// message string to reject the call.
type LLMConditionalFunc func(method, url string, headers, body json.RawMessage) *string

// LLMExecConditionalFunc is a callback that returns true if an execution
// intercept should handle the LLM call, or false to pass through to the
// next intercept or the original implementation.
type LLMExecConditionalFunc func(method, url string, headers, body json.RawMessage) bool

// LLMExecutionFunc is a callback that executes an LLM call, receiving the
// HTTP method, URL, headers, and body as JSON, and returning the response
// JSON or an error.
type LLMExecutionFunc func(method, url string, headers, body json.RawMessage) (json.RawMessage, error)

// StringInterceptFunc is a callback that transforms a single chunk (as a
// string) during a streaming LLM response.
type StringInterceptFunc func(chunk string) string

// CollectorFunc is a callback invoked with each intercepted chunk during a
// streaming LLM response. It is used to accumulate chunks on the Go side for
// aggregation. The chunk string is only valid for the duration of the call.
type CollectorFunc func(chunk string)

// FinalizerFunc is a callback invoked exactly once when a streaming LLM
// response is exhausted. It takes no arguments and must return a JSON string
// representing the aggregated response.
type FinalizerFunc func() string

// EventSubscriberFunc is a callback invoked for each lifecycle event emitted
// by the runtime. The Event pointer is only valid for the duration of the
// callback; callers must not retain it.
type EventSubscriberFunc func(event *Event)

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

//export goToolExecConditionalTrampoline
func goToolExecConditionalTrampoline(userData unsafe.Pointer, name *C.char, argsJSON *C.char) C._Bool {
	fn := lookupClosure(userData).(ToolExecConditionalFunc)
	goName := C.GoString(name)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	return C._Bool(fn(goName, goArgs))
}

//export goToolExecTrampoline
func goToolExecTrampoline(userData unsafe.Pointer, argsJSON *C.char) *C.char {
	fn := lookupClosure(userData).(ToolExecutionFunc)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	result, err := fn(goArgs)
	if err != nil {
		return C.CString(`{"error":"` + err.Error() + `"}`)
	}
	return C.CString(string(result))
}

//export goJSONTrampoline
func goJSONTrampoline(userData unsafe.Pointer, jsonStr *C.char) *C.char {
	fn := lookupClosure(userData).(JSONFunc)
	goJSON := json.RawMessage(C.GoString(jsonStr))
	result := fn(goJSON)
	return C.CString(string(result))
}

//export goEventSubscriberTrampoline
func goEventSubscriberTrampoline(userData unsafe.Pointer, event *C.FfiEvent) {
	fn := lookupClosure(userData).(EventSubscriberFunc)
	goEvent := &Event{ptr: event}
	fn(goEvent)
}

//export goFreeTrampoline
func goFreeTrampoline(userData unsafe.Pointer) {
	unregisterClosure(userData)
}

//export goLlmRequestTrampoline
func goLlmRequestTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.FfiLLMRequest {
	fn := lookupClosure(userData).(LLMRequestFunc)

	method := goString(C.nvagentrt_llm_request_method(request))
	url := goString(C.nvagentrt_llm_request_url(request))
	headers := goJSONOpt(C.nvagentrt_llm_request_headers(request))
	body := goJSONOpt(C.nvagentrt_llm_request_body(request))

	m2, u2, h2, b2 := fn(method, url, headers, body)

	cMethod := C.CString(m2)
	cURL := C.CString(u2)
	cHeaders := C.CString(string(h2))
	cBody := C.CString(string(b2))
	defer C.free(unsafe.Pointer(cMethod))
	defer C.free(unsafe.Pointer(cURL))
	defer C.free(unsafe.Pointer(cHeaders))
	defer C.free(unsafe.Pointer(cBody))

	return C.nvagentrt_llm_request_new(cMethod, cURL, cHeaders, cBody)
}

//export goLlmConditionalTrampoline
func goLlmConditionalTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.char {
	fn := lookupClosure(userData).(LLMConditionalFunc)

	method := goString(C.nvagentrt_llm_request_method(request))
	url := goString(C.nvagentrt_llm_request_url(request))
	headers := goJSONOpt(C.nvagentrt_llm_request_headers(request))
	body := goJSONOpt(C.nvagentrt_llm_request_body(request))

	result := fn(method, url, headers, body)
	if result == nil {
		return nil
	}
	return C.CString(*result)
}

//export goLlmExecConditionalTrampoline
func goLlmExecConditionalTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) C._Bool {
	fn := lookupClosure(userData).(LLMExecConditionalFunc)

	method := goString(C.nvagentrt_llm_request_method(request))
	url := goString(C.nvagentrt_llm_request_url(request))
	headers := goJSONOpt(C.nvagentrt_llm_request_headers(request))
	body := goJSONOpt(C.nvagentrt_llm_request_body(request))

	return C._Bool(fn(method, url, headers, body))
}

//export goLlmExecTrampoline
func goLlmExecTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.char {
	fn := lookupClosure(userData).(LLMExecutionFunc)

	method := goString(C.nvagentrt_llm_request_method(request))
	url := goString(C.nvagentrt_llm_request_url(request))
	headers := goJSONOpt(C.nvagentrt_llm_request_headers(request))
	body := goJSONOpt(C.nvagentrt_llm_request_body(request))

	result, err := fn(method, url, headers, body)
	if err != nil {
		return C.CString(`{"error":"` + err.Error() + `"}`)
	}
	return C.CString(string(result))
}

//export goCollectorTrampoline
func goCollectorTrampoline(chunk *C.char) {
	// The collector callback doesn't use user_data; it is stored separately
	// and invoked directly by the FFI layer. However, for the Go binding we
	// need to route through the closure registry. The closure ID is encoded
	// in the global collectorClosureID variable set before the FFI call.
	goChunk := C.GoString(chunk)
	collectorMu.Lock()
	fn := activeCollector
	collectorMu.Unlock()
	if fn != nil {
		fn(goChunk)
	}
}

//export goFinalizerTrampoline
func goFinalizerTrampoline() *C.char {
	collectorMu.Lock()
	fn := activeFinalizer
	activeFinalizer = nil
	collectorMu.Unlock()
	if fn != nil {
		result := fn()
		return C.CString(result)
	}
	return C.CString("null")
}

//export goStringInterceptTrampoline
func goStringInterceptTrampoline(userData unsafe.Pointer, sseJSON *C.char) *C.char {
	fn := lookupClosure(userData).(StringInterceptFunc)
	goChunk := C.GoString(sseJSON)
	result := fn(goChunk)
	return C.CString(result)
}
