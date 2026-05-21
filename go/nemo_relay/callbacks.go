// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// callbacks.go defines the Go callback type aliases used by the NeMo Relay
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

package nemo_relay

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;

typedef void (*NemoRelayFreeFn)(void* user_data);
typedef char* (*NemoRelayToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolExecFn)(void* user_data, const char* args_json);
typedef FfiLLMRequest* (*NemoRelayLlmRequestCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayLlmConditionalCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayLlmExecFn)(void* user_data, const char* native_json);
typedef char* (*NemoRelayLlmResponseFn)(void* user_data, const char* response_json);
typedef void (*NemoRelayEventSubscriberFn)(void* user_data, const FfiEvent* event);
typedef struct FfiPluginContext FfiPluginContext;

// Middleware chain next function types
typedef char* (*NemoRelayToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NemoRelayToolExecInterceptCb)(void* user_data, const char* args_json, NemoRelayToolExecNextFn next_fn, void* next_ctx);
typedef char* (*NemoRelayLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NemoRelayLlmExecInterceptCb)(void* user_data, const char* native_json, NemoRelayLlmExecNextFn next_fn, void* next_ctx);

// Helper to call the tool exec next function pointer from Go
static inline char* callToolExecNext(NemoRelayToolExecNextFn next_fn, const char* args_json, void* next_ctx) {
	return next_fn(args_json, next_ctx);
}

// Helper to call the LLM exec next function pointer from Go
static inline char* callLlmExecNext(NemoRelayLlmExecNextFn next_fn, const char* native_json, void* next_ctx) {
	return next_fn(native_json, next_ctx);
}

// LLMRequest accessors (also declared in types.go, needed here for trampolines)
extern FfiLLMRequest* nemo_relay_llm_request_new(const char* headers_json, const char* content_json);
extern char* nemo_relay_llm_request_headers(const FfiLLMRequest* ptr);
extern char* nemo_relay_llm_request_content(const FfiLLMRequest* ptr);
extern void nemo_relay_string_free(char* ptr);
extern void nemo_relay_set_last_error_message(const char* msg);

// Codec callback typedefs (kept for trampoline use at execute time)
typedef char* (*NemoRelayCodecDecodeCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayCodecEncodeCb)(void* user_data, const char* annotated_json, const FfiLLMRequest* original_request);
typedef NemoRelayCodecDecodeCb NemoRelayCodecDecodeFn;
typedef NemoRelayCodecEncodeCb NemoRelayCodecEncodeFn;
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
	closureTokenAlloc = func() unsafe.Pointer {
		return C.malloc(C.size_t(unsafe.Sizeof(uintptr(0))))
	}
)

func setLastErrorMessage(msg string) {
	cMsg := C.CString(msg)
	defer C.free(unsafe.Pointer(cMsg))
	C.nemo_relay_set_last_error_message(cMsg)
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
	p := (*uintptr)(closureTokenAlloc())
	if p == nil {
		panic("nemo_relay: failed to allocate callback token")
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
// implement [Event]. The Go binding snapshots FFI event fields before invoking
// the callback, so it is safe to retain the event after the callback returns.
type EventSubscriberFunc func(event Event)

// CodecFunc is a bidirectional codec with decode and encode methods.
// Decode receives the full LLM request (headers + content) as JSON and returns
// the AnnotatedLLMRequest as JSON. Encode receives the annotated request JSON
// and the original request JSON, and returns the merged content JSON.
type CodecFunc struct {
	Decode func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error)
	Encode func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error)
}

// LLMRequestInterceptFunc is a callback for LLM request intercepts with
// the unified annotated-aware signature. It receives the intercept name,
// request headers/content, and optionally the annotated request JSON (nil if
// no Codec resolved). Returns the (possibly modified) headers, content, and
// annotated JSON.
type LLMRequestInterceptFunc func(
	name string,
	headers, content json.RawMessage,
	annotatedJSON json.RawMessage,
) (newHeaders, newContent, newAnnotatedJSON json.RawMessage, err error)

func codecDecodePayload(codec *CodecFunc, headers, content json.RawMessage) (json.RawMessage, error) {
	return codec.Decode(headers, content)
}

func codecEncodePayload(codec *CodecFunc, annotated, originalHeaders, originalContent json.RawMessage) (json.RawMessage, error) {
	return codec.Encode(annotated, originalHeaders, originalContent)
}

func llmRequestInterceptPayload(
	fn LLMRequestInterceptFunc,
	name string,
	headers, content json.RawMessage,
	annotatedJSON json.RawMessage,
) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
	return fn(name, headers, content, annotatedJSON)
}

func pluginValidatePayload(plugin Plugin, pluginConfigJSON json.RawMessage) (json.RawMessage, error) {
	var pluginConfig map[string]any
	if pluginConfigJSON != nil {
		if err := jsonUnmarshal(pluginConfigJSON, &pluginConfig); err != nil {
			return nil, err
		}
	}
	diagnostics, err := plugin.Validate(pluginConfig)
	if err != nil {
		return nil, err
	}
	if diagnostics == nil {
		diagnostics = []ConfigDiagnostic{}
	}
	return jsonMarshal(diagnostics)
}

func pluginRegisterPayload(plugin Plugin, pluginConfigJSON json.RawMessage, ctx *PluginContext) error {
	var pluginConfig map[string]any
	if pluginConfigJSON != nil {
		if err := jsonUnmarshal(pluginConfigJSON, &pluginConfig); err != nil {
			return err
		}
	}
	return plugin.Register(pluginConfig, ctx)
}

func codecDecodeCStringForTest(codec *CodecFunc, request *LLMRequest) *C.char {
	result, err := codecDecodePayload(codec, request.Headers(), request.Content())
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

func codecEncodeCStringForTest(codec *CodecFunc, annotated json.RawMessage, request *LLMRequest) *C.char {
	result, err := codecEncodePayload(codec, annotated, request.Headers(), request.Content())
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

func codecDecodeResultForTest(codec *CodecFunc, request *LLMRequest) (json.RawMessage, error) {
	result := codecDecodeCStringForTest(codec, request)
	if result == nil {
		return nil, lastError()
	}
	defer C.free(unsafe.Pointer(result))
	return json.RawMessage(C.GoString(result)), nil
}

func codecEncodeResultForTest(codec *CodecFunc, annotated json.RawMessage, request *LLMRequest) (json.RawMessage, error) {
	result := codecEncodeCStringForTest(codec, annotated, request)
	if result == nil {
		return nil, lastError()
	}
	defer C.free(unsafe.Pointer(result))
	return json.RawMessage(C.GoString(result)), nil
}

func pluginValidateCStringForTest(plugin Plugin, raw json.RawMessage) *C.char {
	payload, err := pluginValidatePayload(plugin, raw)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(payload))
}

func pluginRegisterStatusForTest(plugin Plugin, raw json.RawMessage, ctx *PluginContext) C.int32_t {
	if err := pluginRegisterPayload(plugin, raw, ctx); err != nil {
		setLastErrorMessage(err.Error())
		return 5
	}
	return 0
}

func pluginValidateResultForTest(plugin Plugin, raw json.RawMessage) (json.RawMessage, error) {
	result := pluginValidateCStringForTest(plugin, raw)
	if result == nil {
		return nil, lastError()
	}
	defer C.free(unsafe.Pointer(result))
	return json.RawMessage(C.GoString(result)), nil
}

func pluginRegisterErrorForTest(plugin Plugin, raw json.RawMessage, ctx *PluginContext) error {
	if pluginRegisterStatusForTest(plugin, raw, ctx) != 0 {
		return lastError()
	}
	return nil
}

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
	cHeaders := C.nemo_relay_llm_request_headers(request)
	cContent := C.nemo_relay_llm_request_content(request)
	goHeaders := json.RawMessage(C.GoString(cHeaders))
	goContent := json.RawMessage(C.GoString(cContent))
	C.nemo_relay_string_free(cHeaders)
	C.nemo_relay_string_free(cContent)

	// Call the Go callback
	newHeaders, newContent := fn(goHeaders, goContent)

	// Create a new FfiLLMRequest from the result
	cNewHeaders := C.CString(string(newHeaders))
	cNewContent := C.CString(string(newContent))
	defer C.free(unsafe.Pointer(cNewHeaders))
	defer C.free(unsafe.Pointer(cNewContent))
	return C.nemo_relay_llm_request_new(cNewHeaders, cNewContent)
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
	cHeaders := C.nemo_relay_llm_request_headers(request)
	cContent := C.nemo_relay_llm_request_content(request)
	goHeaders := json.RawMessage(C.GoString(cHeaders))
	goContent := json.RawMessage(C.GoString(cContent))
	C.nemo_relay_string_free(cHeaders)
	C.nemo_relay_string_free(cContent)

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
func goToolExecInterceptTrampoline(userData unsafe.Pointer, argsJSON *C.char, nextFn C.NemoRelayToolExecNextFn, nextCtx unsafe.Pointer) *C.char {
	fn := lookupClosure(userData).(ToolExecutionInterceptFunc)
	goArgs := json.RawMessage(C.GoString(argsJSON))
	goNext := func(args json.RawMessage) (json.RawMessage, error) {
		cArgs := C.CString(string(args))
		defer C.free(unsafe.Pointer(cArgs))
		result := C.callToolExecNext(nextFn, cArgs, nextCtx)
		if result == nil {
			return nil, lastError()
		}
		defer C.nemo_relay_string_free(result)
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
func goLlmExecInterceptTrampoline(userData unsafe.Pointer, nativeJSON *C.char, nextFn C.NemoRelayLlmExecNextFn, nextCtx unsafe.Pointer) *C.char {
	fn := lookupClosure(userData).(LLMExecutionInterceptFunc)
	goJSON := json.RawMessage(C.GoString(nativeJSON))

	goNext := func(reqJSON json.RawMessage) (json.RawMessage, error) {
		cJSON := C.CString(string(reqJSON))
		defer C.free(unsafe.Pointer(cJSON))

		result := C.callLlmExecNext(nextFn, cJSON, nextCtx)
		if result == nil {
			return nil, lastError()
		}
		defer C.nemo_relay_string_free(result)
		return json.RawMessage(C.GoString(result)), nil
	}

	result, err := fn(goJSON, goNext)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goCodecDecodeTrampoline
func goCodecDecodeTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest) *C.char {
	codec := lookupClosure(userData).(*CodecFunc)
	return codecDecodeCStringForTest(codec, &LLMRequest{ptr: request})
}

//export goCodecEncodeTrampoline
func goCodecEncodeTrampoline(userData unsafe.Pointer, annotatedJSON *C.char, originalRequest *C.FfiLLMRequest) *C.char {
	codec := lookupClosure(userData).(*CodecFunc)
	goAnnotated := json.RawMessage(C.GoString(annotatedJSON))
	return codecEncodeCStringForTest(codec, goAnnotated, &LLMRequest{ptr: originalRequest})
}

//export goLlmRequestInterceptTrampoline
func goLlmRequestInterceptTrampoline(
	userData unsafe.Pointer, name *C.char, request *C.FfiLLMRequest,
	annotatedJSON *C.char, outRequest **C.FfiLLMRequest, outAnnotatedJSON **C.char,
) C.int32_t {
	fn := lookupClosure(userData).(LLMRequestInterceptFunc)
	goName := C.GoString(name)
	cHeaders := C.nemo_relay_llm_request_headers(request)
	cContent := C.nemo_relay_llm_request_content(request)
	goHeaders := json.RawMessage(C.GoString(cHeaders))
	goContent := json.RawMessage(C.GoString(cContent))
	C.nemo_relay_string_free(cHeaders)
	C.nemo_relay_string_free(cContent)
	var goAnnotated json.RawMessage
	if annotatedJSON != nil {
		goAnnotated = json.RawMessage(C.GoString(annotatedJSON))
	}
	newHeaders, newContent, newAnnotated, err := llmRequestInterceptPayload(fn, goName, goHeaders, goContent, goAnnotated)
	if err != nil {
		setLastErrorMessage(err.Error())
		return 5 // NemoRelayStatus::Internal
	}
	// Create output FfiLLMRequest
	cNewHeaders := C.CString(string(newHeaders))
	cNewContent := C.CString(string(newContent))
	defer C.free(unsafe.Pointer(cNewHeaders))
	defer C.free(unsafe.Pointer(cNewContent))
	*outRequest = C.nemo_relay_llm_request_new(cNewHeaders, cNewContent)
	if newAnnotated != nil {
		*outAnnotatedJSON = C.CString(string(newAnnotated))
	}
	return 0 // NemoRelayStatus::Ok
}

//export goPluginValidateTrampoline
func goPluginValidateTrampoline(userData unsafe.Pointer, pluginConfigJSON *C.char) *C.char {
	plugin := lookupClosure(userData).(Plugin)
	var raw json.RawMessage
	if pluginConfigJSON != nil {
		raw = json.RawMessage(C.GoString(pluginConfigJSON))
	}
	return pluginValidateCStringForTest(plugin, raw)
}

//export goPluginRegisterTrampoline
func goPluginRegisterTrampoline(userData unsafe.Pointer, pluginConfigJSON *C.char, ctx *C.FfiPluginContext) C.int32_t {
	plugin := lookupClosure(userData).(Plugin)
	var raw json.RawMessage
	if pluginConfigJSON != nil {
		raw = json.RawMessage(C.GoString(pluginConfigJSON))
	}
	return pluginRegisterStatusForTest(plugin, raw, &PluginContext{ptr: ctx})
}
