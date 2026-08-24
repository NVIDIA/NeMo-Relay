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
typedef struct FfiLlmSanitizeRequestCodec FfiLlmSanitizeRequestCodec;
typedef struct FfiLlmSanitizeResponseCodec FfiLlmSanitizeResponseCodec;
typedef struct NemoRelayLlmSanitizeRequestContext {
	uint32_t codec_kind;
	const char* codec_id;
	const FfiLlmSanitizeRequestCodec* codec;
} NemoRelayLlmSanitizeRequestContext;
typedef struct NemoRelayLlmSanitizeResponseContext {
	uint32_t codec_kind;
	const char* codec_id;
	const FfiLlmSanitizeResponseCodec* codec;
} NemoRelayLlmSanitizeResponseContext;

typedef void (*NemoRelayFreeFn)(void* user_data);
typedef char* (*NemoRelayToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolConditionalFn)(void* user_data, const char* name, const char* args_json);
typedef char* (*NemoRelayToolExecFn)(void* user_data, const char* args_json);
typedef FfiLLMRequest* (*NemoRelayLlmSanitizeRequestCb)(void* user_data, const FfiLLMRequest* request, NemoRelayLlmSanitizeRequestContext context);
typedef char* (*NemoRelayLlmConditionalCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayLlmExecFn)(void* user_data, const char* native_json);
typedef char* (*NemoRelayLlmSanitizeResponseCb)(void* user_data, const char* response_json, NemoRelayLlmSanitizeResponseContext context);
typedef void (*NemoRelayEventSubscriberFn)(void* user_data, const FfiEvent* event);
typedef char* (*NemoRelayEventMetadataInjectorFn)(void* user_data, const FfiEvent* event);
typedef char* (*NemoRelayEventSanitizeFn)(void* user_data, const FfiEvent* event, const char* fields_json);
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
extern void nemo_relay_llm_request_free(FfiLLMRequest* request);
extern char* nemo_relay_llm_request_headers(const FfiLLMRequest* ptr);
extern char* nemo_relay_llm_request_content(const FfiLLMRequest* ptr);
extern void nemo_relay_string_free(char* ptr);
extern void nemo_relay_set_last_error_message(const char* msg);
extern char* nemo_relay_llm_sanitize_request_codec_decode(const FfiLlmSanitizeRequestCodec*, const FfiLLMRequest*);
extern FfiLLMRequest* nemo_relay_llm_sanitize_request_codec_encode(const FfiLlmSanitizeRequestCodec*, const char*, const FfiLLMRequest*);
extern char* nemo_relay_llm_sanitize_response_codec_decode(const FfiLlmSanitizeResponseCodec*, const char*);

// Codec callback typedefs (kept for trampoline use at execute time)
typedef char* (*NemoRelayCodecDecodeCb)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayCodecEncodeCb)(void* user_data, const char* annotated_json, const FfiLLMRequest* original_request);
typedef NemoRelayCodecDecodeCb NemoRelayCodecDecodeFn;
typedef NemoRelayCodecEncodeCb NemoRelayCodecEncodeFn;
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Global closure registry: maps integer IDs to Go closures.
// The ID is passed as void* user_data to C callbacks.
// ---------------------------------------------------------------------------

var errEventMetadataInjectorCallbackNil = errors.New("event metadata injector callback is nil")
var errConditionalMiddlewareGuardrailCallbackNil = errors.New("conditional middleware guardrail callback is nil")

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

// ConditionalMiddlewareGuardrailFunc decides whether one matching global
// runtime registration remains eligible. A nil reason enables the target.
type ConditionalMiddlewareGuardrailFunc func(kinds []RuntimeRegistrationKind, registrationName string) *string

// ToolExecutionFunc is a callback that executes a tool call, receiving the
// arguments as JSON and returning the canonical result or an error.
type ToolExecutionFunc func(args json.RawMessage) (ToolExecutionResult, error)

// ToolExecutionInterceptFunc is a callback for tool execution intercepts
// following the middleware chain pattern. It receives the tool arguments and
// a `next` function. Call `next` to invoke the next intercept in the chain
// (or the original tool implementation if this is the innermost intercept).
// Skip calling `next` to short-circuit the chain entirely. The callback returns
// the canonical outcome containing the tool result, optional annotation, and
// any pending marks.
type ToolExecutionInterceptFunc func(args json.RawMessage, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error)

// LLMCodecKind identifies the active codec state supplied to a sanitizer.
type LLMCodecKind string

const (
	// LLMCodecNone means no codec was active.
	LLMCodecNone LLMCodecKind = "none"
	// LLMCodecBuiltin means a Relay built-in codec was active.
	LLMCodecBuiltin LLMCodecKind = "builtin"
	// LLMCodecRuntime means a runtime-registered codec was active.
	LLMCodecRuntime LLMCodecKind = "runtime"
	// LLMCodecOpaque means an active codec has no registered identity.
	LLMCodecOpaque LLMCodecKind = "opaque"
)

// LLMCodec identifies the codec active for one managed LLM sanitizer callback.
// ID is present for built-in and runtime codec identities.
type LLMCodec struct {
	CodecKind LLMCodecKind
	CodecID   *string
}

// LLMSanitizeRequestContext provides request codec context for one sanitizer call.
type LLMSanitizeRequestContext struct {
	Codec    LLMCodec
	resolved *LLMRequestSanitizeCodec
}

// ResolveCodec returns the active callback-scoped request codec, if any.
func (context LLMSanitizeRequestContext) ResolveCodec() *LLMRequestSanitizeCodec {
	return context.resolved
}

// LLMSanitizeResponseContext provides response codec context for one sanitizer call.
type LLMSanitizeResponseContext struct {
	Codec    LLMCodec
	resolved *LLMResponseSanitizeCodec
}

// ResolveCodec returns the active callback-scoped response codec, if any.
func (context LLMSanitizeResponseContext) ResolveCodec() *LLMResponseSanitizeCodec {
	return context.resolved
}

// ErrLLMSanitizeCodecExpired is returned when a callback-scoped codec
// capability is used after its sanitizer callback has returned.
var ErrLLMSanitizeCodecExpired = errors.New("LLM sanitizer codec capability is no longer active")

type llmSanitizeCodecInvocation struct {
	mu     sync.RWMutex
	active bool
}

func newLLMSanitizeCodecInvocation() *llmSanitizeCodecInvocation {
	return &llmSanitizeCodecInvocation{active: true}
}

func (invocation *llmSanitizeCodecInvocation) acquire() (func(), error) {
	invocation.mu.RLock()
	if !invocation.active {
		invocation.mu.RUnlock()
		return nil, ErrLLMSanitizeCodecExpired
	}
	return invocation.mu.RUnlock, nil
}

func (invocation *llmSanitizeCodecInvocation) invalidate() {
	invocation.mu.Lock()
	invocation.active = false
	invocation.mu.Unlock()
}

// LLMRequestSanitizeCodec is a callback-scoped request codec capability.
type LLMRequestSanitizeCodec struct {
	ptr        unsafe.Pointer
	invocation *llmSanitizeCodecInvocation
}

// LLMResponseSanitizeCodec is a callback-scoped response codec capability.
type LLMResponseSanitizeCodec struct {
	ptr        unsafe.Pointer
	invocation *llmSanitizeCodecInvocation
}

func acquireLLMSanitizeCodec(
	ptr unsafe.Pointer,
	invocation *llmSanitizeCodecInvocation,
) (unsafe.Pointer, func(), error) {
	if ptr == nil || invocation == nil {
		return nil, nil, ErrLLMSanitizeCodecExpired
	}
	release, err := invocation.acquire()
	if err != nil {
		return nil, nil, err
	}
	return ptr, release, nil
}

// LLMRequestFunc sanitizes an emitted LLM request. It receives the request
// first and codec context second; returning omit true removes observability.
type LLMRequestFunc func(request LLMRequestDTO, context LLMSanitizeRequestContext) (sanitized LLMRequestDTO, omit bool)

// LLMResponseFunc sanitizes an emitted LLM response. It receives the response
// first and codec context second; returning omit true removes observability.
type LLMResponseFunc func(response json.RawMessage, context LLMSanitizeResponseContext) (sanitized json.RawMessage, omit bool)

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

// EventMetadata contains flat metadata additions proposed by an injector.
// Relay validates keys and values and never overwrites metadata already present
// on the event.
type EventMetadata map[string]any

// EventMetadataInjectorFunc inspects an event and proposes metadata additions.
// Return nil metadata with a nil error for a valid no-op. Returning an error
// rejects this callback's additions without preventing event delivery.
type EventMetadataInjectorFunc func(event Event) (EventMetadata, error)

// EventSanitizeFields contains the observability fields an event sanitizer may replace.
//
// A sanitizer result replaces all three fields; it does not patch the input fields.
// Start with the provided fields and modify only the fields you intend to change.
// A zero-value json.RawMessage serializes as JSON null and clears its corresponding
// event field, so a newly constructed EventSanitizeFields clears every field left
// at its zero value.
type EventSanitizeFields struct {
	Data            json.RawMessage `json:"data"`
	CategoryProfile json.RawMessage `json:"category_profile"`
	Metadata        json.RawMessage `json:"metadata"`
}

// EventSanitizeFunc sanitizes mark or scope event observability fields.
type EventSanitizeFunc func(event Event, fields EventSanitizeFields) EventSanitizeFields

// CodecFunc is a bidirectional codec with decode and encode methods.
// Decode receives the full LLM request (headers + content) as JSON and returns
// the AnnotatedLLMRequest as JSON. Encode receives the annotated request JSON
// and the original request JSON, and returns the merged content JSON.
type CodecFunc struct {
	Decode func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error)
	Encode func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error)
}

// LLMRequestDTO is the JSON-shaped request used by request intercept outcomes.
type LLMRequestDTO struct {
	Headers json.RawMessage `json:"headers"`
	Content json.RawMessage `json:"content"`
}

// Decode normalizes an opaque request through the active codec.
func (codec *LLMRequestSanitizeCodec) Decode(request LLMRequestDTO) (json.RawMessage, error) {
	if codec == nil {
		return nil, ErrLLMSanitizeCodecExpired
	}
	ptr, release, err := acquireLLMSanitizeCodec(codec.ptr, codec.invocation)
	if err != nil {
		return nil, err
	}
	defer release()
	cHeaders := C.CString(string(request.Headers))
	defer C.free(unsafe.Pointer(cHeaders))
	cContent := C.CString(string(request.Content))
	defer C.free(unsafe.Pointer(cContent))
	cRequest := C.nemo_relay_llm_request_new(cHeaders, cContent)
	if cRequest == nil {
		return nil, lastError()
	}
	defer C.nemo_relay_llm_request_free(cRequest)
	out := C.nemo_relay_llm_sanitize_request_codec_decode((*C.FfiLlmSanitizeRequestCodec)(ptr), cRequest)
	if out == nil {
		return nil, lastError()
	}
	defer C.nemo_relay_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// Encode merges normalized changes onto the original opaque request.
func (codec *LLMRequestSanitizeCodec) Encode(annotated json.RawMessage, original LLMRequestDTO) (LLMRequestDTO, error) {
	if codec == nil {
		return LLMRequestDTO{}, ErrLLMSanitizeCodecExpired
	}
	ptr, release, err := acquireLLMSanitizeCodec(codec.ptr, codec.invocation)
	if err != nil {
		return LLMRequestDTO{}, err
	}
	defer release()
	cHeaders := C.CString(string(original.Headers))
	defer C.free(unsafe.Pointer(cHeaders))
	cContent := C.CString(string(original.Content))
	defer C.free(unsafe.Pointer(cContent))
	cOriginal := C.nemo_relay_llm_request_new(cHeaders, cContent)
	if cOriginal == nil {
		return LLMRequestDTO{}, lastError()
	}
	defer C.nemo_relay_llm_request_free(cOriginal)
	cAnnotated := C.CString(string(annotated))
	defer C.free(unsafe.Pointer(cAnnotated))
	out := C.nemo_relay_llm_sanitize_request_codec_encode((*C.FfiLlmSanitizeRequestCodec)(ptr), cAnnotated, cOriginal)
	if out == nil {
		return LLMRequestDTO{}, lastError()
	}
	defer C.nemo_relay_llm_request_free(out)
	headers := C.nemo_relay_llm_request_headers(out)
	defer C.nemo_relay_string_free(headers)
	content := C.nemo_relay_llm_request_content(out)
	defer C.nemo_relay_string_free(content)
	return LLMRequestDTO{Headers: json.RawMessage(C.GoString(headers)), Content: json.RawMessage(C.GoString(content))}, nil
}

// Decode normalizes an opaque response through the active codec.
func (codec *LLMResponseSanitizeCodec) Decode(response json.RawMessage) (json.RawMessage, error) {
	if codec == nil {
		return nil, ErrLLMSanitizeCodecExpired
	}
	ptr, release, err := acquireLLMSanitizeCodec(codec.ptr, codec.invocation)
	if err != nil {
		return nil, err
	}
	defer release()
	cResponse := C.CString(string(response))
	defer C.free(unsafe.Pointer(cResponse))
	out := C.nemo_relay_llm_sanitize_response_codec_decode((*C.FfiLlmSanitizeResponseCodec)(ptr), cResponse)
	if out == nil {
		return nil, lastError()
	}
	defer C.nemo_relay_string_free(out)
	return json.RawMessage(C.GoString(out)), nil
}

// PendingMarkSpec describes a mark Relay materializes under a managed lifecycle.
type PendingMarkSpec struct {
	Name            string          `json:"name"`
	Category        *string         `json:"category,omitempty"`
	CategoryProfile json.RawMessage `json:"category_profile,omitempty"`
	Data            json.RawMessage `json:"data,omitempty"`
	DataSchema      *DataSchema     `json:"data_schema,omitempty"`
	Metadata        json.RawMessage `json:"metadata,omitempty"`
	Severity        LogSeverity     `json:"severity,omitempty"`
}

// LLMRequestInterceptOutcome is the canonical result of an LLM request intercept.
type LLMRequestInterceptOutcome struct {
	Request                   LLMRequestDTO                 `json:"request"`
	AnnotatedRequest          json.RawMessage               `json:"annotated_request"`
	PendingMarks              []PendingMarkSpec             `json:"pending_marks"`
	OptimizationContributions []LLMOptimizationContribution `json:"optimization_contributions"`
}

// ToolExecutionResult is the canonical application-visible result of a tool
// execution. Result contains the protocol payload. Annotation carries optional
// Relay-adjacent metadata without changing that payload.
type ToolExecutionResult struct {
	Result     json.RawMessage `json:"result"`
	Annotation json.RawMessage `json:"annotation,omitempty"`
}

func normalizeToolExecutionAnnotation(annotation json.RawMessage) json.RawMessage {
	if bytes.Equal(bytes.TrimSpace(annotation), []byte("null")) {
		return nil
	}
	return annotation
}

func normalizeToolExecutionResult(result ToolExecutionResult) ToolExecutionResult {
	result.Annotation = normalizeToolExecutionAnnotation(result.Annotation)
	return result
}

func decodeToolExecutionResult(payload []byte) (ToolExecutionResult, error) {
	var result ToolExecutionResult
	if err := json.Unmarshal(payload, &result); err != nil {
		return ToolExecutionResult{}, err
	}
	if result.Result == nil {
		return ToolExecutionResult{}, errors.New("canonical tool execution result is missing required result field")
	}
	return normalizeToolExecutionResult(result), nil
}

// ToolExecutionInterceptOutcome is the canonical result of a tool execution
// intercept. Result and Annotation are passed to the remaining middleware and
// application. PendingMarks are Relay-owned lifecycle metadata emitted after
// the tool-end event and are not included in the application-visible result.
type ToolExecutionInterceptOutcome struct {
	Result       json.RawMessage   `json:"result"`
	Annotation   json.RawMessage   `json:"annotation,omitempty"`
	PendingMarks []PendingMarkSpec `json:"pending_marks"`
}

// LLMRequestInterceptFunc is a callback for LLM request intercepts. When
// annotatedJSON is non-nil, request.Content is read-only, request.Headers may
// be changed, and the returned annotation is authoritative for provider body
// content. Without an annotation, the full request is writable.
type LLMRequestInterceptFunc func(
	name string,
	request LLMRequestDTO,
	annotatedJSON json.RawMessage,
) (LLMRequestInterceptOutcome, error)

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
) (LLMRequestInterceptOutcome, error) {
	return fn(name, LLMRequestDTO{Headers: headers, Content: content}, annotatedJSON)
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

//export goConditionalMiddlewareGuardrailTrampoline
func goConditionalMiddlewareGuardrailTrampoline(userData unsafe.Pointer, kindsJSON *C.char, registrationName *C.char) *C.char {
	fn := lookupClosure(userData).(ConditionalMiddlewareGuardrailFunc)
	var kinds []RuntimeRegistrationKind
	if err := json.Unmarshal([]byte(C.GoString(kindsJSON)), &kinds); err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	reason := fn(kinds, C.GoString(registrationName))
	if reason == nil {
		return nil
	}
	return C.CString(*reason)
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
	result = normalizeToolExecutionResult(result)
	resultJSON, err := jsonMarshal(result)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(resultJSON))
}

//export goEventSubscriberTrampoline
func goEventSubscriberTrampoline(userData unsafe.Pointer, event *C.FfiEvent) {
	fn := lookupClosure(userData).(EventSubscriberFunc)
	goEvent := newEvent(event)
	fn(goEvent)
}

//export goEventMetadataInjectorTrampoline
func goEventMetadataInjectorTrampoline(userData unsafe.Pointer, event *C.FfiEvent) *C.char {
	fn := lookupClosure(userData).(EventMetadataInjectorFunc)
	metadata, err := fn(newEvent(event))
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	if metadata == nil {
		metadata = EventMetadata{}
	}
	result, err := json.Marshal(metadata)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goEventSanitizeTrampoline
func goEventSanitizeTrampoline(userData unsafe.Pointer, event *C.FfiEvent, fieldsJSON *C.char) *C.char {
	fn := lookupClosure(userData).(EventSanitizeFunc)
	var fields EventSanitizeFields
	if err := json.Unmarshal([]byte(C.GoString(fieldsJSON)), &fields); err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	result, err := json.Marshal(fn(newEvent(event), fields))
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(result))
}

//export goFreeTrampoline
func goFreeTrampoline(userData unsafe.Pointer) {
	unregisterClosure(userData)
}

//export goLlmRequestTrampoline
func goLlmRequestTrampoline(userData unsafe.Pointer, request *C.FfiLLMRequest, context C.NemoRelayLlmSanitizeRequestContext) *C.FfiLLMRequest {
	fn := lookupClosure(userData).(LLMRequestFunc)
	cHeaders := C.nemo_relay_llm_request_headers(request)
	cContent := C.nemo_relay_llm_request_content(request)
	defer C.nemo_relay_string_free(cHeaders)
	defer C.nemo_relay_string_free(cContent)
	invocation := newLLMSanitizeCodecInvocation()
	defer invocation.invalidate()
	sanitized, omit := fn(LLMRequestDTO{
		Headers: json.RawMessage(C.GoString(cHeaders)),
		Content: json.RawMessage(C.GoString(cContent)),
	}, llmSanitizeRequestContextFromC(context, invocation))
	if omit {
		return nil
	}
	cHeaders = C.CString(string(sanitized.Headers))
	cContent = C.CString(string(sanitized.Content))
	defer C.free(unsafe.Pointer(cHeaders))
	defer C.free(unsafe.Pointer(cContent))
	return C.nemo_relay_llm_request_new(cHeaders, cContent)
}

func llmSanitizeRequestContextFromC(
	context C.NemoRelayLlmSanitizeRequestContext,
	invocation *llmSanitizeCodecInvocation,
) LLMSanitizeRequestContext {
	var codecID *string
	if context.codec_id != nil {
		id := C.GoString(context.codec_id)
		codecID = &id
	}
	result := LLMSanitizeRequestContext{Codec: llmCodecIdentity(uint32(context.codec_kind), codecID)}
	if context.codec != nil {
		result.resolved = &LLMRequestSanitizeCodec{
			ptr:        unsafe.Pointer(context.codec),
			invocation: invocation,
		}
	}
	return result
}

func llmCodecIdentity(codecKind uint32, codecID *string) LLMCodec {
	kind := LLMCodecOpaque
	switch codecKind {
	case 0:
		kind = LLMCodecNone
	case 1:
		kind = LLMCodecBuiltin
	case 2:
		kind = LLMCodecRuntime
	case 3:
		kind = LLMCodecOpaque
	}
	return LLMCodec{CodecKind: kind, CodecID: codecID}
}

//export goLlmResponseTrampoline
func goLlmResponseTrampoline(userData unsafe.Pointer, responseJSON *C.char, context C.NemoRelayLlmSanitizeResponseContext) *C.char {
	fn := lookupClosure(userData).(LLMResponseFunc)
	invocation := newLLMSanitizeCodecInvocation()
	defer invocation.invalidate()
	sanitized, omit := fn(
		json.RawMessage(C.GoString(responseJSON)),
		llmSanitizeResponseContextFromC(context, invocation),
	)
	if omit {
		return nil
	}
	return C.CString(string(sanitized))
}

func llmSanitizeResponseContextFromC(
	context C.NemoRelayLlmSanitizeResponseContext,
	invocation *llmSanitizeCodecInvocation,
) LLMSanitizeResponseContext {
	var codecID *string
	if context.codec_id != nil {
		id := C.GoString(context.codec_id)
		codecID = &id
	}
	resolved := (*LLMResponseSanitizeCodec)(nil)
	if context.codec != nil {
		resolved = &LLMResponseSanitizeCodec{
			ptr:        unsafe.Pointer(context.codec),
			invocation: invocation,
		}
	}
	return LLMSanitizeResponseContext{
		Codec:    llmCodecIdentity(uint32(context.codec_kind), codecID),
		resolved: resolved,
	}
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
	goNext := func(args json.RawMessage) (ToolExecutionResult, error) {
		cArgs := C.CString(string(args))
		defer C.free(unsafe.Pointer(cArgs))
		result := C.callToolExecNext(nextFn, cArgs, nextCtx)
		if result == nil {
			return ToolExecutionResult{}, lastError()
		}
		defer C.nemo_relay_string_free(result)
		return decodeToolExecutionResult([]byte(C.GoString(result)))
	}
	outcome, err := fn(goArgs, goNext)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	if outcome.PendingMarks == nil {
		outcome.PendingMarks = []PendingMarkSpec{}
	}
	outcome.Annotation = normalizeToolExecutionAnnotation(outcome.Annotation)
	outcomeJSON, err := jsonMarshal(outcome)
	if err != nil {
		setLastErrorMessage(err.Error())
		return nil
	}
	return C.CString(string(outcomeJSON))
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
	annotatedJSON *C.char, outOutcomeJSON **C.char,
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
	outcome, err := llmRequestInterceptPayload(fn, goName, goHeaders, goContent, goAnnotated)
	if err != nil {
		setLastErrorMessage(err.Error())
		return 5 // NemoRelayStatus::Internal
	}
	if outcome.PendingMarks == nil {
		outcome.PendingMarks = []PendingMarkSpec{}
	}
	if outcome.OptimizationContributions == nil {
		outcome.OptimizationContributions = []LLMOptimizationContribution{}
	}
	outcomeJSON, err := jsonMarshal(outcome)
	if err != nil {
		setLastErrorMessage(err.Error())
		return 5
	}
	*outOutcomeJSON = C.CString(string(outcomeJSON))
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
