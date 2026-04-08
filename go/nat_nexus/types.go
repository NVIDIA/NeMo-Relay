// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// types.go defines the Go-side data types, opaque handle wrappers, and helper
// functions that correspond to the C FFI types exposed by nat-nexus-ffi. Each
// handle struct wraps an opaque C pointer and uses a Go runtime finalizer to
// free the underlying resource automatically when it is garbage-collected.

package nat_nexus

/*
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// Forward declarations matching the FFI crate's C API.
// These map to the opaque types in crates/ffi/src/types.rs.
typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;
typedef struct FfiStream FfiStream;

// Accessors — ScopeHandle
extern char* nat_nexus_scope_handle_uuid(const FfiScopeHandle* ptr);
extern char* nat_nexus_scope_handle_name(const FfiScopeHandle* ptr);
extern int32_t nat_nexus_scope_handle_scope_type(const FfiScopeHandle* ptr);
extern uint32_t nat_nexus_scope_handle_attributes(const FfiScopeHandle* ptr);
extern char* nat_nexus_scope_handle_parent_uuid(const FfiScopeHandle* ptr);
extern char* nat_nexus_scope_handle_data(const FfiScopeHandle* ptr);
extern char* nat_nexus_scope_handle_metadata(const FfiScopeHandle* ptr);
extern void nat_nexus_scope_handle_free(FfiScopeHandle* ptr);

// Accessors — ToolHandle
extern char* nat_nexus_tool_handle_uuid(const FfiToolHandle* ptr);
extern char* nat_nexus_tool_handle_name(const FfiToolHandle* ptr);
extern uint32_t nat_nexus_tool_handle_attributes(const FfiToolHandle* ptr);
extern char* nat_nexus_tool_handle_parent_uuid(const FfiToolHandle* ptr);
extern void nat_nexus_tool_handle_free(FfiToolHandle* ptr);

// Accessors — LLMHandle
extern char* nat_nexus_llm_handle_uuid(const FfiLLMHandle* ptr);
extern char* nat_nexus_llm_handle_name(const FfiLLMHandle* ptr);
extern uint32_t nat_nexus_llm_handle_attributes(const FfiLLMHandle* ptr);
extern char* nat_nexus_llm_handle_parent_uuid(const FfiLLMHandle* ptr);
extern void nat_nexus_llm_handle_free(FfiLLMHandle* ptr);

// LLMRequest
extern FfiLLMRequest* nat_nexus_llm_request_new(const char* headers_json, const char* content_json);
extern char* nat_nexus_llm_request_headers(const FfiLLMRequest* ptr);
extern char* nat_nexus_llm_request_content(const FfiLLMRequest* ptr);
extern void nat_nexus_llm_request_free(FfiLLMRequest* ptr);

// Event accessors
extern char* nat_nexus_event_uuid(const FfiEvent* ptr);
extern char* nat_nexus_event_name(const FfiEvent* ptr);
extern char* nat_nexus_event_kind(const FfiEvent* ptr);
extern uint32_t nat_nexus_event_attributes(const FfiEvent* ptr);
extern char* nat_nexus_event_data(const FfiEvent* ptr);
extern char* nat_nexus_event_metadata(const FfiEvent* ptr);
extern char* nat_nexus_event_timestamp(const FfiEvent* ptr);
extern char* nat_nexus_event_input(const void* ptr);
extern char* nat_nexus_event_output(const void* ptr);
extern char* nat_nexus_event_model_name(const void* ptr);
extern char* nat_nexus_event_tool_call_id(const void* ptr);
extern char* nat_nexus_event_parent_uuid(const void* ptr);
extern char* nat_nexus_event_scope_type(const void* ptr);
extern void nat_nexus_event_free(FfiEvent* ptr);

// String free
extern void nat_nexus_string_free(char* ptr);

// Last error
extern const char* nat_nexus_last_error();

// Stream
extern void nat_nexus_stream_free(FfiStream* stream);
extern int32_t nat_nexus_stream_next(FfiStream* stream, char** out_chunk);
*/
import "C"

import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// ScopeType represents the kind of execution scope. It mirrors the core Rust
// ScopeType enum and is used when pushing a new scope onto the scope stack via
// [PushScope]. The scope type categorizes the scope for observability and
// tracing purposes.
type ScopeType int32

const (
	// ScopeTypeAgent represents a top-level agent scope. This is typically the
	// outermost scope in an agent execution.
	ScopeTypeAgent ScopeType = 0
	// ScopeTypeFunction represents a generic function scope for arbitrary
	// user-defined operations.
	ScopeTypeFunction ScopeType = 1
	// ScopeTypeTool represents a tool invocation scope. Typically used
	// internally by [ToolCall] and [ToolCallExecute].
	ScopeTypeTool ScopeType = 2
	// ScopeTypeLlm represents an LLM call scope. Typically used internally
	// by [LlmCall] and [LlmCallExecute].
	ScopeTypeLlm ScopeType = 3
	// ScopeTypeRetriever represents a retriever scope for RAG-style document
	// retrieval operations.
	ScopeTypeRetriever ScopeType = 4
	// ScopeTypeEmbedder represents an embedder scope for text embedding
	// operations.
	ScopeTypeEmbedder ScopeType = 5
	// ScopeTypeReranker represents a reranker scope for result reranking
	// operations.
	ScopeTypeReranker ScopeType = 6
	// ScopeTypeGuardrail represents a guardrail scope for middleware that
	// sanitizes or gates calls.
	ScopeTypeGuardrail ScopeType = 7
	// ScopeTypeEvaluator represents an evaluator scope for evaluation or
	// scoring operations.
	ScopeTypeEvaluator ScopeType = 8
	// ScopeTypeCustom represents a user-defined custom scope type for
	// domain-specific operations not covered by the built-in types.
	ScopeTypeCustom ScopeType = 9
	// ScopeTypeUnknown represents an unrecognized or invalid scope type.
	// This should not be used directly; it serves as a sentinel value.
	ScopeTypeUnknown ScopeType = 10
)

// Scope attribute bitflags modify scope behavior. They are passed to
// [WithScopeAttributes] and can be combined with bitwise OR.
const (
	// ScopeAttrParallel marks the scope as executing in parallel with its
	// sibling scopes. Observability tools may use this to visualize
	// concurrent branches in a trace.
	ScopeAttrParallel uint32 = 0b01
	// ScopeAttrRelocatable marks the scope as movable across execution
	// contexts (e.g., between goroutines or threads). This is a hint for
	// the runtime that the scope may not complete on the same thread where
	// it was created.
	ScopeAttrRelocatable uint32 = 0b10
)

// Tool attribute bitflags modify tool call behavior. They are passed to
// [WithToolAttributes] and can be combined with bitwise OR.
const (
	// ToolAttrLocal marks the tool as a local (in-process) tool, as opposed
	// to a remote tool invoked over the network.
	ToolAttrLocal uint32 = 0b01
)

// LLM attribute bitflags modify LLM call behavior. They are passed to
// [WithLLMAttributes] and can be combined with bitwise OR.
const (
	// LLMAttrStateless marks the LLM call as stateless, meaning it carries
	// no conversation history and each call is independent.
	LLMAttrStateless uint32 = 0b01
	// LLMAttrStreaming marks the LLM call as a streaming request where the
	// response is delivered incrementally via SSE chunks. This flag is set
	// automatically by [LlmStreamCallExecute].
	LLMAttrStreaming uint32 = 0b10
)

// ScopeHandle wraps an opaque C pointer to a scope handle. It represents a
// single node in the hierarchical scope stack and provides read-only access
// to scope metadata. The underlying C resource is freed automatically via
// a Go runtime finalizer.
type ScopeHandle struct {
	ptr *C.FfiScopeHandle
}

func newScopeHandle(ptr *C.FfiScopeHandle) *ScopeHandle {
	if ptr == nil {
		return nil
	}
	h := &ScopeHandle{ptr: ptr}
	runtime.SetFinalizer(h, func(h *ScopeHandle) {
		if h.ptr != nil {
			C.nat_nexus_scope_handle_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// UUID returns the unique identifier for this scope.
func (h *ScopeHandle) UUID() string { return goString(C.nat_nexus_scope_handle_uuid(h.ptr)) }

// Name returns the human-readable name of this scope.
func (h *ScopeHandle) Name() string { return goString(C.nat_nexus_scope_handle_name(h.ptr)) }

// Type returns the ScopeType of this scope.
func (h *ScopeHandle) Type() ScopeType {
	return ScopeType(C.nat_nexus_scope_handle_scope_type(h.ptr))
}

// Attributes returns the attribute bitflags for this scope.
func (h *ScopeHandle) Attributes() uint32 {
	return uint32(C.nat_nexus_scope_handle_attributes(h.ptr))
}

// ParentUUID returns the UUID of the parent scope, or an empty string if this
// is a root scope.
func (h *ScopeHandle) ParentUUID() string {
	return goStringOpt(C.nat_nexus_scope_handle_parent_uuid(h.ptr))
}

// Data returns the optional data JSON payload attached to this scope.
func (h *ScopeHandle) Data() json.RawMessage {
	return goJSONOpt(C.nat_nexus_scope_handle_data(h.ptr))
}

// Metadata returns the optional metadata JSON payload attached to this scope.
func (h *ScopeHandle) Metadata() json.RawMessage {
	return goJSONOpt(C.nat_nexus_scope_handle_metadata(h.ptr))
}

// ToolHandle wraps an opaque C pointer to a tool call handle. It is returned
// by ToolCall and used to end the call with ToolCallEnd. The underlying C
// resource is freed automatically via a Go runtime finalizer.
type ToolHandle struct {
	ptr *C.FfiToolHandle
}

func newToolHandle(ptr *C.FfiToolHandle) *ToolHandle {
	if ptr == nil {
		return nil
	}
	h := &ToolHandle{ptr: ptr}
	runtime.SetFinalizer(h, func(h *ToolHandle) {
		if h.ptr != nil {
			C.nat_nexus_tool_handle_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// UUID returns the unique identifier for this tool call.
func (h *ToolHandle) UUID() string { return goString(C.nat_nexus_tool_handle_uuid(h.ptr)) }

// Name returns the name of the tool being called.
func (h *ToolHandle) Name() string { return goString(C.nat_nexus_tool_handle_name(h.ptr)) }

// Attributes returns the attribute bitflags for this tool call.
func (h *ToolHandle) Attributes() uint32 {
	return uint32(C.nat_nexus_tool_handle_attributes(h.ptr))
}

// ParentUUID returns the UUID of the parent scope for this tool call.
func (h *ToolHandle) ParentUUID() string {
	return goStringOpt(C.nat_nexus_tool_handle_parent_uuid(h.ptr))
}

// LLMHandle wraps an opaque C pointer to an LLM call handle. It is returned
// by LlmCall and used to end the call with LlmCallEnd. The underlying C
// resource is freed automatically via a Go runtime finalizer.
type LLMHandle struct {
	ptr *C.FfiLLMHandle
}

func newLLMHandle(ptr *C.FfiLLMHandle) *LLMHandle {
	if ptr == nil {
		return nil
	}
	h := &LLMHandle{ptr: ptr}
	runtime.SetFinalizer(h, func(h *LLMHandle) {
		if h.ptr != nil {
			C.nat_nexus_llm_handle_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// UUID returns the unique identifier for this LLM call.
func (h *LLMHandle) UUID() string { return goString(C.nat_nexus_llm_handle_uuid(h.ptr)) }

// Name returns the name of the LLM being called.
func (h *LLMHandle) Name() string { return goString(C.nat_nexus_llm_handle_name(h.ptr)) }

// Attributes returns the attribute bitflags for this LLM call.
func (h *LLMHandle) Attributes() uint32 {
	return uint32(C.nat_nexus_llm_handle_attributes(h.ptr))
}

// ParentUUID returns the UUID of the parent scope for this LLM call.
func (h *LLMHandle) ParentUUID() string {
	return goStringOpt(C.nat_nexus_llm_handle_parent_uuid(h.ptr))
}

// LLMRequest wraps an opaque C pointer to an LLM request. It contains the
// headers and content for an LLM API call. Create instances with
// [NewLLMRequest]. The underlying C resource is freed automatically via a Go
// runtime finalizer.
type LLMRequest struct {
	ptr *C.FfiLLMRequest
}

// NewLLMRequest creates a new LLM request with the given headers map and
// content. The headers and content parameters are serialized to JSON
// internally. Returns nil if the underlying C allocation fails.
//
// Example:
//
//	req := nat_nexus.NewLLMRequest(
//	    map[string]interface{}{"Authorization": "Bearer tok"},
//	    map[string]interface{}{"model": "gpt-4", "messages": []interface{}{}},
//	)
func NewLLMRequest(headers map[string]interface{}, content interface{}) *LLMRequest {
	headersJSON, _ := json.Marshal(headers)
	contentJSON, _ := json.Marshal(content)

	cHeaders := C.CString(string(headersJSON))
	cContent := C.CString(string(contentJSON))
	defer C.free(unsafe.Pointer(cHeaders))
	defer C.free(unsafe.Pointer(cContent))

	ptr := C.nat_nexus_llm_request_new(cHeaders, cContent)
	if ptr == nil {
		return nil
	}
	r := &LLMRequest{ptr: ptr}
	runtime.SetFinalizer(r, func(r *LLMRequest) {
		if r.ptr != nil {
			C.nat_nexus_llm_request_free(r.ptr)
			r.ptr = nil
		}
	})
	return r
}

// Headers returns the request headers as a JSON object.
func (r *LLMRequest) Headers() json.RawMessage {
	return goJSONOpt(C.nat_nexus_llm_request_headers(r.ptr))
}

// Content returns the request content as raw JSON.
func (r *LLMRequest) Content() json.RawMessage {
	return goJSONOpt(C.nat_nexus_llm_request_content(r.ptr))
}

// Event is the common interface implemented by all lifecycle event variants.
// Subscriber callbacks receive one of:
// [ScopeStartEvent], [ScopeEndEvent], [ToolStartEvent], [ToolEndEvent],
// [LLMStartEvent], [LLMEndEvent], or [MarkEvent].
//
// The underlying C pointer is only valid for the duration of the subscriber
// call; callers must copy any data they want to retain.
type Event interface {
	Kind() string
	UUID() string
	Name() string
	ParentUUID() string
	ScopeType() string
	Attributes() uint32
	Data() json.RawMessage
	Metadata() json.RawMessage
	Timestamp() string
	Input() json.RawMessage
	Output() json.RawMessage
	ModelName() string
	ToolCallID() string
}

type eventBase struct {
	ptr *C.FfiEvent
}

func (e eventBase) UUID() string { return goString(C.nat_nexus_event_uuid(e.ptr)) }
func (e eventBase) Name() string { return goStringOpt(C.nat_nexus_event_name(e.ptr)) }
func (e eventBase) Kind() string { return goStringOpt(C.nat_nexus_event_kind(e.ptr)) }
func (e eventBase) ScopeType() string {
	return goStringOpt((*C.char)(C.nat_nexus_event_scope_type(unsafe.Pointer(e.ptr))))
}
func (e eventBase) Attributes() uint32 {
	return uint32(C.nat_nexus_event_attributes(e.ptr))
}
func (e eventBase) Data() json.RawMessage     { return goJSONOpt(C.nat_nexus_event_data(e.ptr)) }
func (e eventBase) Metadata() json.RawMessage { return goJSONOpt(C.nat_nexus_event_metadata(e.ptr)) }
func (e eventBase) Timestamp() string         { return goString(C.nat_nexus_event_timestamp(e.ptr)) }
func (e eventBase) Input() json.RawMessage {
	return goJSONOpt((*C.char)(C.nat_nexus_event_input(unsafe.Pointer(e.ptr))))
}
func (e eventBase) Output() json.RawMessage {
	return goJSONOpt((*C.char)(C.nat_nexus_event_output(unsafe.Pointer(e.ptr))))
}
func (e eventBase) ModelName() string {
	return goStringOpt((*C.char)(C.nat_nexus_event_model_name(unsafe.Pointer(e.ptr))))
}
func (e eventBase) ToolCallID() string {
	return goStringOpt((*C.char)(C.nat_nexus_event_tool_call_id(unsafe.Pointer(e.ptr))))
}
func (e eventBase) ParentUUID() string {
	return goStringOpt((*C.char)(C.nat_nexus_event_parent_uuid(unsafe.Pointer(e.ptr))))
}

type ScopeStartEvent struct{ eventBase }
type ScopeEndEvent struct{ eventBase }
type ToolStartEvent struct{ eventBase }
type ToolEndEvent struct{ eventBase }
type LLMStartEvent struct{ eventBase }
type LLMEndEvent struct{ eventBase }
type MarkEvent struct{ eventBase }

func newEvent(ptr *C.FfiEvent) Event {
	base := eventBase{ptr: ptr}
	switch base.Kind() {
	case "ScopeStart":
		return &ScopeStartEvent{eventBase: base}
	case "ScopeEnd":
		return &ScopeEndEvent{eventBase: base}
	case "ToolStart":
		return &ToolStartEvent{eventBase: base}
	case "ToolEnd":
		return &ToolEndEvent{eventBase: base}
	case "LLMStart":
		return &LLMStartEvent{eventBase: base}
	case "LLMEnd":
		return &LLMEndEvent{eventBase: base}
	default:
		return &MarkEvent{eventBase: base}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// goString converts a library-owned C string to Go string and frees it.
func goString(cstr *C.char) string {
	if cstr == nil {
		return ""
	}
	s := C.GoString(cstr)
	C.nat_nexus_string_free(cstr)
	return s
}

// goStringOpt converts a possibly-null C string; returns "" for null.
func goStringOpt(cstr *C.char) string {
	if cstr == nil {
		return ""
	}
	s := C.GoString(cstr)
	C.nat_nexus_string_free(cstr)
	return s
}

// goJSONOpt converts a possibly-null JSON C string to json.RawMessage.
func goJSONOpt(cstr *C.char) json.RawMessage {
	if cstr == nil {
		return nil
	}
	s := C.GoString(cstr)
	C.nat_nexus_string_free(cstr)
	return json.RawMessage(s)
}
