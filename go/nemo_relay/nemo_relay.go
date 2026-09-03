// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nemo_relay provides Go bindings for the NeMo Relay agent runtime via CGo.
//
// NeMo Relay is a multi-language agent runtime framework that provides execution
// scope management, lifecycle events, and middleware (guardrails and intercepts)
// for tool and LLM calls. The core runtime is written in Rust; this package
// wraps the C FFI layer produced by the nemo-relay-ffi crate.
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
// Build prerequisites: the nemo-relay-ffi library must be built first
// (cargo build --release -p nemo-relay-ffi). The package searches the
// repo-local Cargo target directories automatically.
package nemo_relay

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -L${SRCDIR}/../../target/debug -lnemo_relay_ffi
#cgo windows LDFLAGS: -luserenv -lntdll -lws2_32 -ladvapi32 -lbcrypt
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct FfiScopeHandle FfiScopeHandle;
typedef struct FfiScopeStack FfiScopeStack;
typedef struct FfiThreadScopeStackBinding FfiThreadScopeStackBinding;
typedef struct FfiToolHandle FfiToolHandle;
typedef struct FfiLLMHandle FfiLLMHandle;
typedef struct FfiLLMRequest FfiLLMRequest;
typedef struct FfiEvent FfiEvent;
typedef struct FfiStream FfiStream;
typedef struct FfiCodecHandle FfiCodecHandle;
typedef struct FfiLlmSanitizeRequestCodec FfiLlmSanitizeRequestCodec;
typedef struct FfiLlmSanitizeResponseCodec FfiLlmSanitizeResponseCodec;
typedef struct NemoRelayLlmSanitizeRequestContext { uint32_t codec_kind; const char* codec_id; const FfiLlmSanitizeRequestCodec* codec; } NemoRelayLlmSanitizeRequestContext;
typedef struct NemoRelayLlmSanitizeResponseContext { uint32_t codec_kind; const char* codec_id; const FfiLlmSanitizeResponseCodec* codec; } NemoRelayLlmSanitizeResponseContext;

typedef void (*NemoRelayFreeFn)(void* user_data);

typedef char* (*NemoRelayConditionalMiddlewareGuardrailFn)(void* user_data, const char* kinds_json, const char* registration_name);
extern int32_t nemo_relay_register_conditional_middleware_guardrail(const char* name, const char* kinds_json, const char* registration_name, NemoRelayConditionalMiddlewareGuardrailFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_conditional_middleware_guardrail(const char* name, _Bool* out_removed);
extern int32_t nemo_relay_list_runtime_registrations(const char* kinds_json, char** out_json);

// Core API
extern int32_t nemo_relay_initialize_default_logging(void);
extern int32_t nemo_relay_shutdown_default_logging(void);
extern int32_t nemo_relay_get_handle(FfiScopeHandle** out);
extern int32_t nemo_relay_push_scope(const char* name, int32_t scope_type, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* input_json, const int64_t* timestamp_unix_micros, FfiScopeHandle** out);
extern int32_t nemo_relay_pop_scope(const FfiScopeHandle* handle, const char* output_json, const char* metadata_json, const int64_t* timestamp_unix_micros);
extern int32_t nemo_relay_event(const char* name, const FfiScopeHandle* parent, const char* data_json, const char* metadata_json, const int64_t* timestamp_unix_micros);
extern int32_t nemo_relay_event_v2(const char* name, const FfiScopeHandle* parent, const char* data_json, const char* data_schema_json, const char* metadata_json, const int32_t* severity, const int64_t* timestamp_unix_micros);
extern int32_t nemo_relay_metric_json(const char* name, const FfiScopeHandle* parent, const char* measurements_json, const char* metadata_json, const int64_t* timestamp_unix_micros);

// Tool lifecycle
extern int32_t nemo_relay_tool_call(const char* name, const char* args_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* tool_call_id, const int64_t* timestamp_unix_micros, FfiToolHandle** out);
extern int32_t nemo_relay_tool_call_end(const FfiToolHandle* handle, const char* result_json, const char* data_json, const char* metadata_json, const int64_t* timestamp_unix_micros);

// Tool call execute (with C function pointer callbacks)
typedef char* (*NemoRelayToolExecFn)(void* user_data, const char* args_json);
extern int32_t nemo_relay_tool_call_execute(
	const char* name, const char* args_json,
	NemoRelayToolExecFn func_cb, void* func_user_data, NemoRelayFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	char** out);
extern int32_t nemo_relay_tool_call_execute_v2(
	const char* name, const char* args_json,
	NemoRelayToolExecFn func_cb, void* func_user_data, NemoRelayFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* tool_call_id, char** out);

// LLM lifecycle
typedef void (*NemoRelayCollectorCb)(const char* chunk_json);
typedef struct Option_NemoRelayCollectorCb { NemoRelayCollectorCb cb; } Option_NemoRelayCollectorCb;
typedef char* (*NemoRelayFinalizerCb)();
typedef struct Option_NemoRelayFinalizerCb { NemoRelayFinalizerCb cb; } Option_NemoRelayFinalizerCb;

static inline Option_NemoRelayCollectorCb makeOptCollectorCb(NemoRelayCollectorCb cb) {
	Option_NemoRelayCollectorCb opt = { cb };
	return opt;
}
static inline Option_NemoRelayFinalizerCb makeOptFinalizerCb(NemoRelayFinalizerCb cb) {
	Option_NemoRelayFinalizerCb opt = { cb };
	return opt;
}

extern int32_t nemo_relay_llm_call(const char* name, const char* native_json, const FfiScopeHandle* parent, uint32_t attributes, const char* data_json, const char* metadata_json, const char* model_name, const int64_t* timestamp_unix_micros, FfiLLMHandle** out);
extern int32_t nemo_relay_llm_call_end(const FfiLLMHandle* handle, const char* response_json, const char* data_json, const char* metadata_json, const int64_t* timestamp_unix_micros);

// LLM call execute
typedef char* (*NemoRelayLlmExecFn)(void* user_data, const char* native_json);
typedef char* (*NemoRelayCodecDecodeFn)(void* user_data, const FfiLLMRequest* request);
typedef char* (*NemoRelayCodecEncodeFn)(void* user_data, const char* annotated_json, const FfiLLMRequest* original_request);
extern int32_t nemo_relay_llm_call_execute(
	const char* name, const char* native_json,
	NemoRelayLlmExecFn func_cb, void* func_user_data, NemoRelayFreeFn func_free,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	NemoRelayCodecDecodeFn codec_decode, NemoRelayCodecEncodeFn codec_encode,
	void* codec_user_data, NemoRelayFreeFn codec_free_fn,
	const FfiCodecHandle* response_codec,
	char** out);

// LLM stream execute
extern int32_t nemo_relay_llm_stream_call_execute(
	const char* name, const char* native_json,
	NemoRelayLlmExecFn func_cb, void* func_user_data, NemoRelayFreeFn func_free,
	Option_NemoRelayCollectorCb collector, Option_NemoRelayFinalizerCb finalizer,
	const FfiScopeHandle* parent, uint32_t attributes,
	const char* data_json, const char* metadata_json,
	const char* model_name,
	NemoRelayCodecDecodeFn codec_decode, NemoRelayCodecEncodeFn codec_encode,
	void* codec_user_data, NemoRelayFreeFn codec_free_fn,
	const FfiCodecHandle* response_codec,
	FfiStream** out);

// Built-in codec constructors
extern FfiCodecHandle* nemo_relay_openai_chat_codec_new(void);
extern FfiCodecHandle* nemo_relay_openai_responses_codec_new(void);
extern FfiCodecHandle* nemo_relay_anthropic_messages_codec_new(void);
extern FfiCodecHandle* nemo_relay_gemini_generate_content_codec_new(void);
extern void nemo_relay_codec_free(FfiCodecHandle* handle);

extern void nemo_relay_set_last_error_message(const char* msg);

// Tool guardrails
typedef char* (*NemoRelayToolSanitizeFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nemo_relay_register_tool_sanitize_request_guardrail(const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_tool_sanitize_request_guardrail(const char* name);
extern int32_t nemo_relay_register_tool_sanitize_response_guardrail(const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_tool_sanitize_response_guardrail(const char* name);

typedef char* (*NemoRelayToolConditionalFn)(void* user_data, const char* name, const char* args_json);
extern int32_t nemo_relay_register_tool_conditional_execution_guardrail(const char* name, int32_t priority, NemoRelayToolConditionalFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_tool_conditional_execution_guardrail(const char* name);

// Tool intercepts
extern int32_t nemo_relay_register_tool_request_intercept(const char* name, int32_t priority, _Bool break_chain, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_tool_request_intercept(const char* name);
// Middleware chain intercept callback types (must be declared before use in externs)
typedef char* (*NemoRelayToolExecNextFn)(const char* args_json, void* next_ctx);
typedef char* (*NemoRelayToolExecInterceptCb)(void* user_data, const char* args_json, NemoRelayToolExecNextFn next_fn, void* next_ctx);
extern int32_t nemo_relay_register_tool_execution_intercept(const char* name, int32_t priority, NemoRelayToolExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_deregister_tool_execution_intercept(const char* name);

// LLM guardrails
typedef FfiLLMRequest* (*NemoRelayLlmSanitizeRequestCb)(void* user_data, const FfiLLMRequest* request, NemoRelayLlmSanitizeRequestContext context);
extern int32_t nemo_relay_register_llm_sanitize_request_guardrail(const char* name, int32_t priority, NemoRelayLlmSanitizeRequestCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_llm_sanitize_request_guardrail(const char* name);

typedef char* (*NemoRelayLlmSanitizeResponseCb)(void* user_data, const char* response_json, NemoRelayLlmSanitizeResponseContext context);
extern int32_t nemo_relay_register_llm_sanitize_response_guardrail(const char* name, int32_t priority, NemoRelayLlmSanitizeResponseCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_llm_sanitize_response_guardrail(const char* name);

typedef char* (*NemoRelayLlmConditionalCb)(void* user_data, const FfiLLMRequest* request);
extern int32_t nemo_relay_register_llm_conditional_execution_guardrail(const char* name, int32_t priority, NemoRelayLlmConditionalCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_llm_conditional_execution_guardrail(const char* name);

// LLM intercepts
typedef int32_t (*NemoRelayLlmRequestInterceptCb)(void* user_data, const char* name, const FfiLLMRequest* request, const char* annotated_json, char** out_outcome_json);
extern int32_t nemo_relay_register_llm_request_intercept(const char* name, int32_t priority, _Bool break_chain, NemoRelayLlmRequestInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_llm_request_intercept(const char* name);
typedef char* (*NemoRelayLlmExecNextFn)(const char* native_json, void* next_ctx);
typedef char* (*NemoRelayLlmExecInterceptCb)(void* user_data, const char* native_json, NemoRelayLlmExecNextFn next_fn, void* next_ctx);

extern int32_t nemo_relay_register_llm_execution_intercept(const char* name, int32_t priority, NemoRelayLlmExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_deregister_llm_execution_intercept(const char* name);
extern int32_t nemo_relay_register_llm_stream_execution_intercept(const char* name, int32_t priority, NemoRelayLlmExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_deregister_llm_stream_execution_intercept(const char* name);

// Subscribers
typedef void (*NemoRelayEventSubscriberFn)(void* user_data, const FfiEvent* event);
typedef char* (*NemoRelayEventMetadataInjectorFn)(void* user_data, const FfiEvent* event);
typedef char* (*NemoRelayEventSanitizeFn)(void* user_data, const FfiEvent* event, const char* fields_json);
extern int32_t nemo_relay_register_subscriber(const char* name, NemoRelayEventSubscriberFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_subscriber(const char* name);
extern int32_t nemo_relay_flush_subscribers(void);
extern int32_t nemo_relay_register_event_metadata_injector(const char* name, int32_t priority, NemoRelayEventMetadataInjectorFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_event_metadata_injector(const char* name);
extern int32_t nemo_relay_register_mark_sanitize_guardrail(const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_mark_sanitize_guardrail(const char* name);
extern int32_t nemo_relay_register_scope_sanitize_start_guardrail(const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_scope_sanitize_start_guardrail(const char* name);
extern int32_t nemo_relay_register_scope_sanitize_end_guardrail(const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_deregister_scope_sanitize_end_guardrail(const char* name);

// Scope-local tool guardrails
extern int32_t nemo_relay_scope_register_event_metadata_injector(const char* scope_uuid, const char* name, int32_t priority, NemoRelayEventMetadataInjectorFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_event_metadata_injector(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_mark_sanitize_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_mark_sanitize_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_scope_sanitize_start_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_scope_sanitize_start_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_scope_sanitize_end_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayEventSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_scope_sanitize_end_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_tool_sanitize_request_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_tool_sanitize_request_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_tool_sanitize_response_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_tool_sanitize_response_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_tool_conditional_execution_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayToolConditionalFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_tool_conditional_execution_guardrail(const char* scope_uuid, const char* name);

// Scope-local tool intercepts
extern int32_t nemo_relay_scope_register_tool_request_intercept(const char* scope_uuid, const char* name, int32_t priority, _Bool break_chain, NemoRelayToolSanitizeFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_tool_request_intercept(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_tool_execution_intercept(const char* scope_uuid, const char* name, int32_t priority, NemoRelayToolExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_scope_deregister_tool_execution_intercept(const char* scope_uuid, const char* name);

// Scope-local LLM guardrails
extern int32_t nemo_relay_scope_register_llm_sanitize_request_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayLlmSanitizeRequestCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_llm_sanitize_request_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_llm_sanitize_response_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayLlmSanitizeResponseCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_llm_sanitize_response_guardrail(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_llm_conditional_execution_guardrail(const char* scope_uuid, const char* name, int32_t priority, NemoRelayLlmConditionalCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_llm_conditional_execution_guardrail(const char* scope_uuid, const char* name);

// Scope-local LLM intercepts
extern int32_t nemo_relay_scope_register_llm_request_intercept(const char* scope_uuid, const char* name, int32_t priority, _Bool break_chain, NemoRelayLlmRequestInterceptCb cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_llm_request_intercept(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_llm_execution_intercept(const char* scope_uuid, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_scope_deregister_llm_execution_intercept(const char* scope_uuid, const char* name);
extern int32_t nemo_relay_scope_register_llm_stream_execution_intercept(const char* scope_uuid, const char* name, int32_t priority, NemoRelayLlmExecInterceptCb exec_cb, void* exec_user_data, NemoRelayFreeFn exec_free);
extern int32_t nemo_relay_scope_deregister_llm_stream_execution_intercept(const char* scope_uuid, const char* name);

// Scope-local subscribers
extern int32_t nemo_relay_scope_register_subscriber(const char* scope_uuid, const char* name, NemoRelayEventSubscriberFn cb, void* user_data, NemoRelayFreeFn free_fn);
extern int32_t nemo_relay_scope_deregister_subscriber(const char* scope_uuid, const char* name);

// Standalone middleware chains
extern int32_t nemo_relay_tool_request_intercepts(const char* name, const char* args_json, char** out);
extern int32_t nemo_relay_tool_conditional_execution(const char* name, const char* args_json);
extern int32_t nemo_relay_llm_request_intercepts(const char* name, const char* request_json, char** out);
extern int32_t nemo_relay_llm_conditional_execution(const char* request_json);
// Error
extern const char* nemo_relay_last_error();

// String free
extern void nemo_relay_string_free(char* ptr);

// Scope stack isolation
extern int32_t nemo_relay_scope_stack_create(FfiScopeStack** out);
extern int32_t nemo_relay_capture_propagation_context_json(char** out);
extern int32_t nemo_relay_capture_rootless_propagation_context_json(char** out);
extern int32_t nemo_relay_capture_propagation_context_with_root_json(const char* root_uuid, char** out);
extern int32_t nemo_relay_capture_traceparent(char** out);
extern int32_t nemo_relay_propagation_context_to_traceparent(const char* context_json, char** out);
extern int32_t nemo_relay_scope_stack_create_from_propagation_json(const char* context_json, FfiScopeStack** out);
extern int32_t nemo_relay_scope_stack_set_thread(const FfiScopeStack* stack);
extern int32_t nemo_relay_scope_stack_capture_thread(FfiThreadScopeStackBinding** out);
extern int32_t nemo_relay_scope_stack_restore_thread(FfiThreadScopeStackBinding* binding);
extern _Bool nemo_relay_scope_stack_active(void);
extern void nemo_relay_scope_stack_free(FfiScopeStack* ptr);

// ATIF exporter
extern int32_t nemo_relay_atif_exporter_create(const char*, const char*, const char*, const char*, void**);
extern int32_t nemo_relay_atif_exporter_register(const void*, const char*);
extern int32_t nemo_relay_atif_exporter_deregister(const char*);
extern int32_t nemo_relay_atif_exporter_export(const void*, char**);
extern int32_t nemo_relay_atif_exporter_clear(const void*);
extern void nemo_relay_atif_exporter_free(void*);

// ATOF JSONL exporter
extern int32_t nemo_relay_atof_exporter_create(const char*, const char*, const char*, void**);
extern int32_t nemo_relay_atof_exporter_create_from_json(const char*, void**);
extern int32_t nemo_relay_atof_exporter_register(const void*, const char*);
extern int32_t nemo_relay_atof_exporter_deregister(const char*);
extern int32_t nemo_relay_atof_exporter_force_flush(const void*);
extern int32_t nemo_relay_atof_exporter_shutdown(const void*);
extern int32_t nemo_relay_atof_exporter_path(const void*, char**);
extern void nemo_relay_atof_exporter_free(void*);

// OpenTelemetry subscriber
extern int32_t nemo_relay_otel_subscriber_create(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, void**);
extern int32_t nemo_relay_otel_subscriber_create_with_projection_options(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, const char*, const char*, void**);
extern int32_t nemo_relay_otel_subscriber_create_with_projection_options_v2(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, const char*, const char*, const char*, void**);
extern int32_t nemo_relay_otel_subscriber_create_with_projection_options_v3(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, const char*, const char*, const char*, uint64_t, void**);
extern int32_t nemo_relay_otel_subscriber_create_with_projection_options_v4(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, const char*, const char*, const char*, uint64_t, void**);
extern int32_t nemo_relay_otel_subscriber_register(const void*, const char*);
extern int32_t nemo_relay_otel_subscriber_deregister(const char*);
extern int32_t nemo_relay_otel_subscriber_force_flush(const void*);
extern int32_t nemo_relay_otel_subscriber_runtime_diagnostics_json(const void*, char**);
extern int32_t nemo_relay_otel_subscriber_shutdown(const void*);
extern void nemo_relay_otel_subscriber_free(void*);
extern int32_t nemo_relay_otel_log_subscriber_create(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, uint64_t, uint64_t, uint64_t, uint64_t, void**);
extern int32_t nemo_relay_otel_log_subscriber_create_v2(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, const char*, uint64_t, uint64_t, uint64_t, uint64_t, void**);
extern int32_t nemo_relay_otel_log_subscriber_register(const void*, const char*);
extern int32_t nemo_relay_otel_log_subscriber_deregister(const char*);
extern int32_t nemo_relay_otel_log_subscriber_force_flush(const void*);
extern int32_t nemo_relay_otel_log_subscriber_runtime_diagnostics_json(const void*, char**);
extern int32_t nemo_relay_otel_log_subscriber_shutdown(const void*);
extern void nemo_relay_otel_log_subscriber_free(void*);
extern int32_t nemo_relay_otel_metric_subscriber_create(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, uint64_t, const char*, uint64_t, uint64_t, void**);
extern int32_t nemo_relay_otel_metric_subscriber_create_v2(const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, const char*, uint64_t, uint64_t, const char*, uint64_t, uint64_t, void**);
extern int32_t nemo_relay_otel_metric_subscriber_register(const void*, const char*);
extern int32_t nemo_relay_otel_metric_subscriber_deregister(const char*);
extern int32_t nemo_relay_otel_metric_subscriber_force_flush(const void*);
extern int32_t nemo_relay_otel_metric_subscriber_runtime_diagnostics_json(const void*, char**);
extern int32_t nemo_relay_otel_metric_subscriber_shutdown(const void*);
extern void nemo_relay_otel_metric_subscriber_free(void*);

// Go trampoline forward declarations (defined via //export in callbacks.go)
extern char* goToolSanitizeTrampoline(void*, const char*, const char*);
extern char* goEventMetadataInjectorTrampoline(void*, const FfiEvent*);
extern char* goEventSanitizeTrampoline(void*, const FfiEvent*, const char*);
extern char* goToolConditionalTrampoline(void*, const char*, const char*);
extern char* goConditionalMiddlewareGuardrailTrampoline(void*, const char*, const char*);
extern char* goToolExecTrampoline(void*, const char*);
extern void goEventSubscriberTrampoline(void*, const FfiEvent*);
extern void goFreeTrampoline(void*);
extern FfiLLMRequest* goLlmRequestTrampoline(void*, const FfiLLMRequest*, NemoRelayLlmSanitizeRequestContext);
extern char* goLlmResponseTrampoline(void*, const char*, NemoRelayLlmSanitizeResponseContext);
extern char* goLlmConditionalTrampoline(void*, const FfiLLMRequest*);
extern char* goLlmExecTrampoline(void*, const char*);
extern char* goToolExecInterceptTrampoline(void*, const char*, NemoRelayToolExecNextFn, void*);
extern char* goLlmExecInterceptTrampoline(void*, const char*, NemoRelayLlmExecNextFn, void*);

// Codec trampolines (used at execute time, not registration)
extern char* goCodecDecodeTrampoline(void*, const FfiLLMRequest*);
extern char* goCodecEncodeTrampoline(void*, const char*, const FfiLLMRequest*);
extern int32_t goLlmRequestInterceptTrampoline(
    void*, const char*, const FfiLLMRequest*, const char*, char**);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

const defaultServiceName = "nemo-relay"

func init() {
	if err := checkStatus(C.nemo_relay_initialize_default_logging()); err != nil {
		panic(fmt.Sprintf("failed to initialize NeMo Relay operational logging: %v", err))
	}
}

// ShutdownLogging drains pending operational log records and releases the default logging runtime.
// Callers that configure file sinks should defer ShutdownLogging from main.
func ShutdownLogging() error {
	return checkStatus(C.nemo_relay_shutdown_default_logging())
}

func checkedValue[T any](status int32, value T) (T, error) {
	if err := checkStatus(C.int32_t(status)); err != nil {
		var zero T
		return zero, err
	}
	return value, nil
}

var (
	getHandleFunc = func() (*ScopeHandle, error) {
		var out *C.FfiScopeHandle
		status := C.nemo_relay_get_handle(&out)
		return checkedValue(int32(status), newScopeHandle(out))
	}
	newScopeStackFunc = func() (*ScopeStack, error) {
		var ptr *C.FfiScopeStack
		status := C.nemo_relay_scope_stack_create(&ptr)
		return checkedValue(int32(status), &ScopeStack{ptr: ptr})
	}
	newAtifExporterFunc = func(sessionID, agentName, agentVersion, modelName string) (*AtifExporter, error) {
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
		status := C.nemo_relay_atif_exporter_create(cSessionID, cAgentName, cAgentVersion, cModelName, &ptr)
		return checkedValue(int32(status), &AtifExporter{ptr: ptr})
	}
	newAtofExporterFunc = func(config AtofExporterConfig) (*AtofExporter, error) {
		payload, err := json.Marshal(config)
		if err != nil {
			return nil, err
		}
		cConfig := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cConfig))
		var ptr unsafe.Pointer
		status := C.nemo_relay_atof_exporter_create_from_json(cConfig, &ptr)
		return checkedValue(int32(status), &AtofExporter{ptr: ptr})
	}
)

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

func lastError() error {
	msg := C.nemo_relay_last_error()
	if msg == nil {
		return errors.New("unknown nemo_relay error")
	}
	return errors.New(C.GoString(msg))
}

func checkStatus(status C.int32_t) error {
	if status == 0 {
		return nil
	}
	return lastError()
}

func cTimestampMicros(timestamp time.Time) *C.int64_t {
	ptr := (*C.int64_t)(C.malloc(C.size_t(unsafe.Sizeof(C.int64_t(0)))))
	*ptr = C.int64_t(timestamp.UTC().UnixMicro())
	return ptr
}

func cLogSeverity(severity LogSeverity) (*C.int32_t, error) {
	var value C.int32_t
	switch severity {
	case LogSeverityTrace:
		value = 0
	case LogSeverityDebug:
		value = 1
	case LogSeverityInfo:
		value = 2
	case LogSeverityWarn, LogSeverityWarning:
		value = 3
	case LogSeverityError:
		value = 4
	default:
		return nil, fmt.Errorf("invalid log severity %q", severity)
	}
	ptr := (*C.int32_t)(C.malloc(C.size_t(unsafe.Sizeof(value))))
	*ptr = value
	return ptr, nil
}

// RuntimeRegistrationKind identifies a global runtime registration surface.
type RuntimeRegistrationKind string

const (
	RuntimeRegistrationSubscriber                        RuntimeRegistrationKind = "subscriber"
	RuntimeRegistrationEventMetadataInjector             RuntimeRegistrationKind = "event_metadata_injector"
	RuntimeRegistrationMarkSanitizeGuardrail             RuntimeRegistrationKind = "mark_sanitize_guardrail"
	RuntimeRegistrationScopeSanitizeStartGuardrail       RuntimeRegistrationKind = "scope_sanitize_start_guardrail"
	RuntimeRegistrationScopeSanitizeEndGuardrail         RuntimeRegistrationKind = "scope_sanitize_end_guardrail"
	RuntimeRegistrationToolSanitizeRequestGuardrail      RuntimeRegistrationKind = "tool_sanitize_request_guardrail"
	RuntimeRegistrationToolSanitizeResponseGuardrail     RuntimeRegistrationKind = "tool_sanitize_response_guardrail"
	RuntimeRegistrationToolConditionalExecutionGuardrail RuntimeRegistrationKind = "tool_conditional_execution_guardrail"
	RuntimeRegistrationToolRequestIntercept              RuntimeRegistrationKind = "tool_request_intercept"
	RuntimeRegistrationToolExecutionIntercept            RuntimeRegistrationKind = "tool_execution_intercept"
	RuntimeRegistrationLlmSanitizeRequestGuardrail       RuntimeRegistrationKind = "llm_sanitize_request_guardrail"
	RuntimeRegistrationLlmSanitizeResponseGuardrail      RuntimeRegistrationKind = "llm_sanitize_response_guardrail"
	RuntimeRegistrationLlmConditionalExecutionGuardrail  RuntimeRegistrationKind = "llm_conditional_execution_guardrail"
	RuntimeRegistrationLlmRequestIntercept               RuntimeRegistrationKind = "llm_request_intercept"
	RuntimeRegistrationLlmExecutionIntercept             RuntimeRegistrationKind = "llm_execution_intercept"
	RuntimeRegistrationLlmStreamExecutionIntercept       RuntimeRegistrationKind = "llm_stream_execution_intercept"
)

// RuntimeRegistrationOwnerKind identifies the component category that installed a registration.
type RuntimeRegistrationOwnerKind string

const (
	RuntimeRegistrationOwnerCore      RuntimeRegistrationOwnerKind = "core"
	RuntimeRegistrationOwnerGlobalAPI RuntimeRegistrationOwnerKind = "global_api"
	RuntimeRegistrationOwnerPlugin    RuntimeRegistrationOwnerKind = "plugin"
)

// RuntimeRegistrationOwner describes the component that installed a registration.
type RuntimeRegistrationOwner struct {
	Kind             RuntimeRegistrationOwnerKind `json:"kind"`
	PluginKind       *string                      `json:"plugin_kind"`
	ComponentOrdinal *uint32                      `json:"component_ordinal"`
}

// RuntimeRegistrationIdentity is the structured identity of one global registration.
type RuntimeRegistrationIdentity struct {
	Kind          RuntimeRegistrationKind  `json:"kind"`
	LocalName     string                   `json:"local_name"`
	EffectiveName string                   `json:"effective_name"`
	Owner         RuntimeRegistrationOwner `json:"owner"`
}

// RegisterConditionalMiddlewareGuardrail registers a global eligibility gate.
func RegisterConditionalMiddlewareGuardrail(name string, kinds []RuntimeRegistrationKind, registrationName string, fn ConditionalMiddlewareGuardrailFunc) error {
	if fn == nil {
		return errConditionalMiddlewareGuardrailCallbackNil
	}
	kindsJSON, err := json.Marshal(kinds)
	if err != nil {
		return err
	}
	id := registerClosure(fn)
	cName := C.CString(name)
	cKinds := C.CString(string(kindsJSON))
	cRegistrationName := C.CString(registrationName)
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cKinds))
	defer C.free(unsafe.Pointer(cRegistrationName))
	return checkStatus(C.nemo_relay_register_conditional_middleware_guardrail(
		cName,
		cKinds,
		cRegistrationName,
		C.NemoRelayConditionalMiddlewareGuardrailFn(C.goConditionalMiddlewareGuardrailTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterConditionalMiddlewareGuardrail removes a gate and reports whether it existed.
func DeregisterConditionalMiddlewareGuardrail(name string) (bool, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var removed C.bool
	if err := checkStatus(C.nemo_relay_deregister_conditional_middleware_guardrail(cName, &removed)); err != nil {
		return false, err
	}
	return bool(removed), nil
}

// ListRuntimeRegistrations returns global registrations, optionally filtered by kind.
func ListRuntimeRegistrations(kinds []RuntimeRegistrationKind) ([]RuntimeRegistrationIdentity, error) {
	var cKinds *C.char
	if kinds != nil {
		payload, err := json.Marshal(kinds)
		if err != nil {
			return nil, err
		}
		cKinds = C.CString(string(payload))
		defer C.free(unsafe.Pointer(cKinds))
	}
	var out *C.char
	if err := checkStatus(C.nemo_relay_list_runtime_registrations(cKinds, &out)); err != nil {
		return nil, err
	}
	defer C.nemo_relay_string_free(out)
	var registrations []RuntimeRegistrationIdentity
	if err := json.Unmarshal([]byte(C.GoString(out)), &registrations); err != nil {
		return nil, err
	}
	return registrations, nil
}

// ---------------------------------------------------------------------------
// Scope options (functional options pattern)
// ---------------------------------------------------------------------------

type scopeOptions struct {
	parent     *C.FfiScopeHandle
	attributes uint32
	data       *C.char
	metadata   *C.char
	input      *C.char
	timestamp  *C.int64_t
}

// ScopeOption is a functional option that configures optional parameters for
// [PushScope]. Options are applied in the order they are passed. Available
// options include [WithParent], [WithScopeAttributes], [WithData],
// [WithMetadata], [WithInput], and [WithScopeTimestamp].
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

// WithData stores an arbitrary JSON application data payload on the new scope
// handle. Scope start events use [WithInput] for their semantic event payload.
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

// WithInput attaches an arbitrary JSON semantic input payload to the new scope.
// This is exported as the scope Start event input rather than as scope data.
func WithInput(input json.RawMessage) ScopeOption {
	return func(o *scopeOptions) {
		o.input = C.CString(string(input))
	}
}

// WithScopeTimestamp records an explicit time.Time on the scope handle and
// emitted Start event. The value is converted to UTC Unix microseconds at the
// FFI boundary; sub-microsecond precision is truncated. Omit this option to use
// the current runtime time.
func WithScopeTimestamp(timestamp time.Time) ScopeOption {
	return func(o *scopeOptions) {
		o.timestamp = cTimestampMicros(timestamp)
	}
}

type scopeEndOptions struct {
	output    *C.char
	metadata  *C.char
	timestamp *C.int64_t
}

// ScopeEndOption is a functional option that configures optional parameters for
// [PopScope]. Available options include [WithOutput],
// [WithScopeEndMetadata], and [WithScopeEndTimestamp].
type ScopeEndOption func(*scopeEndOptions)

// WithOutput attaches an arbitrary JSON semantic output payload to the scope end event.
func WithOutput(output json.RawMessage) ScopeEndOption {
	return func(o *scopeEndOptions) {
		o.output = C.CString(string(output))
	}
}

// WithScopeEndMetadata attaches an arbitrary JSON metadata payload to the
// scope End event. When the scope handle already has metadata, object keys in
// this payload overwrite matching existing keys and preserve non-conflicting
// keys.
func WithScopeEndMetadata(metadata json.RawMessage) ScopeEndOption {
	return func(o *scopeEndOptions) {
		o.metadata = C.CString(string(metadata))
	}
}

// WithScopeEndTimestamp records an explicit time.Time on the scope End event.
// The value is converted to UTC Unix microseconds at the FFI boundary;
// sub-microsecond precision is truncated. Omit this option to use the runtime
// default end timestamp.
func WithScopeEndTimestamp(timestamp time.Time) ScopeEndOption {
	return func(o *scopeEndOptions) {
		o.timestamp = cTimestampMicros(timestamp)
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
	return getHandleFunc()
}

// PushScope creates a new scope and pushes it onto the hierarchical scope
// stack. The scope is assigned a unique UUID and emits a Start event to all
// registered subscribers. Use [PopScope] to end the scope. Optional parameters
// can be set via [WithParent], [WithScopeAttributes], [WithData],
// [WithMetadata], [WithInput], and [WithScopeTimestamp].
//
// The name should be a human-readable identifier for the scope (e.g.,
// "my-agent", "search-tool"). The scopeType categorizes the scope for
// observability; see [ScopeType] constants for valid values. [WithData] stores
// application data on the returned handle, while [WithInput] supplies the
// semantic data payload for the Start event.
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
	if o.input != nil {
		defer C.free(unsafe.Pointer(o.input))
	}
	if o.timestamp != nil {
		defer C.free(unsafe.Pointer(o.timestamp))
	}

	var out *C.FfiScopeHandle
	status := C.nemo_relay_push_scope(cName, C.int32_t(scopeType), o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.input, o.timestamp, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newScopeHandle(out), nil
}

// PopScope removes the given scope from the scope stack and emits an End event
// to all registered subscribers. The handle must have been returned by a
// previous call to [PushScope]. Popping scopes out of stack order returns an
// error. Optional end payloads can be attached via [WithOutput] and
// [WithScopeEndMetadata], and an explicit event timestamp can be supplied with
// [WithScopeEndTimestamp].
func PopScope(handle *ScopeHandle, opts ...ScopeEndOption) error {
	o := &scopeEndOptions{}
	for _, opt := range opts {
		opt(o)
	}
	if o.output != nil {
		defer C.free(unsafe.Pointer(o.output))
	}
	if o.metadata != nil {
		defer C.free(unsafe.Pointer(o.metadata))
	}
	if o.timestamp != nil {
		defer C.free(unsafe.Pointer(o.timestamp))
	}
	return checkStatus(C.nemo_relay_pop_scope(handle.ptr, o.output, o.metadata, o.timestamp))
}

// ---------------------------------------------------------------------------
// Event options
// ---------------------------------------------------------------------------

type eventOptions struct {
	parent     *C.FfiScopeHandle
	data       *C.char
	dataSchema *C.char
	metadata   *C.char
	severity   LogSeverity
	timestamp  *C.int64_t
}

// replaceCString replaces an option-owned C string and frees its prior value.
func replaceCString(slot **C.char, value string) {
	if *slot != nil {
		C.free(unsafe.Pointer(*slot))
	}
	*slot = C.CString(value)
}

// DataSchema identifies the name and version of a structured mark payload.
type DataSchema struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// LogSeverity controls the severity of a mark exported as an OTLP log.
type LogSeverity string

const (
	LogSeverityTrace LogSeverity = "trace"
	LogSeverityDebug LogSeverity = "debug"
	LogSeverityInfo  LogSeverity = "info"
	LogSeverityWarn  LogSeverity = "warn"
	// LogSeverityWarning is accepted as an alias for LogSeverityWarn.
	LogSeverityWarning LogSeverity = "warning"
	LogSeverityError   LogSeverity = "error"
)

// EventOption is a functional option that configures optional parameters for
// [EmitEvent]. Available options include [WithEventParent], [WithEventData],
// [WithEventDataSchema], [WithEventMetadata], [WithEventSeverity], and
// [WithEventTimestamp].
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
		replaceCString(&o.data, string(data))
	}
}

// WithEventDataSchema identifies the schema of the event data payload.
func WithEventDataSchema(schema DataSchema) EventOption {
	return func(o *eventOptions) {
		encoded, err := jsonMarshal(schema)
		if err == nil {
			replaceCString(&o.dataSchema, string(encoded))
		}
	}
}

// WithEventMetadata attaches an arbitrary JSON metadata payload to the event.
// Metadata is typically used for operational context (e.g., trace IDs, timing
// hints) as opposed to the primary data payload.
func WithEventMetadata(metadata json.RawMessage) EventOption {
	return func(o *eventOptions) {
		replaceCString(&o.metadata, string(metadata))
	}
}

// WithEventSeverity attaches a typed log severity to the mark. Relay stores
// it authoritatively in the reserved sanitizer-visible metadata key.
func WithEventSeverity(severity LogSeverity) EventOption {
	return func(o *eventOptions) {
		o.severity = severity
	}
}

// WithEventTimestamp records an explicit time.Time on the emitted Mark event.
// The value is converted to UTC Unix microseconds at the FFI boundary;
// sub-microsecond precision is truncated. Omit this option to use the current
// runtime time.
func WithEventTimestamp(timestamp time.Time) EventOption {
	return func(o *eventOptions) {
		o.timestamp = cTimestampMicros(timestamp)
	}
}

// EmitEvent emits an instantaneous Mark event within the current scope. Mark
// events represent point-in-time occurrences (e.g., checkpoints, milestones)
// and are delivered to all registered subscribers. Optional data and metadata
// payloads can be attached via [WithEventData] and [WithEventMetadata]. An
// explicit event timestamp can be supplied with [WithEventTimestamp].
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
	if o.dataSchema != nil {
		defer C.free(unsafe.Pointer(o.dataSchema))
	}
	if o.metadata != nil {
		defer C.free(unsafe.Pointer(o.metadata))
	}
	if o.timestamp != nil {
		defer C.free(unsafe.Pointer(o.timestamp))
	}
	var severity *C.int32_t
	if o.severity != "" {
		var err error
		severity, err = cLogSeverity(o.severity)
		if err != nil {
			return err
		}
		defer C.free(unsafe.Pointer(severity))
	}

	return checkStatus(C.nemo_relay_event_v2(cName, o.parent, o.data, o.dataSchema, o.metadata, severity, o.timestamp))
}

// MetricKind selects the OpenTelemetry instrument used for a measurement.
type MetricKind string

const (
	MetricKindCounter       MetricKind = "counter"
	MetricKindUpDownCounter MetricKind = "up_down_counter"
	MetricKindGauge         MetricKind = "gauge"
	MetricKindHistogram     MetricKind = "histogram"
)

// MetricValueType identifies the numeric representation of a measurement.
type MetricValueType string

const (
	MetricValueTypeU64 MetricValueType = "u64"
	MetricValueTypeI64 MetricValueType = "i64"
	MetricValueTypeF64 MetricValueType = "f64"
)

// MetricMeasurement is one SDK recording operation in an atomic metric mark.
type MetricMeasurement struct {
	Name        string                 `json:"name"`
	Kind        MetricKind             `json:"kind"`
	ValueType   MetricValueType        `json:"value_type"`
	Value       interface{}            `json:"value"`
	Unit        string                 `json:"unit,omitempty"`
	Description string                 `json:"description,omitempty"`
	Attributes  map[string]interface{} `json:"attributes,omitempty"`
	Boundaries  []float64              `json:"boundaries"`
}

type metricOptions struct {
	parent       *C.FfiScopeHandle
	parentHandle *ScopeHandle // prevents GC of the ScopeHandle during the FFI call
	metadata     *C.char
	timestamp    *C.int64_t
}

// MetricOption configures optional fields for [EmitMetric].
type MetricOption func(*metricOptions)

// WithMetricParent sets the containing scope for a metric mark.
func WithMetricParent(parent *ScopeHandle) MetricOption {
	return func(o *metricOptions) {
		if parent != nil {
			o.parent = parent.ptr
			o.parentHandle = parent
		}
	}
}

// WithMetricMetadata attaches JSON metadata to a metric mark. Metadata is not
// converted into metric attributes by the exporter.
func WithMetricMetadata(metadata json.RawMessage) MetricOption {
	return func(o *metricOptions) {
		replaceCString(&o.metadata, string(metadata))
	}
}

// WithMetricTimestamp records an explicit timestamp on the underlying mark.
// SDK-backed metric points use collection timestamps.
func WithMetricTimestamp(timestamp time.Time) MetricOption {
	return func(o *metricOptions) {
		o.timestamp = cTimestampMicros(timestamp)
	}
}

// EmitMetric emits measurements as one atomically validated metric mark.
func EmitMetric(name string, measurements []MetricMeasurement, opts ...MetricOption) error {
	encoded, err := jsonMarshal(measurements)
	if err != nil {
		return err
	}
	o := &metricOptions{}
	for _, opt := range opts {
		opt(o)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cMeasurements := C.CString(string(encoded))
	defer C.free(unsafe.Pointer(cMeasurements))
	if o.metadata != nil {
		defer C.free(unsafe.Pointer(o.metadata))
	}
	if o.timestamp != nil {
		defer C.free(unsafe.Pointer(o.timestamp))
	}
	status := C.nemo_relay_metric_json(cName, o.parent, cMeasurements, o.metadata, o.timestamp)
	runtime.KeepAlive(o.parentHandle)
	return checkStatus(status)
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
	timestamp  *C.int64_t
}

// ToolCallOption is a functional option that configures optional parameters for
// tool call functions ([ToolCall], [ToolCallEnd], [ToolCallExecute]). Available
// options include [WithToolParent], [WithToolAttributes], [WithToolData],
// [WithToolMetadata], [WithToolCallID], and [WithToolTimestamp]. Tool-call IDs
// apply to both manual and managed spans. Explicit timestamps apply only to
// manual [ToolCall] and [ToolCallEnd] spans; managed [ToolCallExecute] spans use
// runtime-generated timestamps.
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

// WithToolAttributes sets attribute bitflags for a tool call. See [ToolAttrRemote]
// for available flags. Multiple flags can be combined with bitwise OR.
func WithToolAttributes(attrs uint32) ToolCallOption {
	return func(o *toolCallOptions) {
		o.attributes = attrs
	}
}

// WithToolData stores an arbitrary JSON application data payload on the manual
// tool handle. Manual Start event data is the sanitized tool arguments; manual
// End event data is the sanitized protocol result unless that value is JSON
// null.
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
// original tool_call request in the conversation. It is recorded on both Start
// and End events for manual and managed tool calls. Omit this option to leave
// the tool call ID unset.
func WithToolCallID(id string) ToolCallOption {
	return func(o *toolCallOptions) {
		o.toolCallID = C.CString(id)
	}
}

// WithToolTimestamp records an explicit time.Time on a manual tool Start or End
// event. The value is converted to UTC Unix microseconds at the FFI boundary;
// sub-microsecond precision is truncated. Omit this option to use the current
// runtime time for Start events or the runtime default for End events.
func WithToolTimestamp(timestamp time.Time) ToolCallOption {
	return func(o *toolCallOptions) {
		o.timestamp = cTimestampMicros(timestamp)
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
	if o.timestamp != nil {
		C.free(unsafe.Pointer(o.timestamp))
	}
}

// ToolCall starts a tool call lifecycle and returns a [ToolHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [ToolCallEnd] when the tool completes. For a higher-level API that
// manages the full lifecycle automatically, use [ToolCallExecute] instead.
//
// The name identifies the tool being invoked, and args contains the tool
// arguments as JSON. The emitted Start event records args after
// sanitize-request guardrails. Request and execution intercepts run only
// through [ToolCallExecute]. Optional parameters can be set via
// [ToolCallOption] values.
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
	status := C.nemo_relay_tool_call(cName, cArgs, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.toolCallID, o.timestamp, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newToolHandle(out), nil
}

// ToolCallEnd completes a tool call that was previously started with [ToolCall].
// It emits an End event to all subscribers with the provided canonical result.
// The handle must have been returned by a prior [ToolCall] invocation. The emitted
// End event records Result after sanitize-response guardrails and carries
// Annotation in the event category profile; [WithToolData] is used only when
// the sanitized result is JSON null. Response intercepts run only through
// [ToolCallExecute].
func ToolCallEnd(handle *ToolHandle, result ToolExecutionResult, opts ...ToolCallOption) error {
	o := &toolCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeToolOpts(o)

	result = normalizeToolExecutionResult(result)
	resultJSON, err := jsonMarshal(result)
	if err != nil {
		return err
	}
	cResult := C.CString(string(resultJSON))
	defer C.free(unsafe.Pointer(cResult))

	return checkStatus(C.nemo_relay_tool_call_end(handle.ptr, cResult, o.data, o.metadata, o.timestamp))
}

// ToolCallExecute runs a complete tool call lifecycle through the full
// middleware pipeline: conditional-execution guardrails (on raw args),
// request intercepts, sanitize-request guardrails for the emitted Start event
// payload, execution intercepts, the provided fn, and sanitize-response
// guardrails for the emitted End event payload.
// On rejection, only a standalone Mark event is emitted (no Start/End pair)
// and GuardrailRejected is returned. This is the recommended high-level API
// for tool invocations. Sanitize guardrails do not rewrite the value passed
// into fn or the value returned to the caller. Use [WithToolCallID] to preserve
// a provider-assigned correlation ID on both managed lifecycle events.
func ToolCallExecute(name string, args json.RawMessage, fn ToolExecutionFunc, opts ...ToolCallOption) (ToolExecutionResult, error) {
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
	status := C.nemo_relay_tool_call_execute_v2(
		cName, cArgs,
		C.NemoRelayToolExecFn(C.goToolExecTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.toolCallID, &out,
	)
	if err := checkStatus(status); err != nil {
		return ToolExecutionResult{}, err
	}
	result, err := decodeToolExecutionResult([]byte(C.GoString(out)))
	if err != nil {
		C.nemo_relay_string_free(out)
		return ToolExecutionResult{}, err
	}
	C.nemo_relay_string_free(out)
	return result, nil
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

type llmCallOptions struct {
	parent              *C.FfiScopeHandle
	attributes          uint32
	data                *C.char
	metadata            *C.char
	modelName           *C.char
	timestamp           *C.int64_t
	codecDecode         C.NemoRelayCodecDecodeFn
	codecEncode         C.NemoRelayCodecEncodeFn
	codecUserData       unsafe.Pointer
	codecFreeFn         C.NemoRelayFreeFn
	responseCodec       *C.FfiCodecHandle
	responseCodecHandle *CodecHandle // prevents GC of the CodecHandle during FFI calls
}

// LLMCallOption is a functional option that configures optional parameters for
// LLM call functions ([LlmCall], [LlmCallEnd], [LlmCallExecute],
// [LlmStreamCallExecute], [LlmConditionalExecution]). Available options include
// [WithLLMParent], [WithLLMAttributes], [WithLLMData], [WithLLMMetadata],
// [WithLLMModelName], [WithLLMCodec], [WithLLMResponseCodec], and
// [WithLLMTimestamp]. [WithLLMTimestamp] affects manual [LlmCall] and
// [LlmCallEnd] spans only; managed execute spans use runtime-generated timestamps.
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
// [LLMAttrStateful] and [LLMAttrStreaming] for available flags. Multiple flags
// can be combined with bitwise OR.
func WithLLMAttributes(attrs uint32) LLMCallOption {
	return func(o *llmCallOptions) {
		o.attributes = attrs
	}
}

// WithLLMData stores an arbitrary JSON application data payload on the manual
// LLM handle. Manual Start event data is the sanitized request; manual End event
// data is the sanitized response unless that value is JSON null.
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

// WithLLMCodec sets the Codec to use for this LLM call. The codec's decode
// and encode callbacks are passed directly to the FFI execute functions.
func WithLLMCodec(codec CodecFunc) LLMCallOption {
	return func(o *llmCallOptions) {
		id := registerClosure(&codec)
		o.codecDecode = C.NemoRelayCodecDecodeFn(C.goCodecDecodeTrampoline)
		o.codecEncode = C.NemoRelayCodecEncodeFn(C.goCodecEncodeTrampoline)
		o.codecUserData = id
		o.codecFreeFn = C.NemoRelayFreeFn(C.goFreeTrampoline)
	}
}

// CodecHandle wraps an opaque FFI codec handle that carries both request
// codec (decode/encode) and response codec (decode_response) implementations.
// Create via [NewOpenAIChatCodec], [NewOpenAIResponsesCodec],
// [NewAnthropicMessagesCodec], or [NewGeminiGenerateContentCodec]. The handle is
// automatically freed when garbage collected.
type CodecHandle struct {
	ptr *C.FfiCodecHandle
}

// NewOpenAIChatCodec creates a codec for the OpenAI Chat Completions API.
//
// The returned handle can be passed to [WithLLMCodec] or
// [WithLLMResponseCodec] to enable structured request and response handling for
// OpenAI Chat payloads.
func NewOpenAIChatCodec() *CodecHandle {
	h := &CodecHandle{ptr: C.nemo_relay_openai_chat_codec_new()}
	runtime.SetFinalizer(h, func(h *CodecHandle) {
		if h.ptr != nil {
			C.nemo_relay_codec_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// NewOpenAIResponsesCodec creates a codec for the OpenAI Responses API.
//
// The returned handle can be passed to [WithLLMCodec] or
// [WithLLMResponseCodec] to enable structured request and response handling for
// OpenAI Responses payloads.
func NewOpenAIResponsesCodec() *CodecHandle {
	h := &CodecHandle{ptr: C.nemo_relay_openai_responses_codec_new()}
	runtime.SetFinalizer(h, func(h *CodecHandle) {
		if h.ptr != nil {
			C.nemo_relay_codec_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// NewAnthropicMessagesCodec creates a codec for the Anthropic Messages API.
//
// The returned handle can be passed to [WithLLMCodec] or
// [WithLLMResponseCodec] to enable structured request and response handling for
// Anthropic Messages payloads.
func NewAnthropicMessagesCodec() *CodecHandle {
	h := &CodecHandle{ptr: C.nemo_relay_anthropic_messages_codec_new()}
	runtime.SetFinalizer(h, func(h *CodecHandle) {
		if h.ptr != nil {
			C.nemo_relay_codec_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// NewGeminiGenerateContentCodec creates a codec for the Gemini generateContent API.
//
// The returned handle can be passed to [WithLLMCodec] or
// [WithLLMResponseCodec] to enable structured request and response handling for
// Gemini generateContent payloads.
func NewGeminiGenerateContentCodec() *CodecHandle {
	h := &CodecHandle{ptr: C.nemo_relay_gemini_generate_content_codec_new()}
	runtime.SetFinalizer(h, func(h *CodecHandle) {
		if h.ptr != nil {
			C.nemo_relay_codec_free(h.ptr)
			h.ptr = nil
		}
	})
	return h
}

// WithLLMResponseCodec sets the response codec for this LLM call.
// Pass a CodecHandle created by [NewOpenAIChatCodec], [NewOpenAIResponsesCodec],
// [NewAnthropicMessagesCodec], or [NewGeminiGenerateContentCodec].
// The codec handle is kept alive for the duration of the FFI call via
// runtime.KeepAlive, so it is safe to pass an inline-constructed handle.
func WithLLMResponseCodec(codec *CodecHandle) LLMCallOption {
	return func(o *llmCallOptions) {
		if codec != nil {
			o.responseCodec = codec.ptr
			o.responseCodecHandle = codec
		}
	}
}

// WithLLMTimestamp records an explicit time.Time on a manual LLM Start or End
// event. The value is converted to UTC Unix microseconds at the FFI boundary;
// sub-microsecond precision is truncated. Omit this option to use the current
// runtime time for Start events or the runtime default for End events.
func WithLLMTimestamp(timestamp time.Time) LLMCallOption {
	return func(o *llmCallOptions) {
		o.timestamp = cTimestampMicros(timestamp)
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
	if o.timestamp != nil {
		C.free(unsafe.Pointer(o.timestamp))
	}
	// responseCodec is borrowed from a CodecHandle kept alive via
	// responseCodecHandle + runtime.KeepAlive — do not free here.
	// Codec closure cleanup is handled by the FFI free_fn callback.
}

// LlmCall starts an LLM call lifecycle and returns an [LLMHandle]. This emits a
// Start event to all subscribers. The caller is responsible for ending the call
// with [LlmCallEnd] when the LLM responds. For a higher-level API that manages
// the full lifecycle automatically, use [LlmCallExecute] or
// [LlmStreamCallExecute] instead.
//
// The name identifies the LLM provider/model, and request is an LLMRequest-shaped
// value ({headers, content}) that will be serialized to JSON. Optional parameters
// can be set via [LLMCallOption] values. The emitted Start event records the
// request after sanitize-request guardrails. Request and execution intercepts
// run only through [LlmCallExecute] and [LlmStreamCallExecute].
func LlmCall(name string, request interface{}, opts ...LLMCallOption) (*LLMHandle, error) {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	requestJSON, err := jsonMarshal(request)
	if err != nil {
		return nil, err
	}

	cName := C.CString(name)
	cRequest := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cName))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.FfiLLMHandle
	status := C.nemo_relay_llm_call(cName, cRequest, o.parent, C.uint32_t(o.attributes), o.data, o.metadata, o.modelName, o.timestamp, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newLLMHandle(out), nil
}

// LlmCallEnd completes an LLM call that was previously started with [LlmCall].
// It emits an End event to all subscribers with the provided response JSON. The
// handle must have been returned by a prior [LlmCall] invocation. The emitted
// End event records response after sanitize-response guardrails; [WithLLMData]
// is used only when the sanitized response is JSON null. Response intercepts
// run only through [LlmCallExecute] and [LlmStreamCallExecute].
func LlmCallEnd(handle *LLMHandle, response json.RawMessage, opts ...LLMCallOption) error {
	o := &llmCallOptions{}
	for _, opt := range opts {
		opt(o)
	}
	defer freeLLMOpts(o)

	cResponse := C.CString(string(response))
	defer C.free(unsafe.Pointer(cResponse))

	return checkStatus(C.nemo_relay_llm_call_end(handle.ptr, cResponse, o.data, o.metadata, o.timestamp))
}

// LlmCallExecute runs a complete LLM call lifecycle through the full
// middleware pipeline: conditional-execution guardrails (on raw request),
// request intercepts, sanitize-request guardrails for the emitted Start event
// payload, execution intercepts, the provided fn, and sanitize-response
// guardrails for the emitted End event payload.
// On rejection, only a standalone Mark event is emitted (no Start/End pair)
// and GuardrailRejected is returned. This is the recommended high-level API
// for non-streaming LLM invocations. Sanitize guardrails do not rewrite the
// request passed into fn or the value returned to the caller.
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
	status := C.nemo_relay_llm_call_execute(
		cName, cRequest,
		C.NemoRelayLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		o.codecDecode, o.codecEncode,
		o.codecUserData, o.codecFreeFn,
		o.responseCodec,
		&out,
	)
	runtime.KeepAlive(o.responseCodecHandle)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := json.RawMessage(C.GoString(out))
	C.nemo_relay_string_free(out)
	return result, nil
}

// LlmStreamCallExecute runs a streaming LLM call lifecycle. Like
// [LlmCallExecute], conditional-execution guardrails run first on the raw
// request. Sanitize-request guardrails affect the emitted Start event payload,
// while sanitize-response guardrails affect only the aggregated End event
// payload. If accepted, it runs the remaining middleware pipeline and returns
// an [LlmStream] that yields individual SSE (Server-Sent Event) chunks.
// Stream execution intercepts are applied to each chunk as it is consumed.
// The caller must call [LlmStream.Next] repeatedly until [io.EOF] is
// returned, or call [LlmStream.Close] to stop early. Close waits for producer
// cleanup, finalizes the partial response, and returns any cleanup error.
//
// The optional collector callback is invoked with each intercepted chunk string,
// allowing the caller to accumulate chunks for aggregation. The optional
// finalizer callback is invoked once when the stream is exhausted or closed
// early and must return a JSON string representing the aggregated response.
// Pass nil for either to use the default no-op behavior.
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

	// Pass nil collector/finalizer to the FFI. The FFI collector/finalizer
	// callbacks lack user_data parameters, making them unsuitable for
	// concurrent streams (all streams would share a single global
	// callback). Instead, we store the collector/finalizer on the
	// returned LlmStream and invoke them from LlmStream.Next(), which
	// provides natural per-stream isolation.
	cCollector := C.makeOptCollectorCb(nil)
	cFinalizer := C.makeOptFinalizerCb(nil)

	var out *C.FfiStream
	status := C.nemo_relay_llm_stream_call_execute(
		cName, cRequest,
		C.NemoRelayLlmExecFn(C.goLlmExecTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
		cCollector,
		cFinalizer,
		o.parent, C.uint32_t(o.attributes),
		o.data, o.metadata,
		o.modelName,
		o.codecDecode, o.codecEncode,
		o.codecUserData, o.codecFreeFn,
		o.responseCodec,
		&out,
	)
	runtime.KeepAlive(o.responseCodecHandle)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return newLlmStream(out, collector, finalizer), nil
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (Tool)
// ---------------------------------------------------------------------------

// RegisterEventMetadataInjector registers a global event metadata injector.
// Injectors run in ascending priority order and may only add metadata keys that
// are not already present on the event.
func RegisterEventMetadataInjector(name string, priority int32, fn EventMetadataInjectorFunc) error {
	if fn == nil {
		return errEventMetadataInjectorCallbackNil
	}
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_event_metadata_injector(
		cName,
		C.int32_t(priority),
		C.NemoRelayEventMetadataInjectorFn(C.goEventMetadataInjectorTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterEventMetadataInjector removes a global event metadata injector.
func DeregisterEventMetadataInjector(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_event_metadata_injector(cName))
}

func registerEventSanitizer(name string, priority int32, fn EventSanitizeFunc, kind int) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var status C.int32_t
	switch kind {
	case 0:
		status = C.nemo_relay_register_mark_sanitize_guardrail(cName, C.int32_t(priority), C.NemoRelayEventSanitizeFn(C.goEventSanitizeTrampoline), id, C.NemoRelayFreeFn(C.goFreeTrampoline))
	case 1:
		status = C.nemo_relay_register_scope_sanitize_start_guardrail(cName, C.int32_t(priority), C.NemoRelayEventSanitizeFn(C.goEventSanitizeTrampoline), id, C.NemoRelayFreeFn(C.goFreeTrampoline))
	default:
		status = C.nemo_relay_register_scope_sanitize_end_guardrail(cName, C.int32_t(priority), C.NemoRelayEventSanitizeFn(C.goEventSanitizeTrampoline), id, C.NemoRelayFreeFn(C.goFreeTrampoline))
	}
	return checkStatus(status)
}

// RegisterMarkSanitizeGuardrail registers a global mark event sanitizer.
func RegisterMarkSanitizeGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return registerEventSanitizer(name, priority, fn, 0)
}

// DeregisterMarkSanitizeGuardrail removes a global mark event sanitizer.
func DeregisterMarkSanitizeGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_mark_sanitize_guardrail(cName))
}

// RegisterScopeSanitizeStartGuardrail registers a global scope-start event sanitizer.
func RegisterScopeSanitizeStartGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return registerEventSanitizer(name, priority, fn, 1)
}

// DeregisterScopeSanitizeStartGuardrail removes a global scope-start event sanitizer.
func DeregisterScopeSanitizeStartGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_scope_sanitize_start_guardrail(cName))
}

// RegisterScopeSanitizeEndGuardrail registers a global scope-end event sanitizer.
func RegisterScopeSanitizeEndGuardrail(name string, priority int32, fn EventSanitizeFunc) error {
	return registerEventSanitizer(name, priority, fn, 2)
}

// DeregisterScopeSanitizeEndGuardrail removes a global scope-end event sanitizer.
func DeregisterScopeSanitizeEndGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_scope_sanitize_end_guardrail(cName))
}

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
	return checkStatus(C.nemo_relay_register_tool_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeRequestGuardrail removes a previously registered tool
// sanitize-request guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_tool_sanitize_request_guardrail(cName))
}

// RegisterToolSanitizeResponseGuardrail registers a guardrail that sanitizes
// tool response data before it is returned to the caller. The callback receives
// the tool name and response JSON and must return the (possibly modified)
// response. Guardrails are invoked in priority order (lower values run first).
func RegisterToolSanitizeResponseGuardrail(name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_tool_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolSanitizeResponseGuardrail removes a previously registered tool
// sanitize-response guardrail by name. Returns a NotFound error if no guardrail
// with the given name is registered.
func DeregisterToolSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_tool_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nemo_relay_register_tool_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayToolConditionalFn(C.goToolConditionalTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolConditionalExecutionGuardrail removes a previously registered
// tool conditional-execution guardrail by name. Returns a NotFound error if no
// guardrail with the given name is registered.
func DeregisterToolConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_tool_conditional_execution_guardrail(cName))
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
	return checkStatus(C.nemo_relay_register_tool_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolRequestIntercept removes a previously registered tool request
// intercept by name.
func DeregisterToolRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_tool_request_intercept(cName))
}

// RegisterToolExecutionIntercept registers an execution intercept following
// the middleware chain pattern. execFn is called with the args and a `next`
// function. Call `next` to invoke the next intercept or original
// implementation; skip calling `next` to short-circuit the chain.
func RegisterToolExecutionIntercept(name string, priority int32, execFn ToolExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_tool_execution_intercept(
		cName, C.int32_t(priority),
		C.NemoRelayToolExecInterceptCb(C.goToolExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterToolExecutionIntercept removes a previously registered tool
// execution intercept by name.
func DeregisterToolExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_tool_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Guardrail/Intercept registration (LLM)
// ---------------------------------------------------------------------------

// RegisterLlmSanitizeRequestGuardrail registers a codec-aware LLM request
// sanitizer. Returning omit true removes only the emitted payload.
func RegisterLlmSanitizeRequestGuardrail(name string, priority int32, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_llm_sanitize_request_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayLlmSanitizeRequestCb(C.goLlmRequestTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeRequestGuardrail removes a previously registered LLM
// sanitize-request guardrail by name.
func DeregisterLlmSanitizeRequestGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_sanitize_request_guardrail(cName))
}

// RegisterLlmSanitizeResponseGuardrail registers a codec-aware LLM response
// sanitizer. Returning omit true removes only the emitted payload.
func RegisterLlmSanitizeResponseGuardrail(name string, priority int32, fn LLMResponseFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_llm_sanitize_response_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayLlmSanitizeResponseCb(C.goLlmResponseTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmSanitizeResponseGuardrail removes a previously registered LLM
// sanitize-response guardrail by name.
func DeregisterLlmSanitizeResponseGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_sanitize_response_guardrail(cName))
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
	return checkStatus(C.nemo_relay_register_llm_conditional_execution_guardrail(
		cName, C.int32_t(priority),
		C.NemoRelayLlmConditionalCb(C.goLlmConditionalTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmConditionalExecutionGuardrail removes a previously registered
// LLM conditional-execution guardrail by name.
func DeregisterLlmConditionalExecutionGuardrail(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_conditional_execution_guardrail(cName))
}

// RegisterLlmRequestIntercept registers an intercept that transforms the LLM
// request (headers, content, and optionally the annotated request) before the
// call is made. Intercepts run in priority order (lower values first). When
// breakChain is true, no lower-priority intercepts in the chain are invoked
// after this one. The callback receives the intercept name, headers, content,
// and annotated JSON (nil if no Codec resolved).
func RegisterLlmRequestIntercept(name string, priority int32, breakChain bool, fn LLMRequestInterceptFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_llm_request_intercept(
		cName, C.int32_t(priority), C._Bool(breakChain),
		C.NemoRelayLlmRequestInterceptCb(C.goLlmRequestInterceptTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmRequestIntercept removes a previously registered LLM request
// intercept by name.
func DeregisterLlmRequestIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_request_intercept(cName))
}

// RegisterLlmExecutionIntercept registers an execution intercept following
// the middleware chain pattern. execFn is called with the request parameters
// and a `next` function. Call `next` to invoke the next intercept or original
// implementation; skip calling `next` to short-circuit the chain.
func RegisterLlmExecutionIntercept(name string, priority int32, execFn LLMExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_llm_execution_intercept(
		cName, C.int32_t(priority),
		C.NemoRelayLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmExecutionIntercept removes a previously registered LLM
// execution intercept by name.
func DeregisterLlmExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_execution_intercept(cName))
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
	return checkStatus(C.nemo_relay_register_llm_stream_execution_intercept(
		cName, C.int32_t(priority),
		C.NemoRelayLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterLlmStreamExecutionIntercept removes a previously registered LLM
// stream execution intercept by name.
func DeregisterLlmStreamExecutionIntercept(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_llm_stream_execution_intercept(cName))
}

// ---------------------------------------------------------------------------
// Subscriber registration
// ---------------------------------------------------------------------------

// RegisterSubscriber registers a named event subscriber that will be called for
// every lifecycle event (Start, End, Mark) emitted by the runtime. Native event
// calls enqueue subscriber delivery and return without waiting for callbacks.
// Subscribers are identified by a unique name; registering a duplicate returns an
// AlreadyExists error. The callback receives an owned [Event] snapshot that is
// safe to retain after the callback returns.
func RegisterSubscriber(name string, fn EventSubscriberFunc) error {
	id := registerClosure(fn)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_register_subscriber(
		cName,
		C.NemoRelayEventSubscriberFn(C.goEventSubscriberTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// DeregisterSubscriber removes a named event subscriber for future emissions.
// Already queued snapshots may still run. Returns a NotFound error if no
// subscriber with the given name is registered.
func DeregisterSubscriber(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_deregister_subscriber(cName))
}

// FlushSubscribers waits for subscriber callbacks queued before this call and
// events emitted transitively by those callbacks to finish. Native
// event-producing APIs enqueue subscriber work and return without waiting for
// observer callbacks. Call this function outside native subscriber callbacks.
// A re-entrant call returns without waiting to avoid blocking the dispatcher,
// so callbacks later in the same dispatch snapshot can still run.
func FlushSubscribers() error {
	return checkStatus(C.nemo_relay_flush_subscribers())
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

// ScopeStack represents an isolated scope stack for per-request/per-goroutine isolation.
// Each ScopeStack has its own root scope and is independent of other scope stacks.
type ScopeStack struct {
	ptr *C.FfiScopeStack
}

// PropagationContext is the versioned, transport-neutral causal context used
// to continue Relay work in another process.
type PropagationContext struct {
	Version    uint16  `json:"version"`
	RootUUID   *string `json:"root_uuid,omitempty"`
	ParentUUID string  `json:"parent_uuid"`
}

// ToJSON serializes a validated propagation context for application-managed transport.
func (context PropagationContext) ToJSON() (string, error) {
	if err := validatePropagationContext(context); err != nil {
		return "", err
	}
	// PropagationContext has only JSON-native fields, so marshaling cannot fail.
	payload, _ := json.Marshal(context)
	return string(payload), nil
}

// ToTraceparent converts a rooted propagation context to a W3C traceparent value.
func (context PropagationContext) ToTraceparent() (string, error) {
	payload, err := context.ToJSON()
	if err != nil {
		return "", err
	}
	cPayload := C.CString(payload)
	defer C.free(unsafe.Pointer(cPayload))
	var out *C.char
	if err := checkStatus(C.nemo_relay_propagation_context_to_traceparent(cPayload, &out)); err != nil {
		return "", err
	}
	defer C.nemo_relay_string_free(out)
	return C.GoString(out), nil
}

// PropagationContextFromJSON deserializes and validates a transport context.
func PropagationContextFromJSON(value string) (PropagationContext, error) {
	var context PropagationContext
	if err := json.Unmarshal([]byte(value), &context); err != nil {
		return PropagationContext{}, err
	}
	if err := validatePropagationContext(context); err != nil {
		return PropagationContext{}, err
	}
	return context, nil
}

func validatePropagationContext(context PropagationContext) error {
	stack, err := NewScopeStackFromPropagation(context)
	if err != nil {
		return err
	}
	stack.Close()
	return nil
}

// CapturePropagationContext captures the current Relay causal parent and root.
func CapturePropagationContext() (PropagationContext, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_capture_propagation_context_json(&out)); err != nil {
		return PropagationContext{}, err
	}
	defer C.nemo_relay_string_free(out)
	return PropagationContextFromJSON(C.GoString(out))
}

// CaptureRootlessPropagationContext captures the current causal parent without
// a propagation root, so a receiver starts a new observability trace.
func CaptureRootlessPropagationContext() (PropagationContext, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_capture_rootless_propagation_context_json(&out)); err != nil {
		return PropagationContext{}, err
	}
	defer C.nemo_relay_string_free(out)
	return PropagationContextFromJSON(C.GoString(out))
}

// CapturePropagationContextWithRoot captures the current parent with an
// application-supplied stable session root. Pass nil when no root is known.
func CapturePropagationContextWithRoot(rootUUID *string) (PropagationContext, error) {
	var cRoot *C.char
	if rootUUID != nil {
		cRoot = C.CString(*rootUUID)
		defer C.free(unsafe.Pointer(cRoot))
	}
	var out *C.char
	if err := checkStatus(C.nemo_relay_capture_propagation_context_with_root_json(cRoot, &out)); err != nil {
		return PropagationContext{}, err
	}
	defer C.nemo_relay_string_free(out)
	return PropagationContextFromJSON(C.GoString(out))
}

// CaptureTraceparent captures the current Relay context as a W3C traceparent value.
func CaptureTraceparent() (string, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_capture_traceparent(&out)); err != nil {
		return "", err
	}
	defer C.nemo_relay_string_free(out)
	return C.GoString(out), nil
}

// NewScopeStack creates a new isolated scope stack.
// The caller must call Close() when done.
func NewScopeStack() (*ScopeStack, error) {
	return newScopeStackFunc()
}

// NewScopeStackFromPropagation creates an isolated stack seeded from a
// received propagation context. The caller must call Close when done.
func NewScopeStackFromPropagation(context PropagationContext) (*ScopeStack, error) {
	payload, err := json.Marshal(context)
	if err != nil {
		return nil, err
	}
	cPayload := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cPayload))
	var ptr *C.FfiScopeStack
	status := C.nemo_relay_scope_stack_create_from_propagation_json(cPayload, &ptr)
	return checkedValue(int32(status), &ScopeStack{ptr: ptr})
}

// Close frees the scope stack. After calling Close, the ScopeStack must not be used.
func (s *ScopeStack) Close() {
	if s.ptr != nil {
		C.nemo_relay_scope_stack_free(s.ptr)
		s.ptr = nil
	}
}

// Run binds this scope stack to the current OS thread and executes fn.
// The calling goroutine is locked to the OS thread for the duration of fn.
// All NeMo Relay scope operations within fn will use this scope stack.
//
// This is the canonical way to propagate a scope stack to a worker goroutine.
func (s *ScopeStack) Run(fn func()) {
	runtime.LockOSThread()
	var binding *C.FfiThreadScopeStackBinding
	if err := checkStatus(C.nemo_relay_scope_stack_capture_thread(&binding)); err != nil {
		runtime.UnlockOSThread()
		panic(err)
	}
	defer func() {
		status := C.nemo_relay_scope_stack_restore_thread(binding)
		if err := checkStatus(status); err != nil {
			runtime.UnlockOSThread()
			panic(err)
		}
		runtime.UnlockOSThread()
	}()
	if err := checkStatus(C.nemo_relay_scope_stack_set_thread(s.ptr)); err != nil {
		panic(err)
	}
	fn()
}

// ScopeStackActive returns true if the current OS thread has an explicitly-bound
// scope stack (set via ScopeStack.Run or directly via set_thread), or false if
// only the auto-created default is present.
//
// This function must be called from a goroutine locked to an OS thread
// (e.g. inside ScopeStack.Run) for the result to be meaningful.
func ScopeStackActive() bool {
	return bool(C.nemo_relay_scope_stack_active())
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
	return newAtifExporterFunc(sessionID, agentName, agentVersion, modelName)
}

// Register registers the exporter as an event subscriber with the given name.
func (e *AtifExporter) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_atif_exporter_register(e.ptr, cName)
	return checkStatus(status)
}

// Deregister removes the exporter subscriber by name.
func (e *AtifExporter) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_atif_exporter_deregister(cName)
	return checkStatus(status)
}

// ExportJSON exports collected events as an ATIF trajectory JSON string.
func (e *AtifExporter) ExportJSON() (json.RawMessage, error) {
	var cOut *C.char
	status := C.nemo_relay_atif_exporter_export(e.ptr, &cOut)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nemo_relay_string_free(cOut)
	return json.RawMessage(C.GoString(cOut)), nil
}

// Clear removes all collected events.
func (e *AtifExporter) Clear() {
	C.nemo_relay_atif_exporter_clear(e.ptr)
}

// Close frees the exporter handle.
func (e *AtifExporter) Close() {
	if e.ptr != nil {
		C.nemo_relay_atif_exporter_free(e.ptr)
		e.ptr = nil
	}
}

// ---------------------------------------------------------------------------
// ATOF JSONL Exporter
// ---------------------------------------------------------------------------

// AtofExporterMode controls how an ATOF JSONL exporter opens its output file.
type AtofExporterMode string

const (
	// AtofExporterModeAppend appends events to an existing file.
	AtofExporterModeAppend AtofExporterMode = "append"
	// AtofExporterModeOverwrite truncates an existing file when the exporter is created.
	AtofExporterModeOverwrite AtofExporterMode = "overwrite"
)

// MarkProjection selects how mark events are represented by full and OpenInference projections.
type MarkProjection string

const (
	MarkProjectionInherit MarkProjection = "inherit"
	MarkProjectionEvent   MarkProjection = "event"
	MarkProjectionTool    MarkProjection = "tool"
)

// OtlpAttributeMapping maps a projected OpenTelemetry attribute to an alias.
type OtlpAttributeMapping struct {
	Key   string `json:"key"`
	Alias string `json:"alias"`
}

// AtofExporterConfig configures one tagged ATOF sink.
type AtofExporterConfig struct {
	Sink AtofSinkConfigurer `json:"-"`
}

// MarshalJSON serializes the selected tagged sink directly, matching the Rust API.
func (config AtofExporterConfig) MarshalJSON() ([]byte, error) {
	if config.Sink == nil {
		return json.Marshal(NewAtofFileSinkConfig())
	}
	return json.Marshal(config.Sink)
}

// AtofSinkConfigurer is one ATOF exporter destination.
type AtofSinkConfigurer interface {
	atofExporterSink()
}

// AtofFileSinkConfig configures one filesystem ATOF JSONL destination.
type AtofFileSinkConfig struct {
	OutputDirectory string           `json:"output_directory,omitempty"`
	Mode            AtofExporterMode `json:"mode,omitempty"`
	Filename        string           `json:"filename,omitempty"`
}

func (AtofFileSinkConfig) atofExporterSink() {
	// This marker method intentionally has no runtime behavior.
}

// MarshalJSON serializes the fixed file sink discriminator.
func (config AtofFileSinkConfig) MarshalJSON() ([]byte, error) {
	type alias AtofFileSinkConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{Type: "file", alias: alias(config)})
}

// AtofEndpointTransport controls how an ATOF streaming endpoint receives events.
type AtofEndpointTransport string

const (
	// AtofEndpointTransportHTTPPost sends each event as one HTTP POST JSONL record.
	AtofEndpointTransportHTTPPost AtofEndpointTransport = "http_post"
	// AtofEndpointTransportWebsocket sends each event as one WebSocket JSON text message.
	AtofEndpointTransportWebsocket AtofEndpointTransport = "websocket"
	// AtofEndpointTransportNDJSON sends events over one long-lived HTTP NDJSON upload.
	AtofEndpointTransportNDJSON AtofEndpointTransport = "ndjson"
)

// AtofEndpointFieldNamePolicy controls endpoint-local field name transformations.
type AtofEndpointFieldNamePolicy string

const (
	// AtofEndpointFieldNamePolicyPreserve sends canonical ATOF field names unchanged.
	AtofEndpointFieldNamePolicyPreserve AtofEndpointFieldNamePolicy = "preserve"
	// AtofEndpointFieldNamePolicyReplaceDots replaces dots in JSON object keys with underscores.
	AtofEndpointFieldNamePolicyReplaceDots AtofEndpointFieldNamePolicy = "replace_dots"
)

// AtofStreamSinkConfig configures one streaming destination for raw ATOF events.
type AtofStreamSinkConfig struct {
	URL             string                      `json:"url"`
	Transport       AtofEndpointTransport       `json:"transport,omitempty"`
	Headers         map[string]string           `json:"headers,omitempty"`
	HeaderEnv       map[string]string           `json:"header_env,omitempty"`
	TimeoutMillis   uint64                      `json:"timeout_millis,omitempty"`
	FieldNamePolicy AtofEndpointFieldNamePolicy `json:"field_name_policy,omitempty"`
}

func (AtofStreamSinkConfig) atofExporterSink() {
	// This marker method intentionally has no runtime behavior.
}

// MarshalJSON serializes the fixed stream sink discriminator.
func (config AtofStreamSinkConfig) MarshalJSON() ([]byte, error) {
	type alias AtofStreamSinkConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{Type: "stream", alias: alias(config)})
}

// NewAtofExporterConfig returns a config initialized with native defaults.
func NewAtofExporterConfig() AtofExporterConfig {
	return AtofExporterConfig{Sink: NewAtofFileSinkConfig()}
}

// NewAtofFileSinkConfig returns a file sink initialized with native defaults.
func NewAtofFileSinkConfig() AtofFileSinkConfig {
	return AtofFileSinkConfig{Mode: AtofExporterModeAppend}
}

// NewAtofStreamSinkConfig returns an HTTP POST stream sink with native defaults.
func NewAtofStreamSinkConfig(url string) AtofStreamSinkConfig {
	return AtofStreamSinkConfig{
		URL:             url,
		Transport:       AtofEndpointTransportHTTPPost,
		TimeoutMillis:   3000,
		FieldNamePolicy: AtofEndpointFieldNamePolicyPreserve,
	}
}

// AtofExporter writes raw NeMo Relay ATOF lifecycle events as JSONL.
type AtofExporter struct {
	ptr unsafe.Pointer
}

// NewAtofExporter creates a new single-sink ATOF exporter.
func NewAtofExporter(config AtofExporterConfig) (*AtofExporter, error) {
	return newAtofExporterFunc(config)
}

// Path returns the JSONL output path, or nil for a stream-backed exporter.
func (e *AtofExporter) Path() (*string, error) {
	var cOut *C.char
	status := C.nemo_relay_atof_exporter_path(e.ptr, &cOut)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	if cOut == nil {
		return nil, nil
	}
	defer C.nemo_relay_string_free(cOut)
	path := C.GoString(cOut)
	return &path, nil
}

// Register registers the exporter as a global event subscriber.
func (e *AtofExporter) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_atof_exporter_register(e.ptr, cName)
	return checkStatus(status)
}

// Deregister removes the exporter subscriber by name.
func (e *AtofExporter) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_atof_exporter_deregister(cName)
	return checkStatus(status)
}

// ForceFlush, outside a native subscriber callback, waits for queued subscriber delivery and then
// flushes the configured file sink or asks the configured stream sink to drain up to its timeout.
// A re-entrant call does not establish the delivery barrier. A stream timeout is logged and does
// not by itself return an error.
func (e *AtofExporter) ForceFlush() error {
	status := C.nemo_relay_atof_exporter_force_flush(e.ptr)
	return checkStatus(status)
}

// Shutdown, outside a native subscriber callback, waits for queued subscriber delivery and then
// flushes the configured file sink or asks the configured stream sink to drain and close up to its
// timeout. A re-entrant call does not establish the delivery barrier. A stream timeout is logged
// and does not by itself return an error.
func (e *AtofExporter) Shutdown() error {
	status := C.nemo_relay_atof_exporter_shutdown(e.ptr)
	return checkStatus(status)
}

// Close frees the exporter handle.
func (e *AtofExporter) Close() {
	if e.ptr != nil {
		C.nemo_relay_atof_exporter_free(e.ptr)
		e.ptr = nil
	}
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

// OpenTelemetryTransport configures which OTLP transport to use.
type OpenTelemetryTransport string

const (
	// OpenTelemetryTransportHTTPBinary uses OTLP/HTTP protobuf export.
	OpenTelemetryTransportHTTPBinary OpenTelemetryTransport = "http_binary"
	// OpenTelemetryTransportGrpc uses OTLP/gRPC export.
	OpenTelemetryTransportGrpc OpenTelemetryTransport = "grpc"
)

// OpenTelemetryType selects the semantic projection emitted by an exporter.
type OpenTelemetryType string

const (
	OpenTelemetryTypeFull          OpenTelemetryType = "full"
	OpenTelemetryTypeGenAI         OpenTelemetryType = "gen_ai"
	OpenTelemetryTypeOpenInference OpenTelemetryType = "openinference"
)

// OpenTelemetryConfig configures the OpenTelemetry subscriber.
//
// Create it with [NewOpenTelemetryConfig], then mutate fields as needed before
// passing it to [NewOpenTelemetrySubscriber].
type OpenTelemetryConfig struct {
	Type      OpenTelemetryType
	Transport OpenTelemetryTransport
	Endpoint  string
	Headers   map[string]string
	// HeaderEnv maps outbound header names to environment variables resolved at activation.
	HeaderEnv               map[string]string
	ResourceAttributes      map[string]string
	ServiceName             string
	ServiceNamespace        string
	ServiceVersion          string
	InstrumentationScope    string
	Timeout                 time.Duration
	CompletedSpanContextTTL *time.Duration
	MarkProjection          MarkProjection
	MarkExcludeNames        []string
	AttributeMappings       []OtlpAttributeMapping
	PromoteMetadataPrefixes []string
}

// NewOpenTelemetryConfig returns a typed config for the required endpoint.
func NewOpenTelemetryConfig(otelType OpenTelemetryType, endpoint string) OpenTelemetryConfig {
	completedSpanContextTTL := 60 * time.Second
	return OpenTelemetryConfig{
		Type:                    otelType,
		Transport:               OpenTelemetryTransportHTTPBinary,
		Endpoint:                endpoint,
		Headers:                 map[string]string{},
		HeaderEnv:               map[string]string{},
		ResourceAttributes:      map[string]string{},
		ServiceName:             "unknown_service",
		InstrumentationScope:    "opentelemetry",
		Timeout:                 3 * time.Second,
		CompletedSpanContextTTL: &completedSpanContextTTL,
		MarkProjection:          MarkProjectionInherit,
		MarkExcludeNames:        []string{"llm.chunk"},
		AttributeMappings:       []OtlpAttributeMapping{},
		PromoteMetadataPrefixes: []string{},
	}
}

// OpenTelemetrySubscriber exports NeMo Relay lifecycle events to an OpenTelemetry server.
type OpenTelemetrySubscriber struct {
	ptr unsafe.Pointer
}

// OpenTelemetryRuntimeDiagnostic is one bounded aggregate of an OTLP exporter
// or event-processing failure.
type OpenTelemetryRuntimeDiagnostic struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Count   uint64 `json:"count"`
}

const openTelemetryEndpointRequiredMessage = "endpoint is required"

func decodeOpenTelemetryRuntimeDiagnostics(out *C.char) ([]OpenTelemetryRuntimeDiagnostic, error) {
	defer C.nemo_relay_string_free(out)
	var diagnostics []OpenTelemetryRuntimeDiagnostic
	if err := jsonUnmarshal([]byte(C.GoString(out)), &diagnostics); err != nil {
		return nil, err
	}
	return diagnostics, nil
}

func normalizeOpenTelemetryConfig(config OpenTelemetryConfig) (OpenTelemetryConfig, error) {
	if config.Transport == "" {
		config.Transport = OpenTelemetryTransportHTTPBinary
	}
	if config.Type == "" {
		return config, fmt.Errorf("type is required")
	}
	if config.Endpoint == "" {
		return config, fmt.Errorf(openTelemetryEndpointRequiredMessage)
	}
	if config.ServiceName == "" {
		config.ServiceName = "unknown_service"
	}
	if config.InstrumentationScope == "" {
		config.InstrumentationScope = "opentelemetry"
	}
	if config.Timeout == 0 {
		config.Timeout = 3 * time.Second
	}
	if config.CompletedSpanContextTTL == nil {
		completedSpanContextTTL := 60 * time.Second
		config.CompletedSpanContextTTL = &completedSpanContextTTL
	}
	if *config.CompletedSpanContextTTL <= 0 {
		return config, fmt.Errorf("completed span context TTL must be greater than 0")
	}
	if err := requireWholeMillisecondDuration("completed span context TTL", *config.CompletedSpanContextTTL); err != nil {
		return config, err
	}
	if config.Headers == nil {
		config.Headers = map[string]string{}
	}
	if config.HeaderEnv == nil {
		config.HeaderEnv = map[string]string{}
	}
	if config.ResourceAttributes == nil {
		config.ResourceAttributes = map[string]string{}
	}
	if config.MarkProjection == "" {
		config.MarkProjection = MarkProjectionInherit
	}
	if config.MarkExcludeNames == nil {
		config.MarkExcludeNames = []string{"llm.chunk"}
	}
	if config.AttributeMappings == nil {
		config.AttributeMappings = []OtlpAttributeMapping{}
	}
	if config.PromoteMetadataPrefixes == nil {
		config.PromoteMetadataPrefixes = []string{}
	}
	return config, nil
}

func optionalCString(value string) *C.char {
	if value == "" {
		return nil
	}
	return C.CString(value)
}

// NewOpenTelemetrySubscriber creates a new OpenTelemetry subscriber from config.
func NewOpenTelemetrySubscriber(config OpenTelemetryConfig) (*OpenTelemetrySubscriber, error) {
	config, err := normalizeOpenTelemetryConfig(config)
	if err != nil {
		return nil, err
	}

	cTransport := C.CString(string(config.Transport))
	defer C.free(unsafe.Pointer(cTransport))
	cType := C.CString(string(config.Type))
	defer C.free(unsafe.Pointer(cType))

	cEndpoint := C.CString(config.Endpoint)
	defer C.free(unsafe.Pointer(cEndpoint))

	headersJSON, err := jsonMarshal(config.Headers)
	if err != nil {
		return nil, err
	}
	cHeadersJSON := C.CString(string(headersJSON))
	defer C.free(unsafe.Pointer(cHeadersJSON))
	headerEnvJSON, err := jsonMarshal(config.HeaderEnv)
	if err != nil {
		return nil, err
	}
	cHeaderEnvJSON := C.CString(string(headerEnvJSON))
	defer C.free(unsafe.Pointer(cHeaderEnvJSON))

	resourceAttrsJSON, err := jsonMarshal(config.ResourceAttributes)
	if err != nil {
		return nil, err
	}
	cResourceAttrsJSON := C.CString(string(resourceAttrsJSON))
	defer C.free(unsafe.Pointer(cResourceAttrsJSON))

	cServiceName := C.CString(config.ServiceName)
	defer C.free(unsafe.Pointer(cServiceName))

	cServiceNamespace := optionalCString(config.ServiceNamespace)
	defer C.free(unsafe.Pointer(cServiceNamespace))

	cServiceVersion := optionalCString(config.ServiceVersion)
	defer C.free(unsafe.Pointer(cServiceVersion))

	cInstrumentationScope := C.CString(config.InstrumentationScope)
	defer C.free(unsafe.Pointer(cInstrumentationScope))
	cMarkProjection := C.CString(string(config.MarkProjection))
	defer C.free(unsafe.Pointer(cMarkProjection))
	markExcludeNamesJSON, err := jsonMarshal(config.MarkExcludeNames)
	if err != nil {
		return nil, err
	}
	cMarkExcludeNamesJSON := C.CString(string(markExcludeNamesJSON))
	defer C.free(unsafe.Pointer(cMarkExcludeNamesJSON))
	attributeMappingsJSON, err := jsonMarshal(config.AttributeMappings)
	if err != nil {
		return nil, err
	}
	cAttributeMappingsJSON := C.CString(string(attributeMappingsJSON))
	defer C.free(unsafe.Pointer(cAttributeMappingsJSON))
	promoteMetadataPrefixesJSON, err := jsonMarshal(config.PromoteMetadataPrefixes)
	if err != nil {
		return nil, err
	}
	cPromoteMetadataPrefixesJSON := C.CString(string(promoteMetadataPrefixesJSON))
	defer C.free(unsafe.Pointer(cPromoteMetadataPrefixesJSON))

	var ptr unsafe.Pointer
	status := C.nemo_relay_otel_subscriber_create_with_projection_options_v4(
		cType,
		cTransport,
		cEndpoint,
		cHeadersJSON,
		cHeaderEnvJSON,
		cResourceAttrsJSON,
		cServiceName,
		cServiceNamespace,
		cServiceVersion,
		cInstrumentationScope,
		C.uint64_t(config.Timeout/time.Millisecond),
		cMarkProjection,
		cMarkExcludeNamesJSON,
		cAttributeMappingsJSON,
		cPromoteMetadataPrefixesJSON,
		C.uint64_t(*config.CompletedSpanContextTTL/time.Millisecond),
		&ptr,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &OpenTelemetrySubscriber{ptr: ptr}, nil
}

// Register registers the subscriber globally with the given name.
func (s *OpenTelemetrySubscriber) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_otel_subscriber_register(s.ptr, cName)
	return checkStatus(status)
}

// Deregister removes the subscriber by name.
func (s *OpenTelemetrySubscriber) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.nemo_relay_otel_subscriber_deregister(cName)
	return checkStatus(status)
}

// ForceFlush flushes finished spans through the underlying exporter. A successful flush updates
// RuntimeDiagnostics with queue drops observed so far.
func (s *OpenTelemetrySubscriber) ForceFlush() error {
	status := C.nemo_relay_otel_subscriber_force_flush(s.ptr)
	return checkStatus(status)
}

// RuntimeDiagnostics returns a bounded snapshot of exporter and event-processing failures.
func (s *OpenTelemetrySubscriber) RuntimeDiagnostics() ([]OpenTelemetryRuntimeDiagnostic, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_otel_subscriber_runtime_diagnostics_json(s.ptr, &out)); err != nil {
		return nil, err
	}
	return decodeOpenTelemetryRuntimeDiagnostics(out)
}

// Shutdown shuts down the underlying tracer provider.
func (s *OpenTelemetrySubscriber) Shutdown() error {
	status := C.nemo_relay_otel_subscriber_shutdown(s.ptr)
	return checkStatus(status)
}

// Close frees the subscriber handle.
func (s *OpenTelemetrySubscriber) Close() {
	if s.ptr != nil {
		C.nemo_relay_otel_subscriber_free(s.ptr)
		s.ptr = nil
	}
}

// OpenTelemetryLogConfig configures an independent OTLP log subscriber.
type OpenTelemetryLogConfig struct {
	Transport OpenTelemetryTransport
	Endpoint  string
	Headers   map[string]string
	// HeaderEnv maps outbound header names to environment variables resolved at activation.
	HeaderEnv               map[string]string
	ResourceAttributes      map[string]string
	ServiceName             string
	ServiceNamespace        string
	ServiceVersion          string
	InstrumentationScope    string
	Timeout                 time.Duration
	MinimumSeverity         LogSeverity
	MaxQueueSize            uint64
	MaxExportBatchSize      uint64
	ScheduledDelay          time.Duration
	CompletedSpanContextTTL time.Duration
}

// NewOpenTelemetryLogConfig returns log settings initialized with native defaults.
func NewOpenTelemetryLogConfig(endpoint string) OpenTelemetryLogConfig {
	return OpenTelemetryLogConfig{
		Transport:               OpenTelemetryTransportHTTPBinary,
		Endpoint:                endpoint,
		Headers:                 map[string]string{},
		HeaderEnv:               map[string]string{},
		ResourceAttributes:      map[string]string{},
		ServiceName:             "unknown_service",
		InstrumentationScope:    "opentelemetry",
		Timeout:                 3 * time.Second,
		MinimumSeverity:         LogSeverityInfo,
		MaxQueueSize:            2048,
		MaxExportBatchSize:      512,
		ScheduledDelay:          time.Second,
		CompletedSpanContextTTL: 60 * time.Second,
	}
}

// OpenTelemetryLogSubscriber exports sanitized non-metric marks as OTLP logs.
type OpenTelemetryLogSubscriber struct {
	ptr unsafe.Pointer
}

// OpenTelemetryMetricTemporality controls SDK metric aggregation temporality.
type OpenTelemetryMetricTemporality string

const (
	OpenTelemetryMetricTemporalityCumulative OpenTelemetryMetricTemporality = "cumulative"
	OpenTelemetryMetricTemporalityDelta      OpenTelemetryMetricTemporality = "delta"
	OpenTelemetryMetricTemporalityLowMemory  OpenTelemetryMetricTemporality = "low_memory"
)

// OpenTelemetryMetricConfig configures an independent OTLP metric subscriber.
type OpenTelemetryMetricConfig struct {
	Transport OpenTelemetryTransport
	Endpoint  string
	Headers   map[string]string
	// HeaderEnv maps outbound header names to environment variables resolved at activation.
	HeaderEnv            map[string]string
	ResourceAttributes   map[string]string
	ServiceName          string
	ServiceNamespace     string
	ServiceVersion       string
	InstrumentationScope string
	Timeout              time.Duration
	ExportInterval       time.Duration
	Temporality          OpenTelemetryMetricTemporality
	MaxInstruments       uint64
	CardinalityLimit     uint64
}

// NewOpenTelemetryMetricConfig returns metric settings initialized with native defaults.
func NewOpenTelemetryMetricConfig(endpoint string) OpenTelemetryMetricConfig {
	return OpenTelemetryMetricConfig{
		Transport:            OpenTelemetryTransportHTTPBinary,
		Endpoint:             endpoint,
		Headers:              map[string]string{},
		HeaderEnv:            map[string]string{},
		ResourceAttributes:   map[string]string{},
		ServiceName:          "unknown_service",
		InstrumentationScope: "opentelemetry",
		Timeout:              3 * time.Second,
		ExportInterval:       60 * time.Second,
		Temporality:          OpenTelemetryMetricTemporalityCumulative,
		MaxInstruments:       256,
		CardinalityLimit:     2000,
	}
}

// OpenTelemetryMetricSubscriber records Relay metric marks through the OTLP metrics SDK.
type OpenTelemetryMetricSubscriber struct {
	ptr unsafe.Pointer
}

type openTelemetrySignalCStrings struct {
	transport            *C.char
	endpoint             *C.char
	headers              *C.char
	headerEnv            *C.char
	resourceAttributes   *C.char
	serviceName          *C.char
	serviceNamespace     *C.char
	serviceVersion       *C.char
	instrumentationScope *C.char
}

type openTelemetrySignalConfig struct {
	transport            OpenTelemetryTransport
	endpoint             string
	headers              map[string]string
	headerEnv            map[string]string
	resourceAttributes   map[string]string
	serviceName          string
	serviceNamespace     string
	serviceVersion       string
	instrumentationScope string
}

func newOpenTelemetrySignalCStrings(config openTelemetrySignalConfig) (openTelemetrySignalCStrings, error) {
	encodedHeaders, err := jsonMarshal(config.headers)
	if err != nil {
		return openTelemetrySignalCStrings{}, err
	}
	encodedHeaderEnv, err := jsonMarshal(config.headerEnv)
	if err != nil {
		return openTelemetrySignalCStrings{}, err
	}
	encodedResources, err := jsonMarshal(config.resourceAttributes)
	if err != nil {
		return openTelemetrySignalCStrings{}, err
	}
	return openTelemetrySignalCStrings{
		transport:            C.CString(string(config.transport)),
		endpoint:             C.CString(config.endpoint),
		headers:              C.CString(string(encodedHeaders)),
		headerEnv:            C.CString(string(encodedHeaderEnv)),
		resourceAttributes:   C.CString(string(encodedResources)),
		serviceName:          C.CString(config.serviceName),
		serviceNamespace:     optionalCString(config.serviceNamespace),
		serviceVersion:       optionalCString(config.serviceVersion),
		instrumentationScope: C.CString(config.instrumentationScope),
	}, nil
}

func (values *openTelemetrySignalCStrings) free() {
	C.free(unsafe.Pointer(values.transport))
	C.free(unsafe.Pointer(values.endpoint))
	C.free(unsafe.Pointer(values.headers))
	C.free(unsafe.Pointer(values.headerEnv))
	C.free(unsafe.Pointer(values.resourceAttributes))
	C.free(unsafe.Pointer(values.serviceName))
	C.free(unsafe.Pointer(values.serviceNamespace))
	C.free(unsafe.Pointer(values.serviceVersion))
	C.free(unsafe.Pointer(values.instrumentationScope))
}

func requireWholeMillisecondDuration(field string, value time.Duration) error {
	if value > 0 && value%time.Millisecond != 0 {
		return fmt.Errorf("%s must be zero or an exact multiple of 1ms", field)
	}
	return nil
}

func normalizeOpenTelemetryLogConfig(config OpenTelemetryLogConfig) (OpenTelemetryLogConfig, error) {
	if config.Transport == "" {
		config.Transport = OpenTelemetryTransportHTTPBinary
	}
	if config.Endpoint == "" {
		return config, fmt.Errorf(openTelemetryEndpointRequiredMessage)
	}
	if config.ServiceName == "" {
		config.ServiceName = "unknown_service"
	}
	if config.InstrumentationScope == "" {
		config.InstrumentationScope = "opentelemetry"
	}
	if config.Timeout == 0 {
		config.Timeout = 3 * time.Second
	}
	if config.Timeout < 0 || config.ScheduledDelay < 0 || config.CompletedSpanContextTTL < 0 {
		return config, fmt.Errorf("durations must not be negative")
	}
	if err := requireWholeMillisecondDuration("timeout", config.Timeout); err != nil {
		return config, err
	}
	if err := requireWholeMillisecondDuration("scheduled delay", config.ScheduledDelay); err != nil {
		return config, err
	}
	if err := requireWholeMillisecondDuration("completed span context TTL", config.CompletedSpanContextTTL); err != nil {
		return config, err
	}
	if config.MinimumSeverity == "" {
		config.MinimumSeverity = LogSeverityInfo
	}
	if config.MaxQueueSize == 0 {
		config.MaxQueueSize = 2048
	}
	if config.MaxExportBatchSize == 0 {
		config.MaxExportBatchSize = 512
	}
	if config.ScheduledDelay == 0 {
		config.ScheduledDelay = time.Second
	}
	if config.CompletedSpanContextTTL == 0 {
		config.CompletedSpanContextTTL = 60 * time.Second
	}
	if config.Headers == nil {
		config.Headers = map[string]string{}
	}
	if config.HeaderEnv == nil {
		config.HeaderEnv = map[string]string{}
	}
	if config.ResourceAttributes == nil {
		config.ResourceAttributes = map[string]string{}
	}
	return config, nil
}

// NewOpenTelemetryLogSubscriber creates an independent OTLP log subscriber.
func NewOpenTelemetryLogSubscriber(config OpenTelemetryLogConfig) (*OpenTelemetryLogSubscriber, error) {
	config, err := normalizeOpenTelemetryLogConfig(config)
	if err != nil {
		return nil, err
	}
	common, err := newOpenTelemetrySignalCStrings(openTelemetrySignalConfig{
		config.Transport, config.Endpoint, config.Headers, config.HeaderEnv, config.ResourceAttributes,
		config.ServiceName, config.ServiceNamespace, config.ServiceVersion, config.InstrumentationScope,
	})
	if err != nil {
		return nil, err
	}
	defer common.free()
	cSeverity := C.CString(string(config.MinimumSeverity))
	defer C.free(unsafe.Pointer(cSeverity))
	var ptr unsafe.Pointer
	status := C.nemo_relay_otel_log_subscriber_create_v2(
		common.transport,
		common.endpoint,
		common.headers,
		common.headerEnv,
		common.resourceAttributes,
		common.serviceName,
		common.serviceNamespace,
		common.serviceVersion,
		common.instrumentationScope,
		C.uint64_t(config.Timeout/time.Millisecond),
		cSeverity,
		C.uint64_t(config.MaxQueueSize),
		C.uint64_t(config.MaxExportBatchSize),
		C.uint64_t(config.ScheduledDelay/time.Millisecond),
		C.uint64_t(config.CompletedSpanContextTTL/time.Millisecond),
		&ptr,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &OpenTelemetryLogSubscriber{ptr: ptr}, nil
}

// Register registers the log subscriber globally with the given name.
func (s *OpenTelemetryLogSubscriber) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_otel_log_subscriber_register(s.ptr, cName))
}

// Deregister removes the log subscriber by name.
func (s *OpenTelemetryLogSubscriber) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_otel_log_subscriber_deregister(cName))
}

// ForceFlush drains Relay delivery and queued log batches. A successful flush updates
// RuntimeDiagnostics with queue drops observed so far.
func (s *OpenTelemetryLogSubscriber) ForceFlush() error {
	return checkStatus(C.nemo_relay_otel_log_subscriber_force_flush(s.ptr))
}

// RuntimeDiagnostics returns a bounded snapshot of exporter and event-processing failures.
func (s *OpenTelemetryLogSubscriber) RuntimeDiagnostics() ([]OpenTelemetryRuntimeDiagnostic, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_otel_log_subscriber_runtime_diagnostics_json(s.ptr, &out)); err != nil {
		return nil, err
	}
	return decodeOpenTelemetryRuntimeDiagnostics(out)
}

// Shutdown drains and shuts down the logger provider.
func (s *OpenTelemetryLogSubscriber) Shutdown() error {
	return checkStatus(C.nemo_relay_otel_log_subscriber_shutdown(s.ptr))
}

// Close frees the log subscriber handle.
func (s *OpenTelemetryLogSubscriber) Close() {
	if s.ptr != nil {
		C.nemo_relay_otel_log_subscriber_free(s.ptr)
		s.ptr = nil
	}
}

func normalizeOpenTelemetryMetricConfig(config OpenTelemetryMetricConfig) (OpenTelemetryMetricConfig, error) {
	if config.Transport == "" {
		config.Transport = OpenTelemetryTransportHTTPBinary
	}
	if config.Endpoint == "" {
		return config, fmt.Errorf(openTelemetryEndpointRequiredMessage)
	}
	if config.ServiceName == "" {
		config.ServiceName = "unknown_service"
	}
	if config.InstrumentationScope == "" {
		config.InstrumentationScope = "opentelemetry"
	}
	if config.Timeout == 0 {
		config.Timeout = 3 * time.Second
	}
	if config.ExportInterval == 0 {
		config.ExportInterval = 60 * time.Second
	}
	if config.Timeout < 0 || config.ExportInterval < 0 {
		return config, fmt.Errorf("durations must not be negative")
	}
	if err := requireWholeMillisecondDuration("timeout", config.Timeout); err != nil {
		return config, err
	}
	if err := requireWholeMillisecondDuration("export interval", config.ExportInterval); err != nil {
		return config, err
	}
	if config.Temporality == "" {
		config.Temporality = OpenTelemetryMetricTemporalityCumulative
	}
	if config.MaxInstruments == 0 {
		config.MaxInstruments = 256
	}
	if config.CardinalityLimit == 0 {
		config.CardinalityLimit = 2000
	}
	if config.Headers == nil {
		config.Headers = map[string]string{}
	}
	if config.HeaderEnv == nil {
		config.HeaderEnv = map[string]string{}
	}
	if config.ResourceAttributes == nil {
		config.ResourceAttributes = map[string]string{}
	}
	return config, nil
}

// NewOpenTelemetryMetricSubscriber creates an independent OTLP metric subscriber.
func NewOpenTelemetryMetricSubscriber(config OpenTelemetryMetricConfig) (*OpenTelemetryMetricSubscriber, error) {
	config, err := normalizeOpenTelemetryMetricConfig(config)
	if err != nil {
		return nil, err
	}
	common, err := newOpenTelemetrySignalCStrings(openTelemetrySignalConfig{
		config.Transport, config.Endpoint, config.Headers, config.HeaderEnv, config.ResourceAttributes,
		config.ServiceName, config.ServiceNamespace, config.ServiceVersion, config.InstrumentationScope,
	})
	if err != nil {
		return nil, err
	}
	defer common.free()
	cTemporality := C.CString(string(config.Temporality))
	defer C.free(unsafe.Pointer(cTemporality))
	var ptr unsafe.Pointer
	status := C.nemo_relay_otel_metric_subscriber_create_v2(
		common.transport,
		common.endpoint,
		common.headers,
		common.headerEnv,
		common.resourceAttributes,
		common.serviceName,
		common.serviceNamespace,
		common.serviceVersion,
		common.instrumentationScope,
		C.uint64_t(config.Timeout/time.Millisecond),
		C.uint64_t(config.ExportInterval/time.Millisecond),
		cTemporality,
		C.uint64_t(config.MaxInstruments),
		C.uint64_t(config.CardinalityLimit),
		&ptr,
	)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &OpenTelemetryMetricSubscriber{ptr: ptr}, nil
}

// Register registers the metric subscriber globally with the given name.
func (s *OpenTelemetryMetricSubscriber) Register(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_otel_metric_subscriber_register(s.ptr, cName))
}

// Deregister removes the metric subscriber by name.
func (s *OpenTelemetryMetricSubscriber) Deregister(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_otel_metric_subscriber_deregister(cName))
}

// ForceFlush drains Relay delivery and collects current metric aggregates.
func (s *OpenTelemetryMetricSubscriber) ForceFlush() error {
	return checkStatus(C.nemo_relay_otel_metric_subscriber_force_flush(s.ptr))
}

// RuntimeDiagnostics returns a bounded snapshot of exporter and event-processing failures.
func (s *OpenTelemetryMetricSubscriber) RuntimeDiagnostics() ([]OpenTelemetryRuntimeDiagnostic, error) {
	var out *C.char
	if err := checkStatus(C.nemo_relay_otel_metric_subscriber_runtime_diagnostics_json(s.ptr, &out)); err != nil {
		return nil, err
	}
	return decodeOpenTelemetryRuntimeDiagnostics(out)
}

// Shutdown performs final collection and shuts down the meter provider.
func (s *OpenTelemetryMetricSubscriber) Shutdown() error {
	return checkStatus(C.nemo_relay_otel_metric_subscriber_shutdown(s.ptr))
}

// Close frees the metric subscriber handle.
func (s *OpenTelemetryMetricSubscriber) Close() {
	if s.ptr != nil {
		C.nemo_relay_otel_metric_subscriber_free(s.ptr)
		s.ptr = nil
	}
}

// ---------------------------------------------------------------------------
// Scope-local guardrail/intercept registration (Tool)
// ---------------------------------------------------------------------------

// ScopeRegisterEventMetadataInjector registers an event metadata injector
// owned by an active scope.
func ScopeRegisterEventMetadataInjector(scopeUUID, name string, priority int32, fn EventMetadataInjectorFunc) error {
	if fn == nil {
		return errEventMetadataInjectorCallbackNil
	}
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_event_metadata_injector(
		cScopeUUID,
		cName,
		C.int32_t(priority),
		C.NemoRelayEventMetadataInjectorFn(C.goEventMetadataInjectorTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterEventMetadataInjector removes an event metadata injector
// owned by an active scope.
func ScopeDeregisterEventMetadataInjector(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_event_metadata_injector(
		cScopeUUID,
		cName,
	))
}

func registerScopeEventSanitizer(scopeUUID, name string, priority int32, fn EventSanitizeFunc, kind int) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	callback := C.NemoRelayEventSanitizeFn(C.goEventSanitizeTrampoline)
	free := C.NemoRelayFreeFn(C.goFreeTrampoline)
	var status C.int32_t
	switch kind {
	case 0:
		status = C.nemo_relay_scope_register_mark_sanitize_guardrail(cScopeUUID, cName, C.int32_t(priority), callback, id, free)
	case 1:
		status = C.nemo_relay_scope_register_scope_sanitize_start_guardrail(cScopeUUID, cName, C.int32_t(priority), callback, id, free)
	default:
		status = C.nemo_relay_scope_register_scope_sanitize_end_guardrail(cScopeUUID, cName, C.int32_t(priority), callback, id, free)
	}
	return checkStatus(status)
}

// ScopeRegisterMarkSanitizeGuardrail registers a scope-local mark sanitizer.
func ScopeRegisterMarkSanitizeGuardrail(scopeUUID, name string, priority int32, fn EventSanitizeFunc) error {
	return registerScopeEventSanitizer(scopeUUID, name, priority, fn, 0)
}

// ScopeDeregisterMarkSanitizeGuardrail removes a scope-local mark sanitizer.
func ScopeDeregisterMarkSanitizeGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_mark_sanitize_guardrail(cScopeUUID, cName))
}

// ScopeRegisterScopeSanitizeStartGuardrail registers a scope-local scope-start sanitizer.
func ScopeRegisterScopeSanitizeStartGuardrail(scopeUUID, name string, priority int32, fn EventSanitizeFunc) error {
	return registerScopeEventSanitizer(scopeUUID, name, priority, fn, 1)
}

// ScopeDeregisterScopeSanitizeStartGuardrail removes a scope-local scope-start sanitizer.
func ScopeDeregisterScopeSanitizeStartGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_scope_sanitize_start_guardrail(cScopeUUID, cName))
}

// ScopeRegisterScopeSanitizeEndGuardrail registers a scope-local scope-end sanitizer.
func ScopeRegisterScopeSanitizeEndGuardrail(scopeUUID, name string, priority int32, fn EventSanitizeFunc) error {
	return registerScopeEventSanitizer(scopeUUID, name, priority, fn, 2)
}

// ScopeDeregisterScopeSanitizeEndGuardrail removes a scope-local scope-end sanitizer.
func ScopeDeregisterScopeSanitizeEndGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_scope_sanitize_end_guardrail(cScopeUUID, cName))
}

// ScopeRegisterToolSanitizeRequestGuardrail registers a scope-local guardrail
// that sanitizes tool request arguments. The guardrail is scoped to the given
// scope UUID and does not affect other scopes.
func ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_tool_sanitize_request_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterToolSanitizeRequestGuardrail removes a scope-local tool
// sanitize-request guardrail by name.
func ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_tool_sanitize_request_guardrail(cScopeUUID, cName))
}

// ScopeRegisterToolSanitizeResponseGuardrail registers a scope-local guardrail
// that sanitizes tool response data.
func ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, name string, priority int32, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_tool_sanitize_response_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterToolSanitizeResponseGuardrail removes a scope-local tool
// sanitize-response guardrail by name.
func ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_tool_sanitize_response_guardrail(cScopeUUID, cName))
}

// ScopeRegisterToolConditionalExecutionGuardrail registers a scope-local
// guardrail that conditionally gates tool execution. Returns nil to allow
// execution, or a non-nil pointer to an error message string to reject.
func ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, name string, priority int32, fn ToolConditionalFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_tool_conditional_execution_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayToolConditionalFn(C.goToolConditionalTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterToolConditionalExecutionGuardrail removes a scope-local tool
// conditional-execution guardrail by name.
func ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_tool_conditional_execution_guardrail(cScopeUUID, cName))
}

// ScopeRegisterToolRequestIntercept registers a scope-local intercept that
// transforms tool request arguments.
func ScopeRegisterToolRequestIntercept(scopeUUID, name string, priority int32, breakChain bool, fn ToolSanitizeFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_tool_request_intercept(
		cScopeUUID, cName, C.int32_t(priority), C._Bool(breakChain),
		C.NemoRelayToolSanitizeFn(C.goToolSanitizeTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterToolRequestIntercept removes a scope-local tool request
// intercept by name.
func ScopeDeregisterToolRequestIntercept(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_tool_request_intercept(cScopeUUID, cName))
}

// ScopeRegisterToolExecutionIntercept registers a scope-local tool execution
// intercept following the middleware chain pattern.
func ScopeRegisterToolExecutionIntercept(scopeUUID, name string, priority int32, execFn ToolExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_tool_execution_intercept(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayToolExecInterceptCb(C.goToolExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterToolExecutionIntercept removes a scope-local tool execution
// intercept by name.
func ScopeDeregisterToolExecutionIntercept(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_tool_execution_intercept(cScopeUUID, cName))
}

// ---------------------------------------------------------------------------
// Scope-local guardrail/intercept registration (LLM)
// ---------------------------------------------------------------------------

// ScopeRegisterLlmSanitizeRequestGuardrail registers a scope-local guardrail
// that sanitizes LLM request data.
func ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, name string, priority int32, fn LLMRequestFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_sanitize_request_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayLlmSanitizeRequestCb(C.goLlmRequestTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmSanitizeRequestGuardrail removes a scope-local LLM
// sanitize-request guardrail by name.
func ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_sanitize_request_guardrail(cScopeUUID, cName))
}

// ScopeRegisterLlmSanitizeResponseGuardrail registers a scope-local guardrail
// that sanitizes LLM response data.
func ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, name string, priority int32, fn LLMResponseFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_sanitize_response_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayLlmSanitizeResponseCb(C.goLlmResponseTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmSanitizeResponseGuardrail removes a scope-local LLM
// sanitize-response guardrail by name.
func ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_sanitize_response_guardrail(cScopeUUID, cName))
}

// ScopeRegisterLlmConditionalExecutionGuardrail registers a scope-local
// guardrail that conditionally gates LLM execution.
func ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, name string, priority int32, fn LLMConditionalFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_conditional_execution_guardrail(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayLlmConditionalCb(C.goLlmConditionalTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmConditionalExecutionGuardrail removes a scope-local LLM
// conditional-execution guardrail by name.
func ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_conditional_execution_guardrail(cScopeUUID, cName))
}

// ScopeRegisterLlmRequestIntercept registers a scope-local intercept that
// transforms the LLM request using the unified annotated-aware signature.
func ScopeRegisterLlmRequestIntercept(scopeUUID, name string, priority int32, breakChain bool, fn LLMRequestInterceptFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_request_intercept(
		cScopeUUID, cName, C.int32_t(priority), C._Bool(breakChain),
		C.NemoRelayLlmRequestInterceptCb(C.goLlmRequestInterceptTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmRequestIntercept removes a scope-local LLM request
// intercept by name.
func ScopeDeregisterLlmRequestIntercept(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_request_intercept(cScopeUUID, cName))
}

// ScopeRegisterLlmExecutionIntercept registers a scope-local LLM execution
// intercept following the middleware chain pattern.
func ScopeRegisterLlmExecutionIntercept(scopeUUID, name string, priority int32, execFn LLMExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_execution_intercept(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmExecutionIntercept removes a scope-local LLM execution
// intercept by name.
func ScopeDeregisterLlmExecutionIntercept(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_execution_intercept(cScopeUUID, cName))
}

// ScopeRegisterLlmStreamExecutionIntercept registers a scope-local streaming
// LLM execution intercept following the middleware chain pattern.
func ScopeRegisterLlmStreamExecutionIntercept(scopeUUID, name string, priority int32, execFn LLMExecutionInterceptFunc) error {
	execID := registerClosure(execFn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_llm_stream_execution_intercept(
		cScopeUUID, cName, C.int32_t(priority),
		C.NemoRelayLlmExecInterceptCb(C.goLlmExecInterceptTrampoline),
		execID,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterLlmStreamExecutionIntercept removes a scope-local LLM stream
// execution intercept by name.
func ScopeDeregisterLlmStreamExecutionIntercept(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_llm_stream_execution_intercept(cScopeUUID, cName))
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registration
// ---------------------------------------------------------------------------

// ScopeRegisterSubscriber registers a scope-local event subscriber. The
// callback receives an owned [Event] snapshot that is safe to retain after the
// callback returns.
func ScopeRegisterSubscriber(scopeUUID, name string, fn EventSubscriberFunc) error {
	id := registerClosure(fn)
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_register_subscriber(
		cScopeUUID, cName,
		C.NemoRelayEventSubscriberFn(C.goEventSubscriberTrampoline),
		id,
		C.NemoRelayFreeFn(C.goFreeTrampoline),
	))
}

// ScopeDeregisterSubscriber removes a scope-local event subscriber by name.
func ScopeDeregisterSubscriber(scopeUUID, name string) error {
	cScopeUUID := C.CString(scopeUUID)
	defer C.free(unsafe.Pointer(cScopeUUID))
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nemo_relay_scope_deregister_subscriber(cScopeUUID, cName))
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
	status := C.nemo_relay_tool_request_intercepts(cName, cArgs, &out)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	defer C.nemo_relay_string_free(out)
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

	status := C.nemo_relay_tool_conditional_execution(cName, cArgs)
	return checkStatus(status)
}

// LlmRequestIntercepts runs the registered LLM request intercept chain on the
// given request (serialized as JSON) and returns the transformed request JSON.
func LlmRequestIntercepts(name string, request json.RawMessage) (LLMRequestInterceptOutcome, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	var out *C.char
	status := C.nemo_relay_llm_request_intercepts(cName, cRequest, &out)
	if err := checkStatus(status); err != nil {
		return LLMRequestInterceptOutcome{}, err
	}
	defer C.nemo_relay_string_free(out)
	var outcome LLMRequestInterceptOutcome
	if err := jsonUnmarshal([]byte(C.GoString(out)), &outcome); err != nil {
		return LLMRequestInterceptOutcome{}, err
	}
	return outcome, nil
}

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all guardrails pass, or an error with the
// rejection reason if blocked. The request should be in LLMRequest JSON format
// ({"headers": {...}, "content": {...}}).
func LlmConditionalExecution(request json.RawMessage) error {
	cRequest := C.CString(string(request))
	defer C.free(unsafe.Pointer(cRequest))

	status := C.nemo_relay_llm_conditional_execution(cRequest)
	return checkStatus(status)
}
