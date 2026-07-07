// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit coverage for the native plugin host ABI adapter.

use super::*;

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay_plugin::{
    NemoRelayNativeLlmNextFn, NemoRelayNativeLlmStreamNextFn, NemoRelayNativeToolNextFn,
};
use serde_json::json;

struct ThreadScopeStackRestore(Option<ThreadScopeStackBinding>);

impl ThreadScopeStackRestore {
    fn capture() -> Self {
        Self(Some(capture_thread_scope_stack()))
    }
}

impl Drop for ThreadScopeStackRestore {
    fn drop(&mut self) {
        if let Some(binding) = self.0.take() {
            restore_thread_scope_stack(binding);
        }
    }
}

fn native_string(value: &str) -> *mut NemoRelayNativeString {
    native_string_from_str(value).expect("native string allocation should succeed")
}

fn assert_last_error_contains(expected: &str) {
    let error = native_last_error_message().expect("native last error should be set");
    assert!(
        error.contains(expected),
        "expected native error '{error}' to contain '{expected}'"
    );
}

#[test]
fn native_string_and_json_helpers_cover_abi_boundaries() {
    clear_native_last_error();
    assert_eq!(
        unsafe { native_string_new(ptr::null(), 0, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_last_error_contains("out string pointer is null");

    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { native_string_new(ptr::null(), 1, &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert!(out.is_null());
    assert_last_error_contains("string data pointer is null");

    let invalid_utf8 = [0xff];
    assert_eq!(
        unsafe { native_string_new(invalid_utf8.as_ptr(), invalid_utf8.len(), &mut out) },
        NemoRelayStatus::InvalidUtf8
    );
    assert!(out.is_null());
    assert_last_error_contains("not valid UTF-8");

    let text = native_string("hello");
    assert_eq!(unsafe { native_string_len(text) }, 5);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(native_string_data(text), 5) },
        b"hello"
    );
    assert!(unsafe { native_string_data(ptr::null()) }.is_null());
    assert_eq!(unsafe { native_string_len(ptr::null()) }, 0);
    assert_eq!(take_native_string(text).unwrap(), "hello");
    unsafe { native_string_free(ptr::null_mut()) };

    let empty = native_string("");
    assert_eq!(read_native_string(empty).unwrap(), "");
    unsafe { native_string_free(empty) };
    assert_eq!(read_native_string(ptr::null()).unwrap(), "");

    let bad = Box::into_raw(Box::new(NativeHostString(vec![0xff]))) as *mut NemoRelayNativeString;
    assert!(read_native_string(bad).is_err());
    assert_eq!(
        optional_json_from_native_string(bad, "bad json"),
        Err(NemoRelayStatus::InvalidUtf8)
    );
    unsafe { native_last_error_set(bad) };
    assert_last_error_contains("not valid UTF-8");
    unsafe { native_string_free(bad) };

    let message = native_string("explicit native error");
    unsafe { native_last_error_set(message) };
    assert_eq!(
        native_last_error_message().as_deref(),
        Some("explicit native error")
    );
    unsafe { native_string_free(message) };
    unsafe { native_last_error_clear() };
    assert!(native_last_error_message().is_none());

    set_native_last_error("specific fallback");
    assert!(
        json_from_native_string(ptr::null_mut(), "generic fallback")
            .unwrap_err()
            .to_string()
            .contains("specific fallback")
    );
    clear_native_last_error();
    assert!(
        json_from_native_string(ptr::null_mut(), "generic fallback")
            .unwrap_err()
            .to_string()
            .contains("generic fallback")
    );

    let invalid_json = native_string("{");
    assert!(
        take_json_from_native_string(invalid_json, "unused")
            .unwrap_err()
            .to_string()
            .contains("invalid JSON")
    );
    assert_eq!(
        optional_json_from_native_string(ptr::null(), "optional"),
        Ok(None)
    );
    let valid_json = native_string(r#"{"value":1}"#);
    assert_eq!(
        optional_json_from_native_string(valid_json, "optional").unwrap(),
        Some(json!({"value": 1}))
    );
    unsafe { native_string_free(valid_json) };
    let invalid_json = native_string("not-json");
    assert_eq!(
        optional_json_from_native_string(invalid_json, "optional"),
        Err(NemoRelayStatus::InvalidJson)
    );
    assert_last_error_contains("optional is not valid JSON");
    unsafe { native_string_free(invalid_json) };

    assert_eq!(
        parse_json_arg(ptr::null(), "null JSON").unwrap_err(),
        NemoRelayStatus::InvalidJson
    );
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(
        parse_llm_request_arg(request_json, "request").unwrap(),
        request
    );
    unsafe { native_string_free(request_json) };
    let wrong_shape = native_string(r#"{"headers":[]}"#);
    assert_eq!(
        parse_llm_request_arg(wrong_shape, "request").unwrap_err(),
        NemoRelayStatus::InvalidJson
    );
    assert_last_error_contains("was not an LLM request");
    unsafe { native_string_free(wrong_shape) };

    assert_eq!(
        write_native_json(&json!({"ok": true}), ptr::null_mut()),
        NemoRelayStatus::NullPointer
    );
    let mut json_out = ptr::null_mut();
    assert_eq!(
        write_native_json(&json!({"ok": true}), &mut json_out),
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(json_out, "unused").unwrap(),
        json!({"ok": true})
    );

    let host_api = unsafe { &*native_host_api() };
    assert_eq!(host_api.abi_version, NEMO_RELAY_NATIVE_ABI_VERSION);
    assert_eq!(
        host_api.struct_size,
        std::mem::size_of::<NemoRelayNativeHostApiV1>()
    );
}

#[test]
fn native_timestamp_scope_type_and_error_mappings_cover_variants() {
    assert_eq!(optional_timestamp_from_native(ptr::null()).unwrap(), None);
    let epoch = 0_i64;
    assert_eq!(
        optional_timestamp_from_native(&epoch).unwrap(),
        DateTime::<Utc>::from_timestamp_micros(0)
    );
    let invalid_timestamp = i64::MAX;
    assert_eq!(
        optional_timestamp_from_native(&invalid_timestamp),
        Err(NemoRelayStatus::InvalidArg)
    );

    for (native, core) in [
        (NemoRelayNativeScopeType::Agent, ScopeType::Agent),
        (NemoRelayNativeScopeType::Function, ScopeType::Function),
        (NemoRelayNativeScopeType::Tool, ScopeType::Tool),
        (NemoRelayNativeScopeType::Llm, ScopeType::Llm),
        (NemoRelayNativeScopeType::Retriever, ScopeType::Retriever),
        (NemoRelayNativeScopeType::Embedder, ScopeType::Embedder),
        (NemoRelayNativeScopeType::Reranker, ScopeType::Reranker),
        (NemoRelayNativeScopeType::Guardrail, ScopeType::Guardrail),
        (NemoRelayNativeScopeType::Evaluator, ScopeType::Evaluator),
        (NemoRelayNativeScopeType::Custom, ScopeType::Custom),
        (NemoRelayNativeScopeType::Unknown, ScopeType::Unknown),
    ] {
        assert_eq!(native_scope_type_to_core(native), core);
    }
    assert!(native_scope_ref(ptr::null()).is_none());

    let status_cases = [
        (NemoRelayStatus::AlreadyExists, "already exists"),
        (NemoRelayStatus::NotFound, "not found"),
        (NemoRelayStatus::ScopeStackEmpty, "scope stack empty"),
        (NemoRelayStatus::GuardrailRejected, "guardrail rejected"),
        (NemoRelayStatus::InvalidArg, "invalid argument"),
        (NemoRelayStatus::Internal, "internal error"),
    ];
    for (status, expected) in status_cases {
        clear_native_last_error();
        assert!(
            flow_error_from_status(status, "fallback")
                .to_string()
                .contains(expected)
        );
    }

    assert_eq!(
        status_from_plugin_error(PluginError::NotFound("missing".into())),
        NemoRelayStatus::NotFound
    );
    assert_eq!(
        status_from_plugin_error(PluginError::Conflict("duplicate".into())),
        NemoRelayStatus::AlreadyExists
    );
    assert_eq!(
        status_from_plugin_error(PluginError::InvalidConfig("invalid".into())),
        NemoRelayStatus::InvalidArg
    );
    let serialization = serde_json::from_str::<Json>("{").unwrap_err();
    assert_eq!(
        status_from_plugin_error(PluginError::Serialization(serialization)),
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        status_from_plugin_error(PluginError::Internal("internal".into())),
        NemoRelayStatus::Internal
    );
    assert_eq!(
        status_from_plugin_error(PluginError::RegistrationFailed("registration".into())),
        NemoRelayStatus::Internal
    );

    for (error, status) in [
        (
            FlowError::AlreadyExists("duplicate".into()),
            NemoRelayStatus::AlreadyExists,
        ),
        (
            FlowError::NotFound("missing".into()),
            NemoRelayStatus::NotFound,
        ),
        (
            FlowError::InvalidArgument("invalid".into()),
            NemoRelayStatus::InvalidArg,
        ),
        (FlowError::ScopeStackEmpty, NemoRelayStatus::ScopeStackEmpty),
        (
            FlowError::GuardrailRejected("blocked".into()),
            NemoRelayStatus::GuardrailRejected,
        ),
        (
            FlowError::Internal("internal".into()),
            NemoRelayStatus::Internal,
        ),
    ] {
        assert_eq!(status_from_flow_error(error), status);
    }
}

unsafe extern "C" fn count_scope_callback(user_data: *mut c_void) -> NemoRelayStatus {
    let calls = unsafe { &*(user_data as *const AtomicUsize) };
    calls.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

#[test]
fn native_scope_stack_abi_covers_lifecycle_and_validation() {
    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *crate::api::runtime::global_context().write().unwrap() =
        crate::api::runtime::NemoRelayContextState::new();
    let _restore = ThreadScopeStackRestore::capture();
    assert_eq!(
        unsafe { native_scope_stack_create(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_set_thread(ptr::null()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_restore_thread(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_scope_stack_with_current(ptr::null(), count_scope_callback, ptr::null_mut())
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_get_current(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );

    let mut stack = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_create(&mut stack) },
        NemoRelayStatus::Ok
    );
    assert!(!stack.is_null());
    assert_eq!(
        unsafe { native_scope_stack_set_thread(stack) },
        NemoRelayStatus::Ok
    );
    assert!(unsafe { native_scope_stack_active() });

    let calls = AtomicUsize::new(0);
    assert_eq!(
        unsafe {
            native_scope_stack_with_current(
                stack,
                count_scope_callback,
                (&calls as *const AtomicUsize).cast_mut().cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let name = native_string("native-scope");
    let data = native_string(r#"{"source":"native"}"#);
    let metadata = native_string(r#"{"test":true}"#);
    let input = native_string(r#"{"input":1}"#);
    let timestamp = 0_i64;
    let mut scope = ptr::null_mut();
    assert_eq!(
        unsafe {
            native_scope_push(
                name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                data,
                metadata,
                input,
                &timestamp,
                &mut scope,
            )
        },
        NemoRelayStatus::Ok
    );
    assert!(!scope.is_null());
    assert_eq!(native_scope_ref(scope).unwrap().name, "native-scope");

    let mut current = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_get_current(&mut current) },
        NemoRelayStatus::Ok
    );
    assert_eq!(native_scope_ref(current).unwrap().name, "native-scope");
    unsafe { native_scope_handle_free(current) };

    let mark_name = native_string("native-mark");
    assert_eq!(
        unsafe { native_emit_mark(mark_name, scope, data, metadata, &timestamp) },
        NemoRelayStatus::Ok
    );
    let output = native_string(r#"{"output":1}"#);
    assert_eq!(
        unsafe { native_scope_pop(scope, output, metadata, &timestamp) },
        NemoRelayStatus::Ok
    );

    let invalid = native_string("not-json");
    let mut invalid_scope = ptr::null_mut();
    assert_eq!(
        unsafe {
            native_scope_push(
                name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                invalid,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut invalid_scope,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(invalid_scope.is_null());
    assert_eq!(
        unsafe { native_scope_pop(ptr::null(), ptr::null(), ptr::null(), ptr::null()) },
        NemoRelayStatus::NullPointer
    );

    let mut binding = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(&mut binding) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { native_scope_stack_restore_thread(binding) },
        NemoRelayStatus::Ok
    );
    let mut disposable_binding = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(&mut disposable_binding) },
        NemoRelayStatus::Ok
    );
    unsafe { native_scope_stack_binding_free(disposable_binding) };

    for value in [name, data, metadata, input, mark_name, output, invalid] {
        unsafe { native_string_free(value) };
    }
    unsafe {
        native_scope_handle_free(scope);
        native_scope_handle_free(ptr::null_mut());
        native_scope_stack_free(stack);
        native_scope_stack_free(ptr::null_mut());
        native_scope_stack_binding_free(ptr::null_mut());
    }
}

unsafe extern "C" fn noop_subscriber(
    _user_data: *mut c_void,
    _event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_json(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _payload_json: *const NemoRelayNativeString,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_conditional(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeToolNextFn,
    _next_ctx: *mut c_void,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_request(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_json(
    _user_data: *mut c_void,
    _payload_json: *const NemoRelayNativeString,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_conditional(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_request_intercept(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _annotated_json: *const NemoRelayNativeString,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmNextFn,
    _next_ctx: *mut c_void,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_stream_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmStreamNextFn,
    _next_ctx: *mut c_void,
    _out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

#[test]
fn native_registration_entrypoints_reject_null_contexts() {
    unsafe {
        assert_eq!(
            native_plugin_context_register_subscriber(
                ptr::null_mut(),
                ptr::null(),
                noop_subscriber,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_request_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_response_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_conditional_execution_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_request_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_request_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_request,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_response_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_conditional_execution_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_request_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                noop_llm_request_intercept,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_stream_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_stream_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
    }
    assert_last_error_contains("plugin context is null");
}

unsafe extern "C" fn tool_json_echo(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let payload = read_native_string(payload_json).unwrap();
    unsafe { *out_json = native_string(&payload) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn tool_json_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    set_native_last_error("tool callback rejected input");
    unsafe { *out_json = native_string(r#"{"discarded":true}"#) };
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn llm_request_echo(
    _user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let request = read_native_string(request_json).unwrap();
    unsafe { *out_request_json = native_string(&request) };
    NemoRelayStatus::Ok
}

#[test]
fn native_callback_helpers_cover_success_error_and_invalid_output() {
    assert_eq!(
        call_tool_json_callback(tool_json_echo, ptr::null_mut(), "tool", &json!({"a": 1})).unwrap(),
        json!({"a": 1})
    );
    assert!(
        call_tool_json_callback(tool_json_error, ptr::null_mut(), "tool", &Json::Null)
            .unwrap_err()
            .to_string()
            .contains("tool callback rejected input")
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    assert_eq!(
        call_llm_request_callback(llm_request_echo, ptr::null_mut(), &request).unwrap(),
        request
    );
}

fn tool_next(output: FlowResult<Json>) -> ToolExecutionNextFn {
    let output = Arc::new(Mutex::new(Some(output)));
    Arc::new(move |_args| {
        let output = output.lock().unwrap().take().unwrap();
        Box::pin(async move { output })
    })
}

fn llm_next(output: FlowResult<Json>) -> LlmExecutionNextFn {
    let output = Arc::new(Mutex::new(Some(output)));
    Arc::new(move |_request| {
        let output = output.lock().unwrap().take().unwrap();
        Box::pin(async move { output })
    })
}

#[test]
fn native_non_streaming_continuations_cover_success_and_error_paths() {
    let args = native_string(r#"{"value":1}"#);
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { native_tool_next(args, ptr::null_mut(), &mut out) },
        NemoRelayStatus::NullPointer
    );
    let next = Box::into_raw(Box::new(tool_next(Ok(json!({"result": 2}))))) as *mut c_void;
    assert_eq!(
        unsafe { native_tool_next(args, next, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"result": 2})
    );
    unsafe { drop(Box::from_raw(next as *mut ToolExecutionNextFn)) };

    let next = Box::into_raw(Box::new(tool_next(Err(FlowError::NotFound(
        "missing".into(),
    ))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_tool_next(args, next, &mut out) },
        NemoRelayStatus::NotFound
    );
    unsafe { drop(Box::from_raw(next as *mut ToolExecutionNextFn)) };
    unsafe { native_string_free(args) };

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();
    let next = Box::into_raw(Box::new(llm_next(Ok(json!({"answer": 42}))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_llm_next(request_json, next, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"answer": 42})
    );
    unsafe { drop(Box::from_raw(next as *mut LlmExecutionNextFn)) };

    let next = Box::into_raw(Box::new(llm_next(Err(FlowError::GuardrailRejected(
        "blocked".into(),
    ))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_llm_next(request_json, next, &mut out) },
        NemoRelayStatus::GuardrailRejected
    );
    unsafe {
        drop(Box::from_raw(next as *mut LlmExecutionNextFn));
        native_string_free(request_json);
    }
}

#[derive(Debug)]
enum NativeStreamItem {
    Json(Json),
    InvalidJson,
    Null,
    Error(NemoRelayStatus),
    End,
}

struct TestNativeStream {
    items: VecDeque<NativeStreamItem>,
    cancel_count: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
}

unsafe extern "C" fn test_native_stream_poll(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &mut *(user_data as *mut TestNativeStream) };
    unsafe { *out_json = ptr::null_mut() };
    match state.items.pop_front().unwrap_or(NativeStreamItem::End) {
        NativeStreamItem::Json(value) => write_native_json(&value, out_json),
        NativeStreamItem::InvalidJson => {
            unsafe { *out_json = native_string("not-json") };
            NemoRelayStatus::Ok
        }
        NativeStreamItem::Null => NemoRelayStatus::Ok,
        NativeStreamItem::Error(status) => status,
        NativeStreamItem::End => NemoRelayStatus::StreamEnd,
    }
}

unsafe extern "C" fn test_native_stream_cancel(user_data: *mut c_void) -> NemoRelayStatus {
    let state = unsafe { &*(user_data as *const TestNativeStream) };
    state.cancel_count.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn test_native_stream_drop(user_data: *mut c_void) {
    let state = unsafe { Box::from_raw(user_data as *mut TestNativeStream) };
    state.drop_count.fetch_add(1, Ordering::SeqCst);
}

fn test_native_stream(
    items: impl IntoIterator<Item = NativeStreamItem>,
) -> (
    NemoRelayNativeLlmStreamV1,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let cancel_count = Arc::new(AtomicUsize::new(0));
    let drop_count = Arc::new(AtomicUsize::new(0));
    let state = Box::new(TestNativeStream {
        items: items.into_iter().collect(),
        cancel_count: cancel_count.clone(),
        drop_count: drop_count.clone(),
    });
    (
        NemoRelayNativeLlmStreamV1 {
            struct_size: std::mem::size_of::<NemoRelayNativeLlmStreamV1>(),
            user_data: Box::into_raw(state).cast(),
            next: Some(test_native_stream_poll),
            cancel: Some(test_native_stream_cancel),
            drop: Some(test_native_stream_drop),
        },
        cancel_count,
        drop_count,
    )
}

#[tokio::test]
async fn native_stream_adapter_covers_chunks_end_errors_and_cancellation() {
    let (raw, cancel_count, drop_count) = test_native_stream([
        NativeStreamItem::Json(json!({"chunk": 1})),
        NativeStreamItem::End,
    ]);
    let mut stream = native_stream_to_relay_stream(raw, None, None).unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap(), json!({"chunk": 1}));
    assert!(stream.next().await.is_none());
    assert_eq!(cancel_count.load(Ordering::SeqCst), 0);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let (raw, cancel_count, drop_count) = test_native_stream([NativeStreamItem::Json(json!({
        "chunk": 2
    }))]);
    let stream = native_stream_to_relay_stream(raw, None, None).unwrap();
    drop(stream);
    assert_eq!(cancel_count.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    for item in [
        NativeStreamItem::InvalidJson,
        NativeStreamItem::Null,
        NativeStreamItem::Error(NemoRelayStatus::InvalidArg),
    ] {
        let (raw, _, drop_count) = test_native_stream([item]);
        let mut stream = native_stream_to_relay_stream(raw, None, None).unwrap();
        assert!(stream.next().await.unwrap().is_err());
        assert!(stream.next().await.is_none());
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    let (mut raw, _, drop_count) = test_native_stream([]);
    raw.struct_size = 0;
    assert!(NativeRelayLlmStream::from_raw(raw, None, None).is_err());
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let (mut raw, _, drop_count) = test_native_stream([]);
    raw.next = None;
    assert!(NativeRelayLlmStream::from_raw(raw, None, None).is_err());
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn relay_stream_adapter_covers_poll_end_error_and_cancel() {
    let stream: LlmJsonStream = Box::pin(tokio_stream::iter(vec![
        Ok(json!({"chunk": 1})),
        Err(FlowError::Internal("stream failed".into())),
    ]));
    let raw = relay_stream_to_native_stream(stream);
    let poll = raw.next.unwrap();
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"chunk": 1})
    );
    out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::Internal
    );
    drop_native_stream(raw);

    let stream: LlmJsonStream = Box::pin(tokio_stream::empty());
    let raw = relay_stream_to_native_stream(stream);
    let poll = raw.next.unwrap();
    out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::StreamEnd
    );
    assert_eq!(
        unsafe { cancel_relay_llm_stream(raw.user_data) },
        NemoRelayStatus::Ok
    );
    drop_native_stream(raw);

    assert_eq!(
        unsafe { poll_relay_llm_stream(ptr::null_mut(), ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { cancel_relay_llm_stream(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    unsafe { drop_relay_llm_stream(ptr::null_mut()) };

    let stream: LlmJsonStream = Box::pin(tokio_stream::empty());
    let raw = relay_stream_to_native_stream(stream);
    let state = unsafe { &*(raw.user_data as *const NativeHostLlmStream) };
    let mutex = state.stream.clone();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _guard = mutex.lock().unwrap();
        panic!("poison native stream lock");
    }));
    assert_eq!(
        unsafe { cancel_relay_llm_stream(raw.user_data) },
        NemoRelayStatus::Internal
    );
    drop_native_stream(raw);
}

#[test]
fn native_stream_continuation_covers_success_and_error() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();

    let next: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async {
            Ok(Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))])) as LlmJsonStream)
        })
    });
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    let mut raw = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe { native_llm_stream_next(request_json, next_ctx, &mut raw) },
        NemoRelayStatus::Ok
    );
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { raw.next.unwrap()(raw.user_data, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"chunk": true})
    );
    drop_native_stream(raw);
    unsafe { drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn)) };

    let next: LlmStreamExecutionNextFn =
        Arc::new(|_request| Box::pin(async { Err(FlowError::NotFound("stream missing".into())) }));
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    raw = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe { native_llm_stream_next(request_json, next_ctx, &mut raw) },
        NemoRelayStatus::NotFound
    );
    unsafe {
        drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn));
        native_string_free(request_json);
    }
    assert_eq!(
        unsafe { native_llm_stream_next(ptr::null(), ptr::null_mut(), ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
}
