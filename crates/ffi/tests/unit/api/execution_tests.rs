// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for execution in the NeMo Relay FFI crate.

use super::*;

/// Observe the middleware-only envelope and forward changed business arguments.
unsafe extern "C" fn context_capture_cb(
    user_data: *mut libc::c_void,
    context_json: *const c_char,
    next_fn: NemoRelayToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    let context: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(context_json) }.to_str().unwrap()).unwrap();
    unsafe { &*user_data.cast::<Mutex<Vec<Json>>>() }
        .lock()
        .unwrap()
        .push(context.clone());
    let mut args = context["args"].clone();
    args["changed"] = json!(true);
    let encoded = cstring(&args.to_string());
    unsafe { next_fn(encoded.as_ptr(), next_ctx) }
}

#[test]
fn execution_context_reaches_c_middleware_but_not_arguments_or_events() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let observed = Arc::new(Mutex::new(Vec::<Json>::new()));
    unsafe {
        let stack = fresh_scope_stack();
        let middleware_name = cstring(&unique_name("ffi_context_middleware"));
        let subscriber_name = cstring(&unique_name("ffi_context_events"));
        assert_status!(
            nemo_relay_register_tool_execution_intercept(
                middleware_name.as_ptr(),
                1,
                context_capture_cb,
                Arc::as_ptr(&observed).cast_mut().cast(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_register_subscriber(
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let name = cstring("ffi_context_tool");
        let args = cstring(r#"{"value":2}"#);
        let call_id = cstring("provider-context-call");
        for supplied in [
            Some(r#"{"data":{"review":{"reference":"context-only-marker"}}}"#),
            Some("{}"),
            None,
        ] {
            let mut out = ptr::null_mut();
            let status = if let Some(value) = supplied {
                let context = cstring(value);
                nemo_relay_tool_call_execute(
                    name.as_ptr(),
                    args.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    call_id.as_ptr(),
                    0,
                    context.as_ptr(),
                    ptr::null(),
                    &mut out,
                )
            } else {
                // Omitted application context leaves the callback envelope unchanged.
                nemo_relay_tool_call_execute(
                    name.as_ptr(),
                    args.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    call_id.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut out,
                )
            };
            assert_status!(status, NemoRelayStatus::Ok);
            let result = returned_json(out);
            assert_eq!(
                result["result"],
                json!({"value":2,"changed":true,"executed":true})
            );
            assert_eq!(result["annotation"], json!({"source":"ffi"}));
        }
        let captures = observed.lock().unwrap();
        assert_eq!(captures.len(), 3);
        assert_eq!(
            captures[0]["execution_context"],
            json!({"data":{"review":{"reference":"context-only-marker"}}})
        );
        assert_eq!(captures[1]["execution_context"], json!({}));
        assert!(captures[2].get("execution_context").is_none());
        for captured in captures.iter() {
            assert_eq!(captured["tool_call_id"], json!("provider-context-call"));
            assert_eq!(captured["args"], json!({"value":2}));
        }
        assert_status!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log());
        assert!(!events.is_empty());
        let serialized = serde_json::to_string(&*events).unwrap();
        assert!(!serialized.contains("context-only-marker"));
        assert!(!serialized.contains("execution_context"));
        drop(events);
        assert_status!(
            nemo_relay_deregister_subscriber(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(middleware_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn execution_context_validation_releases_c_callback_without_running_middleware_or_effect() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counts {
        effects: AtomicUsize,
        middleware: AtomicUsize,
        freed: AtomicUsize,
    }
    unsafe extern "C" fn effect(data: *mut libc::c_void, _args: *const c_char) -> *mut c_char {
        unsafe { &*data.cast::<Counts>() }
            .effects
            .fetch_add(1, Ordering::SeqCst);
        ptr::null_mut()
    }
    unsafe extern "C" fn middleware(
        data: *mut libc::c_void,
        _context: *const c_char,
        _next: NemoRelayToolExecNextFn,
        _next_data: *mut libc::c_void,
    ) -> *mut c_char {
        unsafe { &*data.cast::<Counts>() }
            .middleware
            .fetch_add(1, Ordering::SeqCst);
        ptr::null_mut()
    }
    unsafe extern "C" fn release(data: *mut libc::c_void) {
        unsafe { &*data.cast::<Counts>() }
            .freed
            .fetch_add(1, Ordering::SeqCst);
    }
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let counts = Counts {
        effects: AtomicUsize::new(0),
        middleware: AtomicUsize::new(0),
        freed: AtomicUsize::new(0),
    };
    let data = (&counts as *const Counts).cast_mut().cast();
    unsafe {
        let stack = fresh_scope_stack();
        let middleware_name = cstring(&unique_name("ffi_invalid_context"));
        assert_status!(
            nemo_relay_register_tool_execution_intercept(
                middleware_name.as_ptr(),
                1,
                middleware,
                data,
                None,
            ),
            NemoRelayStatus::Ok
        );
        let mut deep = json!(true);
        for _ in 0..65 {
            deep = json!({"nested":deep});
        }
        let name = cstring("ffi_invalid_context_tool");
        let args = cstring("{}");
        let cases = [
            // Resource refusal must precede even the JSON parser's syntax error.
            (
                format!("{{\"data\":{{\"value\":\"{}", "x".repeat(1_048_576)),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({"data":{"value":"x".repeat(1_048_576)}}).to_string(),
                NemoRelayStatus::InvalidArg,
            ),
            ("{".to_string(), NemoRelayStatus::InvalidJson),
            ("null".to_string(), NemoRelayStatus::InvalidArg),
            ("[]".to_string(), NemoRelayStatus::InvalidArg),
            ("true".to_string(), NemoRelayStatus::InvalidArg),
            (
                r#"{"host_approval":null}"#.to_string(),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({"value":"x".repeat(16_384)}).to_string(),
                NemoRelayStatus::InvalidArg,
            ),
            (deep.to_string(), NemoRelayStatus::InvalidArg),
        ];
        for (index, (input, expected)) in cases.into_iter().enumerate() {
            let input = cstring(&input);
            let mut out = ptr::null_mut();
            assert_status!(
                nemo_relay_tool_call_execute(
                    name.as_ptr(),
                    args.as_ptr(),
                    effect,
                    data,
                    Some(release),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    0,
                    input.as_ptr(),
                    ptr::null(),
                    &mut out,
                ),
                expected
            );
            assert!(out.is_null());
            assert_eq!(counts.effects.load(Ordering::SeqCst), 0);
            assert_eq!(counts.middleware.load(Ordering::SeqCst), 0);
            assert_eq!(counts.freed.load(Ordering::SeqCst), index + 1);
        }
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(middleware_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_execute_parent_data_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_tool_execute_parent");
        let args = cstring(r#"{"value":2}"#);
        let data = cstring(r#"{"source":"tool-execute"}"#);
        let metadata = cstring(r#"{"trace":"tool"}"#);
        let invalid_json = cstring("{");
        let mut out_json = ptr::null_mut();

        assert_status!(
            nemo_relay_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let executed = returned_json(out_json);
        assert_eq!(executed["result"]["executed"], json!(true));
        assert_eq!(executed["annotation"]["source"], json!("ffi"));

        assert_status!(
            nemo_relay_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_status!(
            nemo_relay_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_fail_cb,
                ptr::null_mut(),
                None,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("tool execution callback failed")
        );

        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_execute_v2_propagates_tool_call_id_and_legacy_omits_it() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = cstring(&unique_name("ffi_tool_execute_id_subscriber"));
        assert_status!(
            nemo_relay_register_subscriber(
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let args = cstring(r#"{"value":2}"#);
        let managed_name = cstring("ffi_tool_execute_v2_id");
        let legacy_name = cstring("ffi_tool_execute_legacy_id");
        let tool_call_id = cstring("call_ffi_managed_123");
        let mut out_json = ptr::null_mut();

        assert_status!(
            nemo_relay_tool_call_execute(
                managed_name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                tool_call_id.as_ptr(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let _ = returned_json(out_json);

        assert_status!(
            nemo_relay_tool_call_execute(
                legacy_name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let _ = returned_json(out_json);

        let invalid_utf8 = [0xff_u8, 0];
        assert_status!(
            nemo_relay_tool_call_execute(
                managed_name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                invalid_utf8.as_ptr().cast(),
                0,
                ptr::null(),
                std::ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log()).clone();
        for scope_category in ["start", "end"] {
            let managed_event = events
                .iter()
                .find(|event| {
                    event["json"]["kind"] == json!("scope")
                        && event["json"]["name"] == json!("ffi_tool_execute_v2_id")
                        && event["json"]["scope_category"] == json!(scope_category)
                })
                .unwrap_or_else(|| panic!("missing managed {scope_category} event"));
            assert_eq!(managed_event["tool_call_id"], json!("call_ffi_managed_123"));

            let legacy_event = events
                .iter()
                .find(|event| {
                    event["json"]["kind"] == json!("scope")
                        && event["json"]["name"] == json!("ffi_tool_execute_legacy_id")
                        && event["json"]["scope_category"] == json!(scope_category)
                })
                .unwrap_or_else(|| panic!("missing legacy {scope_category} event"));
            assert!(legacy_event["tool_call_id"].is_null());
        }

        assert_status!(
            nemo_relay_deregister_subscriber(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_execute_codec_parent_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_execute_codec");
        let request = cstring(
            r#"{"headers":{"x-trace":"1"},"content":{"model":"codec-model","prompt":"hello codec"}}"#,
        );
        let data = cstring(r#"{"source":"llm-execute"}"#);
        let metadata = cstring(r#"{"trace":"llm"}"#);
        let model_name = cstring("override-model");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let mut out_json = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                model_name.as_ptr(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                ptr::null_mut(),
                None,
                std::ptr::null(),
                ptr::null(),
                0,
                std::ptr::null(),
                &mut out_json
            ),
            NemoRelayStatus::Ok
        );
        let executed = returned_json(out_json);
        assert_eq!(executed["model_seen"], json!("codec-model"));
        assert_eq!(executed["content"], json!("hello from ffi"));

        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                invalid_json.as_ptr(),
                model_name.as_ptr(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                ptr::null_mut(),
                None,
                std::ptr::null(),
                ptr::null(),
                0,
                std::ptr::null(),
                &mut out_json
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                invalid_utf8.as_ptr() as *const c_char,
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                ptr::null_mut(),
                None,
                std::ptr::null(),
                ptr::null(),
                0,
                std::ptr::null(),
                &mut out_json
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_fail_cb,
                ptr::null_mut(),
                None,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                ptr::null_mut(),
                None,
                std::ptr::null(),
                ptr::null(),
                0,
                std::ptr::null(),
                &mut out_json
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("llm execution callback failed")
        );

        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_stream_execute_response_codec_defaults_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_stream_defaults");
        let request = cstring(
            r#"{"headers":{},"content":{"model":"gpt-ffi","messages":[{"role":"user","content":"hi"}]}}"#,
        );
        let data = cstring(r#"{"stream":true}"#);
        let metadata = cstring(r#"{"trace":"stream"}"#);
        let model_name = cstring("stream-model");
        let invalid_json = cstring("{");
        let response_codec = api::nemo_relay_codec_new(0, std::ptr::null());
        let mut stream = ptr::null_mut();
        let mut chunk = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                Some(llm_exec_openai_chat_cb),
                None,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                std::ptr::null(),
                response_codec,
                0,
                std::ptr::null(),
                &mut stream
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 1);
        let stream_chunk = returned_json(chunk);
        assert_eq!(stream_chunk["id"], json!("chatcmpl-ffi"));
        assert_status!(nemo_relay_stream_close(stream), NemoRelayStatus::Ok);
        assert_status!(nemo_relay_stream_close(stream), NemoRelayStatus::Ok);
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 0);
        assert!(lock_unpoisoned(collected_chunks()).is_empty());
        assert_eq!(*lock_unpoisoned(finalizer_calls()), 0);
        nemo_relay_stream_free(stream);

        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                Some(llm_exec_openai_chat_cb),
                None,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null_mut(),
                None,
                parent,
                1,
                invalid_json.as_ptr(),
                metadata.as_ptr(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                std::ptr::null(),
                response_codec,
                0,
                std::ptr::null(),
                &mut stream
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                Some(llm_exec_fail_cb),
                None,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null_mut(),
                None,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                std::ptr::null(),
                response_codec,
                0,
                std::ptr::null(),
                &mut stream
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("llm execution callback failed")
        );

        types::nemo_relay_codec_free(response_codec);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_registration_and_exporter_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        assert_status!(
            nemo_relay_scope_stack_create(ptr::null(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_set_thread(ptr::null()),
            NemoRelayStatus::NullPointer
        );

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_scope_local");
        let mut scope = ptr::null_mut();
        assert_status!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        let scope_uuid = cstring(&take_string(nemo_relay_scope_handle_uuid(scope)).unwrap());
        let invalid_uuid = cstring("not-a-uuid");

        let global_tool_san_req = cstring(&unique_name("ffi_tool_san_req"));
        assert_status!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::AlreadyExists
        );
        assert_status!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_tool_san_resp = cstring(&unique_name("ffi_tool_san_resp"));
        assert_status!(
            nemo_relay_register_tool_sanitize_response_guardrail(
                global_tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(global_tool_san_resp.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_tool_exec = cstring(&unique_name("ffi_tool_exec"));
        assert_status!(
            nemo_relay_register_tool_execution_intercept(
                global_tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(global_tool_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_san_req = cstring(&unique_name("ffi_llm_san_req"));
        assert_status!(
            nemo_relay_register_llm_sanitize_request_guardrail(
                global_llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(global_llm_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_exec = cstring(&unique_name("ffi_llm_exec"));
        assert_status!(
            nemo_relay_register_llm_execution_intercept(
                global_llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_llm_execution_intercept(global_llm_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_stream_exec = cstring(&unique_name("ffi_llm_stream_exec"));
        assert_status!(
            nemo_relay_register_llm_stream_execution_intercept(
                global_llm_stream_exec.as_ptr(),
                1,
                Some(llm_exec_intercept_cb),
                None,
                ptr::null_mut(),
                None
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_llm_stream_execution_intercept(global_llm_stream_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let scope_tool_san_req = cstring(&unique_name("scope_tool_san_req"));
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                invalid_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_san_resp = cstring(&unique_name("scope_tool_san_resp"));
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_cond = cstring(&unique_name("scope_tool_cond"));
        assert_status!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_req = cstring(&unique_name("scope_tool_req"));
        assert_status!(
            nemo_relay_scope_register_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_exec = cstring(&unique_name("scope_tool_exec"));
        assert_status!(
            nemo_relay_scope_register_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_tool_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_san_req = cstring(&unique_name("scope_llm_san_req"));
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_san_resp = cstring(&unique_name("scope_llm_san_resp"));
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_cond = cstring(&unique_name("scope_llm_cond"));
        assert_status!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_req = cstring(&unique_name("scope_llm_req"));
        assert_status!(
            nemo_relay_scope_register_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_exec = cstring(&unique_name("scope_llm_exec"));
        assert_status!(
            nemo_relay_scope_register_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_stream_exec = cstring(&unique_name("scope_llm_stream_exec"));
        assert_status!(
            nemo_relay_scope_register_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
                1,
                Some(llm_exec_intercept_cb),
                None,
                ptr::null_mut(),
                None
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_subscriber = cstring(&unique_name("scope_subscriber"));
        assert_status!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                scope_subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoRelayStatus::Ok
        );

        let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
        let session = cstring("ffi-session");
        let agent = cstring("ffi-agent");
        let version = cstring("1.0.0");
        assert_status!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                &mut exporter
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                ptr::null_mut()
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atif_exporter_register(ptr::null(), scope_subscriber.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        let mut null_export = ptr::null_mut();
        assert_status!(
            nemo_relay_atif_exporter_export(ptr::null(), &mut null_export),
            NemoRelayStatus::NullPointer
        );
        let exporter_name = cstring(&unique_name("ffi_exporter_sub"));
        assert_status!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::AlreadyExists
        );
        assert_status!(
            nemo_relay_atif_exporter_export(exporter, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atif_exporter_clear(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        let missing_exporter = cstring("missing_exporter");
        assert_status!(
            nemo_relay_atif_exporter_deregister(missing_exporter.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);

        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_relay_stream_next(ptr::null_mut(), &mut chunk), -1);
        assert_eq!(nemo_relay_stream_next(ptr::null_mut(), ptr::null_mut()), -1);

        assert_status!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

/// Check tool skip flags and release the callback when validation fails.
#[test]
fn execution_options_validate_bits_and_release_callback_on_failure() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("ffi-options");
        let args = cstring(r#"{"value":2}"#);
        let mut out = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                NEMO_RELAY_SKIP_CONDITIONAL_EXECUTION
                    | NEMO_RELAY_SKIP_REQUEST_INTERCEPTS
                    | NEMO_RELAY_SKIP_EXECUTION_INTERCEPTS,
                ptr::null(),
                std::ptr::null(),
                &mut out,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out)["annotation"]["source"], json!("ffi"));
        let frees_before = *lock_unpoisoned(plugin_frees());
        assert_status!(
            nemo_relay_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                Some(plugin_free),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                8,
                ptr::null(),
                std::ptr::null(),
                &mut out,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(*lock_unpoisoned(plugin_frees()), frees_before + 1);
        assert!(
            CStr::from_ptr(nemo_relay_last_error())
                .to_str()
                .unwrap()
                .contains("skip-stage")
        );
        nemo_relay_scope_stack_free(stack);
    }
}

/// Count each resource's releases without deallocating it, so duplicate releases are safe to detect.
unsafe extern "C" fn execution_options_count_free(user_data: *mut libc::c_void) {
    let count = unsafe { &*user_data.cast::<std::sync::atomic::AtomicUsize>() };
    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Check that invalid LLM options and mismatched codecs release each callback resource once.
#[test]
fn llm_execution_options_release_resources_on_validation_failure() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("ffi-llm-options");
        let request = cstring(r#"{"headers":{},"content":{"model":"test"}}"#);
        for streaming in [false, true] {
            for (decode, encode, skip_stages, message) in [
                (true, true, 8, "skip-stage"),
                (true, false, 0, "both be provided"),
                (false, true, 0, "both be provided"),
            ] {
                let func_frees = AtomicUsize::new(0);
                let codec_frees = AtomicUsize::new(0);
                let stream_frees = AtomicUsize::new(0);
                let func_data = (&func_frees as *const AtomicUsize).cast_mut().cast();
                let codec_data = (&codec_frees as *const AtomicUsize).cast_mut().cast();
                let stream_data = (&stream_frees as *const AtomicUsize).cast_mut().cast();
                let decode: callable::NemoRelayCodecDecodeFn =
                    if decode { Some(codec_decode_cb) } else { None };
                let encode: callable::NemoRelayCodecEncodeFn =
                    if encode { Some(codec_encode_cb) } else { None };
                let status = if streaming {
                    let mut out = ptr::null_mut();
                    let status = nemo_relay_llm_stream_call_execute(
                        name.as_ptr(),
                        request.as_ptr(),
                        Some(llm_exec_cb),
                        None,
                        func_data,
                        Some(execution_options_count_free),
                        Some(collector_cb),
                        Some(finalizer_cb),
                        stream_data,
                        Some(execution_options_count_free),
                        ptr::null(),
                        0,
                        ptr::null(),
                        ptr::null(),
                        ptr::null(),
                        decode,
                        encode,
                        codec_data,
                        Some(execution_options_count_free),
                        std::ptr::null(),
                        ptr::null(),
                        skip_stages,
                        std::ptr::null(),
                        &mut out,
                    );
                    assert!(out.is_null());
                    status
                } else {
                    let mut out = ptr::null_mut();
                    let status = nemo_relay_llm_call_execute(
                        name.as_ptr(),
                        request.as_ptr(),
                        llm_exec_cb,
                        func_data,
                        Some(execution_options_count_free),
                        ptr::null(),
                        0,
                        ptr::null(),
                        ptr::null(),
                        ptr::null(),
                        decode,
                        encode,
                        codec_data,
                        Some(execution_options_count_free),
                        std::ptr::null(),
                        ptr::null(),
                        skip_stages,
                        std::ptr::null(),
                        &mut out,
                    );
                    assert!(out.is_null());
                    status
                };
                assert_status!(status, NemoRelayStatus::InvalidArg);
                assert!(
                    CStr::from_ptr(nemo_relay_last_error())
                        .to_str()
                        .unwrap()
                        .contains(message)
                );
                assert_eq!(
                    func_frees.load(Ordering::SeqCst),
                    1,
                    "execution callback, streaming={streaming}"
                );
                assert_eq!(
                    codec_frees.load(Ordering::SeqCst),
                    1,
                    "codec callback, streaming={streaming}"
                );
                assert_eq!(
                    stream_frees.load(Ordering::SeqCst),
                    usize::from(streaming),
                    "stream callbacks"
                );
            }
        }
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn execution_context_accepts_c_boundary_whitespace_and_escaped_values() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();
    let overhead = r#"{"data":{"value":""}}"#.len();
    let inputs = [
        json!({"data":{"value":"x".repeat(16_384-overhead)}}).to_string(),
        format!(r#"{{"data":{{"value":"{}"}}}}"#, r"\u0061".repeat(3_000)),
        format!("{}{{}}{}", " ".repeat(20_000), " ".repeat(20_000)),
    ];
    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("c-context-admission");
        let arguments = cstring("{}");
        for input in inputs {
            let context = cstring(&input);
            let mut output = ptr::null_mut();
            assert_status!(
                nemo_relay_tool_call_execute(
                    name.as_ptr(),
                    arguments.as_ptr(),
                    tool_exec_cb,
                    ptr::null_mut(),
                    None,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    0,
                    context.as_ptr(),
                    ptr::null(),
                    &mut output,
                ),
                NemoRelayStatus::Ok
            );
            assert!(!output.is_null());
            nemo_relay_string_free(output);
        }
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn application_context_normalization_validates_borrowed_buffers_and_budgets() {
    use std::ffi::CStr;
    use std::ptr;
    unsafe {
        let mut output = ptr::null_mut();
        for input in [
            b"{}\0{}".as_slice(),
            b"\xff",
            b"{} {}",
            b"NaN",
            b"{\"data\":null}",
        ] {
            let status = nemo_relay_application_execution_context_normalize_json(
                input.as_ptr(),
                input.len(),
                &mut output,
            );
            assert_ne!(status, NemoRelayStatus::Ok);
            assert!(output.is_null());
        }
        let padded = format!("{}{{}}", " ".repeat(1_048_576));
        assert_eq!(
            nemo_relay_application_execution_context_normalize_json(
                padded.as_ptr(),
                padded.len(),
                &mut output
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(CStr::from_ptr(output).to_bytes(), b"{}");
        nemo_relay_string_free(output);
        output = ptr::null_mut();
        assert_eq!(
            nemo_relay_application_execution_context_normalize_json(ptr::null(), 0, &mut output),
            NemoRelayStatus::InvalidArg
        );
        assert!(output.is_null());
        assert_eq!(
            nemo_relay_application_execution_context_normalize_json(
                b"{}".as_ptr(),
                usize::MAX,
                &mut output
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_application_execution_context_normalize_json(
                b"{}".as_ptr(),
                2,
                ptr::null_mut()
            ),
            NemoRelayStatus::NullPointer
        );
    }
}

#[test]
fn application_context_normalization_validates_value_budgets_and_depth() {
    use std::ffi::CStr;
    use std::ptr;
    unsafe {
        let mut output = ptr::null_mut();
        let value = br#""\u0000""#;
        assert_eq!(
            nemo_relay_application_execution_context_value_normalize_json(
                value.as_ptr(),
                value.len(),
                8,
                &mut output
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(CStr::from_ptr(output).to_bytes(), value);
        nemo_relay_string_free(output);
        for budget in [0, 7, 16_385] {
            output = ptr::null_mut();
            assert_eq!(
                nemo_relay_application_execution_context_value_normalize_json(
                    value.as_ptr(),
                    value.len(),
                    budget,
                    &mut output
                ),
                NemoRelayStatus::InvalidArg
            );
            assert!(output.is_null());
        }
        let value = format!("1.{}", "0".repeat(1_048_576));
        assert_eq!(
            nemo_relay_application_execution_context_value_normalize_json(
                value.as_ptr(),
                value.len(),
                3,
                &mut output
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(CStr::from_ptr(output).to_bytes(), b"1.0");
        nemo_relay_string_free(output);
        let deep = format!("{}null{}", "[".repeat(63), "]".repeat(63));
        output = ptr::null_mut();
        assert_eq!(
            nemo_relay_application_execution_context_value_normalize_json(
                deep.as_ptr(),
                deep.len(),
                16_384,
                &mut output
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(output.is_null());
    }
}
