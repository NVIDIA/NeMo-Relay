// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for execution in the NeMo Relay FFI crate.

use super::*;

#[test]
fn test_ffi_tool_execute_parent_data_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_tool_execute_parent");
        let args = cstring(r#"{"value":2}"#);
        let data = cstring(r#"{"source":"tool-execute"}"#);
        let metadata = cstring(r#"{"trace":"tool"}"#);
        let invalid_json = cstring("{");
        let mut out_json = ptr::null_mut();

        assert_eq!(
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
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let executed = returned_json(out_json);
        assert_eq!(executed["executed"], json!(true));

        assert_eq!(
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
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
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
fn test_ffi_llm_execute_codec_parent_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

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

        assert_eq!(
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
                ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let executed = returned_json(out_json);
        assert_eq!(executed["model_seen"], json!("codec-model"));
        assert_eq!(executed["content"], json!("hello from ffi"));

        assert_eq!(
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
                ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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
                ptr::null(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
                ptr::null(),
                &mut out_json,
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_stream_defaults");
        let request = cstring(
            r#"{"headers":{},"content":{"model":"gpt-ffi","messages":[{"role":"user","content":"hi"}]}}"#,
        );
        let data = cstring(r#"{"stream":true}"#);
        let metadata = cstring(r#"{"trace":"stream"}"#);
        let model_name = cstring("stream-model");
        let invalid_json = cstring("{");
        let response_codec = api::nemo_relay_openai_chat_codec_new();
        let mut stream = ptr::null_mut();
        let mut chunk = ptr::null_mut();

        assert_eq!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_openai_chat_cb,
                ptr::null_mut(),
                None,
                None,
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
                response_codec,
                &mut stream,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 1);
        let stream_chunk = returned_json(chunk);
        assert_eq!(stream_chunk["id"], json!("chatcmpl-ffi"));
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 0);
        assert!(lock_unpoisoned(collected_chunks()).is_empty());
        assert_eq!(*lock_unpoisoned(finalizer_calls()), 0);
        nemo_relay_stream_free(stream);

        assert_eq!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_openai_chat_cb,
                ptr::null_mut(),
                None,
                None,
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
                response_codec,
                &mut stream,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_fail_cb,
                ptr::null_mut(),
                None,
                None,
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
                response_codec,
                &mut stream,
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
        assert_eq!(
            nemo_relay_scope_stack_create(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_stack_set_thread(ptr::null()),
            NemoRelayStatus::NullPointer
        );

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_scope_local");
        let mut scope = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::AlreadyExists
        );
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_tool_san_resp = cstring(&unique_name("ffi_tool_san_resp"));
        assert_eq!(
            nemo_relay_register_tool_sanitize_response_guardrail(
                global_tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(global_tool_san_resp.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_tool_exec = cstring(&unique_name("ffi_tool_exec"));
        assert_eq!(
            nemo_relay_register_tool_execution_intercept(
                global_tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_tool_execution_intercept(global_tool_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_san_req = cstring(&unique_name("ffi_llm_san_req"));
        assert_eq!(
            nemo_relay_register_llm_sanitize_request_guardrail(
                global_llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(global_llm_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_exec = cstring(&unique_name("ffi_llm_exec"));
        assert_eq!(
            nemo_relay_register_llm_execution_intercept(
                global_llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_llm_execution_intercept(global_llm_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let global_llm_stream_exec = cstring(&unique_name("ffi_llm_stream_exec"));
        assert_eq!(
            nemo_relay_register_llm_stream_execution_intercept(
                global_llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_llm_stream_execution_intercept(global_llm_stream_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let scope_tool_san_req = cstring(&unique_name("scope_tool_san_req"));
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_san_resp = cstring(&unique_name("scope_tool_san_resp"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_cond = cstring(&unique_name("scope_tool_cond"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_req = cstring(&unique_name("scope_tool_req"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_exec = cstring(&unique_name("scope_tool_exec"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_tool_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_san_req = cstring(&unique_name("scope_llm_san_req"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_san_resp = cstring(&unique_name("scope_llm_san_resp"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_cond = cstring(&unique_name("scope_llm_cond"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_req = cstring(&unique_name("scope_llm_req"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_exec = cstring(&unique_name("scope_llm_exec"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_stream_exec = cstring(&unique_name("scope_llm_stream_exec"));
        assert_eq!(
            nemo_relay_scope_register_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_subscriber = cstring(&unique_name("scope_subscriber"));
        assert_eq!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                scope_subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoRelayStatus::Ok
        );

        let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
        let session = cstring("ffi-session");
        let agent = cstring("ffi-agent");
        let version = cstring("1.0.0");
        assert_eq!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atif_exporter_register(ptr::null(), scope_subscriber.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        let mut null_export = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_export(ptr::null(), &mut null_export),
            NemoRelayStatus::NullPointer
        );
        let exporter_name = cstring(&unique_name("ffi_exporter_sub"));
        assert_eq!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::AlreadyExists
        );
        assert_eq!(
            nemo_relay_atif_exporter_export(exporter, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atif_exporter_clear(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        let missing_exporter = cstring("missing_exporter");
        assert_eq!(
            nemo_relay_atif_exporter_deregister(missing_exporter.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);

        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_relay_stream_next(ptr::null_mut(), &mut chunk), -1);
        assert_eq!(nemo_relay_stream_next(ptr::null_mut(), ptr::null_mut()), -1);

        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}
