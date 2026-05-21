// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for coverage sweeps in the NeMo Relay FFI crate.

use super::*;
use std::ptr;

#[test]
fn test_ffi_scope_and_event_remaining_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let scope_name = cstring("ffi_child_scope_with_parent");
        let data = cstring(r#"{"scope":"child"}"#);
        let metadata = cstring(r#"{"meta":"scope"}"#);
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut child = ptr::null_mut();

        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                parent,
                3,
                data.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
                &mut child,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_scope_handle_parent_uuid(child)).is_some());
        assert_eq!(
            nemo_relay_push_scope(
                invalid,
                NemoRelayScopeType::Function,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut child,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                parent,
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut child,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                parent,
                0,
                ptr::null(),
                invalid_json.as_ptr(),
                ptr::null(),
                &mut child,
            ),
            NemoRelayStatus::InvalidJson
        );

        let event_name = cstring("ffi_event_with_parent");
        assert_eq!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                data.as_ptr(),
                metadata.as_ptr()
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_event(invalid, parent, ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                invalid_json.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                ptr::null(),
                invalid_json.as_ptr()
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
            nemo_relay_pop_scope(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_pop_scope(child, ptr::null()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(child, ptr::null()),
            NemoRelayStatus::NotFound
        );

        nemo_relay_scope_handle_free(child);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_and_llm_parent_utf8_and_shape_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let tool_name = cstring("ffi_tool_call_utf8");
        let tool_args = cstring(r#"{"value":1}"#);
        let tool_result = cstring(r#"{"done":true}"#);
        let tool_data = cstring(r#"{"source":"tool-call"}"#);
        let tool_metadata = cstring(r#"{"trace":"tool-call"}"#);
        let tool_call_id = cstring("tool-call-id");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut tool_handle = ptr::null_mut();

        assert_eq!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                parent,
                1,
                tool_data.as_ptr(),
                tool_metadata.as_ptr(),
                tool_call_id.as_ptr(),
                &mut tool_handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_tool_handle_parent_uuid(tool_handle)).is_some());
        assert_eq!(
            nemo_relay_tool_call(
                invalid,
                tool_args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut tool_handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                invalid,
                &mut tool_handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_tool_call_end(
                tool_handle,
                tool_result.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_tool_call_end(
                tool_handle,
                tool_result.as_ptr(),
                tool_data.as_ptr(),
                tool_metadata.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_name = cstring("ffi_llm_call_utf8");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let invalid_shape = cstring(r#"{"content":{"model":"ffi-model"}}"#);
        let response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);
        let data = cstring(r#"{"source":"llm-call"}"#);
        let metadata = cstring(r#"{"trace":"llm-call"}"#);
        let model_name = cstring("ffi-model-override");
        let mut llm_handle = ptr::null_mut();

        assert_eq!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                request.as_ptr(),
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                model_name.as_ptr(),
                &mut llm_handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_llm_handle_parent_uuid(llm_handle)).is_some());
        assert_eq!(
            nemo_relay_llm_call(
                invalid,
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                invalid,
                &mut llm_handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_llm_call_end(
                llm_handle,
                response.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_llm_call_end(
                llm_handle,
                response.as_ptr(),
                data.as_ptr(),
                metadata.as_ptr()
            ),
            NemoRelayStatus::Ok
        );

        let mut out = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_call_execute(
                llm_name.as_ptr(),
                invalid_shape.as_ptr(),
                llm_exec_cb,
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
                ptr::null(),
                &mut out,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        let mut stream = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_stream_call_execute(
                llm_name.as_ptr(),
                invalid_shape.as_ptr(),
                llm_exec_cb,
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
                ptr::null(),
                &mut stream,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        nemo_relay_tool_handle_free(tool_handle);
        nemo_relay_llm_handle_free(llm_handle);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_global_registry_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid = invalid_utf8.as_ptr() as *const c_char;

    unsafe {
        assert_eq!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                invalid,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_tool_sanitize_response_guardrail(
                invalid,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_tool_conditional_execution_guardrail(
                invalid,
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_tool_request_intercept(
                invalid,
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_tool_request_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_tool_execution_intercept(
                invalid,
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_tool_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_register_llm_sanitize_request_guardrail(
                invalid,
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_llm_sanitize_response_guardrail(
                invalid,
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_llm_conditional_execution_guardrail(
                invalid,
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_llm_request_intercept(
                invalid,
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_request_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_llm_execution_intercept(
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_llm_stream_execution_intercept(
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_llm_stream_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_register_subscriber(invalid, subscriber_cb, ptr::null_mut(), None),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_deregister_subscriber(invalid),
            NemoRelayStatus::InvalidUtf8
        );
    }
}

#[test]
fn test_ffi_scope_registry_invalid_utf8_scope_and_name_sweeps() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid_scope = invalid_utf8.as_ptr() as *const c_char;

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("scope-registry-invalid");
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
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let valid_name = cstring("scope-registry-valid-name");

        assert_eq!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name
            ),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_scope_register_tool_execution_intercept(
                invalid_scope,
                valid_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                invalid_scope,
                valid_name.as_ptr()
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_tool_execution_intercept(
                scope_uuid.as_ptr(),
                invalid_name,
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name,
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name
            ),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_scope_register_llm_execution_intercept(
                invalid_scope,
                valid_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(invalid_scope, valid_name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_llm_execution_intercept(
                scope_uuid.as_ptr(),
                invalid_name,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_scope_register_subscriber(
                invalid_scope,
                valid_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(invalid_scope, valid_name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                invalid_name,
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}
