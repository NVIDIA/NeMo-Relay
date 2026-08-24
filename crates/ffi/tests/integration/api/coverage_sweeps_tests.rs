// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for coverage sweeps in the NeMo Relay FFI crate.

use super::ffi_coverage_support::{ValidationFixtures, assert_otel_signal_lifecycle};
use super::*;
use std::ptr;

#[test]
fn test_ffi_scope_and_event_remaining_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let scope_name = cstring("ffi_child_scope_with_parent");
        let data = cstring(r#"{"scope":"child"}"#);
        let metadata = cstring(r#"{"meta":"scope"}"#);
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut child = ptr::null_mut();

        assert_status!(
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
        assert_status!(
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
        assert_status!(
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
        assert_status!(
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
        assert_status!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                data.as_ptr(),
                metadata.as_ptr()
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_event(invalid, parent, ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                invalid_json.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_event(
                event_name.as_ptr(),
                parent,
                ptr::null(),
                invalid_json.as_ptr()
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_status!(
            nemo_relay_pop_scope(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_pop_scope(child, ptr::null()),
            NemoRelayStatus::Ok
        );
        assert_status!(
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
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let tool_name = cstring("ffi_tool_call_utf8");
        let tool_args = cstring(r#"{"value":1}"#);
        let tool_result = cstring(r#"{"result":{"done":true}}"#);
        let tool_data = cstring(r#"{"source":"tool-call"}"#);
        let tool_metadata = cstring(r#"{"trace":"tool-call"}"#);
        let tool_call_id = cstring("tool-call-id");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut tool_handle = ptr::null_mut();

        assert_status!(
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
        assert_status!(
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
        assert_status!(
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
        assert_status!(
            nemo_relay_tool_call_end(
                tool_handle,
                tool_result.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
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

        assert_status!(
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
        assert_status!(
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
        assert_status!(
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
        assert_status!(
            nemo_relay_llm_call_end(
                llm_handle,
                response.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_call_end(
                llm_handle,
                response.as_ptr(),
                data.as_ptr(),
                metadata.as_ptr()
            ),
            NemoRelayStatus::Ok
        );

        let mut out = ptr::null_mut();
        assert_status!(
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
        assert_status!(
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
        assert_status!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                invalid,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_tool_sanitize_response_guardrail(
                invalid,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_tool_conditional_execution_guardrail(
                invalid,
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_deregister_tool_request_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_tool_execution_intercept(
                invalid,
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
            nemo_relay_register_llm_sanitize_request_guardrail(
                invalid,
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_llm_sanitize_response_guardrail(
                invalid,
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_llm_conditional_execution_guardrail(
                invalid,
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_deregister_llm_request_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_llm_execution_intercept(
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_llm_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_llm_stream_execution_intercept(
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_llm_stream_execution_intercept(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_register_subscriber(invalid, subscriber_cb, ptr::null_mut(), None),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let valid_name = cstring("scope-registry-valid-name");

        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name
            ),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                invalid_scope,
                valid_name.as_ptr()
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                invalid_scope,
                valid_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid_name
            ),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(invalid_scope, valid_name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
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
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
            nemo_relay_scope_register_subscriber(
                invalid_scope,
                valid_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(invalid_scope, valid_name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                invalid_name,
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), invalid_name),
            NemoRelayStatus::InvalidUtf8
        );

        assert_status!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_adaptive_and_observability_entry_points_from_integration_binary() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let config = cstring(
            &json!({
                "version": 1,
                "agent_id": "ffi-adaptive-integration",
                "state": {
                    "backend": {
                        "kind": "in_memory",
                        "config": {}
                    }
                },
                "acg": {
                    "provider": "openai"
                }
            })
            .to_string(),
        );
        let invalid_json = cstring("{");
        let mut out_json = ptr::null_mut();

        assert_status!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));
        assert_status!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_validate_config(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );

        let mut runtime = ptr::null_mut();
        assert_status!(
            nemo_relay_adaptive_runtime_create(config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_create(invalid_json.as_ptr(), &mut runtime),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_adaptive_runtime_create(config.as_ptr(), &mut runtime),
            NemoRelayStatus::Ok
        );
        assert!(!runtime.is_null());
        assert_status!(
            nemo_relay_adaptive_runtime_register(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_register(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_wait_for_idle(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_wait_for_idle(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_report_json(runtime, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_report_json(runtime, &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_adaptive_integration_scope");
        let mut scope = ptr::null_mut();
        assert_status!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_bind_scope(runtime, scope),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_bind_scope(runtime, ptr::null()),
            NemoRelayStatus::NullPointer
        );

        let cache_options = cstring(
            &json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000701",
                "annotated_request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Cache this"
                        }
                    ],
                    "model": "gpt-4.1-mini"
                },
                "agent_id": "ffi-adaptive-integration"
            })
            .to_string(),
        );
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                cache_options.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let facts = returned_json(out_json);
        assert_eq!(facts["provider"], json!("openai"));
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                cache_options.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        for options in [
            json!({
                "provider": "openai",
                "request_id": "not-a-uuid",
                "annotated_request": {},
                "agent_id": "ffi-adaptive-integration"
            }),
            json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000703",
                "annotated_request": {},
                "agent_id": "ffi-adaptive-integration",
                "timestamp": "not-a-timestamp"
            }),
            json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000704",
                "annotated_request": "bad",
                "agent_id": "ffi-adaptive-integration"
            }),
            json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000705"
            }),
        ] {
            let options = cstring(&options.to_string());
            assert!(
                matches!(
                    nemo_relay_adaptive_runtime_build_cache_request_facts(
                        runtime,
                        options.as_ptr(),
                        &mut out_json,
                    ),
                    NemoRelayStatus::InvalidArg | NemoRelayStatus::InvalidJson
                ),
                "expected invalid cache facts options to fail"
            );
        }

        let telemetry_options = cstring(
            &json!({
                "provider": "anthropic",
                "request_id": "00000000-0000-0000-0000-000000000702",
                "usage": {
                    "prompt_tokens": 50,
                    "cache_read_tokens": 10
                },
                "agent_id": "ffi-adaptive-integration",
                "template_version": "v1",
                "toolset_hash": "tools",
                "model_family": "claude",
                "tenant_scope": "tenant"
            })
            .to_string(),
        );
        assert_status!(
            nemo_relay_adaptive_build_cache_telemetry_event(
                telemetry_options.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let event = returned_json(out_json);
        assert_eq!(event["cache_read_tokens"], json!(10));
        assert_eq!(event["hit_rate"], json!(10.0 / 60.0));
        assert_status!(
            nemo_relay_adaptive_build_cache_telemetry_event(
                telemetry_options.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        let no_usage_options = cstring(
            &json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000706",
                "agent_id": "ffi-adaptive-integration",
                "template_version": "v1",
                "toolset_hash": "tools",
                "model_family": "gpt",
                "tenant_scope": "tenant"
            })
            .to_string(),
        );
        assert_status!(
            nemo_relay_adaptive_build_cache_telemetry_event(
                no_usage_options.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json), Json::Null);
        for (options, expected) in [
            (
                json!({
                    "provider": "unsupported",
                    "request_id": "00000000-0000-0000-0000-000000000707",
                    "usage": {},
                    "agent_id": "ffi-adaptive-integration",
                    "template_version": "v1",
                    "toolset_hash": "tools",
                    "model_family": "gpt",
                    "tenant_scope": "tenant"
                }),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "not-a-uuid",
                    "usage": {},
                    "agent_id": "ffi-adaptive-integration",
                    "template_version": "v1",
                    "toolset_hash": "tools",
                    "model_family": "gpt",
                    "tenant_scope": "tenant"
                }),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "00000000-0000-0000-0000-000000000708",
                    "usage": {},
                    "agent_id": "ffi-adaptive-integration",
                    "template_version": "v1",
                    "toolset_hash": "tools",
                    "model_family": "gpt",
                    "tenant_scope": "tenant",
                    "timestamp": "not-a-timestamp"
                }),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "00000000-0000-0000-0000-000000000709",
                    "usage": "bad",
                    "agent_id": "ffi-adaptive-integration",
                    "template_version": "v1",
                    "toolset_hash": "tools",
                    "model_family": "gpt",
                    "tenant_scope": "tenant"
                }),
                NemoRelayStatus::InvalidJson,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "00000000-0000-0000-0000-000000000710",
                    "usage": {},
                    "request_facts": "bad",
                    "agent_id": "ffi-adaptive-integration",
                    "template_version": "v1",
                    "toolset_hash": "tools",
                    "model_family": "gpt",
                    "tenant_scope": "tenant"
                }),
                NemoRelayStatus::InvalidJson,
            ),
        ] {
            let options = cstring(&options.to_string());
            assert_eq!(
                nemo_relay_adaptive_build_cache_telemetry_event(options.as_ptr(), &mut out_json),
                expected
            );
        }
        assert_status!(
            nemo_relay_adaptive_build_cache_telemetry_event(invalid_json.as_ptr(), &mut out_json,),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                ptr::null_mut(),
                cache_options.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_set_latency_sensitivity(0),
            NemoRelayStatus::InvalidArg
        );

        assert_status!(
            nemo_relay_adaptive_runtime_deregister(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_deregister(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_shutdown(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_shutdown(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_report_json(runtime, &mut out_json),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
        types::nemo_relay_adaptive_runtime_free(runtime);

        assert_status!(
            nemo_relay_observability_default_config_json(&mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["version"], json!(4));
        assert_status!(
            nemo_relay_observability_default_config_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["kind"], json!("observability"));
        assert_status!(
            nemo_relay_observability_component_spec_json(
                invalid_json.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );

        let append = cstring("append");
        let bad_mode = cstring("bad-mode");
        let filename = cstring("events.jsonl");
        let happy_dir = temp_dir("ffi-atof-happy");
        let happy_dir_text = happy_dir.to_string_lossy().into_owned();
        let happy_dir = cstring(&happy_dir_text);
        let happy_filename = cstring("happy-events.jsonl");
        let happy_name = cstring(&unique_name("ffi_atof_happy"));
        let mut happy_atof = ptr::null_mut();
        assert_status!(
            nemo_relay_atof_exporter_create(
                happy_dir.as_ptr(),
                append.as_ptr(),
                happy_filename.as_ptr(),
                &mut happy_atof,
            ),
            NemoRelayStatus::Ok
        );
        assert!(!happy_atof.is_null());
        let mut path_ptr = ptr::null_mut();
        assert_status!(
            nemo_relay_atof_exporter_path(happy_atof, &mut path_ptr),
            NemoRelayStatus::Ok
        );
        let path = take_string(path_ptr).expect("expected ATOF exporter path");
        assert!(
            path.ends_with("happy-events.jsonl"),
            "unexpected ATOF exporter path: {path}"
        );
        assert_status!(
            nemo_relay_atof_exporter_register(happy_atof, happy_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_force_flush(happy_atof),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_shutdown(happy_atof),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_deregister(happy_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atof_exporter_free(happy_atof);

        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut atof = ptr::null_mut();
        assert_status!(
            nemo_relay_atof_exporter_create(
                ptr::null(),
                append.as_ptr(),
                filename.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_create(
                ptr::null(),
                bad_mode.as_ptr(),
                filename.as_ptr(),
                &mut atof,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_atof_exporter_create(invalid, append.as_ptr(), filename.as_ptr(), &mut atof),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_create(ptr::null(), invalid, filename.as_ptr(), &mut atof),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_create(ptr::null(), append.as_ptr(), invalid, &mut atof),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_register(ptr::null(), filename.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_force_flush(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_shutdown(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_path(ptr::null(), &mut out_json),
            NemoRelayStatus::NullPointer
        );
    }
}

#[test]
fn test_ffi_lifecycle_validation_paths_from_integration_binary() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let fixtures = ValidationFixtures::new("integration");
        let invalid = fixtures.invalid_utf8();
        let malformed_json = &fixtures.malformed_json;
        let invalid_timestamp = fixtures.invalid_timestamp;
        let scope_name = &fixtures.scope_name;

        assert_status!(
            api::nemo_relay_get_handle(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            api::nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );

        let mut scope = ptr::null_mut();
        for (data, metadata, input) in fixtures.push_scope_json_cases() {
            assert_status!(
                api::nemo_relay_push_scope(
                    scope_name.as_ptr(),
                    NemoRelayScopeType::Custom,
                    ptr::null(),
                    0,
                    data,
                    metadata,
                    input,
                    ptr::null(),
                    &mut scope,
                ),
                NemoRelayStatus::InvalidJson
            );
        }
        assert_status!(
            api::nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::from_ref(&invalid_timestamp),
                &mut scope,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            api::nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        for (output, metadata, timestamp, expected) in fixtures.pop_scope_cases() {
            assert_status!(
                api::nemo_relay_pop_scope(scope, output, metadata, timestamp),
                expected
            );
        }

        let event_name = &fixtures.event_name;
        assert_status!(
            api::nemo_relay_event_v2(
                invalid,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        for (data, schema, metadata) in fixtures.event_json_cases() {
            assert_status!(
                api::nemo_relay_event_v2(
                    event_name.as_ptr(),
                    scope,
                    data,
                    schema,
                    metadata,
                    ptr::null(),
                    ptr::null(),
                ),
                NemoRelayStatus::InvalidJson
            );
        }
        assert_status!(
            api::nemo_relay_event_v2(
                event_name.as_ptr(),
                scope,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::from_ref(&invalid_timestamp),
            ),
            NemoRelayStatus::InvalidArg
        );

        let metric_name = &fixtures.metric_name;
        for (name, values, metadata, timestamp, expected) in fixtures.metric_json_cases() {
            assert_status!(
                api::nemo_relay_metric_json(name, scope, values, metadata, timestamp),
                expected
            );
        }

        let base_measurement = fixtures.base_measurement();
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                ptr::null(),
                ptr::null(),
                1,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::NullPointer
        );
        for measurement in [
            types::NemoRelayMetricMeasurement {
                name: invalid,
                ..base_measurement
            },
            types::NemoRelayMetricMeasurement {
                unit: invalid,
                ..base_measurement
            },
            types::NemoRelayMetricMeasurement {
                description: invalid,
                ..base_measurement
            },
        ] {
            assert_status!(
                api::nemo_relay_metric(
                    metric_name.as_ptr(),
                    scope,
                    ptr::from_ref(&measurement),
                    1,
                    ptr::null(),
                    ptr::null(),
                ),
                NemoRelayStatus::InvalidUtf8
            );
        }
        let histogram = types::NemoRelayMetricMeasurement {
            kind: types::NEMO_RELAY_METRIC_KIND_HISTOGRAM,
            value_type: types::NEMO_RELAY_METRIC_VALUE_TYPE_F64,
            f64_value: 2.5,
            boundaries: fixtures.boundaries.as_ptr(),
            boundaries_len: fixtures.boundaries.len(),
            ..base_measurement
        };
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                scope,
                ptr::from_ref(&histogram),
                1,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        for (metadata, timestamp, expected) in fixtures.native_metric_status_cases() {
            assert_status!(
                api::nemo_relay_metric(
                    metric_name.as_ptr(),
                    scope,
                    ptr::from_ref(&base_measurement),
                    1,
                    metadata,
                    timestamp,
                ),
                expected
            );
        }

        let invalid_attributes = types::NemoRelayMetricMeasurement {
            attributes_json: malformed_json.as_ptr(),
            ..base_measurement
        };
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                scope,
                ptr::from_ref(&invalid_attributes),
                1,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                scope,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            api::nemo_relay_metric_json(
                metric_name.as_ptr(),
                scope,
                fixtures.measurements.as_ptr(),
                fixtures.metadata.as_ptr(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );

        let tool_name = cstring("ffi_integration_validation_tool");
        let tool_args = cstring(r#"{"value":1}"#);
        let tool_result = cstring(r#"{"result":{"ok":true}}"#);
        let mut tool = ptr::null_mut();
        assert_status!(
            api::nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                scope,
                0,
                ptr::null(),
                malformed_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut tool,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            api::nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                scope,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut tool,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            api::nemo_relay_tool_call_end(
                tool,
                tool_result.as_ptr(),
                malformed_json.as_ptr(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            api::nemo_relay_tool_call_end(
                tool,
                tool_result.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            api::nemo_relay_tool_call_end(
                tool,
                tool_result.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(tool);

        let llm_name = cstring("ffi_integration_validation_llm");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);
        let invalid_llm_shape = cstring(r#"{"content":{"model":"ffi-model"}}"#);
        let mut llm = ptr::null_mut();
        for (data, metadata) in [
            (malformed_json.as_ptr(), ptr::null()),
            (ptr::null(), malformed_json.as_ptr()),
        ] {
            assert_status!(
                api::nemo_relay_llm_call(
                    llm_name.as_ptr(),
                    llm_request.as_ptr(),
                    scope,
                    0,
                    data,
                    metadata,
                    ptr::null(),
                    ptr::null(),
                    &mut llm,
                ),
                NemoRelayStatus::InvalidJson
            );
        }
        assert_status!(
            api::nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                scope,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            api::nemo_relay_llm_call_end(
                llm,
                llm_response.as_ptr(),
                malformed_json.as_ptr(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            api::nemo_relay_llm_call_end(
                llm,
                llm_response.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            api::nemo_relay_llm_call_end(
                llm,
                llm_response.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_llm_handle_free(llm);

        let mut out = ptr::null_mut();
        assert_status!(
            api::nemo_relay_llm_request_intercepts(
                llm_name.as_ptr(),
                invalid_llm_shape.as_ptr(),
                &mut out,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            api::nemo_relay_llm_conditional_execution(invalid_llm_shape.as_ptr()),
            NemoRelayStatus::InvalidJson
        );

        assert!(
            api::nemo_relay_llm_sanitize_request_codec_decode(ptr::null(), ptr::null()).is_null()
        );
        assert!(
            api::nemo_relay_llm_sanitize_request_codec_encode(
                ptr::null(),
                ptr::null(),
                ptr::null(),
            )
            .is_null()
        );
        assert!(
            api::nemo_relay_llm_sanitize_response_codec_decode(ptr::null(), ptr::null()).is_null()
        );
        assert_status!(
            api::nemo_relay_stream_close(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );

        assert_status!(
            api::nemo_relay_pop_scope(scope, ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_otel_projection_validation_from_integration_binary() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let otel_type = cstring("full");
        let transport = cstring("http_binary");
        let endpoint = cstring("http://127.0.0.1:4318/v1/traces");
        let headers = cstring(r#"{"authorization":"test-token"}"#);
        let resources = cstring(r#"{"service.instance.id":"ffi-integration"}"#);
        let service_name = cstring("ffi-integration");
        let service_namespace = cstring("nemo-relay");
        let service_version = cstring("test");
        let instrumentation_scope = cstring("ffi.integration");
        let malformed_json = cstring("{");
        let wrong_shape = cstring(r#"{"not":"an-array"}"#);
        let invalid_projection = cstring("invalid");
        let invalid_prefixes = cstring(r#"[""]"#);
        let projection = cstring("event");
        let excluded_names = cstring(r#"["llm.chunk"]"#);
        let mappings = cstring(r#"[{"key":"gen_ai.request.model","alias":"relay.request.model"}]"#);
        let prefixes = cstring(r#"["nv."]"#);
        let mut subscriber = ptr::null_mut();

        macro_rules! assert_projection_status {
            ($projection:expr, $excluded:expr, $mappings:expr, $prefixes:expr, $expected:expr) => {
                assert_status!(
                    nemo_relay_otel_subscriber_create_with_projection_options_v2(
                        otel_type.as_ptr(),
                        transport.as_ptr(),
                        endpoint.as_ptr(),
                        headers.as_ptr(),
                        resources.as_ptr(),
                        service_name.as_ptr(),
                        service_namespace.as_ptr(),
                        service_version.as_ptr(),
                        instrumentation_scope.as_ptr(),
                        250,
                        $projection,
                        $excluded,
                        $mappings,
                        $prefixes,
                        &mut subscriber,
                    ),
                    $expected
                )
            };
        }

        assert_projection_status!(
            invalid_projection.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            NemoRelayStatus::InvalidArg
        );
        assert_projection_status!(
            projection.as_ptr(),
            malformed_json.as_ptr(),
            ptr::null(),
            ptr::null(),
            NemoRelayStatus::InvalidJson
        );
        assert_projection_status!(
            projection.as_ptr(),
            wrong_shape.as_ptr(),
            ptr::null(),
            ptr::null(),
            NemoRelayStatus::InvalidArg
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            malformed_json.as_ptr(),
            ptr::null(),
            NemoRelayStatus::InvalidJson
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            wrong_shape.as_ptr(),
            ptr::null(),
            NemoRelayStatus::InvalidArg
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            mappings.as_ptr(),
            malformed_json.as_ptr(),
            NemoRelayStatus::InvalidJson
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            mappings.as_ptr(),
            wrong_shape.as_ptr(),
            NemoRelayStatus::InvalidArg
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            mappings.as_ptr(),
            invalid_prefixes.as_ptr(),
            NemoRelayStatus::InvalidArg
        );
        assert_projection_status!(
            projection.as_ptr(),
            excluded_names.as_ptr(),
            mappings.as_ptr(),
            prefixes.as_ptr(),
            NemoRelayStatus::Ok
        );
        assert!(!subscriber.is_null());

        let subscriber_name = cstring(&unique_name("ffi_projection_integration"));
        assert_status!(
            nemo_relay_otel_subscriber_register(subscriber, subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_register(subscriber, subscriber_name.as_ptr()),
            NemoRelayStatus::Internal
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        let mut diagnostics = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_runtime_diagnostics_json(ptr::null(), &mut diagnostics),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_otel_subscriber_runtime_diagnostics_json(subscriber, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_otel_subscriber_runtime_diagnostics_json(subscriber, &mut diagnostics),
            NemoRelayStatus::Ok
        );
        assert!(returned_json(diagnostics).is_array());
        assert_status!(
            nemo_relay_otel_subscriber_shutdown(subscriber),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_force_flush(subscriber),
            NemoRelayStatus::Internal
        );
        assert_status!(
            nemo_relay_otel_subscriber_shutdown(subscriber),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_observability_component_and_atof_validation_from_integration_binary() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut out_json = ptr::null_mut();
        assert_status!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        let invalid_config = cstring(r#"{"version":"invalid"}"#);
        assert_status!(
            nemo_relay_observability_component_spec_json(
                invalid_config.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_observability_default_config_json(&mut out_json),
            NemoRelayStatus::Ok
        );
        let valid_config = cstring(&take_string(out_json).expect("expected default config JSON"));
        assert_status!(
            nemo_relay_observability_component_spec_json(
                valid_config.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["kind"], json!("observability"));

        let valid_atof_config = cstring(r#"{"type":"file","filename":"events.jsonl"}"#);
        let invalid_atof_config = cstring(r#"{"mode":17}"#);
        let mut exporter = ptr::null_mut();
        assert_status!(
            nemo_relay_atof_exporter_create_from_json(valid_atof_config.as_ptr(), ptr::null_mut(),),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_create_from_json(invalid_atof_config.as_ptr(), &mut exporter),
            NemoRelayStatus::InvalidJson
        );

        let directory = temp_dir("ffi-atof-integration-validation");
        let directory = cstring(&directory.to_string_lossy());
        let mode = cstring("overwrite");
        let filename = cstring("events.jsonl");
        assert_status!(
            nemo_relay_atof_exporter_create(
                directory.as_ptr(),
                mode.as_ptr(),
                filename.as_ptr(),
                &mut exporter,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_path(exporter, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        let mut path = ptr::null_mut();
        assert_status!(
            nemo_relay_atof_exporter_path(exporter, &mut path),
            NemoRelayStatus::Ok
        );
        let path = take_string(path).expect("expected ATOF exporter path");
        assert!(
            path.ends_with("events.jsonl"),
            "unexpected exporter path: {path}"
        );
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        assert_status!(
            nemo_relay_atof_exporter_register(exporter, invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_deregister(invalid),
            NemoRelayStatus::InvalidUtf8
        );
        types::nemo_relay_atof_exporter_free(exporter);
    }
}

#[test]
fn test_ffi_typed_otel_signal_options_from_integration_binary() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let transport = cstring("http_binary");
        let log_endpoint = cstring("http://127.0.0.1:4318/v1/logs");
        let metric_endpoint = cstring("http://127.0.0.1:4318/v1/metrics");
        let headers = cstring(r#"{"authorization":"test-token"}"#);
        let resources = cstring(r#"{"service.instance.id":"ffi-integration"}"#);
        let service_name = cstring("ffi-integration");
        let service_namespace = cstring("nemo-relay");
        let service_version = cstring("test");
        let instrumentation_scope = cstring("ffi.integration");
        let minimum_severity = cstring("warn");
        let temporality = cstring("delta");
        let fixtures = ValidationFixtures::new("typed_otel_integration");
        let invalid = fixtures.invalid_utf8();
        let log_name = cstring(&unique_name("ffi_typed_log_integration"));
        assert_otel_signal_lifecycle(
            &log_name,
            |out| {
                nemo_relay_otel_log_subscriber_create(
                    transport.as_ptr(),
                    log_endpoint.as_ptr(),
                    headers.as_ptr(),
                    resources.as_ptr(),
                    service_name.as_ptr(),
                    service_namespace.as_ptr(),
                    service_version.as_ptr(),
                    instrumentation_scope.as_ptr(),
                    250,
                    minimum_severity.as_ptr(),
                    64,
                    16,
                    10,
                    60_000,
                    out,
                )
            },
            |subscriber| {
                assert_status!(
                    nemo_relay_otel_log_subscriber_register(subscriber, invalid),
                    NemoRelayStatus::InvalidUtf8
                );
            },
            |subscriber, name| nemo_relay_otel_log_subscriber_register(subscriber, name),
            |name| nemo_relay_otel_log_subscriber_deregister(name),
            |subscriber| nemo_relay_otel_log_subscriber_shutdown(subscriber),
            |subscriber| nemo_relay_otel_log_subscriber_force_flush(subscriber),
            |subscriber| types::nemo_relay_otel_log_subscriber_free(subscriber),
        );

        let metric_name = cstring(&unique_name("ffi_typed_metric_integration"));
        assert_otel_signal_lifecycle(
            &metric_name,
            |out| {
                nemo_relay_otel_metric_subscriber_create(
                    transport.as_ptr(),
                    metric_endpoint.as_ptr(),
                    headers.as_ptr(),
                    resources.as_ptr(),
                    service_name.as_ptr(),
                    service_namespace.as_ptr(),
                    service_version.as_ptr(),
                    instrumentation_scope.as_ptr(),
                    250,
                    20,
                    temporality.as_ptr(),
                    256,
                    128,
                    out,
                )
            },
            |subscriber| {
                assert_status!(
                    nemo_relay_otel_metric_subscriber_register(subscriber, invalid),
                    NemoRelayStatus::InvalidUtf8
                );
            },
            |subscriber, name| nemo_relay_otel_metric_subscriber_register(subscriber, name),
            |name| nemo_relay_otel_metric_subscriber_deregister(name),
            |subscriber| nemo_relay_otel_metric_subscriber_shutdown(subscriber),
            |subscriber| nemo_relay_otel_metric_subscriber_force_flush(subscriber),
            |subscriber| types::nemo_relay_otel_metric_subscriber_free(subscriber),
        );
    }
}
