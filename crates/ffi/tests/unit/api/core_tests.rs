// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_ffi_plugin_config_validate_initialize_and_clear() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let config = cstring(
        &json!({
            "version": 1,
            "components": [
                {
                    "kind": "adaptive",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "state": {
                            "backend": {
                                "kind": "in_memory",
                                "config": {}
                            }
                        },
                        "telemetry": {
                            "learners": ["latency_sensitivity"]
                        },
                        "adaptive_hints": {},
                        "tool_parallelism": {}
                    }
                }
            ]
        })
        .to_string(),
    );

    let mut report_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_validate_plugin_config(config.as_ptr(), &mut report_json) },
        NemoFlowStatus::Ok
    );
    let report = unsafe { returned_json(report_json) };
    assert_eq!(report["diagnostics"], json!([]));

    let mut kinds_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_list_plugin_kinds_json(&mut kinds_json) },
        NemoFlowStatus::Ok
    );
    let kinds = unsafe { returned_json(kinds_json) };
    assert!(
        kinds
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "adaptive"))
    );

    let mut configured_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_initialize_plugins(config.as_ptr(), &mut configured_json) },
        NemoFlowStatus::Ok
    );
    let configured_report = unsafe { returned_json(configured_json) };
    assert_eq!(configured_report["diagnostics"], json!([]));

    let mut active_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_active_plugin_report_json(&mut active_json) },
        NemoFlowStatus::Ok
    );
    let active_report = unsafe { returned_json(active_json) };
    assert_eq!(active_report["diagnostics"], json!([]));

    assert_eq!(nemo_flow_clear_plugin_configuration(), NemoFlowStatus::Ok);

    let mut cleared_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_active_plugin_report_json(&mut cleared_json) },
        NemoFlowStatus::Ok
    );
    assert_eq!(unsafe { returned_json(cleared_json) }, Json::Null);
}

#[test]
fn test_ffi_plugin_top_level_null_and_invalid_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let valid_config = cstring(
        &json!({
            "version": 1,
            "components": []
        })
        .to_string(),
    );
    let invalid_json = cstring("{");
    let invalid_shape = cstring(r#"{"version":"bad","components":"nope"}"#);

    unsafe {
        assert_eq!(
            nemo_flow_validate_plugin_config(valid_config.as_ptr(), ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out_json pointer is null")
        );

        let mut out_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_validate_plugin_config(invalid_json.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_validate_plugin_config(invalid_shape.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("invalid type")
        );

        assert_eq!(
            nemo_flow_initialize_plugins(valid_config.as_ptr(), ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_initialize_plugins(invalid_json.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_initialize_plugins(invalid_shape.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_active_plugin_report_json(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_list_plugin_kinds_json(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_register_plugin(
                ptr::null(),
                None,
                hosted_plugin_register_fail,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_deregister_plugin(ptr::null()),
            NemoFlowStatus::NullPointer
        );
    }
}

#[test]
fn test_ffi_error_paths_and_scope_stack() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        assert_eq!(
            nemo_flow_get_handle(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert!(read_last_error().unwrap().contains("out pointer is null"));

        let name = cstring("ffi_invalid_scope");
        let invalid_json = cstring("{");
        let mut handle = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                name.as_ptr(),
                NemoFlowScopeType::Agent,
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );

        let stack = fresh_scope_stack();
        assert!(nemo_flow_scope_stack_active());

        let mut root = ptr::null_mut();
        assert_eq!(nemo_flow_get_handle(&mut root), NemoFlowStatus::Ok);
        let root_uuid = take_string(nemo_flow_scope_handle_uuid(root)).unwrap();
        assert!(!root_uuid.is_empty());
        assert_eq!(
            nemo_flow_scope_handle_scope_type(root) as i32,
            NemoFlowScopeType::Agent as i32
        );
        assert_eq!(nemo_flow_scope_handle_attributes(root), 0);
        nemo_flow_scope_handle_free(root);

        let scope_name = cstring("ffi_scope");
        let scope_data = cstring(r#"{"scope":true}"#);
        let scope_metadata = cstring(r#"{"meta":"ok"}"#);
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                scope_name.as_ptr(),
                NemoFlowScopeType::Function,
                ptr::null(),
                1,
                scope_data.as_ptr(),
                scope_metadata.as_ptr(),
                &mut scope,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            take_string(nemo_flow_scope_handle_name(scope)).unwrap(),
            "ffi_scope"
        );
        assert_eq!(
            nemo_flow_scope_handle_scope_type(scope) as i32,
            NemoFlowScopeType::Function as i32
        );
        assert_eq!(nemo_flow_scope_handle_attributes(scope), 1);
        assert!(take_string(nemo_flow_scope_handle_parent_uuid(scope)).is_some());
        assert_eq!(
            serde_json::from_str::<Json>(&take_string(nemo_flow_scope_handle_data(scope)).unwrap())
                .unwrap(),
            json!({"scope": true})
        );
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_flow_scope_handle_metadata(scope)).unwrap()
            )
            .unwrap(),
            json!({"meta": "ok"})
        );
        assert_eq!(nemo_flow_pop_scope(scope), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(scope);

        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_lifecycle_execute_and_helpers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = unique_name("ffi_subscriber");
        let subscriber_name_c = cstring(&subscriber_name);
        assert_eq!(
            nemo_flow_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let intercept_name = unique_name("ffi_tool_intercept");
        let intercept_name_c = cstring(&intercept_name);
        assert_eq!(
            nemo_flow_register_tool_request_intercept(
                intercept_name_c.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let conditional_name = unique_name("ffi_tool_conditional");
        let conditional_name_c = cstring(&conditional_name);
        assert_eq!(
            nemo_flow_register_tool_conditional_execution_guardrail(
                conditional_name_c.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let tool_name = cstring("ffi_tool");
        let args = cstring(r#"{"value": 1}"#);
        let mut intercepted_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_request_intercepts(
                tool_name.as_ptr(),
                args.as_ptr(),
                &mut intercepted_out
            ),
            NemoFlowStatus::Ok
        );
        let intercepted_json = returned_json(intercepted_out);
        assert_eq!(intercepted_json["intercepted"], json!(true));

        assert_eq!(
            nemo_flow_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
            NemoFlowStatus::Ok
        );

        let tool_call_id = cstring("call_ffi_123");
        let metadata = cstring(r#"{"source":"ffi-test"}"#);
        let mut handle: *mut FfiToolHandle = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_call(
                tool_name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                1,
                ptr::null(),
                metadata.as_ptr(),
                tool_call_id.as_ptr(),
                &mut handle,
            ),
            NemoFlowStatus::Ok
        );
        assert!(take_string(nemo_flow_tool_handle_uuid(handle)).is_some());
        assert_eq!(
            take_string(nemo_flow_tool_handle_name(handle)).unwrap(),
            "ffi_tool"
        );
        assert_eq!(nemo_flow_tool_handle_attributes(handle), 1);
        assert!(take_string(nemo_flow_tool_handle_parent_uuid(handle)).is_some());

        let result = cstring(r#"{"ok": true}"#);
        assert_eq!(
            nemo_flow_tool_call_end(handle, result.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_tool_handle_free(handle);

        let mut execute_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_call_execute(
                tool_name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut execute_out,
            ),
            NemoFlowStatus::Ok
        );
        let executed_json = returned_json(execute_out);
        assert_eq!(executed_json["intercepted"], json!(true));
        assert_eq!(executed_json["executed"], json!(true));

        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| event["name"] == "ffi_tool"));
        assert!(
            events
                .iter()
                .any(|event| event["tool_call_id"] == "call_ffi_123")
        );
        assert!(
            events
                .iter()
                .any(|event| event["timestamp"].as_str().is_some_and(|s| !s.is_empty()))
        );

        let mark_name = cstring("ffi_mark");
        let mark_data = cstring(r#"{"mark":true}"#);
        let mark_metadata = cstring(r#"{"origin":"ffi"}"#);
        assert_eq!(
            nemo_flow_event(
                mark_name.as_ptr(),
                ptr::null(),
                mark_data.as_ptr(),
                mark_metadata.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );
        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| {
            event["name"] == "ffi_mark"
                && event["kind"] == json!("Mark")
                && event["data"] == json!({"mark": true})
                && event["metadata"] == json!({"origin": "ffi"})
        }));

        assert_eq!(
            nemo_flow_deregister_tool_request_intercept(intercept_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_tool_conditional_execution_guardrail(conditional_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_additional_null_and_invalid_json_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("ffi_edge_paths");
        let args = cstring(r#"{"value": 1}"#);
        let invalid_json = cstring("{");
        let invalid_request_shape = cstring(r#"{"headers":[],"content":"bad"}"#);
        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model"}}"#);
        let mut handle: *mut FfiToolHandle = ptr::null_mut();
        let mut llm_handle: *mut FfiLLMHandle = ptr::null_mut();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut stream: *mut FfiStream = ptr::null_mut();

        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_tool_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, invalid_json.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, args.as_ptr(), invalid_json.as_ptr(), ptr::null(),),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_tool_handle_free(handle);

        assert_eq!(
            nemo_flow_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call_execute(
                name.as_ptr(),
                invalid_json.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out_json,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LLMRequest")
        );

        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_llm_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call_end(llm_handle, invalid_json.as_ptr(), ptr::null(), ptr::null(),),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_llm_call_end(llm_handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_llm_handle_free(llm_handle);

        assert_eq!(
            nemo_flow_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call_execute(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut out_json,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_stream_call_execute(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null(),
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
            NemoFlowStatus::InvalidJson
        );

        nemo_flow_scope_stack_free(stack);
    }
}
