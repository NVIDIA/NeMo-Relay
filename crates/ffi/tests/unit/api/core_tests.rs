// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for core in the NeMo Relay FFI crate.

use super::*;

#[test]
fn test_ffi_plugin_config_validate_initialize_and_clear() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

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
        unsafe { nemo_relay_validate_plugin_config(config.as_ptr(), &mut report_json) },
        NemoRelayStatus::Ok
    );
    let report = unsafe { returned_json(report_json) };
    assert_eq!(report["diagnostics"], json!([]));

    let mut kinds_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_list_plugin_kinds_json(&mut kinds_json) },
        NemoRelayStatus::Ok
    );
    let kinds = unsafe { returned_json(kinds_json) };
    assert!(
        kinds
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "adaptive"))
    );
    assert!(
        kinds
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "observability"))
    );

    let mut configured_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_initialize_plugins(config.as_ptr(), &mut configured_json) },
        NemoRelayStatus::Ok
    );
    let configured_report = unsafe { returned_json(configured_json) };
    assert_eq!(configured_report["diagnostics"], json!([]));

    let mut active_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_active_plugin_report_json(&mut active_json) },
        NemoRelayStatus::Ok
    );
    let active_report = unsafe { returned_json(active_json) };
    assert_eq!(active_report["diagnostics"], json!([]));

    assert_eq!(nemo_relay_clear_plugin_configuration(), NemoRelayStatus::Ok);

    let mut cleared_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_active_plugin_report_json(&mut cleared_json) },
        NemoRelayStatus::Ok
    );
    assert_eq!(unsafe { returned_json(cleared_json) }, Json::Null);
}

#[test]
fn test_ffi_observability_plugin_file_sinks() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();
    let dir = std::env::temp_dir().join(unique_name("ffi_observability_plugin"));
    std::fs::create_dir_all(&dir).unwrap();
    let dir_text = dir.to_string_lossy().into_owned();

    let config = cstring(
        &json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atof": {
                            "enabled": true,
                            "output_directory": dir_text,
                            "filename": "events.jsonl",
                            "mode": "overwrite"
                        },
                        "atif": {
                            "enabled": true,
                            "agent_name": "ffi-agent",
                            "agent_version": "1.2.3",
                            "model_name": "ffi-model",
                            "tool_definitions": [{"name": "search"}],
                            "extra": {"binding": "ffi"},
                            "output_directory": dir_text,
                            "filename_template": "trajectory-{session_id}.json"
                        }
                    }
                }
            ]
        })
        .to_string(),
    );

    unsafe {
        assert_eq!(
            take_string(nemo_relay_observability_plugin_kind()).unwrap(),
            "observability"
        );
        let mut default_config_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_observability_default_config_json(&mut default_config_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(default_config_json)["version"], json!(1));
        let mut component_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, &mut component_json),
            NemoRelayStatus::Ok
        );
        let component = returned_json(component_json);
        assert_eq!(component["kind"], "observability");
        assert_eq!(component["enabled"], true);

        let mut report_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_validate_plugin_config(config.as_ptr(), &mut report_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(report_json)["diagnostics"], json!([]));

        let mut initialized_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_initialize_plugins(config.as_ptr(), &mut initialized_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(initialized_json)["diagnostics"], json!([]));

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi-observability-agent");
        let input = cstring(r#"{"agent":true}"#);
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                input.as_ptr(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        let scope_uuid = take_string(nemo_relay_scope_handle_uuid(scope)).unwrap();

        let mark_name = cstring("ffi-observability-mark");
        let mark_data = cstring(r#"{"step":1}"#);
        assert_eq!(
            nemo_relay_event(mark_name.as_ptr(), scope, mark_data.as_ptr(), ptr::null()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
        assert_eq!(nemo_relay_clear_plugin_configuration(), NemoRelayStatus::Ok);

        let jsonl = std::fs::read_to_string(dir.join("events.jsonl")).unwrap();
        assert_eq!(jsonl.trim().lines().count(), 3);

        let trajectory_path = dir.join(format!("trajectory-{scope_uuid}.json"));
        let trajectory: Json =
            serde_json::from_str(&std::fs::read_to_string(trajectory_path).unwrap()).unwrap();
        assert_eq!(trajectory["agent"]["name"], "ffi-agent");
        assert_eq!(trajectory["agent"]["version"], "1.2.3");
        assert_eq!(trajectory["agent"]["model_name"], "ffi-model");
        assert!(
            trajectory["extra"]
                .to_string()
                .contains("ffi-observability-agent")
        );
    }
}

#[test]
fn test_ffi_observability_plugin_atif_splits_multiple_top_level_agents() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();
    let dir = std::env::temp_dir().join(unique_name("ffi_observability_plugin_multi_agent"));
    std::fs::create_dir_all(&dir).unwrap();
    let dir_text = dir.to_string_lossy().into_owned();

    let config = cstring(
        &json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "atif": {
                            "enabled": true,
                            "output_directory": dir_text,
                            "filename_template": "trajectory-{session_id}.json"
                        }
                    }
                }
            ]
        })
        .to_string(),
    );

    unsafe {
        let mut initialized_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_initialize_plugins(config.as_ptr(), &mut initialized_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(initialized_json)["diagnostics"], json!([]));

        let stack = fresh_scope_stack();

        let first_name = cstring("ffi-first-agent");
        let first_input = cstring(r#"{"agent":"first"}"#);
        let mut first = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                first_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                first_input.as_ptr(),
                &mut first,
            ),
            NemoRelayStatus::Ok
        );
        let first_uuid = take_string(nemo_relay_scope_handle_uuid(first)).unwrap();

        let first_mark = cstring("ffi-first-mark");
        let first_mark_data = cstring(r#"{"agent":"first"}"#);
        assert_eq!(
            nemo_relay_event(
                first_mark.as_ptr(),
                first,
                first_mark_data.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::Ok
        );

        let nested_name = cstring("ffi-nested-agent");
        let nested_input = cstring(r#"{"agent":"nested"}"#);
        let mut nested = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                nested_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                nested_input.as_ptr(),
                &mut nested,
            ),
            NemoRelayStatus::Ok
        );
        let nested_mark = cstring("ffi-nested-mark");
        let nested_mark_data = cstring(r#"{"agent":"nested"}"#);
        assert_eq!(
            nemo_relay_event(
                nested_mark.as_ptr(),
                nested,
                nested_mark_data.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(nested, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(nested);
        assert_eq!(
            nemo_relay_pop_scope(first, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(first);

        let second_name = cstring("ffi-second-agent");
        let second_input = cstring(r#"{"agent":"second"}"#);
        let mut second = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                second_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                second_input.as_ptr(),
                &mut second,
            ),
            NemoRelayStatus::Ok
        );
        let second_uuid = take_string(nemo_relay_scope_handle_uuid(second)).unwrap();
        let second_mark = cstring("ffi-second-mark");
        let second_mark_data = cstring(r#"{"agent":"second"}"#);
        assert_eq!(
            nemo_relay_event(
                second_mark.as_ptr(),
                second,
                second_mark_data.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(second, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(second);
        nemo_relay_scope_stack_free(stack);
        assert_eq!(nemo_relay_clear_plugin_configuration(), NemoRelayStatus::Ok);

        let files = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .ok()
                    .and_then(|entry| entry.file_name().into_string().ok())
                    .is_some_and(|name| name.starts_with("trajectory-"))
            })
            .count();
        assert_eq!(files, 2);

        let first_payload =
            std::fs::read_to_string(dir.join(format!("trajectory-{first_uuid}.json"))).unwrap();
        let second_payload =
            std::fs::read_to_string(dir.join(format!("trajectory-{second_uuid}.json"))).unwrap();
        assert!(first_payload.contains("ffi-first-agent"));
        assert!(first_payload.contains("ffi-nested-agent"));
        assert!(!first_payload.contains("ffi-second-agent"));
        assert!(second_payload.contains("ffi-second-agent"));
        assert!(!second_payload.contains("ffi-first-agent"));
        assert!(!second_payload.contains("ffi-nested-agent"));
    }
}

#[test]
fn test_ffi_plugin_top_level_null_and_invalid_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

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
            nemo_relay_validate_plugin_config(valid_config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out_json pointer is null")
        );

        let mut out_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_validate_plugin_config(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_validate_plugin_config(invalid_shape.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("invalid type")
        );

        assert_eq!(
            nemo_relay_initialize_plugins(valid_config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_initialize_plugins(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_initialize_plugins(invalid_shape.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
            nemo_relay_active_plugin_report_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_list_plugin_kinds_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_register_plugin(
                ptr::null(),
                None,
                plugin_register_fail,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_deregister_plugin(ptr::null()),
            NemoRelayStatus::NullPointer
        );
    }
}

#[test]
fn test_ffi_error_paths_and_scope_stack() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        assert_eq!(
            nemo_relay_get_handle(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert!(read_last_error().unwrap().contains("out pointer is null"));

        let name = cstring("ffi_invalid_scope");
        let invalid_json = cstring("{");
        let mut handle = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidJson
        );

        let stack = fresh_scope_stack();
        assert!(nemo_relay_scope_stack_active());

        let mut root = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut root), NemoRelayStatus::Ok);
        let root_uuid = take_string(nemo_relay_scope_handle_uuid(root)).unwrap();
        assert!(!root_uuid.is_empty());
        assert_eq!(
            nemo_relay_scope_handle_scope_type(root) as i32,
            NemoRelayScopeType::Agent as i32
        );
        assert_eq!(nemo_relay_scope_handle_attributes(root), 0);
        nemo_relay_scope_handle_free(root);

        let scope_name = cstring("ffi_scope");
        let scope_data = cstring(r#"{"scope":true}"#);
        let scope_metadata = cstring(r#"{"meta":"ok"}"#);
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                ptr::null(),
                1,
                scope_data.as_ptr(),
                scope_metadata.as_ptr(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            take_string(nemo_relay_scope_handle_name(scope)).unwrap(),
            "ffi_scope"
        );
        assert_eq!(
            nemo_relay_scope_handle_scope_type(scope) as i32,
            NemoRelayScopeType::Function as i32
        );
        assert_eq!(nemo_relay_scope_handle_attributes(scope), 1);
        assert!(take_string(nemo_relay_scope_handle_parent_uuid(scope)).is_some());
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_relay_scope_handle_data(scope)).unwrap()
            )
            .unwrap(),
            json!({"scope": true})
        );
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_relay_scope_handle_metadata(scope)).unwrap()
            )
            .unwrap(),
            json!({"meta": "ok"})
        );
        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);

        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_pop_scope_merges_scope_metadata() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = unique_name("ffi_scope_end_metadata_subscriber");
        let subscriber_name_c = cstring(&subscriber_name);
        assert_eq!(
            nemo_relay_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let scope_name = cstring("ffi_scope_end_metadata");
        let scope_metadata = cstring(r#"{"a":1,"b":2,"c":3}"#);
        let end_metadata = cstring(r#"{"c":3.5,"d":4}"#);
        let invalid_end_metadata = cstring("{");
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                scope_metadata.as_ptr(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            api::nemo_relay_pop_scope(scope, ptr::null(), end_metadata.as_ptr(), ptr::null(),),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            api::nemo_relay_pop_scope(
                scope,
                ptr::null(),
                invalid_end_metadata.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);

        let events = lock_unpoisoned(event_log()).clone();
        let end_event = events
            .iter()
            .find(|event| {
                event["json"]["kind"] == json!("scope")
                    && event["json"]["name"] == json!("ffi_scope_end_metadata")
                    && event["json"]["scope_category"] == json!("end")
            })
            .unwrap();
        assert_eq!(
            end_event["metadata"],
            json!({"a": 1, "b": 2, "c": 3.5, "d": 4})
        );

        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_event_json_null_pointer_returns_null() {
    unsafe {
        assert!(types::nemo_relay_event_json(ptr::null::<FfiEvent>()).is_null());
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
            nemo_relay_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let intercept_name = unique_name("ffi_tool_intercept");
        let intercept_name_c = cstring(&intercept_name);
        assert_eq!(
            nemo_relay_register_tool_request_intercept(
                intercept_name_c.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let conditional_name = unique_name("ffi_tool_conditional");
        let conditional_name_c = cstring(&conditional_name);
        assert_eq!(
            nemo_relay_register_tool_conditional_execution_guardrail(
                conditional_name_c.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let tool_name = cstring("ffi_tool");
        let args = cstring(r#"{"value": 1}"#);
        let mut intercepted_out = ptr::null_mut();
        assert_eq!(
            nemo_relay_tool_request_intercepts(
                tool_name.as_ptr(),
                args.as_ptr(),
                &mut intercepted_out
            ),
            NemoRelayStatus::Ok
        );
        let intercepted_json = returned_json(intercepted_out);
        assert_eq!(intercepted_json["intercepted"], json!(true));

        assert_eq!(
            nemo_relay_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_call_id = cstring("call_ffi_123");
        let metadata = cstring(r#"{"source":"ffi-test"}"#);
        let mut handle: *mut FfiToolHandle = ptr::null_mut();
        assert_eq!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                1,
                ptr::null(),
                metadata.as_ptr(),
                tool_call_id.as_ptr(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_tool_handle_uuid(handle)).is_some());
        assert_eq!(
            take_string(nemo_relay_tool_handle_name(handle)).unwrap(),
            "ffi_tool"
        );
        assert_eq!(nemo_relay_tool_handle_attributes(handle), 1);
        assert!(take_string(nemo_relay_tool_handle_parent_uuid(handle)).is_some());

        let result = cstring(r#"{"ok": true}"#);
        assert_eq!(
            nemo_relay_tool_call_end(handle, result.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(handle);

        let mut execute_out = ptr::null_mut();
        assert_eq!(
            nemo_relay_tool_call_execute(
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
            NemoRelayStatus::Ok
        );
        let executed_json = returned_json(execute_out);
        assert_eq!(executed_json["intercepted"], json!(true));
        assert_eq!(executed_json["executed"], json!(true));

        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| event["name"] == "ffi_tool"));
        assert!(events.iter().any(|event| {
            event["json"]["kind"] == json!("scope")
                && event["json"]["name"] == json!("ffi_tool")
                && event["json"]["category"] == json!("tool")
        }));
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
            nemo_relay_event(
                mark_name.as_ptr(),
                ptr::null(),
                mark_data.as_ptr(),
                mark_metadata.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| {
            event["name"] == "ffi_mark"
                && event["kind"] == json!("mark")
                && event["json"]["kind"] == json!("mark")
                && event["json"]["name"] == json!("ffi_mark")
                && event["data"] == json!({"mark": true})
                && event["metadata"] == json!({"origin": "ffi"})
        }));

        assert_eq!(
            nemo_relay_deregister_tool_request_intercept(intercept_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(conditional_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_manual_lifecycle_timestamps_accept_unix_micros() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    fn micros(value: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(value)
            .unwrap()
            .timestamp_micros()
    }

    fn observed_micros(event: &Json) -> i64 {
        chrono::DateTime::parse_from_rfc3339(event["timestamp"].as_str().unwrap())
            .unwrap()
            .timestamp_micros()
    }

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = unique_name("ffi_timestamp_subscriber");
        let subscriber_name_c = cstring(&subscriber_name);
        assert_eq!(
            nemo_relay_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let timestamps = [
            micros("2026-01-01T00:00:00.123456Z"),
            micros("2026-01-01T00:00:01.123456Z"),
            micros("2026-01-01T00:00:02.123456Z"),
            micros("2026-01-01T00:00:03.123456Z"),
            micros("2026-01-01T00:00:04.123456Z"),
            micros("2026-01-01T00:00:05.123456Z"),
            micros("2026-01-01T00:00:06.123456Z"),
        ];

        let scope_name = cstring("ffi_ts_scope");
        let mut scope: *mut FfiScopeHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &timestamps[0],
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );

        let mark_name = cstring("ffi_ts_mark");
        assert_eq!(
            api::nemo_relay_event(
                mark_name.as_ptr(),
                scope,
                ptr::null(),
                ptr::null(),
                &timestamps[1],
            ),
            NemoRelayStatus::Ok
        );

        let tool_name = cstring("ffi_ts_tool");
        let tool_args = cstring(r#"{"x":1}"#);
        let mut tool: *mut FfiToolHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &timestamps[2],
                &mut tool,
            ),
            NemoRelayStatus::Ok
        );
        let tool_result = cstring(r#"{"ok":true}"#);
        assert_eq!(
            api::nemo_relay_tool_call_end(
                tool,
                tool_result.as_ptr(),
                ptr::null(),
                ptr::null(),
                &timestamps[3],
            ),
            NemoRelayStatus::Ok
        );

        let llm_name = cstring("ffi_ts_llm");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"messages":[],"model":"test-model"}}"#);
        let mut llm: *mut FfiLLMHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &timestamps[4],
                &mut llm,
            ),
            NemoRelayStatus::Ok
        );
        let llm_response = cstring(r#"{"ok":true}"#);
        assert_eq!(
            api::nemo_relay_llm_call_end(
                llm,
                llm_response.as_ptr(),
                ptr::null(),
                ptr::null(),
                &timestamps[5],
            ),
            NemoRelayStatus::Ok
        );

        assert_eq!(
            api::nemo_relay_pop_scope(scope, ptr::null(), ptr::null(), &timestamps[6]),
            NemoRelayStatus::Ok
        );

        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log()).clone();
        let observed: Vec<_> = events
            .iter()
            .filter(|event| {
                event["name"]
                    .as_str()
                    .is_some_and(|name| name.starts_with("ffi_ts_"))
            })
            .map(|event| {
                (
                    event["name"].as_str().unwrap().to_string(),
                    observed_micros(event),
                )
            })
            .collect();
        assert_eq!(
            observed,
            vec![
                ("ffi_ts_scope".to_string(), timestamps[0]),
                ("ffi_ts_mark".to_string(), timestamps[1]),
                ("ffi_ts_tool".to_string(), timestamps[2]),
                ("ffi_ts_tool".to_string(), timestamps[3]),
                ("ffi_ts_llm".to_string(), timestamps[4]),
                ("ffi_ts_llm".to_string(), timestamps[5]),
                ("ffi_ts_scope".to_string(), timestamps[6]),
            ]
        );

        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(tool);
        nemo_relay_llm_handle_free(llm);
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_manual_lifecycle_timestamps_reject_out_of_range_unix_micros() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    fn assert_invalid_timestamp(status: NemoRelayStatus) {
        assert_eq!(status, NemoRelayStatus::InvalidArg);
        assert!(
            unsafe { read_last_error() }
                .unwrap_or_default()
                .contains("unix microseconds are outside supported range")
        );
    }

    unsafe {
        let stack = fresh_scope_stack();
        let invalid_timestamp = i64::MAX;

        let invalid_scope_name = cstring("ffi_bad_ts_scope");
        let mut invalid_scope: *mut FfiScopeHandle = ptr::null_mut();
        assert_invalid_timestamp(api::nemo_relay_push_scope(
            invalid_scope_name.as_ptr(),
            NemoRelayScopeType::Agent,
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
            &mut invalid_scope,
        ));
        assert!(invalid_scope.is_null());

        let scope_name = cstring("ffi_valid_ts_scope");
        let mut scope: *mut FfiScopeHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Agent,
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

        let mark_name = cstring("ffi_bad_ts_mark");
        assert_invalid_timestamp(api::nemo_relay_event(
            mark_name.as_ptr(),
            scope,
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
        ));

        let invalid_tool_name = cstring("ffi_bad_ts_tool");
        let tool_args = cstring(r#"{"x":1}"#);
        let mut invalid_tool: *mut FfiToolHandle = ptr::null_mut();
        assert_invalid_timestamp(api::nemo_relay_tool_call(
            invalid_tool_name.as_ptr(),
            tool_args.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
            &mut invalid_tool,
        ));
        assert!(invalid_tool.is_null());

        let tool_name = cstring("ffi_valid_ts_tool");
        let mut tool: *mut FfiToolHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut tool,
            ),
            NemoRelayStatus::Ok
        );
        let tool_result = cstring(r#"{"ok":true}"#);
        assert_invalid_timestamp(api::nemo_relay_tool_call_end(
            tool,
            tool_result.as_ptr(),
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
        ));
        assert_eq!(
            api::nemo_relay_tool_call_end(
                tool,
                tool_result.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );

        let invalid_llm_name = cstring("ffi_bad_ts_llm");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"messages":[],"model":"test-model"}}"#);
        let mut invalid_llm: *mut FfiLLMHandle = ptr::null_mut();
        assert_invalid_timestamp(api::nemo_relay_llm_call(
            invalid_llm_name.as_ptr(),
            llm_request.as_ptr(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
            &mut invalid_llm,
        ));
        assert!(invalid_llm.is_null());

        let llm_name = cstring("ffi_valid_ts_llm");
        let mut llm: *mut FfiLLMHandle = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm,
            ),
            NemoRelayStatus::Ok
        );
        let llm_response = cstring(r#"{"ok":true}"#);
        assert_invalid_timestamp(api::nemo_relay_llm_call_end(
            llm,
            llm_response.as_ptr(),
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
        ));
        assert_eq!(
            api::nemo_relay_llm_call_end(
                llm,
                llm_response.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );

        assert_invalid_timestamp(api::nemo_relay_pop_scope(
            scope,
            ptr::null(),
            ptr::null(),
            &invalid_timestamp,
        ));
        assert_eq!(
            api::nemo_relay_pop_scope(scope, ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );

        nemo_relay_tool_handle_free(tool);
        nemo_relay_llm_handle_free(llm);
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
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
            nemo_relay_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_tool_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
            nemo_relay_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_tool_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_tool_call_end(handle, invalid_json.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_tool_call_end(handle, args.as_ptr(), invalid_json.as_ptr(), ptr::null(),),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_tool_call_end(handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(handle);

        assert_eq!(
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
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_tool_call_execute(
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
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_llm_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_llm_call(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        assert_eq!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_llm_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_llm_call_end(llm_handle, invalid_json.as_ptr(), ptr::null(), ptr::null(),),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_llm_call_end(llm_handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_llm_handle_free(llm_handle);

        assert_eq!(
            nemo_relay_llm_call_execute(
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
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_llm_call_execute(
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
            NemoRelayStatus::InvalidJson
        );

        assert_eq!(
            nemo_relay_llm_stream_call_execute(
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
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_llm_stream_call_execute(
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
            NemoRelayStatus::InvalidJson
        );

        nemo_relay_scope_stack_free(stack);
    }
}
