// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin in the NeMo Relay FFI crate.

use super::*;
use nemo_relay::plugin::rollback_registrations;

unsafe extern "C" fn plugin_event_metadata_injector_cb(
    _user_data: *mut libc::c_void,
    event: *const FfiEvent,
) -> *mut c_char {
    let name = unsafe { take_string(nemo_relay_event_name(event)) }.unwrap_or_default();
    CString::new(json!({"ffi.injected": name}).to_string())
        .unwrap()
        .into_raw()
}

#[test]
fn test_ffi_dynamic_plugin_activation_rejects_empty_specs_without_outputs() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let config = cstring(r#"{"version":1,"components":[]}"#);
    let specs = cstring("[]");
    for _ in 0..2 {
        let mut activation = std::ptr::dangling_mut::<FfiPluginActivation>();
        let mut report_json = std::ptr::dangling_mut::<c_char>();
        unsafe {
            assert_status!(
                nemo_relay_initialize_with_dynamic_plugins(
                    config.as_ptr(),
                    specs.as_ptr(),
                    &mut activation,
                    &mut report_json,
                ),
                NemoRelayStatus::InvalidArg
            );
            assert!(activation.is_null());
            assert!(report_json.is_null());
            assert!(
                read_last_error()
                    .unwrap_or_default()
                    .contains("at least one dynamic plugin")
            );
        }
    }
}

#[test]
fn test_ffi_dynamic_plugin_activation_rejects_invalid_inputs_without_outputs() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let config = cstring(r#"{"version":1,"components":[]}"#);
    let specs = cstring("[]");
    let invalid = cstring("not-json");
    unsafe {
        let mut report = std::ptr::dangling_mut::<c_char>();
        assert_status!(
            nemo_relay_initialize_with_dynamic_plugins(
                config.as_ptr(),
                specs.as_ptr(),
                ptr::null_mut(),
                &mut report,
            ),
            NemoRelayStatus::NullPointer
        );
        assert!(report.is_null());

        let mut activation = std::ptr::dangling_mut::<FfiPluginActivation>();
        assert_status!(
            nemo_relay_initialize_with_dynamic_plugins(
                config.as_ptr(),
                specs.as_ptr(),
                &mut activation,
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert!(activation.is_null());

        for (config_json, specs_json, expected_error) in [
            (invalid.as_ptr(), specs.as_ptr(), "invalid JSON"),
            (config.as_ptr(), invalid.as_ptr(), "invalid JSON"),
        ] {
            let mut activation = std::ptr::dangling_mut::<FfiPluginActivation>();
            let mut report = std::ptr::dangling_mut::<c_char>();
            assert_status!(
                nemo_relay_initialize_with_dynamic_plugins(
                    config_json,
                    specs_json,
                    &mut activation,
                    &mut report,
                ),
                NemoRelayStatus::InvalidJson
            );
            assert!(activation.is_null());
            assert!(report.is_null());
            assert!(
                read_last_error()
                    .unwrap_or_default()
                    .contains(expected_error)
            );
        }

        let invalid_shape = cstring(r#"{"plugin_id":"not-an-array"}"#);
        let mut activation = ptr::null_mut();
        let mut report = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_with_dynamic_plugins(
                config.as_ptr(),
                invalid_shape.as_ptr(),
                &mut activation,
                &mut report,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("invalid dynamic plugin specifications")
        );
    }
}

#[test]
fn test_ffi_dynamic_plugin_activation_surfaces_load_failures_and_releases_owner() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let config = cstring(r#"{"version":1,"components":[]}"#);
    let missing_manifest = std::env::temp_dir()
        .join(format!("missing-relay-plugin-{}.toml", Uuid::now_v7()))
        .to_string_lossy()
        .into_owned();
    let specs = cstring(
        &json!([{
            "plugin_id": "fixture_missing",
            "kind": "rust_dynamic",
            "manifest_ref": missing_manifest,
            "config": {}
        }])
        .to_string(),
    );

    unsafe {
        let mut activation = ptr::null_mut();
        let mut report = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_with_dynamic_plugins(
                config.as_ptr(),
                specs.as_ptr(),
                &mut activation,
                &mut report,
            ),
            NemoRelayStatus::NotFound
        );
        assert!(activation.is_null());
        assert!(report.is_null());
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("native plugin load failed")
        );

        let mut retry_activation = ptr::null_mut();
        let mut retry_report = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_with_dynamic_plugins(
                config.as_ptr(),
                specs.as_ptr(),
                &mut retry_activation,
                &mut retry_report,
            ),
            NemoRelayStatus::NotFound
        );
        assert!(retry_activation.is_null());
        assert!(retry_report.is_null());
    }
}

#[test]
fn test_ffi_plugin_registration_validation_and_cleanup() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let plugin_kind = unique_name("ffi_plugin");
    let plugin_kind_c = cstring(&plugin_kind);
    let config = cstring(
        &json!({
            "version": 1,
            "components": [{
                "kind": plugin_kind,
                "enabled": true,
                "config": {}
            }]
        })
        .to_string(),
    );
    let user_data = Box::into_raw(Box::new(7usize)) as *mut libc::c_void;

    unsafe {
        assert_status!(
            nemo_relay_register_plugin(
                plugin_kind_c.as_ptr(),
                Some(plugin_validate_warn),
                plugin_register_subscriber,
                user_data,
                Some(plugin_free),
            ),
            NemoRelayStatus::Ok
        );

        let mut report_json = ptr::null_mut();
        assert_status!(
            nemo_relay_validate_plugin_config(config.as_ptr(), &mut report_json),
            NemoRelayStatus::Ok
        );
        let report = returned_json(report_json);
        assert!(
            report["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.warning")
        );

        let mut init_json = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_plugins(config.as_ptr(), &mut init_json),
            NemoRelayStatus::Ok
        );
        let initialized = returned_json(init_json);
        assert!(
            initialized["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "plugin.warning")
        );

        let mut active_json = ptr::null_mut();
        assert_status!(
            nemo_relay_active_plugin_report_json(&mut active_json),
            NemoRelayStatus::Ok
        );
        let active = returned_json(active_json);
        assert_eq!(active["diagnostics"], initialized["diagnostics"]);

        assert_status!(nemo_relay_clear_plugin_configuration(), NemoRelayStatus::Ok);
        assert_status!(
            nemo_relay_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoRelayStatus::NotFound
        );
    }

    assert_eq!(*lock_unpoisoned(plugin_frees()), 1);
}

#[test]
fn test_ffi_plugin_context_event_metadata_injector_is_rolled_back() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    let subscriber_name = cstring(&unique_name("ffi_plugin_metadata_subscriber"));
    let injector_name = cstring("metadata");
    let mut registrations = PluginRegistrationContext::with_namespace("ffi_plugin::");
    let mut ctx = FfiPluginContext(&mut registrations as *mut _);

    unsafe {
        assert_status!(
            nemo_relay_register_subscriber(
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_event_metadata_injector(
                &mut ctx,
                injector_name.as_ptr(),
                10,
                Some(plugin_event_metadata_injector_cb),
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let active_name = cstring("ffi-plugin-metadata-active");
        assert_status!(
            nemo_relay_event(active_name.as_ptr(), ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        assert_status!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);

        let mut registrations = registrations.into_registrations();
        rollback_registrations(&mut registrations);
        let cleanup_name = cstring("ffi-plugin-metadata-cleanup");
        assert_status!(
            nemo_relay_event(cleanup_name.as_ptr(), ptr::null(), ptr::null(), ptr::null(),),
            NemoRelayStatus::Ok
        );
        assert_status!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);

        let events = lock_unpoisoned(event_log());
        let active = events
            .iter()
            .find(|event| event["name"] == "ffi-plugin-metadata-active")
            .expect("plugin-active mark should be delivered");
        let cleanup = events
            .iter()
            .find(|event| event["name"] == "ffi-plugin-metadata-cleanup")
            .expect("plugin-cleanup mark should be delivered");
        assert_eq!(
            active["metadata"]["ffi.injected"],
            json!("ffi-plugin-metadata-active")
        );
        assert!(cleanup["metadata"].is_null());
        drop(events);

        assert_status!(
            nemo_relay_deregister_subscriber(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
    }
}

#[test]
fn test_ffi_plugin_context_event_metadata_injector_rejects_null_callback() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    let injector_name = cstring("metadata");
    let mut registrations = PluginRegistrationContext::with_namespace("ffi_plugin_null::");
    let mut ctx = FfiPluginContext(&mut registrations as *mut _);

    unsafe {
        assert_status!(
            nemo_relay_plugin_context_register_event_metadata_injector(
                &mut ctx,
                injector_name.as_ptr(),
                10,
                None,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_event_metadata_injector(
                &mut ctx,
                injector_name.as_ptr(),
                10,
                Some(plugin_event_metadata_injector_cb),
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
    }

    let mut registrations = registrations.into_registrations();
    rollback_registrations(&mut registrations);
}

#[test]
fn test_ffi_plugin_validation_failure_modes_are_reported() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    for (suffix, validate_cb, expected_fragment) in [
        (
            "invalid",
            Some(plugin_validate_invalid as callable::NemoRelayPluginValidateCb),
            "invalid diagnostics JSON",
        ),
        (
            "null",
            Some(plugin_validate_null as callable::NemoRelayPluginValidateCb),
            "returned null",
        ),
    ] {
        let plugin_kind = unique_name(&format!("ffi_plugin_{suffix}"));
        let plugin_kind_c = cstring(&plugin_kind);
        let config = cstring(
            &json!({
                "version": 1,
                "components": [{
                    "kind": plugin_kind,
                    "enabled": true,
                    "config": {}
                }]
            })
            .to_string(),
        );
        let user_data = Box::into_raw(Box::new(9usize)) as *mut libc::c_void;

        unsafe {
            assert_status!(
                nemo_relay_register_plugin(
                    plugin_kind_c.as_ptr(),
                    validate_cb,
                    plugin_register_fail,
                    user_data,
                    Some(plugin_free),
                ),
                NemoRelayStatus::Ok
            );

            let mut report_json = ptr::null_mut();
            assert_status!(
                nemo_relay_validate_plugin_config(config.as_ptr(), &mut report_json),
                NemoRelayStatus::Ok
            );
            let report = returned_json(report_json);
            let diag = report["diagnostics"].as_array().unwrap();
            assert!(
                diag.iter().any(|value| {
                    value["code"] == "plugin.validate_failed"
                        && value["message"]
                            .as_str()
                            .is_some_and(|message| message.contains(expected_fragment))
                }),
                "missing expected plugin validation diagnostic: {expected_fragment}"
            );

            assert_status!(
                nemo_relay_deregister_plugin(plugin_kind_c.as_ptr()),
                NemoRelayStatus::Ok
            );
        }
    }

    assert_eq!(*lock_unpoisoned(plugin_frees()), 2);
}

#[test]
fn test_ffi_plugin_without_validate_callback_uses_registration_fallback_error() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let plugin_kind = unique_name("ffi_plugin_no_validate");
    let plugin_kind_c = cstring(&plugin_kind);
    let config = cstring(
        &json!({
            "version": 1,
            "components": [{
                "kind": plugin_kind,
                "enabled": true,
                "config": {}
            }]
        })
        .to_string(),
    );
    let user_data = Box::into_raw(Box::new(11usize)) as *mut libc::c_void;

    unsafe {
        assert_status!(
            nemo_relay_register_plugin(
                plugin_kind_c.as_ptr(),
                None,
                plugin_register_fail,
                user_data,
                Some(plugin_free),
            ),
            NemoRelayStatus::Ok
        );

        let mut report_json = ptr::null_mut();
        assert_status!(
            nemo_relay_validate_plugin_config(config.as_ptr(), &mut report_json),
            NemoRelayStatus::Ok
        );
        let report = returned_json(report_json);
        assert_eq!(report["diagnostics"], json!([]));

        let mut init_json = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_plugins(config.as_ptr(), &mut init_json),
            NemoRelayStatus::Internal
        );
        let err = read_last_error().expect("expected plugin registration failure message");
        assert!(err.contains("register callback failed with status Internal"));

        let mut active_json = ptr::null_mut();
        assert_status!(
            nemo_relay_active_plugin_report_json(&mut active_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(active_json), Json::Null);

        assert_status!(
            nemo_relay_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoRelayStatus::Ok
        );
    }

    assert_eq!(*lock_unpoisoned(plugin_frees()), 1);
}

#[test]
fn test_ffi_plugin_registration_failure_prefers_last_error_message() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_relay_clear_plugin_configuration();

    let plugin_kind = unique_name("ffi_plugin_last_error");
    let plugin_kind_c = cstring(&plugin_kind);
    let config = cstring(
        &json!({
            "version": 1,
            "components": [{
                "kind": plugin_kind,
                "enabled": true,
                "config": {}
            }]
        })
        .to_string(),
    );
    let user_data = Box::into_raw(Box::new(13usize)) as *mut libc::c_void;

    unsafe {
        assert_status!(
            nemo_relay_register_plugin(
                plugin_kind_c.as_ptr(),
                None,
                plugin_register_fail_with_last_error,
                user_data,
                Some(plugin_free),
            ),
            NemoRelayStatus::Ok
        );

        let mut init_json = ptr::null_mut();
        assert_status!(
            nemo_relay_initialize_plugins(config.as_ptr(), &mut init_json),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("plugin register callback set last error explicitly")
        );

        assert_status!(
            nemo_relay_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoRelayStatus::Ok
        );
    }

    assert_eq!(*lock_unpoisoned(plugin_frees()), 1);
}

#[test]
fn test_ffi_plugin_context_helpers_cover_null_and_success_paths() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();

    let name = cstring("registered");
    let llm_name = cstring("llm");
    let tool_name = cstring("tool");

    unsafe {
        assert_status!(
            nemo_relay_plugin_context_register_subscriber(
                ptr::null_mut(),
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_request_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_request_intercept(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_execution_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_execution_intercept(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
    }

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    unsafe {
        assert_status!(
            nemo_relay_plugin_context_register_subscriber(
                &mut ctx,
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_request_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_execution_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
    }

    let mut registrations = inner.into_registrations();
    let registered_names = registrations
        .iter()
        .map(|registration| registration.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        registered_names,
        vec![
            "ffi::registered",
            "ffi::tool",
            "ffi::tool",
            "ffi::tool",
            "ffi::llm",
            "ffi::llm",
            "ffi::llm",
            "ffi::llm",
            "ffi::tool",
            "ffi::llm",
            "ffi::llm",
            "ffi::tool",
        ]
    );
    nemo_relay::plugin::rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
}

#[test]
fn test_ffi_plugin_context_helpers_reject_duplicate_names() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let subscriber_name = cstring("duplicate-subscriber");
    let llm_name = cstring("duplicate-llm");
    let tool_name = cstring("duplicate-tool");

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    unsafe {
        assert_status!(
            nemo_relay_plugin_context_register_subscriber(
                &mut ctx,
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_subscriber(
                &mut ctx,
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                2,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_status!(
            nemo_relay_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                2,
                true,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_status!(
            nemo_relay_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                2,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );
    }
}

#[test]
fn test_ffi_plugin_context_helpers_reject_invalid_utf8_names_in_bulk() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid_name = invalid_utf8.as_ptr() as *const c_char;

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    macro_rules! assert_invalid_name_status {
        ($call:expr) => {{
            assert_status!($call, NemoRelayStatus::InvalidUtf8);
            assert!(read_last_error().unwrap_or_default().contains("utf-8"));
        }};
    }

    unsafe {
        assert_invalid_name_status!(nemo_relay_plugin_context_register_subscriber(
            &mut ctx,
            invalid_name,
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                invalid_name,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
                &mut ctx,
                invalid_name,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
                &mut ctx,
                invalid_name,
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
                &mut ctx,
                invalid_name,
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
                &mut ctx,
                invalid_name,
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
                &mut ctx,
                invalid_name,
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(nemo_relay_plugin_context_register_llm_request_intercept(
            &mut ctx,
            invalid_name,
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_name_status!(nemo_relay_plugin_context_register_tool_request_intercept(
            &mut ctx,
            invalid_name,
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_name_status!(nemo_relay_plugin_context_register_llm_execution_intercept(
            &mut ctx,
            invalid_name,
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_name_status!(
            nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                &mut ctx,
                invalid_name,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_name_status!(nemo_relay_plugin_context_register_tool_execution_intercept(
            &mut ctx,
            invalid_name,
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
    }

    let registrations = inner.into_registrations();
    assert!(registrations.is_empty());
}

#[test]
fn test_ffi_plugin_context_helpers_reject_duplicate_names_in_bulk() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    macro_rules! assert_duplicate {
        ($call:expr) => {{
            assert_status!($call, NemoRelayStatus::Internal);
            assert!(
                read_last_error()
                    .unwrap_or_default()
                    .contains("already exists")
            );
        }};
    }

    unsafe {
        let subscriber_name = cstring("duplicate-subscriber-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_subscriber(
                &mut ctx,
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(nemo_relay_plugin_context_register_subscriber(
            &mut ctx,
            subscriber_name.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));

        let tool_sanitize_req = cstring("duplicate-tool-sanitize-req-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_sanitize_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_sanitize_req.as_ptr(),
                2,
                tool_request_cb,
                ptr::null_mut(),
                None,
            )
        );

        let tool_sanitize_resp = cstring("duplicate-tool-sanitize-resp-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
                &mut ctx,
                tool_sanitize_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
                &mut ctx,
                tool_sanitize_resp.as_ptr(),
                2,
                tool_request_cb,
                ptr::null_mut(),
                None,
            )
        );

        let tool_conditional = cstring("duplicate-tool-conditional-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
                &mut ctx,
                tool_conditional.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
                &mut ctx,
                tool_conditional.as_ptr(),
                2,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );

        let llm_sanitize_req = cstring("duplicate-llm-sanitize-req-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
                &mut ctx,
                llm_sanitize_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
                &mut ctx,
                llm_sanitize_req.as_ptr(),
                2,
                llm_request_cb,
                ptr::null_mut(),
                None,
            )
        );

        let llm_sanitize_resp = cstring("duplicate-llm-sanitize-resp-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
                &mut ctx,
                llm_sanitize_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
                &mut ctx,
                llm_sanitize_resp.as_ptr(),
                2,
                llm_response_cb,
                ptr::null_mut(),
                None,
            )
        );

        let llm_conditional = cstring("duplicate-llm-conditional-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
                &mut ctx,
                llm_conditional.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
                &mut ctx,
                llm_conditional.as_ptr(),
                2,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );

        let llm_request = cstring("duplicate-llm-request-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_request.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(nemo_relay_plugin_context_register_llm_request_intercept(
            &mut ctx,
            llm_request.as_ptr(),
            2,
            true,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));

        let tool_request = cstring("duplicate-tool-request-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_tool_request_intercept(
                &mut ctx,
                tool_request.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(nemo_relay_plugin_context_register_tool_request_intercept(
            &mut ctx,
            tool_request.as_ptr(),
            2,
            true,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));

        let llm_exec = cstring("duplicate-llm-exec-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_execution_intercept(
                &mut ctx,
                llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(nemo_relay_plugin_context_register_llm_execution_intercept(
            &mut ctx,
            llm_exec.as_ptr(),
            2,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));

        let llm_stream_exec = cstring("duplicate-llm-stream-exec-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                &mut ctx,
                llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(
            nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                &mut ctx,
                llm_stream_exec.as_ptr(),
                2,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            )
        );

        let tool_exec = cstring("duplicate-tool-exec-bulk");
        assert_status!(
            nemo_relay_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_duplicate!(nemo_relay_plugin_context_register_tool_execution_intercept(
            &mut ctx,
            tool_exec.as_ptr(),
            2,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
    }

    let mut registrations = inner.into_registrations();
    nemo_relay::plugin::rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
}

#[test]
fn test_ffi_specialized_subscriber_and_exporter_default_and_invalid_name_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let invalid_utf8 = [0xffu8, 0];
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let endpoint = c"http://localhost:4318/v1/traces";

        let mut otel_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut otel_subscriber,
            ),
            NemoRelayStatus::Ok
        );
        let otel_name = cstring(&unique_name("ffi_otel_defaults"));
        assert_status!(
            nemo_relay_otel_subscriber_register(otel_subscriber, invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_otel_subscriber_register(otel_subscriber, otel_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_register(otel_subscriber, otel_name.as_ptr()),
            NemoRelayStatus::Internal
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(otel_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_force_flush(otel_subscriber),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_shutdown(otel_subscriber),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(otel_subscriber);

        let mut oi_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"openinference".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut oi_subscriber,
            ),
            NemoRelayStatus::Ok
        );
        let oi_name = cstring(&unique_name("ffi_oi_defaults"));
        assert_status!(
            nemo_relay_otel_subscriber_register(oi_subscriber, invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_otel_subscriber_register(oi_subscriber, oi_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_register(oi_subscriber, oi_name.as_ptr()),
            NemoRelayStatus::Internal
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_otel_subscriber_deregister(oi_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_force_flush(oi_subscriber),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_otel_subscriber_shutdown(oi_subscriber),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(oi_subscriber);

        let session = cstring("specialized-session");
        let agent = cstring("specialized-agent");
        let version = cstring("1.0.0");
        let mut exporter = ptr::null_mut();
        assert_status!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoRelayStatus::Ok
        );
        let exporter_name = cstring(&unique_name("ffi_exporter_defaults"));
        assert_status!(
            nemo_relay_atif_exporter_register(exporter, invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::AlreadyExists
        );
        assert_status!(
            nemo_relay_atif_exporter_deregister(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atif_exporter_deregister(invalid_name),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);
    }
}

#[test]
fn test_ffi_otel_projection_options_accept_and_validate_legacy_controls() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    unsafe {
        let endpoint = c"http://localhost:4318/v1/traces";
        let exclusions = c"[\"custom.mark\"]";
        let mappings = c"[{\"key\":\"nemo_relay.model_name\",\"alias\":\"model.alias\"}]";
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"tool".as_ptr(),
                exclusions.as_ptr(),
                mappings.as_ptr(),
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);

        let invalid_mappings = c"[{\"key\":\"\",\"alias\":\"model.alias\"}]";
        let mut invalid_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                invalid_mappings.as_ptr(),
                &mut invalid_subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(invalid_subscriber.is_null());
    }
}

#[test]
fn test_ffi_otel_projection_options_v2_accepts_and_validates_metadata_prefixes() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    unsafe {
        let endpoint = c"http://localhost:4318/v1/traces";
        let valid_prefixes = c"[\"nv.\",\"host_\"]";
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options_v2(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                ptr::null(),
                valid_prefixes.as_ptr(),
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);

        for (prefixes, expected_status) in [
            (c"[\"nv.*\"]".as_ptr(), NemoRelayStatus::InvalidArg),
            (c"{".as_ptr(), NemoRelayStatus::InvalidJson),
        ] {
            let mut invalid_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
            assert_status!(
                nemo_relay_otel_subscriber_create_with_projection_options_v2(
                    c"full".as_ptr(),
                    ptr::null(),
                    endpoint.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    0,
                    c"inherit".as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    prefixes,
                    &mut invalid_subscriber,
                ),
                expected_status
            );
            assert!(invalid_subscriber.is_null());
        }
    }
}

#[test]
fn test_ffi_otel_projection_options_v3_configures_completed_context_ttl() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    unsafe {
        let endpoint = c"http://localhost:4318/v1/traces";
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options_v3(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                750,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);

        let mut invalid_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options_v3(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut invalid_subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(invalid_subscriber.is_null());
    }
}

#[test]
fn test_ffi_otel_projection_options_v4_accepts_header_env_and_keeps_values_secret() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_globals();

    let variable = "NEMO_RELAY_FFI_OTEL_HEADER_V4";
    let secret = "relay-ffi-secret";
    let header_env_json = format!(r#"{{"authorization":"{variable}"}}"#);
    let header_env = cstring(&header_env_json);
    let endpoint = c"http://localhost:4318/v1/traces";
    unsafe {
        std::env::set_var(variable, "Bearer activation-value");
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options_v4(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                header_env.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                60_000,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);

        std::env::set_var(variable, format!("{secret}\ninvalid"));
        let mut invalid_subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create_with_projection_options_v4(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                header_env.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                c"inherit".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                60_000,
                &mut invalid_subscriber,
            ),
            NemoRelayStatus::Internal
        );
        assert!(invalid_subscriber.is_null());
        assert!(!read_last_error().unwrap_or_default().contains(secret));
        std::env::remove_var(variable);
    }
}

#[test]
fn test_ffi_specialized_constructor_invalid_utf8_and_malformed_json_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let malformed_json = cstring("{");
        let valid_headers = cstring(r#"{"authorization":"Bearer token"}"#);
        let valid_attrs = cstring(r#"{"deployment.environment":"test"}"#);
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let service = cstring("ffi-agent");
        let namespace = cstring("agents");
        let version = cstring("1.0.0");
        let scope = cstring("ffi-tests");
        let grpc = cstring("grpc");

        let mut otel = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                malformed_json.as_ptr(),
                valid_attrs.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
                1,
                &mut otel,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                valid_headers.as_ptr(),
                malformed_json.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
                1,
                &mut otel,
            ),
            NemoRelayStatus::InvalidJson
        );
        for (transport, endpoint_ptr, service_ptr, namespace_ptr, version_ptr, scope_ptr) in [
            (
                invalid,
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                invalid,
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                invalid,
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                invalid,
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                invalid,
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                invalid,
            ),
        ] {
            assert_status!(
                nemo_relay_otel_subscriber_create(
                    c"full".as_ptr(),
                    transport,
                    endpoint_ptr,
                    valid_headers.as_ptr(),
                    valid_attrs.as_ptr(),
                    service_ptr,
                    namespace_ptr,
                    version_ptr,
                    scope_ptr,
                    1,
                    &mut otel,
                ),
                NemoRelayStatus::InvalidUtf8
            );
        }

        let mut openinference = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"openinference".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                malformed_json.as_ptr(),
                valid_attrs.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
                1,
                &mut openinference,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"openinference".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                valid_headers.as_ptr(),
                malformed_json.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
                1,
                &mut openinference,
            ),
            NemoRelayStatus::InvalidJson
        );
        for (transport, endpoint_ptr, service_ptr, namespace_ptr, version_ptr, scope_ptr) in [
            (
                invalid,
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                invalid,
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                invalid,
                namespace.as_ptr(),
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                invalid,
                version.as_ptr(),
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                invalid,
                scope.as_ptr(),
            ),
            (
                grpc.as_ptr(),
                endpoint.as_ptr(),
                service.as_ptr(),
                namespace.as_ptr(),
                version.as_ptr(),
                invalid,
            ),
        ] {
            assert_status!(
                nemo_relay_otel_subscriber_create(
                    c"openinference".as_ptr(),
                    transport,
                    endpoint_ptr,
                    valid_headers.as_ptr(),
                    valid_attrs.as_ptr(),
                    service_ptr,
                    namespace_ptr,
                    version_ptr,
                    scope_ptr,
                    1,
                    &mut openinference,
                ),
                NemoRelayStatus::InvalidUtf8
            );
        }

        let session = cstring("specialized-session-invalid-utf8");
        let agent = cstring("specialized-agent-invalid-utf8");
        let agent_version = cstring("1.0.0");
        let mut exporter = ptr::null_mut();
        for (session_ptr, agent_ptr, version_ptr, model_ptr) in [
            (invalid, agent.as_ptr(), agent_version.as_ptr(), ptr::null()),
            (
                session.as_ptr(),
                invalid,
                agent_version.as_ptr(),
                ptr::null(),
            ),
            (session.as_ptr(), agent.as_ptr(), invalid, ptr::null()),
            (
                session.as_ptr(),
                agent.as_ptr(),
                agent_version.as_ptr(),
                invalid,
            ),
        ] {
            assert_status!(
                nemo_relay_atif_exporter_create(
                    session_ptr,
                    agent_ptr,
                    version_ptr,
                    model_ptr,
                    &mut exporter,
                ),
                NemoRelayStatus::InvalidUtf8
            );
        }

        let plugin_kind = invalid;
        assert_status!(
            nemo_relay_register_plugin(
                plugin_kind,
                None,
                plugin_register_fail,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_deregister_plugin(plugin_kind),
            NemoRelayStatus::InvalidUtf8
        );
    }
}
