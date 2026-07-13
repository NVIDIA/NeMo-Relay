// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for registry in the NeMo Relay FFI crate.

use super::*;
use nemo_relay::plugin::rollback_registrations;

unsafe extern "C" fn event_sanitize_cb(
    _user_data: *mut libc::c_void,
    event: *const FfiEvent,
    fields_json: *const c_char,
) -> *mut c_char {
    let name = unsafe { take_string(nemo_relay_event_name(event)) }.unwrap_or_default();
    let mut fields: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(fields_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    fields["data"] = json!({"sanitized_by": name});
    fields["category_profile"] = json!({"subtype": "ffi.sanitized"});
    fields["metadata"] = Json::Null;
    CString::new(fields.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn invalid_event_sanitize_cb(
    _user_data: *mut libc::c_void,
    _event: *const FfiEvent,
    _fields_json: *const c_char,
) -> *mut c_char {
    CString::new("not-json").unwrap().into_raw()
}

#[test]
fn test_ffi_event_sanitizer_registries_and_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = cstring(&unique_name("ffi_event_sanitize_subscriber"));
        assert_eq!(
            nemo_relay_register_subscriber(
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let mark_guard = cstring(&unique_name("ffi_mark_sanitize"));
        let start_guard = cstring(&unique_name("ffi_scope_start_sanitize"));
        let end_guard = cstring(&unique_name("ffi_scope_end_sanitize"));
        for status in [
            nemo_relay_register_mark_sanitize_guardrail(
                mark_guard.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(1usize)).cast(),
                Some(plugin_free),
            ),
            nemo_relay_register_scope_sanitize_start_guardrail(
                start_guard.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(2usize)).cast(),
                Some(plugin_free),
            ),
            nemo_relay_register_scope_sanitize_end_guardrail(
                end_guard.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(3usize)).cast(),
                Some(plugin_free),
            ),
        ] {
            assert_eq!(status, NemoRelayStatus::Ok);
        }

        let scope_name = cstring("ffi-global-scope");
        let original = cstring(r#"{"secret":true}"#);
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Custom,
                ptr::null(),
                0,
                original.as_ptr(),
                original.as_ptr(),
                ptr::null(),
                &mut scope,
            ),
            NemoRelayStatus::Ok
        );
        let mark_name = cstring("ffi-global-mark");
        assert_eq!(
            nemo_relay_event(
                mark_name.as_ptr(),
                scope,
                original.as_ptr(),
                original.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);

        let events = lock_unpoisoned(event_log());
        for name in ["ffi-global-scope", "ffi-global-mark"] {
            for event in events.iter().filter(|event| event["name"] == name) {
                assert_eq!(event["data"], json!({"sanitized_by": name}));
                assert_eq!(
                    event["json"]["category_profile"]["subtype"],
                    "ffi.sanitized"
                );
                assert_eq!(event["metadata"], Json::Null);
            }
        }
        for phase in ["start", "end"] {
            assert!(events.iter().any(|event| {
                event["name"] == "ffi-global-scope" && event["json"]["scope_category"] == phase
            }));
        }
        drop(events);

        assert_eq!(
            nemo_relay_deregister_mark_sanitize_guardrail(mark_guard.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_scope_sanitize_start_guardrail(start_guard.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_scope_sanitize_end_guardrail(end_guard.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(*lock_unpoisoned(plugin_frees()), 3);

        let invalid_guard = cstring(&unique_name("ffi_invalid_event_sanitize"));
        assert_eq!(
            nemo_relay_register_mark_sanitize_guardrail(
                invalid_guard.as_ptr(),
                1,
                invalid_event_sanitize_cb,
                Box::into_raw(Box::new(4usize)).cast(),
                Some(plugin_free),
            ),
            NemoRelayStatus::Ok
        );
        let invalid_mark = cstring("ffi-invalid-callback-mark");
        assert_eq!(
            nemo_relay_event(
                invalid_mark.as_ptr(),
                ptr::null(),
                original.as_ptr(),
                original.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_mark_sanitize_guardrail(invalid_guard.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(*lock_unpoisoned(plugin_frees()), 4);

        let mut owner = ptr::null_mut();
        let owner_name = cstring("ffi-event-owner");
        assert_eq!(
            nemo_relay_push_scope(
                owner_name.as_ptr(),
                NemoRelayScopeType::Agent,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut owner,
            ),
            NemoRelayStatus::Ok
        );
        let owner_uuid = cstring(&take_string(nemo_relay_scope_handle_uuid(owner)).unwrap());
        let local_mark = cstring(&unique_name("ffi_local_mark_sanitize"));
        let local_start = cstring(&unique_name("ffi_local_start_sanitize"));
        let local_end = cstring(&unique_name("ffi_local_end_sanitize"));
        for status in [
            nemo_relay_scope_register_mark_sanitize_guardrail(
                owner_uuid.as_ptr(),
                local_mark.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(5usize)).cast(),
                Some(plugin_free),
            ),
            nemo_relay_scope_register_scope_sanitize_start_guardrail(
                owner_uuid.as_ptr(),
                local_start.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(6usize)).cast(),
                Some(plugin_free),
            ),
            nemo_relay_scope_register_scope_sanitize_end_guardrail(
                owner_uuid.as_ptr(),
                local_end.as_ptr(),
                1,
                event_sanitize_cb,
                Box::into_raw(Box::new(7usize)).cast(),
                Some(plugin_free),
            ),
        ] {
            assert_eq!(status, NemoRelayStatus::Ok);
        }
        let child_name = cstring("ffi-local-child");
        let mut child = ptr::null_mut();
        assert_eq!(
            nemo_relay_push_scope(
                child_name.as_ptr(),
                NemoRelayScopeType::Function,
                ptr::null(),
                0,
                original.as_ptr(),
                original.as_ptr(),
                ptr::null(),
                &mut child,
            ),
            NemoRelayStatus::Ok
        );
        let local_mark_name = cstring("ffi-local-mark");
        assert_eq!(
            nemo_relay_event(
                local_mark_name.as_ptr(),
                child,
                original.as_ptr(),
                original.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_pop_scope(child, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(child);
        assert_eq!(
            nemo_relay_scope_deregister_mark_sanitize_guardrail(
                owner_uuid.as_ptr(),
                local_mark.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_deregister_scope_sanitize_start_guardrail(
                owner_uuid.as_ptr(),
                local_start.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_deregister_scope_sanitize_end_guardrail(
                owner_uuid.as_ptr(),
                local_end.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(*lock_unpoisoned(plugin_frees()), 7);

        let invalid_uuid = cstring("not-a-uuid");
        let invalid_name = cstring("invalid-scope-event-sanitizer");
        assert_eq!(
            nemo_relay_scope_register_mark_sanitize_guardrail(
                invalid_uuid.as_ptr(),
                invalid_name.as_ptr(),
                1,
                event_sanitize_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_scope_sanitize_start_guardrail(
                invalid_uuid.as_ptr(),
                invalid_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_register_scope_sanitize_end_guardrail(
                ptr::null(),
                1,
                event_sanitize_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_deregister_mark_sanitize_guardrail(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_register_scope_sanitize_end_guardrail(
                owner_uuid.as_ptr(),
                ptr::null(),
                1,
                event_sanitize_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_deregister_scope_sanitize_end_guardrail(
                owner_uuid.as_ptr(),
                ptr::null(),
            ),
            NemoRelayStatus::NullPointer
        );

        assert_eq!(
            nemo_relay_pop_scope(owner, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(owner);
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log());
        let invalid_callback_event = events
            .iter()
            .find(|event| event["name"] == "ffi-invalid-callback-mark")
            .expect("invalid callback mark should be delivered");
        assert_eq!(invalid_callback_event["data"], json!({"secret": true}));
        assert_eq!(invalid_callback_event["metadata"], json!({"secret": true}));
        for name in ["ffi-local-child", "ffi-local-mark"] {
            for event in events.iter().filter(|event| event["name"] == name) {
                assert_eq!(event["data"], json!({"sanitized_by": name}));
                assert_eq!(
                    event["json"]["category_profile"]["subtype"],
                    "ffi.sanitized"
                );
                assert_eq!(event["metadata"], Json::Null);
            }
        }
        for phase in ["start", "end"] {
            assert!(events.iter().any(|event| {
                event["name"] == "ffi-local-child" && event["json"]["scope_category"] == phase
            }));
        }
        drop(events);

        let mut context = PluginRegistrationContext::with_namespace(unique_name("ffi_plugin_"));
        let mut ffi_context = FfiPluginContext(&mut context);
        for (name, status) in [
            (
                cstring("mark"),
                nemo_relay_plugin_context_register_mark_sanitize_guardrail(
                    &mut ffi_context,
                    cstring("mark").as_ptr(),
                    1,
                    event_sanitize_cb,
                    Box::into_raw(Box::new(8usize)).cast(),
                    Some(plugin_free),
                ),
            ),
            (
                cstring("start"),
                nemo_relay_plugin_context_register_scope_sanitize_start_guardrail(
                    &mut ffi_context,
                    cstring("start").as_ptr(),
                    1,
                    event_sanitize_cb,
                    Box::into_raw(Box::new(9usize)).cast(),
                    Some(plugin_free),
                ),
            ),
            (
                cstring("end"),
                nemo_relay_plugin_context_register_scope_sanitize_end_guardrail(
                    &mut ffi_context,
                    cstring("end").as_ptr(),
                    1,
                    event_sanitize_cb,
                    Box::into_raw(Box::new(10usize)).cast(),
                    Some(plugin_free),
                ),
            ),
        ] {
            assert!(!name.as_bytes().is_empty());
            assert_eq!(status, NemoRelayStatus::Ok);
        }
        assert_eq!(
            nemo_relay_plugin_context_register_mark_sanitize_guardrail(
                ptr::null_mut(),
                invalid_name.as_ptr(),
                1,
                event_sanitize_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_plugin_context_register_mark_sanitize_guardrail(
                &mut ffi_context,
                ptr::null(),
                1,
                event_sanitize_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        let mut registrations = context.into_registrations();
        rollback_registrations(&mut registrations);
        assert_eq!(*lock_unpoisoned(plugin_frees()), 10);

        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_open_telemetry_subscriber_lifecycle_and_errors() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let headers = cstring(r#"{"authorization":"Bearer token"}"#);
        let resource_attributes = cstring(r#"{"deployment.environment":"test"}"#);
        let service_name = cstring("ffi-agent");
        let service_namespace = cstring("agents");
        let service_version = cstring("1.0.0");
        let instrumentation_scope = cstring("ffi-tests");
        let invalid_transport = cstring("invalid");
        let grpc_transport = cstring("grpc");
        let invalid_headers = cstring(r#"{"authorization":1}"#);
        let invalid_resource_attributes = cstring(r#"["not-an-object"]"#);

        assert_eq!(
            nemo_relay_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                invalid_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                invalid_headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                invalid_resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                grpc_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        assert!(!subscriber.is_null());
        nemo_relay_otel_subscriber_free(subscriber);
        subscriber = ptr::null_mut();
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        assert!(!subscriber.is_null());

        let name = cstring(&unique_name("ffi_otel"));
        assert_eq!(
            nemo_relay_otel_subscriber_register(ptr::null(), name.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_otel_subscriber_force_flush(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_otel_subscriber_shutdown(ptr::null()),
            NemoRelayStatus::NullPointer
        );

        assert_eq!(
            nemo_relay_otel_subscriber_register(subscriber, name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_otel_subscriber_deregister(name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_otel_subscriber_deregister(name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_otel_subscriber_force_flush(subscriber),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_otel_subscriber_shutdown(subscriber),
            NemoRelayStatus::Ok
        );
        nemo_relay_otel_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_open_inference_subscriber_lifecycle_and_errors() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut subscriber: *mut FfiOpenInferenceSubscriber = ptr::null_mut();
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let headers = cstring(r#"{"authorization":"Bearer token"}"#);
        let resource_attributes = cstring(r#"{"deployment.environment":"test"}"#);
        let service_name = cstring("ffi-agent");
        let service_namespace = cstring("agents");
        let service_version = cstring("1.0.0");
        let instrumentation_scope = cstring("ffi-tests");
        let invalid_transport = cstring("invalid");
        let grpc_transport = cstring("grpc");
        let invalid_headers = cstring(r#"{"authorization":1}"#);
        let invalid_resource_attributes = cstring(r#"["not-an-object"]"#);

        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                invalid_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                invalid_headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                invalid_resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                grpc_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        assert!(!subscriber.is_null());
        nemo_relay_openinference_subscriber_free(subscriber);
        subscriber = ptr::null_mut();
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
        assert!(!subscriber.is_null());

        let name = cstring(&unique_name("ffi_openinference"));
        assert_eq!(
            nemo_relay_openinference_subscriber_register(ptr::null(), name.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_force_flush(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_shutdown(ptr::null()),
            NemoRelayStatus::NullPointer
        );

        assert_eq!(
            nemo_relay_openinference_subscriber_register(subscriber, name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_deregister(name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_deregister(name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_force_flush(subscriber),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_openinference_subscriber_shutdown(subscriber),
            NemoRelayStatus::Ok
        );
        nemo_relay_openinference_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_helper_rejection_and_null_name_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let args = cstring(r#"{"value": 7}"#);
        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let invalid_json = cstring("{");
        let tool_name = cstring("tool");
        let llm_name = cstring("llm");
        let mut tool_out = ptr::null_mut();
        let mut llm_error_out = ptr::null_mut();

        assert_eq!(
            nemo_relay_tool_request_intercepts(tool_name.as_ptr(), args.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out pointer is null")
        );
        assert_eq!(
            nemo_relay_tool_request_intercepts(ptr::null(), args.as_ptr(), &mut tool_out),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_tool_request_intercepts(
                tool_name.as_ptr(),
                invalid_json.as_ptr(),
                &mut tool_out
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(tool_out.is_null());
        assert_eq!(
            nemo_relay_tool_conditional_execution(ptr::null(), args.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_tool_conditional_execution(tool_name.as_ptr(), invalid_json.as_ptr()),
            NemoRelayStatus::InvalidJson
        );

        let tool_guard = cstring(&unique_name("ffi_tool_reject"));
        assert_eq!(
            nemo_relay_register_tool_conditional_execution_guardrail(
                tool_guard.as_ptr(),
                1,
                tool_reject_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
            NemoRelayStatus::GuardrailRejected
        );
        assert_eq!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(tool_guard.as_ptr()),
            NemoRelayStatus::Ok
        );

        let mut llm_out = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_request_intercepts(ptr::null(), request.as_ptr(), &mut llm_out),
            NemoRelayStatus::Ok
        );
        let llm_json = returned_json(llm_out);
        assert_eq!(llm_json["request"]["content"]["model"], json!("ffi-model"));

        assert_eq!(
            nemo_relay_llm_request_intercepts(llm_name.as_ptr(), request.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out pointer is null")
        );
        assert_eq!(
            nemo_relay_llm_request_intercepts(
                llm_name.as_ptr(),
                invalid_json.as_ptr(),
                &mut llm_error_out
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(llm_error_out.is_null());
        assert_eq!(
            nemo_relay_llm_conditional_execution(invalid_json.as_ptr()),
            NemoRelayStatus::InvalidJson
        );

        let llm_guard = cstring(&unique_name("ffi_llm_reject"));
        assert_eq!(
            nemo_relay_register_llm_conditional_execution_guardrail(
                llm_guard.as_ptr(),
                1,
                llm_reject_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_llm_conditional_execution(request.as_ptr()),
            NemoRelayStatus::GuardrailRejected
        );
        assert_eq!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(llm_guard.as_ptr()),
            NemoRelayStatus::Ok
        );

        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_registration_name_and_uuid_error_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_invalid_arg {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoRelayStatus::InvalidArg);
        };
    }
    macro_rules! assert_null_pointer {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoRelayStatus::NullPointer);
        };
    }

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_error_sweep_scope");
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

        let valid_scope_uuid = cstring(&take_string(nemo_relay_scope_handle_uuid(scope)).unwrap());
        let invalid_scope_uuid = cstring("not-a-uuid");

        assert_null_pointer!(nemo_relay_register_tool_sanitize_request_guardrail(
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_tool_sanitize_request_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_tool_sanitize_response_guardrail(
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_tool_sanitize_response_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_tool_conditional_execution_guardrail(
            ptr::null(),
            1,
            tool_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_tool_conditional_execution_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_tool_request_intercept(
            ptr::null(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_tool_request_intercept(ptr::null()));
        assert_null_pointer!(nemo_relay_register_tool_execution_intercept(
            ptr::null(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_tool_execution_intercept(ptr::null()));
        assert_null_pointer!(nemo_relay_register_llm_sanitize_request_guardrail(
            ptr::null(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_sanitize_request_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_llm_sanitize_response_guardrail(
            ptr::null(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_sanitize_response_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_llm_conditional_execution_guardrail(
            ptr::null(),
            1,
            llm_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_conditional_execution_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_llm_request_intercept(
            ptr::null(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_request_intercept(ptr::null()));
        assert_null_pointer!(nemo_relay_register_llm_execution_intercept(
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_execution_intercept(ptr::null()));
        assert_null_pointer!(nemo_relay_register_llm_stream_execution_intercept(
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_llm_stream_execution_intercept(
            ptr::null()
        ));
        assert_null_pointer!(nemo_relay_register_subscriber(
            ptr::null(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_deregister_subscriber(ptr::null()));

        assert_invalid_arg!(nemo_relay_scope_register_tool_sanitize_request_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_relay_scope_register_tool_sanitize_response_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            )
        );
        assert_invalid_arg!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_arg!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            )
        );
        assert_null_pointer!(nemo_relay_scope_register_tool_request_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_scope_deregister_tool_request_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_relay_scope_register_tool_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_relay_scope_deregister_tool_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_relay_scope_register_llm_sanitize_request_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_relay_scope_register_llm_sanitize_response_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_null_pointer!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            )
        );
        assert_invalid_arg!(nemo_relay_scope_register_llm_request_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_relay_scope_deregister_llm_request_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_relay_scope_register_llm_execution_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_scope_deregister_llm_execution_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_relay_scope_register_llm_stream_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_relay_scope_deregister_llm_stream_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_relay_scope_register_subscriber(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_relay_scope_deregister_subscriber(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));

        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_duplicate_registration_sweep_and_helper_callbacks() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_already_exists {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoRelayStatus::AlreadyExists);
        };
    }

    unsafe extern "C" fn tool_next_passthrough(
        _args_json: *const c_char,
        _next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        CString::new(r#"{"next":true}"#).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_next_passthrough(
        _native_json: *const c_char,
        _next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        CString::new(r#"{"role":"assistant","content":"next","tool_calls":[]}"#)
            .unwrap()
            .into_raw()
    }

    unsafe {
        clear_last_error();
        assert!(read_last_error().is_none());

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_duplicate_scope");
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

        let tool_cond = cstring(&unique_name("dup_tool_cond"));
        assert_eq!(
            nemo_relay_register_tool_conditional_execution_guardrail(
                tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_tool_conditional_execution_guardrail(
            tool_cond.as_ptr(),
            1,
            tool_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(tool_cond.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_req = cstring(&unique_name("dup_tool_req"));
        assert_eq!(
            nemo_relay_register_tool_request_intercept(
                tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_tool_request_intercept(
            tool_req.as_ptr(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_deregister_tool_request_intercept(tool_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_san_resp = cstring(&unique_name("dup_llm_san_resp"));
        assert_eq!(
            nemo_relay_register_llm_sanitize_response_guardrail(
                llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_sanitize_response_guardrail(
            llm_san_resp.as_ptr(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(llm_san_resp.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_cond = cstring(&unique_name("dup_llm_cond"));
        assert_eq!(
            nemo_relay_register_llm_conditional_execution_guardrail(
                llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_conditional_execution_guardrail(
            llm_cond.as_ptr(),
            1,
            llm_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(llm_cond.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_req = cstring(&unique_name("dup_llm_req"));
        assert_eq!(
            nemo_relay_register_llm_request_intercept(
                llm_req.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_request_intercept(
            llm_req.as_ptr(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_deregister_llm_request_intercept(llm_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let subscriber = cstring(&unique_name("dup_subscriber"));
        assert_eq!(
            nemo_relay_register_subscriber(
                subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_subscriber(
            subscriber.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber.as_ptr()),
            NemoRelayStatus::Ok
        );

        let scope_tool_cond = cstring(&unique_name("dup_scope_tool_cond"));
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
        assert_already_exists!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_tool_req = cstring(&unique_name("dup_scope_tool_req"));
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
        assert_already_exists!(nemo_relay_scope_register_tool_request_intercept(
            scope_uuid.as_ptr(),
            scope_tool_req.as_ptr(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_cond = cstring(&unique_name("dup_scope_llm_cond"));
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
        assert_already_exists!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_llm_req = cstring(&unique_name("dup_scope_llm_req"));
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
        assert_already_exists!(nemo_relay_scope_register_llm_request_intercept(
            scope_uuid.as_ptr(),
            scope_llm_req.as_ptr(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let scope_subscriber = cstring(&unique_name("dup_scope_subscriber"));
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
        assert_already_exists!(nemo_relay_scope_register_subscriber(
            scope_uuid.as_ptr(),
            scope_subscriber.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoRelayStatus::Ok
        );

        let session = cstring("dup-session");
        let agent = cstring("dup-agent");
        let version = cstring("1.0.0");
        let mut exporter = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_create(
                ptr::null(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                ptr::null(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut exporter,
            ),
            NemoRelayStatus::NullPointer
        );
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
            nemo_relay_atif_exporter_register(exporter, ptr::null()),
            NemoRelayStatus::NullPointer
        );
        let exporter_name = cstring(&unique_name("dup_exporter_subscriber"));
        assert_eq!(
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_atif_exporter_register(
            exporter,
            exporter_name.as_ptr(),
        ));
        assert_eq!(
            nemo_relay_atif_exporter_deregister(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);

        let args = cstring(r#"{"value":1}"#);
        let tool_intercept_json = take_string(tool_exec_intercept_cb(
            ptr::null_mut(),
            args.as_ptr(),
            tool_next_passthrough,
            ptr::null_mut(),
        ))
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&tool_intercept_json).unwrap(),
            json!({"result": {"next": true}, "pending_marks": []})
        );

        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_intercept_json = take_string(llm_exec_intercept_cb(
            ptr::null_mut(),
            request.as_ptr(),
            llm_next_passthrough,
            ptr::null_mut(),
        ))
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&llm_intercept_json).unwrap(),
            json!({"role":"assistant","content":"next","tool_calls":[]})
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
fn test_ffi_registration_table_sweep_for_remaining_wrappers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_global_guardrail_sweep {
        ($prefix:literal, $register:ident, $deregister:ident, $cb:expr) => {{
            let name = cstring(&unique_name($prefix));
            assert_eq!(
                $register(name.as_ptr(), 1, $cb, ptr::null_mut(), None),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $register(name.as_ptr(), 1, $cb, ptr::null_mut(), None),
                NemoRelayStatus::AlreadyExists
            );
            assert_eq!($deregister(name.as_ptr()), NemoRelayStatus::Ok);
            assert_eq!($deregister(name.as_ptr()), NemoRelayStatus::Ok);
        }};
    }

    macro_rules! assert_global_execution_sweep {
        ($prefix:literal, $register:ident, $deregister:ident, $cb:expr) => {{
            let name = cstring(&unique_name($prefix));
            assert_eq!(
                $register(name.as_ptr(), 1, $cb, ptr::null_mut(), None),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $register(name.as_ptr(), 1, $cb, ptr::null_mut(), None),
                NemoRelayStatus::AlreadyExists
            );
            assert_eq!($deregister(name.as_ptr()), NemoRelayStatus::Ok);
            assert_eq!($deregister(name.as_ptr()), NemoRelayStatus::Ok);
        }};
    }

    macro_rules! assert_scope_guardrail_sweep {
        ($scope_uuid:expr, $prefix:literal, $register:ident, $deregister:ident, $cb:expr) => {{
            let name = cstring(&unique_name($prefix));
            assert_eq!(
                $register(
                    $scope_uuid.as_ptr(),
                    name.as_ptr(),
                    1,
                    $cb,
                    ptr::null_mut(),
                    None,
                ),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $register(
                    $scope_uuid.as_ptr(),
                    name.as_ptr(),
                    1,
                    $cb,
                    ptr::null_mut(),
                    None,
                ),
                NemoRelayStatus::AlreadyExists
            );
            assert_eq!(
                $deregister($scope_uuid.as_ptr(), name.as_ptr()),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $deregister($scope_uuid.as_ptr(), name.as_ptr()),
                NemoRelayStatus::Ok
            );
        }};
    }

    macro_rules! assert_scope_execution_sweep {
        ($scope_uuid:expr, $prefix:literal, $register:ident, $deregister:ident, $cb:expr) => {{
            let name = cstring(&unique_name($prefix));
            assert_eq!(
                $register(
                    $scope_uuid.as_ptr(),
                    name.as_ptr(),
                    1,
                    $cb,
                    ptr::null_mut(),
                    None,
                ),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $register(
                    $scope_uuid.as_ptr(),
                    name.as_ptr(),
                    1,
                    $cb,
                    ptr::null_mut(),
                    None,
                ),
                NemoRelayStatus::AlreadyExists
            );
            assert_eq!(
                $deregister($scope_uuid.as_ptr(), name.as_ptr()),
                NemoRelayStatus::Ok
            );
            assert_eq!(
                $deregister($scope_uuid.as_ptr(), name.as_ptr()),
                NemoRelayStatus::Ok
            );
        }};
    }

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_table_sweep_scope");
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

        assert_global_guardrail_sweep!(
            "table_tool_san_resp",
            nemo_relay_register_tool_sanitize_response_guardrail,
            nemo_relay_deregister_tool_sanitize_response_guardrail,
            tool_request_cb
        );
        assert_global_execution_sweep!(
            "table_tool_exec",
            nemo_relay_register_tool_execution_intercept,
            nemo_relay_deregister_tool_execution_intercept,
            tool_exec_intercept_cb
        );
        assert_global_guardrail_sweep!(
            "table_llm_san_req",
            nemo_relay_register_llm_sanitize_request_guardrail,
            nemo_relay_deregister_llm_sanitize_request_guardrail,
            llm_request_cb
        );
        assert_global_execution_sweep!(
            "table_llm_exec",
            nemo_relay_register_llm_execution_intercept,
            nemo_relay_deregister_llm_execution_intercept,
            llm_exec_intercept_cb
        );
        assert_global_execution_sweep!(
            "table_llm_stream_exec",
            nemo_relay_register_llm_stream_execution_intercept,
            nemo_relay_deregister_llm_stream_execution_intercept,
            llm_exec_intercept_cb
        );

        assert_scope_guardrail_sweep!(
            scope_uuid,
            "table_scope_tool_san_resp",
            nemo_relay_scope_register_tool_sanitize_response_guardrail,
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail,
            tool_request_cb
        );
        assert_scope_execution_sweep!(
            scope_uuid,
            "table_scope_tool_exec",
            nemo_relay_scope_register_tool_execution_intercept,
            nemo_relay_scope_deregister_tool_execution_intercept,
            tool_exec_intercept_cb
        );
        assert_scope_guardrail_sweep!(
            scope_uuid,
            "table_scope_llm_san_req",
            nemo_relay_scope_register_llm_sanitize_request_guardrail,
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail,
            llm_request_cb
        );
        assert_scope_guardrail_sweep!(
            scope_uuid,
            "table_scope_llm_san_resp",
            nemo_relay_scope_register_llm_sanitize_response_guardrail,
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail,
            llm_response_cb
        );
        assert_scope_execution_sweep!(
            scope_uuid,
            "table_scope_llm_exec",
            nemo_relay_scope_register_llm_execution_intercept,
            nemo_relay_scope_deregister_llm_execution_intercept,
            llm_exec_intercept_cb
        );
        assert_scope_execution_sweep!(
            scope_uuid,
            "table_scope_llm_stream_exec",
            nemo_relay_scope_register_llm_stream_execution_intercept,
            nemo_relay_scope_deregister_llm_stream_execution_intercept,
            llm_exec_intercept_cb
        );

        let mut exporter = ptr::null_mut();
        let session = cstring("table-sweep-session");
        let agent = cstring("table-sweep-agent");
        let version = cstring("1.0.0");
        let exporter_name = cstring(&unique_name("table_exporter_subscriber"));
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
            nemo_relay_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);

        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_execute_stream_and_atif_exporter() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();

        let subscriber_name = unique_name("ffi_llm_subscriber");
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

        let mut root = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut root), NemoRelayStatus::Ok);
        nemo_relay_scope_handle_free(root);

        let intercept_name = unique_name("ffi_llm_intercept");
        let intercept_name_c = cstring(&intercept_name);
        assert_eq!(
            nemo_relay_register_llm_request_intercept(
                intercept_name_c.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let conditional_name = unique_name("ffi_llm_conditional");
        let conditional_name_c = cstring(&conditional_name);
        assert_eq!(
            nemo_relay_register_llm_conditional_execution_guardrail(
                conditional_name_c.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let sanitize_name = unique_name("ffi_llm_sanitize");
        let sanitize_name_c = cstring(&sanitize_name);
        assert_eq!(
            nemo_relay_register_llm_sanitize_response_guardrail(
                sanitize_name_c.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );

        let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
        let session = cstring("ffi-session");
        let agent = cstring("ffi-agent");
        let version = cstring("1.0.0");
        let model_name = cstring("ffi-model");
        assert_eq!(
            nemo_relay_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                model_name.as_ptr(),
                &mut exporter,
            ),
            NemoRelayStatus::Ok
        );

        let exporter_sub = unique_name("ffi_exporter");
        let exporter_sub_c = cstring(&exporter_sub);
        assert_eq!(
            nemo_relay_atif_exporter_register(exporter, exporter_sub_c.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_name = cstring("ffi_llm");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let headers = cstring(r#"{"Authorization":"Bearer token"}"#);
        let content = cstring(r#"{"messages":[],"model":"ffi-model"}"#);
        let llm_request = nemo_relay_llm_request_new(headers.as_ptr(), content.as_ptr());
        assert!(!llm_request.is_null());
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_relay_llm_request_headers(llm_request)).unwrap()
            )
            .unwrap(),
            json!({"Authorization": "Bearer token"})
        );
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_relay_llm_request_content(llm_request)).unwrap()
            )
            .unwrap(),
            json!({"messages": [], "model": "ffi-model"})
        );
        nemo_relay_llm_request_free(llm_request);

        let mut helper_out = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_request_intercepts(llm_name.as_ptr(), request.as_ptr(), &mut helper_out),
            NemoRelayStatus::Ok
        );
        let helper_json = returned_json(helper_out);
        assert_eq!(
            helper_json["request"]["content"]["intercepted"],
            json!(true)
        );

        assert_eq!(
            nemo_relay_llm_conditional_execution(request.as_ptr()),
            NemoRelayStatus::Ok
        );

        let mut handle: *mut FfiLLMHandle = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                2,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_llm_handle_uuid(handle)).is_some());
        assert_eq!(
            take_string(nemo_relay_llm_handle_name(handle)).unwrap(),
            "ffi_llm"
        );
        assert_eq!(nemo_relay_llm_handle_attributes(handle), 2);
        assert!(take_string(nemo_relay_llm_handle_parent_uuid(handle)).is_some());

        let response = cstring(r#"{"content":"manual end","role":"assistant","tool_calls":[]}"#);
        assert_eq!(
            nemo_relay_llm_call_end(handle, response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_llm_handle_free(handle);

        let mut execute_out = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_call_execute(
                llm_name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut execute_out,
            ),
            NemoRelayStatus::Ok
        );
        let execute_json = returned_json(execute_out);
        assert_eq!(execute_json["content"], json!("hello from ffi"));
        assert_eq!(execute_json["model_seen"], json!("ffi-model"));
        assert_eq!(nemo_relay_flush_subscribers(), NemoRelayStatus::Ok);
        let events = lock_unpoisoned(event_log()).clone();
        assert!(
            events
                .iter()
                .any(|event| event["output"]["sanitized"] == json!(true))
        );
        assert!(
            events
                .iter()
                .any(|event| event["model_name"] == "ffi-model")
        );

        let mut stream = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_stream_call_execute(
                llm_name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                Some(collector_cb),
                Some(finalizer_cb),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut stream,
            ),
            NemoRelayStatus::Ok
        );
        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 1);
        let chunk_json = returned_json(chunk);
        assert_eq!(chunk_json["content"], json!("hello from ffi"));
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 0);
        nemo_relay_stream_free(stream);

        assert_eq!(lock_unpoisoned(collected_chunks()).len(), 1);
        assert_eq!(*lock_unpoisoned(finalizer_calls()), 1);

        let mut exported = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_export(exporter, &mut exported),
            NemoRelayStatus::Ok
        );
        let trajectory = returned_json(exported);
        assert_eq!(trajectory["schema_version"], json!("ATIF-v1.7"));
        assert!(trajectory["steps"].as_array().unwrap().len() >= 4);

        assert_eq!(
            nemo_relay_atif_exporter_clear(exporter),
            NemoRelayStatus::Ok
        );
        let mut cleared = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_export(exporter, &mut cleared),
            NemoRelayStatus::Ok
        );
        let cleared_json = returned_json(cleared);
        assert_eq!(cleared_json["steps"].as_array().unwrap().len(), 0);

        assert_eq!(
            nemo_relay_atif_exporter_deregister(exporter_sub_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_atif_exporter_free(exporter);
        assert_eq!(
            nemo_relay_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );

        assert_eq!(
            nemo_relay_deregister_llm_request_intercept(intercept_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(conditional_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(sanitize_name_c.as_ptr()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}
