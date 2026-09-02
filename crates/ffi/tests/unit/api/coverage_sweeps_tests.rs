// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for coverage sweeps in the NeMo Relay FFI crate.

use super::ffi_coverage_support::{ValidationFixtures, assert_otel_signal_lifecycle};
use super::*;

const RUNTIME_OWNER_ENV: &str = "NEMO_RELAY_RUNTIME_OWNER";
const BINDING_KIND_ENV: &str = "NEMO_RELAY_BINDING_KIND";

struct EnvGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, original }
    }

    fn remove(key: &'static str) -> Self {
        let original = std::env::var_os(key);
        unsafe { std::env::remove_var(key) };
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

#[test]
fn test_ffi_additional_duplicate_registration_sweeps_for_missing_global_wrappers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_already_exists {
        ($expr:expr) => {
            assert_status!($expr, NemoRelayStatus::AlreadyExists);
        };
    }

    unsafe {
        let tool_san_req = cstring(&unique_name("dup_tool_san_req_extra"));
        assert_status!(
            nemo_relay_register_tool_sanitize_request_guardrail(
                tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_tool_sanitize_request_guardrail(
            tool_san_req.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_san_resp = cstring(&unique_name("dup_tool_san_resp_extra"));
        assert_status!(
            nemo_relay_register_tool_sanitize_response_guardrail(
                tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_tool_sanitize_response_guardrail(
            tool_san_resp.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(tool_san_resp.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_exec = cstring(&unique_name("dup_tool_exec_extra"));
        assert_status!(
            nemo_relay_register_tool_execution_intercept(
                tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_tool_execution_intercept(
            tool_exec.as_ptr(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(tool_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_san_req = cstring(&unique_name("dup_llm_san_req_extra"));
        assert_status!(
            nemo_relay_register_llm_sanitize_request_guardrail(
                llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_sanitize_request_guardrail(
            llm_san_req.as_ptr(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(llm_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_exec = cstring(&unique_name("dup_llm_exec_extra"));
        assert_status!(
            nemo_relay_register_llm_execution_intercept(
                llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_execution_intercept(
            llm_exec.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_llm_execution_intercept(llm_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_stream_exec = cstring(&unique_name("dup_llm_stream_exec_extra"));
        assert_status!(
            nemo_relay_register_llm_stream_execution_intercept(
                llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_register_llm_stream_execution_intercept(
            llm_stream_exec.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_deregister_llm_stream_execution_intercept(llm_stream_exec.as_ptr()),
            NemoRelayStatus::Ok
        );
    }
}

#[test]
fn test_ffi_runtime_owner_conflict_and_llm_shape_error_sweeps() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);
        let scope_uuid = cstring(
            &take_string(nemo_relay_scope_handle_uuid(parent)).expect("scope uuid should exist"),
        );

        let tool_name = cstring("ffi_runtime_owner_tool");
        let tool_args = cstring(r#"{"value":1}"#);
        let tool_result = cstring(r#"{"result":{"ok":true}}"#);
        let llm_name = cstring("ffi_runtime_owner_llm");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);

        let mut tool_handle = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut tool_handle,
            ),
            NemoRelayStatus::Ok
        );

        let mut llm_handle = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::Ok
        );

        let malformed_request = cstring(r#"{"headers":[],"content":"bad"}"#);
        let mut transformed_out = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_request_intercepts(
                llm_name.as_ptr(),
                malformed_request.as_ptr(),
                &mut transformed_out,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );
        assert_status!(
            nemo_relay_llm_conditional_execution(malformed_request.as_ptr()),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        let major = env!("CARGO_PKG_VERSION").split('.').next().unwrap_or("0");
        let conflict_token = format!(
            "pid={};binding=ffi-conflict;version={major}",
            std::process::id()
        );
        let _binding_guard = EnvGuard::remove(BINDING_KIND_ENV);
        let _owner_guard = EnvGuard::set(RUNTIME_OWNER_ENV, &conflict_token);

        let conflict_fragment = "multiple bindings in one process";

        let mut out_json = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_request_intercepts(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                &mut out_json
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        assert_status!(
            nemo_relay_llm_request_intercepts(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                &mut out_json
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        assert_status!(
            nemo_relay_llm_conditional_execution(llm_request.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let mut conflict_scope = ptr::null_mut();
        assert_status!(
            nemo_relay_get_handle(&mut conflict_scope),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );
        assert_status!(
            nemo_relay_push_scope(
                tool_name.as_ptr(),
                NemoRelayScopeType::Function,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut conflict_scope,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );
        assert_status!(
            nemo_relay_event(tool_name.as_ptr(), parent, ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let mut conflict_tool_handle = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut conflict_tool_handle,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );
        assert_status!(
            nemo_relay_tool_call_end(tool_handle, tool_result.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        assert_status!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );
        assert_status!(
            nemo_relay_llm_call_end(llm_handle, llm_response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let global_name = cstring("conflict-global");
        assert_status!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_tool_request_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_tool_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_request_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_llm_stream_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_deregister_subscriber(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );

        let scope_name = cstring("conflict-scope");
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr()
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );

        nemo_relay_tool_handle_free(tool_handle);
        nemo_relay_llm_handle_free(llm_handle);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_additional_duplicate_registration_sweeps_for_missing_scope_wrappers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_already_exists {
        ($expr:expr) => {
            assert_status!($expr, NemoRelayStatus::AlreadyExists);
        };
    }

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("dup_scope_extra");
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

        let tool_san_req = cstring(&unique_name("dup_scope_tool_san_req_extra"));
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_tool_sanitize_request_guardrail(
            scope_uuid.as_ptr(),
            tool_san_req.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                tool_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let tool_san_resp = cstring(&unique_name("dup_scope_tool_san_resp_extra"));
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_tool_sanitize_response_guardrail(
            scope_uuid.as_ptr(),
            tool_san_resp.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                tool_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let tool_exec = cstring(&unique_name("dup_scope_tool_exec_extra"));
        assert_status!(
            nemo_relay_scope_register_tool_execution_intercept(
                scope_uuid.as_ptr(),
                tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_tool_execution_intercept(
            scope_uuid.as_ptr(),
            tool_exec.as_ptr(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                tool_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_san_req = cstring(&unique_name("dup_scope_llm_san_req_extra"));
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_llm_sanitize_request_guardrail(
            scope_uuid.as_ptr(),
            llm_san_req.as_ptr(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                llm_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_san_resp = cstring(&unique_name("dup_scope_llm_san_resp_extra"));
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_llm_sanitize_response_guardrail(
            scope_uuid.as_ptr(),
            llm_san_resp.as_ptr(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                llm_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_exec = cstring(&unique_name("dup_scope_llm_exec_extra"));
        assert_status!(
            nemo_relay_scope_register_llm_execution_intercept(
                scope_uuid.as_ptr(),
                llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_llm_execution_intercept(
            scope_uuid.as_ptr(),
            llm_exec.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                llm_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_stream_exec = cstring(&unique_name("dup_scope_llm_stream_exec_extra"));
        assert_status!(
            nemo_relay_scope_register_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_already_exists!(nemo_relay_scope_register_llm_stream_execution_intercept(
            scope_uuid.as_ptr(),
            llm_stream_exec.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                llm_stream_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
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
fn test_ffi_global_tool_registration_invalid_utf8_name_sweep() {
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
    }
}

#[test]
fn test_ffi_global_llm_and_subscriber_registration_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid = invalid_utf8.as_ptr() as *const c_char;

    unsafe {
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
fn test_ffi_scope_tool_registration_invalid_utf8_scope_uuid_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid_scope = invalid_utf8.as_ptr() as *const c_char;
    let name = cstring("scope-tool-invalid-scope");

    unsafe {
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                invalid_scope,
                name.as_ptr(),
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
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_request_intercept(
                invalid_scope,
                name.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_request_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_execution_intercept(
                invalid_scope,
                name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
    }
}

#[test]
fn test_ffi_scope_llm_and_subscriber_registration_invalid_utf8_scope_uuid_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid_scope = invalid_utf8.as_ptr() as *const c_char;
    let name = cstring("scope-llm-invalid-scope");

    unsafe {
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                invalid_scope,
                name.as_ptr(),
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
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_request_intercept(
                invalid_scope,
                name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_request_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_execution_intercept(
                invalid_scope,
                name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_stream_execution_intercept(
                invalid_scope,
                name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                invalid_scope,
                name.as_ptr()
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_subscriber(
                invalid_scope,
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
    }
}

#[test]
fn test_ffi_scope_tool_registration_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("scope-tool-invalid-name");
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
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;

        assert_status!(
            nemo_relay_scope_register_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid,
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
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid,
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_request_intercept(
                scope_uuid.as_ptr(),
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
            nemo_relay_scope_deregister_tool_request_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_tool_execution_intercept(
                scope_uuid.as_ptr(),
                invalid,
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_tool_execution_intercept(scope_uuid.as_ptr(), invalid),
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
fn test_ffi_scope_llm_and_subscriber_registration_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("scope-llm-invalid-name");
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
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;

        assert_status!(
            nemo_relay_scope_register_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid,
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
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid,
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_request_intercept(
                scope_uuid.as_ptr(),
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
            nemo_relay_scope_deregister_llm_request_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_execution_intercept(
                scope_uuid.as_ptr(),
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_execution_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                invalid,
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                invalid,
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), invalid),
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
fn test_ffi_scope_and_event_parent_and_utf8_paths() {
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
            nemo_relay_pop_scope(child, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(child);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_call_parent_tool_call_id_and_utf8_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_tool_call_utf8");
        let args = cstring(r#"{"value":1}"#);
        let result = cstring(r#"{"result":{"done":true}}"#);
        let data = cstring(r#"{"source":"tool-call"}"#);
        let metadata = cstring(r#"{"trace":"tool-call"}"#);
        let tool_call_id = cstring("tool-call-id");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut handle = ptr::null_mut();

        assert_status!(
            nemo_relay_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                tool_call_id.as_ptr(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_tool_handle_parent_uuid(handle)).is_some());
        assert_status!(
            nemo_relay_tool_call(
                invalid,
                args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                invalid,
                &mut handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_tool_call_end(handle, result.as_ptr(), ptr::null(), invalid_json.as_ptr()),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_tool_call_end(handle, result.as_ptr(), data.as_ptr(), metadata.as_ptr()),
            NemoRelayStatus::Ok
        );

        nemo_relay_tool_handle_free(handle);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_call_parent_model_and_utf8_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_call_utf8");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);
        let data = cstring(r#"{"source":"llm-call"}"#);
        let metadata = cstring(r#"{"trace":"llm-call"}"#);
        let model_name = cstring("ffi-model-override");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut handle = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                model_name.as_ptr(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert!(take_string(nemo_relay_llm_handle_parent_uuid(handle)).is_some());
        assert_status!(
            nemo_relay_llm_call(
                invalid,
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                invalid,
                &mut handle,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_call_end(
                handle,
                response.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_call_end(handle, response.as_ptr(), data.as_ptr(), metadata.as_ptr()),
            NemoRelayStatus::Ok
        );

        nemo_relay_llm_handle_free(handle);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_execute_and_stream_shape_and_out_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let name = cstring("ffi_llm_execute_shape");
        let invalid_shape = cstring(r#"{"content":{"model":"ffi-model"}}"#);
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );

        assert_status!(
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
        let mut out = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                invalid_shape.as_ptr(),
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
                &mut out,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        assert_status!(
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
        let mut stream = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                invalid_shape.as_ptr(),
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
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );
    }
}

#[test]
fn test_ffi_stream_next_reports_error_items() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.blocking_send(Err(nemo_relay::error::FlowError::Internal(
        "ffi stream failed".to_string(),
    )))
    .expect("expected error payload to be queued");
    drop(tx);

    let stream = Box::into_raw(Box::new(FfiStream {
        receiver: tokio::sync::Mutex::new(rx),
        cancel: tokio::sync::watch::channel(false).0,
        closed: tokio::sync::watch::channel(Some(Ok(()))).1,
    }));

    unsafe {
        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), -1);
        assert!(chunk.is_null());
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("ffi stream failed")
        );
        nemo_relay_stream_free(stream);
    }
}

#[test]
fn test_ffi_llm_helper_invalid_shape_and_intercept_failure_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("ffi_llm_helper_error_sweep");
        let valid_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let invalid_shape = cstring(r#"{"headers":[],"content":1}"#);
        let mut out = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_request_intercepts(name.as_ptr(), invalid_shape.as_ptr(), &mut out),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        assert_status!(
            nemo_relay_llm_conditional_execution(invalid_shape.as_ptr()),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        let intercept_name = cstring(&unique_name("ffi_llm_request_intercept_fail"));
        assert_status!(
            nemo_relay_register_llm_request_intercept(
                intercept_name.as_ptr(),
                1,
                false,
                llm_request_intercept_fail_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_llm_request_intercepts(name.as_ptr(), valid_request.as_ptr(), &mut out),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("llm request intercept callback failed")
        );
        assert_status!(
            nemo_relay_deregister_llm_request_intercept(intercept_name.as_ptr()),
            NemoRelayStatus::Ok
        );

        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_helper_and_lifecycle_callback_failure_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let tool_name = cstring("ffi_tool_failure_sweep");
        let tool_args = cstring(r#"{"value":9}"#);
        let llm_name = cstring("ffi_llm_failure_sweep");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);

        let tool_intercept = cstring(&unique_name("ffi_tool_helper_fail"));
        assert_status!(
            nemo_relay_register_tool_request_intercept(
                tool_intercept.as_ptr(),
                1,
                false,
                tool_request_fail_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let mut tool_out = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_request_intercepts(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                &mut tool_out
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("tool sanitize callback failed")
        );
        assert_status!(
            nemo_relay_deregister_tool_request_intercept(tool_intercept.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_intercept = cstring(&unique_name("ffi_llm_helper_fail"));
        assert_status!(
            nemo_relay_register_llm_request_intercept(
                llm_intercept.as_ptr(),
                1,
                false,
                llm_request_intercept_fail_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let mut llm_out = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_request_intercepts(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                &mut llm_out
            ),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("llm request intercept callback failed")
        );
        assert_status!(
            nemo_relay_deregister_llm_request_intercept(llm_intercept.as_ptr()),
            NemoRelayStatus::Ok
        );

        let mut llm_handle = ptr::null_mut();
        assert_status!(
            nemo_relay_llm_call(
                llm_name.as_ptr(),
                llm_request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_llm_call_end(llm_handle, llm_response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_llm_handle_free(llm_handle);

        let mut tool_handle = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_call(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut tool_handle,
            ),
            NemoRelayStatus::Ok
        );

        let tool_result = cstring(r#"{"result":{"done":true}}"#);
        assert_status!(
            nemo_relay_tool_call_end(tool_handle, tool_result.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(tool_handle);

        let invalid_utf8 = [0xffu8, 0];
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let invalid_json = cstring("{");
        let mut exec_out = ptr::null_mut();
        assert_status!(
            nemo_relay_tool_call_execute(
                invalid_name,
                tool_args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                &mut exec_out,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_tool_call_execute(
                tool_name.as_ptr(),
                tool_args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                parent,
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                &mut exec_out,
            ),
            NemoRelayStatus::InvalidJson
        );

        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_scope_registry_missing_scope_and_null_out_sweeps() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_scope_registry_missing_scope_sweep");
        let valid_name = cstring("ffi_missing_scope_registry_name");
        let missing_scope_uuid = cstring(&uuid::Uuid::now_v7().to_string());
        let invalid_utf8 = [0xffu8, 0];
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;

        assert_status!(
            nemo_relay_push_scope(
                scope_name.as_ptr(),
                NemoRelayScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out pointer is null")
        );

        macro_rules! assert_missing_scope {
            ($expr:expr) => {
                assert_status!($expr, NemoRelayStatus::NotFound);
            };
        }

        assert_missing_scope!(nemo_relay_scope_register_tool_sanitize_request_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_tool_sanitize_response_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                missing_scope_uuid.as_ptr(),
                valid_name.as_ptr(),
            )
        );
        assert_missing_scope!(
            nemo_relay_scope_register_tool_conditional_execution_guardrail(
                missing_scope_uuid.as_ptr(),
                valid_name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_missing_scope!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                missing_scope_uuid.as_ptr(),
                valid_name.as_ptr(),
            )
        );
        assert_missing_scope!(nemo_relay_scope_register_tool_request_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_tool_request_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_tool_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_tool_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));

        assert_missing_scope!(nemo_relay_scope_register_llm_sanitize_request_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_llm_sanitize_response_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(
            nemo_relay_scope_register_llm_conditional_execution_guardrail(
                missing_scope_uuid.as_ptr(),
                valid_name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_missing_scope!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                missing_scope_uuid.as_ptr(),
                valid_name.as_ptr(),
            )
        );
        assert_missing_scope!(nemo_relay_scope_register_llm_request_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_llm_request_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_llm_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_llm_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_llm_stream_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_llm_stream_execution_intercept(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));
        assert_missing_scope!(nemo_relay_scope_register_subscriber(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_missing_scope!(nemo_relay_scope_deregister_subscriber(
            missing_scope_uuid.as_ptr(),
            valid_name.as_ptr(),
        ));

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

        assert_status!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                invalid_name,
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
fn test_ffi_llm_lifecycle_additional_error_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_lifecycle_extra");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);
        let invalid_json = cstring("{");
        let mut handle = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                parent,
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                invalid_json.as_ptr(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::InvalidJson
        );

        assert_status!(
            nemo_relay_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                parent,
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_llm_call_end(
                handle,
                response.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_llm_call_end(handle, response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );

        nemo_relay_llm_handle_free(handle);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_execute_and_stream_additional_input_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_status!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_execute_extra");
        let request = cstring(
            r#"{"headers":{"x-trace":"extra"},"content":{"model":"codec-model","prompt":"hello extra"}}"#,
        );
        let data = cstring(r#"{"source":"llm-extra"}"#);
        let metadata = cstring(r#"{"trace":"llm-extra"}"#);
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let invalid_model_name = invalid_utf8.as_ptr() as *const c_char;
        let response_codec = api::nemo_relay_openai_chat_codec_new();
        let mut out_json = ptr::null_mut();
        let mut stream = ptr::null_mut();
        let mut chunk = ptr::null_mut();

        assert_status!(
            nemo_relay_llm_call_execute(
                invalid_name,
                request.as_ptr(),
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
                &mut out_json,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                invalid_json.as_ptr(),
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
                &mut out_json,
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
                0,
                invalid_json.as_ptr(),
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
        assert_status!(
            nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_openai_chat_cb,
                ptr::null_mut(),
                None,
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                response_codec,
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let decoded = returned_json(out_json);
        assert_eq!(decoded["id"], json!("chatcmpl-ffi"));
        assert_eq!(decoded["model"], json!("codec-model"));

        assert_status!(
            nemo_relay_llm_stream_call_execute(
                invalid_name,
                request.as_ptr(),
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
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                invalid_json.as_ptr(),
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
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                parent,
                0,
                ptr::null(),
                invalid_json.as_ptr(),
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
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                parent,
                0,
                ptr::null(),
                ptr::null(),
                invalid_model_name,
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut stream,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                Some(collector_cb),
                Some(finalizer_cb),
                parent,
                1,
                data.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut stream,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 1);
        assert_eq!(returned_json(chunk)["content"], json!("hello from ffi"));
        assert_eq!(nemo_relay_stream_next(stream, &mut chunk), 0);
        nemo_relay_stream_free(stream);

        types::nemo_relay_codec_free(response_codec);
        nemo_relay_scope_handle_free(parent);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_adaptive_runtime_and_cache_helper_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let config = cstring(
            &json!({
                "version": 1,
                "agent_id": "ffi-adaptive-openai",
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
            nemo_relay_adaptive_runtime_register(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_deregister(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_shutdown(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_wait_for_idle(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_report_json(ptr::null_mut(), &mut out_json),
            NemoRelayStatus::NullPointer
        );

        assert_status!(
            nemo_relay_adaptive_validate_config(ptr::null(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_validate_config(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));

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
            nemo_relay_adaptive_runtime_report_json(runtime, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_report_json(runtime, &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));
        assert_status!(
            nemo_relay_adaptive_runtime_register(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_wait_for_idle(runtime),
            NemoRelayStatus::Ok
        );

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_adaptive_bound_scope");
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
            nemo_relay_adaptive_runtime_bind_scope(runtime, ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_runtime_bind_scope(runtime, scope),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_bind_scope(ptr::null_mut(), scope),
            NemoRelayStatus::NullPointer
        );

        let cache_facts_options = cstring(
            &json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000601",
                "annotated_request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Find cache evidence"
                        }
                    ],
                    "model": "gpt-4.1-mini"
                },
                "agent_id": "ffi-adaptive-openai",
                "timestamp": "2026-06-24T12:00:00Z"
            })
            .to_string(),
        );
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                cache_facts_options.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let facts = returned_json(out_json);
        assert_eq!(facts["provider"], "openai");
        assert_eq!(facts["missing_facts"][0], "acg_stability_unavailable");

        let mut null_runtime_json = ptr::null_mut();
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                ptr::null_mut(),
                cache_facts_options.as_ptr(),
                &mut null_runtime_json,
            ),
            NemoRelayStatus::NullPointer
        );
        assert!(null_runtime_json.is_null());

        for (options, expected) in [
            (
                json!({
                    "provider": "openai",
                    "request_id": "not-a-uuid",
                    "annotated_request": {},
                    "agent_id": "ffi-adaptive-openai"
                }),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "00000000-0000-0000-0000-000000000602",
                    "annotated_request": {},
                    "agent_id": "ffi-adaptive-openai",
                    "timestamp": "not-a-timestamp"
                }),
                NemoRelayStatus::InvalidArg,
            ),
            (
                json!({
                    "provider": "openai",
                    "request_id": "00000000-0000-0000-0000-000000000603",
                    "annotated_request": "bad",
                    "agent_id": "ffi-adaptive-openai"
                }),
                NemoRelayStatus::InvalidJson,
            ),
        ] {
            let options = cstring(&options.to_string());
            assert_eq!(
                nemo_relay_adaptive_runtime_build_cache_request_facts(
                    runtime,
                    options.as_ptr(),
                    &mut out_json,
                ),
                expected
            );
        }
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                invalid_json.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                cache_facts_options.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );

        let telemetry_options = cstring(
            &json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000604",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 8,
                    "cache_read_tokens": 25
                },
                "request_facts": facts,
                "agent_id": "ffi-adaptive-openai",
                "template_version": "v1",
                "toolset_hash": "tools",
                "model_family": "gpt",
                "tenant_scope": "tenant",
                "timestamp": "2026-06-24T12:00:01Z"
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
        assert_eq!(event["cache_read_tokens"], json!(25));
        assert_eq!(event["hit_rate"], json!(0.25));

        let no_usage_options = cstring(
            &json!({
                "provider": "openai",
                "request_id": "00000000-0000-0000-0000-000000000605",
                "agent_id": "ffi-adaptive-openai",
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
                    "request_id": "00000000-0000-0000-0000-000000000606",
                    "usage": {},
                    "agent_id": "ffi-adaptive-openai",
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
                    "agent_id": "ffi-adaptive-openai",
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
                    "request_id": "00000000-0000-0000-0000-000000000607",
                    "usage": {},
                    "agent_id": "ffi-adaptive-openai",
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
                    "request_id": "00000000-0000-0000-0000-000000000608",
                    "usage": "bad",
                    "agent_id": "ffi-adaptive-openai",
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
                    "request_id": "00000000-0000-0000-0000-000000000609",
                    "usage": {},
                    "request_facts": "bad",
                    "agent_id": "ffi-adaptive-openai",
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
            nemo_relay_adaptive_build_cache_telemetry_event(
                telemetry_options.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_adaptive_build_cache_telemetry_event(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
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
            nemo_relay_adaptive_runtime_shutdown(runtime),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_adaptive_runtime_shutdown(runtime),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_adaptive_runtime_register(runtime),
            NemoRelayStatus::InvalidArg
        );

        assert_status!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
        types::nemo_relay_adaptive_runtime_free(runtime);
    }
}

#[test]
fn test_ffi_scope_stack_propagation_and_thread_binding_entry_points() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut context_json = ptr::null_mut();
        assert_status!(
            nemo_relay_capture_propagation_context_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_capture_propagation_context_json(&mut context_json),
            NemoRelayStatus::Ok
        );
        let context = returned_json(context_json);
        assert_eq!(context["version"], json!(1));

        let root_uuid = cstring("018f13f0-7c1a-7a80-8000-000000000701");
        assert_status!(
            nemo_relay_capture_propagation_context_with_root_json(
                root_uuid.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        let invalid_root = cstring("not-a-uuid");
        assert_status!(
            nemo_relay_capture_propagation_context_with_root_json(
                invalid_root.as_ptr(),
                &mut context_json,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_capture_propagation_context_with_root_json(
                root_uuid.as_ptr(),
                &mut context_json,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            returned_json(context_json)["root_uuid"],
            root_uuid.to_str().unwrap()
        );
        assert_status!(
            nemo_relay_capture_propagation_context_with_root_json(ptr::null(), &mut context_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(context_json)["root_uuid"], Json::Null);

        let payload = cstring(&context.to_string());
        let invalid_json = cstring("{");
        let mut stack = ptr::null_mut();
        assert_status!(
            nemo_relay_scope_stack_create_from_propagation_json(payload.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_create_from_propagation_json(ptr::null(), &mut stack),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_create_from_propagation_json(invalid_json.as_ptr(), &mut stack),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_scope_stack_create_from_propagation_json(payload.as_ptr(), &mut stack),
            NemoRelayStatus::Ok
        );
        assert!(!stack.is_null());

        assert_status!(
            nemo_relay_scope_stack_set_thread(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_set_thread(stack),
            NemoRelayStatus::Ok
        );
        assert!(nemo_relay_scope_stack_active());

        let mut traceparent = ptr::null_mut();
        assert_status!(
            nemo_relay_capture_traceparent(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_capture_traceparent(&mut traceparent),
            NemoRelayStatus::Ok
        );
        nemo_relay_string_free(traceparent);

        let scope_name = cstring("ffi_traceparent_scope");
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
            nemo_relay_capture_traceparent(&mut traceparent),
            NemoRelayStatus::Ok
        );
        let traceparent_value = take_string(traceparent).expect("traceparent must be allocated");
        assert!(traceparent_value.starts_with("00-"));
        assert!(traceparent_value.ends_with("-01"));
        assert_eq!(traceparent_value.len(), 55);
        assert_status!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);

        let rooted_context = cstring(
            r#"{"version":1,"root_uuid":"018f13f0-7c1a-7a80-8000-000000000701","parent_uuid":"018f13f0-7c1a-7a80-8000-000000000702"}"#,
        );
        let rootless_context =
            cstring(r#"{"version":1,"parent_uuid":"018f13f0-7c1a-7a80-8000-000000000702"}"#);
        assert_status!(
            nemo_relay_propagation_context_to_traceparent(rooted_context.as_ptr(), ptr::null_mut(),),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_propagation_context_to_traceparent(ptr::null(), &mut traceparent),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_propagation_context_to_traceparent(
                rootless_context.as_ptr(),
                &mut traceparent,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_status!(
            nemo_relay_propagation_context_to_traceparent(
                rooted_context.as_ptr(),
                &mut traceparent,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            take_string(traceparent),
            Some("00-018f13f07c1a7a808000000000000701-8000000000000702-01".into())
        );

        let mut binding = ptr::null_mut();
        assert_status!(
            nemo_relay_scope_stack_capture_thread(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_capture_thread(&mut binding),
            NemoRelayStatus::Ok
        );
        assert!(!binding.is_null());
        assert_status!(
            nemo_relay_scope_stack_restore_thread(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_scope_stack_restore_thread(binding),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_observability_exporter_error_lifecycles() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut out_json = ptr::null_mut();
        assert_status!(
            nemo_relay_observability_default_config_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        let invalid_config = cstring("{");
        assert_status!(
            nemo_relay_observability_component_spec_json(
                invalid_config.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );

        let directory = cstring(
            &std::env::temp_dir()
                .join(unique_name("ffi_atof"))
                .display()
                .to_string(),
        );
        let mode = cstring("overwrite");
        let filename = cstring("events.jsonl");
        let mut exporter = ptr::null_mut();
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
            nemo_relay_atof_exporter_register(ptr::null(), filename.as_ptr()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_atof_exporter_path(exporter, ptr::null_mut()),
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
        let subscriber_name = cstring(&unique_name("ffi_atof_subscriber"));
        assert_status!(
            nemo_relay_atof_exporter_register(exporter, subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_register(exporter, subscriber_name.as_ptr()),
            NemoRelayStatus::AlreadyExists
        );
        assert_status!(
            nemo_relay_atof_exporter_deregister(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        assert_status!(
            nemo_relay_atof_exporter_deregister(subscriber_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        types::nemo_relay_atof_exporter_free(exporter);

        let otel_type = cstring("full");
        let endpoint = cstring("http://127.0.0.1:4318/v1/traces");
        let mut subscriber = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                otel_type.as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut subscriber,
            ),
            NemoRelayStatus::Ok
        );
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
        let missing_name = cstring(&unique_name("missing_otel_subscriber"));
        assert_status!(
            nemo_relay_otel_subscriber_deregister(missing_name.as_ptr()),
            NemoRelayStatus::Ok
        );
        types::nemo_relay_otel_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_observability_component_and_constructor_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut out_json = ptr::null_mut();
        let invalid_json = cstring("{");
        let invalid_config = cstring(r#"{"version":"bad"}"#);
        let explicit_config = cstring(
            &json!({
                "version": 1,
                "atof": {
                    "enabled": false
                }
            })
            .to_string(),
        );

        assert_status!(
            nemo_relay_observability_default_config_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, ptr::null_mut(),),
            NemoRelayStatus::NullPointer
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(
                invalid_json.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(
                invalid_config.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_status!(
            nemo_relay_observability_component_spec_json(
                explicit_config.as_ptr(),
                false,
                &mut out_json,
            ),
            NemoRelayStatus::Ok
        );
        let component = returned_json(out_json);
        assert_eq!(component["kind"], "observability");
        assert_eq!(component["enabled"], false);

        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let append = cstring("append");
        let filename = cstring("events.jsonl");
        let bad_mode = cstring("bad-mode");
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
            nemo_relay_atof_exporter_create(invalid, append.as_ptr(), filename.as_ptr(), &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_create(ptr::null(), invalid, filename.as_ptr(), &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );
        assert_status!(
            nemo_relay_atof_exporter_create(ptr::null(), append.as_ptr(), invalid, &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );

        let invalid_map_shape = cstring(r#"["not-an-object"]"#);
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let mut otel = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"full".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                invalid_map_shape.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut otel,
            ),
            NemoRelayStatus::InvalidArg
        );
        let mut openinference = ptr::null_mut();
        assert_status!(
            nemo_relay_otel_subscriber_create(
                c"openinference".as_ptr(),
                ptr::null(),
                endpoint.as_ptr(),
                invalid_map_shape.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                &mut openinference,
            ),
            NemoRelayStatus::InvalidArg
        );
    }
}

#[test]
fn test_ffi_scope_event_and_metric_validation_sweeps() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let fixtures = ValidationFixtures::new("unit");
        let invalid = fixtures.invalid_utf8();
        let malformed_json = &fixtures.malformed_json;
        let metric_name = &fixtures.metric_name;
        let event_name = &fixtures.event_name;
        let scope_name = &fixtures.scope_name;
        let metadata = &fixtures.metadata;
        let invalid_timestamp = fixtures.invalid_timestamp;

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

        assert_status!(
            api::nemo_relay_pop_scope(ptr::null(), ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        for (output, metadata, timestamp, expected) in fixtures.pop_scope_cases() {
            assert_status!(
                api::nemo_relay_pop_scope(scope, output, metadata, timestamp),
                expected
            );
        }

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
        for (data, schema, event_metadata) in fixtures.event_json_cases() {
            assert_status!(
                api::nemo_relay_event_v2(
                    event_name.as_ptr(),
                    scope,
                    data,
                    schema,
                    event_metadata,
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

        for (name, values, metric_metadata, timestamp, expected) in fixtures.metric_json_cases() {
            assert_status!(
                api::nemo_relay_metric_json(name, ptr::null(), values, metric_metadata, timestamp),
                expected
            );
        }
        assert_status!(
            api::nemo_relay_metric_json(
                metric_name.as_ptr(),
                scope,
                fixtures.measurements.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );

        let base_measurement = fixtures.base_measurement();
        assert_status!(
            api::nemo_relay_metric(
                invalid,
                ptr::null(),
                ptr::from_ref(&base_measurement),
                1,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
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
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidArg
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
                    ptr::null(),
                    ptr::from_ref(&measurement),
                    1,
                    ptr::null(),
                    ptr::null(),
                ),
                NemoRelayStatus::InvalidUtf8
            );
        }
        let invalid_attributes = types::NemoRelayMetricMeasurement {
            attributes_json: malformed_json.as_ptr(),
            ..base_measurement
        };
        assert_status!(
            api::nemo_relay_metric(
                metric_name.as_ptr(),
                ptr::null(),
                ptr::from_ref(&invalid_attributes),
                1,
                ptr::null(),
                ptr::null(),
            ),
            NemoRelayStatus::InvalidJson
        );

        let histogram_measurement = types::NemoRelayMetricMeasurement {
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
                ptr::from_ref(&histogram_measurement),
                1,
                metadata.as_ptr(),
                ptr::null(),
            ),
            NemoRelayStatus::Ok
        );
        for (metric_metadata, timestamp, expected) in fixtures.native_metric_status_cases() {
            assert_status!(
                api::nemo_relay_metric(
                    metric_name.as_ptr(),
                    ptr::null(),
                    ptr::from_ref(&base_measurement),
                    1,
                    metric_metadata,
                    timestamp,
                ),
                expected
            );
        }

        assert_status!(
            api::nemo_relay_pop_scope(scope, ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_otel_signal_subscribers_apply_all_typed_options() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let transport = cstring("http_binary");
        let log_endpoint = cstring("http://127.0.0.1:4318/v1/logs");
        let metric_endpoint = cstring("http://127.0.0.1:4318/v1/metrics");
        let headers = cstring(r#"{"authorization":"test-token"}"#);
        let resources = cstring(r#"{"service.instance.id":"ffi-test"}"#);
        let service_name = cstring("ffi-coverage");
        let service_namespace = cstring("nemo-relay");
        let service_version = cstring("test");
        let instrumentation_scope = cstring("ffi.coverage");
        let minimum_severity = cstring("warn");
        let temporality = cstring("delta");

        let log_name = cstring(&unique_name("ffi_typed_log_subscriber"));
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
            |_| {},
            |subscriber, name| nemo_relay_otel_log_subscriber_register(subscriber, name),
            |name| nemo_relay_otel_log_subscriber_deregister(name),
            |subscriber| nemo_relay_otel_log_subscriber_shutdown(subscriber),
            |subscriber| nemo_relay_otel_log_subscriber_force_flush(subscriber),
            |subscriber| types::nemo_relay_otel_log_subscriber_free(subscriber),
        );

        let metric_name = cstring(&unique_name("ffi_typed_metric_subscriber"));
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
            |_| {},
            |subscriber, name| nemo_relay_otel_metric_subscriber_register(subscriber, name),
            |name| nemo_relay_otel_metric_subscriber_deregister(name),
            |subscriber| nemo_relay_otel_metric_subscriber_shutdown(subscriber),
            |subscriber| nemo_relay_otel_metric_subscriber_force_flush(subscriber),
            |subscriber| types::nemo_relay_otel_metric_subscriber_free(subscriber),
        );
    }
}
