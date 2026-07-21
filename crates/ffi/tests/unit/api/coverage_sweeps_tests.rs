// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for coverage sweeps in the NeMo Relay FFI crate.

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
            assert_eq!($expr, NemoRelayStatus::AlreadyExists);
        };
    }

    unsafe {
        let tool_san_req = cstring(&unique_name("dup_tool_san_req_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(tool_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_san_resp = cstring(&unique_name("dup_tool_san_resp_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_response_guardrail(tool_san_resp.as_ptr()),
            NemoRelayStatus::Ok
        );

        let tool_exec = cstring(&unique_name("dup_tool_exec_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_tool_execution_intercept(tool_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_san_req = cstring(&unique_name("dup_llm_san_req_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(llm_san_req.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_exec = cstring(&unique_name("dup_llm_exec_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_llm_execution_intercept(llm_exec.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_stream_exec = cstring(&unique_name("dup_llm_stream_exec_extra"));
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);
        let scope_uuid = cstring(
            &take_string(nemo_relay_scope_handle_uuid(parent)).expect("scope uuid should exist"),
        );

        let tool_name = cstring("ffi_runtime_owner_tool");
        let tool_args = cstring(r#"{"value":1}"#);
        let tool_result = cstring(r#"{"ok":true}"#);
        let llm_name = cstring("ffi_runtime_owner_llm");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);

        let mut tool_handle = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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

        assert_eq!(
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

        assert_eq!(
            nemo_relay_llm_conditional_execution(llm_request.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let mut conflict_scope = ptr::null_mut();
        assert_eq!(
            nemo_relay_get_handle(&mut conflict_scope),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_event(tool_name.as_ptr(), parent, ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let mut conflict_tool_handle = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
            nemo_relay_tool_call_end(tool_handle, tool_result.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        assert_eq!(
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
        assert_eq!(
            nemo_relay_llm_call_end(llm_handle, llm_response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::InvalidArg
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains(conflict_fragment)
        );

        let global_name = cstring("conflict-global");
        assert_eq!(
            nemo_relay_deregister_tool_sanitize_request_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_tool_conditional_execution_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_tool_request_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_tool_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_request_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_sanitize_response_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_conditional_execution_guardrail(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_request_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_llm_stream_execution_intercept(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_deregister_subscriber(global_name.as_ptr()),
            NemoRelayStatus::InvalidArg
        );

        let scope_name = cstring("conflict-scope");
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr()
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_name.as_ptr(),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
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
            assert_eq!($expr, NemoRelayStatus::AlreadyExists);
        };
    }

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("dup_scope_extra");
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

        let tool_san_req = cstring(&unique_name("dup_scope_tool_san_req_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                tool_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let tool_san_resp = cstring(&unique_name("dup_scope_tool_san_resp_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                tool_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let tool_exec = cstring(&unique_name("dup_scope_tool_exec_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                tool_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_san_req = cstring(&unique_name("dup_scope_llm_san_req_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                llm_san_req.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_san_resp = cstring(&unique_name("dup_scope_llm_san_resp_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                llm_san_resp.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_exec = cstring(&unique_name("dup_scope_llm_exec_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                llm_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
        );

        let llm_stream_exec = cstring(&unique_name("dup_scope_llm_stream_exec_extra"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                llm_stream_exec.as_ptr(),
            ),
            NemoRelayStatus::Ok
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
fn test_ffi_global_tool_registration_invalid_utf8_name_sweep() {
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
    }
}

#[test]
fn test_ffi_global_llm_and_subscriber_registration_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid = invalid_utf8.as_ptr() as *const c_char;

    unsafe {
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
fn test_ffi_scope_tool_registration_invalid_utf8_scope_uuid_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let invalid_utf8 = [0xffu8, 0];
    let invalid_scope = invalid_utf8.as_ptr() as *const c_char;
    let name = cstring("scope-tool-invalid-scope");

    unsafe {
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_request_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                invalid_scope,
                name.as_ptr(),
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_request_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(invalid_scope, name.as_ptr()),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                invalid_scope,
                name.as_ptr()
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_subscriber(
                invalid_scope,
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;

        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_request_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_tool_execution_intercept(scope_uuid.as_ptr(), invalid),
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

#[test]
fn test_ffi_scope_llm_and_subscriber_registration_invalid_utf8_name_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("scope-llm-invalid-name");
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
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;

        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                invalid,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_request_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_execution_intercept(scope_uuid.as_ptr(), invalid),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
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
        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                invalid
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_register_subscriber(
                scope_uuid.as_ptr(),
                invalid,
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_scope_deregister_subscriber(scope_uuid.as_ptr(), invalid),
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

#[test]
fn test_ffi_scope_and_event_parent_and_utf8_paths() {
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_tool_call_utf8");
        let args = cstring(r#"{"value":1}"#);
        let result = cstring(r#"{"done":true}"#);
        let data = cstring(r#"{"source":"tool-call"}"#);
        let metadata = cstring(r#"{"trace":"tool-call"}"#);
        let tool_call_id = cstring("tool-call-id");
        let invalid_json = cstring("{");
        let invalid_utf8 = [0xffu8, 0];
        let invalid = invalid_utf8.as_ptr() as *const c_char;
        let mut handle = ptr::null_mut();

        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_tool_call_end(handle, result.as_ptr(), ptr::null(), invalid_json.as_ptr()),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

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

        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_llm_call_end(
                handle,
                response.as_ptr(),
                ptr::null(),
                invalid_json.as_ptr()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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
        let mut out = ptr::null_mut();
        assert_eq!(
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
        let mut stream = ptr::null_mut();
        assert_eq!(
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

        assert_eq!(
            nemo_relay_llm_request_intercepts(name.as_ptr(), invalid_shape.as_ptr(), &mut out),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        assert_eq!(
            nemo_relay_llm_conditional_execution(invalid_shape.as_ptr()),
            NemoRelayStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LlmRequest")
        );

        let intercept_name = cstring(&unique_name("ffi_llm_request_intercept_fail"));
        assert_eq!(
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
        assert_eq!(
            nemo_relay_llm_request_intercepts(name.as_ptr(), valid_request.as_ptr(), &mut out),
            NemoRelayStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("llm request intercept callback failed")
        );
        assert_eq!(
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let tool_name = cstring("ffi_tool_failure_sweep");
        let tool_args = cstring(r#"{"value":9}"#);
        let llm_name = cstring("ffi_llm_failure_sweep");
        let llm_request =
            cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);

        let tool_intercept = cstring(&unique_name("ffi_tool_helper_fail"));
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_tool_request_intercept(tool_intercept.as_ptr()),
            NemoRelayStatus::Ok
        );

        let llm_intercept = cstring(&unique_name("ffi_llm_helper_fail"));
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_deregister_llm_request_intercept(llm_intercept.as_ptr()),
            NemoRelayStatus::Ok
        );

        let mut llm_handle = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
            nemo_relay_llm_call_end(llm_handle, llm_response.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_llm_handle_free(llm_handle);

        let mut tool_handle = ptr::null_mut();
        assert_eq!(
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

        let tool_result = cstring(r#"{"done":true}"#);
        assert_eq!(
            nemo_relay_tool_call_end(tool_handle, tool_result.as_ptr(), ptr::null(), ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_tool_handle_free(tool_handle);

        let invalid_utf8 = [0xffu8, 0];
        let invalid_name = invalid_utf8.as_ptr() as *const c_char;
        let invalid_json = cstring("{");
        let mut exec_out = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
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

        assert_eq!(
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
                assert_eq!($expr, NemoRelayStatus::NotFound);
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

        assert_eq!(
            nemo_relay_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                invalid_name,
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

#[test]
fn test_ffi_llm_lifecycle_additional_error_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let mut parent = ptr::null_mut();
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

        let name = cstring("ffi_llm_lifecycle_extra");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let response = cstring(r#"{"content":"ok","role":"assistant","tool_calls":[]}"#);
        let invalid_json = cstring("{");
        let mut handle = ptr::null_mut();

        assert_eq!(
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
        assert_eq!(
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

        assert_eq!(
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
        assert_eq!(
            nemo_relay_llm_call_end(
                handle,
                response.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null()
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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
        assert_eq!(nemo_relay_get_handle(&mut parent), NemoRelayStatus::Ok);

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

        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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

        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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

        assert_eq!(
            nemo_relay_adaptive_validate_config(ptr::null(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_adaptive_validate_config(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_adaptive_validate_config(config.as_ptr(), &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));

        let mut runtime = ptr::null_mut();
        assert_eq!(
            nemo_relay_adaptive_runtime_create(config.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_create(invalid_json.as_ptr(), &mut runtime),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_create(config.as_ptr(), &mut runtime),
            NemoRelayStatus::Ok
        );
        assert!(!runtime.is_null());

        assert_eq!(
            nemo_relay_adaptive_runtime_report_json(runtime, ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_report_json(runtime, &mut out_json),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(out_json)["diagnostics"], json!([]));
        assert_eq!(
            nemo_relay_adaptive_runtime_register(runtime),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_wait_for_idle(runtime),
            NemoRelayStatus::Ok
        );

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_adaptive_bound_scope");
        let mut scope = ptr::null_mut();
        assert_eq!(
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
        assert_eq!(
            nemo_relay_adaptive_runtime_bind_scope(runtime, ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_bind_scope(runtime, scope),
            NemoRelayStatus::Ok
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_adaptive_runtime_build_cache_request_facts(
                runtime,
                invalid_json.as_ptr(),
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
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
        assert_eq!(
            nemo_relay_adaptive_build_cache_telemetry_event(
                telemetry_options.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_adaptive_build_cache_telemetry_event(invalid_json.as_ptr(), &mut out_json),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_adaptive_set_latency_sensitivity(0),
            NemoRelayStatus::InvalidArg
        );

        assert_eq!(
            nemo_relay_adaptive_runtime_deregister(runtime),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_shutdown(runtime),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_shutdown(runtime),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_adaptive_runtime_register(runtime),
            NemoRelayStatus::InvalidArg
        );

        assert_eq!(
            nemo_relay_pop_scope(scope, ptr::null()),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_handle_free(scope);
        nemo_relay_scope_stack_free(stack);
        types::nemo_relay_adaptive_runtime_free(runtime);
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

        assert_eq!(
            nemo_relay_observability_default_config_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_observability_component_spec_json(ptr::null(), true, ptr::null_mut(),),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_observability_component_spec_json(
                invalid_json.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_observability_component_spec_json(
                invalid_config.as_ptr(),
                true,
                &mut out_json,
            ),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
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

        assert_eq!(
            nemo_relay_atof_exporter_create(
                ptr::null(),
                append.as_ptr(),
                filename.as_ptr(),
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_atof_exporter_create(
                ptr::null(),
                bad_mode.as_ptr(),
                filename.as_ptr(),
                &mut atof,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_atof_exporter_create(invalid, append.as_ptr(), filename.as_ptr(), &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_atof_exporter_create(ptr::null(), invalid, filename.as_ptr(), &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            nemo_relay_atof_exporter_create(ptr::null(), append.as_ptr(), invalid, &mut atof,),
            NemoRelayStatus::InvalidUtf8
        );

        let invalid_map_shape = cstring(r#"["not-an-object"]"#);
        let mut otel = ptr::null_mut();
        assert_eq!(
            nemo_relay_otel_subscriber_create(
                ptr::null(),
                ptr::null(),
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
        assert_eq!(
            nemo_relay_openinference_subscriber_create(
                ptr::null(),
                ptr::null(),
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
