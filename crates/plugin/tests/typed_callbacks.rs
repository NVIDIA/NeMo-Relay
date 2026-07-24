// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-API tests for typed native plugin callback registration.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::ptr::{self, NonNull};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicUsize, Ordering},
};

use nemo_relay_plugin::{
    AnnotatedLlmRequest, CategoryProfile, ConfigDiagnostic, DiagnosticLevel, Event, EventCategory,
    EventSanitizeFields, Json, LlmJsonStream, LlmNext, LlmRequest, LlmRequestInterceptOutcome,
    LlmSanitizeContext, LlmStream, LlmStreamNext, NEMO_RELAY_NATIVE_ABI_VERSION, NativePlugin,
    NemoRelayNativeEventSanitizeCb, NemoRelayNativeEventSubscriberCb, NemoRelayNativeFreeFn,
    NemoRelayNativeHostApiV1, NemoRelayNativeLlmConditionalCb, NemoRelayNativeLlmExecutionCb,
    NemoRelayNativeLlmRequestInterceptCb, NemoRelayNativeLlmSanitizeRequestCb,
    NemoRelayNativeLlmSanitizeResponseCb, NemoRelayNativeLlmStreamExecutionCb,
    NemoRelayNativeLlmStreamV1, NemoRelayNativePluginContext, NemoRelayNativePluginV1,
    NemoRelayNativeScopeHandle, NemoRelayNativeScopeStack, NemoRelayNativeScopeStackBinding,
    NemoRelayNativeScopeType, NemoRelayNativeString, NemoRelayNativeToolConditionalCb,
    NemoRelayNativeToolExecutionCb, NemoRelayNativeToolJsonCb, NemoRelayNativeWithScopeStackCb,
    NemoRelayStatus, PendingMarkSpec, PluginContext, PluginRuntime, ScopeType,
    ToolExecutionInterceptOutcome, ToolNext,
};
use serde_json::{Map, json};

struct TestString(Vec<u8>);

struct RegisteredSubscriber {
    name: String,
    cb: NemoRelayNativeEventSubscriberCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

struct RegisteredEventSanitize {
    name: String,
    priority: i32,
    cb: NemoRelayNativeEventSanitizeCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredEventSanitize {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

impl RegisteredSubscriber {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredToolJson {
    name: String,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeToolJsonCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredToolJson {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredToolConditional {
    name: String,
    priority: i32,
    cb: NemoRelayNativeToolConditionalCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredToolConditional {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredToolExecution {
    name: String,
    priority: i32,
    cb: NemoRelayNativeToolExecutionCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredToolExecution {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmRequest {
    name: String,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeRequestCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmRequest {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmJson {
    name: String,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeResponseCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmJson {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmConditional {
    name: String,
    priority: i32,
    cb: NemoRelayNativeLlmConditionalCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmConditional {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmExecution {
    name: String,
    priority: i32,
    cb: NemoRelayNativeLlmExecutionCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmExecution {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmStreamExecution {
    name: String,
    priority: i32,
    cb: NemoRelayNativeLlmStreamExecutionCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmStreamExecution {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredLlmRequestIntercept {
    name: String,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeLlmRequestInterceptCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredLlmRequestIntercept {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

trait CapturedRegistration {
    unsafe fn free(self);
}

macro_rules! impl_captured_registration {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CapturedRegistration for $ty {
                unsafe fn free(self) {
                    unsafe { <$ty>::free(self) };
                }
            }
        )+
    };
}

impl_captured_registration!(
    RegisteredSubscriber,
    RegisteredEventSanitize,
    RegisteredToolJson,
    RegisteredToolConditional,
    RegisteredToolExecution,
    RegisteredLlmRequest,
    RegisteredLlmJson,
    RegisteredLlmConditional,
    RegisteredLlmExecution,
    RegisteredLlmStreamExecution,
    RegisteredLlmRequestIntercept,
);

fn replace_registration<T: CapturedRegistration>(slot: &Mutex<Option<T>>, registration: T) {
    let previous = {
        let mut slot = slot.lock().unwrap();
        slot.replace(registration)
    };
    if let Some(previous) = previous {
        unsafe { previous.free() };
    }
}

fn clear_registration<T: CapturedRegistration>(slot: &Mutex<Option<T>>) {
    let registration = {
        let mut slot = slot.lock().unwrap();
        slot.take()
    };
    if let Some(registration) = registration {
        unsafe { registration.free() };
    }
}

static TEST_LOCK: Mutex<()> = Mutex::new(());
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);
static REGISTRATION_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static STRING_NEW_REMAINING_SUCCESSES: Mutex<Option<usize>> = Mutex::new(None);
static STRING_NEW_RETURNS_NULL: Mutex<bool> = Mutex::new(false);
static SCOPE_GET_CURRENT_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_GET_CURRENT_RETURNS_NULL: Mutex<bool> = Mutex::new(false);
static SCOPE_PUSH_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_PUSH_RETURNS_NULL: Mutex<bool> = Mutex::new(false);
static SCOPE_POP_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static EMIT_MARK_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_STACK_CREATE_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_STACK_CREATE_RETURNS_NULL: Mutex<bool> = Mutex::new(false);
static SCOPE_STACK_SET_THREAD_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_STACK_CAPTURE_THREAD_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_STACK_CAPTURE_THREAD_RETURNS_NULL: Mutex<bool> = Mutex::new(false);
static SCOPE_STACK_RESTORE_THREAD_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SCOPE_STACK_WITH_CURRENT_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static STRING_LIVE_COUNT: AtomicUsize = AtomicUsize::new(0);
static RUNTIME_CALLS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SCOPE_HANDLE_FREES: AtomicUsize = AtomicUsize::new(0);
static SCOPE_STACK_FREES: AtomicUsize = AtomicUsize::new(0);
static SCOPE_STACK_BINDING_FREES: AtomicUsize = AtomicUsize::new(0);
static SCOPE_STACK_BINDING_RESTORES: AtomicUsize = AtomicUsize::new(0);
static SUBSCRIBER_REGISTRATION: Mutex<Option<RegisteredSubscriber>> = Mutex::new(None);
static EVENT_SANITIZE_REGISTRATION: Mutex<Option<RegisteredEventSanitize>> = Mutex::new(None);
static TOOL_JSON_REGISTRATION: Mutex<Option<RegisteredToolJson>> = Mutex::new(None);
static TOOL_CONDITIONAL_REGISTRATION: Mutex<Option<RegisteredToolConditional>> = Mutex::new(None);
static TOOL_EXECUTION_REGISTRATION: Mutex<Option<RegisteredToolExecution>> = Mutex::new(None);
static LLM_REQUEST_REGISTRATION: Mutex<Option<RegisteredLlmRequest>> = Mutex::new(None);
static LLM_JSON_REGISTRATION: Mutex<Option<RegisteredLlmJson>> = Mutex::new(None);
static LLM_CONDITIONAL_REGISTRATION: Mutex<Option<RegisteredLlmConditional>> = Mutex::new(None);
static LLM_EXECUTION_REGISTRATION: Mutex<Option<RegisteredLlmExecution>> = Mutex::new(None);
static LLM_STREAM_EXECUTION_REGISTRATION: Mutex<Option<RegisteredLlmStreamExecution>> =
    Mutex::new(None);
static LLM_REQUEST_INTERCEPT_REGISTRATION: Mutex<Option<RegisteredLlmRequestIntercept>> =
    Mutex::new(None);

#[test]
fn native_abi_v1_struct_sizes_are_self_describing() {
    assert_eq!(NEMO_RELAY_NATIVE_ABI_VERSION, 2);
    assert_eq!(
        size_of::<NemoRelayNativeHostApiV1>(),
        test_host().struct_size
    );
    assert_eq!(
        size_of::<NemoRelayNativePluginV1>(),
        NemoRelayNativePluginV1::default().struct_size
    );
    assert_eq!(
        size_of::<NemoRelayNativeLlmStreamV1>(),
        NemoRelayNativeLlmStreamV1::default().struct_size
    );
    assert_eq!(NemoRelayStatus::StreamEnd as i32, 10);

    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(align_of::<NemoRelayNativeHostApiV1>(), 8);
        assert_eq!(size_of::<NemoRelayNativeHostApiV1>(), 296);
        assert_eq!(
            host_api_offsets(),
            [
                0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
                152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272,
                280, 288,
            ]
        );
        assert_eq!(align_of::<NemoRelayNativePluginV1>(), 8);
        assert_eq!(size_of::<NemoRelayNativePluginV1>(), 56);
        assert_eq!(plugin_offsets(), [0, 8, 16, 24, 32, 40, 48]);
        assert_eq!(align_of::<NemoRelayNativeLlmStreamV1>(), 8);
        assert_eq!(size_of::<NemoRelayNativeLlmStreamV1>(), 40);
        assert_eq!(stream_offsets(), [0, 8, 16, 24, 32]);
    }

    #[cfg(target_pointer_width = "32")]
    {
        assert_eq!(align_of::<NemoRelayNativeHostApiV1>(), 4);
        assert_eq!(size_of::<NemoRelayNativeHostApiV1>(), 156);
        assert_eq!(
            host_api_offsets(),
            [
                0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
                84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148,
                152,
            ]
        );
        assert_eq!(align_of::<NemoRelayNativePluginV1>(), 4);
        assert_eq!(size_of::<NemoRelayNativePluginV1>(), 28);
        assert_eq!(plugin_offsets(), [0, 4, 8, 12, 16, 20, 24]);
        assert_eq!(align_of::<NemoRelayNativeLlmStreamV1>(), 4);
        assert_eq!(size_of::<NemoRelayNativeLlmStreamV1>(), 20);
        assert_eq!(stream_offsets(), [0, 4, 8, 12, 16]);
    }
}

fn host_api_offsets() -> [usize; 37] {
    [
        offset_of!(NemoRelayNativeHostApiV1, abi_version),
        offset_of!(NemoRelayNativeHostApiV1, struct_size),
        offset_of!(NemoRelayNativeHostApiV1, relay_version),
        offset_of!(NemoRelayNativeHostApiV1, string_new),
        offset_of!(NemoRelayNativeHostApiV1, string_data),
        offset_of!(NemoRelayNativeHostApiV1, string_len),
        offset_of!(NemoRelayNativeHostApiV1, string_free),
        offset_of!(NemoRelayNativeHostApiV1, last_error_clear),
        offset_of!(NemoRelayNativeHostApiV1, last_error_set),
        offset_of!(NemoRelayNativeHostApiV1, plugin_context_register_subscriber),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_tool_sanitize_request_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_tool_sanitize_response_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_tool_conditional_execution_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_tool_request_intercept
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_tool_execution_intercept
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_sanitize_request_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_sanitize_response_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_conditional_execution_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_request_intercept
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_execution_intercept
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_llm_stream_execution_intercept
        ),
        offset_of!(NemoRelayNativeHostApiV1, scope_handle_free),
        offset_of!(NemoRelayNativeHostApiV1, scope_get_current),
        offset_of!(NemoRelayNativeHostApiV1, scope_push),
        offset_of!(NemoRelayNativeHostApiV1, scope_pop),
        offset_of!(NemoRelayNativeHostApiV1, emit_mark),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_create),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_free),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_set_thread),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_capture_thread),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_restore_thread),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_binding_free),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_active),
        offset_of!(NemoRelayNativeHostApiV1, scope_stack_with_current),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_mark_sanitize_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_scope_sanitize_start_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV1,
            plugin_context_register_scope_sanitize_end_guardrail
        ),
    ]
}

fn plugin_offsets() -> [usize; 7] {
    [
        offset_of!(NemoRelayNativePluginV1, struct_size),
        offset_of!(NemoRelayNativePluginV1, plugin_kind),
        offset_of!(NemoRelayNativePluginV1, allows_multiple_components),
        offset_of!(NemoRelayNativePluginV1, user_data),
        offset_of!(NemoRelayNativePluginV1, validate),
        offset_of!(NemoRelayNativePluginV1, register),
        offset_of!(NemoRelayNativePluginV1, drop),
    ]
}

fn stream_offsets() -> [usize; 5] {
    [
        offset_of!(NemoRelayNativeLlmStreamV1, struct_size),
        offset_of!(NemoRelayNativeLlmStreamV1, user_data),
        offset_of!(NemoRelayNativeLlmStreamV1, next),
        offset_of!(NemoRelayNativeLlmStreamV1, cancel),
        offset_of!(NemoRelayNativeLlmStreamV1, drop),
    ]
}

unsafe extern "C" fn test_string_new(
    data: *const u8,
    len: usize,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out.is_null() || (data.is_null() && len > 0) {
        return NemoRelayStatus::NullPointer;
    }
    {
        let mut remaining = STRING_NEW_REMAINING_SUCCESSES.lock().unwrap();
        if let Some(remaining) = remaining.as_mut() {
            if *remaining == 0 {
                return NemoRelayStatus::Internal;
            }
            *remaining -= 1;
        }
    }
    if *STRING_NEW_RETURNS_NULL.lock().unwrap() {
        unsafe { *out = ptr::null_mut() };
        return NemoRelayStatus::Ok;
    }
    let bytes = if len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }.to_vec()
    };
    unsafe { *out = Box::into_raw(Box::new(TestString(bytes))).cast() };
    STRING_LIVE_COUNT.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn test_string_data(value: *const NemoRelayNativeString) -> *const u8 {
    if value.is_null() {
        return ptr::null();
    }
    unsafe { &*(value.cast::<TestString>()) }.0.as_ptr()
}

unsafe extern "C" fn test_string_len(value: *const NemoRelayNativeString) -> usize {
    if value.is_null() {
        return 0;
    }
    unsafe { &*(value.cast::<TestString>()) }.0.len()
}

unsafe extern "C" fn test_string_free(value: *mut NemoRelayNativeString) {
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value.cast::<TestString>()) });
        STRING_LIVE_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn test_last_error_clear() {
    *LAST_ERROR.lock().unwrap() = None;
}

unsafe extern "C" fn test_last_error_set(message: *const NemoRelayNativeString) {
    let host = test_host();
    *LAST_ERROR.lock().unwrap() = read_host_string(&host, message);
}

unsafe extern "C" fn capture_register_subscriber(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    cb: NemoRelayNativeEventSubscriberCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &SUBSCRIBER_REGISTRATION,
            RegisteredSubscriber {
                name,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_tool_json(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &TOOL_JSON_REGISTRATION,
            RegisteredToolJson {
                name,
                priority,
                break_chain: false,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn passthrough_tool_json_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _payload_json: *const NemoRelayNativeString,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_event_sanitize_cb(
    _user_data: *mut c_void,
    _event_json: *const NemoRelayNativeString,
    _fields_json: *const NemoRelayNativeString,
    _out_fields_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_tool_conditional(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeToolConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &TOOL_CONDITIONAL_REGISTRATION,
            RegisteredToolConditional {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_tool_request_intercept(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &TOOL_JSON_REGISTRATION,
            RegisteredToolJson {
                name,
                priority,
                break_chain,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_tool_execution(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeToolExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &TOOL_EXECUTION_REGISTRATION,
            RegisteredToolExecution {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_request(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeRequestCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_REQUEST_REGISTRATION,
            RegisteredLlmRequest {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_json(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeResponseCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_JSON_REGISTRATION,
            RegisteredLlmJson {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_conditional(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_CONDITIONAL_REGISTRATION,
            RegisteredLlmConditional {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_request_intercept(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeLlmRequestInterceptCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_REQUEST_INTERCEPT_REGISTRATION,
            RegisteredLlmRequestIntercept {
                name,
                priority,
                break_chain,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_stream_execution(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmStreamExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_STREAM_EXECUTION_REGISTRATION,
            RegisteredLlmStreamExecution {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_llm_execution(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &LLM_EXECUTION_REGISTRATION,
            RegisteredLlmExecution {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}

unsafe extern "C" fn capture_scope_get_current(
    out: *mut *mut NemoRelayNativeScopeHandle,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_GET_CURRENT_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    RUNTIME_CALLS.lock().unwrap().push("current_scope".into());
    if *SCOPE_GET_CURRENT_RETURNS_NULL.lock().unwrap() {
        unsafe { *out = ptr::null_mut() };
    } else {
        unsafe { *out = Box::into_raw(Box::new(0_u8)).cast() };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_push(
    name: *const NemoRelayNativeString,
    scope_type: NemoRelayNativeScopeType,
    parent: *const NemoRelayNativeScopeHandle,
    attributes: u32,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    input_json: *const NemoRelayNativeString,
    _timestamp_unix_micros: *const i64,
    out: *mut *mut NemoRelayNativeScopeHandle,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_PUSH_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    let name = match required_host_string(&host, name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let data = match optional_host_string(&host, data_json) {
        Ok(data) => data,
        Err(status) => return status,
    };
    let metadata = match optional_host_string(&host, metadata_json) {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    let input = match optional_host_string(&host, input_json) {
        Ok(input) => input,
        Err(status) => return status,
    };
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "push:{name}:{scope_type:?}:{attributes}:parent={}:data={data}:metadata={metadata}:input={input}",
        !parent.is_null()
    ));
    if *SCOPE_PUSH_RETURNS_NULL.lock().unwrap() {
        unsafe { *out = ptr::null_mut() };
    } else {
        unsafe { *out = Box::into_raw(Box::new(0_u8)).cast() };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_pop(
    handle: *const NemoRelayNativeScopeHandle,
    output_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    _timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    if handle.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_POP_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    let output = match optional_host_string(&host, output_json) {
        Ok(output) => output,
        Err(status) => return status,
    };
    let metadata = match optional_host_string(&host, metadata_json) {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    RUNTIME_CALLS
        .lock()
        .unwrap()
        .push(format!("pop:output={output}:metadata={metadata}"));
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_emit_mark(
    name: *const NemoRelayNativeString,
    parent: *const NemoRelayNativeScopeHandle,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    _timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    let status = *EMIT_MARK_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    let name = match required_host_string(&host, name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let data = match optional_host_string(&host, data_json) {
        Ok(data) => data,
        Err(status) => return status,
    };
    let metadata = match optional_host_string(&host, metadata_json) {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "mark:{name}:parent={}:data={data}:metadata={metadata}",
        !parent.is_null()
    ));
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_stack_create(
    out: *mut *mut NemoRelayNativeScopeStack,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_STACK_CREATE_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    RUNTIME_CALLS.lock().unwrap().push("stack_create".into());
    if *SCOPE_STACK_CREATE_RETURNS_NULL.lock().unwrap() {
        unsafe { *out = ptr::null_mut() };
    } else {
        unsafe { *out = Box::into_raw(Box::new(0_u8)).cast() };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_stack_set_thread(
    stack: *const NemoRelayNativeScopeStack,
) -> NemoRelayStatus {
    if stack.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_STACK_SET_THREAD_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    RUNTIME_CALLS
        .lock()
        .unwrap()
        .push("stack_set_thread".into());
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_stack_capture_thread(
    out: *mut *mut NemoRelayNativeScopeStackBinding,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_STACK_CAPTURE_THREAD_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    RUNTIME_CALLS.lock().unwrap().push("stack_capture".into());
    if *SCOPE_STACK_CAPTURE_THREAD_RETURNS_NULL.lock().unwrap() {
        unsafe { *out = ptr::null_mut() };
    } else {
        unsafe { *out = Box::into_raw(Box::new(0_u8)).cast() };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_scope_stack_restore_thread(
    binding: *mut NemoRelayNativeScopeStackBinding,
) -> NemoRelayStatus {
    if binding.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_STACK_RESTORE_THREAD_STATUS.lock().unwrap();
    RUNTIME_CALLS.lock().unwrap().push("stack_restore".into());
    unsafe { drop(Box::from_raw(binding.cast::<u8>())) };
    SCOPE_STACK_BINDING_RESTORES.fetch_add(1, Ordering::SeqCst);
    status
}

unsafe extern "C" fn capture_scope_stack_with_current(
    stack: *const NemoRelayNativeScopeStack,
    cb: NemoRelayNativeWithScopeStackCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    if stack.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let status = *SCOPE_STACK_WITH_CURRENT_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    RUNTIME_CALLS
        .lock()
        .unwrap()
        .push("stack_with_current".into());
    unsafe { cb(user_data) }
}

unsafe extern "C" fn capture_scope_handle_free(handle: *mut NemoRelayNativeScopeHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle.cast::<u8>())) };
        SCOPE_HANDLE_FREES.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn capture_event_sanitize(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeEventSanitizeCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status == NemoRelayStatus::Ok {
        let host = test_host();
        let name = match required_host_string(&host, name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        replace_registration(
            &EVENT_SANITIZE_REGISTRATION,
            RegisteredEventSanitize {
                name,
                priority,
                cb,
                user_data: user_data as usize,
                free_fn,
            },
        );
    }
    status
}
unsafe extern "C" fn capture_scope_stack_free(stack: *mut NemoRelayNativeScopeStack) {
    if !stack.is_null() {
        unsafe { drop(Box::from_raw(stack.cast::<u8>())) };
        SCOPE_STACK_FREES.fetch_add(1, Ordering::SeqCst);
    }
}
unsafe extern "C" fn capture_scope_stack_binding_free(
    binding: *mut NemoRelayNativeScopeStackBinding,
) {
    if !binding.is_null() {
        unsafe { drop(Box::from_raw(binding.cast::<u8>())) };
        SCOPE_STACK_BINDING_FREES.fetch_add(1, Ordering::SeqCst);
    }
}
unsafe extern "C" fn true_scope_stack_active() -> bool {
    true
}

fn test_host() -> NemoRelayNativeHostApiV1 {
    NemoRelayNativeHostApiV1 {
        abi_version: NEMO_RELAY_NATIVE_ABI_VERSION,
        struct_size: std::mem::size_of::<NemoRelayNativeHostApiV1>(),
        relay_version: c"test".as_ptr(),
        string_new: test_string_new,
        string_data: test_string_data,
        string_len: test_string_len,
        string_free: test_string_free,
        last_error_clear: test_last_error_clear,
        last_error_set: test_last_error_set,
        plugin_context_register_subscriber: capture_register_subscriber,
        plugin_context_register_tool_sanitize_request_guardrail: capture_tool_json,
        plugin_context_register_tool_sanitize_response_guardrail: capture_tool_json,
        plugin_context_register_tool_conditional_execution_guardrail: capture_tool_conditional,
        plugin_context_register_tool_request_intercept: capture_tool_request_intercept,
        plugin_context_register_tool_execution_intercept: capture_tool_execution,
        plugin_context_register_llm_sanitize_request_guardrail: capture_llm_request,
        plugin_context_register_llm_sanitize_response_guardrail: capture_llm_json,
        plugin_context_register_llm_conditional_execution_guardrail: capture_llm_conditional,
        plugin_context_register_llm_request_intercept: capture_llm_request_intercept,
        plugin_context_register_llm_execution_intercept: capture_llm_execution,
        plugin_context_register_llm_stream_execution_intercept: capture_llm_stream_execution,
        scope_handle_free: capture_scope_handle_free,
        scope_get_current: capture_scope_get_current,
        scope_push: capture_scope_push,
        scope_pop: capture_scope_pop,
        emit_mark: capture_emit_mark,
        scope_stack_create: capture_scope_stack_create,
        scope_stack_free: capture_scope_stack_free,
        scope_stack_set_thread: capture_scope_stack_set_thread,
        scope_stack_capture_thread: capture_scope_stack_capture_thread,
        scope_stack_restore_thread: capture_scope_stack_restore_thread,
        scope_stack_binding_free: capture_scope_stack_binding_free,
        scope_stack_active: true_scope_stack_active,
        scope_stack_with_current: capture_scope_stack_with_current,
        plugin_context_register_mark_sanitize_guardrail: capture_event_sanitize,
        plugin_context_register_scope_sanitize_start_guardrail: capture_event_sanitize,
        plugin_context_register_scope_sanitize_end_guardrail: capture_event_sanitize,
    }
}

fn begin_test() -> MutexGuard<'static, ()> {
    let guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reset_state();
    guard
}

fn reset_state() {
    clear_registration(&SUBSCRIBER_REGISTRATION);
    clear_registration(&EVENT_SANITIZE_REGISTRATION);
    clear_registration(&TOOL_JSON_REGISTRATION);
    clear_registration(&TOOL_CONDITIONAL_REGISTRATION);
    clear_registration(&TOOL_EXECUTION_REGISTRATION);
    clear_registration(&LLM_REQUEST_REGISTRATION);
    clear_registration(&LLM_JSON_REGISTRATION);
    clear_registration(&LLM_CONDITIONAL_REGISTRATION);
    clear_registration(&LLM_EXECUTION_REGISTRATION);
    clear_registration(&LLM_STREAM_EXECUTION_REGISTRATION);
    clear_registration(&LLM_REQUEST_INTERCEPT_REGISTRATION);
    assert_eq!(
        STRING_LIVE_COUNT.load(Ordering::SeqCst),
        0,
        "previous test leaked host strings"
    );
    *LAST_ERROR.lock().unwrap() = None;
    *REGISTRATION_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    *STRING_NEW_RETURNS_NULL.lock().unwrap() = false;
    *SCOPE_GET_CURRENT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_GET_CURRENT_RETURNS_NULL.lock().unwrap() = false;
    *SCOPE_PUSH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_PUSH_RETURNS_NULL.lock().unwrap() = false;
    *SCOPE_POP_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *EMIT_MARK_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_STACK_CREATE_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_STACK_CREATE_RETURNS_NULL.lock().unwrap() = false;
    *SCOPE_STACK_SET_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_STACK_CAPTURE_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_STACK_CAPTURE_THREAD_RETURNS_NULL.lock().unwrap() = false;
    *SCOPE_STACK_RESTORE_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SCOPE_STACK_WITH_CURRENT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    RUNTIME_CALLS.lock().unwrap().clear();
    SCOPE_HANDLE_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_BINDING_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_BINDING_RESTORES.store(0, Ordering::SeqCst);
}

fn test_context(host: &NemoRelayNativeHostApiV1) -> PluginContext<'_> {
    unsafe {
        PluginContext::from_raw(
            host,
            NonNull::<NemoRelayNativePluginContext>::dangling().as_ptr(),
        )
    }
}

fn read_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let data = unsafe { (host.string_data)(value) };
    let len = unsafe { (host.string_len)(value) };
    if data.is_null() && len > 0 {
        return None;
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    std::str::from_utf8(bytes).ok().map(ToOwned::to_owned)
}

fn required_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> std::result::Result<String, NemoRelayStatus> {
    if value.is_null() {
        return Err(NemoRelayStatus::NullPointer);
    }
    read_host_string(host, value).ok_or(NemoRelayStatus::InvalidArg)
}

fn optional_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> std::result::Result<String, NemoRelayStatus> {
    if value.is_null() {
        return Ok(String::new());
    }
    read_host_string(host, value).ok_or(NemoRelayStatus::InvalidArg)
}

fn host_string(host: &NemoRelayNativeHostApiV1, value: &str) -> *mut NemoRelayNativeString {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut out) };
    assert_eq!(status, NemoRelayStatus::Ok);
    out
}

fn bytes_host_string(host: &NemoRelayNativeHostApiV1, value: &[u8]) -> *mut NemoRelayNativeString {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut out) };
    assert_eq!(status, NemoRelayStatus::Ok);
    out
}

fn json_host_string(host: &NemoRelayNativeHostApiV1, value: Json) -> *mut NemoRelayNativeString {
    host_string(host, &serde_json::to_string(&value).unwrap())
}

fn read_json_and_free(host: &NemoRelayNativeHostApiV1, value: *mut NemoRelayNativeString) -> Json {
    let result: Json = serde_json::from_str(&read_host_string(host, value).unwrap()).unwrap();
    unsafe { (host.string_free)(value) };
    result
}

fn read_string_and_free(
    host: &NemoRelayNativeHostApiV1,
    value: *mut NemoRelayNativeString,
) -> String {
    let result = read_host_string(host, value).unwrap();
    unsafe { (host.string_free)(value) };
    result
}

fn live_host_strings() -> usize {
    STRING_LIVE_COUNT.load(Ordering::SeqCst)
}

fn expect_string_err<T>(result: std::result::Result<T, String>) -> String {
    match result {
        Ok(_) => panic!("operation should have failed"),
        Err(error) => error,
    }
}

fn poll_stream_chunk(
    host: &NemoRelayNativeHostApiV1,
    stream: &NemoRelayNativeLlmStreamV1,
) -> (NemoRelayStatus, Option<Json>) {
    let mut out = ptr::null_mut();
    let status = unsafe { stream.next.unwrap()(stream.user_data, &mut out) };
    let chunk = if out.is_null() {
        None
    } else {
        Some(read_json_and_free(host, out))
    };
    (status, chunk)
}

unsafe fn drop_stream(stream: &mut NemoRelayNativeLlmStreamV1) {
    if let Some(drop_fn) = stream.drop.take() {
        unsafe { drop_fn(stream.user_data) };
    }
    stream.user_data = ptr::null_mut();
}

unsafe extern "C" fn count_stream_drop(user_data: *mut c_void) {
    if !user_data.is_null() {
        unsafe { (&*(user_data as *const AtomicUsize)).fetch_add(1, Ordering::SeqCst) };
    }
}

fn write_json(
    host: &NemoRelayNativeHostApiV1,
    value: &Json,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let encoded = serde_json::to_string(value).unwrap();
    let mut string = ptr::null_mut();
    let status = unsafe { (host.string_new)(encoded.as_ptr(), encoded.len(), &mut string) };
    if status == NemoRelayStatus::Ok {
        unsafe { *out = string };
    }
    status
}

fn take_tool_json_registration() -> RegisteredToolJson {
    TOOL_JSON_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("tool JSON callback should be registered")
}

fn take_subscriber_registration() -> RegisteredSubscriber {
    SUBSCRIBER_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("subscriber callback should be registered")
}

fn take_event_sanitize_registration() -> RegisteredEventSanitize {
    EVENT_SANITIZE_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("event sanitize callback should be registered")
}

fn take_tool_conditional_registration() -> RegisteredToolConditional {
    TOOL_CONDITIONAL_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("tool conditional callback should be registered")
}

fn take_tool_execution_registration() -> RegisteredToolExecution {
    TOOL_EXECUTION_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("tool execution callback should be registered")
}

fn take_llm_request_registration() -> RegisteredLlmRequest {
    LLM_REQUEST_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM request callback should be registered")
}

fn take_llm_json_registration() -> RegisteredLlmJson {
    LLM_JSON_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM JSON callback should be registered")
}

fn take_llm_conditional_registration() -> RegisteredLlmConditional {
    LLM_CONDITIONAL_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM conditional callback should be registered")
}

fn take_llm_execution_registration() -> RegisteredLlmExecution {
    LLM_EXECUTION_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM execution callback should be registered")
}

fn take_llm_request_intercept_registration() -> RegisteredLlmRequestIntercept {
    LLM_REQUEST_INTERCEPT_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM request intercept callback should be registered")
}

fn take_llm_stream_execution_registration() -> RegisteredLlmStreamExecution {
    LLM_STREAM_EXECUTION_REGISTRATION
        .lock()
        .unwrap()
        .take()
        .expect("LLM stream execution callback should be registered")
}

struct PanicOnDrop(&'static str);

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!("{}", self.0);
    }
}

struct PanicIterator {
    _panic_on_drop: PanicOnDrop,
}

impl Iterator for PanicIterator {
    type Item = std::result::Result<Json, String>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[test]
fn llm_stream_from_raw_drops_rejected_streams() {
    let _guard = begin_test();
    let host = test_host();

    let undersized_drop_calls = AtomicUsize::new(0);
    let wrong_size = NemoRelayNativeLlmStreamV1 {
        struct_size: 0,
        user_data: (&undersized_drop_calls as *const AtomicUsize)
            .cast_mut()
            .cast(),
        next: None,
        cancel: None,
        drop: Some(count_stream_drop),
    };
    let err = match unsafe { LlmStream::from_raw(&host, wrong_size) } {
        Ok(_) => panic!("undersized stream should be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("unsupported LLM stream struct size"));
    assert_eq!(undersized_drop_calls.load(Ordering::SeqCst), 0);

    let dropped = Arc::new(AtomicUsize::new(0));
    let mut wrong_size = test_llm_stream(
        &host,
        vec![],
        Arc::new(AtomicUsize::new(0)),
        dropped.clone(),
    );
    wrong_size.struct_size = size_of::<NemoRelayNativeLlmStreamV1>() + 8;
    let err = match unsafe { LlmStream::from_raw(&host, wrong_size) } {
        Ok(_) => panic!("oversized stream should be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("unsupported LLM stream struct size"));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);

    let dropped = Arc::new(AtomicUsize::new(0));
    let mut null_next = test_llm_stream(
        &host,
        vec![],
        Arc::new(AtomicUsize::new(0)),
        dropped.clone(),
    );
    null_next.next = None;
    let err = match unsafe { LlmStream::from_raw(&host, null_next) } {
        Ok(_) => panic!("null-next stream should be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("LLM stream next callback was null"));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn llm_stream_from_raw_polls_iterates_cancels_and_drops() {
    let _guard = begin_test();
    let host = test_host();
    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let raw = manual_llm_stream(
        &host,
        vec![
            ManualStreamPoll::Json(json!({ "chunk": 1 })),
            ManualStreamPoll::Json(json!({ "chunk": 2 })),
            ManualStreamPoll::EndWithJson(json!({ "ignored": true })),
        ],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();

    assert_eq!(stream.next_chunk().unwrap().unwrap()["chunk"], json!(1));
    assert_eq!(stream.next().unwrap().unwrap()["chunk"], json!(2));
    assert!(stream.next().is_none());
    assert!(stream.next_chunk().unwrap().is_none());
    assert!(stream.cancel().is_ok());
    drop(stream);

    assert_eq!(cancelled.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn llm_stream_from_raw_reports_chunk_and_status_failures() {
    let _guard = begin_test();
    let host = test_host();
    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));

    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::NullOk],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    assert_eq!(
        stream.next_chunk().unwrap_err(),
        "LLM stream returned null chunk"
    );
    assert!(stream.next_chunk().unwrap().is_none());
    drop(stream);
    assert_eq!(cancelled.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);

    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::InvalidJson],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    assert_eq!(
        stream.next().unwrap().unwrap_err(),
        "LLM stream returned invalid JSON: InvalidJson"
    );
    assert!(stream.next().is_none());
    drop(stream);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);

    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::StatusWithJson(
            NemoRelayStatus::GuardrailRejected,
            json!({ "discarded": true }),
        )],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    let live_before = live_host_strings();
    assert_eq!(
        stream.next_chunk().unwrap_err(),
        "LLM stream failed: GuardrailRejected"
    );
    assert_eq!(live_host_strings(), live_before);
    drop(stream);
    assert_eq!(dropped.load(Ordering::SeqCst), 3);

    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::Status(NemoRelayStatus::NotFound)],
        NemoRelayStatus::Ok,
        cancelled,
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    assert_eq!(
        stream.next().unwrap().unwrap_err(),
        "LLM stream failed: NotFound"
    );
    drop(stream);
    assert_eq!(dropped.load(Ordering::SeqCst), 4);
}

#[test]
fn llm_stream_cancel_handles_finished_missing_and_failing_callbacks() {
    let _guard = begin_test();
    let host = test_host();

    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::Json(json!({ "chunk": true }))],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    stream.cancel().unwrap();
    stream.cancel().unwrap();
    drop(stream);
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);

    let mut raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::Json(json!({ "chunk": true }))],
        NemoRelayStatus::Ok,
        cancelled.clone(),
        dropped.clone(),
    );
    raw.cancel = None;
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    stream.cancel().unwrap();
    drop(stream);
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);

    let raw = manual_llm_stream(
        &host,
        vec![ManualStreamPoll::Json(json!({ "chunk": true }))],
        NemoRelayStatus::Internal,
        cancelled.clone(),
        dropped.clone(),
    );
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    assert_eq!(
        stream.cancel().unwrap_err(),
        "LLM stream cancel failed: Internal"
    );
    drop(stream);
    assert_eq!(cancelled.load(Ordering::SeqCst), 3);
    assert_eq!(dropped.load(Ordering::SeqCst), 3);
}

#[test]
fn plugin_runtime_scope_mark_and_stack_helpers_call_host() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);
    assert_eq!(
        runtime.host_api().abi_version,
        NEMO_RELAY_NATIVE_ABI_VERSION
    );

    let current = runtime.current_scope().unwrap();
    assert!(!current.as_ptr().is_null());
    drop(current);

    let mut scope = runtime
        .scope(
            "work",
            ScopeType::Tool,
            Some(&json!({ "data": true })),
            Some(&json!({ "metadata": true })),
            Some(&json!({ "input": true })),
        )
        .unwrap();
    assert!(scope.handle().is_some());
    runtime
        .emit_mark(
            "checkpoint",
            Some(&json!({ "mark": true })),
            Some(&json!({ "meta": true })),
        )
        .unwrap();
    scope
        .close(
            Some(&json!({ "output": true })),
            Some(&json!({ "closed": true })),
        )
        .unwrap();
    assert!(scope.handle().is_none());
    scope.close(None, None).unwrap();

    let stack = runtime.create_scope_stack().unwrap();
    assert!(runtime.scope_stack_active());
    let with_current_calls = Arc::new(AtomicUsize::new(0));
    stack
        .with_current({
            let with_current_calls = with_current_calls.clone();
            move || {
                with_current_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .unwrap();
    assert_eq!(with_current_calls.load(Ordering::SeqCst), 1);
    runtime
        .bind_scope_stack_thread(&stack)
        .unwrap()
        .restore()
        .unwrap();
    drop(stack);

    let calls = RUNTIME_CALLS.lock().unwrap().clone();
    assert!(calls.iter().any(|call| call == "current_scope"));
    assert!(calls.iter().any(|call| {
        call.starts_with("push:work:Tool:0:parent=false")
            && call.contains(r#""data":true"#)
            && call.contains(r#""metadata":true"#)
            && call.contains(r#""input":true"#)
    }));
    assert!(calls.iter().any(|call| {
        call.starts_with("mark:checkpoint:parent=false")
            && call.contains(r#""mark":true"#)
            && call.contains(r#""meta":true"#)
    }));
    assert!(calls.iter().any(|call| {
        call.starts_with("pop:")
            && call.contains(r#""output":true"#)
            && call.contains(r#""closed":true"#)
    }));
    assert!(calls.iter().any(|call| call == "stack_create"));
    assert!(calls.iter().any(|call| call == "stack_with_current"));
    assert!(calls.iter().any(|call| call == "stack_capture"));
    assert!(calls.iter().any(|call| call == "stack_set_thread"));
    assert!(calls.iter().any(|call| call == "stack_restore"));
    assert_eq!(SCOPE_HANDLE_FREES.load(Ordering::SeqCst), 2);
    assert_eq!(SCOPE_STACK_FREES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 0);
}

#[test]
fn scope_guard_drops_unclosed_scope_and_maps_scope_types() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);

    assert_eq!(
        [
            NemoRelayNativeScopeType::from(ScopeType::Agent),
            NemoRelayNativeScopeType::from(ScopeType::Function),
            NemoRelayNativeScopeType::from(ScopeType::Tool),
            NemoRelayNativeScopeType::from(ScopeType::Llm),
            NemoRelayNativeScopeType::from(ScopeType::Retriever),
            NemoRelayNativeScopeType::from(ScopeType::Embedder),
            NemoRelayNativeScopeType::from(ScopeType::Reranker),
            NemoRelayNativeScopeType::from(ScopeType::Guardrail),
            NemoRelayNativeScopeType::from(ScopeType::Evaluator),
            NemoRelayNativeScopeType::from(ScopeType::Custom),
            NemoRelayNativeScopeType::from(ScopeType::Unknown),
        ],
        [
            NemoRelayNativeScopeType::Agent,
            NemoRelayNativeScopeType::Function,
            NemoRelayNativeScopeType::Tool,
            NemoRelayNativeScopeType::Llm,
            NemoRelayNativeScopeType::Retriever,
            NemoRelayNativeScopeType::Embedder,
            NemoRelayNativeScopeType::Reranker,
            NemoRelayNativeScopeType::Guardrail,
            NemoRelayNativeScopeType::Evaluator,
            NemoRelayNativeScopeType::Custom,
            NemoRelayNativeScopeType::Unknown,
        ]
    );

    {
        let scope = runtime
            .scope("auto", ScopeType::Agent, None, None, None)
            .unwrap();
        assert!(scope.handle().is_some());
    }

    let calls = RUNTIME_CALLS.lock().unwrap().clone();
    assert!(calls.iter().any(|call| call.starts_with("push:auto:Agent")));
    assert!(calls.iter().any(|call| call == "pop:output=:metadata="));
    assert_eq!(SCOPE_HANDLE_FREES.load(Ordering::SeqCst), 1);
}

#[test]
fn plugin_runtime_reports_scope_host_failures_and_allocation_failures() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);

    *SCOPE_GET_CURRENT_STATUS.lock().unwrap() = NemoRelayStatus::NotFound;
    assert_eq!(
        expect_string_err(runtime.current_scope()),
        "scope_get_current failed: NotFound"
    );
    *SCOPE_GET_CURRENT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SCOPE_GET_CURRENT_RETURNS_NULL.lock().unwrap() = true;
    assert_eq!(
        expect_string_err(runtime.current_scope()),
        "scope_get_current failed: Ok"
    );
    *SCOPE_GET_CURRENT_RETURNS_NULL.lock().unwrap() = false;

    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    assert_eq!(
        expect_string_err(runtime.push_scope("scope", ScopeType::Tool, None, None, None)),
        "failed to allocate scope name"
    );
    assert_eq!(
        runtime.emit_mark("mark", None, None).unwrap_err(),
        "failed to allocate mark name"
    );
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;

    *SCOPE_PUSH_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    assert_eq!(
        expect_string_err(runtime.push_scope("scope", ScopeType::Tool, None, None, None)),
        "scope_push failed: InvalidArg"
    );
    *SCOPE_PUSH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SCOPE_PUSH_RETURNS_NULL.lock().unwrap() = true;
    assert_eq!(
        expect_string_err(runtime.push_scope("scope", ScopeType::Tool, None, None, None)),
        "scope_push failed: Ok"
    );
    *SCOPE_PUSH_RETURNS_NULL.lock().unwrap() = false;

    *EMIT_MARK_STATUS.lock().unwrap() = NemoRelayStatus::Internal;
    assert_eq!(
        runtime.emit_mark("mark", None, None).unwrap_err(),
        "emit_mark failed: Internal"
    );
    *EMIT_MARK_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    let handle = runtime
        .push_scope("scope", ScopeType::Tool, None, None, None)
        .unwrap();
    *SCOPE_POP_STATUS.lock().unwrap() = NemoRelayStatus::ScopeStackEmpty;
    assert_eq!(
        runtime.pop_scope(&handle, None, None).unwrap_err(),
        "scope_pop failed: ScopeStackEmpty"
    );
    *SCOPE_POP_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    drop(handle);

    *SCOPE_STACK_CREATE_STATUS.lock().unwrap() = NemoRelayStatus::Internal;
    assert_eq!(
        expect_string_err(runtime.create_scope_stack()),
        "scope_stack_create failed: Internal"
    );
    *SCOPE_STACK_CREATE_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SCOPE_STACK_CREATE_RETURNS_NULL.lock().unwrap() = true;
    assert_eq!(
        expect_string_err(runtime.create_scope_stack()),
        "scope_stack_create failed: Ok"
    );
    *SCOPE_STACK_CREATE_RETURNS_NULL.lock().unwrap() = false;

    *SCOPE_STACK_CAPTURE_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::NotFound;
    assert_eq!(
        expect_string_err(runtime.capture_scope_stack_thread()),
        "scope_stack_capture_thread failed: NotFound"
    );
    *SCOPE_STACK_CAPTURE_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SCOPE_STACK_CAPTURE_THREAD_RETURNS_NULL.lock().unwrap() = true;
    assert_eq!(
        expect_string_err(runtime.capture_scope_stack_thread()),
        "scope_stack_capture_thread failed: Ok"
    );
    *SCOPE_STACK_CAPTURE_THREAD_RETURNS_NULL.lock().unwrap() = false;

    *STRING_NEW_RETURNS_NULL.lock().unwrap() = true;
    assert_eq!(
        runtime.emit_mark("mark", None, None).unwrap_err(),
        "failed to allocate mark name"
    );
    *STRING_NEW_RETURNS_NULL.lock().unwrap() = false;
}

#[test]
fn scope_stack_with_current_reports_callback_and_host_failures() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);
    let stack = runtime.create_scope_stack().unwrap();
    assert!(!stack.as_ptr().is_null());

    assert_eq!(
        stack
            .with_current(|| Err("scope stack callback failed".into()))
            .unwrap_err(),
        "scope stack callback failed"
    );
    assert_eq!(
        stack
            .with_current(|| panic!("scope stack panic"))
            .unwrap_err(),
        "scope-stack callback panicked"
    );

    *SCOPE_STACK_WITH_CURRENT_STATUS.lock().unwrap() = NemoRelayStatus::NotFound;
    assert_eq!(
        stack.with_current(|| Ok(())).unwrap_err(),
        "scope_stack_with_current failed: NotFound"
    );
}

#[test]
fn scope_stack_thread_binding_restores_on_set_failure_and_reports_restore_failure() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);
    let stack = runtime.create_scope_stack().unwrap();

    *SCOPE_STACK_SET_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    assert_eq!(
        expect_string_err(runtime.bind_scope_stack_thread(&stack)),
        "scope_stack_set_thread failed: InvalidArg"
    );
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 1);
    *SCOPE_STACK_SET_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SCOPE_STACK_RESTORE_THREAD_STATUS.lock().unwrap() = NemoRelayStatus::Internal;
    let guard = runtime.bind_scope_stack_thread(&stack).unwrap();
    assert_eq!(
        guard.restore().unwrap_err(),
        "scope_stack_restore_thread failed: Internal"
    );
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 2);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 0);
}

#[test]
fn scope_stack_bindings_restore_or_free_on_drop() {
    let _guard = begin_test();
    let host = test_host();
    let runtime = PluginRuntime::new(&host);
    let stack = runtime.create_scope_stack().unwrap();

    {
        let _guard = runtime.bind_scope_stack_thread(&stack).unwrap();
    }
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 0);

    let binding = runtime.capture_scope_stack_thread().unwrap();
    drop(binding);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 1);
}

#[test]
fn typed_subscriber_registration_decodes_events() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_subscriber("events", {
        let called = called.clone();
        move |event: &Event| {
            assert_eq!(event.kind(), "mark");
            called.fetch_add(1, Ordering::SeqCst);
        }
    })
    .unwrap();

    let registration = take_subscriber_registration();
    assert_eq!(registration.name, "events");
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    let status = unsafe { (registration.cb)(registration.user_data as *mut c_void, event) };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(called.load(Ordering::SeqCst), 1);

    unsafe {
        (host.string_free)(event);
        registration.free();
    }
}

#[test]
fn repeated_captured_registration_frees_previous_callback_state() {
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let _guard = begin_test();
    let host = test_host();
    let drops = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);

    ctx.register_subscriber("first", {
        let counter = DropCounter(drops.clone());
        move |_event: &Event| {
            let _ = &counter;
        }
    })
    .unwrap();
    ctx.register_subscriber("second", {
        let counter = DropCounter(drops.clone());
        move |_event: &Event| {
            let _ = &counter;
        }
    })
    .unwrap();

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let registration = take_subscriber_registration();
    assert_eq!(registration.name, "second");
    unsafe { registration.free() };
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn typed_tool_sanitize_guardrails_transform_payloads() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_sanitize_request_guardrail("tool-sanitize-request", 4, |name, mut args| {
        assert_eq!(name, "tool");
        args["surface"] = json!("request");
        args
    })
    .unwrap();

    let registration = take_tool_json_registration();
    assert_eq!(registration.name, "tool-sanitize-request");
    assert_eq!(registration.priority, 4);
    assert!(!registration.break_chain);
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["surface"], json!("request"));
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        5,
        |name, mut value| {
            assert_eq!(name, "tool");
            value["surface"] = json!("response");
            value
        },
    )
    .unwrap();

    let registration = take_tool_json_registration();
    assert_eq!(registration.name, "tool-sanitize-response");
    assert_eq!(registration.priority, 5);
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({ "output": true }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["surface"], json!("response"));
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }
}

#[test]
fn typed_event_sanitize_guardrails_transform_fields_for_every_surface() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 4, |event, mut fields| {
        assert_eq!(event.name(), "checkpoint");
        fields.data = Some(json!({"clean": true}));
        fields.category_profile = None;
        fields.metadata = Some(json!({"source": "native"}));
        fields
    })
    .unwrap();

    let registration = take_event_sanitize_registration();
    assert_eq!(registration.name, "mark-sanitize");
    assert_eq!(registration.priority, 4);
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint",
            "data": {"secret": "raw"}
        }),
    );
    let fields = json_host_string(
        &host,
        serde_json::to_value(EventSanitizeFields {
            data: Some(json!({"secret": "raw"})),
            category_profile: Some(CategoryProfile::builder().subtype("raw").build()),
            metadata: None,
        })
        .unwrap(),
    );
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            event,
            fields,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let output: EventSanitizeFields =
        serde_json::from_value(read_json_and_free(&host, out)).unwrap();
    assert_eq!(output.data, Some(json!({"clean": true})));
    assert!(output.category_profile.is_none());
    assert_eq!(output.metadata, Some(json!({"source": "native"})));
    unsafe {
        (host.string_free)(event);
        (host.string_free)(fields);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_scope_sanitize_start_guardrail("scope-start-sanitize", 5, |_, fields| fields)
        .unwrap();
    let registration = take_event_sanitize_registration();
    assert_eq!(registration.name, "scope-start-sanitize");
    assert_eq!(registration.priority, 5);
    unsafe { registration.free() };

    let mut ctx = test_context(&host);
    ctx.register_scope_sanitize_end_guardrail("scope-end-sanitize", 6, |_, fields| fields)
        .unwrap();
    let registration = take_event_sanitize_registration();
    assert_eq!(registration.name, "scope-end-sanitize");
    assert_eq!(registration.priority, 6);
    unsafe { registration.free() };
}

#[test]
fn typed_json_callbacks_report_output_allocation_failures() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_sanitize_request_guardrail("tool-sanitize", 0, |_name, value| value)
        .unwrap();

    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());

    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 0, |_, fields| fields)
        .unwrap();
    let registration = take_event_sanitize_registration();
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    let fields = json_host_string(&host, json!({}));
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            event,
            fields,
            &mut out,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    unsafe {
        (host.string_free)(event);
        (host.string_free)(fields);
        registration.free();
    }
}

#[test]
fn typed_event_sanitize_callback_catches_panics() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 0, |_, _| {
        panic!("event sanitizer panic")
    })
    .unwrap();
    let registration = take_event_sanitize_registration();
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    let fields = json_host_string(&host, json!({}));
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                event,
                fields,
                &mut out,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("event sanitize callback panicked")
    );
    unsafe {
        (host.string_free)(event);
        (host.string_free)(fields);
        registration.free();
    }
}

#[test]
fn typed_tool_conditional_guardrail_returns_optional_reason() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_conditional_execution_guardrail("tool-conditional", 8, |name, args| {
        assert_eq!(name, "tool");
        if args["block"].as_bool().unwrap_or(false) {
            Ok(Some("blocked by policy".into()))
        } else {
            Ok(None)
        }
    })
    .unwrap();

    let registration = take_tool_conditional_registration();
    assert_eq!(registration.name, "tool-conditional");
    assert_eq!(registration.priority, 8);
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "block": false }));
    let sentinel = host_string(&host, "sentinel");
    let mut reason = sentinel;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            &mut reason,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert!(reason.is_null());
    unsafe {
        (host.string_free)(sentinel);
        (host.string_free)(args);
    }

    let args = json_host_string(&host, json!({ "block": true }));
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            &mut reason,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_string_and_free(&host, reason), "blocked by policy");
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        registration.free();
    }
}

#[test]
fn typed_tool_intercept_registration_rewrites_json() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept("tool", 17, true, |_name, mut value| {
        value["typed"] = json!(true);
        Ok(value)
    })
    .unwrap();

    let registration = take_tool_json_registration();
    assert_eq!(registration.name, "tool");
    assert_eq!(registration.priority, 17);
    assert!(registration.break_chain);
    let name = host_string(&host, "");
    let payload = json_host_string(&host, json!({ "input": "value" }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["typed"], json!(true));
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }
}

#[test]
fn typed_tool_intercept_registration_reports_invalid_json() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept("tool", 0, false, |_name, value| Ok(value))
        .unwrap();

    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = host_string(&host, "{not json");
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::InvalidJson);
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }
}

#[test]
fn typed_tool_intercept_reports_null_inputs_separately_from_invalid_utf8() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept("tool", 0, false, |_name, value| Ok(value))
        .unwrap();

    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            ptr::null(),
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::NullPointer);
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool name was null")
    );
    unsafe { (host.string_free)(stale_out) };

    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            ptr::null(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::NullPointer);
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool payload was null")
    );
    unsafe { (host.string_free)(stale_out) };

    let invalid_name = bytes_host_string(&host, b"\xff");
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invalid_name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::InvalidUtf8);
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool name contained invalid UTF-8")
    );

    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(invalid_name);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }
}

#[test]
fn typed_tool_intercept_registration_maps_callback_errors_and_panics() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept("tool", 0, false, |_name, _value| {
        Err("callback failed".into())
    })
    .unwrap();

    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("callback failed")
    );
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept(
        "tool",
        0,
        false,
        |_name, _value| -> Result<Json, String> { panic!("boom") },
    )
    .unwrap();
    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            payload,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool intercept callback panicked")
    );
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }
}

#[test]
fn typed_callback_free_catches_drop_panics() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    let panic_on_drop = PanicOnDrop("typed callback drop panic");
    ctx.register_tool_request_intercept("tool", 0, false, move |_name, value| {
        let _ = &panic_on_drop;
        Ok(value)
    })
    .unwrap();

    let registration = take_tool_json_registration();
    *LAST_ERROR.lock().unwrap() = None;
    unsafe { registration.free() };
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native plugin typed callback state drop panicked")
    );
}

#[test]
fn typed_callbacks_reject_null_abi_pointers_before_decoding_inputs() {
    let _guard = begin_test();
    let host = test_host();

    let mut ctx = test_context(&host);
    ctx.register_subscriber("events", |_event: &Event| {})
        .unwrap();
    let registration = take_subscriber_registration();
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), event) },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(event);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 0, |_, fields| fields)
        .unwrap();
    let registration = take_event_sanitize_registration();
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    let fields = json_host_string(&host, json!({}));
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), event, fields, &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                ptr::null(),
                fields,
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                event,
                ptr::null(),
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                event,
                fields,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(event);
        (host.string_free)(fields);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_sanitize_request_guardrail("tool-sanitize", 0, |_name, value| value)
        .unwrap();
    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), name, payload, &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_request_intercept("tool", 0, false, |_name, value| Ok(value))
        .unwrap();
    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), name, payload, &mut out) },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_conditional_execution_guardrail("tool-conditional", 0, |_name, _value| {
        Ok(None)
    })
    .unwrap();
    let registration = take_tool_conditional_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let mut reason = ptr::null_mut();
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), name, payload, &mut reason) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool-exec", 0, |_name, value, _next| Ok(value.into()))
        .unwrap();
    let registration = take_tool_execution_registration();
    let name = host_string(&host, "tool");
    let payload = json_host_string(&host, json!({}));
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: Arc::new(AtomicUsize::new(0)),
    }));
    assert_eq!(
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                name,
                payload,
                fake_tool_next,
                next_state.cast(),
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                fake_tool_next,
                next_state.cast(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(payload);
        drop(Box::from_raw(next_state));
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail("llm-request", 0, |request, _context| {
        Some(request)
    })
    .unwrap();
    let registration = take_llm_request_registration();
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), request, ptr::null(), &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                request,
                ptr::null(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail("llm-response", 0, |value, _context| Some(value))
        .unwrap();
    let registration = take_llm_json_registration();
    let response = json_host_string(&host, json!({}));
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), response, ptr::null(), &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                response,
                ptr::null(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(response);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 0, |_request| Ok(None))
        .unwrap();
    let registration = take_llm_conditional_registration();
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    assert_eq!(
        unsafe { (registration.cb)(ptr::null_mut(), request, &mut reason) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                request,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm-request-intercept", 0, false, |_name, request, ann| {
        Ok(LlmRequestInterceptOutcome::new(request, ann))
    })
    .unwrap();
    let registration = take_llm_request_intercept_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out_outcome = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                name,
                request,
                ptr::null(),
                &mut out_outcome,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                ptr::null(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept(
        "llm-exec",
        0,
        |_name, request, _next| Ok(request.content),
    )
    .unwrap();
    let registration = take_llm_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: Arc::new(AtomicUsize::new(0)),
    }));
    assert_eq!(
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                name,
                request,
                failing_llm_next,
                next_state.cast(),
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                failing_llm_next,
                next_state.cast(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept("llm-stream", 0, |_name, _request, _next| {
        Ok(Box::new(std::iter::empty()))
    })
    .unwrap();
    let registration = take_llm_stream_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let next_state = Box::into_raw(Box::new(StreamNextState {
        host,
        called: Arc::new(AtomicUsize::new(0)),
        cancelled: Arc::new(AtomicUsize::new(0)),
        dropped: Arc::new(AtomicUsize::new(0)),
    }));
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                name,
                request,
                fake_llm_stream_next,
                next_state.cast(),
                &mut stream,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                fake_llm_stream_next,
                next_state.cast(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_callbacks_report_invalid_json_for_each_decoder_family() {
    let _guard = begin_test();
    let host = test_host();

    let mut ctx = test_context(&host);
    ctx.register_subscriber("events", |_event: &Event| {})
        .unwrap();
    let registration = take_subscriber_registration();
    let event = host_string(&host, "{not json");
    assert_eq!(
        unsafe { (registration.cb)(registration.user_data as *mut c_void, event) },
        NemoRelayStatus::InvalidJson
    );
    unsafe {
        (host.string_free)(event);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 0, |_, fields| fields)
        .unwrap();
    let registration = take_event_sanitize_registration();
    let invalid_event = host_string(&host, "{not json");
    let fields = json_host_string(&host, json!({}));
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                invalid_event,
                fields,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    let event = json_host_string(
        &host,
        json!({
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "00000000-0000-0000-0000-000000000000",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "checkpoint"
        }),
    );
    let invalid_fields = host_string(&host, "{not json");
    out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                event,
                invalid_fields,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(invalid_event);
        (host.string_free)(fields);
        (host.string_free)(event);
        (host.string_free)(invalid_fields);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_sanitize_request_guardrail("tool-sanitize", 0, |_name, value| value)
        .unwrap();
    let registration = take_tool_json_registration();
    let name = host_string(&host, "tool");
    let payload = host_string(&host, "{not json");
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_conditional_execution_guardrail("tool-conditional", 0, |_name, _value| {
        Ok(None)
    })
    .unwrap();
    let registration = take_tool_conditional_registration();
    let name = host_string(&host, "tool");
    let payload = host_string(&host, "{not json");
    let stale_reason = host_string(&host, "stale");
    let mut reason = stale_reason;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                &mut reason,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(reason.is_null());
    unsafe {
        (host.string_free)(stale_reason);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool-exec", 0, |_name, value, _next| Ok(value.into()))
        .unwrap();
    let registration = take_tool_execution_registration();
    let name = host_string(&host, "tool");
    let payload = host_string(&host, "{not json");
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                payload,
                fake_tool_next,
                ptr::null_mut(),
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(payload);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail("llm-request", 0, |request, _context| {
        Some(request)
    })
    .unwrap();
    let registration = take_llm_request_registration();
    let request = host_string(&host, "{not json");
    let context = json_host_string(
        &host,
        serde_json::to_value(LlmSanitizeContext::default()).unwrap(),
    );
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                request,
                context,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(request);
        (host.string_free)(context);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail("llm-response", 0, |value, _context| Some(value))
        .unwrap();
    let registration = take_llm_json_registration();
    let response = host_string(&host, "{not json");
    let context = json_host_string(
        &host,
        serde_json::to_value(LlmSanitizeContext::default()).unwrap(),
    );
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                response,
                context,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(response);
        (host.string_free)(context);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 0, |_request| Ok(None))
        .unwrap();
    let registration = take_llm_conditional_registration();
    let request = host_string(&host, "{not json");
    let stale_reason = host_string(&host, "stale");
    let mut reason = stale_reason;
    assert_eq!(
        unsafe { (registration.cb)(registration.user_data as *mut c_void, request, &mut reason) },
        NemoRelayStatus::InvalidJson
    );
    assert!(reason.is_null());
    unsafe {
        (host.string_free)(stale_reason);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm-request", 0, false, |_name, request, ann| {
        Ok(LlmRequestInterceptOutcome::new(request, ann))
    })
    .unwrap();
    let registration = take_llm_request_intercept_registration();
    let name = host_string(&host, "llm");
    let bad_request = host_string(&host, "{not json");
    let mut out_outcome = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                bad_request,
                ptr::null(),
                &mut out_outcome,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let bad_annotation = host_string(&host, "{not json");
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                bad_annotation,
                &mut out_outcome,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(bad_request);
        (host.string_free)(request);
        (host.string_free)(bad_annotation);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept(
        "llm-exec",
        0,
        |_name, request, _next| Ok(request.content),
    )
    .unwrap();
    let registration = take_llm_execution_registration();
    let name = host_string(&host, "llm");
    let request = host_string(&host, "{not json");
    let stale_out = host_string(&host, r#"{"stale":true}"#);
    let mut out = stale_out;
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                failing_llm_next,
                ptr::null_mut(),
                &mut out,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(out.is_null());
    unsafe {
        (host.string_free)(stale_out);
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept("llm-stream", 0, |_name, _request, _next| {
        Ok(Box::new(std::iter::empty()))
    })
    .unwrap();
    let registration = take_llm_stream_execution_registration();
    let name = host_string(&host, "llm");
    let request = host_string(&host, "{not json");
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                fake_llm_stream_next,
                ptr::null_mut(),
                &mut stream,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }
}

#[test]
fn typed_callbacks_map_additional_callback_errors() {
    let _guard = begin_test();
    let host = test_host();

    let mut ctx = test_context(&host);
    ctx.register_tool_conditional_execution_guardrail("tool-conditional", 0, |_name, _value| {
        Err("tool conditional failed".into())
    })
    .unwrap();
    let registration = take_tool_conditional_registration();
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({}));
    let mut reason = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                args,
                &mut reason,
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool conditional failed")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool-exec", 0, |_name, _value, _next| {
        Err("tool execution failed".into())
    })
    .unwrap();
    let registration = take_tool_execution_registration();
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({}));
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                args,
                fake_tool_next,
                ptr::null_mut(),
                &mut out,
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool execution failed")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 0, |_request| {
        Err("llm conditional failed".into())
    })
    .unwrap();
    let registration = take_llm_conditional_registration();
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut reason = ptr::null_mut();
    assert_eq!(
        unsafe { (registration.cb)(registration.user_data as *mut c_void, request, &mut reason) },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm conditional failed")
    );
    unsafe {
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm-request", 0, false, |_name, _request, _ann| {
        Err("llm request failed".into())
    })
    .unwrap();
    let registration = take_llm_request_intercept_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out_outcome = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                ptr::null(),
                &mut out_outcome,
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm request failed")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept("llm-exec", 0, |_name, _request, _next| {
        Err("llm execution failed".into())
    })
    .unwrap();
    let registration = take_llm_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                failing_llm_next,
                ptr::null_mut(),
                &mut out,
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm execution failed")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept(
        "llm-exec",
        0,
        |_name, request, _next| Ok(request.content),
    )
    .unwrap();
    let registration = take_llm_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                failing_llm_next,
                ptr::null_mut(),
                &mut out,
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(read_json_and_free(&host, out), json!({ "input": true }));
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept("llm-stream", 0, |_name, _request, _next| {
        Err("llm stream failed".into())
    })
    .unwrap();
    let registration = take_llm_stream_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                name,
                request,
                fake_llm_stream_next,
                ptr::null_mut(),
                &mut stream,
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm stream failed")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }
}

struct NextState {
    host: NemoRelayNativeHostApiV1,
    called: Arc<AtomicUsize>,
}

unsafe extern "C" fn fake_tool_next(
    args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    let mut args: Json =
        serde_json::from_str(&read_host_string(&state.host, args_json).unwrap()).unwrap();
    args["next_called"] = json!(true);
    write_json(&state.host, &args, out_json)
}

unsafe extern "C" fn failing_tool_next(
    _args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::GuardrailRejected
}

unsafe extern "C" fn invalid_json_tool_next(
    _args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    let invalid = b"{not json";
    unsafe { (state.host.string_new)(invalid.as_ptr(), invalid.len(), out_json) }
}

unsafe extern "C" fn null_tool_next(
    _args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    unsafe { *out_json = ptr::null_mut() };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn failing_llm_next(
    _request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::GuardrailRejected
}

unsafe extern "C" fn invalid_json_llm_next(
    _request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    let invalid = b"{not json";
    unsafe { (state.host.string_new)(invalid.as_ptr(), invalid.len(), out_json) }
}

unsafe extern "C" fn null_llm_next(
    _request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const NextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    unsafe { *out_json = ptr::null_mut() };
    NemoRelayStatus::Ok
}

struct StreamNextState {
    host: NemoRelayNativeHostApiV1,
    called: Arc<AtomicUsize>,
    cancelled: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

struct TestLlmStreamState {
    host: NemoRelayNativeHostApiV1,
    chunks: Mutex<VecDeque<std::result::Result<Json, String>>>,
    cancelled: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

fn test_llm_stream(
    host: &NemoRelayNativeHostApiV1,
    chunks: Vec<std::result::Result<Json, String>>,
    cancelled: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
) -> NemoRelayNativeLlmStreamV1 {
    let state = Box::new(TestLlmStreamState {
        host: *host,
        chunks: Mutex::new(VecDeque::from(chunks)),
        cancelled,
        dropped,
    });
    NemoRelayNativeLlmStreamV1 {
        struct_size: size_of::<NemoRelayNativeLlmStreamV1>(),
        user_data: Box::into_raw(state).cast(),
        next: Some(poll_test_llm_stream),
        cancel: Some(cancel_test_llm_stream),
        drop: Some(drop_test_llm_stream),
    }
}

unsafe extern "C" fn poll_test_llm_stream(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TestLlmStreamState) };
    let mut chunks = state.chunks.lock().unwrap();
    match chunks.pop_front() {
        Some(Ok(chunk)) => write_json(&state.host, &chunk, out_json),
        Some(Err(message)) => {
            let message = host_string(&state.host, &message);
            unsafe {
                (state.host.last_error_set)(message);
                (state.host.string_free)(message);
            }
            NemoRelayStatus::Internal
        }
        None => NemoRelayStatus::StreamEnd,
    }
}

unsafe extern "C" fn cancel_test_llm_stream(user_data: *mut c_void) -> NemoRelayStatus {
    if user_data.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const TestLlmStreamState) };
    state.cancelled.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn drop_test_llm_stream(user_data: *mut c_void) {
    if !user_data.is_null() {
        let state = unsafe { Box::from_raw(user_data as *mut TestLlmStreamState) };
        state.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn fake_llm_stream_next(
    _request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    if out_stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(next_ctx as *const StreamNextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    unsafe {
        *out_stream = test_llm_stream(
            &state.host,
            vec![Ok(json!({ "chunk": 1 })), Ok(json!({ "chunk": 2 }))],
            state.cancelled.clone(),
            state.dropped.clone(),
        )
    };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn failing_llm_stream_next(
    _request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    _out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    let state = unsafe { &*(next_ctx as *const StreamNextState) };
    state.called.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::GuardrailRejected
}

enum ManualStreamPoll {
    Json(Json),
    InvalidJson,
    NullOk,
    Status(NemoRelayStatus),
    StatusWithJson(NemoRelayStatus, Json),
    End,
    EndWithJson(Json),
}

struct ManualStreamState {
    host: NemoRelayNativeHostApiV1,
    polls: Mutex<VecDeque<ManualStreamPoll>>,
    cancel_status: NemoRelayStatus,
    cancelled: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

fn manual_llm_stream(
    host: &NemoRelayNativeHostApiV1,
    polls: Vec<ManualStreamPoll>,
    cancel_status: NemoRelayStatus,
    cancelled: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
) -> NemoRelayNativeLlmStreamV1 {
    let state = Box::new(ManualStreamState {
        host: *host,
        polls: Mutex::new(VecDeque::from(polls)),
        cancel_status,
        cancelled,
        dropped,
    });
    NemoRelayNativeLlmStreamV1 {
        struct_size: size_of::<NemoRelayNativeLlmStreamV1>(),
        user_data: Box::into_raw(state).cast(),
        next: Some(poll_manual_llm_stream),
        cancel: Some(cancel_manual_llm_stream),
        drop: Some(drop_manual_llm_stream),
    }
}

unsafe extern "C" fn poll_manual_llm_stream(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const ManualStreamState) };
    match state
        .polls
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or(ManualStreamPoll::End)
    {
        ManualStreamPoll::Json(value) => write_json(&state.host, &value, out_json),
        ManualStreamPoll::InvalidJson => {
            let invalid = b"{not json";
            unsafe { (state.host.string_new)(invalid.as_ptr(), invalid.len(), out_json) }
        }
        ManualStreamPoll::NullOk => NemoRelayStatus::Ok,
        ManualStreamPoll::Status(status) => status,
        ManualStreamPoll::StatusWithJson(status, value) => {
            let write_status = write_json(&state.host, &value, out_json);
            if write_status == NemoRelayStatus::Ok {
                status
            } else {
                write_status
            }
        }
        ManualStreamPoll::End => NemoRelayStatus::StreamEnd,
        ManualStreamPoll::EndWithJson(value) => {
            let write_status = write_json(&state.host, &value, out_json);
            if write_status == NemoRelayStatus::Ok {
                NemoRelayStatus::StreamEnd
            } else {
                write_status
            }
        }
    }
}

unsafe extern "C" fn cancel_manual_llm_stream(user_data: *mut c_void) -> NemoRelayStatus {
    if user_data.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const ManualStreamState) };
    state.cancelled.fetch_add(1, Ordering::SeqCst);
    state.cancel_status
}

unsafe extern "C" fn drop_manual_llm_stream(user_data: *mut c_void) {
    if !user_data.is_null() {
        let state = unsafe { Box::from_raw(user_data as *mut ManualStreamState) };
        state.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn typed_tool_execution_registration_calls_next() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool", 23, |_name, args, next: ToolNext<'_>| {
        let result = next.call(args)?;
        Ok(
            ToolExecutionInterceptOutcome::new(result).with_pending_mark(
                PendingMarkSpec::builder()
                    .name("plugin.tool.completed")
                    .category(EventCategory::custom())
                    .category_profile(CategoryProfile {
                        subtype: Some("plugin.tool.pending".into()),
                        ..CategoryProfile::default()
                    })
                    .data(json!({ "saved_tokens": 7 }))
                    .metadata(json!({ "source": "typed-test" }))
                    .build(),
            ),
        )
    })
    .unwrap();

    let registration = take_tool_execution_registration();
    assert_eq!(registration.name, "tool");
    assert_eq!(registration.priority, 23);
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            fake_tool_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(called.load(Ordering::SeqCst), 1);
    let outcome = read_json_and_free(&host, out);
    assert_eq!(outcome["result"]["next_called"], json!(true));
    assert_eq!(outcome["pending_marks"][0]["name"], "plugin.tool.completed");
    assert_eq!(outcome["pending_marks"][0]["category"], "custom");
    assert_eq!(
        outcome["pending_marks"][0]["category_profile"]["subtype"],
        "plugin.tool.pending"
    );
    assert_eq!(outcome["pending_marks"][0]["data"]["saved_tokens"], 7);
    assert_eq!(
        outcome["pending_marks"][0]["metadata"]["source"],
        "typed-test"
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_tool_execution_does_not_publish_partial_outcome() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool", 0, |_name, args, _next| {
        Ok(ToolExecutionInterceptOutcome::new(args))
    })
    .unwrap();

    let registration = take_tool_execution_registration();
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "input": true }));
    let stale_outcome = host_string(&host, r#"{"stale":true}"#);
    let mut out_outcome = stale_outcome;
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let live_before = live_host_strings();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            fake_tool_next,
            ptr::null_mut(),
            &mut out_outcome,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out_outcome.is_null());
    assert_eq!(live_host_strings(), live_before);
    unsafe {
        (host.string_free)(stale_outcome);
        (host.string_free)(name);
        (host.string_free)(args);
        registration.free();
    }
}

#[test]
fn typed_tool_execution_surfaces_next_status_failures() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool", 0, |_name, args, next: ToolNext<'_>| {
        next.call(args).map(Into::into)
    })
    .unwrap();

    let registration = take_tool_execution_registration();
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            failing_tool_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool next failed: GuardrailRejected")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_tool_execution_surfaces_invalid_next_json() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool", 0, |_name, args, next: ToolNext<'_>| {
        next.call(args).map(Into::into)
    })
    .unwrap();

    let registration = take_tool_execution_registration();
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    let live_before = live_host_strings();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            invalid_json_tool_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(live_host_strings(), live_before);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool next returned invalid JSON: InvalidJson")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_tool_execution_surfaces_null_next_output() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_tool_execution_intercept("tool", 0, |_name, args, next: ToolNext<'_>| {
        next.call(args).map(Into::into)
    })
    .unwrap();

    let registration = take_tool_execution_registration();
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "tool");
    let args = json_host_string(&host, json!({ "input": true }));
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            args,
            null_tool_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("tool next returned null output")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(args);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_sanitize_guardrails_transform_request_and_response() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        12,
        |mut request, _context| {
            request.headers.insert("x-policy".into(), json!("sdk"));
            request.content["sanitized"] = json!(true);
            Some(request)
        },
    )
    .unwrap();

    let registration = take_llm_request_registration();
    assert_eq!(registration.name, "llm-sanitize-request");
    assert_eq!(registration.priority, 12);
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let context = json_host_string(
        &host,
        serde_json::to_value(LlmSanitizeContext::default()).unwrap(),
    );
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            request,
            context,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let output = read_json_and_free(&host, out);
    assert_eq!(output["headers"]["x-policy"], json!("sdk"));
    assert_eq!(output["content"]["sanitized"], json!(true));
    unsafe {
        (host.string_free)(request);
        (host.string_free)(context);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        13,
        |mut payload, _context| {
            payload["sanitized"] = json!(true);
            Some(payload)
        },
    )
    .unwrap();

    let registration = take_llm_json_registration();
    assert_eq!(registration.name, "llm-sanitize-response");
    assert_eq!(registration.priority, 13);
    let response = json_host_string(&host, json!({ "output": true }));
    let context = json_host_string(
        &host,
        serde_json::to_value(LlmSanitizeContext::default()).unwrap(),
    );
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            response,
            context,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["sanitized"], json!(true));
    unsafe {
        (host.string_free)(response);
        (host.string_free)(context);
        registration.free();
    }
}

#[test]
fn typed_contextual_llm_sanitize_guardrails_receive_payload_before_context() {
    let _guard = begin_test();
    let host = test_host();
    let context = LlmSanitizeContext {
        has_active_codec: true,
        codec_name: Some("openai_chat".into()),
    };

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail(
        "contextual-request",
        14,
        |mut request, callback_context| {
            assert!(callback_context.has_active_codec);
            assert_eq!(callback_context.codec_name.as_deref(), Some("openai_chat"));
            request.headers.insert("x-contextual".into(), json!(true));
            Some(request)
        },
    )
    .unwrap();

    let registration = take_llm_request_registration();
    assert_eq!(registration.name, "contextual-request");
    assert_eq!(registration.priority, 14);
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let context_json = json_host_string(&host, serde_json::to_value(&context).unwrap());
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            request,
            context_json,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(
        read_json_and_free(&host, out)["headers"]["x-contextual"],
        json!(true)
    );
    unsafe {
        (host.string_free)(request);
        (host.string_free)(context_json);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail(
        "contextual-response",
        15,
        |mut payload, callback_context| {
            assert_eq!(callback_context.codec_name.as_deref(), Some("openai_chat"));
            payload["contextual"] = json!(true);
            Some(payload)
        },
    )
    .unwrap();

    let registration = take_llm_json_registration();
    assert_eq!(registration.name, "contextual-response");
    assert_eq!(registration.priority, 15);
    let response = json_host_string(&host, json!({ "output": true }));
    let context_json = json_host_string(&host, serde_json::to_value(context).unwrap());
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            response,
            context_json,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["contextual"], json!(true));
    unsafe {
        (host.string_free)(response);
        (host.string_free)(context_json);
        registration.free();
    }
}

#[test]
fn typed_contextual_llm_sanitizer_uses_null_output_to_omit_payload() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail("contextual-omit", 16, |_payload, _context| None)
        .unwrap();

    let registration = take_llm_json_registration();
    let response = json_host_string(&host, json!({"secret": "value"}));
    let context = json_host_string(
        &host,
        serde_json::to_value(LlmSanitizeContext::default()).unwrap(),
    );
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            response,
            context,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert!(out.is_null(), "null native output must represent omission");
    unsafe {
        (host.string_free)(response);
        (host.string_free)(context);
        registration.free();
    }
}

#[test]
fn typed_llm_conditional_guardrail_returns_optional_reason() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 14, |request| {
        if request.content["block"].as_bool().unwrap_or(false) {
            Ok(Some("LLM blocked".into()))
        } else {
            Ok(None)
        }
    })
    .unwrap();

    let registration = take_llm_conditional_registration();
    assert_eq!(registration.name, "llm-conditional");
    assert_eq!(registration.priority, 14);
    let request = json_host_string(
        &host,
        serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({ "block": false }),
        })
        .unwrap(),
    );
    let sentinel = host_string(&host, "sentinel");
    let mut reason = sentinel;
    let status =
        unsafe { (registration.cb)(registration.user_data as *mut c_void, request, &mut reason) };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert!(reason.is_null());
    unsafe {
        (host.string_free)(sentinel);
        (host.string_free)(request);
    }

    let request = json_host_string(
        &host,
        serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({ "block": true }),
        })
        .unwrap(),
    );
    let status =
        unsafe { (registration.cb)(registration.user_data as *mut c_void, request, &mut reason) };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_string_and_free(&host, reason), "LLM blocked");
    unsafe {
        (host.string_free)(request);
        registration.free();
    }
}

#[test]
fn typed_llm_execution_surfaces_next_status_failures() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept("llm", 0, |_name, request, next: LlmNext<'_>| {
        next.call(request)
    })
    .unwrap();

    let registration = take_llm_execution_registration();
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            failing_llm_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm next failed: GuardrailRejected")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_execution_surfaces_invalid_next_json() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept("llm", 0, |_name, request, next: LlmNext<'_>| {
        next.call(request)
    })
    .unwrap();

    let registration = take_llm_execution_registration();
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out = ptr::null_mut();
    let live_before = live_host_strings();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            invalid_json_llm_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(live_host_strings(), live_before);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm next returned invalid JSON: InvalidJson")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_execution_surfaces_null_next_output() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_execution_intercept("llm", 31, |_name, request, next: LlmNext<'_>| {
        next.call(request)
    })
    .unwrap();

    let registration = take_llm_execution_registration();
    assert_eq!(registration.name, "llm");
    assert_eq!(registration.priority, 31);
    let next_state = Box::into_raw(Box::new(NextState {
        host,
        called: called.clone(),
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            null_llm_next,
            next_state.cast(),
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out.is_null());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm next returned null output")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_stream_execution_wraps_next_chunks() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        31,
        |_name, request, next: LlmStreamNext<'_>| {
            let stream = next.call(request)?;
            let stream: LlmJsonStream = Box::new(stream.map(|chunk| {
                chunk.map(|mut chunk| {
                    chunk["wrapped"] = json!(true);
                    chunk
                })
            }));
            Ok(stream)
        },
    )
    .unwrap();

    let registration = take_llm_stream_execution_registration();
    assert_eq!(registration.name, "llm-stream");
    assert_eq!(registration.priority, 31);
    let next_state = Box::into_raw(Box::new(StreamNextState {
        host,
        called: called.clone(),
        cancelled: cancelled.clone(),
        dropped: dropped.clone(),
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            fake_llm_stream_next,
            next_state.cast(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(called.load(Ordering::SeqCst), 1);

    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { stream.next.unwrap()(ptr::null_mut(), &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { stream.next.unwrap()(stream.user_data, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );

    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(chunk.unwrap()["wrapped"], json!(true));
    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::Ok);
    let chunk = chunk.unwrap();
    assert_eq!(chunk["chunk"], json!(2));
    assert_eq!(chunk["wrapped"], json!(true));
    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::StreamEnd);
    assert!(chunk.is_none());
    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::StreamEnd);
    assert!(chunk.is_none());
    assert_eq!(
        unsafe { stream.cancel.unwrap()(stream.user_data) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { stream.cancel.unwrap()(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );

    unsafe {
        drop_stream(&mut stream);
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
    assert_eq!(cancelled.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn typed_llm_stream_drop_catches_stream_state_panics() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept("llm-stream", 0, |_name, _request, _next| {
        let stream: LlmJsonStream = Box::new(PanicIterator {
            _panic_on_drop: PanicOnDrop("LLM stream state drop panic"),
        });
        Ok(stream)
    })
    .unwrap();

    let registration = take_llm_stream_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            fake_llm_stream_next,
            ptr::null_mut(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);

    *LAST_ERROR.lock().unwrap() = None;
    unsafe {
        drop_stream(&mut stream);
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native plugin LLM stream state drop panicked")
    );
}

#[test]
fn typed_llm_stream_execution_surfaces_next_failures() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        0,
        |_name, request, next: LlmStreamNext<'_>| {
            let stream = next.call(request)?;
            Ok(Box::new(stream))
        },
    )
    .unwrap();

    let registration = take_llm_stream_execution_registration();
    let next_state = Box::into_raw(Box::new(StreamNextState {
        host,
        called: called.clone(),
        cancelled,
        dropped,
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            failing_llm_stream_next,
            next_state.cast(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::Internal);
    assert_eq!(
        stream.struct_size,
        NemoRelayNativeLlmStreamV1::default().struct_size
    );
    assert!(stream.user_data.is_null());
    assert!(stream.next.is_none());
    assert!(stream.cancel.is_none());
    assert!(stream.drop.is_none());
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("llm stream next failed: GuardrailRejected")
    );
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_stream_execution_surfaces_chunk_errors() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept("llm-stream", 0, |_name, _request, _next| {
        let stream: LlmJsonStream = Box::new(std::iter::once(Err("chunk failed".into())));
        Ok(stream)
    })
    .unwrap();

    let registration = take_llm_stream_execution_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let next_state = Box::into_raw(Box::new(StreamNextState {
        host,
        called: Arc::new(AtomicUsize::new(0)),
        cancelled: Arc::new(AtomicUsize::new(0)),
        dropped: Arc::new(AtomicUsize::new(0)),
    }));
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            fake_llm_stream_next,
            next_state.cast(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(chunk.is_none());
    assert_eq!(LAST_ERROR.lock().unwrap().as_deref(), Some("chunk failed"));

    unsafe {
        drop_stream(&mut stream);
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

#[test]
fn typed_llm_stream_execution_cancels_unconsumed_next_stream() {
    let _guard = begin_test();
    let host = test_host();
    let called = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut ctx = test_context(&host);
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        0,
        |_name, request, next: LlmStreamNext<'_>| {
            let stream = next.call(request)?;
            drop(stream);
            let stream: LlmJsonStream = Box::new(std::iter::empty());
            Ok(stream)
        },
    )
    .unwrap();

    let registration = take_llm_stream_execution_registration();
    let next_state = Box::into_raw(Box::new(StreamNextState {
        host,
        called: called.clone(),
        cancelled: cancelled.clone(),
        dropped: dropped.clone(),
    }));
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut stream = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            fake_llm_stream_next,
            next_state.cast(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(called.load(Ordering::SeqCst), 1);
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let (status, chunk) = poll_stream_chunk(&host, &stream);
    assert_eq!(status, NemoRelayStatus::StreamEnd);
    assert!(chunk.is_none());

    unsafe {
        drop_stream(&mut stream);
        (host.string_free)(name);
        (host.string_free)(request);
        drop(Box::from_raw(next_state));
        registration.free();
    }
}

fn test_llm_request() -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({ "input": true }),
    }
}

fn test_annotated_llm_request() -> AnnotatedLlmRequest {
    serde_json::from_value(json!({ "messages": [] })).unwrap()
}

#[test]
fn typed_llm_request_intercept_does_not_publish_partial_outputs() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm", 0, false, |_name, request, _annotated| {
        Ok(LlmRequestInterceptOutcome::new(
            request,
            Some(test_annotated_llm_request()),
        ))
    })
    .unwrap();

    let registration = take_llm_request_intercept_registration();
    assert_eq!(registration.name, "llm");
    assert_eq!(registration.priority, 0);
    assert!(!registration.break_chain);
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let stale_outcome = host_string(&host, r#"{"stale":"outcome"}"#);
    let mut out_outcome = stale_outcome;
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let live_before = live_host_strings();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            ptr::null(),
            &mut out_outcome,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out_outcome.is_null());
    assert_eq!(live_host_strings(), live_before);
    unsafe {
        (host.string_free)(stale_outcome);
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm", 0, false, |_name, request, _annotated| {
        Ok(LlmRequestInterceptOutcome::new(request, None))
    })
    .unwrap();

    let registration = take_llm_request_intercept_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out_outcome = ptr::null_mut();
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            ptr::null(),
            &mut out_outcome,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(out_outcome.is_null());
    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }
}

#[test]
fn typed_llm_request_intercept_round_trips_request_and_annotations() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm", 19, true, |name, mut request, annotated| {
        assert_eq!(name, "llm");
        assert!(annotated.is_some());
        request.headers.insert("x-mutated".into(), json!(true));
        request.content["rewritten"] = json!(true);
        Ok(LlmRequestInterceptOutcome::new(
            request,
            Some(test_annotated_llm_request()),
        ))
    })
    .unwrap();

    let registration = take_llm_request_intercept_registration();
    assert_eq!(registration.name, "llm");
    assert_eq!(registration.priority, 19);
    assert!(registration.break_chain);
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let annotated = json_host_string(
        &host,
        serde_json::to_value(test_annotated_llm_request()).unwrap(),
    );
    let mut out_outcome = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            annotated,
            &mut out_outcome,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let outcome = read_json_and_free(&host, out_outcome);
    assert_eq!(outcome["request"]["headers"]["x-mutated"], json!(true));
    assert_eq!(outcome["request"]["content"]["rewritten"], json!(true));
    assert_eq!(outcome["annotated_request"]["messages"], json!([]));
    assert_eq!(outcome["pending_marks"], json!([]));

    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        (host.string_free)(annotated);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm", 0, false, |_name, request, _annotated| {
        Ok(LlmRequestInterceptOutcome::new(request, None))
    })
    .unwrap();
    let registration = take_llm_request_intercept_registration();
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let mut out_outcome = host_string(&host, r#"{"stale":true}"#);
    let stale_outcome = out_outcome;
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            ptr::null(),
            &mut out_outcome,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let outcome = read_json_and_free(&host, out_outcome);
    assert!(outcome["annotated_request"].is_null());
    assert_eq!(outcome["request"]["content"]["input"], json!(true));
    assert_eq!(outcome["pending_marks"], json!([]));
    unsafe {
        (host.string_free)(stale_outcome);
        (host.string_free)(name);
        (host.string_free)(request);
        registration.free();
    }
}

#[test]
fn typed_llm_request_intercept_serializes_canonical_outcome() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_request_intercept("llm", 23, false, |_name, mut request, annotated| {
        request.content["rewritten"] = json!(true);
        Ok(
            LlmRequestInterceptOutcome::new(request, annotated).with_pending_mark(
                PendingMarkSpec::builder()
                    .name("plugin.request.rewritten")
                    .data(json!({ "saved_tokens": 7 }))
                    .build(),
            ),
        )
    })
    .unwrap();

    let registration = take_llm_request_intercept_registration();
    assert_eq!(registration.priority, 23);
    assert!(!registration.break_chain);
    let name = host_string(&host, "llm");
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let annotated = json_host_string(
        &host,
        serde_json::to_value(test_annotated_llm_request()).unwrap(),
    );
    let mut out_outcome = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            name,
            request,
            annotated,
            &mut out_outcome,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    let outcome = read_json_and_free(&host, out_outcome);
    assert_eq!(outcome["request"]["content"]["rewritten"], true);
    assert_eq!(outcome["annotated_request"]["messages"], json!([]));
    assert_eq!(
        outcome["pending_marks"][0]["name"],
        "plugin.request.rewritten"
    );
    assert_eq!(outcome["pending_marks"][0]["data"]["saved_tokens"], 7);

    unsafe {
        (host.string_free)(name);
        (host.string_free)(request);
        (host.string_free)(annotated);
        registration.free();
    }
}

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn failed_typed_registration_drops_callback_state() {
    let _guard = begin_test();
    let host = test_host();
    *REGISTRATION_STATUS.lock().unwrap() = NemoRelayStatus::AlreadyExists;
    let drops = Arc::new(AtomicUsize::new(0));
    let drop_counter = DropCounter(drops.clone());
    let mut ctx = test_context(&host);
    let result = ctx.register_tool_request_intercept("duplicate", 0, false, move |_name, value| {
        let _keep_alive = &drop_counter;
        Ok(value)
    });

    assert!(result.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(TOOL_JSON_REGISTRATION.lock().unwrap().is_none());
}

#[test]
fn raw_registration_propagates_name_allocation_status() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let status = unsafe {
        ctx.register_tool_request_intercept_raw(
            "tool",
            0,
            false,
            passthrough_tool_json_cb,
            ptr::null_mut(),
            None,
        )
    };
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;

    assert_eq!(status, NemoRelayStatus::Internal);
    assert!(TOOL_JSON_REGISTRATION.lock().unwrap().is_none());
}

#[test]
fn raw_event_sanitize_registrations_cover_every_surface() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);

    assert_eq!(
        unsafe {
            ctx.register_mark_sanitize_guardrail_raw(
                "raw-mark",
                1,
                passthrough_event_sanitize_cb,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let registration = take_event_sanitize_registration();
    assert_eq!(
        (registration.name.as_str(), registration.priority),
        ("raw-mark", 1)
    );
    unsafe { registration.free() };

    assert_eq!(
        unsafe {
            ctx.register_scope_sanitize_start_guardrail_raw(
                "raw-scope-start",
                2,
                passthrough_event_sanitize_cb,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let registration = take_event_sanitize_registration();
    assert_eq!(
        (registration.name.as_str(), registration.priority),
        ("raw-scope-start", 2)
    );
    unsafe { registration.free() };

    assert_eq!(
        unsafe {
            ctx.register_scope_sanitize_end_guardrail_raw(
                "raw-scope-end",
                3,
                passthrough_event_sanitize_cb,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let registration = take_event_sanitize_registration();
    assert_eq!(
        (registration.name.as_str(), registration.priority),
        ("raw-scope-end", 3)
    );
    unsafe { registration.free() };
}

#[test]
fn typed_registration_name_allocation_failure_drops_callback_state() {
    let _guard = begin_test();
    let host = test_host();
    let drops = Arc::new(AtomicUsize::new(0));
    let drop_counter = DropCounter(drops.clone());
    let mut ctx = test_context(&host);
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let result = ctx.register_tool_request_intercept("tool", 0, false, move |_name, value| {
        let _keep_alive = &drop_counter;
        Ok(value)
    });
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;

    assert!(result.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(TOOL_JSON_REGISTRATION.lock().unwrap().is_none());
}

struct ConstructorPanicPlugin;

impl NativePlugin for ConstructorPanicPlugin {
    fn plugin_kind(&self) -> &str {
        "test.constructor_panic"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

static CONSTRUCTOR_CALLS: AtomicUsize = AtomicUsize::new(0);

struct CountingPlugin;

impl NativePlugin for CountingPlugin {
    fn plugin_kind(&self) -> &str {
        "test.counting"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

struct DiagnosticsPlugin;

impl NativePlugin for DiagnosticsPlugin {
    fn plugin_kind(&self) -> &str {
        "test.diagnostics"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "test.warning".into(),
            component: plugin_config
                .get("component")
                .and_then(Json::as_str)
                .map(ToOwned::to_owned),
            field: Some("component".into()),
            message: "diagnostic from plugin".into(),
        }]
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

struct RegisteringPlugin;

impl NativePlugin for RegisteringPlugin {
    fn plugin_kind(&self) -> &str {
        "test.registering"
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        assert_eq!(plugin_config.get("enabled"), Some(&json!(true)));
        assert_eq!(ctx.host_api().abi_version, NEMO_RELAY_NATIVE_ABI_VERSION);
        assert!(ctx.runtime().scope_stack_active());
        ctx.register_subscriber("registered", |_event: &Event| {})?;
        Ok(())
    }
}

struct RegisterErrorPlugin;

impl NativePlugin for RegisterErrorPlugin {
    fn plugin_kind(&self) -> &str {
        "test.register_error"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Err("register rejected config".into())
    }
}

struct PluginKindPanicPlugin;

impl NativePlugin for PluginKindPanicPlugin {
    fn plugin_kind(&self) -> &str {
        panic!("plugin kind panic")
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

struct AllowsMultiplePanicPlugin;

impl NativePlugin for AllowsMultiplePanicPlugin {
    fn plugin_kind(&self) -> &str {
        "test.allows_multiple_panic"
    }

    fn allows_multiple_components(&self) -> bool {
        panic!("allows multiple panic")
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

struct ValidatePanicPlugin;

impl NativePlugin for ValidatePanicPlugin {
    fn plugin_kind(&self) -> &str {
        "test.validate_panic"
    }

    fn validate(
        &self,
        _plugin_config: &Map<String, Json>,
    ) -> Vec<nemo_relay_plugin::ConfigDiagnostic> {
        panic!("validate panic")
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

struct RegisterPanicPlugin;

impl NativePlugin for RegisterPanicPlugin {
    fn plugin_kind(&self) -> &str {
        "test.register_panic"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        panic!("register panic")
    }
}

struct DropPanicPlugin;

impl Drop for DropPanicPlugin {
    fn drop(&mut self) {
        panic!("plugin state drop panic")
    }
}

impl NativePlugin for DropPanicPlugin {
    fn plugin_kind(&self) -> &str {
        "test.drop_panic"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

nemo_relay_plugin::nemo_relay_plugin!(constructor_counting_entry, || {
    CONSTRUCTOR_CALLS.fetch_add(1, Ordering::SeqCst);
    CountingPlugin
});
nemo_relay_plugin::nemo_relay_plugin!(constructor_panic_entry, || -> ConstructorPanicPlugin {
    panic!("constructor panic")
});
nemo_relay_plugin::nemo_relay_plugin!(plugin_kind_panic_entry, || PluginKindPanicPlugin);
nemo_relay_plugin::nemo_relay_plugin!(allows_multiple_panic_entry, || AllowsMultiplePanicPlugin);

unsafe fn drop_exported_plugin(host: &NemoRelayNativeHostApiV1, plugin: NemoRelayNativePluginV1) {
    unsafe { (host.string_free)(plugin.plugin_kind) };
    if let Some(drop_fn) = plugin.drop {
        unsafe { drop_fn(plugin.user_data) };
    }
}

#[test]
fn direct_export_plugin_validates_host_table_and_kind_allocation() {
    let _guard = begin_test();
    let host = test_host();

    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(ptr::null(), &mut plugin, CountingPlugin) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, ptr::null_mut(), CountingPlugin) },
        NemoRelayStatus::NullPointer
    );

    let mut bad_host = host;
    bad_host.abi_version = NEMO_RELAY_NATIVE_ABI_VERSION + 1;
    let stale_kind = host_string(&host, "stale");
    let mut plugin = NemoRelayNativePluginV1 {
        struct_size: 123,
        plugin_kind: stale_kind,
        allows_multiple_components: false,
        user_data: NonNull::<u8>::dangling().as_ptr().cast(),
        validate: None,
        register: None,
        drop: None,
    };
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&bad_host, &mut plugin, CountingPlugin) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { (host.string_free)(stale_kind) };
    assert!(plugin.plugin_kind.is_null());
    assert!(plugin.user_data.is_null());

    let mut short_host = host;
    short_host.struct_size = size_of::<NemoRelayNativeHostApiV1>() - 1;
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&short_host, &mut plugin, CountingPlugin) },
        NemoRelayStatus::InvalidArg
    );

    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, CountingPlugin) },
        NemoRelayStatus::Internal
    );
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    assert!(plugin.plugin_kind.is_null());
    assert!(plugin.user_data.is_null());
}

#[test]
fn exported_plugin_validate_serializes_diagnostics_and_rejects_invalid_config() {
    let _guard = begin_test();
    let host = test_host();
    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, DiagnosticsPlugin) },
        NemoRelayStatus::Ok
    );
    assert!(!plugin.allows_multiple_components);
    assert_eq!(
        read_host_string(&host, plugin.plugin_kind).as_deref(),
        Some("test.diagnostics")
    );

    let config = json_host_string(&host, json!({ "component": "policy" }));
    let mut diagnostics = ptr::null_mut();
    assert_eq!(
        unsafe { plugin.validate.unwrap()(plugin.user_data, config, &mut diagnostics) },
        NemoRelayStatus::Ok
    );
    let diagnostics: Vec<ConfigDiagnostic> =
        serde_json::from_value(read_json_and_free(&host, diagnostics)).unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Warning);
    assert_eq!(diagnostics[0].component.as_deref(), Some("policy"));
    unsafe { (host.string_free)(config) };

    let config = json_host_string(&host, json!(["not", "object"]));
    let stale = host_string(&host, r#"[{"stale":true}]"#);
    let mut diagnostics = stale;
    assert_eq!(
        unsafe { plugin.validate.unwrap()(plugin.user_data, config, &mut diagnostics) },
        NemoRelayStatus::InvalidJson
    );
    assert!(diagnostics.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("plugin config must be a JSON object")
    );
    unsafe {
        (host.string_free)(stale);
        (host.string_free)(config);
    }

    let config = host_string(&host, "{not json");
    assert_eq!(
        unsafe { plugin.validate.unwrap()(plugin.user_data, config, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    let mut diagnostics = ptr::null_mut();
    assert_eq!(
        unsafe { plugin.validate.unwrap()(ptr::null_mut(), config, &mut diagnostics) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { plugin.validate.unwrap()(plugin.user_data, config, &mut diagnostics) },
        NemoRelayStatus::InvalidJson
    );
    let last_error = LAST_ERROR.lock().unwrap().clone().unwrap();
    assert!(last_error.starts_with("plugin config was invalid JSON:"));
    unsafe {
        (host.string_free)(config);
        drop_exported_plugin(&host, plugin);
    }
}

#[test]
fn exported_plugin_default_validate_returns_empty_diagnostics() {
    let _guard = begin_test();
    let host = test_host();
    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, CountingPlugin) },
        NemoRelayStatus::Ok
    );

    let config = json_host_string(&host, json!({}));
    let mut diagnostics = ptr::null_mut();
    assert_eq!(
        unsafe { plugin.validate.unwrap()(plugin.user_data, config, &mut diagnostics) },
        NemoRelayStatus::Ok
    );
    let diagnostics: Vec<ConfigDiagnostic> =
        serde_json::from_value(read_json_and_free(&host, diagnostics)).unwrap();
    assert!(diagnostics.is_empty());
    unsafe {
        (host.string_free)(config);
        drop_exported_plugin(&host, plugin);
    }
}

#[test]
fn exported_plugin_register_installs_callbacks_and_propagates_errors() {
    let _guard = begin_test();
    let host = test_host();

    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, RegisteringPlugin) },
        NemoRelayStatus::Ok
    );
    let config = json_host_string(&host, json!({ "enabled": true }));
    assert_eq!(
        unsafe {
            plugin.register.unwrap()(
                plugin.user_data,
                config,
                NonNull::<NemoRelayNativePluginContext>::dangling().as_ptr(),
            )
        },
        NemoRelayStatus::Ok
    );
    let registration = take_subscriber_registration();
    assert_eq!(registration.name, "registered");
    unsafe {
        registration.free();
        (host.string_free)(config);
    }

    let config = json_host_string(&host, json!({ "enabled": true }));
    assert_eq!(
        unsafe { plugin.register.unwrap()(plugin.user_data, config, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    unsafe { (host.string_free)(config) };
    unsafe { drop_exported_plugin(&host, plugin) };

    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, RegisterErrorPlugin) },
        NemoRelayStatus::Ok
    );
    let config = json_host_string(&host, json!({}));
    assert_eq!(
        unsafe {
            plugin.register.unwrap()(
                plugin.user_data,
                config,
                NonNull::<NemoRelayNativePluginContext>::dangling().as_ptr(),
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("register rejected config")
    );
    unsafe {
        (host.string_free)(config);
        drop_exported_plugin(&host, plugin);
    }
}

#[test]
fn exported_entry_symbol_validates_args_before_constructor() {
    let _guard = begin_test();
    let host = test_host();
    CONSTRUCTOR_CALLS.store(0, Ordering::SeqCst);

    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { constructor_counting_entry(ptr::null(), &mut plugin) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(CONSTRUCTOR_CALLS.load(Ordering::SeqCst), 0);

    assert_eq!(
        unsafe { constructor_counting_entry(&host, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(CONSTRUCTOR_CALLS.load(Ordering::SeqCst), 0);

    let mut bad_host = host;
    bad_host.abi_version = NEMO_RELAY_NATIVE_ABI_VERSION + 1;
    let stale_kind = host_string(&host, "stale");
    let mut plugin = NemoRelayNativePluginV1 {
        struct_size: 123,
        plugin_kind: stale_kind,
        allows_multiple_components: true,
        user_data: NonNull::<u8>::dangling().as_ptr().cast(),
        validate: None,
        register: None,
        drop: None,
    };
    assert_eq!(
        unsafe { constructor_counting_entry(&bad_host, &mut plugin) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { (host.string_free)(stale_kind) };
    assert_eq!(CONSTRUCTOR_CALLS.load(Ordering::SeqCst), 0);
    let default_plugin = NemoRelayNativePluginV1::default();
    assert_eq!(plugin.struct_size, default_plugin.struct_size);
    assert!(plugin.plugin_kind.is_null());
    assert_eq!(
        plugin.allows_multiple_components,
        default_plugin.allows_multiple_components
    );
    assert!(plugin.user_data.is_null());
    assert!(plugin.validate.is_none());
    assert!(plugin.register.is_none());
    assert!(plugin.drop.is_none());

    let mut short_host = host;
    short_host.struct_size = size_of::<NemoRelayNativeHostApiV1>() - 1;
    assert_eq!(
        unsafe { constructor_counting_entry(&short_host, &mut plugin) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(CONSTRUCTOR_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn exported_entry_symbol_catches_panics() {
    let _guard = begin_test();
    let host = test_host();

    for entry in [
        constructor_panic_entry,
        plugin_kind_panic_entry,
        allows_multiple_panic_entry,
    ] {
        *LAST_ERROR.lock().unwrap() = Some("stale error".into());
        let mut plugin = NemoRelayNativePluginV1::default();
        assert_eq!(
            unsafe { entry(&host, &mut plugin) },
            NemoRelayStatus::Internal
        );
        assert!(plugin.plugin_kind.is_null());
        assert!(plugin.user_data.is_null());
        assert!(plugin.validate.is_none());
        assert!(plugin.register.is_none());
        assert!(plugin.drop.is_none());
        assert_eq!(
            LAST_ERROR.lock().unwrap().as_deref(),
            Some("native plugin entry callback panicked")
        );
    }
}

#[test]
fn plugin_drop_callback_catches_state_drop_panics() {
    let _guard = begin_test();
    let host = test_host();
    let mut plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, DropPanicPlugin) },
        NemoRelayStatus::Ok
    );

    *LAST_ERROR.lock().unwrap() = None;
    unsafe { drop_exported_plugin(&host, plugin) };
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native plugin state drop panicked")
    );
}

#[test]
fn plugin_validate_and_register_panics_replace_last_error() {
    let _guard = begin_test();
    let host = test_host();

    let mut validate_plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe {
            nemo_relay_plugin::export_plugin(&host, &mut validate_plugin, ValidatePanicPlugin)
        },
        NemoRelayStatus::Ok
    );
    *LAST_ERROR.lock().unwrap() = Some("stale error".into());
    let config = json_host_string(&host, json!({}));
    let stale_diagnostics = host_string(&host, r#"[{"stale":true}]"#);
    let mut diagnostics = stale_diagnostics;
    assert_eq!(
        unsafe {
            validate_plugin.validate.unwrap()(validate_plugin.user_data, config, &mut diagnostics)
        },
        NemoRelayStatus::Internal
    );
    assert!(diagnostics.is_null());
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native plugin validate callback panicked")
    );
    unsafe {
        (host.string_free)(stale_diagnostics);
        (host.string_free)(config);
        drop_exported_plugin(&host, validate_plugin);
    }

    let mut register_plugin = NemoRelayNativePluginV1::default();
    assert_eq!(
        unsafe {
            nemo_relay_plugin::export_plugin(&host, &mut register_plugin, RegisterPanicPlugin)
        },
        NemoRelayStatus::Ok
    );
    *LAST_ERROR.lock().unwrap() = Some("stale error".into());
    let config = json_host_string(&host, json!({}));
    assert_eq!(
        unsafe {
            register_plugin.register.unwrap()(
                register_plugin.user_data,
                config,
                NonNull::<NemoRelayNativePluginContext>::dangling().as_ptr(),
            )
        },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native plugin register callback panicked")
    );
    unsafe {
        (host.string_free)(config);
        drop_exported_plugin(&host, register_plugin);
    }
}
