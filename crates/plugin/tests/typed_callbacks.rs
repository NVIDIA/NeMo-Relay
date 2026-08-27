// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-API tests for typed native plugin callback registration.

// This file keeps a complete legacy ABI host harness so binary-compatibility
// tests can share the same callback tables even when one test uses only a
// subset of their fields.
#![allow(dead_code, unused_imports)]

use std::collections::{BTreeSet, VecDeque};
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::ptr::{self, NonNull};
use std::sync::{
    Arc, Condvar, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::task::Poll;
use std::time::{Duration, Instant};

use futures::StreamExt;
use nemo_relay_plugin::{
    AnnotatedLlmRequest, BuiltinLlmCodec, CategoryProfile, ConfigDiagnostic, DataSchema,
    DiagnosticLevel, Event, EventCategory, EventSanitizeFields, Json, LlmCodecIdentity,
    LlmJsonAsyncStream, LlmJsonStream, LlmNext, LlmRequest, LlmRequestInterceptOutcome, LlmStream,
    LlmStreamNext, LogSeverity, MetricKind, MetricMeasurement, MetricValueType,
    NEMO_RELAY_NATIVE_ABI_VERSION, NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE,
    NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY, NativeExecutorConfig, NativePlugin,
    NemoRelayNativeAsyncCallbackState, NemoRelayNativeAsyncCompletion,
    NemoRelayNativeAsyncLlmStreamOpenCb, NemoRelayNativeAsyncLlmStreamPullCb,
    NemoRelayNativeAsyncMiddlewareCb, NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncNextResultCb, NemoRelayNativeAsyncNextStreamCb, NemoRelayNativeAsyncStream,
    NemoRelayNativeAsyncStreamMiddlewareCb, NemoRelayNativeConditionalMiddlewareCb,
    NemoRelayNativeEventSanitizeCb, NemoRelayNativeEventSubscriberCb, NemoRelayNativeFreeFn,
    NemoRelayNativeHostApiV1, NemoRelayNativeHostApiV3, NemoRelayNativeHostApiV4,
    NemoRelayNativeLlmAsyncStream, NemoRelayNativeLlmCodecKind, NemoRelayNativeLlmConditionalCb,
    NemoRelayNativeLlmExecutionCb, NemoRelayNativeLlmRequestCodec,
    NemoRelayNativeLlmRequestInterceptCb, NemoRelayNativeLlmResponseCodec,
    NemoRelayNativeLlmSanitizeRequestCb, NemoRelayNativeLlmSanitizeRequestContext,
    NemoRelayNativeLlmSanitizeResponseCb, NemoRelayNativeLlmSanitizeResponseContext,
    NemoRelayNativeLlmStreamExecutionCb, NemoRelayNativeLlmStreamV1, NemoRelayNativePluginContext,
    NemoRelayNativePluginRuntime, NemoRelayNativePluginV1, NemoRelayNativeScopeHandle,
    NemoRelayNativeScopeStack, NemoRelayNativeScopeStackBinding, NemoRelayNativeScopeType,
    NemoRelayNativeString, NemoRelayNativeToolConditionalCb, NemoRelayNativeToolExecutionCb,
    NemoRelayNativeToolJsonCb, NemoRelayNativeWithScopeStackCb, NemoRelayStatus, PendingMarkSpec,
    PluginContext, PluginRuntime, ScopeType, ToolExecutionInterceptOutcome, ToolExecutionResult,
    ToolNext,
};
use serde_json::{Map, json};

#[test]
fn runtime_registration_dtos_reexport_shared_types() {
    let identity = nemo_relay_plugin::RuntimeRegistrationIdentity {
        kind: nemo_relay_plugin::RuntimeRegistrationKind::Subscriber,
        local_name: "subscriber".into(),
        effective_name: "subscriber".into(),
        owner: nemo_relay_plugin::RuntimeRegistrationOwner {
            kind: nemo_relay_plugin::RuntimeRegistrationOwnerKind::GlobalApi,
            plugin_kind: None,
            component_ordinal: None,
        },
    };
    let _: nemo_relay_types::api::registry::RuntimeRegistrationIdentity = identity;
}

#[test]
fn async_abi_discriminants_reject_unknown_values() {
    use NemoRelayNativeAsyncMiddlewareKind as Kind;

    let middleware_kinds = [
        Kind::ToolSanitizeRequest,
        Kind::ToolSanitizeResponse,
        Kind::ToolConditionalExecution,
        Kind::ToolRequestIntercept,
        Kind::ToolExecutionIntercept,
        Kind::LlmSanitizeRequest,
        Kind::LlmSanitizeResponse,
        Kind::LlmConditionalExecution,
        Kind::LlmRequestIntercept,
        Kind::LlmExecutionIntercept,
        Kind::LlmStreamExecutionIntercept,
        Kind::MarkSanitize,
        Kind::ScopeSanitizeStart,
        Kind::ScopeSanitizeEnd,
        Kind::EventMetadataInjector,
    ];
    for (discriminant, kind) in middleware_kinds.into_iter().enumerate() {
        assert_eq!(kind as u32, discriminant as u32);
        assert_eq!(Kind::try_from(discriminant as u32), Ok(kind));
    }
    assert!(NemoRelayNativeAsyncMiddlewareKind::try_from(15).is_err());
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(1),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    assert!(NemoRelayNativeAsyncCallbackState::try_from(2).is_err());
}

struct DefaultNativePlugin;

impl NativePlugin for DefaultNativePlugin {
    fn plugin_kind(&self) -> &str {
        "test.default"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        Ok(())
    }
}

#[test]
fn native_sdk_defaults_initialize_safe_plugin_contracts() {
    let descriptor = NemoRelayNativePluginV1::default();
    assert_eq!(descriptor.struct_size, size_of::<NemoRelayNativePluginV1>());
    assert!(descriptor.plugin_kind.is_null());
    assert!(descriptor.allows_multiple_components);
    assert!(descriptor.user_data.is_null());
    assert!(descriptor.validate.is_none());
    assert!(descriptor.register.is_none());
    assert!(descriptor.drop.is_none());

    let plugin = DefaultNativePlugin;
    assert!(plugin.allows_multiple_components());
    assert_eq!(plugin.executor_config(), NativeExecutorConfig::default());
    assert!(plugin.validate(&Map::new()).is_empty());
}

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

struct RegisteredAsync {
    kind: NemoRelayNativeAsyncMiddlewareKind,
    name: String,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredAsync {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredAsyncStream {
    name: String,
    priority: i32,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredAsyncStream {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
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
    RegisteredAsync,
    RegisteredAsyncStream,
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
static CONDITIONAL_CALLBACK_RESULTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
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
static ASYNC_REGISTRATIONS: Mutex<Vec<RegisteredAsync>> = Mutex::new(Vec::new());
static ASYNC_STREAM_REGISTRATION: Mutex<Option<RegisteredAsyncStream>> = Mutex::new(None);
static ASYNC_PUSH_BACKPRESSURE: AtomicUsize = AtomicUsize::new(0);
static ASYNC_COMPLETION_RETAINS: AtomicUsize = AtomicUsize::new(0);
static ASYNC_TOOL_NEXT_RESULT: AtomicBool = AtomicBool::new(false);
static UNAVAILABLE_RUNTIME_CONTEXT_CALLS: AtomicUsize = AtomicUsize::new(0);
static UNAVAILABLE_RUNTIME_RETAINS: AtomicUsize = AtomicUsize::new(0);
static UNAVAILABLE_RUNTIME_RELEASES: AtomicUsize = AtomicUsize::new(0);
static UNAVAILABLE_CONTEXT_GATE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn native_abi_struct_sizes_are_self_describing() {
    assert_eq!(NEMO_RELAY_NATIVE_ABI_VERSION, 4);
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
    assert_native_abi_platform_layout();
}

#[cfg(target_pointer_width = "64")]
fn assert_native_abi_platform_layout() {
    assert_type_layout::<NemoRelayNativeHostApiV1>(8, 320);
    assert_eq!(
        host_api_offsets(),
        [
            0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
            160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288,
            296, 304, 312,
        ]
    );
    assert_type_layout::<NemoRelayNativeHostApiV3>(8, 440);
    assert_eq!(
        host_api_v3_offsets(),
        [
            0, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432
        ]
    );
    assert_type_layout::<NemoRelayNativeHostApiV4>(8, 600);
    assert_eq!(offset_of!(NemoRelayNativeHostApiV4, v3), 0);
    assert_eq!(offset_of!(NemoRelayNativeHostApiV4, emit_mark_v2), 512);
    assert_eq!(
        offset_of!(NemoRelayNativeHostApiV4, get_runtime_diagnostics),
        520
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail
        ),
        576
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_runtime_register_conditional_middleware_guardrail_callback
        ),
        584
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail_callback
        ),
        592
    );
    assert_type_layout::<NemoRelayNativePluginV1>(8, 56);
    assert_eq!(plugin_offsets(), [0, 8, 16, 24, 32, 40, 48]);
    assert_type_layout::<NemoRelayNativeLlmStreamV1>(8, 40);
    assert_eq!(stream_offsets(), [0, 8, 16, 24, 32]);
}

#[cfg(target_pointer_width = "32")]
fn assert_native_abi_platform_layout() {
    assert_type_layout::<NemoRelayNativeHostApiV1>(4, 160);
    assert_eq!(
        host_api_offsets(),
        [
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84,
            88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156,
        ]
    );
    assert_type_layout::<NemoRelayNativeHostApiV3>(4, 216);
    assert_eq!(
        host_api_v3_offsets(),
        [
            0, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212
        ]
    );
    assert_type_layout::<NemoRelayNativeHostApiV4>(4, 296);
    assert_eq!(offset_of!(NemoRelayNativeHostApiV4, v3), 0);
    assert_eq!(offset_of!(NemoRelayNativeHostApiV4, emit_mark_v2), 252);
    assert_eq!(
        offset_of!(NemoRelayNativeHostApiV4, get_runtime_diagnostics),
        256
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail
        ),
        284
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_runtime_register_conditional_middleware_guardrail_callback
        ),
        288
    );
    assert_eq!(
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail_callback
        ),
        292
    );
    assert_type_layout::<NemoRelayNativePluginV1>(4, 28);
    assert_eq!(plugin_offsets(), [0, 4, 8, 12, 16, 20, 24]);
    assert_type_layout::<NemoRelayNativeLlmStreamV1>(4, 20);
    assert_eq!(stream_offsets(), [0, 4, 8, 12, 16]);
}

fn assert_type_layout<T>(expected_alignment: usize, expected_size: usize) {
    assert_eq!(align_of::<T>(), expected_alignment);
    assert_eq!(size_of::<T>(), expected_size);
}

#[test]
fn native_abi_v4_extension_is_append_only() {
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(align_of::<NemoRelayNativeHostApiV4>(), 8);
        assert_eq!(size_of::<NemoRelayNativeHostApiV4>(), 600);
        assert_eq!(
            host_api_v4_offsets(),
            [0, 512, 520, 528, 536, 544, 552, 560, 568, 576, 584, 592]
        );
    }

    #[cfg(target_pointer_width = "32")]
    {
        assert_eq!(align_of::<NemoRelayNativeHostApiV4>(), 4);
        assert_eq!(size_of::<NemoRelayNativeHostApiV4>(), 296);
        assert_eq!(
            host_api_v4_offsets(),
            [0, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292]
        );
    }
}

fn host_api_v4_offsets() -> [usize; 12] {
    [
        offset_of!(NemoRelayNativeHostApiV4, v3),
        offset_of!(NemoRelayNativeHostApiV4, emit_mark_v2),
        offset_of!(NemoRelayNativeHostApiV4, get_runtime_diagnostics),
        offset_of!(NemoRelayNativeHostApiV4, plugin_context_runtime),
        offset_of!(NemoRelayNativeHostApiV4, plugin_runtime_retain),
        offset_of!(NemoRelayNativeHostApiV4, plugin_runtime_release),
        offset_of!(NemoRelayNativeHostApiV4, plugin_runtime_list_registrations),
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_runtime_register_conditional_middleware_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_runtime_deregister_conditional_middleware_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail
        ),
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_runtime_register_conditional_middleware_guardrail_callback
        ),
        offset_of!(
            NemoRelayNativeHostApiV4,
            plugin_context_register_conditional_middleware_guardrail_callback
        ),
    ]
}

fn host_api_v3_offsets() -> [usize; 16] {
    [
        offset_of!(NemoRelayNativeHostApiV3, v1),
        offset_of!(NemoRelayNativeHostApiV3, async_completion_resolve_json),
        offset_of!(NemoRelayNativeHostApiV3, async_completion_reject),
        offset_of!(NemoRelayNativeHostApiV3, async_completion_is_cancelled),
        offset_of!(NemoRelayNativeHostApiV3, async_completion_release),
        offset_of!(NemoRelayNativeHostApiV3, async_next_invoke),
        offset_of!(NemoRelayNativeHostApiV3, async_next_release),
        offset_of!(
            NemoRelayNativeHostApiV3,
            plugin_context_register_async_middleware
        ),
        offset_of!(NemoRelayNativeHostApiV3, async_stream_push_json),
        offset_of!(NemoRelayNativeHostApiV3, async_stream_finish),
        offset_of!(NemoRelayNativeHostApiV3, async_stream_reject),
        offset_of!(NemoRelayNativeHostApiV3, async_stream_is_cancelled),
        offset_of!(NemoRelayNativeHostApiV3, async_stream_release),
        offset_of!(NemoRelayNativeHostApiV3, async_next_invoke_stream),
        offset_of!(
            NemoRelayNativeHostApiV3,
            plugin_context_register_async_stream_middleware
        ),
        offset_of!(NemoRelayNativeHostApiV3, async_next_invoke_result),
    ]
}

fn host_api_offsets() -> [usize; 40] {
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
        offset_of!(NemoRelayNativeHostApiV1, llm_request_codec_decode),
        offset_of!(NemoRelayNativeHostApiV1, llm_request_codec_encode),
        offset_of!(NemoRelayNativeHostApiV1, llm_response_codec_decode),
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

unsafe extern "C" fn passthrough_tool_conditional_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_tool_execution_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _next_fn: nemo_relay_plugin::NemoRelayNativeToolNextFn,
    _next_ctx: *mut c_void,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_request_cb(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeRequestContext,
    _out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_response_cb(
    _user_data: *mut c_void,
    _response_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeResponseContext,
    _out_response_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_conditional_cb(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_request_intercept_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _annotated_json: *const NemoRelayNativeString,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_execution_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: nemo_relay_plugin::NemoRelayNativeLlmNextFn,
    _next_ctx: *mut c_void,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn passthrough_llm_stream_execution_cb(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: nemo_relay_plugin::NemoRelayNativeLlmStreamNextFn,
    _next_ctx: *mut c_void,
    _out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn pending_async_middleware_cb(
    _user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    _completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

unsafe extern "C" fn pending_async_stream_middleware_cb(
    _user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    _stream: *const NemoRelayNativeAsyncStream,
) -> u32 {
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

static RAW_ASYNC_REJECTIONS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn count_raw_async_rejection(_user_data: *mut c_void) {
    RAW_ASYNC_REJECTIONS.fetch_add(1, Ordering::SeqCst);
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

unsafe extern "C" fn capture_emit_mark_v2(
    name: *const NemoRelayNativeString,
    parent: *const NemoRelayNativeScopeHandle,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    data_schema_json: *const NemoRelayNativeString,
    severity: *const NemoRelayNativeString,
    _timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
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
    let data_schema = match optional_host_string(&host, data_schema_json) {
        Ok(data_schema) => data_schema,
        Err(status) => return status,
    };
    let severity = match optional_host_string(&host, severity) {
        Ok(severity) => severity,
        Err(status) => return status,
    };
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "mark_v2:{name}:parent={}:data={data}:metadata={metadata}:data_schema={data_schema}:severity={severity}",
        !parent.is_null()
    ));
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_runtime_diagnostics(
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let value =
        c"{\"entries\":[{\"code\":\"otel.example\",\"message\":\"example failure\",\"count\":3}]}";
    unsafe { test_string_new(value.as_ptr().cast(), value.to_bytes().len(), out) }
}

unsafe extern "C" fn capture_malformed_runtime_diagnostics(
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let value = c"{not-json";
    unsafe { test_string_new(value.as_ptr().cast(), value.to_bytes().len(), out) }
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

unsafe extern "C" fn unavailable_request_codec_decode(
    _codec: *const NemoRelayNativeLlmRequestCodec,
    _request: *const NemoRelayNativeString,
    _out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

unsafe extern "C" fn unavailable_request_codec_encode(
    _codec: *const NemoRelayNativeLlmRequestCodec,
    _annotated: *const NemoRelayNativeString,
    _original: *const NemoRelayNativeString,
    _out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

unsafe extern "C" fn unavailable_response_codec_decode(
    _codec: *const NemoRelayNativeLlmResponseCodec,
    _response: *const NemoRelayNativeString,
    _out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

unsafe extern "C" fn successful_request_codec_decode(
    _codec: *const NemoRelayNativeLlmRequestCodec,
    _request: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { test_string_new(c"{}".as_ptr().cast(), 2, out) }
}

unsafe extern "C" fn successful_request_codec_encode(
    _codec: *const NemoRelayNativeLlmRequestCodec,
    _annotated: *const NemoRelayNativeString,
    original: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let bytes = unsafe { &*(original.cast::<TestString>()) }.0.as_slice();
    unsafe { test_string_new(bytes.as_ptr(), bytes.len(), out) }
}

unsafe extern "C" fn successful_response_codec_decode(
    _codec: *const NemoRelayNativeLlmResponseCodec,
    _response: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { test_string_new(c"{}".as_ptr().cast(), 2, out) }
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
        llm_request_codec_decode: unavailable_request_codec_decode,
        llm_request_codec_encode: unavailable_request_codec_encode,
        llm_response_codec_decode: unavailable_response_codec_decode,
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

#[derive(Debug)]
struct MockAsyncCompletion {
    settled: Mutex<Option<std::result::Result<Json, String>>>,
    settled_cv: Condvar,
    cancelled: AtomicBool,
    releases: AtomicUsize,
}

impl MockAsyncCompletion {
    fn new() -> Self {
        Self {
            settled: Mutex::new(None),
            settled_cv: Condvar::new(),
            cancelled: AtomicBool::new(false),
            releases: AtomicUsize::new(0),
        }
    }

    fn raw(&self) -> *const NemoRelayNativeAsyncCompletion {
        ptr::from_ref(self).cast()
    }

    fn wait(&self) -> std::result::Result<Json, String> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut settled = self.settled.lock().unwrap();
        while settled.is_none() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "async middleware did not settle");
            let (next, timeout) = self.settled_cv.wait_timeout(settled, remaining).unwrap();
            settled = next;
            assert!(
                !timeout.timed_out() || settled.is_some(),
                "async middleware did not settle"
            );
        }
        settled.clone().unwrap()
    }

    fn wait_for_release(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.releases.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "completion was not released");
            std::thread::yield_now();
        }
    }
}

struct MockAsyncNext {
    calls: AtomicUsize,
    releases: AtomicUsize,
    pull_stream: *const NemoRelayNativeLlmAsyncStream,
}

impl MockAsyncNext {
    fn raw(&self) -> *const NemoRelayNativeAsyncNext {
        ptr::from_ref(self).cast()
    }
}

#[derive(Debug, PartialEq)]
enum MockOutputEvent {
    Chunk(Json),
    Finished,
    Rejected(String),
}

struct MockAsyncOutput {
    events: Mutex<Vec<MockOutputEvent>>,
    events_cv: Condvar,
    cancelled: AtomicBool,
    releases: AtomicUsize,
}

impl MockAsyncOutput {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            events_cv: Condvar::new(),
            cancelled: AtomicBool::new(false),
            releases: AtomicUsize::new(0),
        }
    }

    fn raw(&self) -> *const NemoRelayNativeAsyncStream {
        ptr::from_ref(self).cast()
    }

    fn wait_terminal(&self) -> Vec<MockOutputEvent> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut events = self.events.lock().unwrap();
        while !matches!(
            events.last(),
            Some(MockOutputEvent::Finished | MockOutputEvent::Rejected(_))
        ) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "async stream did not terminate");
            let (next, timeout) = self.events_cv.wait_timeout(events, remaining).unwrap();
            events = next;
            assert!(
                !timeout.timed_out() || !events.is_empty(),
                "async stream did not progress"
            );
        }
        std::mem::take(&mut *events)
    }

    fn wait_for_release(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.releases.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "async output was not released");
            std::thread::yield_now();
        }
    }
}

struct MockPullStream {
    items: Mutex<VecDeque<std::result::Result<Option<Json>, String>>>,
    cancelled: AtomicBool,
    releases: AtomicUsize,
}

impl MockPullStream {
    fn raw(&self) -> *const NemoRelayNativeLlmAsyncStream {
        ptr::from_ref(self).cast()
    }
}

struct MockNativeStream {
    steps: Mutex<VecDeque<(NemoRelayStatus, Option<Vec<u8>>)>>,
    cancellations: AtomicUsize,
    drops: AtomicUsize,
    cancel_status: NemoRelayStatus,
}

unsafe extern "C" fn poll_native_stream(
    user_data: *mut c_void,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let stream = unsafe { &*user_data.cast::<MockNativeStream>() };
    let (status, payload) = stream
        .steps
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or((NemoRelayStatus::StreamEnd, None));
    if let Some(payload) = payload {
        unsafe { *out = bytes_host_string(&test_host(), &payload) };
    }
    status
}

unsafe extern "C" fn cancel_native_stream(user_data: *mut c_void) -> NemoRelayStatus {
    let stream = unsafe { &*user_data.cast::<MockNativeStream>() };
    stream.cancellations.fetch_add(1, Ordering::SeqCst);
    stream.cancel_status
}

unsafe extern "C" fn drop_native_stream(user_data: *mut c_void) {
    let stream = unsafe { &*user_data.cast::<MockNativeStream>() };
    stream.drops.fetch_add(1, Ordering::SeqCst);
}

fn native_stream_raw(stream: &MockNativeStream) -> NemoRelayNativeLlmStreamV1 {
    NemoRelayNativeLlmStreamV1 {
        struct_size: size_of::<NemoRelayNativeLlmStreamV1>(),
        user_data: ptr::from_ref(stream).cast_mut().cast(),
        next: Some(poll_native_stream),
        cancel: Some(cancel_native_stream),
        drop: Some(drop_native_stream),
    }
}

unsafe extern "C" fn capture_async_completion_resolve(
    completion: *const NemoRelayNativeAsyncCompletion,
    value_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    if completion.is_null() || value_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let completion = unsafe { &*completion.cast::<MockAsyncCompletion>() };
    let host = test_host();
    let value = match read_host_string(&host, value_json)
        .and_then(|value| serde_json::from_str(&value).ok())
    {
        Some(value) => value,
        None => return NemoRelayStatus::InvalidArg,
    };
    *completion.settled.lock().unwrap() = Some(Ok(value));
    completion.settled_cv.notify_all();
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_completion_reject(
    completion: *const NemoRelayNativeAsyncCompletion,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    if completion.is_null() || message.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let completion = unsafe { &*completion.cast::<MockAsyncCompletion>() };
    let message = read_host_string(&test_host(), message).unwrap();
    *completion.settled.lock().unwrap() = Some(Err(message));
    completion.settled_cv.notify_all();
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_completion_cancelled(
    completion: *const NemoRelayNativeAsyncCompletion,
) -> bool {
    !completion.is_null()
        && unsafe { &*completion.cast::<MockAsyncCompletion>() }
            .cancelled
            .load(Ordering::SeqCst)
}

unsafe extern "C" fn capture_async_completion_release(
    completion: *const NemoRelayNativeAsyncCompletion,
) {
    if !completion.is_null() {
        unsafe { &*completion.cast::<MockAsyncCompletion>() }
            .releases
            .fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn capture_async_completion_retain(
    completion: *const NemoRelayNativeAsyncCompletion,
) -> NemoRelayStatus {
    if completion.is_null() {
        NemoRelayStatus::NullPointer
    } else {
        ASYNC_COMPLETION_RETAINS.fetch_add(1, Ordering::SeqCst);
        NemoRelayStatus::Ok
    }
}

unsafe extern "C" fn capture_async_next_invoke(
    _next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> NemoRelayStatus {
    unsafe { capture_async_completion_resolve(completion, invocation_json) }
}

unsafe extern "C" fn capture_async_next_release(next: *const NemoRelayNativeAsyncNext) {
    if !next.is_null() {
        unsafe { &*next.cast::<MockAsyncNext>() }
            .releases
            .fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn capture_register_async_middleware(
    _ctx: *mut NemoRelayNativePluginContext,
    kind: u32,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        if let Some(free_fn) = free_fn {
            unsafe { free_fn(user_data) };
        }
        return status;
    }
    let kind = match NemoRelayNativeAsyncMiddlewareKind::try_from(kind) {
        Ok(kind) => kind,
        Err(()) => return NemoRelayStatus::InvalidArg,
    };
    let name = match required_host_string(&test_host(), name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    ASYNC_REGISTRATIONS.lock().unwrap().push(RegisteredAsync {
        kind,
        name,
        priority,
        break_chain,
        cb,
        user_data: user_data as usize,
        free_fn,
    });
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_stream_push(
    stream: *const NemoRelayNativeAsyncStream,
    chunk_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    if stream.is_null() || chunk_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*stream.cast::<MockAsyncOutput>() };
    if ASYNC_PUSH_BACKPRESSURE
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
            remaining.checked_sub(1)
        })
        .is_ok()
    {
        return NemoRelayStatus::Backpressured;
    }
    let chunk = serde_json::from_str(&read_host_string(&test_host(), chunk_json).unwrap()).unwrap();
    stream
        .events
        .lock()
        .unwrap()
        .push(MockOutputEvent::Chunk(chunk));
    stream.events_cv.notify_all();
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_stream_finish(
    stream: *const NemoRelayNativeAsyncStream,
) -> NemoRelayStatus {
    if stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*stream.cast::<MockAsyncOutput>() };
    stream
        .events
        .lock()
        .unwrap()
        .push(MockOutputEvent::Finished);
    stream.events_cv.notify_all();
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_stream_reject(
    stream: *const NemoRelayNativeAsyncStream,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    if stream.is_null() || message.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*stream.cast::<MockAsyncOutput>() };
    let message = read_host_string(&test_host(), message).unwrap();
    stream
        .events
        .lock()
        .unwrap()
        .push(MockOutputEvent::Rejected(message));
    stream.events_cv.notify_all();
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_stream_cancelled(
    stream: *const NemoRelayNativeAsyncStream,
) -> bool {
    !stream.is_null()
        && unsafe { &*stream.cast::<MockAsyncOutput>() }
            .cancelled
            .load(Ordering::SeqCst)
}

unsafe extern "C" fn capture_async_stream_backpressured(
    _stream: *const NemoRelayNativeAsyncStream,
) -> bool {
    ASYNC_PUSH_BACKPRESSURE.load(Ordering::SeqCst) < 2
}

unsafe extern "C" fn capture_async_stream_release(stream: *const NemoRelayNativeAsyncStream) {
    if !stream.is_null() {
        unsafe { &*stream.cast::<MockAsyncOutput>() }
            .releases
            .fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn unavailable_async_next_stream(
    _next: *const NemoRelayNativeAsyncNext,
    _invocation_json: *const NemoRelayNativeString,
    _stream: *const NemoRelayNativeAsyncStream,
    _cb: NemoRelayNativeAsyncNextStreamCb,
    _user_data: *mut c_void,
) -> NemoRelayStatus {
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn capture_register_async_stream(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *REGISTRATION_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        if let Some(free_fn) = free_fn {
            unsafe { free_fn(user_data) };
        }
        return status;
    }
    let name = match required_host_string(&test_host(), name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    replace_registration(
        &ASYNC_STREAM_REGISTRATION,
        RegisteredAsyncStream {
            name,
            priority,
            cb,
            user_data: user_data as usize,
            free_fn,
        },
    );
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_next_result(
    next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncNextResultCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    if next.is_null() || invocation_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { &*next.cast::<MockAsyncNext>() }
        .calls
        .fetch_add(1, Ordering::SeqCst);
    if ASYNC_TOOL_NEXT_RESULT.load(Ordering::SeqCst) {
        let host = test_host();
        let invocation: Json = read_host_string(&host, invocation_json)
            .and_then(|value| serde_json::from_str(&value).ok())
            .expect("tool next invocation JSON");
        let result = json_host_string(
            &host,
            json!({
                "result": invocation,
                "annotation": {"source": "host"},
            }),
        );
        unsafe { cb(user_data, result, ptr::null()) };
        unsafe { (host.string_free)(result) };
    } else {
        unsafe { cb(user_data, invocation_json, ptr::null()) };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_request_decode(
    _completion: *const NemoRelayNativeAsyncCompletion,
    _request_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    write_json(&test_host(), &json!({}), out)
}

unsafe extern "C" fn capture_async_request_encode(
    _completion: *const NemoRelayNativeAsyncCompletion,
    _annotated_json: *const NemoRelayNativeString,
    original_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out.is_null() || original_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let original = read_host_string(&test_host(), original_json).unwrap();
    let value = serde_json::from_str(&original).unwrap();
    write_json(&test_host(), &value, out)
}

unsafe extern "C" fn capture_async_response_decode(
    _completion: *const NemoRelayNativeAsyncCompletion,
    _response_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    write_json(&test_host(), &json!({}), out)
}

unsafe extern "C" fn capture_async_open_stream(
    next: *const NemoRelayNativeAsyncNext,
    _request_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncLlmStreamOpenCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    if next.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*next.cast::<MockAsyncNext>() }.pull_stream;
    unsafe { cb(user_data, stream, ptr::null()) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_pull_stream(
    stream: *const NemoRelayNativeLlmAsyncStream,
    cb: NemoRelayNativeAsyncLlmStreamPullCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    if stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*stream.cast::<MockPullStream>() };
    match stream.items.lock().unwrap().pop_front().unwrap_or(Ok(None)) {
        Ok(Some(chunk)) => {
            let chunk = json_host_string(&test_host(), chunk);
            unsafe { cb(user_data, chunk, ptr::null(), false) };
            unsafe { (test_host().string_free)(chunk) };
        }
        Ok(None) => unsafe { cb(user_data, ptr::null(), ptr::null(), true) },
        Err(error) => {
            let error = host_string(&test_host(), &error);
            unsafe { cb(user_data, ptr::null(), error, true) };
            unsafe { (test_host().string_free)(error) };
        }
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_cancel_pull_stream(
    stream: *const NemoRelayNativeLlmAsyncStream,
) -> NemoRelayStatus {
    if stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { &*stream.cast::<MockPullStream>() }
        .cancelled
        .store(true, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_async_release_pull_stream(
    stream: *const NemoRelayNativeLlmAsyncStream,
) {
    if !stream.is_null() {
        unsafe { &*stream.cast::<MockPullStream>() }
            .releases
            .fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn unavailable_plugin_context_runtime(
    _ctx: *mut NemoRelayNativePluginContext,
    out: *mut *const NemoRelayNativePluginRuntime,
) -> NemoRelayStatus {
    UNAVAILABLE_RUNTIME_CONTEXT_CALLS.fetch_add(1, Ordering::SeqCst);
    if !out.is_null() {
        unsafe { *out = ptr::null() };
    }
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_runtime_retain(
    _runtime: *const NemoRelayNativePluginRuntime,
) -> NemoRelayStatus {
    UNAVAILABLE_RUNTIME_RETAINS.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_runtime_release(
    _runtime: *const NemoRelayNativePluginRuntime,
) {
    UNAVAILABLE_RUNTIME_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn unavailable_plugin_runtime_list_registrations(
    _runtime: *const NemoRelayNativePluginRuntime,
    _kinds_json: *const NemoRelayNativeString,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_runtime_register_gate(
    _runtime: *const NemoRelayNativePluginRuntime,
    _name: *const NemoRelayNativeString,
    _kinds_json: *const NemoRelayNativeString,
    _registration_name: *const NemoRelayNativeString,
    _reason: *const NemoRelayNativeString,
    _out_handle: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_runtime_deregister_gate(
    _runtime: *const NemoRelayNativePluginRuntime,
    _handle: *const NemoRelayNativeString,
    _out_removed: *mut bool,
) -> NemoRelayStatus {
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_context_register_gate(
    _ctx: *mut NemoRelayNativePluginContext,
    _name: *const NemoRelayNativeString,
    _kinds_json: *const NemoRelayNativeString,
    _registration_name: *const NemoRelayNativeString,
    _reason: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    UNAVAILABLE_CONTEXT_GATE_CALLS.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_runtime_register_gate_callback(
    _runtime: *const NemoRelayNativePluginRuntime,
    _name: *const NemoRelayNativeString,
    _kinds_json: *const NemoRelayNativeString,
    _registration_name: *const NemoRelayNativeString,
    _cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    _out_handle: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if let Some(free_fn) = free_fn {
        unsafe { free_fn(user_data) };
    }
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn unavailable_plugin_context_register_gate_callback(
    _ctx: *mut NemoRelayNativePluginContext,
    _name: *const NemoRelayNativeString,
    _kinds_json: *const NemoRelayNativeString,
    _registration_name: *const NemoRelayNativeString,
    _cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    UNAVAILABLE_CONTEXT_GATE_CALLS.fetch_add(1, Ordering::SeqCst);
    if let Some(free_fn) = free_fn {
        unsafe { free_fn(user_data) };
    }
    NemoRelayStatus::NotFound
}

unsafe extern "C" fn capture_plugin_context_runtime(
    _ctx: *mut NemoRelayNativePluginContext,
    out: *mut *const NemoRelayNativePluginRuntime,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = NonNull::<NemoRelayNativePluginRuntime>::dangling().as_ptr() };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_plugin_runtime_retain(
    _runtime: *const NemoRelayNativePluginRuntime,
) -> NemoRelayStatus {
    UNAVAILABLE_RUNTIME_RETAINS.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_plugin_runtime_release(_runtime: *const NemoRelayNativePluginRuntime) {
    UNAVAILABLE_RUNTIME_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn capture_plugin_runtime_list_registrations(
    _runtime: *const NemoRelayNativePluginRuntime,
    kinds_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let kinds = read_host_string(&test_host(), kinds_json).unwrap();
    RUNTIME_CALLS.lock().unwrap().push(format!("list:{kinds}"));
    unsafe {
        *out_json = json_host_string(
            &test_host(),
            json!([{
                "kind": "subscriber",
                "local_name": "target",
                "effective_name": "nemo-relay-plugin.v1.example:1:target",
                "owner": {
                    "kind": "plugin",
                    "plugin_kind": "example",
                    "component_ordinal": 1
                }
            }]),
        )
    };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_plugin_runtime_register_gate(
    _runtime: *const NemoRelayNativePluginRuntime,
    name: *const NemoRelayNativeString,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    reason: *const NemoRelayNativeString,
    out_handle: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_handle.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let host = test_host();
    let values = [name, kinds_json, registration_name, reason]
        .map(|value| read_host_string(&host, value).unwrap());
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "register:{}:{}:{}:{}",
        values[0], values[1], values[2], values[3]
    ));
    unsafe { *out_handle = host_string(&host, "gate-handle") };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_plugin_runtime_deregister_gate(
    _runtime: *const NemoRelayNativePluginRuntime,
    handle: *const NemoRelayNativeString,
    out_removed: *mut bool,
) -> NemoRelayStatus {
    if out_removed.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let handle = read_host_string(&test_host(), handle).unwrap();
    RUNTIME_CALLS
        .lock()
        .unwrap()
        .push(format!("deregister:{handle}"));
    unsafe { *out_removed = true };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn capture_plugin_context_register_gate(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    reason: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let values = [name, kinds_json, registration_name, reason]
        .map(|value| read_host_string(&host, value).unwrap());
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "initial:{}:{}:{}:{}",
        values[0], values[1], values[2], values[3]
    ));
    NemoRelayStatus::Ok
}

unsafe fn capture_conditional_middleware_decisions(
    cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let mut blocked = ptr::null_mut();
    let status = unsafe { cb(user_data, kinds_json, registration_name, &mut blocked) };
    if status != NemoRelayStatus::Ok {
        if let Some(free_fn) = free_fn {
            unsafe { free_fn(user_data) };
        }
        return status;
    }
    let blocked = if blocked.is_null() {
        "none".to_string()
    } else {
        let value = read_host_string(&host, blocked).unwrap();
        unsafe { (host.string_free)(blocked) };
        value
    };
    let allowed_name = host_string(&host, "documentation-observed-subscriber");
    let mut allowed = ptr::null_mut();
    let status = unsafe { cb(user_data, kinds_json, allowed_name, &mut allowed) };
    unsafe { (host.string_free)(allowed_name) };
    let allowed = if allowed.is_null() {
        "none".to_string()
    } else {
        let value = read_host_string(&host, allowed).unwrap();
        unsafe { (host.string_free)(allowed) };
        value
    };
    RUNTIME_CALLS
        .lock()
        .unwrap()
        .push(format!("decisions:{blocked}:{allowed}"));
    if let Some(free_fn) = free_fn {
        unsafe { free_fn(user_data) };
    }
    status
}

unsafe extern "C" fn capture_plugin_runtime_register_gate_callback(
    _runtime: *const NemoRelayNativePluginRuntime,
    name: *const NemoRelayNativeString,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    out_handle: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_handle.is_null() {
        if let Some(free_fn) = free_fn {
            unsafe { free_fn(user_data) };
        }
        return NemoRelayStatus::NullPointer;
    }
    let host = test_host();
    let values =
        [name, kinds_json, registration_name].map(|value| read_host_string(&host, value).unwrap());
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "register-callback:{}:{}:{}",
        values[0], values[1], values[2]
    ));
    let status = unsafe {
        capture_conditional_middleware_decisions(
            cb,
            user_data,
            free_fn,
            kinds_json,
            registration_name,
        )
    };
    if status == NemoRelayStatus::Ok {
        unsafe { *out_handle = host_string(&host, "gate-handle") };
    }
    status
}

unsafe extern "C" fn capture_plugin_context_register_gate_callback(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let host = test_host();
    let values =
        [name, kinds_json, registration_name].map(|value| read_host_string(&host, value).unwrap());
    RUNTIME_CALLS.lock().unwrap().push(format!(
        "initial-callback:{}:{}:{}",
        values[0], values[1], values[2]
    ));
    unsafe {
        capture_conditional_middleware_decisions(
            cb,
            user_data,
            free_fn,
            kinds_json,
            registration_name,
        )
    }
}

unsafe fn record_conditional_callback_result(
    label: &str,
    cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) {
    let host = test_host();
    let status = unsafe { cb(user_data, kinds_json, registration_name, out_reason) };
    let reason = if out_reason.is_null() || unsafe { *out_reason }.is_null() {
        "none".to_string()
    } else {
        let reason =
            read_host_string(&host, unsafe { *out_reason }).unwrap_or_else(|| "invalid".into());
        unsafe { (host.string_free)(*out_reason) };
        reason
    };
    CONDITIONAL_CALLBACK_RESULTS
        .lock()
        .unwrap()
        .push(format!("{label}:{status:?}:{reason}"));
}

unsafe extern "C" fn exercise_plugin_context_register_gate_callback(
    _ctx: *mut NemoRelayNativePluginContext,
    _name: *const NemoRelayNativeString,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    cb: NemoRelayNativeConditionalMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let host = test_host();
    let allowed_name = host_string(&host, "documentation-observed-subscriber");
    let malformed_kinds = host_string(&host, "not-json");
    let mut out = ptr::null_mut();
    unsafe {
        record_conditional_callback_result(
            "blocked",
            cb,
            user_data,
            kinds_json,
            registration_name,
            &mut out,
        );
        record_conditional_callback_result(
            "allowed",
            cb,
            user_data,
            kinds_json,
            allowed_name,
            &mut out,
        );
        record_conditional_callback_result(
            "malformed-kinds",
            cb,
            user_data,
            malformed_kinds,
            registration_name,
            &mut out,
        );
        record_conditional_callback_result(
            "null-kinds",
            cb,
            user_data,
            ptr::null(),
            registration_name,
            &mut out,
        );
        record_conditional_callback_result(
            "null-name",
            cb,
            user_data,
            kinds_json,
            ptr::null(),
            &mut out,
        );
        record_conditional_callback_result(
            "null-output",
            cb,
            user_data,
            kinds_json,
            registration_name,
            ptr::null_mut(),
        );
    }
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    unsafe {
        record_conditional_callback_result(
            "reason-allocation",
            cb,
            user_data,
            kinds_json,
            registration_name,
            &mut out,
        );
        (host.string_free)(allowed_name);
        (host.string_free)(malformed_kinds);
    }
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;
    if let Some(free_fn) = free_fn {
        unsafe { free_fn(user_data) };
    }
    NemoRelayStatus::Ok
}

fn test_host_v4() -> NemoRelayNativeHostApiV4 {
    let mut v1 = test_host();
    v1.abi_version = 4;
    v1.struct_size = size_of::<NemoRelayNativeHostApiV4>();
    NemoRelayNativeHostApiV4 {
        v3: NemoRelayNativeHostApiV3 {
            v1,
            async_completion_resolve_json: capture_async_completion_resolve,
            async_completion_reject: capture_async_completion_reject,
            async_completion_is_cancelled: capture_async_completion_cancelled,
            async_completion_release: capture_async_completion_release,
            async_next_invoke: capture_async_next_invoke,
            async_next_release: capture_async_next_release,
            plugin_context_register_async_middleware: capture_register_async_middleware,
            async_stream_push_json: capture_async_stream_push,
            async_stream_finish: capture_async_stream_finish,
            async_stream_reject: capture_async_stream_reject,
            async_stream_is_cancelled: capture_async_stream_cancelled,
            async_stream_release: capture_async_stream_release,
            async_next_invoke_stream: unavailable_async_next_stream,
            plugin_context_register_async_stream_middleware: capture_register_async_stream,
            async_next_invoke_result: capture_async_next_result,
        },
        async_completion_llm_request_codec_decode: capture_async_request_decode,
        async_completion_llm_request_codec_encode: capture_async_request_encode,
        async_completion_llm_response_codec_decode: capture_async_response_decode,
        async_next_open_llm_stream: capture_async_open_stream,
        async_llm_stream_pull: capture_async_pull_stream,
        async_llm_stream_cancel: capture_async_cancel_pull_stream,
        async_llm_stream_release: capture_async_release_pull_stream,
        async_completion_retain: capture_async_completion_retain,
        async_stream_is_backpressured: capture_async_stream_backpressured,
        emit_mark_v2: capture_emit_mark_v2,
        get_runtime_diagnostics: capture_runtime_diagnostics,
        plugin_context_runtime: unavailable_plugin_context_runtime,
        plugin_runtime_retain: unavailable_plugin_runtime_retain,
        plugin_runtime_release: unavailable_plugin_runtime_release,
        plugin_runtime_list_registrations: unavailable_plugin_runtime_list_registrations,
        plugin_runtime_register_conditional_middleware_guardrail:
            unavailable_plugin_runtime_register_gate,
        plugin_runtime_deregister_conditional_middleware_guardrail:
            unavailable_plugin_runtime_deregister_gate,
        plugin_context_register_conditional_middleware_guardrail:
            unavailable_plugin_context_register_gate,
        plugin_runtime_register_conditional_middleware_guardrail_callback:
            unavailable_plugin_runtime_register_gate_callback,
        plugin_context_register_conditional_middleware_guardrail_callback:
            unavailable_plugin_context_register_gate_callback,
    }
}

fn test_llm_request() -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({ "prompt": "hello" }),
    }
}

fn take_async_registration(kind: NemoRelayNativeAsyncMiddlewareKind) -> RegisteredAsync {
    let mut registrations = ASYNC_REGISTRATIONS.lock().unwrap();
    let index = registrations
        .iter()
        .position(|registration| registration.kind == kind)
        .unwrap_or_else(|| panic!("missing {kind:?} registration"));
    registrations.remove(index)
}

fn invoke_async_registration(
    host: &NemoRelayNativeHostApiV4,
    registration: &RegisteredAsync,
    invocation: Json,
    next: Option<&MockAsyncNext>,
) -> std::result::Result<Json, String> {
    ASYNC_TOOL_NEXT_RESULT.store(
        registration.kind == NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept,
        Ordering::SeqCst,
    );
    let completion = MockAsyncCompletion::new();
    let invocation = json_host_string(&host.v3.v1, invocation);
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.map_or(ptr::null(), MockAsyncNext::raw),
            completion.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    let result = completion.wait();
    ASYNC_TOOL_NEXT_RESULT.store(false, Ordering::SeqCst);
    completion.wait_for_release();
    assert!(completion.releases.load(Ordering::SeqCst) >= 1);
    result
}

fn begin_test() -> MutexGuard<'static, ()> {
    let guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reset_state();
    guard
}

fn reset_state() {
    ASYNC_COMPLETION_RETAINS.store(0, Ordering::SeqCst);
    UNAVAILABLE_RUNTIME_CONTEXT_CALLS.store(0, Ordering::SeqCst);
    UNAVAILABLE_RUNTIME_RETAINS.store(0, Ordering::SeqCst);
    UNAVAILABLE_RUNTIME_RELEASES.store(0, Ordering::SeqCst);
    UNAVAILABLE_CONTEXT_GATE_CALLS.store(0, Ordering::SeqCst);
    for registration in ASYNC_REGISTRATIONS.lock().unwrap().drain(..) {
        unsafe { registration.free() };
    }
    clear_registration(&ASYNC_STREAM_REGISTRATION);
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
    CONDITIONAL_CALLBACK_RESULTS.lock().unwrap().clear();
    SCOPE_HANDLE_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_BINDING_FREES.store(0, Ordering::SeqCst);
    SCOPE_STACK_BINDING_RESTORES.store(0, Ordering::SeqCst);
    ASYNC_PUSH_BACKPRESSURE.store(0, Ordering::SeqCst);
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

fn native_no_codec_context() -> NemoRelayNativeLlmSanitizeRequestContext {
    NemoRelayNativeLlmSanitizeRequestContext {
        codec_kind: NemoRelayNativeLlmCodecKind::None,
        codec_id: ptr::null(),
        codec: ptr::null(),
    }
}

fn native_no_response_codec_context() -> NemoRelayNativeLlmSanitizeResponseContext {
    NemoRelayNativeLlmSanitizeResponseContext {
        codec_kind: NemoRelayNativeLlmCodecKind::None,
        codec_id: ptr::null(),
        codec: ptr::null(),
    }
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

struct CountDrop(Arc<AtomicUsize>);

impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
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
    assert_scope_runtime_calls(&calls);
    assert_stack_runtime_calls(&calls);
    assert_eq!(SCOPE_HANDLE_FREES.load(Ordering::SeqCst), 2);
    assert_eq!(SCOPE_STACK_FREES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 0);
}

#[test]
fn plugin_runtime_uses_v4_mark_options_extension() {
    let _guard = begin_test();
    let host = test_host_v4();
    let runtime = PluginRuntime::new(&host.v3.v1);
    let data_schema = DataSchema::builder()
        .name("example.mark")
        .version("1")
        .build();

    runtime
        .emit_mark_with_options(
            "checkpoint",
            Some(&json!({ "mark": true })),
            Some(&json!({ "meta": true })),
            Some(&data_schema),
            Some(LogSeverity::Warn),
        )
        .unwrap();

    assert!(RUNTIME_CALLS.lock().unwrap().iter().any(|call| {
        call.starts_with("mark_v2:checkpoint:parent=false")
            && call.contains(r#""mark":true"#)
            && call.contains(r#""meta":true"#)
            && call.contains(r#""name":"example.mark""#)
            && call.contains("severity=warn")
    }));
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn plugin_runtime_emits_metrics_and_requires_v4_diagnostics() {
    let _guard = begin_test();
    let host = test_host_v4();
    PluginRuntime::new(&host.v3.v1)
        .emit_metric(
            "example.requests",
            vec![MetricMeasurement {
                name: "example.requests".into(),
                kind: MetricKind::Counter,
                value_type: MetricValueType::U64,
                value: json!(1),
                unit: Some("{request}".into()),
                description: None,
                attributes: Some(json!({"region": "test"})),
                boundaries: None,
            }],
            Some(&json!({"source": "sdk-test"})),
        )
        .unwrap();

    assert!(RUNTIME_CALLS.lock().unwrap().iter().any(|call| {
        call.starts_with("mark_v2:example.requests:parent=false")
            && call.contains(r#""name":"nemo.relay.metric_measurements""#)
            && call.contains(r#""measurements":[{"attributes":{"region":"test"}"#)
    }));
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);

    let legacy_host = test_host();
    let error = PluginRuntime::new(&legacy_host)
        .runtime_diagnostics()
        .unwrap_err();
    assert!(error.contains("runtime diagnostics require the native host ABI v4"));
}

#[test]
fn plugin_runtime_reads_v4_runtime_diagnostics_extension() {
    let _guard = begin_test();
    let host = test_host_v4();
    let diagnostics = PluginRuntime::new(&host.v3.v1)
        .runtime_diagnostics()
        .unwrap();

    assert_eq!(diagnostics.entries().len(), 1);
    assert_eq!(diagnostics.entries()[0].code, "otel.example");
    assert_eq!(diagnostics.entries()[0].count, 3);
    assert_eq!(
        diagnostics
            .get("otel.example")
            .map(|diagnostic| diagnostic.message.as_str()),
        Some("example failure")
    );
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn plugin_runtime_rejects_malformed_runtime_diagnostics_json() {
    let _guard = begin_test();
    let mut host = test_host_v4();
    host.get_runtime_diagnostics = capture_malformed_runtime_diagnostics;

    let error = PluginRuntime::new(&host.v3.v1)
        .runtime_diagnostics()
        .unwrap_err();
    assert!(
        error.contains("invalid runtime diagnostics result"),
        "{error}"
    );
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn plugin_context_handles_an_unavailable_runtime_capability() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut context = test_context(&host.v3.v1);
    let runtime = context.runtime();
    let kinds = BTreeSet::from([nemo_relay_plugin::RuntimeRegistrationKind::Subscriber]);

    assert_eq!(UNAVAILABLE_RUNTIME_CONTEXT_CALLS.load(Ordering::SeqCst), 1);
    assert!(
        expect_string_err(runtime.list_runtime_registrations(None))
            .contains("host does not support activation-owned runtime gate control")
    );
    assert!(
        expect_string_err(runtime.register_conditional_middleware_guardrail(
            "timer-gate",
            &kinds,
            "target-subscriber",
            |_, _| Some("timer active".into()),
        ))
        .contains("host does not support activation-owned runtime gate control")
    );

    let cloned = runtime.clone();
    drop(cloned);
    drop(runtime);
    assert_eq!(UNAVAILABLE_RUNTIME_RETAINS.load(Ordering::SeqCst), 0);
    assert_eq!(UNAVAILABLE_RUNTIME_RELEASES.load(Ordering::SeqCst), 0);

    let error = context
        .register_conditional_middleware_guardrail(
            "startup-gate",
            &kinds,
            "target-subscriber",
            |_, _| Some("disabled".into()),
        )
        .unwrap_err();
    assert!(error.contains("NotFound"), "{error}");
    assert_eq!(UNAVAILABLE_CONTEXT_GATE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn callback_gates_require_the_appended_v4_table_prefix() {
    let _guard = begin_test();
    let mut host = test_host_v4();
    host.v3.v1.struct_size = offset_of!(
        NemoRelayNativeHostApiV4,
        plugin_runtime_register_conditional_middleware_guardrail_callback
    );
    let mut context = test_context(&host.v3.v1);
    let kinds = BTreeSet::from([nemo_relay_plugin::RuntimeRegistrationKind::Subscriber]);

    let error = context
        .register_conditional_middleware_guardrail(
            "startup-gate",
            &kinds,
            "target-subscriber",
            |_, _| Some("disabled".into()),
        )
        .unwrap_err();
    assert!(
        error.contains("host does not support conditional middleware guardrails"),
        "{error}"
    );
    assert_eq!(UNAVAILABLE_CONTEXT_GATE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn plugin_runtime_registration_controls_cover_success_and_lifecycle() {
    let _guard = begin_test();
    let mut host = test_host_v4();
    host.plugin_context_runtime = capture_plugin_context_runtime;
    host.plugin_runtime_retain = capture_plugin_runtime_retain;
    host.plugin_runtime_release = capture_plugin_runtime_release;
    host.plugin_runtime_list_registrations = capture_plugin_runtime_list_registrations;
    host.plugin_runtime_register_conditional_middleware_guardrail_callback =
        capture_plugin_runtime_register_gate_callback;
    host.plugin_runtime_deregister_conditional_middleware_guardrail =
        capture_plugin_runtime_deregister_gate;
    host.plugin_context_register_conditional_middleware_guardrail_callback =
        capture_plugin_context_register_gate_callback;
    let mut context = test_context(&host.v3.v1);
    let runtime = context.runtime();
    let kinds = BTreeSet::from([nemo_relay_plugin::RuntimeRegistrationKind::Subscriber]);

    let registrations = runtime
        .list_runtime_registrations(Some(&kinds))
        .expect("runtime registrations should decode");
    assert_eq!(
        registrations[0].effective_name,
        "nemo-relay-plugin.v1.example:1:target"
    );
    let handle = runtime
        .register_conditional_middleware_guardrail(
            "timer-gate",
            &kinds,
            "nemo-relay-plugin.v1.example:1:target",
            |_, registration_name| {
                registration_name
                    .contains(":target")
                    .then(|| "timer active".into())
            },
        )
        .expect("dynamic gate should register");
    assert!(
        runtime
            .deregister_conditional_middleware_guardrail(&handle)
            .unwrap()
    );
    context
        .register_conditional_middleware_guardrail(
            "startup-gate",
            &kinds,
            "nemo-relay-plugin.v1.example:1:target",
            |_, registration_name| {
                registration_name
                    .contains(":target")
                    .then(|| "startup disabled".into())
            },
        )
        .expect("initial gate should register");

    let clone = runtime.clone();
    drop(clone);
    drop(runtime);
    assert_eq!(UNAVAILABLE_RUNTIME_RETAINS.load(Ordering::SeqCst), 1);
    assert_eq!(UNAVAILABLE_RUNTIME_RELEASES.load(Ordering::SeqCst), 2);
    let calls = RUNTIME_CALLS.lock().unwrap();
    assert!(calls.iter().any(|call| call.starts_with("list:")));
    assert!(
        calls
            .iter()
            .any(|call| call.starts_with("register-callback:timer-gate:"))
    );
    assert!(calls.iter().any(|call| call == "deregister:gate-handle"));
    assert!(
        calls
            .iter()
            .any(|call| call.starts_with("initial-callback:startup-gate:"))
    );
    assert!(
        calls
            .iter()
            .any(|call| call == "decisions:timer active:none")
    );
    assert!(
        calls
            .iter()
            .any(|call| call == "decisions:startup disabled:none")
    );
    drop(calls);
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn conditional_middleware_callback_trampoline_covers_decisions_and_invalid_inputs() {
    let _guard = begin_test();
    let mut host = test_host_v4();
    host.plugin_context_register_conditional_middleware_guardrail_callback =
        exercise_plugin_context_register_gate_callback;
    let mut context = test_context(&host.v3.v1);
    use nemo_relay_plugin::RuntimeRegistrationKind as Kind;
    let kinds = BTreeSet::from([
        Kind::Subscriber,
        Kind::EventMetadataInjector,
        Kind::MarkSanitizeGuardrail,
        Kind::ScopeSanitizeStartGuardrail,
        Kind::ScopeSanitizeEndGuardrail,
        Kind::ToolSanitizeRequestGuardrail,
        Kind::ToolSanitizeResponseGuardrail,
        Kind::ToolConditionalExecutionGuardrail,
        Kind::ToolRequestIntercept,
        Kind::ToolExecutionIntercept,
        Kind::LlmSanitizeRequestGuardrail,
        Kind::LlmSanitizeResponseGuardrail,
        Kind::LlmConditionalExecutionGuardrail,
        Kind::LlmRequestIntercept,
        Kind::LlmExecutionIntercept,
        Kind::LlmStreamExecutionIntercept,
    ]);

    context
        .register_conditional_middleware_guardrail(
            "startup-gate",
            &kinds,
            "documentation-controlled-subscriber",
            |kinds, registration_name| {
                (kinds.len() == 16
                    && kinds.contains(&Kind::Subscriber)
                    && kinds.contains(&Kind::LlmStreamExecutionIntercept)
                    && registration_name == "documentation-controlled-subscriber")
                    .then(|| "disabled".into())
            },
        )
        .expect("the exercise host should accept callback registration");

    assert_eq!(
        *CONDITIONAL_CALLBACK_RESULTS.lock().unwrap(),
        [
            "blocked:Ok:disabled",
            "allowed:Ok:none",
            "malformed-kinds:InvalidJson:none",
            "null-kinds:NullPointer:none",
            "null-name:NullPointer:none",
            "null-output:NullPointer:none",
            "reason-allocation:Internal:none",
        ]
    );
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn conditional_middleware_callback_panics_are_caught_and_state_is_freed() {
    let _guard = begin_test();
    let mut host = test_host_v4();
    host.plugin_context_register_conditional_middleware_guardrail_callback =
        capture_plugin_context_register_gate_callback;
    let mut context = test_context(&host.v3.v1);
    let kinds = BTreeSet::from([nemo_relay_plugin::RuntimeRegistrationKind::Subscriber]);
    let drops = Arc::new(AtomicUsize::new(0));
    let drop_probe = CountDrop(Arc::clone(&drops));

    let error = context
        .register_conditional_middleware_guardrail(
            "panic-gate",
            &kinds,
            "target-subscriber",
            move |_, _| {
                let _ = &drop_probe;
                panic!("conditional callback panic")
            },
        )
        .unwrap_err();

    assert!(error.contains("Internal"), "{error}");
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}

fn assert_scope_runtime_calls(calls: &[String]) {
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
}

fn assert_stack_runtime_calls(calls: &[String]) {
    assert!(calls.iter().any(|call| call == "stack_create"));
    assert!(calls.iter().any(|call| call == "stack_with_current"));
    assert!(calls.iter().any(|call| call == "stack_capture"));
    assert!(calls.iter().any(|call| call == "stack_set_thread"));
    assert!(calls.iter().any(|call| call == "stack_restore"));
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
#[allow(clippy::cognitive_complexity)] // Exercises each independent native stream ownership outcome.
fn native_llm_stream_validates_callbacks_and_releases_host_resources() {
    let _guard = begin_test();
    let host = test_host();

    let default_stream = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        default_stream.struct_size,
        size_of::<NemoRelayNativeLlmStreamV1>()
    );
    assert!(default_stream.user_data.is_null());
    assert!(default_stream.next.is_none());
    assert!(default_stream.cancel.is_none());
    assert!(default_stream.drop.is_none());

    let invalid_size = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut raw = native_stream_raw(&invalid_size);
    raw.struct_size -= 1;
    assert_eq!(
        expect_string_err(unsafe { LlmStream::from_raw(&host, raw) }),
        format!(
            "unsupported LLM stream struct size: {}",
            size_of::<NemoRelayNativeLlmStreamV1>() - 1
        )
    );
    assert_eq!(invalid_size.drops.load(Ordering::SeqCst), 0);

    let extended_size = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut raw = native_stream_raw(&extended_size);
    raw.struct_size += 1;
    assert_eq!(
        expect_string_err(unsafe { LlmStream::from_raw(&host, raw) }),
        format!(
            "unsupported LLM stream struct size: {}",
            size_of::<NemoRelayNativeLlmStreamV1>() + 1
        )
    );
    assert_eq!(extended_size.drops.load(Ordering::SeqCst), 1);

    let missing_next = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut raw = native_stream_raw(&missing_next);
    raw.next = None;
    assert_eq!(
        expect_string_err(unsafe { LlmStream::from_raw(&host, raw) }),
        "LLM stream next callback was null"
    );
    assert_eq!(missing_next.drops.load(Ordering::SeqCst), 1);

    let stream_state = MockNativeStream {
        steps: Mutex::new(VecDeque::from([
            (NemoRelayStatus::Ok, Some(br#"{"chunk":1}"#.to_vec())),
            (
                NemoRelayStatus::StreamEnd,
                Some(br#"{"ignored":true}"#.to_vec()),
            ),
        ])),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut stream =
        unsafe { LlmStream::from_raw(&host, native_stream_raw(&stream_state)) }.unwrap();
    assert_eq!(stream.next_chunk().unwrap(), Some(json!({ "chunk": 1 })));
    assert_eq!(stream.next_chunk().unwrap(), None);
    assert_eq!(stream.next_chunk().unwrap(), None);
    drop(stream);
    assert_eq!(stream_state.cancellations.load(Ordering::SeqCst), 0);
    assert_eq!(stream_state.drops.load(Ordering::SeqCst), 1);

    let invalid_json = MockNativeStream {
        steps: Mutex::new(VecDeque::from([(NemoRelayStatus::Ok, Some(b"{".to_vec()))])),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut stream =
        unsafe { LlmStream::from_raw(&host, native_stream_raw(&invalid_json)) }.unwrap();
    assert!(
        stream
            .next_chunk()
            .unwrap_err()
            .starts_with("LLM stream returned invalid JSON:")
    );
    drop(stream);
    assert_eq!(invalid_json.cancellations.load(Ordering::SeqCst), 0);
    assert_eq!(invalid_json.drops.load(Ordering::SeqCst), 1);

    let failure = MockNativeStream {
        steps: Mutex::new(VecDeque::from([(
            NemoRelayStatus::Internal,
            Some(br#"{}"#.to_vec()),
        )])),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut stream = unsafe { LlmStream::from_raw(&host, native_stream_raw(&failure)) }.unwrap();
    assert_eq!(
        stream.next().unwrap().unwrap_err(),
        "LLM stream failed: Internal"
    );
    drop(stream);
    assert_eq!(failure.cancellations.load(Ordering::SeqCst), 0);
    assert_eq!(failure.drops.load(Ordering::SeqCst), 1);

    let null_chunk = MockNativeStream {
        steps: Mutex::new(VecDeque::from([(NemoRelayStatus::Ok, None)])),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut stream = unsafe { LlmStream::from_raw(&host, native_stream_raw(&null_chunk)) }.unwrap();
    assert_eq!(
        stream.next().unwrap().unwrap_err(),
        "LLM stream returned null chunk"
    );
    drop(stream);
    assert_eq!(null_chunk.cancellations.load(Ordering::SeqCst), 0);

    let no_cancel = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut raw = native_stream_raw(&no_cancel);
    raw.cancel = None;
    let mut stream = unsafe { LlmStream::from_raw(&host, raw) }.unwrap();
    stream.cancel().unwrap();
    drop(stream);
    assert_eq!(no_cancel.cancellations.load(Ordering::SeqCst), 0);
    assert_eq!(no_cancel.drops.load(Ordering::SeqCst), 1);

    let cancellable = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Ok,
    };
    let mut stream =
        unsafe { LlmStream::from_raw(&host, native_stream_raw(&cancellable)) }.unwrap();
    stream.cancel().unwrap();
    stream.cancel().unwrap();
    drop(stream);
    assert_eq!(cancellable.cancellations.load(Ordering::SeqCst), 1);
    assert_eq!(cancellable.drops.load(Ordering::SeqCst), 1);

    let cancel_failure = MockNativeStream {
        steps: Mutex::new(VecDeque::new()),
        cancellations: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_status: NemoRelayStatus::Internal,
    };
    let mut stream =
        unsafe { LlmStream::from_raw(&host, native_stream_raw(&cancel_failure)) }.unwrap();
    assert_eq!(
        stream.cancel().unwrap_err(),
        "LLM stream cancel failed: Internal"
    );
    drop(stream);
    assert_eq!(cancel_failure.cancellations.load(Ordering::SeqCst), 2);
    assert_eq!(cancel_failure.drops.load(Ordering::SeqCst), 1);
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
#[allow(clippy::cognitive_complexity)] // One table-style test deliberately exercises every surface.
fn typed_async_middleware_registers_and_round_trips_every_surface() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);

    ctx.register_mark_sanitize_guardrail("mark-async", 1, |_event, mut fields| async move {
        tokio::time::sleep(Duration::from_millis(1)).await;
        fields.metadata = Some(json!({ "surface": "mark" }));
        Ok(fields)
    })
    .unwrap();
    ctx.register_scope_sanitize_start_guardrail(
        "scope-start-async",
        2,
        |_event, fields| async move { Ok(fields) },
    )
    .unwrap();
    ctx.register_scope_sanitize_end_guardrail("scope-end-async", 3, |_event, fields| async move {
        Ok(fields)
    })
    .unwrap();
    ctx.register_tool_sanitize_request_guardrail(
        "tool-request-async",
        4,
        |name, mut value| async move {
            value["surface"] = json!(name);
            Ok(value)
        },
    )
    .unwrap();
    ctx.register_tool_sanitize_response_guardrail(
        "tool-response-async",
        5,
        |name, mut value| async move {
            value["surface"] = json!(name);
            Ok(value)
        },
    )
    .unwrap();
    ctx.register_tool_conditional_execution_guardrail(
        "tool-conditional-async",
        6,
        |name, _| async move { Ok(Some(format!("blocked {name}"))) },
    )
    .unwrap();
    ctx.register_tool_request_intercept(
        "tool-intercept-async",
        7,
        true,
        |name, mut value| async move {
            value["intercepted"] = json!(name);
            Ok(value)
        },
    )
    .unwrap();
    ctx.register_tool_execution_intercept(
        "tool-execution-async",
        8,
        |_name, value, next| async move {
            let result = next.call(value).await?;
            Ok(ToolExecutionInterceptOutcome::from(result))
        },
    )
    .unwrap();
    ctx.register_llm_sanitize_request_guardrail(
        "llm-request-async",
        9,
        |request, context| async move {
            assert_eq!(
                context.codec,
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
            );
            let codec = context.resolve_codec().expect("request codec capability");
            let annotated = codec.decode(&request)?;
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(Some(codec.encode(&annotated, &request)?))
        },
    )
    .unwrap();
    ctx.register_llm_sanitize_response_guardrail(
        "llm-response-async",
        10,
        |response, context| async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let codec = context.resolve_codec().expect("response codec capability");
            let _ = codec.decode(&response)?;
            Ok(Some(response))
        },
    )
    .unwrap();
    ctx.register_llm_conditional_execution_guardrail(
        "llm-conditional-async",
        11,
        |request| async move { Ok(Some(format!("blocked {}", request.content["prompt"]))) },
    )
    .unwrap();
    ctx.register_llm_request_intercept(
        "llm-intercept-async",
        12,
        true,
        |_name, mut request, annotated| async move {
            request.headers.insert("x-tested".into(), json!(true));
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        },
    )
    .unwrap();
    ctx.register_llm_execution_intercept(
        "llm-execution-async",
        13,
        |_name, request, next| async move { next.call(request).await },
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "llm-stream-async",
        14,
        |_name, request, next| async move {
            let stream = next.call(request).await?;
            let transformed = stream.map(|item| item.map(|chunk| json!({ "wrapped": chunk })));
            Ok(Box::pin(transformed) as LlmJsonAsyncStream)
        },
    )
    .unwrap();
    ctx.register_event_metadata_injector("event-metadata-async", 15, |event| async move {
        Ok(std::collections::BTreeMap::from([(
            "plugin.event_name".into(),
            json!(event.name()),
        )]))
    })
    .unwrap();

    let metadata = ASYNC_REGISTRATIONS
        .lock()
        .unwrap()
        .iter()
        .map(|registration| {
            (
                registration.kind,
                registration.name.clone(),
                registration.priority,
                registration.break_chain,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(metadata.len(), 14);
    assert_eq!(metadata[0].1, "mark-async");
    assert_eq!(metadata[0].2, 1);
    assert_eq!(
        metadata[6].0,
        NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept
    );
    assert!(metadata[6].3);
    assert_eq!(
        metadata[11].0,
        NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept
    );
    assert!(metadata[11].3);
    {
        let stream_registration = ASYNC_STREAM_REGISTRATION.lock().unwrap();
        let stream_registration = stream_registration.as_ref().unwrap();
        assert_eq!(
            (
                stream_registration.name.as_str(),
                stream_registration.priority
            ),
            ("llm-stream-async", 14)
        );
    }

    let event = json!({
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "00000000-0000-0000-0000-000000000000",
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "checkpoint"
    });
    let metadata_registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::EventMetadataInjector);
    let result = invoke_async_registration(
        &host,
        &metadata_registration,
        json!({ "event": event }),
        None,
    )
    .unwrap();
    assert_eq!(result, json!({ "plugin.event_name": "checkpoint" }));
    unsafe { metadata_registration.free() };

    for kind in [
        NemoRelayNativeAsyncMiddlewareKind::MarkSanitize,
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart,
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd,
    ] {
        let registration = take_async_registration(kind);
        let result = invoke_async_registration(
            &host,
            &registration,
            json!({ "event": event, "fields": { "data": { "visible": true } } }),
            None,
        )
        .unwrap();
        assert_eq!(result["data"], json!({ "visible": true }));
        if kind == NemoRelayNativeAsyncMiddlewareKind::MarkSanitize {
            assert_eq!(result["metadata"], json!({ "surface": "mark" }));
        }
        unsafe { registration.free() };
    }

    for kind in [
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest,
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse,
        NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
    ] {
        let registration = take_async_registration(kind);
        let result = invoke_async_registration(
            &host,
            &registration,
            json!({ "name": "calculator", "value": { "x": 1 } }),
            None,
        )
        .unwrap();
        assert_eq!(result["x"], 1);
        unsafe { registration.free() };
    }

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({ "name": "calculator", "value": {} }),
            None,
        )
        .unwrap(),
        json!("blocked calculator")
    );
    unsafe { registration.free() };

    let pull_stream = MockPullStream {
        items: Mutex::new(VecDeque::from([
            Ok(Some(json!({ "chunk": 1 }))),
            Ok(Some(json!({ "chunk": 2 }))),
            Ok(None),
        ])),
        cancelled: AtomicBool::new(false),
        releases: AtomicUsize::new(0),
    };
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: pull_stream.raw(),
    };
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept);
    let outcome = invoke_async_registration(
        &host,
        &registration,
        json!({ "name": "calculator", "value": { "answer": 42 } }),
        Some(&next),
    );
    let outcome = outcome.unwrap();
    assert_eq!(outcome["result"], json!({ "answer": 42 }));
    assert_eq!(outcome["annotation"], json!({"source": "host"}));
    unsafe { registration.free() };

    let request = test_llm_request();
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({
                "request": request,
                "context": { "codec_kind": "builtin", "codec_id": "openai_chat" }
            }),
            None,
        )
        .unwrap()["content"],
        json!({ "prompt": "hello" })
    );
    assert_eq!(ASYNC_COMPLETION_RETAINS.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({
                "response": { "answer": 42 },
                "context": { "codec_kind": "runtime", "codec_id": "test" }
            }),
            None,
        )
        .unwrap(),
        json!({ "answer": 42 })
    );
    assert_eq!(ASYNC_COMPLETION_RETAINS.load(Ordering::SeqCst), 2);
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({ "request": test_llm_request() }),
            None
        )
        .unwrap(),
        json!("blocked \"hello\"")
    );
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept);
    let result = invoke_async_registration(
        &host,
        &registration,
        json!({ "name": "provider", "request": test_llm_request(), "annotated": null }),
        None,
    )
    .unwrap();
    assert_eq!(result["request"]["headers"]["x-tested"], true);
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({ "name": "provider", "request": test_llm_request() }),
            Some(&next),
        )
        .unwrap()["content"],
        json!({ "prompt": "hello" })
    );
    unsafe { registration.free() };

    let stream_registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    ASYNC_PUSH_BACKPRESSURE.store(2, Ordering::SeqCst);
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    let state = unsafe {
        (stream_registration.cb)(
            stream_registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    assert_eq!(
        output.wait_terminal(),
        vec![
            MockOutputEvent::Chunk(json!({ "wrapped": { "chunk": 1 } })),
            MockOutputEvent::Chunk(json!({ "wrapped": { "chunk": 2 } })),
            MockOutputEvent::Finished,
        ]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    assert_eq!(pull_stream.releases.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 3);
    unsafe { stream_registration.free() };
    drop(ctx);
    assert!(ASYNC_REGISTRATIONS.lock().unwrap().is_empty());
    assert!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst) > 0);
    assert!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst) > 0);
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn typed_async_llm_sanitize_context_decodes_oci_genai_builtin_identity() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);

    ctx.register_llm_sanitize_request_guardrail(
        "llm-request-oci-genai",
        0,
        |request, context| async move {
            assert_eq!(
                context.codec,
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
            );
            Ok(Some(request))
        },
    )
    .unwrap();

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({
                "request": test_llm_request(),
                "context": { "codec_kind": "builtin", "codec_id": "oci_genai" }
            }),
            None,
        )
        .unwrap()["content"],
        json!({ "prompt": "hello" })
    );
    unsafe { registration.free() };
    // The context's retained codec capability is released on the SDK executor
    // after the result completion is delivered, so poll instead of asserting
    // immediately.
    let deadline = Instant::now() + Duration::from_secs(5);
    while live_host_strings() != 0 {
        assert!(
            Instant::now() < deadline,
            "host strings were not released after the sanitize invocation"
        );
        std::thread::yield_now();
    }
}

#[test]
fn typed_async_llm_sanitize_context_decodes_all_builtin_identities() {
    let _guard = begin_test();

    for expected in [
        BuiltinLlmCodec::OpenAiChat,
        BuiltinLlmCodec::OpenAiResponses,
        BuiltinLlmCodec::AnthropicMessages,
        BuiltinLlmCodec::OCIGenAI,
        BuiltinLlmCodec::GeminiGenerateContent,
    ] {
        let host = test_host_v4();
        let mut ctx = test_context(&host.v3.v1);
        ctx.register_llm_sanitize_request_guardrail(
            "llm-request-builtins",
            0,
            move |request, context| async move {
                assert_eq!(context.codec, LlmCodecIdentity::BuiltIn(expected));
                Ok(Some(request))
            },
        )
        .unwrap();

        let registration =
            take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest);
        assert_eq!(
            invoke_async_registration(
                &host,
                &registration,
                json!({
                    "request": test_llm_request(),
                    "context": { "codec_kind": "builtin", "codec_id": expected.id() }
                }),
                None,
            )
            .unwrap()["content"],
            json!({ "prompt": "hello" })
        );
        unsafe { registration.free() };

        let deadline = Instant::now() + Duration::from_secs(5);
        while live_host_strings() != 0 {
            assert!(
                Instant::now() < deadline,
                "host strings were not released after the sanitize invocation"
            );
            std::thread::yield_now();
        }
    }
}

#[test]
fn typed_async_llm_sanitize_context_rejects_unknown_builtin_identity() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_sanitize_request_guardrail(
        "llm-request-unknown",
        0,
        |request, _context| async move { Ok(Some(request)) },
    )
    .unwrap();

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest);
    let error = invoke_async_registration(
        &host,
        &registration,
        json!({
            "request": test_llm_request(),
            "context": { "codec_kind": "builtin", "codec_id": "future_provider" }
        }),
        None,
    )
    .expect_err("unknown built-in codec identities must be rejected");
    assert!(error.contains("unknown built-in LLM codec: future_provider"));
    unsafe { registration.free() };
}

#[test]
fn typed_async_registration_failure_rolls_back_callback_state() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    *REGISTRATION_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;

    let unary_drops = Arc::new(AtomicUsize::new(0));
    let unary_probe = CountDrop(unary_drops.clone());
    let result =
        ctx.register_tool_request_intercept("rejected-unary", 0, false, move |_name, value| {
            let _ = &unary_probe;
            async move { Ok(value) }
        });
    assert!(result.unwrap_err().contains("InvalidArg"));
    assert_eq!(unary_drops.load(Ordering::SeqCst), 1);

    let stream_drops = Arc::new(AtomicUsize::new(0));
    let stream_probe = CountDrop(stream_drops.clone());
    let result = ctx.register_llm_stream_execution_intercept(
        "rejected-stream",
        0,
        move |_name, _request, _next| {
            let _ = &stream_probe;
            async move { Ok(Box::pin(futures::stream::empty()) as LlmJsonAsyncStream) }
        },
    );
    assert!(result.unwrap_err().contains("InvalidArg"));
    assert_eq!(stream_drops.load(Ordering::SeqCst), 1);
    assert!(ASYNC_REGISTRATIONS.lock().unwrap().is_empty());
    assert!(ASYNC_STREAM_REGISTRATION.lock().unwrap().is_none());
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn typed_async_callbacks_isolate_errors_panics_and_invalid_input() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);

    ctx.register_tool_request_intercept("error", 0, false, |_name, _value| async move {
        Err("callback failed".into())
    })
    .unwrap();
    ctx.register_tool_request_intercept("panic", 0, false, |_name, _value| async move {
        tokio::time::sleep(Duration::from_millis(1)).await;
        panic!("future exploded")
    })
    .unwrap();
    ctx.register_tool_request_intercept(
        "decode",
        0,
        false,
        |_name, value| async move { Ok(value) },
    )
    .unwrap();

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({ "name": "tool", "value": {} }),
            None,
        )
        .unwrap_err(),
        "callback failed"
    );
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    assert_eq!(
        invoke_async_registration(
            &host,
            &registration,
            json!({ "name": "tool", "value": {} }),
            None,
        )
        .unwrap_err(),
        "typed native middleware future panicked"
    );
    unsafe { registration.free() };

    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    let error = invoke_async_registration(&host, &registration, json!({ "wrong": true }), None)
        .unwrap_err();
    assert!(
        error.contains("missing field"),
        "unexpected decode error: {error}"
    );
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn typed_async_continuations_are_concurrent_and_executor_owned() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_tool_execution_intercept("concurrent", 0, |_name, _value, next| async move {
        tokio::time::sleep(Duration::from_millis(1)).await;
        let thread = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_string();
        let (left, right) = tokio::join!(
            next.call(json!({ "call": 1 })),
            next.call(json!({ "call": 2 }))
        );
        let left = left?;
        let right = right?;
        Ok(ToolExecutionInterceptOutcome::new(json!({
            "thread": thread,
            "results": [left.result, right.result],
            "annotations": [left.annotation, right.annotation],
        })))
    })
    .unwrap();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept);
    let result = invoke_async_registration(
        &host,
        &registration,
        json!({ "name": "tool", "value": {} }),
        Some(&next),
    )
    .unwrap();
    assert!(
        result["result"]["thread"]
            .as_str()
            .unwrap()
            .starts_with("nemo-relay-plugin-")
    );
    assert_eq!(result["result"]["results"][0], json!({ "call": 1 }));
    assert_eq!(result["result"]["results"][1], json!({ "call": 2 }));
    assert_eq!(
        result["result"]["annotations"],
        json!([{"source": "host"}, {"source": "host"}])
    );
    assert_eq!(next.calls.load(Ordering::SeqCst), 2);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };

    ctx.register_llm_stream_execution_intercept(
        "stream-open-error",
        0,
        |_name, request, next| async move { next.call(request).await },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        output.wait_terminal(),
        vec![MockOutputEvent::Rejected(
            "host returned neither an LLM stream nor an error".into()
        )]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_cancellation_drops_future_and_releases_owned_handles() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    let future_drops = Arc::new(AtomicUsize::new(0));
    ctx.register_tool_execution_intercept("cancel", 0, {
        let future_drops = future_drops.clone();
        move |_name, _value, _next| {
            let probe = CountDrop(future_drops.clone());
            async move {
                let _probe = probe;
                futures::future::pending::<std::result::Result<
                    ToolExecutionInterceptOutcome,
                    String,
                >>()
                .await
            }
        }
    })
    .unwrap();
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept);
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let completion = MockAsyncCompletion::new();
    completion.cancelled.store(true, Ordering::SeqCst);
    let invocation = json_host_string(&host.v3.v1, json!({ "name": "tool", "value": {} }));
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            completion.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    completion.wait_for_release();
    let deadline = Instant::now() + Duration::from_secs(5);
    while future_drops.load(Ordering::SeqCst) == 0 || next.releases.load(Ordering::SeqCst) == 0 {
        assert!(
            Instant::now() < deadline,
            "cancelled state was not reclaimed"
        );
        std::thread::yield_now();
    }
    assert!(completion.settled.lock().unwrap().is_none());
    assert_eq!(completion.releases.load(Ordering::SeqCst), 1);
    assert_eq!(future_drops.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_cancellation_while_awaiting_reclaims_future() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    let started = Arc::new(AtomicBool::new(false));
    let future_drops = Arc::new(AtomicUsize::new(0));
    ctx.register_tool_request_intercept("cancel-await", 0, false, {
        let started = started.clone();
        let future_drops = future_drops.clone();
        move |_name, _value| {
            let started = started.clone();
            let probe = CountDrop(future_drops.clone());
            async move {
                let _probe = probe;
                started.store(true, Ordering::SeqCst);
                futures::future::pending::<std::result::Result<Json, String>>().await
            }
        }
    })
    .unwrap();
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    let completion = MockAsyncCompletion::new();
    let invocation = json_host_string(&host.v3.v1, json!({ "name": "tool", "value": {} }));
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            ptr::null(),
            completion.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    let deadline = Instant::now() + Duration::from_secs(5);
    while !started.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "callback future never started");
        std::thread::yield_now();
    }
    completion.cancelled.store(true, Ordering::SeqCst);
    completion.wait_for_release();
    while future_drops.load(Ordering::SeqCst) == 0 {
        assert!(
            Instant::now() < deadline,
            "cancelled future was not dropped"
        );
        std::thread::yield_now();
    }
    assert!(completion.settled.lock().unwrap().is_none());
    assert_eq!(completion.releases.load(Ordering::SeqCst), 1);
    assert_eq!(future_drops.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_executor_drop_drains_accepted_tasks() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
    let finish_rx = Arc::new(Mutex::new(Some(finish_rx)));
    ctx.register_tool_request_intercept("drain", 0, false, {
        let finish_rx = Arc::clone(&finish_rx);
        move |_name, _value| {
            let finish_rx = Arc::clone(&finish_rx);
            let started_tx = started_tx.clone();
            async move {
                started_tx.send(()).unwrap();
                let finish_rx = finish_rx.lock().unwrap().take().unwrap();
                finish_rx.await.unwrap();
                Ok(json!({"drained": true}))
            }
        }
    })
    .unwrap();
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    let completion = MockAsyncCompletion::new();
    let invocation = json_host_string(&host.v3.v1, json!({ "name": "tool", "value": {} }));
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            ptr::null(),
            completion.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    unsafe { registration.free() };

    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        finish_tx.send(()).unwrap();
    });
    let drop_started = Instant::now();
    drop(ctx);
    assert!(
        drop_started.elapsed() >= Duration::from_millis(50),
        "executor shutdown returned before the accepted task completed"
    );
    releaser.join().unwrap();
    assert_eq!(completion.wait(), Ok(json!({"drained": true})));
}

#[test]
fn typed_async_executor_drop_inside_tokio_runtime_drains_accepted_tasks() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
    let finish_rx = Arc::new(Mutex::new(Some(finish_rx)));
    ctx.register_tool_request_intercept("drain-in-runtime", 0, false, {
        let finish_rx = Arc::clone(&finish_rx);
        move |_name, _value| {
            let finish_rx = Arc::clone(&finish_rx);
            let started_tx = started_tx.clone();
            async move {
                started_tx.send(()).unwrap();
                let finish_rx = finish_rx.lock().unwrap().take().unwrap();
                finish_rx.await.unwrap();
                Ok(json!({"drained": true}))
            }
        }
    })
    .unwrap();
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    let completion = MockAsyncCompletion::new();
    let invocation = json_host_string(&host.v3.v1, json!({ "name": "tool", "value": {} }));
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            ptr::null(),
            completion.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(ctx);

    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        finish_tx.send(()).unwrap();
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let drop_started = Instant::now();
    runtime.block_on(async move { unsafe { registration.free() } });
    assert!(
        drop_started.elapsed() >= Duration::from_millis(50),
        "executor shutdown returned before the accepted task completed"
    );
    releaser.join().unwrap();
    assert_eq!(completion.wait(), Ok(json!({"drained": true})));
}

#[test]
fn typed_async_stream_cancellation_while_polling_releases_output() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    let started = Arc::new(AtomicBool::new(false));
    ctx.register_llm_stream_execution_intercept("cancel-poll", 0, {
        let started = Arc::clone(&started);
        move |_name, _request, _next| {
            let started = Arc::clone(&started);
            async move {
                Ok(Box::pin(futures::stream::poll_fn(move |_| {
                    started.store(true, Ordering::SeqCst);
                    Poll::Pending
                })) as LlmJsonAsyncStream)
            }
        }
    })
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    let deadline = Instant::now() + Duration::from_secs(5);
    while !started.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "returned stream was not polled");
        std::thread::yield_now();
    }
    output.cancelled.store(true, Ordering::SeqCst);
    output.wait_for_release();
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_stream_restores_callback_scope_while_polling_returned_stream() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_stream_execution_intercept(
        "stream-scope",
        0,
        |_name, _request, _next| async move {
            Ok(Box::pin(futures::stream::iter([Ok(json!({ "chunk": 1 }))])) as LlmJsonAsyncStream)
        },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        output.wait_terminal(),
        vec![
            MockOutputEvent::Chunk(json!({ "chunk": 1 })),
            MockOutputEvent::Finished,
        ]
    );
    output.wait_for_release();
    assert!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst) >= 3);
    unsafe { registration.free() };
}

#[test]
fn typed_async_stream_rejects_item_errors_and_releases_output() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_stream_execution_intercept(
        "stream-error",
        0,
        |_name, _request, _next| async move {
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(json!({ "chunk": 1 })),
                Err("stream item failed".into()),
            ])) as LlmJsonAsyncStream)
        },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        output.wait_terminal(),
        vec![
            MockOutputEvent::Chunk(json!({ "chunk": 1 })),
            MockOutputEvent::Rejected("stream item failed".into()),
        ]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_stream_rejects_poll_panics_and_releases_output() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_stream_execution_intercept(
        "stream-panic",
        0,
        |_name, _request, _next| async move {
            let mut polled = false;
            Ok(Box::pin(futures::stream::poll_fn(move |_| {
                if polled {
                    panic!("stream poll panic");
                }
                polled = true;
                Poll::Ready(Some(Ok(json!({ "chunk": 1 }))))
            })) as LlmJsonAsyncStream)
        },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: ptr::null(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        output.wait_terminal(),
        vec![
            MockOutputEvent::Chunk(json!({ "chunk": 1 })),
            MockOutputEvent::Rejected("typed native stream panicked while polling".into()),
        ]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_stream_propagates_downstream_pull_errors() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_stream_execution_intercept(
        "stream-downstream-error",
        0,
        |_name, request, next| async move { next.call(request).await },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let pull_stream = MockPullStream {
        items: Mutex::new(VecDeque::from([Err("downstream pull failed".into())])),
        cancelled: AtomicBool::new(false),
        releases: AtomicUsize::new(0),
    };
    let next = MockAsyncNext {
        calls: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        pull_stream: pull_stream.raw(),
    };
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next.raw(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        output.wait_terminal(),
        vec![MockOutputEvent::Rejected("downstream pull failed".into())]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    assert_eq!(pull_stream.releases.load(Ordering::SeqCst), 1);
    assert_eq!(next.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
}

#[test]
fn typed_async_stream_rejects_missing_continuation() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_llm_stream_execution_intercept(
        "stream-null-next",
        0,
        |_name, _request, _next| async move {
            panic!("stream callback must not run without a continuation")
        },
    )
    .unwrap();
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    let output = MockAsyncOutput::new();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "provider", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            ptr::null(),
            output.raw(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    assert_eq!(
        output.wait_terminal(),
        vec![MockOutputEvent::Rejected(
            "native stream middleware requires a continuation".into()
        )]
    );
    output.wait_for_release();
    assert_eq!(output.releases.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
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
fn raw_callback_registrations_preserve_every_middleware_shape() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);

    unsafe {
        assert_eq!(
            ctx.register_tool_sanitize_request_guardrail_raw(
                "raw-tool-request",
                1,
                passthrough_tool_json_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_tool_json_registration().free();
        assert_eq!(
            ctx.register_tool_sanitize_response_guardrail_raw(
                "raw-tool-response",
                2,
                passthrough_tool_json_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_tool_json_registration().free();
        assert_eq!(
            ctx.register_tool_conditional_execution_guardrail_raw(
                "raw-tool-conditional",
                3,
                passthrough_tool_conditional_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_tool_conditional_registration().free();
        assert_eq!(
            ctx.register_tool_request_intercept_raw(
                "raw-tool-intercept",
                4,
                true,
                passthrough_tool_json_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let tool_intercept = take_tool_json_registration();
        assert!(tool_intercept.break_chain);
        tool_intercept.free();
        assert_eq!(
            ctx.register_tool_execution_intercept_raw(
                "raw-tool-execution",
                5,
                passthrough_tool_execution_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_tool_execution_registration().free();

        assert_eq!(
            ctx.register_llm_sanitize_request_guardrail_raw(
                "raw-llm-request",
                6,
                passthrough_llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_llm_request_registration().free();
        assert_eq!(
            ctx.register_llm_sanitize_response_guardrail_raw(
                "raw-llm-response",
                7,
                passthrough_llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_llm_json_registration().free();
        assert_eq!(
            ctx.register_llm_conditional_execution_guardrail_raw(
                "raw-llm-conditional",
                8,
                passthrough_llm_conditional_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_llm_conditional_registration().free();
        assert_eq!(
            ctx.register_llm_request_intercept_raw(
                "raw-llm-intercept",
                9,
                true,
                passthrough_llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let llm_intercept = take_llm_request_intercept_registration();
        assert!(llm_intercept.break_chain);
        llm_intercept.free();
        assert_eq!(
            ctx.register_llm_execution_intercept_raw(
                "raw-llm-execution",
                10,
                passthrough_llm_execution_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_llm_execution_registration().free();
        assert_eq!(
            ctx.register_llm_stream_execution_intercept_raw(
                "raw-llm-stream",
                11,
                passthrough_llm_stream_execution_cb,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        take_llm_stream_execution_registration().free();
    }
}

#[test]
fn raw_async_callback_registrations_use_the_v3_extension_tables() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);

    assert_eq!(
        unsafe {
            ctx.register_async_middleware_raw(
                NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
                "raw-async-tool",
                1,
                true,
                pending_async_middleware_cb,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let registration =
        take_async_registration(NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept);
    assert_eq!(registration.name, "raw-async-tool");
    assert!(registration.break_chain);
    unsafe { registration.free() };

    assert_eq!(
        unsafe {
            ctx.register_async_stream_middleware_raw(
                "raw-async-stream",
                2,
                pending_async_stream_middleware_cb,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let registration = ASYNC_STREAM_REGISTRATION.lock().unwrap().take().unwrap();
    assert_eq!(registration.name, "raw-async-stream");
    assert_eq!(registration.priority, 2);
    unsafe { registration.free() };

    RAW_ASYNC_REJECTIONS.store(0, Ordering::SeqCst);
    let legacy_host = test_host();
    let mut legacy_ctx = test_context(&legacy_host);
    assert_eq!(
        unsafe {
            legacy_ctx.register_async_middleware_raw(
                NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
                "legacy-async-tool",
                0,
                false,
                pending_async_middleware_cb,
                ptr::null_mut(),
                Some(count_raw_async_rejection),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe {
            legacy_ctx.register_async_stream_middleware_raw(
                "legacy-async-stream",
                0,
                pending_async_stream_middleware_cb,
                ptr::null_mut(),
                Some(count_raw_async_rejection),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(RAW_ASYNC_REJECTIONS.load(Ordering::SeqCst), 2);
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

struct ZeroWorkerPlugin;

impl NativePlugin for ZeroWorkerPlugin {
    fn plugin_kind(&self) -> &str {
        "test.zero_worker"
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 0 }
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
fn direct_export_plugin_rejects_zero_executor_workers() {
    let _guard = begin_test();
    let host = test_host();
    let mut plugin = NemoRelayNativePluginV1::default();

    assert_eq!(
        unsafe { nemo_relay_plugin::export_plugin(&host, &mut plugin, ZeroWorkerPlugin) },
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
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("executor.worker_threads must be greater than zero")
    );
    unsafe {
        (host.string_free)(config);
        drop_exported_plugin(&host, plugin);
    }
}

#[test]
fn native_executor_config_defaults_to_two_workers() {
    assert_eq!(
        NativeExecutorConfig::default(),
        NativeExecutorConfig { worker_threads: 2 }
    );
}

#[test]
fn native_executor_config_reads_component_executor_override() {
    let config = serde_json::from_value(json!({
        "executor": { "worker_threads": 4 }
    }))
    .unwrap();
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&config)
            .unwrap(),
        NativeExecutorConfig { worker_threads: 4 }
    );
    let invalid = serde_json::from_value(json!({
        "executor": { "worker_threads": 0 }
    }))
    .unwrap();
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&invalid)
            .unwrap_err(),
        "executor.worker_threads must be greater than zero"
    );
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
fn exported_entry_symbol_accepts_supported_prior_host_versions() {
    let _guard = begin_test();
    CONSTRUCTOR_CALLS.store(0, Ordering::SeqCst);

    for abi_version in [
        NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY,
        NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE,
        NEMO_RELAY_NATIVE_ABI_VERSION,
    ] {
        let mut host = test_host();
        host.abi_version = abi_version;
        let mut plugin = NemoRelayNativePluginV1::default();

        assert_eq!(
            unsafe { constructor_counting_entry(&host, &mut plugin) },
            NemoRelayStatus::Ok
        );
        unsafe { drop_exported_plugin(&host, plugin) };
    }

    assert_eq!(CONSTRUCTOR_CALLS.load(Ordering::SeqCst), 3);
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
