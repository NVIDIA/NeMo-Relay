// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-API tests for typed native plugin callback registration.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::future::Future;
use std::mem::{align_of, offset_of, size_of};
use std::ptr::{self, NonNull};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use futures::{StreamExt, stream};
use nemo_relay_plugin::{
    AnnotatedLlmRequest, BuiltinLlmCodec, CategoryProfile, ConfigDiagnostic, DiagnosticLevel,
    Event, EventCategory, EventSanitizeFields, Json, LlmCodecIdentity, LlmContinuationFailureV2,
    LlmContinuationInvocationV2, LlmContinuationTargetV2, LlmContinuationV2, LlmJsonStream,
    LlmNext, LlmNonHttpFailureKindV2, LlmRequest, LlmRequestInterceptOutcome, LlmStream,
    LlmStreamContinuationV2, LlmStreamExecutionOutcomeV2, LlmStreamNext,
    NEMO_RELAY_NATIVE_ABI_VERSION, NEMO_RELAY_NATIVE_ABI_VERSION_TARGETED_LLM_CONTINUATIONS,
    NativePlugin, NemoRelayNativeAsyncCallbackState, NemoRelayNativeAsyncCompletion,
    NemoRelayNativeAsyncLlmResultCbV2, NemoRelayNativeAsyncLlmStreamForwardCbV2,
    NemoRelayNativeAsyncLlmStreamNextCbV2, NemoRelayNativeAsyncLlmStreamOpenCbV2,
    NemoRelayNativeAsyncMiddlewareCb, NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncNextResultCb, NemoRelayNativeAsyncNextStreamCb, NemoRelayNativeAsyncStream,
    NemoRelayNativeAsyncStreamMiddlewareCb, NemoRelayNativeAsyncTaskPollCbV2,
    NemoRelayNativeAsyncTaskV2, NemoRelayNativeEventSanitizeCb, NemoRelayNativeEventSubscriberCb,
    NemoRelayNativeFreeFn, NemoRelayNativeHostApiV1, NemoRelayNativeHostApiV3,
    NemoRelayNativeHostApiV4, NemoRelayNativeLlmCodecKind, NemoRelayNativeLlmConditionalCb,
    NemoRelayNativeLlmExecutionCb, NemoRelayNativeLlmRequestCodec,
    NemoRelayNativeLlmRequestInterceptCb, NemoRelayNativeLlmResponseCodec,
    NemoRelayNativeLlmSanitizeRequestCb, NemoRelayNativeLlmSanitizeRequestContext,
    NemoRelayNativeLlmSanitizeResponseCb, NemoRelayNativeLlmSanitizeResponseContext,
    NemoRelayNativeLlmStreamExecutionCb, NemoRelayNativeLlmStreamV1, NemoRelayNativeLlmStreamV2,
    NemoRelayNativePluginContext, NemoRelayNativePluginV1, NemoRelayNativeScopeHandle,
    NemoRelayNativeScopeStack, NemoRelayNativeScopeStackBinding, NemoRelayNativeScopeType,
    NemoRelayNativeString, NemoRelayNativeToolConditionalCb, NemoRelayNativeToolExecutionCb,
    NemoRelayNativeToolJsonCb, NemoRelayNativeWithScopeStackCb, NemoRelayStatus, PendingMarkSpec,
    PluginContext, PluginRuntime, ScopeType, ToolExecutionInterceptOutcome, ToolNext,
};
use serde::de::DeserializeOwned;
use serde_json::{Map, json};

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
    ];
    for (discriminant, kind) in middleware_kinds.into_iter().enumerate() {
        assert_eq!(kind as u32, discriminant as u32);
        assert_eq!(Kind::try_from(discriminant as u32), Ok(kind));
    }
    assert!(NemoRelayNativeAsyncMiddlewareKind::try_from(14).is_err());
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(1),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    assert!(NemoRelayNativeAsyncCallbackState::try_from(2).is_err());
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

struct RegisteredAsyncV2 {
    name: String,
    priority: i32,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredAsyncV2 {
    unsafe fn free(self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.user_data as *mut c_void) };
        }
    }
}

struct RegisteredAsyncStreamV2 {
    name: String,
    priority: i32,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

impl RegisteredAsyncStreamV2 {
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
    RegisteredAsyncV2,
    RegisteredAsyncStreamV2,
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
static ASYNC_V2_REGISTRATION: Mutex<Option<RegisteredAsyncV2>> = Mutex::new(None);
static ASYNC_STREAM_V2_REGISTRATION: Mutex<Option<RegisteredAsyncStreamV2>> = Mutex::new(None);
static SAFE_V2_COMPLETION: Mutex<Option<std::result::Result<Json, String>>> = Mutex::new(None);
static SAFE_V2_COMPLETION_CANCELLED: AtomicBool = AtomicBool::new(false);
static SAFE_V2_OUTPUT: Mutex<Vec<std::result::Result<Json, String>>> = Mutex::new(Vec::new());
#[derive(Clone)]
enum SafeV2ProviderEvent {
    Chunk(Json),
    Done,
    Failure(LlmContinuationFailureV2),
}

static SAFE_V2_PROVIDER_EVENTS: Mutex<VecDeque<SafeV2ProviderEvent>> = Mutex::new(VecDeque::new());
static SAFE_V2_OPEN_FAILURE: Mutex<Option<LlmContinuationFailureV2>> = Mutex::new(None);
static SAFE_V2_OPEN_RETURNS_STREAM_AND_ERROR: AtomicBool = AtomicBool::new(false);
static SAFE_V2_HOLD_STREAM_OPEN_CALLBACK: AtomicBool = AtomicBool::new(false);
static SAFE_V2_HELD_STREAM_OPEN_CALLBACK: Mutex<
    Option<(NemoRelayNativeAsyncLlmStreamOpenCbV2, usize)>,
> = Mutex::new(None);
static SAFE_V2_FORWARDED_REQUESTS: Mutex<Vec<LlmRequest>> = Mutex::new(Vec::new());
static SAFE_V2_HOLD_TARGETED_CALLBACK: AtomicBool = AtomicBool::new(false);
static SAFE_V2_HELD_TARGETED_CALLBACK: Mutex<Option<(NemoRelayNativeAsyncLlmResultCbV2, usize)>> =
    Mutex::new(None);
static SAFE_V2_NEXT_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_COMPLETION_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_OUTPUT_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_PROVIDER_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_OUTPUT_FINISHES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_OUTPUT_CANCELLED: AtomicBool = AtomicBool::new(false);
static SAFE_V2_REGISTRATION_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_TARGETED_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_TARGETED_FAILURE: Mutex<Option<LlmContinuationFailureV2>> = Mutex::new(None);
static SAFE_V2_TARGETED_INVALID_OUTCOME: AtomicBool = AtomicBool::new(false);
static SAFE_V2_PASSTHROUGH_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_PASSTHROUGH_ERROR: Mutex<Option<String>> = Mutex::new(None);
static SAFE_V2_STREAM_OPEN_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_PROVIDER_NEXT_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_PROVIDER_EVENT_JSON: Mutex<Option<String>> = Mutex::new(None);
static SAFE_V2_PROVIDER_INVALID_EVENT: AtomicBool = AtomicBool::new(false);
static SAFE_V2_FORWARD_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_COMPLETION_RESOLVE_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_COMPLETION_REJECT_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_OUTPUT_PUSH_STATUSES: Mutex<VecDeque<NemoRelayStatus>> = Mutex::new(VecDeque::new());
static SAFE_V2_OUTPUT_FINISH_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_OUTPUT_REJECT_STATUSES: Mutex<VecDeque<NemoRelayStatus>> =
    Mutex::new(VecDeque::new());
static SAFE_V2_TASKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static SAFE_V2_TASK_SPAWN_STATUS: Mutex<NemoRelayStatus> = Mutex::new(NemoRelayStatus::Ok);
static SAFE_V2_TASK_RETAINS: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_TASK_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAFE_V2_HELD_TASK_WAKER: Mutex<Option<std::task::Waker>> = Mutex::new(None);

thread_local! {
    static SAFE_V2_CURRENT_TASK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct SafeV2Task {
    refs: AtomicUsize,
    woken: AtomicBool,
    completed: AtomicBool,
    cb: NemoRelayNativeAsyncTaskPollCbV2,
    user_data: usize,
    free_fn: NemoRelayNativeFreeFn,
}

#[test]
fn native_abi_struct_sizes_are_self_describing() {
    assert_eq!(NEMO_RELAY_NATIVE_ABI_VERSION, 3);
    assert_eq!(NEMO_RELAY_NATIVE_ABI_VERSION_TARGETED_LLM_CONTINUATIONS, 4);
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
    assert_eq!(NemoRelayStatus::WouldBlock as i32, 11);

    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(align_of::<NemoRelayNativeHostApiV1>(), 8);
        assert_eq!(size_of::<NemoRelayNativeHostApiV1>(), 320);
        assert_eq!(
            host_api_offsets(),
            [
                0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
                152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272,
                280, 288, 296, 304, 312,
            ]
        );
        assert_eq!(align_of::<NemoRelayNativeHostApiV3>(), 8);
        assert_eq!(size_of::<NemoRelayNativeHostApiV3>(), 440);
        assert_eq!(
            host_api_v3_offsets(),
            [
                0, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432
            ]
        );
        assert_eq!(align_of::<NemoRelayNativeHostApiV4>(), 8);
        assert_eq!(size_of::<NemoRelayNativeHostApiV4>(), 520);
        assert_eq!(
            host_api_v4_offsets(),
            [0, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512]
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
        assert_eq!(size_of::<NemoRelayNativeHostApiV1>(), 160);
        assert_eq!(
            host_api_offsets(),
            [
                0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
                84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148,
                152, 156,
            ]
        );
        assert_eq!(align_of::<NemoRelayNativeHostApiV3>(), 4);
        assert_eq!(size_of::<NemoRelayNativeHostApiV3>(), 216);
        assert_eq!(
            host_api_v3_offsets(),
            [
                0, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212
            ]
        );
        assert_eq!(align_of::<NemoRelayNativeHostApiV4>(), 4);
        assert_eq!(size_of::<NemoRelayNativeHostApiV4>(), 256);
        assert_eq!(
            host_api_v4_offsets(),
            [0, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252]
        );
        assert_eq!(align_of::<NemoRelayNativePluginV1>(), 4);
        assert_eq!(size_of::<NemoRelayNativePluginV1>(), 28);
        assert_eq!(plugin_offsets(), [0, 4, 8, 12, 16, 20, 24]);
        assert_eq!(align_of::<NemoRelayNativeLlmStreamV1>(), 4);
        assert_eq!(size_of::<NemoRelayNativeLlmStreamV1>(), 20);
        assert_eq!(stream_offsets(), [0, 4, 8, 12, 16]);
    }
}

fn host_api_v4_offsets() -> [usize; 11] {
    [
        offset_of!(NemoRelayNativeHostApiV4, v3),
        offset_of!(NemoRelayNativeHostApiV4, async_llm_next_invoke_result_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_llm_next_open_stream_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_llm_stream_next_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_llm_stream_release_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_completion_spawn_task_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_stream_spawn_task_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_task_retain_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_task_wake_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_task_release_v2),
        offset_of!(NemoRelayNativeHostApiV4, async_llm_next_forward_stream_v2),
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
    clear_registration(&ASYNC_V2_REGISTRATION);
    clear_registration(&ASYNC_STREAM_V2_REGISTRATION);
    assert!(
        SAFE_V2_TASKS.lock().unwrap().is_empty(),
        "previous test leaked cooperative host tasks"
    );
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
    *SAFE_V2_COMPLETION.lock().unwrap() = None;
    SAFE_V2_COMPLETION_CANCELLED.store(false, Ordering::SeqCst);
    SAFE_V2_OUTPUT.lock().unwrap().clear();
    SAFE_V2_PROVIDER_EVENTS.lock().unwrap().clear();
    *SAFE_V2_OPEN_FAILURE.lock().unwrap() = None;
    SAFE_V2_OPEN_RETURNS_STREAM_AND_ERROR.store(false, Ordering::SeqCst);
    SAFE_V2_HOLD_STREAM_OPEN_CALLBACK.store(false, Ordering::SeqCst);
    assert!(SAFE_V2_HELD_STREAM_OPEN_CALLBACK.lock().unwrap().is_none());
    SAFE_V2_FORWARDED_REQUESTS.lock().unwrap().clear();
    SAFE_V2_HOLD_TARGETED_CALLBACK.store(false, Ordering::SeqCst);
    assert!(SAFE_V2_HELD_TARGETED_CALLBACK.lock().unwrap().is_none());
    SAFE_V2_NEXT_RELEASES.store(0, Ordering::SeqCst);
    SAFE_V2_COMPLETION_RELEASES.store(0, Ordering::SeqCst);
    SAFE_V2_OUTPUT_RELEASES.store(0, Ordering::SeqCst);
    SAFE_V2_PROVIDER_RELEASES.store(0, Ordering::SeqCst);
    SAFE_V2_OUTPUT_FINISHES.store(0, Ordering::SeqCst);
    SAFE_V2_OUTPUT_CANCELLED.store(false, Ordering::SeqCst);
    *SAFE_V2_REGISTRATION_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_TARGETED_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_TARGETED_FAILURE.lock().unwrap() = None;
    SAFE_V2_TARGETED_INVALID_OUTCOME.store(false, Ordering::SeqCst);
    *SAFE_V2_PASSTHROUGH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_PASSTHROUGH_ERROR.lock().unwrap() = None;
    *SAFE_V2_STREAM_OPEN_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_PROVIDER_NEXT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_PROVIDER_EVENT_JSON.lock().unwrap() = None;
    SAFE_V2_PROVIDER_INVALID_EVENT.store(false, Ordering::SeqCst);
    *SAFE_V2_FORWARD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_COMPLETION_RESOLVE_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    *SAFE_V2_COMPLETION_REJECT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT_PUSH_STATUSES.lock().unwrap().clear();
    *SAFE_V2_OUTPUT_FINISH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT_REJECT_STATUSES.lock().unwrap().clear();
    *SAFE_V2_TASK_SPAWN_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_TASK_RETAINS.store(0, Ordering::SeqCst);
    SAFE_V2_TASK_RELEASES.store(0, Ordering::SeqCst);
    assert!(SAFE_V2_HELD_TASK_WAKER.lock().unwrap().is_none());
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

fn required_host_json<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> std::result::Result<T, NemoRelayStatus> {
    let value = required_host_string(host, value)?;
    serde_json::from_str(&value).map_err(|_| NemoRelayStatus::InvalidJson)
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
    assert_scope_runtime_calls(&calls);
    assert_stack_runtime_calls(&calls);
    assert_eq!(SCOPE_HANDLE_FREES.load(Ordering::SeqCst), 2);
    assert_eq!(SCOPE_STACK_FREES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_RESTORES.load(Ordering::SeqCst), 1);
    assert_eq!(SCOPE_STACK_BINDING_FREES.load(Ordering::SeqCst), 0);
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
fn typed_event_and_tool_callbacks_reject_null_abi_pointers_before_decoding_inputs() {
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
}

#[test]
fn typed_llm_callbacks_reject_null_abi_pointers_before_decoding_inputs() {
    let _guard = begin_test();
    let host = test_host();
    let mut out = ptr::null_mut();
    let mut reason = ptr::null_mut();

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail("llm-request", 0, |request, _context| {
        Some(request)
    })
    .unwrap();
    let registration = take_llm_request_registration();
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    assert_eq!(
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                request,
                native_no_codec_context(),
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                request,
                native_no_codec_context(),
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
        unsafe {
            (registration.cb)(
                ptr::null_mut(),
                response,
                native_no_response_codec_context(),
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            (registration.cb)(
                registration.user_data as *mut c_void,
                response,
                native_no_response_codec_context(),
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
fn typed_subscriber_event_and_tool_sanitize_callbacks_report_invalid_json() {
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
}

#[test]
fn typed_conditional_execution_and_llm_callbacks_report_invalid_json() {
    let _guard = begin_test();
    let host = test_host();

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
    let context = native_no_codec_context();
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
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail("llm-response", 0, |value, _context| Some(value))
        .unwrap();
    let registration = take_llm_json_registration();
    let response = host_string(&host, "{not json");
    let context = native_no_response_codec_context();
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
    let context = native_no_codec_context();
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
    let context = native_no_response_codec_context();
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
        registration.free();
    }
}

#[test]
fn typed_contextual_llm_sanitize_guardrails_receive_payload_before_context() {
    let _guard = begin_test();
    let mut host = test_host();
    host.llm_request_codec_decode = successful_request_codec_decode;
    host.llm_request_codec_encode = successful_request_codec_encode;
    host.llm_response_codec_decode = successful_response_codec_decode;
    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail(
        "contextual-request",
        14,
        |mut request, callback_context| {
            assert_eq!(
                callback_context.codec,
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
            );
            let codec = callback_context
                .resolve_codec()
                .expect("active request codec must resolve");
            let annotated = codec.decode(&request).expect("request decode succeeds");
            request = codec
                .encode(&annotated, &request)
                .expect("request encode succeeds");
            request.headers.insert("x-contextual".into(), json!(true));
            Some(request)
        },
    )
    .unwrap();

    let registration = take_llm_request_registration();
    assert_eq!(registration.name, "contextual-request");
    assert_eq!(registration.priority, 14);
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let context_id = host_string(&host, "openai_chat");
    let request_codec_placeholder = Box::new(0_usize);
    let native_context = NemoRelayNativeLlmSanitizeRequestContext {
        codec_kind: NemoRelayNativeLlmCodecKind::BuiltIn,
        codec_id: context_id,
        codec: std::ptr::from_ref(request_codec_placeholder.as_ref())
            .cast::<NemoRelayNativeLlmRequestCodec>(),
    };
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            request,
            native_context,
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
        (host.string_free)(context_id);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail(
        "contextual-response",
        15,
        |mut payload, callback_context| {
            assert_eq!(
                callback_context.codec,
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
            );
            callback_context
                .resolve_codec()
                .expect("active response codec must resolve")
                .decode(&payload)
                .expect("response decode succeeds");
            payload["contextual"] = json!(true);
            Some(payload)
        },
    )
    .unwrap();

    let registration = take_llm_json_registration();
    assert_eq!(registration.name, "contextual-response");
    assert_eq!(registration.priority, 15);
    let response = json_host_string(&host, json!({ "output": true }));
    let context_id = host_string(&host, "openai_chat");
    let response_codec_placeholder = Box::new(0_usize);
    let native_context = NemoRelayNativeLlmSanitizeResponseContext {
        codec_kind: NemoRelayNativeLlmCodecKind::BuiltIn,
        codec_id: context_id,
        codec: std::ptr::from_ref(response_codec_placeholder.as_ref())
            .cast::<NemoRelayNativeLlmResponseCodec>(),
    };
    let mut out = ptr::null_mut();
    let status = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            response,
            native_context,
            &mut out,
        )
    };
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(read_json_and_free(&host, out)["contextual"], json!(true));
    unsafe {
        (host.string_free)(response);
        (host.string_free)(context_id);
        registration.free();
    }
}

#[test]
fn typed_contextual_llm_sanitizer_uses_null_output_to_omit_payload() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_request_guardrail(
        "contextual-omit-request",
        16,
        |_request, _context| None,
    )
    .unwrap();

    let registration = take_llm_request_registration();
    let request = json_host_string(&host, serde_json::to_value(test_llm_request()).unwrap());
    let context = native_no_codec_context();
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
    assert!(out.is_null(), "null native output must represent omission");
    unsafe {
        (host.string_free)(request);
        registration.free();
    }

    let mut ctx = test_context(&host);
    ctx.register_llm_sanitize_response_guardrail("contextual-omit", 16, |_payload, _context| None)
        .unwrap();

    let registration = take_llm_json_registration();
    let response = json_host_string(&host, json!({"secret": "value"}));
    let context = NemoRelayNativeLlmSanitizeResponseContext {
        codec_kind: NemoRelayNativeLlmCodecKind::None,
        codec_id: ptr::null(),
        codec: ptr::null(),
    };
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
            let stream: LlmJsonStream = Box::new(next.call(request)?);
            Ok(wrap_stream_chunks(stream))
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

    assert_wrapped_stream_chunks(&host, &stream);
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

fn wrap_stream_chunks(stream: LlmJsonStream) -> LlmJsonStream {
    Box::new(stream.map(|chunk| {
        chunk.map(|mut chunk| {
            chunk["wrapped"] = json!(true);
            chunk
        })
    }))
}

fn assert_wrapped_stream_chunks(
    host: &NemoRelayNativeHostApiV1,
    stream: &NemoRelayNativeLlmStreamV1,
) {
    let (status, chunk) = poll_stream_chunk(host, stream);
    assert_eq!(status, NemoRelayStatus::Ok);
    assert_eq!(chunk.unwrap()["wrapped"], json!(true));

    let (status, chunk) = poll_stream_chunk(host, stream);
    assert_eq!(status, NemoRelayStatus::Ok);
    let chunk = chunk.unwrap();
    assert_eq!(chunk["chunk"], json!(2));
    assert_eq!(chunk["wrapped"], json!(true));

    for _ in 0..2 {
        let (status, chunk) = poll_stream_chunk(host, stream);
        assert_eq!(status, NemoRelayStatus::StreamEnd);
        assert!(chunk.is_none());
    }
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

unsafe extern "C" fn safe_v2_completion_resolve(
    _completion: *const NemoRelayNativeAsyncCompletion,
    value_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let value = match required_host_json(&host, value_json) {
        Ok(value) => value,
        Err(status) => return status,
    };
    *SAFE_V2_COMPLETION.lock().unwrap() = Some(Ok(value));
    *SAFE_V2_COMPLETION_RESOLVE_STATUS.lock().unwrap()
}

unsafe extern "C" fn safe_v2_completion_reject(
    _completion: *const NemoRelayNativeAsyncCompletion,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let message = required_host_string(&host, message)
        .unwrap_or_else(|status| format!("invalid rejection: {status:?}"));
    *SAFE_V2_COMPLETION.lock().unwrap() = Some(Err(message));
    *SAFE_V2_COMPLETION_REJECT_STATUS.lock().unwrap()
}

unsafe extern "C" fn safe_v2_completion_is_cancelled(
    _completion: *const NemoRelayNativeAsyncCompletion,
) -> bool {
    SAFE_V2_COMPLETION_CANCELLED.load(Ordering::SeqCst)
}

unsafe extern "C" fn safe_v2_completion_release(
    _completion: *const NemoRelayNativeAsyncCompletion,
) {
    SAFE_V2_COMPLETION_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn safe_v2_async_next_invoke(
    _next: *const NemoRelayNativeAsyncNext,
    _invocation_json: *const NemoRelayNativeString,
    _completion: *const NemoRelayNativeAsyncCompletion,
) -> NemoRelayStatus {
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn safe_v2_next_release(_next: *const NemoRelayNativeAsyncNext) {
    SAFE_V2_NEXT_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe fn safe_v2_reject_registration(
    status: NemoRelayStatus,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    if let Some(free_fn) = free_fn {
        unsafe { free_fn(user_data) };
    }
    status
}

fn safe_v2_registration_name(
    name: *const NemoRelayNativeString,
) -> std::result::Result<String, NemoRelayStatus> {
    let status = *SAFE_V2_REGISTRATION_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return Err(status);
    }
    required_host_string(&test_host(), name)
}

unsafe extern "C" fn safe_v2_register_generic_async(
    _ctx: *mut NemoRelayNativePluginContext,
    kind: u32,
    name: *const NemoRelayNativeString,
    priority: i32,
    _break_chain: bool,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    if kind != NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept as u32 {
        return unsafe {
            safe_v2_reject_registration(NemoRelayStatus::InvalidArg, user_data, free_fn)
        };
    }
    let name = match safe_v2_registration_name(name) {
        Ok(name) => name,
        Err(status) => return unsafe { safe_v2_reject_registration(status, user_data, free_fn) },
    };
    replace_registration(
        &ASYNC_V2_REGISTRATION,
        RegisteredAsyncV2 {
            name,
            priority,
            cb,
            user_data: user_data as usize,
            free_fn,
        },
    );
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_stream_push(
    _stream: *const NemoRelayNativeAsyncStream,
    chunk_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let chunk = match required_host_json(&host, chunk_json) {
        Ok(chunk) => chunk,
        Err(status) => return status,
    };
    let status = safe_v2_output_status(&SAFE_V2_OUTPUT_PUSH_STATUSES);
    if status == NemoRelayStatus::Ok {
        SAFE_V2_OUTPUT.lock().unwrap().push(Ok(chunk));
    }
    status
}

fn safe_v2_output_status(statuses: &Mutex<VecDeque<NemoRelayStatus>>) -> NemoRelayStatus {
    let status = statuses
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or(NemoRelayStatus::Ok);
    if status == NemoRelayStatus::WouldBlock {
        SAFE_V2_CURRENT_TASK.with(|current| {
            let task = current.get();
            if task != 0 {
                let _ = unsafe { safe_v2_task_wake(task as *const NemoRelayNativeAsyncTaskV2) };
            }
        });
    }
    status
}

unsafe extern "C" fn safe_v2_stream_finish(
    _stream: *const NemoRelayNativeAsyncStream,
) -> NemoRelayStatus {
    SAFE_V2_OUTPUT_FINISHES.fetch_add(1, Ordering::SeqCst);
    *SAFE_V2_OUTPUT_FINISH_STATUS.lock().unwrap()
}

unsafe extern "C" fn safe_v2_stream_reject(
    _stream: *const NemoRelayNativeAsyncStream,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let host = test_host();
    let message = required_host_string(&host, message)
        .unwrap_or_else(|status| format!("invalid rejection: {status:?}"));
    let status = safe_v2_output_status(&SAFE_V2_OUTPUT_REJECT_STATUSES);
    if status == NemoRelayStatus::Ok {
        SAFE_V2_OUTPUT.lock().unwrap().push(Err(message));
    }
    status
}

unsafe extern "C" fn safe_v2_stream_is_cancelled(
    _stream: *const NemoRelayNativeAsyncStream,
) -> bool {
    SAFE_V2_OUTPUT_CANCELLED.load(Ordering::SeqCst)
}

unsafe extern "C" fn safe_v2_stream_release(_stream: *const NemoRelayNativeAsyncStream) {
    SAFE_V2_OUTPUT_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn safe_v2_async_next_invoke_stream(
    _next: *const NemoRelayNativeAsyncNext,
    _invocation_json: *const NemoRelayNativeString,
    _stream: *const NemoRelayNativeAsyncStream,
    _cb: NemoRelayNativeAsyncNextStreamCb,
    _user_data: *mut c_void,
) -> NemoRelayStatus {
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn safe_v2_register_generic_stream(
    _ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let name = match safe_v2_registration_name(name) {
        Ok(name) => name,
        Err(status) => return unsafe { safe_v2_reject_registration(status, user_data, free_fn) },
    };
    replace_registration(
        &ASYNC_STREAM_V2_REGISTRATION,
        RegisteredAsyncStreamV2 {
            name,
            priority,
            cb,
            user_data: user_data as usize,
            free_fn,
        },
    );
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_passthrough_result(
    _next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncNextResultCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let status = *SAFE_V2_PASSTHROUGH_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    if let Err(status) = required_host_json::<LlmRequest>(&host, invocation_json) {
        return status;
    }
    if let Some(error) = SAFE_V2_PASSTHROUGH_ERROR.lock().unwrap().clone() {
        let error = host_string(&host, &error);
        unsafe { cb(user_data, ptr::null(), error) };
        unsafe { (host.string_free)(error) };
    } else {
        let value = json_host_string(&host, json!({ "passthrough": true }));
        unsafe { cb(user_data, value, ptr::null()) };
        unsafe { (host.string_free)(value) };
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_targeted_result(
    _next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncLlmResultCbV2,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let status = *SAFE_V2_TARGETED_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    if let Err(status) = required_host_json::<LlmContinuationInvocationV2>(&host, invocation_json) {
        return status;
    }
    if SAFE_V2_HOLD_TARGETED_CALLBACK.load(Ordering::SeqCst) {
        *SAFE_V2_HELD_TARGETED_CALLBACK.lock().unwrap() = Some((cb, user_data as usize));
        return NemoRelayStatus::Ok;
    }
    if SAFE_V2_TARGETED_INVALID_OUTCOME.swap(false, Ordering::SeqCst) {
        unsafe { cb(user_data, ptr::null(), ptr::null()) };
        return NemoRelayStatus::Ok;
    }
    if let Some(error) = SAFE_V2_TARGETED_FAILURE.lock().unwrap().take() {
        let error = json_host_string(&host, serde_json::to_value(error).unwrap());
        unsafe { cb(user_data, ptr::null(), error) };
        unsafe { (host.string_free)(error) };
        return NemoRelayStatus::Ok;
    }
    let response = json_host_string(&host, json!({ "targeted": true }));
    unsafe { cb(user_data, response, ptr::null()) };
    unsafe { (host.string_free)(response) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_stream_open(
    _next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    _output_stream: *const NemoRelayNativeAsyncStream,
    cb: NemoRelayNativeAsyncLlmStreamOpenCbV2,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let status = *SAFE_V2_STREAM_OPEN_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    if let Err(status) = required_host_json::<LlmContinuationInvocationV2>(&host, invocation_json) {
        return status;
    }
    if SAFE_V2_HOLD_STREAM_OPEN_CALLBACK.load(Ordering::SeqCst) {
        *SAFE_V2_HELD_STREAM_OPEN_CALLBACK.lock().unwrap() = Some((cb, user_data as usize));
        return NemoRelayStatus::Ok;
    }
    if let Some(error) = SAFE_V2_OPEN_FAILURE.lock().unwrap().clone() {
        let error = json_host_string(&host, serde_json::to_value(error).unwrap());
        let stream = if SAFE_V2_OPEN_RETURNS_STREAM_AND_ERROR.load(Ordering::SeqCst) {
            NonNull::<NemoRelayNativeLlmStreamV2>::dangling().as_ptr()
        } else {
            ptr::null()
        };
        unsafe { cb(user_data, stream, error) };
        unsafe { (host.string_free)(error) };
        return NemoRelayStatus::Ok;
    }
    unsafe {
        cb(
            user_data,
            NonNull::<NemoRelayNativeLlmStreamV2>::dangling().as_ptr(),
            ptr::null(),
        )
    };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_provider_next(
    _stream: *const NemoRelayNativeLlmStreamV2,
    cb: NemoRelayNativeAsyncLlmStreamNextCbV2,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let status = *SAFE_V2_PROVIDER_NEXT_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    if let Some(event) = SAFE_V2_PROVIDER_EVENT_JSON.lock().unwrap().take() {
        let chunk = host_string(&host, &event);
        unsafe { cb(user_data, chunk, ptr::null(), false) };
        unsafe { (host.string_free)(chunk) };
        return NemoRelayStatus::Ok;
    }
    if SAFE_V2_PROVIDER_INVALID_EVENT.swap(false, Ordering::SeqCst) {
        unsafe { cb(user_data, ptr::null(), ptr::null(), false) };
        return NemoRelayStatus::Ok;
    }
    let event = SAFE_V2_PROVIDER_EVENTS
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or(SafeV2ProviderEvent::Done);
    match event {
        SafeV2ProviderEvent::Chunk(chunk) => {
            let chunk = json_host_string(&host, chunk);
            unsafe { cb(user_data, chunk, ptr::null(), false) };
            unsafe { (host.string_free)(chunk) };
        }
        SafeV2ProviderEvent::Done => unsafe { cb(user_data, ptr::null(), ptr::null(), true) },
        SafeV2ProviderEvent::Failure(error) => {
            let error = json_host_string(&host, serde_json::to_value(error).unwrap());
            unsafe { cb(user_data, ptr::null(), error, true) };
            unsafe { (host.string_free)(error) };
        }
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_provider_release(_stream: *const NemoRelayNativeLlmStreamV2) {
    SAFE_V2_PROVIDER_RELEASES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn safe_v2_spawn_task(
    _owner: *const c_void,
    cb: NemoRelayNativeAsyncTaskPollCbV2,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    let status = *SAFE_V2_TASK_SPAWN_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let task = Box::new(SafeV2Task {
        refs: AtomicUsize::new(1),
        woken: AtomicBool::new(true),
        completed: AtomicBool::new(false),
        cb,
        user_data: user_data as usize,
        free_fn,
    });
    let task = Box::into_raw(task) as usize;
    SAFE_V2_TASKS.lock().unwrap().push(task);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_completion_spawn_task(
    completion: *const NemoRelayNativeAsyncCompletion,
    cb: NemoRelayNativeAsyncTaskPollCbV2,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    unsafe { safe_v2_spawn_task(completion.cast(), cb, user_data, free_fn) }
}

unsafe extern "C" fn safe_v2_stream_spawn_task(
    stream: *const NemoRelayNativeAsyncStream,
    cb: NemoRelayNativeAsyncTaskPollCbV2,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    unsafe { safe_v2_spawn_task(stream.cast(), cb, user_data, free_fn) }
}

unsafe extern "C" fn safe_v2_task_retain(task: *const NemoRelayNativeAsyncTaskV2) {
    if let Some(task) = unsafe { task.cast::<SafeV2Task>().as_ref() } {
        task.refs.fetch_add(1, Ordering::Relaxed);
        SAFE_V2_TASK_RETAINS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn safe_v2_task_wake(task: *const NemoRelayNativeAsyncTaskV2) -> NemoRelayStatus {
    let Some(task) = (unsafe { task.cast::<SafeV2Task>().as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if task.completed.load(Ordering::Acquire) {
        return NemoRelayStatus::Ok;
    }
    task.woken.store(true, Ordering::Release);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn safe_v2_task_release(task: *const NemoRelayNativeAsyncTaskV2) {
    let Some(task_ref) = (unsafe { task.cast::<SafeV2Task>().as_ref() }) else {
        return;
    };
    SAFE_V2_TASK_RELEASES.fetch_add(1, Ordering::SeqCst);
    if task_ref.refs.fetch_sub(1, Ordering::AcqRel) == 1 {
        unsafe { drop(Box::from_raw(task.cast_mut().cast::<SafeV2Task>())) };
    }
}

fn wake_safe_v2_tasks() {
    for task in SAFE_V2_TASKS.lock().unwrap().iter().copied() {
        let _ = unsafe { safe_v2_task_wake(task as *const NemoRelayNativeAsyncTaskV2) };
    }
}

fn drive_safe_v2_tasks() {
    loop {
        let tasks = SAFE_V2_TASKS.lock().unwrap().clone();
        let mut made_progress = false;
        for raw in tasks {
            let task_ptr = raw as *const SafeV2Task;
            let Some(task) = (unsafe { task_ptr.as_ref() }) else {
                continue;
            };
            if !task.woken.swap(false, Ordering::AcqRel) {
                continue;
            }
            made_progress = true;
            unsafe {
                safe_v2_task_retain(task_ptr.cast());
            }
            SAFE_V2_CURRENT_TASK.with(|current| current.set(raw));
            let state = unsafe {
                (task.cb)(
                    task.user_data as *mut c_void,
                    task_ptr.cast::<NemoRelayNativeAsyncTaskV2>(),
                )
            };
            SAFE_V2_CURRENT_TASK.with(|current| current.set(0));
            if NemoRelayNativeAsyncCallbackState::try_from(state)
                == Ok(NemoRelayNativeAsyncCallbackState::Complete)
            {
                task.completed.store(true, Ordering::Release);
                if let Some(free_fn) = task.free_fn {
                    unsafe { free_fn(task.user_data as *mut c_void) };
                }
                SAFE_V2_TASKS.lock().unwrap().retain(|task| *task != raw);
                unsafe { safe_v2_task_release(task_ptr.cast()) };
            }
            unsafe { safe_v2_task_release(task_ptr.cast()) };
        }
        if !made_progress {
            return;
        }
    }
}

unsafe extern "C" fn safe_v2_forward_stream(
    _next: *const NemoRelayNativeAsyncNext,
    request_json: *const NemoRelayNativeString,
    _output_stream: *const NemoRelayNativeAsyncStream,
    cb: NemoRelayNativeAsyncLlmStreamForwardCbV2,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let status = *SAFE_V2_FORWARD_STATUS.lock().unwrap();
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let host = test_host();
    let request = match required_host_json(&host, request_json) {
        Ok(request) => request,
        Err(status) => return status,
    };
    SAFE_V2_FORWARDED_REQUESTS.lock().unwrap().push(request);
    SAFE_V2_OUTPUT_FINISHES.fetch_add(1, Ordering::SeqCst);
    unsafe { cb(user_data) };
    NemoRelayStatus::Ok
}

fn test_host_v4() -> NemoRelayNativeHostApiV4 {
    let mut v1 = test_host();
    v1.abi_version = NEMO_RELAY_NATIVE_ABI_VERSION_TARGETED_LLM_CONTINUATIONS;
    v1.struct_size = size_of::<NemoRelayNativeHostApiV4>();
    NemoRelayNativeHostApiV4 {
        v3: NemoRelayNativeHostApiV3 {
            v1,
            async_completion_resolve_json: safe_v2_completion_resolve,
            async_completion_reject: safe_v2_completion_reject,
            async_completion_is_cancelled: safe_v2_completion_is_cancelled,
            async_completion_release: safe_v2_completion_release,
            async_next_invoke: safe_v2_async_next_invoke,
            async_next_release: safe_v2_next_release,
            plugin_context_register_async_middleware: safe_v2_register_generic_async,
            async_stream_push_json: safe_v2_stream_push,
            async_stream_finish: safe_v2_stream_finish,
            async_stream_reject: safe_v2_stream_reject,
            async_stream_is_cancelled: safe_v2_stream_is_cancelled,
            async_stream_release: safe_v2_stream_release,
            async_next_invoke_stream: safe_v2_async_next_invoke_stream,
            plugin_context_register_async_stream_middleware: safe_v2_register_generic_stream,
            async_next_invoke_result: safe_v2_passthrough_result,
        },
        async_llm_next_invoke_result_v2: safe_v2_targeted_result,
        async_llm_next_open_stream_v2: safe_v2_stream_open,
        async_llm_stream_next_v2: safe_v2_provider_next,
        async_llm_stream_release_v2: safe_v2_provider_release,
        async_completion_spawn_task_v2: safe_v2_completion_spawn_task,
        async_stream_spawn_task_v2: safe_v2_stream_spawn_task,
        async_task_retain_v2: safe_v2_task_retain,
        async_task_wake_v2: safe_v2_task_wake,
        async_task_release_v2: safe_v2_task_release,
        async_llm_next_forward_stream_v2: safe_v2_forward_stream,
    }
}

fn safe_v2_target() -> LlmContinuationTargetV2 {
    LlmContinuationTargetV2 {
        url: "https://provider.example/v1/chat/completions".into(),
        headers: Default::default(),
    }
}

#[test]
fn native_v2_debug_output_redacts_requests_targets_and_credentials() {
    let invocation = LlmContinuationInvocationV2 {
        request: LlmRequest {
            headers: Map::from_iter([("authorization".into(), json!("Bearer request-secret"))]),
            content: json!({"prompt": "request-body-secret"}),
        },
        target: LlmContinuationTargetV2 {
            url: "https://provider.example/v1?api_key=url-secret".into(),
            headers: BTreeMap::from([("authorization".into(), "Bearer target-secret".into())]),
        },
    };

    let debug = format!("{invocation:?}");
    assert!(debug.contains("LlmContinuationInvocationV2"));
    assert!(debug.contains("authorization"));
    for secret in [
        "request-secret",
        "request-body-secret",
        "target-secret",
        "url-secret",
    ] {
        assert!(!debug.contains(secret));
    }
}

#[test]
fn native_v2_retry_helper_uses_provider_neutral_status_semantics() {
    let http_failure = |status| LlmContinuationFailureV2::Http {
        status,
        body: String::new(),
        headers: BTreeMap::new(),
    };
    for status in [408, 425, 429, 500, 502, 503, 504] {
        assert!(LlmContinuationFailureV2::http_status_is_retryable(status));
        assert!(http_failure(status).is_retryable(), "status={status}");
    }
    for status in [400, 401, 404, 409, 422, 501] {
        assert!(!LlmContinuationFailureV2::http_status_is_retryable(status));
        assert!(!http_failure(status).is_retryable(), "status={status}");
    }

    let non_http_failure = |kind| LlmContinuationFailureV2::NonHttp {
        kind,
        message: String::new(),
    };
    assert!(LlmNonHttpFailureKindV2::Transport.is_retryable());
    assert!(non_http_failure(LlmNonHttpFailureKindV2::Transport).is_retryable());
    assert!(LlmNonHttpFailureKindV2::Timeout.is_retryable());
    assert!(non_http_failure(LlmNonHttpFailureKindV2::Timeout).is_retryable());
    for kind in [
        LlmNonHttpFailureKindV2::Cancelled,
        LlmNonHttpFailureKindV2::InvalidRequest,
        LlmNonHttpFailureKindV2::Guardrail,
        LlmNonHttpFailureKindV2::Internal,
    ] {
        assert!(!kind.is_retryable(), "kind={kind:?}");
        assert!(!non_http_failure(kind).is_retryable(), "kind={kind:?}");
    }
}

fn take_safe_v2_buffered_registration() -> RegisteredAsyncV2 {
    ASYNC_V2_REGISTRATION.lock().unwrap().take().unwrap()
}

fn take_safe_v2_stream_registration() -> RegisteredAsyncStreamV2 {
    ASYNC_STREAM_V2_REGISTRATION.lock().unwrap().take().unwrap()
}

fn invoke_safe_v2_buffered(
    host: &NemoRelayNativeHostApiV4,
    registration: &RegisteredAsyncV2,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next,
            completion,
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    state
}

fn invoke_safe_v2_streaming(
    host: &NemoRelayNativeHostApiV4,
    registration: &RegisteredAsyncStreamV2,
    next: *const NemoRelayNativeAsyncNext,
    output: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            next,
            output,
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    state
}

fn run_safe_v2_buffered<F, Fut>(host: &NemoRelayNativeHostApiV4, name: &str, callback: F) -> u32
where
    F: Fn(String, LlmRequest, LlmContinuationV2) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = std::result::Result<Json, String>> + Send + 'static,
{
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2(name, 0, callback)
        .unwrap();
    let registration = take_safe_v2_buffered_registration();
    let state = invoke_safe_v2_buffered(
        host,
        &registration,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
    );
    unsafe { registration.free() };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    state
}

fn run_safe_v2_streaming<F, Fut>(host: &NemoRelayNativeHostApiV4, name: &str, callback: F) -> u32
where
    F: Fn(String, LlmRequest, LlmStreamContinuationV2) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = std::result::Result<LlmStreamExecutionOutcomeV2, String>> + Send + 'static,
{
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(name, 0, callback)
        .unwrap();
    let registration = take_safe_v2_stream_registration();
    let state = invoke_safe_v2_streaming(
        host,
        &registration,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
    );
    unsafe { registration.free() };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    state
}

async fn safe_v2_targeted_provider_stream(
    request: LlmRequest,
    next: LlmStreamContinuationV2,
) -> std::result::Result<LlmStreamExecutionOutcomeV2, String> {
    let provider = next
        .open_stream(LlmContinuationInvocationV2 {
            request,
            target: safe_v2_target(),
        })
        .await
        .map_err(|error| format!("{error:?}"))?;
    Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
        provider.map(|item| item.map_err(|error| format!("{error:?}"))),
    )))
}

fn run_safe_v2_targeted_stream(host: &NemoRelayNativeHostApiV4, name: &str) {
    run_safe_v2_streaming(host, name, |_, request, next| {
        safe_v2_targeted_provider_stream(request, next)
    });
}

fn safe_v2_one_chunk_stream() -> LlmStreamExecutionOutcomeV2 {
    LlmStreamExecutionOutcomeV2::Stream(Box::pin(stream::once(async {
        Ok(json!({ "chunk": true }))
    })))
}

unsafe extern "C" fn raw_v2_buffered_probe(
    _user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    _completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

unsafe extern "C" fn raw_v2_streaming_probe(
    _user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    _output: *const NemoRelayNativeAsyncStream,
) -> u32 {
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

#[test]
fn native_api_v2_uses_generic_raw_registration_as_advanced_escape_hatch() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);

    assert_eq!(
        unsafe {
            ctx.register_async_middleware_raw(
                NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
                "raw-buffered",
                11,
                false,
                raw_v2_buffered_probe,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let buffered = take_safe_v2_buffered_registration();
    assert_eq!(
        (buffered.name.as_str(), buffered.priority),
        ("raw-buffered", 11)
    );
    unsafe { buffered.free() };

    assert_eq!(
        unsafe {
            ctx.register_async_stream_middleware_raw(
                "raw-streaming",
                12,
                raw_v2_streaming_probe,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );
    let streaming = take_safe_v2_stream_registration();
    assert_eq!(
        (streaming.name.as_str(), streaming.priority),
        ("raw-streaming", 12)
    );
    unsafe { streaming.free() };

    let v1 = test_host();
    let mut v1_ctx = test_context(&v1);
    assert_eq!(
        unsafe {
            v1_ctx.register_async_middleware_raw(
                NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
                "unsupported-buffered",
                0,
                false,
                raw_v2_buffered_probe,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe {
            v1_ctx.register_async_stream_middleware_raw(
                "unsupported-streaming",
                0,
                raw_v2_streaming_probe,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::InvalidArg
    );

    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_registration_wraps_targeted_and_passthrough_calls() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("safe-buffered", 7, |name, request, next| async move {
        assert_eq!(name, "managed-llm");
        let targeted = next
            .call(LlmContinuationInvocationV2 {
                request: request.clone(),
                target: safe_v2_target(),
            })
            .await
            .map_err(|error| format!("{error:?}"))?;
        let passthrough = next.call_passthrough(request).await?;
        Ok(json!({ "targeted": targeted, "passthrough": passthrough }))
    })
    .unwrap();
    let registration = take_safe_v2_buffered_registration();
    assert_eq!(registration.name, "safe-buffered");
    assert_eq!(registration.priority, 7);

    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed-llm", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    assert_eq!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Ok(json!({
            "targeted": { "targeted": true },
            "passthrough": { "passthrough": true }
        })))
    );
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_COMPLETION_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_continuation_preserves_flattened_failure_and_rejects_invalid_outcome() {
    let _guard = begin_test();
    let host = test_host_v4();
    let expected = LlmContinuationFailureV2::Http {
        status: 429,
        body: "bounded".into(),
        headers: Default::default(),
    };
    assert_eq!(
        serde_json::to_value(&expected).unwrap(),
        json!({
            "failure_type": "http",
            "status": 429,
            "body": "bounded",
            "headers": {},
        })
    );
    *SAFE_V2_TARGETED_FAILURE.lock().unwrap() = Some(expected.clone());
    run_safe_v2_buffered(&host, "typed-failure", move |_, request, next| {
        let expected = expected.clone();
        async move {
            let error = next
                .call(LlmContinuationInvocationV2 {
                    request,
                    target: safe_v2_target(),
                })
                .await
                .expect_err("the host returned a structured failure");
            assert_eq!(error, expected);
            Ok(json!({ "observed": "typed failure" }))
        }
    });
    assert_eq!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Ok(json!({ "observed": "typed failure" })))
    );

    SAFE_V2_TARGETED_INVALID_OUTCOME.store(true, Ordering::SeqCst);
    run_safe_v2_buffered(&host, "invalid-outcome", |_, request, next| async move {
        let error = next
            .call(LlmContinuationInvocationV2 {
                request,
                target: safe_v2_target(),
            })
            .await
            .expect_err("two null callback values are not a valid outcome");
        Ok(json!({ "error": format!("{error:?}") }))
    });
    let outcome = SAFE_V2_COMPLETION.lock().unwrap().take().unwrap().unwrap();
    assert!(
        outcome["error"]
            .as_str()
            .is_some_and(|error| error.contains("invalid outcome"))
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_continuation_supports_repeated_concurrent_calls() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2(
        "safe-buffered-concurrent",
        0,
        |_, request, next| async move {
            let calls = (0..8).map(|index| {
                let next = next.clone();
                let mut request = request.clone();
                request.content["index"] = json!(index);
                async move {
                    next.call(LlmContinuationInvocationV2 {
                        request,
                        target: safe_v2_target(),
                    })
                    .await
                    .map_err(|error| format!("{error:?}"))
                }
            });
            let results = futures::future::join_all(calls)
                .await
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(json!(results))
        },
    )
    .unwrap();
    let registration = take_safe_v2_buffered_registration();

    let state = invoke_safe_v2_buffered(
        &host,
        &registration,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
    );
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();

    let outcome = SAFE_V2_COMPLETION.lock().unwrap().take().unwrap().unwrap();
    let results = outcome
        .as_array()
        .expect("callback returns one result per call");
    assert_eq!(results.len(), 8);
    assert!(
        results
            .iter()
            .all(|result| result == &json!({ "targeted": true }))
    );
    assert_eq!(
        SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst),
        1,
        "all continuation clones share one retained host handle"
    );
    assert_eq!(SAFE_V2_COMPLETION_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_callback_stops_when_the_caller_cancels() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("safe-buffered-cancel", 0, |_, _, _| async move {
        std::future::pending::<Result<Json, String>>().await
    })
    .unwrap();
    let registration = take_safe_v2_buffered_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    SAFE_V2_COMPLETION_CANCELLED.store(true, Ordering::SeqCst);
    wake_safe_v2_tasks();
    drive_safe_v2_tasks();
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_COMPLETION_RELEASES.load(Ordering::SeqCst), 1);
    assert!(SAFE_V2_COMPLETION.lock().unwrap().is_none());
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_cancelled_buffered_task_releases_completion_when_future_drop_panics() {
    let _guard = begin_test();

    struct PendingFutureWithPanickingDrop;

    impl Future for PendingFutureWithPanickingDrop {
        type Output = std::result::Result<Json, String>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for PendingFutureWithPanickingDrop {
        fn drop(&mut self) {
            panic!("safe callback future drop panic");
        }
    }

    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("panic-on-cancel", 0, |_, _, _| {
        PendingFutureWithPanickingDrop
    })
    .unwrap();
    let registration = take_safe_v2_buffered_registration();
    let state = invoke_safe_v2_buffered(
        &host,
        &registration,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
    );
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();

    SAFE_V2_COMPLETION_CANCELLED.store(true, Ordering::SeqCst);
    wake_safe_v2_tasks();
    drive_safe_v2_tasks();

    assert!(SAFE_V2_TASKS.lock().unwrap().is_empty());
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_COMPLETION_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native API v2 host task state drop panicked")
    );
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_cancelled_targeted_call_releases_next_before_the_host_callback() {
    let _guard = begin_test();
    SAFE_V2_HOLD_TARGETED_CALLBACK.store(true, Ordering::SeqCst);
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("safe-target-cancel", 0, |_, request, next| async move {
        next.call(LlmContinuationInvocationV2 {
            request,
            target: safe_v2_target(),
        })
        .await
        .map_err(|error| format!("{error:?}"))
    })
    .unwrap();
    let registration = take_safe_v2_buffered_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    while SAFE_V2_HELD_TARGETED_CALLBACK.lock().unwrap().is_none() {
        std::thread::yield_now();
    }
    SAFE_V2_COMPLETION_CANCELLED.store(true, Ordering::SeqCst);
    wake_safe_v2_tasks();
    drive_safe_v2_tasks();
    assert_eq!(
        SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst),
        1,
        "callback state must not retain the continuation until the host replies"
    );
    let (targeted_callback, targeted_user_data) = SAFE_V2_HELD_TARGETED_CALLBACK
        .lock()
        .unwrap()
        .take()
        .unwrap();
    let error = json_host_string(
        &host.v3.v1,
        serde_json::to_value(LlmContinuationFailureV2::NonHttp {
            kind: LlmNonHttpFailureKindV2::Cancelled,
            message: "cancelled by host".into(),
        })
        .unwrap(),
    );
    unsafe { targeted_callback(targeted_user_data as *mut c_void, ptr::null(), error) };
    unsafe {
        (host.v3.v1.string_free)(error);
        registration.free();
    }
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_registration_pumps_provider_stream_and_releases_handles() {
    let _guard = begin_test();
    SAFE_V2_PROVIDER_EVENTS.lock().unwrap().extend([
        SafeV2ProviderEvent::Chunk(json!({ "delta": "hello" })),
        SafeV2ProviderEvent::Done,
    ]);
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(
        "safe-stream",
        3,
        |_name, request, next| async move {
            let mut provider = next
                .open_stream(LlmContinuationInvocationV2 {
                    request,
                    target: safe_v2_target(),
                })
                .await
                .map_err(|error| format!("{error:?}"))?;
            let mut chunks = Vec::new();
            while let Some(item) = provider.next().await {
                chunks.push(item.map_err(|error| format!("{error:?}"))?);
            }
            assert!(
                provider.next().await.is_none(),
                "completed streams stay fused"
            );
            Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(stream::iter(
                chunks.into_iter().map(Ok),
            ))))
        },
    )
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    assert_eq!(registration.name, "safe-stream");
    assert_eq!(registration.priority, 3);

    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed-llm", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    assert_eq!(
        *SAFE_V2_OUTPUT.lock().unwrap(),
        vec![Ok(json!({ "delta": "hello" }))]
    );
    assert_eq!(SAFE_V2_OUTPUT_FINISHES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_PROVIDER_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_passthrough_uses_host_owned_forwarding() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(
        "safe-passthrough",
        0,
        |_name, request, _next| async move {
            Ok(LlmStreamExecutionOutcomeV2::Passthrough(request))
        },
    )
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let request = test_llm_request();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "unmanaged", "request": request.clone() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    assert_eq!(*SAFE_V2_FORWARDED_REQUESTS.lock().unwrap(), vec![request]);
    assert!(SAFE_V2_OUTPUT.lock().unwrap().is_empty());
    assert_eq!(SAFE_V2_OUTPUT_FINISHES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_provider_stream_drop_releases_unfinished_production() {
    let _guard = begin_test();
    SAFE_V2_PROVIDER_EVENTS
        .lock()
        .unwrap()
        .push_back(SafeV2ProviderEvent::Chunk(json!({ "unused": true })));
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2("safe-drop", 0, |_name, request, next| async move {
        let provider = next
            .open_stream(LlmContinuationInvocationV2 {
                request,
                target: safe_v2_target(),
            })
            .await
            .map_err(|error| format!("{error:?}"))?;
        drop(provider);
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
            stream::empty(),
        )))
    })
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    drive_safe_v2_tasks();
    assert_eq!(SAFE_V2_PROVIDER_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_open_result_releases_when_waiter_is_cancelled_before_consumption() {
    let _guard = begin_test();
    SAFE_V2_HOLD_STREAM_OPEN_CALLBACK.store(true, Ordering::SeqCst);
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(
        "safe-open-cancel",
        0,
        |_name, request, next| async move {
            let provider = next
                .open_stream(LlmContinuationInvocationV2 {
                    request,
                    target: safe_v2_target(),
                })
                .await
                .map_err(|error| format!("{error:?}"))?;
            Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
                provider.map(|item| item.map_err(|error| format!("{error:?}"))),
            )))
        },
    )
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let state = invoke_safe_v2_streaming(
        &host,
        &registration,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
    );
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();

    let (callback, user_data) = SAFE_V2_HELD_STREAM_OPEN_CALLBACK
        .lock()
        .unwrap()
        .take()
        .expect("the stream-open continuation is pending");
    unsafe {
        callback(
            user_data as *mut c_void,
            NonNull::<NemoRelayNativeLlmStreamV2>::dangling().as_ptr(),
            ptr::null(),
        )
    };
    SAFE_V2_OUTPUT_CANCELLED.store(true, Ordering::SeqCst);
    drive_safe_v2_tasks();

    assert_eq!(SAFE_V2_PROVIDER_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_callback_stops_when_the_caller_cancels() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2("safe-cancel", 0, |_, _, _| async move {
        std::future::pending::<Result<LlmStreamExecutionOutcomeV2, String>>().await
    })
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    let state = unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Pending)
    );
    drive_safe_v2_tasks();
    SAFE_V2_OUTPUT_CANCELLED.store(true, Ordering::SeqCst);
    wake_safe_v2_tasks();
    drive_safe_v2_tasks();
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    assert!(SAFE_V2_OUTPUT.lock().unwrap().is_empty());
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_open_preserves_structured_failure() {
    let _guard = begin_test();
    let expected = LlmContinuationFailureV2::Http {
        status: 429,
        body: "bounded".into(),
        headers: Default::default(),
    };
    *SAFE_V2_OPEN_FAILURE.lock().unwrap() = Some(expected.clone());
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(
        "typed-open-failure",
        0,
        move |_name, request, next| {
            let expected = expected.clone();
            async move {
                let error = match next
                    .open_stream(LlmContinuationInvocationV2 {
                        request,
                        target: safe_v2_target(),
                    })
                    .await
                {
                    Ok(_) => panic!("stream setup should preserve the host failure"),
                    Err(error) => error,
                };
                assert_eq!(error, expected);
                Err("observed typed stream-open failure".into())
            }
        },
    )
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    drive_safe_v2_tasks();
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error == "observed typed stream-open failure"
    ));
    assert_eq!(SAFE_V2_PROVIDER_RELEASES.load(Ordering::SeqCst), 0);
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_malformed_stream_open_releases_the_provider_stream() {
    let _guard = begin_test();
    *SAFE_V2_OPEN_FAILURE.lock().unwrap() = Some(LlmContinuationFailureV2::NonHttp {
        kind: LlmNonHttpFailureKindV2::Internal,
        message: "must not accompany a stream".into(),
    });
    SAFE_V2_OPEN_RETURNS_STREAM_AND_ERROR.store(true, Ordering::SeqCst);
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2(
        "malformed-open",
        0,
        |_, request, next| async move {
            next.open_stream(LlmContinuationInvocationV2 {
                request,
                target: safe_v2_target(),
            })
            .await
            .map(|provider| {
                LlmStreamExecutionOutcomeV2::Stream(Box::pin(
                    provider.map(|item| item.map_err(|error| format!("{error:?}"))),
                ))
            })
            .map_err(|error| format!("{error:?}"))
        },
    )
    .unwrap();
    let registration = take_safe_v2_stream_registration();
    let invocation = json_host_string(
        &host.v3.v1,
        json!({ "name": "managed", "request": test_llm_request() }),
    );
    unsafe {
        (registration.cb)(
            registration.user_data as *mut c_void,
            invocation,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(invocation) };
    drive_safe_v2_tasks();

    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("invalid outcome")
    ));
    assert_eq!(SAFE_V2_PROVIDER_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { registration.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_registration_rejects_a_v1_host_without_leaking_callback_state() {
    let _guard = begin_test();
    let host = test_host();
    let mut ctx = test_context(&host);
    let dropped = Arc::new(AtomicUsize::new(0));
    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let guard = DropCounter(dropped.clone());
    let error = ctx
        .register_async_llm_execution_v2("unsupported", 0, move |_, _, _| {
            let _guard = &guard;
            async { Ok(json!({})) }
        })
        .unwrap_err();
    assert!(error.contains("ABI-v4"));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_failed_host_registration_frees_callback_state_exactly_once() {
    let _guard = begin_test();
    *SAFE_V2_REGISTRATION_STATUS.lock().unwrap() = NemoRelayStatus::AlreadyExists;
    let host = test_host_v4();

    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let buffered_drops = Arc::new(AtomicUsize::new(0));
    let buffered_guard = DropCounter(buffered_drops.clone());
    let mut ctx = test_context(&host.v3.v1);
    assert!(
        ctx.register_async_llm_execution_v2("duplicate", 0, move |_, _, _| {
            let _guard = &buffered_guard;
            async { Ok(json!({})) }
        })
        .unwrap_err()
        .contains("AlreadyExists")
    );
    assert_eq!(buffered_drops.load(Ordering::SeqCst), 1);

    let stream_drops = Arc::new(AtomicUsize::new(0));
    let stream_guard = DropCounter(stream_drops.clone());
    let mut ctx = test_context(&host.v3.v1);
    assert!(
        ctx.register_async_llm_stream_execution_v2("duplicate", 0, move |_, _, _| {
            let _guard = &stream_guard;
            async {
                Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
                    stream::empty(),
                )))
            }
        })
        .unwrap_err()
        .contains("AlreadyExists")
    );
    assert_eq!(stream_drops.load(Ordering::SeqCst), 1);
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_task_spawn_failures_settle_synchronously_and_release_handles() {
    let _guard = begin_test();
    *SAFE_V2_TASK_SPAWN_STATUS.lock().unwrap() = NemoRelayStatus::Internal;
    let host = test_host_v4();

    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("spawn-failure", 0, |_, _, _| async {
        panic!("a failed spawn must not poll the buffered callback")
    })
    .unwrap();
    let buffered = take_safe_v2_buffered_registration();
    let state = invoke_safe_v2_buffered(
        &host,
        &buffered,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
    );
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Complete)
    );
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("task spawn failed: Internal")
    ));
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    assert_eq!(SAFE_V2_COMPLETION_RELEASES.load(Ordering::SeqCst), 0);
    unsafe { buffered.free() };

    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2("stream-spawn-failure", 0, |_, _, _| async {
        panic!("a failed spawn must not poll the streaming callback")
    })
    .unwrap();
    let streaming = take_safe_v2_stream_registration();
    let state = invoke_safe_v2_streaming(
        &host,
        &streaming,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
    );
    assert_eq!(
        NemoRelayNativeAsyncCallbackState::try_from(state),
        Ok(NemoRelayNativeAsyncCallbackState::Complete)
    );
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("task spawn failed: Internal")
    ));
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 2);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    assert!(SAFE_V2_TASKS.lock().unwrap().is_empty());
    unsafe { streaming.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_malformed_invocations_settle_and_release_callback_handles() {
    let _guard = begin_test();
    let host = test_host_v4();
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("malformed-buffered", 0, |_, _, _| async {
        panic!("malformed invocation must not reach the callback")
    })
    .unwrap();
    let buffered = take_safe_v2_buffered_registration();
    let malformed = host_string(&host.v3.v1, "not-json");
    unsafe {
        (buffered.cb)(
            buffered.user_data as *mut c_void,
            malformed,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(malformed) };
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("invalid native API v2 LLM invocation")
    ));
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { buffered.free() };

    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2("malformed-stream", 0, |_, _, _| async {
        panic!("malformed invocation must not reach the callback")
    })
    .unwrap();
    let streaming = take_safe_v2_stream_registration();
    let malformed = host_string(&host.v3.v1, "not-json");
    unsafe {
        (streaming.cb)(
            streaming.user_data as *mut c_void,
            malformed,
            NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
            NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
        )
    };
    unsafe { (host.v3.v1.string_free)(malformed) };
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("invalid native API v2 stream invocation")
    ));
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), 2);
    assert_eq!(SAFE_V2_OUTPUT_RELEASES.load(Ordering::SeqCst), 1);
    unsafe { streaming.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_host_task_honors_wakes_without_plugin_thread_local_state() {
    let _guard = begin_test();
    let host = test_host_v4();
    run_safe_v2_buffered(&host, "self-waking", |_, _, _| async move {
        let first_poll = Arc::new(AtomicBool::new(true));
        std::future::poll_fn(move |cx| {
            if first_poll.swap(false, Ordering::SeqCst) {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            } else {
                std::task::Poll::Ready(())
            }
        })
        .await;
        Ok(json!({ "woke": true }))
    });
    assert_eq!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Ok(json!({ "woke": true })))
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stale_waker_after_completion_is_a_silent_noop() {
    let _guard = begin_test();
    let host = test_host_v4();
    run_safe_v2_buffered(&host, "retained-waker", |_, _, _| async {
        std::future::poll_fn(|context| {
            *SAFE_V2_HELD_TASK_WAKER.lock().unwrap() = Some(context.waker().clone());
            std::task::Poll::Ready(())
        })
        .await;
        Ok(json!({ "done": true }))
    });
    assert!(SAFE_V2_TASKS.lock().unwrap().is_empty());
    *LAST_ERROR.lock().unwrap() = None;
    let waker = SAFE_V2_HELD_TASK_WAKER.lock().unwrap().take().unwrap();
    waker.wake_by_ref();
    assert!(LAST_ERROR.lock().unwrap().is_none());
    drop(waker);
    assert_eq!(SAFE_V2_TASK_RETAINS.load(Ordering::SeqCst), 2);
    assert_eq!(SAFE_V2_TASK_RELEASES.load(Ordering::SeqCst), 3);
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_does_not_settle_a_completion_cancelled_during_callback_polling() {
    let _guard = begin_test();
    let host = test_host_v4();
    run_safe_v2_buffered(&host, "cancel-before-settlement", |_, _, _| async {
        SAFE_V2_COMPLETION_CANCELLED.store(true, Ordering::SeqCst);
        Ok(json!({ "ignored": true }))
    });
    assert!(SAFE_V2_COMPLETION.lock().unwrap().is_none());
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_trampolines_reject_invalid_callback_handles() {
    let _guard = begin_test();
    let host = test_host_v4();

    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("invalid-buffered-handles", 0, |_, _, _| async {
        Ok(json!({}))
    })
    .unwrap();
    let buffered = take_safe_v2_buffered_registration();
    assert_eq!(
        unsafe { (buffered.cb)(ptr::null_mut(), ptr::null(), ptr::null(), ptr::null(),) },
        NemoRelayNativeAsyncCallbackState::Complete as u32
    );
    invoke_safe_v2_buffered(
        &host,
        &buffered,
        ptr::null(),
        NonNull::<NemoRelayNativeAsyncCompletion>::dangling().as_ptr(),
    );
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("NullPointer")
    ));
    let releases = SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst);
    invoke_safe_v2_buffered(
        &host,
        &buffered,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        ptr::null(),
    );
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), releases + 1);
    unsafe { buffered.free() };

    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_stream_execution_v2("invalid-stream-handles", 0, |_, _, _| async {
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
            stream::empty(),
        )))
    })
    .unwrap();
    let streaming = take_safe_v2_stream_registration();
    assert_eq!(
        unsafe { (streaming.cb)(ptr::null_mut(), ptr::null(), ptr::null(), ptr::null(),) },
        NemoRelayNativeAsyncCallbackState::Complete as u32
    );
    invoke_safe_v2_streaming(
        &host,
        &streaming,
        ptr::null(),
        NonNull::<NemoRelayNativeAsyncStream>::dangling().as_ptr(),
    );
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("NullPointer")
    ));
    SAFE_V2_OUTPUT.lock().unwrap().clear();
    let releases = SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst);
    invoke_safe_v2_streaming(
        &host,
        &streaming,
        NonNull::<NemoRelayNativeAsyncNext>::dangling().as_ptr(),
        ptr::null(),
    );
    assert_eq!(SAFE_V2_NEXT_RELEASES.load(Ordering::SeqCst), releases + 1);
    unsafe { streaming.free() };
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_registration_reports_allocation_failure_and_contains_drop_panics() {
    let _guard = begin_test();
    let host = test_host_v4();

    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let mut ctx = test_context(&host.v3.v1);
    assert!(
        ctx.register_async_llm_execution_v2("cannot-allocate", 0, |_, _, _| async {
            Ok(json!({}))
        })
        .unwrap_err()
        .contains("registration name")
    );
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = Some(0);
    let mut ctx = test_context(&host.v3.v1);
    assert!(
        ctx.register_async_llm_stream_execution_v2("cannot-allocate", 0, |_, _, _| async {
            Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
                stream::empty(),
            )))
        })
        .unwrap_err()
        .contains("registration name")
    );
    *STRING_NEW_REMAINING_SUCCESSES.lock().unwrap() = None;

    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("safe callback state drop panic")
        }
    }
    let panic_on_drop = PanicOnDrop;
    let mut ctx = test_context(&host.v3.v1);
    ctx.register_async_llm_execution_v2("drop-panic", 0, move |_, _, _| {
        let _ = &panic_on_drop;
        async { Ok(json!({})) }
    })
    .unwrap();
    let registration = take_safe_v2_buffered_registration();
    unsafe { registration.free() };
    assert_eq!(
        LAST_ERROR.lock().unwrap().as_deref(),
        Some("native API v2 safe callback state drop panicked")
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_continuations_preserve_abi_and_provider_failures() {
    let _guard = begin_test();
    let host = test_host_v4();

    *SAFE_V2_TARGETED_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_buffered(&host, "target-status", |_, request, next| async move {
        next.call(LlmContinuationInvocationV2 {
            request,
            target: safe_v2_target(),
        })
        .await
        .map_err(|error| format!("{error:?}"))
    });
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("InvalidArg")
    ));
    *SAFE_V2_TARGETED_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SAFE_V2_PASSTHROUGH_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_buffered(&host, "passthrough-status", |_, request, next| async move {
        next.call_passthrough(request).await
    });
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("InvalidArg")
    ));
    *SAFE_V2_PASSTHROUGH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *SAFE_V2_PASSTHROUGH_ERROR.lock().unwrap() = Some("provider rejected passthrough".into());
    run_safe_v2_buffered(&host, "passthrough-error", |_, request, next| async move {
        next.call_passthrough(request).await
    });
    assert_eq!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err("provider rejected passthrough".into()))
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_buffered_callback_settlement_is_bounded_and_panic_safe() {
    let _guard = begin_test();
    let host = test_host_v4();

    run_safe_v2_buffered(&host, "panic", |_, _, _| async move {
        panic!("buffered callback panic")
    });
    assert!(matches!(
        SAFE_V2_COMPLETION.lock().unwrap().take(),
        Some(Err(error)) if error.contains("callback panicked")
    ));

    let long_error = "é".repeat(3_000);
    run_safe_v2_buffered(&host, "bounded-error", move |_, _, _| {
        let long_error = long_error.clone();
        async move { Err(long_error) }
    });
    let error = SAFE_V2_COMPLETION
        .lock()
        .unwrap()
        .take()
        .unwrap()
        .unwrap_err();
    assert_eq!(error.len(), 4 * 1024);
    assert!(error.is_char_boundary(error.len()));

    *SAFE_V2_COMPLETION_RESOLVE_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_buffered(&host, "resolve-status", |_, _, _| async {
        Ok(json!({ "ok": true }))
    });
    assert!(
        LAST_ERROR
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("completion failed"))
    );
    *SAFE_V2_COMPLETION_RESOLVE_STATUS.lock().unwrap() = NemoRelayStatus::Ok;

    *LAST_ERROR.lock().unwrap() = None;
    *SAFE_V2_COMPLETION_REJECT_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_buffered(&host, "reject-status", |_, _, _| async {
        Err("rejected".into())
    });
    assert!(
        LAST_ERROR
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("rejection failed"))
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_provider_stream_preserves_late_and_boundary_failures() {
    let _guard = begin_test();
    let host = test_host_v4();

    *SAFE_V2_STREAM_OPEN_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_targeted_stream(&host, "open-status");
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("InvalidArg")
    ));
    *SAFE_V2_STREAM_OPEN_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    *SAFE_V2_PROVIDER_NEXT_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_targeted_stream(&host, "poll-status");
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("InvalidArg")
    ));
    *SAFE_V2_PROVIDER_NEXT_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    SAFE_V2_PROVIDER_EVENTS
        .lock()
        .unwrap()
        .push_back(SafeV2ProviderEvent::Failure(
            LlmContinuationFailureV2::NonHttp {
                kind: LlmNonHttpFailureKindV2::Transport,
                message: "late provider failure".into(),
            },
        ));
    run_safe_v2_targeted_stream(&host, "late-failure");
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("late provider failure")
    ));
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    *SAFE_V2_PROVIDER_EVENT_JSON.lock().unwrap() = Some("not-json".into());
    run_safe_v2_targeted_stream(&host, "malformed-event");
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("stream chunk")
    ));
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    SAFE_V2_PROVIDER_INVALID_EVENT.store(true, Ordering::SeqCst);
    run_safe_v2_targeted_stream(&host, "invalid-event");
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("invalid event")
    ));
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_output_handles_backpressure_cancellation_and_settlement_errors() {
    let _guard = begin_test();
    let host = test_host_v4();

    SAFE_V2_OUTPUT_PUSH_STATUSES
        .lock()
        .unwrap()
        .extend([NemoRelayStatus::WouldBlock, NemoRelayStatus::Ok]);
    run_safe_v2_streaming(&host, "push-backpressure", |_, _, _| async {
        Ok(safe_v2_one_chunk_stream())
    });
    assert_eq!(
        *SAFE_V2_OUTPUT.lock().unwrap(),
        vec![Ok(json!({ "chunk": true }))]
    );
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    for (name, status, expected) in [
        (
            "push-internal",
            NemoRelayStatus::Internal,
            "output push failed: Internal",
        ),
        (
            "push-failure",
            NemoRelayStatus::InvalidArg,
            "output push failed",
        ),
    ] {
        SAFE_V2_OUTPUT_PUSH_STATUSES
            .lock()
            .unwrap()
            .push_back(status);
        run_safe_v2_streaming(&host, name, |_, _, _| async {
            Ok(safe_v2_one_chunk_stream())
        });
        assert!(matches!(
            SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
            [Err(error)] if error.contains(expected)
        ));
        SAFE_V2_OUTPUT.lock().unwrap().clear();
    }

    *SAFE_V2_OUTPUT_FINISH_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_streaming(&host, "finish-failure", |_, _, _| async {
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
            stream::empty(),
        )))
    });
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("output finish failed")
    ));
    *SAFE_V2_OUTPUT_FINISH_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    SAFE_V2_OUTPUT_REJECT_STATUSES
        .lock()
        .unwrap()
        .extend([NemoRelayStatus::WouldBlock, NemoRelayStatus::Ok]);
    run_safe_v2_streaming(&host, "reject-backpressure", |_, _, _| async {
        Err("stream rejected".into())
    });
    assert_eq!(
        *SAFE_V2_OUTPUT.lock().unwrap(),
        vec![Err("stream rejected".into())]
    );
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    SAFE_V2_OUTPUT_REJECT_STATUSES
        .lock()
        .unwrap()
        .push_back(NemoRelayStatus::InvalidArg);
    *LAST_ERROR.lock().unwrap() = None;
    run_safe_v2_streaming(&host, "reject-failure", |_, _, _| async {
        Err("stream rejected".into())
    });
    assert!(
        LAST_ERROR
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("output rejection failed"))
    );
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_stream_cancellation_and_pass_through_failures_settle_once() {
    let _guard = begin_test();
    let host = test_host_v4();

    run_safe_v2_streaming(&host, "cancel-before-push", |_, _, _| async {
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(stream::once(
            async {
                SAFE_V2_OUTPUT_CANCELLED.store(true, Ordering::SeqCst);
                Ok(json!({ "ignored": true }))
            },
        ))))
    });
    assert!(SAFE_V2_OUTPUT.lock().unwrap().is_empty());
    SAFE_V2_OUTPUT_CANCELLED.store(false, Ordering::SeqCst);

    run_safe_v2_streaming(&host, "cancel-before-finish", |_, _, _| async {
        Ok(LlmStreamExecutionOutcomeV2::Stream(Box::pin(
            stream::poll_fn(|_| {
                SAFE_V2_OUTPUT_CANCELLED.store(true, Ordering::SeqCst);
                std::task::Poll::Ready(None)
            }),
        )))
    });
    assert!(SAFE_V2_OUTPUT.lock().unwrap().is_empty());
    SAFE_V2_OUTPUT_CANCELLED.store(false, Ordering::SeqCst);

    *SAFE_V2_FORWARD_STATUS.lock().unwrap() = NemoRelayStatus::InvalidArg;
    run_safe_v2_streaming(&host, "forward-status", |_, request, _| async move {
        Ok(LlmStreamExecutionOutcomeV2::Passthrough(request))
    });
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("InvalidArg")
    ));
    *SAFE_V2_FORWARD_STATUS.lock().unwrap() = NemoRelayStatus::Ok;
    SAFE_V2_OUTPUT.lock().unwrap().clear();

    run_safe_v2_streaming(&host, "stream-panic", |_, _, _| async move {
        panic!("stream callback panic")
    });
    assert!(matches!(
        SAFE_V2_OUTPUT.lock().unwrap().as_slice(),
        [Err(error)] if error.contains("callback panicked")
    ));
    assert_eq!(live_host_strings(), 0);
}

#[test]
fn safe_v2_host_string_failures_do_not_leave_unsettled_handles() {
    let _guard = begin_test();
    let host = test_host_v4();

    run_safe_v2_buffered(&host, "resolve-allocation", |_, _, _| async {
        *STRING_NEW_RETURNS_NULL.lock().unwrap() = true;
        Ok(json!({ "cannot": "allocate" }))
    });
    *STRING_NEW_RETURNS_NULL.lock().unwrap() = false;
    assert!(LAST_ERROR.lock().unwrap().is_none());

    *LAST_ERROR.lock().unwrap() = None;
    run_safe_v2_streaming(&host, "reject-allocation", |_, _, _| async {
        *STRING_NEW_RETURNS_NULL.lock().unwrap() = true;
        Err("cannot allocate rejection".into())
    });
    *STRING_NEW_RETURNS_NULL.lock().unwrap() = false;
    assert!(LAST_ERROR.lock().unwrap().is_none());
    assert_eq!(live_host_strings(), 0);
}
