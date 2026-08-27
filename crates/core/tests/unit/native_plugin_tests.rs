// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit coverage for the native plugin host ABI adapter.

use super::*;

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nemo_relay_plugin::{
    NemoRelayNativeLlmNextFn, NemoRelayNativeLlmSanitizeRequestContext,
    NemoRelayNativeLlmSanitizeResponseContext, NemoRelayNativeLlmStreamNextFn,
    NemoRelayNativeToolNextFn,
};
#[cfg(unix)]
use nemo_relay_plugin::{NemoRelayNativePluginRegisterFn, NemoRelayNativePluginValidateFn};
use serde_json::json;

use crate::api::optimization::{
    LlmOptimizationRecorder, current_llm_optimization_recorder, scope_llm_optimization_recorder,
};
use crate::api::runtime::scope_stack::active_event_uuid;
use crate::api::runtime::subscriber_dispatcher::{
    capture_nested_publication_buffer, with_task_publication_context,
};
use crate::api::runtime::{
    BuiltinLlmCodec, LlmSanitizeRequestContext, LlmSanitizeResponseContext,
    MiddlewareContinuationLease, NemoRelayContextState, TASK_SCOPE_STACK, current_scope_stack,
    global_context, with_active_event_uuid,
};
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::response::AnnotatedLlmResponse;

type RawToolExecutionNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> + Send + Sync>;

fn canonical_tool_next(next: RawToolExecutionNextFn) -> NativeAsyncNextInner {
    NativeAsyncNextInner::Tool(Arc::new(move |value| {
        let future = next(value);
        Box::pin(async move { future.await.map(ToolExecutionResult::from) })
    }))
}

struct ThreadScopeStackRestore(Option<ThreadScopeStackBinding>);

impl ThreadScopeStackRestore {
    fn capture() -> Self {
        Self(Some(capture_thread_scope_stack()))
    }
}

impl Drop for ThreadScopeStackRestore {
    fn drop(&mut self) {
        if let Some(binding) = self.0.take() {
            restore_thread_scope_stack(binding);
        }
    }
}

struct GlobalContextRestore(Option<NemoRelayContextState>);

impl GlobalContextRestore {
    fn replace_with_empty() -> Self {
        let context = global_context();
        let previous =
            std::mem::take(&mut *context.write().unwrap_or_else(|error| error.into_inner()));
        Self(Some(previous))
    }
}

impl Drop for GlobalContextRestore {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            *global_context()
                .write()
                .unwrap_or_else(|error| error.into_inner()) = previous;
        }
    }
}

fn native_string(value: &str) -> *mut NemoRelayNativeString {
    native_string_from_str(value).expect("native string allocation should succeed")
}

fn assert_last_error_contains(expected: &str) {
    let error = native_last_error_message().expect("native last error should be set");
    assert!(
        error.contains(expected),
        "expected native error '{error}' to contain '{expected}'"
    );
}

fn wait_for_native_reaper<T>(value: &Arc<T>, expected_strong_count: usize) {
    for _ in 0..1_000 {
        if Arc::strong_count(value) == expected_strong_count {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(Arc::strong_count(value), expected_strong_count);
}

unsafe extern "C" fn accept_native_stream_item(
    _user_data: *mut c_void,
    _chunk_json: *const NemoRelayNativeString,
    _error: *const NemoRelayNativeString,
    _done: bool,
) -> bool {
    true
}

unsafe extern "C" fn report_native_callback_drop_thread(user_data: *mut c_void) {
    let sender = unsafe { Box::from_raw(user_data.cast::<std::sync::mpsc::Sender<String>>()) };
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string();
    let _ = sender.send(thread_name);
}

#[test]
fn native_async_release_defers_library_guard_drop_to_host_reaper() {
    let (sender, receiver) = std::sync::mpsc::channel::<String>();
    let callback_user_data = Arc::new(NativeCallbackUserData {
        ptr: Box::into_raw(Box::new(sender)).cast(),
        free_fn: Some(report_native_callback_drop_thread),
        _instance: None,
    });
    let (result_sender, _result_receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(result_sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: Some(callback_user_data),
    });
    let raw = Arc::into_raw(completion).cast();
    unsafe { native_async_completion_release(raw) };
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
        "nemo-relay-native-reaper"
    );
}

#[test]
fn native_loader_empty_input_and_stream_context_drop_are_safe() {
    let activation = load_native_plugins(Vec::<NativePluginLoadSpec>::new()).unwrap();
    assert!(activation.plugins.is_empty());

    let next: LlmStreamExecutionNextFn =
        Arc::new(|_| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) }));
    let raw = Box::into_raw(Box::new(next)).cast::<c_void>();
    let context = NativeStreamNextContext::new(raw);
    drop(context);
}

#[test]
fn native_last_error_helpers_store_and_clear_messages() {
    clear_native_last_error();
    assert_eq!(native_last_error_message(), None);
    set_native_last_error("native test error");
    assert_eq!(
        native_last_error_message().as_deref(),
        Some("native test error")
    );
    clear_native_last_error();
    assert_eq!(native_last_error_message(), None);
}

unsafe extern "C" fn complete_native_next_result(
    user_data: *mut c_void,
    value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
) {
    let sender = unsafe {
        Box::from_raw(
            user_data as *mut tokio::sync::oneshot::Sender<std::result::Result<Json, String>>,
        )
    };
    let result = if !error.is_null() {
        Err(read_native_string(error).unwrap())
    } else {
        parse_json_arg(value_json, "native next callback result")
            .map_err(|status| format!("invalid native next callback result: {status:?}"))
    };
    let _ = sender.send(result);
}

type PullOpenResult = std::result::Result<usize, String>;
type PullItemResult = std::result::Result<Option<Json>, String>;

unsafe extern "C" fn complete_pull_stream_open(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmAsyncStream,
    error: *const NemoRelayNativeString,
) {
    let sender =
        unsafe { Box::from_raw(user_data as *mut tokio::sync::oneshot::Sender<PullOpenResult>) };
    let result = if !error.is_null() {
        Err(read_native_string(error).unwrap())
    } else if stream.is_null() {
        Err("pull stream open returned no stream".into())
    } else {
        Ok(stream as usize)
    };
    let _ = sender.send(result);
}

unsafe extern "C" fn count_native_result_callback(
    user_data: *mut c_void,
    _value_json: *const NemoRelayNativeString,
    _error: *const NemoRelayNativeString,
) {
    unsafe { &*user_data.cast::<AtomicUsize>() }.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn count_native_pull_open_callback(
    user_data: *mut c_void,
    _stream: *const NemoRelayNativeLlmAsyncStream,
    _error: *const NemoRelayNativeString,
) {
    unsafe { &*user_data.cast::<AtomicUsize>() }.fetch_add(1, Ordering::SeqCst);
}

struct OwnedCallbackProbe {
    callbacks: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    error: Arc<Mutex<Option<String>>>,
}

struct CallbackProbe {
    callbacks: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    error: Arc<Mutex<Option<String>>>,
}

impl CallbackProbe {
    fn new() -> (*mut c_void, Self) {
        let probe = Self {
            callbacks: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
            error: Arc::new(Mutex::new(None)),
        };
        let user_data = Box::into_raw(Box::new(OwnedCallbackProbe {
            callbacks: Arc::clone(&probe.callbacks),
            drops: Arc::clone(&probe.drops),
            error: Arc::clone(&probe.error),
        }))
        .cast();
        (user_data, probe)
    }
}

impl Drop for OwnedCallbackProbe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn record_owned_result_callback(
    user_data: *mut c_void,
    _value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
) {
    let probe = unsafe { Box::from_raw(user_data.cast::<OwnedCallbackProbe>()) };
    probe.callbacks.fetch_add(1, Ordering::SeqCst);
    if !error.is_null() {
        *probe
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = read_native_string(error).ok();
    }
}

unsafe extern "C" fn record_owned_pull_open_callback(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmAsyncStream,
    error: *const NemoRelayNativeString,
) {
    assert!(stream.is_null(), "cancelled open must not return a stream");
    unsafe { record_owned_result_callback(user_data, ptr::null(), error) };
}

unsafe extern "C" fn complete_pull_stream_item(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) {
    let sender =
        unsafe { Box::from_raw(user_data as *mut tokio::sync::oneshot::Sender<PullItemResult>) };
    let result = if !error.is_null() {
        Err(read_native_string(error).unwrap())
    } else if done {
        Ok(None)
    } else {
        parse_json_arg(chunk_json, "pull stream chunk")
            .map(Some)
            .map_err(|status| format!("invalid pull stream chunk: {status:?}"))
    };
    let _ = sender.send(result);
}

#[derive(Default)]
struct NativeStreamCallbackState {
    error: Mutex<Option<String>>,
    done: AtomicBool,
    callbacks: AtomicUsize,
    notified: tokio::sync::Notify,
}

struct OwnedNativeStreamCallbackState {
    result: Arc<NativeStreamCallbackState>,
    drop_count: Arc<AtomicUsize>,
    stream: *const NemoRelayNativeAsyncStream,
}

impl Drop for OwnedNativeStreamCallbackState {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn record_native_stream_result(
    user_data: *mut c_void,
    _chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool {
    let state = unsafe { &*(user_data as *const NativeStreamCallbackState) };
    state.callbacks.fetch_add(1, Ordering::SeqCst);
    if !error.is_null() {
        *state
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = read_native_string(error).ok();
    }
    state.done.store(done, Ordering::Release);
    state.notified.notify_one();
    true
}

unsafe extern "C" fn record_and_release_native_stream_result(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool {
    if !chunk_json.is_null() {
        return true;
    }
    let state = unsafe { Box::from_raw(user_data as *mut OwnedNativeStreamCallbackState) };
    state.result.callbacks.fetch_add(1, Ordering::SeqCst);
    if !error.is_null() {
        *state
            .result
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = read_native_string(error).ok();
    }
    state.result.done.store(done, Ordering::Release);
    let result = Arc::clone(&state.result);
    unsafe { native_async_stream_release(state.stream) };
    drop(state);
    result.notified.notify_one();
    false
}

unsafe extern "C" fn stop_after_first_native_stream_item(
    user_data: *mut c_void,
    _chunk_json: *const NemoRelayNativeString,
    _error: *const NemoRelayNativeString,
    _done: bool,
) -> bool {
    let callbacks = unsafe { &*(user_data as *const AtomicUsize) };
    callbacks.fetch_add(1, Ordering::SeqCst);
    false
}

#[test]
fn native_async_entrypoints_reject_null_handles() {
    unsafe {
        assert_eq!(
            native_async_completion_resolve_json(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_async_completion_reject(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert!(native_async_completion_is_cancelled(ptr::null()));
        native_async_completion_release(ptr::null());
        native_async_next_release(ptr::null());

        assert_eq!(
            native_async_stream_push_json(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_async_stream_finish(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_async_stream_reject(ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert!(native_async_stream_is_cancelled(ptr::null()));
        native_async_stream_release(ptr::null());

        assert_eq!(
            native_async_next_invoke(ptr::null(), ptr::null(), ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_async_next_invoke_result(
                ptr::null(),
                ptr::null(),
                complete_native_next_result,
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_async_next_invoke_stream(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                accept_native_stream_item,
                ptr::null_mut(),
            ),
            NemoRelayStatus::NullPointer
        );
    }
}

#[cfg(unix)]
unsafe extern "C" fn native_test_validate_invalid_json(
    _user_data: *mut c_void,
    _config: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out = native_string("not-json") };
    NemoRelayStatus::Ok
}

#[cfg(unix)]
unsafe extern "C" fn native_test_validate_error(
    _user_data: *mut c_void,
    _config: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out = native_string("unused") };
    set_native_last_error("validation callback failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn native_test_validate_empty(
    _user_data: *mut c_void,
    _config: *const NemoRelayNativeString,
    _out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_test_register_ok(
    _user_data: *mut c_void,
    _config: *const NemoRelayNativeString,
    _ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

#[cfg(unix)]
unsafe extern "C" fn native_test_register_error(
    _user_data: *mut c_void,
    _config: *const NemoRelayNativeString,
    _ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    set_native_last_error("registration callback failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
fn native_test_adapter(
    validate: Option<NemoRelayNativePluginValidateFn>,
    register: Option<NemoRelayNativePluginRegisterFn>,
) -> NativePluginAdapter {
    let plugin = NemoRelayNativePluginV1 {
        validate,
        register,
        ..Default::default()
    };
    NativePluginAdapter {
        plugin_kind: "test.native.adapter".into(),
        allows_multiple_components: false,
        instance: Arc::new(NativePluginInstance {
            plugin_kind: "test.native.adapter".into(),
            relay_compat: "^0.8".into(),
            allows_multiple_components: false,
            plugin: Mutex::new(plugin),
            _library: libloading::os::unix::Library::this().into(),
        }),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn native_plugin_adapter_covers_validation_and_registration_results() {
    let no_validate = native_test_adapter(None, Some(native_test_register_ok));
    assert_eq!(no_validate.plugin_kind(), "test.native.adapter");
    assert!(!no_validate.allows_multiple_components());
    assert!(no_validate.validate(&Map::new()).is_empty());

    let empty = native_test_adapter(
        Some(native_test_validate_empty),
        Some(native_test_register_ok),
    );
    assert!(empty.validate(&Map::new()).is_empty());

    let invalid = native_test_adapter(
        Some(native_test_validate_invalid_json),
        Some(native_test_register_ok),
    );
    let diagnostics = invalid.validate(&Map::new());
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].message.contains("invalid diagnostics JSON"));

    let failing = native_test_adapter(
        Some(native_test_validate_error),
        Some(native_test_register_error),
    );
    let diagnostics = failing.validate(&Map::new());
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].message, "validation callback failed");

    let mut context = PluginRegistrationContext::new();
    assert!(empty.register(&Map::new(), &mut context).await.is_ok());
    let error = failing
        .register(&Map::new(), &mut context)
        .await
        .expect_err("registration callback should fail");
    assert!(error.to_string().contains("registration callback failed"));

    let missing = native_test_adapter(Some(native_test_validate_empty), None);
    let error = missing
        .register(&Map::new(), &mut context)
        .await
        .expect_err("missing registration callback should fail");
    assert!(
        error
            .to_string()
            .contains("did not return a register callback")
    );
}

#[test]
fn native_loader_helpers_cover_compatibility_descriptor_and_digest_edges() {
    assert_native_compatibility_edges();
    assert_native_descriptor_edges();
    assert_native_digest_edges();
    assert_native_host_api_versions();
}

#[test]
fn native_status_helpers_keep_error_categories_stable() {
    clear_native_last_error();
    assert!(matches!(
        flow_error_from_status(NemoRelayStatus::NotFound, "missing"),
        FlowError::NotFound(message) if message.contains("missing")
    ));
    assert_eq!(
        status_from_plugin_error(PluginError::InvalidConfig("bad".into())),
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        status_from_flow_error(FlowError::Internal("bad".into())),
        NemoRelayStatus::Internal
    );
    assert_eq!(panic_payload_message(&"string panic"), "string panic");
    clear_native_last_error();
}

fn assert_native_compatibility_edges() {
    assert!(validate_relay_compatibility(None).is_err());
    assert!(validate_relay_compatibility(Some(" ")).is_err());
    assert!(validate_relay_compatibility(Some("not a requirement")).is_err());
    assert!(validate_relay_compatibility(Some(">=999.0.0")).is_err());
    let host_requirement = format!("={}", env!("CARGO_PKG_VERSION"));
    assert!(validate_relay_compatibility(Some(&host_requirement)).is_ok());
}

fn assert_native_descriptor_edges() {
    let mut descriptor = NemoRelayNativePluginV1 {
        struct_size: 0,
        ..Default::default()
    };
    assert!(validate_plugin_descriptor("test", &descriptor).is_err());
    descriptor.struct_size = std::mem::size_of::<NemoRelayNativePluginV1>();
    assert!(validate_plugin_descriptor("test", &descriptor).is_err());
    descriptor.plugin_kind = native_string("test");
    assert!(validate_plugin_descriptor("test", &descriptor).is_err());
    descriptor.register = Some(native_test_register_ok);
    assert!(validate_plugin_descriptor("test", &descriptor).is_ok());
    drop_native_plugin_descriptor(&mut descriptor);
}

fn assert_native_digest_edges() {
    let temp = tempfile::tempdir().unwrap();
    let manifest = temp.path().join("plugin.toml");
    let library = temp.path().join("plugin.bin");
    std::fs::write(&library, b"native plugin bytes").unwrap();
    assert_eq!(
        resolve_manifest_relative_path(&manifest, "plugin.bin"),
        library
    );
    assert_eq!(
        resolve_manifest_relative_path(&manifest, temp.path().to_str().unwrap()),
        temp.path()
    );
    assert_eq!(hex_digest([0x00, 0xab, 0xff]), "00abff");
    let digest = hex_digest(Sha256::digest(b"native plugin bytes"));
    assert!(verify_sha256(&library, &format!("sha256:{}", digest.to_uppercase())).is_ok());
    assert!(verify_sha256(&library, "sha256:00").is_err());
    assert!(verify_sha256(&temp.path().join("missing"), "00").is_err());
}

fn assert_native_host_api_versions() {
    let current = native_host_api();
    let frozen_v3 = native_host_api_v3();
    let legacy = native_host_api_v2();
    assert!(!current.is_null());
    assert!(!frozen_v3.is_null());
    assert!(!legacy.is_null());
    assert_eq!(
        unsafe { (*current).abi_version },
        NEMO_RELAY_NATIVE_ABI_VERSION
    );
    assert_eq!(unsafe { (*frozen_v3).abi_version }, 3);
    assert_eq!(
        unsafe { (*legacy).abi_version },
        NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY
    );
    assert_eq!(
        unsafe { (*current).struct_size },
        std::mem::size_of::<NemoRelayNativeHostApiV4>()
    );
    assert_eq!(
        unsafe { (*frozen_v3).struct_size },
        std::mem::size_of::<NemoRelayNativeHostApiV3>()
    );
    assert_eq!(
        unsafe { (*legacy).struct_size },
        std::mem::size_of::<NemoRelayNativeHostApiV1>()
    );
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(std::mem::align_of::<NemoRelayNativeHostApiV4>(), 8);
        assert_eq!(std::mem::size_of::<NemoRelayNativeHostApiV4>(), 600);
        assert_eq!(std::mem::offset_of!(NemoRelayNativeHostApiV4, v3), 0);
        assert_eq!(
            std::mem::offset_of!(NemoRelayNativeHostApiV4, emit_mark_v2),
            512
        );
        assert_eq!(
            std::mem::offset_of!(NemoRelayNativeHostApiV4, get_runtime_diagnostics),
            520
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_context_register_conditional_middleware_guardrail
            ),
            576
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_runtime_register_conditional_middleware_guardrail_callback
            ),
            584
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_context_register_conditional_middleware_guardrail_callback
            ),
            592
        );
    }
    #[cfg(target_pointer_width = "32")]
    {
        assert_eq!(std::mem::align_of::<NemoRelayNativeHostApiV4>(), 4);
        assert_eq!(std::mem::size_of::<NemoRelayNativeHostApiV4>(), 296);
        assert_eq!(std::mem::offset_of!(NemoRelayNativeHostApiV4, v3), 0);
        assert_eq!(
            std::mem::offset_of!(NemoRelayNativeHostApiV4, emit_mark_v2),
            252
        );
        assert_eq!(
            std::mem::offset_of!(NemoRelayNativeHostApiV4, get_runtime_diagnostics),
            256
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_context_register_conditional_middleware_guardrail
            ),
            284
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_runtime_register_conditional_middleware_guardrail_callback
            ),
            288
        );
        assert_eq!(
            std::mem::offset_of!(
                NemoRelayNativeHostApiV4,
                plugin_context_register_conditional_middleware_guardrail_callback
            ),
            292
        );
    }
}

#[tokio::test]
async fn native_async_wait_and_rejection_cover_dropped_and_aborted_continuations() {
    let (sender, receiver) = tokio::sync::oneshot::channel::<FlowResult<Json>>();
    drop(sender);
    let mut wait = NativeAsyncWait {
        completion: Arc::new(NativeAsyncCompletion {
            sender: Mutex::new(None),
            cancelled: AtomicBool::new(false),
            next_invoked: AtomicBool::new(false),
            next_abort: Mutex::new(None),
            continuation_aborts: Mutex::new(HashMap::new()),
            codec: None,
            before_settlement_lock: None,
            _callback_user_data: None,
        }),
        receiver,
        completed: false,
    };
    let error = wait
        .receive()
        .await
        .expect_err("a dropped callback must fail the wait");
    assert!(error.to_string().contains("dropped without settling"));

    let task = tokio::spawn(std::future::pending::<()>());
    let abort = task.abort_handle();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(true),
        next_abort: Mutex::new(Some(abort)),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invalid_message = Box::into_raw(Box::new(NativeHostString(vec![0xff]))).cast();
    assert_eq!(
        unsafe { native_async_completion_reject(completion_ref, invalid_message) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_string_free(invalid_message) };

    let message = native_string("continuation rejected");
    assert_eq!(
        unsafe { native_async_completion_reject(completion_ref, message) },
        NemoRelayStatus::Ok
    );
    unsafe { native_string_free(message) };
    let error = receiver.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("continuation rejected"));
    assert!(task.await.unwrap_err().is_cancelled());
    unsafe { native_async_completion_release(completion_ref) };
}

#[test]
fn native_stream_callback_guard_covers_terminal_drop_modes() {
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let callback_state = NativeStreamCallbackState::default();
    let user_data: *mut c_void = ptr::from_ref(&callback_state).cast_mut().cast();

    let mut inactive = NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: user_data as usize,
        stream: Arc::clone(&stream),
        _library_guard: None,
        active: false,
    };
    inactive.fail("ignored");

    stream.cancelled.store(true, Ordering::Release);
    let mut cancelled = NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: user_data as usize,
        stream: Arc::clone(&stream),
        _library_guard: None,
        active: true,
    };
    cancelled.fail("cancellation owns settlement");
    drop(cancelled);

    stream.cancelled.store(false, Ordering::Release);
    stream.settled.store(true, Ordering::Release);
    drop(NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: user_data as usize,
        stream: Arc::clone(&stream),
        _library_guard: None,
        active: true,
    });

    stream.settled.store(false, Ordering::Release);
    drop(NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: user_data as usize,
        stream,
        _library_guard: None,
        active: true,
    });
    assert_eq!(callback_state.callbacks.load(Ordering::Acquire), 3);
    assert!(callback_state.done.load(Ordering::Acquire));
}

#[tokio::test]
async fn native_async_stream_forwarding_reports_conversion_and_stream_errors() {
    let make_stream_state = || {
        let (sender, _receiver) = tokio::sync::mpsc::channel(1);
        Arc::new(NativeAsyncStream {
            sender: Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
            settled: AtomicBool::new(false),
            backpressured: AtomicBool::new(false),
            downstream_aborts: Mutex::new(HashMap::new()),
            settlement: Mutex::new(()),
            before_settlement_lock: None,
            _callback_user_data: None,
        })
    };

    let conversion_state = NativeStreamCallbackState::default();
    let mut conversion_guard = NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: ptr::from_ref(&conversion_state) as usize,
        stream: make_stream_state(),
        _library_guard: None,
        active: true,
    };
    forward_native_async_next_stream_with(
        LlmJsonStream::new(tokio_stream::iter([Ok(json!({"chunk": true}))])),
        &mut conversion_guard,
        |_| None,
    )
    .await;
    assert!(
        conversion_state
            .error
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("failed to serialize or allocate"))
    );
    assert!(!conversion_state.done.load(Ordering::Acquire));

    let stream_error_state = NativeStreamCallbackState::default();
    let mut stream_error_guard = NativeAsyncStreamCallbackGuard {
        cb: record_native_stream_result,
        user_data: ptr::from_ref(&stream_error_state) as usize,
        stream: make_stream_state(),
        _library_guard: None,
        active: true,
    };
    forward_native_async_next_stream(
        LlmJsonStream::new(tokio_stream::iter([Err(FlowError::Internal(
            "provider stream failed".into(),
        ))])),
        &mut stream_error_guard,
    )
    .await;
    assert!(
        stream_error_state
            .error
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("provider stream failed"))
    );
}

#[tokio::test]
async fn native_async_result_entrypoint_covers_llm_and_stream_continuations() {
    let llm_next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::Llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })),
        tokio::runtime::Handle::current(),
        None,
    ));
    let llm_ref = Arc::into_raw(llm_next) as *const NemoRelayNativeAsyncNext;
    let invalid = native_string("{}");
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                llm_ref,
                invalid,
                complete_native_next_result,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::InvalidJson
    );
    unsafe { native_string_free(invalid) };

    let request = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"message": "hello"}),
        })
        .unwrap(),
    )
    .unwrap();
    let (sender, receiver) = tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                llm_ref,
                request,
                complete_native_next_result,
                Box::into_raw(Box::new(sender)).cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        receiver.await.unwrap().unwrap(),
        json!({"message": "hello"})
    );
    unsafe {
        native_string_free(request);
        native_async_next_release(llm_ref);
    }

    let stream_next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
        })),
        tokio::runtime::Handle::current(),
        None,
    ));
    let stream_ref = Arc::into_raw(stream_next) as *const NemoRelayNativeAsyncNext;
    let invocation = native_string("null");
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                stream_ref,
                invocation,
                complete_native_next_result,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    unsafe {
        native_string_free(invocation);
        native_async_next_release(stream_ref);
    }
}

#[test]
fn rejected_native_next_registration_does_not_invoke_result_or_open_callbacks() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let request = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
        .unwrap(),
    )
    .unwrap();
    let callbacks = AtomicUsize::new(0);

    let mut unary_next = NativeAsyncNext::new(
        NativeAsyncNextInner::Llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })),
        runtime.handle().clone(),
        None,
    );
    unary_next.owner = Some(NativeAsyncNextOwner::Completion(Weak::new()));
    let unary_next = Arc::into_raw(Arc::new(unary_next)) as *const NemoRelayNativeAsyncNext;
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                unary_next,
                request,
                count_native_result_callback,
                ptr::from_ref(&callbacks).cast_mut().cast(),
            )
        },
        NemoRelayStatus::InvalidArg
    );

    let mut stream_next = NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
        })),
        runtime.handle().clone(),
        None,
    );
    stream_next.owner = Some(NativeAsyncNextOwner::Stream(Weak::new()));
    let stream_next = Arc::into_raw(Arc::new(stream_next)) as *const NemoRelayNativeAsyncNext;
    assert_eq!(
        unsafe {
            native_async_next_open_llm_stream(
                stream_next,
                request,
                count_native_pull_open_callback,
                ptr::from_ref(&callbacks).cast_mut().cast(),
            )
        },
        NemoRelayStatus::InvalidArg
    );

    runtime.block_on(tokio::task::yield_now());
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    unsafe {
        native_string_free(request);
        native_async_next_release(unary_next);
        native_async_next_release(stream_next);
    }
}

#[test]
fn accepted_native_callbacks_settle_when_cancelled_before_first_poll() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(completion_tx)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let wait = NativeAsyncWait {
        completion: Arc::clone(&completion),
        receiver: completion_rx,
        completed: false,
    };
    let unary_started = Arc::new(AtomicBool::new(false));
    let unary_next = Arc::new(NativeAsyncNext::with_completion_owner(
        canonical_tool_next({
            let started = Arc::clone(&unary_started);
            Arc::new(move |value| {
                started.store(true, Ordering::SeqCst);
                Box::pin(async move { Ok(value) })
            })
        }),
        runtime.handle().clone(),
        None,
        &completion,
    ));
    let unary_next_ref = Arc::into_raw(unary_next) as *const NemoRelayNativeAsyncNext;
    let unary_invocation = native_string_from_json(&json!({"pending": true})).unwrap();
    let (unary_user_data, unary_probe) = CallbackProbe::new();
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                unary_next_ref,
                unary_invocation,
                record_owned_result_callback,
                unary_user_data,
            )
        },
        NemoRelayStatus::Ok
    );
    drop(wait);

    let (stream_sender, stream_receiver) = tokio::sync::mpsc::channel(1);
    let stream_owner = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(stream_sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let pull_started = Arc::new(AtomicBool::new(false));
    let pull_next = Arc::new(NativeAsyncNext::with_stream_owner(
        NativeAsyncNextInner::LlmStream({
            let started = Arc::clone(&pull_started);
            Arc::new(move |_request| {
                started.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
            })
        }),
        runtime.handle().clone(),
        None,
        &stream_owner,
    ));
    let pull_next_ref = Arc::into_raw(pull_next) as *const NemoRelayNativeAsyncNext;
    let pull_invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
        .unwrap(),
    )
    .unwrap();
    let (pull_user_data, pull_probe) = CallbackProbe::new();
    assert_eq!(
        unsafe {
            native_async_next_open_llm_stream(
                pull_next_ref,
                pull_invocation,
                record_owned_pull_open_callback,
                pull_user_data,
            )
        },
        NemoRelayStatus::Ok
    );
    drop(NativeAsyncStreamReceiver {
        receiver: stream_receiver,
        stream: Arc::clone(&stream_owner),
    });

    runtime.block_on(tokio::task::yield_now());

    for probe in [&unary_probe, &pull_probe] {
        assert_eq!(probe.callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(probe.drops.load(Ordering::SeqCst), 1);
        let error = probe
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .expect("cancelled callback must include an error");
        assert!(error.contains("cancelled"), "{error}");
    }
    assert!(!unary_started.load(Ordering::SeqCst));
    assert!(!pull_started.load(Ordering::SeqCst));
    assert!(
        completion
            .continuation_aborts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
    );
    assert!(
        stream_owner
            .downstream_aborts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
    );

    unsafe {
        native_string_free(unary_invocation);
        native_string_free(pull_invocation);
        native_async_next_release(unary_next_ref);
        native_async_next_release(pull_next_ref);
    }
}

#[test]
fn native_async_stream_entrypoints_cover_closed_full_and_settled_channels() {
    let chunk = native_string("null");

    let no_sender = Arc::new(NativeAsyncStream {
        sender: Mutex::new(None),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let no_sender_ref = Arc::into_raw(no_sender) as *const NemoRelayNativeAsyncStream;
    assert_eq!(
        unsafe { native_async_stream_push_json(no_sender_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe { native_async_stream_finish(no_sender_ref) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe { native_async_stream_reject(no_sender_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_async_stream_release(no_sender_ref) };

    let (full_sender, _full_receiver) = tokio::sync::mpsc::channel::<FlowResult<Json>>(1);
    full_sender.try_send(Ok(Json::Null)).unwrap();
    let full = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(full_sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let full_ref = Arc::into_raw(full) as *const NemoRelayNativeAsyncStream;
    assert_eq!(
        unsafe { native_async_stream_push_json(full_ref, chunk) },
        NemoRelayStatus::Backpressured
    );
    assert!(unsafe { native_async_stream_is_backpressured(full_ref) });
    assert_eq!(
        unsafe { native_async_stream_reject(full_ref, chunk) },
        NemoRelayStatus::Backpressured
    );
    assert!(unsafe { native_async_stream_is_backpressured(full_ref) });
    unsafe { native_async_stream_release(full_ref) };

    let (closed_sender, closed_receiver) = tokio::sync::mpsc::channel::<FlowResult<Json>>(1);
    drop(closed_receiver);
    let closed = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(closed_sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let closed_ref = Arc::into_raw(closed) as *const NemoRelayNativeAsyncStream;
    assert_eq!(
        unsafe { native_async_stream_push_json(closed_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    assert!(!unsafe { native_async_stream_is_backpressured(closed_ref) });
    assert_eq!(
        unsafe { native_async_stream_reject(closed_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_async_stream_release(closed_ref) };

    let (settled_sender, _settled_receiver) = tokio::sync::mpsc::channel(1);
    let settled = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(settled_sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(true),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let settled_ref = Arc::into_raw(settled) as *const NemoRelayNativeAsyncStream;
    assert_eq!(
        unsafe { native_async_stream_push_json(settled_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe { native_async_stream_finish(settled_ref) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe { native_async_stream_reject(settled_ref, chunk) },
        NemoRelayStatus::InvalidArg
    );
    unsafe {
        native_async_stream_release(settled_ref);
        native_string_free(chunk);
    }
}

#[tokio::test]
async fn native_async_result_entrypoint_reports_provider_errors_and_panics() {
    for next_fn in [
        Arc::new(|_value| {
            Box::pin(async { Err(FlowError::Internal("provider failed".into())) })
                as Pin<Box<dyn Future<Output = FlowResult<ToolExecutionResult>> + Send>>
        }) as ToolExecutionNextFn,
        Arc::new(|_value| {
            Box::pin(async {
                panic!("provider panicked");
                #[allow(unreachable_code)]
                Ok(ToolExecutionResult::new(Json::Null))
            }) as Pin<Box<dyn Future<Output = FlowResult<ToolExecutionResult>> + Send>>
        }) as ToolExecutionNextFn,
    ] {
        let next = Arc::new(NativeAsyncNext::new(
            NativeAsyncNextInner::Tool(next_fn),
            tokio::runtime::Handle::current(),
            None,
        ));
        let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
        let invocation = native_string("null");
        let (sender, receiver) =
            tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
        assert_eq!(
            unsafe {
                native_async_next_invoke_result(
                    next_ref,
                    invocation,
                    complete_native_next_result,
                    Box::into_raw(Box::new(sender)).cast(),
                )
            },
            NemoRelayStatus::Ok
        );
        let error = receiver.await.unwrap().unwrap_err();
        assert!(error.contains("provider"));
        unsafe {
            native_string_free(invocation);
            native_async_next_release(next_ref);
        }
    }
}

#[tokio::test]
async fn native_async_stream_next_entrypoint_validates_handle_kind_and_request() {
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(stream) as *const NemoRelayNativeAsyncStream;
    let invocation = native_string("null");

    let tool_next = Arc::new(NativeAsyncNext::new(
        canonical_tool_next(Arc::new(|value| Box::pin(async move { Ok(value) }))),
        tokio::runtime::Handle::current(),
        None,
    ));
    let tool_ref = Arc::into_raw(tool_next) as *const NemoRelayNativeAsyncNext;
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                tool_ref,
                invocation,
                stream_ref,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::InvalidArg
    );

    let stream_next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
        })),
        tokio::runtime::Handle::current(),
        None,
    ));
    let next_ref = Arc::into_raw(stream_next) as *const NemoRelayNativeAsyncNext;
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                ptr::null(),
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::InvalidJson
    );

    unsafe {
        native_string_free(invocation);
        native_async_next_release(tool_ref);
        native_async_next_release(next_ref);
        native_async_stream_release(stream_ref);
    }
}

struct InvokeNativeNextThenReturnState {
    callback_state: u32,
    invoke_status: AtomicUsize,
    started: Mutex<std::sync::mpsc::Receiver<()>>,
}

unsafe extern "C" fn invoke_native_next_then_return_state(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    let state = unsafe { &*user_data.cast::<InvokeNativeNextThenReturnState>() };
    let status = unsafe { native_async_next_invoke(next, invocation_json, completion) };
    state
        .invoke_status
        .store(status as usize, Ordering::Release);
    if status == NemoRelayStatus::Ok {
        let _ = state
            .started
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .recv_timeout(Duration::from_secs(1));
    }
    unsafe { native_async_next_release(next) };
    state.callback_state
}

unsafe extern "C" fn invoke_native_stream_next_then_return_state(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    stream: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let state = unsafe { &*user_data.cast::<InvokeNativeNextThenReturnState>() };
    let request = read_native_string(invocation_json)
        .ok()
        .and_then(|invocation| serde_json::from_str::<Json>(&invocation).ok())
        .and_then(|invocation| invocation.get("request").cloned())
        .and_then(|request| native_string_from_json(&request));
    let status = if let Some(request) = request {
        let status = unsafe {
            native_async_next_invoke_stream(
                next,
                request,
                stream,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        };
        unsafe { native_string_free(request) };
        status
    } else {
        NemoRelayStatus::InvalidJson
    };
    state
        .invoke_status
        .store(status as usize, Ordering::Release);
    if status == NemoRelayStatus::Ok {
        let _ = state
            .started
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .recv_timeout(Duration::from_secs(1));
    }
    unsafe {
        native_async_next_release(next);
        native_async_stream_release(stream);
    }
    state.callback_state
}

unsafe extern "C" fn invoke_detached_next_and_finish_replacement_stream(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    stream: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let invoke_status = unsafe { &*user_data.cast::<AtomicUsize>() };
    let request = read_native_string(invocation_json)
        .ok()
        .and_then(|invocation| serde_json::from_str::<Json>(&invocation).ok())
        .and_then(|invocation| invocation.get("request").cloned())
        .and_then(|request| native_string_from_json(&request));
    let status = if let Some(request) = request {
        let status = unsafe {
            native_async_next_invoke_stream(
                next,
                request,
                stream,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        };
        unsafe { native_string_free(request) };
        status
    } else {
        NemoRelayStatus::InvalidJson
    };
    invoke_status.store(status as usize, Ordering::Release);
    let replacement = native_string_from_json(&json!({"source": "replacement"})).unwrap();
    unsafe {
        let _ = native_async_stream_push_json(stream, replacement);
        native_string_free(replacement);
        let _ = native_async_stream_finish(stream);
        native_async_next_release(next);
        native_async_stream_release(stream);
    }
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

struct FailingNativeCodec;

impl LlmCodec for FailingNativeCodec {
    fn decode(&self, _request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        Err(FlowError::Internal("request decode rejected".into()))
    }

    fn encode(
        &self,
        _annotated: &AnnotatedLlmRequest,
        _original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        Err(FlowError::Internal("request encode rejected".into()))
    }
}

impl LlmResponseCodec for FailingNativeCodec {
    fn decode_response(&self, _response: &Json) -> FlowResult<AnnotatedLlmResponse> {
        Err(FlowError::Internal("response decode rejected".into()))
    }
}

struct PanickingNativeCodec;

impl LlmCodec for PanickingNativeCodec {
    fn decode(&self, _request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        panic!("request decode panic")
    }

    fn encode(
        &self,
        _annotated: &AnnotatedLlmRequest,
        _original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        panic!("request encode panic")
    }
}

impl LlmResponseCodec for PanickingNativeCodec {
    fn decode_response(&self, _response: &Json) -> FlowResult<AnnotatedLlmResponse> {
        panic!("response decode panic")
    }
}

#[test]
fn native_string_and_json_helpers_cover_abi_boundaries() {
    assert_native_string_allocation_boundaries();
    assert_native_error_string_boundaries();
    assert_native_json_parsing_boundaries();
    assert_native_json_output_and_host_api();
}

fn assert_native_string_allocation_boundaries() {
    clear_native_last_error();
    assert_eq!(
        unsafe { native_string_new(ptr::null(), 0, ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_last_error_contains("out string pointer is null");

    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { native_string_new(ptr::null(), 1, &mut out) },
        NemoRelayStatus::NullPointer
    );
    assert!(out.is_null());
    assert_last_error_contains("string data pointer is null");

    let invalid_utf8 = [0xff];
    assert_eq!(
        unsafe { native_string_new(invalid_utf8.as_ptr(), invalid_utf8.len(), &mut out) },
        NemoRelayStatus::InvalidUtf8
    );
    assert!(out.is_null());
    assert_last_error_contains("not valid UTF-8");

    let text = native_string("hello");
    assert_eq!(unsafe { native_string_len(text) }, 5);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(native_string_data(text), 5) },
        b"hello"
    );
    assert!(unsafe { native_string_data(ptr::null()) }.is_null());
    assert_eq!(unsafe { native_string_len(ptr::null()) }, 0);
    assert_eq!(take_native_string(text).unwrap(), "hello");
    unsafe { native_string_free(ptr::null_mut()) };

    let empty = native_string("");
    assert_eq!(read_native_string(empty).unwrap(), "");
    unsafe { native_string_free(empty) };
    assert_eq!(read_native_string(ptr::null()).unwrap(), "");
}

fn assert_native_error_string_boundaries() {
    let bad = Box::into_raw(Box::new(NativeHostString(vec![0xff]))) as *mut NemoRelayNativeString;
    assert!(read_native_string(bad).is_err());
    assert_eq!(
        optional_json_from_native_string(bad, "bad json"),
        Err(NemoRelayStatus::InvalidUtf8)
    );
    unsafe { native_last_error_set(bad) };
    assert_last_error_contains("not valid UTF-8");
    unsafe { native_string_free(bad) };

    let message = native_string("explicit native error");
    unsafe { native_last_error_set(message) };
    assert_eq!(
        native_last_error_message().as_deref(),
        Some("explicit native error")
    );
    unsafe { native_string_free(message) };
    unsafe { native_last_error_clear() };
    assert!(native_last_error_message().is_none());

    set_native_last_error("specific fallback");
    assert!(
        json_from_native_string(ptr::null_mut(), "generic fallback")
            .unwrap_err()
            .to_string()
            .contains("specific fallback")
    );
    clear_native_last_error();
    assert!(
        json_from_native_string(ptr::null_mut(), "generic fallback")
            .unwrap_err()
            .to_string()
            .contains("generic fallback")
    );
}

fn assert_native_json_parsing_boundaries() {
    let invalid_json = native_string("{");
    assert!(
        take_json_from_native_string(invalid_json, "unused")
            .unwrap_err()
            .to_string()
            .contains("invalid JSON")
    );
    assert_eq!(
        optional_json_from_native_string(ptr::null(), "optional"),
        Ok(None)
    );
    let valid_json = native_string(r#"{"value":1}"#);
    assert_eq!(
        optional_json_from_native_string(valid_json, "optional").unwrap(),
        Some(json!({"value": 1}))
    );
    unsafe { native_string_free(valid_json) };
    let invalid_json = native_string("not-json");
    assert_eq!(
        optional_json_from_native_string(invalid_json, "optional"),
        Err(NemoRelayStatus::InvalidJson)
    );
    assert_last_error_contains("optional is not valid JSON");
    unsafe { native_string_free(invalid_json) };

    assert_eq!(
        parse_json_arg(ptr::null(), "null JSON").unwrap_err(),
        NemoRelayStatus::InvalidJson
    );
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(
        parse_llm_request_arg(request_json, "request").unwrap(),
        request
    );
    unsafe { native_string_free(request_json) };
    let wrong_shape = native_string(r#"{"headers":[]}"#);
    assert_eq!(
        parse_llm_request_arg(wrong_shape, "request").unwrap_err(),
        NemoRelayStatus::InvalidJson
    );
    assert_last_error_contains("was not an LLM request");
    unsafe { native_string_free(wrong_shape) };
}

fn assert_native_json_output_and_host_api() {
    assert_eq!(
        write_native_json(&json!({"ok": true}), ptr::null_mut()),
        NemoRelayStatus::NullPointer
    );
    let mut json_out = ptr::null_mut();
    assert_eq!(
        write_native_json(&json!({"ok": true}), &mut json_out),
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(json_out, "unused").unwrap(),
        json!({"ok": true})
    );

    let host_api = unsafe { &*native_host_api() };
    assert_eq!(host_api.abi_version, NEMO_RELAY_NATIVE_ABI_VERSION);
    assert_eq!(
        host_api.struct_size,
        std::mem::size_of::<NemoRelayNativeHostApiV4>()
    );
}

#[test]
fn native_async_next_abi_runs_tool_llm_and_stream_continuations() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases: Vec<(NativeAsyncNextInner, Json, Json)> = vec![
        (
            canonical_tool_next(Arc::new(|value| Box::pin(async move { Ok(value) }))),
            json!({"tool": true}),
            json!({"result": {"tool": true}, "pending_marks": []}),
        ),
        (
            NativeAsyncNextInner::Llm(Arc::new(|request| {
                Box::pin(async move { Ok(request.content) })
            })),
            serde_json::to_value(LlmRequest {
                headers: Map::new(),
                content: json!({"llm": true}),
            })
            .unwrap(),
            json!({"llm": true}),
        ),
    ];

    for (inner, invocation, expected) in cases {
        let next = Arc::new(NativeAsyncNext::new(inner, runtime.handle().clone(), None));
        let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let completion = Arc::new(NativeAsyncCompletion {
            sender: Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
            next_invoked: AtomicBool::new(false),
            next_abort: Mutex::new(None),
            continuation_aborts: Mutex::new(HashMap::new()),
            codec: None,
            before_settlement_lock: None,
            _callback_user_data: None,
        });
        let completion_ref =
            Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
        let invocation = native_string_from_json(&invocation).unwrap();
        assert_eq!(
            unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
            NemoRelayStatus::Ok
        );
        assert_eq!(runtime.block_on(receiver).unwrap().unwrap(), expected);
        unsafe {
            native_string_free(invocation);
            native_async_next_release(next_ref);
            native_async_completion_release(completion_ref);
        }
    }

    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![
                    Ok(json!({"chunk": 1})),
                    Ok(json!({"chunk": 2})),
                ])))
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, _receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"stream": true}),
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::InvalidArg
    );
    assert_last_error_contains("async_next_invoke_stream");
    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_async_next_reports_a_revoked_continuation_without_calling_the_provider() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let (lease, guard) = MiddlewareContinuationLease::capture();
    let next = Arc::new(NativeAsyncNext::new(
        canonical_tool_next({
            let provider_calls = provider_calls.clone();
            Arc::new(move |value| {
                let provider_calls = provider_calls.clone();
                let invocation = lease.begin();
                Box::pin(async move {
                    invocation?
                        .invoke(|| async move {
                            provider_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(value)
                        })
                        .await
                })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(&json!({"tool": true})).unwrap();

    drop(guard);
    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::Ok
    );
    let error = runtime
        .block_on(receiver)
        .expect("native completion should settle")
        .expect_err("revoked continuation should reject");
    assert!(
        error
            .to_string()
            .contains("execution continuation is no longer active")
    );
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_async_next_result_supports_repeated_concurrent_calls() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let next = Arc::new(NativeAsyncNext::new(
        canonical_tool_next({
            let provider_calls = provider_calls.clone();
            Arc::new(move |value| {
                provider_calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    Ok(json!({
                        "value": value,
                        "scope": crate::api::runtime::task_scope_top().uuid.to_string(),
                    }))
                })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let first = native_string_from_json(&json!({"branch": "first"})).unwrap();
    let second = native_string_from_json(&json!({"branch": "second"})).unwrap();
    let (first_tx, first_rx) = tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    let (second_tx, second_rx) =
        tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    let first_stack = create_scope_stack();
    let first_scope = first_stack
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .top()
        .uuid
        .to_string();
    let second_stack = create_scope_stack();
    let second_scope = second_stack
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .top()
        .uuid
        .to_string();

    assert_eq!(
        with_scope_stack(first_stack, || unsafe {
            native_async_next_invoke_result(
                next_ref,
                first,
                complete_native_next_result,
                Box::into_raw(Box::new(first_tx)).cast(),
            )
        }),
        NemoRelayStatus::Ok
    );
    assert_eq!(
        with_scope_stack(second_stack, || unsafe {
            native_async_next_invoke_result(
                next_ref,
                second,
                complete_native_next_result,
                Box::into_raw(Box::new(second_tx)).cast(),
            )
        }),
        NemoRelayStatus::Ok
    );
    let (first_result, second_result) =
        runtime.block_on(async { tokio::join!(first_rx, second_rx) });

    assert_eq!(
        first_result.unwrap().unwrap(),
        json!({
            "result": {
                "value": {"branch": "first"},
                "scope": first_scope,
            },
        })
    );
    assert_eq!(
        second_result.unwrap().unwrap(),
        json!({
            "result": {
                "value": {"branch": "second"},
                "scope": second_scope,
            },
        })
    );
    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);

    unsafe {
        native_string_free(first);
        native_string_free(second);
        native_async_next_release(next_ref);
    }
}

#[test]
fn native_v4_pull_stream_orders_chunks_ends_and_cancels_pending_pulls() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async {
                Ok(LlmJsonStream::new(tokio_stream::iter([
                    Ok(json!({"index": 1})),
                    Ok(json!({"index": 2})),
                ])))
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let request = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
        .unwrap(),
    )
    .unwrap();
    let (open_tx, open_rx) = tokio::sync::oneshot::channel::<PullOpenResult>();
    assert_eq!(
        unsafe {
            native_async_next_open_llm_stream(
                next_ref,
                request,
                complete_pull_stream_open,
                Box::into_raw(Box::new(open_tx)).cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    let stream = runtime.block_on(open_rx).unwrap().unwrap();

    let pull = |stream: usize| {
        let (tx, rx) = tokio::sync::oneshot::channel::<PullItemResult>();
        assert_eq!(
            unsafe {
                native_async_llm_stream_pull(
                    stream as *const NemoRelayNativeLlmAsyncStream,
                    complete_pull_stream_item,
                    Box::into_raw(Box::new(tx)).cast(),
                )
            },
            NemoRelayStatus::Ok
        );
        rx
    };
    assert_eq!(
        runtime.block_on(pull(stream)).unwrap().unwrap(),
        Some(json!({"index": 1}))
    );
    assert_eq!(
        runtime.block_on(pull(stream)).unwrap().unwrap(),
        Some(json!({"index": 2}))
    );
    assert_eq!(runtime.block_on(pull(stream)).unwrap().unwrap(), None);
    unsafe { native_async_llm_stream_release(stream as *const NemoRelayNativeLlmAsyncStream) };

    let pending_next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async {
                Ok(LlmJsonStream::new(futures_util::stream::pending::<
                    FlowResult<Json>,
                >()))
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let pending_next_ref = Arc::into_raw(pending_next) as *const NemoRelayNativeAsyncNext;
    let (open_tx, open_rx) = tokio::sync::oneshot::channel::<PullOpenResult>();
    assert_eq!(
        unsafe {
            native_async_next_open_llm_stream(
                pending_next_ref,
                request,
                complete_pull_stream_open,
                Box::into_raw(Box::new(open_tx)).cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    let pending_stream = runtime.block_on(open_rx).unwrap().unwrap();
    let pending_pull = pull(pending_stream);
    assert_eq!(
        unsafe {
            native_async_llm_stream_cancel(pending_stream as *const NemoRelayNativeLlmAsyncStream)
        },
        NemoRelayStatus::Ok
    );
    let error = runtime
        .block_on(pending_pull)
        .unwrap()
        .expect_err("cancelled pull should report an error");
    assert!(error.contains("cancelled"), "{error}");
    unsafe {
        native_async_llm_stream_release(pending_stream as *const NemoRelayNativeLlmAsyncStream);
        native_string_free(request);
        native_async_next_release(next_ref);
        native_async_next_release(pending_next_ref);
    }
}

#[test]
fn owned_native_result_continuation_is_aborted_when_completion_is_cancelled() {
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(completion_tx)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let wait = NativeAsyncWait {
        completion: Arc::clone(&completion),
        receiver: completion_rx,
        completed: false,
    };
    let started = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let next = Arc::new(NativeAsyncNext::with_completion_owner(
        canonical_tool_next({
            let started = Arc::clone(&started);
            let dropped = Arc::clone(&dropped);
            Arc::new(move |_value| {
                let started = Arc::clone(&started);
                let probe = DropProbe(Arc::clone(&dropped));
                Box::pin(async move {
                    started.store(true, Ordering::Release);
                    let _probe = probe;
                    std::future::pending::<FlowResult<Json>>().await
                })
            })
        }),
        runtime.handle().clone(),
        None,
        &completion,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let invocation = native_string_from_json(&json!({"pending": true})).unwrap();
    let (result_tx, result_rx) =
        tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    assert_eq!(
        unsafe {
            native_async_next_invoke_result(
                next_ref,
                invocation,
                complete_native_next_result,
                Box::into_raw(Box::new(result_tx)).cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    runtime.block_on(async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    });
    drop(wait);
    let result = runtime
        .block_on(result_rx)
        .unwrap()
        .expect_err("cancelled continuation should reject its result callback");
    assert!(result.contains("cancelled"), "{result}");
    assert!(dropped.load(Ordering::Acquire));
    assert!(completion.cancelled.load(Ordering::Acquire));

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
    }
}

#[test]
fn native_async_next_result_uses_captured_scope_on_an_unbound_plugin_thread() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let captured_stack = create_scope_stack();
    let captured_scope = captured_stack
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .top()
        .uuid
        .to_string();
    let next = with_scope_stack(captured_stack, || {
        Arc::new(NativeAsyncNext::new(
            canonical_tool_next(Arc::new(|value| {
                Box::pin(async move {
                    Ok(json!({
                        "value": value,
                        "scope": crate::api::runtime::task_scope_top().uuid.to_string(),
                    }))
                })
            })),
            runtime.handle().clone(),
            None,
        ))
    });
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let invocation = native_string_from_json(&json!({"thread": "plugin"})).unwrap();
    let (sender, receiver) = tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    let next_address = next_ref as usize;
    let invocation_address = invocation as usize;
    let sender_address = Box::into_raw(Box::new(sender)) as usize;

    assert_eq!(
        std::thread::spawn(move || unsafe {
            native_async_next_invoke_result(
                next_address as *const NemoRelayNativeAsyncNext,
                invocation_address as *const NemoRelayNativeString,
                complete_native_next_result,
                sender_address as *mut c_void,
            )
        })
        .join()
        .unwrap(),
        NemoRelayStatus::Ok
    );
    assert_eq!(
        runtime.block_on(receiver).unwrap().unwrap(),
        json!({
            "result": {
                "value": {"thread": "plugin"},
                "scope": captured_scope,
            },
        })
    );

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
    }
}

fn native_continuation_context_observation(
    expected_stack: &ScopeStackHandle,
    expected_event_uuid: uuid::Uuid,
) -> Json {
    let expected_scope_uuid = expected_stack
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .top()
        .uuid;
    let visible_scope_uuid = current_scope_stack()
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .top()
        .uuid;
    json!({
        "scope_stack": visible_scope_uuid == expected_scope_uuid,
        "active_event_uuid": active_event_uuid() == Some(expected_event_uuid),
        "publication_context": crate::api::runtime::subscriber_dispatcher::publication_context::<String>()
            .is_some_and(|context| context.as_str() == "native-continuation"),
        "publication_buffer": capture_nested_publication_buffer().is_some(),
        "optimization_recorder": current_llm_optimization_recorder().is_some(),
    })
}

#[test]
fn native_async_next_preserves_runtime_context_for_unary_and_stream_continuations() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let expected_stack = create_scope_stack();
    let expected_event_uuid = uuid::Uuid::now_v7();
    let expected = json!({
        "scope_stack": true,
        "active_event_uuid": true,
        "publication_context": true,
        "publication_buffer": true,
        "optimization_recorder": true,
    });

    runtime.block_on(TASK_SCOPE_STACK.scope(
        expected_stack.clone(),
        with_task_publication_context(
            Some(Arc::new(String::from("native-continuation"))),
            scope_llm_optimization_recorder(
                LlmOptimizationRecorder::default(),
                with_active_event_uuid(
                    expected_event_uuid,
                    crate::api::runtime::subscriber_dispatcher::with_async_publication_context(
                        crate::api::runtime::subscriber_dispatcher::register_async_publication(),
                        async {
                            let unary_stack = expected_stack.clone();
                            let unary = Arc::new(NativeAsyncNext::new(
                                canonical_tool_next(Arc::new(move |_value| {
                                    let unary_stack = unary_stack.clone();
                                    Box::pin(async move {
                                        Ok(native_continuation_context_observation(
                                            &unary_stack,
                                            expected_event_uuid,
                                        ))
                                    })
                                })),
                                runtime.handle().clone(),
                                None,
                            ));
                            let unary_ref = Arc::into_raw(unary) as *const NemoRelayNativeAsyncNext;
                            let (sender, receiver) = tokio::sync::oneshot::channel();
                            let completion = Arc::new(NativeAsyncCompletion {
                                sender: Mutex::new(Some(sender)),
                                cancelled: AtomicBool::new(false),
                                next_invoked: AtomicBool::new(false),
                                next_abort: Mutex::new(None),
                                continuation_aborts: Mutex::new(HashMap::new()),
                                codec: None,
                                before_settlement_lock: None,
                                _callback_user_data: None,
                            });
                            let completion_ref = Arc::into_raw(Arc::clone(&completion))
                                as *const NemoRelayNativeAsyncCompletion;
                            let invocation = native_string_from_json(&Json::Null).unwrap();
                            assert_eq!(
                                unsafe {
                                    native_async_next_invoke(unary_ref, invocation, completion_ref)
                                },
                                NemoRelayStatus::Ok
                            );
                            assert_eq!(
                                receiver.await.unwrap().unwrap(),
                                json!({"result": expected.clone(), "pending_marks": []})
                            );
                            unsafe {
                                native_string_free(invocation);
                                native_async_next_release(unary_ref);
                                native_async_completion_release(completion_ref);
                            }

                            let stream_stack = expected_stack.clone();
                            let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
                            let observed_tx = Arc::new(Mutex::new(Some(observed_tx)));
                            let stream_next = Arc::new(NativeAsyncNext::new(
                                NativeAsyncNextInner::LlmStream(Arc::new(move |_request| {
                                    let stream_stack = stream_stack.clone();
                                    let observed_tx = observed_tx.clone();
                                    Box::pin(async move {
                                        if let Some(sender) = observed_tx
                                            .lock()
                                            .unwrap_or_else(|error| error.into_inner())
                                            .take()
                                        {
                                            let _ = sender.send(
                                                native_continuation_context_observation(
                                                    &stream_stack,
                                                    expected_event_uuid,
                                                ),
                                            );
                                        }
                                        Ok(LlmJsonStream::new(tokio_stream::empty()))
                                    })
                                })),
                                runtime.handle().clone(),
                                None,
                            ));
                            let stream_next_ref =
                                Arc::into_raw(stream_next) as *const NemoRelayNativeAsyncNext;
                            let (sender, receiver) = tokio::sync::mpsc::channel(1);
                            let stream = Arc::new(NativeAsyncStream {
                                sender: Mutex::new(Some(sender)),
                                cancelled: AtomicBool::new(false),
                                settled: AtomicBool::new(false),
                                backpressured: AtomicBool::new(false),
                                downstream_aborts: Mutex::new(HashMap::new()),
                                settlement: Mutex::new(()),
                                before_settlement_lock: None,
                                _callback_user_data: None,
                            });
                            let stream_ref = Arc::into_raw(Arc::clone(&stream))
                                as *const NemoRelayNativeAsyncStream;
                            let invocation = native_string_from_json(
                                &serde_json::to_value(LlmRequest {
                                    headers: Map::new(),
                                    content: Json::Null,
                                })
                                .unwrap(),
                            )
                            .unwrap();
                            assert_eq!(
                                unsafe {
                                    native_async_next_invoke_stream(
                                        stream_next_ref,
                                        invocation,
                                        stream_ref,
                                        accept_native_stream_item,
                                        ptr::null_mut(),
                                    )
                                },
                                NemoRelayStatus::Ok
                            );
                            assert_eq!(observed_rx.await.unwrap(), expected);
                            drop(NativeAsyncStreamReceiver {
                                receiver,
                                stream: Arc::clone(&stream),
                            });
                            unsafe {
                                native_string_free(invocation);
                                native_async_next_release(stream_next_ref);
                                native_async_stream_release(stream_ref);
                            }
                        },
                    ),
                ),
            ),
        ),
    ));
}

#[test]
fn native_sync_next_preserves_runtime_context_for_unary_and_stream_continuations() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let expected_stack = create_scope_stack();
    let expected_event_uuid = uuid::Uuid::now_v7();
    let expected = json!({
        "scope_stack": true,
        "active_event_uuid": true,
        "publication_context": true,
        "publication_buffer": true,
        "optimization_recorder": true,
    });

    runtime.block_on(TASK_SCOPE_STACK.scope(
        expected_stack.clone(),
        with_task_publication_context(
            Some(Arc::new(String::from("native-continuation"))),
            scope_llm_optimization_recorder(
                LlmOptimizationRecorder::default(),
                with_active_event_uuid(
                    expected_event_uuid,
                    crate::api::runtime::subscriber_dispatcher::with_async_publication_context(
                        crate::api::runtime::subscriber_dispatcher::register_async_publication(),
                        async {
                            let unary_stack = expected_stack.clone();
                            let unary_next: ToolExecutionNextFn = Arc::new(move |_value| {
                                let unary_stack = unary_stack.clone();
                                Box::pin(async move {
                                    Ok(ToolExecutionResult::new(
                                        native_continuation_context_observation(
                                            &unary_stack,
                                            expected_event_uuid,
                                        ),
                                    ))
                                })
                            });
                            let unary_next = unary_next;
                            let invocation = native_string_from_json(&Json::Null).unwrap();
                            let mut output = ptr::null_mut();
                            assert_eq!(
                                unsafe {
                                    native_tool_next(
                                        invocation,
                                        (&unary_next as *const ToolExecutionNextFn)
                                            .cast_mut()
                                            .cast(),
                                        &mut output,
                                    )
                                },
                                NemoRelayStatus::Ok
                            );
                            let observed: Json =
                                serde_json::from_str(&read_native_string(output).unwrap()).unwrap();
                            assert_eq!(observed, json!({"result": expected}));
                            unsafe {
                                native_string_free(invocation);
                                native_string_free(output);
                            }

                            let stream_stack = expected_stack.clone();
                            let stream_next: LlmStreamExecutionNextFn = Arc::new(move |_request| {
                                let stream_stack = stream_stack.clone();
                                Box::pin(async move {
                                    Ok(LlmJsonStream::new(futures_util::stream::once(async move {
                                        Ok(native_continuation_context_observation(
                                            &stream_stack,
                                            expected_event_uuid,
                                        ))
                                    })))
                                })
                            });
                            let request = native_string_from_json(
                                &serde_json::to_value(LlmRequest {
                                    headers: Map::new(),
                                    content: Json::Null,
                                })
                                .unwrap(),
                            )
                            .unwrap();
                            let mut native_stream = NemoRelayNativeLlmStreamV1::default();
                            assert_eq!(
                                unsafe {
                                    native_llm_stream_next(
                                        request,
                                        (&stream_next as *const LlmStreamExecutionNextFn)
                                            .cast_mut()
                                            .cast(),
                                        &mut native_stream,
                                    )
                                },
                                NemoRelayStatus::Ok
                            );
                            unsafe { native_string_free(request) };

                            let mut output = ptr::null_mut();
                            assert_eq!(
                                unsafe {
                                    native_stream.next.unwrap()(
                                        native_stream.user_data,
                                        &mut output,
                                    )
                                },
                                NemoRelayStatus::Ok
                            );
                            let observed: Json =
                                serde_json::from_str(&read_native_string(output).unwrap()).unwrap();
                            assert_eq!(observed, expected);
                            unsafe { native_string_free(output) };
                            drop_native_stream(native_stream);
                        },
                    ),
                ),
            ),
        ),
    ));
}

#[test]
fn native_async_next_panics_settle_unary_and_stream_errors() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let next = Arc::new(NativeAsyncNext::new(
        canonical_tool_next(Arc::new(|_value| {
            Box::pin(async move {
                panic!("native unary next panic");
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(&Json::Null).unwrap();
    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::Ok
    );
    let error = runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(1), receiver).await })
        .expect("panicking unary next should settle")
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("native unary next panic"));
    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }

    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(futures_util::stream::once(async move {
                    panic!("native stream next panic");
                    #[allow(unreachable_code)]
                    Ok(Json::Null)
                })))
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let output_stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let output_stream_ref =
        Arc::into_raw(Arc::clone(&output_stream)) as *const NemoRelayNativeAsyncStream;
    let callback_state = Arc::new(NativeStreamCallbackState::default());
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                output_stream_ref,
                record_native_stream_result,
                Arc::as_ptr(&callback_state).cast_mut().cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    runtime
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(1), callback_state.notified.notified()).await
        })
        .expect("panicking stream next should report an error");
    assert!(!callback_state.done.load(Ordering::Acquire));
    assert!(
        callback_state
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_deref()
            .is_some_and(|error| error.contains("native stream next panic"))
    );
    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_stream_release(output_stream_ref);
    }
}

#[test]
fn native_async_next_is_permanently_one_shot() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let next = Arc::new(NativeAsyncNext::new(
        canonical_tool_next({
            let calls = Arc::clone(&calls);
            Arc::new(move |value| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(value)
                })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(&json!({"value": 1})).unwrap();

    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::Ok
    );
    runtime.block_on(receiver).unwrap().unwrap();
    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::InvalidArg
    );
    runtime.block_on(tokio::task::yield_now());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn cancelled_native_async_next_does_not_start_unary_or_stream_continuations() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let unary_started = Arc::new(AtomicBool::new(false));
    let unary = Arc::new(NativeAsyncNext::new(
        canonical_tool_next({
            let unary_started = unary_started.clone();
            Arc::new(move |value| {
                unary_started.store(true, Ordering::SeqCst);
                Box::pin(async move { Ok(value) })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let unary_ref = Arc::into_raw(unary) as *const NemoRelayNativeAsyncNext;
    let (sender, _receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(true),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let unary_invocation = native_string_from_json(&Json::Null).unwrap();
    assert_eq!(
        unsafe { native_async_next_invoke(unary_ref, unary_invocation, completion_ref) },
        NemoRelayStatus::InvalidArg
    );

    let stream_started = Arc::new(AtomicBool::new(false));
    let stream_next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream({
            let stream_started = stream_started.clone();
            Arc::new(move |_request| {
                stream_started.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let stream_next_ref = Arc::into_raw(stream_next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(true),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let stream_invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                stream_next_ref,
                stream_invocation,
                stream_ref,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    runtime.block_on(tokio::task::yield_now());
    assert!(!unary_started.load(Ordering::SeqCst));
    assert!(!stream_started.load(Ordering::SeqCst));

    drop(NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    });
    unsafe {
        native_string_free(unary_invocation);
        native_string_free(stream_invocation);
        native_async_next_release(unary_ref);
        native_async_next_release(stream_next_ref);
        native_async_completion_release(completion_ref);
        native_async_stream_release(stream_ref);
    }
}

#[test]
fn malformed_llm_next_does_not_consume_the_completion() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::Llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, _receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(&json!({"not": "an llm request"})).unwrap();

    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::InvalidJson
    );
    assert!(
        completion
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
    );
    assert!(!completion.next_invoked.load(Ordering::SeqCst));

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_async_stream_next_supports_repeated_concurrent_calls() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"stream": true}),
        })
        .unwrap(),
    )
    .unwrap();

    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                accept_native_stream_item,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::Ok
    );

    runtime
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if stream
                        .downstream_aborts
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .is_empty()
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
        })
        .expect("completed native stream continuations should prune their abort handles");
    assert!(
        stream
            .downstream_aborts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    );

    drop(NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    });
    runtime.block_on(tokio::task::yield_now());
    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_stream_release(stream_ref);
    }
}

#[test]
fn native_async_stream_settlement_rejects_late_next_and_aborts_in_flight_next() {
    for reject in [false, true] {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let next = Arc::new(NativeAsyncNext::new(
            NativeAsyncNextInner::LlmStream(Arc::new({
                let provider_calls = Arc::clone(&provider_calls);
                move |_request| {
                    let provider_calls = Arc::clone(&provider_calls);
                    let started_tx = Arc::clone(&started_tx);
                    Box::pin(async move {
                        provider_calls.fetch_add(1, Ordering::SeqCst);
                        if let Some(started_tx) = started_tx
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .take()
                        {
                            let _ = started_tx.send(());
                        }
                        std::future::pending().await
                    })
                }
            })),
            runtime.handle().clone(),
            None,
        ));
        let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let stream = Arc::new(NativeAsyncStream {
            sender: Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
            settled: AtomicBool::new(false),
            backpressured: AtomicBool::new(false),
            downstream_aborts: Mutex::new(HashMap::new()),
            settlement: Mutex::new(()),
            before_settlement_lock: None,
            _callback_user_data: None,
        });
        let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
        let invocation = native_string_from_json(
            &serde_json::to_value(LlmRequest {
                headers: Map::new(),
                content: json!({"stream": true}),
            })
            .unwrap(),
        )
        .unwrap();
        let callback_state = NativeStreamCallbackState::default();

        assert_eq!(
            unsafe {
                native_async_next_invoke_stream(
                    next_ref,
                    invocation,
                    stream_ref,
                    record_native_stream_result,
                    (&callback_state as *const NativeStreamCallbackState)
                        .cast_mut()
                        .cast(),
                )
            },
            NemoRelayStatus::Ok
        );
        runtime.block_on(started_rx).unwrap();

        if reject {
            let message = native_string("replacement rejected");
            assert_eq!(
                unsafe { native_async_stream_reject(stream_ref, message) },
                NemoRelayStatus::Ok
            );
            unsafe { native_string_free(message) };
        } else {
            assert_eq!(
                unsafe { native_async_stream_finish(stream_ref) },
                NemoRelayStatus::Ok
            );
        }

        runtime
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(1), callback_state.notified.notified())
                    .await
            })
            .expect("settlement should terminate the in-flight continuation callback");
        assert_eq!(callback_state.callbacks.load(Ordering::SeqCst), 1);
        assert!(
            callback_state
                .error
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_deref()
                .is_some_and(|error| error.contains("settled"))
        );
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert!(
            stream
                .downstream_aborts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
        );
        assert_eq!(
            unsafe {
                native_async_next_invoke_stream(
                    next_ref,
                    invocation,
                    stream_ref,
                    accept_native_stream_item,
                    ptr::null_mut(),
                )
            },
            NemoRelayStatus::InvalidArg
        );
        runtime.block_on(tokio::task::yield_now());
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);

        drop(NativeAsyncStreamReceiver {
            receiver,
            stream: Arc::clone(&stream),
        });
        unsafe {
            native_string_free(invocation);
            native_async_next_release(next_ref);
            native_async_stream_release(stream_ref);
        }
    }
}

#[test]
fn native_async_stream_next_stops_callbacks_after_false() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(|_request| {
            Box::pin(async {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![
                    Ok(json!({"chunk": 1})),
                    Ok(json!({"chunk": 2})),
                ])))
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"stream": true}),
        })
        .unwrap(),
    )
    .unwrap();
    let callbacks = AtomicUsize::new(0);

    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                stop_after_first_native_stream_item,
                (&callbacks as *const AtomicUsize).cast_mut().cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    runtime.block_on(async {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    });
    assert_eq!(callbacks.load(Ordering::SeqCst), 1);

    drop(NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    });
    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_stream_release(stream_ref);
    }
}

#[test]
fn native_async_stream_in_flight_cancellation_releases_callback_state() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(move |_request| {
            let started_tx = Arc::clone(&started_tx);
            Box::pin(async move {
                if let Some(started_tx) = started_tx.lock().unwrap().take() {
                    let _ = started_tx.send(());
                }
                std::future::pending().await
            })
        })),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"stream": true}),
        })
        .unwrap(),
    )
    .unwrap();
    let callback_state = Arc::new(NativeStreamCallbackState::default());
    let drop_count = Arc::new(AtomicUsize::new(0));
    let callback_user_data = Box::into_raw(Box::new(OwnedNativeStreamCallbackState {
        result: Arc::clone(&callback_state),
        drop_count: Arc::clone(&drop_count),
        stream: stream_ref,
    }));

    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                record_and_release_native_stream_result,
                callback_user_data.cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    runtime.block_on(started_rx).unwrap();
    drop(NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    });
    runtime
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(1), callback_state.notified.notified()).await
        })
        .expect("in-flight cancellation should deliver a terminal callback");
    runtime.block_on(tokio::task::yield_now());
    assert_eq!(callback_state.callbacks.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    assert!(!callback_state.done.load(Ordering::Acquire));
    assert!(
        callback_state
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_deref()
            .is_some_and(|error| error.contains("cancelled"))
    );
    wait_for_native_reaper(&stream, 1);

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
    }
}

#[test]
fn native_async_stream_cancellation_before_first_poll_releases_callback_state() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::LlmStream(Arc::new(move |_request| Box::pin(std::future::pending()))),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"stream": true}),
        })
        .unwrap(),
    )
    .unwrap();
    let callback_state = Arc::new(NativeStreamCallbackState::default());
    let drop_count = Arc::new(AtomicUsize::new(0));
    let callback_user_data = Box::into_raw(Box::new(OwnedNativeStreamCallbackState {
        result: Arc::clone(&callback_state),
        drop_count: Arc::clone(&drop_count),
        stream: stream_ref,
    }));

    assert_eq!(
        unsafe {
            native_async_next_invoke_stream(
                next_ref,
                invocation,
                stream_ref,
                record_and_release_native_stream_result,
                callback_user_data.cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    // The current-thread runtime has not been driven, so cancellation happens
    // before the spawned continuation can be polled for the first time.
    drop(NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    });
    runtime
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(1), callback_state.notified.notified()).await
        })
        .expect("pre-poll cancellation should deliver a terminal callback");
    runtime.block_on(tokio::task::yield_now());
    assert_eq!(callback_state.callbacks.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    assert!(!callback_state.done.load(Ordering::Acquire));
    assert!(
        callback_state
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_deref()
            .is_some_and(|error| error.contains("cancelled"))
    );
    wait_for_native_reaper(&stream, 1);

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
    }
}

#[test]
fn native_async_completion_abi_rejects_invalid_duplicate_and_cancelled_settlement() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invalid = native_string("not-json");
    assert_eq!(
        unsafe { native_async_completion_resolve_json(completion_ref, invalid) },
        NemoRelayStatus::InvalidJson
    );
    let value = native_string(r#"{"ok":true}"#);
    assert_eq!(
        unsafe { native_async_completion_resolve_json(completion_ref, value) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { native_async_completion_resolve_json(completion_ref, value) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        runtime.block_on(receiver).unwrap().unwrap(),
        json!({"ok": true})
    );
    unsafe {
        native_string_free(invalid);
        native_string_free(value);
        native_async_completion_release(completion_ref);
    }

    let (sender, _receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(true),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    assert!(unsafe { native_async_completion_is_cancelled(completion_ref) });
    assert!(unsafe { native_async_completion_is_cancelled(ptr::null()) });
    assert_eq!(
        unsafe { native_async_completion_reject(completion_ref, ptr::null()) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_async_completion_release(completion_ref) };
}

#[test]
fn completed_native_async_wait_is_not_marked_cancelled() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let value = native_string(r#"{"ok":true}"#);
    let mut wait = NativeAsyncWait {
        completion: Arc::clone(&completion),
        receiver,
        completed: false,
    };

    assert_eq!(
        unsafe { native_async_completion_resolve_json(completion_ref, value) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        runtime.block_on(wait.receive()).unwrap(),
        json!({"ok": true})
    );
    drop(wait);
    assert!(!completion.cancelled.load(Ordering::Acquire));
    assert!(!unsafe { native_async_completion_is_cancelled(completion_ref) });

    unsafe {
        native_string_free(value);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_async_completion_cancellation_wins_resolve_and_reject_settlement_races() {
    type SettleFn = unsafe extern "C" fn(
        *const NemoRelayNativeAsyncCompletion,
        *const NemoRelayNativeString,
    ) -> NemoRelayStatus;

    for (settle, argument) in [
        (
            native_async_completion_resolve_json as SettleFn,
            native_string(r#"{"ok":true}"#),
        ),
        (
            native_async_completion_reject as SettleFn,
            native_string("cancelled"),
        ),
    ] {
        let (sender, _receiver) = tokio::sync::oneshot::channel();
        let settlement_checkpoint = Arc::new(std::sync::Barrier::new(2));
        let completion = Arc::new(NativeAsyncCompletion {
            sender: Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
            next_invoked: AtomicBool::new(false),
            next_abort: Mutex::new(None),
            continuation_aborts: Mutex::new(HashMap::new()),
            codec: None,
            before_settlement_lock: Some(Arc::clone(&settlement_checkpoint)),
            _callback_user_data: None,
        });
        let completion_ref =
            Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
        let cancellation_guard = completion
            .next_abort
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let completion_address = completion_ref as usize;
        let argument_address = argument as usize;
        let settlement = std::thread::spawn(move || unsafe {
            settle(
                completion_address as *const NemoRelayNativeAsyncCompletion,
                argument_address as *const NemoRelayNativeString,
            )
        });

        // The settlement thread has parsed its argument and reached the exact
        // boundary before acquiring next_abort. The held guard now forces it
        // to observe cancellation after the lock is released.
        settlement_checkpoint.wait();
        completion.cancelled.store(true, Ordering::Release);
        drop(cancellation_guard);

        assert_eq!(settlement.join().unwrap(), NemoRelayStatus::InvalidArg);
        assert!(
            completion
                .sender
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some()
        );
        unsafe {
            native_string_free(argument);
            native_async_completion_release(completion_ref);
        }
    }
}

#[test]
fn cancelling_completion_aborts_pending_native_next() {
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let next = Arc::new(NativeAsyncNext::new(
        NativeAsyncNextInner::Llm({
            let started = Arc::clone(&started);
            let dropped = Arc::clone(&dropped);
            Arc::new(move |_request| {
                let started = Arc::clone(&started);
                let probe = DropProbe(Arc::clone(&dropped));
                Box::pin(async move {
                    started.store(true, Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    drop(probe);
                    unreachable!("pending continuation only exits when aborted")
                })
            })
        }),
        runtime.handle().clone(),
        None,
    ));
    let next_ref = Arc::into_raw(next) as *const NemoRelayNativeAsyncNext;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: None,
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let invocation = native_string_from_json(
        &serde_json::to_value(LlmRequest {
            headers: Map::new(),
            content: json!({"pending": true}),
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        unsafe { native_async_next_invoke(next_ref, invocation, completion_ref) },
        NemoRelayStatus::Ok
    );
    runtime.block_on(tokio::task::yield_now());
    assert!(started.load(Ordering::SeqCst));

    drop(NativeAsyncWait {
        completion: Arc::clone(&completion),
        receiver,
        completed: false,
    });
    runtime.block_on(tokio::task::yield_now());
    assert!(completion.cancelled.load(Ordering::SeqCst));
    assert!(dropped.load(Ordering::SeqCst));
    assert!(
        completion
            .next_abort
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_none()
    );

    unsafe {
        native_string_free(invocation);
        native_async_next_release(next_ref);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_async_callback_contract_errors_abort_an_invoked_next() {
    struct DropSignal(std::sync::mpsc::Sender<()>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    for (callback_state, expected_error) in [
        (
            NemoRelayNativeAsyncCallbackState::Complete as u32,
            "returned Complete without settling",
        ),
        (99, "returned an invalid state"),
    ] {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (dropped_tx, dropped_rx) = std::sync::mpsc::channel();
        let state = InvokeNativeNextThenReturnState {
            callback_state,
            invoke_status: AtomicUsize::new(NemoRelayStatus::Internal as usize),
            started: Mutex::new(started_rx),
        };
        let user_data = Arc::new(NativeCallbackUserData {
            ptr: (&state as *const InvokeNativeNextThenReturnState)
                .cast_mut()
                .cast(),
            free_fn: None,
            _instance: None,
        });

        let error = runtime
            .block_on(invoke_native_async_callback(
                invoke_native_next_then_return_state,
                user_data,
                json!({}),
                Some(canonical_tool_next(Arc::new({
                    let started_tx = started_tx.clone();
                    let dropped_tx = dropped_tx.clone();
                    move |_value| {
                        let started_tx = started_tx.clone();
                        let guard = DropSignal(dropped_tx.clone());
                        Box::pin(async move {
                            let _guard = guard;
                            let _ = started_tx.send(());
                            std::future::pending::<FlowResult<Json>>().await
                        })
                    }
                }))),
                None,
            ))
            .unwrap_err();

        assert!(error.to_string().contains(expected_error));
        assert_eq!(
            state.invoke_status.load(Ordering::Acquire),
            NemoRelayStatus::Ok as usize
        );
        dropped_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("contract error should abort and drop the pending native next");
    }
}

#[test]
fn native_async_stream_contract_errors_abort_an_invoked_next() {
    struct DropSignal(std::sync::mpsc::Sender<()>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    for (callback_state, expected_error) in [
        (
            NemoRelayNativeAsyncCallbackState::Complete as u32,
            "returned Complete without finishing",
        ),
        (99, "panicked or returned an invalid state"),
    ] {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (dropped_tx, dropped_rx) = std::sync::mpsc::channel();
        let state = InvokeNativeNextThenReturnState {
            callback_state,
            invoke_status: AtomicUsize::new(NemoRelayStatus::Internal as usize),
            started: Mutex::new(started_rx),
        };
        let user_data = Arc::new(NativeCallbackUserData {
            ptr: (&state as *const InvokeNativeNextThenReturnState)
                .cast_mut()
                .cast(),
            free_fn: None,
            _instance: None,
        });
        let wrapped = wrap_native_incremental_llm_stream_execution_with_user_data(
            invoke_native_stream_next_then_return_state,
            user_data,
        );
        let next: LlmStreamExecutionNextFn = Arc::new({
            let started_tx = started_tx.clone();
            let dropped_tx = dropped_tx.clone();
            move |_request| {
                let started_tx = started_tx.clone();
                let guard = DropSignal(dropped_tx.clone());
                Box::pin(async move {
                    let _guard = guard;
                    let _ = started_tx.send(());
                    std::future::pending::<FlowResult<LlmJsonStream>>().await
                })
            }
        });

        let result = runtime.block_on(wrapped(
            "contract-error",
            LlmRequest {
                headers: Map::new(),
                content: Json::Null,
            },
            next,
        ));
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("contract error unexpectedly returned a stream"),
        };

        assert!(error.to_string().contains(expected_error));
        assert_eq!(
            state.invoke_status.load(Ordering::Acquire),
            NemoRelayStatus::Ok as usize
        );
        dropped_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("contract error should abort and drop the pending native stream next");
    }
}

#[test]
fn native_replacement_stream_does_not_wait_for_a_detached_pending_next() {
    struct DropSignal(std::sync::mpsc::Sender<()>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (dropped_tx, dropped_rx) = std::sync::mpsc::channel();
    let invoke_status = AtomicUsize::new(NemoRelayStatus::Internal as usize);
    let user_data = Arc::new(NativeCallbackUserData {
        ptr: (&invoke_status as *const AtomicUsize).cast_mut().cast(),
        free_fn: None,
        _instance: None,
    });
    let wrapped = wrap_native_incremental_llm_stream_execution_with_user_data(
        invoke_detached_next_and_finish_replacement_stream,
        user_data,
    );
    let next: LlmStreamExecutionNextFn = Arc::new(move |_request| {
        let guard = DropSignal(dropped_tx.clone());
        Box::pin(async move {
            let _guard = guard;
            std::future::pending::<FlowResult<LlmJsonStream>>().await
        })
    });

    let mut stream = runtime
        .block_on(wrapped(
            "detached-next",
            LlmRequest {
                headers: Map::new(),
                content: Json::Null,
            },
            next,
        ))
        .expect("replacement stream must not wait for detached downstream construction");
    assert_eq!(
        runtime.block_on(stream.next()).unwrap().unwrap(),
        json!({"source": "replacement"})
    );
    assert!(runtime.block_on(stream.next()).is_none());
    drop(stream);

    assert_eq!(
        invoke_status.load(Ordering::Acquire),
        NemoRelayStatus::Ok as usize
    );
    match dropped_rx.recv_timeout(Duration::from_secs(1)) {
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("closing replacement output must cancel detached downstream next");
        }
    }
}

#[test]
fn native_async_stream_settlement_cannot_succeed_after_cancellation() {
    #[derive(Clone, Copy)]
    enum Settlement {
        Push(usize),
        Finish,
        Reject(usize),
    }

    for settlement in [
        Settlement::Push(native_string(r#"{"chunk":1}"#) as usize),
        Settlement::Finish,
        Settlement::Reject(native_string("cancelled") as usize),
    ] {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let settlement_checkpoint = Arc::new(std::sync::Barrier::new(2));
        let stream = Arc::new(NativeAsyncStream {
            sender: Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
            settled: AtomicBool::new(false),
            backpressured: AtomicBool::new(false),
            downstream_aborts: Mutex::new(HashMap::new()),
            settlement: Mutex::new(()),
            before_settlement_lock: Some(Arc::clone(&settlement_checkpoint)),
            _callback_user_data: None,
        });
        let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
        let cancellation_guard = stream
            .settlement
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let stream_address = stream_ref as usize;
        let operation = std::thread::spawn(move || unsafe {
            match settlement {
                Settlement::Push(chunk) => native_async_stream_push_json(
                    stream_address as *const NemoRelayNativeAsyncStream,
                    chunk as *const NemoRelayNativeString,
                ),
                Settlement::Finish => {
                    native_async_stream_finish(stream_address as *const NemoRelayNativeAsyncStream)
                }
                Settlement::Reject(message) => native_async_stream_reject(
                    stream_address as *const NemoRelayNativeAsyncStream,
                    message as *const NemoRelayNativeString,
                ),
            }
        });

        settlement_checkpoint.wait();
        stream.cancelled.store(true, Ordering::Release);
        drop(cancellation_guard);

        assert_eq!(operation.join().unwrap(), NemoRelayStatus::InvalidArg);
        assert!(
            stream
                .sender
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some()
        );

        let argument = match settlement {
            Settlement::Push(argument) | Settlement::Reject(argument) => Some(argument),
            Settlement::Finish => None,
        };
        drop(NativeAsyncStreamReceiver {
            receiver,
            stream: Arc::clone(&stream),
        });
        unsafe {
            if let Some(argument) = argument {
                native_string_free(argument as *mut NemoRelayNativeString);
            }
            native_async_stream_release(stream_ref);
        }
    }
}

#[test]
fn native_async_stream_push_is_bounded_retryable_and_incremental() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let stream = Arc::new(NativeAsyncStream {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        settled: AtomicBool::new(false),
        backpressured: AtomicBool::new(false),
        downstream_aborts: Mutex::new(HashMap::new()),
        settlement: Mutex::new(()),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let stream_ref = Arc::into_raw(Arc::clone(&stream)) as *const NemoRelayNativeAsyncStream;
    let first_chunk = native_string(r#"{"chunk":1}"#);
    let second_chunk = native_string(r#"{"chunk":2}"#);
    assert_eq!(
        unsafe { native_async_stream_push_json(stream_ref, first_chunk) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { native_async_stream_push_json(stream_ref, second_chunk) },
        NemoRelayStatus::Backpressured
    );
    assert_last_error_contains("backpressured");
    let mut receiver = NativeAsyncStreamReceiver {
        receiver,
        stream: Arc::clone(&stream),
    };
    assert_eq!(
        runtime.block_on(receiver.next()).unwrap().unwrap(),
        json!({"chunk": 1})
    );
    assert_eq!(
        unsafe { native_async_stream_push_json(stream_ref, second_chunk) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        runtime.block_on(receiver.next()).unwrap().unwrap(),
        json!({"chunk": 2})
    );
    assert_eq!(
        unsafe { native_async_stream_finish(stream_ref) },
        NemoRelayStatus::Ok
    );
    assert!(runtime.block_on(receiver.next()).is_none());
    drop(receiver);
    assert!(unsafe { native_async_stream_is_cancelled(stream_ref) });
    assert_eq!(
        unsafe { native_async_stream_push_json(stream_ref, first_chunk) },
        NemoRelayStatus::InvalidArg
    );
    unsafe {
        native_string_free(first_chunk);
        native_string_free(second_chunk);
        native_async_stream_release(stream_ref);
    }
}

#[test]
fn native_timestamp_scope_type_and_error_mappings_cover_variants() {
    assert_eq!(optional_timestamp_from_native(ptr::null()).unwrap(), None);
    let epoch = 0_i64;
    assert_eq!(
        optional_timestamp_from_native(&epoch).unwrap(),
        DateTime::<Utc>::from_timestamp_micros(0)
    );
    let invalid_timestamp = i64::MAX;
    assert_eq!(
        optional_timestamp_from_native(&invalid_timestamp),
        Err(NemoRelayStatus::InvalidArg)
    );

    for (native, core) in [
        (NemoRelayNativeScopeType::Agent, ScopeType::Agent),
        (NemoRelayNativeScopeType::Function, ScopeType::Function),
        (NemoRelayNativeScopeType::Tool, ScopeType::Tool),
        (NemoRelayNativeScopeType::Llm, ScopeType::Llm),
        (NemoRelayNativeScopeType::Retriever, ScopeType::Retriever),
        (NemoRelayNativeScopeType::Embedder, ScopeType::Embedder),
        (NemoRelayNativeScopeType::Reranker, ScopeType::Reranker),
        (NemoRelayNativeScopeType::Guardrail, ScopeType::Guardrail),
        (NemoRelayNativeScopeType::Evaluator, ScopeType::Evaluator),
        (NemoRelayNativeScopeType::Custom, ScopeType::Custom),
        (NemoRelayNativeScopeType::Unknown, ScopeType::Unknown),
    ] {
        assert_eq!(native_scope_type_to_core(native), core);
    }
    assert!(native_scope_ref(ptr::null()).is_none());

    let status_cases = [
        (NemoRelayStatus::AlreadyExists, "already exists"),
        (NemoRelayStatus::NotFound, "not found"),
        (NemoRelayStatus::ScopeStackEmpty, "scope stack empty"),
        (NemoRelayStatus::GuardrailRejected, "guardrail rejected"),
        (NemoRelayStatus::InvalidArg, "invalid argument"),
        (NemoRelayStatus::Internal, "internal error"),
    ];
    for (status, expected) in status_cases {
        clear_native_last_error();
        assert!(
            flow_error_from_status(status, "fallback")
                .to_string()
                .contains(expected)
        );
    }

    assert_eq!(
        status_from_plugin_error(PluginError::NotFound("missing".into())),
        NemoRelayStatus::NotFound
    );
    assert_eq!(
        status_from_plugin_error(PluginError::Conflict("duplicate".into())),
        NemoRelayStatus::AlreadyExists
    );
    assert_eq!(
        status_from_plugin_error(PluginError::InvalidConfig("invalid".into())),
        NemoRelayStatus::InvalidArg
    );
    let serialization = serde_json::from_str::<Json>("{").unwrap_err();
    assert_eq!(
        status_from_plugin_error(PluginError::Serialization(serialization)),
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        status_from_plugin_error(PluginError::Internal("internal".into())),
        NemoRelayStatus::Internal
    );
    assert_eq!(
        status_from_plugin_error(PluginError::RegistrationFailed("registration".into())),
        NemoRelayStatus::Internal
    );

    for (error, status) in [
        (
            FlowError::AlreadyExists("duplicate".into()),
            NemoRelayStatus::AlreadyExists,
        ),
        (
            FlowError::NotFound("missing".into()),
            NemoRelayStatus::NotFound,
        ),
        (
            FlowError::InvalidArgument("invalid".into()),
            NemoRelayStatus::InvalidArg,
        ),
        (FlowError::ScopeStackEmpty, NemoRelayStatus::ScopeStackEmpty),
        (
            FlowError::GuardrailRejected("blocked".into()),
            NemoRelayStatus::GuardrailRejected,
        ),
        (
            FlowError::Internal("internal".into()),
            NemoRelayStatus::Internal,
        ),
    ] {
        assert_eq!(status_from_flow_error(error), status);
    }
}

unsafe extern "C" fn count_scope_callback(user_data: *mut c_void) -> NemoRelayStatus {
    let calls = unsafe { &*(user_data as *const AtomicUsize) };
    calls.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn fail_scope_callback(_user_data: *mut c_void) -> NemoRelayStatus {
    NemoRelayStatus::InvalidArg
}

#[test]
fn native_scope_stack_abi_covers_lifecycle_and_validation() {
    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let _global_context_restore = GlobalContextRestore::replace_with_empty();
    let _restore = ThreadScopeStackRestore::capture();

    assert_native_scope_stack_null_validation();
    let stack = create_active_native_scope_stack();
    let strings = NativeScopeTestStrings::new();
    let scope = assert_native_scope_lifecycle(&strings);
    assert_native_scope_push_validation(&strings);
    assert_native_scope_pop_and_mark_validation(scope, &strings);
    assert_native_scope_stack_binding_lifecycle();
    free_native_scope_test_resources(stack, scope, strings);
}

fn assert_native_scope_stack_null_validation() {
    assert_eq!(
        unsafe { native_scope_stack_create(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_set_thread(ptr::null()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_restore_thread(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_scope_stack_with_current(ptr::null(), count_scope_callback, ptr::null_mut())
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { native_scope_get_current(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_scope_push(
                ptr::null(),
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
}

fn create_active_native_scope_stack() -> *mut NemoRelayNativeScopeStack {
    let mut stack = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_create(&mut stack) },
        NemoRelayStatus::Ok
    );
    assert!(!stack.is_null());
    assert_eq!(
        unsafe { native_scope_stack_set_thread(stack) },
        NemoRelayStatus::Ok
    );
    assert!(unsafe { native_scope_stack_active() });

    let calls = AtomicUsize::new(0);
    assert_eq!(
        unsafe {
            native_scope_stack_with_current(
                stack,
                count_scope_callback,
                (&calls as *const AtomicUsize).cast_mut().cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        unsafe { native_scope_stack_with_current(stack, fail_scope_callback, ptr::null_mut()) },
        NemoRelayStatus::InvalidArg
    );
    assert_last_error_contains("scope-stack callback returned InvalidArg");

    stack
}

struct NativeScopeTestStrings {
    name: *mut NemoRelayNativeString,
    data: *mut NemoRelayNativeString,
    metadata: *mut NemoRelayNativeString,
    input: *mut NemoRelayNativeString,
    mark_name: *mut NemoRelayNativeString,
    output: *mut NemoRelayNativeString,
    invalid: *mut NemoRelayNativeString,
}

impl NativeScopeTestStrings {
    fn new() -> Self {
        Self {
            name: native_string("native-scope"),
            data: native_string(r#"{"source":"native"}"#),
            metadata: native_string(r#"{"test":true}"#),
            input: native_string(r#"{"input":1}"#),
            mark_name: native_string("native-mark"),
            output: native_string(r#"{"output":1}"#),
            invalid: native_string("not-json"),
        }
    }
}

fn assert_native_scope_lifecycle(
    strings: &NativeScopeTestStrings,
) -> *mut NemoRelayNativeScopeHandle {
    let timestamp = 0_i64;
    let mut scope = ptr::null_mut();
    assert_eq!(
        unsafe {
            native_scope_push(
                strings.name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                strings.data,
                strings.metadata,
                strings.input,
                &timestamp,
                &mut scope,
            )
        },
        NemoRelayStatus::Ok
    );
    assert!(!scope.is_null());
    assert_eq!(native_scope_ref(scope).unwrap().name, "native-scope");

    let mut current = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_get_current(&mut current) },
        NemoRelayStatus::Ok
    );
    assert_eq!(native_scope_ref(current).unwrap().name, "native-scope");
    unsafe { native_scope_handle_free(current) };

    assert_eq!(
        unsafe {
            native_emit_mark(
                strings.mark_name,
                scope,
                strings.data,
                strings.metadata,
                &timestamp,
            )
        },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { native_scope_pop(scope, strings.output, strings.metadata, &timestamp) },
        NemoRelayStatus::Ok
    );

    scope
}

fn assert_native_scope_push_validation(strings: &NativeScopeTestStrings) {
    let mut invalid_scope = ptr::null_mut();
    assert_eq!(
        unsafe {
            native_scope_push(
                strings.name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                strings.invalid,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut invalid_scope,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(invalid_scope.is_null());
    let invalid_const = strings.invalid.cast_const();
    for (data_arg, metadata_arg, input_arg) in [
        (ptr::null(), invalid_const, ptr::null()),
        (ptr::null(), ptr::null(), invalid_const),
    ] {
        assert_eq!(
            unsafe {
                native_scope_push(
                    strings.name,
                    NemoRelayNativeScopeType::Custom,
                    ptr::null(),
                    0,
                    data_arg,
                    metadata_arg,
                    input_arg,
                    ptr::null(),
                    &mut invalid_scope,
                )
            },
            NemoRelayStatus::InvalidJson
        );
        assert!(invalid_scope.is_null());
    }
    let invalid_timestamp = i64::MAX;
    assert_eq!(
        unsafe {
            native_scope_push(
                strings.name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &invalid_timestamp,
                &mut invalid_scope,
            )
        },
        NemoRelayStatus::InvalidArg
    );

    let invalid_name = Box::into_raw(Box::new(NativeHostString(vec![0xff]))).cast();
    assert_eq!(
        unsafe {
            native_scope_push(
                invalid_name,
                NemoRelayNativeScopeType::Custom,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut invalid_scope,
            )
        },
        NemoRelayStatus::InvalidUtf8
    );
    assert_eq!(
        unsafe {
            native_emit_mark(
                invalid_name,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            )
        },
        NemoRelayStatus::InvalidUtf8
    );
    unsafe { native_string_free(invalid_name) };
}

fn assert_native_scope_pop_and_mark_validation(
    scope: *mut NemoRelayNativeScopeHandle,
    strings: &NativeScopeTestStrings,
) {
    let invalid_timestamp = i64::MAX;
    assert_eq!(
        unsafe { native_scope_pop(scope, strings.invalid, ptr::null(), ptr::null()) },
        NemoRelayStatus::InvalidJson
    );
    assert_eq!(
        unsafe { native_scope_pop(scope, ptr::null(), strings.invalid, ptr::null()) },
        NemoRelayStatus::InvalidJson
    );
    assert_eq!(
        unsafe { native_scope_pop(scope, ptr::null(), ptr::null(), &invalid_timestamp) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe {
            native_emit_mark(
                strings.mark_name,
                scope,
                strings.invalid,
                ptr::null(),
                ptr::null(),
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert_eq!(
        unsafe {
            native_emit_mark(
                strings.mark_name,
                scope,
                ptr::null(),
                strings.invalid,
                ptr::null(),
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert_eq!(
        unsafe {
            native_emit_mark(
                strings.mark_name,
                scope,
                ptr::null(),
                ptr::null(),
                &invalid_timestamp,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    let malformed_schema = native_string(r#"{"name":7,"version":"1"}"#);
    assert_eq!(
        unsafe {
            native_emit_mark_v2(
                strings.mark_name,
                scope,
                ptr::null(),
                ptr::null(),
                malformed_schema,
                ptr::null(),
                ptr::null(),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_string_free(malformed_schema) };
    let invalid_severity = native_string("fatal");
    assert_eq!(
        unsafe {
            native_emit_mark_v2(
                strings.mark_name,
                scope,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                invalid_severity,
                ptr::null(),
            )
        },
        NemoRelayStatus::InvalidArg
    );
    unsafe { native_string_free(invalid_severity) };
    assert_eq!(
        unsafe { native_scope_pop(ptr::null(), ptr::null(), ptr::null(), ptr::null()) },
        NemoRelayStatus::NullPointer
    );
}

#[test]
fn native_runtime_diagnostics_callback_returns_owned_json() {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        unsafe { native_get_runtime_diagnostics(&mut out) },
        NemoRelayStatus::Ok
    );
    assert!(!out.is_null());

    let value: Json = serde_json::from_str(&read_native_string(out).unwrap()).unwrap();
    assert!(value["entries"].is_array());
    unsafe { native_string_free(out) };
}

fn assert_native_scope_stack_binding_lifecycle() {
    let mut binding = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(&mut binding) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { native_scope_stack_restore_thread(binding) },
        NemoRelayStatus::Ok
    );
    let mut disposable_binding = ptr::null_mut();
    assert_eq!(
        unsafe { native_scope_stack_capture_thread(&mut disposable_binding) },
        NemoRelayStatus::Ok
    );
    unsafe { native_scope_stack_binding_free(disposable_binding) };
}

fn free_native_scope_test_resources(
    stack: *mut NemoRelayNativeScopeStack,
    scope: *mut NemoRelayNativeScopeHandle,
    strings: NativeScopeTestStrings,
) {
    for value in [
        strings.name,
        strings.data,
        strings.metadata,
        strings.input,
        strings.mark_name,
        strings.output,
        strings.invalid,
    ] {
        unsafe { native_string_free(value) };
    }
    unsafe {
        native_scope_handle_free(scope);
        native_scope_handle_free(ptr::null_mut());
        native_scope_stack_free(stack);
        native_scope_stack_free(ptr::null_mut());
        native_scope_stack_binding_free(ptr::null_mut());
    }
}

unsafe extern "C" fn noop_subscriber(
    _user_data: *mut c_void,
    _event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_json(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _payload_json: *const NemoRelayNativeString,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_conditional(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_tool_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeToolNextFn,
    _next_ctx: *mut c_void,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_request(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeRequestContext,
    _out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_json(
    _user_data: *mut c_void,
    _payload_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeResponseContext,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_conditional(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_request_intercept(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _annotated_json: *const NemoRelayNativeString,
    _out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmNextFn,
    _next_ctx: *mut c_void,
    _out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn noop_llm_stream_execution(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmStreamNextFn,
    _next_ctx: *mut c_void,
    _out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

#[test]
fn native_registration_entrypoints_reject_null_contexts() {
    unsafe {
        assert_eq!(
            native_plugin_context_register_subscriber(
                ptr::null_mut(),
                ptr::null(),
                noop_subscriber,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_request_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_response_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_conditional_execution_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_request_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_tool_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_tool_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_request_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_request,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_response_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_conditional_execution_guardrail(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_request_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                noop_llm_request_intercept,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_llm_stream_execution_intercept(
                ptr::null_mut(),
                ptr::null(),
                0,
                noop_llm_stream_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
    }
    assert_last_error_contains("plugin context is null");
}

#[cfg(unix)]
#[test]
fn native_registration_entrypoints_reject_invalid_host_contexts_and_names() {
    let instance = Arc::new(NativePluginInstance {
        plugin_kind: "test.native".into(),
        relay_compat: "^0.8".into(),
        allows_multiple_components: false,
        plugin: Mutex::new(NemoRelayNativePluginV1::default()),
        _library: libloading::os::unix::Library::this().into(),
    });
    let mut invalid_host = NativeHostPluginContext {
        ctx: ptr::null_mut(),
        instance: Arc::clone(&instance),
    };
    assert_eq!(
        unsafe {
            native_plugin_context_register_subscriber(
                ptr::from_mut(&mut invalid_host).cast(),
                ptr::null(),
                noop_subscriber,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::NullPointer
    );

    let mut registration = PluginRegistrationContext::new();
    let mut host = NativeHostPluginContext {
        ctx: ptr::from_mut(&mut registration),
        instance,
    };
    let ctx = ptr::from_mut(&mut host).cast();

    assert_registration_entrypoints_reject_invalid_names(ctx);
    assert_registration_entrypoints_accept_valid_names(ctx);
    assert_async_registration_entrypoints_validate_contracts(ctx);
    assert_async_request_registration_rejects_legacy_relay_contract();
}

#[cfg(unix)]
fn assert_registration_entrypoints_reject_invalid_names(ctx: *mut NemoRelayNativePluginContext) {
    let invalid_name = Box::into_raw(Box::new(NativeHostString(vec![0xff]))).cast();
    unsafe {
        assert_eq!(
            native_plugin_context_register_subscriber(
                ctx,
                invalid_name,
                noop_subscriber,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_request_guardrail(
                ctx,
                invalid_name,
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_response_guardrail(
                ctx,
                invalid_name,
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_tool_conditional_execution_guardrail(
                ctx,
                invalid_name,
                0,
                noop_tool_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_tool_request_intercept(
                ctx,
                invalid_name,
                0,
                false,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_tool_execution_intercept(
                ctx,
                invalid_name,
                0,
                noop_tool_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_request_guardrail(
                ctx,
                invalid_name,
                0,
                noop_llm_request,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_response_guardrail(
                ctx,
                invalid_name,
                0,
                noop_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_conditional_execution_guardrail(
                ctx,
                invalid_name,
                0,
                noop_llm_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_request_intercept(
                ctx,
                invalid_name,
                0,
                false,
                noop_llm_request_intercept,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_execution_intercept(
                ctx,
                invalid_name,
                0,
                noop_llm_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_llm_stream_execution_intercept(
                ctx,
                invalid_name,
                0,
                noop_llm_stream_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_async_stream_middleware(
                ctx,
                invalid_name,
                0,
                invoke_native_stream_next_then_return_state,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        assert_eq!(
            native_plugin_context_register_async_middleware(
                ctx,
                NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest as u32,
                invalid_name,
                0,
                false,
                resolve_async_static_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidUtf8
        );
        drop(Box::from_raw(invalid_name as *mut NativeHostString));
    }
}

#[cfg(unix)]
fn assert_registration_entrypoints_accept_valid_names(ctx: *mut NemoRelayNativePluginContext) {
    unsafe {
        let name = native_string("registered");
        assert_eq!(
            native_plugin_context_register_subscriber(
                ctx,
                name,
                noop_subscriber,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_request_guardrail(
                ctx,
                name,
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_tool_sanitize_response_guardrail(
                ctx,
                name,
                0,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_tool_conditional_execution_guardrail(
                ctx,
                name,
                0,
                noop_tool_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_tool_request_intercept(
                ctx,
                name,
                0,
                false,
                noop_tool_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_tool_execution_intercept(
                ctx,
                name,
                0,
                noop_tool_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_request_guardrail(
                ctx,
                name,
                0,
                noop_llm_request,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_sanitize_response_guardrail(
                ctx,
                name,
                0,
                noop_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_conditional_execution_guardrail(
                ctx,
                name,
                0,
                noop_llm_conditional,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_request_intercept(
                ctx,
                name,
                0,
                false,
                noop_llm_request_intercept,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_execution_intercept(
                ctx,
                name,
                0,
                noop_llm_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            native_plugin_context_register_llm_stream_execution_intercept(
                ctx,
                name,
                0,
                noop_llm_stream_execution,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        native_string_free(name);
    }
}

#[cfg(unix)]
fn assert_async_registration_entrypoints_validate_contracts(
    ctx: *mut NemoRelayNativePluginContext,
) {
    unsafe {
        assert_eq!(
            native_plugin_context_register_async_stream_middleware(
                ptr::null_mut(),
                ptr::null(),
                0,
                invoke_native_stream_next_then_return_state,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            native_plugin_context_register_async_middleware(
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
                false,
                resolve_async_static_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::NullPointer
        );

        let name = native_string("async-registered");
        let rejected_user_data_frees = AtomicUsize::new(0);
        assert_eq!(
            native_plugin_context_register_async_middleware(
                ctx,
                u32::MAX,
                name,
                0,
                false,
                resolve_async_static_json,
                (&rejected_user_data_frees as *const AtomicUsize)
                    .cast_mut()
                    .cast(),
                Some(count_user_data_free),
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(rejected_user_data_frees.load(Ordering::SeqCst), 1);
        assert_eq!(
            native_plugin_context_register_async_middleware(
                ctx,
                NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept as u32,
                name,
                0,
                false,
                resolve_async_static_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            native_plugin_context_register_async_middleware(
                ctx,
                NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest as u32,
                name,
                0,
                false,
                resolve_async_static_json,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        let stream_name = native_string("async-stream-registered");
        assert_eq!(
            native_plugin_context_register_async_stream_middleware(
                ctx,
                stream_name,
                0,
                invoke_native_stream_next_then_return_state,
                ptr::null_mut(),
                None,
            ),
            NemoRelayStatus::Ok
        );
        native_string_free(name);
        native_string_free(stream_name);
    }
}

#[cfg(unix)]
fn assert_async_request_registration_rejects_legacy_relay_contract() {
    let instance = Arc::new(NativePluginInstance {
        plugin_kind: "test.native.legacy".into(),
        relay_compat: "^0.5".into(),
        allows_multiple_components: false,
        plugin: Mutex::new(NemoRelayNativePluginV1::default()),
        _library: libloading::os::unix::Library::this().into(),
    });
    let mut registration = PluginRegistrationContext::new();
    let mut host = NativeHostPluginContext {
        ctx: ptr::from_mut(&mut registration),
        instance,
    };
    let name = native_string("legacy-request");
    assert_eq!(
        unsafe {
            native_plugin_context_register_async_middleware(
                ptr::from_mut(&mut host).cast(),
                NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept as u32,
                name,
                0,
                false,
                resolve_async_static_json,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert_last_error_contains("excludes Relay 0.5");
    unsafe { native_string_free(name) };
}

#[cfg(unix)]
unsafe extern "C" fn resolve_async_static_json(
    user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    assert_eq!(
        unsafe { native_async_completion_resolve_json(completion, user_data.cast()) },
        NemoRelayStatus::Ok
    );
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

#[cfg(unix)]
#[tokio::test]
async fn native_async_wrappers_validate_callback_result_shapes() {
    let instance = Arc::new(NativePluginInstance {
        plugin_kind: "test.native.async".into(),
        relay_compat: "^0.8".into(),
        allows_multiple_components: false,
        plugin: Mutex::new(NemoRelayNativePluginV1::default()),
        _library: libloading::os::unix::Library::this().into(),
    });
    let result = native_string("true");
    let user_data = result.cast();

    let tool_json = wrap_native_async_tool_json(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert_eq!(
        tool_json("tool".into(), json!({})).await.unwrap(),
        json!(true)
    );

    let tool_conditional = wrap_native_async_tool_conditional(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert!(
        tool_conditional("tool".into(), json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("expected string or null")
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"messages": []}),
    };
    let llm_conditional = wrap_native_async_llm_conditional(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert!(
        llm_conditional(request.clone())
            .await
            .unwrap_err()
            .to_string()
            .contains("expected string or null")
    );

    let sanitize_request = wrap_native_async_llm_sanitize_request(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert!(
        sanitize_request(
            request.clone(),
            LlmSanitizeRequestContext::for_request_codec(None),
        )
        .await
        .is_err()
    );

    let sanitize_response = wrap_native_async_llm_sanitize_response(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert_eq!(
        sanitize_response(
            json!({"response": true}),
            LlmSanitizeResponseContext::for_response_codec(None),
        )
        .await
        .unwrap(),
        Some(json!(true))
    );

    let request_intercept = wrap_native_async_llm_request_intercept(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert!(
        request_intercept("model".into(), request, None)
            .await
            .unwrap_err()
            .to_string()
            .contains("invalid native async LLM intercept outcome")
    );

    let tool_execution = wrap_native_async_tool_execution(
        Arc::clone(&instance),
        resolve_async_static_json,
        user_data,
        None,
    );
    assert!(
        tool_execution("tool", json!({}), tool_next(Ok(Json::Null)))
            .await
            .unwrap_err()
            .to_string()
            .contains("invalid native async tool outcome")
    );

    let fields = EventSanitizeFields::default();
    let fields_result = native_string_from_json(&serde_json::to_value(&fields).unwrap()).unwrap();
    let event_sanitize = wrap_native_async_event_sanitize(
        Arc::clone(&instance),
        resolve_async_static_json,
        fields_result.cast(),
        None,
    );
    let event = Event::Mark(crate::api::event::MarkEvent::new(
        crate::api::event::BaseEvent::builder()
            .name("native-async-event")
            .build(),
        None,
        None,
    ));
    assert_eq!(
        event_sanitize(Arc::new(event), fields.clone())
            .await
            .unwrap(),
        fields
    );

    drop(event_sanitize);
    drop(tool_execution);
    drop(request_intercept);
    drop(sanitize_response);
    drop(sanitize_request);
    drop(llm_conditional);
    drop(tool_conditional);
    drop(tool_json);
    unsafe {
        native_string_free(fields_result);
        native_string_free(result);
    }
}

#[test]
fn native_codec_operations_report_json_and_codec_failures() {
    let openai_request_codec =
        NativeHostLlmRequestCodec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmCodec>);
    let openai_response_codec =
        NativeHostLlmResponseCodec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>);
    let failing_request_codec =
        NativeHostLlmRequestCodec(Arc::new(FailingNativeCodec) as Arc<dyn LlmCodec>);
    let failing_response_codec =
        NativeHostLlmResponseCodec(Arc::new(FailingNativeCodec) as Arc<dyn LlmResponseCodec>);
    let invalid_json = native_string("not-json");
    let mut output = ptr::null_mut();

    assert_eq!(
        unsafe {
            native_llm_request_codec_decode(
                ptr::from_ref(&openai_request_codec).cast(),
                invalid_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("invalid request JSON");

    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(
                ptr::from_ref(&openai_response_codec).cast(),
                invalid_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("invalid response JSON");
    unsafe { native_string_free(invalid_json) };

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}]
        }),
    };
    let request_json = native_string(&serde_json::to_string(&request).unwrap());
    assert_eq!(
        unsafe {
            native_llm_request_codec_decode(
                ptr::from_ref(&failing_request_codec).cast(),
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("request decode rejected");

    let annotated = OpenAIChatCodec.decode(&request).unwrap();
    let annotated_json = native_string(&serde_json::to_string(&annotated).unwrap());
    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::from_ref(&failing_request_codec).cast(),
                annotated_json,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("request encode rejected");

    let response_json = native_string(
        r#"{"id":"chatcmpl-test","model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"secret"},"finish_reason":"stop"}]}"#,
    );
    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(
                ptr::from_ref(&failing_response_codec).cast(),
                response_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("response decode rejected");

    unsafe {
        native_string_free(request_json);
        native_string_free(annotated_json);
        native_string_free(response_json);
    }
}

#[test]
fn native_v4_completion_scoped_codecs_enforce_direction_and_expiration() {
    let (sender, _receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        continuation_aborts: Mutex::new(HashMap::new()),
        codec: Some(NativeAsyncCodecCapability::Request(
            Arc::new(OpenAIChatCodec) as Arc<dyn LlmCodec>,
        )),
        before_settlement_lock: None,
        _callback_user_data: None,
    });
    let completion_ref =
        Arc::into_raw(Arc::clone(&completion)) as *const NemoRelayNativeAsyncCompletion;
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}]
        }),
    };
    let request_json = native_string(&serde_json::to_string(&request).unwrap());
    let mut output = ptr::null_mut();

    assert_eq!(
        unsafe {
            native_async_completion_llm_request_codec_decode(
                completion_ref,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Ok
    );
    let annotated: AnnotatedLlmRequest =
        serde_json::from_str(&read_native_string(output).unwrap()).unwrap();
    unsafe { native_string_free(output) };
    let annotated_json = native_string(&serde_json::to_string(&annotated).unwrap());
    assert_eq!(
        unsafe {
            native_async_completion_llm_request_codec_encode(
                completion_ref,
                annotated_json,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Ok
    );
    unsafe { native_string_free(output) };

    let sentinel = native_string("sentinel");
    output = sentinel;
    assert_eq!(
        unsafe {
            native_async_completion_llm_response_codec_decode(
                completion_ref,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert!(output.is_null());
    unsafe { native_string_free(sentinel) };

    completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    let sentinel = native_string("expired");
    output = sentinel;
    assert_eq!(
        unsafe {
            native_async_completion_llm_request_codec_decode(
                completion_ref,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert!(output.is_null());
    unsafe {
        native_string_free(sentinel);
        native_string_free(request_json);
        native_string_free(annotated_json);
        native_async_completion_release(completion_ref);
    }
}

#[test]
fn native_codec_operations_contain_codec_panics() {
    let request_codec =
        NativeHostLlmRequestCodec(Arc::new(PanickingNativeCodec) as Arc<dyn LlmCodec>);
    let response_codec =
        NativeHostLlmResponseCodec(Arc::new(PanickingNativeCodec) as Arc<dyn LlmResponseCodec>);
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}]
        }),
    };
    let request_json = native_string(&serde_json::to_string(&request).unwrap());
    let annotated = OpenAIChatCodec.decode(&request).unwrap();
    let annotated_json = native_string(&serde_json::to_string(&annotated).unwrap());
    let response_json = native_string(
        r#"{"id":"chatcmpl-test","model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"secret"},"finish_reason":"stop"}]}"#,
    );
    let mut output = ptr::null_mut();

    assert_eq!(
        unsafe {
            native_llm_request_codec_decode(
                ptr::from_ref(&request_codec).cast(),
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("request codec decode panicked");

    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::from_ref(&request_codec).cast(),
                annotated_json,
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("request codec encode panicked");

    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(
                ptr::from_ref(&response_codec).cast(),
                response_json,
                &mut output,
            )
        },
        NemoRelayStatus::Internal
    );
    assert!(output.is_null());
    assert_last_error_contains("response codec decode panicked");

    unsafe {
        native_string_free(request_json);
        native_string_free(annotated_json);
        native_string_free(response_json);
    }
}

#[test]
fn native_codec_operations_clear_output_slots_on_null_arguments() {
    let request_codec = NativeHostLlmRequestCodec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmCodec>);
    let response_codec =
        NativeHostLlmResponseCodec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>);
    let request_json = native_string(
        r#"{"headers":{},"content":{"model":"gpt-test","messages":[{"role":"user","content":"hello"}]}}"#,
    );
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    };
    let annotated = OpenAIChatCodec.decode(&request).unwrap();
    let annotated_json = native_string(&serde_json::to_string(&annotated).unwrap());

    assert_eq!(
        unsafe { native_llm_request_codec_decode(ptr::null(), request_json, &mut ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_request_codec_decode(
                ptr::from_ref(&request_codec).cast(),
                request_json,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::null(),
                annotated_json,
                request_json,
                &mut ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::from_ref(&request_codec).cast(),
                annotated_json,
                ptr::null(),
                &mut ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::from_ref(&request_codec).cast(),
                annotated_json,
                request_json,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(ptr::null(), request_json, &mut ptr::null_mut())
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(
                ptr::from_ref(&response_codec).cast(),
                request_json,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );

    let request_decode_sentinel = native_string("request-decode-sentinel");
    let mut output = request_decode_sentinel;
    set_native_last_error("stale request decode error");
    assert_eq!(
        unsafe {
            native_llm_request_codec_decode(
                ptr::from_ref(&request_codec).cast(),
                ptr::null(),
                &mut output,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert!(output.is_null());
    assert_last_error_contains("request codec decode request is null");
    unsafe { native_string_free(request_decode_sentinel) };

    let request_encode_sentinel = native_string("request-encode-sentinel");
    output = request_encode_sentinel;
    set_native_last_error("stale request encode error");
    assert_eq!(
        unsafe {
            native_llm_request_codec_encode(
                ptr::from_ref(&request_codec).cast(),
                ptr::null(),
                request_json,
                &mut output,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert!(output.is_null());
    assert_last_error_contains("request codec encode annotated request is null");
    unsafe { native_string_free(request_encode_sentinel) };

    let response_decode_sentinel = native_string("response-decode-sentinel");
    output = response_decode_sentinel;
    set_native_last_error("stale response decode error");
    assert_eq!(
        unsafe {
            native_llm_response_codec_decode(
                ptr::from_ref(&response_codec).cast(),
                ptr::null(),
                &mut output,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert!(output.is_null());
    assert_last_error_contains("response codec decode response is null");
    unsafe {
        native_string_free(response_decode_sentinel);
        native_string_free(annotated_json);
        native_string_free(request_json);
    }
}

unsafe extern "C" fn tool_json_echo(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let payload = read_native_string(payload_json).unwrap();
    unsafe { *out_json = native_string(&payload) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn tool_json_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    set_native_last_error("tool callback rejected input");
    unsafe { *out_json = native_string(r#"{"discarded":true}"#) };
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn tool_conditional_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_reason = native_string("discarded reason") };
    set_native_last_error("tool conditional failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn tool_conditional_reason(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_reason = native_string("blocked by tool policy") };
    NemoRelayStatus::Ok
}

#[cfg(unix)]
unsafe extern "C" fn llm_conditional_error(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_reason = native_string("discarded reason") };
    set_native_last_error("LLM conditional failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn llm_conditional_reason(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_reason = native_string("blocked by LLM policy") };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_error(
    _user_data: *mut c_void,
    _request_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeRequestContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_request_json = native_string(r#"{"discarded":true}"#) };
    set_native_last_error("LLM request sanitizer failed");
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn llm_response_error(
    _user_data: *mut c_void,
    _response_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeResponseContext,
    out_response_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_response_json = native_string(r#"{"discarded":true}"#) };
    set_native_last_error("LLM response sanitizer failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn tool_execution_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeToolNextFn,
    _next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_outcome_json = native_string(r#"{"discarded":true}"#) };
    set_native_last_error("tool execution failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn llm_request_intercept_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _annotated_json: *const NemoRelayNativeString,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_outcome_json = native_string(r#"{"discarded":true}"#) };
    set_native_last_error("LLM request intercept failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn llm_execution_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmNextFn,
    _next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_json = native_string(r#"{"discarded":true}"#) };
    set_native_last_error("LLM execution failed");
    NemoRelayStatus::InvalidArg
}

#[cfg(unix)]
unsafe extern "C" fn llm_stream_execution_error(
    _user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    _request_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeLlmStreamNextFn,
    _next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    unsafe { *out_stream = NemoRelayNativeLlmStreamV1::default() };
    set_native_last_error("LLM stream execution failed");
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn llm_request_echo(
    _user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeRequestContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let request = read_native_string(request_json).unwrap();
    unsafe { *out_request_json = native_string(&request) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_alias(
    _user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeRequestContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_request_json = request_json.cast_mut() };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_codec_round_trip(
    _user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeRequestContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    assert!(!context.codec.is_null());
    let mut annotated = ptr::null_mut();
    let status =
        unsafe { native_llm_request_codec_decode(context.codec, request_json, &mut annotated) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let status = unsafe {
        native_llm_request_codec_encode(context.codec, annotated, request_json, out_request_json)
    };
    unsafe { native_string_free(annotated) };
    status
}

unsafe extern "C" fn llm_response_codec_decode_and_echo(
    _user_data: *mut c_void,
    response_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeResponseContext,
    out_response_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    assert!(!context.codec.is_null());
    let mut annotated = ptr::null_mut();
    let status =
        unsafe { native_llm_response_codec_decode(context.codec, response_json, &mut annotated) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    unsafe { native_string_free(annotated) };
    let response = read_native_string(response_json).unwrap();
    unsafe { *out_response_json = native_string(&response) };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_response_alias(
    _user_data: *mut c_void,
    response_json: *const NemoRelayNativeString,
    _context: NemoRelayNativeLlmSanitizeResponseContext,
    out_response_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { *out_response_json = response_json.cast_mut() };
    NemoRelayStatus::Ok
}

#[test]
fn native_callback_helpers_cover_success_error_and_invalid_output() {
    assert_eq!(
        call_tool_json_callback(tool_json_echo, ptr::null_mut(), "tool", &json!({"a": 1})).unwrap(),
        json!({"a": 1})
    );
    assert!(
        call_tool_json_callback(tool_json_error, ptr::null_mut(), "tool", &Json::Null)
            .unwrap_err()
            .to_string()
            .contains("tool callback rejected input")
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    assert_eq!(
        call_llm_sanitize_request_callback(
            llm_request_echo,
            ptr::null_mut(),
            &request,
            LlmSanitizeRequestContext::default(),
        )
        .unwrap(),
        Some(request.clone())
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "alias"}),
    };
    assert_eq!(
        call_llm_sanitize_request_callback(
            llm_request_alias,
            ptr::null_mut(),
            &request,
            LlmSanitizeRequestContext::default(),
        )
        .unwrap(),
        Some(request.clone())
    );

    let response = json!({"message": "alias"});
    assert_eq!(
        call_llm_sanitize_response_callback(
            llm_response_alias,
            ptr::null_mut(),
            &response,
            LlmSanitizeResponseContext::default(),
        )
        .unwrap(),
        Some(response.clone())
    );

    assert!(
        call_llm_sanitize_request_callback(
            llm_request_error,
            ptr::null_mut(),
            &request,
            LlmSanitizeRequestContext::default(),
        )
        .unwrap_err()
        .to_string()
        .contains("LLM request sanitizer failed")
    );
    assert_eq!(
        call_llm_sanitize_request_callback(
            noop_llm_request,
            ptr::null_mut(),
            &request,
            LlmSanitizeRequestContext::default(),
        )
        .unwrap(),
        None
    );
    assert!(
        call_llm_sanitize_response_callback(
            llm_response_error,
            ptr::null_mut(),
            &response,
            LlmSanitizeResponseContext::default(),
        )
        .unwrap_err()
        .to_string()
        .contains("LLM response sanitizer failed")
    );
    assert_eq!(
        call_llm_sanitize_response_callback(
            noop_json,
            ptr::null_mut(),
            &response,
            LlmSanitizeResponseContext::default(),
        )
        .unwrap(),
        None
    );
}

#[cfg(unix)]
#[tokio::test]
async fn native_callback_wrappers_release_error_outputs_and_preserve_reasons() {
    let instance = Arc::new(NativePluginInstance {
        plugin_kind: "test.native.callback-errors".into(),
        relay_compat: "^0.8".into(),
        allows_multiple_components: false,
        plugin: Mutex::new(NemoRelayNativePluginV1::default()),
        _library: libloading::os::unix::Library::this().into(),
    });
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };

    let tool_conditional = wrap_tool_conditional_fn(
        Arc::clone(&instance),
        tool_conditional_error,
        ptr::null_mut(),
        None,
    );
    assert!(
        tool_conditional("tool".into(), json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("tool conditional failed")
    );
    let tool_conditional = wrap_tool_conditional_fn(
        Arc::clone(&instance),
        tool_conditional_reason,
        ptr::null_mut(),
        None,
    );
    assert_eq!(
        tool_conditional("tool".into(), json!({})).await.unwrap(),
        Some("blocked by tool policy".into())
    );

    let tool_execution = wrap_tool_execution_fn(
        Arc::clone(&instance),
        tool_execution_error,
        ptr::null_mut(),
        None,
    );
    assert!(
        tool_execution("tool", json!({}), tool_next(Ok(Json::Null)))
            .await
            .unwrap_err()
            .to_string()
            .contains("tool execution failed")
    );

    let llm_conditional = wrap_llm_conditional_fn(
        Arc::clone(&instance),
        llm_conditional_error,
        ptr::null_mut(),
        None,
    );
    assert!(
        llm_conditional(request.clone())
            .await
            .unwrap_err()
            .to_string()
            .contains("LLM conditional failed")
    );
    let llm_conditional = wrap_llm_conditional_fn(
        Arc::clone(&instance),
        llm_conditional_reason,
        ptr::null_mut(),
        None,
    );
    assert_eq!(
        llm_conditional(request.clone()).await.unwrap(),
        Some("blocked by LLM policy".into())
    );

    let request_intercept = wrap_llm_request_intercept_fn(
        Arc::clone(&instance),
        llm_request_intercept_error,
        ptr::null_mut(),
        None,
    );
    assert!(
        request_intercept("model".into(), request.clone(), None)
            .await
            .unwrap_err()
            .to_string()
            .contains("LLM request intercept failed")
    );

    let llm_execution = wrap_llm_execution_fn(
        Arc::clone(&instance),
        llm_execution_error,
        ptr::null_mut(),
        None,
    );
    assert!(
        llm_execution("model", request.clone(), llm_next(Ok(Json::Null)))
            .await
            .unwrap_err()
            .to_string()
            .contains("LLM execution failed")
    );

    let stream_next: LlmStreamExecutionNextFn =
        Arc::new(|_| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) }));
    let llm_stream_execution =
        wrap_llm_stream_execution_fn(instance, llm_stream_execution_error, ptr::null_mut(), None);
    assert!(
        llm_stream_execution("model", request, stream_next)
            .await
            .err()
            .expect("native stream callback should fail")
            .to_string()
            .contains("LLM stream execution failed")
    );
}

#[test]
fn native_sanitizer_context_resolves_directional_codecs() {
    let codec = Arc::new(OpenAIChatCodec);
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}],
            "preserve": true
        }),
    };
    let sanitized = call_llm_sanitize_request_callback(
        llm_request_codec_round_trip,
        ptr::null_mut(),
        &request,
        LlmSanitizeRequestContext::for_request_codec(Some(codec.clone())),
    )
    .unwrap()
    .expect("native request sanitizer returns a request");
    assert_eq!(sanitized, request);

    let response = json!({
        "id": "chatcmpl-test",
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "secret"},
            "finish_reason": "stop"
        }]
    });
    let sanitized = call_llm_sanitize_response_callback(
        llm_response_codec_decode_and_echo,
        ptr::null_mut(),
        &response,
        LlmSanitizeResponseContext::for_response_codec(Some(codec)),
    )
    .unwrap()
    .expect("native response sanitizer returns a response");
    assert_eq!(sanitized, response);
}

#[test]
fn native_llm_sanitize_context_preserves_all_codec_identity_states() {
    let cases = [
        (
            LlmCodecIdentity::None,
            NemoRelayNativeLlmCodecKind::None,
            None,
        ),
        (
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
            NemoRelayNativeLlmCodecKind::BuiltIn,
            Some("openai_chat"),
        ),
        (
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses),
            NemoRelayNativeLlmCodecKind::BuiltIn,
            Some("openai_responses"),
        ),
        (
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages),
            NemoRelayNativeLlmCodecKind::BuiltIn,
            Some("anthropic_messages"),
        ),
        (
            LlmCodecIdentity::Runtime("com.example.chat.v1".into()),
            NemoRelayNativeLlmCodecKind::Runtime,
            Some("com.example.chat.v1"),
        ),
        (
            LlmCodecIdentity::Opaque,
            NemoRelayNativeLlmCodecKind::Opaque,
            None,
        ),
    ];

    for (codec, expected_kind, expected_id) in cases {
        let (codec_kind, context_id) =
            native_llm_codec_identity(&codec).expect("native context conversion should succeed");
        assert_eq!(codec_kind, expected_kind);
        assert_eq!(
            context_id.map(|id| read_native_string(id).unwrap()),
            expected_id.map(str::to_owned),
        );
        if let Some(context_id) = context_id {
            unsafe { native_string_free(context_id) };
        }
    }
}

#[test]
fn native_async_llm_sanitize_context_uses_stable_codec_envelope() {
    assert_eq!(
        native_async_codec_identity(&LlmCodecIdentity::None),
        json!({"codec_kind": "none", "codec_id": null})
    );
    assert_eq!(
        native_async_codec_identity(&LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)),
        json!({"codec_kind": "builtin", "codec_id": "openai_chat"})
    );
    assert_eq!(
        native_async_codec_identity(&LlmCodecIdentity::Runtime("com.example.chat.v1".into())),
        json!({"codec_kind": "runtime", "codec_id": "com.example.chat.v1"})
    );
    assert_eq!(
        native_async_codec_identity(&LlmCodecIdentity::Opaque),
        json!({"codec_kind": "opaque", "codec_id": null})
    );
}

unsafe extern "C" fn count_user_data_free(user_data: *mut c_void) {
    let count = unsafe { &*(user_data as *const AtomicUsize) };
    count.fetch_add(1, Ordering::SeqCst);
}

#[derive(Default)]
struct ConditionalCallbackTestState {
    calls: Mutex<Vec<(BTreeSet<RuntimeRegistrationKind>, String)>>,
    frees: AtomicUsize,
}

struct ConditionalCallbackTestUserData {
    state: Arc<ConditionalCallbackTestState>,
}

unsafe extern "C" fn free_conditional_callback_test_user_data(user_data: *mut c_void) {
    let user_data = unsafe { Box::from_raw(user_data.cast::<ConditionalCallbackTestUserData>()) };
    user_data.state.frees.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn exercise_native_conditional_callback(
    user_data: *mut c_void,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_reason.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_reason = ptr::null_mut() };
    let user_data = unsafe { &*user_data.cast::<ConditionalCallbackTestUserData>() };
    let kinds = match read_native_string(kinds_json)
        .ok()
        .and_then(|kinds| serde_json::from_str(&kinds).ok())
    {
        Some(kinds) => kinds,
        None => return NemoRelayStatus::InvalidJson,
    };
    let registration_name = match read_native_string(registration_name) {
        Ok(registration_name) => registration_name,
        Err(_) => return NemoRelayStatus::InvalidUtf8,
    };
    user_data
        .state
        .calls
        .lock()
        .unwrap()
        .push((kinds, registration_name.clone()));
    match registration_name.as_str() {
        "blocked" => {
            unsafe { *out_reason = native_string("disabled") };
            NemoRelayStatus::Ok
        }
        "error-with-output" => {
            unsafe { *out_reason = native_string("ignored") };
            set_native_last_error("injected callback error");
            NemoRelayStatus::Internal
        }
        "invalid-output" => {
            unsafe { *out_reason = Box::into_raw(Box::new(NativeHostString(vec![0xff]))).cast() };
            NemoRelayStatus::Ok
        }
        _ => NemoRelayStatus::Ok,
    }
}

unsafe extern "C" fn unreachable_native_conditional_callback(
    _user_data: *mut c_void,
    _kinds_json: *const NemoRelayNativeString,
    _registration_name: *const NemoRelayNativeString,
    _out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

#[test]
fn native_async_registration_user_data_guard_frees_or_transfers_exactly_once() {
    let frees = AtomicUsize::new(0);
    {
        let _guard = NativeCallbackUserDataGuard::new(
            (&frees as *const AtomicUsize).cast_mut().cast(),
            Some(count_user_data_free),
        );
    }
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    let transferred = NativeCallbackUserDataGuard::new(
        (&frees as *const AtomicUsize).cast_mut().cast(),
        Some(count_user_data_free),
    )
    .transfer();
    assert_eq!(frees.load(Ordering::SeqCst), 1);
    unsafe { transferred.1.unwrap()(transferred.0) };
    assert_eq!(frees.load(Ordering::SeqCst), 2);
}

#[test]
fn native_conditional_callback_wrapper_blocks_allows_fails_open_and_releases_state() {
    let state = Arc::new(ConditionalCallbackTestState::default());
    let user_data = Box::into_raw(Box::new(ConditionalCallbackTestUserData {
        state: Arc::clone(&state),
    }))
    .cast();
    let user_data = Arc::new(NativeCallbackUserData {
        ptr: user_data,
        free_fn: Some(free_conditional_callback_test_user_data),
        _instance: None,
    });
    let callback = wrap_conditional_middleware_guardrail_callback(
        "fixture_native".into(),
        exercise_native_conditional_callback,
        user_data,
    );
    let kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
    let live_before = native_string_live_allocations();

    assert_eq!(callback(&kinds, "blocked"), Some("disabled".into()));
    assert_eq!(callback(&kinds, "allowed"), None);
    assert_eq!(callback(&kinds, "error-with-output"), None);
    assert_eq!(callback(&kinds, "invalid-output"), None);
    assert_eq!(native_string_live_allocations(), live_before);

    let calls_before_allocation_failures = state.calls.lock().unwrap().len();
    fail_native_string_allocation_after(0);
    assert_eq!(callback(&kinds, "input-allocation-failure"), None);
    fail_native_string_allocation_after(1);
    assert_eq!(callback(&kinds, "name-allocation-failure"), None);
    assert_eq!(
        state.calls.lock().unwrap().len(),
        calls_before_allocation_failures,
        "the plugin callback must not run when host input allocation fails"
    );
    assert_eq!(native_string_live_allocations(), live_before);

    let calls = state.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 4);
    assert!(calls.iter().all(|(actual_kinds, _)| actual_kinds == &kinds));
    drop(calls);
    drop(callback);
    assert_eq!(state.frees.load(Ordering::SeqCst), 1);
}

#[test]
fn native_conditional_callback_registration_consumes_state_on_every_rejected_path() {
    let frees = AtomicUsize::new(0);
    let user_data = || (&frees as *const AtomicUsize).cast_mut().cast();
    let free_fn = Some(count_user_data_free as unsafe extern "C" fn(*mut c_void));
    let mut out = ptr::null_mut();

    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                ptr::null_mut(),
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    assert_eq!(
        unsafe {
            native_plugin_context_register_conditional_middleware_guardrail_callback(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(frees.load(Ordering::SeqCst), 2);

    let inactive_runtime = NativeHostPluginRuntime {
        namespace: "test:".into(),
        encode_local_names: false,
        instance: Weak::new(),
        active: AtomicBool::new(false),
        gates: Mutex::new(HashMap::new()),
    };
    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                (&inactive_runtime as *const NativeHostPluginRuntime).cast(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                &mut out,
            )
        },
        NemoRelayStatus::NotFound
    );
    assert_eq!(frees.load(Ordering::SeqCst), 3);

    let active_runtime = NativeHostPluginRuntime {
        namespace: "test:".into(),
        encode_local_names: false,
        instance: Weak::new(),
        active: AtomicBool::new(true),
        gates: Mutex::new(HashMap::new()),
    };
    let name = native_string("gate");
    let kinds = native_string(r#"["subscriber"]"#);
    let malformed_kinds = native_string("not-json");
    let registration_name = native_string("target");
    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                (&active_runtime as *const NativeHostPluginRuntime).cast(),
                ptr::null(),
                kinds,
                registration_name,
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(frees.load(Ordering::SeqCst), 4);

    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                (&active_runtime as *const NativeHostPluginRuntime).cast(),
                name,
                kinds,
                ptr::null(),
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                &mut out,
            )
        },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(frees.load(Ordering::SeqCst), 5);

    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                (&active_runtime as *const NativeHostPluginRuntime).cast(),
                name,
                malformed_kinds,
                registration_name,
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                &mut out,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(frees.load(Ordering::SeqCst), 6);

    assert_eq!(
        unsafe {
            native_plugin_runtime_register_conditional_middleware_guardrail_callback(
                (&active_runtime as *const NativeHostPluginRuntime).cast(),
                name,
                kinds,
                registration_name,
                unreachable_native_conditional_callback,
                user_data(),
                free_fn,
                &mut out,
            )
        },
        NemoRelayStatus::NotFound
    );
    assert_eq!(frees.load(Ordering::SeqCst), 7);
    unsafe {
        native_string_free(name);
        native_string_free(malformed_kinds);
        native_string_free(registration_name);
        native_string_free(kinds);
    }
}

#[test]
fn native_llm_sanitizer_input_allocation_failures_release_codec_ids() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let identity = LlmCodecIdentity::Runtime("com.example.chat.v1".into());
    let live_before = native_string_live_allocations();

    fail_native_string_allocation_after(1);
    let request_error = call_llm_sanitize_request_callback(
        llm_request_alias,
        ptr::null_mut(),
        &request,
        LlmSanitizeRequestContext::with_identity(identity.clone()),
    )
    .unwrap_err();
    assert!(
        request_error
            .to_string()
            .contains("failed to allocate native LLM request")
    );
    assert_eq!(native_string_live_allocations(), live_before);

    fail_native_string_allocation_after(1);
    let response_error = call_llm_sanitize_response_callback(
        llm_response_alias,
        ptr::null_mut(),
        &json!({"model": "test"}),
        LlmSanitizeResponseContext::with_identity(identity),
    )
    .unwrap_err();
    assert!(
        response_error
            .to_string()
            .contains("failed to allocate native LLM response")
    );
    assert_eq!(native_string_live_allocations(), live_before);
}

fn tool_next(output: FlowResult<Json>) -> ToolExecutionNextFn {
    let output = Arc::new(Mutex::new(Some(output)));
    Arc::new(move |_args| {
        let output = output.lock().unwrap().take().unwrap();
        Box::pin(async move { output.map(ToolExecutionResult::from) })
    })
}

fn llm_next(output: FlowResult<Json>) -> LlmExecutionNextFn {
    let output = Arc::new(Mutex::new(Some(output)));
    Arc::new(move |_request| {
        let output = output.lock().unwrap().take().unwrap();
        Box::pin(async move { output })
    })
}

#[test]
fn native_non_streaming_continuations_cover_success_and_error_paths() {
    let args = native_string(r#"{"value":1}"#);
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { native_tool_next(args, ptr::null_mut(), &mut out) },
        NemoRelayStatus::NullPointer
    );
    let next = Box::into_raw(Box::new(tool_next(Ok(json!({"result": 2}))))) as *mut c_void;
    assert_eq!(
        unsafe { native_tool_next(args, next, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"result": {"result": 2}})
    );
    unsafe { drop(Box::from_raw(next as *mut ToolExecutionNextFn)) };

    let next = Box::into_raw(Box::new(tool_next(Err(FlowError::NotFound(
        "missing".into(),
    ))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_tool_next(args, next, &mut out) },
        NemoRelayStatus::NotFound
    );
    unsafe { drop(Box::from_raw(next as *mut ToolExecutionNextFn)) };

    let panicking_next: ToolExecutionNextFn =
        Arc::new(|_| Box::pin(async { panic!("tool next panic") }));
    let next = Box::into_raw(Box::new(panicking_next)) as *mut c_void;
    assert_eq!(
        unsafe { native_tool_next(args, next, &mut out) },
        NemoRelayStatus::Internal
    );
    assert_last_error_contains("native tool next panicked");
    unsafe { drop(Box::from_raw(next as *mut ToolExecutionNextFn)) };
    unsafe { native_string_free(args) };

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();
    let next = Box::into_raw(Box::new(llm_next(Ok(json!({"answer": 42}))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_llm_next(request_json, next, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"answer": 42})
    );
    unsafe { drop(Box::from_raw(next as *mut LlmExecutionNextFn)) };

    let next = Box::into_raw(Box::new(llm_next(Err(FlowError::GuardrailRejected(
        "blocked".into(),
    ))))) as *mut c_void;
    out = ptr::null_mut();
    assert_eq!(
        unsafe { native_llm_next(request_json, next, &mut out) },
        NemoRelayStatus::GuardrailRejected
    );
    unsafe { drop(Box::from_raw(next as *mut LlmExecutionNextFn)) };

    let panicking_next: LlmExecutionNextFn =
        Arc::new(|_| Box::pin(async { panic!("LLM next panic") }));
    let next = Box::into_raw(Box::new(panicking_next)) as *mut c_void;
    assert_eq!(
        unsafe { native_llm_next(request_json, next, &mut out) },
        NemoRelayStatus::Internal
    );
    assert_last_error_contains("native LLM next panicked");
    unsafe {
        drop(Box::from_raw(next as *mut LlmExecutionNextFn));
        native_string_free(request_json);
    }
}

#[derive(Debug)]
enum NativeStreamItem {
    Json(Json),
    InvalidJson,
    Null,
    Error(NemoRelayStatus),
    ErrorWithJson(NemoRelayStatus),
    End,
    EndWithJson,
}

struct TestNativeStream {
    items: VecDeque<NativeStreamItem>,
    cancel_count: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
}

unsafe extern "C" fn test_native_stream_poll(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    let state = unsafe { &mut *(user_data as *mut TestNativeStream) };
    unsafe { *out_json = ptr::null_mut() };
    match state.items.pop_front().unwrap_or(NativeStreamItem::End) {
        NativeStreamItem::Json(value) => write_native_json(&value, out_json),
        NativeStreamItem::InvalidJson => {
            unsafe { *out_json = native_string("not-json") };
            NemoRelayStatus::Ok
        }
        NativeStreamItem::Null => NemoRelayStatus::Ok,
        NativeStreamItem::Error(status) => status,
        NativeStreamItem::ErrorWithJson(status) => {
            unsafe { *out_json = native_string(r#"{"discarded":true}"#) };
            status
        }
        NativeStreamItem::End => NemoRelayStatus::StreamEnd,
        NativeStreamItem::EndWithJson => {
            unsafe { *out_json = native_string(r#"{"discarded":true}"#) };
            NemoRelayStatus::StreamEnd
        }
    }
}

unsafe extern "C" fn test_native_stream_cancel(user_data: *mut c_void) -> NemoRelayStatus {
    let state = unsafe { &*(user_data as *const TestNativeStream) };
    state.cancel_count.fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn test_native_stream_drop(user_data: *mut c_void) {
    let state = unsafe { Box::from_raw(user_data as *mut TestNativeStream) };
    state.drop_count.fetch_add(1, Ordering::SeqCst);
}

fn test_native_stream(
    items: impl IntoIterator<Item = NativeStreamItem>,
) -> (
    NemoRelayNativeLlmStreamV1,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let cancel_count = Arc::new(AtomicUsize::new(0));
    let drop_count = Arc::new(AtomicUsize::new(0));
    let state = Box::new(TestNativeStream {
        items: items.into_iter().collect(),
        cancel_count: cancel_count.clone(),
        drop_count: drop_count.clone(),
    });
    (
        NemoRelayNativeLlmStreamV1 {
            struct_size: std::mem::size_of::<NemoRelayNativeLlmStreamV1>(),
            user_data: Box::into_raw(state).cast(),
            next: Some(test_native_stream_poll),
            cancel: Some(test_native_stream_cancel),
            drop: Some(test_native_stream_drop),
        },
        cancel_count,
        drop_count,
    )
}

#[tokio::test]
async fn native_stream_adapter_covers_chunks_end_errors_and_cancellation() {
    let (raw, cancel_count, drop_count) = test_native_stream([
        NativeStreamItem::Json(json!({"chunk": 1})),
        NativeStreamItem::End,
    ]);
    let mut stream = native_stream_to_relay_stream(raw, None, None).unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap(), json!({"chunk": 1}));
    assert!(stream.next().await.is_none());
    assert_eq!(cancel_count.load(Ordering::SeqCst), 0);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let (raw, cancel_count, drop_count) = test_native_stream([NativeStreamItem::Json(json!({
        "chunk": 2
    }))]);
    let stream = native_stream_to_relay_stream(raw, None, None).unwrap();
    drop(stream);
    assert_eq!(cancel_count.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    for item in [
        NativeStreamItem::InvalidJson,
        NativeStreamItem::Null,
        NativeStreamItem::Error(NemoRelayStatus::InvalidArg),
        NativeStreamItem::ErrorWithJson(NemoRelayStatus::InvalidArg),
    ] {
        let (raw, _, drop_count) = test_native_stream([item]);
        let mut stream = native_stream_to_relay_stream(raw, None, None).unwrap();
        assert!(stream.next().await.unwrap().is_err());
        assert!(stream.next().await.is_none());
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    let (mut raw, _, drop_count) = test_native_stream([]);
    raw.struct_size = 0;
    assert!(NativeRelayLlmStream::from_raw(raw, None, None).is_err());
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let (mut raw, _, drop_count) = test_native_stream([]);
    raw.next = None;
    assert!(NativeRelayLlmStream::from_raw(raw, None, None).is_err());
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let (raw, _, drop_count) = test_native_stream([NativeStreamItem::EndWithJson]);
    let mut stream = native_stream_to_relay_stream(raw, None, None).unwrap();
    assert!(stream.next().await.is_none());
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);

    let mut invalid = NativeRelayLlmStream {
        raw: NemoRelayNativeLlmStreamV1::default(),
        finished: false,
        _next_ctx: None,
        _callback_user_data: None,
    };
    assert!(invalid.next().await.unwrap().is_err());
    assert!(invalid.next().await.is_none());
}

#[tokio::test]
async fn relay_stream_adapter_covers_poll_end_error_and_cancel() {
    let stream = LlmJsonStream::new(tokio_stream::iter(vec![
        Ok(json!({"chunk": 1})),
        Err(FlowError::Internal("stream failed".into())),
    ]));
    let raw = relay_stream_to_native_stream(stream);
    let poll = raw.next.unwrap();
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"chunk": 1})
    );
    out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::Internal
    );
    drop_native_stream(raw);

    let stream = LlmJsonStream::new(tokio_stream::empty());
    let raw = relay_stream_to_native_stream(stream);
    let poll = raw.next.unwrap();
    out = ptr::null_mut();
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::StreamEnd
    );
    assert_eq!(
        unsafe { poll(raw.user_data, &mut out) },
        NemoRelayStatus::StreamEnd
    );
    assert_eq!(
        unsafe { cancel_relay_llm_stream(raw.user_data) },
        NemoRelayStatus::Ok
    );
    drop_native_stream(raw);

    assert_eq!(
        unsafe { poll_relay_llm_stream(ptr::null_mut(), ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    assert_eq!(
        unsafe { cancel_relay_llm_stream(ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
    unsafe { drop_relay_llm_stream(ptr::null_mut()) };

    let stream = LlmJsonStream::new(tokio_stream::empty());
    let raw = relay_stream_to_native_stream(stream);
    let state = unsafe { &*(raw.user_data as *const NativeHostLlmStream) };
    let mutex = state.stream.clone();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _guard = mutex.lock().unwrap();
        panic!("poison native stream lock");
    }));
    assert_eq!(
        unsafe { raw.next.unwrap()(raw.user_data, &mut out) },
        NemoRelayStatus::Internal
    );
    assert_eq!(
        unsafe { cancel_relay_llm_stream(raw.user_data) },
        NemoRelayStatus::Internal
    );
    drop_native_stream(raw);
}

#[test]
fn native_stream_continuation_covers_success_and_error() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "test"}),
    };
    let request_json = native_string_from_json(&serde_json::to_value(&request).unwrap()).unwrap();

    let next: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async {
            Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                "chunk": true
            }))])))
        })
    });
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    let mut raw = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe { native_llm_stream_next(request_json, next_ctx, &mut raw) },
        NemoRelayStatus::Ok
    );
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { raw.next.unwrap()(raw.user_data, &mut out) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        take_json_from_native_string(out, "unused").unwrap(),
        json!({"chunk": true})
    );
    drop_native_stream(raw);
    unsafe { drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn)) };

    let next: LlmStreamExecutionNextFn =
        Arc::new(|_request| Box::pin(async { Err(FlowError::NotFound("stream missing".into())) }));
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    raw = NemoRelayNativeLlmStreamV1::default();
    assert_eq!(
        unsafe { native_llm_stream_next(request_json, next_ctx, &mut raw) },
        NemoRelayStatus::NotFound
    );
    unsafe { drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn)) };

    let next: LlmStreamExecutionNextFn =
        Arc::new(|_| Box::pin(async { panic!("stream next panic") }));
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    assert_eq!(
        unsafe { native_llm_stream_next(request_json, next_ctx, &mut raw) },
        NemoRelayStatus::Internal
    );
    assert_last_error_contains("native LLM stream next panicked");
    unsafe {
        drop(Box::from_raw(next_ctx as *mut LlmStreamExecutionNextFn));
        native_string_free(request_json);
    }
    assert_eq!(
        unsafe { native_llm_stream_next(ptr::null(), ptr::null_mut(), ptr::null_mut()) },
        NemoRelayStatus::NullPointer
    );
}
