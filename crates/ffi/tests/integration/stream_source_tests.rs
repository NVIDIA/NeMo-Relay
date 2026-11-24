// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::{
    ptr,
    sync::{
        Arc, Condvar,
        atomic::{AtomicUsize, Ordering},
    },
};

struct SourceState {
    index: AtomicUsize,
    started: Option<std::sync::mpsc::Sender<()>>,
    closed: Arc<AtomicUsize>,
    freed: Arc<AtomicUsize>,
    blocking: bool,
    cancelled: Mutex<bool>,
    wake: Condvar,
}
unsafe extern "C" fn read(data: *mut libc::c_void, output: *mut *mut c_char) -> i32 {
    let state = unsafe { &*(data as *const SourceState) };
    if state.blocking {
        if let Some(started) = &state.started {
            let _ = started.send(());
        }
        let mut cancelled = state.cancelled.lock().unwrap();
        while !*cancelled {
            cancelled = state.wake.wait(cancelled).unwrap();
        }
        return 0;
    }
    let index = state.index.fetch_add(1, Ordering::SeqCst);
    if index >= 3 {
        return 0;
    }
    unsafe { *output = json_to_c_string(&json!(index)) };
    1
}
unsafe extern "C" fn cancel(data: *mut libc::c_void) {
    let state = unsafe { &*(data as *const SourceState) };
    *state.cancelled.lock().unwrap() = true;
    state.wake.notify_all();
}
unsafe extern "C" fn close(data: *mut libc::c_void) -> NemoRelayStatus {
    unsafe { &*(data as *const SourceState) }
        .closed
        .fetch_add(1, Ordering::SeqCst);
    NemoRelayStatus::Ok
}
unsafe extern "C" fn free(data: *mut libc::c_void) {
    let state = unsafe { Box::from_raw(data as *mut SourceState) };
    state.freed.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn owned_source_preserves_all_chunks_and_closes_once() {
    let closed = Arc::new(AtomicUsize::new(0));
    let freed = Arc::new(AtomicUsize::new(0));
    let state = Box::new(SourceState {
        index: AtomicUsize::new(0),
        started: None,
        closed: closed.clone(),
        freed: freed.clone(),
        blocking: false,
        cancelled: Mutex::new(false),
        wake: Condvar::new(),
    });
    unsafe {
        let source = nemo_relay_stream_source_create(
            Some(read),
            Some(cancel),
            Some(close),
            Box::into_raw(state).cast(),
            Some(free),
        );
        for index in 0..3 {
            let mut output = ptr::null_mut();
            assert_eq!(nemo_relay_stream_source_next(source, &mut output), 1);
            assert_eq!(
                serde_json::from_str::<Json>(&CStr::from_ptr(output).to_string_lossy()).unwrap(),
                json!(index)
            );
            nemo_relay_string_free(output);
        }
        let mut output = ptr::null_mut();
        assert_eq!(nemo_relay_stream_source_next(source, &mut output), 0);
        assert_eq!(nemo_relay_stream_source_close(source), NemoRelayStatus::Ok);
        assert_eq!(nemo_relay_stream_source_close(source), NemoRelayStatus::Ok);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(freed.load(Ordering::SeqCst), 1);
        nemo_relay_stream_source_free(source);
    }
}

#[test]
fn source_cancellation_joins_blocked_read_before_freeing_callback_state() {
    let closed = Arc::new(AtomicUsize::new(0));
    let freed = Arc::new(AtomicUsize::new(0));
    let (started, waiting) = std::sync::mpsc::channel();
    let state = Box::new(SourceState {
        index: AtomicUsize::new(0),
        started: Some(started),
        closed: closed.clone(),
        freed: freed.clone(),
        blocking: true,
        cancelled: Mutex::new(false),
        wake: Condvar::new(),
    });
    unsafe {
        let source = nemo_relay_stream_source_create(
            Some(read),
            Some(cancel),
            Some(close),
            Box::into_raw(state).cast(),
            Some(free),
        );
        let address = source as usize;
        let reader = std::thread::spawn(move || {
            let mut output = ptr::null_mut();
            nemo_relay_stream_source_next(address as *const FfiLlmStreamSource, &mut output)
        });
        waiting
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        nemo_relay_stream_source_cancel(source);
        assert_eq!(nemo_relay_stream_source_close(source), NemoRelayStatus::Ok);
        assert_eq!(reader.join().unwrap(), 0);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(freed.load(Ordering::SeqCst), 1);
        nemo_relay_stream_source_free(source);
    }
}

#[test]
fn retained_recorder_outlives_the_borrowed_wrapper() {
    use nemo_relay_runtime::api::optimization::LlmOptimizationRecorder;
    unsafe {
        let borrowed = FfiLlmOptimizationRecorder(LlmOptimizationRecorder::default());
        let recorder = nemo_relay_llm_optimization_recorder_retain(&borrowed);
        drop(borrowed);
        let mut accepted = false;
        assert_eq!(
            nemo_relay_llm_optimization_record(
                recorder,
                c"{\"producer\":\"ffi-deferred\",\"kind\":\"custom\",\"applied\":true}".as_ptr(),
                &mut accepted
            ),
            NemoRelayStatus::Ok
        );
        assert!(accepted);
        nemo_relay_llm_optimization_recorder_free(recorder);
    }
}

unsafe extern "C" fn passthrough(
    _data: *mut libc::c_void,
    _name: *const c_char,
    request: *const c_char,
    _context: NemoRelayLlmExecutionContext,
    next: NemoRelayLlmExecNextFn,
    next_context: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next(request, next_context) }
}
#[test]
fn single_value_intercept_adapter_preserves_the_upstream_tail() {
    use tokio_stream::StreamExt;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let intercept = wrap_llm_stream_exec_intercept_fn(passthrough, ptr::null_mut(), None);
    let next: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async {
            Ok(nemo_relay_runtime::api::runtime::LlmJsonStream::new(
                tokio_stream::iter(vec![Ok(json!(1)), Ok(json!(2)), Ok(json!(3))]),
            ))
        })
    });
    runtime.block_on(async {
        let mut stream = intercept(
            "adapter",
            LlmRequest {
                headers: Default::default(),
                content: Json::Null,
            },
            nemo_relay_runtime::api::runtime::LlmExecutionContext::default(),
            next,
        )
        .await
        .unwrap();
        for expected in 1..=3 {
            assert_eq!(stream.next().await.unwrap().unwrap(), json!(expected));
        }
        assert!(stream.next().await.is_none());
        stream.close().await.unwrap();
    });
}

#[test]
fn source_close_emits_events_in_its_creation_context_without_a_read() {
    use nemo_relay_runtime::api::runtime::{
        TASK_SCOPE_STACK, create_scope_stack, with_active_event_uuid,
    };
    use nemo_relay_runtime::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    unsafe extern "C" fn eof(_: *mut libc::c_void, _: *mut *mut c_char) -> i32 {
        0
    }
    unsafe extern "C" fn emit(_: *mut libc::c_void) -> NemoRelayStatus {
        unsafe {
            nemo_relay_event(
                c"ffi-source-cleanup-context".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            )
        }
    }
    let (sent, received) = std::sync::mpsc::channel();
    register_subscriber(
        "ffi-source-cleanup-context",
        Arc::new(move |event| {
            if event.name() == "ffi-source-cleanup-context" {
                let _ = sent.send(event.parent_uuid());
            }
        }),
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let expected_event = uuid::Uuid::now_v7();
    let source = runtime.block_on(TASK_SCOPE_STACK.scope(
        create_scope_stack(),
        with_active_event_uuid(expected_event, async {
            unsafe {
                nemo_relay_stream_source_create(Some(eof), None, Some(emit), ptr::null_mut(), None)
            }
        }),
    ));
    unsafe {
        assert_eq!(nemo_relay_stream_source_close(source), NemoRelayStatus::Ok);
        nemo_relay_stream_source_free(source);
    }
    flush_subscribers().unwrap();
    deregister_subscriber("ffi-source-cleanup-context").unwrap();
    assert_eq!(
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        Some(expected_event)
    );
}

struct FailingCleanupStream {
    closed: Arc<AtomicUsize>,
    failure: FlowError,
}
impl tokio_stream::Stream for FailingCleanupStream {
    type Item = Result<Json, FlowError>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(Some(Ok(json!(1))))
    }
}
impl nemo_relay_types::api::runtime::LlmStreamInner for FailingCleanupStream {
    fn close(
        self: std::pin::Pin<&mut Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), FlowError>> + Send + '_>>
    {
        self.closed.fetch_add(1, Ordering::SeqCst);
        let failure = self.failure.clone();
        Box::pin(async move { Err(failure) })
    }
}
unsafe extern "C" fn fail_after_next(
    _: *mut libc::c_void,
    _: *const c_char,
    request: *const c_char,
    _: NemoRelayLlmExecutionContext,
    next: NemoRelayLlmExecNextFn,
    next_context: *mut libc::c_void,
) -> *mut c_char {
    let value = unsafe { next(request, next_context) };
    unsafe { nemo_relay_string_free(value) };
    set_last_error("interceptor failed");
    unsafe {
        nemo_relay_set_last_error_cause_ids([11, 12].as_ptr(), 2);
    }
    ptr::null_mut()
}

#[test]
fn single_value_adapter_preserves_interceptor_and_cleanup_failures() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    unsafe extern "C" fn cleanup_failure(_: *mut libc::c_void, _: *const c_char) -> *mut c_char {
        set_last_error("source cleanup failed");
        unsafe {
            nemo_relay_set_last_error_cause_ids([12, 13].as_ptr(), 2);
        }
        ptr::null_mut()
    }
    let callback = wrap_llm_exec_fn(cleanup_failure, ptr::null_mut(), None);
    let failure = runtime
        .block_on(callback(LlmRequest {
            headers: Default::default(),
            content: Json::Null,
        }))
        .unwrap_err();
    let closed = Arc::new(AtomicUsize::new(0));
    let source_closed = closed.clone();
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let closed = source_closed.clone();
        let failure = failure.clone();
        Box::pin(async move {
            Ok(
                nemo_relay_runtime::api::runtime::LlmJsonStream::from_closeable(
                    FailingCleanupStream { closed, failure },
                ),
            )
        })
    });
    let intercept = wrap_llm_stream_exec_intercept_fn(fail_after_next, ptr::null_mut(), None);
    let result = runtime.block_on(intercept(
        "cleanup-failure",
        LlmRequest {
            headers: Default::default(),
            content: Json::Null,
        },
        nemo_relay_runtime::api::runtime::LlmExecutionContext::default(),
        next,
    ));
    let Err(error) = result else {
        panic!("expected interceptor failure")
    };
    let message = error.to_string();
    assert!(message.contains("interceptor failed"), "{message}");
    assert!(message.contains("source cleanup failed"), "{message}");
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert_eq!(
        message,
        "internal error: internal error: interceptor failed; stream cleanup also failed: internal error: source cleanup failed"
    );
    assert_eq!(status_from_error(&error), NemoRelayStatus::Internal);
    let mut count = 0;
    let ids = unsafe { nemo_relay_last_error_cause_ids(&mut count) };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(ids, count) },
        &[11, 12, 13]
    );
}

#[test]
fn missing_provider_releases_all_transferred_callback_state() {
    unsafe extern "C" fn release(data: *mut libc::c_void) {
        let calls = unsafe { Box::from_raw(data.cast::<Arc<AtomicUsize>>()) };
        calls.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" fn decode(_: *mut libc::c_void, _: *const FfiLLMRequest) -> *mut c_char {
        ptr::null_mut()
    }
    unsafe extern "C" fn encode(
        _: *mut libc::c_void,
        _: *const c_char,
        _: *const FfiLLMRequest,
    ) -> *mut c_char {
        ptr::null_mut()
    }
    let frees = [
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ];
    let state = |index: usize| Box::into_raw(Box::new(frees[index].clone())).cast::<libc::c_void>();
    let mut stream = ptr::null_mut();
    let status = unsafe {
        nemo_relay_llm_stream_call_execute(
            c"missing-provider".as_ptr(),
            c"{\"headers\":{},\"content\":{}}".as_ptr(),
            None,
            None,
            state(0),
            Some(release),
            None,
            None,
            state(1),
            Some(release),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            Some(decode),
            Some(encode),
            state(2),
            Some(release),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            &mut stream,
        )
    };
    assert_eq!(status, NemoRelayStatus::InvalidArg);
    assert!(stream.is_null());
    for count in &frees {
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn source_cleanup_failure_survives_cancellation_and_repeated_close() {
    unsafe extern "C" fn fail(data: *mut libc::c_void) -> NemoRelayStatus {
        unsafe { close(data) };
        set_last_error("provider cleanup failed");
        NemoRelayStatus::Internal
    }
    let closed = Arc::new(AtomicUsize::new(0));
    let freed = Arc::new(AtomicUsize::new(0));
    let state = Box::new(SourceState {
        index: AtomicUsize::new(0),
        started: None,
        closed: closed.clone(),
        freed: freed.clone(),
        blocking: false,
        cancelled: Mutex::new(false),
        wake: Condvar::new(),
    });
    unsafe {
        let source = nemo_relay_stream_source_create(
            Some(read),
            Some(cancel),
            Some(fail),
            Box::into_raw(state).cast(),
            Some(free),
        );
        nemo_relay_stream_source_cancel(source);
        for _ in 0..2 {
            assert_eq!(
                nemo_relay_stream_source_close(source),
                NemoRelayStatus::Internal
            );
            assert!(
                last_error_message()
                    .unwrap()
                    .contains("provider cleanup failed")
            );
        }
        nemo_relay_stream_source_free(source);
    }
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert_eq!(freed.load(Ordering::SeqCst), 1);
}

#[test]
fn invalid_stream_registrations_release_callback_state() {
    unsafe extern "C" fn count_free(data: *mut libc::c_void) {
        unsafe { &*(data as *const AtomicUsize) }.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" fn source_intercept(
        _: *mut libc::c_void,
        _: *const c_char,
        _: *const c_char,
        _: NemoRelayLlmExecutionContext,
        _: NemoRelayLlmStreamExecNextFn,
        _: *mut libc::c_void,
    ) -> *mut FfiLlmStreamSource {
        ptr::null_mut()
    }
    let invalid_utf8 = [0xff_u8, 0];
    for stream_callback in [
        None,
        Some(source_intercept as NemoRelayLlmStreamExecInterceptCb),
    ] {
        let freed = AtomicUsize::new(0);
        let data = (&freed as *const AtomicUsize).cast_mut().cast();
        unsafe {
            assert_eq!(
                nemo_relay_register_llm_stream_execution_intercept(
                    invalid_utf8.as_ptr().cast(),
                    0,
                    None,
                    stream_callback,
                    data,
                    Some(count_free)
                ),
                NemoRelayStatus::InvalidUtf8
            );
            assert_eq!(freed.load(Ordering::SeqCst), 1);
            assert_eq!(
                nemo_relay_scope_register_llm_stream_execution_intercept(
                    c"not-a-uuid".as_ptr(),
                    c"name".as_ptr(),
                    0,
                    None,
                    stream_callback,
                    data,
                    Some(count_free)
                ),
                NemoRelayStatus::InvalidArg
            );
            assert_eq!(freed.load(Ordering::SeqCst), 2);
            assert_eq!(
                nemo_relay_scope_register_llm_stream_execution_intercept(
                    c"00000000-0000-0000-0000-000000000000".as_ptr(),
                    invalid_utf8.as_ptr().cast(),
                    0,
                    None,
                    stream_callback,
                    data,
                    Some(count_free)
                ),
                NemoRelayStatus::InvalidUtf8
            );
            assert_eq!(freed.load(Ordering::SeqCst), 3);
            assert_eq!(
                nemo_relay_plugin_context_register_llm_stream_execution_intercept(
                    ptr::null_mut(),
                    c"name".as_ptr(),
                    0,
                    None,
                    stream_callback,
                    data,
                    Some(count_free)
                ),
                NemoRelayStatus::NullPointer
            );
            assert_eq!(freed.load(Ordering::SeqCst), 4);
        }
    }
}

#[test]
fn source_close_retains_creation_scope_after_stack_changes() {
    use nemo_relay_runtime::api::runtime::{TASK_SCOPE_STACK, create_scope_stack};
    use nemo_relay_runtime::api::scope::{PushScopeParams, push_scope};
    use nemo_relay_runtime::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    unsafe extern "C" fn eof(_: *mut libc::c_void, _: *mut *mut c_char) -> i32 {
        0
    }
    unsafe extern "C" fn emit(_: *mut libc::c_void) -> NemoRelayStatus {
        unsafe {
            nemo_relay_event(
                c"source-creation-scope".as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
            )
        }
    }
    let (sent, received) = std::sync::mpsc::channel();
    register_subscriber(
        "source-creation-scope",
        Arc::new(move |event| {
            if event.name() == "source-creation-scope" {
                let _ = sent.send(event.parent_uuid());
            }
        }),
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let expected = runtime.block_on(TASK_SCOPE_STACK.scope(create_scope_stack(), async {
        let scope_a = push_scope(
            PushScopeParams::builder()
                .name("source-scope-a")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let source = unsafe {
            nemo_relay_stream_source_create(Some(eof), None, Some(emit), ptr::null_mut(), None)
        };
        assert!(!source.is_null());
        let _scope_b = push_scope(
            PushScopeParams::builder()
                .name("source-scope-b")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        unsafe {
            assert_eq!(nemo_relay_stream_source_close(source), NemoRelayStatus::Ok);
            nemo_relay_stream_source_free(source);
        }
        scope_a.uuid
    }));
    flush_subscribers().unwrap();
    deregister_subscriber("source-creation-scope").unwrap();
    assert_eq!(
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        Some(expected)
    );
}
