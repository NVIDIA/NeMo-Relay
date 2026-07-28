// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for callable private in the NeMo Relay FFI crate.

use super::*;

unsafe extern "C" fn complete_without_settling(
    _user_data: *mut libc::c_void,
    _invocation_json: *const c_char,
    _completion: *const NemoRelayAsyncCompletion,
) -> NemoRelayAsyncCallbackState {
    NemoRelayAsyncCallbackState::Complete
}

unsafe extern "C" fn retain_pending_completion(
    user_data: *mut libc::c_void,
    _invocation_json: *const c_char,
    completion: *const NemoRelayAsyncCompletion,
) -> NemoRelayAsyncCallbackState {
    let slot = unsafe { &*user_data.cast::<std::sync::atomic::AtomicUsize>() };
    slot.store(completion as usize, Ordering::Release);
    NemoRelayAsyncCallbackState::Pending
}

unsafe extern "C" fn send_next_result(
    user_data: *mut libc::c_void,
    value_json: *const c_char,
    error_message: *const c_char,
) {
    let sender = unsafe {
        Box::from_raw(
            user_data.cast::<tokio::sync::oneshot::Sender<std::result::Result<Json, String>>>(),
        )
    };
    let result = if error_message.is_null() {
        serde_json::from_str(unsafe { CStr::from_ptr(value_json) }.to_str().unwrap())
            .map_err(|error| error.to_string())
    } else {
        Err(unsafe { CStr::from_ptr(error_message) }
            .to_string_lossy()
            .into_owned())
    };
    let _ = sender.send(result);
}

#[test]
fn test_callable_private_helper_paths() {
    clear_last_error();
    let err = json_result_from_ptr(std::ptr::null_mut(), "fallback helper message").unwrap_err();
    assert!(err.to_string().contains("fallback helper message"));

    assert_eq!(ptr_to_opt_string(std::ptr::null_mut()), None);

    let raw = CString::new("ffi-string").unwrap().into_raw();
    assert_eq!(ptr_to_opt_string(raw), Some("ffi-string".into()));
    unsafe { nemo_relay_string_free_internal(raw) };
}

#[test]
fn async_completion_abi_rejects_invalid_duplicate_and_cancelled_settlements() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NemoRelayAsyncCompletion {
        sender: std::sync::Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
    });
    let completion_ref = Arc::into_raw(Arc::clone(&completion));
    let invalid_json = CString::new("not-json").unwrap();
    assert_eq!(
        unsafe { nemo_relay_async_completion_resolve_json(completion_ref, invalid_json.as_ptr()) },
        NemoRelayStatus::InvalidJson
    );
    let value = CString::new(r#"{"ok":true}"#).unwrap();
    assert_eq!(
        unsafe { nemo_relay_async_completion_resolve_json(completion_ref, value.as_ptr()) },
        NemoRelayStatus::Ok
    );
    assert_eq!(
        unsafe { nemo_relay_async_completion_resolve_json(completion_ref, value.as_ptr()) },
        NemoRelayStatus::InvalidArg
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    assert_eq!(
        runtime.block_on(receiver).unwrap().unwrap(),
        serde_json::json!({"ok": true})
    );
    unsafe { nemo_relay_async_completion_release(completion_ref) };

    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NemoRelayAsyncCompletion {
        sender: std::sync::Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
    });
    let completion_ref = Arc::into_raw(Arc::clone(&completion));
    assert_eq!(
        unsafe { nemo_relay_async_completion_reject(completion_ref, std::ptr::null()) },
        NemoRelayStatus::Ok
    );
    assert!(
        runtime
            .block_on(receiver)
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("async C callback rejected")
    );
    unsafe { nemo_relay_async_completion_release(completion_ref) };

    let retained_completion = std::sync::atomic::AtomicUsize::new(0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut invocation = Box::pin(invoke_async_json(
            retain_pending_completion,
            Arc::new(UserData {
                ptr: (&retained_completion as *const std::sync::atomic::AtomicUsize)
                    .cast_mut()
                    .cast(),
                free_fn: None,
            }),
            serde_json::json!({}),
        ));
        tokio::select! {
            biased;
            result = &mut invocation => panic!("pending callback unexpectedly settled: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        drop(invocation);
    });
    let completion_ref =
        retained_completion.load(Ordering::Acquire) as *const NemoRelayAsyncCompletion;
    assert!(!completion_ref.is_null());
    assert!(unsafe { nemo_relay_async_completion_is_cancelled(completion_ref) });
    assert!(unsafe { nemo_relay_async_completion_is_cancelled(std::ptr::null()) });
    let value = CString::new(r#"{"late":true}"#).unwrap();
    assert_eq!(
        unsafe { nemo_relay_async_completion_resolve_json(completion_ref, value.as_ptr()) },
        NemoRelayStatus::InvalidArg
    );
    assert_eq!(
        unsafe { nemo_relay_async_completion_reject(completion_ref, std::ptr::null()) },
        NemoRelayStatus::InvalidArg
    );
    unsafe { nemo_relay_async_completion_release(completion_ref) };
}

#[test]
fn async_callback_wrappers_reject_complete_callbacks_without_settlement() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(invoke_async_json(
            complete_without_settling,
            Arc::new(UserData {
                ptr: std::ptr::null_mut(),
                free_fn: None,
            }),
            serde_json::json!({}),
        ))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("returned Complete without settling")
    );
}

#[test]
fn async_next_invocation_supports_tool_llm_and_stream_continuations() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let cases: Vec<(AsyncNextInner, CString, serde_json::Value)> = vec![
        (
            AsyncNextInner::Tool(Arc::new(|value| Box::pin(async move { Ok(value) }))),
            CString::new(r#"{"tool":true}"#).unwrap(),
            serde_json::json!({"tool": true}),
        ),
        (
            AsyncNextInner::Llm(Arc::new(|request| {
                Box::pin(async move { Ok(request.content) })
            })),
            CString::new(
                serde_json::to_string(&LlmRequest {
                    headers: serde_json::Map::new(),
                    content: serde_json::json!({"llm": true}),
                })
                .unwrap(),
            )
            .unwrap(),
            serde_json::json!({"llm": true}),
        ),
        (
            AsyncNextInner::LlmStream(Arc::new(|_request| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![
                        Ok(serde_json::json!({"chunk": 1})),
                        Ok(serde_json::json!({"chunk": 2})),
                    ])))
                })
            })),
            CString::new(
                serde_json::to_string(&LlmRequest {
                    headers: serde_json::Map::new(),
                    content: serde_json::json!({"stream": true}),
                })
                .unwrap(),
            )
            .unwrap(),
            serde_json::json!([{"chunk": 1}, {"chunk": 2}]),
        ),
    ];

    for (inner, invocation, expected) in cases {
        let next = Arc::new(NemoRelayAsyncNext {
            inner,
            runtime: runtime.handle().clone(),
        });
        let next_ref = Arc::into_raw(next);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let completion = Arc::new(NemoRelayAsyncCompletion {
            sender: std::sync::Mutex::new(Some(sender)),
            cancelled: AtomicBool::new(false),
        });
        let completion_ref = Arc::into_raw(Arc::clone(&completion));
        assert_eq!(
            unsafe { nemo_relay_async_next_invoke(next_ref, invocation.as_ptr(), completion_ref) },
            NemoRelayStatus::Ok
        );
        assert_eq!(runtime.block_on(receiver).unwrap().unwrap(), expected);
        unsafe {
            nemo_relay_async_next_release(next_ref);
            nemo_relay_async_completion_release(completion_ref);
        }
    }
}

#[test]
fn async_next_callback_reports_tool_llm_and_stream_results() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases: Vec<(AsyncNextInner, CString, Json)> = vec![
        (
            AsyncNextInner::Tool(Arc::new(|value| Box::pin(async move { Ok(value) }))),
            CString::new(r#"{"tool":true}"#).unwrap(),
            serde_json::json!({"tool": true}),
        ),
        (
            AsyncNextInner::Llm(Arc::new(|request| {
                Box::pin(async move { Ok(request.content) })
            })),
            CString::new(r#"{"headers":{},"content":{"llm":true}}"#).unwrap(),
            serde_json::json!({"llm": true}),
        ),
        (
            AsyncNextInner::LlmStream(Arc::new(|_request| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                        serde_json::json!({"stream": true}),
                    )])))
                })
            })),
            CString::new(r#"{"headers":{},"content":{}}"#).unwrap(),
            serde_json::json!([{ "stream": true }]),
        ),
    ];
    for (inner, invocation, expected) in cases {
        let next = Arc::new(NemoRelayAsyncNext {
            inner,
            runtime: runtime.handle().clone(),
        });
        let next_ref = Arc::into_raw(next);
        let (sender, receiver) =
            tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
        assert_eq!(
            unsafe {
                nemo_relay_async_next_invoke_callback(
                    next_ref,
                    invocation.as_ptr(),
                    send_next_result,
                    Box::into_raw(Box::new(sender)).cast(),
                )
            },
            NemoRelayStatus::Ok
        );
        assert_eq!(runtime.block_on(receiver).unwrap().unwrap(), expected);
        unsafe { nemo_relay_async_next_release(next_ref) };
    }

    let next = Arc::new(NemoRelayAsyncNext {
        inner: AsyncNextInner::Tool(Arc::new(|_value| {
            Box::pin(async { Err(FlowError::Internal("next failed".into())) })
        })),
        runtime: runtime.handle().clone(),
    });
    let next_ref = Arc::into_raw(next);
    let invocation = CString::new("{}").unwrap();
    let (sender, receiver) = tokio::sync::oneshot::channel::<std::result::Result<Json, String>>();
    assert_eq!(
        unsafe {
            nemo_relay_async_next_invoke_callback(
                next_ref,
                invocation.as_ptr(),
                send_next_result,
                Box::into_raw(Box::new(sender)).cast(),
            )
        },
        NemoRelayStatus::Ok
    );
    assert!(
        runtime
            .block_on(receiver)
            .unwrap()
            .unwrap_err()
            .contains("next failed")
    );
    unsafe { nemo_relay_async_next_release(next_ref) };
}
