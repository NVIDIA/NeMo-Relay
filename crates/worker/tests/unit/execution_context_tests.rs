// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn endpoint_announcement_creates_parents_and_reports_conflicting_paths() {
    let directory = std::env::temp_dir().join(format!(
        "relay-worker-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let endpoint = "http://127.0.0.1:4317";
    let path = directory.as_path().join("nested/endpoint.txt");
    write_endpoint_file(&path, endpoint).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), endpoint);
    let occupied = directory.as_path().join("occupied");
    std::fs::write(&occupied, b"existing file").unwrap();
    let error = write_endpoint_file(&occupied.join("endpoint.txt"), endpoint).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("failed to create worker endpoint file directory")
    );
    let error = write_endpoint_file(directory.as_path(), endpoint).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("failed to write worker endpoint file")
    );
    assert_eq!(std::fs::read(&occupied).unwrap(), b"existing file");
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(unix)]
#[test]
fn worker_socket_cleanup_removes_only_socket_files() {
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixListener;

    let directory = std::env::temp_dir().join(format!(
        "relay-worker-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let socket = directory.as_path().join("worker.sock");
    remove_stale_socket(&socket).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    drop(listener);
    remove_stale_socket(&socket).unwrap();
    assert!(!socket.exists());
    let regular = directory.as_path().join("regular");
    std::fs::write(&regular, b"preserved").unwrap();
    let linked = directory.as_path().join("linked");
    symlink(&regular, &linked).unwrap();
    for path in [&regular, &linked, &directory] {
        assert!(
            remove_stale_socket(path)
                .unwrap_err()
                .to_string()
                .contains("exists and is not a socket")
        );
    }
    assert_eq!(std::fs::read(&regular).unwrap(), b"preserved");
    assert!(
        std::fs::symlink_metadata(linked)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// Builds a finite-limit runtime whose host RPCs cannot connect during local tests.
fn disconnected_runtime() -> PluginRuntime {
    PluginRuntime {
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "http://127.0.0.1:1".into(),
        host_channel: Arc::new(OnceCell::new()),
        max_message_bytes: nemo_relay_worker_proto::WorkerCallbackMessageLimits::default(),
        callback_name: None,
        diagnostics: Arc::new(OnceCell::new()),
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    }
}

#[tokio::test]
async fn callback_context_follows_stream_polling_without_widening_sibling_tasks() {
    use futures_util::StreamExt;
    use serde_json::json;

    async fn exercise(name: &str) {
        let mut stream = TASK_CALLBACK_NAME
            .scope(Some(name.to_owned()), async {
                ScopedJsonStream::new(
                    Box::pin(futures_util::stream::once(async {
                        tokio::task::yield_now().await;
                        with_thread_scope(&None, || Ok(json!(current_callback_name())))
                    })),
                    None,
                )
            })
            .await;
        TASK_CALLBACK_NAME
            .scope(Some("other_callback".into()), async {
                assert_eq!(stream.next().await.unwrap().unwrap(), json!(name));
                assert_eq!(current_callback_name().as_deref(), Some("other_callback"));
            })
            .await;
        assert!(current_callback_name().is_none());
    }

    tokio::join!(exercise("large.guardrail-α"), exercise("small_guardrail"));
    assert!(THREAD_CALLBACK_NAME.with(|name| name.borrow().is_none()));
}

fn llm_payload(
    execution_codec_context: Option<Box<nemo_relay_worker_proto::v1::LlmExecutionCodecContext>>,
) -> LlmPayload {
    LlmPayload {
        model_name: "model".into(),
        request: None,
        annotated_request: None,
        response: None,
        sanitize_context: None,
        execution_codec_context,
    }
}

#[test]
fn malformed_execution_context_fields_are_rejected() {
    let request = || nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
        codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity::default()),
        codec_capability_id: None,
    };
    let response = || nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
        codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity::default()),
        codec_capability_id: None,
    };
    let cases = [
        (None, "execution context is missing"),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: None,
                response: Some(response()),
            }),
            "request context is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext::default()),
                response: Some(response()),
            }),
            "request codec identity is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext::default()),
            }),
            "response codec identity is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: None,
            }),
            "response context is missing",
        ),
    ];

    for (context, expected) in cases {
        let error = llm_payload(context.map(Box::new))
            .execution_context(&disconnected_runtime(), "invocation", true)
            .unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected '{expected}' in '{error}'"
        );
    }
}

#[test]
fn execution_context_preserves_codec_identities_and_capability_availability() {
    let cases = [
        (
            LlmCodecKind::Unspecified,
            None,
            None,
            LlmCodecIdentity::None,
            false,
        ),
        (
            LlmCodecKind::Opaque,
            None,
            Some("opaque"),
            LlmCodecIdentity::Opaque,
            true,
        ),
        (
            LlmCodecKind::Builtin,
            Some("openai_chat"),
            Some("builtin"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
            true,
        ),
    ];

    for (kind, id, capability, expected, resolves) in cases {
        let codec = || nemo_relay_worker_proto::v1::LlmCodecIdentity {
            kind: kind as i32,
            id: id.map(str::to_owned),
        };
        let capability = |direction: &str| capability.map(|value| format!("{value}-{direction}"));
        let context = llm_payload(Some(Box::new(
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                    codec: Some(codec()),
                    codec_capability_id: capability("request"),
                }),
                response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
                    codec: Some(codec()),
                    codec_capability_id: capability("response"),
                }),
            },
        )))
        .execution_context(&disconnected_runtime(), "invocation", true)
        .unwrap();

        let debug = format!("{context:?}");
        assert!(debug.contains("request_codec"));
        assert!(debug.contains("response_codec"));
        assert!(!debug.contains("-request"));
        assert!(!debug.contains("-response"));
        assert_eq!(context.request_codec().codec, expected);
        assert_eq!(context.request_codec().resolve_codec().is_some(), resolves);
        let response = context.response_codec().expect("unary response context");
        assert_eq!(response.codec, expected);
        assert_eq!(response.resolve_codec().is_some(), resolves);
    }
}

#[test]
fn streaming_execution_context_has_no_response_codec() {
    let payload = llm_payload(Some(Box::new(
        nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
            request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Builtin as i32,
                    id: Some("openai_chat".into()),
                }),
                codec_capability_id: Some("request-capability".into()),
            }),
            response: None,
        },
    )));

    let context = payload
        .execution_context(&disconnected_runtime(), "invocation", false)
        .unwrap();
    assert!(context.request_codec().resolve_codec().is_some());
    assert!(context.response_codec().is_none());
}

#[test]
fn worker_sdk_http_error_prefix_survives_callback_and_next_round_trip() {
    for status in 400..=599 {
        let message = http_error_message(status, "provider failure").unwrap();
        let wire = sdk_error_to_worker(WorkerSdkError::Callback(message.clone()));
        assert_eq!(wire.message, message);
        assert!(!wire.retryable);
        let returned = sdk_error_to_worker(worker_error_to_sdk(wire));
        assert_eq!(returned.message, message);
    }
    let ordinary = sdk_error_to_worker(WorkerSdkError::Callback("plain failure".into()));
    assert_eq!(ordinary.message, "callback failed: plain failure");
}

#[test]
fn worker_sdk_cancellation_keeps_its_code_even_with_http_prefix() {
    let wire = WorkerError {
        code: "worker.cancelled".into(),
        message: "HTTP 403: denied".into(),
        retryable: false,
        downstream_continuation_id: None,
    };
    let returned = sdk_error_to_worker(worker_error_to_sdk(wire));
    assert!(
        returned
            .message
            .starts_with("callback failed: worker.cancelled:")
    );
}

#[tokio::test]
async fn cancellation_acknowledgement_matches_acceptance_after_cleanup() {
    struct TestPlugin;

    impl WorkerPlugin for TestPlugin {
        fn plugin_id(&self) -> &str {
            "test.cancellation"
        }

        fn register(&self, _ctx: &mut PluginContext, _config: &Json) -> Result<()> {
            Ok(())
        }
    }

    for repeated in [false, true] {
        for reason in ["", "stream abandoned"] {
            let task = tokio::spawn(std::future::pending::<()>());
            let (completion_tx, completion_rx) = watch::channel(false);
            let (shutdown, _) = watch::channel(None);
            let service = WorkerService {
                plugin: Arc::new(TestPlugin),
                runtime: disconnected_runtime(),
                handlers: Arc::new(Mutex::new(WorkerHandlers::default())),
                active_invocations: Arc::new(Mutex::new(HashMap::from([(
                    "invocation".into(),
                    ActiveInvocation {
                        generation: 0,
                        abort_handle: task.abort_handle(),
                        stream_cancel: None,
                        completion: completion_rx,
                        cancel_requested: repeated,
                    },
                )]))),
                next_invocation_generation: Arc::new(AtomicU64::new(1)),
                shutdown,
            };
            let cancellation = service.cancel_invocation(Request::new(CancelInvocationRequest {
                activation_id: "activation".into(),
                auth_token: "token".into(),
                invocation_id: "invocation".into(),
                reason: reason.into(),
            }));
            tokio::pin!(cancellation);
            assert!(futures_util::poll!(&mut cancellation).is_pending());
            completion_tx.send_replace(true);
            let ack = cancellation.await.unwrap().into_inner();
            assert_eq!(ack.accepted, !repeated);
            let expected = if repeated {
                "invocation is not active".into()
            } else if reason.is_empty() {
                "cancellation accepted".into()
            } else {
                format!("cancellation accepted: {reason}")
            };
            assert_eq!(ack.message, expected);
            assert!(task.await.unwrap_err().is_cancelled());
        }
    }
}

#[tokio::test]
async fn shutdown_aborts_active_invocations() {
    struct TestPlugin;

    impl WorkerPlugin for TestPlugin {
        fn plugin_id(&self) -> &str {
            "test.shutdown"
        }

        fn register(&self, _ctx: &mut PluginContext, _config: &Json) -> Result<()> {
            Ok(())
        }
    }

    let task = tokio::spawn(std::future::pending::<()>());
    let (_, completion) = watch::channel(false);
    let (stream_cancel, stream_cancel_rx) = watch::channel(false);
    let (shutdown, mut shutdown_rx) = watch::channel(None);
    let active_invocations = Arc::new(Mutex::new(HashMap::from([(
        "invocation".into(),
        ActiveInvocation {
            generation: 0,
            abort_handle: task.abort_handle(),
            stream_cancel: Some(stream_cancel),
            completion,
            cancel_requested: false,
        },
    )])));
    let service = WorkerService {
        plugin: Arc::new(TestPlugin),
        runtime: disconnected_runtime(),
        handlers: Arc::new(Mutex::new(WorkerHandlers::default())),
        active_invocations: active_invocations.clone(),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
        shutdown,
    };

    let ack = service
        .shutdown(Request::new(ShutdownRequest {
            activation_id: "activation".into(),
            auth_token: "token".into(),
            reason: "test".into(),
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(ack.accepted);
    assert_eq!(
        *shutdown_rx.borrow_and_update(),
        Some(ShutdownReason::HostRequest)
    );
    assert!(*stream_cancel_rx.borrow());
    assert!(
        active_invocations
            .lock()
            .unwrap()
            .get("invocation")
            .unwrap()
            .cancel_requested
    );
    assert!(task.await.unwrap_err().is_cancelled());

    let late_task = tokio::spawn(std::future::pending::<()>());
    let (_, late_completion) = watch::channel(false);
    let error = service
        .track_invocation("late", late_task.abort_handle(), None, late_completion)
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unavailable);
    late_task.abort();
    assert!(late_task.await.unwrap_err().is_cancelled());
}

#[test]
fn the_first_shutdown_reason_wins() {
    let (shutdown, receiver) = watch::channel(None);
    request_shutdown(&shutdown, ShutdownReason::HostRequest);
    request_shutdown(&shutdown, ShutdownReason::HostLost);
    assert_eq!(*receiver.borrow(), Some(ShutdownReason::HostRequest));
}

fn gate_test_plugin() -> Arc<dyn WorkerPlugin> {
    struct TestPlugin;

    impl WorkerPlugin for TestPlugin {
        fn plugin_id(&self) -> &str {
            "test.gate"
        }

        fn register(&self, _ctx: &mut PluginContext, _config: &Json) -> Result<()> {
            Ok(())
        }
    }

    Arc::new(TestPlugin)
}

fn shutting_down_service(handlers: WorkerHandlers) -> WorkerService {
    let (shutdown, _) = watch::channel(Some(ShutdownReason::HostRequest));
    WorkerService {
        plugin: gate_test_plugin(),
        runtime: disconnected_runtime(),
        handlers: Arc::new(Mutex::new(handlers)),
        active_invocations: Arc::new(Mutex::new(HashMap::new())),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
        shutdown,
    }
}

fn gate_invoke_request(registration_name: &str, surface: RegistrationSurface) -> InvokeRequest {
    InvokeRequest {
        activation_id: "activation".into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: None,
        auth_token: "token".into(),
        payload: None,
    }
}

#[tokio::test]
async fn gated_invocation_runs_after_open_and_aborts_on_drop() {
    use std::sync::atomic::AtomicBool;

    for open in [false, true] {
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let (completion_tx, mut completion_rx) = watch::channel(false);
        let (task, gate) = spawn_gated_invocation(
            async move {
                flag.store(true, Ordering::SeqCst);
                "ran"
            },
            InvocationCompletion::new(completion_tx),
        );
        tokio::task::yield_now().await;
        assert!(!ran.load(Ordering::SeqCst));
        if open {
            gate.open();
            assert_eq!(task.await.unwrap(), "ran");
        } else {
            drop(gate);
            assert!(task.await.unwrap_err().is_cancelled());
        }
        assert_eq!(ran.load(Ordering::SeqCst), open);
        completion_rx.changed().await.unwrap();
        assert!(*completion_rx.borrow());
    }
}

#[tokio::test]
async fn invoke_after_shutdown_is_unavailable_and_skips_the_callback() {
    use std::sync::atomic::AtomicBool;

    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let mut handlers = WorkerHandlers::default();
    handlers.subscribers.insert(
        "subscriber".into(),
        Arc::new(move |_event: &Event| flag.store(true, Ordering::SeqCst)),
    );
    let service = shutting_down_service(handlers);
    let event = Event::Mark(nemo_relay_types::api::event::MarkEvent::new(
        nemo_relay_types::api::event::BaseEvent::builder()
            .name("gated")
            .data(serde_json::json!({}))
            .build(),
        None,
        None,
    ));
    let mut request = gate_invoke_request("subscriber", RegistrationSurface::Subscriber);
    request.payload = Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Event(
        json_envelope("nemo.relay.Event@1", &event).unwrap(),
    ));
    let error = service.invoke(Request::new(request)).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unavailable);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(!ran.load(Ordering::SeqCst));
    assert!(service.active_invocations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invoke_stream_after_shutdown_is_unavailable_and_skips_the_handler() {
    use std::sync::atomic::AtomicBool;

    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let mut handlers = WorkerHandlers::default();
    handlers.llm_stream_executions.insert(
        "stream".into(),
        Arc::new(move |_model: &str, _request, _context, _next| {
            flag.store(true, Ordering::SeqCst);
            Box::pin(async { Err(WorkerSdkError::Callback("unused".into())) })
        }),
    );
    let service = shutting_down_service(handlers);
    let llm_request = LlmRequest {
        headers: Default::default(),
        content: serde_json::json!({"prompt": "hello"}),
    };
    let mut request =
        gate_invoke_request("stream", RegistrationSurface::LlmStreamExecutionIntercept);
    request.payload = Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(
        nemo_relay_worker_proto::v1::LlmInvocation {
            model_name: "model".into(),
            request: Some(json_envelope(LLM_REQUEST_SCHEMA, &llm_request).unwrap()),
            annotated_request: None,
            response: None,
            sanitize_context: None,
            execution_codec_context: Some(Box::new(
                nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                    request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                        codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                            kind: LlmCodecKind::Unspecified as i32,
                            id: None,
                        }),
                        codec_capability_id: None,
                    }),
                    response: None,
                },
            )),
        },
    ));
    let error = service
        .invoke_stream(Request::new(request))
        .await
        .err()
        .expect("a stream invocation is rejected during shutdown");
    assert_eq!(error.code(), tonic::Code::Unavailable);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(!ran.load(Ordering::SeqCst));
    assert!(service.active_invocations.lock().unwrap().is_empty());
}
