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

#[test]
fn execution_context_preserves_codec_identities_and_capabilities() {
    for (kind, id, expected, resolves) in [
        ("none", None, LlmCodecIdentity::None, false),
        ("opaque", None, LlmCodecIdentity::Opaque, true),
        (
            "builtin",
            Some("openai_chat".into()),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
            true,
        ),
    ] {
        let codec = wire::LlmCodecContext {
            codec: Some(wire::LlmCodecReference {
                kind: kind.into(),
                id,
            }),
            capability_id: resolves.then(|| "capability".into()),
        };
        let payload = LlmPayload {
            request_codec: Some(codec.clone()),
            response_codec: Some(codec),
            ..Default::default()
        };
        let context = payload
            .execution_context(&disconnected_runtime(), "invocation", true)
            .unwrap();
        assert_eq!(context.request_codec().codec, expected);
        assert_eq!(context.request_codec().resolve_codec().is_some(), resolves);
        assert_eq!(context.response_codec().unwrap().codec, expected);
        let payload = LlmPayload {
            response_codec: None,
            ..payload
        };
        assert!(
            payload
                .execution_context(&disconnected_runtime(), "invocation", true)
                .is_err()
        );
        assert!(
            payload
                .execution_context(&disconnected_runtime(), "invocation", false)
                .unwrap()
                .response_codec()
                .is_none()
        );
    }
    assert!(
        LlmPayload::default()
            .execution_context(&disconnected_runtime(), "invocation", false)
            .is_err()
    );
    let invalid = LlmPayload {
        request_codec: Some(wire::LlmCodecContext {
            codec: None,
            capability_id: None,
        }),
        ..Default::default()
    };
    assert!(invalid.sanitize_request_context("invocation").is_err());
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
        http_status: None,
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
            let service = WorkerService {
                health_reporter: tonic_health::server::HealthReporter::new(),
                shutdown: Arc::new(tokio::sync::Notify::new()),
                draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                registration: Arc::new(tokio::sync::Mutex::new(None)),
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
            };
            let cancellation = service.cancel_invocation(Request::new(CancelInvocationRequest {
                credentials: Some(wire::Credentials {
                    activation_id: "activation".into(),
                    auth_token: "token".into(),
                }),
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
async fn shutdown_rejects_stream_handoff_from_a_completed_opener() {
    struct TestPlugin;
    impl WorkerPlugin for TestPlugin {
        fn plugin_id(&self) -> &str {
            "test.shutdown_handoff"
        }
        fn register(&self, _ctx: &mut PluginContext, _config: &Json) -> Result<()> {
            Ok(())
        }
    }
    let service = WorkerService {
        health_reporter: tonic_health::server::HealthReporter::new(),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        registration: Arc::new(tokio::sync::Mutex::new(None)),
        plugin: Arc::new(TestPlugin),
        runtime: disconnected_runtime(),
        handlers: Arc::new(Mutex::new(WorkerHandlers::default())),
        active_invocations: Arc::new(Mutex::new(HashMap::new())),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
    };
    let opener = tokio::spawn(async {});
    let opener_abort = opener.abort_handle();
    opener.await.unwrap();
    let (completed, completion) = watch::channel(false);
    let (cancel, _) = watch::channel(false);
    let generation = service
        .track_invocation(
            "stream",
            opener_abort,
            Some(cancel.clone()),
            completion.clone(),
        )
        .unwrap();
    let shutdown = service.shutdown(Request::new(ShutdownRequest {
        credentials: Some(wire::Credentials {
            activation_id: "activation".into(),
            auth_token: "token".into(),
        }),
        reason: "test stream handoff".into(),
    }));
    tokio::pin!(shutdown);
    // Hold completion open to exercise replacement after the shutdown snapshot.
    assert!(futures_util::poll!(&mut shutdown).is_pending());
    let pump = tokio::spawn(std::future::pending::<()>());
    let replaced = service
        .replace_invocation(
            "stream",
            generation,
            ActiveInvocation {
                generation,
                abort_handle: pump.abort_handle(),
                stream_cancel: Some(cancel),
                completion,
                cancel_requested: false,
            },
        )
        .unwrap();
    pump.abort();
    completed.send_replace(true);
    shutdown.await.unwrap();
    assert!(!replaced, "shutdown must fence the stream producer handoff");
}

#[tokio::test]
async fn repeated_shutdown_does_not_reopen_health_watches() {
    use tonic_health::pb::health_server::Health;
    use tonic_health::pb::{HealthCheckRequest, health_check_response::ServingStatus};

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
    let (completion_tx, completion_rx) = watch::channel(false);
    let service = WorkerService {
        health_reporter: tonic_health::server::HealthReporter::new(),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        registration: Arc::new(tokio::sync::Mutex::new(None)),
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
                cancel_requested: false,
            },
        )]))),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
    };
    service
        .health_reporter
        .set_serving::<PluginWorkerServer<WorkerService>>()
        .await;
    let health =
        tonic_health::server::HealthService::from_health_reporter(service.health_reporter.clone());
    let mut watches = Vec::new();
    for name in [
        "",
        <PluginWorkerServer<WorkerService> as tonic::server::NamedService>::NAME,
    ] {
        let mut watch = health
            .watch(Request::new(HealthCheckRequest {
                service: name.into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            watch.next().await.unwrap().unwrap().status,
            ServingStatus::Serving as i32
        );
        watches.push(watch);
    }
    let shutdown_request = || {
        Request::new(ShutdownRequest {
            credentials: Some(wire::Credentials {
                activation_id: "activation".into(),
                auth_token: "token".into(),
            }),
            reason: "test repeated shutdown".into(),
        })
    };
    let first = service.shutdown(shutdown_request());
    let concurrent = service.shutdown(shutdown_request());
    tokio::pin!(first, concurrent);
    // Hold cleanup open so both calls enter Shutdown before either completes.
    assert!(futures_util::poll!(&mut first).is_pending());
    assert!(futures_util::poll!(&mut concurrent).is_pending());
    completion_tx.send_replace(true);
    let (first, concurrent) = tokio::join!(first, concurrent);
    first.unwrap();
    concurrent.unwrap();
    service.shutdown(shutdown_request()).await.unwrap();
    for mut watch in watches {
        assert_eq!(
            watch.next().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        assert!(watch.next().await.is_none());
    }
    for name in [
        "",
        <PluginWorkerServer<WorkerService> as tonic::server::NamedService>::NAME,
    ] {
        assert_eq!(
            health
                .check(Request::new(HealthCheckRequest {
                    service: name.into()
                }))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::NotFound
        );
    }
    assert!(task.await.unwrap_err().is_cancelled());
}
