// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise foreign stream cleanup without changing managed stream close-result caching.
use super::*;
use crate::dynamic::native_library::{CheckedNativeLibrary, NativeLibraryArtifact};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;

#[tokio::test]
async fn shutdown_drains_synchronous_runtime_calls_before_collecting_scopes() {
    let owner = HostRuntime::new("checked-runtime-admission".into()).unwrap();
    let stack = create_scope_stack();
    let blocked_stack = stack.clone();
    let (locked, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let blocker = std::thread::spawn(move || {
        let _lock = blocked_stack.write().unwrap();
        locked.send(()).unwrap();
        wait.recv().unwrap();
    });
    ready.await.unwrap();
    let worker_stack = stack.clone();
    let runtime = nemo_relay_plugin::PluginRuntime::from_abi(owner.runtime_box());
    let worker = std::thread::spawn(move || {
        with_scope_stack(worker_stack, || {
            runtime
                .push_scope(
                    "admitted-runtime-scope",
                    ScopeType::Custom,
                    None,
                    None,
                    None,
                )
                .map(|_| ())
        })
    });
    let admitted = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while owner.executor.active_tasks() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    let shutdown = owner.shutdown();
    tokio::pin!(shutdown);
    let pending = futures_util::poll!(&mut shutdown).is_pending();
    let stale = owner
        .runtime_box()
        .call_sync("stack.create".into(), abi::Json::Null);
    release.send(()).unwrap();
    blocker.join().unwrap();
    if let Err(error) = worker.join().unwrap() {
        assert!(error.contains("revoked"), "{error}");
    }
    shutdown.await.unwrap();

    assert!(admitted.is_ok(), "foreign runtime calls must be tracked");
    assert!(pending, "shutdown must wait for admitted runtime calls");
    assert!(stale.is_err(), "revoked runtime calls must be rejected");
    assert!(owner.executor.is_closed());
    assert!(owner.state.lock().unwrap().scopes.is_empty());
    with_scope_stack(stack, || {
        assert_eq!(
            current_scope_stack_snapshot()
                .unwrap()
                .unwrap()
                .scopes
                .len(),
            1
        );
    });
}

struct CaptureRuntime {
    base: a::RuntimeBox,
    callbacks: Arc<Mutex<Vec<a::Registration>>>,
    component: Arc<Mutex<Option<a::ComponentBox>>>,
}
impl a::Runtime for CaptureRuntime {
    fn call_sync(&self, op: RString, args: abi::Json) -> a::Result<abi::Json> {
        self.base.call_sync(op, args)
    }
    fn call(&self, op: RString, args: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        self.base.call(op, args)
    }
    fn diagnostics(&self) -> abi::diagnostics::DiagnosticsBox {
        self.base.diagnostics()
    }
    fn capture_scope(&self) -> a::Result<a::ScopeBox> {
        self.base.capture_scope()
    }
    fn bind_component(&self, component: a::ComponentBox) -> a::Result<()> {
        *self.component.lock().unwrap() = Some(component);
        a::Result::ROk(())
    }
    fn register_callback(&self, registration: a::Registration) -> a::Result<()> {
        self.callbacks.lock().unwrap().push(registration);
        a::Result::ROk(())
    }
    fn register_gate(&self, registration: a::Registration) -> a::Result<RString> {
        self.base.register_gate(registration)
    }
}
struct Source {
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    polled: Arc<AtomicBool>,
    pending: bool,
    fail_once: bool,
}
impl Stream for Source {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polled.store(true, Ordering::SeqCst);
        if self.pending {
            Poll::Pending
        } else {
            Poll::Ready(Some(Ok(json!({"token":"checked"}))))
        }
    }
}
impl LlmStreamInner for Source {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let calls = self.calls.clone();
        let fail = self.fail_once;
        Box::pin(async move {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 && fail {
                Err(err("checked downstream close failure"))
            } else {
                Ok(())
            }
        })
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}
fn library() -> CheckedNativeLibrary {
    let path = std::env::var_os("NEMO_RELAY_TEST_NATIVE_PLUGIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(format!(
                    "{}nemo_relay_plugin_fixture{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
        });
    let file = PathBuf::from(path.file_name().unwrap());
    let bytes = std::fs::read(path).expect("run just build-test-plugin-fixtures");
    CheckedNativeLibrary::load(
        NativeLibraryArtifact::new(file.clone(), [(file, bytes)]).unwrap(),
        "v1.0",
        "fixture_native",
    )
    .unwrap()
}

#[tokio::test]
async fn stream_cleanup_retries_errors_and_cancels_a_live_pull() {
    // OS image validation is setup, outside the asynchronous cleanup deadline.
    let library = library();
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        for (pending, raw) in [(false, false), (true, false), (false, true), (true, true)] {
            let owner = HostRuntime::new("checked-stream-test".into()).unwrap();
            let callbacks = Arc::new(Mutex::new(Vec::new()));
            let bound = Arc::new(Mutex::new(None));
            let runtime = a::Runtime_TO::from_value(
                CaptureRuntime {
                    base: owner.runtime_box(),
                    callbacks: callbacks.clone(),
                    component: bound.clone(),
                },
                TD_Opaque,
            );
            let registered = library
                .create_plugin()
                .unwrap()
                .register(
                    into_abi(json!({"stream_only_fixture":!raw, "raw_stream_only_fixture":raw})),
                    runtime,
                )
                .await
                .into_result()
                .unwrap();
            assert!(registered.error.is_none());
            let descriptor = callbacks.lock().unwrap().pop().unwrap();
            let callback = descriptor.callback;
            let calls = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicBool::new(false));
            let polled = Arc::new(AtomicBool::new(false));
            let source_calls = calls.clone();
            let source_dropped = dropped.clone();
            let source_polled = polled.clone();
            let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
                let calls = source_calls.clone();
                let dropped = source_dropped.clone();
                let polled = source_polled.clone();
                Box::pin(async move {
                    Ok(LlmJsonStream::from_closeable(Source {
                        calls,
                        dropped,
                        polled,
                        pending,
                        fail_once: !pending,
                    }))
                })
            });
            if raw {
                let (lease, _guard) = MiddlewareContinuationLease::capture();
                let continuation = a::Continuation_TO::from_value(
                    Continuation {
                        downstream: Default::default(),
                        next: Arc::new(Next::Stream(next.clone())),
                        lease,
                        owner: owner.clone(),
                    },
                    TD_Opaque,
                );
                let (i, _life) = invocation(
                    &owner,
                    a::Surface::LlmStreamExecutionIntercept,
                    "raw-stream-panic".into(),
                    abi::Json::Null,
                    None,
                    None,
                    None,
                );
                let error = callback
                    .open_stream(i, continuation)
                    .await
                    .err()
                    .expect("raw stream panic must become an error");
                assert!(
                    error
                        .message
                        .contains("synchronous stream callback panicked")
                );
            }
            let (lease, _guard) = MiddlewareContinuationLease::capture();
            let next = a::Continuation_TO::from_value(
                Continuation {
                    downstream: Default::default(),
                    next: Arc::new(Next::Stream(next)),
                    lease,
                    owner: owner.clone(),
                },
                TD_Opaque,
            );
            let (i, _life) = invocation(
                &owner,
                a::Surface::LlmStreamExecutionIntercept,
                "checked-stream".into(),
                encode(nemo_relay_types::api::llm::LlmRequest {
                    headers: Default::default(),
                    content: json!({}),
                })
                .unwrap(),
                None,
                None,
                None,
            );
            let mut stream = callback.open_stream(i, next).await.into_result().unwrap();
            let mut pulling = Box::pin(stream.pull());
            if pending {
                tokio::select! {
                    result = &mut pulling => panic!("pending source completed early: {result:?}"),
                    _ = async { while !polled.load(Ordering::SeqCst) { tokio::task::yield_now().await; } } => {}
                }
                stream.close().await.into_result().unwrap();
                assert!(
                    pulling.await.is_err(),
                    "close must cancel a pull even when its waiter is retained"
                );
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            } else {
                assert!(pulling.await.into_result().unwrap().is_some());
                let error = stream.close().await.into_result().unwrap_err();
                assert!(error.message.contains("checked downstream close failure"));
                assert!(!dropped.load(Ordering::SeqCst));
                stream.close().await.into_result().unwrap();
                assert_eq!(calls.load(Ordering::SeqCst), 2);
            }
            assert!(dropped.load(Ordering::SeqCst));
            drop(callback);
            let bound = bound.lock().unwrap().take().unwrap();
            let (first, second) = tokio::join!(registered.component.shutdown(), bound.shutdown());
            first.into_result().unwrap();
            second.into_result().unwrap();
            stream.close().await.into_result().unwrap();
            owner.shutdown().await.unwrap();
        }
    })
    .await
    .expect("stream close must drain its own pending polls");
}

#[tokio::test]
async fn callback_observations_exclude_propagated_downstream_errors_and_revocation() {
    use nemo_relay_diagnostics::{
        DiagnosticQuery, Diagnostics, DiagnosticsSettings, SourceIdentity,
    };
    let context = Diagnostics::from_settings(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: None,
            process_role: "test".into(),
            relay_version: "test".into(),
        },
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let owner = HostRuntime::with_diagnostics(
        "checked-observation".into(),
        context.clone(),
        false,
        "test.plugin".into(),
    )
    .unwrap();
    *owner.observer.lock().unwrap() = super::super::diagnostics::PluginDiagnostics::new(
        &context,
        "test.plugin",
        nemo_relay_diagnostics::PluginExecution::NativeAbi,
        Uuid::now_v7(),
    );
    let downstream = DownstreamErrors::default();
    let failure: a::Result<()> = downstream_boundary(Err(err("downstream failure")), &downstream);
    assert!(owner.callback_result(failure, &downstream).is_err());
    assert!(
        owner
            .callback_result::<()>(
                a::Result::RErr(a::Error::new("plugin", "author callback failed")),
                &downstream
            )
            .is_err()
    );
    let scope = owner.scope();
    let called = Arc::new(AtomicBool::new(false));
    let seen = called.clone();
    let (lease, _guard) = MiddlewareContinuationLease::capture();
    let next = Continuation {
        next: Arc::new(Next::Tool(Arc::new(move |_| {
            seen.store(true, Ordering::Release);
            Box::pin(async {
                Ok(nemo_relay_types::api::tool::ToolExecutionResult::new(
                    Json::Null,
                ))
            })
        }))),
        lease,
        downstream: Default::default(),
        owner: owner.clone(),
    };
    owner.revoke().unwrap();
    assert!(scope.enter().is_err());
    assert!(
        a::Continuation::call_tool(&next, abi::Json::Null)
            .await
            .is_err()
    );
    assert!(!called.load(Ordering::Acquire));
    assert!(
        owner
            .callback_result::<()>(
                a::Result::RErr(a::Error::new("revoked", "stale callback")),
                &downstream
            )
            .is_err()
    );
    let snapshot = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(snapshot.lifetime.totals.failures_total.value.get(), 1);
    owner.shutdown().await.unwrap();
}

#[test]
fn checked_errors_preserve_typed_http_status_without_a_json_envelope() {
    let mut error = a::Error::new("http", "upstream response");
    error.http_status = ROption::RSome(429);
    let error = local::<()>(a::Result::RErr(error)).unwrap_err();
    let FlowError::Upstream(failure) = error else {
        panic!("expected typed HTTP failure");
    };
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.body, "upstream response");
}

#[tokio::test]
async fn codec_capabilities_are_revoked_when_the_component_shuts_down() {
    struct Dummy;
    impl nemo_relay_types::codec::traits::LlmCodec for Dummy {
        fn decode(
            &self,
            _: &nemo_relay_types::api::llm::LlmRequest,
        ) -> Result<nemo_relay_types::codec::request::AnnotatedLlmRequest> {
            Ok(Default::default())
        }
        fn encode(
            &self,
            _: &nemo_relay_types::codec::request::AnnotatedLlmRequest,
            original: &nemo_relay_types::api::llm::LlmRequest,
        ) -> Result<nemo_relay_types::api::llm::LlmRequest> {
            Ok(original.clone())
        }
    }
    let owner = HostRuntime::new("codec-review".into()).unwrap();
    let (invocation, _lifetime) = invocation(
        &owner,
        a::Surface::LlmSanitizeRequest,
        "review".into(),
        abi::Json::Null,
        None,
        Some(LlmRequestContext::for_request_codec(Some(Arc::new(Dummy)))),
        None,
    );
    let codec = invocation.context.request_codec.into_option().unwrap();
    owner.shutdown().await.unwrap();
    let result = codec.decode(
        encode(nemo_relay_types::api::llm::LlmRequest {
            headers: Default::default(),
            content: json!({}),
        })
        .unwrap(),
    );
    assert!(
        result.is_err(),
        "codec still executes after successful component shutdown"
    );
}

#[tokio::test]
async fn revoked_scope_cleanup_retains_ownership_until_the_stack_can_unwind() {
    let owner = HostRuntime::new("checked-scope-cleanup".into()).unwrap();
    let runtime = nemo_relay_plugin::PluginRuntime::from_abi(owner.runtime_box());
    TASK_SCOPE_STACK
        .scope(create_scope_stack(), async {
            let outer = runtime
                .push_scope("native-outer", ScopeType::Custom, None, None, None)
                .unwrap();
            let inner = runtime
                .push_scope("native-inner", ScopeType::Custom, None, None, None)
                .unwrap();
            let blocker = push_scope(
                PushScopeParams::builder()
                    .name("caller-owned")
                    .scope_type(ScopeType::Custom)
                    .build(),
            )
            .unwrap();
            owner.revoke().unwrap();
            assert!(runtime.pop_scope(&inner, None, None).is_err());
            assert!(
                owner.shutdown().await.is_err(),
                "caller scope blocks cleanup"
            );
            assert_eq!(owner.state.lock().unwrap().scopes.len(), 2);
            assert!(!owner.executor.is_closed());
            pop_scope(PopScopeParams::builder().handle_uuid(&blocker.uuid).build()).unwrap();
            owner.shutdown().await.unwrap();
            assert!(owner.state.lock().unwrap().scopes.is_empty());
            assert!(owner.executor.is_closed());
            assert_eq!(
                current_scope_stack_snapshot()
                    .unwrap()
                    .unwrap()
                    .scopes
                    .len(),
                1
            );
            assert!(
                runtime.pop_scope(&outer, None, None).is_err(),
                "retained handles stay revoked"
            );
        })
        .await;
}

struct InlineCallback {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}
impl a::Callback for InlineCallback {
    fn invoke_sync(&self, _: a::Invocation) -> a::Result<abi::Json> {
        self.entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
        self.release.lock().unwrap().recv().unwrap();
        a::Result::ROk(abi::Json::Null)
    }
    fn invoke(
        &self,
        _: a::Invocation,
        _: ROption<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        async { a::Result::RErr(a::Error::new("test", "callback is synchronous")) }.into_ffi()
    }
    fn invoke_tool(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        async { a::Result::RErr(a::Error::new("test", "callback is synchronous")) }.into_ffi()
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        async { a::Result::RErr(a::Error::new("test", "callback is synchronous")) }.into_ffi()
    }
}
struct ShutdownProbe(Arc<AtomicBool>);
impl a::Component for ShutdownProbe {
    fn shutdown(&self) -> FfiFuture<a::Result<()>> {
        self.0.store(true, Ordering::Release);
        async { a::Result::ROk(()) }.into_ffi()
    }
}

#[tokio::test]
async fn component_close_retains_ownership_until_a_raw_inline_subscriber_leaves() {
    let runtime = HostRuntime::new("inline-close".into()).unwrap();
    let shut_down = Arc::new(AtomicBool::new(false));
    *runtime.component.lock().unwrap() = Some(a::Component_TO::from_value(
        ShutdownProbe(shut_down.clone()),
        TD_Opaque,
    ));
    let component = ComponentOwner {
        component: runtime.component.clone(),
        runtime: runtime.clone(),
    };
    let (entered, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let owned = runtime
        .own_callback(a::Callback_TO::from_value(
            InlineCallback {
                entered: Mutex::new(Some(entered)),
                release: Mutex::new(wait),
            },
            TD_Opaque,
        ))
        .unwrap();
    let callback = subscriber_callback(owned.clone(), runtime.clone());
    let thread = std::thread::spawn(move || {
        callback(&nemo_relay_types::api::event::Event::Mark(
            nemo_relay_types::api::event::MarkEvent::builder().build(),
        ));
    });
    ready.await.unwrap();
    let first = component.close();
    let retained = runtime.component.lock().unwrap().is_some();
    let prematurely_stopped = shut_down.load(Ordering::Acquire);
    let retained_callback = owned.value.lock().unwrap().is_some();
    release.send(()).unwrap();
    thread.join().unwrap();
    component.close().unwrap();
    assert!(first.unwrap_err().to_string().contains("close retry"));
    assert!(retained, "failed close must retain the component");
    assert!(
        retained_callback,
        "failed close must retain admitted callback state"
    );
    assert!(owned.value.lock().unwrap().is_none());
    assert!(
        !prematurely_stopped,
        "component resources must outlive admitted callbacks"
    );
    assert!(shut_down.load(Ordering::Acquire));
    assert!(runtime.executor.is_closed());
}

struct CallbackStateProbe {
    dropped: Arc<AtomicBool>,
    invoked: Arc<AtomicBool>,
    _runtime: Arc<HostRuntime>,
}
impl Drop for CallbackStateProbe {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}
impl a::Callback for CallbackStateProbe {
    fn invoke_sync(&self, _: a::Invocation) -> a::Result<abi::Json> {
        self.invoked.store(true, Ordering::Release);
        a::Result::ROk(abi::Json::Null)
    }
    fn invoke(
        &self,
        _: a::Invocation,
        _: ROption<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        self.invoked.store(true, Ordering::Release);
        async { a::Result::RErr(a::Error::new("test", "sync callback")) }.into_ffi()
    }
    fn invoke_tool(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        self.invoked.store(true, Ordering::Release);
        async { a::Result::RErr(a::Error::new("test", "sync callback")) }.into_ffi()
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        self.invoked.store(true, Ordering::Release);
        async { a::Result::RErr(a::Error::new("test", "sync callback")) }.into_ffi()
    }
}
#[tokio::test]
async fn close_releases_queued_native_callback_state_without_waiting_for_dispatch() {
    use nemo_relay_runtime::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    let (entered, ready) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (release, wait) = std::sync::mpsc::channel();
    let wait = Mutex::new(wait);
    register_subscriber(
        "review-queue-blocker",
        Arc::new(move |event| {
            if event.name() == "review-block" {
                entered.lock().unwrap().take().unwrap().send(()).unwrap();
                wait.lock().unwrap().recv().unwrap();
            }
        }),
    )
    .unwrap();
    event(EmitMarkEventParams::builder().name("review-block").build()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), ready)
        .await
        .unwrap()
        .unwrap();
    let runtime = HostRuntime::new("review-queued".into()).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let invoked = Arc::new(AtomicBool::new(false));
    let mut context = crate::HostPluginRegistrationContext::new();
    apply_registration(
        &mut context,
        a::Registration {
            name: "review-native-queued".into(),
            surface: a::Surface::Subscriber,
            priority: 0,
            break_chain: false,
            options: abi::Json::Null,
            callback: a::Callback_TO::from_value(
                CallbackStateProbe {
                    dropped: dropped.clone(),
                    invoked: invoked.clone(),
                    _runtime: runtime.clone(),
                },
                TD_Opaque,
            ),
        },
        runtime.clone(),
    )
    .unwrap();
    event(
        EmitMarkEventParams::builder()
            .name("review-queued-event")
            .build(),
    )
    .unwrap();
    let mut registrations = context.into_registrations();
    let rollback = crate::rollback_registrations(&mut registrations);
    let close = ComponentOwner {
        component: runtime.component.clone(),
        runtime: runtime.clone(),
    }
    .close();
    let dropped_at_close = dropped.load(Ordering::Acquire);
    release.send(()).unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber("review-queue-blocker").unwrap();
    assert!(rollback.callbacks_cleared());
    close.unwrap();
    assert!(!invoked.load(Ordering::Acquire));
    assert!(dropped.load(Ordering::Acquire));
    assert!(
        dropped_at_close,
        "successful component close must release owned native callback state"
    );
}

#[tokio::test]
async fn retained_middleware_and_gate_proxies_release_foreign_state_on_shutdown() {
    let owner = HostRuntime::new("retained-proxies".into()).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let invoked = Arc::new(AtomicBool::new(false));
    let callback = owner
        .own_callback(a::Callback_TO::from_value(
            CallbackStateProbe {
                dropped: dropped.clone(),
                invoked: invoked.clone(),
                _runtime: owner.clone(),
            },
            TD_Opaque,
        ))
        .unwrap();
    let gate = gate_callback(callback.clone(), owner.clone());
    let middleware = unary::<Json>(
        owner.clone(),
        callback,
        Call {
            surface: a::Surface::ToolRequestIntercept,
            name: "retained".into(),
            value: abi::Json::Null,
            id: None,
            request: None,
            response: None,
            next: None,
        },
    );
    owner.shutdown().await.unwrap();
    assert!(
        dropped.load(Ordering::Acquire),
        "unpolled middleware must not retain foreign state"
    );
    assert!(owner.state.lock().unwrap().callbacks.is_empty());
    assert_eq!(gate(&BTreeSet::new(), "retained"), None);
    assert!(middleware.await.is_err());
    assert!(
        !invoked.load(Ordering::Acquire),
        "stale proxies cannot call foreign code"
    );
}

#[tokio::test]
async fn response_codec_capabilities_preserve_results_and_reject_invalid_dispatch() {
    use nemo_relay_types::codec::{response::AnnotatedLlmResponse, traits::LlmResponseCodec};
    struct ResponseCodec;
    impl LlmResponseCodec for ResponseCodec {
        fn decode_response(&self, value: &Json) -> Result<AnnotatedLlmResponse> {
            if value == "panic" {
                panic!("codec author panic");
            }
            if value != &json!({"provider":"response"}) {
                return Err(err("unexpected response"));
            }
            Ok(Default::default())
        }
    }
    let owner = HostRuntime::new("response-codec-contract".into()).unwrap();
    let (call, lifetime) = invocation(
        &owner,
        a::Surface::LlmSanitizeResponse,
        "response".into(),
        abi::Json::Null,
        None,
        None,
        Some(LlmResponseContext::for_response_codec(Some(Arc::new(
            ResponseCodec,
        )))),
    );
    let codec = call.context.response_codec.into_option().unwrap();
    let identity = from_abi(codec.identity().into_result().unwrap()).unwrap();
    assert_eq!(identity["available"], true);
    let decoded: AnnotatedLlmResponse = decode(
        codec
            .decode(into_abi(json!({"provider":"response"})))
            .into_result()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(decoded, AnnotatedLlmResponse::default());
    assert!(codec.encode(abi::Json::Null, abi::Json::Null).is_err());
    assert!(codec.decode(into_abi(json!({"bad":true}))).is_err());
    let panic = codec
        .decode(into_abi(json!("panic")))
        .into_result()
        .unwrap_err();
    assert!(panic.message.contains("codec callback panicked"));
    drop(lifetime);
    assert!(codec.identity().is_err());
    assert!(codec.decode(abi::Json::Null).is_err());
    for (request, response) in [
        (Some(LlmRequestContext::for_request_codec(None)), None),
        (None, Some(LlmResponseContext::for_response_codec(None))),
    ] {
        let (call, _lifetime) = invocation(
            &owner,
            a::Surface::LlmSanitizeResponse,
            "absent".into(),
            abi::Json::Null,
            None,
            request,
            response,
        );
        let codec = call
            .context
            .request_codec
            .into_option()
            .or_else(|| call.context.response_codec.into_option())
            .unwrap();
        assert_eq!(
            from_abi(codec.identity().into_result().unwrap()).unwrap()["available"],
            false
        );
        assert!(codec.decode(abi::Json::Null).is_err());
        assert!(codec.encode(abi::Json::Null, abi::Json::Null).is_err());
    }
    owner.shutdown().await.unwrap();
}

#[tokio::test]
async fn registration_transactions_reject_late_binding_full_queues_and_cancelled_replies() {
    let owner = HostRuntime::new("registration-contract".into()).unwrap();
    let raw = owner.runtime_box();
    let component =
        || a::Component_TO::from_value(ShutdownProbe(Arc::new(AtomicBool::new(false))), TD_Opaque);
    let descriptor = || a::Registration {
        name: "callback".into(),
        surface: a::Surface::Subscriber,
        priority: 0,
        break_chain: false,
        options: abi::Json::Null,
        callback: a::Callback_TO::from_value(
            CallbackStateProbe {
                dropped: Arc::new(AtomicBool::new(false)),
                invoked: Arc::new(AtomicBool::new(false)),
                _runtime: owner.clone(),
            },
            TD_Opaque,
        ),
    };
    assert!(raw.bind_component(component()).is_err());
    assert!(raw.register_callback(descriptor()).is_err());
    assert!(raw.register_gate(descriptor()).is_err());
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    *owner.registration.lock().unwrap() = Some(sender.clone());
    assert!(raw.bind_component(component()).is_ok());
    assert!(raw.bind_component(component()).is_err());
    let (reply, _) = std::sync::mpsc::sync_channel(1);
    sender
        .try_send(RegistrationRequest {
            registration: descriptor(),
            reply,
        })
        .unwrap();
    assert_eq!(
        raw.register_callback(descriptor())
            .into_result()
            .unwrap_err()
            .code
            .as_str(),
        "capacity"
    );
    drop(receiver.try_recv().unwrap());
    let request = descriptor();
    let caller = owner.runtime_box();
    let pending = std::thread::spawn(move || caller.register_callback(request));
    let request = receiver.recv().await.unwrap();
    drop(request);
    let error = pending.join().unwrap().into_result().unwrap_err();
    assert!(error.message.contains("cancelled"));
    drop(receiver);
    assert!(raw.register_callback(descriptor()).is_err());
    owner.registration.lock().unwrap().take();
    owner.component.lock().unwrap().take();
    owner.shutdown().await.unwrap();
    assert!(raw.bind_component(component()).is_err());
}

#[test]
fn raw_registration_can_submit_callbacks_during_construction_and_polling() {
    use crate::Plugin as _;
    struct RawPlugin {
        pending: Option<(Arc<tokio::sync::Notify>, Arc<AtomicBool>)>,
    }
    fn descriptor(name: &str) -> a::Registration {
        a::Registration {
            name: name.into(),
            surface: a::Surface::Subscriber,
            priority: 0,
            break_chain: false,
            options: abi::Json::Null,
            callback: a::Callback_TO::from_value(
                CallbackStateProbe {
                    dropped: Arc::new(AtomicBool::new(false)),
                    invoked: Arc::new(AtomicBool::new(false)),
                    _runtime: HostRuntime::new("raw-registration-probe".into()).unwrap(),
                },
                TD_Opaque,
            ),
        }
    }
    impl a::Plugin for RawPlugin {
        fn allows_multiple_components(&self) -> bool {
            true
        }
        fn validate(&self, _: abi::Json) -> abi::abi_stable::std_types::RVec<a::ConfigDiagnostic> {
            Default::default()
        }
        fn register(
            &self,
            _: abi::Json,
            runtime: a::RuntimeBox,
        ) -> FfiFuture<a::Result<a::RegisteredComponent>> {
            runtime
                .register_callback(descriptor("construction"))
                .into_result()
                .unwrap();
            let pending = self.pending.clone();
            async move {
                runtime
                    .register_callback(descriptor("polling"))
                    .into_result()
                    .unwrap();
                if let Some((started, dropped)) = pending {
                    struct Probe(Arc<AtomicBool>);
                    impl Drop for Probe {
                        fn drop(&mut self) {
                            self.0.store(true, Ordering::Release);
                        }
                    }
                    let _probe = Probe(dropped);
                    started.notify_one();
                    std::future::pending::<()>().await;
                }
                a::Result::ROk(a::RegisteredComponent {
                    error: ROption::RNone,
                    registrations: Default::default(),
                    component: a::Component_TO::from_value(
                        ComponentExecutor::inline("raw-registration"),
                        TD_Opaque,
                    ),
                })
            }
            .into_ffi()
        }
    }
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async { for cancel in [false, true] {
            let started = Arc::new(tokio::sync::Notify::new());
            let dropped = Arc::new(AtomicBool::new(false));
            let plugin = CheckedPlugin {
                kind: "raw-registration".into(),
                resident_path: Default::default(),
                plugin: Arc::new(a::Plugin_TO::from_value(RawPlugin {
                    pending: cancel.then(|| (started.clone(), dropped.clone())),
                }, TD_Opaque)),
                diagnostics: nemo_relay_diagnostics::Diagnostics::process_default().clone(),
                read: false,
                observer: Default::default(),
            };
            let mut context = crate::HostPluginRegistrationContext::new();
            let mut registration = plugin.register(&Default::default(), &mut context);
            if cancel {
                tokio::select! {
                    result = &mut registration => panic!("pending registration finished early: {result:?}"),
                    () = started.notified() => {}
                }
            } else {
                (&mut registration).await.unwrap();
            }
            drop(registration);
            let mut registrations = context.into_registrations();
            assert_eq!(registrations.len(), 3);
            assert!(crate::rollback_registrations(&mut registrations).callbacks_cleared());
            if cancel { assert!(dropped.load(Ordering::Acquire)); }
        }});
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("raw registration must not block its transaction driver");
    worker.join().unwrap();
}

/// SDK background tasks inherit real Relay scope and remain owned after their join handle is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_background_tasks_preserve_scope_and_are_cancelled_with_their_component() {
    use nemo_relay_plugin::{Plugin as NativePlugin, PluginContext, PluginTaskSpawner};
    struct DroppedTask(Arc<AtomicBool>);
    impl Drop for DroppedTask {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    struct Author {
        started: Mutex<Option<tokio::sync::oneshot::Sender<Json>>>,
        dropped: Arc<AtomicBool>,
        spawner: Arc<Mutex<Option<PluginTaskSpawner>>>,
    }
    impl NativePlugin for Author {
        fn plugin_kind(&self) -> &str {
            "background-test"
        }
        fn executor_config(&self) -> Option<nemo_relay_plugin::NativeExecutorConfig> {
            Some(Default::default())
        }
        fn register<'a>(
            &'a self,
            _: &'a serde_json::Map<String, Json>,
            ctx: &'a mut PluginContext<'_>,
        ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
            Box::pin(async move {
                *self.spawner.lock().unwrap() = Some(ctx.task_spawner());
                let started = self.started.lock().unwrap().take().unwrap();
                let runtime = ctx.runtime();
                let probe = DroppedTask(self.dropped.clone());
                let task = ctx.spawn(async move {
                    let _probe = probe;
                    tokio::task::yield_now().await;
                    started
                        .send(
                            serde_json::to_value(runtime.current_scope_stack_snapshot()?).unwrap(),
                        )
                        .unwrap();
                    std::future::pending::<()>().await;
                    Ok(())
                })?;
                drop(task);
                Ok(())
            })
        }
    }
    let host = HostRuntime::new("background-scope".into()).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let spawner = Arc::new(Mutex::new(None));
    let (started, ready) = tokio::sync::oneshot::channel();
    let author = nemo_relay_plugin::checked::Adapter(Arc::new(Author {
        started: Mutex::new(Some(started)),
        dropped: dropped.clone(),
        spawner: spawner.clone(),
    }));
    let callbacks = Arc::new(Mutex::new(Vec::new()));
    let bound = Arc::new(Mutex::new(None));
    let runtime = a::Runtime_TO::from_value(
        CaptureRuntime {
            base: host.runtime_box(),
            callbacks,
            component: bound.clone(),
        },
        TD_Opaque,
    );
    TASK_SCOPE_STACK
        .scope(create_scope_stack(), async {
            let sdk = nemo_relay_plugin::PluginRuntime::from_abi(host.runtime_box());
            let mut scope = sdk
                .scope("background-parent", ScopeType::Custom, None, None, None)
                .unwrap();
            let expected =
                serde_json::to_value(sdk.current_scope_stack_snapshot().unwrap()).unwrap();
            let registered =
                a::Plugin::register(&author, abi::Json::Object(Default::default()), runtime)
                    .await
                    .into_result()
                    .unwrap();
            assert!(registered.error.is_none());
            let observed = tokio::time::timeout(std::time::Duration::from_secs(5), ready)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(observed, expected);
            registered.component.shutdown().await.into_result().unwrap();
            assert!(dropped.load(Ordering::Acquire));
            assert!(
                spawner
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .spawn(async { Ok(()) })
                    .is_err()
            );
            bound.lock().unwrap().take();
            scope.close(None, None).unwrap();
        })
        .await;
    host.shutdown().await.unwrap();
}
