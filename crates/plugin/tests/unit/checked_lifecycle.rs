// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::AtomicUsize;
thread_local! { static BOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
struct Scope;
struct Lease(Option<bool>);
impl a::ScopeLease for Lease {
    fn restore(&mut self) -> a::Result<()> {
        if let Some(previous) = self.0.take() {
            BOUND.set(previous);
        }
        a::Result::Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = a::ScopeLease::restore(self);
    }
}
impl a::Scope for Scope {
    fn enter(&self) -> a::Result<a::ScopeLeaseBox> {
        a::Result::Ok(a::ScopeLeaseBox::new(Lease(Some(BOUND.replace(true)))))
    }
    fn snapshot(&self) -> a::Result<Option<abi::Json>> {
        a::Result::Ok(Option::None)
    }
}
pub(crate) struct Next;
impl a::Continuation for Next {
    fn call(&self, _: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        unreachable!()
    }
    fn call_tool(&self, _: abi::Json) -> FfiFuture<a::Result<a::ToolExecutionResult>> {
        unreachable!()
    }
    fn open_stream(&self, _: abi::Json) -> FfiFuture<a::Result<a::StreamBox>> {
        unreachable!()
    }
}
struct Probe(Arc<AtomicUsize>);
impl Stream for Probe {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_drains_stream_opening_before_closing_resources() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 2 }, "shutdown-race")
            .unwrap();
    let registry = Arc::new(Mutex::new(Vec::new()));
    let owner = Owner {
        executor: executor.clone(),
        streams: registry.clone(),
    };
    let dropped = Arc::new(AtomicUsize::new(0));
    let (entered, ready) = tokio::sync::oneshot::channel();
    let entered = Arc::new(Mutex::new(Some(entered)));
    let (release, receive) = std::sync::mpsc::channel();
    let receive = Arc::new(Mutex::new(receive));
    let probe = dropped.clone();
    let callback = Callback {
        executor: executor.clone(),
        streams: registry.clone(),
        unary: None,
        sync: None,
        stream: Some(Arc::new(move |_, _| {
            let entered = entered.clone();
            let receive = receive.clone();
            let probe = probe.clone();
            Box::pin(async move {
                entered.lock().unwrap().take().unwrap().send(()).unwrap();
                receive.lock().unwrap().recv().unwrap();
                Ok((
                    Box::pin(Probe(probe)) as LlmJsonAsyncStream,
                    Arc::new(Mutex::new(Vec::new())),
                ))
            })
        })),
    };
    let invocation = a::Invocation {
        surface: a::Surface::LlmStreamExecutionIntercept,
        name: "race".into(),
        value: abi::Json::Null,
        tool_call_id: Option::None,
        context: a::Context {
            scope: a::ScopeBox::new(Scope),
            request_codec: Option::None,
            response_codec: Option::None,
        },
    };
    let opening = a::Callback::open_stream(&callback, invocation, a::ContinuationBox::new(Next));
    let opening = tokio::spawn(opening);
    ready.await.unwrap();
    let closing = a::Component::shutdown(&owner);
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), closing)
        .await
        .unwrap()
        .unwrap();
    assert!(opening.await.unwrap().is_err());
    assert!(executor.is_closed());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(registry.lock().unwrap().is_empty());
}

struct ClosingStream(Arc<AtomicBool>);
impl a::Stream for ClosingStream {
    fn pull(&mut self) -> FfiFuture<a::Result<Option<abi::Json>>> {
        async { a::Result::Ok(Option::None) }.into_ffi()
    }
    fn close(&mut self) -> FfiFuture<a::Result<()>> {
        let closed = self.0.clone();
        async move {
            if !BOUND.get() {
                return a::Result::Err(failure("raw close must enter the invocation scope"));
            }
            tokio::task::yield_now().await;
            if !BOUND.get() {
                return a::Result::Err(failure("raw close must restore scope after Pending"));
            }
            closed.store(true, Ordering::Release);
            a::Result::Ok(())
        }
        .into_ffi()
    }
}

#[tokio::test]
async fn raw_stream_cleanup_restores_scope_on_each_poll() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "raw-close").unwrap();
    let closed = Arc::new(AtomicBool::new(false));
    let sources = Arc::new(Mutex::new(Vec::new()));
    SourceStream::new(
        a::StreamBox::new(ClosingStream(closed.clone())),
        &executor,
        Arc::new(a::ScopeBox::new(Scope)),
        &sources,
    );
    let state = Arc::new(StreamState {
        value: tokio::sync::Mutex::new(StreamValue {
            stream: None,
            terminal: false,
        }),
        scope: Arc::new(a::ScopeBox::new(Scope)),
        sources,
        closing: Mutex::new(None),
        closed: AtomicBool::new(false),
        cancelled: tokio_util::sync::CancellationToken::new(),
        pulls: Mutex::new(Vec::new()),
    });
    state.close(&executor).await.unwrap();
    assert!(closed.load(Ordering::Acquire));
    assert!(state.closed.load(Ordering::Acquire));
    a::Component::shutdown(&executor).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_drains_inline_and_cancelled_blocking_callbacks() {
    for mode in ["subscriber", "raw-subscriber", "raw-async"] {
        let executor =
            ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, mode).unwrap();
        let registry = Arc::new(Mutex::new(Vec::new()));
        let owner = Owner {
            executor: executor.clone(),
            streams: registry.clone(),
        };
        let (entered, ready) = tokio::sync::oneshot::channel();
        let entered = Arc::new(Mutex::new(Some(entered)));
        let (release, wait) = std::sync::mpsc::channel();
        let wait = Arc::new(Mutex::new(wait));
        let work = Arc::new(move || {
            entered.lock().unwrap().take().unwrap().send(()).unwrap();
            wait.lock().unwrap().recv().unwrap();
            Ok(Json::Null)
        });
        let callback = if mode == "subscriber" {
            a::CallbackBox::new(Callback {
                executor: executor.clone(),
                streams: registry,
                unary: None,
                stream: None,
                sync: Some(Arc::new(move |_| work())),
            })
        } else {
            a::CallbackBox::new(RawCallback {
                executor: executor.clone(),
                callback: Arc::new(move |_, _| work().map(into_abi)),
            })
        };
        let invocation = a::Invocation {
            surface: a::Surface::Subscriber,
            name: "inline".into(),
            value: abi::Json::Null,
            tool_call_id: Option::None,
            context: a::Context {
                scope: a::ScopeBox::new(Scope),
                request_codec: Option::None,
                response_codec: Option::None,
            },
        };
        let (thread, foreign) = if mode == "raw-async" {
            (None, Some(callback.invoke(invocation, Option::None)))
        } else {
            (
                Some(std::thread::spawn(move || callback.invoke_sync(invocation))),
                None,
            )
        };
        ready.await.unwrap();
        drop(foreign); // Cancel the async waiter while its blocking callback still runs.
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            a::Component::shutdown(&owner),
        )
        .await;
        let closed_while_running = matches!(&result, Ok(v) if v.is_ok());
        release.send(()).unwrap();
        if let Some(thread) = thread {
            thread.join().unwrap().unwrap();
        }
        a::Component::shutdown(&owner).await.unwrap();
        assert!(
            executor.enter_sync().is_err(),
            "revoked callbacks cannot enter"
        );
        assert_eq!(executor.active_tasks(), 0);
        assert!(!closed_while_running, "shutdown must drain {mode}");
    }
}

struct SourceProbe {
    dropped: Arc<AtomicUsize>,
    attempts: Arc<AtomicUsize>,
    fail_first: bool,
}
impl Drop for SourceProbe {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl a::Stream for SourceProbe {
    fn pull(&mut self) -> FfiFuture<a::Result<Option<abi::Json>>> {
        async { a::Result::Ok(Option::None) }.into_ffi()
    }
    fn close(&mut self) -> FfiFuture<a::Result<()>> {
        let attempts = self.attempts.clone();
        let fail_first = self.fail_first;
        async move {
            tokio::task::yield_now().await;
            assert!(
                BOUND.get(),
                "source cleanup must restore scope after Pending"
            );
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 && fail_first {
                a::Result::Err(failure("retry close"))
            } else {
                a::Result::Ok(())
            }
        }
        .into_ffi()
    }
}
struct RepeatedSource(Arc<AtomicUsize>);
impl a::Continuation for RepeatedSource {
    fn call(&self, _: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        unreachable!()
    }
    fn call_tool(&self, _: abi::Json) -> FfiFuture<a::Result<a::ToolExecutionResult>> {
        unreachable!()
    }
    fn open_stream(&self, _: abi::Json) -> FfiFuture<a::Result<a::StreamBox>> {
        let stream = a::StreamBox::new(SourceProbe {
            dropped: self.0.clone(),
            attempts: Arc::new(AtomicUsize::new(0)),
            fail_first: false,
        });
        async move { a::Result::Ok(stream) }.into_ffi()
    }
}
#[tokio::test]
async fn repeated_stream_sources_release_successful_cleanup_before_outer_close() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "sources").unwrap();
    let sources = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let next = LlmStreamNext {
        next: Arc::new(a::ContinuationBox::new(RepeatedSource(dropped.clone()))),
        sources: sources.clone(),
        executor: executor.clone(),
        scope: Arc::new(a::ScopeBox::new(Scope)),
    };
    for count in 1..=100 {
        let mut stream = next
            .call(LlmRequest {
                headers: Map::new(),
                content: Json::Null,
            })
            .await
            .unwrap();
        // Exercise both terminal streams retained by their caller and early drops.
        if count % 2 == 0 {
            assert!(stream.next().await.is_none());
        } else {
            drop(stream);
            stream = Box::pin(futures::stream::empty());
        }
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while dropped.load(Ordering::SeqCst) != count || !sources.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(stream);
    }
    a::Component::shutdown(&executor).await.unwrap();
}
#[tokio::test]
async fn failed_source_cleanup_retains_ownership_for_retry() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "source-retry").unwrap();
    let sources = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    let source = SourceStream::new(
        a::StreamBox::new(SourceProbe {
            dropped: dropped.clone(),
            attempts: attempts.clone(),
            fail_first: true,
        }),
        &executor,
        Arc::new(a::ScopeBox::new(Scope)),
        &sources,
    );
    assert!(source.close().await.is_err());
    assert_eq!(sources.lock().unwrap().len(), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    source.close().await.unwrap();
    assert!(sources.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    a::Component::shutdown(&executor).await.unwrap();
}

pub(crate) fn raw_invocation(value: Json) -> a::Invocation {
    a::Invocation {
        surface: a::Surface::ToolExecutionIntercept,
        name: "raw-tool".into(),
        value: into_abi(value),
        tool_call_id: Option::Some("call-id".into()),
        context: a::Context {
            scope: a::ScopeBox::new(Scope),
            request_codec: Option::None,
            response_codec: Option::None,
        },
    }
}

/// Raw synchronous descriptors preserve tool results, marks, panic errors, and admission fencing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_callback_contract_rejects_invalid_dispatch_and_preserves_outcomes() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "raw-contract").unwrap();
    for mode in ["success", "null-annotation", "failure", "panic"] {
        let callback = RawCallback {
            executor: executor.clone(),
            callback: Arc::new(move |invocation, _| {
                assert!(BOUND.get(), "the callback must enter its originating scope");
                assert_eq!(invocation.tool_call_id.unwrap().as_str(), "call-id");
                match mode {
                    "failure" => Err("raw callback failure".into()),
                    "panic" => panic!("raw callback panic"),
                    _ => Ok(into_abi(serde_json::json!({
                        "result":{"value":42},
                        "annotation":if mode == "success" { serde_json::json!({"source":"raw"}) } else { Json::Null },
                        "pending_marks":[{"name":"raw-mark"}]
                    }))),
                }
            }),
        };
        let outcome = a::Callback::invoke_tool(
            &callback,
            raw_invocation(Json::Null),
            a::ContinuationBox::new(Next),
        )
        .await;
        if mode == "failure" || mode == "panic" {
            assert!(
                outcome
                    .err()
                    .unwrap()
                    .message
                    .contains(if mode == "failure" {
                        "failure"
                    } else {
                        "panicked"
                    })
            );
        } else {
            let outcome = outcome.unwrap();
            assert_eq!(
                from_abi(outcome.result.result).unwrap(),
                serde_json::json!({"value":42})
            );
            assert_eq!(outcome.result.annotation.is_some(), mode == "success");
            let marks: Vec<crate::PendingMarkSpec> = outcome
                .pending_marks
                .into_iter()
                .map(|value| decode(value).unwrap())
                .collect();
            assert_eq!(marks[0].name, "raw-mark");
            a::Callback::invoke(&callback, raw_invocation(Json::Null), Option::None)
                .await
                .unwrap();
            a::Callback::invoke_sync(&callback, raw_invocation(Json::Null)).unwrap();
            assert!(!BOUND.get(), "the caller's scope must be restored");
        }
        assert!(
            a::Callback::open_stream(
                &callback,
                raw_invocation(Json::Null),
                a::ContinuationBox::new(Next)
            )
            .await
            .err()
            .unwrap()
            .message
            .contains("synchronous streaming")
        );
    }
    a::Component::shutdown(&executor).await.unwrap();
    let callback = RawCallback {
        executor: executor.clone(),
        callback: Arc::new(|_, _| panic!("revoked callback ran")),
    };
    assert!(a::Callback::invoke_sync(&callback, raw_invocation(Json::Null)).is_err());
    assert_eq!(executor.active_tasks(), 0);
}

#[test]
fn native_author_defaults_validate_executor_configuration() {
    struct Author;
    impl Plugin for Author {
        fn plugin_kind(&self) -> &str {
            "executor.defaults"
        }
        fn register<'a>(
            &'a self,
            _: &'a Map<String, Json>,
            _: &'a mut PluginContext<'_>,
        ) -> PluginFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }
    }
    assert!(Author.allows_multiple_components());
    assert!(Author.executor_config().is_none());
    assert!(
        Author
            .executor_config_for_component(&Map::new())
            .unwrap()
            .is_none()
    );
    let enabled = serde_json::json!({"executor":{"worker_threads":1}})
        .as_object()
        .unwrap()
        .clone();
    assert_eq!(
        Author
            .executor_config_for_component(&enabled)
            .unwrap()
            .unwrap()
            .worker_threads,
        1
    );
    assert!(Author.validate(&Map::new()).is_empty());
    let invalid = serde_json::json!({"executor":{"worker_threads":0}})
        .as_object()
        .unwrap()
        .clone();
    let diagnostics = Author.validate(&invalid);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].field.as_deref(),
        Some("executor.worker_threads")
    );
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
}

#[test]
fn checked_author_factory_contains_panics_and_rejects_falsified_identity() {
    struct Author;
    impl Plugin for Author {
        fn plugin_kind(&self) -> &str {
            "author"
        }
        fn register<'a>(
            &'a self,
            _: &'a Map<String, Json>,
            _: &'a mut PluginContext<'_>,
        ) -> PluginFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }
    }
    assert!(create_plugin(|| Author, "falsified").is_err());
    let panic = create_plugin(|| -> Author { panic!("constructor failed") }, "author")
        .err()
        .unwrap();
    assert!(panic.message.contains("constructor panicked"));
    let adapter = Adapter(Arc::new(Author));
    let diagnostics = a::Plugin::validate(&adapter, into_abi(serde_json::json!([])));
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].is_error);
    assert!(a::Plugin::allows_multiple_components(&adapter));
}

#[test]
fn invocation_codec_adapters_preserve_typed_values_and_missing_capabilities() {
    struct Codec(bool);
    impl a::Codec for Codec {
        fn identity(&self) -> a::Result<abi::Json> {
            boundary(encode(
                serde_json::json!({"identity":LlmCodecIdentity::Opaque,"available":true}),
            ))
        }
        fn decode(&self, _: abi::Json) -> a::Result<abi::Json> {
            if self.0 {
                boundary(encode(
                    nemo_relay_types::codec::response::AnnotatedLlmResponse::default(),
                ))
            } else {
                boundary(encode(AnnotatedLlmRequest::default()))
            }
        }
        fn encode(&self, _: abi::Json, original: abi::Json) -> a::Result<abi::Json> {
            a::Result::Ok(original)
        }
    }
    let codec = |response| Option::Some(a::CodecBox::new(Codec(response)));
    let context = LlmExecutionContext::new(codec(false), codec(true)).unwrap();
    assert_eq!(context.request_codec().codec, LlmCodecIdentity::Opaque);
    assert_eq!(
        context.response_codec().unwrap().codec,
        LlmCodecIdentity::Opaque
    );
    let request = LlmRequest {
        headers: Default::default(),
        content: serde_json::json!({"provider":"request","unknown":[u64::MAX]}),
    };
    let encoder = context.request_codec().resolve_codec().unwrap();
    let annotated = encoder.decode(&request).unwrap();
    assert_eq!(annotated, AnnotatedLlmRequest::default());
    assert_eq!(encoder.encode(&annotated, &request).unwrap(), request);
    let decoder = context.response_codec().unwrap().resolve_codec().unwrap();
    assert_eq!(
        decoder
            .decode(&serde_json::json!({"provider":"response"}))
            .unwrap(),
        nemo_relay_types::codec::response::AnnotatedLlmResponse::default()
    );
    let request_context = LlmSanitizeRequestContext::new(codec(false)).unwrap();
    assert!(request_context.resolve_codec().is_some());
    let response_context = LlmSanitizeResponseContext::new(codec(true)).unwrap();
    assert!(response_context.resolve_codec().is_some());
    let missing = LlmExecutionContext::new(Option::None, Option::None).unwrap();
    assert_eq!(missing.request_codec().codec, LlmCodecIdentity::None);
    assert!(missing.request_codec().resolve_codec().is_none());
    assert!(missing.response_codec().is_none());
    assert!(
        LlmSanitizeResponseContext::new(Option::None)
            .unwrap()
            .resolve_codec()
            .is_none()
    );
}
