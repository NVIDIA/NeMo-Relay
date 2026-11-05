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
        a::Result::ROk(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = a::ScopeLease::restore(self);
    }
}
impl a::Scope for Scope {
    fn enter(&self) -> a::Result<a::ScopeLeaseBox> {
        a::Result::ROk(a::ScopeLease_TO::from_value(
            Lease(Some(BOUND.replace(true))),
            TD_Opaque,
        ))
    }
    fn snapshot(&self) -> a::Result<ROption<abi::Json>> {
        a::Result::ROk(ROption::RNone)
    }
}
struct Next;
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
        tool_call_id: ROption::RNone,
        context: a::Context {
            scope: a::Scope_TO::from_value(Scope, TD_Opaque),
            request_codec: ROption::RNone,
            response_codec: ROption::RNone,
        },
    };
    let opening = a::Callback::open_stream(
        &callback,
        invocation,
        a::Continuation_TO::from_value(Next, TD_Opaque),
    );
    ready.await.unwrap();
    let closing = a::Component::shutdown(&owner);
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), closing)
        .await
        .unwrap()
        .into_result()
        .unwrap();
    let mut stream = opening.await.into_result().unwrap();
    assert!(executor.is_closed());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(registry.lock().unwrap().is_empty());
    stream.close().await.into_result().unwrap();
}

struct ClosingStream(Arc<AtomicBool>);
impl a::Stream for ClosingStream {
    fn pull(&mut self) -> FfiFuture<a::Result<ROption<abi::Json>>> {
        async { a::Result::ROk(ROption::RNone) }.into_ffi()
    }
    fn close(&mut self) -> FfiFuture<a::Result<()>> {
        let closed = self.0.clone();
        async move {
            if !BOUND.get() {
                return a::Result::RErr(failure("raw close must enter the invocation scope"));
            }
            tokio::task::yield_now().await;
            if !BOUND.get() {
                return a::Result::RErr(failure("raw close must restore scope after Pending"));
            }
            closed.store(true, Ordering::Release);
            a::Result::ROk(())
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
        a::Stream_TO::from_value(ClosingStream(closed.clone()), TD_Opaque),
        &executor,
        Arc::new(a::Scope_TO::from_value(Scope, TD_Opaque)),
        &sources,
    );
    let state = Arc::new(StreamState {
        value: tokio::sync::Mutex::new(StreamValue {
            stream: None,
            terminal: false,
        }),
        scope: Arc::new(a::Scope_TO::from_value(Scope, TD_Opaque)),
        sources,
        closing: Mutex::new(None),
        closed: AtomicBool::new(false),
        cancelled: tokio_util::sync::CancellationToken::new(),
    });
    state.close(&executor).await.into_result().unwrap();
    assert!(closed.load(Ordering::Acquire));
    assert!(state.closed.load(Ordering::Acquire));
    a::Component::shutdown(&executor)
        .await
        .into_result()
        .unwrap();
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
            a::Callback_TO::from_value(
                Callback {
                    executor: executor.clone(),
                    streams: registry,
                    unary: None,
                    stream: None,
                    sync: Some(Arc::new(move |_| work())),
                },
                TD_Opaque,
            )
        } else {
            a::Callback_TO::from_value(
                RawCallback {
                    executor: executor.clone(),
                    callback: Arc::new(move |_, _| work().map(into_abi)),
                },
                TD_Opaque,
            )
        };
        let invocation = a::Invocation {
            surface: a::Surface::Subscriber,
            name: "inline".into(),
            value: abi::Json::Null,
            tool_call_id: ROption::RNone,
            context: a::Context {
                scope: a::Scope_TO::from_value(Scope, TD_Opaque),
                request_codec: ROption::RNone,
                response_codec: ROption::RNone,
            },
        };
        let (thread, foreign) = if mode == "raw-async" {
            (None, Some(callback.invoke(invocation, ROption::RNone)))
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
            thread.join().unwrap().into_result().unwrap();
        }
        a::Component::shutdown(&owner).await.into_result().unwrap();
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
    fn pull(&mut self) -> FfiFuture<a::Result<ROption<abi::Json>>> {
        async { a::Result::ROk(ROption::RNone) }.into_ffi()
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
                a::Result::RErr(failure("retry close"))
            } else {
                a::Result::ROk(())
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
        let stream = a::Stream_TO::from_value(
            SourceProbe {
                dropped: self.0.clone(),
                attempts: Arc::new(AtomicUsize::new(0)),
                fail_first: false,
            },
            TD_Opaque,
        );
        async move { a::Result::ROk(stream) }.into_ffi()
    }
}
#[tokio::test]
async fn repeated_stream_sources_release_successful_cleanup_before_outer_close() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "sources").unwrap();
    let sources = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let next = LlmStreamNext {
        next: Arc::new(a::Continuation_TO::from_value(
            RepeatedSource(dropped.clone()),
            TD_Opaque,
        )),
        sources: sources.clone(),
        executor: executor.clone(),
        scope: Arc::new(a::Scope_TO::from_value(Scope, TD_Opaque)),
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
    a::Component::shutdown(&executor)
        .await
        .into_result()
        .unwrap();
}
#[tokio::test]
async fn failed_source_cleanup_retains_ownership_for_retry() {
    let executor =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "source-retry").unwrap();
    let sources = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    let source = SourceStream::new(
        a::Stream_TO::from_value(
            SourceProbe {
                dropped: dropped.clone(),
                attempts: attempts.clone(),
                fail_first: true,
            },
            TD_Opaque,
        ),
        &executor,
        Arc::new(a::Scope_TO::from_value(Scope, TD_Opaque)),
        &sources,
    );
    assert!(source.close().await.is_err());
    assert_eq!(sources.lock().unwrap().len(), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    source.close().await.unwrap();
    assert!(sources.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    a::Component::shutdown(&executor)
        .await
        .into_result()
        .unwrap();
}
