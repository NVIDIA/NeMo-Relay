// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identical conformance checks compiled by current and future hosts.
use abi::{
    Field, Json,
    future::{FfiFuture, FutureExt},
    v1_0::*,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "foreign future did not wake/complete"
        );
        std::thread::park_timeout(Duration::from_millis(10));
    }
}
async fn pending_poll() {
    let mut ready = false;
    std::future::poll_fn(move |cx| {
        if ready {
            Poll::Ready(())
        } else {
            ready = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}
fn poll_pending<F: Future + Unpin>(future: &mut F) {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    assert!(
        Pin::new(future)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
}
#[derive(Default)]
struct Counts {
    registered: AtomicUsize,
    cancelled: AtomicUsize,
    shutdown: AtomicUsize,
    closed: AtomicUsize,
    dropped: AtomicUsize,
}
struct Runtime {
    counts: Arc<Counts>,
    bound: Arc<Mutex<Option<ComponentBox>>>,
}
impl abi::v1_0::Runtime for Runtime {
    fn call_sync(&self, operation: String, args: Json) -> Result<Json> {
        match operation.as_str() {
            "fixture.register" => {
                self.counts.registered.fetch_add(1, Ordering::SeqCst);
            }
            "fixture.cancelled" => {
                self.counts.cancelled.fetch_add(1, Ordering::SeqCst);
            }
            "fixture.shutdown" => {
                self.counts.shutdown.fetch_add(1, Ordering::SeqCst);
            }
            _ => panic!("unexpected runtime operation"),
        }
        Result::Ok(args)
    }
    fn call(&self, operation: String, args: Json) -> FfiFuture<Result<Json>> {
        let result = self.call_sync(operation, args);
        async move {
            pending_poll().await;
            result
        }
        .into_ffi()
    }
    fn diagnostics(&self) -> abi::diagnostics::DiagnosticsBox {
        unreachable!()
    }
    fn capture_scope(&self) -> Result<ScopeBox> {
        Result::Ok(scope())
    }
    fn bind_component(&self, component: ComponentBox) -> Result<()> {
        assert!(self.bound.lock().unwrap().replace(component).is_none());
        Result::Ok(())
    }
    fn register_callback(&self, _: Registration) -> Result<()> {
        unreachable!()
    }
    fn register_gate(&self, _: Registration) -> Result<String> {
        unreachable!()
    }
}
struct Scope;
struct Lease;
impl abi::v1_0::ScopeLease for Lease {
    fn restore(&mut self) -> Result<()> {
        Result::Ok(())
    }
}
impl abi::v1_0::Scope for Scope {
    fn enter(&self) -> Result<ScopeLeaseBox> {
        Result::Ok(ScopeLeaseBox::new(Lease))
    }
    fn snapshot(&self) -> Result<Option<Json>> {
        Result::Ok(Option::Some(Json::String("scope".into())))
    }
}
fn scope() -> ScopeBox {
    ScopeBox::new(Scope)
}
struct Next(Arc<Counts>);
impl Continuation for Next {
    fn call(&self, value: Json) -> FfiFuture<Result<Json>> {
        async move {
            pending_poll().await;
            if value == Json::Null {
                Result::Err(Error::new("downstream", "expected failure"))
            } else {
                Result::Ok(value)
            }
        }
        .into_ffi()
    }
    fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>> {
        async move {
            pending_poll().await;
            Result::Ok(ToolExecutionResult {
                result: value,
                annotation: Option::Some(Json::Unsigned(u64::MAX)),
            })
        }
        .into_ffi()
    }
    fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>> {
        let counts = self.0.clone();
        async move {
            pending_poll().await;
            Result::Ok(StreamBox::new(Source {
                value: Some(value),
                counts,
                closed: false,
            }))
        }
        .into_ffi()
    }
}
struct Source {
    value: Option<Json>,
    counts: Arc<Counts>,
    closed: bool,
}
impl Drop for Source {
    fn drop(&mut self) {
        self.counts.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl abi::v1_0::Stream for Source {
    fn pull(&mut self) -> FfiFuture<Result<Option<Json>>> {
        let value = self.value.take();
        async move {
            pending_poll().await;
            Result::Ok(value)
        }
        .into_ffi()
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        let first = !std::mem::replace(&mut self.closed, true);
        let counts = self.counts.clone();
        async move {
            pending_poll().await;
            if first {
                counts.closed.fetch_add(1, Ordering::SeqCst);
            }
            Result::Ok(())
        }
        .into_ffi()
    }
}
fn next(counts: &Arc<Counts>) -> ContinuationBox {
    ContinuationBox::new(Next(counts.clone()))
}
fn invocation(surface: Surface, value: Json) -> Invocation {
    Invocation {
        surface,
        name: "fixture".into(),
        value,
        tool_call_id: Option::Some("call-id".into()),
        context: invocation_context(),
    }
}
fn invocation_context() -> abi::v1_0::Context {
    abi::v1_0::Context {
        scope: scope(),
        request_codec: Option::None,
        response_codec: Option::None,
    }
}

pub fn assert_behavior(plugin: PluginBox) {
    // Repeat with the same plugin definition to prove fresh component state.
    assert_component(&plugin);
    assert_component(&plugin);
}

fn assert_component(plugin: &PluginBox) {
    let counts = Arc::new(Counts::default());
    let bound = Arc::new(Mutex::new(None));
    let payload = Json::Object(vec![Field {
        key: "Unicode λ\0".into(),
        value: Json::Array(vec![
            Json::Signed(i64::MIN),
            Json::Unsigned(u64::MAX),
            Json::Float(0.5),
        ]),
    }]);
    let registered = block_on(plugin.register(
        payload.clone(),
        RuntimeBox::new(Runtime {
            counts: counts.clone(),
            bound: bound.clone(),
        }),
    ))
    .unwrap();
    assert!(registered.error.is_none());
    assert_eq!(counts.registered.load(Ordering::SeqCst), 1);
    assert_eq!(registered.registrations.len(), 4);
    let registrations = registered.registrations;
    assert_eq!(
        registrations[0]
            .callback
            .invoke_sync(invocation(Surface::Subscriber, payload.clone()))
            .unwrap(),
        payload
    );
    let callback = &registrations[1].callback;
    drop(callback.invoke(
        invocation(Surface::LlmExecutionIntercept, payload.clone()),
        Option::None,
    ));
    assert_eq!(counts.cancelled.load(Ordering::SeqCst), 1);
    let mut cancelled = callback.invoke(
        invocation(Surface::LlmExecutionIntercept, payload.clone()),
        Option::None,
    );
    poll_pending(&mut cancelled);
    drop(cancelled);
    assert_eq!(counts.cancelled.load(Ordering::SeqCst), 2);
    let mut first = callback.invoke(
        invocation(Surface::LlmExecutionIntercept, payload.clone()),
        Option::Some(next(&counts)),
    );
    poll_pending(&mut first);
    let second = callback.invoke(
        invocation(Surface::LlmExecutionIntercept, payload.clone()),
        Option::Some(next(&counts)),
    );
    assert_eq!(block_on(second).unwrap(), payload);
    assert_eq!(block_on(first).unwrap(), payload);
    let error = block_on(callback.invoke(
        invocation(Surface::LlmExecutionIntercept, Json::Null),
        Option::Some(next(&counts)),
    ))
    .unwrap_err();
    assert_eq!(error.code.as_str(), "downstream");
    let tool = block_on(registrations[2].callback.invoke_tool(
        invocation(Surface::ToolExecutionIntercept, payload.clone()),
        next(&counts),
    ))
    .unwrap();
    assert_eq!(tool.result.result, payload);
    assert_eq!(
        tool.result.annotation,
        Option::Some(Json::Unsigned(u64::MAX))
    );
    assert!(tool.pending_marks.is_empty());
    assert_stream(&registrations[3].callback, &payload, &counts);
    block_on(registered.component.shutdown()).unwrap();
    block_on(registered.component.shutdown()).unwrap();
    assert_eq!(counts.shutdown.load(Ordering::SeqCst), 1);
    let error = block_on(callback.invoke(
        invocation(Surface::LlmExecutionIntercept, payload.clone()),
        Option::None,
    ))
    .unwrap_err();
    assert_eq!(error.code.as_str(), "revoked");
    drop(registrations);
    drop(registered.component);
    bound.lock().unwrap().take();
}

fn assert_stream(callback: &CallbackBox, payload: &Json, counts: &Arc<Counts>) {
    let mut stream = block_on(callback.open_stream(
        invocation(Surface::LlmStreamExecutionIntercept, payload.clone()),
        next(counts),
    ))
    .unwrap();
    assert_eq!(
        block_on(stream.pull()).unwrap(),
        Option::Some(payload.clone())
    );
    assert_eq!(block_on(stream.pull()).unwrap(), Option::None);
    block_on(stream.close()).unwrap();
    block_on(stream.close()).unwrap();
    drop(stream);
    assert_eq!(counts.closed.load(Ordering::SeqCst), 1);
    assert_eq!(counts.dropped.load(Ordering::SeqCst), 1);
}
