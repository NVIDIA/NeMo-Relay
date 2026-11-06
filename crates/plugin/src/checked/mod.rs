// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SDK implementation of the checked native interface.
#![allow(missing_docs)]

use crate::{
    AnnotatedLlmRequest, ConfigDiagnostic, DataSchema, DiagnosticLevel, Event, EventSanitizeFields,
    Json, LlmCodecIdentity, LlmRequest, LlmRequestInterceptOutcome, LogSeverity,
    METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope, MetricMeasurement,
    NativeExecutorConfig, PluginFuture, PluginLogLevel, RuntimeDiagnostics,
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, ScopeStackSnapshot, ScopeType,
    ToolExecutionInterceptOutcome, ToolExecutionResult,
    abi::{
        self,
        abi_stable::{
            sabi_trait::prelude::TD_Opaque,
            std_types::{ROption, RVec},
        },
        async_ffi::{FfiFuture, FutureExt as _},
        v1_0 as a,
    },
    abi_executor::{ComponentExecutor, DirectCancellation},
    abi_value::{from_abi, into_abi},
};
use futures::{
    FutureExt as _, Stream, StreamExt,
    future::{BoxFuture, Shared},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Map;
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

pub type Result<T> = std::result::Result<T, String>;
pub use nemo_relay_types::api::tool::ToolExecutionContext;
pub type LlmJsonAsyncStream = Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;
fn encode<T: Serialize>(value: T) -> Result<abi::Json> {
    serde_json::to_value(value)
        .map(into_abi)
        .map_err(|e| e.to_string())
}
fn decode<T: DeserializeOwned>(value: abi::Json) -> Result<T> {
    serde_json::from_value(from_abi(value)?).map_err(|e| e.to_string())
}
fn failure(error: impl ToString) -> a::Error {
    let message = error.to_string();
    let status =
        nemo_relay_types::plugin::parse_http_error_message(&message).map(|(status, _)| status);
    let mut error = a::Error::new("plugin", message);
    error.http_status = status.into();
    error
}
fn boundary<T>(value: Result<T>) -> a::Result<T> {
    value.map_err(failure).into()
}
/// Restore SDK errors from ABI results while retaining upstream HTTP status in their messages.
fn local<T>(value: a::Result<T>) -> Result<T> {
    value.into_result().map_err(|e| {
        let message = e.message.to_string();
        if let Some(status) = e.http_status.into_option()
            && nemo_relay_types::plugin::parse_http_error_message(&message).is_none()
        {
            return crate::http_error_message(status, &message).unwrap_or(message);
        }
        message
    })
}

pub trait Plugin: Send + Sync + 'static {
    fn plugin_kind(&self) -> &str;
    fn allows_multiple_components(&self) -> bool {
        true
    }
    /// Opt into a shared plugin Tokio runtime; inline execution needs no runtime.
    fn executor_config(&self) -> Option<NativeExecutorConfig> {
        None
    }
    fn executor_config_for_component(
        &self,
        config: &Map<String, Json>,
    ) -> Result<Option<NativeExecutorConfig>> {
        let defaults = self.executor_config();
        if config.contains_key("executor") {
            defaults
                .unwrap_or_default()
                .with_component_config(config)
                .map(Some)
        } else {
            defaults
                .map(|config| config.with_component_config(&Map::new()))
                .transpose()
        }
    }
    fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        self.executor_config_for_component(config)
            .err()
            .map(|message| ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "native_executor_config.invalid".into(),
                component: None,
                field: Some("executor.worker_threads".into()),
                message,
            })
            .into_iter()
            .collect()
    }
    fn register<'a>(
        &'a self,
        config: &'a Map<String, Json>,
        ctx: &'a mut PluginContext<'_>,
    ) -> PluginFuture<'a, ()>;
}
pub use Plugin as NativePlugin;

pub struct PluginContext<'a> {
    registrations: Vec<a::Registration>,
    runtime: PluginRuntime,
    executor: ComponentExecutor,
    streams: Arc<Mutex<Vec<Arc<StreamState>>>>,
    _borrow: PhantomData<&'a mut ()>,
}
/// Cloneable admission handle for background tasks owned by one native component.
#[derive(Clone)]
pub struct PluginTaskSpawner {
    runtime: PluginRuntime,
    executor: ComponentExecutor,
}
impl PluginTaskSpawner {
    /// Spawn work with the current Relay scope and cancel it when the component closes.
    /// Dropping the join handle detaches the waiter; component ownership remains tracked.
    pub fn spawn<F, T>(&self, future: F) -> Result<tokio::task::JoinHandle<Result<T>>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let scope = local(self.runtime.raw.capture_scope())?;
        self.executor
            .spawn(async move { local(scoped(scope, async move { boundary(future.await) }).await) })
    }
}

enum UserResult {
    Value(Json),
    Tool(ToolExecutionInterceptOutcome),
}
type Unary = dyn Fn(
        a::Invocation,
        ROption<a::ContinuationBox>,
    ) -> Pin<Box<dyn Future<Output = Result<UserResult>> + Send>>
    + Send
    + Sync;
type SyncCallback = dyn Fn(a::Invocation) -> Result<Json> + Send + Sync;
type Streaming = dyn Fn(
        a::Invocation,
        a::ContinuationBox,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(LlmJsonAsyncStream, Arc<Mutex<Vec<Arc<SourceStream>>>>)>>
                + Send,
        >,
    > + Send
    + Sync;
struct Callback {
    executor: ComponentExecutor,
    streams: Arc<Mutex<Vec<Arc<StreamState>>>>,
    unary: Option<Arc<Unary>>,
    sync: Option<Arc<SyncCallback>>,
    stream: Option<Arc<Streaming>>,
}

// The host lease stays on this poll's thread. Only the scope capability and
// suspended future move between polling threads.
fn scoped<F: Future>(scope: a::ScopeBox, future: F) -> ScopedFuture<F> {
    ScopedFuture {
        scope,
        future: Some(Box::pin(future)),
        cleanup: false,
    }
}
struct ScopedFuture<F> {
    scope: a::ScopeBox,
    future: Option<Pin<Box<F>>>,
    cleanup: bool,
}
impl<F> Unpin for ScopedFuture<F> {}
impl<F: Future<Output = a::Result<T>>, T> Future for ScopedFuture<F> {
    type Output = a::Result<T>;
    /// Enter the invocation scope for each user-future poll and restore the previous thread binding afterward.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let lease = match this.scope.enter().into_result() {
            Ok(v) => Some(v),
            Err(_) if this.cleanup => None,
            Err(e) => return Poll::Ready(a::Result::RErr(e)),
        };
        let result = this
            .future
            .as_mut()
            .expect("scoped future owns its state")
            .as_mut()
            .poll(cx);
        if let Some(mut lease) = lease
            && let Err(e) = lease.restore().into_result()
        {
            return Poll::Ready(a::Result::RErr(e));
        }
        result
    }
}
impl<F> Drop for ScopedFuture<F> {
    /// Attempt to bind the captured scope while dropping the user future, then restore the previous thread binding.
    fn drop(&mut self) {
        let lease = self.scope.enter().into_result().ok();
        self.future.take();
        if let Some(mut lease) = lease {
            let _ = lease.restore();
        }
    }
}
impl a::Callback for Callback {
    /// Poll a unary callback under its captured scope and optional SDK runtime context.
    fn invoke(
        &self,
        invocation: a::Invocation,
        next: ROption<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        let callback = self.unary.clone();
        // Move scope out without replacing it with a fake identifier.
        let scope = Arc::new(invocation.context.scope);
        let invocation = a::Invocation {
            context: a::Context {
                scope: a::Scope_TO::from_value(SharedScope(scope.clone()), TD_Opaque),
                ..invocation.context
            },
            ..invocation
        };
        self.executor.execute_direct(scoped(
            a::Scope_TO::from_value(SharedScope(scope), TD_Opaque),
            async move {
                boundary(
                    async {
                        let callback = callback.ok_or("callback is synchronous or streaming")?;
                        match callback(invocation, next).await? {
                            UserResult::Value(v) => Ok(into_abi(v)),
                            UserResult::Tool(_) => {
                                Err("tool callback requires typed invocation".into())
                            }
                        }
                    }
                    .await,
                )
            },
        ))
    }

    fn invoke_tool(
        &self,
        invocation: a::Invocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        let callback = self.unary.clone();
        let scope = Arc::new(invocation.context.scope);
        let invocation = a::Invocation {
            context: a::Context {
                scope: a::Scope_TO::from_value(SharedScope(scope.clone()), TD_Opaque),
                ..invocation.context
            },
            ..invocation
        };
        self.executor.execute_direct(scoped(
            a::Scope_TO::from_value(SharedScope(scope), TD_Opaque),
            async move {
                boundary(
                    async {
                        let callback = callback.ok_or("callback is not a tool interceptor")?;
                        let UserResult::Tool(value) =
                            callback(invocation, ROption::RSome(next)).await?
                        else {
                            return Err("tool callback returned an invalid outcome".into());
                        };
                        Ok(a::ToolExecutionOutcome {
                            result: a::ToolExecutionResult {
                                result: into_abi(value.result),
                                annotation: value
                                    .annotation
                                    .filter(|v| !v.is_null())
                                    .map(into_abi)
                                    .into(),
                            },
                            pending_marks: value
                                .pending_marks
                                .into_iter()
                                .map(encode)
                                .collect::<Result<RVec<_>>>()?,
                        })
                    }
                    .await,
                )
            },
        ))
    }

    fn invoke_sync(&self, invocation: a::Invocation) -> a::Result<abi::Json> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _call = local(self.executor.enter_sync())?;
            self.executor.with_context(|| {
                let mut lease = local(invocation.context.scope.enter())?;
                let value = self.sync.as_ref().ok_or("callback is not synchronous")?(invocation);
                local(lease.restore())?;
                value.map(into_abi)
            })?
        }))
        .unwrap_or_else(|_| Err("native synchronous callback panicked".into()));
        boundary(result)
    }
    fn open_stream(
        &self,
        invocation: a::Invocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        let callback = self.stream.clone();
        let executor = self.executor.clone();
        let registry = self.streams.clone();
        // Retain invocation scope for each stream pull, including Pending polls.
        let scope = Arc::new(invocation.context.scope);
        let bound = scope.clone();
        let invocation = a::Invocation {
            context: a::Context {
                scope: a::Scope_TO::from_value(SharedScope(bound), TD_Opaque),
                ..invocation.context
            },
            ..invocation
        };
        self.executor.execute_direct(async move {
            let result = async {
                let callback = callback.ok_or("callback is not streaming")?;
                let (stream, sources) = local(
                    scoped(
                        a::Scope_TO::from_value(SharedScope(scope.clone()), TD_Opaque),
                        async move { boundary(callback(invocation, next).await) },
                    )
                    .await,
                )?;
                let state = Arc::new(StreamState {
                    value: tokio::sync::Mutex::new(StreamValue {
                        stream: Some(stream),
                        terminal: false,
                    }),
                    scope,
                    sources,
                    closing: Mutex::new(None),
                    closed: AtomicBool::new(false),
                    cancelled: tokio_util::sync::CancellationToken::new(),
                    pulls: Mutex::new(Vec::new()),
                });
                {
                    let mut registry = registry.lock().unwrap_or_else(|e| e.into_inner());
                    registry.retain(|v| !v.closed.load(Ordering::Acquire));
                    registry.push(state.clone());
                }
                Ok(a::Stream_TO::from_value(
                    OutputStream { state, executor },
                    TD_Opaque,
                ))
            }
            .await;
            boundary(result)
        })
    }
}
struct SharedScope(Arc<a::ScopeBox>);
impl a::Scope for SharedScope {
    fn enter(&self) -> a::Result<a::ScopeLeaseBox> {
        self.0.enter()
    }
    fn snapshot(&self) -> a::Result<ROption<abi::Json>> {
        self.0.snapshot()
    }
}
struct StreamValue {
    stream: Option<LlmJsonAsyncStream>,
    terminal: bool,
}
type CloseFuture = Shared<BoxFuture<'static, a::Result<()>>>;
struct StreamState {
    value: tokio::sync::Mutex<StreamValue>,
    scope: Arc<a::ScopeBox>,
    sources: Arc<Mutex<Vec<Arc<SourceStream>>>>,
    closing: Mutex<Option<CloseFuture>>,
    closed: AtomicBool,
    cancelled: tokio_util::sync::CancellationToken,
    pulls: Mutex<Vec<DirectCancellation>>,
}
impl StreamState {
    /// Cancel stream pulls and close retained sources under the invocation scope, preserving failed cleanup for retry.
    fn close(self: &Arc<Self>, executor: &ComponentExecutor) -> CloseFuture {
        self.cancelled.cancel();
        let pulls = std::mem::take(&mut *self.pulls.lock().unwrap_or_else(|e| e.into_inner()));
        for pull in pulls {
            pull.cancel();
        }
        let mut closing = self.closing.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(future) = closing.as_ref()
            && future.peek().is_none_or(|result| result.is_ok())
        {
            return future.clone();
        }
        let state = self.clone();
        // Cleanup still runs after revocation. Bind scope on each poll while
        // authority is available, including polls of a raw stream's close future.
        let scope = a::Scope_TO::from_value(SharedScope(state.scope.clone()), TD_Opaque);
        let mut cleanup = scoped(scope, async move {
            {
                let mut value = state.value.lock().await;
                value.stream.take();
                value.terminal = true;
            }
            let sources = state
                .sources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            for source in sources {
                if let Err(error) = source.close().await {
                    return a::Result::RErr(failure(error));
                }
            }
            state
                .sources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            state.closed.store(true, Ordering::Release);
            a::Result::ROk(())
        });
        cleanup.cleanup = true;
        let foreign = executor.execute_cleanup(cleanup);
        let future = foreign.boxed().shared();
        *closing = Some(future.clone());
        future
    }
}
struct OutputStream {
    state: Arc<StreamState>,
    executor: ComponentExecutor,
}
impl a::Stream for OutputStream {
    /// Poll one output item under the invocation scope and make cancellation, EOF, or an error terminal.
    fn pull(&mut self) -> FfiFuture<a::Result<ROption<abi::Json>>> {
        let mut pulls = self.state.pulls.lock().unwrap_or_else(|e| e.into_inner());
        if self.state.cancelled.is_cancelled() {
            return async { a::Result::RErr(a::Error::new("closed", "native stream is closing")) }
                .into_ffi();
        }
        let state = self.state.clone();
        let scope = state.scope.clone();
        let (future, cancellation) = self.executor.execute_direct_tracked(scoped(a::Scope_TO::from_value(SharedScope(scope),TD_Opaque),async move{
            let mut value=state.value.lock().await;
            if value.terminal{return a::Result::ROk(ROption::RNone)}
            let Some(stream)=value.stream.as_mut()else{return a::Result::ROk(ROption::RNone)};
            tokio::select!{biased;
                ()=state.cancelled.cancelled()=>a::Result::RErr(a::Error::new("closed","native stream is closing")),
                next=stream.next()=>match next{
                    Some(Ok(v))=>a::Result::ROk(ROption::RSome(into_abi(v))),
                    Some(Err(e))=>{value.terminal=true;a::Result::RErr(failure(e))},
                    None=>{value.terminal=true;a::Result::ROk(ROption::RNone)},
                }
            }
        }));
        pulls.retain(DirectCancellation::is_alive);
        pulls.push(cancellation);
        future
    }
    fn close(&mut self) -> FfiFuture<a::Result<()>> {
        let future = self.state.close(&self.executor);
        future.into_ffi()
    }
}
impl Drop for OutputStream {
    fn drop(&mut self) {
        drop(self.state.close(&self.executor));
    }
}
struct Owner {
    executor: ComponentExecutor,
    streams: Arc<Mutex<Vec<Arc<StreamState>>>>,
}
impl a::Component for Owner {
    /// Stop user admission, drain work and stream cleanup, then shut down the component executor.
    fn shutdown(&self) -> FfiFuture<a::Result<()>> {
        self.executor.stop_admission();
        let executor = self.executor.clone();
        let registry = self.streams.clone();
        let retry = self.executor.clone();
        let driver = self.executor.execute_cleanup(async move {
            executor.drain_user_tasks().await;
            let streams = registry.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let closing = streams
                .into_iter()
                .map(|state| state.close(&executor))
                .collect::<Vec<_>>();
            for future in closing {
                if let Err(error) = future.await.into_result() {
                    return a::Result::RErr(error);
                }
            }
            registry
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|v| !v.closed.load(Ordering::Acquire));
            a::Result::ROk(a::Component::shutdown(&executor))
        });
        async move {
            match driver.await.into_result() {
                Ok(future) => future.await,
                Err(_) if retry.is_shutting_down() => a::Component::shutdown(&retry).await,
                Err(error) => a::Result::RErr(error),
            }
        }
        .into_ffi()
    }
}

impl PluginContext<'_> {
    fn push(
        &mut self,
        name: &str,
        surface: a::Surface,
        priority: i32,
        break_chain: bool,
        options: Json,
        callback: Callback,
    ) -> Result<()> {
        local(self.runtime.raw.register_callback(a::Registration {
            name: name.into(),
            surface,
            priority,
            break_chain,
            options: into_abi(options),
            callback: a::Callback_TO::from_value(callback, TD_Opaque),
        }))
    }

    fn unary(
        &mut self,
        name: &str,
        surface: a::Surface,
        priority: i32,
        break_chain: bool,
        callback: Arc<Unary>,
    ) -> Result<()> {
        self.push(
            name,
            surface,
            priority,
            break_chain,
            Json::Null,
            Callback {
                executor: self.executor.clone(),
                streams: self.streams.clone(),
                unary: Some(callback),
                sync: None,
                stream: None,
            },
        )
    }
    pub fn runtime(&self) -> PluginRuntime {
        self.runtime.clone()
    }
    /// Obtain a cloneable task spawner for use inside registered middleware.
    pub fn task_spawner(&self) -> PluginTaskSpawner {
        PluginTaskSpawner {
            runtime: self.runtime.clone(),
            executor: self.executor.clone(),
        }
    }
    /// Spawn background work whose scope, cancellation, and cleanup belong to this component.
    pub fn spawn<F, T>(&self, future: F) -> Result<tokio::task::JoinHandle<Result<T>>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.task_spawner().spawn(future)
    }

    pub fn register_subscriber<F: Fn(&Event) + Send + Sync + 'static>(
        &mut self,
        name: &str,
        callback: F,
    ) -> Result<()> {
        self.push(
            name,
            a::Surface::Subscriber,
            0,
            false,
            Json::Null,
            Callback {
                executor: self.executor.clone(),
                streams: self.streams.clone(),
                unary: None,
                stream: None,
                sync: Some(Arc::new(move |i| {
                    callback(&decode(i.value)?);
                    Ok(Json::Null)
                })),
            },
        )
    }
    pub fn register_conditional_middleware_guardrail<F>(
        &mut self,
        name: &str,
        kinds: &BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String> + Send + Sync + 'static,
    {
        self.push(
            name,
            a::Surface::ConditionalMiddleware,
            0,
            false,
            serde_json::json!({"kinds":kinds,"registration_name":registration_name}),
            Callback {
                executor: self.executor.clone(),
                streams: self.streams.clone(),
                unary: None,
                stream: None,
                sync: Some(Arc::new(move |i| {
                    let kinds = decode(i.value)?;
                    serde_json::to_value(callback(&kinds, i.name.as_str()))
                        .map_err(|e| e.to_string())
                })),
            },
        )
    }
}

impl PluginContext<'_> {
    pub fn register_event_metadata_injector<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BTreeMap<String, Json>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::EventMetadataInjector,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(Arc::new(decode(i.value)?)).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_mark_sanitize_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::MarkSanitize,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = {
                        let v = from_abi(i.value)?;
                        callback(
                            Arc::new(
                                serde_json::from_value(v["event"].clone())
                                    .map_err(|e| e.to_string())?,
                            ),
                            serde_json::from_value(v["fields"].clone())
                                .map_err(|e| e.to_string())?,
                        )
                        .await
                    }?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_scope_sanitize_start_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ScopeSanitizeStart,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = {
                        let v = from_abi(i.value)?;
                        callback(
                            Arc::new(
                                serde_json::from_value(v["event"].clone())
                                    .map_err(|e| e.to_string())?,
                            ),
                            serde_json::from_value(v["fields"].clone())
                                .map_err(|e| e.to_string())?,
                        )
                        .await
                    }?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_scope_sanitize_end_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Arc<Event>, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ScopeSanitizeEnd,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = {
                        let v = from_abi(i.value)?;
                        callback(
                            Arc::new(
                                serde_json::from_value(v["event"].clone())
                                    .map_err(|e| e.to_string())?,
                            ),
                            serde_json::from_value(v["fields"].clone())
                                .map_err(|e| e.to_string())?,
                        )
                        .await
                    }?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_tool_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ToolSanitizeRequest,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(i.name.to_string(), from_abi(i.value)?).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_tool_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ToolSanitizeResponse,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(i.name.to_string(), from_abi(i.value)?).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_tool_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ToolConditionalExecution,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(i.name.to_string(), from_abi(i.value)?).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_tool_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ToolRequestIntercept,
            priority,
            break_chain,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(i.name.to_string(), from_abi(i.value)?).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_tool_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(ToolExecutionContext, ToolNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolExecutionInterceptOutcome>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::ToolExecutionIntercept,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(
                        ToolExecutionContext {
                            tool_name: i.name.to_string(),
                            args: from_abi(i.value)?,
                            tool_call_id: i.tool_call_id.into_option().map(|v| v.to_string()),
                        },
                        ToolNext(Arc::new(
                            _next.into_option().ok_or("missing tool continuation")?,
                        )),
                    )
                    .await?;
                    Ok(UserResult::Tool(value))
                })
            }),
        )
    }
    pub fn register_llm_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(LlmRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::LlmConditionalExecution,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(decode(i.value)?).await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_llm_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, Option<AnnotatedLlmRequest>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmRequestInterceptOutcome>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::LlmRequestIntercept,
            priority,
            break_chain,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = {
                        let v = from_abi(i.value)?;
                        callback(
                            i.name.to_string(),
                            serde_json::from_value(v["request"].clone())
                                .map_err(|e| e.to_string())?,
                            serde_json::from_value(v["annotated"].clone())
                                .map_err(|e| e.to_string())?,
                        )
                        .await
                    }?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_llm_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(LlmRequest, LlmSanitizeRequestContext<'static>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<LlmRequest>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::LlmSanitizeRequest,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(
                        decode(i.value)?,
                        LlmSanitizeRequestContext::new(i.context.request_codec)?,
                    )
                    .await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_llm_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Json, LlmSanitizeResponseContext<'static>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<Json>>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::LlmSanitizeResponse,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(
                        from_abi(i.value)?,
                        LlmSanitizeResponseContext::new(i.context.response_codec)?,
                    )
                    .await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_llm_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmExecutionContext, LlmNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        self.unary(
            name,
            a::Surface::LlmExecutionIntercept,
            priority,
            false,
            Arc::new(move |i, _next| {
                let callback = callback.clone();
                Box::pin(async move {
                    let value = callback(
                        i.name.to_string(),
                        decode(i.value)?,
                        LlmExecutionContext::new(
                            i.context.request_codec,
                            i.context.response_codec,
                        )?,
                        LlmNext(Arc::new(
                            _next.into_option().ok_or("missing LLM continuation")?,
                        )),
                    )
                    .await?;
                    serde_json::to_value(value)
                        .map(UserResult::Value)
                        .map_err(|e| e.to_string())
                })
            }),
        )
    }
    pub fn register_llm_stream_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(String, LlmRequest, LlmExecutionContext, LlmStreamNext) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<LlmJsonAsyncStream>> + Send + 'static,
    {
        let callback = Arc::new(callback);
        let executor = self.executor.clone();
        self.push(
            name,
            a::Surface::LlmStreamExecutionIntercept,
            priority,
            false,
            Json::Null,
            Callback {
                executor: self.executor.clone(),
                streams: self.streams.clone(),
                unary: None,
                sync: None,
                stream: Some(Arc::new(move |i, next| {
                    let callback = callback.clone();
                    let executor = executor.clone();
                    Box::pin(async move {
                        let sources = Arc::new(Mutex::new(Vec::new()));
                        let stream = callback(
                            i.name.to_string(),
                            decode(i.value)?,
                            LlmExecutionContext::new(
                                i.context.request_codec,
                                i.context.response_codec,
                            )?,
                            LlmStreamNext {
                                next: Arc::new(next),
                                sources: sources.clone(),
                                executor,
                                scope: Arc::new(i.context.scope),
                            },
                        )
                        .await?;
                        Ok((stream, sources))
                    })
                })),
            },
        )
    }
}

#[derive(Clone)]
pub struct ToolNext(Arc<a::ContinuationBox>);
impl ToolNext {
    pub async fn call(&self, args: Json) -> Result<ToolExecutionResult> {
        let value = local(self.0.call_tool(into_abi(args)).await)?;
        Ok(ToolExecutionResult {
            result: from_abi(value.result)?,
            annotation: value
                .annotation
                .into_option()
                .map(from_abi)
                .transpose()?
                .filter(|v| !v.is_null()),
        })
    }
}

#[derive(Clone)]
pub struct LlmNext(Arc<a::ContinuationBox>);
impl LlmNext {
    pub async fn call(&self, request: LlmRequest) -> Result<Json> {
        from_abi(local(self.0.call(encode(request)?).await)?)
    }
}
#[derive(Clone)]
pub struct LlmStreamNext {
    next: Arc<a::ContinuationBox>,
    sources: Arc<Mutex<Vec<Arc<SourceStream>>>>,
    executor: ComponentExecutor,
    scope: Arc<a::ScopeBox>,
}
impl LlmStreamNext {
    pub async fn call(&self, request: LlmRequest) -> Result<LlmJsonAsyncStream> {
        let stream = local(self.next.open_stream(encode(request)?).await)?;
        let source = SourceStream::new(stream, &self.executor, self.scope.clone(), &self.sources);
        Ok(Box::pin(InputStream {
            source,
            pending: None,
            terminal: false,
        }))
    }
}
type SourceClose = Shared<BoxFuture<'static, Result<()>>>;
struct SourceState {
    stream: Option<a::StreamBox>,
    closing: Option<SourceClose>,
}
struct SourceStream {
    state: Mutex<SourceState>,
    executor: ComponentExecutor,
    scope: Arc<a::ScopeBox>,
    sources: std::sync::Weak<Mutex<Vec<Arc<SourceStream>>>>,
    closed: AtomicBool,
}
impl SourceStream {
    fn new(
        stream: a::StreamBox,
        executor: &ComponentExecutor,
        scope: Arc<a::ScopeBox>,
        sources: &Arc<Mutex<Vec<Arc<SourceStream>>>>,
    ) -> Arc<Self> {
        let source = Arc::new(Self {
            state: Mutex::new(SourceState {
                stream: Some(stream),
                closing: None,
            }),
            executor: executor.clone(),
            scope,
            sources: Arc::downgrade(sources),
            closed: AtomicBool::new(false),
        });
        sources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(source.clone());
        source
    }
    /// Share source-stream cleanup and release the foreign source only after close succeeds.
    fn close(self: &Arc<Self>) -> SourceClose {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(future) = &state.closing
            && future.peek().is_none_or(|v| v.is_ok())
        {
            return future.clone();
        }
        let Some(stream) = state.stream.as_mut() else {
            return async { Ok(()) }.boxed().shared();
        };
        let foreign = stream.close();
        let source = self.clone();
        let scope = a::Scope_TO::from_value(SharedScope(self.scope.clone()), TD_Opaque);
        let mut cleanup = scoped(scope, async move {
            if let Err(error) = foreign.await.into_result() {
                return a::Result::RErr(error);
            }
            source
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .stream
                .take();
            source.closed.store(true, Ordering::Release);
            if let Some(sources) = source.sources.upgrade() {
                sources
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .retain(|v| !v.closed.load(Ordering::Acquire));
            }
            a::Result::ROk(())
        });
        cleanup.cleanup = true;
        // Drive cleanup even when an exhausted/dropped stream has no waiter.
        // Failed closes keep their source in the registry for shutdown to retry.
        let foreign = self.executor.execute_cleanup(cleanup);
        let future = async move { local(foreign.await) }.boxed().shared();
        state.closing = Some(future.clone());
        future
    }
}
struct InputStream {
    source: Arc<SourceStream>,
    pending: Option<FfiFuture<a::Result<ROption<abi::Json>>>>,
    terminal: bool,
}
impl Unpin for InputStream {}
impl Stream for InputStream {
    type Item = Result<Json>;
    /// Poll one foreign source item and stop reading after the source closes or reaches a terminal result.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        if self.pending.is_none() {
            let future = self
                .source
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .stream
                .as_mut()
                .map(|stream| stream.pull());
            let Some(future) = future else {
                self.terminal = true;
                return Poll::Ready(None);
            };
            self.pending = Some(future);
        }
        let result = match Pin::new(self.pending.as_mut().unwrap()).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(v) => v,
        };
        self.pending.take();
        match local(result) {
            Ok(ROption::RSome(v)) => Poll::Ready(Some(from_abi(v))),
            Ok(ROption::RNone) => {
                self.terminal = true;
                drop(self.source.close());
                Poll::Ready(None)
            }
            Err(e) => {
                self.terminal = true;
                drop(self.source.close());
                Poll::Ready(Some(Err(e)))
            }
        }
    }
}
impl Drop for InputStream {
    fn drop(&mut self) {
        self.pending.take();
        drop(self.source.close());
    }
}

mod codec;
pub use codec::*;
mod runtime;
pub use runtime::*;

/// Adapt an author plugin only after bootstrap/module verification.
pub struct Adapter<P>(pub Arc<P>);
impl<P: Plugin> a::Plugin for Adapter<P> {
    fn allows_multiple_components(&self) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.0.allows_multiple_components()
        }))
        .unwrap_or(false)
    }
    fn validate(&self, config: abi::Json) -> RVec<a::ConfigDiagnostic> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let config = decode(config)?;
            Ok::<_, String>(self.0.validate(&config))
        }))
        .unwrap_or_else(|_| Err("native validation panicked".into()));
        let diagnostics = result.unwrap_or_else(|message| {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.native_validate_failed".into(),
                component: None,
                field: None,
                message,
            }]
        });
        diagnostics
            .into_iter()
            .map(|d| a::ConfigDiagnostic {
                is_error: d.level == DiagnosticLevel::Error,
                code: d.code.into(),
                component: d.component.map(Into::into).into(),
                field: d.field.map(Into::into).into(),
                message: d.message.into(),
            })
            .collect()
    }
    /// Bind component ownership and run registration off the caller, with optional Tokio support.
    fn register(
        &self,
        config: abi::Json,
        runtime: a::RuntimeBox,
    ) -> FfiFuture<a::Result<a::RegisteredComponent>> {
        let setup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let config: Map<String, Json> = decode(config)?;
            let executor = match self.0.executor_config_for_component(&config)? {
                Some(config) => ComponentExecutor::new(config, self.0.plugin_kind())?,
                None => ComponentExecutor::inline(self.0.plugin_kind()),
            };
            let scope = local(runtime.capture_scope())?;
            Ok::<_, String>((config, executor, scope))
        }))
        .unwrap_or_else(|_| Err("native registration setup panicked".into()));
        let (config, executor, scope) = match setup {
            Ok(v) => v,
            Err(e) => return async move { a::Result::RErr(failure(e)) }.into_ffi(),
        };
        let plugin = self.0.clone();
        let owner = executor.clone();
        let streams = Arc::new(Mutex::new(Vec::new()));
        if let Err(error) = runtime
            .bind_component(a::Component_TO::from_value(
                Owner {
                    executor: executor.clone(),
                    streams: streams.clone(),
                },
                TD_Opaque,
            ))
            .into_result()
        {
            return async move { a::Result::RErr(error) }.into_ffi();
        }
        let cleanup = executor.clone();
        let execution = executor.execute(scoped(scope, async move {
            let diagnostic_transport = Arc::new(runtime.diagnostics());
            let mut ctx = PluginContext {
                registrations: vec![],
                runtime: PluginRuntime {
                    raw: Arc::new(runtime),
                    diagnostic_transport,
                    diagnostic_state: Arc::new(crate::diagnostics::SdkState::default()),
                    executor: Some(owner.clone()),
                },
                executor: owner.clone(),
                streams: streams.clone(),
                _borrow: PhantomData,
            };
            let result =
                std::panic::AssertUnwindSafe(async { plugin.register(&config, &mut ctx).await })
                    .catch_unwind()
                    .await;
            let error = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(failure(error)),
                Err(_) => Some(failure("native registration callback panicked")),
            };
            a::Result::ROk(a::RegisteredComponent {
                error: error.into(),
                registrations: ctx.registrations.into(),
                component: a::Component_TO::from_value(
                    Owner {
                        executor: owner,
                        streams,
                    },
                    TD_Opaque,
                ),
            })
        }));
        async move {
            let result = execution.await;
            if result.is_err() {
                let closed = a::Component::shutdown(&cleanup).await;
                if closed.is_err() {
                    return closed.map(|_| unreachable!());
                }
            }
            result
        }
        .into_ffi()
    }
}

/// Construct a checked plugin, reject an identity mismatch, and contain constructor panics in the SDK.
pub fn create_plugin<P: Plugin>(
    constructor: impl FnOnce() -> P,
    kind: &str,
) -> a::Result<a::PluginBox> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let plugin = constructor();
        let _ = plugin.allows_multiple_components();
        if plugin.plugin_kind() != kind {
            return a::Result::RErr(failure(
                "exported plugin kind disagrees with implementation",
            ));
        }
        a::Result::ROk(a::Plugin_TO::from_value(
            Adapter(Arc::new(plugin)),
            TD_Opaque,
        ))
    }))
    .unwrap_or_else(|_| a::Result::RErr(failure("native plugin constructor panicked")))
}

/// Export the checked standard header with SDK-derived contract version.
#[macro_export]
macro_rules! nemo_relay_checked_plugin {
    (kind = $kind:literal, $constructor:expr) => {
        mod __nemo_relay_checked_export {
            use super::*;
            use $crate::abi::{
                self,
                abi_stable::{
                    library::{CheckTypeLayout, LibHeader, RootModuleResult},
                    prefix_type::PrefixTypeTrait,
                    sabi_types::RRef,
                    std_types::{ROption, RStr},
                },
                v1_0,
            };
            extern "C" fn descriptor() -> abi::PluginDescriptor {
                abi::PluginDescriptor {
                    native_api: abi::NATIVE_API_VERSION,
                    plugin_kind: $kind.into(),
                }
            }
            extern "C" fn create() -> v1_0::Result<v1_0::PluginBox> {
                $crate::checked::create_plugin($constructor, $kind)
            }
            extern "C" fn query(name: RStr<'_>) -> ROption<RRef<'static, LibHeader>> {
                if name.as_str() != v1_0::FACTORY_MODULE {
                    return ROption::RNone;
                }
                static HEADER: std::sync::OnceLock<LibHeader> = std::sync::OnceLock::new();
                ROption::RSome(RRef::new(HEADER.get_or_init(|| {
                    LibHeader::from_module(v1_0::Factory { create }.leak_into_prefix())
                })))
            }
            fn bootstrap() -> abi::Bootstrap_Ref {
                abi::Bootstrap {
                    descriptor,
                    query_module: query,
                }
                .leak_into_prefix()
            }
            extern "C" fn erased() -> RootModuleResult {
                abi::abi_stable::library::__call_root_module_loader(bootstrap)
            }
            // This is abi_stable's standard exported header symbol. The dependency ABI
            // generation is pinned by the interface crate. Layout checking is mandatory.
            #[unsafe(no_mangle)]
            pub(super) static _1as_0lib_1header_0root_bmodule_bloader: LibHeader = unsafe {
                LibHeader::from_constructor::<abi::Bootstrap_Ref>(erased, CheckTypeLayout::Yes)
            };
        }
    };
}

impl PluginContext<'_> {
    /// Register a synchronous callback using only checked ABI invocation values.
    /// The Rust callback and its captured state remain in the originating SDK.
    /// Subscribers run inline; synchronous middleware uses blocking offload.
    pub fn register_synchronous_callback<F>(
        &mut self,
        name: &str,
        surface: a::Surface,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(a::Invocation, ROption<a::ContinuationBox>) -> Result<abi::Json>
            + Send
            + Sync
            + 'static,
    {
        if surface == a::Surface::LlmStreamExecutionIntercept {
            return Err("use register_synchronous_stream_callback for streaming callbacks".into());
        }
        let callback = Arc::new(callback);
        local(self.runtime.raw.register_callback(a::Registration {
            name: name.into(),
            surface,
            priority,
            break_chain,
            options: abi::Json::Null,
            callback: a::Callback_TO::from_value(
                RawCallback {
                    callback,
                    executor: self.executor.clone(),
                },
                TD_Opaque,
            ),
        }))
    }
}
impl PluginContext<'_> {
    /// Register a synchronous stream factory returning a checked stream descriptor.
    /// Offload construction; poll returned stream futures under component scope and runtime context.
    pub fn register_synchronous_stream_callback<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(a::Invocation, a::ContinuationBox) -> Result<a::StreamBox> + Send + Sync + 'static,
    {
        let callback = Arc::new(callback);
        let executor = self.executor.clone();
        self.push(
            name,
            a::Surface::LlmStreamExecutionIntercept,
            priority,
            false,
            Json::Null,
            Callback {
                executor: self.executor.clone(),
                streams: self.streams.clone(),
                unary: None,
                sync: None,
                stream: Some(Arc::new(move |invocation, next| {
                    let callback = callback.clone();
                    let executor = executor.clone();
                    Box::pin(async move {
                        let scope = Arc::new(invocation.context.scope);
                        let invocation = a::Invocation {
                            context: a::Context {
                                scope: a::Scope_TO::from_value(
                                    SharedScope(scope.clone()),
                                    TD_Opaque,
                                ),
                                ..invocation.context
                            },
                            ..invocation
                        };
                        let bound = scope.clone();
                        let stream = local(
                            executor
                                .execute_blocking(move || {
                                    boundary(
                                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                            || {
                                                let mut lease = local(bound.enter())?;
                                                let result = callback(invocation, next);
                                                local(lease.restore())?;
                                                result
                                            },
                                        ))
                                        .unwrap_or_else(
                                            |_| Err("synchronous stream callback panicked".into()),
                                        ),
                                    )
                                })
                                .await,
                        )?;
                        let sources = Arc::new(Mutex::new(Vec::new()));
                        let source = SourceStream::new(stream, &executor, scope, &sources);
                        Ok((
                            Box::pin(InputStream {
                                source,
                                pending: None,
                                terminal: false,
                            }) as LlmJsonAsyncStream,
                            sources,
                        ))
                    })
                })),
            },
        )
    }
}

type RawSync =
    dyn Fn(a::Invocation, ROption<a::ContinuationBox>) -> Result<abi::Json> + Send + Sync;
struct RawCallback {
    callback: Arc<RawSync>,
    executor: ComponentExecutor,
}
fn raw_sync(
    executor: &ComponentExecutor,
    callback: &RawSync,
    invocation: a::Invocation,
    next: ROption<a::ContinuationBox>,
) -> a::Result<abi::Json> {
    boundary(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _call = local(executor.enter_sync())?;
            executor.with_context(|| {
                let mut lease = local(invocation.context.scope.enter())?;
                let result = callback(invocation, next);
                local(lease.restore())?;
                result
            })?
        }))
        .unwrap_or_else(|_| Err("raw synchronous callback panicked".into())),
    )
}
impl a::Callback for RawCallback {
    /// Offload a synchronous tool callback and decode its canonical result and pending marks.
    fn invoke_tool(
        &self,
        invocation: a::Invocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        let callback = self.callback.clone();
        let executor = self.executor.clone();
        let work = self.executor.execute_blocking(move || {
            raw_sync(&executor, &*callback, invocation, ROption::RSome(next))
        });
        async move {
            let value = local(work.await);
            boundary(value.and_then(|v| {
                let value: ToolExecutionInterceptOutcome = decode(v)?;
                Ok(a::ToolExecutionOutcome {
                    result: a::ToolExecutionResult {
                        result: into_abi(value.result),
                        annotation: value
                            .annotation
                            .filter(|v| !v.is_null())
                            .map(into_abi)
                            .into(),
                    },
                    pending_marks: value
                        .pending_marks
                        .into_iter()
                        .map(encode)
                        .collect::<Result<RVec<_>>>()?,
                })
            }))
        }
        .into_ffi()
    }
    fn invoke_sync(&self, invocation: a::Invocation) -> a::Result<abi::Json> {
        raw_sync(&self.executor, &*self.callback, invocation, ROption::RNone)
    }
    /// Offload a synchronous callback with its supplied optional continuation.
    fn invoke(
        &self,
        invocation: a::Invocation,
        next: ROption<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        let callback = self.callback.clone();
        let executor = self.executor.clone();
        self.executor
            .execute_blocking(move || raw_sync(&executor, &*callback, invocation, next))
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        async {
            a::Result::RErr(failure(
                "synchronous streaming callbacks must return a checked Stream descriptor",
            ))
        }
        .into_ffi()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/checked_lifecycle.rs"]
mod lifecycle_tests;
