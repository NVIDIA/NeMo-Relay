// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component-owned work on shared runtimes for the checked native ABI.
//!
//! Middleware futures are polled directly with optional library-local Tokio context.
//! Foreign future destruction releases unpolled work. Poll panics are
//! converted here, in the originating library; panicking destructors and foreign
//! wakers retain async-ffi's abort limitations.

use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};

use futures::{FutureExt as _, task::AtomicWaker};
use nemo_relay_plugin_abi::{
    async_ffi::{FfiFuture, FutureExt as _},
    v1_0::{Component, Error, Result as AbiResult},
};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::{
    sync::CancellationToken,
    task::{TaskTracker, task_tracker::TaskTrackerToken},
};

use crate::NativeExecutorConfig;

#[derive(Clone)]
enum Shutdown {
    Running,
    Closing,
    Closed,
    Failed(String),
}

struct SharedRuntime {
    runtime: Runtime,
    worker_threads: usize,
}

#[derive(Clone)]
struct RuntimeLease {
    handle: Handle,
    // The host can borrow its current runtime without taking ownership of it.
    _owner: Option<Arc<SharedRuntime>>,
}

/// Acquire a weakly cached runtime, requiring active users to agree on its worker count.
fn shared_runtime(config: NativeExecutorConfig, kind: Option<&str>) -> crate::Result<RuntimeLease> {
    static RUNTIMES: OnceLock<Mutex<HashMap<Option<String>, Weak<SharedRuntime>>>> =
        OnceLock::new();
    let mut runtimes = RUNTIMES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    runtimes.retain(|_, runtime| runtime.strong_count() > 0);
    let key = kind.map(str::to_owned);
    let name = kind.unwrap_or("native-host");
    let owner = if let Some(owner) = runtimes.get(&key).and_then(Weak::upgrade) {
        if owner.worker_threads != config.worker_threads {
            return Err(format!(
                "Tokio-enabled components of native plugin '{name}' must use the same executor.worker_threads ({})",
                owner.worker_threads
            ));
        }
        owner
    } else {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.worker_threads)
            .thread_name(format!("nemo-relay-plugin-{name}"))
            .enable_all()
            .build()
            .map_err(|e| format!("failed to start native plugin runtime: {e}"))?;
        let owner = Arc::new(SharedRuntime {
            runtime,
            worker_threads: config.worker_threads,
        });
        runtimes.insert(key, Arc::downgrade(&owner));
        owner
    };
    Ok(RuntimeLease {
        handle: owner.runtime.handle().clone(),
        _owner: Some(owner),
    })
}

trait DirectTask: Send + Sync {
    fn cancel(&self, component: bool);
}

/// Weak cancellation authority for one directly polled operation.
#[derive(Clone, Default)]
pub struct DirectCancellation(Option<Weak<dyn DirectTask>>);
impl DirectCancellation {
    /// Release operation state even if its foreign waiter is retained and no longer polled.
    pub fn cancel(&self) {
        if let Some(task) = self.0.as_ref().and_then(Weak::upgrade) {
            task.cancel(false);
        }
    }
    /// Whether a foreign waiter still retains this operation's state.
    pub fn is_alive(&self) -> bool {
        self.0.as_ref().is_some_and(|task| task.strong_count() > 0)
    }
}

struct DirectSlot<T> {
    future: Option<Pin<Box<dyn Future<Output = AbiResult<T>> + Send>>>,
    runtime: Option<RuntimeLease>,
    admission: Option<SynchronousCall>,
}
impl<T> DirectSlot<T> {
    /// Destroy user state under its runtime context before releasing shutdown admission.
    fn finish(&mut self) {
        let handle = self.runtime.as_ref().map(|runtime| runtime.handle.clone());
        let _entered = handle.as_ref().map(Handle::enter);
        self.future.take();
        // Release the runtime reference before allowing shutdown to finish its drain.
        self.runtime.take();
        self.admission.take();
    }
}
struct DirectState<T> {
    slot: Mutex<DirectSlot<T>>,
    cancelled: AtomicU8,
    waker: AtomicWaker,
}
const POLLING: u8 = 0b100;
const CANCELLATION: u8 = 0b011;

impl<T: Send> DirectTask for DirectState<T> {
    /// Revoke the future without waiting on a poll that might be revoking itself.
    fn cancel(&self, component: bool) {
        let first = self
            .cancelled
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & CANCELLATION == 0).then_some(state | if component { 1 } else { 2 })
            });
        // An active poll owns cleanup until it atomically clears POLLING after
        // unlocking the slot. Self-cancellation therefore never waits on itself.
        if let Ok(previous) = first
            && previous & POLLING == 0
        {
            self.slot.lock().unwrap_or_else(|e| e.into_inner()).finish();
        }
        self.waker.wake();
    }
}
impl<T> DirectState<T> {
    /// Translate the first cancellation reason into the checked ABI's cancellation or revocation error.
    fn error(&self) -> Error {
        if self.cancelled.load(Ordering::Acquire) & CANCELLATION == 2 {
            Error::new("cancelled", "native operation was cancelled")
        } else {
            Error::new("revoked", "component is shut down")
        }
    }
}
struct DirectFuture<T: Send>(Arc<DirectState<T>>);
impl<T: Send> Future for DirectFuture<T> {
    type Output = AbiResult<T>;
    /// Poll on the caller, bind optional Tokio context, and contain originating-library panics.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.waker.register(cx.waker());
        let polled = {
            let mut slot = self.0.slot.lock().unwrap_or_else(|e| e.into_inner());
            let previous = self.0.cancelled.fetch_or(POLLING, Ordering::AcqRel);
            if previous & CANCELLATION != 0 || slot.future.is_none() {
                slot.finish();
                Poll::Ready(AbiResult::RErr(self.0.error()))
            } else {
                let handle = slot.runtime.as_ref().map(|runtime| runtime.handle.clone());
                let _entered = handle.as_ref().map(Handle::enter);
                let polled = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    slot.future.as_mut().unwrap().as_mut().poll(cx)
                }))
                .unwrap_or_else(|_| {
                    Poll::Ready(AbiResult::RErr(Error::new(
                        "panic",
                        "native plugin future panicked",
                    )))
                });
                if polled.is_ready() {
                    slot.finish();
                }
                polled
            }
        };
        // This RMW and the first cancellation select the cleanup owner. If
        // cancellation wins, this poll finishes; otherwise the canceller locks
        // the now-unlocked slot itself. A retained Pending waiter is irrelevant.
        if self.0.cancelled.fetch_and(!POLLING, Ordering::AcqRel) & CANCELLATION != 0 {
            self.0
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .finish();
            return Poll::Ready(AbiResult::RErr(self.0.error()));
        }
        polled
    }
}
impl<T: Send> Drop for DirectFuture<T> {
    fn drop(&mut self) {
        self.0.cancel(true);
    }
}

struct State {
    accepting: bool,
    teardown_started: bool,
    runtime: Option<RuntimeLease>,
    host: bool,
    direct: Vec<Weak<dyn DirectTask>>,
    background: Vec<AbortHandle>,
}

struct Inner {
    state: Mutex<State>,
    tasks: TaskTracker,
    user_tasks: TaskTracker,
    cancelled: CancellationToken,
    shutdown: watch::Sender<Shutdown>,
    thread_name: String,
}

/// Component task owner used by checked ABI adapters on a shared or borrowed runtime.
///
/// Clones share one component; construct a new executor for every registration.
/// Explicit shutdown is retryable if the teardown thread cannot be started.
#[derive(Clone)]
pub struct ComponentExecutor {
    inner: Arc<Inner>,
}

/// Admission guard for inline work owned by this library.
/// Dropping the guard releases both user-work and executor cleanup barriers.
pub struct SynchronousCall {
    _user: TaskTrackerToken,
    _task: TaskTrackerToken,
}

impl ComponentExecutor {
    /// Acquire the SDK library runtime and create independent component task ownership.
    pub fn new(config: NativeExecutorConfig, plugin_kind: &str) -> crate::Result<Self> {
        if config.worker_threads == 0 {
            return Err("executor.worker_threads must be greater than zero".into());
        }
        let runtime = shared_runtime(config, Some(plugin_kind))?;
        Ok(Self::with_runtime(Some(runtime), false, plugin_kind))
    }

    /// Create component ownership without a plugin Tokio runtime or worker threads.
    pub fn inline(plugin_kind: &str) -> Self {
        Self::with_runtime(None, false, plugin_kind)
    }

    /// Track host work without creating a component runtime. Detached cleanup uses a shared fallback.
    pub fn for_host() -> Self {
        Self::with_runtime(None, true, "native-host")
    }

    fn with_runtime(runtime: Option<RuntimeLease>, host: bool, kind: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    accepting: true,
                    teardown_started: false,
                    runtime,
                    host,
                    direct: Vec::new(),
                    background: Vec::new(),
                }),
                tasks: TaskTracker::new(),
                user_tasks: TaskTracker::new(),
                cancelled: CancellationToken::new(),
                shutdown: watch::channel(Shutdown::Running).0,
                thread_name: format!("nemo-relay-plugin-{kind}"),
            }),
        }
    }

    /// Borrow current host context or acquire detached host cleanup support; inline SDK work needs neither.
    fn runtime(state: &mut State, cleanup: bool) -> crate::Result<Option<RuntimeLease>> {
        if state.host
            && !cleanup
            && let Ok(handle) = Handle::try_current()
        {
            return Ok(Some(RuntimeLease {
                handle,
                _owner: None,
            }));
        }
        if state.host && state.runtime.is_none() {
            state.runtime = Some(shared_runtime(NativeExecutorConfig::default(), None)?);
        }
        Ok(state.runtime.clone())
    }

    /// Poll middleware on its caller, entering the optional SDK runtime for every poll and drop.
    /// Component revocation also destroys retained or unpolled futures before shutdown completes.
    pub fn execute_direct<T: Send + 'static>(
        &self,
        future: impl Future<Output = AbiResult<T>> + Send + 'static,
    ) -> FfiFuture<AbiResult<T>> {
        self.execute_direct_tracked(future).0
    }

    /// Return a direct future with independent cancellation authority for stream close.
    pub fn execute_direct_tracked<T: Send + 'static>(
        &self,
        future: impl Future<Output = AbiResult<T>> + Send + 'static,
    ) -> (FfiFuture<AbiResult<T>>, DirectCancellation) {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let runtime = if state.accepting {
            Self::runtime(&mut state, false)
        } else {
            Err("component is shut down".into())
        };
        let runtime = match runtime {
            Ok(runtime) => runtime,
            Err(message) => {
                return (
                    async move { AbiResult::RErr(Error::new("revoked", message)) }.into_ffi(),
                    DirectCancellation::default(),
                );
            }
        };
        let task = Arc::new(DirectState {
            slot: Mutex::new(DirectSlot {
                future: Some(Box::pin(future)),
                runtime,
                admission: Some(SynchronousCall {
                    _user: self.inner.user_tasks.token(),
                    _task: self.inner.tasks.token(),
                }),
            }),
            cancelled: AtomicU8::new(0),
            waker: AtomicWaker::new(),
        });
        state.direct.retain(|task| task.strong_count() > 0);
        let erased: Arc<dyn DirectTask> = task.clone();
        let cancellation = Arc::downgrade(&erased);
        state.direct.push(cancellation.clone());
        (
            DirectFuture(task).into_ffi(),
            DirectCancellation(Some(cancellation)),
        )
    }

    /// Spawn background work with component ownership; dropping its join handle detaches only the waiter.
    pub fn spawn<F>(&self, future: F) -> crate::Result<JoinHandle<F::Output>>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.accepting {
            return Err("component is shut down".into());
        }
        let runtime = Self::runtime(&mut state, false)?.ok_or_else(||
            "background tasks require a Tokio runtime; configure Plugin::executor_config or the executor object".to_owned())?;
        let task = self
            .inner
            .tasks
            .spawn_on(self.inner.user_tasks.track_future(future), &runtime.handle);
        state.background.retain(|task| !task.is_finished());
        state.background.push(task.abort_handle());
        Ok(task)
    }

    /// Offload a synchronous callback and retain admission until its captured state is released.
    /// Runtime-free components use a temporary thread; Tokio-enabled components share the runtime's blocking pool.
    pub fn execute_blocking<T: Send + 'static>(
        &self,
        callback: impl FnOnce() -> AbiResult<T> + Send + 'static,
    ) -> FfiFuture<AbiResult<T>> {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.accepting {
            return async { AbiResult::RErr(Error::new("revoked", "component is shut down")) }
                .into_ffi();
        }
        let runtime = match Self::runtime(&mut state, false) {
            Ok(runtime) => runtime,
            Err(message) => {
                return async move { AbiResult::RErr(Error::new("runtime", message)) }.into_ffi();
            }
        };
        let admission = SynchronousCall {
            _user: self.inner.user_tasks.token(),
            _task: self.inner.tasks.token(),
        };
        let (send, receive) = oneshot::channel();
        let work = move || {
            let _admission = admission;
            let value = std::panic::catch_unwind(AssertUnwindSafe(callback)).unwrap_or_else(|_| {
                AbiResult::RErr(Error::new("panic", "native blocking callback panicked"))
            });
            let _ = send.send(value);
        };
        if let Some(runtime) = runtime {
            runtime.handle.spawn_blocking(work);
        } else if let Err(error) = std::thread::Builder::new()
            .name(format!("{}-blocking", self.inner.thread_name))
            .spawn(work)
        {
            return async move { AbiResult::RErr(Error::new("runtime", error.to_string())) }
                .into_ffi();
        }
        drop(state);
        async move {
            receive.await.unwrap_or_else(|_| {
                AbiResult::RErr(Error::new("cancelled", "blocking callback stopped"))
            })
        }
        .into_ffi()
    }

    /// Enter this library's runtime while constructing or invoking synchronous user work.
    pub fn with_context<T>(&self, invoke: impl FnOnce() -> T) -> crate::Result<T> {
        let runtime = {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            Self::runtime(&mut state, false)?
        };
        let handle = runtime.as_ref().map(|runtime| runtime.handle.clone());
        let _entered = handle.as_ref().map(Handle::enter);
        Ok(invoke())
    }

    /// Admit a synchronous callback on its calling thread until the guard drops.
    /// Admission is atomic with shutdown, just like asynchronous user work.
    pub fn enter_sync(&self) -> AbiResult<SynchronousCall> {
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.accepting {
            return AbiResult::RErr(Error::new("revoked", "component is shut down"));
        }
        AbiResult::ROk(SynchronousCall {
            _user: self.inner.user_tasks.token(),
            _task: self.inner.tasks.token(),
        })
    }

    /// Spawn immediately on the owning executor and return an ABI-safe receiver.
    ///
    /// Capture cancellation ownership outside the returned async block: dropping
    /// an unpolled foreign future must still cancel the already admitted task.
    pub fn execute<T: Send + 'static>(
        &self,
        future: impl Future<Output = AbiResult<T>> + Send + 'static,
    ) -> FfiFuture<AbiResult<T>> {
        let invocation = CancellationToken::new();
        let guard = invocation.clone().drop_guard();
        let component = self.inner.cancelled.clone();
        let (send, receive) = oneshot::channel();
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.accepting {
            let runtime = match Self::runtime(&mut state, false) {
                Ok(runtime) => runtime,
                Err(message) => {
                    return async move { AbiResult::RErr(Error::new("runtime", message)) }
                        .into_ffi();
                }
            };
            let future = self.inner.user_tasks.track_future(async move {
                let result = tokio::select! {
                    biased;
                    () = component.cancelled() => AbiResult::RErr(Error::new("revoked", "component is shutting down")),
                    () = invocation.cancelled() => return,
                    result = AssertUnwindSafe(future).catch_unwind() => result.unwrap_or_else(|_| AbiResult::RErr(Error::new("panic", "native plugin future panicked"))),
                };
                let _ = send.send(result);
            });
            if let Err(message) = self.dispatch(runtime, future) {
                return async move { AbiResult::RErr(Error::new("runtime", message)) }.into_ffi();
            }
        } else {
            let _ = send.send(AbiResult::RErr(Error::new(
                "revoked",
                "component is shut down",
            )));
        }
        drop(state);
        async move {
            let _guard = guard;
            receive.await.unwrap_or_else(|_| {
                AbiResult::RErr(Error::new("cancelled", "native plugin task stopped"))
            })
        }
        .into_ffi()
    }

    /// Run offloaded work on the configured runtime or on a temporary thread for runtime-free components.
    fn dispatch(
        &self,
        runtime: Option<RuntimeLease>,
        future: impl Future<Output = ()> + Send + 'static,
    ) -> crate::Result<()> {
        let future = self.inner.tasks.track_future(future);
        if let Some(runtime) = runtime {
            runtime.handle.spawn(future);
        } else {
            std::thread::Builder::new()
                .name(format!("{}-work", self.inner.thread_name))
                .spawn(move || futures::executor::block_on(future))
                .map_err(|error| format!("cannot start native work thread: {error}"))?;
        }
        Ok(())
    }

    /// Admit cleanup whose completion is required even if its foreign waiter drops.
    /// Cleanup tasks ignore component cancellation and are drained during shutdown.
    pub fn execute_cleanup<T: Send + 'static>(
        &self,
        future: impl Future<Output = AbiResult<T>> + Send + 'static,
    ) -> FfiFuture<AbiResult<T>> {
        let (send, receive) = oneshot::channel();
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.teardown_started {
            let runtime = match Self::runtime(&mut state, true) {
                Ok(runtime) => runtime,
                Err(message) => {
                    return async move { AbiResult::RErr(Error::new("runtime", message)) }
                        .into_ffi();
                }
            };
            let future = async move {
                let result = AssertUnwindSafe(future)
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|_| {
                        AbiResult::RErr(Error::new("panic", "native cleanup future panicked"))
                    });
                let _ = send.send(result);
            };
            if let Err(message) = self.dispatch(runtime, future) {
                return async move { AbiResult::RErr(Error::new("runtime", message)) }.into_ffi();
            }
        } else {
            let _ = send.send(AbiResult::RErr(Error::new(
                "revoked",
                "cleanup admission is stopped",
            )));
        }
        drop(state);
        async move {
            receive
                .await
                .unwrap_or_else(|_| AbiResult::RErr(Error::new("shutdown", "cleanup task stopped")))
        }
        .into_ffi()
    }

    /// Stop user work while retaining cleanup admission for a failed-close retry.
    pub fn stop_admission(&self) {
        let (direct, background) = {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            state.accepting = false;
            (
                std::mem::take(&mut state.direct),
                std::mem::take(&mut state.background),
            )
        };
        self.inner.user_tasks.close();
        self.inner.cancelled.cancel();
        for task in direct.into_iter().filter_map(|task| task.upgrade()) {
            task.cancel(true);
        }
        for task in background {
            task.abort();
        }
    }

    /// Stop and drain user tasks while keeping cleanup admission available.
    /// Call from outside user work, before taking the final resource snapshot.
    pub async fn drain_user_tasks(&self) {
        self.stop_admission();
        self.inner.user_tasks.wait().await;
    }

    /// Stop task admission and start runtime teardown on a dedicated thread.
    /// Retain runtime ownership if that thread cannot start.
    fn begin_shutdown(&self) -> std::result::Result<(), String> {
        self.stop_admission();
        let inner = &self.inner;
        let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.teardown_started {
            return Ok(());
        }
        state.accepting = false;
        inner.cancelled.cancel();
        inner.tasks.close();
        inner.shutdown.send_replace(Shutdown::Closing);
        // The runtime remains owned by State until the thread starts. A failed
        // spawn must not drop a Tokio runtime on the calling host/plugin worker.
        let owner = Arc::clone(inner);
        state.teardown_started = true;
        match std::thread::Builder::new()
            .name(format!("{}-shutdown", inner.thread_name))
            .spawn(move || {
                let runtime = owner
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .runtime
                    .take();
                futures::executor::block_on(owner.tasks.wait());
                drop(runtime);
                owner.shutdown.send_replace(Shutdown::Closed);
            }) {
            Ok(_) => Ok(()),
            Err(error) => {
                state.teardown_started = false;
                let message = format!("failed to start native plugin teardown thread: {error}");
                inner
                    .shutdown
                    .send_replace(Shutdown::Failed(message.clone()));
                Err(message)
            }
        }
    }

    /// Whether the dedicated teardown thread has taken responsibility for the runtime.
    pub fn is_shutting_down(&self) -> bool {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .teardown_started
    }

    /// Whether component teardown has drained every admitted task and released its runtime lease.
    pub fn is_closed(&self) -> bool {
        matches!(*self.inner.shutdown.borrow(), Shutdown::Closed)
    }

    /// Number of admitted tasks whose owned state has not yet been released.
    pub fn active_tasks(&self) -> usize {
        self.inner.tasks.len()
    }
}

impl Component for ComponentExecutor {
    /// Start teardown immediately and return a foreign future that reports when executor shutdown completes.
    fn shutdown(&self) -> FfiFuture<AbiResult<()>> {
        // Start teardown even if the caller drops the shutdown future unpolled.
        let started = self.begin_shutdown();
        let mut status = self.inner.shutdown.subscribe();
        async move {
            if let Err(message) = started {
                return AbiResult::RErr(Error::new("shutdown", message));
            }
            loop {
                let current = status.borrow_and_update().clone();
                match current {
                    Shutdown::Closed => return AbiResult::ROk(()),
                    Shutdown::Failed(message) => {
                        return AbiResult::RErr(Error::new("shutdown", message));
                    }
                    Shutdown::Running | Shutdown::Closing => {}
                }
                if status.changed().await.is_err() {
                    return AbiResult::RErr(Error::new("shutdown", "component owner disappeared"));
                }
            }
        }
        .into_ffi()
    }
}

impl Drop for Inner {
    /// Drain the last executor owner on a teardown thread so a Tokio worker never destroys its own runtime.
    fn drop(&mut self) {
        // Arc decides which caller is last atomically. Testing strong_count in
        // ComponentExecutor::drop would race when the last two clones drop at
        // once, potentially destroying a Tokio runtime on a host worker.
        self.cancelled.cancel();
        self.tasks.close();
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        for task in std::mem::take(&mut state.direct)
            .into_iter()
            .filter_map(|task| task.upgrade())
        {
            task.cancel(true);
        }
        for task in std::mem::take(&mut state.background) {
            task.abort();
        }
        if let Some(runtime) = state.runtime.take() {
            let retained = Arc::new(Mutex::new(Some(runtime)));
            let worker = retained.clone();
            let tasks = self.tasks.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("{}-shutdown", self.thread_name))
                .spawn(move || {
                    let runtime = worker
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                        .expect("implicit teardown owns its runtime");
                    futures::executor::block_on(tasks.wait());
                    drop(runtime);
                });
            if spawned.is_err() {
                // With no remaining component handle there is no caller to
                // retry. Retain failed cleanup ownership rather than destroying
                // its runtime on an arbitrary host Tokio worker.
                std::mem::forget(retained);
            }
        }
    }
}
