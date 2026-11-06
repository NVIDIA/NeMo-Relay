// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component-local execution for the checked native ABI.
//!
//! User futures are polled only by this library's Tokio executor. Foreign future
//! destruction cancels work even before the first foreign poll. Poll panics are
//! converted here, in the originating library; panicking destructors and foreign
//! wakers retain async-ffi's abort limitations.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use futures::FutureExt as _;
use nemo_relay_plugin_abi::{
    async_ffi::{FfiFuture, FutureExt as _},
    v1_0::{Component, Error, Result as AbiResult},
};
use tokio::runtime::Runtime;
use tokio::sync::{oneshot, watch};
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

struct State {
    accepting: bool,
    teardown_started: bool,
    runtime: Option<Runtime>,
}

struct Inner {
    state: Mutex<State>,
    tasks: TaskTracker,
    user_tasks: TaskTracker,
    cancelled: CancellationToken,
    shutdown: watch::Sender<Shutdown>,
    thread_name: String,
}

/// Executor owner used by checked ABI component adapters.
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
    /// Create a component executor in the SDK library that owns its futures.
    pub fn new(config: NativeExecutorConfig, plugin_kind: &str) -> crate::Result<Self> {
        if config.worker_threads == 0 {
            return Err("executor.worker_threads must be greater than zero".into());
        }
        let thread_name = format!("nemo-relay-plugin-{plugin_kind}");
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.worker_threads)
            .thread_name(thread_name.clone())
            .enable_all()
            .build()
            .map_err(|error| format!("failed to start native plugin executor: {error}"))?;
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    accepting: true,
                    teardown_started: false,
                    runtime: Some(runtime),
                }),
                tasks: TaskTracker::new(),
                user_tasks: TaskTracker::new(),
                cancelled: CancellationToken::new(),
                shutdown: watch::channel(Shutdown::Running).0,
                thread_name,
            }),
        })
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
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.accepting {
            let runtime = state
                .runtime
                .as_ref()
                .expect("admitting component has a runtime");
            // Admission and tracking are atomic with respect to shutdown.
            self.inner.tasks.spawn_on(self.inner.user_tasks.track_future(async move {
                let result = tokio::select! {
                    biased;
                    () = component.cancelled() => AbiResult::RErr(Error::new("revoked", "component is shutting down")),
                    () = invocation.cancelled() => return,
                    result = AssertUnwindSafe(future).catch_unwind() => result.unwrap_or_else(|_| {
                        AbiResult::RErr(Error::new("panic", "native plugin future panicked"))
                    }),
                };
                let _ = send.send(result);
            }), runtime.handle());
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

    /// Admit cleanup whose completion is required even if its foreign waiter drops.
    /// Cleanup tasks ignore component cancellation and are drained during shutdown.
    pub fn execute_cleanup<T: Send + 'static>(
        &self,
        future: impl Future<Output = AbiResult<T>> + Send + 'static,
    ) -> FfiFuture<AbiResult<T>> {
        let (send, receive) = oneshot::channel();
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(runtime) = state.runtime.as_ref()
            && !state.teardown_started
        {
            self.inner.tasks.spawn_on(
                async move {
                    let result = AssertUnwindSafe(future)
                        .catch_unwind()
                        .await
                        .unwrap_or_else(|_| {
                            AbiResult::RErr(Error::new("panic", "native cleanup future panicked"))
                        });
                    let _ = send.send(result);
                },
                runtime.handle(),
            );
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
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .accepting = false;
        self.inner.user_tasks.close();
        self.inner.cancelled.cancel();
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
                if let Some(runtime) = runtime {
                    runtime.block_on(owner.tasks.wait());
                    drop(runtime);
                }
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

    /// Whether teardown has drained every admitted task and destroyed the runtime.
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
                    runtime.block_on(tasks.wait());
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
