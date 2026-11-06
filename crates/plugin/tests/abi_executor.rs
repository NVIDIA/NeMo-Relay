// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked ABI executor ownership, cancellation, and teardown.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use nemo_relay_plugin::abi::v1_0::{Component, Result as AbiResult};
use nemo_relay_plugin::{NativeExecutorConfig, abi_executor::ComponentExecutor};

struct Dropped(Arc<AtomicBool>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
fn executor() -> ComponentExecutor {
    ComponentExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "lifecycle-test").unwrap()
}
async fn close(executor: &ComponentExecutor) {
    tokio::time::timeout(Duration::from_secs(5), executor.shutdown())
        .await
        .unwrap()
        .into_result()
        .unwrap();
    assert!(executor.is_closed());
    assert_eq!(executor.active_tasks(), 0);
}

#[tokio::test]
async fn user_future_runs_on_its_own_executor_and_panics_become_errors() {
    let executor = executor();
    let name = executor
        .execute(async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            AbiResult::ROk(std::thread::current().name().unwrap().to_owned())
        })
        .await
        .into_result()
        .unwrap();
    assert!(name.starts_with("nemo-relay-plugin-lifecycle-test"));
    let panic = executor
        .execute::<()>(async {
            panic!("user callback");
        })
        .await
        .into_result()
        .unwrap_err();
    assert_eq!(panic.code.as_str(), "panic");
    close(&executor).await;
}

#[tokio::test]
async fn dropping_unpolled_foreign_future_cancels_already_running_user_future() {
    let executor = executor();
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = Dropped(dropped.clone());
    let (started, ready) = tokio::sync::oneshot::channel();
    let future = executor.execute(async move {
        let _probe = probe;
        started.send(()).unwrap();
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    ready.await.unwrap();
    drop(future); // The foreign future has never been polled.
    tokio::time::timeout(Duration::from_secs(5), async {
        while !dropped.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    close(&executor).await;
}

#[tokio::test]
async fn concurrent_shutdown_drains_tasks_and_stale_executor_is_inert() {
    let executor = executor();
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = Dropped(dropped.clone());
    let pending = executor.execute(async move {
        let _probe = probe;
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    let (first, second) = tokio::join!(executor.shutdown(), executor.shutdown());
    first.into_result().unwrap();
    second.into_result().unwrap();
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(executor.active_tasks(), 0);
    assert!(executor.is_closed());
    assert_eq!(
        pending.await.into_result().unwrap_err().code.as_str(),
        "revoked"
    );
    let called = Arc::new(AtomicBool::new(false));
    let marker = called.clone();
    let denied = executor.execute(async move {
        marker.store(true, Ordering::Release);
        AbiResult::ROk(())
    });
    assert_eq!(
        denied.await.into_result().unwrap_err().code.as_str(),
        "revoked"
    );
    assert!(!called.load(Ordering::Acquire));
    close(&executor).await;
    let fresh = self::executor();
    fresh
        .execute(async { AbiResult::ROk(()) })
        .await
        .into_result()
        .unwrap();
    close(&fresh).await;
}

#[tokio::test]
async fn shutdown_started_from_own_executor_does_not_wait_on_itself() {
    let executor = executor();
    let owner = executor.clone();
    let (send, receive) = tokio::sync::oneshot::channel();
    let pending = executor.execute(async move {
        // Dropping an unpolled shutdown future still starts teardown. Destruction
        // runs on the dedicated thread after this admitted task is released.
        drop(owner.shutdown());
        send.send(()).unwrap();
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    receive.await.unwrap();
    close(&executor).await;
    assert!(pending.await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_last_handle_drops_teardown_off_host_workers() {
    let executor = executor();
    let first = executor.clone();
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let first_barrier = barrier.clone();
    let pending = executor.execute(async {
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    let one = tokio::spawn(async move {
        first_barrier.wait().await;
        drop(first);
    });
    let two = tokio::spawn(async move {
        barrier.wait().await;
        drop(executor);
    });
    one.await.unwrap();
    two.await.unwrap();
    let error = tokio::time::timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .into_result()
        .unwrap_err();
    assert_eq!(error.code.as_str(), "revoked");
}

#[tokio::test]
async fn cleanup_survives_dropped_waiters_and_remains_admissible_after_revocation() {
    let executor = executor();
    executor.stop_admission();
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = Dropped(dropped.clone());
    let (send, receive) = tokio::sync::oneshot::channel();
    let cleanup = executor.execute_cleanup(async move {
        let _probe = probe;
        receive.await.unwrap();
        AbiResult::ROk(())
    });
    drop(cleanup);
    let denied = executor
        .execute(async { AbiResult::ROk(()) })
        .await
        .into_result()
        .unwrap_err();
    assert_eq!(denied.code.as_str(), "revoked");
    send.send(()).unwrap();
    close(&executor).await;
    assert!(dropped.load(Ordering::Acquire));
}

/// Direct polling stays on the caller while timers and spawned work use the SDK runtime.
#[tokio::test]
async fn direct_polling_preserves_caller_thread_and_plugin_runtime_context() {
    let executor = executor();
    let runtime_id = executor
        .execute(async { AbiResult::ROk(tokio::runtime::Handle::current().id()) })
        .await
        .into_result()
        .unwrap();
    let caller = std::thread::current().id();
    let observed = executor
        .execute_direct(async move {
            assert_eq!(std::thread::current().id(), caller);
            tokio::time::sleep(Duration::from_millis(1)).await;
            assert_eq!(std::thread::current().id(), caller);
            AbiResult::ROk(tokio::runtime::Handle::current().id())
        })
        .await
        .into_result()
        .unwrap();
    assert_eq!(runtime_id, observed);
    let error = executor
        .execute_direct::<()>(async {
            panic!("direct callback");
        })
        .await
        .into_result()
        .unwrap_err();
    assert_eq!(error.code.as_str(), "panic");
    close(&executor).await;
}

/// Retained foreign futures must release captured state even if they were never polled.
#[tokio::test]
async fn direct_unpolled_drop_and_shutdown_release_future_ownership() {
    let executor = executor();
    let first = Arc::new(AtomicBool::new(false));
    let probe = Dropped(first.clone());
    let future = executor.execute_direct(async move {
        let _probe = probe;
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    assert!(!first.load(Ordering::Acquire));
    drop(future);
    assert!(first.load(Ordering::Acquire));
    assert_eq!(executor.active_tasks(), 0);
    let second = Arc::new(AtomicBool::new(false));
    let probe = Dropped(second.clone());
    let retained = executor.execute_direct(async move {
        let _probe = probe;
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    close(&executor).await;
    assert!(second.load(Ordering::Acquire));
    assert_eq!(
        retained.await.into_result().unwrap_err().code.as_str(),
        "revoked"
    );
    // A stale wrapper cannot retain the shared runtime after component close.
    let fresh =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 2 }, "lifecycle-test")
            .unwrap();
    close(&fresh).await;
}

/// Component cancellation must not destroy a runtime still used by another component.
#[tokio::test]
async fn shared_runtime_keeps_component_tasks_and_shutdown_isolated() {
    let first = executor();
    let second = executor();
    let a = first
        .execute(async { AbiResult::ROk(tokio::runtime::Handle::current().id()) })
        .await
        .into_result()
        .unwrap();
    let b = second
        .execute(async { AbiResult::ROk(tokio::runtime::Handle::current().id()) })
        .await
        .into_result()
        .unwrap();
    assert_eq!(a, b);
    assert!(
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 2 }, "lifecycle-test")
            .is_err()
    );
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = Dropped(dropped.clone());
    let (started, ready) = tokio::sync::oneshot::channel();
    let task = first
        .spawn(async move {
            let _probe = probe;
            started.send(()).unwrap();
            std::future::pending::<()>().await
        })
        .unwrap();
    ready.await.unwrap();
    close(&first).await;
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(dropped.load(Ordering::Acquire));
    second
        .execute_direct(async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            AbiResult::ROk(())
        })
        .await
        .into_result()
        .unwrap();
    close(&second).await;
    let fresh =
        ComponentExecutor::new(NativeExecutorConfig { worker_threads: 2 }, "lifecycle-test")
            .unwrap();
    let id = fresh
        .execute(async { AbiResult::ROk(tokio::runtime::Handle::current().id()) })
        .await
        .into_result()
        .unwrap();
    assert_ne!(a, id);
    close(&fresh).await;
}

/// Self-revocation during a direct poll must not wait on the poll's own mutex.
#[tokio::test]
async fn direct_future_can_start_its_own_shutdown() {
    let executor = executor();
    let owner = executor.clone();
    let result = executor.execute_direct(async move {
        drop(owner.shutdown());
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    assert!(
        tokio::time::timeout(Duration::from_secs(5), result)
            .await
            .unwrap()
            .is_err()
    );
    close(&executor).await;
}

/// Host components use the caller's runtime, including when later calls use a fresh runtime.
#[test]
fn host_component_does_not_capture_the_activation_runtime() {
    let executor = ComponentExecutor::for_host();
    for _ in 0..2 {
        let host = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let expected = host.handle().id();
        host.block_on(async {
            assert_eq!(
                executor
                    .execute_direct(async {
                        AbiResult::ROk(tokio::runtime::Handle::current().id())
                    })
                    .await
                    .into_result()
                    .unwrap(),
                expected
            );
        });
    }
    futures::executor::block_on(executor.shutdown())
        .into_result()
        .unwrap();
    assert!(executor.is_closed());
}

/// Inline middleware, registration, blocking callbacks, and cleanup need no Tokio runtime.
#[test]
fn inline_components_do_not_create_a_tokio_runtime() {
    let executor = ComponentExecutor::inline("runtime-free-test");
    assert!(
        executor
            .with_context(|| tokio::runtime::Handle::try_current().is_err())
            .unwrap()
    );
    futures::executor::block_on(async {
        executor
            .execute_direct(async {
                assert!(tokio::runtime::Handle::try_current().is_err());
                AbiResult::ROk(())
            })
            .await
            .into_result()
            .unwrap();
        executor
            .execute(async {
                assert!(tokio::runtime::Handle::try_current().is_err());
                AbiResult::ROk(())
            })
            .await
            .into_result()
            .unwrap();
        executor
            .execute_blocking(|| {
                assert!(tokio::runtime::Handle::try_current().is_err());
                AbiResult::ROk(())
            })
            .await
            .into_result()
            .unwrap();
        executor
            .execute_cleanup(async {
                assert!(tokio::runtime::Handle::try_current().is_err());
                AbiResult::ROk(())
            })
            .await
            .into_result()
            .unwrap();
        assert!(executor.spawn(async {}).is_err());
        executor.shutdown().await.into_result().unwrap();
    });
    assert!(executor.is_closed());
    assert_eq!(executor.active_tasks(), 0);
}

/// Operation cancellation releases a retained waiter without revoking the rest of its component.
#[test]
fn direct_operation_cancellation_is_independent_of_component_shutdown() {
    let executor = ComponentExecutor::inline("operation-cancel-test");
    let dropped = Arc::new(AtomicBool::new(false));
    let probe = Dropped(dropped.clone());
    let (future, cancellation) = executor.execute_direct_tracked(async move {
        let _probe = probe;
        std::future::pending::<()>().await;
        AbiResult::ROk(())
    });
    cancellation.cancel();
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(executor.active_tasks(), 0);
    futures::executor::block_on(async {
        assert_eq!(
            future.await.into_result().unwrap_err().code.as_str(),
            "cancelled"
        );
        executor
            .execute_direct(async { AbiResult::ROk(()) })
            .await
            .into_result()
            .unwrap();
        executor.shutdown().await.into_result().unwrap();
    });
}

#[test]
fn cancellation_racing_a_pending_poll_releases_retained_waiters() {
    let executor = ComponentExecutor::inline("cancel-poll-race");
    for _ in 0..256 {
        let dropped = Arc::new(AtomicBool::new(false));
        let probe = Dropped(dropped.clone());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let polling = barrier.clone();
        let (mut future, cancellation) = executor.execute_direct_tracked(async move {
            let _probe = probe;
            futures::future::poll_fn(move |_| {
                polling.wait();
                std::task::Poll::<()>::Pending
            })
            .await;
            AbiResult::ROk(())
        });
        let poller = std::thread::spawn(move || {
            let waker = futures::task::noop_waker();
            let mut context = std::task::Context::from_waker(&waker);
            let _ = std::pin::Pin::new(&mut future).poll(&mut context);
            future
        });
        barrier.wait();
        cancellation.cancel();
        let retained = poller.join().unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(executor.active_tasks(), 0);
        drop(retained);
    }
    futures::executor::block_on(executor.shutdown())
        .into_result()
        .unwrap();
}
