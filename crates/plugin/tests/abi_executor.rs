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
