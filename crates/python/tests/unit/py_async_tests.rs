// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

#[test]
fn stream_driver_handles_native_timer_wakeups_and_cancellation() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let asyncio = py.import("asyncio").unwrap();
        let event_loop = asyncio.call_method0("new_event_loop").unwrap();
        asyncio
            .call_method1("set_event_loop", (&event_loop,))
            .unwrap();
        pyo3_async_runtimes::tokio::run_until_complete(event_loop.clone(), async move {
            let coroutine = Python::attach(|py| {
                future_into_py(py, async {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    Ok(42)
                })
                .map(Bound::unbind)
            })?;
            let future = Python::attach(|py| {
                pyo3_async_runtimes::tokio::into_future(coroutine.into_bound(py))
            })?;
            let value = future.await?;
            assert_eq!(Python::attach(|py| value.extract::<i32>(py).unwrap()), 42);

            struct Dropped(Option<tokio::sync::oneshot::Sender<()>>);
            impl Drop for Dropped {
                fn drop(&mut self) {
                    if let Some(sender) = self.0.take() {
                        let _ = sender.send(());
                    }
                }
            }
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let guard = Dropped(Some(sender));
            let future = Python::attach(|py| {
                let coroutine = future_into_py(py, async move {
                    let _guard = guard;
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    Ok(())
                })?;
                let timeout = py
                    .import("asyncio")?
                    .call_method1("wait_for", (coroutine, 0.01))?;
                pyo3_async_runtimes::tokio::into_future(timeout)
            })?;
            let error = future.await.unwrap_err();
            assert!(Python::attach(
                |py| error.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py)
            ));
            tokio::time::timeout(Duration::from_secs(1), receiver)
                .await
                .unwrap()
                .unwrap();
            Ok(())
        })
        .unwrap();
        asyncio
            .call_method1("set_event_loop", (py.None(),))
            .unwrap();
        event_loop.call_method0("close").unwrap();
    });
}
