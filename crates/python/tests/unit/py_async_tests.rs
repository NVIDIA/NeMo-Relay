// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

#[pyclass]
struct WakeBurst {
    waker: Arc<Mutex<Option<Waker>>>,
}

#[pymethods]
impl WakeBurst {
    fn __call__(&self, py: Python<'_>) {
        let waker = self.waker.lock().unwrap().clone().unwrap();
        py.detach(move || {
            std::thread::spawn(move || {
                for _ in 0..100 {
                    waker.wake_by_ref();
                }
            })
            .join()
            .unwrap();
        });
    }
}

#[test]
fn stream_driver_coalesces_wake_bursts_without_losing_later_wakes() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let asyncio = py.import("asyncio").unwrap();
        let event_loop = asyncio.call_method0("new_event_loop").unwrap();
        let waker = Arc::new(Mutex::new(None));
        let captured = Arc::clone(&waker);
        let mut polls = 0;
        let driver = EventLoopFuture {
            future: Mutex::new(Some(Box::pin(std::future::poll_fn(move |cx| {
                if polls == 2 {
                    return Poll::Ready(Python::attach(|py| 42.into_py_any(py)));
                }
                polls += 1;
                *captured.lock().unwrap() = Some(cx.waker().clone());
                Poll::Pending
            })))),
            event_loop: event_loop.clone().unbind(),
            // Exercise local polling even on platforms with unknown stack bounds.
            local_polling: Some(true),
            fallback: None,
        };
        let helper = PyModule::from_code(
            py,
            pyo3::ffi::c_str!(
                "import asyncio
async def exercise(driver, burst):
    loop = asyncio.get_running_loop()
    original = loop.call_soon_threadsafe
    scheduled = []
    fail_next = False
    def schedule(callback, *args, **kwargs):
        nonlocal fail_next
        scheduled.append(callback)
        if fail_next:
            fail_next = False
            raise RuntimeError('injected scheduling failure')
        return original(callback, *args, **kwargs)
    loop.call_soon_threadsafe = schedule
    try:
        for round in range(2):
            ready, waiter = driver.poll()
            assert not ready
            before = len(scheduled)
            # A failed scheduling attempt must allow the next wake to retry.
            fail_next = round == 0
            attempts = 2 if fail_next else 1
            burst()
            assert len(scheduled) == before + attempts
            await asyncio.wait_for(waiter, 1)
            # The callback must reset its flag, even though this waiter is done.
            burst()
            assert len(scheduled) == before + attempts + 1
            await asyncio.sleep(0)
        assert driver.poll() == (True, 42)
    finally:
        driver.cancel()
        loop.call_soon_threadsafe = original
"
            ),
            pyo3::ffi::c_str!("wake_coalescing_test.py"),
            pyo3::ffi::c_str!("wake_coalescing_test"),
        )
        .unwrap();
        let coroutine = helper
            .getattr("exercise")
            .unwrap()
            .call1((driver, WakeBurst { waker }))
            .unwrap();
        let result = event_loop.call_method1("run_until_complete", (coroutine,));
        event_loop.call_method0("close").unwrap();
        result.unwrap();
    });
}

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
