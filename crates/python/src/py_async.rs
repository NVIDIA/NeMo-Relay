// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive streaming bridge futures on their originating Python event loop.
//!
//! Tokio still provides native I/O and timers, but ready chunks do not need a
//! worker or blocking-pool round trip to get back to Python.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

type PythonFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

/// Query stack bounds once per OS thread, including Linux's pthread attribute
/// lookup, rather than paying for that lookup on every consumer chunk.
fn caller_stack_base() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: pthread_self identifies the current live thread, and these
        // functions return its stack's upper address and allocation size.
        unsafe {
            let thread = libc::pthread_self();
            let top = libc::pthread_get_stackaddr_np(thread) as usize;
            let size = libc::pthread_get_stacksize_np(thread);
            Some(top.saturating_sub(size))
        }
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: the attribute storage is initialized by pthread_getattr_np
        // before it is read, and destroyed after querying the current stack.
        unsafe {
            let mut attr = std::mem::MaybeUninit::<libc::pthread_attr_t>::uninit();
            if libc::pthread_getattr_np(libc::pthread_self(), attr.as_mut_ptr()) != 0 {
                return None;
            }
            let mut attr = attr.assume_init();
            let mut base = std::ptr::null_mut();
            let mut size = 0;
            let result = libc::pthread_attr_getstack(&attr, &mut base, &mut size);
            libc::pthread_attr_destroy(&mut attr);
            (result == 0).then_some(base as usize)
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        // Keep the established Tokio path where native stack bounds are unknown.
        None
    }
}

/// Small Python-created thread stacks cannot safely poll the native pipeline.
fn caller_has_stack_headroom() -> bool {
    thread_local! {
        static STACK_BASE: std::cell::OnceCell<Option<usize>> = const { std::cell::OnceCell::new() };
    }
    let base = STACK_BASE.with(|base| *base.get_or_init(caller_stack_base));
    let marker = 0_u8;
    base.is_some_and(|base| (&marker as *const u8 as usize).saturating_sub(base) >= 512 * 1024)
}

pub(crate) fn on_event_loop(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> bool {
    py.import("asyncio")
        .and_then(|asyncio| asyncio.call_method0("get_running_loop"))
        .is_ok_and(|running| running.is(event_loop))
}

#[pyclass]
struct FutureWake {
    waiter: Py<PyAny>,
}

#[pymethods]
impl FutureWake {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        let waiter = self.waiter.bind(py);
        if !waiter.call_method0("done")?.is_truthy()? {
            waiter.call_method1("set_result", (py.None(),))?;
        }
        Ok(())
    }
}

struct EventLoopWake {
    event_loop: Py<PyAny>,
    callback: Py<FutureWake>,
}

impl Wake for EventLoopWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        Python::attach(|py| {
            let event_loop = self.event_loop.bind(py);
            if on_event_loop(py, event_loop) {
                let _ = self.callback.borrow(py).__call__(py);
            } else if !event_loop
                .call_method0("is_closed")
                .and_then(|closed| closed.is_truthy())
                .unwrap_or(true)
            {
                let _ = event_loop.call_method1("call_soon_threadsafe", (self.callback.bind(py),));
            }
        });
    }
}

#[pyclass]
struct EventLoopFuture {
    future: Mutex<Option<PythonFuture>>,
    event_loop: Py<PyAny>,
    local_polling: Option<bool>,
    fallback: Option<Py<PyAny>>,
}

#[pymethods]
impl EventLoopFuture {
    fn poll(&mut self, py: Python<'_>) -> PyResult<(bool, Py<PyAny>)> {
        if !on_event_loop(py, self.event_loop.bind(py)) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "stream future must be awaited on its originating event loop",
            ));
        }
        let waiter = self
            .event_loop
            .bind(py)
            .call_method0("create_future")?
            .unbind();
        let waker = Waker::from(Arc::new(EventLoopWake {
            event_loop: self.event_loop.clone_ref(py),
            callback: Py::new(
                py,
                FutureWake {
                    waiter: waiter.clone_ref(py),
                },
            )?,
        }));
        let mut context = Context::from_waker(&waker);
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let _runtime = runtime.enter();
        let slot = self.future.get_mut().expect("stream future lock poisoned");
        if !*self
            .local_polling
            .get_or_insert_with(caller_has_stack_headroom)
            && self.fallback.is_none()
        {
            let future = slot.take().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("stream future already completed")
            })?;
            let fallback = pyo3_async_runtimes::tokio::future_into_py(py, future)?.unbind();
            *slot = Some(task_result(fallback.bind(py))?);
            self.fallback = Some(fallback);
        }
        let future = slot.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("stream future already completed")
        })?;
        // Translate native panics just as the Tokio/PyO3 bridge does, and drop
        // the future on both completion and cancellation to release resources.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            future.as_mut().poll(&mut context)
        }));
        match result {
            Ok(Poll::Pending) => Ok((false, waiter)),
            Ok(Poll::Ready(result)) => {
                slot.take();
                result.map(|value| (true, value))
            }
            Err(panic) => {
                slot.take();
                Err(pyo3_async_runtimes::err::RustPanic::new_err(
                    crate::py_api::panic_message(panic.as_ref()).to_owned(),
                ))
            }
        }
    }

    fn cancel(&mut self, py: Python<'_>) {
        if let Some(fallback) = self.fallback.take() {
            let _ = fallback.bind(py).call_method0("cancel");
        }
        self.future
            .get_mut()
            .expect("stream future lock poisoned")
            .take();
    }
}

pub(crate) fn future_into_py<'py, F, T>(py: Python<'py>, future: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    future_into_py_with_locals(py, locals, future)
}

pub(crate) fn future_into_py_with_locals<'py, F, T>(
    py: Python<'py>,
    locals: pyo3_async_runtimes::TaskLocals,
    future: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let event_loop = locals.event_loop(py).unbind();
    let future = pyo3_async_runtimes::tokio::scope(locals, async move {
        let value = future.await?;
        Python::attach(|py| value.into_py_any(py))
    });
    let driver = EventLoopFuture {
        future: Mutex::new(Some(Box::pin(future))),
        event_loop,
        local_polling: None,
        fallback: None,
    };
    py.import("nemo_relay._event_sanitizer_context")?
        .getattr("drive_stream_future")?
        .call1((driver,))
}

#[pyclass]
struct TaskComplete {
    sender: Option<tokio::sync::oneshot::Sender<PyResult<Py<PyAny>>>>,
}

#[pymethods]
impl TaskComplete {
    fn __call__(&mut self, task: &Bound<'_, PyAny>) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(task.call_method0("result").map(Bound::unbind));
        }
    }
}

/// Await an already scheduled Python task without another scheduling hop.
pub(crate) fn task_result(task: &Bound<'_, PyAny>) -> PyResult<PythonFuture> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    task.call_method1(
        "add_done_callback",
        (TaskComplete {
            sender: Some(sender),
        },),
    )?;
    Ok(Box::pin(async move {
        receiver.await.map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Python stream task ended without a result")
        })?
    }))
}

#[cfg(test)]
#[path = "../tests/unit/py_async_tests.rs"]
mod tests;
