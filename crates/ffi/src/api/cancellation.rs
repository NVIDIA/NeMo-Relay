// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cooperative cancellation of managed calls.
use super::*;

/// An owned, thread-safe cancellation signal. Calls clone it when they start.
/// cancelling it does not free it. Release with `nemo_relay_cancellation_free`.
#[derive(Clone)]
pub struct FfiCancellation(pub(crate) tokio::sync::watch::Sender<bool>);

/// Create a fresh cancellation signal.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_cancellation_create() -> *mut FfiCancellation {
    let (signal, _) = tokio::sync::watch::channel(false);
    Box::into_raw(Box::new(FfiCancellation(signal)))
}
/// Signal cancellation. It is safe to call more than once.
/// Foreign callbacks must poll the signal or cancel their provider request.
/// Relay cannot force them to stop.
/// # Safety
/// The signal must be live or null. Do not free it during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_cancellation_cancel(signal: *const FfiCancellation) {
    if let Some(signal) = unsafe { signal.as_ref() } {
        signal.0.send_replace(true);
    }
}
/// Test whether a signal has been cancelled.
/// # Safety
/// Signal must be live or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_cancellation_is_cancelled(
    signal: *const FfiCancellation,
) -> bool {
    unsafe { signal.as_ref() }.is_some_and(|signal| *signal.0.borrow())
}
/// Release an owned signal after concurrent foreign accesses have stopped.
/// # Safety
/// Signal must be owned, live, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_cancellation_free(signal: *mut FfiCancellation) {
    if !signal.is_null() {
        drop(unsafe { Box::from_raw(signal) });
    }
}

pub(crate) async fn run<T: Send + 'static>(
    signal: Option<FfiCancellation>,
    future: impl Future<Output = FlowResult<T>> + Send + 'static,
) -> FlowResult<T> {
    let Some(signal) = signal else {
        return future.await;
    };
    let mut receiver = signal.0.subscribe();
    if *receiver.borrow() {
        return Err(FlowError::Internal("execution cancelled".into()));
    }
    // Poll in the caller's task to preserve its scope and publication context.
    tokio::select! {
        biased;
        _ = receiver.changed() => Err(FlowError::Internal("execution cancelled".into())),
        result = future => result,
    }
}
