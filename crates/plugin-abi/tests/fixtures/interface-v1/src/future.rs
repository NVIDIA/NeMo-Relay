// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Library-local futures with async-ffi futures and wakers at the native boundary.
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

/// An owned future polled on the caller with the originating adapter's context.
pub struct FfiFuture<T>(Pin<Box<dyn Future<Output = T> + Send + 'static>>);
impl<T> Future for FfiFuture<T> {
    type Output = T;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        self.0.as_mut().poll(cx)
    }
}
/// Convert adapter work into a library-local owned future.
pub trait FutureExt: Future + Send + Sized + 'static {
    /// Retain work until completion or cancellation by drop.
    fn into_ffi(self) -> FfiFuture<Self::Output> {
        FfiFuture(Box::pin(self))
    }
}
impl<F: Future + Send + 'static> FutureExt for F {}

/// Frozen async-ffi owner with checked generation and source-owned storage.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FutureWire {
    /// Required async-ffi ABI generation.
    pub abi_version: u32,
    /// Source-allocated future storage.
    pub instance: *mut std::ffi::c_void,
    /// Move the foreign future out and release storage with the source allocator.
    pub take: unsafe extern "C" fn(*mut std::ffi::c_void) -> async_ffi::FfiFuture<xabi::XabiResult>,
    /// Release an unclaimed future with the source allocator.
    pub release: unsafe extern "C" fn(*mut std::ffi::c_void),
}
use crate::transfer::{Transfer, Value};
use xabi::XabiType;
// Keep the dependency generation named in the checked schema explicit.
const _: () = assert!(async_ffi::ABI_VERSION == 2);
unsafe extern "C" fn take_future(
    instance: *mut std::ffi::c_void,
) -> async_ffi::FfiFuture<xabi::XabiResult> {
    *unsafe { Box::from_raw(instance.cast::<async_ffi::FfiFuture<xabi::XabiResult>>()) }
}
unsafe extern "C" fn release_future(instance: *mut std::ffi::c_void) {
    if !instance.is_null() {
        drop(unsafe { Box::from_raw(instance.cast::<async_ffi::FfiFuture<xabi::XabiResult>>()) });
    }
}
unsafe impl<T: Transfer + Send + 'static> Transfer for FfiFuture<crate::v1_0::Result<T>> {
    type Wire = FutureWire;
    fn lower(self) -> FutureWire {
        let future = async_ffi::FfiFuture::new(async move {
            match self.await {
                Ok(value) => xabi::XabiResult::ok(Value(value).into_payload()),
                Err(error) => xabi::XabiResult {
                    code: xabi::ERR_EXPORT,
                    payload: Value(error).into_payload(),
                },
            }
        });
        FutureWire {
            abi_version: async_ffi::ABI_VERSION,
            instance: Box::into_raw(Box::new(future)).cast(),
            take: take_future,
            release: release_future,
        }
    }
    unsafe fn take(wire: &mut FutureWire) -> xabi::Result<Self> {
        if wire.abi_version != async_ffi::ABI_VERSION || wire.instance.is_null() {
            return Err(xabi::Error::AbiMismatch(
                "async-ffi future generation".into(),
            ));
        }
        let mut future = Some(unsafe {
            (wire.take)(std::mem::replace(&mut wire.instance, std::ptr::null_mut()))
        });
        Ok(std::future::poll_fn(move |cx| {
            // async-ffi catches panics in the source library and rethrows locally.
            // Preserve Relay's fallible future behavior at this receiving boundary.
            let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Pin::new(
                    future
                        .as_mut()
                        .expect("foreign future polled after completion"),
                )
                .poll(cx)
            }));
            if !matches!(polled, Ok(Poll::Pending)) {
                // Release completed or panicked source work even if the caller retains us.
                future.take();
            }
            match polled {
                Err(_) => Poll::Ready(Err(crate::v1_0::Error::new(
                    "abi",
                    "foreign future poll panicked",
                ))),
                Ok(Poll::Pending) => Poll::Pending,
                Ok(Poll::Ready(result)) => Poll::Ready(if result.code == xabi::OK {
                    unsafe { Value::<T>::from_payload(result.payload) }
                        .map(|value| value.0)
                        .map_err(|error| crate::v1_0::Error::new("abi", error.to_string()))
                } else if result.code == xabi::ERR_EXPORT {
                    Err(
                        unsafe { Value::<crate::v1_0::Error>::from_payload(result.payload) }
                            .map(|value| value.0)
                            .unwrap_or_else(|error| {
                                crate::v1_0::Error::new("abi", error.to_string())
                            }),
                    )
                } else {
                    unsafe { (result.payload.free)(result.payload.ptr, result.payload.len) };
                    Err(crate::v1_0::Error::new(
                        "abi",
                        format!("foreign future returned unknown status {}", result.code),
                    ))
                }),
            }
        })
        .into_ffi())
    }
    unsafe fn release(wire: &mut FutureWire) {
        if !wire.instance.is_null() {
            unsafe { (wire.release)(std::mem::replace(&mut wire.instance, std::ptr::null_mut())) }
        }
    }
}
