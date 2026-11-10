// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Library-local futures. Only xabi's generated foreign futures cross the boundary.
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

/// Frozen xabi future descriptor with explicit single-owner transfer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FutureWire {
    /// Size of the xabi future descriptor in bytes.
    pub size: usize,
    /// xabi future descriptor generation.
    pub abi_version: u32,
    /// Exclusively owned future state in its originating library.
    pub instance: *mut std::ffi::c_void,
    /// Poll the source future using an xabi waker and result slot.
    pub poll: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const xabi::XabiWaker,
        *mut xabi::XabiResult,
    ) -> i32,
    /// Release the future in its originating library, including unpolled work.
    pub release: unsafe extern "C" fn(*mut std::ffi::c_void),
}
use crate::transfer::{Transfer, Value};
unsafe impl<T: Transfer + Send + 'static> Transfer for FfiFuture<crate::v1_0::Result<T>> {
    type Wire = FutureWire;
    fn lower(self) -> FutureWire {
        let future =
            xabi::XabiFuture::from_result_value(
                async move { self.await.map(Value).map_err(Value) },
            );
        FutureWire {
            size: future.size,
            abi_version: future.abi_version,
            instance: future.instance,
            poll: future.poll,
            release: future.release,
        }
    }
    unsafe fn take(wire: &mut FutureWire) -> xabi::Result<Self> {
        let foreign = xabi::XabiFuture {
            size: wire.size,
            abi_version: wire.abi_version,
            instance: wire.instance,
            poll: wire.poll,
            release: wire.release,
        };
        // Validate while the containing record still owns rollback. Construction
        // cannot fail after the same descriptor has passed this check.
        foreign.validate()?;
        let future = xabi::XabiTypedFuture::<Value<crate::v1_0::Error>, Value<T>>::new(foreign)?;
        wire.instance = std::ptr::null_mut();
        Ok(async move {
            future.await.map(|v| v.0).map_err(|e| match e {
                xabi::XabiCallError::Export(Value(e)) => e,
                xabi::XabiCallError::Runtime(e) => crate::v1_0::Error::new("abi", e.to_string()),
            })
        }
        .into_ffi())
    }
    unsafe fn release(wire: &mut FutureWire) {
        if !wire.instance.is_null() {
            unsafe { (wire.release)(std::mem::replace(&mut wire.instance, std::ptr::null_mut())) }
        }
    }
}
