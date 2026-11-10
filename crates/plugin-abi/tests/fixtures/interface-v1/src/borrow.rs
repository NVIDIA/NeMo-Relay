// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Call-scoped views used by bounded, synchronous diagnostic operations.
//!
//! These views support synchronous argument transfer only. Encoding an owned
//! payload panics; decoding an owned payload releases it and returns an error.
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

// Owned payloads cannot retain these borrows or safely copy their struct padding.
macro_rules! call_only_payload {
    () => {
        fn into_payload(self) -> xabi::XabiOwnedBytes {
            panic!("borrowed ABI values support synchronous argument transfer only")
        }
        unsafe fn from_payload(payload: xabi::XabiOwnedBytes) -> xabi::Result<Self> {
            unsafe { (payload.free)(payload.ptr, payload.len) };
            Err(xabi::Error::AbiMismatch(
                "borrowed ABI values support synchronous argument transfer only".into(),
            ))
        }
    };
}
pub(crate) use call_only_payload;

#[repr(C)]
#[derive(Clone, Copy)]
/// UTF-8 text borrowed for one synchronous call.
pub struct Str<'a> {
    data: *const u8,
    len: usize,
    lifetime: PhantomData<&'a str>,
}
impl<'a> Str<'a> {
    /// Borrow valid UTF-8 text without copying it.
    pub const fn from_str(value: &'a str) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
            lifetime: PhantomData,
        }
    }
    /// Access the text during its original borrow.
    pub fn as_str(&self) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.data, self.len)) }
    }
}
impl<'a> From<&'a str> for Str<'a> {
    fn from(value: &'a str) -> Self {
        Self::from_str(value)
    }
}
impl Deref for Str<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl std::fmt::Debug for Str<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}
impl std::fmt::Display for Str<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

#[repr(C)]
/// Read-only elements borrowed for one synchronous call.
pub struct Slice<'a, T> {
    data: *const T,
    len: usize,
    lifetime: PhantomData<&'a [T]>,
}
impl<T> Copy for Slice<'_, T> {}
impl<T> Clone for Slice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Slice<'a, T> {
    /// Borrow a contiguous slice without copying its elements.
    pub const fn from_slice(value: &'a [T]) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
            lifetime: PhantomData,
        }
    }
    /// Access the elements while the original borrow is live.
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}
impl<'a, T> From<&'a [T]> for Slice<'a, T> {
    fn from(value: &'a [T]) -> Self {
        Self::from_slice(value)
    }
}
impl<T> Deref for Slice<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

#[repr(C)]
/// Exclusive elements borrowed for one synchronous call.
pub struct SliceMut<'a, T> {
    data: *mut T,
    len: usize,
    lifetime: PhantomData<&'a mut [T]>,
}
impl<'a, T> SliceMut<'a, T> {
    /// Borrow caller-owned storage exclusively for this call.
    pub fn from_mut_slice(value: &'a mut [T]) -> Self {
        Self {
            data: value.as_mut_ptr(),
            len: value.len(),
            lifetime: PhantomData,
        }
    }
    /// Access the caller-owned storage exclusively.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }
    /// Access the elements while the original borrow is live.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}
impl<T> Deref for SliceMut<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T> DerefMut for SliceMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Copyable raw view; it conveys exclusive access only for a synchronous call.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OutputWire {
    data: *mut u8,
    len: usize,
}
impl xabi::XabiType for SliceMut<'_, u8> {
    call_only_payload!();
    type Wire = OutputWire;
    const WIRE_TYPE_NAME: &'static str = "relay.OutputWire.v1";
    fn into_wire(self) -> OutputWire {
        OutputWire {
            data: self.data,
            len: self.len,
        }
    }
    unsafe fn from_wire(wire: *const OutputWire) -> xabi::Result<Self> {
        let wire = unsafe { wire.as_ref() }.ok_or(xabi::Error::NullPointer("output buffer"))?;
        Ok(Self {
            data: wire.data,
            len: wire.len,
            lifetime: PhantomData,
        })
    }
}

/// Opaque transport copy. The erased lifetime cannot be accessed through safe Rust.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BorrowedWire<T: Copy>(T);
impl<T: Copy> BorrowedWire<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }
    /// The borrowed data must still be alive and confined to the original call.
    pub(crate) unsafe fn get(&self) -> T {
        self.0
    }
}

/// Borrowed arguments are never stored by these synchronous generated methods.
macro_rules! borrowed {
    ($ty:ident) => {
        impl xabi::XabiType for $ty<'_> {
            $crate::borrow::call_only_payload!();
            type Wire = $crate::borrow::BorrowedWire<$ty<'static>>;
            const WIRE_TYPE_NAME: &'static str = concat!("relay.", stringify!($ty), ".v1");
            fn into_wire(self) -> Self::Wire {
                unsafe {
                    $crate::borrow::BorrowedWire::new(std::mem::transmute::<$ty<'_>, $ty<'static>>(
                        self,
                    ))
                }
            }
            unsafe fn from_wire(wire: *const Self::Wire) -> xabi::Result<Self> {
                let wire =
                    unsafe { wire.as_ref() }.ok_or(xabi::Error::NullPointer(stringify!($ty)))?;
                Ok(unsafe { wire.get() })
            }
        }
    };
}
pub(crate) use borrowed;
borrowed!(Str);

/// Caller-owned diagnostic reply. Error replies never publish a partial value.
#[repr(C)]
#[derive(Default)]
pub struct Reply<T> {
    /// Zero on success; otherwise a frozen diagnostic status code.
    pub status: u32,
    /// Result storage, meaningful only when status is zero.
    pub value: T,
}
impl<T> Reply<T> {
    /// Read the result only if the callee reported success.
    pub fn into_result(self) -> Result<T, u32> {
        if self.status == 0 {
            Ok(self.value)
        } else {
            Err(self.status)
        }
    }
}
/// Exclusive caller storage valid for one synchronous call.
pub struct Output<'a, T> {
    data: *mut T,
    lifetime: PhantomData<&'a mut T>,
}
impl<'a, T> Output<'a, T> {
    /// Borrow exclusive reply storage for one synchronous call.
    pub fn new(value: &'a mut T) -> Self {
        Self {
            data: value,
            lifetime: PhantomData,
        }
    }
}
impl<T> Output<'_, Reply<T>> {
    /// Publish a complete result or an error status.
    pub fn write(&mut self, value: Result<T, u32>) {
        let output = unsafe { &mut *self.data };
        match value {
            Ok(value) => {
                output.value = value;
                output.status = 0
            }
            Err(status) => output.status = status,
        }
    }
}
#[repr(C)]
/// Opaque pointer to caller-owned synchronous reply storage.
pub struct ReplyWire<T> {
    data: *mut T,
}
impl<T> Copy for ReplyWire<T> {}
impl<T> Clone for ReplyWire<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: 'static> xabi::XabiType for Output<'_, T> {
    call_only_payload!();
    type Wire = ReplyWire<T>;
    const WIRE_TYPE_NAME: &'static str = "relay.ReplyWire.v1";
    fn into_wire(self) -> Self::Wire {
        ReplyWire { data: self.data }
    }
    unsafe fn from_wire(wire: *const Self::Wire) -> xabi::Result<Self> {
        let wire = unsafe { wire.as_ref() }.ok_or(xabi::Error::NullPointer("diagnostic reply"))?;
        if wire.data.is_null() {
            return Err(xabi::Error::NullPointer("diagnostic reply storage"));
        }
        Ok(Self {
            data: wire.data,
            lifetime: PhantomData,
        })
    }
}
