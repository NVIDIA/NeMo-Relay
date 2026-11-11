// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed ownership transfer. Every allocation is reclaimed in its originating library.
//!
//! These records are included in the checked contract before any foreign value is
//! decoded. A successful take clears each owner; a failed take releases everything
//! that remains, including owners nested in records, options, and vectors.

use std::{ffi::c_void, marker::PhantomData, mem::MaybeUninit};
use xabi::{XabiOwnedBytes, XabiType};

/// A Rust value with a frozen C representation and transactional ownership hooks.
///
/// # Safety
/// `Wire` must have a C-compatible layout included in the contract schema. `take`
/// must clear transferred resources even when a later field fails. `release`
/// must accept both complete and partially consumed records, exactly once.
pub unsafe trait Transfer: Sized {
    /// Copyable transport record; copying does not duplicate ownership.
    type Wire: Copy + 'static;
    /// Move the value into its transport record.
    fn lower(self) -> Self::Wire;
    /// Claim a value from an exclusively owned, compatible record.
    ///
    /// # Safety
    /// The record must come from `lower` and must not have been consumed.
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self>;
    /// Release resources not already claimed by `take`.
    ///
    /// # Safety
    /// The record must be exclusively owned and not previously released.
    unsafe fn release(wire: &mut Self::Wire);
}

/// Call-scoped owner used while constructing and decoding compound records.
pub struct Guard<T: Transfer> {
    wire: Option<T::Wire>,
    marker: PhantomData<T>,
}
impl<T: Transfer> Guard<T> {
    /// Convert one value while retaining rollback ownership.
    pub fn new(value: T) -> Self {
        Self {
            wire: Some(value.lower()),
            marker: PhantomData,
        }
    }
    /// Transfer rollback ownership to the containing record.
    pub fn into_raw(mut self) -> T::Wire {
        self.wire.take().unwrap()
    }
}
impl<T: Transfer> Drop for Guard<T> {
    /// Roll back a partially built record unless ownership moved to its parent.
    fn drop(&mut self) {
        if let Some(wire) = &mut self.wire {
            unsafe { T::release(wire) }
        }
    }
}

/// Opaque owner of a checked typed record. Its release function belongs to its allocator.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OwnedWire {
    /// Source-owned allocation; a null pointer means no retained allocation.
    pub data: *mut c_void,
    /// Release remaining nested owners and storage in the source library.
    pub release: unsafe extern "C" fn(*mut c_void),
}
unsafe extern "C" fn empty_release(_: *mut c_void) {}
impl OwnedWire {
    /// Construct a record that owns no allocation.
    pub fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            release: empty_release,
        }
    }
}
struct Owner(OwnedWire);
impl Drop for Owner {
    fn drop(&mut self) {
        unsafe { (self.0.release)(self.0.data) }
    }
}
/// Release unclaimed nested fields before freeing their source-owned record.
unsafe extern "C" fn release_record<T: Transfer>(data: *mut c_void) {
    if !data.is_null() {
        let mut wire = unsafe { Box::from_raw(data.cast::<T::Wire>()) };
        unsafe { T::release(&mut wire) };
    }
}
/// Release a typed payload and its remaining nested owners in the source library.
unsafe extern "C" fn release_payload(data: *mut u8, _: usize) {
    if !data.is_null() {
        drop(Owner(*unsafe { Box::from_raw(data.cast::<OwnedWire>()) }));
    }
}

/// A typed xabi value with rollback for all nested resources.
pub struct Value<T>(pub T);
impl<T: std::fmt::Display> std::fmt::Display for Value<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl<T: Transfer> XabiType for Value<T> {
    type Wire = OwnedWire;
    const WIRE_TYPE_NAME: &'static str = "relay.OwnedWire.v1";
    fn into_wire(self) -> OwnedWire {
        OwnedWire {
            data: Box::into_raw(Box::new(self.0.lower())).cast(),
            release: release_record::<T>,
        }
    }
    unsafe fn from_wire(wire: *const OwnedWire) -> xabi::Result<Self> {
        let wire = unsafe { wire.as_ref() }.ok_or(xabi::Error::NullPointer("owned record"))?;
        let owner = Owner(*wire);
        let value = unsafe { owner.0.data.cast::<T::Wire>().as_mut() }
            .ok_or(xabi::Error::NullPointer("typed record"))?;
        unsafe { T::take(value) }.map(Self)
    }
    unsafe fn xabi_take_from_wire(wire: *mut OwnedWire) -> xabi::Result<Self> {
        let wire = unsafe { wire.as_mut() }.ok_or(xabi::Error::NullPointer("owned record"))?;
        let owned = std::mem::replace(wire, OwnedWire::empty());
        unsafe { Self::from_wire(&owned) }
    }
    /// Clear the wire owner before releasing its storage to prevent a second release.
    unsafe fn xabi_drop_wire(wire: *mut OwnedWire) {
        if let Some(wire) = unsafe { wire.as_mut() } {
            drop(Owner(std::mem::replace(wire, OwnedWire::empty())));
        }
    }
    fn into_payload(self) -> XabiOwnedBytes {
        // The payload points at an aligned record, not copied struct padding.
        XabiOwnedBytes {
            ptr: Box::into_raw(Box::new(self.into_wire())).cast(),
            len: std::mem::size_of::<OwnedWire>(),
            free: release_payload,
        }
    }
    /// Validate payload storage and release it on both successful and failed decoding.
    unsafe fn from_payload(payload: XabiOwnedBytes) -> xabi::Result<Self> {
        struct Payload(XabiOwnedBytes);
        impl Drop for Payload {
            fn drop(&mut self) {
                unsafe { (self.0.free)(self.0.ptr, self.0.len) }
            }
        }
        let owner = Payload(payload);
        if owner.0.len != std::mem::size_of::<OwnedWire>()
            || owner.0.ptr.is_null()
            || !owner.0.ptr.cast::<OwnedWire>().is_aligned()
        {
            return Err(xabi::Error::AbiMismatch("invalid typed payload".into()));
        }
        unsafe { Self::xabi_take_from_wire(owner.0.ptr.cast()) }
    }
}

macro_rules! scalar {
    ($($ty:ty),*) => { $(unsafe impl Transfer for $ty {
        type Wire = Self;
        fn lower(self) -> Self { self }
        unsafe fn take(wire: &mut Self) -> xabi::Result<Self> { Ok(*wire) }
        unsafe fn release(_: &mut Self) {}
    })* };
}
scalar!((), u8, u16, u32, u64, i32, i64, usize);
unsafe impl Transfer for bool {
    type Wire = u8;
    fn lower(self) -> u8 {
        u8::from(self)
    }
    /// Accept only canonical zero and one bytes as Boolean values.
    unsafe fn take(wire: &mut u8) -> xabi::Result<Self> {
        match *wire {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(xabi::Error::AbiMismatch("invalid boolean".into())),
        }
    }
    unsafe fn release(_: &mut u8) {}
}

/// Tagged optional wire value with explicit initialization state.
#[repr(C)]
pub struct OptionalWire<W: Copy> {
    /// Zero for absent; one when the value field is initialized.
    pub present: u8,
    /// Initialized only when the presence tag is one.
    pub value: MaybeUninit<W>,
}
impl<W: Copy> Copy for OptionalWire<W> {}
impl<W: Copy> Clone for OptionalWire<W> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T: Transfer> Transfer for Option<T> {
    type Wire = OptionalWire<T::Wire>;
    /// Initialize optional storage only when a value is present.
    fn lower(self) -> Self::Wire {
        match self {
            Some(value) => OptionalWire {
                present: 1,
                value: MaybeUninit::new(value.lower()),
            },
            None => OptionalWire {
                present: 0,
                value: MaybeUninit::uninit(),
            },
        }
    }
    /// Decode the presence tag before reading optional storage.
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        match wire.present {
            0 => Ok(None),
            1 => unsafe { T::take(wire.value.assume_init_mut()) }.map(Some),
            _ => Err(xabi::Error::AbiMismatch("invalid optional value".into())),
        }
    }
    /// Release initialized optional storage and mark it absent.
    unsafe fn release(wire: &mut Self::Wire) {
        if wire.present == 1 {
            unsafe { T::release(wire.value.assume_init_mut()) };
            wire.present = 0;
        }
    }
}

/// Source-owned contiguous wire elements with a source release callback.
#[repr(C)]
pub struct VectorWire<W> {
    /// Source-owned allocation; a null pointer means no retained allocation.
    pub data: *mut W,
    /// Number of initialized elements in the source allocation.
    pub len: usize,
    /// Release remaining nested owners and storage in the source library.
    pub release: unsafe extern "C" fn(*mut W, usize),
}
impl<W> Copy for VectorWire<W> {}
impl<W> Clone for VectorWire<W> {
    fn clone(&self) -> Self {
        *self
    }
}
/// Release each remaining element before freeing the source-owned vector.
unsafe extern "C" fn release_vector<T: Transfer>(data: *mut T::Wire, len: usize) {
    if !data.is_null() {
        let mut values = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, len)) };
        for value in &mut values {
            unsafe { T::release(value) };
        }
    }
}
impl<W> VectorWire<W> {
    /// Construct a record that owns no allocation.
    pub fn empty<T: Transfer<Wire = W>>() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
            release: release_vector::<T>,
        }
    }
}
unsafe impl<T: Transfer> Transfer for Vec<T> {
    type Wire = VectorWire<T::Wire>;
    /// Build a wire vector with rollback if lowering any element fails.
    fn lower(self) -> Self::Wire {
        struct Builder<T: Transfer>(Vec<T::Wire>);
        impl<T: Transfer> Drop for Builder<T> {
            /// Release elements already lowered when vector construction stops early.
            fn drop(&mut self) {
                for wire in &mut self.0 {
                    unsafe { T::release(wire) };
                }
            }
        }
        let mut builder = Builder::<T>(Vec::with_capacity(self.len()));
        for value in self {
            builder.0.push(value.lower());
        }
        let values = std::mem::take(&mut builder.0).into_boxed_slice();
        let len = values.len();
        VectorWire {
            data: Box::into_raw(values).cast(),
            len,
            release: release_vector::<T>,
        }
    }
    /// Decode each vector element while leaving rollback ownership with the source.
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        if wire.data.is_null() {
            return if wire.len == 0 {
                Ok(Vec::new())
            } else {
                Err(xabi::Error::NullPointer("vector"))
            };
        }
        let values = unsafe { std::slice::from_raw_parts_mut(wire.data, wire.len) };
        values
            .iter_mut()
            .map(|value| unsafe { T::take(value) })
            .collect()
    }
    unsafe fn release(wire: &mut Self::Wire) {
        let owned = std::mem::replace(wire, VectorWire::empty::<T>());
        unsafe { (owned.release)(owned.data, owned.len) }
    }
}
unsafe impl Transfer for String {
    type Wire = <Vec<u8> as Transfer>::Wire;
    fn lower(self) -> Self::Wire {
        // Bytes have no nested owners, so preserve the source allocation directly.
        let bytes = self.into_bytes().into_boxed_slice();
        let len = bytes.len();
        VectorWire {
            data: Box::into_raw(bytes).cast(),
            len,
            release: release_vector::<u8>,
        }
    }
    /// Copy valid UTF-8 into local storage while the source retains its byte allocation.
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        if wire.data.is_null() {
            return if wire.len == 0 {
                Ok(String::new())
            } else {
                Err(xabi::Error::NullPointer("string"))
            };
        }
        let bytes = unsafe { std::slice::from_raw_parts(wire.data, wire.len) };
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|e| xabi::Error::Export(e.to_string()))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { Vec::<u8>::release(wire) }
    }
}

/// Define a checked record with rollback for each of its fields.
macro_rules! record {
    ($wire:ident, $ty:ty { $($field:ident: $ft:ty),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $wire { $(pub $field: <$ft as $crate::transfer::Transfer>::Wire),* }
        impl $wire {
            /// Append compiler-derived record and field layouts to the contract.
            pub(crate) fn append_layout(result: &mut String) {
                result.push_str(&format!(";{}:{}/{}", stringify!($wire),
                    std::mem::size_of::<Self>(), std::mem::align_of::<Self>()));
                $(
                    // Check the generated field type without reading its storage.
                    let storage = std::mem::MaybeUninit::<Self>::uninit();
                    let _: *const <$ft as $crate::transfer::Transfer>::Wire = unsafe {
                        std::ptr::addr_of!((*storage.as_ptr()).$field)
                    };
                    result.push_str(&format!(";{}.{}:{}/{}/{}", stringify!($wire), stringify!($field),
                        std::mem::offset_of!(Self, $field),
                        std::mem::size_of::<<$ft as $crate::transfer::Transfer>::Wire>(),
                        std::mem::align_of::<<$ft as $crate::transfer::Transfer>::Wire>()));
                )*
            }
        }
        unsafe impl $crate::transfer::Transfer for $ty {
            type Wire = $wire;
            fn lower(self) -> Self::Wire {
                $(let $field = $crate::transfer::Guard::<$ft>::new(self.$field);)*
                $wire { $($field: $field.into_raw()),* }
            }
            unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
                Ok(Self { $($field: unsafe { <$ft as $crate::transfer::Transfer>::take(&mut wire.$field) }?),* })
            }
            unsafe fn release(wire: &mut Self::Wire) { $(unsafe { <$ft as $crate::transfer::Transfer>::release(&mut wire.$field) };)* }
        }
    };
}
pub(crate) use record;

/// Unit result carried by the typed transport.
pub type Unit = ();
