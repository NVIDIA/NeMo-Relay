// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocation guarantees for pending foreign future polls and borrowed wake-ups.
use nemo_relay_plugin_abi::{
    Json,
    future::{FfiFuture, FutureExt as _},
    transfer::Value,
    v1_0,
    xabi::XabiType,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn allocated() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
struct WakeCount(AtomicUsize);
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Repeated pending polls must deliver borrowed wake-ups without allocating transport storage.
#[test]
fn pending_foreign_polls_and_borrowed_wakes_do_not_allocate() {
    let source = std::future::poll_fn(|cx| {
        cx.waker().wake_by_ref();
        Poll::<v1_0::Result<Json>>::Pending
    })
    .into_ffi();
    let payload = Value(source).into_payload();
    let mut remote = unsafe { Value::<FfiFuture<v1_0::Result<Json>>>::from_payload(payload) }
        .unwrap()
        .0;
    let wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..64 {
        assert!(Pin::new(&mut remote).poll(&mut cx).is_pending());
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None)).unwrap();
    assert_eq!(wakes.0.load(Ordering::Relaxed), 64);
    assert_eq!(allocations, 0, "pending foreign polls allocated");
}
