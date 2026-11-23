// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requested Rust heap bytes for a single-threaded measurement window.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Heap;
#[global_allocator]
static HEAP: Heap = Heap;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

fn allocate(size: usize) {
    REQUESTED.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: forward each operation to System with the original pointer and layout.
unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocate(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocate(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, old, size) };
        if !ptr.is_null() {
            LIVE.fetch_sub(old.size(), Relaxed);
            allocate(size);
        }
        ptr
    }
}

pub fn measure<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    let initial = LIVE.load(Relaxed);
    PEAK.store(initial, Relaxed);
    let before = REQUESTED.load(Relaxed);
    let result = operation();
    (
        result,
        REQUESTED.load(Relaxed) - before,
        PEAK.load(Relaxed) - initial,
    )
}
