// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-thread allocator request traffic, not RSS, live bytes, or worker-thread totals.
//! Allocation calls/bytes include realloc's new request; frees count explicit
//! dealloc calls only. Realloc's old size is reported separately, not as a free.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationStats {
    pub allocation_calls: usize,
    pub allocated_bytes: usize,
    pub free_calls: usize,
    pub freed_bytes: usize,
    pub realloc_calls: usize,
    pub realloc_old_bytes: usize,
}
thread_local! {
    static TRACKED: Cell<Option<AllocationStats>> = const { Cell::new(None) };
}
fn note(update: impl FnOnce(&mut AllocationStats)) {
    let _ = TRACKED.try_with(|tracked| {
        if let Some(mut stats) = tracked.get() {
            update(&mut stats);
            tracked.set(Some(stats));
        }
    });
}
fn allocation(bytes: usize) {
    note(|stats| {
        stats.allocation_calls = stats.allocation_calls.saturating_add(1);
        stats.allocated_bytes = stats.allocated_bytes.saturating_add(bytes);
    });
}
struct CountingAllocator;
// SAFETY: requests are forwarded unchanged to System. TLS accounting does not
// allocate, retain pointers, or run user callbacks inside allocator operations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocation(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        note(|stats| {
            stats.free_calls = stats.free_calls.saturating_add(1);
            stats.freed_bytes = stats.freed_bytes.saturating_add(layout.size());
        });
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocation(size);
        note(|stats| {
            stats.realloc_calls = stats.realloc_calls.saturating_add(1);
            stats.realloc_old_bytes = stats.realloc_old_bytes.saturating_add(layout.size());
        });
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub fn allocation_stats_during<T>(body: impl FnOnce() -> T) -> (T, AllocationStats) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TRACKED.with(|tracked| tracked.set(None));
        }
    }
    TRACKED.with(|tracked| assert!(tracked.replace(Some(AllocationStats::default())).is_none()));
    let reset = Reset;
    let value = body();
    let stats = TRACKED.with(|tracked| tracked.replace(None).unwrap());
    drop(reset);
    (value, stats)
}
pub fn allocations_during<T>(body: impl FnOnce() -> T) -> (T, usize) {
    let (value, stats) = allocation_stats_during(body);
    (value, stats.allocation_calls)
}

#[test]
fn allocator_traffic_distinguishes_realloc_from_explicit_free() {
    let (_, stats) = allocation_stats_during(|| {
        // SAFETY: valid layouts, checked allocation results, and matching final layout.
        unsafe {
            let initial = Layout::from_size_align(8, 8).unwrap();
            let pointer = std::alloc::alloc(initial);
            assert!(!pointer.is_null());
            let pointer = std::alloc::realloc(std::hint::black_box(pointer), initial, 32);
            assert!(!pointer.is_null());
            std::alloc::dealloc(
                std::hint::black_box(pointer),
                Layout::from_size_align(32, 8).unwrap(),
            );
        }
    });
    assert_eq!(
        stats,
        AllocationStats {
            allocation_calls: 2,
            allocated_bytes: 40,
            free_calls: 1,
            freed_bytes: 32,
            realloc_calls: 1,
            realloc_old_bytes: 8,
        }
    );
}
