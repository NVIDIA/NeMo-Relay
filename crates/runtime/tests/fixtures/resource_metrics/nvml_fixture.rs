// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI32, Ordering};

static INITIALIZE_RESULT: AtomicI32 = AtomicI32::new(0);

#[unsafe(no_mangle)]
pub extern "C" fn fixture_initialize_result(result: i32) {
    INITIALIZE_RESULT.store(result, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn nvmlInit_v2() -> i32 {
    INITIALIZE_RESULT.load(Ordering::Relaxed)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvmlDeviceGetCount_v2(out: *mut u32) -> i32 {
    unsafe { out.write(0) };
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn nvmlDeviceGetHandleByIndex_v2(_: u32, _: *mut *mut c_void) -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn nvmlDeviceGetUUID(_: *mut c_void, _: *mut c_char, _: u32) -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn nvmlDeviceGetMemoryInfo(_: *mut c_void, _: *mut c_void) -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn nvmlDeviceGetUtilizationRates(_: *mut c_void, _: *mut c_void) -> i32 {
    1
}
