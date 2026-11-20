// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An obsolete development host rejects the changed v1.0 schema before construction.
fn main() {
    let path = std::env::args_os().nth(1).expect("plugin path");
    let library = unsafe { abi::xabi::ModuleHandle::load(path) }.unwrap();
    let bootstrap = unsafe {
        library.get::<unsafe extern "C" fn() -> *const abi::bootstrap::Bootstrap>(
            abi::bootstrap::SYMBOL,
        )
    }
    .unwrap();
    let count =
        unsafe { library.get::<unsafe extern "C" fn() -> usize>(b"fixture_creation_count\0") }
            .unwrap();
    assert_eq!(unsafe { count() }, 0);
    let error = unsafe { abi::bootstrap::check(bootstrap()) }
        .err()
        .expect("obsolete development host must reject changed schema");
    assert!(error.contains("incompatible native contract"), "{error}");
    assert_eq!(unsafe { count() }, 0, "rejection must precede construction");
}
