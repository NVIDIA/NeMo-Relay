// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for FFI tests.

use std::future::Future;

/// Resolve a future on a fresh current-thread runtime.
///
/// This helper cannot drive wrappers whose futures call
/// [`tokio::task::block_in_place`], such as execution-intercept trampolines;
/// those tests must use a multi-thread runtime.
pub(crate) fn resolve<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}
