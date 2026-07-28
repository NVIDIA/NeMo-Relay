// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;

pub fn ready<T: Send + 'static>(
    value: T,
) -> Pin<Box<dyn Future<Output = nemo_relay::error::Result<T>> + Send>> {
    Box::pin(async move { Ok(value) })
}

#[allow(dead_code)]
pub fn ready_result<T: Send + 'static>(
    value: nemo_relay::error::Result<T>,
) -> Pin<Box<dyn Future<Output = nemo_relay::error::Result<T>> + Send>> {
    Box::pin(async move { value })
}
