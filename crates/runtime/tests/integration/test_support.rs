// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;

pub fn ready<T: Send + 'static>(
    value: T,
) -> Pin<Box<dyn Future<Output = nemo_relay_types::error::Result<T>> + Send>> {
    Box::pin(async move { Ok(value) })
}

#[allow(dead_code)]
pub fn ready_result<T: Send + 'static>(
    value: nemo_relay_types::error::Result<T>,
) -> Pin<Box<dyn Future<Output = nemo_relay_types::error::Result<T>> + Send>> {
    Box::pin(async move { value })
}

#[allow(dead_code)]
pub fn initialize_runtime_services() {
    nemo_relay_runtime::composition::install_runtime_services(|| {
        nemo_relay_runtime::composition::RuntimeServices {
            pricing_snapshot: || nemo_relay_model_pricing::active_pricing_resolver(),
            infer_model_provider: nemo_relay_model_pricing::infer_model_provider,
            attach_estimated_cost: nemo_relay_model_pricing::attach_estimated_cost_for_provider,
        }
    });
}
