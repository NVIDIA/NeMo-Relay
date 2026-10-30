// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component fixtures for runtime implementation tests.

#![allow(dead_code)]

use nemo_relay_runtime as rt;

pub(crate) fn ensure_runtime_services() {
    rt::composition::install_runtime_services(services);
}

pub(crate) fn services() -> rt::composition::RuntimeServices {
    nemo_relay_diagnostics::integration::install_resource_memory(
        nemo_relay_resource_metrics::manager::latest_memory_view,
    );
    rt::composition::RuntimeServices {
        pricing_snapshot: || nemo_relay_model_pricing::active_pricing_resolver(),
        infer_model_provider: nemo_relay_model_pricing::infer_model_provider,
        attach_estimated_cost: nemo_relay_model_pricing::attach_estimated_cost_for_provider,
    }
}
