// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional observability integrations for NeMo Relay Core.

#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
pub(crate) fn test_mutex() -> &'static Mutex<()> {
    crate::shared_runtime::runtime_owner_test_mutex()
}

pub mod atif;
pub mod atof;
#[cfg(feature = "openinference")]
pub mod openinference;
#[cfg(feature = "otel")]
pub mod otel;
pub mod plugin_component;
