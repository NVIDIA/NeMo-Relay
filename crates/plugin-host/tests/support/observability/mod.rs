// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core-specific integration views and fixtures for observability.

#![allow(unused_imports)]
pub(crate) use crate::observability_runtime::{core_host, event_view};
pub(crate) use nemo_relay_observability::*;

pub(crate) mod atif;
pub(crate) mod atof;
pub(crate) mod openinference;
pub(crate) mod otel;
pub(crate) mod otel_genai;
pub(crate) mod otel_logs;
pub(crate) mod otel_metrics;
pub(crate) mod otel_signal;
pub(crate) mod plugin_component;
pub(crate) mod runtime_diagnostics;

#[path = "../../integration/observability/exporter_parity_tests.rs"]
mod exporter_parity_tests;

pub(crate) fn test_mutex() -> &'static std::sync::Mutex<()> {
    crate::test_support::runtime_owner_test_mutex()
}
