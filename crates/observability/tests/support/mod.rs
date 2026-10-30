// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test views for exercising exporters independently of core's provider resolver.

pub(crate) use crate::*;
use nemo_relay_types::api::event::Event;

pub(crate) fn event_view(event: &Event) -> crate::normalization::EventView<'_> {
    crate::normalization::EventView::new(event, &crate::host::DetachedHost)
}
pub(crate) fn detached_host() -> std::sync::Arc<dyn crate::host::ObservabilityHost> {
    std::sync::Arc::new(crate::host::DetachedHost)
}
pub(crate) fn test_mutex() -> &'static std::sync::Mutex<()> {
    static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &MUTEX
}

pub(crate) mod atif;

pub(crate) mod atof;

pub(crate) mod otel;

pub(crate) mod openinference;

pub(crate) mod otel_genai;

pub(crate) mod otel_logs;

pub(crate) mod otel_metrics;

pub(crate) mod otel_signal;

pub(crate) mod plugin_component;

pub(crate) fn pricing_test_mutex() -> &'static std::sync::Mutex<()> {
    static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &MUTEX
}
