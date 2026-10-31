// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod plugin_host;
pub(crate) fn runtime_owner_test_mutex() -> &'static std::sync::Mutex<()> {
    static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &MUTEX
}

pub(crate) fn lock_test_logging() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

pub(crate) const OTEL_RUNTIME_DELIVERY_FAILURE_MARKER: &str =
    "OpenTelemetry runtime delivery failures";
pub(crate) fn pricing_test_mutex() -> &'static std::sync::Mutex<()> {
    crate::initialize_runtime();
    static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &MUTEX
}
