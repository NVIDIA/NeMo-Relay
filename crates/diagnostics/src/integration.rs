// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional integrations supplied by the embedding application.
use crate::{Diagnostics, ResourceMemoryView};
use std::sync::OnceLock;

/// Logging attachment without a dependency on a logging backend.
pub struct LoggingIntegration {
    /// Attach an existing logger to the selected diagnostic context.
    pub attach: fn(&Diagnostics),
    /// Report attachment coverage without starting logging.
    pub coverage: fn() -> &'static str,
}
static LOGGING: OnceLock<LoggingIntegration> = OnceLock::new();
static RESOURCE_MEMORY: OnceLock<fn(Option<u32>) -> ResourceMemoryView> = OnceLock::new();

/// Install the logging integration once, preserving the selected backend.
pub fn install_logging(integration: LoggingIntegration) {
    let _ = LOGGING.set(integration);
}
/// Supply existing resource samples without starting collection.
pub fn install_resource_memory(provider: fn(Option<u32>) -> ResourceMemoryView) {
    let _ = RESOURCE_MEMORY.set(provider);
}
pub(crate) fn attach_logging(context: &Diagnostics) {
    if let Some(integration) = LOGGING.get() {
        (integration.attach)(context);
    }
}
pub(crate) fn logging_coverage() -> &'static str {
    LOGGING
        .get()
        .map(|integration| (integration.coverage)())
        .unwrap_or("not_owned_or_closed")
}
pub(crate) fn resource_memory(pid: Option<u32>) -> ResourceMemoryView {
    RESOURCE_MEMORY
        .get()
        .map(|provider| provider(pid))
        .unwrap_or_else(ResourceMemoryView::disabled)
}
