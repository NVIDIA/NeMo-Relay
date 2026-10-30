// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-global access to the shared runtime context state.
//!
//! The public API layer uses this module to resolve the single
//! [`NemoRelayContextState`] instance that owns middleware registrations and
//! runtime extensions for the current process.

use std::sync::{Arc, RwLock};

use crate::api::runtime::state::NemoRelayContextState;

static GLOBAL_CONTEXT: std::sync::OnceLock<Arc<RwLock<NemoRelayContextState>>> =
    std::sync::OnceLock::new();

/// Return the process-global runtime context state handle.
///
/// This lazily initializes the shared [`NemoRelayContextState`] on first use and
/// returns a cloned [`Arc`] handle to the same underlying [`RwLock`] on every
/// subsequent call.
///
/// # Returns
/// An [`Arc`] pointing at the singleton [`RwLock`] that stores the runtime
/// context state for the current process.
///
/// # Notes
/// All callers share the same underlying state. Mutations made through one
/// handle are visible through every other handle returned by this function.
pub fn global_context() -> Arc<RwLock<NemoRelayContextState>> {
    GLOBAL_CONTEXT
        .get_or_init(|| Arc::new(RwLock::new(NemoRelayContextState::new())))
        .clone()
}
