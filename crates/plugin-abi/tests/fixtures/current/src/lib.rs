// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Current v1.0 contract, built independently from the host test binary.
#[path = "../../behavior_plugin.rs"]
mod behavior;
use std::sync::atomic::{AtomicUsize, Ordering};
static CREATED: AtomicUsize = AtomicUsize::new(0);
/// Inspect whether the host constructed plugin state before checking the schema.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_creation_count() -> usize { CREATED.load(Ordering::SeqCst) }
abi::export_factory!(kind = "fixture", create = || {
    CREATED.fetch_add(1, Ordering::SeqCst);
    Ok(abi::v1_0::PluginBox::new(behavior::Example))
});
