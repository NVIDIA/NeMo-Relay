// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal native sidecar and consumer, compiled independently by the fixture recipe.
#[cfg(not(native_sidecar))]
#[cfg_attr(native_main_image, link(name = "relay_resident_image"))]
#[cfg_attr(native_delay_import, link(name = "relay_delayed_dependency"))]
#[cfg_attr(
    not(any(native_main_image, native_delay_import)),
    link(name = "relay_native_dependency")
)]
unsafe extern "C" {
    fn relay_dependency_value() -> u32;
}

#[cfg(native_sidecar)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_dependency_value() -> u32 {
    if cfg!(native_alternate) { 17 } else { 7 }
}

#[cfg(not(native_sidecar))]
#[unsafe(no_mangle)]
pub extern "C" fn relay_consumer_value() -> u32 {
    // SAFETY: the linked fixture has the matching C signature and no shared state.
    unsafe { relay_dependency_value() + 1 }
}
