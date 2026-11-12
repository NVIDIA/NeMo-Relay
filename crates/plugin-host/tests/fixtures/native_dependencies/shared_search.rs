// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared dependency graphs with different inherited search paths.

#[cfg(search_leaf)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_d_value() -> u32 {
    if cfg!(search_alternate) { 2 } else { 1 }
}

#[cfg(search_shared)]
#[link(name = "relay_search_d")]
unsafe extern "C" {
    fn relay_search_d_value() -> u32;
}
#[cfg(search_shared)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_c_value() -> u32 {
    // SAFETY: fixture dependencies export matching C signatures.
    unsafe { relay_search_d_value() }
}

#[cfg(any(search_extra, search_b, all(search_a, not(search_deep))))]
#[link(name = "relay_search_c")]
unsafe extern "C" {
    fn relay_search_c_value() -> u32;
}
#[cfg(search_extra)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_x_value() -> u32 {
    // SAFETY: fixture dependencies export matching C signatures.
    unsafe { relay_search_c_value() }
}

#[cfg(all(search_a, search_deep))]
#[link(name = "relay_search_x")]
unsafe extern "C" {
    fn relay_search_x_value() -> u32;
}
#[cfg(search_a)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_a_value() -> u32 {
    // SAFETY: fixture dependencies export matching C signatures.
    unsafe {
        #[cfg(search_deep)]
        {
            relay_search_x_value()
        }
        #[cfg(not(search_deep))]
        {
            relay_search_c_value()
        }
    }
}
#[cfg(search_b)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_b_value() -> u32 {
    // SAFETY: fixture dependencies export matching C signatures.
    unsafe { relay_search_c_value() }
}

#[cfg(search_main)]
#[link(name = "relay_search_a")]
unsafe extern "C" {
    fn relay_search_a_value() -> u32;
}
#[cfg(search_main)]
#[link(name = "relay_search_b")]
unsafe extern "C" {
    fn relay_search_b_value() -> u32;
}
#[cfg(search_main)]
#[unsafe(no_mangle)]
pub extern "C" fn relay_search_value() -> u32 {
    // SAFETY: fixture dependencies export matching C signatures.
    unsafe { 10 * relay_search_a_value() + relay_search_b_value() }
}
