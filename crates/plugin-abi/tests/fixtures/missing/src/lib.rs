// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invalid native declaration; rejection must precede construction.
static EXPORTS: [abi::xabi::XabiExport; 0] = [];
static MANIFEST: abi::xabi::XabiManifest = abi::xabi::XabiManifest::new(&EXPORTS);
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_native_bootstrap() -> *const abi::bootstrap::Bootstrap {
    static HEADER: std::sync::OnceLock<abi::bootstrap::Bootstrap> = std::sync::OnceLock::new();
    HEADER.get_or_init(|| abi::bootstrap::Bootstrap {
        size: std::mem::size_of::<abi::bootstrap::Bootstrap>(),
        generation: 1,
        major: 1,
        minor: 0,
        kind: abi::xabi::XabiStr::from_static("fixture"),
        schema: abi::xabi::XabiStr::from_static(abi::bootstrap::schema()),
        manifest: &MANIFEST,
    })
}
