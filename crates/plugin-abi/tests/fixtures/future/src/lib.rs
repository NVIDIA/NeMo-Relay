// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Later minor fixture with an additional checked factory module.
#[path = "../../behavior_plugin.rs"]
mod behavior;
fn create() -> abi::v1_0::Result<abi::v1_0::PluginBox> {
    Ok(abi::v1_0::PluginBox::new(behavior::Example))
}
unsafe extern "C" fn make() -> *mut std::ffi::c_void {
    abi::v1_1::export(create)
}
static EXPORTS: [abi::xabi::XabiExport; 1] = [abi::xabi::XabiExport::new(
    abi::xabi::XabiStr::from_static(abi::v1_1::XabiV1AbiTraitFutureFactory::ID),
    abi::xabi::XabiStr::from_static("nemo.relay.native.factory.v1.1"),
    1,
    1,
    0,
    make,
)];
static MANIFEST: abi::xabi::XabiManifest = abi::xabi::XabiManifest::new(&EXPORTS);
#[unsafe(no_mangle)]
pub extern "C" fn xabi_manifest() -> *const abi::xabi::XabiManifest {
    &MANIFEST
}
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_native_bootstrap() -> *const abi::bootstrap::Bootstrap {
    static HEADER: std::sync::OnceLock<abi::bootstrap::Bootstrap> = std::sync::OnceLock::new();
    HEADER.get_or_init(|| abi::bootstrap::Bootstrap {
        size: std::mem::size_of::<abi::bootstrap::Bootstrap>(),
        generation: 1,
        major: 1,
        minor: 1,
        kind: abi::xabi::XabiStr::from_static("fixture"),
        schema: abi::xabi::XabiStr::from_static(abi::v1_1::schema()),
        manifest: &MANIFEST,
    })
}
