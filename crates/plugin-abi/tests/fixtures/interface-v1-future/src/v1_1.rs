// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A separately retained later module; v1.0 definitions remain frozen.
use crate::{
    transfer::Value,
    v1_0::{Error, PluginBox, Result},
};
use xabi::XabiType;
#[xabi::xabi(id = "nemo.relay.native.v1.1.Factory", version = 1)]
pub trait FutureFactory {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<Error>>;
    fn revision(&self) -> u32;
}
struct Adapter(fn() -> Result<PluginBox>);
impl FutureFactory for Adapter {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<Error>> {
        (self.0)().map(Value).map_err(Value)
    }
    fn revision(&self) -> u32 {
        1
    }
}
pub fn export(create: fn() -> Result<PluginBox>) -> *mut std::ffi::c_void {
    <Adapter as FutureFactory>::__xabi_export(Adapter(create))
}
pub fn schema() -> &'static str {
    static SCHEMA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        let mut result = crate::bootstrap::schema().to_owned();
        result.push_str(include_str!("v1_1.rs"));
        let mut layouts = Vec::new();
        (XabiV1AbiTraitFutureFactory::XABI_LAYOUT.collect)(&mut layouts);
        result.push_str(&format!("{layouts:?}"));
        result
    })
}
pub unsafe fn load(manifest: &xabi::XabiManifest) -> XabiV1OwnedTraitFutureFactory {
    let exports = unsafe { manifest.exports.as_slice() }.unwrap();
    let export = exports
        .iter()
        .find(|e| unsafe { e.abi_id.as_str() }.unwrap() == XabiV1AbiTraitFutureFactory::ID)
        .unwrap();
    assert_eq!(export.contract_version, 1);
    let wire = XabiV1OwnedRefTraitFutureFactory {
        size: std::mem::size_of::<XabiV1OwnedRefTraitFutureFactory>(),
        abi_version: 1,
        vtable: unsafe { (export.make)() }.cast(),
    };
    unsafe { XabiV1OwnedTraitFutureFactory::from_wire(&wire) }.unwrap()
}
