// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A v1.0 descriptor carried by an incompatible abi_stable generation.
#![allow(non_camel_case_types, non_local_definitions)]
use abi_stable::{
    StableAbi, export_root_module,
    library::{LibHeader, RootModule},
    prefix_type::PrefixTypeTrait,
    sabi_types::{RRef, VersionStrings},
    std_types::{ROption, RStr, RString},
};

#[repr(C)]
#[derive(StableAbi)]
pub struct NativeApiVersion {
    pub major: u32,
    pub minor: u32,
}
#[repr(C)]
#[derive(StableAbi)]
pub struct PluginDescriptor {
    pub native_api: NativeApiVersion,
    pub plugin_kind: RString,
}
#[repr(C)]
#[derive(StableAbi)]
#[sabi(kind(Prefix(prefix_ref = "Bootstrap_Ref")))]
pub struct Bootstrap {
    pub descriptor: extern "C" fn() -> PluginDescriptor,
    #[sabi(last_prefix_field)]
    pub query_module: extern "C" fn(RStr<'_>) -> ROption<RRef<'static, LibHeader>>,
}
impl RootModule for Bootstrap_Ref {
    abi_stable::declare_root_module_statics! { Bootstrap_Ref }
    const BASE_NAME: &'static str = "nemo_relay_plugin";
    const NAME: &'static str = "nemo.relay.native.bootstrap.v1";
    const VERSION_STRINGS: VersionStrings = VersionStrings::new("1.0.0");
}
extern "C" fn descriptor() -> PluginDescriptor {
    PluginDescriptor {
        native_api: NativeApiVersion { major: 1, minor: 0 },
        plugin_kind: "fixture".into(),
    }
}
extern "C" fn query(_: RStr<'_>) -> ROption<RRef<'static, LibHeader>> {
    panic!("incompatible dependency ABI must be rejected before module discovery")
}
#[export_root_module]
pub fn module() -> Bootstrap_Ref {
    Bootstrap {
        descriptor,
        query_module: query,
    }
    .leak_into_prefix()
}
