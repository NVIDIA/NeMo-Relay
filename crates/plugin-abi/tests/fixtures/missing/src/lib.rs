// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatible bootstrap which omits the required factory module.
use abi::abi_stable::{
    export_root_module,
    library::LibHeader,
    prefix_type::PrefixTypeTrait,
    sabi_types::RRef,
    std_types::{ROption, RStr},
};
use abi::{Bootstrap, Bootstrap_Ref, NativeApiVersion, PluginDescriptor};

extern "C" fn descriptor() -> PluginDescriptor {
    PluginDescriptor {
        native_api: NativeApiVersion::new(1, 0),
        plugin_kind: "fixture".into(),
    }
}
extern "C" fn query(_: RStr<'_>) -> ROption<RRef<'static, LibHeader>> {
    ROption::RNone
}
#[export_root_module]
pub fn module() -> Bootstrap_Ref {
    Bootstrap {
        descriptor,
        query_module: query,
    }
    .leak_into_prefix()
}
