// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host compiled against a synthetic later interface release.
use abi::{
    Bootstrap_Ref, NativeApiVersion as Api, abi_stable::library::lib_header_from_path,
    validate_descriptor,
};
#[path = "../../behavior_host.rs"]
mod behavior;
fn main() {
    let path = std::env::args_os().nth(1).expect("plugin path");
    let bootstrap = lib_header_from_path(std::path::Path::new(&path))
        .unwrap()
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let descriptor = (bootstrap.descriptor())();
    validate_descriptor(
        &descriptor,
        descriptor.native_api,
        "fixture",
        Api::new(1, 1),
    )
    .unwrap();
    let module = if descriptor.native_api.minor == 0 {
        "nemo.relay.native.factory.v1.0"
    } else {
        "nemo.relay.native.factory.v1.1"
    };
    let module = (bootstrap.query_module())(module.into())
        .into_option()
        .unwrap();
    if descriptor.native_api.minor == 0 {
        let factory = module
            .get()
            .init_root_module::<abi::v1_0::Factory_Ref>()
            .unwrap();
        behavior::assert_behavior((factory.create())().into_result().unwrap());
    } else {
        let factory = module
            .get()
            .init_root_module::<abi::v1_1::Factory_Ref>()
            .unwrap();
        assert_eq!((factory.revision())(), 1);
        behavior::assert_behavior((factory.v1_0().create())().into_result().unwrap());
    }
}
