// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host whose interface package release differs from unchanged plugins' package releases.

#[path = "../../behavior_host.rs"]
mod behavior;

/// Check the explicit contract and full plugin behavior across Cargo package generations.
fn main() {
    let path = std::env::args_os().nth(1).expect("plugin path");
    let bootstrap = abi::abi_stable::library::lib_header_from_path(std::path::Path::new(&path))
        .unwrap()
        .init_root_module::<abi::Bootstrap_Ref>()
        .unwrap();
    let descriptor = (bootstrap.descriptor())();
    abi::validate_descriptor(
        &descriptor,
        abi::NATIVE_API_VERSION,
        "fixture",
        abi::NATIVE_API_VERSION,
    )
    .unwrap();
    let module = (bootstrap.query_module())(abi::v1_0::FACTORY_MODULE.into())
        .into_option()
        .unwrap();
    let factory = module
        .get()
        .init_root_module::<abi::v1_0::Factory_Ref>()
        .unwrap();
    behavior::assert_behavior((factory.create())().into_result().unwrap());
}
