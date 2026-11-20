// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently compiled host for frozen binary compatibility.
#[path = "../../behavior_host.rs"]
mod behavior;
/// Construct the invocation from this host's checked interface generation.
fn native_test_invocation(surface: abi::v1_0::Surface, value: abi::Json, context: abi::v1_0::Context) -> abi::v1_0::Invocation {
    abi::v1_0::Invocation {surface, name: "fixture".into(), value, tool_call_id: Some("call-id".into()), execution_context: None, context}
}

fn main() {
    let path = std::env::args_os().nth(1).expect("plugin path");
    let library = unsafe { abi::xabi::ModuleHandle::load(path) }.unwrap();
    let raw = unsafe {
        library.get::<unsafe extern "C" fn() -> *const abi::bootstrap::Bootstrap>(
            abi::bootstrap::SYMBOL,
        )
    }
    .unwrap();
    let (descriptor, manifest) = unsafe { abi::bootstrap::check(raw()) }.unwrap();
    std::mem::forget(library);
    abi::validate_descriptor(
        &descriptor,
        descriptor.native_api,
        "fixture",
        abi::NATIVE_API_VERSION,
    )
    .unwrap();
    let factory = unsafe { abi::bootstrap::factory(manifest) }.unwrap();
    behavior::assert_behavior(factory.create().unwrap());
}
