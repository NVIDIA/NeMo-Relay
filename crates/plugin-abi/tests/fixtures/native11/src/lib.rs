// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual current v1.1 contract, built independently from the host test binary.
use abi::{Json, future::FfiFuture, v1_0 as a, v1_1 as b};
use std::sync::atomic::{AtomicUsize, Ordering};
static CREATED: AtomicUsize = AtomicUsize::new(0);
/// Inspect whether a host invoked construction before checking the contract.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_creation_count() -> usize {
    CREATED.load(Ordering::SeqCst)
}
struct Example;
impl b::Plugin for Example {
    fn allows_multiple_components(&self) -> bool {
        true
    }
    fn validate(&self, _: Json) -> Vec<a::ConfigDiagnostic> {
        vec![]
    }
    fn register(&self, _: Json, _: b::RuntimeBox) -> FfiFuture<a::Result<a::RegisteredComponent>> {
        panic!("factory compatibility does not register a component")
    }
}
abi::export_factory_v1_1!(
    kind = "fixture",
    create = || {
        CREATED.fetch_add(1, Ordering::SeqCst);
        Ok(b::PluginBox::new(Example))
    }
);
