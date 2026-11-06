// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generate checked ABI metadata from the explicit native contract.

#[path = "src/contract.rs"]
mod contract;

/// Parse the canonical contract identifier before generating any ABI metadata.
fn contract_numbers() -> (u32, u32) {
    let (major, minor) = contract::NATIVE_API
        .strip_prefix('v')
        .and_then(|value| value.split_once('.'))
        .expect("native contract must use v<major>.<minor>");
    let major: u32 = major
        .parse()
        .expect("native contract major must be an integer");
    let minor: u32 = minor
        .parse()
        .expect("native contract minor must be an integer");
    assert!(major > 0, "native contract major must be positive");
    assert_eq!(
        contract::NATIVE_API,
        format!("v{major}.{minor}"),
        "native contract must use canonical notation"
    );
    (major, minor)
}

/// Derive descriptor constants and checked layout versions from the native contract.
fn main() {
    let source = std::path::Path::new(file!())
        .parent()
        .unwrap()
        .join("src/contract.rs");
    println!("cargo:rerun-if-changed={}", source.display());
    let (major, minor) = contract_numbers();
    // abi_stable's derives embed CARGO_PKG_VERSION in every ItemInfo, including
    // generated trait objects. Override that compilation metadata here so all
    // layouts use the contract generation. Cargo's package version remains the
    // workspace release version in manifests, registries, and cargo metadata.
    for (name, value) in [
        ("CARGO_PKG_VERSION", format!("{major}.{minor}.0")),
        ("CARGO_PKG_VERSION_MAJOR", major.to_string()),
        ("CARGO_PKG_VERSION_MINOR", minor.to_string()),
        ("CARGO_PKG_VERSION_PATCH", "0".into()),
        ("CARGO_PKG_VERSION_PRE", String::new()),
        ("CARGO_PKG_VERSION_BUILD", String::new()),
    ] {
        println!("cargo:rustc-env={name}={value}");
    }
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(output.join("native_api.rs"), format!(
        "/// Contract emitted by the SDK; authors cannot lower it independently.\npub const NATIVE_API_VERSION: NativeApiVersion = NativeApiVersion::new({major}, {minor});\n"
    )).unwrap();
}
