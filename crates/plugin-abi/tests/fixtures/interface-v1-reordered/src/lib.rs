// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen native plugin contracts, independent of Relay's runtime and release train.
//!
//! Every boundary type must implement `StableAbi`. Tokio objects, ordinary Rust
//! futures, runtime traits, and serde_json values must remain behind adapters.

#![allow(non_camel_case_types, non_local_definitions)]

pub mod v1_0;
pub mod diagnostics;
mod value;
mod version;

pub use value::{Field, Json};
pub use version::{NativeApiVersion, ParseNativeApiVersionError, NATIVE_API, NATIVE_API_VERSION};
pub use abi_stable;
pub use async_ffi;

use abi_stable::{library::{LibHeader, RootModule}, sabi_types::{RRef, VersionStrings}, std_types::{ROption, RStr, RString}, StableAbi};

/// Metadata available without constructing a plugin component or executor.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct PluginDescriptor {
    /// Native SDK contract required by this artifact.
    pub native_api: NativeApiVersion,
    /// Plugin kind, checked against the authored manifest.
    pub plugin_kind: RString,
}

/// Frozen bootstrap for native ABI major 1. Never add fields to this type.
#[repr(C)]
#[derive(StableAbi)]
#[sabi(kind(Prefix(prefix_ref = Bootstrap_Ref)))]
pub struct Bootstrap {
    /// Pure metadata; must not construct plugin state.
    pub descriptor: extern "C" fn() -> PluginDescriptor,
    /// Discover an immutable, layout-checked module by its exact versioned ID.
    #[sabi(last_prefix_field)]
    pub query_module: extern "C" fn(RStr<'_>) -> ROption<RRef<'static, LibHeader>>,
}

impl RootModule for Bootstrap_Ref {
    abi_stable::declare_root_module_statics! { Bootstrap_Ref }
    const BASE_NAME: &'static str = "nemo_relay_plugin";
    const NAME: &'static str = "nemo.relay.native.bootstrap.v1";
    // The bootstrap version is frozen independently of this package's minor.
    const VERSION_STRINGS: VersionStrings = VersionStrings::new("1.0.0");
}

/// Validate SDK age and manifest/binary agreement before invoking a factory.
pub fn validate_descriptor(
    descriptor: &PluginDescriptor,
    manifest_api: NativeApiVersion,
    manifest_kind: &str,
    host_api: NativeApiVersion,
) -> Result<(), String> {
    if descriptor.native_api != manifest_api {
        return Err(format!("manifest native API {manifest_api} disagrees with binary {}", descriptor.native_api));
    }
    if descriptor.plugin_kind.as_str() != manifest_kind {
        return Err(format!("manifest kind '{manifest_kind}' disagrees with binary '{}'", descriptor.plugin_kind));
    }
    if !host_api.supports(manifest_api) {
        return Err(format!("plugin requires native API {manifest_api}; host supports {host_api}"));
    }
    Ok(())
}
