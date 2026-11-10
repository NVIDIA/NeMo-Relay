// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked, versioned xabi contracts for native plugins.

pub mod bootstrap;
pub mod borrow;
mod contract;
pub mod diagnostics;
pub mod future;
pub mod handles;
pub mod transfer;
pub mod v1_0;
mod value;
mod values;
mod version;
pub mod wire;
pub use contract::NATIVE_API;
pub use value::{Field, Json};
pub use version::{NATIVE_API_VERSION, NativeApiVersion, ParseNativeApiVersionError};
pub use xabi;

/// Metadata checked before constructing plugin state.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Minimum native contract required by this SDK.
    pub native_api: NativeApiVersion,
    /// Plugin identity checked against its manifest.
    pub plugin_kind: String,
}
/// Validate SDK age and manifest/binary agreement before invoking a factory.
pub fn validate_descriptor(
    descriptor: &PluginDescriptor,
    manifest_api: NativeApiVersion,
    manifest_kind: &str,
    host_api: NativeApiVersion,
) -> Result<(), String> {
    if descriptor.native_api != manifest_api {
        return Err(format!(
            "manifest native API {manifest_api} disagrees with binary {}",
            descriptor.native_api
        ));
    }
    if descriptor.plugin_kind.as_str() != manifest_kind {
        return Err(format!(
            "manifest kind '{manifest_kind}' disagrees with binary '{}'",
            descriptor.plugin_kind
        ));
    }
    if !host_api.supports(manifest_api) {
        return Err(format!(
            "plugin requires native API {manifest_api}; host supports {host_api}"
        ));
    }
    Ok(())
}

/// Export immutable checked metadata and the standard xabi manifest.
#[macro_export]
macro_rules! export_factory {
    (kind = $kind:literal, create = $factory:expr) => {
        mod __nemo_relay_checked_export {
            use super::*;
            use $crate::{v1_0, xabi};
            fn create() -> v1_0::Result<v1_0::PluginBox> {
                ($factory)()
            }
            unsafe extern "C" fn make() -> *mut std::ffi::c_void {
                std::panic::catch_unwind(|| $crate::handles::export_factory(create))
                    .unwrap_or(std::ptr::null_mut())
            }
            static EXPORTS: [xabi::XabiExport; 1] = [xabi::XabiExport::new(
                xabi::XabiStr::from_static($crate::wire::XabiV1AbiTraitWireFactory::ID),
                xabi::XabiStr::from_static(v1_0::FACTORY_MODULE),
                1,
                1,
                0,
                make,
            )];
            static MANIFEST: xabi::XabiManifest = xabi::XabiManifest::new(&EXPORTS);
            #[unsafe(no_mangle)]
            pub extern "C" fn xabi_manifest() -> *const xabi::XabiManifest {
                &MANIFEST
            }
            #[unsafe(no_mangle)]
            pub extern "C" fn nemo_relay_native_bootstrap() -> *const $crate::bootstrap::Bootstrap {
                static BOOTSTRAP: std::sync::OnceLock<$crate::bootstrap::Bootstrap> =
                    std::sync::OnceLock::new();
                BOOTSTRAP.get_or_init(|| $crate::bootstrap::Bootstrap {
                    size: std::mem::size_of::<$crate::bootstrap::Bootstrap>(),
                    generation: 1,
                    major: $crate::NATIVE_API_VERSION.major,
                    minor: $crate::NATIVE_API_VERSION.minor,
                    kind: xabi::XabiStr::from_static($kind),
                    schema: xabi::XabiStr::from_static($crate::bootstrap::sdk_schema()),
                    manifest: &MANIFEST,
                })
            }
        }
    };
}
