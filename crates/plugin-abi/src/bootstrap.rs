// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen bootstrap metadata, checked before invoking an xabi constructor.
use crate::{NativeApiVersion, PluginDescriptor};

/// Null-terminated symbol for the frozen bootstrap entry point.
pub const SYMBOL: &[u8] = b"nemo_relay_native_bootstrap\0";
/// This record is frozen for ABI major 1. New contracts use separate schemas.
#[repr(C)]
pub struct Bootstrap {
    /// Size of this bootstrap record in bytes.
    pub size: usize,
    /// Bootstrap layout generation, independent of native contract versions.
    pub generation: u32,
    /// Required native ABI major.
    pub major: u32,
    /// Required native ABI minor.
    pub minor: u32,
    /// Static plugin kind matched against its manifest.
    pub kind: xabi::XabiStr,
    /// Static complete schema for the declared native contract.
    pub schema: xabi::XabiStr,
    /// Static xabi export manifest owned by the resident library.
    pub manifest: *const xabi::XabiManifest,
}
unsafe impl Send for Bootstrap {}
unsafe impl Sync for Bootstrap {}

/// Complete checked schema, independent of the package release number.
pub fn schema() -> &'static str {
    static SCHEMA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        let mut result = String::from(include_str!(concat!(env!("OUT_DIR"), "/native_schema.txt")));
        result.push_str(&format!(
            "\npointer:{};endian:{};xabi-generation:{};",
            std::mem::size_of::<usize>(),
            cfg!(target_endian = "little"),
            xabi::ABI_VERSION
        ));
        let append_layouts = include!(concat!(env!("OUT_DIR"), "/native_layout.rs"));
        append_layouts(&mut result);
        result
    })
}

/// Select an immutable schema for the plugin's exact SDK contract.
pub fn schema_for(api: NativeApiVersion) -> Result<&'static str, String> {
    match (api.major, api.minor) {
        (1, 0) => Ok(schema()),
        _ => Err(format!(
            "plugin requires native API {api}; host supports {}",
            crate::NATIVE_API_VERSION
        )),
    }
}
/// The export macro emits the build SDK's contract without an author override.
pub fn sdk_schema() -> &'static str {
    schema_for(crate::NATIVE_API_VERSION).expect("SDK contract must have an interface")
}

/// Check frozen metadata without constructing a factory or component.
///
/// # Safety
/// The pointer must refer to a static bootstrap in a trusted resident library.
pub unsafe fn check(
    raw: *const Bootstrap,
) -> Result<(PluginDescriptor, &'static xabi::XabiManifest), String> {
    let (descriptor, bootstrap) = unsafe { inspect(raw) }?;
    let manifest = unsafe { check_contract(bootstrap) }?;
    Ok((descriptor, manifest))
}

/// Inspect the frozen descriptor before comparing manifest declarations.
///
/// # Safety
/// The pointer must refer to a static bootstrap in a trusted resident library.
pub unsafe fn inspect(
    raw: *const Bootstrap,
) -> Result<(PluginDescriptor, &'static Bootstrap), String> {
    if raw.is_null() {
        return Err("native bootstrap is null".into());
    }
    if unsafe { raw.cast::<usize>().read() } != std::mem::size_of::<Bootstrap>() {
        return Err("incompatible native bootstrap size".into());
    }
    let bootstrap = unsafe { &*raw };
    if bootstrap.generation != 1 {
        return Err("unsupported native bootstrap generation".into());
    }
    let descriptor = PluginDescriptor {
        native_api: NativeApiVersion::new(bootstrap.major, bootstrap.minor),
        plugin_kind: unsafe { bootstrap.kind.as_str() }
            .map_err(|e| e.to_string())?
            .into(),
    };
    Ok((descriptor, bootstrap))
}

/// Validate the exact contract before any factory export is invoked.
///
/// # Safety
/// All pointers in the bootstrap must belong to a trusted resident library.
pub unsafe fn check_contract(bootstrap: &Bootstrap) -> Result<&'static xabi::XabiManifest, String> {
    let api = NativeApiVersion::new(bootstrap.major, bootstrap.minor);
    let actual = unsafe { bootstrap.schema.as_str() }.map_err(|e| e.to_string())?;
    if actual != schema_for(api)? {
        return Err("incompatible native contract: type layouts, signatures, ownership, or dependency ABI differ".into());
    }
    let manifest = unsafe { bootstrap.manifest.as_ref() }.ok_or("native xabi manifest is null")?;
    manifest.validate().map_err(|e| e.to_string())?;
    Ok(manifest)
}

/// Select the factory after checking every required export.
///
/// # Safety
/// The manifest and all exports must remain resident for the process lifetime.
pub unsafe fn factory(manifest: &xabi::XabiManifest) -> Result<crate::handles::FactoryBox, String> {
    let exports = unsafe { manifest.exports.as_slice() }.map_err(|e| e.to_string())?;
    let mut selected = None;
    for export in exports {
        export.validate().map_err(|e| e.to_string())?;
        let id = unsafe { export.abi_id.as_str() }.map_err(|e| e.to_string())?;
        if id == crate::wire::XabiV1AbiTraitWireFactory::ID {
            if selected.is_some() {
                return Err("duplicate native factory module".into());
            }
            if export.contract_version != 1 {
                return Err("unsupported native factory version".into());
            }
            selected = Some(export);
        }
    }
    let export = selected.ok_or("native plugin is missing required factory module")?;
    unsafe { crate::handles::load_factory(export) }.map_err(|e| e.to_string())
}
