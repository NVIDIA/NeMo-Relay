// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native dynamic plugin loader and host-side ABI adapter.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;
use std::ffi::c_void;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};

use futures_util::FutureExt;

use crate::api::event::{Event, EventSanitizeFields};
use crate::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use crate::api::runtime::{
    EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity, LlmConditionalFn, LlmExecutionFn,
    LlmExecutionNextFn, LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestContext,
    LlmSanitizeRequestFn, LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, MiddlewareContinuationContext, ToolConditionalFn, ToolExecutionFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::runtime::{
    ScopeStackHandle, ThreadScopeStackBinding, capture_thread_scope_stack, create_scope_stack,
    restore_thread_scope_stack, scope_stack_active, set_thread_scope_stack, with_scope_stack,
};
use crate::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeHandle, ScopeType,
};
use crate::api::scope::{event as emit_scope_mark, get_handle, pop_scope, push_scope};
use crate::api::tool::ToolExecutionInterceptOutcome;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result as FlowResult};
use crate::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistrationContext,
    deregister_plugin_registration_checked, register_plugin_tracked,
};
use chrono::{DateTime, Utc};
use libloading::{Library, Symbol};
use nemo_relay_plugin::{
    NEMO_RELAY_NATIVE_ABI_VERSION, NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY,
    NemoRelayNativeAsyncCallbackState, NemoRelayNativeAsyncCompletion,
    NemoRelayNativeAsyncMiddlewareCb, NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncNextResultCb, NemoRelayNativeAsyncNextStreamCb, NemoRelayNativeAsyncStream,
    NemoRelayNativeAsyncStreamMiddlewareCb, NemoRelayNativeEventSanitizeCb,
    NemoRelayNativeEventSubscriberCb, NemoRelayNativeFreeFn, NemoRelayNativeHostApiV1,
    NemoRelayNativeHostApiV3, NemoRelayNativeLlmCodecKind, NemoRelayNativeLlmConditionalCb,
    NemoRelayNativeLlmExecutionCb, NemoRelayNativeLlmRequestCodec,
    NemoRelayNativeLlmRequestInterceptCb, NemoRelayNativeLlmResponseCodec,
    NemoRelayNativeLlmSanitizeRequestCb, NemoRelayNativeLlmSanitizeRequestContext,
    NemoRelayNativeLlmSanitizeResponseCb, NemoRelayNativeLlmSanitizeResponseContext,
    NemoRelayNativeLlmStreamExecutionCb, NemoRelayNativeLlmStreamV1, NemoRelayNativePluginContext,
    NemoRelayNativePluginEntry, NemoRelayNativePluginV1, NemoRelayNativeScopeHandle,
    NemoRelayNativeScopeStack, NemoRelayNativeScopeStackBinding, NemoRelayNativeScopeType,
    NemoRelayNativeString, NemoRelayNativeToolConditionalCb, NemoRelayNativeToolExecutionCb,
    NemoRelayNativeToolJsonCb, NemoRelayNativeWithScopeStackCb, NemoRelayStatus,
};
use semver::{Version, VersionReq};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use tokio_stream::{Stream, StreamExt};

use super::{
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginManifestLoad,
    DynamicPluginTeardownOutcome, deregister_tracked_registrations_checked,
    validate_annotated_request_consumer_compatibility,
};

/// Native plugin load request derived from host dynamic-plugin state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePluginLoadSpec {
    /// Expected plugin kind.
    pub plugin_id: String,
    /// Path to the authored `relay-plugin.toml`.
    pub manifest_ref: String,
}

/// Owns native dynamic libraries registered into the plugin registry.
///
/// Dropping this value deregisters the native plugin kinds before unloading
/// their libraries. Clear active plugin configuration before dropping it so
/// runtime callbacks cannot outlive their code.
pub struct NativePluginActivation {
    plugins: Vec<Arc<NativePluginInstance>>,
    plugin_registrations: Vec<(String, u64)>,
}

impl NativePluginActivation {
    /// Returns `true` when no native plugins were loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Consumes the activation and deregisters loaded plugin kinds.
    pub fn clear(self) {}

    pub(crate) fn deregister_plugin_kinds_checked(&mut self) -> DynamicPluginTeardownOutcome {
        deregister_tracked_registrations_checked(&mut self.plugin_registrations, "native")
    }

    #[cfg(test)]
    pub(super) fn with_plugin_kind_for_test(plugin_kind: impl Into<String>) -> Self {
        Self {
            plugins: Vec::new(),
            plugin_registrations: vec![(plugin_kind.into(), 0)],
        }
    }
}

impl Drop for NativePluginActivation {
    fn drop(&mut self) {
        for (plugin_kind, registration_id) in self.plugin_registrations.iter().rev() {
            let _ = deregister_plugin_registration_checked(plugin_kind, *registration_id);
        }
    }
}

/// Loads native dynamic plugins and registers their plugin kinds.
///
/// The returned activation must be kept alive until after active plugin
/// configuration has been cleared.
pub fn load_native_plugins<I>(specs: I) -> crate::plugin::Result<NativePluginActivation>
where
    I: IntoIterator<Item = NativePluginLoadSpec>,
{
    let mut activation = NativePluginActivation {
        plugins: Vec::new(),
        plugin_registrations: Vec::new(),
    };
    for spec in specs {
        let instance = load_one_native_plugin(&spec)?;
        let plugin_kind = instance.plugin_kind.clone();
        let registration_id = register_plugin_tracked(Arc::new(NativePluginAdapter {
            plugin_kind: plugin_kind.clone(),
            allows_multiple_components: instance.allows_multiple_components,
            instance: instance.clone(),
        }))?;
        activation.plugins.push(instance);
        activation
            .plugin_registrations
            .push((plugin_kind, registration_id));
    }
    Ok(activation)
}

struct NativePluginAdapter {
    plugin_kind: String,
    allows_multiple_components: bool,
    instance: Arc<NativePluginInstance>,
}

impl Plugin for NativePluginAdapter {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn allows_multiple_components(&self) -> bool {
        self.allows_multiple_components
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let plugin = self
            .instance
            .plugin
            .lock()
            .expect("native plugin lock poisoned");
        let Some(validate) = plugin.validate else {
            return vec![];
        };
        clear_native_last_error();
        let Some(config_json) = native_string_from_json(&Json::Object(plugin_config.clone()))
        else {
            return vec![native_error_diagnostic(
                &self.plugin_kind,
                "plugin.native_validate_failed",
                "failed to serialize plugin config",
            )];
        };
        let mut out = ptr::null_mut();
        let status = unsafe { validate(plugin.user_data, config_json, &mut out) };
        unsafe { native_string_free(config_json) };
        if status != NemoRelayStatus::Ok {
            if !out.is_null() {
                unsafe { native_string_free(out) };
            }
            let message = native_last_error_message()
                .unwrap_or_else(|| format!("native validate callback returned {status:?}"));
            return vec![native_error_diagnostic(
                &self.plugin_kind,
                "plugin.native_validate_failed",
                &message,
            )];
        }
        if out.is_null() {
            return vec![];
        }
        let diagnostics = read_native_string(out)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<ConfigDiagnostic>>(&text).ok())
            .unwrap_or_else(|| {
                vec![native_error_diagnostic(
                    &self.plugin_kind,
                    "plugin.native_validate_failed",
                    "native validate callback returned invalid diagnostics JSON",
                )]
            });
        unsafe { native_string_free(out) };
        diagnostics
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = crate::plugin::Result<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let plugin = self.instance.plugin.lock().map_err(|err| {
                PluginError::Internal(format!("native plugin lock poisoned: {err}"))
            })?;
            let register = plugin.register.ok_or_else(|| {
                PluginError::RegistrationFailed(format!(
                    "native plugin '{}' did not return a register callback",
                    self.plugin_kind
                ))
            })?;
            clear_native_last_error();
            let config_json =
                native_string_from_json(&Json::Object(plugin_config)).ok_or_else(|| {
                    PluginError::RegistrationFailed("failed to serialize plugin config".into())
                })?;
            let mut native_ctx = NativeHostPluginContext {
                ctx: ctx as *mut _,
                instance: self.instance.clone(),
            };
            let status = unsafe {
                register(
                    plugin.user_data,
                    config_json,
                    &mut native_ctx as *mut _ as *mut NemoRelayNativePluginContext,
                )
            };
            unsafe { native_string_free(config_json) };
            if status == NemoRelayStatus::Ok {
                Ok(())
            } else {
                let message = native_last_error_message()
                    .unwrap_or_else(|| format!("native register callback returned {status:?}"));
                Err(PluginError::RegistrationFailed(message))
            }
        })
    }
}

fn native_error_diagnostic(plugin_kind: &str, code: &str, message: &str) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: code.into(),
        component: Some(plugin_kind.into()),
        field: None,
        message: message.into(),
    }
}

struct NativePluginInstance {
    plugin_kind: String,
    relay_compat: String,
    allows_multiple_components: bool,
    plugin: Mutex<NemoRelayNativePluginV1>,
    _library: Library,
}

unsafe impl Send for NativePluginInstance {}
unsafe impl Sync for NativePluginInstance {}

impl Drop for NativePluginInstance {
    fn drop(&mut self) {
        if let Ok(mut plugin) = self.plugin.lock() {
            drop_native_plugin_descriptor(&mut plugin);
        }
    }
}

fn drop_native_plugin_descriptor(plugin: &mut NemoRelayNativePluginV1) {
    if let Some(drop_fn) = plugin.drop.take() {
        unsafe { drop_fn(plugin.user_data) };
        plugin.user_data = ptr::null_mut();
    }
    if !plugin.plugin_kind.is_null() {
        unsafe { native_string_free(plugin.plugin_kind) };
        plugin.plugin_kind = ptr::null_mut();
    }
}

fn load_one_native_plugin(
    spec: &NativePluginLoadSpec,
) -> crate::plugin::Result<Arc<NativePluginInstance>> {
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&spec.manifest_ref)?;
    if manifest.plugin.id.trim() != spec.plugin_id {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin manifest id '{}' does not match expected id '{}'",
            manifest.plugin.id, spec.plugin_id
        )));
    }
    if manifest.plugin.kind != DynamicPluginKind::RustDynamic {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin '{}' is kind {}; native loader only supports rust_dynamic",
            spec.plugin_id, manifest.plugin.kind
        )));
    }
    validate_relay_compatibility(manifest.compat.relay.as_deref())?;
    let relay_compat = manifest
        .compat
        .relay
        .as_deref()
        .expect("validated native manifest must declare compat.relay")
        .to_string();
    if manifest.compat.native_api.as_deref().map(str::trim) != Some("1") {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin '{}' declares unsupported compat.native_api '{}'; expected 1",
            spec.plugin_id,
            manifest.compat.native_api.as_deref().unwrap_or("")
        )));
    }
    let DynamicPluginManifestLoad::RustDynamic(load) = &manifest.load else {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin '{}' has invalid rust_dynamic load contract",
            spec.plugin_id
        )));
    };
    let manifest_path = PathBuf::from(&manifest_ref);
    let library_path = resolve_manifest_relative_path(
        &manifest_path,
        load.library
            .as_deref()
            .ok_or_else(|| PluginError::InvalidConfig("load.library is required".into()))?,
    );
    if !library_path.exists() {
        return Err(PluginError::NotFound(format!(
            "native plugin library '{}' does not exist",
            library_path.display()
        )));
    }
    if let Some(expected_digest) = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
    {
        verify_sha256(&library_path, expected_digest)?;
    }
    let symbol = load
        .symbol
        .as_deref()
        .ok_or_else(|| PluginError::InvalidConfig("load.symbol is required".into()))?;

    let library = unsafe { Library::new(&library_path) }.map_err(|err| {
        PluginError::Internal(format!(
            "failed to load native plugin library '{}': {err}",
            library_path.display()
        ))
    })?;
    let mut plugin = NemoRelayNativePluginV1::default();
    unsafe {
        let entry: Symbol<NemoRelayNativePluginEntry> =
            library.get(symbol.as_bytes()).map_err(|err| {
                PluginError::NotFound(format!(
                    "native plugin symbol '{symbol}' not found in '{}': {err}",
                    library_path.display()
                ))
            })?;
        let mut status = entry(native_host_api(), &mut plugin);
        // SDKs compiled against ABI v2 correctly reject a v3 table. Retry
        // their entry point with the frozen v2 prefix instead of making a
        // runtime upgrade a breaking change for installed native plugins.
        if status == NemoRelayStatus::InvalidArg {
            drop_native_plugin_descriptor(&mut plugin);
            status = entry(native_host_api_legacy(), &mut plugin);
        }
        if status != NemoRelayStatus::Ok {
            drop_native_plugin_descriptor(&mut plugin);
            return Err(PluginError::RegistrationFailed(format!(
                "native plugin entry symbol '{symbol}' failed: {}",
                native_last_error_message().unwrap_or_else(|| format!("{status:?}"))
            )));
        }
    }
    if let Err(err) = validate_plugin_descriptor(&spec.plugin_id, &plugin) {
        drop_native_plugin_descriptor(&mut plugin);
        return Err(err);
    }
    let plugin_kind = match read_native_string(plugin.plugin_kind) {
        Ok(plugin_kind) => plugin_kind,
        Err(err) => {
            drop_native_plugin_descriptor(&mut plugin);
            return Err(err);
        }
    };
    if plugin_kind != spec.plugin_id {
        drop_native_plugin_descriptor(&mut plugin);
        return Err(PluginError::InvalidConfig(format!(
            "native plugin returned kind '{plugin_kind}' but manifest id is '{}'",
            spec.plugin_id
        )));
    }
    Ok(Arc::new(NativePluginInstance {
        plugin_kind,
        relay_compat,
        allows_multiple_components: plugin.allows_multiple_components,
        plugin: Mutex::new(plugin),
        _library: library,
    }))
}

fn validate_relay_compatibility(relay: Option<&str>) -> crate::plugin::Result<()> {
    let relay = relay
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| PluginError::InvalidConfig("compat.relay is required".into()))?;
    let req = VersionReq::parse(relay).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid compat.relay version requirement: {err}"))
    })?;
    let version = Version::parse(env!("CARGO_PKG_VERSION"))
        .map_err(|err| PluginError::Internal(format!("failed to parse host version: {err}")))?;
    if req.matches(&version) {
        Ok(())
    } else {
        Err(PluginError::InvalidConfig(format!(
            "native plugin requires relay '{relay}' but host version is {version}"
        )))
    }
}

fn validate_plugin_descriptor(
    plugin_id: &str,
    plugin: &NemoRelayNativePluginV1,
) -> crate::plugin::Result<()> {
    if plugin.struct_size < std::mem::size_of::<NemoRelayNativePluginV1>() {
        return Err(PluginError::InvalidConfig(format!(
            "native plugin '{plugin_id}' returned incompatible plugin descriptor size {}",
            plugin.struct_size
        )));
    }
    if plugin.plugin_kind.is_null() {
        return Err(PluginError::InvalidConfig(format!(
            "native plugin '{plugin_id}' returned a null plugin_kind"
        )));
    }
    if plugin.register.is_none() {
        return Err(PluginError::InvalidConfig(format!(
            "native plugin '{plugin_id}' returned no register callback"
        )));
    }
    Ok(())
}

fn resolve_manifest_relative_path(manifest_path: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn verify_sha256(path: &Path, expected: &str) -> crate::plugin::Result<()> {
    let expected = expected
        .trim()
        .strip_prefix("sha256:")
        .unwrap_or(expected.trim());
    let bytes = std::fs::read(path).map_err(|err| {
        PluginError::Internal(format!("failed to read '{}': {err}", path.display()))
    })?;
    let actual = hex_digest(Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(PluginError::InvalidConfig(format!(
            "native plugin library '{}' sha256 mismatch",
            path.display()
        )))
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[repr(C)]
struct NativeHostPluginContext {
    ctx: *mut PluginRegistrationContext,
    instance: Arc<NativePluginInstance>,
}

struct NativeHostString(Vec<u8>);

struct NativeHostLlmRequestCodec(Arc<dyn LlmCodec>);
struct NativeHostLlmResponseCodec(Arc<dyn LlmResponseCodec>);

struct NativeHostScopeHandle(ScopeHandle);

struct NativeHostScopeStack(ScopeStackHandle);

struct NativeHostScopeStackBinding(ThreadScopeStackBinding);

thread_local! {
    static NATIVE_LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    #[cfg(test)]
    static NATIVE_STRING_LIVE_ALLOCATIONS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    #[cfg(test)]
    static NATIVE_STRING_FAIL_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}

fn set_native_last_error(message: impl Into<String>) {
    NATIVE_LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(message.into()));
}

fn clear_native_last_error() {
    NATIVE_LAST_ERROR.with(|cell| *cell.borrow_mut() = None);
}

fn native_last_error_message() -> Option<String> {
    NATIVE_LAST_ERROR.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
fn fail_native_string_allocation_after(successful_allocations: usize) {
    NATIVE_STRING_FAIL_AFTER.with(|cell| cell.set(Some(successful_allocations)));
}

#[cfg(test)]
fn native_string_live_allocations() -> usize {
    NATIVE_STRING_LIVE_ALLOCATIONS.with(|allocations| allocations.borrow().len())
}

unsafe extern "C" fn native_string_new(
    data: *const u8,
    len: usize,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out.is_null() {
        set_native_last_error("out string pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    if data.is_null() && len > 0 {
        set_native_last_error("string data pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let bytes: &[u8] = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    if let Err(err) = std::str::from_utf8(bytes) {
        set_native_last_error(format!("string data is not valid UTF-8: {err}"));
        return NemoRelayStatus::InvalidUtf8;
    }
    #[cfg(test)]
    let should_fail = NATIVE_STRING_FAIL_AFTER.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            true
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            false
        }
        None => false,
    });
    #[cfg(test)]
    if should_fail {
        set_native_last_error("injected native string allocation failure");
        return NemoRelayStatus::Internal;
    }
    let handle = Box::new(NativeHostString(bytes.to_vec()));
    unsafe { *out = Box::into_raw(handle) as *mut NemoRelayNativeString };
    #[cfg(test)]
    NATIVE_STRING_LIVE_ALLOCATIONS.with(|allocations| {
        allocations.borrow_mut().insert(unsafe { *out } as usize);
    });
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_string_data(value: *const NemoRelayNativeString) -> *const u8 {
    if value.is_null() {
        return ptr::null();
    }
    let value = unsafe { &*(value as *const NativeHostString) };
    value.0.as_ptr()
}

unsafe extern "C" fn native_string_len(value: *const NemoRelayNativeString) -> usize {
    if value.is_null() {
        return 0;
    }
    let value = unsafe { &*(value as *const NativeHostString) };
    value.0.len()
}

unsafe extern "C" fn native_string_free(value: *mut NemoRelayNativeString) {
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value as *mut NativeHostString) });
        #[cfg(test)]
        NATIVE_STRING_LIVE_ALLOCATIONS.with(|allocations| {
            allocations.borrow_mut().remove(&(value as usize));
        });
    }
}

unsafe extern "C" fn native_last_error_clear() {
    clear_native_last_error();
}

unsafe extern "C" fn native_last_error_set(message: *const NemoRelayNativeString) {
    match read_native_string(message) {
        Ok(message) => set_native_last_error(message),
        Err(err) => set_native_last_error(err.to_string()),
    }
}

unsafe extern "C" fn native_llm_request_codec_decode(
    codec: *const NemoRelayNativeLlmRequestCodec,
    request_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("request codec decode output pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    if codec.is_null() {
        set_native_last_error("request codec decode capability is null");
        return NemoRelayStatus::NullPointer;
    }
    if request_json.is_null() {
        set_native_last_error("request codec decode request is null");
        return NemoRelayStatus::NullPointer;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> std::result::Result<_, String> {
        let request: LlmRequest = serde_json::from_str(
            &read_native_string(request_json).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid request JSON: {error}"))?;
        let codec = unsafe { &*(codec as *const NativeHostLlmRequestCodec) };
        let annotated = codec
            .0
            .decode(&request)
            .map_err(|error| error.to_string())?;
        let annotated = serde_json::to_value(annotated).map_err(|error| error.to_string())?;
        native_string_from_json(&annotated)
            .ok_or_else(|| "failed to allocate decoded request".to_string())
    }));
    match result {
        Ok(Ok(value)) => {
            unsafe { *out = value };
            NemoRelayStatus::Ok
        }
        Ok(Err(error)) => {
            set_native_last_error(format!("request codec decode failed: {error}"));
            NemoRelayStatus::Internal
        }
        Err(_) => {
            set_native_last_error("request codec decode panicked");
            NemoRelayStatus::Internal
        }
    }
}

unsafe extern "C" fn native_llm_request_codec_encode(
    codec: *const NemoRelayNativeLlmRequestCodec,
    annotated_json: *const NemoRelayNativeString,
    original_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("request codec encode output pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    if codec.is_null() {
        set_native_last_error("request codec encode capability is null");
        return NemoRelayStatus::NullPointer;
    }
    if annotated_json.is_null() {
        set_native_last_error("request codec encode annotated request is null");
        return NemoRelayStatus::NullPointer;
    }
    if original_json.is_null() {
        set_native_last_error("request codec encode original request is null");
        return NemoRelayStatus::NullPointer;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> std::result::Result<_, String> {
        let annotated: AnnotatedLlmRequest = serde_json::from_str(
            &read_native_string(annotated_json).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid annotated request JSON: {error}"))?;
        let original: LlmRequest = serde_json::from_str(
            &read_native_string(original_json).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid original request JSON: {error}"))?;
        let codec = unsafe { &*(codec as *const NativeHostLlmRequestCodec) };
        let request = codec
            .0
            .encode(&annotated, &original)
            .map_err(|error| error.to_string())?;
        let request = serde_json::to_value(request).map_err(|error| error.to_string())?;
        native_string_from_json(&request)
            .ok_or_else(|| "failed to allocate encoded request".to_string())
    }));
    match result {
        Ok(Ok(value)) => {
            unsafe { *out = value };
            NemoRelayStatus::Ok
        }
        Ok(Err(error)) => {
            set_native_last_error(format!("request codec encode failed: {error}"));
            NemoRelayStatus::Internal
        }
        Err(_) => {
            set_native_last_error("request codec encode panicked");
            NemoRelayStatus::Internal
        }
    }
}

unsafe extern "C" fn native_llm_response_codec_decode(
    codec: *const NemoRelayNativeLlmResponseCodec,
    response_json: *const NemoRelayNativeString,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("response codec decode output pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    if codec.is_null() {
        set_native_last_error("response codec decode capability is null");
        return NemoRelayStatus::NullPointer;
    }
    if response_json.is_null() {
        set_native_last_error("response codec decode response is null");
        return NemoRelayStatus::NullPointer;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> std::result::Result<_, String> {
        let response: Json = serde_json::from_str(
            &read_native_string(response_json).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid response JSON: {error}"))?;
        let codec = unsafe { &*(codec as *const NativeHostLlmResponseCodec) };
        let annotated = codec
            .0
            .decode_response(&response)
            .map_err(|error| error.to_string())?;
        let annotated = serde_json::to_value(annotated).map_err(|error| error.to_string())?;
        native_string_from_json(&annotated)
            .ok_or_else(|| "failed to allocate decoded response".to_string())
    }));
    match result {
        Ok(Ok(value)) => {
            unsafe { *out = value };
            NemoRelayStatus::Ok
        }
        Ok(Err(error)) => {
            set_native_last_error(format!("response codec decode failed: {error}"));
            NemoRelayStatus::Internal
        }
        Err(_) => {
            set_native_last_error("response codec decode panicked");
            NemoRelayStatus::Internal
        }
    }
}

fn native_host_api() -> *const NemoRelayNativeHostApiV1 {
    static HOST_API: OnceLock<NemoRelayNativeHostApiV3> = OnceLock::new();
    &HOST_API.get_or_init(build_native_host_api_v3).v1 as *const NemoRelayNativeHostApiV1
}

fn native_host_api_legacy() -> *const NemoRelayNativeHostApiV1 {
    static HOST_API: OnceLock<NemoRelayNativeHostApiV1> = OnceLock::new();
    HOST_API.get_or_init(build_native_host_api_legacy) as *const _
}

fn build_native_host_api_legacy() -> NemoRelayNativeHostApiV1 {
    static RELAY_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    NemoRelayNativeHostApiV1 {
        abi_version: NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY,
        struct_size: std::mem::size_of::<NemoRelayNativeHostApiV1>(),
        relay_version: RELAY_VERSION.as_ptr().cast(),
        string_new: native_string_new,
        string_data: native_string_data,
        string_len: native_string_len,
        string_free: native_string_free,
        last_error_clear: native_last_error_clear,
        last_error_set: native_last_error_set,
        llm_request_codec_decode: native_llm_request_codec_decode,
        llm_request_codec_encode: native_llm_request_codec_encode,
        llm_response_codec_decode: native_llm_response_codec_decode,
        plugin_context_register_subscriber: native_plugin_context_register_subscriber,
        plugin_context_register_tool_sanitize_request_guardrail:
            native_plugin_context_register_tool_sanitize_request_guardrail,
        plugin_context_register_tool_sanitize_response_guardrail:
            native_plugin_context_register_tool_sanitize_response_guardrail,
        plugin_context_register_tool_conditional_execution_guardrail:
            native_plugin_context_register_tool_conditional_execution_guardrail,
        plugin_context_register_tool_request_intercept:
            native_plugin_context_register_tool_request_intercept,
        plugin_context_register_tool_execution_intercept:
            native_plugin_context_register_tool_execution_intercept,
        plugin_context_register_llm_sanitize_request_guardrail:
            native_plugin_context_register_llm_sanitize_request_guardrail,
        plugin_context_register_llm_sanitize_response_guardrail:
            native_plugin_context_register_llm_sanitize_response_guardrail,
        plugin_context_register_llm_conditional_execution_guardrail:
            native_plugin_context_register_llm_conditional_execution_guardrail,
        plugin_context_register_llm_request_intercept:
            native_plugin_context_register_llm_request_intercept,
        plugin_context_register_llm_execution_intercept:
            native_plugin_context_register_llm_execution_intercept,
        plugin_context_register_llm_stream_execution_intercept:
            native_plugin_context_register_llm_stream_execution_intercept,
        scope_handle_free: native_scope_handle_free,
        scope_get_current: native_scope_get_current,
        scope_push: native_scope_push,
        scope_pop: native_scope_pop,
        emit_mark: native_emit_mark,
        scope_stack_create: native_scope_stack_create,
        scope_stack_free: native_scope_stack_free,
        scope_stack_set_thread: native_scope_stack_set_thread,
        scope_stack_capture_thread: native_scope_stack_capture_thread,
        scope_stack_restore_thread: native_scope_stack_restore_thread,
        scope_stack_binding_free: native_scope_stack_binding_free,
        scope_stack_active: native_scope_stack_active,
        scope_stack_with_current: native_scope_stack_with_current,
        plugin_context_register_mark_sanitize_guardrail:
            native_plugin_context_register_mark_sanitize_guardrail,
        plugin_context_register_scope_sanitize_start_guardrail:
            native_plugin_context_register_scope_sanitize_start_guardrail,
        plugin_context_register_scope_sanitize_end_guardrail:
            native_plugin_context_register_scope_sanitize_end_guardrail,
    }
}

fn build_native_host_api_v3() -> NemoRelayNativeHostApiV3 {
    let mut v1 = build_native_host_api_legacy();
    v1.abi_version = NEMO_RELAY_NATIVE_ABI_VERSION;
    v1.struct_size = std::mem::size_of::<NemoRelayNativeHostApiV3>();
    NemoRelayNativeHostApiV3 {
        v1,
        async_completion_resolve_json: native_async_completion_resolve_json,
        async_completion_reject: native_async_completion_reject,
        async_completion_is_cancelled: native_async_completion_is_cancelled,
        async_completion_release: native_async_completion_release,
        async_next_invoke: native_async_next_invoke,
        async_next_release: native_async_next_release,
        plugin_context_register_async_middleware: native_plugin_context_register_async_middleware,
        async_stream_push_json: native_async_stream_push_json,
        async_stream_finish: native_async_stream_finish,
        async_stream_reject: native_async_stream_reject,
        async_stream_is_cancelled: native_async_stream_is_cancelled,
        async_stream_release: native_async_stream_release,
        async_next_invoke_stream: native_async_next_invoke_stream,
        plugin_context_register_async_stream_middleware:
            native_plugin_context_register_async_stream_middleware,
        async_next_invoke_result: native_async_next_invoke_result,
    }
}

fn read_native_string(value: *const NemoRelayNativeString) -> crate::plugin::Result<String> {
    if value.is_null() {
        return Ok(String::new());
    }
    let value = unsafe { &*(value as *const NativeHostString) };
    std::str::from_utf8(&value.0)
        .map(str::to_owned)
        .map_err(|err| {
            PluginError::InvalidConfig(format!("native string is not valid UTF-8: {err}"))
        })
}

fn native_string_from_str(value: &str) -> Option<*mut NemoRelayNativeString> {
    let mut out = ptr::null_mut();
    let status = unsafe { native_string_new(value.as_ptr(), value.len(), &mut out) };
    (status == NemoRelayStatus::Ok).then_some(out)
}

fn native_string_from_json(value: &Json) -> Option<*mut NemoRelayNativeString> {
    serde_json::to_string(value)
        .ok()
        .and_then(|value| native_string_from_str(&value))
}

fn json_from_native_string(value: *mut NemoRelayNativeString, fallback: &str) -> FlowResult<Json> {
    if value.is_null() {
        return Err(FlowError::Internal(
            native_last_error_message().unwrap_or_else(|| fallback.into()),
        ));
    }
    let text = read_native_string(value).map_err(|err| FlowError::Internal(err.to_string()))?;
    serde_json::from_str(&text).map_err(|err| FlowError::Internal(format!("invalid JSON: {err}")))
}

fn take_native_string(value: *mut NemoRelayNativeString) -> FlowResult<String> {
    let result = read_native_string(value).map_err(|err| FlowError::Internal(err.to_string()));
    unsafe { native_string_free(value) };
    result
}

fn take_json_from_native_string(
    value: *mut NemoRelayNativeString,
    fallback: &str,
) -> FlowResult<Json> {
    let result = json_from_native_string(value, fallback);
    unsafe { native_string_free(value) };
    result
}

unsafe fn free_native_sanitizer_strings(
    input: *mut NemoRelayNativeString,
    codec_id: Option<*mut NemoRelayNativeString>,
    output: *mut NemoRelayNativeString,
) {
    unsafe { native_string_free(input) };
    if let Some(codec_id) = codec_id
        && codec_id != input
    {
        unsafe { native_string_free(codec_id) };
    }
    if !output.is_null() && output != input && Some(output) != codec_id {
        unsafe { native_string_free(output) };
    }
}

fn optional_json_from_native_string(
    value: *const NemoRelayNativeString,
    field: &str,
) -> Result<Option<Json>, NemoRelayStatus> {
    if value.is_null() {
        return Ok(None);
    }
    let text = read_native_string(value).map_err(|err| {
        set_native_last_error(err.to_string());
        NemoRelayStatus::InvalidUtf8
    })?;
    serde_json::from_str(&text).map(Some).map_err(|err| {
        set_native_last_error(format!("{field} is not valid JSON: {err}"));
        NemoRelayStatus::InvalidJson
    })
}

fn optional_timestamp_from_native(
    timestamp_unix_micros: *const i64,
) -> Result<Option<DateTime<Utc>>, NemoRelayStatus> {
    if timestamp_unix_micros.is_null() {
        return Ok(None);
    }
    DateTime::<Utc>::from_timestamp_micros(unsafe { ptr::read(timestamp_unix_micros) })
        .map(Some)
        .ok_or_else(|| {
            set_native_last_error("timestamp unix microseconds are outside supported range");
            NemoRelayStatus::InvalidArg
        })
}

fn native_scope_type_to_core(scope_type: NemoRelayNativeScopeType) -> ScopeType {
    match scope_type {
        NemoRelayNativeScopeType::Agent => ScopeType::Agent,
        NemoRelayNativeScopeType::Function => ScopeType::Function,
        NemoRelayNativeScopeType::Tool => ScopeType::Tool,
        NemoRelayNativeScopeType::Llm => ScopeType::Llm,
        NemoRelayNativeScopeType::Retriever => ScopeType::Retriever,
        NemoRelayNativeScopeType::Embedder => ScopeType::Embedder,
        NemoRelayNativeScopeType::Reranker => ScopeType::Reranker,
        NemoRelayNativeScopeType::Guardrail => ScopeType::Guardrail,
        NemoRelayNativeScopeType::Evaluator => ScopeType::Evaluator,
        NemoRelayNativeScopeType::Custom => ScopeType::Custom,
        NemoRelayNativeScopeType::Unknown => ScopeType::Unknown,
    }
}

fn native_scope_ref<'a>(handle: *const NemoRelayNativeScopeHandle) -> Option<&'a ScopeHandle> {
    if handle.is_null() {
        return None;
    }
    Some(&unsafe { &*(handle as *const NativeHostScopeHandle) }.0)
}

unsafe extern "C" fn native_scope_handle_free(handle: *mut NemoRelayNativeScopeHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle as *mut NativeHostScopeHandle) });
    }
}

unsafe extern "C" fn native_scope_get_current(
    out: *mut *mut NemoRelayNativeScopeHandle,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("out scope handle pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    match get_handle() {
        Ok(handle) => {
            unsafe { *out = Box::into_raw(Box::new(NativeHostScopeHandle(handle))).cast() };
            NemoRelayStatus::Ok
        }
        Err(err) => status_from_flow_error(err),
    }
}

unsafe extern "C" fn native_scope_push(
    name: *const NemoRelayNativeString,
    scope_type: NemoRelayNativeScopeType,
    parent: *const NemoRelayNativeScopeHandle,
    attributes: u32,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    input_json: *const NemoRelayNativeString,
    timestamp_unix_micros: *const i64,
    out: *mut *mut NemoRelayNativeScopeHandle,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("out scope handle pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let data = match optional_json_from_native_string(data_json, "scope data") {
        Ok(data) => data,
        Err(status) => return status,
    };
    let metadata = match optional_json_from_native_string(metadata_json, "scope metadata") {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    let input = match optional_json_from_native_string(input_json, "scope input") {
        Ok(input) => input,
        Err(status) => return status,
    };
    let timestamp = match optional_timestamp_from_native(timestamp_unix_micros) {
        Ok(timestamp) => timestamp,
        Err(status) => return status,
    };
    let parent_ref = native_scope_ref(parent);
    match push_scope(
        PushScopeParams::builder()
            .name(&name)
            .scope_type(native_scope_type_to_core(scope_type))
            .parent_opt(parent_ref)
            .attributes(ScopeAttributes::from_bits_truncate(attributes))
            .data_opt(data)
            .metadata_opt(metadata)
            .input_opt(input)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(handle) => {
            unsafe { *out = Box::into_raw(Box::new(NativeHostScopeHandle(handle))).cast() };
            NemoRelayStatus::Ok
        }
        Err(err) => status_from_flow_error(err),
    }
}

unsafe extern "C" fn native_scope_pop(
    handle: *const NemoRelayNativeScopeHandle,
    output_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_native_last_error();
    if handle.is_null() {
        set_native_last_error("scope handle is null");
        return NemoRelayStatus::NullPointer;
    }
    let output = match optional_json_from_native_string(output_json, "scope output") {
        Ok(output) => output,
        Err(status) => return status,
    };
    let metadata = match optional_json_from_native_string(metadata_json, "scope metadata") {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    let timestamp = match optional_timestamp_from_native(timestamp_unix_micros) {
        Ok(timestamp) => timestamp,
        Err(status) => return status,
    };
    let handle = unsafe { &*(handle as *const NativeHostScopeHandle) };
    match pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.0.uuid)
            .output_opt(output)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_flow_error(err),
    }
}

unsafe extern "C" fn native_emit_mark(
    name: *const NemoRelayNativeString,
    parent: *const NemoRelayNativeScopeHandle,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_native_last_error();
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let data = match optional_json_from_native_string(data_json, "mark data") {
        Ok(data) => data,
        Err(status) => return status,
    };
    let metadata = match optional_json_from_native_string(metadata_json, "mark metadata") {
        Ok(metadata) => metadata,
        Err(status) => return status,
    };
    let timestamp = match optional_timestamp_from_native(timestamp_unix_micros) {
        Ok(timestamp) => timestamp,
        Err(status) => return status,
    };
    let parent_ref = native_scope_ref(parent);
    match emit_scope_mark(
        EmitMarkEventParams::builder()
            .name(&name)
            .parent_opt(parent_ref)
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_flow_error(err),
    }
}

unsafe extern "C" fn native_scope_stack_create(
    out: *mut *mut NemoRelayNativeScopeStack,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("out scope stack pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = Box::into_raw(Box::new(NativeHostScopeStack(create_scope_stack()))).cast();
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_scope_stack_free(stack: *mut NemoRelayNativeScopeStack) {
    if !stack.is_null() {
        drop(unsafe { Box::from_raw(stack as *mut NativeHostScopeStack) });
    }
}

unsafe extern "C" fn native_scope_stack_set_thread(
    stack: *const NemoRelayNativeScopeStack,
) -> NemoRelayStatus {
    clear_native_last_error();
    if stack.is_null() {
        set_native_last_error("scope stack is null");
        return NemoRelayStatus::NullPointer;
    }
    let stack = unsafe { &*(stack as *const NativeHostScopeStack) };
    set_thread_scope_stack(stack.0.clone());
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_scope_stack_capture_thread(
    out: *mut *mut NemoRelayNativeScopeStackBinding,
) -> NemoRelayStatus {
    clear_native_last_error();
    if out.is_null() {
        set_native_last_error("out scope stack binding pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = Box::into_raw(Box::new(NativeHostScopeStackBinding(
            capture_thread_scope_stack(),
        )))
        .cast();
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_scope_stack_restore_thread(
    binding: *mut NemoRelayNativeScopeStackBinding,
) -> NemoRelayStatus {
    clear_native_last_error();
    if binding.is_null() {
        set_native_last_error("scope stack binding is null");
        return NemoRelayStatus::NullPointer;
    }
    let binding = unsafe { Box::from_raw(binding as *mut NativeHostScopeStackBinding) };
    restore_thread_scope_stack(binding.0);
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_scope_stack_binding_free(
    binding: *mut NemoRelayNativeScopeStackBinding,
) {
    if !binding.is_null() {
        drop(unsafe { Box::from_raw(binding as *mut NativeHostScopeStackBinding) });
    }
}

unsafe extern "C" fn native_scope_stack_active() -> bool {
    scope_stack_active()
}

unsafe extern "C" fn native_scope_stack_with_current(
    stack: *const NemoRelayNativeScopeStack,
    cb: NemoRelayNativeWithScopeStackCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    clear_native_last_error();
    if stack.is_null() {
        set_native_last_error("scope stack is null");
        return NemoRelayStatus::NullPointer;
    }
    let stack = unsafe { &*(stack as *const NativeHostScopeStack) };
    let status = with_scope_stack(stack.0.clone(), || unsafe { cb(user_data) });
    if status != NemoRelayStatus::Ok && native_last_error_message().is_none() {
        set_native_last_error(format!("native scope-stack callback returned {status:?}"));
    }
    status
}

fn flow_error_from_status(status: NemoRelayStatus, fallback: &str) -> FlowError {
    let message = native_last_error_message().unwrap_or_else(|| format!("{fallback}: {status:?}"));
    match status {
        NemoRelayStatus::AlreadyExists => FlowError::AlreadyExists(message),
        NemoRelayStatus::NotFound => FlowError::NotFound(message),
        NemoRelayStatus::ScopeStackEmpty => FlowError::ScopeStackEmpty,
        NemoRelayStatus::GuardrailRejected => FlowError::GuardrailRejected(message),
        NemoRelayStatus::InvalidArg => FlowError::InvalidArgument(message),
        _ => FlowError::Internal(message),
    }
}

fn status_from_plugin_error(err: PluginError) -> NemoRelayStatus {
    set_native_last_error(err.to_string());
    match err {
        PluginError::NotFound(_) => NemoRelayStatus::NotFound,
        PluginError::Conflict(_) => NemoRelayStatus::AlreadyExists,
        PluginError::InvalidConfig(_) | PluginError::Serialization(_) => {
            NemoRelayStatus::InvalidArg
        }
        PluginError::Internal(_) | PluginError::RegistrationFailed(_) => NemoRelayStatus::Internal,
    }
}

fn status_from_flow_error(err: FlowError) -> NemoRelayStatus {
    set_native_last_error(err.to_string());
    match err {
        FlowError::AlreadyExists(_) => NemoRelayStatus::AlreadyExists,
        FlowError::NotFound(_) => NemoRelayStatus::NotFound,
        FlowError::InvalidArgument(_) => NemoRelayStatus::InvalidArg,
        FlowError::ScopeStackEmpty => NemoRelayStatus::ScopeStackEmpty,
        FlowError::GuardrailRejected(_) => NemoRelayStatus::GuardrailRejected,
        FlowError::Upstream(_) | FlowError::Internal(_) => NemoRelayStatus::Internal,
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload")
}

fn native_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("native plugin runtime should build")
    })
}

fn spawn_with_continuation_context<C, F, T>(
    context: MiddlewareContinuationContext,
    callback: C,
) -> std::thread::JoinHandle<T>
where
    C: FnOnce() -> F + Send + 'static,
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let binding = capture_thread_scope_stack();
    std::thread::spawn(move || {
        restore_thread_scope_stack(binding);
        native_runtime().block_on(context.invoke(callback))
    })
}

struct NativeCallbackUserData {
    ptr: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    _instance: Option<Arc<NativePluginInstance>>,
}

struct NativeCallbackUserDataGuard {
    ptr: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    armed: bool,
}

impl NativeCallbackUserDataGuard {
    fn new(ptr: *mut c_void, free_fn: NemoRelayNativeFreeFn) -> Self {
        Self {
            ptr,
            free_fn,
            armed: true,
        }
    }

    fn transfer(mut self) -> (*mut c_void, NemoRelayNativeFreeFn) {
        self.armed = false;
        (self.ptr, self.free_fn)
    }
}

impl Drop for NativeCallbackUserDataGuard {
    fn drop(&mut self) {
        if self.armed
            && let Some(free_fn) = self.free_fn
        {
            unsafe { free_fn(self.ptr) };
        }
    }
}

unsafe impl Send for NativeCallbackUserData {}
unsafe impl Sync for NativeCallbackUserData {}

impl Drop for NativeCallbackUserData {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.ptr) };
        }
    }
}

fn make_user_data(
    instance: Arc<NativePluginInstance>,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> Arc<NativeCallbackUserData> {
    Arc::new(NativeCallbackUserData {
        ptr: user_data,
        free_fn,
        _instance: Some(instance),
    })
}

const NATIVE_ASYNC_STREAM_CHANNEL_CAPACITY: usize = 64;

struct NativeAsyncCompletion {
    sender: Mutex<Option<tokio::sync::oneshot::Sender<FlowResult<Json>>>>,
    cancelled: AtomicBool,
    next_invoked: AtomicBool,
    next_abort: Mutex<Option<tokio::task::AbortHandle>>,
    #[cfg(test)]
    before_settlement_lock: Option<Arc<std::sync::Barrier>>,
    // A pending native callback can continue running after its completion
    // wakes the awaiting task. Keep the callback's dynamic-library instance
    // alive until native code explicitly releases this handle.
    _callback_user_data: Option<Arc<NativeCallbackUserData>>,
}

struct NativeAsyncWait {
    completion: Arc<NativeAsyncCompletion>,
    receiver: tokio::sync::oneshot::Receiver<FlowResult<Json>>,
    completed: bool,
}

impl NativeAsyncWait {
    async fn receive(&mut self) -> FlowResult<Json> {
        let result = (&mut self.receiver).await.map_err(|_| {
            FlowError::Internal("native async callback dropped without settling".into())
        })?;
        self.completed = true;
        result
    }
}

impl Drop for NativeAsyncWait {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut next_abort = self
            .completion
            .next_abort
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.completion.cancelled.store(true, Ordering::Release);
        if let Some(abort) = next_abort.take() {
            abort.abort();
        }
    }
}

enum NativeAsyncNextInner {
    Tool(ToolExecutionNextFn),
    Llm(LlmExecutionNextFn),
    LlmStream(LlmStreamExecutionNextFn),
}

struct NativeAsyncNext {
    inner: NativeAsyncNextInner,
    runtime: tokio::runtime::Handle,
    context: MiddlewareContinuationContext,
    // The native callback owns this handle independently of its completion.
    // Retaining the library here prevents an unload while it still uses `next`.
    _callback_user_data: Option<Arc<NativeCallbackUserData>>,
}

impl NativeAsyncNext {
    fn new(
        inner: NativeAsyncNextInner,
        runtime: tokio::runtime::Handle,
        callback_user_data: Option<Arc<NativeCallbackUserData>>,
    ) -> Self {
        Self {
            inner,
            runtime,
            context: MiddlewareContinuationContext::capture(),
            _callback_user_data: callback_user_data,
        }
    }
}

struct NativeAsyncStream {
    sender: Mutex<Option<tokio::sync::mpsc::Sender<FlowResult<Json>>>>,
    cancelled: AtomicBool,
    settled: AtomicBool,
    downstream_aborts: Mutex<HashMap<tokio::task::Id, tokio::task::AbortHandle>>,
    settlement: Mutex<()>,
    #[cfg(test)]
    before_settlement_lock: Option<Arc<std::sync::Barrier>>,
    _callback_user_data: Option<Arc<NativeCallbackUserData>>,
}

struct NativeAsyncStreamReceiver {
    receiver: tokio::sync::mpsc::Receiver<FlowResult<Json>>,
    stream: Arc<NativeAsyncStream>,
}

struct NativeAsyncStreamCallbackGuard {
    cb: NemoRelayNativeAsyncNextStreamCb,
    user_data: usize,
    stream: Arc<NativeAsyncStream>,
    _library_guard: Option<Arc<NativeCallbackUserData>>,
    active: bool,
}

impl NativeAsyncStreamCallbackGuard {
    fn finish(&mut self) {
        self.active = false;
    }

    fn fail(&mut self, error: &str) {
        if !self.active {
            return;
        }
        // Cancellation owns terminal delivery. Leave the guard active so its
        // Drop implementation can notify the plugin and release callback data.
        if self.stream.cancelled.load(Ordering::Acquire) {
            return;
        }
        if let Some(message) = native_string_from_str(error) {
            unsafe {
                let _ = (self.cb)(self.user_data as *mut c_void, ptr::null(), message, false);
                native_string_free(message);
            }
            self.active = false;
        }
    }
}

impl Drop for NativeAsyncStreamCallbackGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if self.stream.cancelled.load(Ordering::Acquire) {
            if let Some(message) =
                native_string_from_str("native async stream continuation was cancelled")
            {
                unsafe {
                    let _ = (self.cb)(self.user_data as *mut c_void, ptr::null(), message, false);
                    native_string_free(message);
                }
            }
        } else if self.stream.settled.load(Ordering::Acquire) {
            if let Some(message) =
                native_string_from_str("native async stream continuation output settled")
            {
                unsafe {
                    let _ = (self.cb)(self.user_data as *mut c_void, ptr::null(), message, false);
                    native_string_free(message);
                }
            }
        } else {
            unsafe {
                let _ = (self.cb)(
                    self.user_data as *mut c_void,
                    ptr::null(),
                    ptr::null(),
                    true,
                );
            }
        }
    }
}

impl Stream for NativeAsyncStreamReceiver {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

impl Drop for NativeAsyncStreamReceiver {
    fn drop(&mut self) {
        let _settlement = self
            .stream
            .settlement
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut downstream_aborts = self
            .stream
            .downstream_aborts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.stream.cancelled.store(true, Ordering::Release);
        for (_, abort) in downstream_aborts.drain() {
            abort.abort();
        }
        drop(downstream_aborts);
        self.stream
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }
}

async fn invoke_native_async_callback(
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: Arc<NativeCallbackUserData>,
    invocation: Json,
    next: Option<NativeAsyncNextInner>,
) -> FlowResult<Json> {
    let runtime = if next.is_some() {
        Some(tokio::runtime::Handle::try_current().map_err(|error| {
            FlowError::Internal(format!(
                "native async intercept requires a Tokio runtime: {error}"
            ))
        })?)
    } else {
        None
    };
    let invocation = native_string_from_json(&invocation)
        .ok_or_else(|| FlowError::Internal("failed to allocate native async invocation".into()))?
        as usize;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let completion = Arc::new(NativeAsyncCompletion {
        sender: Mutex::new(Some(sender)),
        cancelled: AtomicBool::new(false),
        next_invoked: AtomicBool::new(false),
        next_abort: Mutex::new(None),
        #[cfg(test)]
        before_settlement_lock: None,
        _callback_user_data: Some(user_data.clone()),
    });
    let mut wait = NativeAsyncWait {
        completion: Arc::clone(&completion),
        receiver,
        completed: false,
    };
    let completion_ref = Arc::into_raw(completion.clone()) as usize;
    let next_ref = match (next, runtime) {
        (Some(inner), Some(runtime)) => Some(Arc::into_raw(Arc::new(NativeAsyncNext::new(
            inner,
            runtime,
            Some(user_data.clone()),
        ))) as usize),
        (None, None) => None,
        _ => unreachable!("runtime is present exactly for native async intercepts"),
    };
    let state = match catch_unwind(AssertUnwindSafe(|| unsafe {
        cb(
            user_data.ptr,
            invocation as *const NemoRelayNativeString,
            next_ref
                .map(|next| next as *const NemoRelayNativeAsyncNext)
                .unwrap_or(ptr::null()),
            completion_ref as *const NemoRelayNativeAsyncCompletion,
        )
    })) {
        Ok(state) => state,
        Err(_) => {
            unsafe {
                drop(Arc::from_raw(
                    completion_ref as *const NativeAsyncCompletion,
                ));
                native_string_free(invocation as *mut NemoRelayNativeString);
            }
            return Err(FlowError::Internal("native async callback panicked".into()));
        }
    };
    unsafe { native_string_free(invocation as *mut NemoRelayNativeString) };
    let state = match NemoRelayNativeAsyncCallbackState::try_from(state) {
        Ok(state) => state,
        Err(()) => {
            unsafe {
                drop(Arc::from_raw(
                    completion_ref as *const NativeAsyncCompletion,
                ));
            }
            return Err(FlowError::Internal(
                "native async callback returned an invalid state".into(),
            ));
        }
    };
    if state == NemoRelayNativeAsyncCallbackState::Complete {
        unsafe {
            drop(Arc::from_raw(
                completion_ref as *const NativeAsyncCompletion,
            ));
        }
        if completion
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
        {
            return Err(FlowError::Internal(
                "native async callback returned Complete without settling".into(),
            ));
        }
    }
    wait.receive().await
}

unsafe extern "C" fn native_async_completion_resolve_json(
    completion: *const NemoRelayNativeAsyncCompletion,
    value_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let Some(completion) = (unsafe { (completion as *const NativeAsyncCompletion).as_ref() })
    else {
        return NemoRelayStatus::NullPointer;
    };
    let value = match parse_json_arg(value_json, "native async completion result") {
        Ok(value) => value,
        Err(status) => return status,
    };
    #[cfg(test)]
    if let Some(barrier) = &completion.before_settlement_lock {
        barrier.wait();
    }
    let mut next_abort = completion
        .next_abort
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if let Some(abort) = next_abort.take() {
        abort.abort();
    }
    let Some(sender) = completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
    else {
        return NemoRelayStatus::InvalidArg;
    };
    let _ = sender.send(Ok(value));
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_async_completion_reject(
    completion: *const NemoRelayNativeAsyncCompletion,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    let Some(completion) = (unsafe { (completion as *const NativeAsyncCompletion).as_ref() })
    else {
        return NemoRelayStatus::NullPointer;
    };
    let message = if message.is_null() {
        "native async callback rejected".to_string()
    } else {
        match read_native_string(message) {
            Ok(message) => message,
            Err(error) => {
                set_native_last_error(error.to_string());
                return NemoRelayStatus::InvalidArg;
            }
        }
    };
    #[cfg(test)]
    if let Some(barrier) = &completion.before_settlement_lock {
        barrier.wait();
    }
    let mut next_abort = completion
        .next_abort
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if let Some(abort) = next_abort.take() {
        abort.abort();
    }
    let Some(sender) = completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
    else {
        return NemoRelayStatus::InvalidArg;
    };
    let _ = sender.send(Err(FlowError::Internal(message)));
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_async_completion_is_cancelled(
    completion: *const NemoRelayNativeAsyncCompletion,
) -> bool {
    unsafe { (completion as *const NativeAsyncCompletion).as_ref() }
        .is_none_or(|completion| completion.cancelled.load(Ordering::Acquire))
}

unsafe extern "C" fn native_async_completion_release(
    completion: *const NemoRelayNativeAsyncCompletion,
) {
    if !completion.is_null() {
        unsafe { drop(Arc::from_raw(completion as *const NativeAsyncCompletion)) };
    }
}

unsafe extern "C" fn native_async_next_release(next: *const NemoRelayNativeAsyncNext) {
    if !next.is_null() {
        unsafe { drop(Arc::from_raw(next as *const NativeAsyncNext)) };
    }
}

unsafe extern "C" fn native_async_stream_push_json(
    stream: *const NemoRelayNativeAsyncStream,
    chunk_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    clear_native_last_error();
    let Some(stream) = (unsafe { (stream as *const NativeAsyncStream).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if stream.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let chunk = match parse_json_arg(chunk_json, "native async stream chunk") {
        Ok(chunk) => chunk,
        Err(status) => return status,
    };
    #[cfg(test)]
    if let Some(barrier) = &stream.before_settlement_lock {
        barrier.wait();
    }
    let _settlement = stream
        .settlement
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if stream.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if stream.settled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let sender = stream
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    let Some(sender) = sender else {
        return NemoRelayStatus::InvalidArg;
    };
    match sender.try_send(Ok(chunk)) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            set_native_last_error(
                "native async stream is backpressured; retry the chunk after the consumer advances",
            );
            NemoRelayStatus::Internal
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => NemoRelayStatus::InvalidArg,
    }
}

unsafe extern "C" fn native_async_stream_finish(
    stream: *const NemoRelayNativeAsyncStream,
) -> NemoRelayStatus {
    let Some(stream) = (unsafe { (stream as *const NativeAsyncStream).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    #[cfg(test)]
    if let Some(barrier) = &stream.before_settlement_lock {
        barrier.wait();
    }
    let _settlement = stream
        .settlement
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if stream.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if stream.settled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if stream
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
        .is_some()
    {
        stream.settled.store(true, Ordering::Release);
        let mut downstream_aborts = stream
            .downstream_aborts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for (_, abort) in downstream_aborts.drain() {
            abort.abort();
        }
        NemoRelayStatus::Ok
    } else {
        NemoRelayStatus::InvalidArg
    }
}

unsafe extern "C" fn native_async_stream_reject(
    stream: *const NemoRelayNativeAsyncStream,
    message: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    clear_native_last_error();
    let Some(stream) = (unsafe { (stream as *const NativeAsyncStream).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    let message =
        read_native_string(message).unwrap_or_else(|_| "native async stream rejected".to_string());
    #[cfg(test)]
    if let Some(barrier) = &stream.before_settlement_lock {
        barrier.wait();
    }
    let _settlement = stream
        .settlement
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if stream.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    if stream.settled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let mut sender_guard = stream
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let Some(sender) = sender_guard.as_ref() else {
        return NemoRelayStatus::InvalidArg;
    };
    match sender.try_send(Err(FlowError::Internal(message))) {
        Ok(()) => {
            sender_guard.take();
            stream.settled.store(true, Ordering::Release);
            let mut downstream_aborts = stream
                .downstream_aborts
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            for (_, abort) in downstream_aborts.drain() {
                abort.abort();
            }
            NemoRelayStatus::Ok
        }
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            set_native_last_error(
                "native async stream is backpressured; retry rejection after the consumer advances",
            );
            NemoRelayStatus::Internal
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => NemoRelayStatus::InvalidArg,
    }
}

unsafe extern "C" fn native_async_stream_is_cancelled(
    stream: *const NemoRelayNativeAsyncStream,
) -> bool {
    unsafe { (stream as *const NativeAsyncStream).as_ref() }
        .is_none_or(|stream| stream.cancelled.load(Ordering::Acquire))
}

unsafe extern "C" fn native_async_stream_release(stream: *const NemoRelayNativeAsyncStream) {
    if !stream.is_null() {
        unsafe { drop(Arc::from_raw(stream as *const NativeAsyncStream)) };
    }
}

/// Invokes the runtime continuation without blocking the calling native thread.
unsafe extern "C" fn native_async_next_invoke(
    next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if completion.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let invocation = match parse_json_arg(invocation_json, "native async next invocation") {
        Ok(value) => value,
        Err(status) => return status,
    };
    if matches!(&next.inner, NativeAsyncNextInner::LlmStream(_)) {
        set_native_last_error(
            "stream continuations require async_next_invoke_stream; completion-based next cannot buffer a stream",
        );
        return NemoRelayStatus::InvalidArg;
    }
    enum Invocation {
        Tool(Json),
        Llm(LlmRequest),
    }
    let invocation = match &next.inner {
        NativeAsyncNextInner::Tool(_) => Invocation::Tool(invocation),
        NativeAsyncNextInner::Llm(_) => match serde_json::from_value(invocation) {
            Ok(request) => Invocation::Llm(request),
            Err(error) => {
                set_native_last_error(error.to_string());
                return NemoRelayStatus::InvalidJson;
            }
        },
        NativeAsyncNextInner::LlmStream(_) => unreachable!("stream continuations were rejected"),
    };
    unsafe { Arc::increment_strong_count(completion as *const NativeAsyncCompletion) };
    let completion = unsafe { Arc::from_raw(completion as *const NativeAsyncCompletion) };
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let future: Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> =
        match (&next.inner, invocation) {
            (NativeAsyncNextInner::Tool(next), Invocation::Tool(invocation)) => {
                let next = next.clone();
                Box::pin(async move {
                    serde_json::to_value(ToolExecutionInterceptOutcome::new(
                        next(invocation).await?,
                    ))
                    .map_err(|error| {
                        FlowError::Internal(format!(
                            "failed to serialize native async tool outcome: {error}"
                        ))
                    })
                })
            }
            (NativeAsyncNextInner::Llm(next), Invocation::Llm(request)) => {
                let next = next.clone();
                Box::pin(async move { next(request).await })
            }
            _ => unreachable!("native next invocation kind matched its continuation"),
        };
    let continuation_context = match next.context.isolated_for_current_invocation() {
        Ok(context) => context,
        Err(error) => return status_from_flow_error(error),
    };
    let mut abort_guard = completion
        .next_abort
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if completion.cancelled.load(Ordering::Acquire) {
        return NemoRelayStatus::InvalidArg;
    }
    let unsettled = completion
        .sender
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_some();
    if !unsettled || completion.next_invoked.swap(true, Ordering::AcqRel) {
        set_native_last_error("native async next was already invoked for this completion");
        return NemoRelayStatus::InvalidArg;
    }
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();
    let completion_for_task = Arc::clone(&completion);
    let task = next.runtime.spawn(async move {
        if start_rx.await.is_err() {
            return;
        }
        let result = AssertUnwindSafe(continuation_context.run(future))
            .catch_unwind()
            .await;
        let result = result.unwrap_or_else(|payload| {
            Err(FlowError::Internal(format!(
                "native async next continuation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        completion_for_task
            .next_abort
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(sender) = completion_for_task
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = sender.send(result);
        }
    });
    let abort = task.abort_handle();
    *abort_guard = Some(abort.clone());
    if completion.cancelled.load(Ordering::Acquire) {
        abort_guard.take();
        abort.abort();
        return NemoRelayStatus::InvalidArg;
    }
    let _ = start_tx.send(());
    NemoRelayStatus::Ok
}

/// Invokes a unary continuation with an independent per-call result callback.
unsafe extern "C" fn native_async_next_invoke_result(
    next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncNextResultCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    let invocation = match parse_json_arg(invocation_json, "native async next invocation") {
        Ok(value) => value,
        Err(status) => return status,
    };
    let future: Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> = match &next.inner {
        NativeAsyncNextInner::Tool(next_fn) => {
            let next_fn = next_fn.clone();
            Box::pin(async move { next_fn(invocation).await })
        }
        NativeAsyncNextInner::Llm(next_fn) => {
            let request = match serde_json::from_value(invocation) {
                Ok(request) => request,
                Err(error) => {
                    set_native_last_error(error.to_string());
                    return NemoRelayStatus::InvalidJson;
                }
            };
            let next_fn = next_fn.clone();
            Box::pin(async move { next_fn(request).await })
        }
        NativeAsyncNextInner::LlmStream(_) => {
            set_native_last_error(
                "stream continuations require async_next_invoke_stream; unary result callbacks cannot buffer a stream",
            );
            return NemoRelayStatus::InvalidArg;
        }
    };
    let continuation_context = match next.context.isolated_for_current_invocation() {
        Ok(context) => context,
        Err(error) => return status_from_flow_error(error),
    };
    let user_data = user_data as usize;
    let _library_guard = next._callback_user_data.clone();
    next.runtime.spawn(async move {
        let result = AssertUnwindSafe(continuation_context.run(future))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                Err(FlowError::Internal(format!(
                    "native async next continuation panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            });
        match result {
            Ok(value) => {
                if let Some(value) = native_string_from_json(&value) {
                    unsafe {
                        cb(user_data as *mut c_void, value, ptr::null());
                        native_string_free(value);
                    }
                } else if let Some(error) =
                    native_string_from_str("failed to allocate native async next result")
                {
                    unsafe {
                        cb(user_data as *mut c_void, ptr::null(), error);
                        native_string_free(error);
                    }
                }
            }
            Err(error) => {
                if let Some(error) = native_string_from_str(&error.to_string()) {
                    unsafe {
                        cb(user_data as *mut c_void, ptr::null(), error);
                        native_string_free(error);
                    }
                }
            }
        }
        drop(_library_guard);
    });
    NemoRelayStatus::Ok
}

unsafe extern "C" fn native_async_next_invoke_stream(
    next: *const NemoRelayNativeAsyncNext,
    invocation_json: *const NemoRelayNativeString,
    output_stream: *const NemoRelayNativeAsyncStream,
    cb: NemoRelayNativeAsyncNextStreamCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    if output_stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { Arc::increment_strong_count(output_stream as *const NativeAsyncStream) };
    let output_stream = unsafe { Arc::from_raw(output_stream as *const NativeAsyncStream) };
    let NativeAsyncNextInner::LlmStream(next_fn) = &next.inner else {
        return NemoRelayStatus::InvalidArg;
    };
    let request = match parse_json_arg(invocation_json, "native async stream next invocation")
        .and_then(|value| {
            serde_json::from_value(value).map_err(|error| {
                set_native_last_error(error.to_string());
                NemoRelayStatus::InvalidJson
            })
        }) {
        Ok(request) => request,
        Err(status) => return status,
    };
    let continuation_context = match next.context.isolated_for_current_invocation() {
        Ok(context) => context,
        Err(error) => return status_from_flow_error(error),
    };
    let _settlement = output_stream
        .settlement
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if output_stream.cancelled.load(Ordering::Acquire)
        || output_stream.settled.load(Ordering::Acquire)
        || output_stream
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_none()
    {
        return NemoRelayStatus::InvalidArg;
    }
    let mut downstream_aborts = output_stream
        .downstream_aborts
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let next_fn = next_fn.clone();
    let user_data = user_data as usize;
    let output_stream_for_task = Arc::clone(&output_stream);
    let output_stream_for_cleanup = Arc::clone(&output_stream);
    let callback_guard = NativeAsyncStreamCallbackGuard {
        cb,
        user_data,
        stream: output_stream_for_task,
        _library_guard: next._callback_user_data.clone(),
        active: true,
    };
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();
    let task = next.runtime.spawn(async move {
        if start_rx.await.is_err() {
            return;
        }
        continuation_context
            .run(deliver_native_async_next_stream(
                next_fn,
                request,
                callback_guard,
            ))
            .await;
        output_stream_for_cleanup
            .downstream_aborts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&tokio::task::id());
    });
    let abort = task.abort_handle();
    downstream_aborts.insert(task.id(), abort);
    let _ = start_tx.send(());
    NemoRelayStatus::Ok
}

async fn deliver_native_async_next_stream(
    next_fn: LlmStreamExecutionNextFn,
    request: LlmRequest,
    mut callback_guard: NativeAsyncStreamCallbackGuard,
) {
    let result = AssertUnwindSafe(async {
        match next_fn(request).await {
            Ok(stream) => forward_native_async_next_stream(stream, &mut callback_guard).await,
            Err(error) => callback_guard.fail(&error.to_string()),
        }
    })
    .catch_unwind()
    .await;
    if let Err(payload) = result {
        callback_guard.fail(&format!(
            "native async stream continuation panicked: {}",
            panic_payload_message(payload.as_ref())
        ));
    }
}

async fn forward_native_async_next_stream(
    stream: LlmJsonStream,
    callback_guard: &mut NativeAsyncStreamCallbackGuard,
) {
    forward_native_async_next_stream_with(stream, callback_guard, native_string_from_json).await;
}

async fn forward_native_async_next_stream_with(
    mut stream: LlmJsonStream,
    callback_guard: &mut NativeAsyncStreamCallbackGuard,
    to_native_string: impl Fn(&Json) -> Option<*mut NemoRelayNativeString>,
) {
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                let Some(chunk) = to_native_string(&chunk) else {
                    callback_guard.fail(
                        "failed to serialize or allocate native async stream continuation chunk",
                    );
                    return;
                };
                let keep_going = unsafe {
                    (callback_guard.cb)(
                        callback_guard.user_data as *mut c_void,
                        chunk,
                        ptr::null(),
                        false,
                    )
                };
                unsafe { native_string_free(chunk) };
                if !keep_going {
                    callback_guard.finish();
                    return;
                }
            }
            Err(error) => {
                callback_guard.fail(&error.to_string());
                return;
            }
        }
    }
    unsafe {
        let _ = (callback_guard.cb)(
            callback_guard.user_data as *mut c_void,
            ptr::null(),
            ptr::null(),
            true,
        );
    }
    callback_guard.finish();
}

fn wrap_native_async_tool_json(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolSanitizeFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, value| {
        let user_data = user_data.clone();
        Box::pin(async move {
            let value = invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"name": name, "value": value}),
                None,
            )
            .await?;
            Ok(value)
        })
    })
}

fn wrap_native_async_tool_conditional(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolConditionalFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, value| {
        let user_data = user_data.clone();
        Box::pin(async move {
            match invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"name": name, "value": value}),
                None,
            )
            .await?
            {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => Err(FlowError::Internal(format!(
                    "native async tool conditional callback returned {other}; expected string or null"
                ))),
            }
        })
    })
}

fn wrap_native_async_llm_conditional(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmConditionalFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request| {
        let user_data = user_data.clone();
        Box::pin(async move {
            match invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"request": request}),
                None,
            )
            .await?
            {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => Err(FlowError::Internal(format!(
                    "native async LLM conditional callback returned {other}; expected string or null"
                ))),
            }
        })
    })
}

fn wrap_native_async_llm_sanitize_request(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeRequestFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request, context| {
        let user_data = user_data.clone();
        let codec = native_async_codec_identity(context.codec());
        Box::pin(async move {
            let value = invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"request": request, "context": codec}),
                None,
            )
            .await?;
            if value.is_null() {
                Ok(None)
            } else {
                serde_json::from_value(value)
                    .map(Some)
                    .map_err(|error| FlowError::Internal(error.to_string()))
            }
        })
    })
}

fn wrap_native_async_llm_sanitize_response(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeResponseFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |response, context| {
        let user_data = user_data.clone();
        let codec = native_async_codec_identity(context.codec());
        Box::pin(async move {
            let value = invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"response": response, "context": codec}),
                None,
            )
            .await?;
            Ok((!value.is_null()).then_some(value))
        })
    })
}

fn native_async_codec_identity(identity: &LlmCodecIdentity) -> Json {
    match identity {
        LlmCodecIdentity::None => {
            serde_json::json!({"codec_kind": "none", "codec_id": Json::Null})
        }
        LlmCodecIdentity::BuiltIn(codec) => {
            serde_json::json!({"codec_kind": "builtin", "codec_id": codec.id()})
        }
        LlmCodecIdentity::Runtime(id) => {
            serde_json::json!({"codec_kind": "runtime", "codec_id": id})
        }
        LlmCodecIdentity::Opaque => {
            serde_json::json!({"codec_kind": "opaque", "codec_id": Json::Null})
        }
    }
}

fn wrap_native_async_llm_request_intercept(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmRequestInterceptFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, annotated| {
        let user_data = user_data.clone();
        Box::pin(async move {
            serde_json::from_value(
                invoke_native_async_callback(
                    cb,
                    user_data,
                    serde_json::json!({
                        "name": name,
                        "request": request,
                        "annotated": annotated,
                    }),
                    None,
                )
                .await?,
            )
            .map_err(|error| {
                FlowError::Internal(format!(
                    "invalid native async LLM intercept outcome: {error}"
                ))
            })
        })
    })
}

fn wrap_native_async_event_sanitize(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> EventSanitizeFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |event, fields| {
        let user_data = user_data.clone();
        Box::pin(async move {
            serde_json::from_value(
                invoke_native_async_callback(
                    cb,
                    user_data,
                    serde_json::json!({"event": event, "fields": fields}),
                    None,
                )
                .await?,
            )
            .map_err(|error| {
                FlowError::Internal(format!("invalid native async event fields: {error}"))
            })
        })
    })
}

fn wrap_native_async_tool_execution(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, args, next| {
        let user_data = user_data.clone();
        let invocation = serde_json::json!({"name": name, "value": args});
        Box::pin(async move {
            serde_json::from_value(
                invoke_native_async_callback(
                    cb,
                    user_data,
                    invocation,
                    Some(NativeAsyncNextInner::Tool(next)),
                )
                .await?,
            )
            .map_err(|error| {
                FlowError::Internal(format!("invalid native async tool outcome: {error}"))
            })
        })
    })
}

fn wrap_native_async_llm_execution(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, next| {
        let user_data = user_data.clone();
        let name = name.to_owned();
        Box::pin(async move {
            invoke_native_async_callback(
                cb,
                user_data,
                serde_json::json!({"name": name, "request": request}),
                Some(NativeAsyncNextInner::Llm(next)),
            )
            .await
        })
    })
}

fn wrap_native_incremental_llm_stream_execution(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmStreamExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    wrap_native_incremental_llm_stream_execution_with_user_data(cb, user_data)
}

fn wrap_native_incremental_llm_stream_execution_with_user_data(
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: Arc<NativeCallbackUserData>,
) -> LlmStreamExecutionFn {
    Arc::new(move |name, request, next| {
        let user_data = user_data.clone();
        let name = name.to_owned();
        Box::pin(async move {
            let (sender, receiver) =
                tokio::sync::mpsc::channel(NATIVE_ASYNC_STREAM_CHANNEL_CAPACITY);
            let stream = Arc::new(NativeAsyncStream {
                sender: Mutex::new(Some(sender)),
                cancelled: AtomicBool::new(false),
                settled: AtomicBool::new(false),
                downstream_aborts: Mutex::new(HashMap::new()),
                settlement: Mutex::new(()),
                #[cfg(test)]
                before_settlement_lock: None,
                _callback_user_data: Some(user_data.clone()),
            });
            let output = NativeAsyncStreamReceiver {
                receiver,
                stream: Arc::clone(&stream),
            };
            let state = {
                let invocation =
                    native_string_from_json(&serde_json::json!({"name": name, "request": request}))
                        .ok_or_else(|| {
                            FlowError::Internal(
                                "failed to allocate native async stream invocation".into(),
                            )
                        })?;
                let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
                    FlowError::Internal(format!(
                        "native async stream intercept requires a Tokio runtime: {error}"
                    ))
                })?;
                let next_ref = Arc::into_raw(Arc::new(NativeAsyncNext::new(
                    NativeAsyncNextInner::LlmStream(next),
                    runtime,
                    Some(user_data.clone()),
                )));
                let stream_ref = Arc::into_raw(stream.clone());
                let state = catch_unwind(AssertUnwindSafe(|| unsafe {
                    cb(
                        user_data.ptr,
                        invocation,
                        next_ref as *const NemoRelayNativeAsyncNext,
                        stream_ref as *const NemoRelayNativeAsyncStream,
                    )
                }));
                unsafe { native_string_free(invocation) };
                state
                    .ok()
                    .and_then(|state| NemoRelayNativeAsyncCallbackState::try_from(state).ok())
                    .ok_or_else(|| {
                        FlowError::Internal(
                            "native async stream callback panicked or returned an invalid state"
                                .into(),
                        )
                    })?
            };
            if state == NemoRelayNativeAsyncCallbackState::Complete
                && stream
                    .sender
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_some()
            {
                return Err(FlowError::Internal(
                    "native async stream callback returned Complete without finishing".into(),
                ));
            }
            Ok(LlmJsonStream::new(output))
        })
    })
}

unsafe extern "C" fn native_plugin_context_register_async_stream_middleware(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeAsyncStreamMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let user_data_guard = NativeCallbackUserDataGuard::new(user_data, free_fn);
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let (user_data, free_fn) = user_data_guard.transfer();
    let context = unsafe { &mut *host_ctx.ctx };
    match context.register_llm_stream_execution_intercept(
        &name,
        priority,
        wrap_native_incremental_llm_stream_execution(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_plugin_error(error),
    }
}

unsafe extern "C" fn native_plugin_context_register_async_middleware(
    ctx: *mut NemoRelayNativePluginContext,
    kind: u32,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeAsyncMiddlewareCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    // The host owns callback user data as soon as registration is attempted,
    // including malformed and incompatible registrations.
    let user_data_guard = NativeCallbackUserDataGuard::new(user_data, free_fn);
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let kind = match NemoRelayNativeAsyncMiddlewareKind::try_from(kind) {
        Ok(kind) => kind,
        Err(()) => {
            set_native_last_error("invalid native async middleware kind");
            return NemoRelayStatus::InvalidArg;
        }
    };
    if kind == NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept {
        set_native_last_error(
            "completion-based LLM stream middleware is unsupported; use plugin_context_register_async_stream_middleware",
        );
        return NemoRelayStatus::InvalidArg;
    }
    if kind == NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept
        && let Err(error) = validate_annotated_request_consumer_compatibility(
            &instance.relay_compat,
            &instance.plugin_kind,
        )
    {
        return status_from_plugin_error(error);
    }
    let (user_data, free_fn) = user_data_guard.transfer();
    let context = unsafe { &mut *host_ctx.ctx };
    let registration = match kind {
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest => context
            .register_tool_sanitize_request_guardrail(
                &name,
                priority,
                wrap_native_async_tool_json(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse => context
            .register_tool_sanitize_response_guardrail(
                &name,
                priority,
                wrap_native_async_tool_json(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution => context
            .register_tool_conditional_execution_guardrail(
                &name,
                priority,
                wrap_native_async_tool_conditional(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept => context
            .register_tool_request_intercept(
                &name,
                priority,
                break_chain,
                wrap_native_async_tool_json(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept => context
            .register_tool_execution_intercept(
                &name,
                priority,
                wrap_native_async_tool_execution(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest => context
            .register_llm_sanitize_request_guardrail(
                &name,
                priority,
                wrap_native_async_llm_sanitize_request(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse => context
            .register_llm_sanitize_response_guardrail(
                &name,
                priority,
                wrap_native_async_llm_sanitize_response(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution => context
            .register_llm_conditional_execution_guardrail(
                &name,
                priority,
                wrap_native_async_llm_conditional(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept => context
            .register_llm_request_intercept(
                &name,
                priority,
                break_chain,
                wrap_native_async_llm_request_intercept(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept => context
            .register_llm_execution_intercept(
                &name,
                priority,
                wrap_native_async_llm_execution(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept => {
            unreachable!("completion-based stream middleware was rejected before registration")
        }
        NemoRelayNativeAsyncMiddlewareKind::MarkSanitize => context
            .register_mark_sanitize_guardrail(
                &name,
                priority,
                wrap_native_async_event_sanitize(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart => context
            .register_scope_sanitize_start_guardrail(
                &name,
                priority,
                wrap_native_async_event_sanitize(instance, cb, user_data, free_fn),
            ),
        NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd => context
            .register_scope_sanitize_end_guardrail(
                &name,
                priority,
                wrap_native_async_event_sanitize(instance, cb, user_data, free_fn),
            ),
    };
    match registration {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_plugin_error(error),
    }
}

fn host_ctx_mut<'a>(
    ctx: *mut NemoRelayNativePluginContext,
) -> Result<&'a mut NativeHostPluginContext, NemoRelayStatus> {
    if ctx.is_null() {
        set_native_last_error("plugin context is null");
        return Err(NemoRelayStatus::NullPointer);
    }
    let ctx = unsafe { &mut *(ctx as *mut NativeHostPluginContext) };
    if ctx.ctx.is_null() {
        set_native_last_error("plugin context inner pointer is null");
        return Err(NemoRelayStatus::NullPointer);
    }
    Ok(ctx)
}

fn read_name(name: *const NemoRelayNativeString) -> Result<String, NemoRelayStatus> {
    read_native_string(name).map_err(|err| {
        set_native_last_error(err.to_string());
        NemoRelayStatus::InvalidUtf8
    })
}

unsafe extern "C" fn native_plugin_context_register_subscriber(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    cb: NemoRelayNativeEventSubscriberCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_subscriber(
        &name,
        wrap_event_subscriber(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

macro_rules! native_tool_json_context_register {
    ($fn_name:ident, $ctx_method:ident) => {
        unsafe extern "C" fn $fn_name(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeToolJsonCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus {
            clear_native_last_error();
            let host_ctx = match host_ctx_mut(ctx) {
                Ok(ctx) => ctx,
                Err(status) => return status,
            };
            let instance = host_ctx.instance.clone();
            let ctx = unsafe { &mut *host_ctx.ctx };
            let name = match read_name(name) {
                Ok(name) => name,
                Err(status) => return status,
            };
            match ctx.$ctx_method(
                &name,
                priority,
                wrap_tool_json_fn(instance, cb, user_data, free_fn),
            ) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(err) => status_from_plugin_error(err),
            }
        }
    };
}

native_tool_json_context_register!(
    native_plugin_context_register_tool_sanitize_request_guardrail,
    register_tool_sanitize_request_guardrail
);
native_tool_json_context_register!(
    native_plugin_context_register_tool_sanitize_response_guardrail,
    register_tool_sanitize_response_guardrail
);

unsafe extern "C" fn native_plugin_context_register_tool_conditional_execution_guardrail(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeToolConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_tool_conditional_execution_guardrail(
        &name,
        priority,
        wrap_tool_conditional_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_tool_request_intercept(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_tool_request_intercept(
        &name,
        priority,
        break_chain,
        wrap_tool_intercept_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_tool_execution_intercept(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeToolExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_tool_execution_intercept(
        &name,
        priority,
        wrap_tool_execution_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_sanitize_request_guardrail(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeRequestCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_sanitize_request_guardrail(
        &name,
        priority,
        wrap_llm_sanitize_request_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_sanitize_response_guardrail(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmSanitizeResponseCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_sanitize_response_guardrail(
        &name,
        priority,
        wrap_llm_sanitize_response_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_conditional_execution_guardrail(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_conditional_execution_guardrail(
        &name,
        priority,
        wrap_llm_conditional_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_request_intercept(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayNativeLlmRequestInterceptCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    if let Err(error) = validate_annotated_request_consumer_compatibility(
        &instance.relay_compat,
        &instance.plugin_kind,
    ) {
        return status_from_plugin_error(error);
    }
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        wrap_llm_request_intercept_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_execution_intercept(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_execution_intercept(
        &name,
        priority,
        wrap_llm_execution_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_stream_execution_intercept(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeLlmStreamExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> NemoRelayStatus {
    clear_native_last_error();
    let host_ctx = match host_ctx_mut(ctx) {
        Ok(ctx) => ctx,
        Err(status) => return status,
    };
    let instance = host_ctx.instance.clone();
    let ctx = unsafe { &mut *host_ctx.ctx };
    let name = match read_name(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match ctx.register_llm_stream_execution_intercept(
        &name,
        priority,
        wrap_llm_stream_execution_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

macro_rules! native_event_sanitize_context_register {
    ($fn_name:ident, $ctx_method:ident) => {
        unsafe extern "C" fn $fn_name(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeEventSanitizeCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus {
            clear_native_last_error();
            let host_ctx = match host_ctx_mut(ctx) {
                Ok(ctx) => ctx,
                Err(status) => return status,
            };
            let instance = host_ctx.instance.clone();
            let ctx = unsafe { &mut *host_ctx.ctx };
            let name = match read_name(name) {
                Ok(name) => name,
                Err(status) => return status,
            };
            match ctx.$ctx_method(
                &name,
                priority,
                wrap_event_sanitize_fn(instance, cb, user_data, free_fn),
            ) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(err) => status_from_plugin_error(err),
            }
        }
    };
}

native_event_sanitize_context_register!(
    native_plugin_context_register_mark_sanitize_guardrail,
    register_mark_sanitize_guardrail
);
native_event_sanitize_context_register!(
    native_plugin_context_register_scope_sanitize_start_guardrail,
    register_scope_sanitize_start_guardrail
);
native_event_sanitize_context_register!(
    native_plugin_context_register_scope_sanitize_end_guardrail,
    register_scope_sanitize_end_guardrail
);

fn wrap_event_subscriber(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeEventSubscriberCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> EventSubscriberFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |event: &Event| {
        let event_json = serde_json::to_value(event).unwrap_or(Json::Null);
        if let Some(event_string) = native_string_from_json(&event_json) {
            let status = unsafe { cb(user_data.ptr, event_string) };
            if status != NemoRelayStatus::Ok {
                set_native_last_error(format!("native subscriber callback returned {status:?}"));
            }
            unsafe { native_string_free(event_string) };
        }
    })
}

fn wrap_event_sanitize_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeEventSanitizeCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> EventSanitizeFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |event, fields| {
        let user_data = user_data.clone();
        Box::pin(async move { call_event_sanitize_callback(cb, user_data.ptr, &event, &fields) })
    })
}

fn call_event_sanitize_callback(
    cb: NemoRelayNativeEventSanitizeCb,
    user_data: *mut c_void,
    event: &Event,
    fields: &EventSanitizeFields,
) -> FlowResult<EventSanitizeFields> {
    clear_native_last_error();
    let event_json = serde_json::to_value(event)
        .map_err(|err| FlowError::Internal(format!("failed to encode native event: {err}")))?;
    let fields_json = serde_json::to_value(fields).map_err(|err| {
        FlowError::Internal(format!("failed to encode native event fields: {err}"))
    })?;
    let event = native_string_from_json(&event_json)
        .ok_or_else(|| FlowError::Internal("failed to allocate native event".into()))?;
    let fields = match native_string_from_json(&fields_json) {
        Some(fields) => fields,
        None => {
            unsafe { native_string_free(event) };
            return Err(FlowError::Internal(
                "failed to allocate native event fields".into(),
            ));
        }
    };
    let mut out = ptr::null_mut();
    let status = unsafe { cb(user_data, event, fields, &mut out) };
    unsafe {
        native_string_free(event);
        native_string_free(fields);
    }
    if status != NemoRelayStatus::Ok {
        if !out.is_null() {
            unsafe { native_string_free(out) };
        }
        return Err(flow_error_from_status(
            status,
            "native event sanitizer failed",
        ));
    }
    let value = take_json_from_native_string(out, "native event sanitizer returned null")?;
    serde_json::from_value(value)
        .map_err(|err| FlowError::Internal(format!("invalid event sanitize fields: {err}")))
}

fn wrap_tool_json_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolSanitizeFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, payload| {
        let user_data = user_data.clone();
        Box::pin(async move { call_tool_json_callback(cb, user_data.ptr, &name, &payload) })
    })
}

fn wrap_tool_intercept_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolInterceptFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, payload| {
        let user_data = user_data.clone();
        Box::pin(async move { call_tool_json_callback(cb, user_data.ptr, &name, &payload) })
    })
}

fn call_tool_json_callback(
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    name: &str,
    payload: &Json,
) -> FlowResult<Json> {
    clear_native_last_error();
    let name = native_string_from_str(name)
        .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
    let payload = native_string_from_json(payload)
        .ok_or_else(|| FlowError::Internal("failed to allocate native payload".into()))?;
    let mut out = ptr::null_mut();
    let status = unsafe { cb(user_data, name, payload, &mut out) };
    unsafe {
        native_string_free(name);
        native_string_free(payload);
    }
    if status != NemoRelayStatus::Ok {
        if !out.is_null() {
            unsafe { native_string_free(out) };
        }
        return Err(flow_error_from_status(
            status,
            "native JSON callback failed",
        ));
    }
    take_json_from_native_string(out, "native JSON callback returned null")
}

fn wrap_tool_conditional_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolConditionalFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, args| {
        let user_data = user_data.clone();
        Box::pin(async move {
            clear_native_last_error();
            let name_string = native_string_from_str(&name)
                .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
            let args_string = native_string_from_json(&args)
                .ok_or_else(|| FlowError::Internal("failed to allocate native args".into()))?;
            let mut out = ptr::null_mut();
            let status = unsafe { cb(user_data.ptr, name_string, args_string, &mut out) };
            unsafe {
                native_string_free(name_string);
                native_string_free(args_string);
            }
            if status != NemoRelayStatus::Ok {
                if !out.is_null() {
                    unsafe { native_string_free(out) };
                }
                return Err(flow_error_from_status(
                    status,
                    "native tool conditional failed",
                ));
            }
            if out.is_null() {
                Ok(None)
            } else {
                let reason = take_native_string(out)?;
                Ok(Some(reason))
            }
        })
    })
}

fn wrap_tool_execution_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, args, next| {
        let name = name.to_owned();
        let user_data = user_data.clone();
        Box::pin(async move {
            clear_native_last_error();
            let name_string = native_string_from_str(&name)
                .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
            let args_string = native_string_from_json(&args)
                .ok_or_else(|| FlowError::Internal("failed to allocate native args".into()))?;
            let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
            let mut out_outcome = ptr::null_mut();
            let status = unsafe {
                cb(
                    user_data.ptr,
                    name_string,
                    args_string,
                    native_tool_next,
                    next_ctx,
                    &mut out_outcome,
                )
            };
            unsafe {
                drop(Box::from_raw(next_ctx as *mut ToolExecutionNextFn));
                native_string_free(name_string);
                native_string_free(args_string);
            }
            if status != NemoRelayStatus::Ok {
                if !out_outcome.is_null() {
                    unsafe { native_string_free(out_outcome) };
                }
                return Err(flow_error_from_status(
                    status,
                    "native tool execution failed",
                ));
            }
            let outcome_json = take_json_from_native_string(
                out_outcome,
                "native tool execution returned null outcome",
            )?;
            serde_json::from_value::<ToolExecutionInterceptOutcome>(outcome_json).map_err(|err| {
                FlowError::Internal(format!("invalid native tool execution outcome JSON: {err}"))
            })
        })
    })
}

unsafe extern "C" fn native_tool_next(
    args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if next_ctx.is_null() || out_json.is_null() {
        set_native_last_error("native tool next received null pointer");
        return NemoRelayStatus::NullPointer;
    }
    let args = match parse_json_arg(args_json, "native tool next args") {
        Ok(args) => args,
        Err(status) => return status,
    };
    let next = unsafe { (*(next_ctx as *const ToolExecutionNextFn)).clone() };
    let context = MiddlewareContinuationContext::capture();
    let result = spawn_with_continuation_context(context, move || next(args)).join();
    match result {
        Ok(Ok(result)) => write_native_json(&result, out_json),
        Ok(Err(err)) => status_from_flow_error(err),
        Err(_) => {
            set_native_last_error("native tool next panicked");
            NemoRelayStatus::Internal
        }
    }
}

fn wrap_llm_sanitize_request_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmSanitizeRequestCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeRequestFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request, context| {
        let user_data = user_data.clone();
        Box::pin(
            async move { call_llm_sanitize_request_callback(cb, user_data.ptr, &request, context) },
        )
    })
}

fn wrap_llm_sanitize_response_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmSanitizeResponseCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeResponseFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |payload, context| {
        let user_data = user_data.clone();
        Box::pin(async move {
            call_llm_sanitize_response_callback(cb, user_data.ptr, &payload, context)
        })
    })
}

fn call_llm_sanitize_request_callback(
    cb: NemoRelayNativeLlmSanitizeRequestCb,
    user_data: *mut c_void,
    request: &LlmRequest,
    context: LlmSanitizeRequestContext,
) -> FlowResult<Option<LlmRequest>> {
    clear_native_last_error();
    let codec = context.resolve_codec().map(NativeHostLlmRequestCodec);
    let (codec_kind, context_id) = native_llm_codec_identity(context.codec())?;
    let request_json = match serde_json::to_value(request) {
        Ok(request_json) => request_json,
        Err(err) => {
            if let Some(context_id) = context_id {
                unsafe { native_string_free(context_id) };
            }
            return Err(FlowError::Internal(format!(
                "failed to serialize LLM request: {err}"
            )));
        }
    };
    let request_string = match native_string_from_json(&request_json) {
        Some(request_string) => request_string,
        None => {
            if let Some(context_id) = context_id {
                unsafe { native_string_free(context_id) };
            }
            return Err(FlowError::Internal(
                "failed to allocate native LLM request".into(),
            ));
        }
    };
    let context = NemoRelayNativeLlmSanitizeRequestContext {
        codec_kind,
        codec_id: context_id.map_or(ptr::null(), |value| value.cast_const()),
        codec: codec
            .as_ref()
            .map_or(ptr::null(), |value| std::ptr::from_ref(value).cast()),
    };
    let mut out = ptr::null_mut();
    let status = unsafe { cb(user_data, request_string, context, &mut out) };
    if status != NemoRelayStatus::Ok {
        unsafe { free_native_sanitizer_strings(request_string, context_id, out) };
        return Err(flow_error_from_status(
            status,
            "native LLM sanitize-request callback failed",
        ));
    }
    if out.is_null() {
        unsafe { free_native_sanitizer_strings(request_string, context_id, out) };
        return Ok(None);
    }
    let result_json =
        json_from_native_string(out, "native LLM sanitize-request returned invalid JSON");
    unsafe { free_native_sanitizer_strings(request_string, context_id, out) };
    let result_json = result_json?;
    serde_json::from_value(result_json)
        .map(Some)
        .map_err(|err| FlowError::Internal(format!("invalid LLM request JSON: {err}")))
}

fn call_llm_sanitize_response_callback(
    cb: NemoRelayNativeLlmSanitizeResponseCb,
    user_data: *mut c_void,
    payload: &Json,
    context: LlmSanitizeResponseContext,
) -> FlowResult<Option<Json>> {
    clear_native_last_error();
    let codec = context.resolve_codec().map(NativeHostLlmResponseCodec);
    let (codec_kind, context_id) = native_llm_codec_identity(context.codec())?;
    let payload_string = match native_string_from_json(payload) {
        Some(payload_string) => payload_string,
        None => {
            if let Some(context_id) = context_id {
                unsafe { native_string_free(context_id) };
            }
            return Err(FlowError::Internal(
                "failed to allocate native LLM response".into(),
            ));
        }
    };
    let context = NemoRelayNativeLlmSanitizeResponseContext {
        codec_kind,
        codec_id: context_id.map_or(ptr::null(), |value| value.cast_const()),
        codec: codec
            .as_ref()
            .map_or(ptr::null(), |value| std::ptr::from_ref(value).cast()),
    };
    let mut out = ptr::null_mut();
    let status = unsafe { cb(user_data, payload_string, context, &mut out) };
    if status != NemoRelayStatus::Ok {
        unsafe { free_native_sanitizer_strings(payload_string, context_id, out) };
        return Err(flow_error_from_status(
            status,
            "native LLM sanitize-response callback failed",
        ));
    }
    if out.is_null() {
        unsafe { free_native_sanitizer_strings(payload_string, context_id, out) };
        return Ok(None);
    }
    let result = json_from_native_string(out, "native LLM sanitize-response returned invalid JSON");
    unsafe { free_native_sanitizer_strings(payload_string, context_id, out) };
    result.map(Some)
}

fn native_llm_codec_identity(
    context: &LlmCodecIdentity,
) -> FlowResult<(
    NemoRelayNativeLlmCodecKind,
    Option<*mut NemoRelayNativeString>,
)> {
    let (codec_kind, codec_id) = match context {
        LlmCodecIdentity::None => (NemoRelayNativeLlmCodecKind::None, None),
        LlmCodecIdentity::BuiltIn(codec) => {
            (NemoRelayNativeLlmCodecKind::BuiltIn, Some(codec.id()))
        }
        LlmCodecIdentity::Runtime(id) => (NemoRelayNativeLlmCodecKind::Runtime, Some(id.as_str())),
        LlmCodecIdentity::Opaque => (NemoRelayNativeLlmCodecKind::Opaque, None),
    };
    let codec_id =
        match codec_id {
            Some(codec_id) => Some(native_string_from_str(codec_id).ok_or_else(|| {
                FlowError::Internal("failed to allocate native LLM codec ID".into())
            })?),
            None => None,
        };
    Ok((codec_kind, codec_id))
}

fn wrap_llm_conditional_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmConditionalFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request| {
        let user_data = user_data.clone();
        Box::pin(async move {
            clear_native_last_error();
            let request_json = serde_json::to_value(request).map_err(|err| {
                FlowError::Internal(format!("failed to serialize LLM request: {err}"))
            })?;
            let request_string = native_string_from_json(&request_json).ok_or_else(|| {
                FlowError::Internal("failed to allocate native LLM request".into())
            })?;
            let mut out = ptr::null_mut();
            let status = unsafe { cb(user_data.ptr, request_string, &mut out) };
            unsafe { native_string_free(request_string) };
            if status != NemoRelayStatus::Ok {
                if !out.is_null() {
                    unsafe { native_string_free(out) };
                }
                return Err(flow_error_from_status(
                    status,
                    "native LLM conditional failed",
                ));
            }
            if out.is_null() {
                Ok(None)
            } else {
                let reason = take_native_string(out)?;
                Ok(Some(reason))
            }
        })
    })
}

fn wrap_llm_request_intercept_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmRequestInterceptCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmRequestInterceptFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, annotated| {
        let user_data = user_data.clone();
        Box::pin(async move {
            clear_native_last_error();
            let name_string = native_string_from_str(&name)
                .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
            let request_json = serde_json::to_value(&request).map_err(|err| {
                FlowError::Internal(format!("failed to serialize LLM request: {err}"))
            })?;
            let request_string = native_string_from_json(&request_json).ok_or_else(|| {
                FlowError::Internal("failed to allocate native LLM request".into())
            })?;
            let annotated_string = match &annotated {
                Some(annotated) => {
                    let value = serde_json::to_value(annotated).map_err(|err| {
                        FlowError::Internal(format!("failed to serialize annotated request: {err}"))
                    })?;
                    native_string_from_json(&value).ok_or_else(|| {
                        FlowError::Internal("failed to allocate annotated request".into())
                    })?
                }
                None => ptr::null_mut(),
            };
            let mut out_outcome = ptr::null_mut();
            let status = unsafe {
                cb(
                    user_data.ptr,
                    name_string,
                    request_string,
                    annotated_string,
                    &mut out_outcome,
                )
            };
            unsafe {
                native_string_free(name_string);
                native_string_free(request_string);
                native_string_free(annotated_string);
            }
            if status != NemoRelayStatus::Ok {
                unsafe {
                    native_string_free(out_outcome);
                }
                return Err(flow_error_from_status(
                    status,
                    "native LLM request intercept failed",
                ));
            }
            let outcome_json = json_from_native_string(
                out_outcome,
                "native LLM request intercept returned null outcome",
            );
            unsafe {
                native_string_free(out_outcome);
            }
            serde_json::from_value::<LlmRequestInterceptOutcome>(outcome_json?).map_err(|err| {
                FlowError::Internal(format!("invalid LLM request intercept outcome JSON: {err}"))
            })
        })
    })
}

fn wrap_llm_execution_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, next| {
        let name = name.to_owned();
        let user_data = user_data.clone();
        Box::pin(async move { call_llm_execution_callback(cb, &user_data, &name, &request, next) })
    })
}

fn call_llm_execution_callback(
    cb: NemoRelayNativeLlmExecutionCb,
    user_data: &NativeCallbackUserData,
    name: &str,
    request: &LlmRequest,
    next: LlmExecutionNextFn,
) -> FlowResult<Json> {
    clear_native_last_error();
    let name_string = native_string_from_str(name)
        .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
    let request_json = serde_json::to_value(request)
        .map_err(|err| FlowError::Internal(format!("failed to serialize LLM request: {err}")))?;
    let request_string = native_string_from_json(&request_json)
        .ok_or_else(|| FlowError::Internal("failed to allocate native LLM request".into()))?;
    let next_ctx = Box::into_raw(Box::new(next)) as *mut c_void;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cb(
            user_data.ptr,
            name_string,
            request_string,
            native_llm_next,
            next_ctx,
            &mut out,
        )
    };
    unsafe {
        drop(Box::from_raw(next_ctx as *mut LlmExecutionNextFn));
        native_string_free(name_string);
        native_string_free(request_string);
    }
    if status != NemoRelayStatus::Ok {
        if !out.is_null() {
            unsafe { native_string_free(out) };
        }
        return Err(flow_error_from_status(
            status,
            "native LLM execution failed",
        ));
    }
    take_json_from_native_string(out, "native LLM execution returned null")
}

unsafe extern "C" fn native_llm_next(
    request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if next_ctx.is_null() || out_json.is_null() {
        set_native_last_error("native LLM next received null pointer");
        return NemoRelayStatus::NullPointer;
    }
    let request = match parse_llm_request_arg(request_json, "native LLM next request") {
        Ok(request) => request,
        Err(status) => return status,
    };
    let next = unsafe { (*(next_ctx as *const LlmExecutionNextFn)).clone() };
    let context = MiddlewareContinuationContext::capture();
    let result = spawn_with_continuation_context(context, move || next(request)).join();
    match result {
        Ok(Ok(result)) => write_native_json(&result, out_json),
        Ok(Err(err)) => status_from_flow_error(err),
        Err(_) => {
            set_native_last_error("native LLM next panicked");
            NemoRelayStatus::Internal
        }
    }
}
fn wrap_llm_stream_execution_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmStreamExecutionCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmStreamExecutionFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, next| {
        let name = name.to_owned();
        let user_data = user_data.clone();
        Box::pin(
            async move { call_llm_stream_execution_callback(cb, user_data, &name, &request, next) },
        )
    })
}

fn call_llm_stream_execution_callback(
    cb: NemoRelayNativeLlmStreamExecutionCb,
    user_data: Arc<NativeCallbackUserData>,
    name: &str,
    request: &LlmRequest,
    next: LlmStreamExecutionNextFn,
) -> FlowResult<LlmJsonStream> {
    clear_native_last_error();
    let name_string = native_string_from_str(name)
        .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
    let request_json = serde_json::to_value(request)
        .map_err(|err| FlowError::Internal(format!("failed to serialize LLM request: {err}")))?;
    let request_string = native_string_from_json(&request_json)
        .ok_or_else(|| FlowError::Internal("failed to allocate native LLM request".into()))?;
    let next_ctx = NativeStreamNextContext::new(Box::into_raw(Box::new(next)) as *mut c_void);
    let mut out = NemoRelayNativeLlmStreamV1::default();
    let status = unsafe {
        cb(
            user_data.ptr,
            name_string,
            request_string,
            native_llm_stream_next,
            next_ctx.ptr,
            &mut out,
        )
    };
    unsafe {
        native_string_free(name_string);
        native_string_free(request_string);
    }
    if status != NemoRelayStatus::Ok {
        drop_native_stream(out);
        return Err(flow_error_from_status(
            status,
            "native LLM stream execution failed",
        ));
    }
    native_stream_to_relay_stream(out, Some(next_ctx), Some(user_data))
}

unsafe extern "C" fn native_llm_stream_next(
    request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus {
    if next_ctx.is_null() || out_stream.is_null() {
        set_native_last_error("native LLM stream next received null pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_stream = NemoRelayNativeLlmStreamV1::default() };
    let request = match parse_llm_request_arg(request_json, "native LLM stream next request") {
        Ok(request) => request,
        Err(status) => return status,
    };
    let next = unsafe { (*(next_ctx as *const LlmStreamExecutionNextFn)).clone() };
    let context = MiddlewareContinuationContext::capture();
    let stream_context = context.clone();
    let result = spawn_with_continuation_context(context, move || next(request)).join();
    match result {
        Ok(Ok(stream)) => {
            unsafe {
                *out_stream = relay_stream_to_native_stream_with_context(stream, stream_context)
            };
            NemoRelayStatus::Ok
        }
        Ok(Err(err)) => status_from_flow_error(err),
        Err(_) => {
            set_native_last_error("native LLM stream next panicked");
            NemoRelayStatus::Internal
        }
    }
}

struct NativeRelayLlmStream {
    raw: NemoRelayNativeLlmStreamV1,
    finished: bool,
    _next_ctx: Option<NativeStreamNextContext>,
    _callback_user_data: Option<Arc<NativeCallbackUserData>>,
}

unsafe impl Send for NativeRelayLlmStream {}

impl NativeRelayLlmStream {
    fn from_raw(
        raw: NemoRelayNativeLlmStreamV1,
        next_ctx: Option<NativeStreamNextContext>,
        callback_user_data: Option<Arc<NativeCallbackUserData>>,
    ) -> FlowResult<Self> {
        if raw.struct_size != std::mem::size_of::<NemoRelayNativeLlmStreamV1>() {
            let struct_size = raw.struct_size;
            drop_native_stream(raw);
            return Err(FlowError::Internal(format!(
                "unsupported native LLM stream struct size: {}",
                struct_size
            )));
        }
        if raw.next.is_none() {
            drop_native_stream(raw);
            return Err(FlowError::Internal(
                "native LLM stream next callback was null".into(),
            ));
        }
        Ok(Self {
            raw,
            finished: false,
            _next_ctx: next_ctx,
            _callback_user_data: callback_user_data,
        })
    }

    fn finish(&mut self) {
        self.finished = true;
        if let Some(drop_fn) = self.raw.drop.take() {
            unsafe { drop_fn(self.raw.user_data) };
        }
        self.raw.user_data = ptr::null_mut();
    }
}

impl Stream for NativeRelayLlmStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        let Some(next) = self.raw.next else {
            self.finish();
            return Poll::Ready(Some(Err(FlowError::Internal(
                "native LLM stream next callback was null".into(),
            ))));
        };
        let mut out = ptr::null_mut();
        let status = unsafe { next(self.raw.user_data, &mut out) };
        match status {
            NemoRelayStatus::Ok => {
                if out.is_null() {
                    let error = FlowError::Internal(
                        native_last_error_message()
                            .unwrap_or_else(|| "native LLM stream returned null chunk".into()),
                    );
                    self.finish();
                    return Poll::Ready(Some(Err(error)));
                }
                let result =
                    take_json_from_native_string(out, "native LLM stream returned null chunk");
                if result.is_err() {
                    self.finish();
                }
                Poll::Ready(Some(result))
            }
            NemoRelayStatus::StreamEnd => {
                if !out.is_null() {
                    unsafe { native_string_free(out) };
                }
                self.finish();
                Poll::Ready(None)
            }
            status => {
                if !out.is_null() {
                    unsafe { native_string_free(out) };
                }
                let error = flow_error_from_status(status, "native LLM stream poll failed");
                self.finish();
                Poll::Ready(Some(Err(error)))
            }
        }
    }
}

impl Drop for NativeRelayLlmStream {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = self.raw.cancel
        {
            let _ = unsafe { cancel(self.raw.user_data) };
        }
        self.finish();
    }
}

fn native_stream_to_relay_stream(
    raw: NemoRelayNativeLlmStreamV1,
    next_ctx: Option<NativeStreamNextContext>,
    callback_user_data: Option<Arc<NativeCallbackUserData>>,
) -> FlowResult<LlmJsonStream> {
    Ok(LlmJsonStream::new(NativeRelayLlmStream::from_raw(
        raw,
        next_ctx,
        callback_user_data,
    )?))
}

fn drop_native_stream(mut raw: NemoRelayNativeLlmStreamV1) {
    if let Some(drop_fn) = raw.drop.take() {
        unsafe { drop_fn(raw.user_data) };
    }
}

struct NativeHostLlmStream {
    stream: Arc<Mutex<Option<LlmJsonStream>>>,
    context: MiddlewareContinuationContext,
}

struct NativeStreamNextContext {
    ptr: *mut c_void,
}

unsafe impl Send for NativeStreamNextContext {}

impl NativeStreamNextContext {
    fn new(ptr: *mut c_void) -> Self {
        Self { ptr }
    }
}

impl Drop for NativeStreamNextContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            drop(unsafe { Box::from_raw(self.ptr as *mut LlmStreamExecutionNextFn) });
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
fn relay_stream_to_native_stream(stream: LlmJsonStream) -> NemoRelayNativeLlmStreamV1 {
    relay_stream_to_native_stream_with_context(stream, MiddlewareContinuationContext::capture())
}

fn relay_stream_to_native_stream_with_context(
    stream: LlmJsonStream,
    context: MiddlewareContinuationContext,
) -> NemoRelayNativeLlmStreamV1 {
    let state = Box::new(NativeHostLlmStream {
        stream: Arc::new(Mutex::new(Some(stream))),
        context,
    });
    NemoRelayNativeLlmStreamV1 {
        struct_size: std::mem::size_of::<NemoRelayNativeLlmStreamV1>(),
        user_data: Box::into_raw(state).cast(),
        next: Some(poll_relay_llm_stream),
        cancel: Some(cancel_relay_llm_stream),
        drop: Some(drop_relay_llm_stream),
    }
}

unsafe extern "C" fn poll_relay_llm_stream(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_json.is_null() {
        set_native_last_error("native host LLM stream poll received null pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const NativeHostLlmStream) };
    let stream = state.stream.clone();
    let context = state.context.clone();
    let result = spawn_with_continuation_context(context, move || async move {
        let Some(mut current) = stream
            .lock()
            .map_err(|_| FlowError::Internal("native host LLM stream lock poisoned".into()))?
            .take()
        else {
            return Ok(None);
        };
        match current.next().await {
            Some(Ok(chunk)) => {
                *stream.lock().map_err(|_| {
                    FlowError::Internal("native host LLM stream lock poisoned".into())
                })? = Some(current);
                Ok(Some(chunk))
            }
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }
    })
    .join();
    match result {
        Ok(Ok(Some(chunk))) => write_native_json(&chunk, out_json),
        Ok(Ok(None)) => NemoRelayStatus::StreamEnd,
        Ok(Err(err)) => status_from_flow_error(err),
        Err(_) => {
            set_native_last_error("native host LLM stream poll panicked");
            NemoRelayStatus::Internal
        }
    }
}

unsafe extern "C" fn cancel_relay_llm_stream(user_data: *mut c_void) -> NemoRelayStatus {
    if user_data.is_null() {
        set_native_last_error("native host LLM stream cancel received null pointer");
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const NativeHostLlmStream) };
    match state.stream.lock() {
        Ok(mut stream) => {
            stream.take();
            NemoRelayStatus::Ok
        }
        Err(_) => {
            set_native_last_error("native host LLM stream lock poisoned");
            NemoRelayStatus::Internal
        }
    }
}

unsafe extern "C" fn drop_relay_llm_stream(user_data: *mut c_void) {
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data as *mut NativeHostLlmStream) });
    }
}

fn parse_json_arg(
    value: *const NemoRelayNativeString,
    label: &str,
) -> Result<Json, NemoRelayStatus> {
    let text = match read_native_string(value) {
        Ok(text) => text,
        Err(err) => {
            set_native_last_error(err.to_string());
            return Err(NemoRelayStatus::InvalidUtf8);
        }
    };
    serde_json::from_str(&text).map_err(|err| {
        set_native_last_error(format!("{label} was invalid JSON: {err}"));
        NemoRelayStatus::InvalidJson
    })
}

fn parse_llm_request_arg(
    value: *const NemoRelayNativeString,
    label: &str,
) -> Result<LlmRequest, NemoRelayStatus> {
    let value = parse_json_arg(value, label)?;
    serde_json::from_value(value).map_err(|err| {
        set_native_last_error(format!("{label} was not an LLM request: {err}"));
        NemoRelayStatus::InvalidJson
    })
}

fn write_native_json(value: &Json, out: *mut *mut NemoRelayNativeString) -> NemoRelayStatus {
    if out.is_null() {
        set_native_last_error("out JSON pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let Some(handle) = native_string_from_json(value) else {
        set_native_last_error("failed to serialize native JSON output");
        return NemoRelayStatus::Internal;
    };
    unsafe { *out = handle };
    NemoRelayStatus::Ok
}

#[cfg(test)]
#[path = "../../../tests/unit/native_plugin_tests.rs"]
mod tests;
