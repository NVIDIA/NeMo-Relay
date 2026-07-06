// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native dynamic plugin loader and host-side ABI adapter.

use std::cell::RefCell;
use std::ffi::c_void;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};

use chrono::{DateTime, Utc};
use libloading::{Library, Symbol};
use nemo_relay_plugin::{
    NEMO_RELAY_NATIVE_ABI_VERSION, NemoRelayNativeEventSubscriberCb, NemoRelayNativeFreeFn,
    NemoRelayNativeHostApiV1, NemoRelayNativeJsonCb, NemoRelayNativeLlmConditionalCb,
    NemoRelayNativeLlmExecutionCb, NemoRelayNativeLlmRequestCb,
    NemoRelayNativeLlmRequestInterceptCb, NemoRelayNativeLlmStreamExecutionCb,
    NemoRelayNativeLlmStreamV1, NemoRelayNativePluginContext, NemoRelayNativePluginEntry,
    NemoRelayNativePluginV1, NemoRelayNativeScopeHandle, NemoRelayNativeScopeStack,
    NemoRelayNativeScopeStackBinding, NemoRelayNativeScopeType, NemoRelayNativeString,
    NemoRelayNativeToolConditionalCb, NemoRelayNativeToolExecutionCb, NemoRelayNativeToolJsonCb,
    NemoRelayNativeWithScopeStackCb, NemoRelayStatus,
};
use semver::{Version, VersionReq};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use tokio_stream::{Stream, StreamExt};

use crate::api::event::Event;
use crate::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use crate::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmExecutionNextFn, LlmJsonStream,
    LlmRequestInterceptFn, LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, ToolConditionalFn, ToolExecutionFn, ToolExecutionNextFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::runtime::{
    ScopeStackHandle, ThreadScopeStackBinding, capture_thread_scope_stack, create_scope_stack,
    current_scope_stack, restore_thread_scope_stack, scope_stack_active, set_thread_scope_stack,
    with_scope_stack,
};
use crate::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeHandle, ScopeType,
};
use crate::api::scope::{event as emit_scope_mark, get_handle, pop_scope, push_scope};
use crate::api::tool::ToolExecutionInterceptOutcome;
use crate::error::{FlowError, Result as FlowResult};
use crate::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistrationContext,
    deregister_plugin, deregister_plugin_checked, register_plugin,
};

use super::{
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginManifestLoad,
    DynamicPluginTeardownOutcome,
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
    plugin_kinds: Vec<String>,
}

impl NativePluginActivation {
    /// Returns `true` when no native plugins were loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Consumes the activation and deregisters loaded plugin kinds.
    pub fn clear(self) {}

    pub(crate) fn deregister_plugin_kinds_checked(&mut self) -> DynamicPluginTeardownOutcome {
        let mut outcome = DynamicPluginTeardownOutcome::success();
        let plugin_kinds = std::mem::take(&mut self.plugin_kinds);
        for plugin_kind in plugin_kinds.into_iter().rev() {
            match deregister_plugin_checked(&plugin_kind) {
                Ok(true) => {}
                Ok(false) => outcome.record_error(
                    format!(
                        "native plugin kind '{plugin_kind}' was not registered during teardown"
                    ),
                    true,
                ),
                Err(error) => outcome.record_error(
                    format!("failed to deregister native plugin kind '{plugin_kind}': {error}"),
                    false,
                ),
            }
        }
        outcome
    }

    #[cfg(test)]
    pub(super) fn with_plugin_kind_for_test(plugin_kind: impl Into<String>) -> Self {
        Self {
            plugins: Vec::new(),
            plugin_kinds: vec![plugin_kind.into()],
        }
    }
}

impl Drop for NativePluginActivation {
    fn drop(&mut self) {
        for plugin_kind in self.plugin_kinds.iter().rev() {
            let _ = deregister_plugin(plugin_kind);
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
        plugin_kinds: Vec::new(),
    };
    for spec in specs {
        let instance = load_one_native_plugin(&spec)?;
        let plugin_kind = instance.plugin_kind.clone();
        register_plugin(Arc::new(NativePluginAdapter {
            plugin_kind: plugin_kind.clone(),
            allows_multiple_components: instance.allows_multiple_components,
            instance: instance.clone(),
        }))?;
        activation.plugins.push(instance);
        activation.plugin_kinds.push(plugin_kind);
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
        let status = entry(native_host_api(), &mut plugin);
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

struct NativeHostScopeHandle(ScopeHandle);

struct NativeHostScopeStack(ScopeStackHandle);

struct NativeHostScopeStackBinding(ThreadScopeStackBinding);

thread_local! {
    static NATIVE_LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
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
    let handle = Box::new(NativeHostString(bytes.to_vec()));
    unsafe { *out = Box::into_raw(handle) as *mut NemoRelayNativeString };
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

fn native_host_api() -> *const NemoRelayNativeHostApiV1 {
    static HOST_API: OnceLock<NemoRelayNativeHostApiV1> = OnceLock::new();
    static RELAY_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    HOST_API.get_or_init(|| NemoRelayNativeHostApiV1 {
        abi_version: NEMO_RELAY_NATIVE_ABI_VERSION,
        struct_size: std::mem::size_of::<NemoRelayNativeHostApiV1>(),
        relay_version: RELAY_VERSION.as_ptr().cast(),
        string_new: native_string_new,
        string_data: native_string_data,
        string_len: native_string_len,
        string_free: native_string_free,
        last_error_clear: native_last_error_clear,
        last_error_set: native_last_error_set,
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
    }) as *const _
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
        FlowError::Internal(_) => NemoRelayStatus::Internal,
    }
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

fn spawn_with_current_scope<T>(f: impl FnOnce() -> T + Send + 'static) -> std::thread::JoinHandle<T>
where
    T: Send + 'static,
{
    let binding = capture_thread_scope_stack();
    let visible_stack = scope_stack_active().then(current_scope_stack);
    std::thread::spawn(move || {
        restore_thread_scope_stack(binding);
        if let Some(stack) = visible_stack {
            with_scope_stack(stack, f)
        } else {
            f()
        }
    })
}

struct NativeCallbackUserData {
    ptr: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
    _instance: Arc<NativePluginInstance>,
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
        _instance: instance,
    })
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
    cb: NemoRelayNativeLlmRequestCb,
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
        wrap_llm_request_fn(instance, cb, user_data, free_fn),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(err),
    }
}

unsafe extern "C" fn native_plugin_context_register_llm_sanitize_response_guardrail(
    ctx: *mut NemoRelayNativePluginContext,
    name: *const NemoRelayNativeString,
    priority: i32,
    cb: NemoRelayNativeJsonCb,
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
        wrap_json_fn(instance, cb, user_data, free_fn),
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

fn wrap_tool_json_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolSanitizeFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, payload| {
        call_tool_json_callback(cb, user_data.ptr, name, &payload).unwrap_or(Json::Null)
    })
}

fn wrap_tool_intercept_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeToolJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> ToolInterceptFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, payload| call_tool_json_callback(cb, user_data.ptr, name, &payload))
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
        clear_native_last_error();
        let name_string = native_string_from_str(name)
            .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
        let args_string = native_string_from_json(args)
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
    let result = spawn_with_current_scope(move || native_runtime().block_on(next(args))).join();
    match result {
        Ok(Ok(result)) => write_native_json(&result, out_json),
        Ok(Err(err)) => status_from_flow_error(err),
        Err(_) => {
            set_native_last_error("native tool next panicked");
            NemoRelayStatus::Internal
        }
    }
}

fn wrap_llm_request_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmRequestCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeRequestFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request| {
        call_llm_request_callback(cb, user_data.ptr, &request).unwrap_or_else(|_| LlmRequest {
            headers: Map::new(),
            content: Json::Null,
        })
    })
}

fn call_llm_request_callback(
    cb: NemoRelayNativeLlmRequestCb,
    user_data: *mut c_void,
    request: &LlmRequest,
) -> FlowResult<LlmRequest> {
    clear_native_last_error();
    let request_json = serde_json::to_value(request)
        .map_err(|err| FlowError::Internal(format!("failed to serialize LLM request: {err}")))?;
    let request_string = native_string_from_json(&request_json)
        .ok_or_else(|| FlowError::Internal("failed to allocate native LLM request".into()))?;
    let mut out = ptr::null_mut();
    let status = unsafe { cb(user_data, request_string, &mut out) };
    unsafe { native_string_free(request_string) };
    if status != NemoRelayStatus::Ok {
        if !out.is_null() {
            unsafe { native_string_free(out) };
        }
        return Err(flow_error_from_status(
            status,
            "native LLM request callback failed",
        ));
    }
    let result_json =
        take_json_from_native_string(out, "native LLM request callback returned null")?;
    serde_json::from_value(result_json)
        .map_err(|err| FlowError::Internal(format!("invalid LLM request JSON: {err}")))
}

fn wrap_json_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeJsonCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmSanitizeResponseFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |payload| {
        clear_native_last_error();
        let payload_string = native_string_from_json(&payload);
        let Some(payload_string) = payload_string else {
            return Json::Null;
        };
        let mut out = ptr::null_mut();
        let status = unsafe { cb(user_data.ptr, payload_string, &mut out) };
        unsafe { native_string_free(payload_string) };
        if status != NemoRelayStatus::Ok {
            if !out.is_null() {
                unsafe { native_string_free(out) };
            }
            return Json::Null;
        }
        take_json_from_native_string(out, "native JSON callback returned null")
            .unwrap_or(Json::Null)
    })
}

fn wrap_llm_conditional_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmConditionalCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmConditionalFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |request| {
        clear_native_last_error();
        let request_json = serde_json::to_value(request).map_err(|err| {
            FlowError::Internal(format!("failed to serialize LLM request: {err}"))
        })?;
        let request_string = native_string_from_json(&request_json)
            .ok_or_else(|| FlowError::Internal("failed to allocate native LLM request".into()))?;
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
}

fn wrap_llm_request_intercept_fn(
    instance: Arc<NativePluginInstance>,
    cb: NemoRelayNativeLlmRequestInterceptCb,
    user_data: *mut c_void,
    free_fn: NemoRelayNativeFreeFn,
) -> LlmRequestInterceptFn {
    let user_data = make_user_data(instance, user_data, free_fn);
    Arc::new(move |name, request, annotated| {
        clear_native_last_error();
        let name_string = native_string_from_str(name)
            .ok_or_else(|| FlowError::Internal("failed to allocate native name".into()))?;
        let request_json = serde_json::to_value(&request).map_err(|err| {
            FlowError::Internal(format!("failed to serialize LLM request: {err}"))
        })?;
        let request_string = native_string_from_json(&request_json)
            .ok_or_else(|| FlowError::Internal("failed to allocate native LLM request".into()))?;
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
    let result = spawn_with_current_scope(move || native_runtime().block_on(next(request))).join();
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
    let result = spawn_with_current_scope(move || native_runtime().block_on(next(request))).join();
    match result {
        Ok(Ok(stream)) => {
            unsafe { *out_stream = relay_stream_to_native_stream(stream) };
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
    Ok(Box::pin(NativeRelayLlmStream::from_raw(
        raw,
        next_ctx,
        callback_user_data,
    )?) as LlmJsonStream)
}

fn drop_native_stream(mut raw: NemoRelayNativeLlmStreamV1) {
    if let Some(drop_fn) = raw.drop.take() {
        unsafe { drop_fn(raw.user_data) };
    }
}

struct NativeHostLlmStream {
    stream: Arc<Mutex<Option<LlmJsonStream>>>,
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

fn relay_stream_to_native_stream(stream: LlmJsonStream) -> NemoRelayNativeLlmStreamV1 {
    let state = Box::new(NativeHostLlmStream {
        stream: Arc::new(Mutex::new(Some(stream))),
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
    let result = spawn_with_current_scope(move || {
        native_runtime().block_on(async move {
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
