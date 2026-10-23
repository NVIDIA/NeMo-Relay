// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Append-only native diagnostic ABI. All input spans are borrowed for one call.
//! Nonempty spans and output pointers must be valid for their stated size; arrays
//! must be aligned. No callback retains plugin memory. Integer tags are validated.

use crate::{NemoRelayNativeHostApiV8, NemoRelayNativePluginRuntime};

/// Independent diagnostic transport version; execution ABI remains 8.
pub const CONTRACT_VERSION: u32 = 1;
/// Maximum simultaneously live component registrations per activation.
pub const MAX_COMPONENTS: usize = 8;
/// Maximum declared codes per component.
pub const MAX_CODES: usize = 32;
/// Maximum cumulative extension declarations per activation.
pub const MAX_EXTENSIONS: usize = 16;
/// Maximum classes per extension.
pub const MAX_CLASSES: usize = 16;
/// Maximum total text bytes in a native request.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
/// Maximum concurrently admitted authoring sessions per host process.
pub const MAX_SESSIONS: usize = 32;
/// Maximum concurrent native data calls per host process.
pub const MAX_CALLS: usize = 4;

/// Borrowed bytes. A null pointer is permitted only for an empty span.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Bytes {
    /// Readable byte address.
    pub data: *const u8,
    /// Length in bytes, not characters.
    pub len: usize,
}
impl Bytes {
    /// Borrow UTF-8 for the duration of a host call.
    pub fn text(value: &str) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// One declaration. Kind 0 selects a built-in; kind 1 declares an extension.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Code {
    /// 0 = built-in, 1 = extension.
    pub kind: u32,
    /// 0 = unspecified, 1 = success, 2 = failure (extension only).
    pub outcome: u32,
    /// Built-in canonical name or local extension name.
    pub name: Bytes,
    /// Extension description; empty for a built-in.
    pub description: Bytes,
    /// Array of canonical failure-class names.
    pub classes: *const Bytes,
    /// Number of class spans.
    pub class_count: usize,
}
/// Borrowed registration, whose complete vocabulary is admitted together.
#[repr(C)]
pub struct Registration {
    /// Stable local component name.
    pub name: Bytes,
    /// Code array; resulting code handles are one-based positions.
    pub codes: *const Code,
    /// Number of codes.
    pub code_count: usize,
}
/// Borrowed observation. Empty optional strings mean absent values.
#[repr(C)]
pub struct Observation {
    /// One-based code position in the declaration.
    pub code: u32,
    /// 0 = unspecified, 1 = success, 2 = failure.
    pub outcome: u32,
    /// 0 = trace, 1 = debug, 2 = info, 3 = warning, 4 = error.
    pub severity: u32,
    /// Positive occurrence delta.
    pub count: u64,
    /// Canonical class name, or empty.
    pub failure_class: Bytes,
    /// Diagnostic-only UTF-8 detail. Core reports retention truncation.
    pub detail: Bytes,
    /// Optional correlation; never an aggregation key.
    pub request_id: Bytes,
    /// Optional correlation; never an aggregation key.
    pub session_id: Bytes,
}
/// Allocation-free receipt written only after acceptance.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Receipt {
    /// Source-local recording sequence.
    pub sequence: u64,
    /// Updated occurrence total if present.
    pub total: u64,
    /// Boolean 0/1 indicating a code total.
    pub total_present: u32,
    /// Boolean 0/1 indicating total saturation.
    pub saturated: u32,
    /// 0 = retained, 1 = truncated, 2 = dropped evidence.
    pub evidence: u32,
}
/// Non-sensitive recording capabilities; no source-wide diagnostic content.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Capabilities {
    /// Diagnostic transport version.
    pub contract_version: u32,
    /// 0 = disabled, 1 = available, 2 = unavailable.
    pub availability: u32,
    /// Explicit host read grant as boolean 0/1.
    pub can_read_context: u32,
    /// Maximum simultaneously live component registrations per activation.
    pub max_components: u32,
    /// Maximum codes per component.
    pub max_codes: u32,
    /// Maximum cumulative extension declarations per activation.
    pub max_extensions: u32,
    /// Maximum aggregate request text bytes.
    pub max_request_bytes: u32,
    /// Maximum retained detail bytes.
    pub max_detail_bytes: u32,
    /// Maximum retained correlation bytes per field.
    pub max_correlation_bytes: u32,
    /// Maximum serialized canonical response bytes.
    pub max_response_bytes: usize,
    /// Maximum query bytes, independent of registration requests.
    pub max_query_bytes: usize,
    /// Process-wide session ceiling.
    pub max_sessions: u32,
    /// Process-wide data-call ceiling.
    pub max_calls: u32,
    /// Maximum admitted native session backing, excluding core and SDK memory.
    pub adapter_max_session_bytes: usize,
    /// Session backing currently held by this activation, excluding core storage.
    pub adapter_session_bytes: usize,
    /// Fixed registration parsing arrays on the host stack; excludes call frames
    /// and the separately bounded canonical query decoder.
    pub adapter_registration_scratch_bytes: usize,
}
/// Frozen diagnostics callback table. Status 0 means success; other values are
/// defined by `DiagnosticError::from_status`. Read kinds: 0 discovery, 1 snapshot,
/// 2 memory. Read output is caller-owned; `written` is zero on any failure.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NativeDiagnosticsApiV1 {
    /// Actual size of this table.
    pub struct_size: usize,
    /// Independent diagnostic transport version.
    pub contract_version: u32,
    /// Read non-sensitive capabilities for a live activation.
    pub capabilities:
        unsafe extern "C" fn(*const NemoRelayNativePluginRuntime, *mut Capabilities) -> u32,
    /// Register one bounded component; output token is zero on failure.
    pub register: unsafe extern "C" fn(
        *const NemoRelayNativePluginRuntime,
        *const Registration,
        *mut u64,
    ) -> u32,
    /// Record once; a failed return must not be automatically retried.
    pub record: unsafe extern "C" fn(
        *const NemoRelayNativePluginRuntime,
        u64,
        *const Observation,
        *mut Receipt,
    ) -> u32,
    /// Close a component without requiring a data-call permit.
    pub close: unsafe extern "C" fn(*const NemoRelayNativePluginRuntime, u64) -> u32,
    /// Copy a canonical page into a caller-owned destination.
    pub read: unsafe extern "C" fn(
        *const NemoRelayNativePluginRuntime,
        u32,
        Bytes,
        *mut u8,
        usize,
        *mut usize,
    ) -> u32,
}
/// Size-discovered tail of the unchanged ABI-8 execution table.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NativeHostApiV8WithDiagnostics {
    /// Frozen ABI-8 prefix, including every existing callback.
    pub v8: NemoRelayNativeHostApiV8,
    /// Immutable host-lifetime diagnostic table; null means unsupported.
    pub diagnostics_api: *const NativeDiagnosticsApiV1,
}
// SAFETY: these immutable tables reference only host-lifetime thread-safe callbacks.
unsafe impl Send for NativeHostApiV8WithDiagnostics {}
unsafe impl Sync for NativeHostApiV8WithDiagnostics {}
