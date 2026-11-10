// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, bounded synchronous diagnostics for native v1.0.
/// Allocation-free receipt written only after acceptance.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DiagnosticReceipt {
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
pub struct DiagnosticCapabilities {
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

/// Borrowed code declaration; spans are valid only for this synchronous call.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DiagnosticCode<'a> {
    /// Frozen built-in/extension classification.
    pub kind: u32,
    /// Frozen outcome tag.
    pub outcome: u32,
    /// Canonical built-in or local extension name.
    pub name: crate::borrow::Str<'a>,
    /// Declaration-only safe description.
    pub description: crate::borrow::Str<'a>,
    /// Canonical failure class names.
    pub classes: crate::borrow::Slice<'a, crate::borrow::Str<'a>>,
}
/// One complete component declaration.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DiagnosticRegistration<'a> {
    /// Component-local key.
    pub name: crate::borrow::Str<'a>,
    /// Complete bounded vocabulary.
    pub codes: crate::borrow::Slice<'a, DiagnosticCode<'a>>,
}
/// Borrowed observation; the receiver retains only bounded copies.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DiagnosticObservation<'a> {
    /// One-based declared code handle.
    pub code: u32,
    /// Frozen outcome tag.
    pub outcome: u32,
    /// Frozen severity tag.
    pub severity: u32,
    /// Positive occurrence delta.
    pub count: u64,
    /// Canonical failure class or empty.
    pub failure_class: crate::borrow::Str<'a>,
    /// Restricted evidence, never an aggregation key.
    pub detail: crate::borrow::Str<'a>,
    /// Optional correlation; empty means absent.
    pub request_id: crate::borrow::Str<'a>,
    /// Optional correlation; empty means absent.
    pub session_id: crate::borrow::Str<'a>,
}
/// Component-owned authority. Status codes preserve the diagnostic error vocabulary.
pub trait Diagnostics: Send + Sync {
    /// Non-sensitive limits and the host-selected whole-context read grant.
    fn capabilities(&self) -> std::result::Result<DiagnosticCapabilities, u32>;
    /// Admit one bounded component and all code metadata together.
    fn register(&self, registration: DiagnosticRegistration<'_>) -> std::result::Result<u64, u32>;
    /// Submit once; an uncertain error must not be automatically retried.
    fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
    ) -> std::result::Result<DiagnosticReceipt, u32>;
    /// Close a declaration without changing source-wide totals.
    fn close(&self, token: u64) -> std::result::Result<(), u32>;
    /// Copy a canonical read into caller-owned storage; no partial output is published.
    fn read(
        &self,
        kind: u32,
        query: crate::borrow::Str<'_>,
        output: crate::borrow::SliceMut<'_, u8>,
    ) -> std::result::Result<usize, u32>;
}
/// Owned revocable diagnostic capability.
pub use crate::handles::DiagnosticsBox;
crate::borrow::borrowed!(DiagnosticCode);
crate::borrow::borrowed!(DiagnosticRegistration);
crate::borrow::borrowed!(DiagnosticObservation);
