// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native diagnostic authoring and explicitly authorized canonical reads.
//! Recording is synchronous, state-only, and independent of operational logs.
//! Disabled/unsupported facades skip factories. Diagnostic errors must not be
//! propagated as plugin execution failures unless the application chooses so.

pub mod abi;
use crate::PluginRuntime;
use abi::{Bytes, NativeDiagnosticsApiV1};
pub use nemo_relay_types::diagnostics::{
    BuiltinDiagnosticCode, Classification, ClassificationError, CommonPluginFailureCode,
    DiagnosticAvailability, DiagnosticClassifier, DiagnosticCorrelation, DiagnosticCount,
    DiagnosticDiscovery, DiagnosticMemoryPage, DiagnosticOutcome, DiagnosticQuery,
    DiagnosticReadReason, DiagnosticSeverity, DiagnosticSnapshot, EvidenceDisposition,
    FailureClass, FailureCode, PluginFailureCode, RecordOutcome, RecordReceipt,
};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

/// Diagnostic operation error. These values never classify plugin execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticError {
    /// Host has no compatible native diagnostics extension.
    Unsupported,
    /// Diagnostic adapter backing could not be admitted.
    Unavailable,
    /// Activation or registration was closed.
    Closed,
    /// Host did not grant whole-context read access.
    PermissionDenied,
    /// Re-entrant call or exhausted bounded admission; no automatic retry occurs.
    Busy,
    /// Registration or request exceeds its capacity.
    Capacity,
    /// Input or handle is invalid.
    Invalid,
    /// A local component name is already registered.
    Conflict,
    /// Caller-owned output buffer is too small.
    BufferTooSmall,
    /// Internal adapter failure; acceptance is not safe to retry.
    Internal,
    /// Canonical core read reason.
    Read(DiagnosticReadReason),
}
impl DiagnosticError {
    /// Stable bounded error label without producer-supplied text.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Unavailable => "unavailable",
            Self::Closed => "closed",
            Self::PermissionDenied => "permission_denied",
            Self::Busy => "busy",
            Self::Capacity => "capacity",
            Self::Invalid => "invalid",
            Self::Conflict => "conflict",
            Self::BufferTooSmall => "buffer_too_small",
            Self::Internal => "internal",
            Self::Read(reason) => reason.as_str(),
        }
    }
    /// Convert the frozen native status representation; unknown values fail closed.
    pub fn from_status(status: u32) -> Self {
        match status {
            2 => Self::Unsupported,
            3 => Self::Unavailable,
            4 => Self::Closed,
            5 => Self::PermissionDenied,
            6 => Self::Busy,
            7 => Self::Capacity,
            8 => Self::Invalid,
            9 => Self::Conflict,
            10 => Self::BufferTooSmall,
            100 => Self::Read(DiagnosticReadReason::Disabled),
            101 => Self::Read(DiagnosticReadReason::QueryDisabled),
            102 => Self::Read(DiagnosticReadReason::UnsupportedSchema),
            103 => Self::Read(DiagnosticReadReason::InvalidQuery),
            104 => Self::Read(DiagnosticReadReason::InvalidCursor),
            105 => Self::Read(DiagnosticReadReason::Busy),
            106 => Self::Read(DiagnosticReadReason::SnapshotCapacity),
            107 => Self::Read(DiagnosticReadReason::InvalidLimits),
            108 => Self::Read(DiagnosticReadReason::AllocationFailed),
            109 => Self::Read(DiagnosticReadReason::Internal),
            _ => Self::Internal,
        }
    }
    /// Frozen native representation; 0 and 1 are reserved for accepted/disabled.
    pub fn status(self) -> u32 {
        match self {
            Self::Unsupported => 2,
            Self::Unavailable => 3,
            Self::Closed => 4,
            Self::PermissionDenied => 5,
            Self::Busy => 6,
            Self::Capacity => 7,
            Self::Invalid => 8,
            Self::Conflict => 9,
            Self::BufferTooSmall => 10,
            Self::Internal => 11,
            Self::Read(reason) => match reason {
                DiagnosticReadReason::Disabled => 100,
                DiagnosticReadReason::QueryDisabled => 101,
                DiagnosticReadReason::UnsupportedSchema => 102,
                DiagnosticReadReason::InvalidQuery => 103,
                DiagnosticReadReason::InvalidCursor => 104,
                DiagnosticReadReason::Busy => 105,
                DiagnosticReadReason::SnapshotCapacity => 106,
                DiagnosticReadReason::InvalidLimits => 107,
                DiagnosticReadReason::AllocationFailed => 108,
                _ => 109,
            },
        }
    }
}
impl std::fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.reason())
    }
}
impl std::error::Error for DiagnosticError {}
type Result<T> = std::result::Result<T, DiagnosticError>;

/// One declared code. Extension namespace and component ownership are host-assigned.
#[derive(Clone, Copy)]
pub enum DiagnosticCode<'a> {
    /// Existing semantic code; host restricts plugin authors to common plugin failures.
    Builtin(BuiltinDiagnosticCode),
    /// A distinct plugin-internal operation.
    Extension {
        /// Stable local name (at most 32 UTF-8 bytes).
        name: &'a str,
        /// Safe metadata retained once (at most 256 UTF-8 bytes).
        description: &'a str,
        /// Declared outcome; success is independent evidence, not recovery.
        outcome: DiagnosticOutcome,
        /// Allowed causes; empty for a nonfailure extension.
        allowed_classes: &'a [FailureClass],
    },
}
/// Borrowed component declaration evaluated only when collection is available.
pub struct ComponentDeclaration<'a> {
    /// Stable name, excluding request/session identity.
    pub name: &'a str,
    /// Complete bounded vocabulary; code handles are one-based positions.
    pub codes: &'a [DiagnosticCode<'a>],
}
/// Typed observation with borrowed detail and shared caller correlation.
pub struct Observation<'a> {
    /// One-based code position returned by registration.
    pub code: u32,
    /// Observed outcome, checked against the declaration.
    pub outcome: DiagnosticOutcome,
    /// Severity is independent of outcome.
    pub severity: DiagnosticSeverity,
    /// Evidence-based cause, never inferred from text.
    pub failure_class: Option<FailureClass>,
    /// Positive input delta, not an accumulated total.
    pub count: NonZeroU64,
    /// Restricted diagnostic-only detail; core reports any truncation.
    pub detail: &'a str,
    /// Optional request/session identifiers, excluded from aggregation keys.
    pub correlation: DiagnosticCorrelation<'a>,
}
impl<'a> Observation<'a> {
    /// Author a failed internal operation with error severity.
    pub fn failure(code: u32, class: FailureClass, detail: &'a str, count: NonZeroU64) -> Self {
        Self {
            code,
            outcome: DiagnosticOutcome::Failure,
            severity: DiagnosticSeverity::Error,
            failure_class: Some(class),
            count,
            detail,
            correlation: DiagnosticCorrelation::default(),
        }
    }
    /// Author independent successful evidence.
    pub fn success(code: u32, detail: &'a str, count: NonZeroU64) -> Self {
        Self {
            code,
            outcome: DiagnosticOutcome::Success,
            severity: DiagnosticSeverity::Info,
            failure_class: None,
            count,
            detail,
            correlation: DiagnosticCorrelation::default(),
        }
    }
}
/// Local delivery totals, not core observation or retention totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeliveryStats {
    /// Local authoring admission failures before registration/recording reaches the host.
    pub dropped: u64,
    /// Explicit host refusals of registration or recording.
    pub rejected: u64,
    /// Internal authoring failures whose acceptance is unsafe to retry.
    pub uncertain: u64,
}
// SDK admission precedes producer factories, scratch arrays, and read buffers.
// These limits are independent of the host's process-wide data-call permits.
const MAX_SDK_CALLS: usize = abi::MAX_CALLS;
const MAX_NESTED_ACTIVATIONS: usize = abi::MAX_SESSIONS;
thread_local! {
    static ACTIVE_CALLS: RefCell<[usize; MAX_NESTED_ACTIVATIONS]> =
        const { RefCell::new([0; MAX_NESTED_ACTIVATIONS]) };
}

#[derive(Default)]
pub(crate) struct SdkState {
    active_calls: AtomicUsize,
    reading: AtomicBool,
    dropped: AtomicU64,
    rejected: AtomicU64,
    uncertain: AtomicU64,
}
fn increment(value: &AtomicU64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_add(1))
    });
}
struct CallGuard<'a> {
    state: &'a SdkState,
    slot: usize,
    // Thread-local entries must be removed on the thread that inserted them.
    _thread: PhantomData<Rc<()>>,
}
impl Drop for CallGuard<'_> {
    fn drop(&mut self) {
        let _ = ACTIVE_CALLS.try_with(|keys| keys.borrow_mut()[self.slot] = 0);
        self.state.active_calls.fetch_sub(1, Ordering::Release);
    }
}
struct ReadGuard<'a>(&'a AtomicBool);
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Cloneable activation-scoped diagnostic facade. It does not start a service.
/// Clones share four concurrent-call slots and one in-flight read-buffer slot.
/// Same-thread recursion into the same activation returns [`DiagnosticError::Busy`].
#[derive(Clone)]
pub struct Diagnostics {
    runtime: PluginRuntime,
}
impl Diagnostics {
    pub(crate) fn new(runtime: PluginRuntime) -> Self {
        Self { runtime }
    }
    fn api(&self) -> Result<NativeDiagnosticsApiV1> {
        let api = self
            .runtime
            .diagnostics_api
            .ok_or(DiagnosticError::Unsupported)?;
        if self.runtime.capability.is_null() {
            return Err(DiagnosticError::Closed);
        }
        Ok(api)
    }
    /// Non-sensitive limits and the separately resolved read grant.
    pub fn capabilities(&self) -> Result<abi::Capabilities> {
        let api = self.api()?;
        let mut result = abi::Capabilities::default();
        let status = unsafe { (api.capabilities)(self.runtime.capability, &mut result) };
        if status != 0 {
            return Err(DiagnosticError::from_status(status));
        }
        if result.contract_version != abi::CONTRACT_VERSION
            || result.availability > 2
            || result.can_read_context > 1
            || result.max_codes as usize > abi::MAX_CODES
            || result.max_request_bytes as usize > abi::MAX_REQUEST_BYTES
        {
            return Err(DiagnosticError::Unsupported);
        }
        Ok(result)
    }
    /// Live host availability; startup enablement remains host-owned.
    pub fn availability(&self) -> DiagnosticAvailability {
        match self.capabilities() {
            Ok(c) if c.availability == 1 => DiagnosticAvailability::Available,
            Ok(c) if c.availability == 0 => DiagnosticAvailability::Disabled,
            Err(DiagnosticError::Unsupported) => DiagnosticAvailability::Unsupported,
            _ => DiagnosticAvailability::Unavailable,
        }
    }
    fn inactive(&self) -> Result<bool> {
        match self.capabilities() {
            Ok(c) if c.availability == 0 => Ok(true),
            Ok(c) if c.availability == 1 => Ok(false),
            Err(DiagnosticError::Unsupported) => Ok(true),
            Err(error) => Err(error),
            _ => Err(DiagnosticError::Unavailable),
        }
    }
    fn enter(&self) -> Result<CallGuard<'_>> {
        let state = &self.runtime.diagnostic_state;
        let key = Arc::as_ptr(state) as usize;
        ACTIVE_CALLS
            .try_with(|keys| {
                let mut keys = keys.borrow_mut();
                if keys.contains(&key) {
                    return Err(DiagnosticError::Busy);
                }
                let slot = keys
                    .iter()
                    .position(|key| *key == 0)
                    .ok_or(DiagnosticError::Busy)?;
                state
                    .active_calls
                    .fetch_update(Ordering::Acquire, Ordering::Relaxed, |n| {
                        (n < MAX_SDK_CALLS).then_some(n + 1)
                    })
                    .map_err(|_| DiagnosticError::Busy)?;
                keys[slot] = key;
                Ok(CallGuard {
                    state,
                    slot,
                    _thread: PhantomData,
                })
            })
            .unwrap_or(Err(DiagnosticError::Busy))
    }
    fn refused_locally(&self, error: DiagnosticError) -> DiagnosticError {
        increment(&self.runtime.diagnostic_state.dropped);
        error
    }
    fn returned(&self, status: u32) -> DiagnosticError {
        let error = DiagnosticError::from_status(status);
        increment(if error == DiagnosticError::Internal {
            &self.runtime.diagnostic_state.uncertain
        } else {
            &self.runtime.diagnostic_state.rejected
        });
        error
    }
    /// Fixed local delivery totals shared by runtime/component clones.
    pub fn delivery_stats(&self) -> DeliveryStats {
        let s = &self.runtime.diagnostic_state;
        DeliveryStats {
            dropped: s.dropped.load(Ordering::Relaxed),
            rejected: s.rejected.load(Ordering::Relaxed),
            uncertain: s.uncertain.load(Ordering::Relaxed),
        }
    }
    /// Admit a component and its vocabulary, skipping the factory when inert.
    pub fn component_lazy<'a>(
        &self,
        make: impl FnOnce() -> ComponentDeclaration<'a>,
    ) -> Result<DiagnosticComponent> {
        if self.inactive().map_err(|e| self.refused_locally(e))? {
            return Ok(DiagnosticComponent::new(self.clone(), 0));
        }
        let _guard = self.enter().map_err(|e| self.refused_locally(e))?;
        let declaration = make();
        if declaration.codes.is_empty() {
            increment(&self.runtime.diagnostic_state.dropped);
            return Err(DiagnosticError::Invalid);
        }
        if declaration.codes.len() > abi::MAX_CODES {
            increment(&self.runtime.diagnostic_state.dropped);
            return Err(DiagnosticError::Capacity);
        }
        let mut codes = [abi::Code::default(); abi::MAX_CODES];
        let mut classes = [[Bytes::default(); abi::MAX_CLASSES]; abi::MAX_CODES];
        let mut bytes = declaration.name.len();
        for (index, definition) in declaration.codes.iter().enumerate() {
            let output = &mut codes[index];
            match definition {
                DiagnosticCode::Builtin(code) => {
                    output.name = Bytes::text(code.canonical_name());
                }
                DiagnosticCode::Extension {
                    name,
                    description,
                    outcome,
                    allowed_classes,
                } => {
                    if allowed_classes.len() > abi::MAX_CLASSES {
                        increment(&self.runtime.diagnostic_state.dropped);
                        return Err(DiagnosticError::Capacity);
                    }
                    output.kind = 1;
                    output.outcome = outcome_tag(*outcome);
                    output.name = Bytes::text(name);
                    output.description = Bytes::text(description);
                    for (slot, class) in classes[index].iter_mut().zip(*allowed_classes) {
                        *slot = Bytes::text(class.as_str());
                    }
                    output.classes = classes[index].as_ptr();
                    output.class_count = allowed_classes.len();
                    for class in *allowed_classes {
                        bytes = bytes.saturating_add(class.as_str().len());
                    }
                }
            }
            bytes = bytes
                .saturating_add(output.name.len)
                .saturating_add(output.description.len);
        }
        if bytes > abi::MAX_REQUEST_BYTES {
            increment(&self.runtime.diagnostic_state.dropped);
            return Err(DiagnosticError::Capacity);
        }
        let raw = abi::Registration {
            name: Bytes::text(declaration.name),
            codes: codes.as_ptr(),
            code_count: declaration.codes.len(),
        };
        let mut handle = 0;
        let status = unsafe { (self.api()?.register)(self.runtime.capability, &raw, &mut handle) };
        match status {
            0 if handle != 0 => Ok(DiagnosticComponent::new(self.clone(), handle)),
            1 => Ok(DiagnosticComponent::new(self.clone(), 0)),
            _ => Err(self.returned(status)),
        }
    }
    /// Obtain a reader only when the host explicitly grants whole-context access.
    pub fn reader(&self) -> Result<DiagnosticReader> {
        if self.capabilities()?.can_read_context != 1 {
            return Err(DiagnosticError::PermissionDenied);
        }
        Ok(DiagnosticReader {
            diagnostics: self.clone(),
        })
    }
}
struct ComponentInner {
    diagnostics: Diagnostics,
    handle: u64,
    closed: AtomicBool,
}
impl ComponentInner {
    fn close(&self) -> Result<()> {
        if self.handle == 0 || self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let api = self.diagnostics.api()?;
        let status = unsafe { (api.close)(self.diagnostics.runtime.capability, self.handle) };
        if status == 0 || status == 4 {
            Ok(())
        } else {
            Err(DiagnosticError::from_status(status))
        }
    }
}
impl Drop for ComponentInner {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
/// Shared producer registration. Last drop closes it; explicit close fences clones.
#[derive(Clone)]
pub struct DiagnosticComponent {
    inner: Arc<ComponentInner>,
}
impl DiagnosticComponent {
    fn new(diagnostics: Diagnostics, handle: u64) -> Self {
        Self {
            inner: Arc::new(ComponentInner {
                diagnostics,
                handle,
                closed: AtomicBool::new(false),
            }),
        }
    }
    /// Record an occurrence delta once. No retry, log, or exporter projection.
    pub fn try_record_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome> {
        if self.inner.handle == 0 {
            return Ok(RecordOutcome::Disabled);
        }
        if self.inner.closed.load(Ordering::Acquire) {
            increment(&self.inner.diagnostics.runtime.diagnostic_state.dropped);
            return Err(DiagnosticError::Closed);
        }
        let d = &self.inner.diagnostics;
        let _guard = d.enter().map_err(|e| d.refused_locally(e))?;
        let value = make();
        let class = value.failure_class.map_or("", FailureClass::as_str);
        let request = value.correlation.request_id.as_deref().unwrap_or("");
        let session = value.correlation.session_id.as_deref().unwrap_or("");
        if [
            value.detail.len(),
            request.len(),
            session.len(),
            class.len(),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
            > abi::MAX_REQUEST_BYTES
        {
            increment(&d.runtime.diagnostic_state.dropped);
            return Err(DiagnosticError::Capacity);
        }
        let raw = abi::Observation {
            code: value.code,
            outcome: outcome_tag(value.outcome),
            severity: match value.severity {
                DiagnosticSeverity::Trace => 0,
                DiagnosticSeverity::Debug => 1,
                DiagnosticSeverity::Info => 2,
                DiagnosticSeverity::Warning => 3,
                DiagnosticSeverity::Error => 4,
            },
            count: value.count.get(),
            failure_class: Bytes::text(class),
            detail: Bytes::text(value.detail),
            request_id: Bytes::text(request),
            session_id: Bytes::text(session),
        };
        let mut receipt = abi::Receipt::default();
        let status = unsafe {
            (d.api()?.record)(d.runtime.capability, self.inner.handle, &raw, &mut receipt)
        };
        match status {
            0 => {
                let evidence = match receipt.evidence {
                    0 => EvidenceDisposition::Retained,
                    1 => EvidenceDisposition::Truncated,
                    2 => EvidenceDisposition::Dropped,
                    _ => return Err(d.returned(11)),
                };
                if receipt.sequence == 0 || receipt.total_present > 1 || receipt.saturated > 1 {
                    return Err(d.returned(11));
                }
                Ok(RecordOutcome::Recorded(RecordReceipt {
                    sequence: receipt.sequence,
                    evidence,
                    code_total: (receipt.total_present == 1).then_some(DiagnosticCount {
                        value: receipt.total.into(),
                        saturated: receipt.saturated == 1,
                    }),
                }))
            }
            1 => Ok(RecordOutcome::Disabled),
            _ => Err(d.returned(status)),
        }
    }
    /// Close explicitly; this never resets source totals or classifies recovery.
    pub fn close(&self) -> Result<()> {
        self.inner.close()
    }
}
fn outcome_tag(outcome: DiagnosticOutcome) -> u32 {
    match outcome {
        DiagnosticOutcome::Unknown => 0,
        DiagnosticOutcome::Success => 1,
        DiagnosticOutcome::Failure => 2,
    }
}

/// Authorized read facade; each call independently checks live host authority.
#[derive(Clone)]
pub struct DiagnosticReader {
    diagnostics: Diagnostics,
}
impl DiagnosticReader {
    /// Canonical discovery for the selected source.
    pub fn discovery(&self) -> Result<DiagnosticDiscovery> {
        self.decode(0, &DiagnosticQuery::default())
    }
    /// Independently owned canonical snapshot, outside the host memory ceiling.
    pub fn snapshot(&self, query: &DiagnosticQuery) -> Result<DiagnosticSnapshot> {
        self.decode(1, query)
    }
    /// Independently owned canonical memory page; starts no resource collector.
    pub fn inspect_memory(&self, query: &DiagnosticQuery) -> Result<DiagnosticMemoryPage> {
        self.decode(2, query)
    }
    /// Complete canonical JSON snapshot, preserving exact decimal counter strings.
    pub fn snapshot_json(&self, query: &DiagnosticQuery) -> Result<String> {
        self.read(1, query)
    }
    /// Complete canonical JSON memory page.
    pub fn inspect_memory_json(&self, query: &DiagnosticQuery) -> Result<String> {
        self.read(2, query)
    }
    fn decode<T: serde::de::DeserializeOwned>(
        &self,
        kind: u32,
        query: &DiagnosticQuery,
    ) -> Result<T> {
        serde_json::from_str(&self.read(kind, query)?).map_err(|_| DiagnosticError::Internal)
    }
    fn read(&self, kind: u32, query: &DiagnosticQuery) -> Result<String> {
        let d = &self.diagnostics;
        let _guard = d.enter()?;
        let reading = &d.runtime.diagnostic_state.reading;
        if reading.swap(true, Ordering::Acquire) {
            return Err(DiagnosticError::Busy);
        }
        let _read = ReadGuard(reading);
        let caps = d.capabilities()?;
        let mut encoded = BoundedBuffer::new(caps.max_query_bytes)?;
        serde_json::to_writer(&mut encoded, query)
            .map_err(|_| DiagnosticError::Read(DiagnosticReadReason::InvalidQuery))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(caps.max_response_bytes)
            .map_err(|_| DiagnosticError::Read(DiagnosticReadReason::AllocationFailed))?;
        output.resize(caps.max_response_bytes, 0);
        let mut written = 0;
        let status = unsafe {
            (d.api()?.read)(
                d.runtime.capability,
                kind,
                Bytes {
                    data: encoded.bytes.as_ptr(),
                    len: encoded.bytes.len(),
                },
                output.as_mut_ptr(),
                output.len(),
                &mut written,
            )
        };
        if status != 0 {
            return Err(DiagnosticError::from_status(status));
        }
        if written > output.len() {
            return Err(DiagnosticError::Internal);
        }
        output.truncate(written);
        String::from_utf8(output).map_err(|_| DiagnosticError::Internal)
    }
}
struct BoundedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}
impl BoundedBuffer {
    fn new(limit: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(limit)
            .map_err(|_| DiagnosticError::Read(DiagnosticReadReason::AllocationFailed))?;
        Ok(Self { bytes, limit })
    }
}
impl std::io::Write for BoundedBuffer {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if data.len() > self.limit - self.bytes.len() {
            return Err(std::io::ErrorKind::OutOfMemory.into());
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
