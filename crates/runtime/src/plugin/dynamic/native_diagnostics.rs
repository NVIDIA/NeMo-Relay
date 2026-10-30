// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native diagnostic bridge: borrowed bounded inputs, one core recording boundary.

use super::{NativeHostPluginRuntime, NemoRelayNativePluginRuntime, native_runtime_ref};
use crate::diagnostics::*;
use nemo_relay_plugin::diagnostics::{DiagnosticError as Error, abi};
use std::num::NonZeroU64;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use uuid::Uuid;

type Result<T> = std::result::Result<T, Error>;
static SESSIONS: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static TOKENS: AtomicU64 = AtomicU64::new(0);
struct Permit(&'static AtomicUsize);
impl Permit {
    fn acquire(counter: &'static AtomicUsize, limit: usize) -> Result<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| {
                (n < limit).then_some(n + 1)
            })
            .map(|_| Self(counter))
            .map_err(|_| Error::Busy)
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One registration owner's authority. Revocation drops all strong core handles.
pub(super) struct NativeDiagnostics {
    state: Mutex<Option<Activation>>,
}
struct Activation {
    context: Diagnostics,
    read: bool,
    instance: Uuid,
    responsibility: [u8; 128],
    responsibility_len: usize,
    registering: bool,
    session: Option<Box<[Session]>>,
}
struct Session {
    _permit: Permit,
    components: [Option<Registered>; abi::MAX_COMPONENTS],
    extensions: usize,
    metadata: usize,
}
struct Registered {
    token: u64,
    key: [u8; 128],
    key_len: usize,
    owner: ComponentOwner,
    recorder: ComponentRecorder,
    codes: [Option<Code>; abi::MAX_CODES],
}
impl Drop for Registered {
    fn drop(&mut self) {
        let _ = self.owner.close();
    }
}
#[derive(Clone, Copy)]
enum Code {
    Builtin(BuiltinDiagnosticCode),
    Extension(ExtensionHandle, DiagnosticOutcome),
}
#[derive(Clone, Copy)]
enum Parsed<'a> {
    Empty,
    Builtin(BuiltinDiagnosticCode),
    Extension {
        name: &'a str,
        description: &'a str,
        outcome: DiagnosticOutcome,
        classes: [FailureClass; abi::MAX_CLASSES],
        class_count: usize,
    },
}
impl NativeDiagnostics {
    pub(super) fn new(context: Diagnostics, read: bool, responsibility: &str) -> Self {
        // Descriptive host attribution is bounded independently of local keys.
        let mut len = responsibility
            .len()
            .min(context.limits().max_identity_bytes)
            .min(128);
        while !responsibility.is_char_boundary(len) {
            len -= 1;
        }
        let mut name = [0; 128];
        name[..len].copy_from_slice(&responsibility.as_bytes()[..len]);
        Self {
            state: Mutex::new(Some(Activation {
                context,
                read,
                instance: Uuid::now_v7(),
                responsibility: name,
                responsibility_len: len,
                registering: true,
                session: None,
            })),
        }
    }
    pub(super) fn revoke(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
    pub(super) fn registered(&self) {
        if let Some(active) = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            active.registering = false;
        }
    }
    fn with<T>(&self, f: impl FnOnce(&mut Activation) -> Result<T>) -> Result<T> {
        let mut state = self.state.lock().map_err(|_| Error::Internal)?;
        f(state.as_mut().ok_or(Error::Closed)?)
    }
}
impl Activation {
    fn session(&mut self) -> Result<&mut Session> {
        if self.session.is_none() {
            let permit =
                Permit::acquire(&SESSIONS, abi::MAX_SESSIONS).map_err(|_| Error::Unavailable)?;
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(1)
                .map_err(|_| Error::Unavailable)?;
            storage.push(Session {
                _permit: permit,
                components: std::array::from_fn(|_| None),
                extensions: 0,
                metadata: 0,
            });
            self.session = Some(storage.into_boxed_slice());
        }
        Ok(&mut self.session.as_mut().expect("admitted session")[0])
    }
    fn capabilities(&self) -> abi::Capabilities {
        let limits = self.context.limits();
        abi::Capabilities {
            contract_version: abi::CONTRACT_VERSION,
            availability: u32::from(
                self.context.reader().availability() == DiagnosticAvailability::Available,
            ),
            can_read_context: u32::from(self.read),
            max_components: abi::MAX_COMPONENTS as u32,
            max_codes: abi::MAX_CODES as u32,
            max_extensions: abi::MAX_EXTENSIONS as u32,
            max_request_bytes: abi::MAX_REQUEST_BYTES as u32,
            max_detail_bytes: limits.max_detail_bytes as u32,
            max_correlation_bytes: limits.max_correlation_bytes as u32,
            max_response_bytes: limits.max_snapshot_bytes,
            max_query_bytes: MAX_DIAGNOSTIC_QUERY_BYTES,
            max_sessions: abi::MAX_SESSIONS as u32,
            max_calls: abi::MAX_CALLS as u32,
            adapter_max_session_bytes: abi::MAX_SESSIONS * std::mem::size_of::<Session>(),
            adapter_session_bytes: self
                .session
                .as_ref()
                .map_or(0, |_| std::mem::size_of::<Session>()),
            adapter_registration_scratch_bytes: std::mem::size_of::<[Parsed<'_>; abi::MAX_CODES]>()
                + std::mem::size_of::<[BuiltinDiagnosticCode; abi::MAX_CODES]>(),
        }
    }
}

fn guarded(f: impl FnOnce() -> Result<u32>) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or(Err(Error::Internal))
        .unwrap_or_else(Error::status)
}
unsafe fn runtime<'a>(
    runtime: *const NemoRelayNativePluginRuntime,
) -> Result<&'a NativeHostPluginRuntime> {
    native_runtime_ref(runtime).map_err(|_| Error::Closed)
}
unsafe fn text<'a>(value: abi::Bytes, budget: &mut usize) -> Result<&'a str> {
    if value.len > *budget || value.len > isize::MAX as usize {
        return Err(Error::Capacity);
    }
    *budget -= value.len;
    if value.len == 0 {
        return Ok("");
    }
    if value.data.is_null() {
        return Err(Error::Invalid);
    }
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(value.data, value.len) })
        .map_err(|_| Error::Invalid)
}
unsafe fn array<'a, T>(data: *const T, len: usize, cap: usize) -> Result<&'a [T]> {
    if len > cap {
        return Err(Error::Capacity);
    }
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() || !(data as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(Error::Invalid);
    }
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}
fn outcome(tag: u32) -> Result<DiagnosticOutcome> {
    match tag {
        0 => Ok(DiagnosticOutcome::Unknown),
        1 => Ok(DiagnosticOutcome::Success),
        2 => Ok(DiagnosticOutcome::Failure),
        _ => Err(Error::Invalid),
    }
}
fn core_error(error: DiagnosticError) -> Error {
    match error {
        DiagnosticError::ComponentCapacity
        | DiagnosticError::KeyCapacity
        | DiagnosticError::MetadataCapacity
        | DiagnosticError::WindowCapacity
        | DiagnosticError::SequenceExhausted => Error::Capacity,
        DiagnosticError::ComponentClosed => Error::Closed,
        DiagnosticError::Allocation => Error::Unavailable,
        DiagnosticError::InvalidComponentKey
        | DiagnosticError::InvalidCode
        | DiagnosticError::InvalidClassification
        | DiagnosticError::InvalidContext => Error::Invalid,
        _ => Error::Internal,
    }
}
unsafe extern "C" fn capabilities(
    runtime_ptr: *const NemoRelayNativePluginRuntime,
    out: *mut abi::Capabilities,
) -> u32 {
    guarded(|| {
        if out.is_null() {
            return Err(Error::Invalid);
        }
        unsafe {
            out.write(abi::Capabilities::default());
        }
        let runtime = unsafe { runtime(runtime_ptr)? };
        runtime.diagnostics.with(|a| {
            unsafe {
                out.write(a.capabilities());
            }
            Ok(0)
        })
    })
}
unsafe extern "C" fn register(
    runtime_ptr: *const NemoRelayNativePluginRuntime,
    input: *const abi::Registration,
    out: *mut u64,
) -> u32 {
    guarded(|| {
        if out.is_null() {
            return Err(Error::Invalid);
        }
        unsafe {
            out.write(0);
        }
        let runtime = unsafe { runtime(runtime_ptr)? };
        let _call = Permit::acquire(&CALLS, abi::MAX_CALLS)?;
        runtime.diagnostics.with(|a| {
            if a.context.reader().availability() == DiagnosticAvailability::Disabled {
                return Ok(1);
            }
            if input.is_null() {
                return Err(Error::Invalid);
            }
            let input = unsafe { &*input };
            let mut budget = abi::MAX_REQUEST_BYTES;
            let key = unsafe { text(input.name, &mut budget)? };
            if key.is_empty()
                || key.len() > a.context.limits().max_identity_bytes
                || key.len() > 128
            {
                return Err(Error::Invalid);
            }
            let ParsedDefinitions {
                codes: parsed,
                builtin,
                builtin_count,
                extension_count,
                count: definition_count,
            } = unsafe { parse_definitions(input, &mut budget)? };
            let metadata = abi::MAX_REQUEST_BYTES - budget;
            let context = a.context.clone();
            let instance = a.instance;
            let responsibility = (a.responsibility_len > 0).then(|| {
                std::str::from_utf8(&a.responsibility[..a.responsibility_len])
                    .expect("bounded host namespace")
                    .to_owned()
            });
            let session = a.session()?;
            if session
                .components
                .iter()
                .flatten()
                .any(|v| &v.key[..v.key_len] == key.as_bytes())
            {
                return Err(Error::Conflict);
            }
            let slot = session
                .components
                .iter()
                .position(Option::is_none)
                .ok_or(Error::Capacity)?;
            if session.extensions + extension_count > abi::MAX_EXTENSIONS
                || session.metadata + metadata > abi::MAX_REQUEST_BYTES
            {
                return Err(Error::Capacity);
            }
            let token = TOKENS
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| Error::Capacity)?
                + 1;
            let mut registration = ComponentRegistration::new(key, &builtin[..builtin_count]);
            registration.ownership.plugin_instance_id = Some(instance);
            registration.ownership.responsibility = responsibility;
            let (owner, recorder) = context
                .register_owned_component(registration)
                .map_err(core_error)?;
            let mut registered = Registered {
                token,
                key: [0; 128],
                key_len: key.len(),
                owner,
                recorder,
                codes: [None; abi::MAX_CODES],
            };
            registered.key[..key.len()].copy_from_slice(key.as_bytes());
            let mut namespace = [0u8; 39];
            namespace[..7].copy_from_slice(b"native_");
            instance.simple().encode_lower(&mut namespace[7..]);
            let namespace = std::str::from_utf8(&namespace).expect("UUID namespace");
            register_codes(&mut registered, &parsed[..definition_count], namespace)?;
            session.extensions += extension_count;
            session.metadata += metadata;
            session.components[slot] = Some(registered);
            unsafe {
                out.write(token);
            }
            Ok(0)
        })
    })
}
unsafe extern "C" fn record(
    runtime_ptr: *const NemoRelayNativePluginRuntime,
    token: u64,
    input: *const abi::Observation,
    out: *mut abi::Receipt,
) -> u32 {
    guarded(|| {
        if out.is_null() {
            return Err(Error::Invalid);
        }
        unsafe {
            out.write(abi::Receipt::default());
        }
        let runtime = unsafe { runtime(runtime_ptr)? };
        let _call = Permit::acquire(&CALLS, abi::MAX_CALLS)?;
        runtime.diagnostics.with(|a| {
            if a.context.reader().availability() == DiagnosticAvailability::Disabled {
                return Ok(1);
            }
            if input.is_null() {
                return Err(Error::Invalid);
            }
            let input = unsafe { &*input };
            let mut budget = abi::MAX_REQUEST_BYTES;
            let detail = unsafe { text(input.detail, &mut budget)? };
            let request = unsafe { text(input.request_id, &mut budget)? };
            let session_id = unsafe { text(input.session_id, &mut budget)? };
            let class = unsafe { text(input.failure_class, &mut budget)? };
            let class = if class.is_empty() {
                None
            } else {
                Some(FailureClass::from_canonical_name(class).ok_or(Error::Invalid)?)
            };
            let count = NonZeroU64::new(input.count).ok_or(Error::Invalid)?;
            let outcome = outcome(input.outcome)?;
            let component = a
                .session
                .as_ref()
                .and_then(|s| s[0].components.iter().flatten().find(|c| c.token == token))
                .ok_or(Error::Closed)?;
            let code = input
                .code
                .checked_sub(1)
                .and_then(|n| component.codes.get(n as usize))
                .and_then(|v| *v)
                .ok_or(Error::Invalid)?;
            let mut observation = code_observation(code, class, outcome, detail, count)?;
            observation.severity = match input.severity {
                0 => DiagnosticSeverity::Trace,
                1 => DiagnosticSeverity::Debug,
                2 => DiagnosticSeverity::Info,
                3 => DiagnosticSeverity::Warning,
                4 => DiagnosticSeverity::Error,
                _ => return Err(Error::Invalid),
            };
            observation.context = PluginObservationContext {
                execution: Some(PluginExecution::NativeAbi.into()),
                phase: Some(
                    if a.registering {
                        PluginPhase::Activation
                    } else {
                        PluginPhase::Runtime
                    }
                    .into(),
                ),
            };
            observation.correlation = DiagnosticCorrelation {
                request_id: (!request.is_empty()).then(|| request.into()),
                session_id: (!session_id.is_empty()).then(|| session_id.into()),
            };
            match component
                .recorder
                .record_without_projection(observation)
                .map_err(core_error)?
            {
                RecordOutcome::Disabled => Ok(1),
                RecordOutcome::Recorded(receipt) => {
                    let total = receipt.code_total.unwrap_or_default();
                    unsafe {
                        out.write(abi::Receipt {
                            sequence: receipt.sequence,
                            total: total.value.get(),
                            total_present: u32::from(receipt.code_total.is_some()),
                            saturated: u32::from(total.saturated),
                            evidence: match receipt.evidence {
                                EvidenceDisposition::Retained => 0,
                                EvidenceDisposition::Truncated => 1,
                                EvidenceDisposition::Dropped => 2,
                            },
                        });
                    }
                    Ok(0)
                }
            }
        })
    })
}
unsafe extern "C" fn close(runtime_ptr: *const NemoRelayNativePluginRuntime, token: u64) -> u32 {
    guarded(|| {
        let runtime = unsafe { runtime(runtime_ptr)? };
        runtime.diagnostics.with(|a| {
            let session = a.session.as_mut().ok_or(Error::Closed)?;
            let slot = session[0]
                .components
                .iter_mut()
                .find(|v| v.as_ref().is_some_and(|c| c.token == token))
                .ok_or(Error::Closed)?;
            slot.take();
            Ok(0)
        })
    })
}
struct Output<'a> {
    bytes: &'a mut [u8],
    written: usize,
}
impl std::io::Write for Output<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.bytes.len() - self.written {
            return Err(std::io::ErrorKind::OutOfMemory.into());
        }
        self.bytes[self.written..self.written + bytes.len()].copy_from_slice(bytes);
        self.written += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
unsafe extern "C" fn read(
    runtime_ptr: *const NemoRelayNativePluginRuntime,
    kind: u32,
    query: abi::Bytes,
    output: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> u32 {
    guarded(|| {
        if written.is_null() {
            return Err(Error::Invalid);
        }
        unsafe {
            written.write(0);
        }
        let runtime = unsafe { runtime(runtime_ptr)? };
        let _call = Permit::acquire(&CALLS, abi::MAX_CALLS)?;
        runtime.diagnostics.with(|a| {
            if !a.read {
                return Err(Error::PermissionDenied);
            }
            if a.context.reader().availability() == DiagnosticAvailability::Disabled {
                return Err(Error::Read(DiagnosticReadReason::Disabled));
            }
            if output.is_null() || capacity > isize::MAX as usize {
                return Err(Error::Invalid);
            }
            let mut budget = MAX_DIAGNOSTIC_QUERY_BYTES;
            let query = unsafe { text(query, &mut budget) }
                .map_err(|_| Error::Read(DiagnosticReadReason::InvalidQuery))?;
            let query = decode_diagnostic_query(query).map_err(Error::Read)?;
            let limit = capacity.min(a.context.limits().max_snapshot_bytes);
            let destination = unsafe { std::slice::from_raw_parts_mut(output, limit) };
            let reader = a.context.reader();
            let mut out = Output {
                bytes: destination,
                written: 0,
            };
            match kind {
                0 => {
                    serde_json::to_writer(&mut out, &reader.discovery())
                        .map_err(|_| Error::BufferTooSmall)?;
                }
                1 => {
                    let page = reader
                        .snapshot(&query)
                        .map_err(|e| Error::Read(e.read_reason()))?;
                    std::io::Write::write_all(&mut out, page.as_bytes())
                        .map_err(|_| Error::BufferTooSmall)?;
                }
                2 => {
                    let page = reader
                        .inspect_memory(&query)
                        .map_err(|e| Error::Read(e.read_reason()))?;
                    std::io::Write::write_all(&mut out, page.as_bytes())
                        .map_err(|_| Error::BufferTooSmall)?;
                }
                _ => return Err(Error::Invalid),
            }
            unsafe {
                written.write(out.written);
            }
            Ok(0)
        })
    })
}
/// Static host-lifetime callback table; the outer ABI tail is size-discovered.
pub(super) static API: abi::NativeDiagnosticsApiV1 = abi::NativeDiagnosticsApiV1 {
    struct_size: std::mem::size_of::<abi::NativeDiagnosticsApiV1>(),
    contract_version: abi::CONTRACT_VERSION,
    capabilities,
    register,
    record,
    close,
    read,
};

fn code_observation<'a>(
    code: Code,
    class: Option<FailureClass>,
    outcome: DiagnosticOutcome,
    detail: &'a str,
    count: NonZeroU64,
) -> Result<Observation<'a>> {
    let observation = match code {
        Code::Builtin(code) => {
            if outcome != DiagnosticOutcome::Failure {
                return Err(Error::Invalid);
            }
            Observation::new(
                Classification::new(code, class).map_err(|_| Error::Invalid)?,
                detail,
                count,
            )
        }
        Code::Extension(handle, expected) => {
            if outcome != expected {
                return Err(Error::Invalid);
            }
            Observation::extension(handle, class, outcome, detail, count)
        }
    };
    Ok(observation)
}

fn register_codes(
    registered: &mut Registered,
    parsed: &[Parsed<'_>],
    namespace: &str,
) -> Result<()> {
    for (i, definition) in parsed.iter().enumerate() {
        let result = match definition {
            Parsed::Builtin(code) => Ok(Code::Builtin(*code)),
            Parsed::Extension {
                name,
                description,
                outcome,
                classes,
                class_count,
            } => registered
                .recorder
                .register_extension(ExtensionRegistration {
                    namespace,
                    name,
                    description,
                    outcome: *outcome,
                    allowed_classes: &classes[..*class_count],
                })
                .map_err(core_error)
                .and_then(|v| {
                    v.map(|h| Code::Extension(h, *outcome))
                        .ok_or(Error::Unavailable)
                }),
            Parsed::Empty => Err(Error::Internal),
        };
        match result {
            Ok(code) => registered.codes[i] = Some(code),
            Err(error) => {
                registered.owner.discard_unpublished();
                return Err(error);
            }
        }
    }
    Ok(())
}

struct ParsedDefinitions<'a> {
    codes: [Parsed<'a>; abi::MAX_CODES],
    builtin: [BuiltinDiagnosticCode; abi::MAX_CODES],
    builtin_count: usize,
    extension_count: usize,
    count: usize,
}

unsafe fn parse_definitions<'a>(
    input: &'a abi::Registration,
    budget: &mut usize,
) -> Result<ParsedDefinitions<'a>> {
    let definitions = unsafe { array(input.codes, input.code_count, abi::MAX_CODES)? };
    if definitions.is_empty() {
        return Err(Error::Invalid);
    }
    let mut parsed = [Parsed::Empty; abi::MAX_CODES];
    let mut builtin = [BuiltinDiagnosticCode::all()[0]; abi::MAX_CODES];
    let mut builtin_count = 0;
    let mut extension_count = 0;
    for (index, def) in definitions.iter().enumerate() {
        let name = unsafe { text(def.name, budget)? };
        match def.kind {
            0 => {
                if def.description.len != 0 || def.class_count != 0 || def.outcome != 0 {
                    return Err(Error::Invalid);
                }
                let code =
                    BuiltinDiagnosticCode::from_canonical_name(name).ok_or(Error::Invalid)?;
                if !matches!(
                    code,
                    BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
                        _
                    )))
                ) || builtin[..builtin_count].contains(&code)
                {
                    return Err(Error::Invalid);
                }
                builtin[builtin_count] = code;
                builtin_count += 1;
                parsed[index] = Parsed::Builtin(code);
            }
            1 => {
                let description = unsafe { text(def.description, budget)? };
                if name.is_empty()
                    || name.len() > 32
                    || description.len() > 256
                    || parsed[..index]
                        .iter()
                        .any(|v| matches!(v, Parsed::Extension { name: n, .. } if *n == name))
                {
                    return Err(Error::Invalid);
                }
                parsed[index] = unsafe { parse_extension(def, name, description, budget)? };
                extension_count += 1;
            }
            _ => return Err(Error::Invalid),
        }
    }
    Ok(ParsedDefinitions {
        codes: parsed,
        builtin,
        builtin_count,
        extension_count,
        count: definitions.len(),
    })
}

unsafe fn parse_extension<'a>(
    def: &abi::Code,
    name: &'a str,
    description: &'a str,
    budget: &mut usize,
) -> Result<Parsed<'a>> {
    let raw_classes = unsafe { array(def.classes, def.class_count, abi::MAX_CLASSES)? };
    let mut classes = [FailureClass::Unknown; abi::MAX_CLASSES];
    for (i, value) in raw_classes.iter().enumerate() {
        classes[i] = FailureClass::from_canonical_name(unsafe { text(*value, budget)? })
            .ok_or(Error::Invalid)?;
        if classes[..i].contains(&classes[i]) {
            return Err(Error::Invalid);
        }
    }
    let outcome = outcome(def.outcome)?;
    if (outcome == DiagnosticOutcome::Failure) == raw_classes.is_empty() {
        return Err(Error::Invalid);
    }
    Ok(Parsed::Extension {
        name,
        description,
        outcome,
        classes,
        class_count: raw_classes.len(),
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/native_diagnostics_tests.rs"]
mod tests;
