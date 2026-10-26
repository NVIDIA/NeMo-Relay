// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One validated recording boundary and bounded evidence preparation.

use super::{
    BuiltinDiagnosticCode, CapturedOrigin, ComponentHandle, DiagnosticDetailRef, DiagnosticError,
    DiagnosticOutcome, EvidenceDisposition, FailureClass, Inner, Observation, ObservationCode,
    PublicationWorkspace, RecordReceipt, State, StoredCode, StoredFailureClass, StoredObservation,
    StoredPluginContext,
};
use chrono::Utc;
use std::fmt;

pub(super) fn bounded_str(value: &str, max: usize) -> &str {
    let mut end = value.len().min(max);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

pub(super) fn prepare_detail<'a>(
    detail: DiagnosticDetailRef<'a>,
    buffer: &'a mut [u8],
    max: usize,
) -> (Option<&'a str>, bool) {
    match detail {
        DiagnosticDetailRef::None => (None, false),
        DiagnosticDetailRef::Text(text) => {
            let value = bounded_str(text, max);
            (Some(value), value.len() != text.len())
        }
        DiagnosticDetailRef::Display(value) => {
            struct BoundedFormat<'a> {
                buffer: &'a mut [u8],
                len: usize,
                truncated: bool,
            }
            impl fmt::Write for BoundedFormat<'_> {
                fn write_str(&mut self, value: &str) -> fmt::Result {
                    let part = bounded_str(value, self.buffer.len().saturating_sub(self.len));
                    self.buffer[self.len..self.len + part.len()].copy_from_slice(part.as_bytes());
                    self.len += part.len();
                    if part.len() < value.len() {
                        self.truncated = true;
                        Err(fmt::Error)
                    } else {
                        Ok(())
                    }
                }
            }
            let mut output = BoundedFormat {
                buffer: &mut buffer[..max],
                len: 0,
                truncated: false,
            };
            let failed = fmt::write(&mut output, format_args!("{value}")).is_err();
            let len = output.len;
            let truncated = output.truncated || failed;
            (
                Some(std::str::from_utf8(&buffer[..len]).expect("formatter copies complete UTF-8")),
                truncated,
            )
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep the borrowed recording inputs explicit at the single acceptance boundary."
)]
pub(super) fn record_observation(
    state: &mut State,
    inner: &Inner,
    handle: ComponentHandle,
    requested_origin: Option<CapturedOrigin>,
    observation: &Observation<'_>,
    detail: Option<&str>,
    truncated: bool,
    projection_requested: bool,
    publication: Option<&mut PublicationWorkspace>,
) -> Result<RecordReceipt, DiagnosticError> {
    let component = match state
        .component(handle)
        .filter(|v| v.closed_at.is_none())
        .copied()
    {
        Some(component) => component,
        None => {
            state.loss.stale_observations.add(1);
            return Err(DiagnosticError::ComponentClosed);
        }
    };
    let origin = requested_origin.unwrap_or_else(|| component.origin());
    let context = StoredPluginContext::try_from(&observation.context)?;
    let (code, class) = resolve_observation_code(state, inner, handle, observation)?;
    let counter_index = if let Some(code) = code {
        Some(
            state
                .component_counters(handle)
                .find(|(_, counter)| counter.code == code)
                .map(|(index, _)| index.slot())
                .ok_or(DiagnosticError::InvalidCode)?,
        )
    } else {
        None
    };
    let sequence = state.next_sequence()?;
    let observed_at = Utc::now();
    let tick = inner.tick();
    state.lifetime.record(
        observation.outcome,
        observation.count.get(),
        observed_at,
        tick,
    );
    let current = state.components[handle.slot]
        .as_mut()
        .expect("validated component");
    current.lifetime.record(
        observation.outcome,
        observation.count.get(),
        observed_at,
        tick,
    );
    current
        .refresh
        .totals
        .record(observation.outcome, observation.count.get());
    state.record_window(handle, tick, observation.outcome, observation.count.get());
    let code_total = counter_index.map(|index| {
        let counter = state.counters[index].as_mut().expect("declared counter");
        counter.occurrences.add(observation.count.get());
        counter.last_observed_at = Some(observed_at);
        counter.occurrences
    });
    let Some(code) = code else {
        state
            .loss
            .classifications_overflowed
            .add(observation.count.get());
        state.loss.events_omitted.add(1);
        return Ok(RecordReceipt {
            sequence,
            code_total: None,
            evidence: EvidenceDisposition::Dropped,
        });
    };
    if truncated {
        state.loss.messages_truncated.add(1);
    }
    let session = observation
        .correlation
        .session_id
        .as_deref()
        .map(|value| bounded_str(value, inner.limits.max_correlation_bytes));
    let request = observation
        .correlation
        .request_id
        .as_deref()
        .map(|value| bounded_str(value, inner.limits.max_correlation_bytes));
    let session_truncated = observation
        .correlation
        .session_id
        .as_deref()
        .zip(session)
        .is_some_and(|(a, b)| a.len() != b.len());
    let request_truncated = observation
        .correlation
        .request_id
        .as_deref()
        .zip(request)
        .is_some_and(|(a, b)| a.len() != b.len());
    if session_truncated {
        state.loss.correlations_truncated.add(1);
    }
    if request_truncated {
        state.loss.correlations_truncated.add(1);
    }
    if let Some(workspace) = publication {
        if !workspace.fill(
            component.instance_id,
            code,
            class,
            sequence,
            observed_at,
            observation,
        ) {
            state.loss.projections_suppressed.add(1);
        }
    } else if projection_requested {
        state.loss.projection_deliveries_omitted.add(1);
    }
    let evidence = if inner.limits.max_events == 0 {
        state.loss.events_omitted.add(1);
        EvidenceDisposition::Dropped
    } else {
        let retained = state.retain_event_text([detail, session, request]);
        if detail.is_some() && retained.detail.is_none() {
            state.loss.details_omitted.add(1);
        }
        let correlation_omitted = (session.is_some() && retained.session_id.is_none()) as u64
            + (request.is_some() && retained.request_id.is_none()) as u64;
        state.loss.correlations_omitted.add(correlation_omitted);
        let event = StoredObservation {
            sequence,
            observed_at,
            remote: observation.remote.map(|mut remote| {
                remote.received_at = observed_at;
                remote.accepted_elapsed_millis = tick.into();
                remote
            }),
            count: observation.count.get(),
            component: handle,
            origin,
            code,
            failure_class: class,
            context,
            severity: observation.severity,
            outcome: observation.outcome,
            text: retained,
            detail_truncated: truncated,
        };
        if let Some(evicted) = state.recent.push_back(event) {
            state.evict_event(evicted);
        }
        if truncated
            || session_truncated
            || request_truncated
            || detail.is_some() && retained.detail.is_none()
            || correlation_omitted > 0
        {
            EvidenceDisposition::Truncated
        } else {
            EvidenceDisposition::Retained
        }
    };
    Ok(RecordReceipt {
        sequence,
        code_total,
        evidence,
    })
}

fn resolve_observation_code(
    state: &State,
    inner: &Inner,
    handle: ComponentHandle,
    observation: &Observation<'_>,
) -> Result<(Option<StoredCode>, Option<StoredFailureClass>), DiagnosticError> {
    let (code, class) = match &observation.code {
        ObservationCode::Builtin(classification) => {
            let code = BuiltinDiagnosticCode::from_canonical_name(classification.code())
                .ok_or(DiagnosticError::InvalidCode)?;
            let class = classification
                .failure_class()
                .map(|name| {
                    FailureClass::from_canonical_name(name)
                        .ok_or(DiagnosticError::InvalidClassification)
                })
                .transpose()?;
            if !code.allows_class(class) {
                return Err(DiagnosticError::InvalidClassification);
            }
            (
                Some(StoredCode::Builtin(code)),
                class.map(StoredFailureClass::Builtin),
            )
        }
        ObservationCode::Extension(extension, class) => {
            let (extension, class) = (*extension, *class);
            if extension.context_id != inner.context_id || extension.component != handle {
                return Err(DiagnosticError::InvalidCode);
            }
            let descriptor = state
                .extensions
                .get(extension.slot)
                .and_then(Option::as_ref)
                .filter(|v| v.generation == extension.generation && v.component == handle)
                .ok_or(DiagnosticError::InvalidCode)?;
            if descriptor.outcome == DiagnosticOutcome::Failure {
                if class.is_none_or(|class| !descriptor.allowed_classes.contains(class)) {
                    return Err(DiagnosticError::InvalidClassification);
                }
            } else if class.is_some() {
                return Err(DiagnosticError::InvalidClassification);
            }
            (
                Some(StoredCode::Extension {
                    slot: extension.slot,
                    generation: extension.generation,
                }),
                class.map(StoredFailureClass::Builtin),
            )
        }
        ObservationCode::OpaqueExtension(extension, class) => {
            let (extension, class) = (*extension, *class);
            if extension.context_id != inner.context_id
                || extension.component != handle
                || class.extension != extension
            {
                return Err(DiagnosticError::InvalidClassification);
            }
            let descriptor = state
                .extensions
                .get(extension.slot)
                .and_then(Option::as_ref)
                .filter(|v| v.generation == extension.generation && v.component == handle)
                .ok_or(DiagnosticError::InvalidCode)?;
            let cause = StoredFailureClass::Extension {
                slot: extension.slot,
                generation: extension.generation,
                index: u8::try_from(class.index)
                    .map_err(|_| DiagnosticError::InvalidClassification)?,
            };
            if descriptor.outcome != DiagnosticOutcome::Failure || state.class_name(cause).is_none()
            {
                return Err(DiagnosticError::InvalidClassification);
            }
            (
                Some(StoredCode::Extension {
                    slot: extension.slot,
                    generation: extension.generation,
                }),
                Some(cause),
            )
        }
        ObservationCode::Overflow => (None, None),
    };
    Ok((code, class))
}
