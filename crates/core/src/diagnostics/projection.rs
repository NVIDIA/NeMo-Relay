// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded publication workspaces, subscription lifetimes, and delivery.

use super::{
    Backend, CapturedOrigin, Classification, ComponentIdentity, ComponentOwnership, ComponentSlot,
    DiagnosticCorrelation, DiagnosticDetail, DiagnosticError, DiagnosticEvent, DiagnosticLimits,
    DiagnosticOrigin, DiagnosticOutcome, DiagnosticSeverity, ID_BYTES, Inner, Observation,
    ObservationTime, PluginObservationContext, ProjectionError, State, StoredCode,
    StoredFailureClass, StoredPluginContext, SuccessCode, empty_slots, lock,
};
use chrono::Utc;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::Weak;
use uuid::Uuid;

const MAX_PROJECTION_DEPTH: usize = 32;

pub(super) type Projection =
    Arc<dyn Fn(&DiagnosticEvent) -> Result<(), ProjectionError> + Send + Sync + 'static>;
pub(super) struct Projections {
    pub(super) next_id: u64,
    pub(super) slots: Box<[Option<(u64, Projection)>]>,
}
/// Bounded projection subscription, or inert disabled-mode handle.
///
/// Dropping unsubscribes future publication. A callback copied before removal
/// may still finish; adapters can add stronger shutdown ordering when needed.
///
/// Retain the guard for the subscription's intended lifetime:
///
/// ```compile_fail
/// #![deny(unused_must_use)]
/// use nemo_relay::diagnostics::{DiagnosticError, Diagnostics};
/// fn subscribe(diagnostics: &Diagnostics) -> Result<(), DiagnosticError> {
///     diagnostics.subscribe_projection(|_| {})?; // Immediately unsubscribes.
///     Ok(())
/// }
/// ```
#[must_use = "retain this guard to keep the diagnostic projection subscribed"]
pub struct ProjectionGuard {
    pub(super) inner: Weak<Inner>,
    pub(super) id: Option<u64>,
}
impl fmt::Debug for ProjectionGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectionGuard").finish_non_exhaustive()
    }
}
impl Drop for ProjectionGuard {
    fn drop(&mut self) {
        if let (Some(inner), Some(id)) = (self.inner.upgrade(), self.id)
            && let Some(backend) = &inner.backend
        {
            let removed = {
                let mut projections = lock(&backend.projections);
                projections
                    .slots
                    .iter_mut()
                    .find(|slot| slot.as_ref().is_some_and(|(key, _)| *key == id))
                    .and_then(Option::take)
            };
            // Captured user values can reenter logging/diagnostics on drop.
            drop(removed);
        }
    }
}

pub(super) struct PublicationWorkspace {
    pub(super) event: DiagnosticEvent,
    pub(super) code: String,
    pub(super) detail: String,
    pub(super) class: String,
    pub(super) session: String,
    pub(super) request: String,
    pub(super) origin_configuration: String,
    pub(super) origin_responsibility: String,
    pub(super) callbacks: Box<[Option<Projection>]>,
    pub(super) ready: bool,
}
pub(super) fn reserved_string(capacity: usize) -> Result<String, DiagnosticError> {
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| DiagnosticError::Allocation)?;
    Ok(value)
}
impl PublicationWorkspace {
    pub(super) fn required_bytes(limits: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        let text = ID_BYTES
            .checked_mul(4)
            .and_then(|v| v.checked_add(32))
            .and_then(|v| v.checked_add(limits.max_detail_bytes))
            .and_then(|v| v.checked_add(limits.max_correlation_bytes.checked_mul(2)?))
            .ok_or(DiagnosticError::InvalidLimits)?;
        text.checked_add(
            limits
                .max_projections
                .checked_mul(size_of::<Option<Projection>>())
                .ok_or(DiagnosticError::InvalidLimits)?,
        )
        .ok_or(DiagnosticError::InvalidLimits)
    }
    pub(super) fn new(limits: &DiagnosticLimits) -> Result<Self, DiagnosticError> {
        Ok(Self {
            event: DiagnosticEvent {
                sequence: 0.into(),
                observed_at: Utc::now(),
                component: ComponentIdentity {
                    key: reserved_string(ID_BYTES)?,
                    instance_id: Uuid::nil(),
                    generation: 0.into(),
                },
                origin: DiagnosticOrigin {
                    refresh_generation: 0.into(),
                    ownership: ComponentOwnership::default(),
                },
                classification: Classification::success(SuccessCode::LoggingInitialized),
                severity: DiagnosticSeverity::Info,
                outcome: DiagnosticOutcome::Unknown,
                context: PluginObservationContext::default(),
                detail: None,
                count: 0.into(),
                correlation: DiagnosticCorrelation::default(),
            },
            code: reserved_string(ID_BYTES)?,
            detail: reserved_string(limits.max_detail_bytes)?,
            class: reserved_string(32)?,
            session: reserved_string(limits.max_correlation_bytes)?,
            request: reserved_string(limits.max_correlation_bytes)?,
            origin_configuration: reserved_string(ID_BYTES)?,
            origin_responsibility: reserved_string(ID_BYTES)?,
            callbacks: empty_slots(limits.max_projections)?,
            ready: false,
        })
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        self.event.component.key.capacity()
            + self.code.capacity()
            + self.detail.capacity()
            + self.class.capacity()
            + self.session.capacity()
            + self.request.capacity()
            + self.origin_configuration.capacity()
            + self.origin_responsibility.capacity()
            + self.callbacks.len() * size_of::<Option<Projection>>()
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Copy validated observation fields into the existing projection workspace."
    )]
    pub(super) fn fill(
        &mut self,
        state: &State,
        component: &ComponentSlot,
        origin: CapturedOrigin,
        code: StoredCode,
        class: Option<StoredFailureClass>,
        sequence: u64,
        observed_at: ObservationTime,
        observation: &Observation<'_>,
        context: StoredPluginContext,
        detail: Option<&str>,
        truncated: bool,
        session: Option<&str>,
        request: Option<&str>,
    ) {
        self.event.sequence = sequence.into();
        self.event.observed_at = observed_at;
        self.event.component.key.clear();
        self.event.component.key.push_str(component.key.as_str());
        self.event.component.instance_id = component.instance_id;
        self.event.component.generation = component.handle.generation.into();
        self.event.origin.refresh_generation = origin.generation.into();
        self.event.origin.ownership.subscriber_instance_id =
            origin.ownership.subscriber_instance_id;
        self.event.origin.ownership.plugin_instance_id = origin.ownership.plugin_instance_id;
        if let Some(value) = origin.ownership.configuration_id {
            self.origin_configuration.clear();
            self.origin_configuration.push_str(value.as_str());
            self.event.origin.ownership.configuration_id =
                Some(std::mem::take(&mut self.origin_configuration));
        }
        if let Some(value) = origin.ownership.responsibility {
            self.origin_responsibility.clear();
            self.origin_responsibility.push_str(value.as_str());
            self.event.origin.ownership.responsibility =
                Some(std::mem::take(&mut self.origin_responsibility));
        }
        self.code.clear();
        self.code
            .push_str(state.code_name(code).expect("admitted code"));
        let failure_class = class.map(|class| {
            self.class.clear();
            self.class
                .push_str(state.class_name(class).expect("admitted class"));
            Cow::Owned(std::mem::take(&mut self.class))
        });
        self.event.classification = Classification::from_opaque_names(
            Cow::Owned(std::mem::take(&mut self.code)),
            failure_class,
        );
        self.event.context = context.into();
        self.event.severity = observation.severity;
        self.event.outcome = observation.outcome;
        self.event.count = observation.count.get().into();
        if let Some(detail) = detail {
            self.detail.clear();
            self.detail.push_str(detail);
            self.event.detail = Some(DiagnosticDetail {
                message: std::mem::take(&mut self.detail),
                truncated,
            });
        }
        if let Some(value) = session {
            self.session.clear();
            self.session.push_str(value);
            self.event.correlation.session_id = Some(Cow::Owned(std::mem::take(&mut self.session)));
        }
        if let Some(value) = request {
            self.request.clear();
            self.request.push_str(value);
            self.event.correlation.request_id = Some(Cow::Owned(std::mem::take(&mut self.request)));
        }
        self.ready = true;
    }
    pub(super) fn reset(&mut self) {
        if let Some(value) = self.event.origin.ownership.configuration_id.take() {
            self.origin_configuration = value;
            self.origin_configuration.clear();
        }
        if let Some(value) = self.event.origin.ownership.responsibility.take() {
            self.origin_responsibility = value;
            self.origin_responsibility.clear();
        }
        if let Some(value) = self.event.detail.take() {
            self.detail = value.message;
            self.detail.clear();
        }
        let (code, class) = std::mem::replace(
            &mut self.event.classification,
            Classification::success(SuccessCode::LoggingInitialized),
        )
        .into_names();
        if let Cow::Owned(code) = code {
            self.code = code;
            self.code.clear();
        }
        if let Some(value) = class {
            self.class = value.into_owned();
            self.class.clear();
        }
        if let Some(value) = self.event.correlation.session_id.take() {
            self.session = value.into_owned();
            self.session.clear();
        }
        if let Some(value) = self.event.correlation.request_id.take() {
            self.request = value.into_owned();
            self.request.clear();
        }
        for callback in &mut self.callbacks {
            *callback = None;
        }
        self.ready = false;
    }
}
/// Owns one bounded publication permit, following the query SnapshotLease.
/// Cleanup runs before taking the pool lock: callback captures may reenter.
pub(super) struct PublicationLease<'a> {
    pub(super) backend: &'a Backend,
    pub(super) index: usize,
    pub(super) workspace: Option<PublicationWorkspace>,
}
impl PublicationLease<'_> {
    pub(super) fn workspace_mut(&mut self) -> &mut PublicationWorkspace {
        self.workspace.as_mut().expect("live publication lease")
    }
}
impl Drop for PublicationLease<'_> {
    fn drop(&mut self) {
        if let Some(mut workspace) = self.workspace.take() {
            workspace.reset();
            lock(&self.backend.publication)[self.index] = Some(workspace);
        }
    }
}
pub(super) fn take_publication(backend: &Backend) -> Option<PublicationLease<'_>> {
    let (index, workspace) = {
        let mut pool = lock(&backend.publication);
        let index = pool.iter().position(Option::is_some)?;
        (index, pool[index].take().expect("available workspace"))
    };
    let mut lease = PublicationLease {
        backend,
        index,
        workspace: Some(workspace),
    };
    let projections = lock(&backend.projections);
    for (target, source) in lease
        .workspace_mut()
        .callbacks
        .iter_mut()
        .zip(projections.slots.iter())
    {
        *target = source.as_ref().map(|(_, callback)| Arc::clone(callback));
    }
    Some(lease)
}
thread_local! {static PROJECTING:RefCell<(usize,[usize;MAX_PROJECTION_DEPTH])>=const{RefCell::new((0,[0;MAX_PROJECTION_DEPTH]))};}
struct ProjectionScope;
impl Drop for ProjectionScope {
    fn drop(&mut self) {
        PROJECTING.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.0 -= 1;
        });
    }
}
pub(super) fn publish(inner: &Inner, workspace: &mut PublicationWorkspace) {
    let service = std::ptr::from_ref(inner) as usize;
    let entered = PROJECTING.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.0 == MAX_PROJECTION_DEPTH || stack.1[..stack.0].contains(&service) {
            false
        } else {
            let index = stack.0;
            stack.1[index] = service;
            stack.0 += 1;
            true
        }
    });
    let Some(backend) = &inner.backend else {
        return;
    };
    if !entered {
        lock(&backend.state).loss.projections_suppressed.add(1);
        return;
    }
    let _scope = ProjectionScope;
    for callback in workspace.callbacks.iter().flatten() {
        match catch_unwind(AssertUnwindSafe(|| callback(&workspace.event))) {
            Ok(Ok(())) => {}
            Ok(Err(_)) => lock(&backend.state)
                .loss
                .projection_deliveries_omitted
                .add(1),
            Err(_) => lock(&backend.state).loss.projections_panicked.add(1),
        }
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use crate::diagnostics::{Diagnostics, DiagnosticsSettings};

    #[test]
    fn publication_lease_restores_payload_and_permit_after_unwind() {
        let diagnostics = Diagnostics::from_settings(
            Diagnostics::default().source().clone(),
            DiagnosticsSettings {
                enabled: true,
                limits: DiagnosticLimits {
                    max_projection_workspaces: 1,
                    ..Default::default()
                },
            },
        )
        .unwrap();
        let backend = diagnostics.inner.backend.as_ref().unwrap();
        let reserved = diagnostics.memory_usage().reserved_bytes;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut lease = take_publication(backend).unwrap();
            assert!(take_publication(backend).is_none());
            let workspace = lease.workspace_mut();
            workspace.detail.push_str("retained payload");
            workspace.event.detail = Some(DiagnosticDetail {
                message: std::mem::take(&mut workspace.detail),
                truncated: false,
            });
            workspace.ready = true;
            panic!("unwind while holding a publication permit");
        }));
        assert!(outcome.is_err());
        let mut lease = take_publication(backend).unwrap();
        let workspace = lease.workspace_mut();
        assert!(!workspace.ready);
        assert!(workspace.event.detail.is_none());
        assert!(workspace.detail.is_empty());
        assert!(workspace.detail.capacity() >= diagnostics.limits().max_detail_bytes);
        drop(lease);
        assert_eq!(diagnostics.memory_usage().projection_workspaces, 0);
        assert_eq!(diagnostics.memory_usage().reserved_bytes, reserved);
    }
}
