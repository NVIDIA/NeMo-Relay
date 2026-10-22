// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded publication workspaces, subscription lifetimes, and delivery.

use super::{
    Backend, Classification, DiagnosticError, DiagnosticLimits, DiagnosticOutcome,
    DiagnosticOutputSummary, DiagnosticSeverity, Inner, Observation, ObservationTime,
    ProjectionError, StoredCode, StoredFailureClass, SuccessCode, empty_slots, lock,
};
use chrono::Utc;
use std::cell::RefCell;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::Weak;
use uuid::Uuid;

const MAX_PROJECTION_DEPTH: usize = 32;

pub(super) type Projection =
    Arc<dyn Fn(&DiagnosticOutputSummary) -> Result<(), ProjectionError> + Send + Sync + 'static>;
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
    pub(super) event: DiagnosticOutputSummary,
    pub(super) callbacks: Box<[Option<Projection>]>,
    pub(super) ready: bool,
}
impl PublicationWorkspace {
    pub(super) fn required_bytes(limits: &DiagnosticLimits) -> Result<usize, DiagnosticError> {
        limits
            .max_projections
            .checked_mul(size_of::<Option<Projection>>())
            .ok_or(DiagnosticError::InvalidLimits)
    }
    pub(super) fn new(limits: &DiagnosticLimits) -> Result<Self, DiagnosticError> {
        Ok(Self {
            event: DiagnosticOutputSummary {
                sequence: 0.into(),
                observed_at: Utc::now(),
                component_instance_id: Uuid::nil(),
                classification: Classification::success(SuccessCode::LoggingInitialized),
                severity: DiagnosticSeverity::Info,
                outcome: DiagnosticOutcome::Unknown,
                count: 0.into(),
            },
            callbacks: empty_slots(limits.max_projections)?,
            ready: false,
        })
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        self.callbacks.len() * size_of::<Option<Projection>>()
    }
    pub(super) fn fill(
        &mut self,
        component_instance_id: Uuid,
        code: StoredCode,
        class: Option<StoredFailureClass>,
        sequence: u64,
        observed_at: ObservationTime,
        observation: &Observation<'_>,
    ) -> bool {
        // This branch never reads detail, names, ownership, or original correlations.
        let StoredCode::Builtin(code) = code else {
            return false;
        };
        let class = match class {
            Some(StoredFailureClass::Builtin(class)) => Some(class),
            Some(StoredFailureClass::Extension { .. }) => return false,
            None => None,
        };
        let Ok(classification) = Classification::new(code, class) else {
            return false;
        };
        self.event = DiagnosticOutputSummary {
            sequence: sequence.into(),
            observed_at,
            component_instance_id,
            classification,
            severity: observation.severity,
            outcome: observation.outcome,
            count: observation.count.get().into(),
        };
        self.ready = true;
        true
    }
    pub(super) fn reset(&mut self) {
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
            workspace.ready = true;
            panic!("unwind while holding a publication permit");
        }));
        assert!(outcome.is_err());
        let mut lease = take_publication(backend).unwrap();
        let workspace = lease.workspace_mut();
        assert!(!workspace.ready);
        drop(lease);
        assert_eq!(diagnostics.memory_usage().projection_workspaces, 0);
        assert_eq!(diagnostics.memory_usage().reserved_bytes, reserved);
    }
}
