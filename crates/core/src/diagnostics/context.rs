// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context ownership, component capabilities, and the public runtime facade.

use super::{
    CapturedOrigin, ComponentHandle, ComponentIdentity, ComponentOwnership, ComponentRegistration,
    DiagnosticAvailability, DiagnosticError, DiagnosticLimits, DiagnosticMemoryUsage,
    DiagnosticNumber, DiagnosticOutputSummary, DiagnosticQuery, DiagnosticsSettings,
    ExtensionClassHandle, ExtensionHandle, ExtensionRegistration, ID_BYTES, MAX_DETAIL_BYTES,
    Observation, Projection, ProjectionError, ProjectionGuard, Projections, PublicationLease,
    PublicationWorkspace, RecordOutcome, SnapshotLease, SourceIdentity, State, bounded_str,
    close_component, context_layout_bytes, diagnostic_memory_usage, empty_slots, lifecycle,
    prepare_detail, publish, read, record_observation, refresh_component, register_component,
    register_extension, take_publication,
};
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::Weak;
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone)]
/// Independent diagnostics context with immutable startup enablement.
///
/// Clones share this context only. State and budgets are independent of logger
/// lifetime and of any other explicitly isolated embedded contexts.
pub struct Diagnostics {
    pub(super) inner: Arc<Inner>,
}
pub(super) struct Inner {
    pub(super) context_id: Uuid,
    pub(super) source: SourceIdentity,
    pub(super) limits: DiagnosticLimits,
    pub(super) started: Instant,
    pub(super) backend: Option<Backend>,
}
pub(super) struct Backend {
    pub(super) state: Mutex<State>,
    pub(super) queries: Option<Arc<read::QueryPool>>,
    pub(super) projections: Mutex<Projections>,
    pub(super) publication: Mutex<Box<[Option<PublicationWorkspace>]>>,
    pub(super) reserved_bytes: usize,
}
impl Inner {
    pub(super) fn tick(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}
impl fmt::Debug for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Diagnostics")
            .field("source", &self.inner.source)
            .field("enabled", &self.inner.backend.is_some())
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
/// Read capability for one context, without recording or lifecycle authority.
///
/// Clones keep the context alive. Materialization requires a bounded lease;
/// disabled collection is explicit instead of fabricated empty evidence.
pub struct DiagnosticsReader {
    pub(super) inner: Arc<Inner>,
}
impl fmt::Debug for DiagnosticsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiagnosticsReader")
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
/// Producer capability scoped to one admitted registration.
///
/// Clones share its lease; the last recorder drop closes the component.
/// Producers cannot refresh or replace registrations. Disabled recorders
/// have no active identity and skip lazy observation factories.
pub struct ComponentRecorder {
    pub(super) inner: Arc<Inner>,
    pub(super) lease: Option<Arc<ComponentLease>>,
}
impl fmt::Debug for ComponentRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentRecorder")
            .field("identity", &self.identity())
            .finish_non_exhaustive()
    }
}
pub(super) struct ComponentLease {
    pub(super) inner: Weak<Inner>,
    pub(super) handle: ComponentHandle,
    pub(super) identity: ComponentIdentity,
}
impl Drop for ComponentLease {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            let _ = close_component(&inner, self.handle);
        }
    }
}
/// Restricted lifecycle capability returned only to the registration owner.
pub struct ComponentOwner {
    pub(super) inner: Weak<Inner>,
    pub(super) handle: Option<ComponentHandle>,
}
impl ComponentOwner {
    // Roll back an unpublished transport registration. No observer may have
    // recorded through this owner; retained/public registrations use close().
    pub(crate) fn discard_unpublished(&self) {
        if let (Some(inner), Some(handle)) = (self.inner.upgrade(), self.handle)
            && let Some(backend) = &inner.backend
        {
            lock(&backend.state).discard_unobserved_component(handle);
        }
    }
    /// Close this registration without erasing retained evidence.
    ///
    /// Returns false for already closed/unavailable or disabled handles. Later
    /// recording receives an explicit closed-handle failure.
    pub fn close(&self) -> Result<bool, DiagnosticError> {
        match (self.inner.upgrade(), self.handle) {
            (Some(inner), Some(handle)) => close_component(&inner, handle),
            _ => Ok(false),
        }
    }
    /// Reset the current since-refresh summary after a committed reload.
    ///
    /// Lifetime totals, code counters, rolling windows, and retained evidence
    /// remain available. The generation marks the new observation-time period;
    /// captured operations retain their original attribution. Disabled handles
    /// return None without allocating or changing diagnostic state.
    pub fn refresh(
        &self,
        ownership: ComponentOwnership,
    ) -> Result<Option<DiagnosticNumber>, DiagnosticError> {
        let Some(inner) = self.inner.upgrade() else {
            return Err(DiagnosticError::ComponentClosed);
        };
        let Some(handle) = self.handle else {
            return Ok(None);
        };
        refresh_component(&inner, handle, ownership).map(|value| Some(value.into()))
    }
}
/// An operation's bounded original configuration attribution.
///
/// A refresh never invalidates this recorder. Observations update lifetime and
/// current since-refresh totals at recording time while preserving their origin
/// on retained events. Closing the registration remains explicit.
#[derive(Clone, Debug)]
pub struct OriginRecorder {
    pub(super) recorder: ComponentRecorder,
    pub(super) origin: Option<CapturedOrigin>,
}
impl OriginRecorder {
    /// Record one completion with its captured origin and optional projections.
    /// The synchronous factory is skipped while collection is disabled.
    pub fn record_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.recorder.record_inner_lazy(make, true, self.origin)
    }
    /// Record with captured origin while bypassing optional output projections.
    pub fn record_without_projection_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.recorder.record_inner_lazy(make, false, self.origin)
    }
}

static PROCESS_DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();
impl Diagnostics {
    /// Construct a context using the startup owner's selected mode and capacities.
    ///
    /// Enabled mode validates arithmetic and admits backend storage before use.
    /// The circular-history allocation uses the global allocation-error handler
    /// on allocation failure; fallible pool reservations return an error.
    /// Disabled mode retains lightweight source/settings state without registry,
    /// history, detail, query, or projection pools. Source role/version metadata
    /// is capped at 128 UTF-8 bytes; the supplied UUID is preserved.
    pub fn from_settings(
        source: SourceIdentity,
        settings: DiagnosticsSettings,
    ) -> Result<Self, DiagnosticError> {
        let source = SourceIdentity {
            instance_id: source.instance_id,
            process_id: source.process_id,
            process_role: bounded_str(&source.process_role, ID_BYTES).to_owned(),
            relay_version: bounded_str(&source.relay_version, ID_BYTES).to_owned(),
        };
        let backend = if settings.enabled {
            settings.limits.validate()?;
            let state = State::new(&settings.limits)?;
            let queries = if settings.limits.max_queries == 0 {
                None
            } else {
                Some(
                    read::QueryPool::new(&settings.limits)
                        .map_err(|_| DiagnosticError::Allocation)?,
                )
            };
            let mut publication = empty_slots(settings.limits.max_projection_workspaces)?;
            for slot in &mut publication {
                *slot = Some(PublicationWorkspace::new(&settings.limits)?);
            }
            let projections = Projections {
                next_id: 0,
                slots: empty_slots(settings.limits.max_projections)?,
            };
            let reserved_bytes = state
                .reserved_bytes()
                .checked_add(queries.as_ref().map_or(0, |pool| pool.reserved_bytes()))
                .and_then(|v| {
                    v.checked_add(
                        publication
                            .iter()
                            .flatten()
                            .map(PublicationWorkspace::reserved_bytes)
                            .sum::<usize>(),
                    )
                })
                .and_then(|v| {
                    v.checked_add(projections.slots.len() * size_of::<Option<(u64, Projection)>>())
                })
                .and_then(|v| {
                    v.checked_add(publication.len() * size_of::<Option<PublicationWorkspace>>())
                })
                .and_then(|v| v.checked_add(context_layout_bytes()))
                .and_then(|v| {
                    v.checked_add(source.process_role.capacity() + source.relay_version.capacity())
                })
                .ok_or(DiagnosticError::InvalidLimits)?;
            if reserved_bytes > settings.limits.max_owned_bytes {
                return Err(DiagnosticError::InvalidLimits);
            }
            Some(Backend {
                state: Mutex::new(state),
                queries,
                projections: Mutex::new(projections),
                publication: Mutex::new(publication),
                reserved_bytes,
            })
        } else {
            None
        };
        Ok(Self {
            inner: Arc::new(Inner {
                context_id: Uuid::now_v7(),
                source,
                limits: settings.limits,
                started: Instant::now(),
                backend,
            }),
        })
    }
    pub(crate) fn install_context(context: Self) -> Result<(), DiagnosticError> {
        PROCESS_DIAGNOSTICS
            .set(context)
            .map_err(|_| DiagnosticError::AlreadyInitialized)
    }
    /// Inspect a selected context without consuming an uninitialized default.
    pub fn selected_process_default() -> Option<&'static Self> {
        PROCESS_DIAGNOSTICS.get()
    }
    /// Install before handing out default handles. Once used, inert handles are
    /// never silently upgraded to a different startup mode.
    pub fn install_process_default(
        source: SourceIdentity,
        settings: DiagnosticsSettings,
    ) -> Result<&'static Self, DiagnosticError> {
        if PROCESS_DIAGNOSTICS.get().is_some() {
            return Err(DiagnosticError::AlreadyInitialized);
        }
        let context = Self::from_settings(source, settings)?;
        PROCESS_DIAGNOSTICS
            .set(context)
            .map_err(|_| DiagnosticError::AlreadyInitialized)?;
        Ok(PROCESS_DIAGNOSTICS.get().expect("installed default"))
    }
    /// Return the installed process context or initialize a disabled default.
    ///
    /// Install enabled settings before the first lookup. Early disabled handles
    /// are never implicitly upgraded by logger or plugin construction.
    pub fn process_default() -> &'static Self {
        PROCESS_DIAGNOSTICS.get_or_init(Self::default)
    }
    /// Configured capacities, independently of current occupancy or disablement.
    pub fn limits(&self) -> &DiagnosticLimits {
        &self.inner.limits
    }
    /// Source incarnation shared by every observation from this context.
    pub fn source(&self) -> &SourceIdentity {
        &self.inner.source
    }
    /// Issue a read capability without materializing or allocating a query page.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.inner),
        }
    }
    /// Capture a bounded page through this context's reader.
    ///
    /// The lease holds a query permit until dropped. Reads never reset counters
    /// or pin history; configured disablement is returned explicitly. The query
    /// is borrowed only during capture and can be reused after this call.
    pub fn snapshot(&self, query: &DiagnosticQuery) -> Result<SnapshotLease, DiagnosticError> {
        self.reader().snapshot(query)
    }
    /// Reserve a component vocabulary and return its producer lease.
    ///
    /// Admission cannot displace active summaries. The final recorder drop closes
    /// the component. Use [`Self::register_owned_component`] for explicit close
    /// and refresh authority. Disabled mode returns an inert recorder.
    pub fn register_component(
        &self,
        registration: ComponentRegistration<'_>,
    ) -> Result<ComponentRecorder, DiagnosticError> {
        self.register_owned_component(registration)
            .map(|(_, recorder)| recorder)
    }
    /// Reserve a component and separately issue owner and producer capabilities.
    ///
    /// The owner can close or rotate this registration. Disabled mode issues inert
    /// handles without an active identity or backend reservation. Admission
    /// failures are explicit and recorded in reserved loss accounting.
    pub fn register_owned_component(
        &self,
        registration: ComponentRegistration<'_>,
    ) -> Result<(ComponentOwner, ComponentRecorder), DiagnosticError> {
        let Some(backend) = &self.inner.backend else {
            return Ok((
                ComponentOwner {
                    inner: Arc::downgrade(&self.inner),
                    handle: None,
                },
                ComponentRecorder {
                    inner: Arc::clone(&self.inner),
                    lease: None,
                },
            ));
        };
        let mut state = lock(&backend.state);
        let admitted = register_component(&mut state, &self.inner, registration);
        let (handle, identity) = match admitted {
            Ok(value) => value,
            Err(error) => {
                state.loss.components_rejected.add(1);
                return Err(error);
            }
        };
        Ok((
            ComponentOwner {
                inner: Arc::downgrade(&self.inner),
                handle: Some(handle),
            },
            ComponentRecorder {
                inner: Arc::clone(&self.inner),
                lease: Some(Arc::new(ComponentLease {
                    inner: Arc::downgrade(&self.inner),
                    handle,
                    identity,
                })),
            },
        ))
    }
    /// Attach a bounded subscriber to approved operational summaries.
    /// Restricted retained evidence is never supplied to projection callbacks.
    ///
    /// Callbacks run outside the store lock, may run concurrently, and have isolated
    /// panics. Disabled mode retains no callback and returns an inert guard.
    /// Arbitrary callback captures and retained copies remain caller-owned.
    pub fn subscribe_projection(
        &self,
        callback: impl Fn(&DiagnosticOutputSummary) + Send + Sync + 'static,
    ) -> Result<ProjectionGuard, DiagnosticError> {
        self.subscribe_fallible_projection(move |event| {
            callback(event);
            Ok(())
        })
    }
    /// Attach a subscriber that reports bounded conversion or delivery failure.
    ///
    /// Errors update projection-loss accounting without undoing numeric acceptance.
    /// Separate publication workspaces cannot pin history. Disabled mode retains
    /// no callback and returns an inert guard.
    pub fn subscribe_fallible_projection(
        &self,
        callback: impl Fn(&DiagnosticOutputSummary) -> Result<(), ProjectionError>
        + Send
        + Sync
        + 'static,
    ) -> Result<ProjectionGuard, DiagnosticError> {
        let Some(backend) = &self.inner.backend else {
            return Ok(ProjectionGuard {
                inner: Weak::new(),
                id: None,
            });
        };
        let mut projections = lock(&backend.projections);
        let index = projections
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(DiagnosticError::ProjectionCapacity)?;
        let id = projections
            .next_id
            .checked_add(1)
            .ok_or(DiagnosticError::ProjectionCapacity)?;
        projections.next_id = id;
        projections.slots[index] = Some((id, Arc::new(callback)));
        Ok(ProjectionGuard {
            inner: Arc::downgrade(&self.inner),
            id: Some(id),
        })
    }
    /// Inspect reserved backend bytes and occupancy without changing evidence.
    ///
    /// Inspection scans admitted slots and detail-block metadata without allocating.
    /// Disabled contexts report zero backend capacity. Lightweight facade state,
    /// caller-owned copies, thread stacks, and independent outputs are excluded.
    pub fn memory_usage(&self) -> DiagnosticMemoryUsage {
        diagnostic_memory_usage(&self.inner)
    }
}
impl Default for Diagnostics {
    fn default() -> Self {
        Self::from_settings(
            SourceIdentity {
                instance_id: Uuid::now_v7(),
                process_id: Some(std::process::id()),
                process_role: "runtime".into(),
                relay_version: env!("CARGO_PKG_VERSION").into(),
            },
            DiagnosticsSettings::default(),
        )
        .expect("disabled construction needs no diagnostic backend")
    }
}
impl DiagnosticsReader {
    /// Discover configured capabilities without a snapshot lease or state mutation.
    pub fn discovery(&self) -> super::DiagnosticDiscovery {
        let limits = &self.inner.limits;
        let enabled = self.inner.backend.is_some();
        super::DiagnosticDiscovery {
            discovery_version: super::DIAGNOSTIC_DISCOVERY_VERSION,
            snapshot_version: super::DIAGNOSTIC_SCHEMA_VERSION,
            catalogue_version: super::DIAGNOSTIC_CATALOGUE_VERSION,
            source: self.inner.source.clone(),
            collection_enabled: enabled,
            queries_enabled: enabled && limits.max_queries > 0,
            max_components: (limits.max_components as u64).into(),
            max_events: (limits.max_events as u64).into(),
            detail_pool_bytes: (limits.detail_pool_bytes as u64).into(),
            max_queries: (limits.max_queries as u64).into(),
            max_snapshot_bytes: (limits.max_snapshot_bytes as u64).into(),
            max_owned_bytes: (limits.max_owned_bytes as u64).into(),
            memory_inspection: true,
            restricted_detail: true,
            producer_support: [
                ("logging", true),
                ("plugin.configuration", true),
                ("plugin.activation", true),
                ("plugin.native", true),
                ("plugin.native.emission", true),
                ("plugin.policy", true),
                ("plugin.worker.host", cfg!(feature = "worker-grpc")),
                ("plugin.worker.emission", cfg!(feature = "worker-grpc")),
                ("otel", false),
                ("resource_metrics", false),
            ]
            .into_iter()
            .map(|(name, supported)| super::DiagnosticProducerSupport {
                name: name.into(),
                supported,
            })
            .collect(),
            max_query_components: (limits.max_query_components as u64).into(),
            max_query_events: (limits.max_query_events as u64).into(),
            detailed_memory_inspection: true,
            resource_correlation: true,
        }
    }

    /// Inspect reserved backing bytes and occupancy without modifying evidence.
    pub fn memory_usage(&self) -> DiagnosticMemoryUsage {
        diagnostic_memory_usage(&self.inner)
    }
    /// Inspect configured capacities separately from actual backing allocation.
    pub fn limits(&self) -> &DiagnosticLimits {
        &self.inner.limits
    }
    /// Capture a bounded consistent page with continuation and coverage.
    ///
    /// A held lease occupies one workspace. Serialization follows store unlocking;
    /// polling neither increments counters nor resets refresh summaries. Disabled collection
    /// returns [`DiagnosticError::Disabled`]. An enabled context with no query
    /// pool returns [`DiagnosticError::QueryDisabled`].
    pub fn snapshot(&self, query: &DiagnosticQuery) -> Result<SnapshotLease, DiagnosticError> {
        read::capture(&self.inner, query)
    }
    /// Report configured availability without materializing a snapshot.
    ///
    /// Available means enabled, not complete producer integration or deployment health.
    pub fn availability(&self) -> DiagnosticAvailability {
        if self.inner.backend.is_some() {
            DiagnosticAvailability::Available
        } else {
            DiagnosticAvailability::Disabled
        }
    }
}
impl ComponentRecorder {
    /// Borrow the original registration identity, or None while disabled.
    ///
    /// Closure or retention cannot relabel this handle as another registration.
    pub fn identity(&self) -> Option<&ComponentIdentity> {
        self.lease.as_ref().map(|lease| &lease.identity)
    }
    /// Read the same source context without granting lifecycle authority.
    pub fn reader(&self) -> DiagnosticsReader {
        DiagnosticsReader {
            inner: Arc::clone(&self.inner),
        }
    }
    /// Capture bounded configuration attribution for an operation's completion.
    ///
    /// Refreshing does not pin history or invalidate the captured origin.
    /// Disabled collection returns an inert recorder without allocating.
    pub fn capture_origin(&self) -> Result<OriginRecorder, DiagnosticError> {
        let origin = if let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) {
            let state = lock(&backend.state);
            let component = state
                .component(lease.handle)
                .filter(|v| v.closed_at.is_none())
                .ok_or(DiagnosticError::ComponentClosed)?;
            Some(component.origin())
        } else {
            None
        };
        Ok(OriginRecorder {
            recorder: self.clone(),
            origin,
        })
    }
    /// Admit namespaced extension metadata and its component code counter.
    ///
    /// The handle retains registration/descriptor generation scope. Disabled mode
    /// returns None without backend allocation. Explicit capacity failure may be
    /// reported using a summary-only classification overflow observation.
    pub fn register_extension(
        &self,
        registration: ExtensionRegistration<'_>,
    ) -> Result<Option<ExtensionHandle>, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(None);
        };
        let mut state = lock(&backend.state);
        register_extension(&mut state, &self.inner, lease.handle, registration).map(Some)
    }
    /// Admit a custom failure-class name for a registered extension. Each code
    /// has four fixed class slots, each at most 32 bytes, within metadata budgets.
    /// Disabled contexts return None without registering metadata.
    pub fn register_extension_class(
        &self,
        extension: ExtensionHandle,
        name: &str,
    ) -> Result<Option<ExtensionClassHandle>, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(None);
        };
        lifecycle::register_extension_class(
            &mut lock(&backend.state),
            &self.inner,
            lease.handle,
            extension,
            name,
        )
        .map(Some)
    }
    /// Build and record one observation synchronously only when enabled.
    ///
    /// Reserved summaries accept positive deltas independently of history pressure.
    /// Projections follow store unlocking. Disabled mode skips the factory and
    /// returns [`RecordOutcome::Disabled`] without loss accounting.
    pub fn record_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_inner_lazy(make, true, None)
    }
    /// Use the same recording boundary while bypassing every projection.
    ///
    /// Sink feedback and existing operational emitters use this path to avoid output
    /// recursion or duplicates. Disabled mode skips the factory.
    pub fn record_without_projection_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_inner_lazy(make, false, None)
    }
    /// Record constructed evidence and attempt optional projections.
    ///
    /// Disabled mode discards it but cannot undo caller construction work.
    /// Built-in producers use [`Self::record_lazy`] for diagnostic-only work.
    pub fn record(&self, observation: Observation<'_>) -> Result<RecordOutcome, DiagnosticError> {
        self.record_lazy(|| observation)
    }
    /// Record constructed evidence without any projection callback.
    ///
    /// Prefer [`Self::record_without_projection_lazy`] to skip construction while
    /// disabled. Counting and retention use the same canonical boundary.
    pub fn record_without_projection(
        &self,
        observation: Observation<'_>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        self.record_without_projection_lazy(|| observation)
    }
    fn record_inner_lazy<'a>(
        &self,
        make: impl FnOnce() -> Observation<'a>,
        project: bool,
        origin: Option<CapturedOrigin>,
    ) -> Result<RecordOutcome, DiagnosticError> {
        let (Some(backend), Some(lease)) = (&self.inner.backend, &self.lease) else {
            return Ok(RecordOutcome::Disabled);
        };
        let observation = make();
        let mut formatted = [0u8; MAX_DETAIL_BYTES];
        let (detail, truncated) = prepare_detail(
            observation.detail,
            &mut formatted,
            self.inner.limits.max_detail_bytes,
        );
        let projection_requested =
            project && lock(&backend.projections).slots.iter().any(Option::is_some);
        let publication = if projection_requested {
            take_publication(backend)
        } else {
            None
        };
        let mut publication = publication;
        let recorded = {
            let mut state = lock(&backend.state);
            let result = record_observation(
                &mut state,
                &self.inner,
                lease.handle,
                origin,
                &observation,
                detail,
                truncated,
                projection_requested,
                publication.as_mut().map(PublicationLease::workspace_mut),
            );
            if result.is_err() {
                state.loss.observations_rejected.add(1);
            }
            result
        };
        if let Some(publication) = publication.as_mut() {
            let workspace = publication.workspace_mut();
            if recorded.is_ok() && workspace.ready {
                publish(&self.inner, workspace);
            }
        }
        recorded.map(RecordOutcome::Recorded)
    }
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl DiagnosticsReader {
    /// Inspect bounded per-entry/component backing and independently copied resource context.
    /// Selection never restricts whole-context totals or grants inspection authority.
    /// Snapshot and memory reads share query permits. No collection is started.
    pub fn inspect_memory(
        &self,
        query: &DiagnosticQuery,
    ) -> Result<super::MemoryInspectionLease, DiagnosticError> {
        read::memory_capture(&self.inner, query)
    }
}
