// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use thiserror::Error;

use super::lifecycle::{McpSessionId, ResolvedTarget, RouteState, RouteStateKind, WorkerTarget};
use crate::daemon::common::identity::{Fingerprint, TokenDigest};
use crate::daemon::common::protocol::{BrokerDirective, WorkerLaunch};

const DEFAULT_RETRY_AFTER_MS: u64 = 100;
const MAX_ROUTE_BINDINGS: usize = 4_096;
const MAX_MCP_REFERENCES_PER_ROUTE: usize = 1_024;

/// An authenticated MCP registration applied idempotently by session ID.
#[derive(Debug, Clone)]
pub(crate) struct McpRegistration {
    pub(crate) fingerprint: Fingerprint,
    pub(crate) token_digest: TokenDigest,
    pub(crate) session_id: McpSessionId,
    pub(crate) lease_expires_at_unix_ms: u64,
}

/// A lock-bounded broker registry keyed by stable user-machine fingerprint.
pub(crate) struct Registry {
    global_pass_through: bool,
    retry_after_ms: u64,
    route_capacity: usize,
    inner: RwLock<RegistryInner>,
}

impl Registry {
    /// Authorizes a worker recovery without mutating route state.
    ///
    /// The returned permit captures the exact route generation and must be presented again when
    /// publishing the worker after its authenticated readiness probe.
    pub(crate) fn authorize_worker_recovery(
        &self,
        fingerprint: Fingerprint,
        worker_id: &str,
    ) -> Result<RecoveryPermit, RegistryError> {
        let inner = self.read();
        let route = inner
            .routes
            .get(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        if route.refs.is_empty() {
            return Err(RegistryError::NoLiveMcpReferences);
        }
        match &route.state {
            RouteState::Activating { launch, .. } => Ok(RecoveryPermit::Activating {
                activation_id: launch.activation_id.clone(),
            }),
            RouteState::Ready { target } if target.worker_id() == worker_id => {
                Ok(RecoveryPermit::ExistingWorker {
                    worker_id: worker_id.to_owned(),
                    recovering: false,
                })
            }
            RouteState::Recovering {
                target: Some(target),
                ..
            } if target.worker_id() == worker_id => Ok(RecoveryPermit::ExistingWorker {
                worker_id: worker_id.to_owned(),
                recovering: true,
            }),
            _ => Err(RegistryError::RecoveryNotAuthorized),
        }
    }

    /// Publishes a recovered worker only if the preflighted route generation is unchanged.
    pub(crate) fn publish_recovered_worker(
        &self,
        fingerprint: Fingerprint,
        permit: &RecoveryPermit,
        target: Arc<WorkerTarget>,
    ) -> Result<Option<String>, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        if route.refs.is_empty() {
            return Err(RegistryError::NoLiveMcpReferences);
        }
        let authorized = match (&route.state, permit) {
            (
                RouteState::Activating { launch, .. },
                RecoveryPermit::Activating { activation_id },
            ) => launch.activation_id == *activation_id,
            (
                RouteState::Ready { target },
                RecoveryPermit::ExistingWorker {
                    worker_id,
                    recovering: false,
                },
            ) => target.worker_id() == worker_id,
            (
                RouteState::Recovering {
                    target: Some(target),
                    ..
                },
                RecoveryPermit::ExistingWorker {
                    worker_id,
                    recovering: true,
                },
            ) => target.worker_id() == worker_id,
            _ => false,
        };
        if !authorized {
            return Err(RegistryError::RecoveryGenerationChanged);
        }
        let canceled_activation = match &route.state {
            RouteState::Activating { launch, .. } => Some(launch.activation_id.clone()),
            _ => None,
        };
        route.state = RouteState::Ready { target };
        Ok(canceled_activation)
    }

    /// Creates an empty registry.
    pub(crate) fn new(global_pass_through: bool) -> Self {
        Self {
            global_pass_through,
            retry_after_ms: DEFAULT_RETRY_AFTER_MS,
            route_capacity: MAX_ROUTE_BINDINGS,
            inner: RwLock::new(RegistryInner::default()),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_route_capacity(mut self, route_capacity: usize) -> Self {
        self.route_capacity = route_capacity;
        self
    }

    /// Changes the retry hint used by wait directives.
    #[cfg(test)]
    pub(crate) fn with_retry_after_ms(mut self, retry_after_ms: u64) -> Self {
        self.retry_after_ms = retry_after_ms;
        self
    }

    /// Restores a persisted token/fingerprint binding without creating a live reference.
    #[cfg(test)]
    pub(crate) fn restore_binding(
        &self,
        fingerprint: Fingerprint,
        token_digest: TokenDigest,
    ) -> Result<(), RegistryError> {
        let mut inner = self.write();
        evict_inactive_routes_at_capacity(&mut inner, fingerprint, self.route_capacity);
        validate_binding(&inner, fingerprint, token_digest)?;
        validate_capacity(&inner, fingerprint, self.route_capacity)?;
        inner.tokens.insert(token_digest, fingerprint);
        inner
            .routes
            .entry(fingerprint)
            .or_insert_with(|| RouteEntry::new(token_digest, self.global_pass_through));
        Ok(())
    }

    /// Registers or renews an MCP and returns the daemon's authoritative directive.
    ///
    /// The launch plan is used only when this call wins the empty-route singleflight.
    pub(crate) fn register_mcp(
        &self,
        registration: McpRegistration,
        launch: WorkerLaunch,
    ) -> Result<BrokerDirective, RegistryError> {
        let mut inner = self.write();
        evict_inactive_routes_at_capacity(
            &mut inner,
            registration.fingerprint,
            self.route_capacity,
        );
        validate_binding(&inner, registration.fingerprint, registration.token_digest)?;
        validate_capacity(&inner, registration.fingerprint, self.route_capacity)?;
        inner
            .tokens
            .insert(registration.token_digest, registration.fingerprint);
        let route = inner
            .routes
            .entry(registration.fingerprint)
            .or_insert_with(|| {
                RouteEntry::new(registration.token_digest, self.global_pass_through)
            });
        if !route.refs.contains_key(&registration.session_id)
            && route.refs.len() >= MAX_MCP_REFERENCES_PER_ROUTE
        {
            return Err(RegistryError::McpReferenceCapacityReached);
        }
        route.refs.insert(
            registration.session_id.clone(),
            registration.lease_expires_at_unix_ms,
        );
        Ok(route.directive_for(&registration.session_id, launch, self.retry_after_ms))
    }

    /// Renews an existing MCP reference without changing its lifecycle state.
    pub(crate) fn renew_mcp(
        &self,
        fingerprint: Fingerprint,
        session_id: &McpSessionId,
        lease_expires_at_unix_ms: u64,
    ) -> Result<(), RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let expiry = route
            .refs
            .get_mut(session_id)
            .ok_or(RegistryError::UnknownMcpSession)?;
        *expiry = lease_expires_at_unix_ms;
        Ok(())
    }

    /// Releases an MCP reference and begins teardown when the final reference leaves.
    pub(crate) fn release_mcp(
        &self,
        fingerprint: Fingerprint,
        session_id: &McpSessionId,
        drain_deadline_unix_ms: u64,
    ) -> Result<ReleaseAction, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        if route.refs.remove(session_id).is_none() {
            return Ok(ReleaseAction::NoChange);
        }
        Ok(route.after_reference_removed(session_id, drain_deadline_unix_ms))
    }

    /// Expires dead MCP leases and returns any resulting teardown or ownership actions.
    pub(crate) fn expire_mcp_leases(
        &self,
        now_unix_ms: u64,
        drain_deadline_unix_ms: u64,
    ) -> Vec<(Fingerprint, ReleaseAction)> {
        let mut inner = self.write();
        let mut actions = Vec::new();
        for (fingerprint, route) in &mut inner.routes {
            let expired: Vec<_> = route
                .refs
                .iter()
                .filter_map(|(session, expiry)| (*expiry <= now_unix_ms).then_some(session.clone()))
                .collect();
            let removed_owner = match &route.state {
                RouteState::Activating { owner, .. }
                    if expired.iter().any(|session| session == owner) =>
                {
                    Some(owner.clone())
                }
                RouteState::Recovering {
                    owner: Some(owner), ..
                } if expired.iter().any(|session| session == owner) => Some(owner.clone()),
                _ => None,
            };
            let Some(removed_session) = removed_owner.or_else(|| expired.first().cloned()) else {
                continue;
            };
            for session in expired {
                route.refs.remove(&session);
            }
            let action = route.after_reference_removed(&removed_session, drain_deadline_unix_ms);
            if !matches!(action, ReleaseAction::NoChange) {
                actions.push((*fingerprint, action));
            }
        }
        actions
    }

    /// Publishes a worker only when its one-time activation ID matches the active generation.
    pub(crate) fn mark_worker_ready(
        &self,
        fingerprint: Fingerprint,
        activation_id: &str,
        target: Arc<WorkerTarget>,
    ) -> Result<(), RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        if route.refs.is_empty() {
            return Err(RegistryError::NoLiveMcpReferences);
        }
        match &route.state {
            RouteState::Activating { launch, .. } if launch.activation_id == activation_id => {
                route.state = RouteState::Ready { target };
                Ok(())
            }
            RouteState::Activating { .. } => Err(RegistryError::ActivationMismatch),
            state => Err(RegistryError::InvalidState {
                expected: RouteStateKind::Activating,
                actual: state.kind(),
            }),
        }
    }

    /// Converts a failed authenticated activation into shared transient pass-through.
    pub(crate) fn mark_activation_failed(
        &self,
        fingerprint: Fingerprint,
        activation_id: &str,
    ) -> Result<(), RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        match &route.state {
            RouteState::Activating { launch, .. } if launch.activation_id == activation_id => {
                route.state = if route.refs.is_empty() {
                    RouteState::Empty
                } else {
                    RouteState::PassThrough { permanent: false }
                };
                Ok(())
            }
            RouteState::Activating { .. } => Err(RegistryError::ActivationMismatch),
            state => Err(RegistryError::InvalidState {
                expected: RouteStateKind::Activating,
                actual: state.kind(),
            }),
        }
    }

    /// Expires activation grants and moves every still-referenced route to transient pass-through.
    ///
    /// This is deliberately separate from expiring the server's secret-bearing grant table: the
    /// broker lifecycle must never remain `Activating` after its signed launch deadline passes.
    pub(crate) fn expire_activations(&self, now_unix_ms: u64) -> Vec<ExpiredActivation> {
        let mut inner = self.write();
        let mut expired = Vec::new();
        for (fingerprint, route) in &mut inner.routes {
            let activation_id = match &route.state {
                RouteState::Activating { launch, .. } if launch.deadline_unix_ms <= now_unix_ms => {
                    Some(launch.activation_id.clone())
                }
                _ => None,
            };
            let Some(activation_id) = activation_id else {
                continue;
            };
            route.state = if route.refs.is_empty() {
                RouteState::Empty
            } else {
                RouteState::PassThrough { permanent: false }
            };
            expired.push(ExpiredActivation {
                fingerprint: *fingerprint,
                activation_id,
            });
        }
        expired
    }

    /// Converts an authenticated worker communication failure into route-wide pass-through.
    ///
    /// The worker ID prevents a delayed failure from an old stream from displacing a newer ready
    /// generation. An activation that raced the failed request is returned so its grant can be
    /// revoked by the control plane.
    pub(crate) fn mark_worker_communication_failed(
        &self,
        fingerprint: Fingerprint,
        worker_id: &str,
    ) -> Result<Option<String>, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let state = std::mem::replace(&mut route.state, RouteState::Empty);
        let canceled_activation = match state {
            RouteState::Ready { target } if target.worker_id() == worker_id => None,
            RouteState::Draining {
                target,
                deadline_unix_ms,
            } if target.worker_id() == worker_id => {
                route.state = RouteState::Draining {
                    target,
                    deadline_unix_ms,
                };
                return Err(RegistryError::InvalidState {
                    expected: RouteStateKind::Ready,
                    actual: RouteStateKind::Draining,
                });
            }
            RouteState::Recovering { target, .. }
                if target
                    .as_ref()
                    .is_none_or(|target| target.worker_id() == worker_id) =>
            {
                None
            }
            RouteState::Activating { launch, .. } => Some(launch.activation_id),
            RouteState::PassThrough { permanent } => {
                route.state = RouteState::PassThrough { permanent };
                return Ok(None);
            }
            RouteState::Empty => return Ok(None),
            RouteState::Ready { target } => {
                route.state = RouteState::Ready { target };
                return Err(RegistryError::WorkerMismatch);
            }
            RouteState::Draining {
                target,
                deadline_unix_ms,
            } => {
                route.state = RouteState::Draining {
                    target,
                    deadline_unix_ms,
                };
                return Err(RegistryError::WorkerMismatch);
            }
            RouteState::Recovering {
                target,
                owner,
                deadline_unix_ms,
            } => {
                route.state = RouteState::Recovering {
                    target,
                    owner,
                    deadline_unix_ms,
                };
                return Err(RegistryError::WorkerMismatch);
            }
        };
        route.state = if route.refs.is_empty() {
            RouteState::Empty
        } else {
            RouteState::PassThrough { permanent: false }
        };
        Ok(canceled_activation)
    }

    /// Forces an authenticated route into transient pass-through after activation setup fails.
    pub(crate) fn mark_route_pass_through(
        &self,
        fingerprint: Fingerprint,
    ) -> Result<Option<String>, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        if matches!(&route.state, RouteState::PassThrough { permanent: true }) {
            return Ok(None);
        }
        let canceled_activation = match &route.state {
            RouteState::Activating { launch, .. } => Some(launch.activation_id.clone()),
            _ => None,
        };
        route.state = if route.refs.is_empty() {
            RouteState::Empty
        } else {
            RouteState::PassThrough { permanent: false }
        };
        Ok(canceled_activation)
    }

    /// Records a ready worker failure and nominates one live MCP to relaunch it.
    pub(crate) fn worker_failed(
        &self,
        fingerprint: Fingerprint,
        worker_id: &str,
        recovery_deadline_unix_ms: u64,
    ) -> Result<WorkerFailureAction, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let state = std::mem::replace(&mut route.state, RouteState::Empty);
        match state {
            RouteState::Ready { target } if target.worker_id() == worker_id => {
                let owner = route.refs.keys().next().cloned();
                if let Some(owner) = owner {
                    route.state = RouteState::Recovering {
                        target: None,
                        owner: Some(owner.clone()),
                        deadline_unix_ms: recovery_deadline_unix_ms,
                    };
                    Ok(WorkerFailureAction::NominateMcp { session_id: owner })
                } else {
                    Ok(WorkerFailureAction::RouteEmpty)
                }
            }
            RouteState::Ready { target } => {
                route.state = RouteState::Ready { target };
                Err(RegistryError::WorkerMismatch)
            }
            other => {
                let actual = other.kind();
                route.state = other;
                Err(RegistryError::InvalidState {
                    expected: RouteStateKind::Ready,
                    actual,
                })
            }
        }
    }

    /// Installs a fresh activation plan after the broker nominates a replacement owner.
    pub(crate) fn begin_relaunch(
        &self,
        fingerprint: Fingerprint,
        session_id: &McpSessionId,
        launch: WorkerLaunch,
    ) -> Result<BrokerDirective, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        match &route.state {
            RouteState::Recovering {
                owner: Some(owner), ..
            } if owner == session_id => {
                route.state = RouteState::Activating {
                    owner: session_id.clone(),
                    launch: launch.clone(),
                };
                Ok(launch.into_directive())
            }
            RouteState::Recovering { .. } => Err(RegistryError::NotLaunchOwner),
            state => Err(RegistryError::InvalidState {
                expected: RouteStateKind::Recovering,
                actual: state.kind(),
            }),
        }
    }

    /// Places a restored route into bounded daemon-restart recovery.
    #[cfg(test)]
    pub(crate) fn begin_recovery(
        &self,
        fingerprint: Fingerprint,
        target: Option<Arc<WorkerTarget>>,
        deadline_unix_ms: u64,
    ) -> Result<(), RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let owner = route.refs.keys().next().cloned();
        route.state = RouteState::Recovering {
            target,
            owner,
            deadline_unix_ms,
        };
        Ok(())
    }

    /// Ends restart recovery at its deadline and returns the required next action.
    #[cfg(test)]
    pub(crate) fn finish_recovery(
        &self,
        fingerprint: Fingerprint,
        now_unix_ms: u64,
        drain_deadline_unix_ms: u64,
    ) -> Result<RecoveryAction, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let state = std::mem::replace(&mut route.state, RouteState::Empty);
        match state {
            RouteState::Recovering {
                target,
                deadline_unix_ms,
                ..
            } if now_unix_ms >= deadline_unix_ms => match (target, route.refs.is_empty()) {
                (Some(target), false) => {
                    route.state = RouteState::Ready { target };
                    Ok(RecoveryAction::WorkerRecovered)
                }
                (Some(target), true) => {
                    route.state = RouteState::Draining {
                        target: Arc::clone(&target),
                        deadline_unix_ms: drain_deadline_unix_ms,
                    };
                    Ok(RecoveryAction::BeginDrain {
                        target,
                        deadline_unix_ms: drain_deadline_unix_ms,
                    })
                }
                (None, false) => {
                    let session_id = route
                        .refs
                        .keys()
                        .next()
                        .expect("route has live references")
                        .clone();
                    route.state = RouteState::Recovering {
                        target: None,
                        owner: Some(session_id.clone()),
                        deadline_unix_ms,
                    };
                    Ok(RecoveryAction::NominateMcp { session_id })
                }
                (None, true) => Ok(RecoveryAction::RouteEmpty),
            },
            RouteState::Recovering {
                target,
                owner,
                deadline_unix_ms,
            } => {
                route.state = RouteState::Recovering {
                    target,
                    owner,
                    deadline_unix_ms,
                };
                Err(RegistryError::RecoveryInProgress)
            }
            other => {
                let actual = other.kind();
                route.state = other;
                Err(RegistryError::InvalidState {
                    expected: RouteStateKind::Recovering,
                    actual,
                })
            }
        }
    }

    /// Completes a drained worker after all requests finish or the deadline elapses.
    pub(crate) fn finish_draining(
        &self,
        fingerprint: Fingerprint,
        now_unix_ms: u64,
    ) -> Result<DrainCompletion, RegistryError> {
        let mut inner = self.write();
        let route = inner
            .routes
            .get_mut(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        match &route.state {
            RouteState::Draining {
                target,
                deadline_unix_ms,
            } if target.in_flight() != 0 && now_unix_ms < *deadline_unix_ms => {
                return Err(RegistryError::DrainInProgress);
            }
            RouteState::Draining { .. } => {}
            state => {
                return Err(RegistryError::InvalidState {
                    expected: RouteStateKind::Draining,
                    actual: state.kind(),
                });
            }
        }
        route.state = RouteState::Empty;
        Ok(route
            .refs
            .keys()
            .next()
            .cloned()
            .map_or(DrainCompletion::RouteEmpty, |session_id| {
                DrainCompletion::ActivationRequired { session_id }
            }))
    }

    /// Resolves and acquires a route from the request header without polling its body.
    pub(crate) fn resolve_target(
        &self,
        token_digest: &TokenDigest,
    ) -> Result<ResolvedTarget, ResolveError> {
        let inner = self.read();
        let fingerprint = inner
            .tokens
            .get(token_digest)
            .ok_or(ResolveError::UnknownToken)?;
        let route = inner
            .routes
            .get(fingerprint)
            .ok_or(ResolveError::UnknownToken)?;
        match &route.state {
            RouteState::Ready { target } => {
                Ok(ResolvedTarget::Worker(target.acquire(*fingerprint)))
            }
            RouteState::PassThrough { .. } if !route.refs.is_empty() => {
                Ok(ResolvedTarget::PassThrough)
            }
            RouteState::PassThrough { .. } => {
                Err(ResolveError::Unavailable(RouteStateKind::PassThrough))
            }
            state => Err(ResolveError::Unavailable(state.kind())),
        }
    }

    /// Returns a credential-free route snapshot for status and tests.
    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        fingerprint: Fingerprint,
    ) -> Result<RouteSnapshot, RegistryError> {
        let inner = self.read();
        let route = inner
            .routes
            .get(&fingerprint)
            .ok_or(RegistryError::UnknownRoute)?;
        let (launch_owner, endpoint, in_flight) = match &route.state {
            RouteState::Activating { owner, .. } => (Some(owner.clone()), None, 0),
            RouteState::Ready { target } | RouteState::Draining { target, .. } => {
                (None, Some(target.endpoint().to_owned()), target.in_flight())
            }
            RouteState::Recovering { target, owner, .. } => (
                owner.clone(),
                target.as_ref().map(|target| target.endpoint().to_owned()),
                target.as_ref().map_or(0, |target| target.in_flight()),
            ),
            RouteState::Empty | RouteState::PassThrough { .. } => (None, None, 0),
        };
        Ok(RouteSnapshot {
            state: route.state.kind(),
            reference_count: route.refs.len(),
            launch_owner,
            endpoint,
            in_flight,
        })
    }

    fn read(&self) -> RwLockReadGuard<'_, RegistryInner> {
        self.inner.read().unwrap_or_else(|error| error.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, RegistryInner> {
        self.inner
            .write()
            .unwrap_or_else(|error| error.into_inner())
    }
}

fn evict_inactive_routes_at_capacity(
    inner: &mut RegistryInner,
    incoming: Fingerprint,
    capacity: usize,
) {
    if inner.routes.contains_key(&incoming) || inner.routes.len() < capacity {
        return;
    }
    let removable = inner.routes.iter().find_map(|(fingerprint, route)| {
        (route.refs.is_empty()
            && matches!(
                route.state,
                RouteState::Empty | RouteState::PassThrough { permanent: true }
            ))
        .then_some((*fingerprint, route.token_digest))
    });
    if let Some((fingerprint, token_digest)) = removable {
        inner.routes.remove(&fingerprint);
        inner.tokens.remove(&token_digest);
    }
}

#[derive(Default)]
struct RegistryInner {
    routes: HashMap<Fingerprint, RouteEntry>,
    tokens: HashMap<TokenDigest, Fingerprint>,
}

struct RouteEntry {
    token_digest: TokenDigest,
    refs: BTreeMap<McpSessionId, u64>,
    state: RouteState,
}

impl RouteEntry {
    fn new(token_digest: TokenDigest, global_pass_through: bool) -> Self {
        Self {
            token_digest,
            refs: BTreeMap::new(),
            state: if global_pass_through {
                RouteState::PassThrough { permanent: true }
            } else {
                RouteState::Empty
            },
        }
    }

    fn directive_for(
        &mut self,
        session_id: &McpSessionId,
        launch: WorkerLaunch,
        retry_after_ms: u64,
    ) -> BrokerDirective {
        match &self.state {
            RouteState::Empty => {
                self.state = RouteState::Activating {
                    owner: session_id.clone(),
                    launch: launch.clone(),
                };
                launch.into_directive()
            }
            RouteState::Activating {
                owner,
                launch: active_launch,
            } if owner == session_id => active_launch.clone().into_directive(),
            RouteState::Activating { .. }
            | RouteState::Draining { .. }
            | RouteState::Recovering { target: None, .. } => {
                BrokerDirective::WaitForWorker { retry_after_ms }
            }
            RouteState::Ready { target } => BrokerDirective::ReuseWorker {
                endpoint: target.endpoint().to_owned(),
            },
            RouteState::PassThrough { .. } => BrokerDirective::UsePassThrough,
            RouteState::Recovering {
                target: Some(target),
                ..
            } => {
                let endpoint = target.endpoint().to_owned();
                self.state = RouteState::Ready {
                    target: Arc::clone(target),
                };
                BrokerDirective::ReuseWorker { endpoint }
            }
        }
    }

    fn after_reference_removed(
        &mut self,
        removed_session: &McpSessionId,
        drain_deadline_unix_ms: u64,
    ) -> ReleaseAction {
        if !self.refs.is_empty() {
            return self.transfer_owner_if_needed(removed_session);
        }
        let state = std::mem::replace(&mut self.state, RouteState::Empty);
        match state {
            RouteState::Activating { launch, .. } => ReleaseAction::CancelActivation {
                activation_id: launch.activation_id,
            },
            RouteState::Ready { target } => {
                self.state = RouteState::Draining {
                    target: Arc::clone(&target),
                    deadline_unix_ms: drain_deadline_unix_ms,
                };
                ReleaseAction::BeginDrain {
                    target,
                    deadline_unix_ms: drain_deadline_unix_ms,
                }
            }
            RouteState::Recovering {
                target: Some(target),
                ..
            } => {
                self.state = RouteState::Draining {
                    target: Arc::clone(&target),
                    deadline_unix_ms: drain_deadline_unix_ms,
                };
                ReleaseAction::BeginDrain {
                    target,
                    deadline_unix_ms: drain_deadline_unix_ms,
                }
            }
            RouteState::PassThrough { permanent: true } => {
                self.state = RouteState::PassThrough { permanent: true };
                ReleaseAction::NoChange
            }
            RouteState::Draining {
                target,
                deadline_unix_ms,
            } => {
                self.state = RouteState::Draining {
                    target,
                    deadline_unix_ms,
                };
                ReleaseAction::NoChange
            }
            RouteState::Empty
            | RouteState::PassThrough { permanent: false }
            | RouteState::Recovering { target: None, .. } => ReleaseAction::NoChange,
        }
    }

    fn transfer_owner_if_needed(&mut self, removed_session: &McpSessionId) -> ReleaseAction {
        let replacement = self
            .refs
            .keys()
            .next()
            .expect("route has live references")
            .clone();
        match &mut self.state {
            RouteState::Activating { owner, launch } if owner == removed_session => {
                *owner = replacement.clone();
                ReleaseAction::TransferActivation {
                    session_id: replacement,
                    directive: launch.clone().into_directive(),
                }
            }
            RouteState::Recovering { owner, .. }
                if owner.as_ref().is_some_and(|owner| owner == removed_session) =>
            {
                *owner = Some(replacement.clone());
                ReleaseAction::NominateMcp {
                    session_id: replacement,
                }
            }
            _ => ReleaseAction::NoChange,
        }
    }
}

fn validate_binding(
    inner: &RegistryInner,
    fingerprint: Fingerprint,
    token_digest: TokenDigest,
) -> Result<(), RegistryError> {
    if inner
        .tokens
        .get(&token_digest)
        .is_some_and(|existing| *existing != fingerprint)
    {
        return Err(RegistryError::TokenAlreadyBound);
    }
    if inner
        .routes
        .get(&fingerprint)
        .is_some_and(|existing| !existing.token_digest.matches(&token_digest))
    {
        return Err(RegistryError::FingerprintTokenMismatch);
    }
    Ok(())
}

fn validate_capacity(
    inner: &RegistryInner,
    fingerprint: Fingerprint,
    route_capacity: usize,
) -> Result<(), RegistryError> {
    if !inner.routes.contains_key(&fingerprint) && inner.routes.len() >= route_capacity {
        return Err(RegistryError::RouteCapacityReached);
    }
    Ok(())
}

/// A credential-free lifecycle snapshot.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteSnapshot {
    pub(crate) state: RouteStateKind,
    pub(crate) reference_count: usize,
    pub(crate) launch_owner: Option<McpSessionId>,
    pub(crate) endpoint: Option<String>,
    pub(crate) in_flight: usize,
}

/// An activation grant whose signed launch deadline elapsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpiredActivation {
    pub(crate) fingerprint: Fingerprint,
    pub(crate) activation_id: String,
}

/// Work required after releasing or expiring an MCP reference.
#[derive(Debug)]
pub(crate) enum ReleaseAction {
    NoChange,
    CancelActivation {
        activation_id: String,
    },
    BeginDrain {
        target: Arc<WorkerTarget>,
        deadline_unix_ms: u64,
    },
    TransferActivation {
        session_id: McpSessionId,
        directive: BrokerDirective,
    },
    NominateMcp {
        session_id: McpSessionId,
    },
}

/// Work required after a ready worker disconnects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorkerFailureAction {
    NominateMcp { session_id: McpSessionId },
    RouteEmpty,
}

/// Exact route generation authorized to attempt worker recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecoveryPermit {
    Activating { activation_id: String },
    ExistingWorker { worker_id: String, recovering: bool },
}

/// Work required when bounded daemon-restart recovery ends.
#[cfg(test)]
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum RecoveryAction {
    WorkerRecovered,
    BeginDrain {
        target: Arc<WorkerTarget>,
        deadline_unix_ms: u64,
    },
    NominateMcp {
        session_id: McpSessionId,
    },
    RouteEmpty,
}

/// Result of completing a worker drain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DrainCompletion {
    RouteEmpty,
    ActivationRequired { session_id: McpSessionId },
}

/// Registry mutation failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum RegistryError {
    #[error("the route token is already bound to a different user-machine fingerprint")]
    TokenAlreadyBound,
    #[error("the user-machine fingerprint is already bound to a different route token")]
    FingerprintTokenMismatch,
    #[error("the broker route does not exist")]
    UnknownRoute,
    #[error("the broker route binding capacity has been reached")]
    RouteCapacityReached,
    #[error("the MCP session is not registered on this route")]
    UnknownMcpSession,
    #[error("the route MCP reference capacity has been reached")]
    McpReferenceCapacityReached,
    #[error("the route no longer has a live MCP reference")]
    NoLiveMcpReferences,
    #[error("the worker activation ID does not match the active generation")]
    ActivationMismatch,
    #[error("the worker ID does not match the active generation")]
    WorkerMismatch,
    #[error("this MCP session is not the nominated launch owner")]
    NotLaunchOwner,
    #[error("this route generation is not eligible for worker recovery")]
    RecoveryNotAuthorized,
    #[error("the route generation changed during worker recovery")]
    RecoveryGenerationChanged,
    #[error("the route is {actual:?}; expected {expected:?}")]
    InvalidState {
        expected: RouteStateKind,
        actual: RouteStateKind,
    },
    #[error("worker recovery is still within its grace period")]
    #[cfg(test)]
    RecoveryInProgress,
    #[error("the worker still has in-flight requests before its drain deadline")]
    DrainInProgress,
}

/// Request-route resolution failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ResolveError {
    #[error("the route token is unknown")]
    UnknownToken,
    #[error("the authenticated route is not ready: {0:?}")]
    Unavailable(RouteStateKind),
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/registry_tests.rs"]
mod tests;
