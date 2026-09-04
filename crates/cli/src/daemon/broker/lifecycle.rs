// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::daemon::common::identity::Fingerprint;
use crate::daemon::common::protocol::{SensitiveString, WorkerLaunch};
use crate::daemon::common::transport::PooledClient;
#[cfg(test)]
use crate::daemon::common::transport::pooled_client;

/// Stable identity for one connected MCP process.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct McpSessionId(String);

impl McpSessionId {
    /// Constructs a non-empty MCP session identifier.
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, LifecycleError> {
        let value = value.into();
        if value.is_empty() {
            return Err(LifecycleError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    /// Returns the wire representation.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// An immutable destination for one registered worker generation.
pub(crate) struct WorkerTarget {
    worker_id: String,
    endpoint: String,
    session_token: SensitiveString,
    client: Arc<PooledClient>,
    in_flight: AtomicUsize,
}

impl WorkerTarget {
    /// Creates an authenticated worker target.
    #[cfg(test)]
    pub(crate) fn new(
        worker_id: impl Into<String>,
        endpoint: impl Into<String>,
        session_token: SensitiveString,
    ) -> Result<Self, LifecycleError> {
        let client = pooled_client().map_err(|_| LifecycleError::TransportInitialization)?;
        Self::with_client(worker_id, endpoint, session_token, client)
    }

    /// Creates a target with a standalone pool for deterministic transport tests.
    #[cfg(test)]
    pub(crate) fn with_client(
        worker_id: impl Into<String>,
        endpoint: impl Into<String>,
        session_token: SensitiveString,
        client: PooledClient,
    ) -> Result<Self, LifecycleError> {
        Self::with_shared_client(worker_id, endpoint, session_token, Arc::new(client))
    }

    /// Creates a target that retains a handle to the daemon's process-wide worker pool service.
    pub(crate) fn with_shared_client(
        worker_id: impl Into<String>,
        endpoint: impl Into<String>,
        session_token: SensitiveString,
        client: Arc<PooledClient>,
    ) -> Result<Self, LifecycleError> {
        let worker_id = worker_id.into();
        let endpoint = endpoint.into();
        if worker_id.is_empty() || endpoint.is_empty() {
            return Err(LifecycleError::EmptyIdentifier);
        }
        Ok(Self {
            worker_id,
            endpoint,
            session_token,
            client,
            in_flight: AtomicUsize::new(0),
        })
    }

    /// Returns the worker generation identifier.
    pub(crate) fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Returns the daemon-reachable worker endpoint.
    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Returns the internal daemon-to-worker credential.
    pub(crate) fn session_token(&self) -> &str {
        self.session_token.expose()
    }

    /// Returns the shared pool selected for this worker's transport trust identity.
    pub(crate) fn client(&self) -> &PooledClient {
        self.client.as_ref()
    }

    /// Returns the number of requests accepted by the broker and not yet dropped.
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    pub(super) fn acquire(self: &Arc<Self>, fingerprint: Fingerprint) -> WorkerRequest {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        WorkerRequest {
            fingerprint,
            target: Arc::clone(self),
        }
    }
}

impl fmt::Debug for WorkerTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerTarget")
            .field("worker_id", &self.worker_id)
            .field("endpoint", &self.endpoint)
            .field("session_token", &self.session_token)
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

/// An accepted request's ownership of one worker target.
pub(crate) struct WorkerRequest {
    fingerprint: Fingerprint,
    target: Arc<WorkerTarget>,
}

impl WorkerRequest {
    /// Returns the stable route identity that selected this worker.
    pub(crate) const fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// Returns the immutable worker target.
    pub(crate) fn target(&self) -> &Arc<WorkerTarget> {
        &self.target
    }

    /// Returns the internal daemon-to-worker credential.
    pub(crate) fn session_token(&self) -> &str {
        self.target.session_token()
    }
}

impl fmt::Debug for WorkerRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerRequest")
            .field("fingerprint", &self.fingerprint)
            .field("target", &self.target)
            .finish()
    }
}

impl Drop for WorkerRequest {
    fn drop(&mut self) {
        self.target.in_flight.fetch_sub(1, Ordering::Release);
    }
}

/// A route destination resolved before any request-body frame is polled.
#[derive(Debug)]
pub(crate) enum ResolvedTarget {
    Worker(WorkerRequest),
    PassThrough,
}

/// The externally useful category of a route's lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteStateKind {
    Empty,
    Activating,
    Ready,
    Draining,
    PassThrough,
    Recovering,
}

/// Internal state for one fingerprint route.
#[derive(Debug)]
pub(crate) enum RouteState {
    Empty,
    Activating {
        owner: McpSessionId,
        launch: WorkerLaunch,
    },
    Ready {
        target: Arc<WorkerTarget>,
    },
    Draining {
        target: Arc<WorkerTarget>,
        deadline_unix_ms: u64,
    },
    PassThrough {
        permanent: bool,
    },
    Recovering {
        target: Option<Arc<WorkerTarget>>,
        owner: Option<McpSessionId>,
        deadline_unix_ms: u64,
    },
}

impl RouteState {
    /// Returns the state category without exposing credentials or mutable internals.
    pub(crate) const fn kind(&self) -> RouteStateKind {
        match self {
            Self::Empty => RouteStateKind::Empty,
            Self::Activating { .. } => RouteStateKind::Activating,
            Self::Ready { .. } => RouteStateKind::Ready,
            Self::Draining { .. } => RouteStateKind::Draining,
            Self::PassThrough { .. } => RouteStateKind::PassThrough,
            Self::Recovering { .. } => RouteStateKind::Recovering,
        }
    }
}

/// Validation failures for strongly typed lifecycle identifiers and targets.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum LifecycleError {
    #[error("a lifecycle identifier or endpoint cannot be empty")]
    EmptyIdentifier,
    #[error("failed to construct a worker transport pool")]
    #[cfg(test)]
    TransportInitialization,
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/lifecycle_tests.rs"]
mod tests;
