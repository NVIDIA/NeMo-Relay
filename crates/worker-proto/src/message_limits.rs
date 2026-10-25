// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::task::{Context, Poll};

use tonic::codegen::{Service, http};

use crate::{
    DEFAULT_WORKER_MAX_MESSAGE_BYTES, WORKER_MAX_MESSAGE_BYTES_ENV, parse_worker_max_message_bytes,
    worker_max_message_bytes_from_env_with_default,
};

/// Host overrides keyed by exact gRPC method name, as a JSON object of byte limits.
pub const WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV: &str =
    "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES_BY_NAME";

/// Resolved worker transport limits shared by the host and maintained SDKs.
#[derive(Debug, Clone)]
pub struct WorkerRpcMessageLimits {
    /// Fallback for RPCs without a named override.
    pub default_bytes: usize,
    /// Overrides keyed by exact, case-sensitive gRPC method name.
    pub by_name: BTreeMap<String, usize>,
}

impl Default for WorkerRpcMessageLimits {
    fn default() -> Self {
        Self {
            default_bytes: DEFAULT_WORKER_MAX_MESSAGE_BYTES,
            by_name: BTreeMap::new(),
        }
    }
}

impl WorkerRpcMessageLimits {
    /// Resolve host named/default overrides ahead of plugin named/default settings.
    ///
    /// # Errors
    /// Rejects invalid finite byte limits or malformed RPC names/override JSON.
    pub fn from_env_with_defaults(mut self) -> Result<Self, String> {
        self.validate()?;
        let host_default_present = std::env::var_os(WORKER_MAX_MESSAGE_BYTES_ENV).is_some();
        self.default_bytes = worker_max_message_bytes_from_env_with_default(self.default_bytes)?;
        if host_default_present {
            self.by_name.clear();
        }
        match std::env::var(WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV) {
            Ok(value) => {
                let overrides: BTreeMap<String, usize> =
                    serde_json::from_str(&value).map_err(|_| named_limit_error())?;
                self.by_name.extend(overrides);
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => return Err(named_limit_error()),
        }
        self.validate()?;
        Ok(self)
    }

    /// Return the finite send/receive limit for one RPC.
    #[must_use]
    pub fn for_rpc(&self, name: &str) -> usize {
        self.by_name
            .get(name)
            .copied()
            .unwrap_or(self.default_bytes)
    }

    fn validate(&self) -> Result<(), String> {
        parse_worker_max_message_bytes(&self.default_bytes.to_string())?;
        for (name, value) in &self.by_name {
            if name.is_empty()
                || !name.starts_with(|c: char| c.is_ascii_alphabetic())
                || !name.chars().all(|c| c.is_ascii_alphanumeric())
                || parse_worker_max_message_bytes(&value.to_string()).is_err()
            {
                return Err(named_limit_error());
            }
        }
        Ok(())
    }
}

fn named_limit_error() -> String {
    format!(
        "{WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV} must map gRPC method names to integers in 1..=2147483647"
    )
}

/// Select a generated tonic server's codec limits before decoding each RPC body.
#[derive(Clone)]
pub struct RpcMessageLimitService<S> {
    inner: S,
    limits: Arc<WorkerRpcMessageLimits>,
    configure: fn(S, usize) -> S,
}

impl<S> RpcMessageLimitService<S> {
    /// Wrap a generated tonic server with its send/receive configuration function.
    pub fn new(inner: S, limits: WorkerRpcMessageLimits, configure: fn(S, usize) -> S) -> Self {
        Self {
            inner,
            limits: Arc::new(limits),
            configure,
        }
    }
}

impl<S: tonic::server::NamedService> tonic::server::NamedService for RpcMessageLimitService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S, B> Service<http::Request<B>> for RpcMessageLimitService<S>
where
    S: Service<http::Request<B>> + Clone,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let method = request.uri().path().rsplit('/').next().unwrap_or_default();
        // Generated tonic servers are always ready; their clones share the handler.
        let mut service = (self.configure)(self.inner.clone(), self.limits.for_rpc(method));
        service.call(request)
    }
}
