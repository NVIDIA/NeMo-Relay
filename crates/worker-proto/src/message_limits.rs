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

/// Host overrides keyed by exact callback registration name, as a JSON byte-limit map.
pub const WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV: &str =
    "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES_BY_NAME";

/// Resolved worker transport limits shared by the host and maintained SDKs.
#[derive(Debug, Clone)]
pub struct WorkerCallbackMessageLimits {
    /// Fallback for callbacks without an override and for control RPCs.
    pub default_bytes: usize,
    /// Overrides keyed by exact callback registration name, like callback timeouts.
    pub by_name: BTreeMap<String, usize>,
}

impl Default for WorkerCallbackMessageLimits {
    fn default() -> Self {
        Self {
            default_bytes: DEFAULT_WORKER_MAX_MESSAGE_BYTES,
            by_name: BTreeMap::new(),
        }
    }
}

impl WorkerCallbackMessageLimits {
    /// Resolve host named/default overrides ahead of plugin named/default settings.
    ///
    /// # Errors
    /// Rejects invalid finite byte limits or malformed callback names/override JSON.
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

    /// Return the finite send/receive limit for one callback (empty for control RPCs).
    #[must_use]
    pub fn for_callback(&self, name: &str) -> usize {
        self.by_name
            .get(name)
            .copied()
            .unwrap_or(self.default_bytes)
    }

    fn validate(&self) -> Result<(), String> {
        parse_worker_max_message_bytes(&self.default_bytes.to_string())?;
        for (name, value) in &self.by_name {
            if name.trim().is_empty()
                || name != name.trim()
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
        "{WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV} must map non-empty callback registration names without surrounding whitespace to integers in 1..=2147483647"
    )
}

/// Select a generated tonic server's codec limits before decoding each RPC body.
#[derive(Clone)]
pub struct CallbackMessageLimitService<S> {
    inner: S,
    limits: Arc<WorkerCallbackMessageLimits>,
    configure: fn(S, usize) -> S,
}

impl<S> CallbackMessageLimitService<S> {
    /// Wrap a generated tonic server with its send/receive configuration function.
    pub fn new(
        inner: S,
        limits: WorkerCallbackMessageLimits,
        configure: fn(S, usize) -> S,
    ) -> Self {
        Self {
            inner,
            limits: Arc::new(limits),
            configure,
        }
    }
}

impl<S: tonic::server::NamedService> tonic::server::NamedService
    for CallbackMessageLimitService<S>
{
    const NAME: &'static str = S::NAME;
}

impl<S, B> Service<http::Request<B>> for CallbackMessageLimitService<S>
where
    S: Service<http::Request<B>> + Clone + tonic::server::NamedService,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let method = request.uri().path().rsplit('/').next().unwrap_or_default();
        let metadata = tonic::metadata::MetadataMap::from_headers(request.headers().clone());
        let callback = if S::NAME == "nemo.relay.worker.v1.PluginWorker"
            && !matches!(method, "Invoke" | "InvokeStream")
        {
            None
        } else {
            callback_name(&metadata)
        };
        // Generated tonic servers are always ready; their clones share the handler.
        let mut service = (self.configure)(
            self.inner.clone(),
            self.limits
                .for_callback(callback.as_deref().unwrap_or_default()),
        );
        service.call(request)
    }
}

/// Binary metadata preserves exact UTF-8 registration names before body decoding.
pub const WORKER_CALLBACK_METADATA: &str = "nemo-relay-worker-callback-bin";

/// Read callback identity independently of an RPC method name.
#[must_use]
pub fn callback_name(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
    let value = metadata
        .get_bin(WORKER_CALLBACK_METADATA)?
        .to_bytes()
        .ok()?;
    String::from_utf8(value.to_vec()).ok()
}

/// Attach an invocation's registered callback name to a request.
pub fn callback_request<T>(message: T, name: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request.metadata_mut().insert_bin(
        WORKER_CALLBACK_METADATA,
        tonic::metadata::MetadataValue::from_bytes(name.as_bytes()),
    );
    request
}

/// Metadata interceptor for callback-scoped worker-to-host runtime calls.
#[derive(Clone)]
pub struct WorkerCallbackMetadata(pub Option<String>);

impl tonic::service::Interceptor for WorkerCallbackMetadata {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        if let Some(name) = &self.0 {
            request.metadata_mut().insert_bin(
                WORKER_CALLBACK_METADATA,
                tonic::metadata::MetadataValue::from_bytes(name.as_bytes()),
            );
        }
        Ok(request)
    }
}

/// Reject mismatched callback metadata; named limits require the updated host.
///
/// # Errors
/// Returns InvalidArgument for a missing required or mismatched callback identity.
pub fn validate_callback_request<T>(
    request: &tonic::Request<T>,
    name: &str,
    limits: &WorkerCallbackMessageLimits,
) -> Result<(), tonic::Status> {
    match callback_name(request.metadata()) {
        Some(actual) if actual != name => Err(tonic::Status::invalid_argument(
            "callback metadata does not match registration_name",
        )),
        None if limits.by_name.contains_key(name) => Err(tonic::Status::invalid_argument(
            "named message limits require callback metadata",
        )),
        _ => Ok(()),
    }
}
