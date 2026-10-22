// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authorized bounded reads. This adapter never touches execution activity or readiness.
use crate::configuration::BootstrapChallengeKey;
use axum::{
    Router,
    body::Body,
    extract::{OriginalUri, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use nemo_relay::diagnostics::*;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

pub(crate) const PREFIX: &str = "/_nemo_relay/v1/diagnostics";
pub(crate) const AUTH_HEADER: &str = "x-nemo-relay-diagnostics-read";
const MAX_QUERY_BYTES: usize = 4096;
const DISCOVERY_BYTES: usize = 8192;

#[derive(Clone)]
pub(crate) struct DiagnosticsHttp {
    reader: DiagnosticsReader,
    key: BootstrapChallengeKey,
    pool: Arc<Mutex<Vec<Vec<u8>>>>,
    capacity: usize,
    coverage: Arc<Vec<&'static str>>,
    closed: Arc<std::sync::atomic::AtomicBool>,
    shutdown: Arc<tokio::sync::Notify>,
    connections: Arc<tokio::sync::Semaphore>,
}
impl DiagnosticsHttp {
    pub(crate) fn new(
        reader: DiagnosticsReader,
        key: BootstrapChallengeKey,
        coverage: Vec<&'static str>,
    ) -> Result<Self, DiagnosticError> {
        let limits = reader.limits();
        let capacity = limits.max_snapshot_bytes.max(DISCOVERY_BYTES);
        // Discovery remains available with no query workspaces.
        let concurrency = limits.max_queries.max(1);
        let bytes = capacity
            .checked_mul(concurrency)
            .ok_or(DiagnosticError::InvalidLimits)?;
        if bytes > limits.max_owned_bytes {
            return Err(DiagnosticError::InvalidLimits);
        }
        let mut pool = Vec::new();
        pool.try_reserve_exact(concurrency)
            .map_err(|_| DiagnosticError::Allocation)?;
        for _ in 0..concurrency {
            let mut slot = Vec::new();
            slot.try_reserve_exact(capacity)
                .map_err(|_| DiagnosticError::Allocation)?;
            pool.push(slot);
        }
        Ok(Self {
            reader,
            key,
            pool: Arc::new(Mutex::new(pool)),
            capacity,
            coverage: Arc::new(coverage),
            closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            connections: Arc::new(tokio::sync::Semaphore::new(concurrency)),
        })
    }
    /// Plain HTTP never returns diagnostic content or accepts a bearer credential.
    /// The existing bootstrap TLS upgrade supplies the bounded authorized transport.
    pub(crate) fn unsecured_router(self) -> Router {
        Router::new()
            .route(PREFIX, get(tls_required))
            .route(&format!("{PREFIX}/components"), get(tls_required))
            .route(&format!("{PREFIX}/events"), get(tls_required))
            .route(&format!("{PREFIX}/capabilities"), get(tls_required))
    }
    pub(crate) fn router(self) -> Router {
        Router::new()
            .route(PREFIX, get(snapshot))
            .route(&format!("{PREFIX}/components"), get(snapshot))
            .route(&format!("{PREFIX}/events"), get(snapshot))
            .route(&format!("{PREFIX}/capabilities"), get(discovery))
            .with_state(self)
    }
    pub(crate) fn close(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        self.connections.close();
        self.shutdown.notify_waiters();
    }
    pub(crate) fn try_connection(
        &self,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::TryAcquireError> {
        self.connections.clone().try_acquire_owned()
    }
    pub(crate) async fn wait_closed(&self) {
        let notified = self.shutdown.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !self.closed.load(std::sync::atomic::Ordering::Acquire) {
            notified.await;
        }
    }
    fn authorize(&self, headers: &HeaderMap) -> Result<(), ReadError> {
        if headers.contains_key(header::ORIGIN) {
            return Err(ReadError(StatusCode::FORBIDDEN, "forbidden"));
        }
        let value = headers
            .get(AUTH_HEADER)
            .ok_or(ReadError(StatusCode::UNAUTHORIZED, "unauthorized"))?;
        if value.len() > 128
            || !value
                .to_str()
                .is_ok_and(|v| self.key.verify_diagnostics_token(v))
        {
            return Err(ReadError(StatusCode::FORBIDDEN, "forbidden"));
        }
        if self.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(ReadError(StatusCode::SERVICE_UNAVAILABLE, "shutting_down"));
        }
        Ok(())
    }
    fn reserve(&self) -> Result<ResponseSlot, ReadError> {
        let bytes = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .ok_or(ReadError(
                StatusCode::TOO_MANY_REQUESTS,
                "response_capacity",
            ))?;
        Ok(ResponseSlot {
            bytes,
            pool: self.pool.clone(),
            capacity: self.capacity,
        })
    }
}
struct ResponseSlot {
    bytes: Vec<u8>,
    pool: Arc<Mutex<Vec<Vec<u8>>>>,
    capacity: usize,
}
impl AsRef<[u8]> for ResponseSlot {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for ResponseSlot {
    fn drop(&mut self) {
        self.bytes.clear();
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(std::mem::take(&mut self.bytes));
    }
}
impl std::io::Write for ResponseSlot {
    fn write(&mut self, value: &[u8]) -> std::io::Result<usize> {
        if value.len() > self.capacity.saturating_sub(self.bytes.len()) {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        self.bytes.extend_from_slice(value);
        Ok(value.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn response(slot: ResponseSlot) -> Response {
    // The backing slot stays charged until the last network/body reference drops.
    let mut response = Response::new(Body::from(bytes::Bytes::from_owner(slot)));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert("x-content-type-options", "nosniff".parse().unwrap());
    response
}
struct ReadError(StatusCode, &'static str);
impl IntoResponse for ReadError {
    fn into_response(self) -> Response {
        error(self.0, self.1)
    }
}
fn error(status: StatusCode, reason: &'static str) -> Response {
    let mut response = (
        status,
        axum::Json(serde_json::json!({"error": {"reason": reason}})),
    )
        .into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    if status == StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, "1".parse().unwrap());
    }
    response
}
fn reader_error(value: DiagnosticError) -> Response {
    let (status, reason) = match value {
        DiagnosticError::Disabled => (StatusCode::SERVICE_UNAVAILABLE, "disabled"),
        DiagnosticError::QueryDisabled => (StatusCode::SERVICE_UNAVAILABLE, "query_disabled"),
        DiagnosticError::QueryCapacity => (StatusCode::TOO_MANY_REQUESTS, "query_capacity"),
        DiagnosticError::InvalidQuery => (StatusCode::BAD_REQUEST, "invalid_query"),
        DiagnosticError::InvalidCursor => (StatusCode::CONFLICT, "invalid_cursor"),
        DiagnosticError::SnapshotCapacity => {
            (StatusCode::UNPROCESSABLE_ENTITY, "snapshot_capacity")
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "reader_failed"),
    };
    error(status, reason)
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HttpQuery {
    schema_version: Option<u32>,
    component_key: Option<String>,
    component_instance_id: Option<uuid::Uuid>,
    code: Option<String>,
    failure_class: Option<String>,
    after_sequence: Option<DiagnosticNumber>,
    max_components: Option<u32>,
    max_events: Option<u32>,
    max_response_bytes: Option<u32>,
    continuation: Option<DiagnosticCursor>,
    domain: Option<DiagnosticDomain>,
    plugin_family: Option<PluginDiagnosticFamily>,
    ownership: ComponentOwnership,
    origin_generation: Option<DiagnosticNumber>,
}
fn decode(uri: &http::Uri) -> Result<DiagnosticQuery, ReadError> {
    // One percent-encoded JSON query avoids ambiguous duplicate-key/array URL encodings.
    let raw = uri.query().unwrap_or("");
    if raw.len() > MAX_QUERY_BYTES {
        return Err(ReadError(StatusCode::BAD_REQUEST, "query_too_large"));
    }
    let decoded = if raw.is_empty() {
        std::borrow::Cow::Borrowed("{}")
    } else {
        let raw = raw
            .strip_prefix("query=")
            .ok_or(ReadError(StatusCode::BAD_REQUEST, "invalid_query"))?;
        percent_encoding::percent_decode_str(raw)
            .decode_utf8()
            .map_err(|_| ReadError(StatusCode::BAD_REQUEST, "invalid_query"))?
    };
    let query: HttpQuery = serde_json::from_str(&decoded)
        .map_err(|_| ReadError(StatusCode::BAD_REQUEST, "invalid_query"))?;
    if query
        .schema_version
        .is_some_and(|v| v != DIAGNOSTIC_SCHEMA_VERSION)
    {
        return Err(ReadError(StatusCode::NOT_ACCEPTABLE, "unsupported_schema"));
    }
    for text in [&query.component_key, &query.code, &query.failure_class]
        .into_iter()
        .flatten()
    {
        if text.len() > 128 {
            return Err(ReadError(StatusCode::BAD_REQUEST, "invalid_selector"));
        }
    }
    Ok(DiagnosticQuery {
        component_key: query.component_key,
        component_instance_id: query.component_instance_id,
        code: query.code,
        failure_class: query.failure_class,
        after_sequence: query.after_sequence,
        max_components: query.max_components,
        max_events: query.max_events,
        max_response_bytes: query.max_response_bytes,
        continuation: query.continuation,
        domain: query.domain,
        plugin_family: query.plugin_family,
        ownership: query.ownership,
        origin_generation: query.origin_generation,
    })
}
async fn snapshot(
    State(adapter): State<DiagnosticsHttp>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = adapter.authorize(&headers) {
        return response.into_response();
    }
    let mut slot = match adapter.reserve() {
        Ok(slot) => slot,
        Err(response) => return response.into_response(),
    };
    let mut query = match decode(&uri) {
        Ok(query) => query,
        Err(response) => return response.into_response(),
    };
    if uri.path().ends_with("/components") {
        query.max_events = Some(0);
    }
    if uri.path().ends_with("/events") {
        query.max_components = Some(0);
    }
    let limits = adapter.reader.limits();
    if query
        .max_response_bytes
        .is_some_and(|v| v as usize > limits.max_snapshot_bytes)
        || query
            .max_components
            .is_some_and(|v| v as usize > limits.max_query_components)
        || query
            .max_events
            .is_some_and(|v| v as usize > limits.max_query_events)
    {
        return error(StatusCode::BAD_REQUEST, "invalid_page_limit");
    }
    let page = match adapter.reader.snapshot(&query) {
        Ok(page) => page,
        Err(e) => return reader_error(e),
    };
    use std::io::Write;
    if slot.write_all(page.as_bytes()).is_err() {
        return reader_error(DiagnosticError::SnapshotCapacity);
    }
    drop(page); // No core permit crosses network backpressure.
    response(slot)
}
async fn discovery(
    State(adapter): State<DiagnosticsHttp>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = adapter.authorize(&headers) {
        return response.into_response();
    }
    if uri.query().is_some() {
        return error(StatusCode::BAD_REQUEST, "invalid_query");
    }
    let mut slot = match adapter.reserve() {
        Ok(slot) => slot,
        Err(response) => return response.into_response(),
    };
    #[derive(serde::Serialize)]
    struct Capabilities<'a> {
        #[serde(flatten)]
        core: DiagnosticDiscovery,
        producers: &'a [&'static str],
        deferred: [&'static str; 4],
        inspection_policy: &'static str,
        #[serde(serialize_with = "serialize_limits")]
        limits: &'a DiagnosticLimits,
        response_slots: usize,
        response_bytes: usize,
    }
    let value = Capabilities {
        core: adapter.reader.discovery(),
        producers: &adapter.coverage,
        deferred: ["otel", "daemon", "bindings", "dynamic_plugin_protocol"],
        inspection_policy: "whole_context_private_bootstrap",
        limits: adapter.reader.limits(),
        response_slots: adapter.reader.limits().max_queries.max(1),
        response_bytes: adapter.capacity,
    };
    if serde_json::to_writer(&mut slot, &value).is_err() {
        return reader_error(DiagnosticError::SnapshotCapacity);
    }
    response(slot)
}

#[cfg(test)]
#[path = "../tests/coverage/shared/diagnostics_http_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/coverage/shared/diagnostic_cursor_tests.rs"]
mod cursor_tests;

async fn tls_required() -> Response {
    error(
        StatusCode::UPGRADE_REQUIRED,
        "authenticated_transport_required",
    )
}

#[cfg(test)]
#[path = "../tests/coverage/shared/diagnostic_http_lifecycle_tests.rs"]
mod lifecycle_tests;

// Explicitly enumerate the profile: adding a core capacity requires revisiting discovery.
fn serialize_limits<S: serde::Serializer>(
    limits: &DiagnosticLimits,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let DiagnosticLimits {
        max_components,
        max_counters,
        max_events,
        max_window_buckets,
        max_extensions,
        max_metadata_bytes,
        max_identity_bytes,
        max_detail_bytes,
        max_correlation_bytes,
        detail_pool_bytes,
        detail_block_bytes,
        max_queries,
        max_query_components,
        max_query_events,
        max_query_counters,
        max_query_window_buckets,
        max_snapshot_bytes,
        query_text_bytes,
        max_projections,
        max_projection_workspaces,
        max_owned_bytes,
    } = limits;
    let mut map = serializer.serialize_map(Some(21))?;
    map.serialize_entry(
        "max_components",
        &DiagnosticNumber::from(*max_components as u64),
    )?;
    map.serialize_entry(
        "max_counters",
        &DiagnosticNumber::from(*max_counters as u64),
    )?;
    map.serialize_entry("max_events", &DiagnosticNumber::from(*max_events as u64))?;
    map.serialize_entry(
        "max_window_buckets",
        &DiagnosticNumber::from(*max_window_buckets as u64),
    )?;
    map.serialize_entry(
        "max_extensions",
        &DiagnosticNumber::from(*max_extensions as u64),
    )?;
    map.serialize_entry(
        "max_metadata_bytes",
        &DiagnosticNumber::from(*max_metadata_bytes as u64),
    )?;
    map.serialize_entry(
        "max_identity_bytes",
        &DiagnosticNumber::from(*max_identity_bytes as u64),
    )?;
    map.serialize_entry(
        "max_detail_bytes",
        &DiagnosticNumber::from(*max_detail_bytes as u64),
    )?;
    map.serialize_entry(
        "max_correlation_bytes",
        &DiagnosticNumber::from(*max_correlation_bytes as u64),
    )?;
    map.serialize_entry(
        "detail_pool_bytes",
        &DiagnosticNumber::from(*detail_pool_bytes as u64),
    )?;
    map.serialize_entry(
        "detail_block_bytes",
        &DiagnosticNumber::from(*detail_block_bytes as u64),
    )?;
    map.serialize_entry("max_queries", &DiagnosticNumber::from(*max_queries as u64))?;
    map.serialize_entry(
        "max_query_components",
        &DiagnosticNumber::from(*max_query_components as u64),
    )?;
    map.serialize_entry(
        "max_query_events",
        &DiagnosticNumber::from(*max_query_events as u64),
    )?;
    map.serialize_entry(
        "max_query_counters",
        &DiagnosticNumber::from(*max_query_counters as u64),
    )?;
    map.serialize_entry(
        "max_query_window_buckets",
        &DiagnosticNumber::from(*max_query_window_buckets as u64),
    )?;
    map.serialize_entry(
        "max_snapshot_bytes",
        &DiagnosticNumber::from(*max_snapshot_bytes as u64),
    )?;
    map.serialize_entry(
        "query_text_bytes",
        &DiagnosticNumber::from(*query_text_bytes as u64),
    )?;
    map.serialize_entry(
        "max_projections",
        &DiagnosticNumber::from(*max_projections as u64),
    )?;
    map.serialize_entry(
        "max_projection_workspaces",
        &DiagnosticNumber::from(*max_projection_workspaces as u64),
    )?;
    map.serialize_entry(
        "max_owned_bytes",
        &DiagnosticNumber::from(*max_owned_bytes as u64),
    )?;
    map.end()
}
