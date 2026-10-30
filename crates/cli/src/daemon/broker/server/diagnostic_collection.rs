// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed Relay worker collection; this does not use the dynamic-plugin worker protocol.
use super::super::collection::{self, Buffer, CacheKey, PollOutcome, Ticket};
use super::*;
use crate::diagnostics_http::{AUTH_HEADER, error};
use http_body_util::BodyExt;
use nemo_relay_types::diagnostics::{
    DiagnosticHttpError, DiagnosticHttpReason as HttpReason, MAX_DIAGNOSTIC_HTTP_ERROR_BYTES,
};
use std::io::Write;
use tokio::time::Instant;

const COLLECTION_POLL_INTERVAL: Duration = Duration::from_millis(250);
const COLLECTION_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Closes only diagnostic work on every serve exit, including setup failure/cancellation.
pub(super) struct Lifetime {
    collector: Option<Arc<collection::Collector>>,
    adapter: Option<crate::diagnostics_http::DiagnosticsHttp>,
    task: Option<tokio::task::AbortHandle>,
}
impl Lifetime {
    pub(super) fn new(state: &DaemonState, task: Option<&tokio::task::JoinHandle<()>>) -> Self {
        Self {
            collector: state.collector.clone(),
            adapter: state.diagnostics_http.clone(),
            task: task.map(|t| t.abort_handle()),
        }
    }
}
impl Drop for Lifetime {
    fn drop(&mut self) {
        if let Some(c) = &self.collector {
            c.close();
        }
        if let Some(a) = &self.adapter {
            a.close();
        }
        if let Some(t) = &self.task {
            t.abort();
        }
    }
}

pub(super) fn grant(
    state: &DaemonState,
    offer: Option<&diag::DiagnosticOffer>,
    generation: &str,
) -> Option<diag::DiagnosticReadGrant> {
    state.collector.as_ref()?;
    let offer = offer.filter(|o| {
        o.enabled
            && o.query_available
            && o.whole_context_read
            && o.schema_version == nemo_relay_runtime::diagnostics::DIAGNOSTIC_SCHEMA_VERSION
            && o.max_page_bytes > 0
    })?;
    diag::DiagnosticReadGrant::issue(
        offer.source,
        generation,
        offer.max_page_bytes.min(collection::PAGE_BYTES),
    )
}
fn key(session: &WorkerControlSession) -> Option<CacheKey> {
    if !session.diagnostic_connected {
        return None;
    }
    let grant = session.diagnostic_grant.as_ref()?;
    CacheKey::new(
        &session.worker_id,
        &session.generation_grant.generation_id,
        grant.source,
        grant.fence(),
    )
}
struct PollGuard {
    collector: Arc<collection::Collector>,
    ticket: Option<Ticket>,
}
impl Drop for PollGuard {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            lock(&self.collector.cache).abandon(ticket);
        }
    }
}

pub(super) async fn run(state: std::sync::Weak<DaemonState>) {
    // Freshness is an age classification, not a per-worker polling SLA: the 15-second
    // cutoff in collection.rs is well above this cadence, but POLLS bounds concurrency.
    let mut interval = tokio::time::interval(COLLECTION_POLL_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut jobs = tokio::task::JoinSet::new();
    let mut cursor = String::new();
    loop {
        tokio::select! {
            _ = interval.tick() => {},
            _ = jobs.join_next(), if !jobs.is_empty() => continue,
        }
        let Some(state) = state.upgrade() else {
            return;
        };
        let Some(collector) = &state.collector else {
            return;
        };
        if collector.is_closed() {
            return;
        }
        for _ in jobs.len()..collection::POLLS {
            // Lock order is sessions -> cache, also used at publication and reads.
            // Copy only one bounded target at a time; no retained fleet-sized queue.
            let selected = {
                let sessions = lock(&state.worker_sessions);
                let next = sessions
                    .values()
                    .filter(|s| {
                        s.worker_id > cursor
                            && !s.diagnostic_poll_stopped
                            && s.diagnostic_connected
                            && s.diagnostic_grant.is_some()
                    })
                    .min_by(|a, b| a.worker_id.cmp(&b.worker_id));
                match next {
                    None => {
                        cursor.clear();
                        None
                    }
                    Some(session) => {
                        cursor.clone_from(&session.worker_id);
                        let cache_key = key(session).expect("selected negotiated source");
                        collector.ingress.reserve().and_then(|buffer| {
                            let ticket = lock(&collector.cache).begin(cache_key, Instant::now())?;
                            Some((
                                Arc::clone(&session.pending_target),
                                session.diagnostic_grant.clone().unwrap(),
                                PollGuard {
                                    collector: Arc::clone(collector),
                                    ticket: Some(ticket),
                                },
                                buffer,
                            ))
                        })
                    }
                }
            };
            if let Some((target, grant, guard, buffer)) = selected {
                jobs.spawn(poll(Arc::clone(&state), target, grant, guard, buffer));
            }
        }
    }
    // JoinSet aborts outstanding jobs; their guards and buffers return on cancellation.
}

async fn poll(
    state: Arc<DaemonState>,
    target: Arc<WorkerTarget>,
    grant: diag::DiagnosticReadGrant,
    mut guard: PollGuard,
    buffer: Buffer,
) {
    let result = tokio::time::timeout(COLLECTION_REQUEST_TIMEOUT, fetch(&target, &grant, buffer))
        .await
        .unwrap_or(Err(PollOutcome::Timeout));
    let mut sessions = lock(&state.worker_sessions);
    let ticket = guard.ticket.take().expect("one completion");
    let current = sessions
        .get(target.worker_id())
        .and_then(key)
        .is_some_and(|current| current.matches(ticket.key()));
    if current
        && result.as_ref().is_err_and(|outcome| outcome.terminal())
        && let Some(session) = sessions.get_mut(target.worker_id())
    {
        session.diagnostic_poll_stopped = true;
    }
    let mut cache = lock(&guard.collector.cache);
    if current && !guard.collector.is_closed() {
        if let Err(outcome) = &result {
            let class = match outcome {
                PollOutcome::Timeout => Some(FailureClass::Timeout),
                PollOutcome::Oversized
                | PollOutcome::InvalidPage
                | PollOutcome::EnvelopeTooSmall => Some(FailureClass::InvalidInput),
                PollOutcome::Disabled | PollOutcome::ShuttingDown | PollOutcome::Unreachable => {
                    Some(FailureClass::Unavailable)
                }
                PollOutcome::Unauthorized => Some(FailureClass::Authentication),
                PollOutcome::Unsupported => Some(FailureClass::Unsupported),
                PollOutcome::Busy => None,
                PollOutcome::Pending | PollOutcome::Succeeded => Some(FailureClass::Unknown),
            };
            if let Some(class) = class {
                diag::record(
                    DaemonFailureCode::Collection,
                    class,
                    "Worker diagnostic collection attempt failed",
                );
            }
        }
        cache.complete(ticket, result, Instant::now(), now_unix_ms());
    } else {
        cache.abandon(ticket);
    }
}

async fn fetch(
    target: &WorkerTarget,
    grant: &diag::DiagnosticReadGrant,
    mut buffer: Buffer,
) -> Result<Buffer, PollOutcome> {
    // Same authenticated endpoint and pooled, pinned transport as forwarding, with an
    // independent credential. No readiness request, redirect, decompressor, or DTO expansion.
    let query = format!("{{\"max_response_bytes\":{}}}", grant.max_page_bytes);
    let query = percent_encoding::utf8_percent_encode(&query, percent_encoding::NON_ALPHANUMERIC);
    let uri = format!(
        "{}{}?query={query}",
        target.endpoint().trim_end_matches('/'),
        crate::diagnostics_http::PREFIX
    );
    let request = Request::builder()
        .uri(uri)
        .method(Method::GET)
        .header(AUTH_HEADER, grant.token(now_unix_ms()))
        .header(http::header::ACCEPT_ENCODING, "identity")
        .body(box_body(Body::empty()))
        .map_err(|_| PollOutcome::InvalidPage)?;
    let response = target
        .client()
        .request(request)
        .await
        .map_err(|_| PollOutcome::Unreachable)?;
    // Transport parser allocations belong to the shared HTTP allowance. Bound accepted
    // headers before using them and bound every body frame before copying to owned storage.
    let header_bytes = response.headers().iter().try_fold(0usize, |n, (k, v)| {
        n.checked_add(k.as_str().len() + v.len())
    });
    if header_bytes.is_none_or(|n| n > 16 * 1024) {
        return Err(PollOutcome::Oversized);
    }
    let status = response.status();
    let max_bytes = if status == StatusCode::OK {
        grant.max_page_bytes
    } else {
        MAX_DIAGNOSTIC_HTTP_ERROR_BYTES
    };
    if response
        .headers()
        .get_all(http::header::CONTENT_ENCODING)
        .iter()
        .any(|v| v != "identity")
    {
        return Err(PollOutcome::InvalidPage);
    }
    if status == StatusCode::OK
        && response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .is_none_or(|v| v != "application/json")
    {
        return Err(PollOutcome::InvalidPage);
    }
    if response
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .is_some_and(|v| {
            v.to_str()
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .is_none_or(|n| n > max_bytes)
        })
    {
        return Err(PollOutcome::Oversized);
    }
    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| PollOutcome::Unreachable)?;
        if let Ok(bytes) = frame.into_data() {
            if bytes.len() > max_bytes.saturating_sub(buffer.bytes.len()) {
                return Err(PollOutcome::Oversized);
            }
            buffer
                .write_all(&bytes)
                .map_err(|_| PollOutcome::Oversized)?;
        } else {
            return Err(PollOutcome::InvalidPage);
        }
    }
    if status != StatusCode::OK {
        return Err(poll_error(status.as_u16(), &buffer.bytes));
    }
    collection::validate_page(&buffer.bytes, grant.source).map_err(|_| PollOutcome::InvalidPage)?;
    Ok(buffer)
}

// A known reason determines semantics. Only old endpoints with no error body use
// status-only compatibility fallbacks; an unknown 503 must remain retryable.
fn poll_error(status: u16, body: &[u8]) -> PollOutcome {
    if let Some(error) = DiagnosticHttpError::from_response(status, body) {
        return match error.error.reason {
            HttpReason::Unsupported | HttpReason::UnsupportedSchema => PollOutcome::Unsupported,
            HttpReason::Disabled
            | HttpReason::QueryDisabled
            | HttpReason::HttpReadDisabled
            | HttpReason::CollectionDisabled => PollOutcome::Disabled,
            HttpReason::ShuttingDown => PollOutcome::ShuttingDown,
            HttpReason::Unauthorized | HttpReason::Forbidden => PollOutcome::Unauthorized,
            HttpReason::Busy => PollOutcome::Busy,
            HttpReason::SnapshotCapacity => PollOutcome::EnvelopeTooSmall,
            _ => PollOutcome::Unreachable,
        };
    }
    match (status, body.is_empty()) {
        (404 | 406 | 501, true) => PollOutcome::Unsupported,
        (401 | 403, _) => PollOutcome::Unauthorized,
        (429, true) => PollOutcome::Busy,
        (422, true) => PollOutcome::EnvelopeTooSmall,
        _ => PollOutcome::Unreachable,
    }
}

pub(super) async fn read(
    State(state): State<Arc<DaemonState>>,
    AxumPath(worker_id): AxumPath<String>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
    headers: HeaderMap,
) -> Response<Body> {
    // Authenticate the full selected context before revealing assignment/cache metadata.
    let Some(adapter) = &state.diagnostics_http else {
        return error(HttpReason::HttpReadDisabled);
    };
    if let Err(e) = adapter.authorize(&headers) {
        return e.into_response();
    }
    if uri.query().is_some() {
        return error(HttpReason::InvalidQuery);
    }
    let Some(collector) = &state.collector else {
        return error(HttpReason::CollectionDisabled);
    };
    if collector.is_closed() {
        return error(HttpReason::ShuttingDown);
    }
    let Some(mut output) = collector.responses.reserve() else {
        return error(HttpReason::Busy);
    };
    {
        let sessions = lock(&state.worker_sessions);
        let Some(session) = sessions.get(&worker_id) else {
            return error(HttpReason::Missing);
        };
        let Some(offer) = &session.diagnostic_offer else {
            return error(HttpReason::Unsupported);
        };
        if !offer.enabled || !offer.whole_context_read {
            return error(HttpReason::Disabled);
        }
        if !offer.query_available {
            return error(HttpReason::QueryDisabled);
        }
        if offer.schema_version != nemo_relay_runtime::diagnostics::DIAGNOSTIC_SCHEMA_VERSION {
            return error(HttpReason::UnsupportedSchema);
        }
        let Some(key) = key(session) else {
            return error(HttpReason::ReadAuthorityUnavailable);
        };
        if lock(&collector.cache)
            .write_page(&key, Instant::now(), &mut output)
            .is_err()
        {
            return error(HttpReason::SnapshotCapacity);
        }
    }
    // No state lock, source lease, or grant survives into the response body.
    let mut response = Response::new(Body::from(Bytes::from_owner(output)));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    response
}

#[cfg(test)]
#[path = "../../../../tests/coverage/daemon/diagnostic_collection_tests.rs"]
mod tests;
