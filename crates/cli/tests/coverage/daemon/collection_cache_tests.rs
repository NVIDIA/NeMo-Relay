// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay::diagnostics::*;
fn source() -> Diagnostics {
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap()
}
fn key(source: uuid::Uuid, name: &str) -> CacheKey {
    CacheKey::new(
        name,
        &uuid::Uuid::now_v7().to_string(),
        source,
        TokenDigest::from_token(b"control"),
    )
    .unwrap()
}
fn page(context: &Diagnostics, pool: &BufferPool) -> Buffer {
    let mut buffer = pool.reserve().unwrap();
    buffer
        .write_all(context.snapshot(&Default::default()).unwrap().as_bytes())
        .unwrap();
    validate_page(&buffer.bytes, context.source().instance_id).unwrap();
    buffer
}
#[test]
fn collector_budget_includes_staged_session_metadata_without_panicking() {
    assert!(owned_bytes() >= MAX_STAGED_WORKER_SESSIONS * PER_SESSION_METADATA_ALLOWANCE_BYTES);
}
#[test]
fn collection_buffers_return_on_body_drop_and_enforce_byte_caps() {
    let pool = BufferPool::new(1, 16).unwrap();
    let mut buffer = pool.reserve().unwrap();
    buffer.write_all(&[1; 16]).unwrap();
    assert!(buffer.write_all(&[2]).is_err());
    assert!(pool.reserve().is_none());
    let body = bytes::Bytes::from_owner(buffer);
    let second = body.clone();
    drop(body);
    assert!(pool.reserve().is_none());
    drop(second);
    assert!(pool.reserve().is_some());
}
#[test]
fn collection_parser_bounds_structure_and_preserves_future_codes() {
    let context = source();
    let pool = BufferPool::new(1, PAGE_BYTES).unwrap();
    let buffer = page(&context, &pool);
    assert!(validate_page(&buffer.bytes, uuid::Uuid::now_v7()).is_err());
    assert!(validate_page(&vec![b' '; PAGE_BYTES + 1], context.source().instance_id).is_err());
    assert!(
        validate_page(
            format!("{}0{}", "[".repeat(40), "]".repeat(40)).as_bytes(),
            context.source().instance_id
        )
        .is_err()
    );
    assert!(
        validate_page(
            format!("[{}]", "0,".repeat(MAX_JSON_NODES) + "0").as_bytes(),
            context.source().instance_id
        )
        .is_err()
    );
    let mut value: serde_json::Value = serde_json::from_slice(&buffer.bytes).unwrap();
    value["future_field"] = serde_json::json!({"unknown_code":"future.failure"});
    assert!(
        validate_page(
            &serde_json::to_vec(&value).unwrap(),
            context.source().instance_id
        )
        .is_ok()
    );
}
#[tokio::test(start_paused = true)]
async fn collection_refreshes_same_sequence_without_recounting_and_keeps_last_page() {
    let context = source();
    let c = context
        .register_component(ComponentRegistration::new(
            "test",
            &[BuiltinDiagnosticCode::Failure(FailureCode::Daemon(
                DaemonFailureCode::Forward,
            ))],
        ))
        .unwrap();
    for n in [2, 3] {
        c.record_without_projection(Observation::new(
            Classification::failure(
                FailureCode::Daemon(DaemonFailureCode::Forward),
                FailureClass::Unknown,
            )
            .unwrap(),
            "private",
            std::num::NonZeroU64::new(n).unwrap(),
        ))
        .unwrap();
    }
    let pool = BufferPool::new(POLLS, PAGE_BYTES).unwrap();
    let responses = BufferPool::new(POLLS, RESPONSE_BYTES).unwrap();
    let mut cache = Cache::new().unwrap();
    let key = key(context.source().instance_id, "worker");
    for wall in [20_000, 10_000] {
        let ticket = cache.begin(key.clone(), Instant::now()).unwrap();
        assert!(cache.begin(key.clone(), Instant::now()).is_none());
        cache.complete(ticket, Ok(page(&context, &pool)), Instant::now(), wall);
        tokio::time::advance(std::time::Duration::from_secs(7)).await;
    }
    let ticket = cache.begin(key.clone(), Instant::now()).unwrap();
    cache.complete(ticket, Err(PollOutcome::Timeout), Instant::now(), 1);
    let mut response = responses.reserve().unwrap();
    cache
        .write_page(&key, Instant::now(), &mut response)
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&response.bytes).unwrap();
    assert_eq!(value["collection"]["received_unix_millis"], "10000");
    assert_eq!(value["collection"]["latest_poll"], "timeout");
    let snapshot: DiagnosticSnapshot = serde_json::from_value(value["snapshot"].clone()).unwrap();
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .failures_total
            .value
            .get(),
        5
    );
    assert_eq!(
        context
            .snapshot(&Default::default())
            .unwrap()
            .to_snapshot()
            .unwrap()
            .sequence,
        snapshot.sequence
    );
}
#[test]
fn collection_churn_reuses_fixed_slots_and_fences_replaced_sources() {
    let context = source();
    let pool = BufferPool::new(POLLS, PAGE_BYTES).unwrap();
    let responses = BufferPool::new(POLLS, RESPONSE_BYTES).unwrap();
    let mut cache = Cache::new().unwrap();
    let first = key(context.source().instance_id, "original");
    for n in 0..SOURCES * 4 {
        let key = if n == 0 {
            first.clone()
        } else {
            key(context.source().instance_id, &format!("worker-{n}"))
        };
        let ticket = cache.begin(key, Instant::now()).unwrap();
        cache.complete(ticket, Ok(page(&context, &pool)), Instant::now(), 1);
    }
    assert_eq!(cache.slots.len(), SOURCES);
    assert_eq!(cache.evictions as usize, SOURCES * 3);
    assert!(cache.slots.iter().all(|s| s.bytes.capacity() == PAGE_BYTES));
    let mut response = responses.reserve().unwrap();
    cache
        .write_page(&first, Instant::now(), &mut response)
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&response.bytes).unwrap();
    assert!(value["snapshot"].is_null());
    let ticket = cache.begin(first.clone(), Instant::now()).unwrap();
    cache.abandon(ticket);
    assert!(!cache.slots.iter().any(|s| s.polling));
    assert!(!first.matches(&key(context.source().instance_id, "original")));
}
