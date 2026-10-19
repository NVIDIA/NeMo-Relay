// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// Suspend a preparation after it takes exclusive session ownership, as when it waits to restore.
async fn detached_gateway_preparation(manager: &SessionManager, id: &str) -> InFlightSession {
    let gate = session_gate(&manager.session_gates, id)
        .await
        .lock_owned()
        .await;
    let activity = manager.session_activity.begin();
    let mut sessions = manager.inner.lock().await;
    let session = sessions.remove(id).unwrap();
    manager.gateway_preparing.lock().unwrap().insert(id.into());
    InFlightSession::for_gateway(
        manager.inner.clone(),
        id.into(),
        session,
        gate,
        activity,
        manager.gateway_preparing.clone(),
    )
}

#[tokio::test]
async fn gateway_preparation_keeps_fallback_identity_without_blocking_other_sessions() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();
    let original = manager
        .prepare_gateway_call(&headers, llm_start())
        .await
        .unwrap();
    let in_flight = detached_gateway_preparation(&manager, &original.session_id).await;
    let unlabelled = manager.prepare_gateway_call(
        &headers,
        LlmGatewayStart {
            session_id: None,
            ..llm_start()
        },
    );
    tokio::pin!(unlabelled);
    assert!(futures_util::poll!(&mut unlabelled).is_pending());

    let other = tokio::time::timeout(
        Duration::from_secs(2),
        manager.prepare_gateway_call(
            &headers,
            LlmGatewayStart {
                session_id: Some("independent".into()),
                ..llm_start()
            },
        ),
    )
    .await
    .expect("unrelated session must not wait for preparation")
    .unwrap();
    assert_eq!(other.session_id, "independent");
    drop(in_flight.restore().await);
    let resumed = unlabelled.await.unwrap();
    assert_eq!(resumed.session_id, original.session_id);
    assert!(manager.gateway_preparing.lock().unwrap().is_empty());
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn cancelled_gateway_preparation_restores_session_and_releases_call_reservation() {
    let manager = SessionManager::new(session_test_config());
    let original = manager
        .prepare_gateway_call(&HeaderMap::new(), llm_start())
        .await
        .unwrap();
    let mut in_flight = detached_gateway_preparation(&manager, &original.session_id).await;
    let prep = in_flight
        .session_mut()
        .prepare_gateway_call(llm_start())
        .await
        .unwrap();
    in_flight.gateway_call_prepared(prep.owner_subagent_id);
    let directory = manager.inner.lock().await;
    let mut restore = Box::pin(in_flight.restore());
    assert!(futures_util::poll!(&mut restore).is_pending());
    drop(restore);
    assert!(manager.session_activity.has_active());
    assert!(!manager.gateway_preparing.lock().unwrap().is_empty());
    drop(directory);
    tokio::time::timeout(
        Duration::from_secs(2),
        manager.session_activity.wait_for_idle(),
    )
    .await
    .expect("cancelled preparation must restore its session");
    let directory = manager.inner.lock().await;
    assert_eq!(directory[&original.session_id].active_gateway_calls, 1);
    drop(directory);
    assert!(manager.gateway_preparing.lock().unwrap().is_empty());
    assert!(
        session_gate(&manager.session_gates, &original.session_id)
            .await
            .try_lock()
            .is_ok()
    );
    manager.close_all("test_shutdown").await.unwrap();
}
