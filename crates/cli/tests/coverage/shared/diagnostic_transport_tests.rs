// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_support::EnvScope;
use nemo_relay::diagnostics::{DiagnosticLimits, Diagnostics, DiagnosticsSettings};

#[tokio::test]
async fn inspection_reuses_tls_without_touching_gateway_activity_or_query_state() {
    let directory = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[("XDG_CONFIG_HOME", Some(directory.path().as_os_str()))]);
    let key = BootstrapChallengeKey::load().unwrap();
    let tls = crate::gateway::tls::RelayTlsIdentity::load_or_create()
        .unwrap()
        .server_config()
        .unwrap();
    let context = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_queries: 0,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut state = AppState::new(GatewayConfig::default());
    state.bootstrap_challenge_key = Some(key.clone());
    state.bootstrap_tls = Some(tls);
    state.local_address = Some(address);
    state.diagnostics_http =
        Some(crate::diagnostics_http::DiagnosticsHttp::new(context.reader(), key, vec![]).unwrap());
    let activity = state.last_activity.clone();
    let before = *activity.lock().unwrap();
    let app = router_with_state(state);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let result = tokio::task::spawn_blocking(move || {
        crate::gateway::client::get_diagnostics(
            &format!("http://{address}"),
            &format!("{}/capabilities", crate::diagnostics_http::PREFIX),
            8192,
        )
    })
    .await
    .unwrap()
    .unwrap();
    server.abort();
    assert_eq!(result.status, 200);
    let discovery: serde_json::Value = serde_json::from_slice(&result.body).unwrap();
    assert_eq!(discovery["collection_enabled"], true);
    assert_eq!(discovery["queries_enabled"], false);
    assert_eq!(*activity.lock().unwrap(), before);
    assert_eq!(context.memory_usage().query_workspaces, 0);
}

#[tokio::test]
async fn inspection_never_creates_bootstrap_state_or_starts_a_gateway() {
    let directory = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(directory.path().as_os_str())),
        ("NEMO_RELAY_BOOTSTRAP_STATE_DIR", None),
    ]);
    // An existing foreign listener, with no private credentials, gets no request.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let path = format!("{}/capabilities", crate::diagnostics_http::PREFIX);
    assert!(
        crate::gateway::client::get_diagnostics(&format!("http://{address}"), &path, 8192).is_err()
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    drop(listener);
    // No acquisition path is called when there is no listener at all.
    assert!(
        crate::gateway::client::get_diagnostics(&format!("http://{address}"), &path, 8192).is_err()
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}
