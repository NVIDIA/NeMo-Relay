// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::identity::PublicIdentity;

fn fingerprint() -> Fingerprint {
    PublicIdentity::from_bytes(&[9; 32])
        .expect("identity")
        .fingerprint()
}

#[test]
fn worker_request_accounts_for_exact_lifetime() {
    let target = Arc::new(
        WorkerTarget::new(
            "worker-1",
            "http://127.0.0.1:41000",
            SensitiveString::new("worker-secret").expect("secret"),
        )
        .expect("target"),
    );
    assert_eq!(target.in_flight(), 0);
    let request = target.acquire(fingerprint());
    assert_eq!(target.in_flight(), 1);
    assert_eq!(request.fingerprint(), fingerprint());
    assert_eq!(request.target().endpoint(), "http://127.0.0.1:41000");
    assert_eq!(request.session_token(), "worker-secret");
    assert!(!format!("{request:?}").contains("worker-secret"));
    drop(request);
    assert_eq!(target.in_flight(), 0);
}

#[test]
fn identifiers_and_targets_reject_empty_values() {
    assert_eq!(McpSessionId::new(""), Err(LifecycleError::EmptyIdentifier));
    assert_eq!(
        McpSessionId::new("mcp-1").expect("session").as_str(),
        "mcp-1"
    );
    assert_eq!(
        WorkerTarget::new(
            "",
            "http://127.0.0.1:1",
            SensitiveString::new("secret").expect("secret")
        )
        .unwrap_err(),
        LifecycleError::EmptyIdentifier
    );
    assert_eq!(
        WorkerTarget::new("worker", "", SensitiveString::new("secret").unwrap()).unwrap_err(),
        LifecycleError::EmptyIdentifier
    );
    assert!(matches!(
        SensitiveString::new(""),
        Err(crate::daemon::common::protocol::ProtocolError::MissingSensitiveValue)
    ));
}

#[test]
fn worker_target_prepares_generation_routing_and_preserves_raw_paths() {
    for endpoint in ["http://127.0.0.1:41000", "https://[::1]:41000/"] {
        let target = WorkerTarget::new(
            "generation",
            endpoint,
            SensitiveString::new("secret").unwrap(),
        )
        .unwrap();
        let request: http::Uri = "/v1/responses/a%2Fb?key=x%2Fy&key=z".parse().unwrap();
        let destination = target.request_uri(&request).unwrap();
        assert_eq!(destination.path_and_query(), request.path_and_query());
        assert_eq!(destination.authority(), target.origin.authority());
        assert_eq!(target.authentication(), "secret");
        assert!(target.authentication().is_sensitive());
        assert!(Arc::ptr_eq(
            &target.shared_worker_id(),
            &target.shared_worker_id()
        ));
        assert!(!format!("{target:?}").contains("secret"));
    }
    assert_eq!(
        WorkerTarget::new(
            "worker",
            "relative",
            SensitiveString::new("secret").unwrap()
        )
        .unwrap_err(),
        LifecycleError::InvalidDestination
    );
    assert_eq!(
        WorkerTarget::new(
            "worker",
            "http://127.0.0.1:1",
            SensitiveString::new("bad\ncredential").unwrap()
        )
        .unwrap_err(),
        LifecycleError::InvalidCredential
    );
}
