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
    assert!(
        WorkerTarget::new(
            "",
            "http://127.0.0.1:1",
            SensitiveString::new("secret").expect("secret")
        )
        .is_err()
    );
}
