// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::Ipv4Addr;
use std::sync::Barrier;

use super::*;
use crate::daemon::common::identity::PublicIdentity;
use crate::daemon::common::protocol::SensitiveString;

fn fingerprint(byte: u8) -> Fingerprint {
    PublicIdentity::from_bytes(&[byte; 32])
        .expect("public identity")
        .fingerprint()
}

fn session(name: &str) -> McpSessionId {
    McpSessionId::new(name).expect("session")
}

fn launch(name: &str) -> WorkerLaunch {
    WorkerLaunch {
        activation_id: name.to_owned(),
        activation_token: SensitiveString::new(format!("{name}-secret")).expect("secret"),
        deadline_unix_ms: 15_000,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    }
}

fn registration(
    fingerprint: Fingerprint,
    token_digest: TokenDigest,
    session_id: &str,
) -> McpRegistration {
    McpRegistration {
        fingerprint,
        token_digest,
        session_id: session(session_id),
        lease_expires_at_unix_ms: 30_000,
    }
}

fn worker(worker_id: &str) -> Arc<WorkerTarget> {
    Arc::new(
        WorkerTarget::new(
            worker_id,
            "http://127.0.0.1:41000",
            SensitiveString::new("internal-session-token").expect("token"),
        )
        .expect("worker target"),
    )
}

#[test]
fn first_mcp_wins_singleflight_and_retries_idempotently() {
    let registry = Registry::new(false).with_retry_after_ms(25);
    let fingerprint = fingerprint(1);
    let token = TokenDigest::from_token(b"token-1");

    let first = registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("first"))
        .expect("first registration");
    assert!(matches!(
        first,
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "first"
    ));

    let retry = registry
        .register_connected_mcp(
            registration(fingerprint, token, "mcp-a"),
            launch("must-not-replace"),
        )
        .expect("idempotent retry");
    assert!(matches!(
        retry,
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "first"
    ));

    let concurrent = registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("second"))
        .expect("concurrent registration");
    assert_eq!(
        concurrent,
        BrokerDirective::WaitForWorker { retry_after_ms: 25 }
    );
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot"),
        RouteSnapshot {
            state: RouteStateKind::Activating,
            reference_count: 2,
            launch_owner: Some(session("mcp-a")),
            endpoint: None,
            in_flight: 0,
        }
    );
}

#[test]
fn concurrent_registrations_issue_exactly_one_launch() {
    const MCP_COUNT: usize = 32;
    let registry = Arc::new(Registry::new(false));
    let barrier = Arc::new(Barrier::new(MCP_COUNT));
    let fingerprint = fingerprint(13);
    let token = TokenDigest::from_token(b"token-13");
    let handles: Vec<_> = (0..MCP_COUNT)
        .map(|index| {
            let registry = Arc::clone(&registry);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                registry
                    .register_connected_mcp(
                        registration(fingerprint, token, &format!("mcp-{index:02}")),
                        launch(&format!("launch-{index:02}")),
                    )
                    .expect("registration")
            })
        })
        .collect();
    let directives: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("thread"))
        .collect();

    assert_eq!(
        directives
            .iter()
            .filter(|directive| matches!(directive, BrokerDirective::LaunchWorker { .. }))
            .count(),
        1
    );
    assert_eq!(
        directives
            .iter()
            .filter(|directive| matches!(directive, BrokerDirective::WaitForWorker { .. }))
            .count(),
        MCP_COUNT - 1
    );
    assert_eq!(
        registry
            .snapshot(fingerprint)
            .expect("snapshot")
            .reference_count,
        MCP_COUNT
    );
}

#[test]
fn ready_worker_is_reused_and_request_guard_counts_in_flight() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(2);
    let token = TokenDigest::from_token(b"token-2");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
        .expect("registration");
    let target = worker("worker-1");
    registry
        .mark_worker_ready(fingerprint, "launch", Arc::clone(&target))
        .expect("worker ready");

    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"))
            .expect("reuse"),
        BrokerDirective::ReuseWorker {
            endpoint: "http://127.0.0.1:41000".to_owned()
        }
    );
    let request = match registry.resolve_target(&token).expect("resolved") {
        ResolvedTarget::Worker(request) => request,
        ResolvedTarget::PassThrough => panic!("expected worker"),
    };
    assert_eq!(target.in_flight(), 1);
    assert_eq!(request.session_token(), "internal-session-token");
    drop(request);
    assert_eq!(target.in_flight(), 0);
}

#[test]
fn worker_status_snapshots_report_assigned_workers_in_worker_id_order() {
    let registry = Registry::new(false);
    let ready_fingerprint = fingerprint(40);
    let ready_token = TokenDigest::from_token(b"ready-status-token");
    registry
        .register_connected_mcp(
            registration(ready_fingerprint, ready_token, "ready-mcp"),
            launch("ready-activation"),
        )
        .unwrap();
    registry
        .mark_worker_ready(ready_fingerprint, "ready-activation", worker("worker-z"))
        .unwrap();

    let recovering_fingerprint = fingerprint(41);
    registry
        .restore_binding(
            recovering_fingerprint,
            TokenDigest::from_token(b"recovering-status-token"),
        )
        .unwrap();
    registry
        .begin_recovery(recovering_fingerprint, Some(worker("worker-m")), 15_000)
        .unwrap();

    let draining_fingerprint = fingerprint(42);
    let draining_token = TokenDigest::from_token(b"draining-status-token");
    registry
        .register_connected_mcp(
            registration(draining_fingerprint, draining_token, "draining-mcp"),
            launch("draining-activation"),
        )
        .unwrap();
    registry
        .mark_worker_ready(
            draining_fingerprint,
            "draining-activation",
            worker("worker-a"),
        )
        .unwrap();
    registry
        .release_mcp(draining_fingerprint, &session("draining-mcp"), 15_000)
        .unwrap();

    let snapshots = registry.worker_status_snapshots();
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| (snapshot.worker_id.as_str(), snapshot.state))
            .collect::<Vec<_>>(),
        vec![
            ("worker-a", RouteStateKind::Draining),
            ("worker-m", RouteStateKind::Recovering),
            ("worker-z", RouteStateKind::Ready),
        ]
    );
}

#[test]
fn worker_status_snapshots_omit_routes_without_assigned_workers() {
    let registry = Registry::new(false);
    let fallback_fingerprint = fingerprint(43);
    registry
        .register_connected_mcp(
            registration(
                fallback_fingerprint,
                TokenDigest::from_token(b"fallback-status-token"),
                "fallback-mcp",
            ),
            launch("fallback-activation"),
        )
        .unwrap();
    registry
        .mark_activation_failed(fallback_fingerprint, "fallback-activation")
        .unwrap();

    let recovering_fingerprint = fingerprint(44);
    registry
        .restore_binding(
            recovering_fingerprint,
            TokenDigest::from_token(b"unassigned-recovery-token"),
        )
        .unwrap();
    registry
        .begin_recovery(recovering_fingerprint, None, 15_000)
        .unwrap();

    assert!(registry.worker_status_snapshots().is_empty());
}

#[test]
fn token_bindings_cannot_be_reassigned_to_another_fingerprint() {
    let registry = Registry::new(false);
    let first_fingerprint = fingerprint(3);
    let other_fingerprint = fingerprint(4);
    let token = TokenDigest::from_token(b"stable-token");
    let joined = TokenDigest::from_token(b"different-token");
    registry
        .restore_binding(first_fingerprint, token)
        .expect("binding");
    registry
        .restore_binding(first_fingerprint, joined)
        .expect("a second token joins the same route");
    for copied in [token, joined] {
        assert_eq!(
            registry.restore_binding(other_fingerprint, copied),
            Err(RegistryError::TokenAlreadyBound)
        );
        assert_eq!(
            registry.register_connected_mcp(
                registration(other_fingerprint, copied, "copied"),
                launch("copied"),
            ),
            Err(RegistryError::TokenAlreadyBound)
        );
    }
}

#[test]
fn tokens_for_one_identity_share_its_route_and_worker() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(40);
    let old = TokenDigest::from_token(b"old-token");
    let new = TokenDigest::from_token(b"new-token");
    assert!(matches!(
        registry
            .register_connected_mcp(registration(fingerprint, old, "mcp-old"), launch("first"))
            .unwrap(),
        BrokerDirective::LaunchWorker { .. }
    ));
    assert!(matches!(
        registry
            .register_connected_mcp(registration(fingerprint, new, "mcp-new"), launch("second"))
            .unwrap(),
        BrokerDirective::WaitForWorker { .. }
    ));
    assert_eq!(registry.snapshot(fingerprint).unwrap().reference_count, 2);
    registry
        .mark_worker_ready(fingerprint, "first", worker("worker-shared"))
        .unwrap();
    for token in [old, new] {
        let ResolvedTarget::Worker(request) = registry.resolve_target(&token).unwrap() else {
            panic!("expected the shared worker");
        };
        assert_eq!(request.fingerprint(), fingerprint);
        assert_eq!(request.target().worker_id(), "worker-shared");
    }

    // Old and new sessions interleave across rounds without rebinding or blocking each other.
    for round in 0..4 {
        for (token, name) in [(old, "old"), (new, "new")] {
            let id = format!("{name}-{round}");
            assert!(matches!(
                registry
                    .register_connected_mcp(registration(fingerprint, token, &id), launch(&id))
                    .unwrap(),
                BrokerDirective::ReuseWorker { .. }
            ));
            assert!(matches!(
                registry.resolve_target(&old).unwrap(),
                ResolvedTarget::Worker(_)
            ));
            assert!(matches!(
                registry.resolve_target(&new).unwrap(),
                ResolvedTarget::Worker(_)
            ));
            registry
                .release_mcp(fingerprint, &session(&id), 1_000)
                .unwrap();
        }
    }
    assert_eq!(registry.worker_status_snapshots().len(), 1);
}

#[test]
fn identity_token_limit_rejects_only_new_tokens() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(41);
    let tokens: Vec<_> = (0..=MAX_TOKENS_PER_IDENTITY)
        .map(|index| TokenDigest::from_token(format!("token-{index}").as_bytes()))
        .collect();
    for (index, token) in tokens[..MAX_TOKENS_PER_IDENTITY].iter().enumerate() {
        registry
            .register_connected_mcp(
                registration(fingerprint, *token, &format!("mcp-{index}")),
                launch(&format!("launch-{index}")),
            )
            .unwrap();
    }
    let over_limit = tokens[MAX_TOKENS_PER_IDENTITY];
    assert_eq!(
        registry.register_connected_mcp(
            registration(fingerprint, over_limit, "mcp-over"),
            launch("over"),
        ),
        Err(RegistryError::RouteTokenLimitReached)
    );
    assert!(matches!(
        registry.resolve_target(&over_limit),
        Err(ResolveError::UnknownToken)
    ));
    // Every bound token keeps registering at the limit.
    for token in &tokens[..MAX_TOKENS_PER_IDENTITY] {
        registry
            .register_connected_mcp(registration(fingerprint, *token, "mcp-again"), launch("x"))
            .unwrap();
    }
}

#[test]
fn final_reference_enters_non_revivable_drain() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(5);
    let token = TokenDigest::from_token(b"token-5");
    let first_session = session("mcp-a");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("first"))
        .expect("register");
    let target = worker("worker-1");
    registry
        .mark_worker_ready(fingerprint, "first", Arc::clone(&target))
        .expect("ready");
    let request = match registry.resolve_target(&token).expect("request") {
        ResolvedTarget::Worker(request) => request,
        ResolvedTarget::PassThrough => panic!("expected worker"),
    };

    assert!(matches!(
        registry
            .release_mcp(fingerprint, &first_session, 2_000)
            .expect("release"),
        ReleaseAction::BeginDrain {
            deadline_unix_ms: 2_000,
            ..
        }
    ));
    assert!(matches!(
        registry.resolve_target(&token),
        Err(ResolveError::Unavailable(RouteStateKind::Draining))
    ));
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("second"))
            .expect("wait during drain"),
        BrokerDirective::WaitForWorker {
            retry_after_ms: DEFAULT_RETRY_AFTER_MS
        }
    );
    registry
        .release_mcp(fingerprint, &session("mcp-b"), 2_000)
        .expect("release waiting MCP");
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Draining
    );
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-c"), launch("second"))
            .expect("replacement waits during drain"),
        BrokerDirective::WaitForWorker {
            retry_after_ms: DEFAULT_RETRY_AFTER_MS
        }
    );
    assert_eq!(
        registry.finish_draining(fingerprint, 1_999),
        Err(RegistryError::DrainInProgress)
    );
    drop(request);
    assert_eq!(
        registry.finish_draining(fingerprint, 1_999),
        Ok(DrainCompletion::ActivationRequired {
            session_id: session("mcp-c")
        })
    );
    assert!(matches!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-c"), launch("second"))
            .expect("new generation"),
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "second"
    ));
}

#[test]
fn activation_failure_is_shared_pass_through_until_zero_refs() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(6);
    let token = TokenDigest::from_token(b"token-6");
    let session_id = session("mcp-a");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("failed"))
        .expect("register");
    registry
        .mark_activation_failed(fingerprint, "failed")
        .expect("failure");
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    assert_eq!(
        registry
            .register_connected_mcp(
                registration(fingerprint, token, "mcp-concurrent"),
                launch("ignored")
            )
            .unwrap(),
        BrokerDirective::UsePassThrough
    );
    registry
        .release_mcp(fingerprint, &session_id, 2_000)
        .expect("release");
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    registry
        .release_mcp(fingerprint, &session("mcp-concurrent"), 2_001)
        .expect("last release");
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Empty
    );
    assert!(matches!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("retry"))
            .expect("retry"),
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "retry"
    ));
}

#[test]
fn global_pass_through_never_activates_or_accepts_workers() {
    let registry = Registry::new(true);
    let fingerprint = fingerprint(7);
    let token = TokenDigest::from_token(b"token-7");
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
            .expect("registration"),
        BrokerDirective::UsePassThrough
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    assert_eq!(
        registry.mark_worker_ready(fingerprint, "unused", worker("worker-1")),
        Err(RegistryError::InvalidState {
            expected: RouteStateKind::Activating,
            actual: RouteStateKind::PassThrough,
        })
    );
}

#[test]
fn worker_crash_nominates_one_live_mcp_and_relaunches() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(8);
    let token = TokenDigest::from_token(b"token-8");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("first"))
        .expect("first");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
        .expect("second");
    registry
        .mark_worker_ready(fingerprint, "first", worker("worker-1"))
        .expect("ready");
    assert_eq!(
        registry
            .worker_failed(fingerprint, "worker-1", 10_000)
            .expect("failure"),
        WorkerFailureAction::NominateMcp {
            session_id: session("mcp-a")
        }
    );
    assert_eq!(
        registry.begin_relaunch(fingerprint, &session("mcp-b"), launch("replacement")),
        Err(RegistryError::NotLaunchOwner)
    );
    assert!(matches!(
        registry
            .begin_relaunch(fingerprint, &session("mcp-a"), launch("replacement"))
            .expect("relaunch"),
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "replacement"
    ));
}

#[test]
fn expired_launch_owner_revokes_its_grant_before_fresh_handoff() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(9);
    let token = TokenDigest::from_token(b"token-9");
    let mut first = registration(fingerprint, token, "mcp-a");
    first.lease_expires_at_unix_ms = 100;
    let mut second = registration(fingerprint, token, "mcp-b");
    second.lease_expires_at_unix_ms = 1_000;
    registry
        .register_connected_mcp(first, launch("launch"))
        .expect("first");
    registry
        .register_connected_mcp(second, launch("unused"))
        .expect("second");

    let actions = registry.expire_mcp_leases(100, 2_000);
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0].1,
        ReleaseAction::CancelActivation { activation_id } if activation_id == "launch"
    ));
    assert_eq!(
        registry
            .snapshot(fingerprint)
            .expect("snapshot")
            .reference_count,
        1
    );
    assert!(registry.expire_mcp_leases(100, 2_000).is_empty());
}

#[test]
fn simultaneous_lease_expiry_emits_one_terminal_action() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(10);
    let token = TokenDigest::from_token(b"token-10");
    let mut first = registration(fingerprint, token, "mcp-a");
    first.lease_expires_at_unix_ms = 100;
    let mut second = registration(fingerprint, token, "mcp-b");
    second.lease_expires_at_unix_ms = 100;
    registry
        .register_connected_mcp(first, launch("launch"))
        .expect("first");
    registry
        .register_connected_mcp(second, launch("unused"))
        .expect("second");

    let actions = registry.expire_mcp_leases(100, 2_000);
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0].1,
        ReleaseAction::CancelActivation { activation_id } if activation_id == "launch"
    ));
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot"),
        RouteSnapshot {
            state: RouteStateKind::Empty,
            reference_count: 0,
            launch_owner: None,
            endpoint: None,
            in_flight: 0,
        }
    );
}

#[test]
fn recovery_waits_for_deadline_then_nominates_a_live_mcp() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(11);
    let token = TokenDigest::from_token(b"token-11");
    registry
        .restore_binding(fingerprint, token)
        .expect("persisted binding");
    registry
        .begin_recovery(fingerprint, None, 1_000)
        .expect("recovery");
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
            .expect("reconnecting MCP"),
        BrokerDirective::WaitForWorker {
            retry_after_ms: DEFAULT_RETRY_AFTER_MS
        }
    );
    assert!(matches!(
        registry.finish_recovery(fingerprint, 999, 2_000),
        Err(RegistryError::RecoveryInProgress)
    ));
    assert!(matches!(
        registry
            .finish_recovery(fingerprint, 1_000, 2_000)
            .expect("recovery deadline"),
        RecoveryAction::NominateMcp { session_id } if session_id == session("mcp-a")
    ));
    assert!(matches!(
        registry
            .begin_relaunch(fingerprint, &session("mcp-a"), launch("replacement"))
            .expect("replacement activation"),
        BrokerDirective::LaunchWorker { activation_id, .. } if activation_id == "replacement"
    ));
}

#[test]
fn recovered_worker_becomes_ready_when_an_mcp_reconnects() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(12);
    let token = TokenDigest::from_token(b"token-12");
    registry
        .restore_binding(fingerprint, token)
        .expect("persisted binding");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("restart"))
        .expect("reconnecting MCP");
    let permit = registry
        .authorize_worker_recovery(fingerprint, "worker-recovered")
        .expect("recovery authorization");
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &permit, None, worker("worker-recovered"))
            .expect("worker registration"),
        Some("restart".to_owned())
    );
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
            .expect("reconnecting MCP"),
        BrokerDirective::ReuseWorker {
            endpoint: "http://127.0.0.1:41000".to_owned()
        }
    );
    registry
        .renew_mcp(fingerprint, &session("mcp-a"), 50_000)
        .expect("renewal");
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Ready
    );
}

#[test]
fn recovered_worker_preserves_its_launch_activation_across_route_replacement() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(120);
    let token = TokenDigest::from_token(b"token-120");
    registry
        .restore_binding(fingerprint, token)
        .expect("persisted binding");
    registry
        .register_connected_mcp(
            registration(fingerprint, token, "mcp-a"),
            launch("replacement-activation"),
        )
        .expect("activation");
    let permit = registry
        .authorize_worker_recovery(fingerprint, "worker-recovered")
        .expect("recovery authorization");
    assert_eq!(
        registry
            .publish_recovered_worker(
                fingerprint,
                &permit,
                Some("original-activation"),
                worker("worker-recovered"),
            )
            .expect("worker publication"),
        Some("replacement-activation".to_owned())
    );

    assert_eq!(
        registry
            .cancel_activation(fingerprint, &session("mcp-a"), "original-activation")
            .expect("activation cancellation status"),
        crate::daemon::common::control::ActivationCancellation::Published
    );
}

#[test]
fn recovered_worker_without_references_is_not_authorized() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(18);
    let token = TokenDigest::from_token(b"token-18");
    registry
        .restore_binding(fingerprint, token)
        .expect("persisted binding");
    registry
        .begin_recovery(fingerprint, None, 100)
        .expect("recovery");
    assert_eq!(
        registry.authorize_worker_recovery(fingerprint, "worker-recovered"),
        Err(RegistryError::NoLiveMcpReferences)
    );
}

#[test]
fn expired_activation_enters_transient_pass_through_until_all_references_leave() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(14);
    let token = TokenDigest::from_token(b"token-14");
    let mut expiring_launch = launch("expiring");
    expiring_launch.deadline_unix_ms = 100;
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), expiring_launch)
        .expect("registration");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"))
        .expect("second registration");

    assert!(registry.expire_activations(99).is_empty());
    assert_eq!(
        registry.expire_activations(100),
        vec![ExpiredActivation {
            fingerprint,
            activation_id: "expiring".to_owned(),
        }]
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    assert_eq!(
        registry
            .register_connected_mcp(
                registration(fingerprint, token, "mcp-c"),
                launch("must-not-launch"),
            )
            .expect("pass-through registration"),
        BrokerDirective::UsePassThrough
    );

    for session_id in ["mcp-a", "mcp-b", "mcp-c"] {
        registry
            .release_mcp(fingerprint, &session(session_id), 1_000)
            .expect("release");
    }
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Empty
    );
}

#[test]
fn authenticated_worker_communication_failure_is_route_wide_pass_through() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(15);
    let token = TokenDigest::from_token(b"token-15");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
        .expect("registration");
    registry
        .mark_worker_ready(fingerprint, "launch", worker("worker-failed"))
        .expect("ready");

    assert_eq!(
        registry
            .mark_worker_communication_failed(fingerprint, "worker-failed")
            .expect("communication failure"),
        None
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    assert_eq!(
        registry.mark_worker_communication_failed(fingerprint, "worker-failed"),
        Ok(None)
    );
    registry
        .release_mcp(fingerprint, &session("mcp-a"), 1_000)
        .expect("release");
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Draining
    );
}

#[test]
fn delayed_failure_from_old_worker_does_not_displace_new_ready_generation() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(16);
    let token = TokenDigest::from_token(b"token-16");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
        .expect("registration");
    registry
        .mark_worker_ready(fingerprint, "launch", worker("worker-old"))
        .expect("ready");
    registry
        .worker_failed(fingerprint, "worker-old", 10_000)
        .expect("worker failure");
    registry
        .begin_relaunch(fingerprint, &session("mcp-a"), launch("replacement"))
        .expect("replacement launch");
    registry
        .mark_worker_ready(fingerprint, "replacement", worker("worker-new"))
        .expect("replacement ready");

    assert_eq!(
        registry.mark_worker_communication_failed(fingerprint, "worker-old"),
        Err(RegistryError::WorkerMismatch)
    );
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Ready
    );
}

#[test]
fn recovered_worker_supersedes_restart_activation_without_a_second_worker() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(17);
    let token = TokenDigest::from_token(b"token-17");
    registry
        .register_connected_mcp(
            registration(fingerprint, token, "mcp-a"),
            launch("restart-activation"),
        )
        .expect("reconnected MCP");

    let permit = registry
        .authorize_worker_recovery(fingerprint, "worker-survivor")
        .expect("recovery authorization");
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &permit, None, worker("worker-survivor"))
            .expect("recovered worker"),
        Some("restart-activation".to_owned())
    );
    assert_eq!(
        registry
            .register_connected_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"),)
            .expect("reuse recovered worker"),
        BrokerDirective::ReuseWorker {
            endpoint: "http://127.0.0.1:41000".to_owned(),
        }
    );
}

#[test]
fn recovery_requires_a_live_known_route_and_rejects_permanent_pass_through() {
    let unknown = Registry::new(false);
    assert_eq!(
        unknown.authorize_worker_recovery(fingerprint(21), "worker"),
        Err(RegistryError::UnknownRoute)
    );

    let pass_through = Registry::new(true);
    let fingerprint = fingerprint(22);
    let token = TokenDigest::from_token(b"token-22");
    pass_through
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("unused"))
        .expect("pass-through registration");
    assert_eq!(
        pass_through.authorize_worker_recovery(fingerprint, "worker"),
        Err(RegistryError::RecoveryNotAuthorized)
    );
}

#[test]
fn pass_through_route_is_not_routable_without_a_live_mcp_reference() {
    let registry = Registry::new(true);
    let fingerprint = fingerprint(23);
    let token = TokenDigest::from_token(b"token-23");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("unused"))
        .expect("registration");
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    registry
        .release_mcp(fingerprint, &session("mcp"), 1_000)
        .expect("release");
    assert!(matches!(
        registry.resolve_target(&token),
        Err(ResolveError::Unavailable(RouteStateKind::PassThrough))
    ));
}

#[test]
fn stable_route_bindings_are_bounded_without_permitting_rebinding() {
    let registry = Registry::new(false).with_route_capacity(1);
    let first = fingerprint(24);
    let second = fingerprint(25);
    let token = TokenDigest::from_token(b"bounded-token");
    registry
        .register_connected_mcp(registration(first, token, "mcp-a"), launch("first"))
        .expect("first route");
    assert_eq!(
        registry.register_connected_mcp(
            registration(second, TokenDigest::from_token(b"another-token"), "mcp-b"),
            launch("second"),
        ),
        Err(RegistryError::RouteCapacityReached)
    );
    assert_eq!(
        registry.register_connected_mcp(registration(second, token, "mcp-c"), launch("rebind")),
        Err(RegistryError::TokenAlreadyBound)
    );
}

#[test]
fn capacity_pressure_evicts_only_a_zero_reference_empty_route() {
    let registry = Registry::new(false).with_route_capacity(1);
    let first = fingerprint(26);
    let first_token = TokenDigest::from_token(b"first-token");
    registry
        .register_connected_mcp(registration(first, first_token, "mcp-a"), launch("first"))
        .expect("first route");
    registry
        .release_mcp(first, &session("mcp-a"), 1_000)
        .expect("release empty activation");

    let second = fingerprint(27);
    let second_token = TokenDigest::from_token(b"second-token");
    assert!(matches!(
        registry
            .register_connected_mcp(
                registration(second, second_token, "mcp-b"),
                launch("second"),
            )
            .expect("inactive route should be evicted"),
        BrokerDirective::LaunchWorker { .. }
    ));
    assert!(matches!(
        registry.resolve_target(&first_token),
        Err(ResolveError::UnknownToken)
    ));
}

#[test]
fn evicting_a_route_unbinds_every_token() {
    let registry = Registry::new(false).with_route_capacity(1);
    let first = fingerprint(42);
    let tokens = [
        TokenDigest::from_token(b"evicted-a"),
        TokenDigest::from_token(b"evicted-b"),
    ];
    for (index, token) in tokens.iter().enumerate() {
        let id = format!("mcp-{index}");
        registry
            .register_connected_mcp(registration(first, *token, &id), launch(&id))
            .unwrap();
        registry.release_mcp(first, &session(&id), 1_000).unwrap();
    }
    let second = fingerprint(43);
    registry
        .register_connected_mcp(
            registration(second, TokenDigest::from_token(b"evictor"), "mcp-b"),
            launch("second"),
        )
        .expect("inactive route should be evicted");
    for token in tokens {
        assert!(matches!(
            registry.resolve_target(&token),
            Err(ResolveError::UnknownToken)
        ));
        // A freed token may bind to another identity once its route is gone.
        assert_eq!(registry.restore_binding(second, token), Ok(()));
    }
}

#[test]
fn route_wide_pass_through_cancels_activation_and_preserves_permanent_routes() {
    let registry = Registry::new(false);
    let route_fingerprint = fingerprint(28);
    let token = TokenDigest::from_token(b"token-28");
    registry
        .register_connected_mcp(
            registration(route_fingerprint, token, "mcp"),
            launch("cancel-me"),
        )
        .unwrap();
    assert_eq!(
        registry.mark_route_pass_through(route_fingerprint).unwrap(),
        Some("cancel-me".into())
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    assert_eq!(
        registry.mark_route_pass_through(route_fingerprint).unwrap(),
        None
    );

    let permanent = Registry::new(true);
    let permanent_fingerprint = fingerprint(29);
    permanent
        .register_connected_mcp(
            registration(
                permanent_fingerprint,
                TokenDigest::from_token(b"token-29"),
                "mcp",
            ),
            launch("unused"),
        )
        .unwrap();
    assert_eq!(
        permanent
            .mark_route_pass_through(permanent_fingerprint)
            .unwrap(),
        None
    );
    assert_eq!(
        permanent.snapshot(permanent_fingerprint).unwrap().state,
        RouteStateKind::PassThrough
    );
}

#[test]
fn recovery_completion_covers_live_empty_and_draining_routes() {
    let live = Registry::new(false);
    let live_fingerprint = fingerprint(30);
    let live_token = TokenDigest::from_token(b"token-30");
    live.register_connected_mcp(
        registration(live_fingerprint, live_token, "mcp"),
        launch("launch"),
    )
    .unwrap();
    live.begin_recovery(live_fingerprint, Some(worker("survivor")), 100)
        .unwrap();
    assert!(matches!(
        live.finish_recovery(live_fingerprint, 100, 200).unwrap(),
        RecoveryAction::WorkerRecovered
    ));

    let empty = Registry::new(false);
    let empty_fingerprint = fingerprint(31);
    empty
        .restore_binding(empty_fingerprint, TokenDigest::from_token(b"token-31"))
        .unwrap();
    empty.begin_recovery(empty_fingerprint, None, 100).unwrap();
    assert!(matches!(
        empty.finish_recovery(empty_fingerprint, 100, 200).unwrap(),
        RecoveryAction::RouteEmpty
    ));

    let draining = Registry::new(false);
    let draining_fingerprint = fingerprint(32);
    draining
        .restore_binding(draining_fingerprint, TokenDigest::from_token(b"token-32"))
        .unwrap();
    draining
        .begin_recovery(draining_fingerprint, Some(worker("survivor")), 100)
        .unwrap();
    assert!(matches!(
        draining
            .finish_recovery(draining_fingerprint, 100, 200)
            .unwrap(),
        RecoveryAction::BeginDrain {
            deadline_unix_ms: 200,
            ..
        }
    ));
    assert_eq!(
        draining.finish_draining(draining_fingerprint, 200).unwrap(),
        DrainCompletion::RouteEmpty
    );
}

#[test]
fn current_directive_promotes_a_recovered_target_before_reusing_it() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(33);
    let token = TokenDigest::from_token(b"token-33");
    let mcp = session("mcp");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("launch"))
        .unwrap();
    registry
        .begin_recovery(fingerprint, Some(worker("survivor")), 100)
        .unwrap();

    assert_eq!(
        registry.current_directive(fingerprint, &mcp).unwrap(),
        BrokerDirective::ReuseWorker {
            endpoint: "http://127.0.0.1:41000".to_owned(),
        }
    );
    assert_eq!(
        registry.snapshot(fingerprint).unwrap().state,
        RouteStateKind::Ready
    );
}

#[test]
fn registry_rejects_stale_worker_generations_and_invalid_state_transitions() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(33);
    let token = TokenDigest::from_token(b"token-33");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("active"))
        .unwrap();
    assert_eq!(
        registry.mark_worker_ready(fingerprint, "stale", worker("worker")),
        Err(RegistryError::ActivationMismatch)
    );
    assert_eq!(
        registry.mark_activation_failed(fingerprint, "stale"),
        Err(RegistryError::ActivationMismatch)
    );
    assert_eq!(
        registry.worker_failed(fingerprint, "worker", 100),
        Ok(WorkerFailureAction::Superseded)
    );
    assert_eq!(
        registry.begin_relaunch(fingerprint, &session("mcp"), launch("new")),
        Err(RegistryError::InvalidState {
            expected: RouteStateKind::Recovering,
            actual: RouteStateKind::Activating,
        })
    );
    assert_eq!(
        registry.finish_draining(fingerprint, 100),
        Err(RegistryError::InvalidState {
            expected: RouteStateKind::Draining,
            actual: RouteStateKind::Activating,
        })
    );
    assert_eq!(
        registry.mark_worker_communication_failed(fingerprint, "worker"),
        Err(RegistryError::WorkerMismatch)
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
}

#[test]
fn registry_unknown_route_and_session_errors_are_explicit() {
    let registry = Registry::new(false);
    let unknown = fingerprint(34);
    assert_eq!(registry.snapshot(unknown), Err(RegistryError::UnknownRoute));
    assert_eq!(
        registry.renew_mcp(unknown, &session("missing"), 100),
        Err(RegistryError::UnknownRoute)
    );
    assert!(matches!(
        registry.release_mcp(unknown, &session("missing"), 100),
        Err(RegistryError::UnknownRoute)
    ));
    assert_eq!(
        registry.mark_route_pass_through(unknown),
        Err(RegistryError::UnknownRoute)
    );
    assert!(matches!(
        registry.finish_recovery(unknown, 100, 200),
        Err(RegistryError::UnknownRoute)
    ));

    let fingerprint = fingerprint(35);
    registry
        .register_connected_mcp(
            registration(fingerprint, TokenDigest::from_token(b"token-35"), "known"),
            launch("launch"),
        )
        .unwrap();
    assert_eq!(
        registry.renew_mcp(fingerprint, &session("missing"), 100),
        Err(RegistryError::UnknownMcpSession)
    );
    assert!(matches!(
        registry
            .release_mcp(fingerprint, &session("missing"), 100)
            .unwrap(),
        ReleaseAction::NoChange
    ));
}

#[test]
fn recovery_permits_cover_existing_ready_and_recovering_worker_generations() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(36);
    let token = TokenDigest::from_token(b"token-36");
    registry
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("launch"))
        .unwrap();
    registry
        .mark_worker_ready(fingerprint, "launch", worker("survivor"))
        .unwrap();

    let ready = registry
        .authorize_worker_recovery(fingerprint, "survivor")
        .expect("ready worker permit");
    assert!(matches!(
        ready,
        RecoveryPermit::ExistingWorker {
            recovering: false,
            ..
        }
    ));
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &ready, None, worker("survivor"))
            .unwrap(),
        None
    );

    registry
        .begin_recovery(fingerprint, Some(worker("survivor")), 1_000)
        .unwrap();
    let recovering = registry
        .authorize_worker_recovery(fingerprint, "survivor")
        .expect("recovering worker permit");
    assert!(matches!(
        recovering,
        RecoveryPermit::ExistingWorker {
            recovering: true,
            ..
        }
    ));
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &recovering, None, worker("survivor"))
            .unwrap(),
        None
    );

    registry
        .begin_recovery(fingerprint, Some(worker("new-generation")), 2_000)
        .unwrap();
    assert_eq!(
        registry.publish_recovered_worker(fingerprint, &recovering, None, worker("survivor")),
        Err(RegistryError::RecoveryGenerationChanged)
    );
}

#[test]
fn communication_failures_preserve_draining_and_mismatched_recovery_generations() {
    let registry = Registry::new(false);
    let draining_fingerprint = fingerprint(37);
    let token = TokenDigest::from_token(b"token-37");
    registry
        .register_connected_mcp(
            registration(draining_fingerprint, token, "mcp"),
            launch("launch"),
        )
        .unwrap();
    registry
        .mark_worker_ready(draining_fingerprint, "launch", worker("active"))
        .unwrap();
    registry
        .release_mcp(draining_fingerprint, &session("mcp"), 1_000)
        .unwrap();
    assert_eq!(
        registry.mark_worker_communication_failed(draining_fingerprint, "active"),
        Err(RegistryError::InvalidState {
            expected: RouteStateKind::Ready,
            actual: RouteStateKind::Draining,
        })
    );
    assert_eq!(
        registry.mark_worker_communication_failed(draining_fingerprint, "different"),
        Err(RegistryError::WorkerMismatch)
    );

    let recovering = Registry::new(false);
    let fingerprint = fingerprint(38);
    let token = TokenDigest::from_token(b"token-38");
    recovering
        .register_connected_mcp(registration(fingerprint, token, "mcp"), launch("launch"))
        .unwrap();
    recovering
        .begin_recovery(fingerprint, Some(worker("survivor")), 1_000)
        .unwrap();
    assert_eq!(
        recovering.mark_worker_communication_failed(fingerprint, "different"),
        Err(RegistryError::WorkerMismatch)
    );
    assert_eq!(
        recovering
            .snapshot(fingerprint)
            .unwrap()
            .endpoint
            .as_deref(),
        Some("http://127.0.0.1:41000")
    );
    assert_eq!(
        recovering
            .mark_worker_communication_failed(fingerprint, "survivor")
            .unwrap(),
        None
    );
    assert!(matches!(
        recovering.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
}

#[tokio::test(start_paused = true)]
async fn startup_has_no_deadline_reports_progress_and_cuts_over_after_readiness() {
    for strict in [false, true] {
        let registry = Registry::new(false).with_require_worker(strict);
        let fingerprint = fingerprint(80);
        let token = TokenDigest::from_token(b"slow-startup");
        let mut grant = launch("slow");
        grant.deadline_unix_ms = u64::MAX;
        registry
            .register_connected_mcp(registration(fingerprint, token, "owner"), grant)
            .unwrap();
        let started = tokio::time::Instant::now();
        tokio::time::advance(std::time::Duration::from_secs(59)).await;
        assert!(registry.expire_activations(u64::MAX - 1).is_empty());
        assert!(registry.activation_candidates(u64::MAX).is_empty());
        assert!(
            registry
                .activation_progress(tokio::time::Instant::now())
                .is_empty()
        );
        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            registry.activation_progress(tokio::time::Instant::now()),
            vec![(fingerprint, 60_000)]
        );
        assert!(
            registry
                .activation_progress(tokio::time::Instant::now())
                .is_empty()
        );
        tokio::time::advance(std::time::Duration::from_secs(60)).await;
        assert_eq!(
            registry.activation_progress(tokio::time::Instant::now()),
            vec![(fingerprint, 120_000)]
        );
        assert_eq!(
            registry.startup_elapsed_ms(fingerprint),
            started.elapsed().as_millis() as u64
        );
        if strict {
            assert!(matches!(
                registry.resolve_target(&token),
                Err(ResolveError::Unavailable(RouteStateKind::Activating))
            ));
        } else {
            assert!(matches!(
                registry.resolve_target(&token),
                Ok(ResolvedTarget::PassThrough)
            ));
        }
        registry
            .mark_worker_ready(fingerprint, "slow", worker("published"))
            .unwrap();
        assert!(matches!(
            registry.resolve_target(&token),
            Ok(ResolvedTarget::Worker(_))
        ));
        assert!(
            registry
                .activation_progress(tokio::time::Instant::now())
                .is_empty()
        );
    }
}

#[test]
fn disconnected_references_cannot_launch_and_owner_loss_requires_a_fresh_grant() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(81);
    let token = TokenDigest::from_token(b"owner-handoff");
    registry
        .register_connected_mcp(registration(fingerprint, token, "a"), launch("old"))
        .unwrap();
    registry
        .register_connected_mcp(registration(fingerprint, token, "b"), launch("unused"))
        .unwrap();
    assert_eq!(
        registry.mcp_disconnected(fingerprint, &session("a")),
        Some("old".into())
    );
    assert_eq!(
        registry.activation_candidates(u64::MAX),
        vec![(fingerprint, session("b"))]
    );
    assert_eq!(
        registry.begin_relaunch(fingerprint, &session("a"), launch("wrong")),
        Err(RegistryError::NotLaunchOwner)
    );
    registry
        .begin_relaunch(fingerprint, &session("b"), launch("fresh"))
        .unwrap();
    assert_eq!(
        registry.mark_worker_ready(fingerprint, "old", worker("stale")),
        Err(RegistryError::ActivationMismatch)
    );
    assert_eq!(
        registry.mark_activation_failed(fingerprint, "old"),
        Err(RegistryError::ActivationMismatch)
    );
    registry.mcp_disconnected(fingerprint, &session("b"));
    assert_eq!(registry.snapshot(fingerprint).unwrap().reference_count, 2);
    assert!(registry.activation_candidates(u64::MAX).is_empty());
    registry.mcp_connected(fingerprint, &session("a"));
    assert_eq!(
        registry.activation_candidates(u64::MAX),
        vec![(fingerprint, session("a"))]
    );
}

#[test]
fn actual_failures_retry_beyond_three_attempts_with_capped_backoff() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(82);
    let token = TokenDigest::from_token(b"retry");
    registry
        .register_connected_mcp(registration(fingerprint, token, "owner"), launch("first"))
        .unwrap();
    registry
        .mark_activation_failed(fingerprint, "first")
        .unwrap();
    assert!(registry.activation_candidates(0).is_empty());
    for attempt in 2..=12 {
        let id = format!("retry-{attempt}");
        assert_eq!(
            registry.activation_candidates(u64::MAX),
            vec![(fingerprint, session("owner"))]
        );
        registry
            .begin_relaunch(fingerprint, &session("owner"), launch(&id))
            .unwrap();
        assert!(registry.activation_candidates(u64::MAX).is_empty());
        registry.mark_activation_failed(fingerprint, &id).unwrap();
        let (issued, delay) = registry.activation_retry(fingerprint);
        assert_eq!(issued, attempt);
        assert!(delay <= 60_000);
    }
    assert_eq!(activation_retry_delay(1), 1000);
    assert_eq!(activation_retry_delay(2), 2000);
    assert_eq!(activation_retry_delay(u32::MAX), 60_000);
    registry
        .begin_relaunch(fingerprint, &session("owner"), launch("success"))
        .unwrap();
    registry
        .mark_worker_ready(fingerprint, "success", worker("new"))
        .unwrap();
    assert_eq!(registry.activation_retry(fingerprint).0, 0);
}

#[test]
fn cancellation_and_delayed_worker_cleanup_are_fenced_by_publication_and_identity() {
    use crate::daemon::common::control::ActivationCancellation;
    let registry = Registry::new(false);
    let fingerprint = fingerprint(83);
    let token = TokenDigest::from_token(b"cancel-race");
    registry
        .register_connected_mcp(registration(fingerprint, token, "owner"), launch("first"))
        .unwrap();
    assert_eq!(
        registry
            .cancel_activation(fingerprint, &session("owner"), "first")
            .unwrap(),
        ActivationCancellation::Cancelled
    );
    assert!(
        registry
            .mark_worker_ready(fingerprint, "first", worker("first"))
            .is_err()
    );
    registry
        .begin_relaunch(fingerprint, &session("owner"), launch("second"))
        .unwrap();
    registry
        .mark_worker_ready(fingerprint, "second", worker("second"))
        .unwrap();
    assert_eq!(
        registry
            .cancel_activation(fingerprint, &session("owner"), "second")
            .unwrap(),
        ActivationCancellation::Published
    );
    assert_eq!(
        registry
            .cancel_activation(fingerprint, &session("owner"), "first")
            .unwrap(),
        ActivationCancellation::Superseded
    );
    registry
        .release_mcp(fingerprint, &session("owner"), 100)
        .unwrap();
    assert_eq!(
        registry.worker_failed(fingerprint, "second", 200).unwrap(),
        WorkerFailureAction::Draining
    );
    registry
        .finish_draining_worker(fingerprint, "second", 100)
        .unwrap();
    assert_eq!(
        registry.worker_failed(fingerprint, "second", 200).unwrap(),
        WorkerFailureAction::AlreadyRemoved
    );
    registry
        .register_connected_mcp(
            registration(fingerprint, token, "new-owner"),
            launch("third"),
        )
        .unwrap();
    assert_eq!(
        registry
            .finish_draining_worker(fingerprint, "second", 300)
            .unwrap(),
        DrainCompletion::AlreadyRemoved
    );
    assert_eq!(
        registry.mark_worker_communication_failed(fingerprint, "second"),
        Err(RegistryError::WorkerMismatch)
    );
    assert_eq!(
        registry.snapshot(fingerprint).unwrap().state,
        RouteStateKind::Activating
    );
    registry
        .mark_worker_ready(fingerprint, "third", worker("third"))
        .unwrap();
    assert_eq!(
        registry.worker_failed(fingerprint, "unknown", 400).unwrap(),
        WorkerFailureAction::Superseded
    );
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::Worker(_))
    ));
}
