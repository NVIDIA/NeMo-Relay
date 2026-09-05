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
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("first"))
        .expect("first registration");
    assert!(matches!(
        first,
        BrokerDirective::LaunchWorker {
            ref activation_id,
            ..
        } if activation_id == "first"
    ));

    let retry = registry
        .register_mcp(
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
        .register_mcp(registration(fingerprint, token, "mcp-b"), launch("second"))
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
                    .register_mcp(
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
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
        .expect("registration");
    let target = worker("worker-1");
    registry
        .mark_worker_ready(fingerprint, "launch", Arc::clone(&target))
        .expect("worker ready");

    assert_eq!(
        registry
            .register_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"))
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
fn token_and_fingerprint_bindings_cannot_be_reassigned() {
    let registry = Registry::new(false);
    let first_fingerprint = fingerprint(3);
    let other_fingerprint = fingerprint(4);
    let token = TokenDigest::from_token(b"stable-token");
    registry
        .restore_binding(first_fingerprint, token)
        .expect("binding");
    assert_eq!(
        registry.restore_binding(other_fingerprint, token),
        Err(RegistryError::TokenAlreadyBound)
    );
    assert_eq!(
        registry.restore_binding(
            first_fingerprint,
            TokenDigest::from_token(b"different-token")
        ),
        Err(RegistryError::FingerprintTokenMismatch)
    );
}

#[test]
fn final_reference_enters_non_revivable_drain() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(5);
    let token = TokenDigest::from_token(b"token-5");
    let first_session = session("mcp-a");
    registry
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("first"))
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
            .register_mcp(registration(fingerprint, token, "mcp-b"), launch("second"))
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
            .register_mcp(registration(fingerprint, token, "mcp-c"), launch("second"))
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
            .register_mcp(registration(fingerprint, token, "mcp-c"), launch("second"))
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
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("failed"))
        .expect("register");
    registry
        .mark_activation_failed(fingerprint, "failed")
        .expect("failure");
    assert!(matches!(
        registry.resolve_target(&token),
        Ok(ResolvedTarget::PassThrough)
    ));
    registry
        .release_mcp(fingerprint, &session_id, 2_000)
        .expect("release");
    assert_eq!(
        registry.snapshot(fingerprint).expect("snapshot").state,
        RouteStateKind::Empty
    );
    assert!(matches!(
        registry
            .register_mcp(registration(fingerprint, token, "mcp-b"), launch("retry"))
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
            .register_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
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
        .register_mcp(registration(fingerprint, token, "mcp-b"), launch("first"))
        .expect("first");
    registry
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
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
fn expired_launch_owner_is_transferred_idempotently() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(9);
    let token = TokenDigest::from_token(b"token-9");
    let mut first = registration(fingerprint, token, "mcp-a");
    first.lease_expires_at_unix_ms = 100;
    let mut second = registration(fingerprint, token, "mcp-b");
    second.lease_expires_at_unix_ms = 1_000;
    registry
        .register_mcp(first, launch("launch"))
        .expect("first");
    registry
        .register_mcp(second, launch("unused"))
        .expect("second");

    let actions = registry.expire_mcp_leases(100, 2_000);
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0].1,
        ReleaseAction::TransferActivation {
            session_id,
            directive: BrokerDirective::LaunchWorker { activation_id, .. },
        } if session_id == &session("mcp-b") && activation_id == "launch"
    ));
    assert_eq!(
        registry
            .snapshot(fingerprint)
            .expect("snapshot")
            .reference_count,
        1
    );
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
        .register_mcp(first, launch("launch"))
        .expect("first");
    registry
        .register_mcp(second, launch("unused"))
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
            .register_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
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
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("restart"))
        .expect("reconnecting MCP");
    let permit = registry
        .authorize_worker_recovery(fingerprint, "worker-recovered")
        .expect("recovery authorization");
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &permit, worker("worker-recovered"))
            .expect("worker registration"),
        Some("restart".to_owned())
    );
    assert_eq!(
        registry
            .register_mcp(registration(fingerprint, token, "mcp-a"), launch("unused"))
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
        .register_mcp(registration(fingerprint, token, "mcp-a"), expiring_launch)
        .expect("registration");
    registry
        .register_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"))
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
            .register_mcp(
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
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
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
        RouteStateKind::Empty
    );
}

#[test]
fn delayed_failure_from_old_worker_does_not_displace_new_ready_generation() {
    let registry = Registry::new(false);
    let fingerprint = fingerprint(16);
    let token = TokenDigest::from_token(b"token-16");
    registry
        .register_mcp(registration(fingerprint, token, "mcp-a"), launch("launch"))
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
        .register_mcp(
            registration(fingerprint, token, "mcp-a"),
            launch("restart-activation"),
        )
        .expect("reconnected MCP");

    let permit = registry
        .authorize_worker_recovery(fingerprint, "worker-survivor")
        .expect("recovery authorization");
    assert_eq!(
        registry
            .publish_recovered_worker(fingerprint, &permit, worker("worker-survivor"))
            .expect("recovered worker"),
        Some("restart-activation".to_owned())
    );
    assert_eq!(
        registry
            .register_mcp(registration(fingerprint, token, "mcp-b"), launch("unused"),)
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
        .register_mcp(registration(fingerprint, token, "mcp"), launch("unused"))
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
        .register_mcp(registration(fingerprint, token, "mcp"), launch("unused"))
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
        .register_mcp(registration(first, token, "mcp-a"), launch("first"))
        .expect("first route");
    assert_eq!(
        registry.register_mcp(
            registration(second, TokenDigest::from_token(b"another-token"), "mcp-b"),
            launch("second"),
        ),
        Err(RegistryError::RouteCapacityReached)
    );
    assert_eq!(
        registry.register_mcp(registration(second, token, "mcp-c"), launch("rebind")),
        Err(RegistryError::TokenAlreadyBound)
    );
}

#[test]
fn capacity_pressure_evicts_only_a_zero_reference_empty_route() {
    let registry = Registry::new(false).with_route_capacity(1);
    let first = fingerprint(26);
    let first_token = TokenDigest::from_token(b"first-token");
    registry
        .register_mcp(registration(first, first_token, "mcp-a"), launch("first"))
        .expect("first route");
    registry
        .release_mcp(first, &session("mcp-a"), 1_000)
        .expect("release empty activation");

    let second = fingerprint(27);
    let second_token = TokenDigest::from_token(b"second-token");
    assert!(matches!(
        registry
            .register_mcp(
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
