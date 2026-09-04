// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::control::now_unix_ms;
use crate::daemon::common::protocol::SensitiveString;

fn bootstrap(bind_ip: Ipv4Addr, port: u16) -> WorkerBootstrap {
    WorkerBootstrap {
        activation_id: "activation".into(),
        activation_token: SensitiveString::new("secret").expect("secret"),
        deadline_unix_ms: now_unix_ms().saturating_add(10_000),
        bind_ip,
        port,
        advertise_address: None,
    }
}

#[test]
fn default_worker_network_matches_loopback_ephemeral_grant() {
    let options = Options {
        daemon_address: "http://127.0.0.1:47632".into(),
        bind: Ipv4Addr::LOCALHOST,
        port: None,
        advertise_address: None,
    };
    assert_eq!(
        validate_bootstrap(&options, &bootstrap(Ipv4Addr::LOCALHOST, 0)).expect("valid grant"),
        "127.0.0.1:0".parse().expect("socket")
    );
}

#[test]
fn worker_network_must_match_activation_grant() {
    let options = Options {
        daemon_address: "http://127.0.0.1:47632".into(),
        bind: Ipv4Addr::LOCALHOST,
        port: Some(4444),
        advertise_address: None,
    };
    assert!(validate_bootstrap(&options, &bootstrap(Ipv4Addr::LOCALHOST, 0)).is_err());
}

#[test]
fn daemon_wall_clock_deadline_is_not_rejected_by_the_worker_clock() {
    let options = Options {
        daemon_address: "http://127.0.0.1:47632".into(),
        bind: Ipv4Addr::LOCALHOST,
        port: None,
        advertise_address: None,
    };
    let mut expired = bootstrap(Ipv4Addr::LOCALHOST, 0);
    expired.deadline_unix_ms = now_unix_ms();
    assert!(validate_bootstrap(&options, &expired).is_ok());
}
