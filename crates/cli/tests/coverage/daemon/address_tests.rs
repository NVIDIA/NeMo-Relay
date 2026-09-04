// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn daemon_target_requires_tls_away_from_loopback() {
    assert!(daemon_url("http://127.0.0.1:47632").is_ok());
    assert!(daemon_url("https://relay.example.com:443").is_ok());
    assert!(daemon_url("http://relay.example.com:47632").is_err());
    assert!(daemon_url("https://0.0.0.0:47632").is_err());
    assert!(daemon_url("https://relay.example.com").is_err());
}

#[test]
fn worker_port_zero_is_implicit_only() {
    assert_eq!(
        worker_socket(Ipv4Addr::LOCALHOST, None).unwrap(),
        "127.0.0.1:0".parse().unwrap()
    );
    assert!(worker_socket(Ipv4Addr::LOCALHOST, Some(0)).is_err());
    assert!(worker_socket(Ipv4Addr::new(10, 0, 0, 1), None).is_err());
}

#[test]
fn unspecified_worker_requires_concrete_advertisement() {
    let local: SocketAddr = "0.0.0.0:43210".parse().unwrap();
    assert!(worker_advertised_address(local, None).is_err());
    assert_eq!(
        worker_advertised_address(local, Some("worker.example.com")).unwrap(),
        "worker.example.com:43210"
    );
}
