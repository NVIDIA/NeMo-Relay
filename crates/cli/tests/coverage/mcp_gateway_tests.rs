// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn recovery_tracks_gateway_instances_instead_of_the_local_starter() {
    let mut recovery = RecoveryState::new("first".into());

    recovery.observe("first".into()).unwrap();
    recovery.require_restart().unwrap();
    recovery.observe("second".into()).unwrap();
    assert_eq!(recovery.instance_id(), "second");
    assert!(recovery.require_restart().is_err());
}

#[test]
fn observing_two_replacements_exhausts_the_single_restart_allowance() {
    let mut recovery = RecoveryState::new("first".into());
    recovery.observe("second".into()).unwrap();

    let error = recovery.observe("third".into()).unwrap_err();

    assert!(error.to_string().contains("replaced again"));
}

#[tokio::test(start_paused = true)]
async fn production_heartbeat_recovers_after_one_thirty_second_interval() {
    let (restarted_tx, restarted_rx) = tokio::sync::oneshot::channel();
    let mut restarted_tx = Some(restarted_tx);
    let monitor = tokio::spawn(maintain_gateway_with(
        "127.0.0.1:47632".parse().unwrap(),
        "http://gateway".into(),
        Duration::from_secs(30),
        |_url| async { Ok(false) },
        move |address| {
            let sender = restarted_tx.take();
            async move {
                if let Some(sender) = sender {
                    let _ = sender.send(());
                }
                Ok(crate::sidecar::GatewayEndpoint {
                    address,
                    url: "http://recovered".into(),
                    instance_id: "recovered".into(),
                })
            }
        },
    ));

    tokio::time::advance(Duration::from_secs(30)).await;
    restarted_rx.await.unwrap();
    assert!(!monitor.is_finished());
    monitor.abort();
}

#[tokio::test]
async fn concurrent_clients_consume_the_same_replacement_allowance() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    let current = Arc::new(Mutex::new(Some("first".to_string())));
    let restart_count = Arc::new(AtomicUsize::new(0));
    let observed_replacement = Arc::new(AtomicUsize::new(0));
    let mut monitors = Vec::new();
    for _ in 0..2 {
        let current_for_health = current.clone();
        let current_for_restart = current.clone();
        let restart_count = restart_count.clone();
        let observed_replacement = observed_replacement.clone();
        monitors.push(tokio::spawn(maintain_gateway_instances_with_generation(
            "127.0.0.1:47632".parse().unwrap(),
            crate::sidecar::GatewayEndpoint {
                address: "127.0.0.1:47632".parse().unwrap(),
                url: "http://gateway".into(),
                instance_id: "first".into(),
            },
            Duration::from_millis(1),
            move |_url, expected| {
                let current = current_for_health.lock().unwrap().clone();
                if expected == "second" && current.as_deref() == Some("second") {
                    observed_replacement.fetch_add(1, Ordering::SeqCst);
                }
                async move { Ok(current) }
            },
            move |address| {
                let current = current_for_restart.clone();
                let restart_count = restart_count.clone();
                async move {
                    let mut current = current.lock().unwrap();
                    let started = current.is_none();
                    if started {
                        *current = Some("second".into());
                        restart_count.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(crate::sidecar::GatewayEndpoint {
                        address,
                        url: "http://gateway".into(),
                        instance_id: current.clone().unwrap(),
                    })
                }
            },
            || async { Ok(()) },
        )));
    }

    *current.lock().unwrap() = None;
    tokio::time::timeout(Duration::from_secs(2), async {
        while observed_replacement.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(restart_count.load(Ordering::SeqCst), 1);

    *current.lock().unwrap() = None;
    for monitor in monitors {
        let error = tokio::time::timeout(Duration::from_secs(2), monitor)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("after its coordinated restart"));
    }
    assert_eq!(restart_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropping_gateway_lease_aborts_its_monitor() {
    struct NotifyOnDrop(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for NotifyOnDrop {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
    let monitor = tokio::spawn(async move {
        let _notify = NotifyOnDrop(Some(dropped_tx));
        let _ = started_tx.send(());
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    });
    started_rx.await.unwrap();

    drop(GatewayLease { monitor });

    tokio::time::timeout(Duration::from_secs(1), dropped_rx)
        .await
        .expect("gateway monitor was not aborted when its lease dropped")
        .unwrap();
}
