// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn recovery_requires_three_consecutive_failures() {
    let mut recovery = RecoveryState::default();

    assert!(!recovery.record_failure().unwrap());
    assert!(!recovery.record_failure().unwrap());
    recovery.record_healthy();
    assert!(!recovery.record_failure().unwrap());
    assert!(!recovery.record_failure().unwrap());
    assert!(recovery.record_failure().unwrap());
}

#[test]
fn rediscovery_preserves_the_single_restart_allowance() {
    let mut recovery = RecoveryState::default();
    for _ in 0..2 {
        assert!(!recovery.record_failure().unwrap());
    }
    assert!(recovery.record_failure().unwrap());
    recovery.record_recovery(false);

    for _ in 0..2 {
        assert!(!recovery.record_failure().unwrap());
    }
    assert!(recovery.record_failure().unwrap());
    recovery.record_recovery(true);

    for _ in 0..2 {
        assert!(!recovery.record_failure().unwrap());
    }
    let error = recovery.record_failure().unwrap_err();
    assert!(error.to_string().contains("after its coordinated restart"));
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
