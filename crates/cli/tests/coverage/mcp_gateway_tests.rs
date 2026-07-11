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
        move |address, _expected_instance| {
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

#[tokio::test(start_paused = true)]
async fn lifecycle_retirement_is_checked_before_a_healthy_heartbeat() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let health_calls = Arc::new(AtomicUsize::new(0));
    let health_calls_for_probe = health_calls.clone();
    let monitor = tokio::spawn(maintain_gateway_instances_with_generation(
        "127.0.0.1:47632".parse().unwrap(),
        crate::sidecar::GatewayEndpoint {
            address: "127.0.0.1:47632".parse().unwrap(),
            url: "http://gateway".into(),
            instance_id: "first".into(),
        },
        Duration::from_secs(30),
        move |_url, _expected| {
            health_calls_for_probe.fetch_add(1, Ordering::SeqCst);
            async { Ok(Some("replacement".into())) }
        },
        |_address, _expected| async { panic!("retired lifecycle attempted recovery") },
        || async { Err(CliError::Launch("cohort retired".into())) },
    ));

    tokio::time::advance(Duration::from_secs(30)).await;
    let error = monitor.await.unwrap().unwrap_err();

    assert!(error.to_string().contains("cohort retired"));
    assert_eq!(health_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lifecycle_retirement_during_health_is_checked_before_adoption() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let retired = Arc::new(AtomicBool::new(false));
    let health_calls = Arc::new(AtomicUsize::new(0));
    let retired_during_health = retired.clone();
    let health_calls_for_probe = health_calls.clone();
    let retired_for_verification = retired.clone();
    let monitor = tokio::spawn(maintain_gateway_instances_with_generation(
        "127.0.0.1:47632".parse().unwrap(),
        crate::sidecar::GatewayEndpoint {
            address: "127.0.0.1:47632".parse().unwrap(),
            url: "http://gateway".into(),
            instance_id: "first".into(),
        },
        Duration::from_millis(1),
        move |_url, _expected| {
            health_calls_for_probe.fetch_add(1, Ordering::SeqCst);
            retired_during_health.store(true, Ordering::SeqCst);
            async { Ok(Some("replacement".into())) }
        },
        |_address, _expected| async { panic!("retired lifecycle attempted recovery") },
        move || {
            let retired = retired_for_verification.load(Ordering::SeqCst);
            async move {
                if retired {
                    Err(CliError::Launch("cohort retired during health".into()))
                } else {
                    Ok(())
                }
            }
        },
    ));

    let error = tokio::time::timeout(Duration::from_secs(1), monitor)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();

    assert!(error.to_string().contains("retired during health"));
    assert_eq!(health_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn production_lifecycle_verifier_rejects_a_rotated_cohort() {
    use std::ffi::OsString;

    struct EnvRestore {
        previous: Option<OsString>,
    }
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            // SAFETY: This test holds the repository-wide environment mutex.
            unsafe {
                match self.previous.take() {
                    Some(value) => {
                        std::env::set_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV, value)
                    }
                    None => std::env::remove_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV),
                }
            }
        }
    }

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let state = tempfile::tempdir().unwrap();
    let previous = std::env::var_os(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV);
    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe { std::env::set_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV, state.path()) };
    let _restore = EnvRestore { previous };
    let url = "http://127.0.0.1:47632";
    let old = crate::sidecar::EndpointLease::acquire(state.path(), url).unwrap();
    let cohort = old.cohort_id().to_string();
    crate::sidecar::stop_owned_sidecar_and_reset(url).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(verify_lifecycle_async(None, cohort, url.into()))
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("retired by an integration update"),
        "{error}"
    );
}

#[test]
fn generation_transaction_polling_is_cancellable_for_clean_mcp_shutdown() {
    struct EnvRestore(Option<std::ffi::OsString>);
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            // SAFETY: The test holds the repository-wide environment mutex.
            unsafe {
                match self.0.take() {
                    Some(value) => {
                        std::env::set_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV, value)
                    }
                    None => std::env::remove_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV),
                }
            }
        }
    }

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let previous = std::env::var_os(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV);
    // SAFETY: This test holds the repository-wide environment mutex.
    unsafe { std::env::set_var(crate::sidecar::BOOTSTRAP_STATE_DIR_ENV, &state) };
    let _restore = EnvRestore(previous);
    let gateway_url = "http://127.0.0.1:47632";
    let endpoint_lease = crate::sidecar::EndpointLease::acquire(&state, gateway_url).unwrap();
    let cohort = endpoint_lease.cohort_id().to_string();
    let path = dir
        .path()
        .join(crate::install_generation::GENERATION_FILE_NAME);
    crate::install_generation::write_new_generation(&path).unwrap();
    let generation = crate::install_generation::InstallGeneration::capture(path.clone()).unwrap();
    let mut retirement = crate::install_generation::GenerationRetirement::acquire(&path)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let verification = tokio::spawn(verify_lifecycle_async(
            Some(generation),
            cohort,
            gateway_url.into(),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!verification.is_finished());
        verification.abort();
        let error = tokio::time::timeout(Duration::from_millis(250), verification)
            .await
            .expect("generation lifecycle poll ignored MCP cancellation")
            .unwrap_err();
        assert!(error.is_cancelled());
    });
    retirement.restore_after_rollback().unwrap();
}

#[test]
fn endpoint_transaction_polling_is_cancellable_for_clean_mcp_shutdown() {
    struct EnvRestore(Option<std::ffi::OsString>);
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            // SAFETY: The test holds the repository-wide environment mutex.
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                    None => std::env::remove_var("XDG_CONFIG_HOME"),
                }
            }
        }
    }

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let previous = std::env::var_os("XDG_CONFIG_HOME");
    // SAFETY: The test holds the repository-wide environment mutex.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", dir.path()) };
    let _restore = EnvRestore(previous);
    let state = crate::sidecar::sidecar_state_dir().unwrap();
    let url = "http://127.0.0.1:47632";
    let lease = crate::sidecar::EndpointLease::acquire(&state, url).unwrap();
    let cohort = lease.cohort_id().to_string();
    let transaction = crate::sidecar::lock_sidecar_endpoint(&state, url).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let verification = tokio::spawn(verify_lifecycle_async(None, cohort, url.into()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!verification.is_finished());
        verification.abort();
        let error = tokio::time::timeout(Duration::from_millis(250), verification)
            .await
            .expect("endpoint lifecycle poll ignored MCP cancellation")
            .unwrap_err();
        assert!(error.is_cancelled());
    });
    drop(transaction);
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
            move |address, _expected_instance| {
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
