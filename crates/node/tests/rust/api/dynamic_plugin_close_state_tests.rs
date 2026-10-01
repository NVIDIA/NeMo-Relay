// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn close_result_is_published_before_a_retry_can_reset_completion() {
    let activation = CorePluginHostActivation::initialize_exact(PluginConfig::default())
        .await
        .expect("empty plugin host must initialize");
    let state = Arc::new(DynamicPluginCloseState::new(activation));
    let activation = {
        let mut status = state
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let activation = match &mut *status {
            DynamicPluginCloseStatus::Active(activation) => {
                activation.take().expect("activation must be owned")
            }
            DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => {
                panic!("new activation must be active")
            }
        };
        *status = DynamicPluginCloseStatus::Closing;
        activation
    };

    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let finish_state = Arc::clone(&state);
    let finish = std::thread::spawn(move || {
        finish_state.finish_with_hook(Some(activation), Err("first close failed".into()), || {
            entered_tx.send(()).expect("test must observe publication");
            release_rx.recv().expect("test must release publication");
        });
    });

    entered_rx
        .recv()
        .expect("finish must reach completion publication");
    assert!(matches!(
        state.status.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    release_tx
        .send(())
        .expect("finish thread must still be waiting");
    finish.join().expect("finish thread must not panic");

    let mut activation = {
        let mut status = state
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let activation = match &mut *status {
            DynamicPluginCloseStatus::Active(activation) => activation
                .take()
                .expect("failed close must remain retryable"),
            DynamicPluginCloseStatus::Closing | DynamicPluginCloseStatus::Closed => {
                panic!("failed close must restore the active state")
            }
        };
        state.completion.send_replace(None);
        *status = DynamicPluginCloseStatus::Closing;
        activation
    };
    assert!(state.completion.borrow().is_none());

    activation.close().expect("retry cleanup must succeed");
}
