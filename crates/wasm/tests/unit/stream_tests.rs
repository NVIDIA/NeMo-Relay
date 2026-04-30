// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for stream in the NeMo Flow WebAssembly crate.

#[cfg(target_arch = "wasm32")]
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(future)
}

#[cfg(target_arch = "wasm32")]
#[test]
fn next_returns_value_done_false_and_then_done_true() {
    use super::*;
    use serde_json::json;

    block_on(async {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tx.send(Ok(json!({"chunk": 1}))).await.unwrap();
        drop(tx);

        let stream = LlmStream {
            receiver: tokio::sync::Mutex::new(rx),
        };
        let first = stream.next().await.unwrap();
        assert_eq!(
            crate::convert::js_to_json(&first).unwrap(),
            json!({"chunk": 1})
        );

        let second = stream.next().await.unwrap();
        assert!(second.is_null());
    });
}

#[cfg(target_arch = "wasm32")]
#[test]
fn next_returns_js_error_for_stream_errors() {
    use super::*;

    block_on(async {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(Err(nemo_flow::error::FlowError::Internal(
            "stream failed".to_string(),
        )))
        .await
        .unwrap();
        drop(tx);

        let stream = LlmStream {
            receiver: tokio::sync::Mutex::new(rx),
        };
        let err = stream.next().await.unwrap_err();
        assert!(err.as_string().unwrap().contains("stream failed"));
    });
}
