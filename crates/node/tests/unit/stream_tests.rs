// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_flow::error::FlowError;
use serde_json::json;

#[tokio::test]
async fn next_yields_values_and_then_finishes() {
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    tx.send(Ok(json!({"chunk": 1}))).await.unwrap();
    drop(tx);

    let stream = LlmStream {
        receiver: tokio::sync::Mutex::new(rx),
    };

    assert_eq!(stream.next().await.unwrap(), Some(json!({"chunk": 1})));
    assert_eq!(stream.next().await.unwrap(), None);
}

#[tokio::test]
async fn next_converts_flow_errors_into_napi_errors() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.send(Err(FlowError::Internal("stream failed".into())))
        .await
        .unwrap();
    drop(tx);

    let stream = LlmStream {
        receiver: tokio::sync::Mutex::new(rx),
    };

    let err = stream.next().await.unwrap_err();
    assert!(err.to_string().contains("stream failed"));
}
