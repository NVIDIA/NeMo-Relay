// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_flow::FlowError;
use serde_json::json;

#[test]
fn test_next_yields_value_then_end_of_stream() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tx.send(Ok(json!({"chunk": 1}))).await.unwrap();
        drop(tx);

        let stream = LlmStream {
            receiver: tokio::sync::Mutex::new(rx),
        };

        assert_eq!(stream.next().await.unwrap(), Some(json!({"chunk": 1})));
        assert_eq!(stream.next().await.unwrap(), None);
    });
}

#[test]
fn test_next_converts_stream_errors() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
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
    });
}
