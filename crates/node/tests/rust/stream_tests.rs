// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay::api::runtime::LlmStreamInner;

struct CancelDuringPoll {
    cancel: Option<tokio::sync::watch::Sender<bool>>,
    item: Option<FlowResult<Json>>,
}

impl Stream for CancelDuringPoll {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // The select has already polled cancellation when it calls into the stream.
        if let Some(cancel) = self.cancel.take() {
            cancel.send_replace(true);
        }
        Poll::Ready(self.item.take())
    }
}

impl LlmStreamInner for CancelDuringPoll {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        Box::pin(async { Err(FlowError::Internal("cleanup boom".into())) })
    }
}

#[test]
fn forwarding_preserves_errors_when_cancel_arrives_during_a_poll() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for cancel_during_poll in [false, true] {
            for collector_fails in [false, true] {
                let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
                let (tx, mut rx) = tokio::sync::mpsc::channel(1);
                let (requests, request_rx) = tokio::sync::mpsc::channel(1);
                let (closed, closed_rx) = tokio::sync::watch::channel(None);
                requests.send(()).await.unwrap();
                drop(requests);
                let item = if collector_fails {
                    Err(FlowError::Internal("collector boom".into()))
                } else {
                    Ok(serde_json::json!({"token": "first"}))
                };
                let stream = LlmJsonStream::from_closeable(CancelDuringPoll {
                    cancel: cancel_during_poll.then(|| cancel.clone()),
                    item: Some(item),
                });

                forward_stream_to_channel(stream, tx, request_rx, cancel_rx, closed).await;

                if cancel_during_poll {
                    assert!(rx.recv().await.is_none(), "close must end the pending read");
                } else {
                    let delivered = rx.recv().await.unwrap();
                    if collector_fails {
                        assert!(
                            delivered
                                .unwrap_err()
                                .to_string()
                                .contains("collector boom")
                        );
                    } else {
                        assert_eq!(delivered.unwrap(), serde_json::json!({"token": "first"}));
                    }
                }
                let outcome = closed_rx.borrow().clone().unwrap().unwrap_err();
                let expected = if cancel_during_poll && collector_fails {
                    "collector boom"
                } else {
                    "cleanup boom"
                };
                assert_eq!(outcome.to_string(), format!("internal error: {expected}"));
            }
        }
    });
}
