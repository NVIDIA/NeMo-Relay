// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming commit behavior.

use std::time::Duration;

use tokio::sync::{oneshot, watch};
use tokio_stream::StreamExt;

use super::*;

#[tokio::test]
async fn cache_commit_does_not_delay_stream_completion() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (cancel, _) = watch::channel(false);
    let (_, closed) = watch::channel(None::<FlowResult<()>>);
    let (release, wait) = oneshot::channel();
    let commit: CacheCommit = Box::pin(async move {
        let _ = wait.await;
    });
    assert!(tx.send(TeeMessage::Commit(commit)).await.is_ok());

    let mut stream = ResponseCacheReceiver {
        receiver: ReceiverStream::new(rx),
        cancel,
        closed,
        finished: false,
    };
    assert!(
        tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("cache write must not delay stream completion")
            .is_none()
    );
    let _ = release.send(());
}
