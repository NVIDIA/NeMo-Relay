// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming commit behavior.

use std::time::Duration;

use tokio::sync::{oneshot, watch};
use tokio_stream::StreamExt;

use super::*;

#[tokio::test]
async fn write_behind_returns_eof_before_cache_commit_completes() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (cancel, _) = watch::channel(false);
    let (_, closed) = watch::channel(None::<FlowResult<()>>);
    let (release, wait) = oneshot::channel();
    let (commit_done, committed) = oneshot::channel();
    let commit: CacheCommit = Box::pin(async move {
        let _ = wait.await;
        let _ = commit_done.send(());
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
            .expect("write-behind cache publication must not delay stream completion")
            .is_none()
    );
    release
        .send(())
        .expect("detached cache commit must still be waiting");
    tokio::time::timeout(Duration::from_secs(1), committed)
        .await
        .expect("detached cache commit must resume after release")
        .expect("detached cache commit must run to completion");
}
