// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test(flavor = "current_thread")]
async fn redis_backend_new_rejects_invalid_urls_before_connecting() {
    let err = RedisBackend::new("not-a-redis-url", "prefix:")
        .await
        .err()
        .expect("expected invalid redis url to fail");

    match err {
        AdaptiveError::Storage(message) => {
            assert!(message.contains("redis client"));
        }
        other => panic!("unexpected redis constructor error: {other}"),
    }
}
