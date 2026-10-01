// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
fn key(id: usize) -> CompletionKey {
    CompletionKey {
        owner: "a".into(),
        session: "s".into(),
        kind: "tool",
        agent_kind: AgentKind::Codex,
        invocation: id.to_string(),
    }
}
#[test]
fn cache_evicts_oldest_and_expires_at_five_minutes() {
    let mut cache = CompletionCache::default();
    for id in 0..=CAPACITY {
        cache.record(key(id));
    }
    assert_eq!(cache.entries.len(), CAPACITY);
    assert!(!cache.contains(&key(0)));
    assert!(cache.contains(&key(CAPACITY)));
    cache.expire(Instant::now() + RETENTION);
    assert!(cache.entries.is_empty());
    assert!(cache.order.is_empty());
}
#[test]
fn lifecycle_and_owner_identifiers_remain_independent() {
    let mut cache = CompletionCache::default();
    let tool = key(1);
    cache.record(tool.clone());
    let mut other = tool.clone();
    other.kind = "turn";
    assert!(!cache.contains(&other));
    other = tool.clone();
    other.owner = "b".into();
    assert!(!cache.contains(&other));
    other = tool.clone();
    other.agent_kind = AgentKind::ClaudeCode;
    assert!(!cache.contains(&other));
    other = tool;
    other.session = "other".into();
    assert!(!cache.contains(&other));
}
