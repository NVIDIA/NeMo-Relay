// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource routing boundaries shared by OTLP logs and metrics.

use super::*;
use crate::api::event::{BaseEvent, MarkEvent, ScopeEvent};
use crate::api::scope::ScopeType;

fn scope(
    id: Uuid,
    parent: Option<Uuid>,
    category: ScopeCategory,
    timestamp: DateTime<Utc>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(id)
            .parent_uuid_opt(parent)
            .name("scope")
            .timestamp(timestamp)
            .build(),
        category,
        Vec::new(),
        ScopeType::Agent.into(),
        None,
    ))
}

fn mark(parent: Uuid, timestamp: DateTime<Utc>) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent)
            .name("late")
            .timestamp(timestamp)
            .build(),
        None,
        None,
    ))
}

#[test]
fn completed_resource_routes_expire_before_lookup_for_marks_and_child_starts() {
    let ttl = Duration::from_secs(60);
    let now = Utc::now();
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    for child_start in [false, true] {
        let mut lineage = SignalResourceLineage::new();
        let root = Uuid::now_v7();
        lineage.process(
            &scope(root, None, ScopeCategory::Start, now),
            ttl,
            &diagnostics,
            || Some("root"),
        );
        assert_eq!(
            lineage.process(
                &scope(root, None, ScopeCategory::End, now),
                ttl,
                &diagnostics,
                || None
            ),
            Some("root")
        );
        assert_eq!(
            lineage.process(
                &mark(root, now + chrono::Duration::seconds(60)),
                ttl,
                &diagnostics,
                || None
            ),
            Some("root")
        );
        let after_ttl = now + chrono::Duration::seconds(61);
        let event = if child_start {
            scope(Uuid::now_v7(), Some(root), ScopeCategory::Start, after_ttl)
        } else {
            mark(root, after_ttl)
        };
        assert_eq!(lineage.process(&event, ttl, &diagnostics, || None), None);
        assert!(lineage.completed.is_empty());
        assert!(lineage.completed_expiry_index.is_empty());
    }
}

#[test]
fn active_resource_scope_capacity_preserves_existing_scopes_and_recovers_on_end() {
    let mut lineage = SignalResourceLineage::new();
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let ttl = Duration::from_secs(60);
    let now = Utc::now();
    let mut ids = Vec::new();
    for _ in 0..MAX_ACTIVE_RESOURCE_SCOPES {
        let id = Uuid::now_v7();
        assert_eq!(
            lineage.process(
                &scope(id, None, ScopeCategory::Start, now),
                ttl,
                &diagnostics,
                || Some("root")
            ),
            Some("root")
        );
        ids.push(id);
    }
    let much_later = now + chrono::Duration::days(1);
    // Repeated starts preserve the original identity even at capacity.
    assert_eq!(
        lineage.process(
            &scope(ids[0], None, ScopeCategory::Start, much_later),
            ttl,
            &diagnostics,
            || panic!("must reuse tracked route")
        ),
        Some("root")
    );
    let rejected = Uuid::now_v7();
    for id in [rejected, Uuid::now_v7(), Uuid::now_v7()] {
        assert_eq!(
            lineage.process(
                &scope(id, Some(ids[0]), ScopeCategory::Start, much_later),
                ttl,
                &diagnostics,
                || panic!("must not create a provider at capacity")
            ),
            None
        );
    }
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES);
    assert_eq!(
        diagnostics
            .snapshot()
            .get("otel.resource_metadata_active_scope_limit")
            .unwrap()
            .count,
        3
    );
    assert_eq!(
        lineage.process(&mark(ids[0], much_later), ttl, &diagnostics, || None),
        Some("root")
    );
    assert_eq!(
        lineage.process(
            &scope(ids[0], None, ScopeCategory::End, much_later),
            ttl,
            &diagnostics,
            || None
        ),
        Some("root")
    );
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES - 1);
    // An untracked scope stays on the base resource even after capacity returns.
    assert_eq!(
        lineage.process(&mark(rejected, much_later), ttl, &diagnostics, || None),
        None
    );
    assert_eq!(
        lineage.process(
            &scope(
                Uuid::now_v7(),
                Some(ids[1]),
                ScopeCategory::Start,
                much_later
            ),
            ttl,
            &diagnostics,
            || panic!("must inherit parent")
        ),
        Some("root")
    );
    assert_eq!(lineage.active.len(), MAX_ACTIVE_RESOURCE_SCOPES);
}
