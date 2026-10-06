// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded tombstones for completion retries and late starts, using source identities only.
use crate::events::{AgentKind, NormalizedEvent};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const CAPACITY: usize = 4_096;
const RETENTION: Duration = Duration::from_secs(300);

#[derive(Clone, PartialEq, Eq, Hash)]
/// Scope source identifiers to their authenticated owner, session, harness, and lifecycle.
pub(super) struct CompletionKey {
    owner: String,
    session: String,
    kind: &'static str,
    agent_kind: AgentKind,
    invocation: String,
}

#[derive(Default)]
pub(super) struct CompletionCache {
    entries: HashMap<CompletionKey, Instant>,
    order: VecDeque<(CompletionKey, Instant)>,
}

impl CompletionCache {
    fn expire(&mut self, now: Instant) {
        while self
            .order
            .front()
            .is_some_and(|(_, at)| now.saturating_duration_since(*at) >= RETENTION)
        {
            let (key, at) = self.order.pop_front().expect("front exists");
            if self.entries.get(&key) == Some(&at) {
                self.entries.remove(&key);
            }
        }
    }
    pub(super) fn contains(&mut self, key: &CompletionKey) -> bool {
        self.expire(Instant::now());
        self.entries.contains_key(key)
    }
    /// Record a completion once so retries do not extend its retention period.
    pub(super) fn record(&mut self, key: CompletionKey) {
        let now = Instant::now();
        self.expire(now);
        if self.entries.contains_key(&key) {
            return;
        }
        while self.entries.len() >= CAPACITY {
            if let Some((old, at)) = self.order.pop_front()
                && self.entries.get(&old) == Some(&at)
            {
                self.entries.remove(&old);
            }
        }
        self.entries.insert(key.clone(), now);
        self.order.push_back((key, now));
    }
}

/// Correlate retries and late starts using stable source IDs, excluding generated IDs.
pub(super) fn completion_key(
    event: &NormalizedEvent,
    owner: Option<&str>,
) -> Option<CompletionKey> {
    use NormalizedEvent::*;
    let (kind, agent_kind, invocation) = match event {
        ToolStarted(event) | ToolEnded(event) => {
            if event
                .metadata
                .get("tool_call_id_generated")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                || event.tool_call_id.is_empty()
            {
                return None;
            }
            ("tool", event.agent_kind, event.tool_call_id.clone())
        }
        SubagentStarted(event) | SubagentEnded(event) => {
            if event
                .metadata
                .get("subagent_id_generated")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                return None;
            }
            ("subagent", event.agent_kind, event.subagent_id.clone())
        }
        AgentStarted(event) | AgentEnded(event) => (
            "agent",
            event.agent_kind,
            source_id(&event.payload, &["lifecycle_id", "agent_invocation_id"])?,
        ),
        TurnStarted(event) | TurnEnded(event) => (
            "turn",
            event.agent_kind,
            source_id(&event.payload, &["turn_id", "generation_id"])?,
        ),
        _ => return None,
    };
    if invocation.is_empty() {
        return None;
    }
    Some(CompletionKey {
        owner: owner.unwrap_or("").to_owned(),
        session: event.session_id().to_owned(),
        kind,
        agent_kind,
        invocation,
    })
}

// Child hooks can trail task completion, after the owning Session has been discarded.
// Reuse the bounded subagent completion identity to reject the whole late hook stream.
pub(super) fn completed_child_key(
    event: &NormalizedEvent,
    owner: Option<&str>,
) -> Option<CompletionKey> {
    use NormalizedEvent::*;
    let id = match event {
        PromptSubmitted(event)
        | TurnStarted(event)
        | TurnEnded(event)
        | Compaction(event)
        | Notification(event)
        | HookMark(event) => super::alignment::aliased_turn_subagent_id(event).or_else(|| {
            event
                .metadata
                .get("agent_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        }),
        LlmHint(event) => event.subagent_id.clone().or_else(|| event.agent_id.clone()),
        ToolStarted(event) | ToolEnded(event) => event.subagent_id.clone(),
        _ => None,
    }?;
    let agent_kind = super::event_agent_kind(event);
    if agent_kind != AgentKind::Codex || id == event.session_id() || id.is_empty() {
        return None;
    }
    Some(CompletionKey {
        owner: owner.unwrap_or("").to_owned(),
        session: event.session_id().to_owned(),
        kind: "subagent",
        agent_kind,
        invocation: id,
    })
}

fn source_id(payload: &serde_json::Value, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        payload
            .get(*name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

pub(super) fn is_completion(event: &NormalizedEvent) -> bool {
    matches!(
        event,
        NormalizedEvent::AgentEnded(_)
            | NormalizedEvent::TurnEnded(_)
            | NormalizedEvent::SubagentEnded(_)
            | NormalizedEvent::ToolEnded(_)
    )
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/completion_tests.rs"]
mod tests;
