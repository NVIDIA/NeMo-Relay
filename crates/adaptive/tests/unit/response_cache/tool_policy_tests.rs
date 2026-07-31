// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused policy-resolution tests for the tool-result response cache.

use super::*;
use crate::response_cache::config::ToolClass;
use std::collections::BTreeMap;

fn response_cache() -> ResponseCacheConfig {
    ResponseCacheConfig {
        ttl_seconds: 3600,
        bypass_rate: 0.0,
        ..ResponseCacheConfig::default()
    }
}

fn class(cacheable: bool, members: &[&str]) -> ToolClass {
    ToolClass {
        cacheable,
        members: members.iter().map(|member| member.to_string()).collect(),
        ..ToolClass::default()
    }
}

#[test]
fn policy_resolution_inherits_class_values_and_honors_an_override() {
    let mut classes = BTreeMap::new();
    classes.insert(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            ttl_seconds: Some(300),
            bypass_rate: Some(0.2),
            arg_skip: vec!["request_id".to_string()],
            members: vec!["docs_*".to_string()],
        },
    );
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_lookup".to_string(),
        ToolOverride {
            cacheable: Some(false),
            arg_skip: Some(vec![]),
            tool_version: Some("v2".to_string()),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };

    let unclassified = resolve_policy("send_email", &response_cache(), &tools);
    assert!(!unclassified.cacheable);
    assert_eq!(unclassified.ttl, Duration::from_secs(3600));

    let class_only = resolve_policy("docs_search", &response_cache(), &tools);
    assert!(class_only.cacheable);
    assert_eq!(class_only.ttl, Duration::from_secs(300));
    assert_eq!(class_only.bypass_rate, 0.2);
    assert_eq!(class_only.arg_skip, ["request_id"]);

    let overridden = resolve_policy("docs_lookup", &response_cache(), &tools);
    assert!(!overridden.cacheable);
    assert_eq!(overridden.ttl, Duration::from_secs(300));
    assert_eq!(overridden.bypass_rate, 0.2);
    assert!(overridden.arg_skip.is_empty());
    assert_eq!(overridden.tool_version.as_deref(), Some("v2"));
}

#[test]
fn exact_and_specific_pattern_rules_choose_one_policy() {
    let mut classes = BTreeMap::new();
    classes.insert(
        "catch_all".to_string(),
        ToolClass {
            ttl_seconds: Some(100),
            members: vec!["*".to_string()],
            ..class(true, &[])
        },
    );
    classes.insert(
        "docs".to_string(),
        ToolClass {
            ttl_seconds: Some(60),
            members: vec!["docs_*".to_string()],
            ..class(true, &[])
        },
    );
    classes.insert(
        "private".to_string(),
        ToolClass {
            ttl_seconds: Some(10),
            members: vec!["docs_private".to_string()],
            ..class(true, &[])
        },
    );
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_*".to_string(),
        ToolOverride {
            ttl_seconds: Some(20),
            ..ToolOverride::default()
        },
    );
    overrides.insert(
        "docs_private".to_string(),
        ToolOverride {
            ttl_seconds: Some(5),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };

    assert_eq!(
        resolve_policy("docs_private", &response_cache(), &tools).ttl,
        Duration::from_secs(5),
        "exact class and override entries win"
    );
    assert_eq!(
        resolve_policy("docs_search", &response_cache(), &tools).ttl,
        Duration::from_secs(20),
        "the more-specific wildcard class and override win"
    );
    assert_eq!(
        resolve_policy("other", &response_cache(), &tools).ttl,
        Duration::from_secs(100)
    );
}

#[test]
fn wildcard_matching_and_overlap_cover_edge_cases() {
    for (pattern, name, expected) in [
        ("*", "", true),
        ("docs_*", "docs_lookup", true),
        ("docs_*", "doc_lookup", false),
        ("get_*_price", "get_stock_price", true),
        ("get_*_price", "get_price", false),
        ("a*a", "a", false),
        ("a*a", "aba", true),
        ("Docs_*", "docs_lookup", false),
    ] {
        assert_eq!(
            wildcard_match(pattern, name),
            expected,
            "{pattern:?}, {name:?}"
        );
    }
    assert!(wildcard_patterns_overlap("*_email", "send_*"));
    assert!(!wildcard_patterns_overlap("docs_*", "send_*"));
    assert!(wildcard_patterns_overlap("é*", "*é"));
    assert_eq!(wildcard_rank("*é*").0, 1);
}
