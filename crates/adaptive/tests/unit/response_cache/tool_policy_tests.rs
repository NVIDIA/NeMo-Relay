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
            tool_version: Some("class-v1".to_string()),
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
            tool_version: Some("tool-v2".to_string()),
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
    assert_eq!(class_only.tool_version.as_deref(), Some("class-v1"));
    assert_eq!(class_only.arg_skip, ["request_id"]);

    let overridden = resolve_policy("docs_lookup", &response_cache(), &tools);
    assert!(!overridden.cacheable);
    assert_eq!(overridden.ttl, Duration::from_secs(300));
    assert_eq!(overridden.bypass_rate, 0.2);
    assert!(overridden.arg_skip.is_empty());
    assert_eq!(overridden.tool_version.as_deref(), Some("tool-v2"));
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
        ("*_price", "stock_price", true),
        ("*stock*", "get_stock_price", true),
        ("get_*_price", "get_stock_price", false),
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
    for pattern in ["docs_lookup", "docs_*", "*_lookup", "*docs*", "*"] {
        assert!(is_supported_tool_pattern(pattern), "{pattern:?}");
    }
    for pattern in ["**", "docs_*_lookup", "docs**", "*docs**"] {
        assert!(!is_supported_tool_pattern(pattern), "{pattern:?}");
    }
}

// The rest of the policy cases stay in the crate test tree so they retain
// private-module access without putting tests under src.

fn policy_response_cache(ttl_seconds: u64, bypass_rate: f64) -> ResponseCacheConfig {
    ResponseCacheConfig {
        ttl_seconds,
        bypass_rate,
        ..ResponseCacheConfig::default()
    }
}

#[test]
fn unclassified_tool_falls_into_the_default_bucket_uncached() {
    let tools = ToolCacheConfig::default();
    let policy = resolve_policy("anything", &policy_response_cache(3600, 0.0), &tools);
    assert!(
        !policy.cacheable,
        "an unknown tool must default to not cached"
    );
}

#[test]
fn class_membership_makes_a_tool_cacheable() {
    let mut classes = BTreeMap::new();
    classes.insert("read_only".to_string(), class(true, &["docs_lookup"]));
    classes.insert(
        "volatile".to_string(),
        ToolClass {
            cacheable: true,
            ttl_seconds: Some(300),
            bypass_rate: Some(0.2),
            members: vec!["get_weather".to_string()],
            ..ToolClass::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    let policy = resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools);
    assert!(policy.cacheable);
    assert_eq!(policy.ttl, Duration::from_secs(3600));
    assert_eq!(policy.bypass_rate, 0.0);
    let policy = resolve_policy("get_weather", &policy_response_cache(3600, 0.0), &tools);
    assert_eq!(policy.ttl, Duration::from_secs(300));
    assert_eq!(policy.bypass_rate, 0.2);
}

#[test]
fn per_tool_override_wins_over_its_class() {
    let mut classes = BTreeMap::new();
    classes.insert(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            arg_skip: vec!["request_id".to_string()],
            members: vec!["docs_lookup".to_string()],
            ..ToolClass::default()
        },
    );
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_lookup".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ttl_seconds: Some(30),
            bypass_rate: Some(0.25),
            tool_version: Some("v2".to_string()),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };
    let policy = resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools);
    assert!(!policy.cacheable, "override cacheable=false must win");
    assert_eq!(policy.ttl, Duration::from_secs(30));
    assert_eq!(policy.bypass_rate, 0.25);
    assert_eq!(policy.tool_version.as_deref(), Some("v2"));
    assert_eq!(policy.arg_skip, vec!["request_id".to_string()]);
}

#[test]
fn override_arg_skip_replaces_the_class_list() {
    let mut classes = BTreeMap::new();
    classes.insert(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            arg_skip: vec!["session_id".to_string()],
            members: vec!["lookup".to_string()],
            ..ToolClass::default()
        },
    );
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "lookup".to_string(),
        ToolOverride {
            arg_skip: Some(vec![]),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };
    let policy = resolve_policy("lookup", &policy_response_cache(3600, 0.0), &tools);
    assert!(
        policy.arg_skip.is_empty(),
        "an override arg_skip (even empty) replaces the class list"
    );
}

#[test]
fn wildcard_match_table() {
    let cases = [
        ("*", "", true),
        ("*", "anything", true),
        ("docs_*", "docs_lookup", true),
        ("docs_*", "docs_", true),
        ("docs_*", "doc_lookup", false),
        ("*_price", "stock_price", true),
        ("*_price", "price", false),
        ("*stock*", "get_stock_price", true),
        ("*stock*", "get_price", false),
        ("get_*_price", "get_stock_price", false),
        ("a*a", "aba", false),
        ("Docs_*", "docs_lookup", false), // case-sensitive
        ("abc*", "abc*", true),           // no escaping: '*' matches itself via the span
    ];
    for (pattern, name, expected) in cases {
        assert_eq!(
            wildcard_match(pattern, name),
            expected,
            "wildcard_match({pattern:?}, {name:?})"
        );
    }
}

#[test]
fn wildcard_overlap_table() {
    let cases = [
        ("*_email", "send_*", true),
        ("delete_*", "*_record", true),
        ("docs_*", "send_*", false),
        ("*docs*", "send_*", true),
        ("é*", "*é", true),
        ("foo*", "bar*", false),
        ("*foo", "*bar", false),
        ("a*b*c", "a*c", false),
    ];
    for (left, right, expected) in cases {
        assert_eq!(
            wildcard_patterns_overlap(left, right),
            expected,
            "wildcard_patterns_overlap({left:?}, {right:?})"
        );
    }
}

#[test]
fn wildcard_rank_counts_unicode_characters_not_utf8_bytes() {
    assert_eq!(wildcard_rank("*é*").0, 1);
    assert_eq!(wildcard_rank("*éé*").0, 2);
    assert_eq!(wildcard_rank("*💡*").0, 1);
}

#[test]
fn wildcard_member_classifies_a_matching_tool() {
    let mut classes = BTreeMap::new();
    classes.insert("read_only".to_string(), class(true, &["docs_*"]));
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    assert!(resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools).cacheable);
    assert!(
        !resolve_policy("send_email", &policy_response_cache(3600, 0.0), &tools).cacheable,
        "a non-matching tool still falls through to default"
    );
}

#[test]
fn exact_member_beats_any_wildcard_match() {
    let mut classes = BTreeMap::new();
    classes.insert("a_wildcards".to_string(), class(true, &["docs_*"]));
    classes.insert("b_exact".to_string(), class(false, &["docs_lookup"]));
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    let policy = resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools);
    assert!(
        !policy.cacheable,
        "the exact member's class must win over a matching wildcard"
    );
}

#[test]
fn most_specific_wildcard_wins() {
    let mut classes = BTreeMap::new();
    classes.insert("a_catch_all".to_string(), class(false, &["*"]));
    classes.insert("b_docs".to_string(), class(true, &["docs_*"]));
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    assert!(
        resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools).cacheable,
        "the pattern with more literal characters must win"
    );
    assert!(!resolve_policy("send_email", &policy_response_cache(3600, 0.0), &tools).cacheable);
}

#[test]
fn equal_literals_fewer_stars_then_smaller_pattern_break_ties() {
    let mut classes = BTreeMap::new();
    classes.insert("two_stars".to_string(), class(false, &["*ab*"]));
    classes.insert("one_star".to_string(), class(true, &["ab*"]));
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    assert!(
        resolve_policy("ab", &policy_response_cache(3600, 0.0), &tools).cacheable,
        "with equal literal counts the pattern with fewer stars must win"
    );

    let mut classes = BTreeMap::new();
    classes.insert("suffix".to_string(), class(false, &["*x"]));
    classes.insert("prefix".to_string(), class(true, &["x*"]));
    let tools = ToolCacheConfig {
        classes,
        ..ToolCacheConfig::default()
    };
    assert!(
        !resolve_policy("x", &policy_response_cache(3600, 0.0), &tools).cacheable,
        "'*x' sorts before 'x*', so the suffix class must win the tie"
    );
}

#[test]
fn override_patterns_apply_with_exact_keys_winning() {
    let mut classes = BTreeMap::new();
    classes.insert("read_only".to_string(), class(true, &["docs_*"]));
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_secret_*".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ..ToolOverride::default()
        },
    );
    overrides.insert(
        "docs_secret_audit".to_string(),
        ToolOverride {
            cacheable: Some(true),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };
    let cacheable =
        |name: &str| resolve_policy(name, &policy_response_cache(3600, 0.0), &tools).cacheable;
    assert!(
        !cacheable("docs_secret_dump"),
        "a pattern override must apply to the tools it matches"
    );
    assert!(
        cacheable("docs_secret_audit"),
        "an exact override key must win over a matching pattern"
    );
    assert!(
        cacheable("docs_lookup"),
        "tools no override matches keep their class policy"
    );
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_*".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ..ToolOverride::default()
        },
    );
    let mut classes = BTreeMap::new();
    classes.insert("read_only".to_string(), class(true, &["docs_*"]));
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };
    assert!(
        !resolve_policy("docs_*", &policy_response_cache(3600, 0.0), &tools).cacheable,
        "the literal name `docs_*` resolves its exact entry"
    );
    assert!(!resolve_policy("docs_lookup", &policy_response_cache(3600, 0.0), &tools).cacheable);
}

#[test]
fn most_specific_override_pattern_wins() {
    let mut classes = BTreeMap::new();
    classes.insert("read_only".to_string(), class(true, &["docs_*"]));
    let mut overrides = BTreeMap::new();
    overrides.insert(
        "docs_*".to_string(),
        ToolOverride {
            cacheable: Some(true),
            ..ToolOverride::default()
        },
    );
    overrides.insert(
        "docs_secret_*".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ..ToolOverride::default()
        },
    );
    let tools = ToolCacheConfig {
        classes,
        overrides,
        ..ToolCacheConfig::default()
    };
    assert!(
        !resolve_policy(
            "docs_secret_dump",
            &policy_response_cache(3600, 0.0),
            &tools
        )
        .cacheable,
        "`docs_secret_*` (more literal characters) must beat `docs_*`"
    );
}
