// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct PriorityItem {
    priority: i32,
    value: String,
}

#[test]
fn test_sorted_registry() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);

    reg.register(
        "b".into(),
        PriorityItem {
            priority: 20,
            value: "B".into(),
        },
    )
    .unwrap();

    reg.register(
        "a".into(),
        PriorityItem {
            priority: 10,
            value: "A".into(),
        },
    )
    .unwrap();

    reg.register(
        "c".into(),
        PriorityItem {
            priority: 30,
            value: "C".into(),
        },
    )
    .unwrap();

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A", "B", "C"]);

    // duplicate
    assert!(
        reg.register(
            "a".into(),
            PriorityItem {
                priority: 5,
                value: "A2".into(),
            },
        )
        .is_err()
    );

    // deregister
    assert!(reg.deregister("b"));
    assert!(!reg.deregister("b"));

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A", "C"]);
}

#[test]
fn test_empty_registry() {
    let reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    let sorted = reg.sorted_values();
    assert!(sorted.is_empty());
}

#[test]
fn test_contains() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    assert!(!reg.contains("x"));
    reg.register(
        "x".into(),
        PriorityItem {
            priority: 1,
            value: "X".into(),
        },
    )
    .unwrap();
    assert!(reg.contains("x"));
    assert!(!reg.contains("y"));
    reg.deregister("x");
    assert!(!reg.contains("x"));
}

#[test]
fn test_negative_priorities() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "pos".into(),
        PriorityItem {
            priority: 10,
            value: "P".into(),
        },
    )
    .unwrap();
    reg.register(
        "neg".into(),
        PriorityItem {
            priority: -5,
            value: "N".into(),
        },
    )
    .unwrap();
    reg.register(
        "zero".into(),
        PriorityItem {
            priority: 0,
            value: "Z".into(),
        },
    )
    .unwrap();

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["N", "Z", "P"]);
}

#[test]
fn test_re_register_after_deregister() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "a".into(),
        PriorityItem {
            priority: 10,
            value: "A1".into(),
        },
    )
    .unwrap();
    reg.deregister("a");
    reg.register(
        "a".into(),
        PriorityItem {
            priority: 5,
            value: "A2".into(),
        },
    )
    .unwrap();
    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A2"]);
}

#[test]
fn test_deregister_nonexistent() {
    let mut reg = SortedRegistry::<PriorityItem>::new(|item| item.priority);
    assert!(!reg.deregister("nope"));
}

#[test]
fn test_duplicate_error_message() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "dup".into(),
        PriorityItem {
            priority: 1,
            value: "D".into(),
        },
    )
    .unwrap();
    let err = reg
        .register(
            "dup".into(),
            PriorityItem {
                priority: 2,
                value: "D2".into(),
            },
        )
        .unwrap_err();
    assert!(err.contains("dup"));
    assert!(err.contains("already exists"));
}

#[test]
fn test_sorted_values_caching() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "a".into(),
        PriorityItem {
            priority: 1,
            value: "A".into(),
        },
    )
    .unwrap();
    // Sort order is already maintained eagerly by register()
    let s1: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(s1, vec!["A"]);
    // Second call returns the same result (no re-sort needed)
    let s2: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(s2, vec!["A"]);
}

#[test]
fn test_many_entries_ordering() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    for i in (0..20).rev() {
        reg.register(
            format!("item_{i}"),
            PriorityItem {
                priority: i,
                value: format!("V{i}"),
            },
        )
        .unwrap();
    }
    let sorted: Vec<i32> = reg.sorted_values().iter().map(|i| i.priority).collect();
    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(sorted, expected);
}

#[test]
fn test_same_priority_stable() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "x".into(),
        PriorityItem {
            priority: 1,
            value: "X".into(),
        },
    )
    .unwrap();
    reg.register(
        "y".into(),
        PriorityItem {
            priority: 1,
            value: "Y".into(),
        },
    )
    .unwrap();
    // Both should be present
    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted.len(), 2);
    assert!(sorted.contains(&"X"));
    assert!(sorted.contains(&"Y"));
}

#[test]
fn test_get_and_remove_cover_lookup_and_resort_paths() {
    let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
    reg.register(
        "alpha".into(),
        PriorityItem {
            priority: 2,
            value: "A".into(),
        },
    )
    .unwrap();
    reg.register(
        "beta".into(),
        PriorityItem {
            priority: 1,
            value: "B".into(),
        },
    )
    .unwrap();

    assert_eq!(reg.get("beta").map(|item| item.value.as_str()), Some("B"));
    assert!(reg.get("missing").is_none());

    let removed = reg.remove("beta").unwrap();
    assert_eq!(removed.value, "B");
    assert!(reg.remove("beta").is_none());

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|item| item.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A"]);
}
