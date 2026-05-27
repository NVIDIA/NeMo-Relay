// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for registry in the NeMo Relay core crate.

use super::*;

struct PriorityItem {
    name: String,
    priority: i32,
    value: String,
}

impl RegistryEntry for PriorityItem {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

fn item(name: &str, priority: i32, value: &str) -> PriorityItem {
    PriorityItem {
        name: name.into(),
        priority,
        value: value.into(),
    }
}

#[test]
fn test_sorted_registry() {
    let mut reg = SortedRegistry::new();

    reg.register(item("b", 20, "B")).unwrap();

    reg.register(item("a", 10, "A")).unwrap();

    reg.register(item("c", 30, "C")).unwrap();

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A", "B", "C"]);

    // duplicate
    assert!(reg.register(item("a", 5, "A2")).is_err());

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
    let reg = SortedRegistry::<PriorityItem>::new();
    let sorted = reg.sorted_values();
    assert!(sorted.is_empty());
}

#[test]
fn test_default_registry_is_empty() {
    let reg = SortedRegistry::<PriorityItem>::default();
    assert!(reg.sorted_values().is_empty());
}

#[test]
fn test_negative_priorities() {
    let mut reg = SortedRegistry::new();
    reg.register(item("pos", 10, "P")).unwrap();
    reg.register(item("neg", -5, "N")).unwrap();
    reg.register(item("zero", 0, "Z")).unwrap();

    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["N", "Z", "P"]);
}

#[test]
fn test_re_register_after_deregister() {
    let mut reg = SortedRegistry::new();
    reg.register(item("a", 10, "A1")).unwrap();
    reg.deregister("a");
    reg.register(item("a", 5, "A2")).unwrap();
    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["A2"]);
}

#[test]
fn test_deregister_nonexistent() {
    let mut reg = SortedRegistry::<PriorityItem>::new();
    assert!(!reg.deregister("nope"));
}

#[test]
fn test_duplicate_error_message() {
    let mut reg = SortedRegistry::new();
    reg.register(item("dup", 1, "D")).unwrap();
    let err = reg.register(item("dup", 2, "D2")).unwrap_err();
    assert!(err.contains("dup"));
    assert!(err.contains("already exists"));
}

#[test]
fn test_sorted_values_caching() {
    let mut reg = SortedRegistry::new();
    reg.register(item("a", 1, "A")).unwrap();
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
    let mut reg = SortedRegistry::new();
    for i in (0..20).rev() {
        reg.register(PriorityItem {
            name: format!("item_{i}"),
            priority: i,
            value: format!("V{i}"),
        })
        .unwrap();
    }
    let sorted: Vec<i32> = reg.sorted_values().iter().map(|i| i.priority).collect();
    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(sorted, expected);
}

#[test]
fn test_same_priority_stable() {
    let mut reg = SortedRegistry::new();
    reg.register(item("x", 1, "X")).unwrap();
    reg.register(item("y", 1, "Y")).unwrap();
    let sorted: Vec<&str> = reg
        .sorted_values()
        .iter()
        .map(|i| i.value.as_str())
        .collect();
    assert_eq!(sorted, vec!["X", "Y"]);
}
