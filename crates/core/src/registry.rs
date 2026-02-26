// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority-sorted named registry.
//!
//! [`SortedRegistry`] is the backbone data structure for all guardrail and intercept
//! registries in the NVAgentRT runtime. It stores entries by unique name and provides
//! iteration in ascending priority order, with lazy re-sorting when the registry is
//! mutated.

use std::collections::HashMap;

/// A named registry that maintains a cached sort order by priority.
///
/// Items are stored by unique string name and sorted by an integer priority
/// extracted via a caller-provided function. The sort is performed lazily:
/// modifications mark the registry as dirty, and the next call to
/// [`sorted_values`](SortedRegistry::sorted_values) re-sorts only if needed.
///
/// # Priority ordering
///
/// Entries are sorted in **ascending** priority order (lower numbers run first).
/// This means a guardrail with priority `1` executes before one with priority `10`.
///
/// # Uniqueness
///
/// Names must be unique within a registry. Attempting to [`register`](SortedRegistry::register)
/// a duplicate name returns an error. Use [`deregister`](SortedRegistry::deregister) first
/// to remove an existing entry before re-registering.
pub struct SortedRegistry<T> {
    entries: HashMap<String, T>,
    sorted_keys: Vec<String>,
    dirty: bool,
    priority_fn: fn(&T) -> i32,
}

impl<T> SortedRegistry<T> {
    /// Creates a new empty registry with the given priority extraction function.
    ///
    /// The `priority_fn` is called on each entry to determine its sort key.
    /// Lower values are sorted first (ascending order).
    pub fn new(priority_fn: fn(&T) -> i32) -> Self {
        Self {
            entries: HashMap::new(),
            sorted_keys: Vec::new(),
            dirty: false,
            priority_fn,
        }
    }

    /// Register a new entry. Returns Err if the name already exists.
    pub fn register(&mut self, name: String, entry: T) -> Result<(), String> {
        if self.entries.contains_key(&name) {
            return Err(format!("{name} already exists"));
        }
        self.entries.insert(name, entry);
        self.dirty = true;
        Ok(())
    }

    /// Deregister an entry by name. Returns true if it existed.
    pub fn deregister(&mut self, name: &str) -> bool {
        if self.entries.remove(name).is_some() {
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Return entries sorted by priority (ascending).
    pub fn sorted_values(&mut self) -> Vec<&T> {
        if self.dirty {
            let pf = self.priority_fn;
            let entries = &self.entries;
            let mut keys: Vec<String> = entries.keys().cloned().collect();
            keys.sort_by_key(|k| pf(entries.get(k).unwrap()));
            self.sorted_keys = keys;
            self.dirty = false;
        }
        self.sorted_keys
            .iter()
            .filter_map(|k| self.entries.get(k))
            .collect()
    }

    /// Returns `true` if an entry with the given name exists in the registry.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
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
        assert!(reg
            .register(
                "a".into(),
                PriorityItem {
                    priority: 5,
                    value: "A2".into(),
                },
            )
            .is_err());

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
        let mut reg = SortedRegistry::new(|item: &PriorityItem| item.priority);
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
        // First call sorts
        let s1: Vec<&str> = reg
            .sorted_values()
            .iter()
            .map(|i| i.value.as_str())
            .collect();
        assert_eq!(s1, vec!["A"]);
        // Second call uses cache (same result)
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
}
