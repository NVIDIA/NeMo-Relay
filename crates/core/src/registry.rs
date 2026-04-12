// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority-sorted named registry.
//!
//! [`SortedRegistry`] is the backbone data structure for all guardrail and intercept
//! registries in the NeMo Flow runtime. It stores entries by unique name and provides
//! iteration in ascending priority order, with eager re-sorting on every mutation.

use std::collections::HashMap;

/// A named registry that maintains a sorted order by priority.
///
/// Items are stored by unique string name and sorted by an integer priority
/// extracted via a caller-provided function. The sort is performed eagerly:
/// every [`register`](SortedRegistry::register) or
/// [`deregister`](SortedRegistry::deregister) call re-sorts immediately, so
/// [`sorted_values`](SortedRegistry::sorted_values) is a read-only lookup.
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
            priority_fn,
        }
    }

    /// Re-sorts the cached key order by priority. Called eagerly on every mutation.
    fn resort(&mut self) {
        let pf = self.priority_fn;
        let entries = &self.entries;
        let mut keys: Vec<String> = entries.keys().cloned().collect();
        keys.sort_by_key(|k| pf(entries.get(k).unwrap()));
        self.sorted_keys = keys;
    }

    /// Register a new entry. Returns Err if the name already exists.
    pub fn register(&mut self, name: String, entry: T) -> Result<(), String> {
        if self.entries.contains_key(&name) {
            return Err(format!("{name} already exists"));
        }
        self.entries.insert(name, entry);
        self.resort();
        Ok(())
    }

    /// Deregister an entry by name. Returns true if it existed.
    pub fn deregister(&mut self, name: &str) -> bool {
        if self.entries.remove(name).is_some() {
            self.resort();
            true
        } else {
            false
        }
    }

    /// Return entries sorted by priority (ascending).
    ///
    /// This is a read-only operation — the sort order is maintained eagerly
    /// on every [`register`](SortedRegistry::register) / [`deregister`](SortedRegistry::deregister) call.
    pub fn sorted_values(&self) -> Vec<&T> {
        self.sorted_keys
            .iter()
            .filter_map(|k| self.entries.get(k))
            .collect()
    }

    /// Returns a shared reference to an entry by name.
    pub fn get(&self, name: &str) -> Option<&T> {
        self.entries.get(name)
    }

    /// Remove and return an entry by name.
    pub fn remove(&mut self, name: &str) -> Option<T> {
        let removed = self.entries.remove(name);
        if removed.is_some() {
            self.resort();
        }
        removed
    }

    /// Returns `true` if an entry with the given name exists in the registry.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

#[cfg(test)]
#[path = "../tests/unit/registry_tests.rs"]
mod tests;
