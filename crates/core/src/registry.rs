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
    /// Create a new empty registry with the given priority extraction function.
    ///
    /// The runtime calls `priority_fn` on each stored entry to determine its
    /// sort key. Lower values are ordered first.
    ///
    /// # Parameters
    /// - `priority_fn`: Function used to extract the integer priority from a
    ///   stored entry.
    ///
    /// # Returns
    /// A new empty [`SortedRegistry`] with no entries.
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

    /// Register a new entry under a unique name.
    ///
    /// # Parameters
    /// - `name`: Unique name used to address the entry later.
    /// - `entry`: Value to store in the registry.
    ///
    /// # Returns
    /// `Ok(())` when the entry was inserted.
    ///
    /// # Errors
    /// Returns `Err(String)` when `name` is already present in the registry.
    ///
    /// # Notes
    /// Successful registration eagerly re-sorts the cached priority order.
    pub fn register(&mut self, name: String, entry: T) -> Result<(), String> {
        if self.entries.contains_key(&name) {
            return Err(format!("{name} already exists"));
        }
        self.entries.insert(name, entry);
        self.resort();
        Ok(())
    }

    /// Deregister an entry by name.
    ///
    /// # Parameters
    /// - `name`: Name of the entry to remove.
    ///
    /// # Returns
    /// `true` when an entry was removed and `false` when `name` was not
    /// present.
    ///
    /// # Notes
    /// Successful removal eagerly re-sorts the cached priority order.
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
    ///
    /// # Returns
    /// A newly allocated [`Vec`] of shared references ordered from lowest
    /// priority to highest priority.
    pub fn sorted_values(&self) -> Vec<&T> {
        self.sorted_keys
            .iter()
            .filter_map(|k| self.entries.get(k))
            .collect()
    }

    /// Return named entries sorted by priority (ascending).
    ///
    /// # Returns
    /// A newly allocated [`Vec`] of `(name, entry)` pairs ordered from lowest
    /// priority to highest priority.
    pub(crate) fn sorted_entries(&self) -> Vec<(&str, &T)> {
        self.sorted_keys
            .iter()
            .filter_map(|key| self.entries.get(key).map(|entry| (key.as_str(), entry)))
            .collect()
    }

    /// Return a shared reference to an entry by name.
    ///
    /// # Parameters
    /// - `name`: Name of the entry to resolve.
    ///
    /// # Returns
    /// `Some(&T)` when an entry exists under `name`, otherwise `None`.
    pub fn get(&self, name: &str) -> Option<&T> {
        self.entries.get(name)
    }

    /// Remove and return an entry by name.
    ///
    /// # Parameters
    /// - `name`: Name of the entry to remove.
    ///
    /// # Returns
    /// `Some(T)` when an entry was removed, otherwise `None`.
    ///
    /// # Notes
    /// Successful removal eagerly re-sorts the cached priority order.
    pub fn remove(&mut self, name: &str) -> Option<T> {
        let removed = self.entries.remove(name);
        if removed.is_some() {
            self.resort();
        }
        removed
    }

    /// Report whether an entry with the given name exists.
    ///
    /// # Parameters
    /// - `name`: Name to test for membership.
    ///
    /// # Returns
    /// `true` when the registry contains `name`, otherwise `false`.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

#[cfg(test)]
#[path = "../tests/unit/registry_tests.rs"]
mod tests;
