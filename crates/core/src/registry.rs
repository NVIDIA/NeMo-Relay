// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority-sorted named registry.
//!
//! [`SortedRegistry`] is the backbone data structure for all guardrail and
//! intercept registries in the NeMo Relay runtime. It stores self-describing
//! entries by unique name and provides iteration in ascending priority order,
//! with eager re-sorting on every mutation.

use std::collections::HashMap;

/// A named, priority-ordered registry entry.
///
/// Registry entries carry their own identity so snapshots can be cloned or
/// copied without pairing a separately stored map key with the entry value.
pub(crate) trait RegistryEntry {
    /// Unique entry name within one registry.
    fn name(&self) -> &str;

    /// Entry priority. Lower values run earlier.
    fn priority(&self) -> i32;
}

/// A named registry that maintains a sorted order by priority.
///
/// Items are stored by their embedded [`RegistryEntry::name`] value and sorted
/// by [`RegistryEntry::priority`]. The sort is performed eagerly: every
/// [`register`](SortedRegistry::register) or
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
pub(crate) struct SortedRegistry<T: RegistryEntry> {
    entries: HashMap<String, T>,
    sorted_keys: Vec<String>,
}

impl<T: RegistryEntry> SortedRegistry<T> {
    /// Create a new empty registry.
    ///
    /// # Returns
    /// A new empty [`SortedRegistry`] with no entries.
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            sorted_keys: Vec::new(),
        }
    }

    /// Re-sorts the cached key order by priority. Called eagerly on every mutation.
    fn resort(&mut self) {
        let entries = &self.entries;
        let mut keys: Vec<String> = entries.keys().cloned().collect();
        keys.sort_by(|left, right| {
            let left_entry = entries.get(left).unwrap();
            let right_entry = entries.get(right).unwrap();
            left_entry
                .priority()
                .cmp(&right_entry.priority())
                .then_with(|| left.cmp(right))
        });
        self.sorted_keys = keys;
    }

    /// Register a new entry under its embedded name.
    ///
    /// # Parameters
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
    pub(crate) fn register(&mut self, entry: T) -> Result<(), String> {
        let name = entry.name().to_string();
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
    pub(crate) fn deregister(&mut self, name: &str) -> bool {
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
    pub(crate) fn sorted_values(&self) -> Vec<&T> {
        self.sorted_keys
            .iter()
            .filter_map(|k| self.entries.get(k))
            .collect()
    }
}

impl<T: RegistryEntry> Default for SortedRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../tests/unit/registry_tests.rs"]
mod tests;
