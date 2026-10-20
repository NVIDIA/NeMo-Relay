// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority-sorted named registry.
//!
//! [`SortedRegistry`] is the backbone data structure for all guardrail and
//! intercept registries in the NeMo Relay runtime. It stores self-describing
//! entries by unique name and provides iteration in ascending priority order,
//! with eager re-sorting on every mutation.

use std::collections::HashMap;
use std::sync::Arc;

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
/// Clones share immutable storage. A successful mutation copies shared storage,
/// leaving snapshots unchanged; failed mutations keep sharing the original storage.
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
#[derive(Clone)]
pub(crate) struct SortedRegistry<T: RegistryEntry> {
    // Empty registries remain allocation-free. Snapshots share populated storage until mutation.
    storage: Option<Arc<RegistryStorage<T>>>,
}

#[derive(Clone)]
struct RegistryStorage<T: RegistryEntry> {
    entries: HashMap<String, T>,
    sorted_keys: Vec<String>,
}

impl<T: RegistryEntry> SortedRegistry<T> {
    /// Create a new empty registry.
    ///
    /// # Returns
    /// A new empty [`SortedRegistry`] with no entries.
    pub(crate) fn new() -> Self {
        Self { storage: None }
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
    pub(crate) fn register(&mut self, entry: T) -> Result<(), String>
    where
        T: Clone,
    {
        let name = entry.name();
        if self
            .storage
            .as_ref()
            .is_some_and(|storage| storage.entries.contains_key(name))
        {
            return Err(format!("{name} already exists"));
        }
        let name = name.to_string();
        let storage = self.storage.get_or_insert_with(|| {
            Arc::new(RegistryStorage {
                entries: HashMap::new(),
                sorted_keys: Vec::new(),
            })
        });
        let storage = Arc::make_mut(storage);
        storage.entries.insert(name, entry);
        storage.resort();
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
    pub(crate) fn deregister(&mut self, name: &str) -> bool
    where
        T: Clone,
    {
        let Some(storage) = self.storage.as_mut() else {
            return false;
        };
        if !storage.entries.contains_key(name) {
            return false;
        }
        let storage = Arc::make_mut(storage);
        storage.entries.remove(name);
        storage.resort();
        true
    }

    /// Return entries sorted by priority (ascending).
    ///
    /// This is a read-only operation — the sort order is maintained eagerly
    /// on every [`register`](SortedRegistry::register) / [`deregister`](SortedRegistry::deregister) call.
    ///
    /// # Returns
    /// Shared references ordered from lowest priority to highest priority,
    /// without allocating an intermediate collection.
    pub(crate) fn sorted_values(&self) -> impl Iterator<Item = &T> {
        self.storage.iter().flat_map(|storage| {
            storage
                .sorted_keys
                .iter()
                .filter_map(|key| storage.entries.get(key))
        })
    }

    /// Whether this registry contains no entries, without scanning retained storage.
    pub(crate) fn is_empty(&self) -> bool {
        self.storage
            .as_ref()
            .is_none_or(|storage| storage.entries.is_empty())
    }

    /// Return every entry without imposing priority order.
    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.storage
            .iter()
            .flat_map(|storage| storage.entries.values())
    }
}

impl<T: RegistryEntry> RegistryStorage<T> {
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
}

impl<T: RegistryEntry> Default for SortedRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../tests/unit/registry_tests.rs"]
mod tests;
