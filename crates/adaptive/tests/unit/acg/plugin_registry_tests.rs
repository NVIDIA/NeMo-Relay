// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin registry in the NeMo Relay adaptive crate.

use std::sync::Arc;

use crate::acg::error::AcgError;
use crate::acg::plugin::{PluginInput, PluginOutput, ProviderPlugin};
use crate::acg::types::{ReasonCode, TranslationReport};

use super::PluginRegistry;

/// A minimal mock plugin for testing.
struct MockPlugin {
    id: String,
    name: String,
}

impl MockPlugin {
    fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
        }
    }
}

impl ProviderPlugin for MockPlugin {
    fn plugin_id(&self) -> &str {
        &self.id
    }

    fn plugin_name(&self) -> &str {
        &self.name
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput> {
        Ok(PluginOutput {
            translated_request: input.rewritten_request.clone(),
            translation_report: TranslationReport::all_ignored(
                input.intent_bundle,
                self.plugin_id(),
                ReasonCode::NotRelevant,
                None,
            ),
        })
    }

    fn capabilities(&self) -> crate::acg::capability::BackendCapabilities {
        crate::acg::capability::BackendCapabilities::none(&self.id)
    }
}

fn assert_send_sync<T: Send + Sync>() {}

// -------------------------------------------------------------------
// new() creates an empty registry
// -------------------------------------------------------------------

#[test]
fn test_new_creates_empty_registry() {
    let registry = PluginRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

// -------------------------------------------------------------------
// register() stores a plugin and get() retrieves it
// -------------------------------------------------------------------

#[test]
fn test_register_and_get() {
    let mut registry = PluginRegistry::new();
    let plugin: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("test", "Test Plugin"));
    registry.register(plugin).unwrap();

    let retrieved = registry.get("test");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().plugin_id(), "test");
}

// -------------------------------------------------------------------
// register() with duplicate ID returns error
// -------------------------------------------------------------------

#[test]
fn test_register_duplicate_returns_error() {
    let mut registry = PluginRegistry::new();
    let p1: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("dupe", "Plugin 1"));
    let p2: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("dupe", "Plugin 2"));

    registry.register(p1).unwrap();
    let err = registry.register(p2).unwrap_err();

    match err {
        AcgError::PluginAlreadyRegistered(id) => assert_eq!(id, "dupe"),
        other => panic!("expected PluginAlreadyRegistered, got: {other:?}"),
    }
}

// -------------------------------------------------------------------
// get() with unknown ID returns None
// -------------------------------------------------------------------

#[test]
fn test_get_unknown_returns_none() {
    let registry = PluginRegistry::new();
    assert!(registry.get("nonexistent").is_none());
}

// -------------------------------------------------------------------
// list_plugin_ids() returns all registered plugin IDs (sorted)
// -------------------------------------------------------------------

#[test]
fn test_list_plugin_ids() {
    let mut registry = PluginRegistry::new();
    let p1: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("beta", "Beta Plugin"));
    let p2: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("alpha", "Alpha Plugin"));
    let p3: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("gamma", "Gamma Plugin"));

    registry.register(p1).unwrap();
    registry.register(p2).unwrap();
    registry.register(p3).unwrap();

    let ids = registry.list_plugin_ids();
    assert_eq!(ids, vec!["alpha", "beta", "gamma"]);
}

// -------------------------------------------------------------------
// deregister() removes a plugin and returns true
// -------------------------------------------------------------------

#[test]
fn test_deregister_removes_plugin() {
    let mut registry = PluginRegistry::new();
    let plugin: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("remove-me", "Remove Me"));
    registry.register(plugin).unwrap();

    assert!(registry.deregister("remove-me"));
    assert!(registry.get("remove-me").is_none());
    assert!(registry.is_empty());
}

// -------------------------------------------------------------------
// deregister() with unknown ID returns false
// -------------------------------------------------------------------

#[test]
fn test_deregister_unknown_returns_false() {
    let mut registry = PluginRegistry::new();
    assert!(!registry.deregister("nonexistent"));
}

// -------------------------------------------------------------------
// PluginRegistry is Send + Sync
// -------------------------------------------------------------------

#[test]
fn test_plugin_registry_is_send_sync() {
    assert_send_sync::<PluginRegistry>();
}

// -------------------------------------------------------------------
// Default trait creates empty registry
// -------------------------------------------------------------------

#[test]
fn test_default_creates_empty_registry() {
    let registry = PluginRegistry::default();
    assert!(registry.is_empty());
}

// -------------------------------------------------------------------
// Debug impl shows plugin IDs
// -------------------------------------------------------------------

#[test]
fn test_debug_shows_plugin_ids() {
    let mut registry = PluginRegistry::new();
    let p: Arc<dyn ProviderPlugin> = Arc::new(MockPlugin::new("debug-test", "Debug Test"));
    registry.register(p).unwrap();

    let debug = format!("{registry:?}");
    assert!(debug.contains("PluginRegistry"));
    assert!(debug.contains("debug-test"));
}
