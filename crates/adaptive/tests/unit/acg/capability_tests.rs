// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for capability in the NeMo Relay adaptive crate.

use super::*;

fn assert_send_sync<T: Send + Sync>() {}

// -------------------------------------------------------------------
// ProviderFeature serde round-trip for all variants
// -------------------------------------------------------------------

#[test]
fn test_provider_feature_serde_round_trip() {
    let features = [
        ProviderFeature::ExplicitCacheBreakpoints,
        ProviderFeature::AutomaticPrefixCaching,
        ProviderFeature::RetentionTiers,
        ProviderFeature::PriorityScheduling,
        ProviderFeature::ModelRouting,
        ProviderFeature::DeferredToolLoading,
        ProviderFeature::FileReferences,
        ProviderFeature::StructuredOutput,
        ProviderFeature::PrefixAffinityHints,
        ProviderFeature::StreamingTokenCounts,
    ];

    for feature in &features {
        let json = serde_json::to_string(feature).unwrap();
        let back: ProviderFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(back, *feature, "round-trip failed for {feature:?}");
    }
}

// -------------------------------------------------------------------
// ProviderFeature is Copy, Eq, Hash
// -------------------------------------------------------------------

#[test]
fn test_provider_feature_is_copy_eq_hash() {
    let f = ProviderFeature::ExplicitCacheBreakpoints;
    let f2 = f; // Copy
    assert_eq!(f, f2); // Eq

    // Hash: can be used in HashSet
    let mut set = HashSet::new();
    set.insert(f);
    assert!(set.contains(&f2));
}

// -------------------------------------------------------------------
// BackendCapabilities::none() creates empty feature set
// -------------------------------------------------------------------

#[test]
fn test_backend_capabilities_none() {
    let caps = BackendCapabilities::none("passthrough");
    assert_eq!(caps.backend_id, "passthrough");
    assert!(caps.supported_features.is_empty());
    assert!(caps.model_families.is_empty());
}

// -------------------------------------------------------------------
// BackendCapabilities::supports() checks backend-level
// -------------------------------------------------------------------

#[test]
fn test_backend_capabilities_supports() {
    let mut features = HashSet::new();
    features.insert(ProviderFeature::ExplicitCacheBreakpoints);
    features.insert(ProviderFeature::StreamingTokenCounts);

    let caps = BackendCapabilities {
        backend_id: "test".to_string(),
        supported_features: features,
        model_families: HashMap::new(),
    };

    assert!(caps.supports(ProviderFeature::ExplicitCacheBreakpoints));
    assert!(caps.supports(ProviderFeature::StreamingTokenCounts));
    assert!(!caps.supports(ProviderFeature::AutomaticPrefixCaching));
}

// -------------------------------------------------------------------
// ModelFamilyCapabilities::supports() checks model-level
// -------------------------------------------------------------------

#[test]
fn test_model_family_supports() {
    let mut features = HashSet::new();
    features.insert(ProviderFeature::AutomaticPrefixCaching);

    let caps = ModelFamilyCapabilities {
        model_family: "gpt-4o".to_string(),
        supported_features: features,
        max_cache_breakpoints: None,
        min_cacheable_tokens: None,
        cache_economics: None,
    };

    assert!(caps.supports(ProviderFeature::AutomaticPrefixCaching));
    assert!(!caps.supports(ProviderFeature::StructuredOutput));
}

// -------------------------------------------------------------------
// BackendCapabilities::model_supports() checks model-family first,
// falls back to backend-level
// -------------------------------------------------------------------

#[test]
fn test_backend_model_supports_fallback() {
    let mut backend_features = HashSet::new();
    backend_features.insert(ProviderFeature::StreamingTokenCounts);

    let mut model_features = HashSet::new();
    model_features.insert(ProviderFeature::AutomaticPrefixCaching);

    let mut caps = BackendCapabilities {
        backend_id: "test".to_string(),
        supported_features: backend_features,
        model_families: HashMap::new(),
    };

    caps.add_model_family(ModelFamilyCapabilities {
        model_family: "special-model".to_string(),
        supported_features: model_features,
        max_cache_breakpoints: None,
        min_cacheable_tokens: None,
        cache_economics: None,
    });

    // Known model family: uses model-level features (NOT backend fallback)
    assert!(caps.model_supports("special-model", ProviderFeature::AutomaticPrefixCaching));
    assert!(!caps.model_supports("special-model", ProviderFeature::StreamingTokenCounts));

    // Unknown model family: falls back to backend-level
    assert!(caps.model_supports("unknown-model", ProviderFeature::StreamingTokenCounts));
    assert!(!caps.model_supports("unknown-model", ProviderFeature::AutomaticPrefixCaching));
}

// -------------------------------------------------------------------
// CapabilityRegistry::new() is empty
// -------------------------------------------------------------------

#[test]
fn test_registry_new_is_empty() {
    let registry = CapabilityRegistry::new();
    assert!(registry.list_backend_ids().is_empty());
}

// -------------------------------------------------------------------
// register_backend() adds backend, get_backend() retrieves it
// -------------------------------------------------------------------

#[test]
fn test_registry_register_and_get() {
    let mut registry = CapabilityRegistry::new();
    let caps = BackendCapabilities::none("test-backend");
    registry.register_backend(caps);

    let retrieved = registry.get_backend("test-backend");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().backend_id, "test-backend");
}

// -------------------------------------------------------------------
// supports_feature() checks backend-level
// -------------------------------------------------------------------

#[test]
fn test_registry_supports_feature() {
    let mut registry = CapabilityRegistry::new();
    let mut features = HashSet::new();
    features.insert(ProviderFeature::StructuredOutput);

    registry.register_backend(BackendCapabilities {
        backend_id: "test".to_string(),
        supported_features: features,
        model_families: HashMap::new(),
    });

    assert!(registry.supports_feature("test", ProviderFeature::StructuredOutput));
    assert!(!registry.supports_feature("test", ProviderFeature::ModelRouting));
    assert!(!registry.supports_feature("nonexistent", ProviderFeature::StructuredOutput));
}

// -------------------------------------------------------------------
// model_supports_feature() checks model-family with backend fallback
// -------------------------------------------------------------------

#[test]
fn test_registry_model_supports_feature() {
    let mut registry = CapabilityRegistry::new();

    let mut backend_features = HashSet::new();
    backend_features.insert(ProviderFeature::StreamingTokenCounts);

    let mut model_features = HashSet::new();
    model_features.insert(ProviderFeature::ExplicitCacheBreakpoints);

    let mut caps = BackendCapabilities {
        backend_id: "test".to_string(),
        supported_features: backend_features,
        model_families: HashMap::new(),
    };

    caps.add_model_family(ModelFamilyCapabilities {
        model_family: "fancy".to_string(),
        supported_features: model_features,
        max_cache_breakpoints: Some(2),
        min_cacheable_tokens: None,
        cache_economics: None,
    });

    registry.register_backend(caps);

    // Model-family level
    assert!(registry.model_supports_feature(
        "test",
        "fancy",
        ProviderFeature::ExplicitCacheBreakpoints
    ));
    // Fallback to backend for unknown model
    assert!(registry.model_supports_feature(
        "test",
        "unknown",
        ProviderFeature::StreamingTokenCounts
    ));
    // Nonexistent backend
    assert!(!registry.model_supports_feature(
        "missing",
        "any",
        ProviderFeature::StreamingTokenCounts
    ));
}

// -------------------------------------------------------------------
// with_defaults() has Anthropic backend
// -------------------------------------------------------------------

#[test]
fn test_defaults_has_anthropic() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry
        .get_backend("anthropic")
        .expect("anthropic not found");

    assert!(anthropic.supports(ProviderFeature::ExplicitCacheBreakpoints));
    assert!(anthropic.supports(ProviderFeature::RetentionTiers));
    assert!(anthropic.supports(ProviderFeature::StreamingTokenCounts));
    assert!(!anthropic.supports(ProviderFeature::AutomaticPrefixCaching));
}

// -------------------------------------------------------------------
// with_defaults() has OpenAI backend
// -------------------------------------------------------------------

#[test]
fn test_defaults_has_openai() {
    let registry = CapabilityRegistry::with_defaults();
    let openai = registry.get_backend("openai").expect("openai not found");

    assert!(openai.supports(ProviderFeature::AutomaticPrefixCaching));
    assert!(openai.supports(ProviderFeature::StreamingTokenCounts));
    assert!(openai.supports(ProviderFeature::StructuredOutput));
    assert!(!openai.supports(ProviderFeature::ExplicitCacheBreakpoints));
}

// -------------------------------------------------------------------
// Anthropic claude-3.5-sonnet model family has correct limits
// -------------------------------------------------------------------

#[test]
fn test_defaults_anthropic_claude_35_sonnet() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let sonnet = anthropic
        .model_families
        .get("claude-3.5-sonnet")
        .expect("claude-3.5-sonnet not found");

    assert_eq!(sonnet.max_cache_breakpoints, Some(4));
    assert_eq!(sonnet.min_cacheable_tokens, Some(1024));
    assert!(sonnet.supports(ProviderFeature::ExplicitCacheBreakpoints));
}

// -------------------------------------------------------------------
// OpenAI gpt-4o model family has AutomaticPrefixCaching
// -------------------------------------------------------------------

#[test]
fn test_defaults_openai_gpt4o() {
    let registry = CapabilityRegistry::with_defaults();
    let openai = registry.get_backend("openai").unwrap();
    let gpt4o = openai
        .model_families
        .get("gpt-4o")
        .expect("gpt-4o not found");

    assert!(gpt4o.supports(ProviderFeature::AutomaticPrefixCaching));
    assert!(gpt4o.supports(ProviderFeature::StreamingTokenCounts));
    assert!(gpt4o.supports(ProviderFeature::StructuredOutput));
}

// -------------------------------------------------------------------
// OpenAI o1 model family has only StreamingTokenCounts
// -------------------------------------------------------------------

#[test]
fn test_defaults_openai_o1() {
    let registry = CapabilityRegistry::with_defaults();
    let openai = registry.get_backend("openai").unwrap();
    let o1 = openai.model_families.get("o1").expect("o1 not found");

    assert!(o1.supports(ProviderFeature::StreamingTokenCounts));
    assert!(!o1.supports(ProviderFeature::AutomaticPrefixCaching));
    assert!(!o1.supports(ProviderFeature::StructuredOutput));
}

// -------------------------------------------------------------------
// CapabilityRegistry is Send + Sync
// -------------------------------------------------------------------

#[test]
fn test_capability_registry_is_send_sync() {
    assert_send_sync::<CapabilityRegistry>();
}

// -------------------------------------------------------------------
// list_backend_ids() returns sorted list
// -------------------------------------------------------------------

#[test]
fn test_list_backend_ids_sorted() {
    let registry = CapabilityRegistry::with_defaults();
    let ids = registry.list_backend_ids();
    assert_eq!(ids, vec!["anthropic", "openai"]);
}

// -------------------------------------------------------------------
// Default trait creates empty registry
// -------------------------------------------------------------------

#[test]
fn test_default_creates_empty_registry() {
    let registry = CapabilityRegistry::default();
    assert!(registry.list_backend_ids().is_empty());
}

// -------------------------------------------------------------------
// Current Anthropic model families (2026)
// -------------------------------------------------------------------

#[test]
fn test_defaults_anthropic_claude_sonnet_4() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-sonnet-4")
        .expect("claude-sonnet-4 not found");

    assert_eq!(family.min_cacheable_tokens, Some(1024));
    assert_eq!(family.max_cache_breakpoints, Some(4));
    assert!(family.supports(ProviderFeature::ExplicitCacheBreakpoints));
}

#[test]
fn test_defaults_anthropic_claude_sonnet_45() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-sonnet-4.5")
        .expect("claude-sonnet-4.5 not found");

    assert_eq!(family.min_cacheable_tokens, Some(1024));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_sonnet_46() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-sonnet-4.6")
        .expect("claude-sonnet-4.6 not found");

    assert_eq!(family.min_cacheable_tokens, Some(2048));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_opus_4() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-opus-4")
        .expect("claude-opus-4 not found");

    assert_eq!(family.min_cacheable_tokens, Some(1024));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_opus_45() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-opus-4.5")
        .expect("claude-opus-4.5 not found");

    assert_eq!(family.min_cacheable_tokens, Some(4096));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_opus_46() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-opus-4.6")
        .expect("claude-opus-4.6 not found");

    assert_eq!(family.min_cacheable_tokens, Some(4096));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_haiku_45() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-haiku-4.5")
        .expect("claude-haiku-4.5 not found");

    assert_eq!(family.min_cacheable_tokens, Some(4096));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}

#[test]
fn test_defaults_anthropic_claude_haiku_35() {
    let registry = CapabilityRegistry::with_defaults();
    let anthropic = registry.get_backend("anthropic").unwrap();
    let family = anthropic
        .model_families
        .get("claude-haiku-3.5")
        .expect("claude-haiku-3.5 not found");

    assert_eq!(family.min_cacheable_tokens, Some(2048));
    assert_eq!(family.max_cache_breakpoints, Some(4));
}
