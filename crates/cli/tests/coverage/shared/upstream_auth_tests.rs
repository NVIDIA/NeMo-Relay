// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(unix)]
pub(crate) fn helper(script: &str) -> ApiKeyHelper {
    ApiKeyHelper {
        command: "/bin/sh".into(),
        args: vec!["-c".into(), script.into()],
        timeout_ms: 10_000,
        cache_ttl_ms: 0,
        cache: Default::default(),
        auth_scheme: None,
    }
}

#[cfg(unix)]
#[tokio::test]
async fn per_request_tokens_replace_inbound_auth_for_both_providers() {
    for (anthropic, scheme, name) in [
        (false, None, "authorization"),
        (true, None, "x-api-key"),
        (true, Some(AuthScheme::Bearer), "authorization"),
    ] {
        let mut generator = helper("printf 'token-%s\\n' $$");
        generator.auth_scheme = scheme;
        let mut config = GatewayConfig::default();
        if anthropic {
            config.anthropic_api_key_helper = Some(generator);
        } else {
            config.openai_api_key_helper = Some(generator);
        }
        let resolve = || async {
            let mut headers = HeaderMap::new();
            for header in CREDENTIAL_HEADERS {
                headers.insert(header, HeaderValue::from_static("caller-credential"));
            }
            apply(&mut headers, anthropic, true, &config).await.unwrap();
            assert_eq!(headers.len(), 1);
            assert!(headers[name].is_sensitive());
            headers[name].to_str().unwrap().to_owned()
        };
        let first = resolve().await;
        let second = resolve().await;
        assert_ne!(first, second);
        let (third, fourth) = tokio::join!(resolve(), resolve());
        assert_ne!(third, fourth);
        assert_ne!(first, third);
        if name == "authorization" {
            assert!(first.starts_with("Bearer token-"));
        } else {
            assert!(first.starts_with("token-"));
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn helper_failures_stop_before_dispatch_and_do_not_expose_output() {
    for cache_ttl_ms in [0, 60_000] {
        for script in [
            "exit 7",
            "printf ''",
            "printf 'secret\\nsecond'",
            "printf 'secret token'",
            "printf '\\377'",
            "head -c 20000 /dev/zero",
            "sleep 5",
        ] {
            let mut generator = helper(script);
            generator.timeout_ms = 100;
            generator.cache_ttl_ms = cache_ttl_ms;
            let config = GatewayConfig {
                openai_api_key_helper: Some(generator),
                ..GatewayConfig::default()
            };
            let mut headers = HeaderMap::new();
            let started = std::time::Instant::now();
            let error = apply(&mut headers, false, true, &config).await.unwrap_err();
            assert!(!error.to_string().contains("secret"));
            assert!(headers.is_empty());
            assert!(started.elapsed() < Duration::from_secs(2));
        }
    }
    let config = GatewayConfig {
        openai_api_key_helper: Some(ApiKeyHelper {
            command: "/missing/relay-helper".into(),
            ..helper("")
        }),
        ..GatewayConfig::default()
    };
    assert!(
        apply(&mut HeaderMap::new(), false, true, &config)
            .await
            .unwrap_err()
            .to_string()
            .contains("could not be started")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn forbidden_destinations_do_not_execute_helpers() {
    let config = GatewayConfig {
        openai_api_key_helper: Some(helper("exit 9")),
        ..GatewayConfig::default()
    };
    let mut headers =
        HeaderMap::from_iter([(AUTHORIZATION, HeaderValue::from_static("Bearer caller"))]);
    apply(&mut headers, false, false, &config).await.unwrap();
    assert_eq!(headers[AUTHORIZATION], "Bearer caller");
}

#[cfg(unix)]
#[tokio::test]
async fn environment_credentials_are_ignored_only_when_provider_helper_is_enabled() {
    let _environment = crate::test_support::EnvScope::set(&[
        (
            "OPENAI_API_KEY",
            Some(std::ffi::OsStr::new("ambient-openai")),
        ),
        (
            "ANTHROPIC_AUTH_TOKEN",
            Some(std::ffi::OsStr::new("ambient-anthropic-token")),
        ),
        (
            "ANTHROPIC_API_KEY",
            Some(std::ffi::OsStr::new("ambient-anthropic-key")),
        ),
    ]);
    let mut config = GatewayConfig::default();
    let mut headers = HeaderMap::new();
    apply(&mut headers, false, true, &config).await.unwrap();
    assert_eq!(headers[AUTHORIZATION], "Bearer ambient-openai");
    headers.clear();
    apply(&mut headers, true, true, &config).await.unwrap();
    assert_eq!(headers["x-api-key"], "ambient-anthropic-key");
    config.openai_api_key_helper = Some(helper("printf helper-token"));
    config.anthropic_api_key_helper = Some(helper("printf helper-token"));
    for anthropic in [false, true] {
        let mut headers = HeaderMap::from_iter([
            (
                AUTHORIZATION,
                HeaderValue::from_static("Bearer ambient-anthropic-token"),
            ),
            (
                HeaderName::from_static("x-api-key"),
                HeaderValue::from_static("ambient-anthropic-key"),
            ),
        ]);
        apply(&mut headers, anthropic, true, &config).await.unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(
            headers[if anthropic {
                "x-api-key"
            } else {
                "authorization"
            }],
            if anthropic {
                "helper-token"
            } else {
                "Bearer helper-token"
            }
        );
    }
    config.openai_api_key_helper = Some(helper("exit 7"));
    let mut headers = HeaderMap::new();
    assert!(apply(&mut headers, false, true, &config).await.is_err());
    assert!(headers.is_empty());
    // OpenAI helper selection does not disable the other provider's env fallback.
    config.anthropic_api_key_helper = None;
    apply(&mut headers, true, true, &config).await.unwrap();
    assert_eq!(headers["x-api-key"], "ambient-anthropic-key");
}

#[tokio::test]
async fn static_api_keys_use_provider_schemes() {
    let config = GatewayConfig {
        openai_api_key: Some("static-openai".into()),
        anthropic_api_key: Some("static-anthropic".into()),
        ..GatewayConfig::default()
    };
    let mut headers = HeaderMap::new();
    apply(&mut headers, false, true, &config).await.unwrap();
    assert_eq!(headers[AUTHORIZATION], "Bearer static-openai");
    headers.clear();
    apply(&mut headers, true, true, &config).await.unwrap();
    assert_eq!(headers["x-api-key"], "static-anthropic");
}

#[cfg(unix)]
#[tokio::test]
async fn helper_cache_shares_concurrent_refreshes_and_expires() {
    for anthropic in [false, true] {
        let mut generator = helper("printf 'token-%s' $$");
        generator.cache_ttl_ms = 60_000;
        let mut config = GatewayConfig::default();
        if anthropic {
            config.anthropic_api_key_helper = Some(generator);
        } else {
            config.openai_api_key_helper = Some(generator);
        }
        let resolve = |config: GatewayConfig| async move {
            let mut headers = HeaderMap::new();
            apply(&mut headers, anthropic, true, &config).await.unwrap();
            let name = if anthropic {
                "x-api-key"
            } else {
                "authorization"
            };
            assert!(headers[name].is_sensitive());
            headers[name].clone()
        };
        let tokens = futures_util::future::join_all((0..12).map(|_| resolve(config.clone()))).await;
        assert!(tokens.iter().all(|token| token == tokens[0]));
        assert_eq!(resolve(config.clone()).await, tokens[0]);
        // Advance only between subprocess invocations: paused time can otherwise auto-advance
        // the helper timeout while waiting for operating-system process I/O.
        tokio::time::pause();
        tokio::time::advance(Duration::from_millis(60_000)).await;
        tokio::time::resume();
        let refreshed = resolve(config.clone()).await;
        assert_ne!(refreshed, tokens[0]);
        assert_eq!(resolve(config).await, refreshed);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn helper_cache_shares_failures_without_reusing_expired_or_ambient_tokens() {
    let _environment = crate::test_support::EnvScope::set(&[(
        "OPENAI_API_KEY",
        Some(std::ffi::OsStr::new("ambient-secret")),
    )]);
    let root = tempfile::tempdir().unwrap();
    let count = root.path().join("count");
    let fail = root.path().join("fail");
    let mut generator = helper(
        "printf x >> \"$1\"; if [ -f \"$2\" ]; then printf secret >&2; exit 7; fi; printf 'token-%s' $$",
    );
    generator.args.extend([
        "helper".into(),
        count.to_str().unwrap().into(),
        fail.to_str().unwrap().into(),
    ]);
    generator.cache_ttl_ms = 60_000;
    let config = GatewayConfig {
        openai_api_key_helper: Some(generator),
        ..GatewayConfig::default()
    };
    let mut headers = HeaderMap::new();
    apply(&mut headers, false, true, &config).await.unwrap();
    std::fs::write(&fail, "").unwrap();
    let mut still_valid = HeaderMap::new();
    apply(&mut still_valid, false, true, &config).await.unwrap();
    assert_eq!(still_valid[AUTHORIZATION], headers[AUTHORIZATION]);
    assert_eq!(std::fs::read(&count).unwrap(), b"x");
    tokio::time::pause();
    tokio::time::advance(Duration::from_millis(60_000)).await;
    tokio::time::resume();

    let results = futures_util::future::join_all((0..12).map(|_| async {
        let mut headers = HeaderMap::new();
        let error = apply(&mut headers, false, true, &config)
            .await
            .unwrap_err()
            .to_string();
        assert!(headers.is_empty());
        assert!(!error.contains("secret"));
        error
    }))
    .await;
    assert!(results.iter().all(|error| error == &results[0]));
    assert_eq!(std::fs::read(&count).unwrap(), b"xx");
    std::fs::remove_file(fail).unwrap();
    apply(&mut HeaderMap::new(), false, true, &config)
        .await
        .unwrap();
    assert_eq!(std::fs::read(count).unwrap(), b"xxx");
}

#[cfg(unix)]
#[tokio::test]
async fn helper_cache_identity_and_serialization_keep_credentials_isolated() {
    let mut generator = helper("printf 'token-%s' $$");
    generator.cache_ttl_ms = 60_000;
    let first = generator
        .resolve_token("openai", "https://one.example/v1")
        .await
        .unwrap();
    let serialized = serde_json::to_string(&generator).unwrap();
    assert!(!serialized.contains(&first));
    assert!(!serialized.contains("expires_at"));
    assert!(!serialized.contains("\"cache\""));
    assert!(!format!("{generator:?}").contains(&first));
    let independent: ApiKeyHelper = serde_json::from_str(&serialized).unwrap();
    assert_ne!(
        independent
            .resolve_token("openai", "https://one.example/v1")
            .await
            .unwrap(),
        first
    );
    for variant in 0..7 {
        let mut changed = generator.clone();
        let mut provider = "openai";
        let mut base = "https://one.example/v1";
        match variant {
            0 => provider = "anthropic",
            1 => base = "https://one.example/v2",
            2 => changed.command = "/bin/bash".into(),
            3 => changed.args.push("unused-argument".into()),
            4 => changed.timeout_ms += 1,
            5 => changed.cache_ttl_ms += 1,
            _ => changed.auth_scheme = Some(AuthScheme::Bearer),
        }
        let next = changed.resolve_token(provider, base).await.unwrap();
        assert_ne!(next, first);
        assert_eq!(changed.resolve_token(provider, base).await.unwrap(), next);
        // Reverting identity must not resurrect the original credential.
        assert_ne!(
            generator
                .resolve_token("openai", "https://one.example/v1")
                .await
                .unwrap(),
            first
        );
    }
    // Explicit zero preserves uncached behavior even on a previously cached runtime clone.
    generator.cache_ttl_ms = 0;
    let (one, two) = tokio::join!(
        generator.resolve_token("openai", "https://one.example/v1"),
        generator.resolve_token("openai", "https://one.example/v1")
    );
    assert_ne!(one.unwrap(), two.unwrap());
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_cache_waiter_does_not_cancel_shared_refresh() {
    let root = tempfile::tempdir().unwrap();
    let count = root.path().join("count");
    let release = root.path().join("release");
    let mut generator =
        helper("printf x >> \"$1\"; while [ ! -f \"$2\" ]; do sleep 0.01; done; printf token");
    generator.timeout_ms = 5000;
    generator.cache_ttl_ms = 60_000;
    generator.args.extend([
        "helper".into(),
        count.to_str().unwrap().into(),
        release.to_str().unwrap().into(),
    ]);
    let clone = generator.clone();
    let waiter =
        tokio::spawn(async move { clone.resolve_token("openai", "https://example.com").await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !count.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    std::fs::write(release, "").unwrap();
    assert_eq!(
        generator
            .resolve_token("openai", "https://example.com")
            .await
            .unwrap(),
        "token"
    );
    assert_eq!(std::fs::read(count).unwrap(), b"x");
}

#[test]
fn omitted_and_zero_cache_ttl_are_equivalent() {
    for setting in ["", "\ncache_ttl_ms = 0"] {
        let helper: ApiKeyHelper = toml::from_str(&format!("command = 'helper'{setting}")).unwrap();
        assert_eq!(helper.cache_ttl_ms, 0);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn replaced_helper_does_not_publish_an_old_in_flight_credential() {
    let root = tempfile::tempdir().unwrap();
    let started = root.path().join("started");
    let release = root.path().join("release");
    let mut old =
        helper("touch \"$1\"; while [ ! -f \"$2\" ]; do sleep 0.01; done; printf old-token");
    old.timeout_ms = 5000;
    old.cache_ttl_ms = 60_000;
    old.args.extend([
        "helper".into(),
        started.to_str().unwrap().into(),
        release.to_str().unwrap().into(),
    ]);
    let mut replacement = old.clone();
    replacement.args = vec!["-c".into(), "printf new-token".into()];
    let waiter =
        tokio::spawn(async move { old.resolve_token("openai", "https://example.com").await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !started.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        replacement
            .resolve_token("openai", "https://example.com")
            .await
            .unwrap(),
        "new-token"
    );
    std::fs::write(release, "").unwrap();
    assert_eq!(waiter.await.unwrap().unwrap(), "old-token");
    assert_eq!(
        replacement
            .resolve_token("openai", "https://example.com")
            .await
            .unwrap(),
        "new-token"
    );
}
