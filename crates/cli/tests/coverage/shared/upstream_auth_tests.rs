// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(unix)]
pub(crate) fn helper(script: &str) -> ApiKeyHelper {
    ApiKeyHelper {
        command: "/bin/sh".into(),
        args: vec!["-c".into(), script.into()],
        timeout_ms: 1000,
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
