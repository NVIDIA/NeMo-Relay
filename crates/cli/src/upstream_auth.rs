// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-local upstream credentials, shared by personal gateways and daemon dispatchers.

use std::process::Stdio;
use std::time::Duration;

use http::{HeaderMap, HeaderName, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use crate::configuration::GatewayConfig;
use crate::error::CliError;

const MAX_TOKEN_BYTES: u64 = 16 * 1024;
const CREDENTIAL_HEADERS: [&str; 4] =
    ["authorization", "x-api-key", "api-key", "anthropic-api-key"];

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ApiKeyHelper {
    pub(crate) command: String,
    #[serde(default)]
    pub(crate) args: Vec<String>,
    #[serde(default = "default_timeout_ms")]
    pub(crate) timeout_ms: u64,
    /// Absent means the provider's standard scheme (OpenAI bearer, Anthropic x-api-key).
    #[serde(default)]
    pub(crate) auth_scheme: Option<AuthScheme>,
}

// Arguments may contain secrets. Never render them in Debug-derived config diagnostics.
impl std::fmt::Debug for ApiKeyHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKeyHelper { configured: true }")
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AuthScheme {
    Bearer,
    ApiKey,
}

fn default_timeout_ms() -> u64 {
    10_000
}

pub(crate) fn validate_config(config: &GatewayConfig) -> Result<(), CliError> {
    for (provider, key, header, helper) in [
        (
            "openai",
            &config.openai_api_key,
            &config.openai_auth_header,
            &config.openai_api_key_helper,
        ),
        (
            "anthropic",
            &config.anthropic_api_key,
            &config.anthropic_auth_header,
            &config.anthropic_api_key_helper,
        ),
    ] {
        if usize::from(key.is_some())
            + usize::from(header.is_some())
            + usize::from(helper.is_some())
            > 1
        {
            return Err(CliError::Config(format!(
                "upstream.{provider}: api_key, auth_header, and api_key_helper are mutually exclusive"
            )));
        }
        if let Some(helper) = helper {
            if helper.command.trim().is_empty() || helper.timeout_ms == 0 {
                return Err(CliError::Config(format!(
                    "upstream.{provider}_api_key_helper requires a nonempty command and positive timeout_ms"
                )));
            }
            if provider == "openai" && matches!(helper.auth_scheme, Some(AuthScheme::ApiKey)) {
                return Err(CliError::Config(
                    "OpenAI API key helpers require bearer authentication".into(),
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn warn_ignored_environment(config: &GatewayConfig) {
    for (enabled, names) in [
        (
            config.openai_api_key_helper.is_some(),
            &["OPENAI_API_KEY", "NEMO_RELAY_OPENAI_AUTH_HEADER"][..],
        ),
        (
            config.anthropic_api_key_helper.is_some(),
            &[
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "NEMO_RELAY_ANTHROPIC_AUTH_HEADER",
            ][..],
        ),
    ] {
        if enabled {
            for name in names {
                if std::env::var_os(name).is_some() {
                    log::warn!(
                        target: "nemo_relay.configuration",
                        event = "provider_environment_auth_ignored",
                        environment_variable = *name,
                        reason = "api_key_helper_configured";
                        "Provider environment credential ignored because an API key helper is configured"
                    );
                }
            }
        }
    }
}

pub(crate) fn helper(config: &GatewayConfig, anthropic: bool) -> Option<&ApiKeyHelper> {
    if anthropic {
        config.anthropic_api_key_helper.as_ref()
    } else {
        config.openai_api_key_helper.as_ref()
    }
}

pub(crate) fn has_replacement(config: &GatewayConfig, anthropic: bool) -> bool {
    if anthropic {
        config.anthropic_api_key_helper.is_some()
            || config.anthropic_api_key.is_some()
            || config.anthropic_auth_header.is_some()
    } else {
        config.openai_api_key_helper.is_some()
            || config.openai_api_key.is_some()
            || config.openai_auth_header.is_some()
    }
}

/// Resolve immediately before dispatch. No result is stored or shared across requests.
pub(crate) async fn apply(
    headers: &mut HeaderMap,
    anthropic: bool,
    allow_configured_auth: bool,
    config: &GatewayConfig,
) -> Result<(), CliError> {
    if !allow_configured_auth {
        return Ok(());
    }
    validate_config(config)?;
    let (provider, helper, key, configured_header) = if anthropic {
        (
            "anthropic",
            &config.anthropic_api_key_helper,
            &config.anthropic_api_key,
            &config.anthropic_auth_header,
        )
    } else {
        (
            "openai",
            &config.openai_api_key_helper,
            &config.openai_api_key,
            &config.openai_auth_header,
        )
    };
    let (name, mut value) = if let Some(helper) = helper {
        let token = helper.token(provider).await?;
        token_header(
            &token,
            helper.auth_scheme.unwrap_or(if anthropic {
                AuthScheme::ApiKey
            } else {
                AuthScheme::Bearer
            }),
        )?
    } else {
        if crate::provider_auth::has_provider_credential(headers) {
            return Ok(());
        }
        if let Some(value) = configured_header {
            (
                AUTHORIZATION,
                HeaderValue::from_str(value)
                    .map_err(|_| CliError::Config(format!("invalid {provider} auth header")))?,
            )
        } else if let Some(key) = key {
            token_header(
                key,
                if anthropic {
                    AuthScheme::ApiKey
                } else {
                    AuthScheme::Bearer
                },
            )?
        } else if let Some(key) = std::env::var(if anthropic {
            "ANTHROPIC_API_KEY"
        } else {
            "OPENAI_API_KEY"
        })
        .ok()
        .filter(|key| !key.trim().is_empty())
        {
            token_header(
                &key,
                if anthropic {
                    AuthScheme::ApiKey
                } else {
                    AuthScheme::Bearer
                },
            )?
        } else {
            return Ok(());
        }
    };
    for name in CREDENTIAL_HEADERS {
        headers.remove(name);
    }
    value.set_sensitive(true);
    headers.insert(name, value);
    Ok(())
}

/// Validate configured keys with the same token rules used at request dispatch.
pub(crate) fn validate_api_key(name: &str, value: String) -> Result<String, CliError> {
    token_header(&value, AuthScheme::Bearer).map_err(|_| {
        CliError::Config(format!(
            "{name} must contain one valid bare token without whitespace"
        ))
    })?;
    Ok(value.trim().to_owned())
}

fn token_header(token: &str, scheme: AuthScheme) -> Result<(HeaderName, HeaderValue), CliError> {
    let token = token.trim();
    if token.is_empty()
        || token.chars().any(char::is_control)
        || token.chars().any(char::is_whitespace)
    {
        return Err(CliError::Launch(
            "Upstream API key is invalid. Expected one bare token without whitespace.".into(),
        ));
    }
    let (name, value) = match scheme {
        AuthScheme::Bearer => (AUTHORIZATION, format!("Bearer {token}")),
        AuthScheme::ApiKey => (HeaderName::from_static("x-api-key"), token.to_owned()),
    };
    let value = HeaderValue::from_str(&value).map_err(|_| {
        CliError::Launch("Upstream API key contains an invalid header value.".into())
    })?;
    Ok((name, value))
}

impl ApiKeyHelper {
    async fn token(&self, provider: &str) -> Result<String, CliError> {
        let fail = |reason: &str| CliError::Launch(format!("{provider} API key helper {reason}"));
        let mut command = tokio::process::Command::new(&self.command);
        #[cfg(unix)]
        command.process_group(0);
        command
            .args(&self.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            // Helper stderr can contain tokens; neither forward nor retain it.
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        let (mut child, _tree) = crate::process::spawn_helper(&mut command).await.map_err(
            |_| fail("could not be started in an owned process tree; check command and arguments"),
        )?;
        #[cfg(not(windows))]
        let mut child = command
            .spawn()
            .map_err(|_| fail("could not be started; check command and arguments"))?;
        #[cfg(unix)]
        let _group = HelperProcessGroup(child.id().expect("live helper child") as i32);
        let stdout = child.stdout.take().expect("piped helper stdout");
        let result = tokio::time::timeout(Duration::from_millis(self.timeout_ms), async {
            let mut output = Vec::new();
            stdout
                .take(MAX_TOKEN_BYTES + 1)
                .read_to_end(&mut output)
                .await
                .map_err(|_| fail("stdout could not be read"))?;
            if output.len() as u64 > MAX_TOKEN_BYTES {
                return Err(fail("output exceeds 16 KiB"));
            }
            let status = child
                .wait()
                .await
                .map_err(|_| fail("could not be awaited"))?;
            if !status.success() {
                return Err(fail("exited unsuccessfully"));
            }
            let output = String::from_utf8(output).map_err(|_| fail("output is not UTF-8"))?;
            let token = output.trim();
            // Validate without ever including the output in an error.
            token_header(token, AuthScheme::Bearer).map_err(|_| {
                fail("returned an invalid token. Expected one bare token without whitespace")
            })?;
            Ok(token.to_owned())
        })
        .await;
        match result {
            Ok(Ok(token)) => Ok(token),
            other => {
                let _ = child.kill().await;
                match other {
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err(fail("timed out")),
                    Ok(Ok(_)) => unreachable!(),
                }
            }
        }
    }
}

// Shell pipelines may have descendants holding stdout open. Own their process group so
// timeout, output overflow, request cancellation, and normal completion all clean them up.
#[cfg(unix)]
struct HelperProcessGroup(i32);

#[cfg(unix)]
impl Drop for HelperProcessGroup {
    fn drop(&mut self) {
        // SAFETY: The helper was spawned into its own group, identified by its positive PID.
        unsafe {
            libc::kill(-self.0, libc::SIGKILL);
        }
    }
}

#[cfg(test)]
#[path = "../tests/coverage/shared/upstream_auth_tests.rs"]
pub(crate) mod tests;
