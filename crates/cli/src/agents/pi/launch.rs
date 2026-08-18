// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transparent launch for pi.
//!
//! pi differs from Codex and Claude Code in two ways that shape this module.
//!
//! **Hooks cannot be injected from outside.** Codex takes `--config hooks.*=...`
//! and Claude Code reads a settings file, so their launchers can install hook
//! commands directly. pi has no native hook-configuration file and its external
//! stream is observation-only, so hook calls must originate inside an extension.
//! Launch therefore loads the NeMo Relay extension with `-e` and passes the
//! gateway URL through the environment for it to read.
//!
//! **Model traffic cannot be redirected by a flag or a generic env var.** pi
//! resolves `baseUrl` per model from its generated catalog; the only documented
//! override points are per-provider (`AZURE_OPENAI_BASE_URL`, `LLAMA_BASE_URL`)
//! or a provider registered by an extension. So redirection is the extension's
//! job too, and this module only supplies the URL.

use std::path::PathBuf;

use crate::error::CliError;
use crate::process::{PreparedAgentLaunch, insert_after_host};

/// Environment variable the pi extension reads to find the gateway.
pub(crate) const PI_GATEWAY_URL_ENV: &str = "NEMO_RELAY_PI_GATEWAY_URL";

/// Environment variable pointing pi at the NeMo Relay extension entry point.
pub(crate) const PI_EXTENSION_PATH_ENV: &str = "NEMO_RELAY_PI_EXTENSION";

/// Upstream this gateway forwards OpenAI-compatible traffic to.
pub(crate) const PI_OPENAI_UPSTREAM_ENV: &str = "NEMO_RELAY_PI_OPENAI_UPSTREAM";

/// Upstream this gateway forwards Anthropic traffic to.
pub(crate) const PI_ANTHROPIC_UPSTREAM_ENV: &str = "NEMO_RELAY_PI_ANTHROPIC_UPSTREAM";

pub(crate) fn prepare(
    launch: &mut PreparedAgentLaunch,
    gateway_url: &str,
    gateway: &crate::configuration::GatewayConfig,
) -> Result<(), CliError> {
    set_env(launch, PI_GATEWAY_URL_ENV, gateway_url);

    // Tell the extension what this gateway actually forwards to.
    //
    // Redirection is only correct when the gateway's upstream is the same endpoint the selected
    // model would otherwise call: the gateway resolves one OpenAI base and one Anthropic base from
    // static configuration (`ProviderRoute::upstream_url`) and there is no per-request override a
    // client can set -- inbound internal dispatch headers are stripped. Without these two values
    // the extension would have to redirect blind, and pointing (say) an NVIDIA model at a gateway
    // configured for `api.openai.com` breaks a session that worked a moment earlier.
    set_env(launch, PI_OPENAI_UPSTREAM_ENV, &gateway.openai_base_url);
    set_env(
        launch,
        PI_ANTHROPIC_UPSTREAM_ENV,
        &gateway.anthropic_base_url,
    );

    // `-e` is the right loader here: it is trust-ungated, loads before
    // discovery, and survives `--no-extensions`, so a launched session gets the
    // extension regardless of the user's own pi configuration.
    let Some(path) = extension_path() else {
        return Err(CliError::Launch(format!(
            "could not locate the NeMo Relay pi extension; set {PI_EXTENSION_PATH_ENV} to its \
             entry point, or install it with `pi install <source>` and launch pi directly"
        )));
    };
    let rendered = path.display().to_string();
    set_env(launch, PI_EXTENSION_PATH_ENV, &rendered);
    insert_after_host(
        &mut launch.argv,
        launch.host_index,
        ["-e".to_string(), rendered],
    );

    // Redirection is conditional, so say what the condition is rather than promising LLM spans.
    launch.notes.push(format!(
        "pi tool and turn activity is reported to NeMo Relay by the extension. Model calls are \
         routed through the gateway only when the selected model's provider already targets this \
         gateway's upstream (openai={openai}, anthropic={anthropic}); pi resolves a base URL per \
         model from a generated catalog, and the gateway forwards to one statically configured \
         upstream per API family. A model on any other provider keeps calling its own endpoint \
         and produces no LLM spans -- select a matching model, or start the gateway with \
         --openai-base-url / --anthropic-base-url pointing at that provider",
        openai = gateway.openai_base_url,
        anthropic = gateway.anthropic_base_url,
    ));
    Ok(())
}

fn set_env(launch: &mut PreparedAgentLaunch, name: &str, value: &str) {
    launch.env.retain(|(existing, _)| existing != name);
    launch.env.push((name.to_string(), value.to_string()));
}

/// Resolve the extension entry point, preferring an explicit override.
///
/// There is no installed location to fall back on the way Codex and Claude Code
/// have one, because pi extensions live in the user's own configuration
/// directories rather than in a NeMo Relay-managed plugin root.
fn extension_path() -> Option<PathBuf> {
    std::env::var_os(PI_EXTENSION_PATH_ENV)
        .map(PathBuf::from)
        .filter(|path| path.exists())
}
