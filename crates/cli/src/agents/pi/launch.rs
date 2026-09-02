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
    // Still sent even though a launched session can name its own upstream (see
    // `pi::alignment`): the named path is what makes an arbitrary provider work, and these two
    // are what let the extension recognise the case where naming is unnecessary because the
    // gateway already fronts the model's endpoint. Without them every redirect would carry a
    // header, including the common one where it changes nothing.
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
            "could not locate the NeMo Relay pi extension at a load path that is not \
             trust-gated; run `nemo-relay install pi`, install it yourself with `pi \
             install <path to crates/cli/assets/pi-extension>` (without `--local`), or set \
             {PI_EXTENSION_PATH_ENV} to its entry point. A project-scoped install is \
             deliberately not used here: `-e` is never trust-gated, so passing one would load \
             code pi itself would not trust. `nemo-relay doctor pi` reports what was found"
        )));
    };
    // `-e` loads *in addition to* discovery, so a second copy of this extension elsewhere on the
    // machine is a second package to pi and loads beside this one, posting every hook twice: each
    // turn closed as superseded by its own duplicate, and the inline-shell gate deciding one
    // command twice. There is no way to suppress it from here that does not also drop the user's
    // own extensions -- `--no-extensions` keeps `-e` but discards everything discovered -- so
    // refuse and name both copies rather than launch a session whose trace is doubled.
    if let Some(duplicate) = conflicting_extension_path(&path) {
        return Err(CliError::Launch(format!(
            "two copies of the NeMo Relay pi extension would load in the same session: {} and \
             {}. pi de-duplicates its extension set by path, not by package, so both register \
             hooks and every turn, tool and inline-shell event is reported twice. Remove one \
             copy, or point {PI_EXTENSION_PATH_ENV} at the one you keep",
            path.display(),
            duplicate.display()
        )));
    }
    // A project-scoped copy is the one duplicate that cannot be decided from here. pi loads
    // `<cwd>/.pi/extensions` only for a trusted project, and that decision is made inside the
    // run -- `-a` overrides it, `defaultProjectTrust` can pre-answer it, and "trust this session
    // only" persists nothing -- so refusing would block every launch in an untrusted project
    // over a copy that will not load. Say what happens if the project *is* trusted instead.
    for site in super::doctor::project_copies_beside(&current_dir(), &path) {
        launch.notes.push(format!(
            "{} is a second copy of this extension under the project's `.pi/`. pi loads it \
             beside the one passed with `-e` whenever the project is trusted, and then every \
             turn, tool and inline-shell event is reported twice. Remove it, or run in an \
             untrusted project",
            site.display()
        ));
    }

    let rendered = path.display().to_string();
    set_env(launch, PI_EXTENSION_PATH_ENV, &rendered);
    insert_after_host(
        &mut launch.argv,
        launch.host_index,
        ["-e".to_string(), rendered],
    );

    // A launched session can name its own upstream, so this names the cases that still do not
    // reach the gateway rather than the far larger set that once did not.
    launch.notes.push(format!(
        "pi tool and turn activity is reported to NeMo Relay by the extension. Model calls are \
         routed through the gateway: a model already targeting this gateway's upstream \
         (openai={openai}, anthropic={anthropic}) is redirected as it is, and a model on any \
         other provider is redirected with its own endpoint named on each request. Two cases \
         still produce no LLM spans -- a model whose API the gateway has no route for, and a \
         provider whose models span more than one endpoint unless both --openai-base-url and \
         --anthropic-base-url already point at it, because a provider is redirected as a whole. \
         The model_redirect mark on the session scope names the outcome either way",
        openai = gateway.openai_base_url,
        anthropic = gateway.anthropic_base_url,
    ));
    Ok(())
}

fn set_env(launch: &mut PreparedAgentLaunch, name: &str, value: &str) {
    launch.env.retain(|(existing, _)| existing != name);
    launch.env.push((name.to_string(), value.to_string()));
}

/// Resolve the extension entry point, from the same places `doctor` looks.
///
/// It reads no environment variable of its own, deliberately. A second read is
/// how the two drifted: `doctor` resolved a user-scope install and reported the
/// setup as ready, while launching refused to start because only the variable
/// counted here -- and the variable is not something any document tells a user
/// to set. `launchable_extension_path` also excludes the project scope, which
/// `-e` would load past pi's trust gate.
fn extension_path() -> Option<PathBuf> {
    super::doctor::launchable_extension_path(&current_dir())
}

/// The other copy pi would load beside the launched one, if there is one.
fn conflicting_extension_path(launched: &std::path::Path) -> Option<PathBuf> {
    super::doctor::conflicting_extension_site(&current_dir(), launched)
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
