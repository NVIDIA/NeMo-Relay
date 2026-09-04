// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable deployment artifacts for administrator-managed daemon integrations.
//!
//! Personal `nemo-relay install` deliberately remains separate. A managed bundle is rendered
//! once for one deployment and then distributed by the administrator. Refresh and doctor only
//! validate it: changing a deployed v1 artifact in place is an error, and an incompatible contract
//! after v1 publication needs a separately named v2 bundle.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::common::state::{ROUTE_TOKEN_ENV, RouteCredential};
use crate::error::CliError;

pub(crate) const BUNDLE_FAMILY: &str = "nemo-relay-managed-v1";
pub(crate) const MANIFEST_FILE: &str = "nemo-relay-managed-v1.manifest.json";
const SCHEMA_VERSION: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_ARTIFACTS: usize = 256;
const CLAUDE_CUSTOM_HEADERS_ENV: &str = "ANTHROPIC_CUSTOM_HEADERS";
const ROUTE_TOKEN_HEADER: &str = "x-nemo-relay-client-token";
const PI_DAEMON_ADDRESS_PLACEHOLDER: &str = "__NEMO_RELAY_DAEMON_ADDRESS__";
const PI_DISPATCHER_PLACEHOLDER: &str = "__NEMO_RELAY_DISPATCHER_COMMAND__";

// Part of the v1 artifact family. Once v1 is published, new host events belong in a v2 family:
// silently changing these lists would make an upgrade rewrite enterprise-managed plugin bytes.
const CODEX_HOOK_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "PreCompact",
    "PostCompact",
];
const CLAUDE_HOOK_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "UserPromptExpansion",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "SubagentStart",
    "SubagentStop",
    "Notification",
    "Stop",
    "PreCompact",
    "PostCompact",
    "SessionEnd",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ManagedAgent {
    Codex,
    ClaudeCode,
    Pi,
}

impl ManagedAgent {
    const fn hook_argument(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude",
            Self::Pi => "pi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ManagedPlatform {
    Linux,
    Macos,
    Windows,
}

impl ManagedPlatform {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Macos => "macos",
            Self::Windows => "windows",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManagedBundleSpec {
    daemon_address: String,
    dispatcher_command: String,
    platform: ManagedPlatform,
    agents: BTreeSet<ManagedAgent>,
}

impl ManagedBundleSpec {
    pub(crate) fn new(
        daemon_address: impl Into<String>,
        dispatcher_command: impl Into<String>,
        platform: ManagedPlatform,
        agents: impl IntoIterator<Item = ManagedAgent>,
    ) -> Result<Self, CliError> {
        let daemon_address = normalize_daemon_address(&daemon_address.into())?;
        let dispatcher_command = dispatcher_command.into();
        validate_dispatcher(&dispatcher_command, platform)?;
        let agents = agents.into_iter().collect::<BTreeSet<_>>();
        if agents.is_empty() {
            return Err(CliError::Config(
                "a managed daemon bundle must target at least one agent".into(),
            ));
        }
        Ok(Self {
            daemon_address,
            dispatcher_command,
            platform,
            agents,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManagedBundleValidation {
    pub(crate) artifact_count: usize,
    pub(crate) daemon_address: String,
    pub(crate) platform: ManagedPlatform,
    pub(crate) sha256: ManagedBundleDigest,
}

/// SHA-256 over the canonical, length-prefixed manifest and artifact byte stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManagedBundleDigest(String);

impl std::fmt::Display for ManagedBundleDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for ManagedBundleDigest {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(
                "managed bundle SHA-256 must be exactly 64 lowercase hexadecimal characters".into(),
            );
        }
        Ok(Self(value.to_owned()))
    }
}

#[derive(Debug, Clone)]
struct RenderedArtifact {
    agent: ManagedAgent,
    path: &'static str,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedManifest {
    schema_version: u32,
    family: String,
    daemon_address: String,
    dispatcher_command: String,
    platform: ManagedPlatform,
    agents: Vec<ManagedAgent>,
    artifacts: Vec<ManifestArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArtifact {
    agent: ManagedAgent,
    path: String,
    byte_length: u64,
    sha256: String,
}

struct RenderedBundle {
    manifest: Vec<u8>,
    artifacts: Vec<RenderedArtifact>,
}

/// Creates a bundle only when the destination does not exist.
///
/// If the destination already exists, it is validated byte-for-byte and left untouched. The
/// returned digest is suitable for separately provisioning `doctor --managed-bundle-sha256`.
pub(crate) fn write_new_bundle(
    root: &Path,
    spec: &ManagedBundleSpec,
) -> Result<ManagedBundleDigest, CliError> {
    let expected = render_bundle(spec)?;
    let expected_digest = rendered_bundle_digest(&expected);
    if root.exists() {
        validate_bundle_files(root, false, None)?;
        let actual_manifest = read_bounded(&root.join(MANIFEST_FILE), MAX_MANIFEST_BYTES)?;
        if actual_manifest != expected.manifest {
            return Err(CliError::Config(format!(
                "refused to replace existing managed bundle {} with different deployment bytes; use a separately named artifact",
                root.display()
            )));
        }
        return Ok(expected_digest);
    }
    let parent = root.parent().ok_or_else(|| {
        CliError::Config(format!(
            "managed bundle destination {} has no parent",
            root.display()
        ))
    })?;
    fs::create_dir_all(parent)?;
    let name = root
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            CliError::Config("managed bundle destination is not valid Unicode".into())
        })?;
    let stage = parent.join(format!(".{name}.{}.tmp", Uuid::now_v7()));
    fs::create_dir(&stage)?;
    let result = write_rendered_bundle(&stage, expected).and_then(|()| {
        fs::rename(&stage, root).map_err(|error| {
            CliError::Config(format!(
                "failed to publish managed bundle {}: {error}",
                root.display()
            ))
        })
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    result.map(|()| expected_digest)
}

/// Managed refresh is validation-only; it never regenerates or overwrites v1 artifacts.
pub(crate) fn refresh_bundle(
    root: &Path,
    expected_sha256: &ManagedBundleDigest,
) -> Result<ManagedBundleValidation, CliError> {
    validate_bundle(root, expected_sha256)
}

/// Validates the administrator-provided digest, every exact artifact byte, and managed env.
pub(crate) fn validate_bundle(
    root: &Path,
    expected_sha256: &ManagedBundleDigest,
) -> Result<ManagedBundleValidation, CliError> {
    validate_bundle_files(root, true, Some(expected_sha256))
}

fn validate_bundle_files(
    root: &Path,
    validate_environment: bool,
    expected_sha256: Option<&ManagedBundleDigest>,
) -> Result<ManagedBundleValidation, CliError> {
    reject_non_directory_or_symlink(root)?;
    let manifest_path = root.join(MANIFEST_FILE);
    reject_symlink(&manifest_path)?;
    let manifest_bytes = read_bounded(&manifest_path, MAX_MANIFEST_BYTES)?;
    let manifest: ManagedManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        CliError::Config(format!(
            "managed bundle manifest {} is invalid: {error}",
            manifest_path.display()
        ))
    })?;
    if manifest.schema_version != SCHEMA_VERSION || manifest.family != BUNDLE_FAMILY {
        return Err(CliError::Config(format!(
            "managed bundle {} is not the supported {BUNDLE_FAMILY} artifact family",
            root.display()
        )));
    }
    if manifest.artifacts.len() > MAX_ARTIFACTS {
        return Err(CliError::Config(format!(
            "managed bundle manifest has more than {MAX_ARTIFACTS} artifacts"
        )));
    }
    let spec = ManagedBundleSpec::new(
        manifest.daemon_address,
        manifest.dispatcher_command,
        manifest.platform,
        manifest.agents,
    )?;
    let expected = render_bundle(&spec)?;
    if manifest_bytes != expected.manifest {
        return Err(CliError::Config(format!(
            "managed bundle manifest {} differs from the canonical {BUNDLE_FAMILY} bytes",
            manifest_path.display()
        )));
    }
    let expected_paths = expected
        .artifacts
        .iter()
        .map(|artifact| artifact.path)
        .chain(std::iter::once(MANIFEST_FILE))
        .collect::<BTreeSet<_>>();
    let actual_paths = bundle_files(root)?;
    if actual_paths != expected_paths {
        let missing = expected_paths
            .difference(&actual_paths)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = actual_paths
            .difference(&expected_paths)
            .copied()
            .collect::<Vec<_>>();
        return Err(CliError::Config(format!(
            "managed bundle file set differs from its immutable manifest (missing: {}; unexpected: {})",
            display_paths(&missing),
            display_paths(&unexpected)
        )));
    }
    for artifact in &expected.artifacts {
        let path = root.join(artifact.path);
        reject_symlink(&path)?;
        let actual = read_bounded(&path, artifact.bytes.len() as u64)?;
        if actual != artifact.bytes {
            return Err(CliError::Config(format!(
                "managed artifact {} differs from its exact canonical bytes",
                path.display()
            )));
        }
    }
    let sha256 = rendered_bundle_digest(&expected);
    if let Some(expected_sha256) = expected_sha256
        && expected_sha256 != &sha256
    {
        return Err(CliError::Config(format!(
            "managed bundle SHA-256 mismatch: expected {expected_sha256}, calculated {sha256}"
        )));
    }
    if validate_environment {
        validate_managed_environment(&spec)?;
    }
    Ok(ManagedBundleValidation {
        artifact_count: expected.artifacts.len(),
        daemon_address: spec.daemon_address,
        platform: spec.platform,
        sha256,
    })
}

fn validate_managed_environment(spec: &ManagedBundleSpec) -> Result<(), CliError> {
    let credential = RouteCredential::from_environment()?;
    if !spec.agents.contains(&ManagedAgent::ClaudeCode) {
        return Ok(());
    }
    let custom_headers = std::env::var(CLAUDE_CUSTOM_HEADERS_ENV).map_err(|_| {
        CliError::Config(format!(
            "managed Claude Code integration requires {CLAUDE_CUSTOM_HEADERS_ENV}; enterprise bootstrap must derive it from {ROUTE_TOKEN_ENV}"
        ))
    })?;
    let matches = custom_headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.trim().eq_ignore_ascii_case(ROUTE_TOKEN_HEADER))
        .map(|(_, value)| value.trim())
        .collect::<Vec<_>>();
    if matches.as_slice() != [credential.expose()] {
        return Err(CliError::Config(format!(
            "{CLAUDE_CUSTOM_HEADERS_ENV} must contain exactly one {ROUTE_TOKEN_HEADER} header whose value matches {ROUTE_TOKEN_ENV}"
        )));
    }
    Ok(())
}

fn render_bundle(spec: &ManagedBundleSpec) -> Result<RenderedBundle, CliError> {
    let mut artifacts = Vec::new();
    for agent in &spec.agents {
        artifacts.extend(render_agent(*agent, spec)?);
    }
    artifacts.sort_by_key(|artifact| artifact.path);
    let manifest = ManagedManifest {
        schema_version: SCHEMA_VERSION,
        family: BUNDLE_FAMILY.into(),
        daemon_address: spec.daemon_address.clone(),
        dispatcher_command: spec.dispatcher_command.clone(),
        platform: spec.platform,
        agents: spec.agents.iter().copied().collect(),
        artifacts: artifacts
            .iter()
            .map(|artifact| ManifestArtifact {
                agent: artifact.agent,
                path: artifact.path.into(),
                byte_length: artifact.bytes.len() as u64,
                sha256: sha256_hex(&artifact.bytes),
            })
            .collect(),
    };
    Ok(RenderedBundle {
        manifest: json_bytes(&manifest)?,
        artifacts,
    })
}

fn render_agent(
    agent: ManagedAgent,
    spec: &ManagedBundleSpec,
) -> Result<Vec<RenderedArtifact>, CliError> {
    match agent {
        ManagedAgent::Codex => render_codex(spec),
        ManagedAgent::ClaudeCode => render_claude(spec),
        ManagedAgent::Pi => render_pi(spec),
    }
}

fn render_codex(spec: &ManagedBundleSpec) -> Result<Vec<RenderedArtifact>, CliError> {
    let mcp = json!({
        "nemo-relay": {
            "command": spec.dispatcher_command,
            "args": ["daemon", "mcp", "--daemon-address", spec.daemon_address],
            "env_vars": [ROUTE_TOKEN_ENV],
            "required": true,
            "startup_timeout_sec": 20
        }
    });
    let plugin = plugin_manifest("codex");
    let settings = format!(
        "model_provider = \"nemo-relay-managed-v1\"\n\n[model_providers.nemo-relay-managed-v1]\nname = \"NeMo Relay Managed\"\nbase_url = {}\nwire_api = \"responses\"\nrequires_openai_auth = true\nsupports_websockets = false\nenv_http_headers = {{ {} = {} }}\n",
        toml_string(&format!("{}/v1", spec.daemon_address)),
        toml_string(ROUTE_TOKEN_HEADER),
        toml_string(ROUTE_TOKEN_ENV),
    );
    Ok(vec![
        artifact(
            ManagedAgent::Codex,
            "codex/plugin-v1/.codex-plugin/plugin.json",
            json_bytes(&plugin)?,
        ),
        artifact(
            ManagedAgent::Codex,
            "codex/plugin-v1/.mcp.json",
            json_bytes(&mcp)?,
        ),
        artifact(
            ManagedAgent::Codex,
            "codex/plugin-v1/hooks/hooks.json",
            hook_bytes(ManagedAgent::Codex, CODEX_HOOK_EVENTS, spec)?,
        ),
        artifact(
            ManagedAgent::Codex,
            "codex/settings-v1/config.toml",
            settings.into_bytes(),
        ),
    ])
}

fn render_claude(spec: &ManagedBundleSpec) -> Result<Vec<RenderedArtifact>, CliError> {
    let mcp = json!({
        "mcpServers": {
            "nemo-relay": {
                "command": spec.dispatcher_command,
                "args": ["daemon", "mcp", "--daemon-address", spec.daemon_address],
                "env": { (ROUTE_TOKEN_ENV): format!("${{{ROUTE_TOKEN_ENV}}}") },
                "alwaysLoad": true
            }
        }
    });
    let settings = json!({
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "env": { "ANTHROPIC_BASE_URL": spec.daemon_address }
    });
    Ok(vec![
        artifact(
            ManagedAgent::ClaudeCode,
            "claude-code/plugin-v1/.claude-plugin/plugin.json",
            json_bytes(&plugin_manifest("claude-code"))?,
        ),
        artifact(
            ManagedAgent::ClaudeCode,
            "claude-code/plugin-v1/.mcp.json",
            json_bytes(&mcp)?,
        ),
        artifact(
            ManagedAgent::ClaudeCode,
            "claude-code/plugin-v1/hooks/hooks.json",
            hook_bytes(ManagedAgent::ClaudeCode, CLAUDE_HOOK_EVENTS, spec)?,
        ),
        artifact(
            ManagedAgent::ClaudeCode,
            "claude-code/settings-v1/managed-settings.json",
            json_bytes(&settings)?,
        ),
    ])
}

fn render_pi(spec: &ManagedBundleSpec) -> Result<Vec<RenderedArtifact>, CliError> {
    let config = render_pi_config(spec)?;
    Ok(vec![
        artifact(
            ManagedAgent::Pi,
            "pi/extension-v1/README.md",
            include_bytes!("pi_extension/README.md").to_vec(),
        ),
        artifact(
            ManagedAgent::Pi,
            "pi/extension-v1/index.ts",
            include_bytes!("pi_extension/index.ts").to_vec(),
        ),
        artifact(
            ManagedAgent::Pi,
            "pi/extension-v1/managed-config.json",
            config,
        ),
        artifact(
            ManagedAgent::Pi,
            "pi/extension-v1/package.json",
            include_bytes!("pi_extension/package.json").to_vec(),
        ),
        artifact(
            ManagedAgent::Pi,
            "pi/extension-v1/tsconfig.json",
            include_bytes!("pi_extension/tsconfig.json").to_vec(),
        ),
    ])
}

fn render_pi_config(spec: &ManagedBundleSpec) -> Result<Vec<u8>, CliError> {
    let template = include_str!("pi_extension/managed-config.json");
    let rendered = replace_json_string_value(
        template,
        PI_DAEMON_ADDRESS_PLACEHOLDER,
        &spec.daemon_address,
    )?;
    let rendered = replace_json_string_value(
        &rendered,
        PI_DISPATCHER_PLACEHOLDER,
        &spec.dispatcher_command,
    )?;
    if rendered.contains("__NEMO_RELAY_") {
        return Err(CliError::Config(
            "managed Pi configuration contains an unrendered deployment placeholder".into(),
        ));
    }
    serde_json::from_str::<Value>(&rendered).map_err(|error| {
        CliError::Config(format!(
            "rendered managed Pi configuration is invalid JSON: {error}"
        ))
    })?;
    Ok(rendered.into_bytes())
}

fn replace_json_string_value(
    template: &str,
    placeholder: &str,
    value: &str,
) -> Result<String, CliError> {
    let placeholder = serde_json::to_string(placeholder).map_err(|error| {
        CliError::Config(format!("failed to encode managed Pi placeholder: {error}"))
    })?;
    if template.matches(&placeholder).count() != 1 {
        return Err(CliError::Config(
            "managed Pi configuration must contain each deployment placeholder exactly once".into(),
        ));
    }
    let value = serde_json::to_string(value).map_err(|error| {
        CliError::Config(format!(
            "failed to encode managed Pi deployment value: {error}"
        ))
    })?;
    Ok(template.replacen(&placeholder, &value, 1))
}

fn plugin_manifest(agent: &str) -> Value {
    json!({
        "name": "nemo-relay-managed-v1",
        "version": "1.0.0",
        "description": format!("Immutable NeMo Relay managed integration for {agent}."),
        "author": { "name": "NVIDIA Corporation and Affiliates" },
        "license": "Apache-2.0",
        "mcpServers": "./.mcp.json"
    })
}

fn hook_bytes(
    agent: ManagedAgent,
    events: &[&str],
    spec: &ManagedBundleSpec,
) -> Result<Vec<u8>, CliError> {
    let fail_open = hook_command(agent, spec, false);
    let fail_closed = hook_command(agent, spec, true);
    let hooks = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if matches!(
                *event,
                "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PermissionRequest"
            ) {
                group.insert("matcher".into(), json!("*"));
            }
            let command = if matches!(*event, "PreToolUse" | "PermissionRequest") {
                &fail_closed
            } else {
                &fail_open
            };
            group.insert(
                "hooks".into(),
                json!([{ "type": "command", "command": command, "timeout": 30 }]),
            );
            (
                (*event).to_string(),
                Value::Array(vec![Value::Object(group)]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json_bytes(&json!({ "hooks": hooks }))
}

fn hook_command(agent: ManagedAgent, spec: &ManagedBundleSpec, fail_closed: bool) -> String {
    format!(
        "{} daemon hook {} --daemon-address {} {}",
        spec.dispatcher_command,
        agent.hook_argument(),
        spec.daemon_address,
        if fail_closed {
            "--fail-closed"
        } else {
            "--fail-open"
        }
    )
}

fn artifact(agent: ManagedAgent, path: &'static str, bytes: Vec<u8>) -> RenderedArtifact {
    RenderedArtifact { agent, path, bytes }
}

fn write_rendered_bundle(root: &Path, bundle: RenderedBundle) -> Result<(), CliError> {
    for artifact in bundle.artifacts {
        write_new_file(&root.join(artifact.path), &artifact.bytes)?;
    }
    write_new_file(&root.join(MANIFEST_FILE), &bundle.manifest)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let parent = path.parent().ok_or_else(|| {
        CliError::Config(format!("managed artifact {} has no parent", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            CliError::Config(format!(
                "refused to overwrite managed artifact {}: {error}",
                path.display()
            ))
        })?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn normalize_daemon_address(raw: &str) -> Result<String, CliError> {
    let raw = raw.trim_end_matches('/');
    super::common::address::daemon_url(raw)?;
    Ok(raw.to_string())
}

fn validate_dispatcher(command: &str, platform: ManagedPlatform) -> Result<(), CliError> {
    let normalized = command.replace('\\', "/").to_ascii_lowercase();
    let windows_drive_absolute = normalized.as_bytes().get(1) == Some(&b':')
        && normalized
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
        && normalized.as_bytes().get(2) == Some(&b'/');
    let absolute = match platform {
        ManagedPlatform::Linux | ManagedPlatform::Macos => command.starts_with('/'),
        ManagedPlatform::Windows => windows_drive_absolute || normalized.starts_with("//"),
    };
    let platform_separators_are_valid =
        matches!(platform, ManagedPlatform::Windows) || !command.contains('\\');
    let parts = normalized.split('/').filter(|part| !part.is_empty());
    let has_relative_component = parts
        .clone()
        .any(|component| matches!(component, "." | ".."));
    let forbidden_root = match platform {
        ManagedPlatform::Linux | ManagedPlatform::Macos => [
            "/tmp",
            "/var/tmp",
            "/private/tmp",
            "/home",
            "/users",
            "/root",
            "/run/user",
            "/var/folders",
        ]
        .iter()
        .any(|root| normalized == *root || normalized.starts_with(&format!("{root}/"))),
        ManagedPlatform::Windows => {
            let path = if windows_drive_absolute {
                &normalized[2..]
            } else {
                normalized.as_str()
            };
            ["/users", "/temp", "/tmp", "/windows/temp"]
                .iter()
                .any(|root| path == *root || path.starts_with(&format!("{root}/")))
        }
    };
    let points_to_directory = normalized.ends_with('/');
    if command.is_empty()
        || !absolute
        || !platform_separators_are_valid
        || command.chars().any(char::is_whitespace)
        || has_relative_component
        || forbidden_root
        || points_to_directory
        || command.chars().any(|character| {
            !character.is_ascii_alphanumeric()
                && !matches!(character, '/' | '\\' | ':' | '.' | '_' | '-')
        })
    {
        return Err(CliError::Config(
            "managed dispatcher command must be an absolute, platform-appropriate, shell-safe administrator path outside user and temporary directories"
                .into(),
        ));
    }
    Ok(())
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, CliError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| CliError::Config(format!("failed to render managed artifact: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn toml_string(value: &str) -> String {
    format!("{value:?}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    lowercase_hex(&Sha256::digest(bytes))
}

fn lowercase_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn rendered_bundle_digest(bundle: &RenderedBundle) -> ManagedBundleDigest {
    let mut hasher = Sha256::new();
    update_bundle_digest(&mut hasher, MANIFEST_FILE.as_bytes(), &bundle.manifest);
    for artifact in &bundle.artifacts {
        update_bundle_digest(&mut hasher, artifact.path.as_bytes(), &artifact.bytes);
    }
    ManagedBundleDigest(lowercase_hex(&hasher.finalize()))
}

fn update_bundle_digest(hasher: &mut Sha256, name: &[u8], bytes: &[u8]) {
    hasher.update(
        u64::try_from(name.len())
            .expect("managed artifact names fit in u64")
            .to_be_bytes(),
    );
    hasher.update(name);
    hasher.update(
        u64::try_from(bytes.len())
            .expect("managed artifacts fit in u64")
            .to_be_bytes(),
    );
    hasher.update(bytes);
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, CliError> {
    let mut file = OpenOptions::new().read(true).open(path).map_err(|error| {
        CliError::Config(format!(
            "failed to read managed artifact {}: {error}",
            path.display()
        ))
    })?;
    let length = file.metadata()?.len();
    if length > maximum {
        return Err(CliError::Config(format!(
            "managed artifact {} exceeds its expected size limit",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn reject_symlink(path: &Path) -> Result<(), CliError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        CliError::Config(format!(
            "failed to inspect managed artifact {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CliError::Config(format!(
            "managed artifact {} must be a regular file, not a symlink",
            path.display()
        )));
    }
    Ok(())
}

fn reject_non_directory_or_symlink(path: &Path) -> Result<(), CliError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        CliError::Config(format!(
            "failed to inspect managed bundle {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CliError::Config(format!(
            "managed bundle {} must be a directory, not a symlink",
            path.display()
        )));
    }
    Ok(())
}

fn bundle_files(root: &Path) -> Result<BTreeSet<&'static str>, CliError> {
    let expected = all_artifact_paths();
    let expected_lookup = expected.iter().copied().collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    let mut visited = 0_usize;
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            visited += 1;
            if visited > MAX_ARTIFACTS * 4 {
                return Err(CliError::Config(
                    "managed bundle contains too many filesystem entries".into(),
                ));
            }
            let entry = entry?;
            let metadata = entry.file_type()?;
            if metadata.is_symlink() {
                return Err(CliError::Config(format!(
                    "managed bundle entry {} must not be a symlink",
                    entry.path().display()
                )));
            }
            if metadata.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !metadata.is_file() {
                return Err(CliError::Config(format!(
                    "managed bundle entry {} must be a regular file",
                    entry.path().display()
                )));
            }
            let path = entry.path();
            let relative = path
                .strip_prefix(root)
                .map_err(|_| CliError::Config("managed bundle entry escaped its root".into()))?;
            if relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(CliError::Config(format!(
                    "managed bundle entry {} has an invalid path",
                    relative.display()
                )));
            }
            let relative = relative.to_string_lossy().replace('\\', "/");
            let canonical = expected_lookup
                .get(relative.as_str())
                .copied()
                .ok_or_else(|| {
                    CliError::Config(format!(
                        "managed bundle contains unexpected artifact {relative}"
                    ))
                })?;
            actual.insert(canonical);
        }
    }
    Ok(actual)
}

fn all_artifact_paths() -> BTreeSet<&'static str> {
    [
        MANIFEST_FILE,
        "codex/plugin-v1/.codex-plugin/plugin.json",
        "codex/plugin-v1/.mcp.json",
        "codex/plugin-v1/hooks/hooks.json",
        "codex/settings-v1/config.toml",
        "claude-code/plugin-v1/.claude-plugin/plugin.json",
        "claude-code/plugin-v1/.mcp.json",
        "claude-code/plugin-v1/hooks/hooks.json",
        "claude-code/settings-v1/managed-settings.json",
        "pi/extension-v1/README.md",
        "pi/extension-v1/index.ts",
        "pi/extension-v1/managed-config.json",
        "pi/extension-v1/package.json",
        "pi/extension-v1/tsconfig.json",
    ]
    .into_iter()
    .collect()
}

fn display_paths(paths: &[&str]) -> String {
    if paths.is_empty() {
        "none".into()
    } else {
        paths.join(", ")
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/managed_tests.rs"]
mod tests;
