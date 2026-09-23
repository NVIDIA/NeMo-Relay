// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-release downloads and owned bundle installation for dynamic plugins.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use flate2::read::GzDecoder;
use nemo_relay::plugin::dynamic::DynamicPluginCheckState;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::*;
use crate::configuration::{global_plugin_config_path, user_plugin_config_path};
use crate::plugins::ConfigurationScope;
use crate::plugins::config_io::remove_dynamic_plugin_reference_path;

const RECEIPT: &str = ".install.json";
const MANAGED_DIR: &str = "installed-plugins";
const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_UNPACKED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;

#[derive(Debug, Clone)]
struct GithubSource {
    repo: String,
    tag: String,
}

/// A source supplies a verified archive; installation owns extraction and lifecycle state.
trait BundleSource {
    fn fetch(&self, download: &Path) -> Result<VerifiedBundle, CliError>;
}

struct VerifiedBundle {
    archive: PathBuf,
    asset: String,
    tag: String,
    sha256: String,
}

struct GithubRelease<'a> {
    source: GithubSource,
    runner: &'a dyn GithubRunner,
}

impl GithubSource {
    fn parse(source: &str) -> Result<Self, CliError> {
        let rest = source
            .strip_prefix("github:")
            .ok_or_else(|| error("expected github:<owner>/<repo>@<tag>"))?;
        let (repo, tag) = rest
            .rsplit_once('@')
            .ok_or_else(|| error("source must include an exact release tag"))?;
        let mut parts = repo.split('/');
        let owner = parts.next().unwrap_or_default();
        let name = parts.next().unwrap_or_default();
        if parts.next().is_some()
            || !safe_segment(owner)
            || !safe_segment(name)
            || !safe_segment(tag)
        {
            return Err(error(
                "invalid GitHub source; expected github:<owner>/<repo>@<tag> using letters, digits, '.', '_' or '-'",
            ));
        }
        Ok(Self {
            repo: repo.to_owned(),
            tag: tag.to_owned(),
        })
    }
}

fn safe_segment(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('-')
        && !value.contains("..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct ManagedReceipt {
    schema_version: u32,
    plugin_id: String,
    scope: String,
    pub(super) source: String,
    pub(super) tag: String,
    asset: String,
    sha256: String,
    owned_directory: String,
}

#[derive(Deserialize)]
struct Release {
    #[serde(rename = "tagName")]
    tag_name: String,
    #[serde(rename = "isDraft")]
    is_draft: bool,
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
}

trait GithubRunner {
    fn run(&self, args: &[&str]) -> Result<Vec<u8>, CliError>;
}

impl BundleSource for GithubRelease<'_> {
    fn fetch(&self, download: &Path) -> Result<VerifiedBundle, CliError> {
        let release_bytes = self.runner.run(&[
            "release",
            "view",
            &self.source.tag,
            "--repo",
            &self.source.repo,
            "--json",
            "tagName,isDraft,assets",
        ])?;
        let release: Release = serde_json::from_slice(&release_bytes)
            .map_err(|err| error(format!("invalid GitHub release response: {err}")))?;
        if release.is_draft || release.tag_name != self.source.tag {
            return Err(error(
                "release is draft or tag does not match the requested exact tag",
            ));
        }
        let platform = platform()?;
        let asset = select_archive_asset(&release, &self.source.tag, platform)?;
        let checksum_name = format!("{asset}.sha256");
        let metadata_name = format!("{asset}.json");
        for name in [&asset, &checksum_name, &metadata_name] {
            if release
                .assets
                .iter()
                .filter(|entry| entry.name == *name)
                .count()
                != 1
            {
                return Err(error(format!(
                    "release must contain exactly one {name} asset"
                )));
            }
        }
        let download_str = download
            .to_str()
            .ok_or_else(|| error("download path is not UTF-8"))?;
        self.runner.run(&[
            "release",
            "download",
            &self.source.tag,
            "--repo",
            &self.source.repo,
            "--dir",
            download_str,
            "--pattern",
            &asset,
            "--pattern",
            &checksum_name,
            "--pattern",
            &metadata_name,
        ])?;
        let archive = download.join(&asset);
        if fs::metadata(&archive)?.len() > MAX_ARCHIVE_BYTES {
            return Err(error("plugin archive exceeds 512 MiB"));
        }
        let digest = sha256_file(&archive)?;
        let checksum = fs::read_to_string(download.join(&checksum_name))?;
        if checksum != format!("{digest}  {asset}\n") {
            return Err(error("plugin archive checksum mismatch"));
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(download.join(&metadata_name))?)
                .map_err(|err| error(format!("invalid release metadata: {err}")))?;
        let metadata_tag = format!(
            "{}-{}",
            metadata["name"].as_str().unwrap_or_default(),
            metadata["version"].as_str().unwrap_or_default()
        );
        if metadata_tag != self.source.tag
            || metadata["platform"] != platform
            || metadata["sha256"] != digest
            || metadata["verified"] != true
        {
            return Err(error(
                "release metadata does not match the requested tag, platform, or archive digest",
            ));
        }
        Ok(VerifiedBundle {
            archive,
            asset,
            tag: self.source.tag.clone(),
            sha256: digest,
        })
    }
}

fn select_archive_asset(release: &Release, tag: &str, platform: &str) -> Result<String, CliError> {
    let zip = format!("{tag}-{platform}.zip");
    let tarball = format!("{tag}-{platform}.tar.gz");
    let matches = release
        .assets
        .iter()
        .filter(|asset| asset.name == zip || asset.name == tarball)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [asset] => Ok(asset.name.clone()),
        _ => Err(error(format!(
            "release must contain exactly one ZIP or tar.gz archive for {platform}"
        ))),
    }
}

struct Gh;

impl GithubRunner for Gh {
    fn run(&self, args: &[&str]) -> Result<Vec<u8>, CliError> {
        let output = Command::new("gh")
            .args(args)
            .env_remove("GH_DEBUG")
            .env_remove("DEBUG")
            .output()
            .map_err(|err| {
                error(format!(
                    "could not run GitHub CLI; install gh and authenticate: {err}"
                ))
            })?;
        if !output.status.success() {
            let mut message = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            for name in ["GH_TOKEN", "GITHUB_TOKEN"] {
                if let Ok(secret) = std::env::var(name)
                    && !secret.is_empty()
                {
                    message = message.replace(&secret, "[redacted]");
                }
            }
            return Err(error(format!("GitHub CLI failed: {}", message)));
        }
        Ok(output.stdout)
    }
}

fn error(message: impl Into<String>) -> CliError {
    CliError::Config(message.into())
}

pub(crate) fn install(
    source: String,
    scope: ConfigurationScope,
    no_enable: bool,
    server: &GatewayOverrides,
) -> Result<(), CliError> {
    install_with_runner(source, scope, no_enable, server, &Gh)
}

fn install_with_runner(
    source: String,
    scope: ConfigurationScope,
    no_enable: bool,
    server: &GatewayOverrides,
    runner: &impl GithubRunner,
) -> Result<(), CliError> {
    let source_ref = match source.split_once(':').map(|(kind, _)| kind) {
        Some("github") => GithubSource::parse(&source)?,
        _ => {
            return Err(error(
                "unsupported plugin source; expected github:<owner>/<repo>@<tag>",
            ));
        }
    };
    install_from_source(
        source,
        scope,
        no_enable,
        server,
        &GithubRelease {
            source: source_ref,
            runner,
        },
    )
}

fn install_from_source(
    source: String,
    scope: ConfigurationScope,
    no_enable: bool,
    server: &GatewayOverrides,
    bundle_source: &impl BundleSource,
) -> Result<(), CliError> {
    if matches!(scope, ConfigurationScope::Invalid) {
        return Err(error("choose only one of --user or --global"));
    }
    if lifecycle_plugin_config_path(server).is_some() {
        return Err(error(
            "plugins install uses --user or --global; --config and --plugin-config-path are not supported",
        ));
    }
    let config_path = config_path(scope)?;
    if scope == ConfigurationScope::Global {
        ensure_global_config_directory(&config_path)?;
    }
    let managed = managed_root(&config_path)?;
    fs::create_dir_all(&managed)?;
    let staging = managed.join(format!(".staging-{}", uuid::Uuid::now_v7()));
    fs::create_dir(&staging)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o700))?;
    }
    let mut guard = Cleanup(Some(staging.clone()));
    let download = staging.join("download");
    fs::create_dir(&download)?;
    let VerifiedBundle {
        archive,
        asset,
        tag,
        sha256: digest,
    } = bundle_source.fetch(&download)?;
    let payload = staging.join("payload");
    fs::create_dir(&payload)?;
    let bundle = extract_archive(&archive, &payload)?;
    let manifest_path = bundle.join("relay-plugin.toml");
    let (manifest, _) = load_manifest_for_action("install", &manifest_path)?;
    validate_bundle_paths(&manifest, &bundle)?;
    let id = manifest.plugin.id.trim().to_owned();
    check_relay_version(&manifest)?;
    load_config_schema_for_manifest(
        &manifest,
        manifest_path
            .to_str()
            .ok_or_else(|| error("manifest path is not UTF-8"))?,
    )?;
    let policy = evaluate_dynamic_plugin_host_policy(
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
        &manifest,
    );
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_path.to_string_lossy(), &policy);
    if trust.integrity != DynamicPluginCheckState::Valid {
        return Err(error(
            trust
                .failure()
                .map(|failure| failure.display(&id).to_string())
                .unwrap_or_else(|| "plugin artifact failed integrity verification".into()),
        ));
    }
    let scopes = load_scoped_registries(None)?;
    if find_record_by_id(&scopes, &id)?.is_some_and(|record| !record.record.is_tombstoned()) {
        return Err(error(format!(
            "dynamic plugin '{id}' is already registered"
        )));
    }
    let final_dir = managed.join(id_key(&id));
    if final_dir.exists() {
        return Err(error(format!(
            "managed installation for '{id}' already exists; uninstall it first"
        )));
    }
    let receipt = ManagedReceipt {
        schema_version: 1,
        plugin_id: id.clone(),
        scope: scope_name(scope).into(),
        source,
        tag,
        asset,
        sha256: digest,
        owned_directory: final_dir.display().to_string(),
    };
    fs::write(
        staging.join(RECEIPT),
        serde_json::to_vec_pretty(&receipt).map_err(|err| error(err.to_string()))?,
    )?;
    fs::remove_dir_all(&download)?;
    fs::rename(&staging, &final_dir)?;
    guard.0 = Some(final_dir.clone());
    if scope == ConfigurationScope::Global {
        make_global_bundle_readable(&managed, &final_dir)?;
    }
    let final_manifest = final_dir
        .join("payload")
        .join(
            bundle
                .file_name()
                .ok_or_else(|| error("bundle root has no filename"))?,
        )
        .join("relay-plugin.toml");
    add_verified_install(
        PluginsAddRequest {
            scope,
            path: final_manifest.clone(),
        },
        server,
    )?;
    guard.0 = None;
    if !no_enable {
        let mut resolved = resolve_plugins_config_with_path(None, None)?;
        crate::plugins::policy::apply_secure_runtime_defaults(&mut resolved.dynamic_plugin_policy);
        let policy =
            evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
        let trust =
            evaluate_dynamic_plugin_trust(&manifest, &final_manifest.to_string_lossy(), &policy);
        if !policy.policy_satisfied || !trust.is_satisfied() {
            let reason = policy
                .failure()
                .map(|failure| failure.display(&id).to_string())
                .or_else(|| {
                    trust
                        .failure()
                        .map(|failure| failure.display(&id).to_string())
                })
                .unwrap_or_else(|| "host policy blocks activation".into());
            return Err(error(format!(
                "Installed and registered '{id}' disabled; activation failed: {reason}. Configure plugin trust, then run `nemo-relay plugins enable {id}`"
            )));
        }
        if let Err(err) = enable(PluginsEnableRequest { id: id.clone() }, server) {
            return Err(error(format!(
                "Installed and registered '{id}' disabled; activation failed: {err}. Configure plugin trust, then run `nemo-relay plugins enable {id}`"
            )));
        }
    }
    println!("Installed dynamic plugin {id} from {}", receipt.source);
    Ok(())
}

pub(crate) fn uninstall(
    id: String,
    scope: ConfigurationScope,
    server: &GatewayOverrides,
) -> Result<(), CliError> {
    let scope = if scope == ConfigurationScope::Default {
        ConfigurationScope::User
    } else {
        scope
    };
    if scope == ConfigurationScope::Invalid {
        return Err(error("choose only one of --user or --global"));
    }
    if lifecycle_plugin_config_path(server).is_some() {
        return Err(error(
            "plugins uninstall does not support --config or --plugin-config-path",
        ));
    }
    let config = config_path(scope)?;
    let root = managed_root(&config)?.join(id_key(&id));
    let receipt = read_receipt(&root)?.ok_or_else(|| {
        error(format!(
            "'{id}' is not a CLI-managed plugin in the selected scope"
        ))
    })?;
    if receipt.plugin_id != id || receipt.scope != scope_name(scope) {
        return Err(error("managed install receipt identity mismatch"));
    }
    verify_owned_root(&root, &receipt)?;
    let canonical_root = fs::canonicalize(&root)?;
    let other_config = match scope {
        ConfigurationScope::User => Some(global_plugin_config_path()),
        ConfigurationScope::Global => user_plugin_config_path(),
        _ => None,
    };
    let scopes = load_scoped_registries_matching(None, |registry_scope| {
        scope_matches(registry_scope, scope)
    })?;
    if let Some(other_config) = other_config {
        // Only declarations in the other scope can activate a plugin. Its
        // registry may be private (for example, a root-owned 0600 system state
        // file), so still check declarations when registry state is unreadable.
        // Other I/O and parse errors still fail closed.
        let other_scopes = match load_scoped_registries_matching(None, |registry_scope| {
            !scope_matches(registry_scope, scope)
        }) {
            Ok(scopes) => scopes,
            Err(CliError::Io(err)) if err.kind() == io::ErrorKind::PermissionDenied => Vec::new(),
            Err(err) => return Err(err),
        };
        reject_other_scope_references(&root, &canonical_root, &other_config, &other_scopes)?;
    }
    let selected = scopes
        .iter()
        .find(|item| item.plugins_toml_path == config)
        .ok_or_else(|| error("selected plugin lifecycle scope is unavailable"))?;
    let entry = selected.registry.get(&id);
    if let Some(record) = entry
        && !record.is_tombstoned()
    {
        let manifest = record
            .source
            .manifest_ref
            .as_deref()
            .ok_or_else(|| error("registered plugin has no manifest reference"))?;
        if !is_owned_manifest_ref(&root, &canonical_root, Path::new(manifest)) {
            return Err(error(
                "registered plugin does not belong to this managed installation",
            ));
        }
        remove_scoped(PluginsRemoveRequest { id: id.clone() }, scope, server)?;
    }
    // Registry state can be lost independently of plugins.toml. Remove only
    // references to files in this receipt-owned bundle before deleting it.
    for manifest_ref in dynamic_manifest_refs(&config)? {
        if is_owned_manifest_ref(&root, &canonical_root, &manifest_ref) {
            remove_dynamic_plugin_reference_path(&config, &manifest_ref)?;
        }
    }
    remove_managed_environment_for_plugin(&selected.state_path, &id).map_err(error)?;
    fs::remove_dir_all(&root)?;
    println!("Uninstalled dynamic plugin {id}");
    Ok(())
}

fn reject_other_scope_references(
    root: &Path,
    canonical_root: &Path,
    other_config: &Path,
    scopes: &[ScopedRegistry],
) -> Result<(), CliError> {
    for manifest_ref in dynamic_manifest_refs(other_config)? {
        if is_owned_manifest_ref(root, canonical_root, &manifest_ref) {
            return Err(error(format!(
                "cannot uninstall: {} still references this managed bundle",
                other_config.display()
            )));
        }
    }
    for other_scope in scopes
        .iter()
        .filter(|scope| scope.plugins_toml_path == other_config)
    {
        for record in other_scope.registry.cloned_records(false) {
            if record
                .source
                .manifest_ref
                .as_deref()
                .is_some_and(|manifest| {
                    is_owned_manifest_ref(root, canonical_root, Path::new(manifest))
                })
            {
                return Err(error(format!(
                    "cannot uninstall: {} still has a live registry record for this managed bundle",
                    other_config.display()
                )));
            }
        }
    }
    Ok(())
}

fn is_owned_manifest_ref(root: &Path, canonical_root: &Path, manifest_ref: &Path) -> bool {
    match fs::canonicalize(manifest_ref) {
        Ok(path) => path.starts_with(canonical_root),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let relative = manifest_ref
                .strip_prefix(canonical_root)
                .or_else(|_| manifest_ref.strip_prefix(root));
            let Ok(relative) = relative else {
                return false;
            };
            let parts = relative.components().collect::<Vec<_>>();
            matches!(
                parts.as_slice(),
                [Component::Normal(payload), Component::Normal(_), Component::Normal(manifest)]
                    if *payload == "payload" && *manifest == "relay-plugin.toml"
            )
        }
        Err(_) => false,
    }
}

pub(super) fn receipt_for(entry: &ScopedDynamicPluginRecord) -> Option<ManagedReceipt> {
    let root = managed_root(&entry.plugins_toml_path)
        .ok()?
        .join(id_key(&entry.record.metadata.id));
    let receipt = read_receipt(&root).ok().flatten()?;
    if receipt.plugin_id != entry.record.metadata.id || receipt.scope != entry.scope.to_string() {
        return None;
    }
    verify_owned_root(&root, &receipt).ok()?;
    let manifest = fs::canonicalize(entry.record.source.manifest_ref.as_deref()?).ok()?;
    if !manifest.starts_with(fs::canonicalize(root).ok()?) {
        return None;
    }
    Some(receipt)
}

fn config_path(scope: ConfigurationScope) -> Result<PathBuf, CliError> {
    match scope {
        ConfigurationScope::Default | ConfigurationScope::User => {
            user_plugin_config_path().ok_or_else(|| error("cannot determine user config directory"))
        }
        ConfigurationScope::Global => Ok(global_plugin_config_path()),
        ConfigurationScope::Invalid => Err(error("choose only one of --user or --global")),
    }
}

fn scope_name(scope: ConfigurationScope) -> &'static str {
    if scope == ConfigurationScope::Global {
        "global"
    } else {
        "user"
    }
}

fn managed_root(config: &Path) -> Result<PathBuf, CliError> {
    Ok(config
        .parent()
        .ok_or_else(|| error("plugin config has no parent directory"))?
        .join(MANAGED_DIR))
}

fn ensure_global_config_directory(config: &Path) -> Result<(), CliError> {
    let directory = config
        .parent()
        .ok_or_else(|| error("plugin config has no parent directory"))?;
    fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

#[cfg(unix)]
fn make_global_bundle_readable(managed: &Path, final_dir: &Path) -> Result<(), CliError> {
    use std::os::unix::fs::PermissionsExt;

    let directory_permissions = fs::Permissions::from_mode(0o755);
    fs::set_permissions(managed, directory_permissions.clone())?;
    let mut pending = vec![final_dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        fs::set_permissions(&directory, directory_permissions.clone())?;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    fs::set_permissions(final_dir.join(RECEIPT), fs::Permissions::from_mode(0o644))?;
    Ok(())
}

#[cfg(not(unix))]
fn make_global_bundle_readable(_managed: &Path, _final_dir: &Path) -> Result<(), CliError> {
    Ok(())
}

fn validate_bundle_paths(manifest: &DynamicPluginManifest, bundle: &Path) -> Result<(), CliError> {
    let check = |value: &str, directory: bool| -> Result<(), CliError> {
        let path = Path::new(value);
        if path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(error(format!(
                "bundle manifest path '{value}' must remain inside the bundle"
            )));
        }
        let root = fs::canonicalize(bundle)?;
        let resolved = fs::canonicalize(bundle.join(path)).map_err(|err| {
            error(format!(
                "bundle manifest path '{value}' is unavailable: {err}"
            ))
        })?;
        if !resolved.starts_with(&root)
            || if directory {
                !resolved.is_dir()
            } else {
                !resolved.is_file()
            }
        {
            return Err(error(format!(
                "bundle manifest path '{value}' is not a bundled {}",
                if directory { "directory" } else { "file" }
            )));
        }
        Ok(())
    };
    if let Some(source) = manifest.source.as_ref() {
        if let Some(path) = source.manifest_root.as_deref() {
            check(path, true)?;
        }
        if let Some(path) = source.artifact.as_deref() {
            check(path, false)?;
        }
    }
    if let Some(schema) = manifest.config_schema.as_ref() {
        check(&schema.path, false)?;
    }
    if let Some(signature) = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
    {
        check(signature, false)?;
    }
    match &manifest.load {
        DynamicPluginManifestLoad::RustDynamic(load) => {
            if let Some(path) = load.library.as_deref() {
                check(path, false)?;
            }
        }
        DynamicPluginManifestLoad::Worker(load) if load.runtime != Some(WorkerRuntime::Python) => {
            if let Some(path) = load.entrypoint.as_deref() {
                check(path, false)?;
            }
        }
        DynamicPluginManifestLoad::Worker(_) => {}
    }
    Ok(())
}

fn check_relay_version(manifest: &DynamicPluginManifest) -> Result<(), CliError> {
    let version =
        Version::parse(env!("CARGO_PKG_VERSION")).map_err(|err| error(err.to_string()))?;
    let range = manifest.compat.relay.as_deref().unwrap_or_default();
    let requirement = VersionReq::parse(range).map_err(|err| error(err.to_string()))?;
    if !requirement.matches(&version) {
        return Err(error(format!(
            "plugin '{}' requires Relay {range}, running version is {version}",
            manifest.plugin.id
        )));
    }
    Ok(())
}

fn id_key(id: &str) -> String {
    Sha256::digest(id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_receipt(root: &Path) -> Result<Option<ManagedReceipt>, CliError> {
    let path = root.join(RECEIPT);
    if !path.exists() {
        return Ok(None);
    }
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() > 64 * 1024 {
        return Err(error(
            "managed install receipt must be a regular file under 64 KiB",
        ));
    }
    let receipt = serde_json::from_slice(&fs::read(path)?)
        .map_err(|err| error(format!("invalid managed install receipt: {err}")))?;
    Ok(Some(receipt))
}

fn verify_owned_root(root: &Path, receipt: &ManagedReceipt) -> Result<(), CliError> {
    if receipt.schema_version != 1
        || Path::new(&receipt.owned_directory) != root
        || fs::symlink_metadata(root)?.file_type().is_symlink()
    {
        return Err(error("managed install receipt is not owned by this scope"));
    }
    let parent = fs::canonicalize(
        root.parent()
            .ok_or_else(|| error("managed directory has no parent"))?,
    )?;
    let actual = fs::canonicalize(root)?;
    if actual.parent() != Some(parent.as_path()) {
        return Err(error("managed install path escapes its scope"));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, CliError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn platform() -> Result<&'static str, CliError> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("linux", "aarch64") => Ok("linux-arm64"),
        ("macos", "aarch64") => Ok("macos-arm64"),
        ("windows", "x86_64") => Ok("windows-x86_64"),
        ("windows", "aarch64") => Ok("windows-arm64"),
        _ => Err(error(
            "no published plugin bundle platform matches this host",
        )),
    }
}

struct ExtractBudget {
    entries: usize,
    bytes: u64,
    seen: HashSet<PathBuf>,
    root: Option<String>,
}

impl ExtractBudget {
    fn destination(&mut self, base: &Path, raw: &Path, size: u64) -> Result<PathBuf, CliError> {
        let parts = raw.components().collect::<Vec<_>>();
        if parts.is_empty()
            || parts.len() > 128
            || raw.to_string_lossy().contains('\\')
            || parts
                .iter()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(error("unsafe archive path"));
        }
        let first = parts[0].as_os_str().to_string_lossy().to_string();
        if self.root.get_or_insert(first.clone()) != &first {
            return Err(error("archive contains multiple bundle roots"));
        }
        if !self.seen.insert(raw.to_path_buf()) {
            return Err(error("duplicate archive member"));
        }
        self.entries += 1;
        self.bytes = self
            .bytes
            .checked_add(size)
            .ok_or_else(|| error("archive size overflow"))?;
        if self.entries > MAX_ENTRIES || self.bytes > MAX_UNPACKED_BYTES {
            return Err(error("plugin bundle exceeds extraction limits"));
        }
        Ok(base.join(raw))
    }
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<PathBuf, CliError> {
    let mut budget = ExtractBudget {
        entries: 0,
        bytes: 0,
        seen: HashSet::new(),
        root: None,
    };
    if archive.extension().is_some_and(|ext| ext == "zip") {
        let mut zip =
            zip::ZipArchive::new(File::open(archive)?).map_err(|err| error(err.to_string()))?;
        for index in 0..zip.len() {
            let mut entry = zip.by_index(index).map_err(|err| error(err.to_string()))?;
            let name = entry.name().trim_end_matches('/');
            let mode = entry.unix_mode().unwrap_or(0);
            if !matches!(mode & 0o170000, 0 | 0o040000 | 0o100000) {
                return Err(error("archive links and special files are not allowed"));
            }
            if mode & 0o170000 == 0o040000 && !entry.is_dir() {
                return Err(error("archive directory metadata disagrees with its path"));
            }
            let path = budget.destination(destination, Path::new(name), entry.size())?;
            if entry.is_dir() {
                fs::create_dir_all(&path)?;
            } else {
                fs::create_dir_all(
                    path.parent()
                        .ok_or_else(|| error("archive member has no parent"))?,
                )?;
                let size = entry.size();
                write_bounded(&mut entry, &path, size)?;
                set_executable(&path, mode)?;
            }
        }
    } else {
        let decoder = GzDecoder::new(File::open(archive)?);
        let mut tar = tar::Archive::new(decoder);
        for entry in tar.entries().map_err(|err| error(err.to_string()))? {
            let mut entry = entry.map_err(|err| error(err.to_string()))?;
            let kind = entry.header().entry_type();
            if !kind.is_file() && !kind.is_dir() {
                return Err(error("archive links and special files are not allowed"));
            }
            let raw = entry
                .path()
                .map_err(|err| error(err.to_string()))?
                .into_owned();
            let size = entry.size();
            let mode = entry
                .header()
                .mode()
                .map_err(|err| error(err.to_string()))?;
            let path = budget.destination(destination, &raw, size)?;
            if kind.is_dir() {
                fs::create_dir_all(&path)?;
            } else {
                fs::create_dir_all(
                    path.parent()
                        .ok_or_else(|| error("archive member has no parent"))?,
                )?;
                write_bounded(&mut entry, &path, size)?;
                set_executable(&path, mode)?;
            }
        }
    }
    let root = budget
        .root
        .ok_or_else(|| error("plugin archive is empty"))?;
    let bundle = destination.join(root);
    if !bundle.is_dir() || !bundle.join("relay-plugin.toml").is_file() {
        return Err(error(
            "archive must contain one bundle root with relay-plugin.toml",
        ));
    }
    Ok(bundle)
}

fn write_bounded(reader: &mut impl Read, path: &Path, expected: u64) -> Result<(), CliError> {
    let mut file = File::create(path)?;
    let actual = io::copy(&mut reader.take(expected.saturating_add(1)), &mut file)?;
    if actual != expected {
        return Err(error("archive member size mismatch"));
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path, mode: u32) -> Result<(), CliError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(
        path,
        fs::Permissions::from_mode(if mode & 0o111 != 0 { 0o755 } else { 0o644 }),
    )?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path, _mode: u32) -> Result<(), CliError> {
    Ok(())
}

struct Cleanup(Option<PathBuf>);
impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/shared/plugins_installation_tests.rs"]
mod tests;
