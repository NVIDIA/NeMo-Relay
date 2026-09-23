// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::OsString;
use std::sync::MutexGuard;

#[test]
fn source_parser_requires_an_exact_safe_tag() {
    let parsed = GithubSource::parse("github:NVIDIA/NeMo-Relay-Plugins@sample-0.1.0").unwrap();
    assert_eq!(parsed.repo, "NVIDIA/NeMo-Relay-Plugins");
    assert_eq!(parsed.tag, "sample-0.1.0");
    for invalid in [
        "github:NVIDIA/repo",
        "github:NVIDIA/repo@",
        "github:NVIDIA/../repo@tag",
        "github:NVIDIA/repo@--help",
        "https://github.com/NVIDIA/repo",
    ] {
        assert!(GithubSource::parse(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn release_asset_selection_detects_format_and_rejects_ambiguity() {
    let tag = "sample-0.1.0";
    let platform = "linux-x86_64";
    let mut release = Release {
        tag_name: tag.into(),
        is_draft: false,
        assets: vec![ReleaseAsset {
            name: format!("{tag}-{platform}.zip"),
        }],
    };
    assert_eq!(
        select_archive_asset(&release, tag, platform).unwrap(),
        format!("{tag}-{platform}.zip")
    );
    release.assets[0].name = format!("{tag}-{platform}.tar.gz");
    assert_eq!(
        select_archive_asset(&release, tag, platform).unwrap(),
        format!("{tag}-{platform}.tar.gz")
    );
    release.assets.push(ReleaseAsset {
        name: format!("{tag}-{platform}.zip"),
    });
    assert!(select_archive_asset(&release, tag, platform).is_err());
}

#[cfg(unix)]
#[test]
fn global_bundle_permissions_expose_final_bundle_but_keep_staging_private() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("system").join("plugins.toml");
    fs::create_dir(config.parent().unwrap()).unwrap();
    fs::set_permissions(config.parent().unwrap(), fs::Permissions::from_mode(0o700)).unwrap();
    ensure_global_config_directory(&config).unwrap();
    assert_eq!(
        fs::metadata(config.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o755
    );
    let managed = config.parent().unwrap().join("installed-plugins");
    let final_dir = managed.join("final");
    let payload = final_dir.join("payload");
    let bundle = payload.join("bundle");
    let staging = managed.join(".staging-private");
    fs::create_dir_all(&bundle).unwrap();
    fs::create_dir(&staging).unwrap();
    fs::write(final_dir.join(RECEIPT), b"receipt").unwrap();
    for directory in [&managed, &final_dir, &bundle, &staging] {
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
    }
    make_global_bundle_readable(&managed, &final_dir).unwrap();
    for directory in [&managed, &final_dir, &payload, &bundle] {
        assert_eq!(
            fs::metadata(directory).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
    assert_eq!(
        fs::metadata(&staging).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(final_dir.join(RECEIPT))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
}

#[cfg(unix)]
#[test]
fn global_registry_saves_are_readable_even_if_prior_state_was_private() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let scope = ScopedRegistry {
        scope: RegistryScope::Global,
        plugins_toml_path: temp.path().join("plugins.toml"),
        state_path: temp.path().join(".dynamic-plugins.json"),
        registry: nemo_relay::plugin::dynamic::DynamicPluginRegistry::new(),
    };
    scope.save().unwrap();
    assert_eq!(
        fs::metadata(&scope.state_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
    fs::set_permissions(&scope.state_path, fs::Permissions::from_mode(0o600)).unwrap();
    scope.save().unwrap();
    assert_eq!(
        fs::metadata(&scope.state_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
}

#[cfg(unix)]
#[test]
fn global_python_environment_permissions_leave_symlinks_and_attestation_digest_intact() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let temp = tempfile::tempdir().unwrap();
    let system = temp.path().join("system");
    fs::create_dir(&system).unwrap();
    fs::set_permissions(&system, fs::Permissions::from_mode(0o755)).unwrap();
    let parent = system.join(".dynamic-plugin-environments");
    let environment = parent.join("plugin");
    let bin = environment.join("bin");
    let lib = environment.join("lib");
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir(&lib).unwrap();
    let python = bin.join("python");
    let module = lib.join("module.py");
    let outside = temp.path().join("outside");
    fs::write(&python, b"executable").unwrap();
    fs::write(&module, b"module").unwrap();
    fs::write(&outside, b"outside").unwrap();
    let attestation = environment.join(ENVIRONMENT_ATTESTATION_FILE);
    fs::write(&attestation, b"attestation").unwrap();
    symlink(&outside, environment.join("linked-file")).unwrap();
    for directory in [&parent, &environment, &bin, &lib] {
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
    }
    fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
    for file in [&module, &outside] {
        fs::set_permissions(file, fs::Permissions::from_mode(0o600)).unwrap();
    }

    let digest = super::super::environment::environment_tree_digest(&environment).unwrap();
    make_global_environment_readable(&environment).unwrap();
    assert_eq!(
        super::super::environment::environment_tree_digest(&environment).unwrap(),
        digest
    );
    for directory in [&parent, &environment, &bin, &lib] {
        assert_eq!(
            fs::metadata(directory).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
    assert_eq!(
        fs::metadata(&python).unwrap().permissions().mode() & 0o777,
        0o755
    );
    assert_eq!(
        fs::metadata(&module).unwrap().permissions().mode() & 0o777,
        0o644
    );
    assert_eq!(
        fs::metadata(&outside).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(
        fs::symlink_metadata(environment.join("linked-file"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[test]
fn extraction_rejects_unsafe_members() {
    let temp = tempfile::tempdir().unwrap();
    let mut budget = ExtractBudget {
        entries: 0,
        bytes: 0,
        seen: HashSet::new(),
        root: None,
    };
    assert!(
        budget
            .destination(temp.path(), Path::new("bundle/../escape"), 0)
            .is_err()
    );
    assert!(
        budget
            .destination(temp.path(), Path::new("/absolute"), 0)
            .is_err()
    );
    assert!(
        budget
            .destination(temp.path(), Path::new("bundle\\escape"), 0)
            .is_err()
    );
    assert!(!temp.path().join("escape").exists());
    budget
        .destination(temp.path(), Path::new("bundle/first"), 0)
        .unwrap();
    assert!(
        budget
            .destination(temp.path(), Path::new("other/second"), 0)
            .unwrap_err()
            .to_string()
            .contains("multiple bundle roots")
    );
}

#[test]
fn extraction_rejects_links_and_special_files() {
    use std::io::Write;

    let temp = tempfile::tempdir().unwrap();
    for kind in [tar::EntryType::Symlink, tar::EntryType::Link] {
        let archive = temp.path().join(format!("link-{}.tar.gz", kind.as_byte()));
        let encoder = flate2::write::GzEncoder::new(
            File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(kind);
        header.set_size(0);
        builder
            .append_link(&mut header, "sample/link", "target")
            .unwrap();
        builder.into_inner().unwrap().finish().unwrap();
        assert!(
            extract_archive(&archive, &temp.path().join("out"))
                .unwrap_err()
                .to_string()
                .contains("links and special files")
        );
    }

    let archive = temp.path().join("special.tar.gz");
    let encoder = flate2::write::GzEncoder::new(
        File::create(&archive).unwrap(),
        flate2::Compression::default(),
    );
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Block);
    header.set_size(0);
    builder
        .append_data(&mut header, "sample/device", std::io::empty())
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap();
    assert!(
        extract_archive(&archive, &temp.path().join("out"))
            .unwrap_err()
            .to_string()
            .contains("links and special files")
    );

    let archive = temp.path().join("link.zip");
    let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
    writer
        .add_symlink(
            "sample/link",
            "target",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
    writer.finish().unwrap().flush().unwrap();
    assert!(
        extract_archive(&archive, &temp.path().join("out"))
            .unwrap_err()
            .to_string()
            .contains("links and special files")
    );
}

#[test]
fn extraction_rejects_member_size_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    for (data, expected) in [(b"short".as_slice(), 6), (b"longer".as_slice(), 5)] {
        let mut reader = std::io::Cursor::new(data);
        assert!(
            write_bounded(&mut reader, &temp.path().join("member"), expected)
                .unwrap_err()
                .to_string()
                .contains("archive member size mismatch")
        );
    }
}

#[test]
fn uninstall_rejects_a_manifest_reference_in_another_scope() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("installed-plugins").join("managed");
    let manifest = root
        .join("payload")
        .join("bundle")
        .join("relay-plugin.toml");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::write(&manifest, "manifest_version = 1\n").unwrap();
    let other_config = temp.path().join("other").join("plugins.toml");
    fs::create_dir_all(other_config.parent().unwrap()).unwrap();
    fs::write(
        &other_config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {}\n",
            toml::Value::String(manifest.display().to_string())
        ),
    )
    .unwrap();

    let error =
        reject_other_scope_references(&root, &fs::canonicalize(&root).unwrap(), &other_config, &[])
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("still references this managed bundle")
    );
    assert!(manifest.exists());
    assert!(
        fs::read_to_string(&other_config)
            .unwrap()
            .contains("relay-plugin.toml")
    );
}

#[test]
fn zip_extraction_keeps_one_bundle_root() {
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("sample.zip");
    let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    writer
        .start_file("sample/relay-plugin.toml", options)
        .unwrap();
    writer.write_all(b"manifest_version = 1\n").unwrap();
    writer.finish().unwrap();
    let bundle = extract_archive(&archive, &temp.path().join("out")).unwrap();
    assert_eq!(
        fs::read(bundle.join("relay-plugin.toml")).unwrap(),
        b"manifest_version = 1\n"
    );
}

struct FakeGh {
    assets: PathBuf,
    tag: String,
    asset: String,
    release_override: Option<serde_json::Value>,
}

impl GithubRunner for FakeGh {
    fn run(&self, args: &[&str]) -> Result<Vec<u8>, CliError> {
        match args.get(1).copied() {
            Some("view") => Ok(
                serde_json::to_vec(&self.release_override.clone().unwrap_or_else(|| {
                    serde_json::json!({
                        "tagName": self.tag, "isDraft": false,
                        "assets": [
                            {"name": self.asset},
                            {"name": format!("{}.sha256", self.asset)},
                            {"name": format!("{}.json", self.asset)}
                        ]
                    })
                }))
                .unwrap(),
            ),
            Some("download") => {
                let dir = args
                    .windows(2)
                    .find(|pair| pair[0] == "--dir")
                    .map(|pair| pair[1])
                    .unwrap();
                for suffix in ["", ".sha256", ".json"] {
                    fs::copy(
                        self.assets.join(format!("{}{}", self.asset, suffix)),
                        Path::new(dir).join(format!("{}{}", self.asset, suffix)),
                    )?;
                }
                Ok(Vec::new())
            }
            _ => Err(error("unexpected gh command")),
        }
    }
}

#[test]
fn fake_github_release_rejects_invalid_release_and_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let tag = "sample-0.1.0";
    let asset = format!("{tag}-{}.zip", platform().unwrap());
    let archive = temp.path().join(&asset);
    fs::write(&archive, b"archive bytes").unwrap();
    let digest = sha256_file(&archive).unwrap();
    fs::write(
        temp.path().join(format!("{asset}.sha256")),
        format!("{digest}  {asset}\n"),
    )
    .unwrap();
    let valid_metadata = serde_json::json!({
        "name": "sample", "version": "0.1.0", "platform": platform().unwrap(),
        "sha256": digest, "verified": true
    });
    let metadata_path = temp.path().join(format!("{asset}.json"));
    fs::write(&metadata_path, serde_json::to_vec(&valid_metadata).unwrap()).unwrap();
    let mut runner = FakeGh {
        assets: temp.path().to_path_buf(),
        tag: tag.into(),
        asset: asset.clone(),
        release_override: None,
    };
    let fetch = |runner: &FakeGh| {
        let download = tempfile::tempdir().unwrap();
        GithubRelease {
            source: GithubSource::parse(&format!("github:NVIDIA/repo@{tag}")).unwrap(),
            runner,
        }
        .fetch(download.path())
    };

    for release in [
        serde_json::json!({
            "tagName": tag, "isDraft": true,
            "assets": [{"name": asset}]
        }),
        serde_json::json!({
            "tagName": "other-0.1.0", "isDraft": false,
            "assets": [{"name": asset}]
        }),
        serde_json::json!({
            "tagName": tag, "isDraft": false,
            "assets": [
                {"name": asset},
                {"name": format!("{asset}.sha256")},
                {"name": format!("{asset}.sha256")},
                {"name": format!("{asset}.json")}
            ]
        }),
    ] {
        runner.release_override = Some(release);
        assert!(fetch(&runner).is_err());
    }
    runner.release_override = None;

    for (field, replacement) in [
        ("platform", serde_json::json!("wrong-platform")),
        ("sha256", serde_json::json!("wrong-digest")),
        ("verified", serde_json::json!(false)),
    ] {
        let mut metadata = valid_metadata.clone();
        metadata[field] = replacement;
        fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        assert!(
            fetch(&runner)
                .err()
                .unwrap()
                .to_string()
                .contains("release metadata does not match")
        );
    }
}

struct FakeBundleSource<'a> {
    archive: &'a Path,
}

impl BundleSource for FakeBundleSource<'_> {
    fn fetch(&self, download: &Path) -> Result<VerifiedBundle, CliError> {
        let archive = download.join("fixture.tar.gz");
        fs::copy(self.archive, &archive)?;
        Ok(VerifiedBundle {
            sha256: sha256_file(&archive)?,
            archive,
            asset: "fixture.tar.gz".into(),
            tag: "fixture-0.1.0".into(),
        })
    }
}

struct UserConfigEnv {
    _cwd: crate::test_support::CwdTestScope,
    _lock: MutexGuard<'static, ()>,
    previous: Option<OsString>,
    previous_system: Option<PathBuf>,
}

impl UserConfigEnv {
    fn set(dir: &Path) -> Self {
        let cwd = crate::test_support::CwdTestScope::locked();
        let lock = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        let previous_system =
            crate::configuration::set_test_system_config_dir(Some(dir.join("system")));
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", dir);
        }
        Self {
            _cwd: cwd,
            _lock: lock,
            previous,
            previous_system,
        }
    }
}

impl Drop for UserConfigEnv {
    fn drop(&mut self) {
        crate::configuration::set_test_system_config_dir(self.previous_system.take());
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var("XDG_CONFIG_HOME", previous);
            } else {
                std::env::remove_var("XDG_CONFIG_HOME");
            }
        }
    }
}

#[test]
fn fake_github_release_installs_disabled_and_uninstalls_after_remove() {
    let temp = tempfile::tempdir().unwrap();
    let _env = UserConfigEnv::set(&temp.path().join("config"));
    let bundle = temp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    let artifact = b"#!/bin/sh\nexit 0\n";
    fs::write(bundle.join("worker.sh"), artifact).unwrap();
    let artifact_digest = Sha256::digest(artifact)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    fs::write(
        bundle.join("relay-plugin.toml"),
        format!(
            r#"
manifest_version = 1
[plugin]
id = "tests.github_install"
kind = "worker"
[compat]
relay = ">=0.8.0,<1.0"
worker_protocol = "grpc-v1"
[defaults]
enabled = false
[capabilities]
items = ["plugin_worker"]
[source]
artifact = "worker.sh"
[integrity]
sha256 = "sha256:{artifact_digest}"
[load]
runtime = "command"
entrypoint = "worker.sh"
"#
        ),
    )
    .unwrap();
    let mut authored = DynamicPluginManifest::parse_toml(
        &fs::read_to_string(bundle.join("relay-plugin.toml")).unwrap(),
    )
    .unwrap();
    validate_bundle_paths(&authored, &bundle).unwrap();
    authored.source.as_mut().unwrap().artifact = Some("../outside".into());
    assert!(validate_bundle_paths(&authored, &bundle).is_err());
    authored.compat.relay = Some(">=999.0.0".into());
    assert!(check_relay_version(&authored).is_err());
    let tag = "sample-0.1.0";
    let asset = format!("{tag}-{}.tar.gz", platform().unwrap());
    let archive = temp.path().join(&asset);
    let encoder = flate2::write::GzEncoder::new(
        File::create(&archive).unwrap(),
        flate2::Compression::default(),
    );
    let mut builder = tar::Builder::new(encoder);
    builder.append_dir_all("sample", &bundle).unwrap();
    builder.into_inner().unwrap().finish().unwrap();
    let digest = sha256_file(&archive).unwrap();
    fs::write(
        temp.path().join(format!("{asset}.sha256")),
        format!("{digest}  {asset}\n"),
    )
    .unwrap();
    fs::write(temp.path().join(format!("{asset}.json")), serde_json::to_vec(&serde_json::json!({
        "name":"sample", "version":"0.1.0", "platform":platform().unwrap(), "sha256":digest, "verified":true
    })).unwrap()).unwrap();
    let runner = FakeGh {
        assets: temp.path().to_path_buf(),
        tag: tag.into(),
        asset,
        release_override: None,
    };
    let server = GatewayOverrides::default();
    install_with_runner(
        format!("github:NVIDIA/NeMo-Relay-Plugins@{tag}"),
        ConfigurationScope::User,
        true,
        &server,
        &runner,
    )
    .unwrap();
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "tests.github_install")
        .unwrap()
        .unwrap();
    assert!(!record.record.spec.enabled);
    assert_eq!(
        record.record.status.validation.integrity,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(receipt_for(&record).is_some());
    let other_config = temp.path().join("other").join("plugins.toml");
    let other_scope = ScopedRegistry {
        scope: RegistryScope::Global,
        plugins_toml_path: other_config.clone(),
        state_path: temp.path().join("other").join(".dynamic-plugins.json"),
        registry: nemo_relay::plugin::dynamic::DynamicPluginRegistry::from_records(vec![
            record.record.clone(),
        ])
        .unwrap(),
    };
    let root = managed_root(&record.plugins_toml_path)
        .unwrap()
        .join(id_key("tests.github_install"));
    let error = reject_other_scope_references(
        &root,
        &fs::canonicalize(&root).unwrap(),
        &other_config,
        &[other_scope],
    )
    .unwrap_err();
    assert!(error.to_string().contains("live registry record"));
    assert!(root.exists());
    remove_scoped(
        PluginsRemoveRequest {
            id: "tests.github_install".into(),
        },
        ConfigurationScope::User,
        &server,
    )
    .unwrap();
    uninstall(
        "tests.github_install".into(),
        ConfigurationScope::Default,
        &server,
    )
    .unwrap();
    assert!(receipt_for(&record).is_none());

    fs::write(
        temp.path().join(format!("{}.sha256", runner.asset)),
        "bad checksum\n",
    )
    .unwrap();
    let checksum_error = install_with_runner(
        format!("github:NVIDIA/NeMo-Relay-Plugins@{tag}"),
        ConfigurationScope::User,
        true,
        &server,
        &runner,
    )
    .unwrap_err();
    assert!(checksum_error.to_string().contains("checksum mismatch"));
    assert!(
        !managed_root(&config_path(ConfigurationScope::User).unwrap())
            .unwrap()
            .join(id_key("tests.github_install"))
            .exists()
    );

    fs::write(
        temp.path().join(format!("{}.sha256", runner.asset)),
        format!("{digest}  {}\n", runner.asset),
    )
    .unwrap();
    assert_unsigned_activation_is_blocked(tag, &runner, &server);

    install_from_source(
        "fixture:sample@0.1.0".into(),
        ConfigurationScope::User,
        true,
        &server,
        &FakeBundleSource { archive: &archive },
    )
    .unwrap();
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "tests.github_install")
        .unwrap()
        .unwrap();
    let receipt = receipt_for(&record).unwrap();
    assert_eq!(receipt.source, "fixture:sample@0.1.0");
    assert_eq!(receipt.tag, "fixture-0.1.0");
    assert!(
        fs::read_to_string(&record.plugins_toml_path)
            .unwrap()
            .contains("relay-plugin.toml")
    );
    let environment = record
        .state_path
        .parent()
        .unwrap()
        .join(MANAGED_ENVIRONMENTS_DIR)
        .join(id_key("tests.github_install"));
    fs::create_dir_all(&environment).unwrap();
    fs::remove_file(&record.state_path).unwrap();
    fs::remove_file(record.record.source.manifest_ref.as_ref().unwrap()).unwrap();
    uninstall(
        "tests.github_install".into(),
        ConfigurationScope::User,
        &server,
    )
    .unwrap();
    assert!(
        !fs::read_to_string(&record.plugins_toml_path)
            .unwrap()
            .contains("relay-plugin.toml")
    );
    assert!(!environment.exists());
}

fn assert_unsigned_activation_is_blocked(tag: &str, runner: &FakeGh, server: &GatewayOverrides) {
    let activation_error = install_with_runner(
        format!("github:NVIDIA/NeMo-Relay-Plugins@{tag}"),
        ConfigurationScope::User,
        false,
        server,
        runner,
    )
    .unwrap_err();
    assert!(
        activation_error
            .to_string()
            .contains("Installed and registered")
    );
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "tests.github_install")
        .unwrap()
        .unwrap();
    assert!(!record.record.spec.enabled);
    let enable_error = enable(
        PluginsEnableRequest {
            id: "tests.github_install".into(),
        },
        server,
    )
    .unwrap_err();
    assert!(enable_error.to_string().contains("signature"));
    let scopes = load_scoped_registries(None).unwrap();
    assert!(
        !find_record_by_id(&scopes, "tests.github_install")
            .unwrap()
            .unwrap()
            .record
            .spec
            .enabled
    );
    uninstall(
        "tests.github_install".into(),
        ConfigurationScope::User,
        server,
    )
    .unwrap();
}
