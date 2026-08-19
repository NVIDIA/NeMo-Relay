// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;

use super::*;
use crate::test_support::EnvScope;

/// Isolate the three environment variables that steer extension discovery, so a
/// developer's own pi install cannot make these pass or fail.
fn scoped(extension: Option<&OsStr>, agent_dir: Option<&OsStr>) -> EnvScope {
    EnvScope::set(&[
        (PI_EXTENSION_PATH_ENV, extension),
        (PI_AGENT_DIR_ENV, agent_dir),
    ])
}

#[test]
fn a_project_scoped_extension_is_reported_because_pi_will_not_say_so() {
    let temp = tempfile::tempdir().unwrap();
    let project_extensions = temp.path().join(".pi").join("extensions");
    std::fs::create_dir_all(&project_extensions).unwrap();
    std::fs::write(project_extensions.join("nemo-relay.ts"), "export default 1").unwrap();
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));
    let sites = extension_sites(temp.path());

    // This is the whole point of the check: pi drops this extension with a bare
    // conditional in every non-interactive mode, never reports it, and the
    // extension cannot report it either because it is not running.
    assert!(
        sites
            .iter()
            .any(|site| site.scope == ExtensionScope::Project),
        "a project-scoped extension must be reported: {sites:?}"
    );
}

#[test]
fn an_empty_project_directory_is_not_reported() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".pi").join("extensions")).unwrap();
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));

    // A `.pi/extensions` directory that pi created and nothing was ever put in
    // is not a finding; warning about it would train users to ignore the check.
    assert!(
        extension_sites(temp.path()).is_empty(),
        "an empty project extensions directory must not be reported"
    );
}

#[test]
fn the_explicit_path_is_reported_as_ungated() {
    let temp = tempfile::tempdir().unwrap();
    let entry = temp.path().join("index.ts");
    std::fs::write(&entry, "export default 1").unwrap();
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(Some(entry.as_os_str()), Some(empty_home.as_os_str()));
    let sites = extension_sites(temp.path());

    // `-e` loads first in pi's precedence order and survives `--no-extensions`,
    // so an extension reached this way is never subject to project trust --
    // which is exactly why the launcher uses it.
    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::Explicit);
    assert_eq!(sites[0].path, entry);
}

#[test]
fn a_user_scope_install_is_reported_as_ungated() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let user_extensions = agent_dir.join("extensions");
    std::fs::create_dir_all(&user_extensions).unwrap();
    std::fs::write(user_extensions.join("nemo-relay.ts"), "export default 1").unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = extension_sites(temp.path());

    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::User);
}

#[test]
fn an_explicit_path_that_does_not_exist_is_not_reported() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("gone.ts");
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(Some(missing.as_os_str()), Some(empty_home.as_os_str()));

    // A stale environment variable is worse than none: it would report an
    // ungated load path for a file pi cannot read.
    assert!(extension_sites(temp.path()).is_empty());
    assert!(!extension_configured());
}

#[test]
fn the_gateway_url_matches_what_the_extension_resolves() {
    let _env = EnvScope::set(&[(PI_GATEWAY_URL_ENV, None)]);
    // Kept in step with `configFromEnv` in integrations/pi/src/gateway-client.ts;
    // a drift here would probe an endpoint the extension never posts to.
    assert_eq!(gateway_url(None), "http://127.0.0.1:4040");
}

#[test]
fn the_gateway_url_honors_the_launcher_variable_and_strips_trailing_slashes() {
    let _env = EnvScope::set(&[(
        PI_GATEWAY_URL_ENV,
        Some(OsStr::new("http://gateway.test:9999///")),
    )]);
    assert_eq!(gateway_url(None), "http://gateway.test:9999");
}

#[test]
fn the_gateway_url_follows_a_configured_bind_rather_than_the_default_port() {
    let _env = EnvScope::set(&[(PI_GATEWAY_URL_ENV, None)]);
    // The launcher sets the environment variable *from* the resolved config, so a
    // preflight that only read the variable would report a working gateway as down
    // for anyone who changed `bind`.
    assert_eq!(
        gateway_url(Some("127.0.0.1:8123".parse().unwrap())),
        "http://127.0.0.1:8123"
    );
}

#[test]
fn a_wildcard_bind_is_probed_on_loopback_where_pi_actually_runs() {
    let _env = EnvScope::set(&[(PI_GATEWAY_URL_ENV, None)]);
    // `http://0.0.0.0:4040` is not a dialable address; the gateway bound that way is
    // reachable on loopback, which is where the pi extension posts from.
    assert_eq!(
        gateway_url(Some("0.0.0.0:4040".parse().unwrap())),
        "http://127.0.0.1:4040"
    );
}

#[test]
fn the_environment_variable_wins_over_a_configured_bind() {
    let _env = EnvScope::set(&[(
        PI_GATEWAY_URL_ENV,
        Some(OsStr::new("http://elsewhere:1234")),
    )]);
    // Someone who set the variable by hand is pointing pi somewhere deliberately.
    assert_eq!(
        gateway_url(Some("127.0.0.1:4040".parse().unwrap())),
        "http://elsewhere:1234"
    );
}
