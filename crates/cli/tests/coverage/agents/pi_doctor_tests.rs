// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;

use super::*;
use crate::test_support::EnvScope;

/// Isolate the three environment variables that steer extension discovery, so a
/// developer's own pi install cannot make these pass or fail.
/// A directory pi would see as this extension: a package manifest naming it.
fn write_relay_package(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("package.json"), r#"{"name": "nemo-relay-pi"}"#).unwrap();
    std::fs::write(dir.join("index.ts"), "export default 1").unwrap();
}

/// Somebody else's pi extension, installed the same way.
fn write_other_package(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("package.json"),
        r#"{"name": "someone-elses-pi-thing"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("index.ts"), "export default 1").unwrap();
}

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
    write_relay_package(&project_extensions.join("nemo-relay"));
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));
    let sites = relay_extension_sites(temp.path());

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
        relay_extension_sites(temp.path()).is_empty(),
        "an empty project extensions directory must not be reported"
    );
}

#[test]
fn the_explicit_path_is_reported_as_ungated() {
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("checkout");
    write_relay_package(&package);
    let entry = package.join("index.ts");
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(Some(entry.as_os_str()), Some(empty_home.as_os_str()));
    let sites = relay_extension_sites(temp.path());

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
    write_relay_package(&user_extensions.join("nemo-relay"));

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

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
    assert!(relay_extension_sites(temp.path()).is_empty());
    assert!(!extension_configured());
}

// `pi install` writes nothing into the extension directories -- it appends the
// source to a `packages` array in settings.json. Scanning only the directories
// therefore told a user who had just run the *recommended* install command that
// no extension was found.
#[test]
fn a_pi_install_at_user_scope_is_found_in_settings_not_in_a_directory() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("checkout"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": ["../checkout"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::User);
}

// The dangerous half: `pi install --local` records the package in the project's
// settings, which is trust-gated exactly like `.pi/extensions`. Before this, the
// check could not see it at all.
#[test]
fn a_local_pi_install_is_reported_as_project_scoped() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".pi")).unwrap();
    write_relay_package(&temp.path().join("checkout"));
    std::fs::write(
        temp.path().join(".pi").join("settings.json"),
        r#"{"packages": ["../checkout"]}"#,
    )
    .unwrap();
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert!(
        sites
            .iter()
            .any(|site| site.scope == ExtensionScope::Project),
        "a --local install is trust-gated and must be reported: {sites:?}"
    );
}

#[test]
fn settings_without_packages_are_not_a_finding() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    let _env = scoped(None, Some(agent_dir.as_os_str()));

    // Every shape pi can leave behind that means "nothing installed", plus a
    // malformed file: a parse failure is pi's to report, not doctor's to fail on.
    for body in [
        r#"{}"#,
        r#"{"packages": []}"#,
        r#"{"packages": "not-an-array"}"#,
        "{ truncated",
    ] {
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();
        assert!(
            relay_extension_sites(temp.path()).is_empty(),
            "settings body {body} must not be reported as an install"
        );
    }
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

// The check is about *this* extension, not about pi having extensions. Before it
// matched on the package name, an unrelated install produced a Relay Pass -- and a
// project-scoped one produced a Relay trust warning about a file that has nothing
// to do with Relay.
#[test]
fn somebody_elses_pi_extension_is_not_reported_as_ours() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    write_other_package(&agent_dir.join("extensions").join("other"));
    write_other_package(&temp.path().join(".pi").join("extensions").join("other"));
    std::fs::create_dir_all(temp.path().join(".pi")).unwrap();
    write_other_package(&temp.path().join("elsewhere"));
    std::fs::write(
        temp.path().join(".pi").join("settings.json"),
        r#"{"packages": ["../elsewhere"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    assert!(
        relay_extension_sites(temp.path()).is_empty(),
        "an unrelated pi package must not be reported as the Relay extension"
    );
    assert!(!extension_configured());
}

// The explicit route was the one hole in that name check: any path that existed
// counted, so a stale variable made doctor Pass *and* made the launcher inject the
// path with `-e` -- a green check describing a session with no Relay code in it.
#[test]
fn an_unrelated_extension_named_by_the_environment_is_not_reported_as_ours() {
    let temp = tempfile::tempdir().unwrap();
    let other = temp.path().join("other");
    write_other_package(&other);
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(
        Some(other.join("index.ts").as_os_str()),
        Some(empty_home.as_os_str()),
    );
    assert!(relay_extension_sites(temp.path()).is_empty());
    assert!(!extension_configured());
    assert!(launchable_extension_path(temp.path()).is_none());
}

// `-e` is never trust-gated, so falling back to a project-scoped site would make
// the launcher run code pi refused to load -- undoing the gate this module reports.
#[test]
fn the_launch_path_never_promotes_a_project_scoped_extension() {
    let temp = tempfile::tempdir().unwrap();
    let project_extensions = temp.path().join(".pi").join("extensions");
    std::fs::create_dir_all(&project_extensions).unwrap();
    write_relay_package(&project_extensions.join("nemo-relay"));
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));
    assert!(!relay_extension_sites(temp.path()).is_empty());
    assert!(launchable_extension_path(temp.path()).is_none());
}

// pi resolves an `-e` argument as a package source, so an npm specifier that
// `pi install` recorded would be fetched and installed by a launch, not loaded.
#[test]
fn the_launch_path_skips_an_installed_source_that_is_not_a_path() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": ["npm:nemo-relay-pi"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    assert!(!relay_extension_sites(temp.path()).is_empty());
    assert!(launchable_extension_path(temp.path()).is_none());
}

// A user-scope install is what the README's install routes produce, and none of
// them set an environment variable -- so the launcher has to find one without it.
// An explicit path still wins, because someone who set it meant it.
#[test]
fn the_launch_path_prefers_the_explicit_route_over_a_user_scope_install() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let installed = agent_dir.join("extensions").join("nemo-relay");
    write_relay_package(&installed);

    let _discovered = scoped(None, Some(agent_dir.as_os_str()));
    assert_eq!(launchable_extension_path(temp.path()), Some(installed));
    drop(_discovered);

    let explicit = temp.path().join("checkout");
    write_relay_package(&explicit);
    let _env = scoped(Some(explicit.as_os_str()), Some(agent_dir.as_os_str()));
    assert_eq!(launchable_extension_path(temp.path()), Some(explicit));
}

// The headline status and the load-path check must agree: they now answer from the
// same resolution, so one `doctor` run cannot say "not located" beside a Pass.
#[test]
fn the_headline_status_agrees_with_the_load_path_check() {
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("checkout");
    write_relay_package(&package);
    let entry = package.join("index.ts");
    let empty_home = temp.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _found = scoped(Some(entry.as_os_str()), Some(empty_home.as_os_str()));
    assert!(extension_configured());
    assert!(hook_status().unwrap().contains("resolved at"));
    drop(_found);

    let _missing = scoped(None, Some(empty_home.as_os_str()));
    assert!(!extension_configured());
    assert!(hook_status().unwrap().contains("not located"));
}
