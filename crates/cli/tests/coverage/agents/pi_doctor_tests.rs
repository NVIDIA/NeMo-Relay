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
    // The declared entry point matters: it is what a `packages` filter's patterns are matched
    // against, so a fixture without one reads as "cannot tell" rather than as enabled.
    std::fs::write(
        dir.join("package.json"),
        r#"{"name": "nemo-relay-pi", "pi": {"extensions": ["./index.ts"]}}"#,
    )
    .unwrap();
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

// pi's own configuration selector rewrites a string entry into the object form the
// moment a user toggles any resource of that package, so this shape is not a hand
// edit -- and while only strings were read, one keystroke in pi's UI made doctor
// report an installed extension as missing and made the launcher refuse to start.
#[test]
fn an_object_form_package_entry_is_found() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    let checkout = temp.path().join("checkout");
    write_relay_package(&checkout);
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": [{"source": "../checkout"}]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::User);
    assert_eq!(sites[0].filter, SettingsFilter::Loads);
    // Compared canonically: a recorded source is relative to the settings file, so the
    // resolved path keeps the `..` pi itself would resolve away.
    assert_eq!(
        launchable_extension_path(temp.path()).map(|path| std::fs::canonicalize(path).unwrap()),
        Some(std::fs::canonicalize(&checkout).unwrap())
    );
}

// A non-empty pattern list is matched against the package manifest, which this
// module does not read. Reporting it as loaded is the deliberate direction: a false
// negative here is the exact failure the whole module exists to prevent.
// The other side of the same table: a `+` force-include, a bare include naming the entry, and
// an `autoload: false` delta that adds it back all leave pi loading it.
#[test]
fn an_object_form_entry_whose_patterns_leave_it_loaded_is_not_reported_as_disabled() {
    for body in [
        r#"{"packages": [{"source": "../checkout", "extensions": ["+index.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "extensions": ["index.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "autoload": false, "extensions": ["+index.ts"]}]}"#,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        write_relay_package(&temp.path().join("checkout"));
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();

        let _env = scoped(None, Some(agent_dir.as_os_str()));
        let sites = relay_extension_sites(temp.path());

        assert_eq!(sites.len(), 1, "{body}: {sites:?}");
        assert_eq!(sites[0].filter, SettingsFilter::Loads, "{body}");
    }
}

// An include list that never names this entry leaves pi loading nothing from the package --
// step 1 keeps only what the includes match. Decidable without a glob matcher, and previously
// reported as a plain Pass.
#[test]
fn an_include_list_that_omits_this_entry_is_reported_as_disabled() {
    for body in [
        r#"{"packages": [{"source": "../checkout", "extensions": ["other.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "autoload": false, "extensions": ["+other.ts"]}]}"#,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        write_relay_package(&temp.path().join("checkout"));
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();

        let _env = scoped(None, Some(agent_dir.as_os_str()));
        let sites = relay_extension_sites(temp.path());
        assert_eq!(sites[0].filter, SettingsFilter::Excluded, "{body}");
    }
}

// A glob is pi's to evaluate, not this module's -- but saying Pass would claim pi loads the
// extension on no evidence, which is the claim this whole module exists to stop making.
#[test]
fn a_glob_filter_is_reported_as_undecided_rather_than_as_loaded() {
    for body in [
        r#"{"packages": [{"source": "../checkout", "extensions": ["!*.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "extensions": ["src/*.ts"]}]}"#,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        write_relay_package(&temp.path().join("checkout"));
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();

        let _env = scoped(None, Some(agent_dir.as_os_str()));
        let sites = relay_extension_sites(temp.path());
        assert_eq!(sites[0].filter, SettingsFilter::Undecided, "{body}");
    }
}

// Choosing the first site regardless of its filter manufactured the duplicate it then refused:
// `-e` re-enables the disabled copy, and the enabled one becomes a genuine second load.
#[test]
fn the_launch_path_prefers_a_copy_pi_already_loads() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("off"));
    let live = temp.path().join("live");
    write_relay_package(&live);
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": [{"source": "../off", "extensions": []}, "../live"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let launched = launchable_extension_path(temp.path()).unwrap();

    assert_eq!(
        std::fs::canonicalize(&launched).unwrap(),
        std::fs::canonicalize(&live).unwrap(),
        "the enabled copy must win, so the disabled one stays disabled"
    );
    assert_eq!(conflicting_extension_site(temp.path(), &launched), None);
}

// A third registration route, a sibling of `packages` in the same file, read by a generic loop
// over pi's four resource types rather than by anything named after extensions. Missing it meant
// doctor said "not located" and the launcher refused to start for a user pi loads fine.
#[test]
fn an_extensions_entry_in_settings_is_found_in_both_scopes() {
    // User scope: entries resolve against the agent directory.
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("checkout"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"extensions": ["../checkout/index.ts"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());
    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::User);
    assert_eq!(sites[0].filter, SettingsFilter::Loads);
    drop(_env);

    // Project scope: entries resolve against `<cwd>/.pi`, and pi trust-gates them.
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join(".pi")).unwrap();
    write_relay_package(&project.path().join("checkout"));
    std::fs::write(
        project.path().join(".pi").join("settings.json"),
        r#"{"extensions": ["../checkout/index.ts"]}"#,
    )
    .unwrap();
    let empty_home = project.path().join("home");
    std::fs::create_dir_all(&empty_home).unwrap();

    let _env = scoped(None, Some(empty_home.as_os_str()));
    let sites = relay_extension_sites(project.path());
    assert!(
        sites
            .iter()
            .any(|site| site.scope == ExtensionScope::Project),
        "a project `extensions` entry is trust-gated and must be reported: {sites:?}"
    );
}

// pi treats a directory entry as a *container* of extensions and walks it, so a parent directory
// registers the package inside it.
#[test]
fn an_extensions_entry_naming_a_container_directory_is_walked() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    let container = temp.path().join("integrations");
    write_relay_package(&container.join("pi"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"extensions": ["../integrations"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());
    assert_eq!(sites.len(), 1, "{sites:?}");
}

// The array filters its collected set with the same globbing `packages` filters use, over files
// this module does not enumerate the way pi does -- so a pattern makes it undecidable, not a Pass.
#[test]
fn an_extensions_entry_carrying_a_pattern_is_reported_as_undecided() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("checkout"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"extensions": ["../checkout/index.ts", "-index.ts"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());
    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].filter, SettingsFilter::Undecided);
}

// B1: two copies inside ONE source. pi resolves every distinct package source, so both load
// and post every hook twice -- and stopping at the first match hid exactly that.
#[test]
fn two_copies_recorded_in_one_settings_file_are_both_reported() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("one"));
    write_relay_package(&temp.path().join("two"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": ["../one", "../two"]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 2, "{sites:?}");
    let launched = launchable_extension_path(temp.path()).unwrap();
    assert!(conflicting_extension_site(temp.path(), &launched).is_some());
}

#[test]
fn two_copies_in_one_extensions_directory_are_both_reported() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let extensions = agent_dir.join("extensions");
    write_relay_package(&extensions.join("nemo-relay"));
    write_relay_package(&extensions.join("nemo-relay-copy"));

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 2, "{sites:?}");
    let launched = launchable_extension_path(temp.path()).unwrap();
    assert!(conflicting_extension_site(temp.path(), &launched).is_some());
}

// B2: a copy pi's own settings switch off is reliably *not* a load, so counting it refused a
// launch over a copy that was never going to register a hook.
#[test]
fn a_copy_pi_switched_off_is_not_a_second_copy() {
    for body in [
        r#"{"packages": [{"source": "../off", "extensions": []}]}"#,
        r#"{"packages": [{"source": "../off", "autoload": false}]}"#,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        write_relay_package(&temp.path().join("off"));
        let explicit = temp.path().join("checkout");
        write_relay_package(&explicit);
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();

        let _env = scoped(Some(explicit.as_os_str()), Some(agent_dir.as_os_str()));
        assert_eq!(
            conflicting_extension_site(temp.path(), &explicit),
            None,
            "{body}"
        );
    }
}

#[test]
fn an_object_form_entry_with_extension_patterns_is_still_found() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    write_relay_package(&temp.path().join("checkout"));
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"packages": [{"source": "../checkout", "extensions": ["index.ts"], "skills": []}]}"#,
    )
    .unwrap();

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 1, "{sites:?}");
    assert_eq!(sites[0].filter, SettingsFilter::Loads);
}

// Installed and switched off is not the same as absent, and must not be reported as
// either a plain Pass or a missing install. The launch path deliberately still uses
// it: `-e` applies no settings filter.
#[test]
fn an_object_form_entry_whose_extensions_are_disabled_is_reported_as_disabled() {
    for body in [
        r#"{"packages": [{"source": "../checkout", "extensions": []}]}"#,
        r#"{"packages": [{"source": "../checkout", "autoload": false}]}"#,
        // What pi's own configuration selector writes when a user switches this extension off:
        // `-<path>`, a force-exclude pi applies last and unconditionally. Reading a non-empty
        // list as "enabled" reported a plain Pass for a package pi loads nothing from.
        r#"{"packages": [{"source": "../checkout", "extensions": ["-index.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "extensions": ["-./index.ts"]}]}"#,
        r#"{"packages": [{"source": "../checkout", "autoload": false, "extensions": ["-index.ts"]}]}"#,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        let checkout = temp.path().join("checkout");
        write_relay_package(&checkout);
        std::fs::write(agent_dir.join("settings.json"), body).unwrap();

        let _env = scoped(None, Some(agent_dir.as_os_str()));
        let sites = relay_extension_sites(temp.path());

        assert_eq!(sites.len(), 1, "{body}: {sites:?}");
        assert_eq!(sites[0].filter, SettingsFilter::Excluded, "{body}");
        // Still launchable, deliberately: `-e` applies no settings filter, so the launcher
        // instruments a session the user's own `pi` runs are missing.
        assert_eq!(
            launchable_extension_path(temp.path()).map(|path| std::fs::canonicalize(path).unwrap()),
            Some(std::fs::canonicalize(&checkout).unwrap()),
            "{body}"
        );
    }
}

// Two ungated copies can be live at once -- a variable someone set at a checkout
// months ago, and the user-scope install the README recommends. Both are reported,
// because the order is the contract the launch path reads.
#[test]
fn a_distinct_explicit_copy_and_a_user_install_are_both_reported_explicit_first() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let installed = agent_dir.join("extensions").join("nemo-relay");
    write_relay_package(&installed);
    let explicit = temp.path().join("checkout");
    write_relay_package(&explicit);

    let _env = scoped(Some(explicit.as_os_str()), Some(agent_dir.as_os_str()));
    let sites = relay_extension_sites(temp.path());

    assert_eq!(sites.len(), 2, "{sites:?}");
    assert_eq!(sites[0].scope, ExtensionScope::Explicit);
    assert_eq!(sites[0].path, explicit);
    assert_eq!(sites[1].scope, ExtensionScope::User);

    // And that is exactly the case pi cannot see: it de-duplicates by path, so both
    // load, and every hook is posted twice.
    assert_eq!(
        conflicting_extension_site(temp.path(), &explicit),
        Some(installed)
    );
}

// One install is reachable both as its directory and as the entry file inside it,
// and pi resolves both to the same file through the `pi.extensions` manifest. That
// is one copy, and refusing to launch it would be a false alarm on the setup the
// launcher itself produces.
#[test]
fn the_same_install_reached_two_ways_is_not_a_second_copy() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let installed = agent_dir.join("extensions").join("nemo-relay");
    write_relay_package(&installed);
    let entry = installed.join("index.ts");

    let _env = scoped(Some(entry.as_os_str()), Some(agent_dir.as_os_str()));
    assert_eq!(conflicting_extension_site(temp.path(), &entry), None);
}

// pi canonicalizes with `realpathSync`, so a symlinked copy is one copy to pi and
// must be one copy here.
#[cfg(unix)]
#[test]
fn a_symlinked_copy_is_not_a_second_copy() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let installed = agent_dir.join("extensions").join("nemo-relay");
    write_relay_package(&installed);
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&installed, &link).unwrap();

    let _env = scoped(Some(link.as_os_str()), Some(agent_dir.as_os_str()));
    assert_eq!(conflicting_extension_site(temp.path(), &link), None);
}

// pi loads a project-scoped copy only for a trusted project, so it is not reliably
// a second load -- and refusing on it would block launches that are fine. The trust
// warning already names it.
#[test]
fn a_project_scoped_copy_is_not_a_second_copy() {
    let temp = tempfile::tempdir().unwrap();
    let agent_dir = temp.path().join("agent");
    let installed = agent_dir.join("extensions").join("nemo-relay");
    write_relay_package(&installed);
    write_relay_package(
        &temp
            .path()
            .join(".pi")
            .join("extensions")
            .join("nemo-relay"),
    );

    let _env = scoped(None, Some(agent_dir.as_os_str()));
    assert_eq!(conflicting_extension_site(temp.path(), &installed), None);
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
