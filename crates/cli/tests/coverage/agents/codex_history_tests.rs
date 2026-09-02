// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage for the experimental Codex thread-history provider migration.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use super::*;
use crate::test_support::EnvScope;

/// Redirects the Codex home and the Relay user config directory into a
/// temporary tree, and seeds a thread database with the given providers.
struct CodexHistoryScope {
    _env: EnvScope,
    home: TempDir,
    database: String,
}

impl CodexHistoryScope {
    fn enter(threads: &[(&str, &str)]) -> Self {
        Self::enter_named(STATE_DB_FILE, threads)
    }

    fn enter_named(database: &str, threads: &[(&str, &str)]) -> Self {
        let home = tempfile::tempdir().expect("temporary home");
        let codex_home = home.path().join(".codex");
        std::fs::create_dir_all(&codex_home).expect("codex home");
        let config_home = home.path().join(".config");
        std::fs::create_dir_all(&config_home).expect("config home");
        let env = EnvScope::set(&[
            ("HOME", Some(home.path().as_os_str())),
            ("USERPROFILE", Some(home.path().as_os_str())),
            ("CODEX_HOME", Some(codex_home.as_os_str())),
            ("XDG_CONFIG_HOME", Some(config_home.as_os_str())),
            ("APPDATA", Some(config_home.as_os_str())),
        ]);
        let scope = Self {
            _env: env,
            home,
            database: database.to_string(),
        };
        seed_database(&scope.database(), threads);
        scope
    }

    fn database(&self) -> std::path::PathBuf {
        self.home.path().join(".codex").join(&self.database)
    }

    fn providers(&self) -> Vec<(String, String)> {
        let raw = sqlite(
            &self.database(),
            "SELECT id, model_provider FROM threads ORDER BY id;",
        );
        raw.lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let (id, provider) = line.split_once('|').expect("delimited row");
                (id.to_string(), provider.to_string())
            })
            .collect()
    }

    fn journal(&self) -> Option<Value> {
        read_journal().expect("journal reads")
    }

    fn backup_dirs(&self) -> Vec<std::path::PathBuf> {
        let mut dirs = std::fs::read_dir(self.home.path().join(".codex"))
            .expect("codex home listing")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(OsStr::to_str)
                        .is_some_and(|name| name.starts_with("nemo-relay-history-backup-"))
            })
            .collect::<Vec<_>>();
        dirs.sort();
        dirs
    }
}

/// Creates a minimal `threads` table carrying only the columns under test.
fn seed_database(database: &Path, threads: &[(&str, &str)]) {
    let mut sql =
        String::from("CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL);");
    for (id, provider) in threads {
        sql.push_str(&format!(
            "INSERT INTO threads (id, model_provider) VALUES ({}, {});",
            sql_string(id),
            sql_string(provider)
        ));
    }
    sqlite(database, &sql);
}

fn sqlite(database: &Path, sql: &str) -> String {
    let output = Command::new("sqlite3")
        .arg("-noheader")
        .arg("-batch")
        .arg(database)
        .arg(sql)
        .output()
        .expect("sqlite3 runs");
    assert!(
        output.status.success(),
        "sqlite3 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 sqlite3 output")
}

/// Skips a test when the host has no `sqlite3`, which the migration requires.
fn sqlite3_available() -> bool {
    Command::new("sqlite3")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

macro_rules! require_sqlite3_or_skip {
    () => {
        if !sqlite3_available() {
            eprintln!("skipping: no sqlite3 on PATH");
            return;
        }
    };
}

#[test]
fn migrate_moves_openai_threads_onto_the_relay_provider() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[
        ("thread-a", OPENAI_PROVIDER),
        ("thread-b", OPENAI_PROVIDER),
        ("thread-c", "some-other-provider"),
    ]);

    let outcome = migrate_to_relay(/*dry_run*/ false, /*database*/ None)
        .expect("migration succeeds")
        .expect("migration reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a", "thread-b"]);
    assert_eq!(
        scope.providers(),
        vec![
            ("thread-a".to_string(), RELAY_PROVIDER.to_string()),
            ("thread-b".to_string(), RELAY_PROVIDER.to_string()),
            ("thread-c".to_string(), "some-other-provider".to_string()),
        ],
        "only threads on the built-in provider move"
    );
}

#[test]
fn migrate_records_a_journal_that_uninstall_can_infer() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);
    assert!(
        !migration_recorded(),
        "no migration is recorded before one runs"
    );

    migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");

    assert!(migration_recorded(), "uninstall can infer the migration");
    let journal = scope.journal().expect("journal exists");
    assert_eq!(journal["version"], json!(JOURNAL_VERSION));
    assert_eq!(journal["from"], json!(OPENAI_PROVIDER));
    assert_eq!(journal["to"], json!(RELAY_PROVIDER));
    assert_eq!(journal["threadIds"], json!(["thread-a"]));
    assert_eq!(
        journal["database"],
        json!(scope.database()),
        "the journal pins the database it rewrote"
    );
}

#[test]
fn migrate_backs_up_the_database_before_rewriting_it() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);

    migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");

    let backups = scope.backup_dirs();
    assert_eq!(backups.len(), 1, "exactly one backup directory is created");
    let copied = backups[0].join(STATE_DB_FILE);
    assert!(copied.exists(), "the database itself is copied");
    assert_eq!(
        sqlite(&copied, "SELECT model_provider FROM threads;").trim(),
        OPENAI_PROVIDER,
        "the backup holds the pre-migration provider"
    );
}

#[test]
fn migrate_reports_nothing_to_do_without_built_in_provider_threads() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", RELAY_PROVIDER)]);

    let outcome =
        migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");

    assert_eq!(outcome, None, "an empty migration reports no outcome");
    assert!(
        !migration_recorded(),
        "an empty migration records no journal to reverse"
    );
    assert!(
        scope.backup_dirs().is_empty(),
        "an empty migration does not back up"
    );
}

#[test]
fn dry_run_migration_reports_without_touching_the_database() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);

    let outcome = migrate_to_relay(/*dry_run*/ true, /*database*/ None)
        .expect("dry run succeeds")
        .expect("dry run reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a"]);
    assert_eq!(
        scope.providers(),
        vec![("thread-a".to_string(), OPENAI_PROVIDER.to_string())],
        "a dry run leaves the database untouched"
    );
    assert!(!migration_recorded(), "a dry run records no journal");
    assert!(scope.backup_dirs().is_empty(), "a dry run does not back up");
}

#[test]
fn restore_returns_migrated_threads_to_the_built_in_provider() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[
        ("thread-a", OPENAI_PROVIDER),
        ("thread-b", "some-other-provider"),
    ]);
    migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");

    let outcome = restore_from_relay(/*dry_run*/ false, /*database*/ None)
        .expect("restore succeeds")
        .expect("restore reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a"]);
    assert_eq!(
        scope.providers(),
        vec![
            ("thread-a".to_string(), OPENAI_PROVIDER.to_string()),
            ("thread-b".to_string(), "some-other-provider".to_string()),
        ],
        "the round trip restores the original providers"
    );
    assert!(
        !migration_recorded(),
        "a completed restore clears the journal"
    );
}

#[test]
fn restore_also_moves_threads_created_while_relay_was_installed() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("pre-install", OPENAI_PROVIDER)]);
    migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");
    // A thread Codex recorded under the Relay provider after the migration ran.
    sqlite(
        &scope.database(),
        &format!(
            "INSERT INTO threads (id, model_provider) VALUES ('relay-era', {});",
            sql_string(RELAY_PROVIDER)
        ),
    );

    let outcome = restore_from_relay(/*dry_run*/ false, /*database*/ None)
        .expect("restore succeeds")
        .expect("restore reports an outcome");

    assert_eq!(
        outcome.thread_ids,
        vec!["pre-install", "relay-era"],
        "reversal covers threads the migration never recorded"
    );
    assert_eq!(
        scope.providers(),
        vec![
            ("pre-install".to_string(), OPENAI_PROVIDER.to_string()),
            ("relay-era".to_string(), OPENAI_PROVIDER.to_string()),
        ],
        "uninstall leaves no thread pointing at a provider that no longer exists"
    );
}

#[test]
fn restore_without_a_recorded_migration_is_a_no_op() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", RELAY_PROVIDER)]);

    let outcome =
        restore_from_relay(/*dry_run*/ false, /*database*/ None).expect("restore succeeds");

    assert_eq!(outcome, None, "no journal means nothing to reverse");
    assert_eq!(
        scope.providers(),
        vec![("thread-a".to_string(), RELAY_PROVIDER.to_string())],
        "an uninstall without a migration leaves the database untouched"
    );
}

#[test]
fn dry_run_restore_reports_without_touching_the_database() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);
    migrate_to_relay(/*dry_run*/ false, /*database*/ None).expect("migration succeeds");

    let outcome = restore_from_relay(/*dry_run*/ true, /*database*/ None)
        .expect("dry run succeeds")
        .expect("dry run reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a"]);
    assert_eq!(
        scope.providers(),
        vec![("thread-a".to_string(), RELAY_PROVIDER.to_string())],
        "a dry run leaves the database untouched"
    );
    assert!(
        migration_recorded(),
        "a dry run keeps the journal so the real reversal still runs"
    );
}

#[test]
fn migrate_fails_when_codex_has_no_thread_database() {
    let home = tempfile::tempdir().expect("temporary home");
    let codex_home = home.path().join(".codex");
    std::fs::create_dir_all(&codex_home).expect("codex home");
    let _env = EnvScope::set(&[
        ("HOME", Some(home.path().as_os_str())),
        ("USERPROFILE", Some(home.path().as_os_str())),
        ("CODEX_HOME", Some(codex_home.as_os_str())),
        ("XDG_CONFIG_HOME", Some(home.path().as_os_str())),
        ("APPDATA", Some(home.path().as_os_str())),
    ]);

    let error = migrate_to_relay(/*dry_run*/ false, /*database*/ None)
        .expect_err("missing database is an error");

    assert!(
        error.contains("no Codex thread database at"),
        "unexpected error: {error}"
    );
}

#[test]
fn sql_string_escapes_embedded_quotes() {
    assert_eq!(sql_string("openai"), "'openai'");
    assert_eq!(sql_string("o'brien"), "'o''brien'");
}

#[test]
fn a_named_database_resolves_inside_the_codex_home() {
    require_sqlite3_or_skip!();
    // The schema generation Codex might move to next.
    let scope = CodexHistoryScope::enter_named(
        "state_6.sqlite",
        &[("thread-a", OPENAI_PROVIDER), ("thread-b", OPENAI_PROVIDER)],
    );

    let outcome = migrate_to_relay(/*dry_run*/ false, Some(Path::new("state_6.sqlite")))
        .expect("migration succeeds")
        .expect("migration reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a", "thread-b"]);
    assert_eq!(
        scope.providers(),
        vec![
            ("thread-a".to_string(), RELAY_PROVIDER.to_string()),
            ("thread-b".to_string(), RELAY_PROVIDER.to_string()),
        ]
    );
    assert_eq!(
        scope.journal().expect("journal exists")["database"],
        json!(scope.database()),
        "the journal pins the overridden database"
    );
}

#[test]
fn a_full_path_database_is_used_as_given() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("ignored", OPENAI_PROVIDER)]);
    let elsewhere = scope.home.path().join("elsewhere.sqlite");
    seed_database(&elsewhere, &[("thread-a", OPENAI_PROVIDER)]);

    let outcome = migrate_to_relay(/*dry_run*/ false, Some(&elsewhere))
        .expect("migration succeeds")
        .expect("migration reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a"]);
    assert_eq!(
        scope.providers(),
        vec![("ignored".to_string(), OPENAI_PROVIDER.to_string())],
        "the default database is left alone"
    );
}

#[test]
fn restore_prefers_the_database_the_migration_recorded() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter_named("state_6.sqlite", &[("thread-a", OPENAI_PROVIDER)]);
    migrate_to_relay(/*dry_run*/ false, Some(Path::new("state_6.sqlite")))
        .expect("migration succeeds");

    // No override here: uninstall infers both the reversal and its target.
    let outcome = restore_from_relay(/*dry_run*/ false, /*database*/ None)
        .expect("restore succeeds")
        .expect("restore reports an outcome");

    assert_eq!(outcome.thread_ids, vec!["thread-a"]);
    assert_eq!(
        scope.providers(),
        vec![("thread-a".to_string(), OPENAI_PROVIDER.to_string())],
        "reversal follows the journal rather than the default file name"
    );
}

#[test]
fn migrate_rejects_a_database_without_a_threads_table() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);
    let unrelated = scope.home.path().join("unrelated.sqlite");
    sqlite(&unrelated, "CREATE TABLE notes (id TEXT);");

    let error = migrate_to_relay(/*dry_run*/ false, Some(&unrelated))
        .expect_err("an unrelated database is rejected");

    assert!(
        error.contains("has no `threads` table"),
        "unexpected error: {error}"
    );
}

#[test]
fn migrate_rejects_a_threads_table_without_a_provider_column() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);
    let future = scope.home.path().join("state_9.sqlite");
    sqlite(
        &future,
        "CREATE TABLE threads (id TEXT, provider_ref TEXT);",
    );

    let error = migrate_to_relay(/*dry_run*/ false, Some(&future))
        .expect_err("an unsupported schema is rejected");

    assert!(
        error.contains("has no `model_provider` column"),
        "unexpected error: {error}"
    );
}

#[test]
fn migrate_fails_cleanly_while_another_writer_holds_the_lock() {
    require_sqlite3_or_skip!();
    let scope = CodexHistoryScope::enter(&[("thread-a", OPENAI_PROVIDER)]);
    // Stands in for a running Codex: an open write transaction on the database.
    let mut holder = Command::new("sqlite3")
        .arg(scope.database())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("sqlite3 spawns");
    {
        use std::io::Write;
        let stdin = holder.stdin.as_mut().expect("sqlite3 stdin");
        writeln!(stdin, "BEGIN IMMEDIATE;").expect("write lock statement");
        stdin.flush().expect("flush lock statement");
    }
    // Give the holder time to take the lock before competing for it.
    std::thread::sleep(std::time::Duration::from_millis(500));

    let error = migrate_to_relay(/*dry_run*/ false, /*database*/ None)
        .expect_err("a locked database is an error");

    let _ = holder.kill();
    let _ = holder.wait();
    assert!(
        error.contains("locked") || error.contains("busy"),
        "unexpected error: {error}"
    );
    assert!(
        error.contains("quit any running Codex session"),
        "the error should say how to recover: {error}"
    );
    assert_eq!(
        scope.providers(),
        vec![("thread-a".to_string(), OPENAI_PROVIDER.to_string())],
        "a locked migration leaves the database unchanged"
    );
    assert!(
        !migration_recorded(),
        "a locked migration records no journal to reverse"
    );
}
