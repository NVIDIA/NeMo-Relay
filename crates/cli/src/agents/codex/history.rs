// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Codex thread-history provider migration.
//!
//! Codex records the provider that produced each thread in
//! `threads.model_provider` and filters its resume picker by the provider that
//! is currently active. Installing the Relay integration switches Codex to the
//! `nemo-relay-openai` provider, so threads recorded under the built-in
//! `openai` provider stop appearing in the picker even though they remain
//! resumable by id.
//!
//! `nemo-relay install codex --migrate-history` rewrites the recorded provider
//! so that pre-install history stays visible. Every migration writes a journal
//! next to the other Relay user state; `nemo-relay uninstall codex` reads that
//! journal and reverses the rewrite without needing the flag again.
//!
//! This is experimental. Codex owns the schema, offers no supported API for
//! changing a thread's provider, and may change the storage layout in any
//! release. Until <https://github.com/openai/codex/issues/27381> is resolved
//! upstream, editing the database directly is the only available mechanism.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::host::codex_home_dir;

/// Codex's built-in provider id, used before the Relay integration is installed.
pub(crate) const OPENAI_PROVIDER: &str = "openai";
/// The provider id Relay installs into `config.toml`.
pub(crate) const RELAY_PROVIDER: &str = "nemo-relay-openai";

/// Codex's thread index. The numeric suffix is a Codex schema generation, so a
/// future Codex release can move this to `state_6.sqlite` or later. Override the
/// default with `--history-database` when that happens.
const STATE_DB_FILE: &str = "state_5.sqlite";
/// Journal recording migrations so uninstall can infer and reverse them.
const JOURNAL_FILE: &str = "codex-history-migration.json";
const JOURNAL_VERSION: u64 = 1;

/// Milliseconds `sqlite3` waits for a competing writer before giving up.
const BUSY_TIMEOUT_MS: u64 = 5_000;

/// Outcome of a migration or reversal, for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MigrationOutcome {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) thread_ids: Vec<String>,
}

impl MigrationOutcome {
    fn describe(&self) -> String {
        format!(
            "{} Codex thread(s) from provider `{}` to `{}`",
            self.thread_ids.len(),
            self.from,
            self.to
        )
    }
}

/// Rewrites pre-install `openai` threads to the Relay provider.
///
/// Returns `None` when there was nothing to migrate.
pub(crate) fn migrate_to_relay(
    dry_run: bool,
    database: Option<&Path>,
) -> Result<Option<MigrationOutcome>, String> {
    let database = resolve_database(database)?;
    if !database.exists() {
        return Err(format!(
            "no Codex thread database at {}; run Codex at least once before migrating history, or \
             name the current database with `--history-database`",
            database.display()
        ));
    }
    require_sqlite3()?;
    ensure_thread_schema(&database)?;
    let thread_ids = thread_ids_for_provider(&database, OPENAI_PROVIDER)?;
    if thread_ids.is_empty() {
        println!(
            "no Codex threads are recorded under provider `{OPENAI_PROVIDER}`; nothing to migrate."
        );
        return Ok(None);
    }
    let outcome = MigrationOutcome {
        from: OPENAI_PROVIDER.to_string(),
        to: RELAY_PROVIDER.to_string(),
        thread_ids,
    };
    if dry_run {
        println!(
            "would move {} in {}",
            outcome.describe(),
            database.display()
        );
        return Ok(Some(outcome));
    }
    let backup = back_up_database(&database)?;
    // Record the journal before the database changes. Writing it afterwards
    // leaves no way back when the write fails: the threads have already moved,
    // and uninstall infers reversal from the journal, so it would silently
    // decline to reverse anything.
    write_journal(&database, &backup, &outcome)?;
    if let Err(error) = update_provider(&database, OPENAI_PROVIDER, RELAY_PROVIDER) {
        // The update is a single transaction, so a failure changed nothing and
        // the journal describes a migration that never happened. Drop it, but
        // do not let cleanup mask the original failure.
        let _ = clear_journal(/*dry_run*/ false);
        return Err(error);
    }
    println!("moved {} in {}", outcome.describe(), database.display());
    println!(
        "backed up the previous thread database to {}",
        backup.display()
    );
    println!(
        "`nemo-relay uninstall codex` reverses this automatically; no flag is needed at uninstall."
    );
    Ok(Some(outcome))
}

/// Reverses a recorded migration, returning threads to the built-in provider.
///
/// Reversal moves every thread still recorded under [`RELAY_PROVIDER`] back to
/// [`OPENAI_PROVIDER`], not only the ids captured at migration time. Threads
/// created while Relay was installed carry the Relay provider legitimately, but
/// uninstall removes that provider from `config.toml`, so leaving them behind
/// would hide them from the picker — the same defect the migration exists to
/// fix, mirrored.
///
/// Returns `None` when no migration was recorded or there is nothing to move.
pub(crate) fn restore_from_relay(
    dry_run: bool,
    database: Option<&Path>,
) -> Result<Option<MigrationOutcome>, String> {
    let Some(journal) = read_journal()? else {
        return Ok(None);
    };
    // An explicit override wins, then the database the migration pinned, then
    // the default. The journal matters most here: it names the database that
    // was actually rewritten, even if the default has since moved on.
    let database = match database {
        Some(database) => resolve_database(Some(database))?,
        None => match journal_database(&journal) {
            Some(database) => database,
            None => resolve_database(None)?,
        },
    };
    if !database.exists() {
        clear_journal(dry_run)?;
        return Err(format!(
            "recorded Codex thread database {} no longer exists; discarded the migration journal",
            database.display()
        ));
    }
    require_sqlite3()?;
    ensure_thread_schema(&database)?;
    let thread_ids = thread_ids_for_provider(&database, RELAY_PROVIDER)?;
    if thread_ids.is_empty() {
        clear_journal(dry_run)?;
        return Ok(None);
    }
    let outcome = MigrationOutcome {
        from: RELAY_PROVIDER.to_string(),
        to: OPENAI_PROVIDER.to_string(),
        thread_ids,
    };
    if dry_run {
        println!(
            "would move {} in {}",
            outcome.describe(),
            database.display()
        );
        return Ok(Some(outcome));
    }
    let backup = back_up_database(&database)?;
    update_provider(&database, RELAY_PROVIDER, OPENAI_PROVIDER)?;
    clear_journal(dry_run)?;
    println!("moved {} in {}", outcome.describe(), database.display());
    println!(
        "backed up the previous thread database to {}",
        backup.display()
    );
    Ok(Some(outcome))
}

/// Reports whether a migration is recorded, so uninstall can infer the flag.
///
/// Uninstall infers reversal from [`restore_from_relay`] reading the same
/// journal, so this exists for tests and future doctor reporting.
#[cfg(test)]
pub(crate) fn migration_recorded() -> bool {
    matches!(read_journal(), Ok(Some(_)))
}

/// Resolves the thread database to operate on.
///
/// A bare file name such as `state_6.sqlite` resolves inside the Codex home,
/// which is the common case when Codex bumps its schema generation. Anything
/// with a directory component is used as given.
fn resolve_database(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let Some(explicit) = explicit else {
        return Ok(codex_home_dir()?.join(STATE_DB_FILE));
    };
    if explicit.components().count() == 1 && explicit.is_relative() {
        return Ok(codex_home_dir()?.join(explicit));
    }
    Ok(explicit.to_path_buf())
}

/// Rejects a database that does not carry the columns this migration rewrites.
///
/// The path can come from `--history-database`, so a typo or an unrelated
/// SQLite file should fail before anything is copied or written.
fn ensure_thread_schema(database: &Path) -> Result<(), String> {
    let columns = run_sqlite(database, "PRAGMA table_info(threads);")?;
    if columns.trim().is_empty() {
        return Err(format!(
            "{} has no `threads` table; it does not look like a Codex thread database",
            database.display()
        ));
    }
    let has_provider = columns
        .lines()
        .filter_map(|line| line.split('|').nth(1))
        .any(|column| column == "model_provider");
    if !has_provider {
        return Err(format!(
            "the `threads` table in {} has no `model_provider` column; this Codex schema is not \
             supported",
            database.display()
        ));
    }
    Ok(())
}

fn journal_path() -> Result<PathBuf, String> {
    crate::configuration::user_config_dir()
        .map(|path| path.join(JOURNAL_FILE))
        .ok_or_else(|| {
            "cannot determine the user configuration directory for the Codex history migration \
             journal"
                .to_string()
        })
}

fn require_sqlite3() -> Result<(), String> {
    match Command::new("sqlite3").arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "`sqlite3 --version` failed with status {}; Codex history migration requires a working \
             sqlite3 on PATH",
            output.status
        )),
        Err(error) => Err(format!(
            "Codex history migration requires the `sqlite3` command on PATH, but it could not be \
             run: {error}"
        )),
    }
}

/// Runs one or more statements against `database` and returns stdout.
///
/// `BEGIN IMMEDIATE` takes the write lock up front so a running Codex causes a
/// clean `database is locked` failure instead of a partial rewrite.
fn run_sqlite(database: &Path, sql: &str) -> Result<String, String> {
    // `.timeout` rather than `PRAGMA busy_timeout`: the pragma emits its value
    // as a result row, which would contaminate the rows callers parse.
    let mut child = Command::new("sqlite3")
        // Stop at a failed `BEGIN IMMEDIATE`. Otherwise the SQLite shell keeps
        // running subsequent statements and masks a lock error with a failed
        // `COMMIT`, notably on Windows.
        .arg("-bail")
        .arg("-noheader")
        .arg("-batch")
        .arg("-cmd")
        .arg(format!(".timeout {BUSY_TIMEOUT_MS}"))
        .arg(database)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "failed to run sqlite3 against {}: {error}",
                database.display()
            )
        })?;
    let mut stdin = child.stdin.take().expect("piped sqlite3 stdin");
    stdin.write_all(sql.as_bytes()).map_err(|error| {
        format!(
            "failed to write SQL to sqlite3 for {}: {error}",
            database.display()
        )
    })?;
    drop(stdin);
    let output = child.wait_with_output().map_err(|error| {
        format!(
            "failed to wait for sqlite3 against {}: {error}",
            database.display()
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let hint = if stderr.contains("locked") || stderr.contains("busy") {
            " — quit any running Codex session and retry"
        } else {
            ""
        };
        return Err(format!(
            "sqlite3 failed against {}: {stderr}{hint}",
            database.display()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("sqlite3 returned non-UTF-8 output: {error}"))
}

fn thread_ids_for_provider(database: &Path, provider: &str) -> Result<Vec<String>, String> {
    let sql = format!(
        "SELECT id FROM threads WHERE model_provider = {};",
        sql_string(provider)
    );
    Ok(run_sqlite(database, &sql)?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect())
}

fn update_provider(database: &Path, from: &str, to: &str) -> Result<(), String> {
    let sql = format!(
        "BEGIN IMMEDIATE;\nUPDATE threads SET model_provider = {} WHERE model_provider = {};\nCOMMIT;",
        sql_string(to),
        sql_string(from)
    );
    run_sqlite(database, &sql).map(|_| ())
}

/// Copies the database and its WAL sidecars to a timestamped directory.
///
/// A checkpoint runs first so the copied main database is self-contained, but
/// the sidecars are copied too: a checkpoint can be declined by a reader.
fn back_up_database(database: &Path) -> Result<PathBuf, String> {
    run_sqlite(database, "PRAGMA wal_checkpoint(TRUNCATE);")?;
    let parent = database
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", database.display()))?;
    let backup_dir = parent.join(format!("nemo-relay-history-backup-{}", unix_timestamp()));
    fs::create_dir_all(&backup_dir)
        .map_err(|error| format!("failed to create {}: {error}", backup_dir.display()))?;
    for suffix in ["", "-wal", "-shm"] {
        let mut source = database.as_os_str().to_os_string();
        source.push(suffix);
        let source = PathBuf::from(source);
        if !source.exists() {
            continue;
        }
        let file_name = source
            .file_name()
            .ok_or_else(|| format!("{} has no file name", source.display()))?;
        fs::copy(&source, backup_dir.join(file_name)).map_err(|error| {
            format!(
                "failed to back up {} into {}: {error}",
                source.display(),
                backup_dir.display()
            )
        })?;
    }
    Ok(backup_dir)
}

fn write_journal(database: &Path, backup: &Path, outcome: &MigrationOutcome) -> Result<(), String> {
    let path = journal_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(&json!({
        "version": JOURNAL_VERSION,
        "database": database,
        "backup": backup,
        "migratedAt": unix_timestamp(),
        "from": outcome.from,
        "to": outcome.to,
        "threadIds": outcome.thread_ids,
    }))
    .map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    crate::filesystem::atomic_write_private(&path, &bytes)
}

fn read_journal() -> Result<Option<Value>, String> {
    let path = journal_path()?;
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
            .map(Some)
            .map_err(|error| {
                format!(
                    "failed to parse the Codex history migration journal {}: {error}",
                    path.display()
                )
            }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "failed to read the Codex history migration journal {}: {error}",
            path.display()
        )),
    }
}

fn journal_database(journal: &Value) -> Option<PathBuf> {
    journal
        .get("database")
        .and_then(Value::as_str)
        .map(PathBuf::from)
}

fn clear_journal(dry_run: bool) -> Result<(), String> {
    let path = journal_path()?;
    if dry_run {
        println!("remove {}", path.display());
        return Ok(());
    }
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to remove the Codex history migration journal {}: {error}",
            path.display()
        )),
    }
}

/// Quotes a value as a SQL string literal, doubling embedded single quotes.
fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/codex_history_tests.rs"]
mod tests;
