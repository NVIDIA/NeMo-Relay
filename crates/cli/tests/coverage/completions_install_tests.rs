// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Mutex;

use clap_complete::Shell;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn zsh_uses_zdotdir_when_set() {
    let path = completion_path(
        Shell::Zsh,
        Some(OsString::from("/home/u")),
        Some(OsString::from("/home/u/dot")),
    )
    .unwrap();
    assert_eq!(path, PathBuf::from("/home/u/dot/.zfunc/_nemo-flow"));
}

#[test]
fn zsh_falls_back_to_home_without_zdotdir() {
    let path = completion_path(Shell::Zsh, Some(OsString::from("/home/u")), None).unwrap();
    assert_eq!(path, PathBuf::from("/home/u/.zfunc/_nemo-flow"));
}

#[test]
fn bash_uses_home_dot_bash_completion_d() {
    let path = completion_path(Shell::Bash, Some(OsString::from("/home/u")), None).unwrap();
    assert_eq!(path, PathBuf::from("/home/u/.bash_completion.d/nemo-flow"));
}

#[test]
fn fish_uses_xdg_config_fish_completions() {
    let path = completion_path(Shell::Fish, Some(OsString::from("/home/u")), None).unwrap();
    assert_eq!(
        path,
        PathBuf::from("/home/u/.config/fish/completions/nemo-flow.fish")
    );
}

#[test]
fn powershell_is_rejected() {
    let error = completion_path(Shell::PowerShell, Some(OsString::from("/home/u")), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not support"), "error was: {error}");
}

#[test]
fn detect_shell_recognises_known_basenames() {
    assert_eq!(
        detect_shell(Some(OsString::from("/bin/zsh"))).unwrap(),
        Shell::Zsh
    );
    assert_eq!(
        detect_shell(Some(OsString::from("/usr/local/bin/bash"))).unwrap(),
        Shell::Bash
    );
    assert_eq!(
        detect_shell(Some(OsString::from("/opt/homebrew/bin/fish"))).unwrap(),
        Shell::Fish
    );
}

#[test]
fn detect_shell_rejects_unknown_shell() {
    let error = detect_shell(Some(OsString::from("/bin/tcsh")))
        .unwrap_err()
        .to_string();
    assert!(error.contains("tcsh"), "error was: {error}");
}

#[test]
fn detect_shell_rejects_missing_shell_env() {
    let error = detect_shell(None).unwrap_err().to_string();
    assert!(error.contains("$SHELL is not set"), "error was: {error}");
}

#[test]
fn write_atomic_creates_target_and_removes_temp_file() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("nemo-flow");

    write_atomic(&target, b"complete -c nemo-flow").unwrap();

    assert_eq!(std::fs::read(&target).unwrap(), b"complete -c nemo-flow");
    assert!(!target.with_file_name(".nemo-flow.tmp").exists());
}

#[test]
fn install_writes_detected_shell_completion() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let old_home = std::env::var_os("HOME");
    let old_zdotdir = std::env::var_os("ZDOTDIR");
    let old_shell = std::env::var_os("SHELL");

    unsafe {
        std::env::set_var("HOME", temp.path());
        std::env::remove_var("ZDOTDIR");
        std::env::set_var("SHELL", "/bin/zsh");
    }

    let path = install(None).unwrap();

    restore_env("HOME", old_home);
    restore_env("ZDOTDIR", old_zdotdir);
    restore_env("SHELL", old_shell);

    assert_eq!(path, temp.path().join(".zfunc/_nemo-flow"));
    let script = std::fs::read_to_string(path).unwrap();
    assert!(script.contains("nemo-flow"));
}

fn restore_env(key: &str, value: Option<OsString>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}
