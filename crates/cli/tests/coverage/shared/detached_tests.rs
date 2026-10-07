// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::io::Write;
use std::os::windows::io::{AsRawHandle, FromRawHandle};

#[test]
fn redirected_worker_stderr_preserves_file_and_pipe_output() {
    let mut file = tempfile::tempfile().unwrap();
    launch_stderr_fixture(file.as_raw_handle());
    use std::io::{Read, Seek};
    file.rewind().unwrap();
    let mut output = String::new();
    file.read_to_string(&mut output).unwrap();
    assert_eq!(output, "detached worker stderr");

    let mut read = std::ptr::null_mut();
    let mut write = std::ptr::null_mut();
    // SAFETY: The output pointers are valid; File takes ownership on success.
    assert_ne!(
        unsafe {
            windows_sys::Win32::System::Pipes::CreatePipe(
                &mut read,
                &mut write,
                std::ptr::null(),
                0,
            )
        },
        0
    );
    let mut reader = unsafe { std::fs::File::from_raw_handle(read) };
    let writer = unsafe { std::fs::File::from_raw_handle(write) };
    launch_stderr_fixture(writer.as_raw_handle());
    drop(writer);
    output.clear();
    reader.read_to_string(&mut output).unwrap();
    assert_eq!(output, "detached worker stderr");
}

fn launch_stderr_fixture(source: windows_sys::Win32::Foundation::HANDLE) {
    let stderr = worker_stderr(source).unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "process::detached::tests::detached_worker_fixture",
            "--nocapture",
        ])
        .env("NEMO_RELAY_TEST_CONSOLE_WORKER_CHILD", "1");
    let (mut child, bootstrap) = spawn_worker_detached(&command, &stderr).unwrap();
    drop(bootstrap);
    assert!(child.wait().unwrap().success());
}

#[test]
fn detached_worker_fixture() {
    if std::env::var_os("NEMO_RELAY_TEST_CONSOLE_WORKER_CHILD").is_some() {
        use std::io::Read;
        std::io::stdin().read_to_end(&mut Vec::new()).unwrap();
        std::io::stderr()
            .write_all(b"detached worker stderr")
            .unwrap();
    }
}

#[test]
fn console_worker_launcher_fixture() {
    if std::env::var_os("NEMO_RELAY_TEST_CONSOLE_WORKER_LAUNCHER").is_none() {
        return;
    }
    use windows_sys::Win32::System::Console::{
        AllocConsole, FreeConsole, GetConsoleMode, STD_ERROR_HANDLE, SetStdHandle,
    };
    // This runs in an isolated process so changing the console cannot affect other tests.
    unsafe { FreeConsole() };
    assert_ne!(unsafe { AllocConsole() }, 0);
    // AllocConsole preserves redirected standard handles when STARTF_USESTDHANDLES was used.
    // Explicitly select the new console buffer to exercise the launcher's console stderr path.
    for readable in [true, false] {
        let console = std::fs::OpenOptions::new()
            .read(readable)
            .write(true)
            .open("CONOUT$")
            .unwrap();
        // SAFETY: The console file remains live throughout the launch.
        assert_ne!(
            unsafe { SetStdHandle(STD_ERROR_HANDLE, console.as_raw_handle()) },
            0
        );
        let mut mode = 0;
        assert_eq!(
            unsafe { GetConsoleMode(console.as_raw_handle(), &mut mode) } != 0,
            readable
        );
        let stderr = inherited_stderr().unwrap();
        assert_eq!(
            unsafe { GetConsoleMode(stderr.as_raw_handle(), &mut mode) },
            0
        );
        launch_stderr_fixture(stderr.as_raw_handle());
    }
    unsafe { FreeConsole() };
}

#[tokio::test]
async fn console_stderr_allows_detached_worker_launch() {
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process::detached::tests::console_worker_launcher_fixture",
                "--nocapture",
            ])
            .env("NEMO_RELAY_TEST_CONSOLE_WORKER_LAUNCHER", "1")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn windows_spawn_error_identifies_operation_and_os_code() {
    // SAFETY: The last-error value is local to this test's thread.
    unsafe { windows_sys::Win32::Foundation::SetLastError(6) };
    let error = windows_spawn_error("CreateProcessW");
    assert_eq!(error.kind(), std::io::Error::from_raw_os_error(6).kind());
    assert!(error.to_string().contains("CreateProcessW"));
    assert!(error.to_string().contains("os error 6"));
}
