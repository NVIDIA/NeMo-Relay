// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(unix)]
async fn wait_for_published_pid(path: &std::path::Path, process: &str) -> i32 {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(pid) = std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
        {
            return pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{process} did not publish a complete PID"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[test]
fn wrapper_probe_uses_last_host_token_and_validates_opaque_wrappers() {
    assert_eq!(
        version_probe_argv(
            CodingAgent::Codex,
            &command_argv("npm exec --package @openai/codex -- codex exec")
        ),
        [
            "npm",
            "exec",
            "--package",
            "@openai/codex",
            "--",
            "codex",
            "--version"
        ]
    );
    assert_eq!(
        version_probe_argv(
            CodingAgent::Codex,
            &command_argv("custom-codex-wrapper --profile dev")
        ),
        ["custom-codex-wrapper", "--profile", "dev", "--version"]
    );
    assert_eq!(
        version_probe_argv(CodingAgent::Codex, &[]),
        ["codex", "--version"]
    );
}

#[test]
fn platform_resolution_supports_explicit_paths_and_windows_pathext() {
    let temp = tempfile::tempdir().unwrap();
    let shim = temp.path().join("codex.CMD");
    std::fs::write(&shim, "").unwrap();

    assert_eq!(
        resolve_executable_for_platform(
            "codex",
            Some(temp.path().as_os_str()),
            Some(std::ffi::OsStr::new(".EXE;.CMD")),
            true,
        ),
        Some(shim.clone())
    );
    assert_eq!(
        resolve_executable_for_platform(
            shim.to_str().unwrap(),
            None,
            Some(std::ffi::OsStr::new(".EXE;.CMD")),
            true,
        ),
        Some(shim)
    );
    assert_eq!(resolve_executable_for_platform("", None, None, false), None);
}

#[cfg(unix)]
#[tokio::test]
async fn supervised_wait_terminates_descendants_left_by_a_wrapper() {
    let temp = tempfile::tempdir().unwrap();
    let descendant_pid_path = temp.path().join("descendant.pid");
    let argv = vec![
        "sh".into(),
        "-c".into(),
        "sleep 30 & echo $! > \"$1\"; exit 0".into(),
        "sh".into(),
        descendant_pid_path.display().to_string(),
    ];
    let mut command = tokio_command(&argv);
    let mut child = SupervisedChild::spawn(&mut command).await.unwrap();

    let status = tokio::time::timeout(std::time::Duration::from_secs(5), child.wait())
        .await
        .expect("wrapper did not exit")
        .unwrap();

    assert!(status.success());
    let pid = wait_for_published_pid(&descendant_pid_path, "wrapper descendant").await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        // SAFETY: Signal 0 performs an existence check and does not alter the target process.
        let result = unsafe { libc::kill(pid, 0) };
        if result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "wrapper descendant {pid} survived normal wrapper exit"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[cfg(unix)]
#[tokio::test]
async fn supervision_error_terminates_and_reaps_the_child_tree_first() {
    let temp = tempfile::tempdir().unwrap();
    let child_pid_path = temp.path().join("child.pid");
    let argv = vec![
        "sh".into(),
        "-c".into(),
        "echo $$ > \"$1\"; exec sleep 30".into(),
        "sh".into(),
        child_pid_path.display().to_string(),
    ];
    let mut command = tokio_command(&argv);
    let mut child = SupervisedChild::spawn(&mut command).await.unwrap();
    let pid = wait_for_published_pid(&child_pid_path, "supervised child").await;

    let error = child
        .inject_wait_error_for_test(std::io::Error::other("injected wait failure"))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("injected wait failure"));
    // SAFETY: Signal zero only checks whether the reaped test child still exists.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}

#[cfg(windows)]
#[tokio::test]
async fn windows_supervision_assigns_before_a_wrapper_can_spawn_a_descendant() {
    let temp = tempfile::tempdir().unwrap();
    let descendant_pid_path = temp.path().join("descendant.pid");
    let release_path = temp.path().join("release-wrapper");
    let wrapper = temp.path().join("spawn-descendant.ps1");
    std::fs::write(
        &wrapper,
        r#"$ErrorActionPreference = 'Stop'
$start = [System.Diagnostics.ProcessStartInfo]::new()
$start.FileName = (Get-Process -Id $PID).Path
$start.Arguments = '-NoProfile -NonInteractive -Command "Start-Sleep -Seconds 30"'
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$child = [System.Diagnostics.Process]::Start($start)
Set-Content -LiteralPath $args[0] -Value $child.Id -Encoding ASCII -NoNewline
$deadline = [DateTime]::UtcNow.AddSeconds(15)
while (-not (Test-Path -LiteralPath $args[1])) {
    if ([DateTime]::UtcNow -ge $deadline) {
        throw 'Relay test did not release the wrapper'
    }
    Start-Sleep -Milliseconds 20
}
"#,
    )
    .unwrap();
    let argv = vec![
        "powershell.exe".into(),
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-ExecutionPolicy".into(),
        "Bypass".into(),
        "-File".into(),
        wrapper.display().to_string(),
        descendant_pid_path.display().to_string(),
        release_path.display().to_string(),
    ];
    let mut command = tokio_command(&argv);
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut child = SupervisedChild::spawn(&mut command).await.unwrap();

    let publish_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let descendant_pid = loop {
        if let Some(process_id) = read_windows_process_id(&descendant_pid_path) {
            break process_id;
        }
        if std::time::Instant::now() >= publish_deadline {
            let _ = child.terminate().await;
            panic!("PowerShell wrapper did not publish its child PID");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    };
    let descendant = match WindowsTestProcess::open(descendant_pid) {
        Ok(descendant) => descendant,
        Err(error) => {
            let _ = child.terminate().await;
            panic!("could not retain wrapper descendant {descendant_pid}: {error}");
        }
    };
    std::fs::write(release_path, b"ready").unwrap();

    let status = match tokio::time::timeout(std::time::Duration::from_secs(15), child.wait()).await
    {
        Ok(status) => status.unwrap(),
        Err(_) => {
            let _ = child.terminate().await;
            panic!("PowerShell wrapper did not exit");
        }
    };

    assert!(status.success());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let escaped = loop {
        if !descendant.is_active().unwrap() {
            break false;
        }
        if std::time::Instant::now() >= deadline {
            break true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    };
    if escaped {
        descendant.terminate();
    }
    assert!(
        !escaped,
        "wrapper descendant {descendant_pid} survived Job Object termination"
    );
}

#[cfg(windows)]
fn read_windows_process_id(path: &std::path::Path) -> Option<u32> {
    std::fs::read_to_string(path).ok()?.parse().ok()
}

#[cfg(windows)]
struct WindowsTestProcess {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
impl WindowsTestProcess {
    fn open(process_id: u32) -> std::io::Result<Self> {
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
        };

        // SAFETY: The wrapper published this live descendant PID while waiting for the test to
        // release it. Holding this handle prevents PID reuse from redirecting later cleanup.
        let handle = unsafe {
            OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE | PROCESS_TERMINATE,
                0,
                process_id,
            )
        };
        if handle.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(Self { handle })
        }
    }

    fn is_active(&self) -> std::io::Result<bool> {
        use windows_sys::Win32::Foundation::STILL_ACTIVE;
        use windows_sys::Win32::System::Threading::GetExitCodeProcess;

        let mut exit_code = 0;
        // SAFETY: `handle` remains live for this guard and `exit_code` is writable storage.
        if unsafe { GetExitCodeProcess(self.handle, &mut exit_code) } == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(exit_code == STILL_ACTIVE as u32)
        }
    }

    fn terminate(&self) {
        use windows_sys::Win32::System::Threading::{TerminateProcess, WaitForSingleObject};

        if matches!(self.is_active(), Ok(false)) {
            return;
        }
        // SAFETY: `handle` identifies the original finite test descendant and was opened with
        // termination and synchronization rights.
        unsafe {
            TerminateProcess(self.handle, 1);
            WaitForSingleObject(self.handle, 5_000);
        }
    }
}

#[cfg(windows)]
impl Drop for WindowsTestProcess {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;

        self.terminate();
        // SAFETY: The guard uniquely owns this handle and closes it exactly once.
        unsafe { CloseHandle(self.handle) };
    }
}

#[cfg(windows)]
#[test]
fn windows_command_shim_preserves_metacharacter_arguments() {
    let temp = tempfile::tempdir().unwrap();
    let shim = temp.path().join("agent shim.cmd");
    let marker = temp.path().join("completed.txt");
    std::fs::write(
        &shim,
        "@echo off\r\n\
         @if not \"%~1\"==\"space & value\" exit /b 11\r\n\
         @if not \"%~2\"==\"caret^value\" exit /b 12\r\n\
         @if not \"%~3\"==\"%%TOKEN%%\" exit /b 13\r\n\
         @echo ok>\"%NEMO_RELAY_ARGV_MARKER%\"\r\n",
    )
    .unwrap();
    let argv = vec![
        shim.display().to_string(),
        "space & value".into(),
        "caret^value".into(),
        "%TOKEN%".into(),
    ];
    let status = std_command(&argv)
        .env("NEMO_RELAY_ARGV_MARKER", &marker)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(std::fs::read_to_string(marker).unwrap().trim(), "ok");
}
