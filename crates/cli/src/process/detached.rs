// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal cross-platform process detachment for the shared gateway.

use std::process::Command;

#[cfg(not(windows))]
use std::process::Child;

#[cfg(windows)]
use std::sync::Mutex;

#[cfg(windows)]
static SIDECAR_SPAWN_LOCK: Mutex<()> = Mutex::new(());

#[cfg(windows)]
fn wide_nul(value: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn append_quoted(command_line: &mut Vec<u16>, value: &std::ffi::OsStr) {
    use std::os::windows::ffi::OsStrExt;

    const BACKSLASH: u16 = b'\\' as u16;
    const QUOTE: u16 = b'"' as u16;
    command_line.push(QUOTE);
    let mut slashes = 0;
    for unit in value.encode_wide() {
        if unit == BACKSLASH {
            slashes += 1;
            continue;
        }
        if unit == QUOTE {
            command_line.extend(std::iter::repeat_n(BACKSLASH, slashes * 2 + 1));
        } else {
            command_line.extend(std::iter::repeat_n(BACKSLASH, slashes));
        }
        slashes = 0;
        command_line.push(unit);
    }
    command_line.extend(std::iter::repeat_n(BACKSLASH, slashes * 2));
    command_line.push(QUOTE);
}

#[cfg(windows)]
fn environment_block(command: &Command) -> Vec<u16> {
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStrExt;

    let mut environment = BTreeMap::<String, (OsString, OsString)>::new();
    for (name, value) in std::env::vars_os() {
        environment.insert(name.to_string_lossy().to_uppercase(), (name, value));
    }
    for (name, value) in command.get_envs() {
        let key = name.to_string_lossy().to_uppercase();
        if let Some(value) = value {
            environment.insert(key, (name.to_owned(), value.to_owned()));
        } else {
            environment.remove(&key);
        }
    }
    let mut block = Vec::new();
    for (_, (name, value)) in environment {
        block.extend(name.encode_wide());
        block.push(b'=' as u16);
        block.extend(value.encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

#[cfg(windows)]
struct AttributeList(*mut std::ffi::c_void);

#[cfg(windows)]
impl Drop for AttributeList {
    fn drop(&mut self) {
        // SAFETY: The list was initialized successfully and is still live.
        unsafe { windows_sys::Win32::System::Threading::DeleteProcThreadAttributeList(self.0) };
    }
}

#[cfg(not(windows))]
pub(crate) type DetachedChild = Child;

#[cfg(windows)]
pub(crate) struct DetachedChild {
    process: windows_sys::Win32::Foundation::HANDLE,
    thread: windows_sys::Win32::Foundation::HANDLE,
    id: u32,
}

#[cfg(windows)]
unsafe impl Send for DetachedChild {}

#[cfg(windows)]
impl DetachedChild {
    pub(crate) fn id(&self) -> u32 {
        self.id
    }

    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        use std::os::windows::process::ExitStatusExt;
        use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
        use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};

        // SAFETY: `process` is owned by this value until Drop.
        match unsafe { WaitForSingleObject(self.process, 0) } {
            WAIT_TIMEOUT => Ok(None),
            WAIT_OBJECT_0 => {
                let mut code = 0;
                // SAFETY: The process handle and output pointer are valid.
                if unsafe { GetExitCodeProcess(self.process, &mut code) } == 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(Some(std::process::ExitStatus::from_raw(code)))
                }
            }
            _ => Err(std::io::Error::last_os_error()),
        }
    }

    pub(crate) fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};

        // SAFETY: `process` is owned by this value until Drop.
        if unsafe { WaitForSingleObject(self.process, INFINITE) } != WAIT_OBJECT_0 {
            return Err(std::io::Error::last_os_error());
        }
        self.try_wait()?.ok_or_else(|| {
            std::io::Error::other("detached gateway was still running after a completed wait")
        })
    }
}

#[cfg(windows)]
impl Drop for DetachedChild {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;

        // SAFETY: Both handles were returned by CreateProcessW and are owned here.
        unsafe {
            CloseHandle(self.thread);
            CloseHandle(self.process);
        }
    }
}

#[cfg(windows)]
fn spawn_detached_with_handle_list(command: &Command) -> std::io::Result<DetachedChild> {
    use std::ffi::c_void;
    use std::fs::OpenOptions;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{HANDLE, HANDLE_FLAG_INHERIT, SetHandleInformation};
    use windows_sys::Win32::System::Threading::{
        CREATE_UNICODE_ENVIRONMENT, CreateProcessW, EXTENDED_STARTUPINFO_PRESENT,
        InitializeProcThreadAttributeList, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION,
        STARTF_USESTDHANDLES, STARTUPINFOEXW, UpdateProcThreadAttribute,
    };

    let stdin = OpenOptions::new().read(true).open(r"\\.\NUL")?;
    let stdout = OpenOptions::new().write(true).open(r"\\.\NUL")?;
    let handles = [
        stdin.as_raw_handle().cast::<c_void>(),
        stdout.as_raw_handle().cast::<c_void>(),
    ];
    for handle in handles {
        // SAFETY: These are live handles owned by `stdin` and `stdout`.
        if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    let mut attribute_bytes = 0;
    // SAFETY: A null first call obtains the required allocation size.
    unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut attribute_bytes) };
    if attribute_bytes == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let words = attribute_bytes.div_ceil(std::mem::size_of::<usize>());
    let mut attribute_storage = vec![0_usize; words];
    let attribute_pointer = attribute_storage.as_mut_ptr().cast::<c_void>();
    // SAFETY: The aligned allocation has the size returned by the sizing call.
    if unsafe { InitializeProcThreadAttributeList(attribute_pointer, 1, 0, &mut attribute_bytes) }
        == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let attribute_list = AttributeList(attribute_pointer);
    // SAFETY: `handles` remains live through CreateProcessW and contains only the intended stdio.
    if unsafe {
        UpdateProcThreadAttribute(
            attribute_list.0,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
            handles.as_ptr().cast(),
            std::mem::size_of_val(&handles),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }

    let program = wide_nul(command.get_program());
    let mut command_line = Vec::new();
    append_quoted(&mut command_line, command.get_program());
    for argument in command.get_args() {
        command_line.push(b' ' as u16);
        append_quoted(&mut command_line, argument);
    }
    command_line.push(0);
    let environment = environment_block(command);
    let current_dir = command
        .get_current_dir()
        .map(|path| wide_nul(path.as_os_str()));

    let mut startup = STARTUPINFOEXW::default();
    startup.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = handles[0] as HANDLE;
    startup.StartupInfo.hStdOutput = handles[1] as HANDLE;
    startup.StartupInfo.hStdError = handles[1] as HANDLE;
    startup.lpAttributeList = attribute_list.0;
    let mut process = PROCESS_INFORMATION::default();
    let (in_job, limits) = current_windows_job_limits();
    let (creation_flags, _) = windows_creation_flags(in_job, limits);
    // SAFETY: Every pointer references initialized storage that remains live for this call.
    let created = unsafe {
        CreateProcessW(
            program.as_ptr(),
            command_line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            creation_flags | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            environment.as_ptr().cast(),
            current_dir
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            (&raw const startup).cast(),
            &mut process,
        )
    };
    let create_error = (created == 0).then(std::io::Error::last_os_error);
    for handle in handles {
        // SAFETY: The handles remain live; clear inheritance before releasing the spawn lock.
        unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0) };
    }
    if let Some(error) = create_error {
        return Err(error);
    }
    Ok(DetachedChild {
        process: process.hProcess,
        thread: process.hThread,
        id: process.dwProcessId,
    })
}

#[cfg(windows)]
pub(crate) fn spawn_detached(command: &mut Command) -> std::io::Result<DetachedChild> {
    let _spawn_guard = SIDECAR_SPAWN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    spawn_detached_with_handle_list(command)
}

#[cfg(not(windows))]
pub(crate) fn spawn_detached(command: &mut Command) -> std::io::Result<DetachedChild> {
    command.spawn()
}

#[cfg(unix)]
pub(crate) fn configure_detached(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    // SAFETY: setsid is async-signal-safe and runs in the post-fork child before exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_CREATE_NO_WINDOW: u32 = 0x0800_0000;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK: u32 = 0x0000_0800;
#[cfg(any(test, windows))]
pub(crate) const WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK: u32 = 0x0000_1000;

#[cfg(any(test, windows))]
pub(crate) fn windows_creation_flags(in_job: bool, job_limit_flags: Option<u32>) -> (u32, bool) {
    let base = WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW;
    if !in_job {
        return (base, false);
    }
    match job_limit_flags {
        Some(flags) if flags & WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK != 0 => {
            (base | WINDOWS_CREATE_BREAKAWAY_FROM_JOB, false)
        }
        Some(flags) if flags & WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK != 0 => (base, false),
        Some(_) | None => (base, true),
    }
}

#[cfg(windows)]
fn current_windows_job_limits() -> (bool, Option<u32>) {
    use windows_sys::Win32::System::JobObjects::{
        IsProcessInJob, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut in_job = 0;
    // SAFETY: The pseudo current-process handle and null current-job handle are valid here.
    if unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } == 0 {
        return (true, None);
    }
    if in_job == 0 {
        return (false, Some(0));
    }
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    // SAFETY: The output buffer matches the requested information class.
    let queried = unsafe {
        QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectExtendedLimitInformation,
            std::ptr::from_mut(&mut limits).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        )
    };
    if queried == 0 {
        (true, None)
    } else {
        (true, Some(limits.BasicLimitInformation.LimitFlags))
    }
}

#[cfg(windows)]
pub(crate) fn configure_detached(_command: &mut Command) {
    let (in_job, limits) = current_windows_job_limits();
    let (_, limited_lifetime) = windows_creation_flags(in_job, limits);
    if limited_lifetime {
        eprintln!(
            "warning: the current Windows Job Object does not permit process breakaway; the shared Relay gateway lifetime is limited to the host job"
        );
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn configure_detached(_command: &mut Command) {}

pub(crate) fn terminate_tree(child: &mut DetachedChild) {
    #[cfg(unix)]
    {
        let process_group = -(child.id() as i32);
        // SAFETY: Detached gateways call setsid, so the child PID is the process-group ID.
        if unsafe { libc::kill(process_group, libc::SIGKILL) } == -1 {
            let _ = child.kill();
        }
    }
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .status();
        if !status.is_ok_and(|status| status.success()) {
            eprintln!(
                "failed to terminate detached gateway process tree {} with taskkill",
                child.id()
            );
            return;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}
