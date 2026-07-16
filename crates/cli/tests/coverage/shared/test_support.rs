// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[must_use]
pub(crate) struct CwdTestScope {
    _guard: MutexGuard<'static, ()>,
    prev: Option<PathBuf>,
}

impl CwdTestScope {
    pub(crate) fn locked() -> Self {
        Self {
            _guard: lock_cwd(),
            prev: None,
        }
    }

    pub(crate) fn enter(path: &Path) -> Self {
        let guard = lock_cwd();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self {
            _guard: guard,
            prev: Some(prev),
        }
    }
}

impl Drop for CwdTestScope {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev
            && let Err(error) = std::env::set_current_dir(prev)
        {
            CWD_RESTORE_FAILED.store(true, Ordering::SeqCst);
            if std::thread::panicking() {
                eprintln!("failed to restore current_dir to {prev:?}: {error}");
            } else {
                panic!("failed to restore current_dir to {prev:?}: {error}");
            }
        }
    }
}

#[must_use]
pub(crate) struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    pub(crate) fn set(values: &[(&'static str, Option<&OsStr>)]) -> Self {
        let guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut previous = Vec::with_capacity(values.len());
        for &(name, value) in values {
            previous.push((name, std::env::var_os(name)));
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..).rev() {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

pub(crate) static CWD_TEST_LOCK: Mutex<()> = Mutex::new(());
static CWD_RESTORE_FAILED: AtomicBool = AtomicBool::new(false);
pub(crate) static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());
pub(crate) static PLUGIN_CONFIG_TEST_LOCK: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

fn lock_cwd() -> MutexGuard<'static, ()> {
    let guard = CWD_TEST_LOCK.lock().expect("CWD_TEST_LOCK poisoned");
    assert!(
        !CWD_RESTORE_FAILED.load(Ordering::SeqCst),
        "current_dir restore failed in a previous test; aborting to prevent cross-test contamination",
    );
    guard
}

pub(crate) fn accept_bounded(listener: &TcpListener) -> std::net::TcpStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                return stream;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for connection"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept failed: {error}"),
        }
    }
}

pub(crate) fn read_headers(stream: &mut std::net::TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "connection closed before complete HTTP headers");
        request.extend_from_slice(&buffer[..count]);
    }
    String::from_utf8(request).unwrap()
}

pub(crate) fn header(request: &str, name: &str) -> String {
    request
        .lines()
        .find_map(|line| {
            let (candidate, value) = line.split_once(':')?;
            candidate
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
        .unwrap_or_else(|| panic!("missing {name} header in {request:?}"))
}
