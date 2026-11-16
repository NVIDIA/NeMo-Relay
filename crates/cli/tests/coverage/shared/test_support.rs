// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, Once};
use std::time::{Duration, Instant};

static TEST_LOGGING: Once = Once::new();

/// Exercise the explicit HTTP host policy over a local connection.
///
/// The absolute DNS name `localhost.` resolves to loopback (RFC 6761), but requires the explicit
/// host allowance because it is neither the literal `localhost` nor a loopback IP in the URL.
/// Avoid binding the default route's interface: it can be a VPN tunnel or change during a test.
pub(crate) async fn http_model_listener() -> (tokio::net::TcpListener, http::uri::Authority) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("localhost.:{}", listener.local_addr().unwrap().port())
        .parse::<http::uri::Authority>()
        .unwrap();
    assert!(crate::configuration::remote_http(
        &reqwest::Url::parse(&format!("http://{address}")).unwrap()
    ));
    (listener, address)
}

pub(crate) fn enable_operational_logs() {
    TEST_LOGGING.call_once(|| {
        let runtime =
            nemo_relay_logging::init_logging(&nemo_relay_logging::LoggingConfig::default())
                .expect("test logging should initialize");
        Box::leak(Box::new(runtime));
    });
}

/// Raise this test process's soft open-file limit toward `minimum`.
///
/// Loopback stress tests that hold many HTTP/1.1 connections at once run the
/// client, relay, and provider in one process, so each concurrent stream holds
/// several descriptors. macOS defaults the soft limit to 256, which is too low
/// for those tests. This is best effort: the soft limit is raised up to the hard
/// limit, and a warning is printed if that is still below `minimum`.
#[cfg(unix)]
pub(crate) fn ensure_open_file_limit(minimum: u64) {
    let minimum = libc::rlim_t::try_from(minimum).unwrap_or(libc::rlim_t::MAX);
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is an initialized `rlimit` that `getrlimit` writes into.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut limit) } != 0 {
        return;
    }
    if limit.rlim_cur >= minimum {
        return;
    }
    let target = minimum.min(limit.rlim_max);
    if target < minimum {
        eprintln!(
            "open-file hard limit {} is below the {minimum} this test needs; it may fail with `Too many open files`",
            limit.rlim_max
        );
    }
    limit.rlim_cur = target;
    // SAFETY: `limit` is an initialized `rlimit` whose soft limit does not exceed its hard limit.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &raw const limit) } != 0 {
        eprintln!(
            "could not raise the open-file soft limit to {target}: {}",
            std::io::Error::last_os_error()
        );
    }
}

/// No open-file soft limit to raise on this platform.
#[cfg(not(unix))]
pub(crate) fn ensure_open_file_limit(_minimum: u64) {}

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

#[path = "plugin_host.rs"]
pub(crate) mod plugin_host;
