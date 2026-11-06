// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The browser sign-in: an authorization code with PKCE (RFC 7636), returned to a loopback
//! redirect (RFC 8252) or pasted back into the terminal when the browser runs on another device.

use std::time::Duration;

use axum::Router;
use axum::extract::{Query, State};
use axum::response::Html;
use axum::routing::get;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use super::auth_error;
use crate::error::CliError;

/// How long the command waits for the approval.
pub(crate) const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);
const CALLBACK_PATH: &str = "/callback";

const DONE_PAGE: &str = "<!doctype html><title>NeMo Relay</title><p>Signed in. You can close this tab and return to the terminal.</p>";
const FAILED_PAGE: &str = "<!doctype html><title>NeMo Relay</title><p>The sign-in was not completed. Return to the terminal.</p>";
const FOREIGN_PAGE: &str = "<!doctype html><title>NeMo Relay</title><p>This page belongs to another sign-in attempt. Use the link the terminal printed.</p>";

/// How a browser sign-in reaches the user and comes back.
pub(crate) struct BrowserApproval<'a> {
    /// Loopback port for the redirect; 0 picks a free one.
    pub(crate) port: u16,
    /// Opens the authorization address; returns whether a browser started.
    pub(crate) open: &'a (dyn Fn(&str) -> bool + Send + Sync),
    /// Redirect addresses pasted into the terminal.
    pub(crate) pasted: mpsc::Receiver<String>,
    pub(crate) timeout: Duration,
}

/// 32 random bytes as unpadded base64url: the PKCE verifier and the OAuth state value.
pub(crate) fn random_value() -> Result<String, CliError> {
    let mut bytes = [0u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| auth_error("the system random number generator failed".into()))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// The S256 code challenge for a verifier.
pub(crate) fn code_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// The query of a redirect back from the issuer.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
pub(crate) struct Redirect {
    pub(crate) code: Option<String>,
    pub(crate) state: Option<String>,
    pub(crate) iss: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) error_description: Option<String>,
}

/// Reads a pasted redirect address. Accepts the address with or without angle brackets.
pub(crate) fn parse_pasted(text: &str) -> Option<Redirect> {
    let trimmed = text
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>')
        .trim();
    let url = reqwest::Url::parse(trimmed).ok()?;
    let mut redirect = Redirect::default();
    for (key, value) in url.query_pairs() {
        let value = Some(value.into_owned());
        match key.as_ref() {
            "code" => redirect.code = value,
            "state" => redirect.state = value,
            "iss" => redirect.iss = value,
            "error" => redirect.error = value,
            "error_description" => redirect.error_description = value,
            _ => {}
        }
    }
    (redirect.code.is_some() || redirect.error.is_some()).then_some(redirect)
}

/// The loopback listener the issuer redirects to.
pub(crate) struct Callback {
    pub(crate) redirect_uri: String,
    received: mpsc::Receiver<Redirect>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

#[derive(Clone)]
struct CallbackState {
    expected_state: String,
    sender: mpsc::Sender<Redirect>,
}

impl Callback {
    /// Listens on `127.0.0.1:<port>`; port 0 picks a free one.
    pub(crate) async fn listen(port: u16, expected_state: &str) -> Result<Self, CliError> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .map_err(|error| {
                auth_error(format!(
                    "cannot listen on 127.0.0.1:{port} for the sign-in redirect: {error}"
                ))
            })?;
        let port = listener.local_addr()?.port();
        let (sender, received) = mpsc::channel(4);
        let (shutdown, stop) = tokio::sync::oneshot::channel::<()>();
        let app = Router::new()
            .route(CALLBACK_PATH, get(receive))
            .with_state(CallbackState {
                expected_state: expected_state.to_owned(),
                sender,
            });
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stop.await;
                })
                .await;
        });
        Ok(Self {
            redirect_uri: format!("http://127.0.0.1:{port}{CALLBACK_PATH}"),
            received,
            shutdown: Some(shutdown),
        })
    }

    /// Waits for the redirect that carries this attempt's state, from the browser or from a
    /// pasted address. A paste for another attempt is reported through `show` and ignored.
    pub(crate) async fn wait(
        mut self,
        expected_state: &str,
        mut pasted: mpsc::Receiver<String>,
        mut show: impl FnMut(String),
        timeout: Duration,
    ) -> Result<Redirect, CliError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut paste_open = true;
        let result = loop {
            tokio::select! {
                Some(redirect) = self.received.recv() => break Ok(redirect),
                line = pasted.recv(), if paste_open => match line {
                    None => paste_open = false,
                    Some(line) if line.trim().is_empty() => {}
                    Some(line) => match parse_pasted(&line) {
                        Some(redirect) if redirect.state.as_deref() == Some(expected_state) => {
                            break Ok(redirect);
                        }
                        Some(_) => show("That address belongs to another sign-in attempt. Paste the address of the tab this command opened.".into()),
                        None => show("That is not a sign-in redirect. Paste the whole http://127.0.0.1:<port>/callback?code=...&state=... address from the browser.".into()),
                    },
                },
                () = tokio::time::sleep_until(deadline) => {
                    break Err(auth_error("timed out after 5 minutes waiting for the sign-in; run the login again".into()));
                }
            }
        };
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        result
    }
}

async fn receive(
    State(state): State<CallbackState>,
    Query(redirect): Query<Redirect>,
) -> Html<&'static str> {
    if redirect.state.as_deref() != Some(state.expected_state.as_str()) {
        return Html(FOREIGN_PAGE);
    }
    let page = if redirect.code.is_some() {
        DONE_PAGE
    } else {
        FAILED_PAGE
    };
    let _ = state.sender.send(redirect).await;
    Html(page)
}

/// Lines typed into the terminal, read on a thread of their own so a blocked read never holds
/// the runtime open. The channel closes at end of input.
pub(crate) fn terminal_lines() -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel(4);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        while matches!(stdin.read_line(&mut line), Ok(read) if read > 0) {
            if sender.blocking_send(std::mem::take(&mut line)).is_err() {
                break;
            }
        }
    });
    receiver
}

/// Opens the address in the default browser. Returns whether a browser was started.
pub(crate) fn open_browser(url: &str) -> bool {
    let mut command = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        let mut command = std::process::Command::new("rundll32");
        command.arg("url.dll,FileProtocolHandler");
        command
    } else if std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()
    {
        std::process::Command::new("xdg-open")
    } else {
        return false;
    };
    command
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}
