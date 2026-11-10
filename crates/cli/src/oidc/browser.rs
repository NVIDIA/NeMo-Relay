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
use tokio::sync::{mpsc, oneshot};

use super::client::OidcClient;
use super::error::SignInError;
use super::session::Session;
use crate::error::CliError;

/// How long the command waits for the user to approve the sign-in.
pub(crate) const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);
/// Bytes of randomness in the PKCE verifier and in the OAuth state.
const RANDOM_BYTES: usize = 32;
const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PATH: &str = "/callback";

const PAGE_RECEIVED: &str = "<!doctype html><title>NeMo Relay</title><p>The sign-in response was received. Return to the terminal to see the result.</p>";
const PAGE_NOT_COMPLETED: &str = "<!doctype html><title>NeMo Relay</title><p>The sign-in was not completed. Return to the terminal.</p>";
const PAGE_FOREIGN_ATTEMPT: &str = "<!doctype html><title>NeMo Relay</title><p>This page belongs to another sign-in attempt. Use the link the terminal printed.</p>";

const PASTE_FOREIGN_ATTEMPT: &str = "That address belongs to another sign-in attempt. Paste the address of the tab this command opened.";
const PASTE_NOT_A_REDIRECT: &str = "That is not a sign-in redirect. Paste the whole http://127.0.0.1:<port>/callback?code=...&state=... address from the browser.";

/// Opens the authorization address; returns whether a browser started.
pub(crate) type Opener = dyn Fn(&str) -> bool + Send + Sync;

/// How a browser sign-in reaches the user and comes back.
pub(crate) struct BrowserApproval<'a> {
    /// Loopback port for the redirect; 0 picks a free one.
    pub(crate) port: u16,
    pub(crate) open: &'a Opener,
    /// Redirect addresses pasted into the terminal.
    pub(crate) pasted: mpsc::Receiver<String>,
    pub(crate) timeout: Duration,
}

impl BrowserApproval<'static> {
    /// The approval `nemo-relay auth login` uses: the system browser unless `open_browser` is
    /// false, and redirect addresses typed into the terminal.
    pub(crate) fn interactive(open_browser: bool, port: u16) -> Self {
        Self {
            port,
            open: if open_browser {
                &open_in_system_browser
            } else {
                &print_only
            },
            pasted: terminal_lines(),
            timeout: APPROVAL_TIMEOUT,
        }
    }
}

/// Runs the browser sign-in and returns the issued tokens.
pub(crate) async fn sign_in(
    client: &OidcClient,
    scope: &str,
    state: &str,
    approval: BrowserApproval<'_>,
    mut show: impl FnMut(String),
) -> Result<Session, CliError> {
    let verifier = random_value()?;
    let callback = Callback::listen(approval.port, state).await?;
    let redirect_uri = callback.redirect_uri.clone();
    let url = client.authorization_url(scope, &redirect_uri, state, &code_challenge(&verifier))?;
    announce(&url, (approval.open)(&url), &redirect_uri, &mut show);
    let redirect = callback
        .wait(approval.pasted, &mut show, approval.timeout)
        .await?;
    let code = redirect.into_code(client.metadata_issuer())?;
    client.exchange_code(&code, &redirect_uri, &verifier).await
}

fn announce(url: &str, opened: bool, redirect_uri: &str, show: &mut impl FnMut(String)) {
    show(if opened {
        format!("Opened the browser to sign in. If it did not open, visit:\n{url}")
    } else {
        format!("To sign in, open this address in a browser on any device:\n{url}")
    });
    show(format!(
        "If the browser runs on another device, its last page fails to load. Paste that page's address ({redirect_uri}?code=...) here and press Enter."
    ));
}

/// Unpadded base64url of fresh random bytes: the PKCE verifier and the OAuth state value.
pub(crate) fn random_value() -> Result<String, SignInError> {
    let mut bytes = [0u8; RANDOM_BYTES];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| SignInError::NoRandomness)?;
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

impl Redirect {
    /// The authorization code, once the redirect is known to come from `issuer` and approve.
    /// RFC 9207 compares `iss` with the metadata issuer as an exact string.
    fn into_code(self, issuer: &str) -> Result<String, SignInError> {
        if self.iss.is_some_and(|iss| iss != issuer) {
            return Err(SignInError::ForeignIssuer);
        }
        if let Some(error) = &self.error {
            return Err(SignInError::from_redirect_error(
                error,
                self.error_description.as_deref(),
            ));
        }
        self.code.ok_or(SignInError::MissingCode)
    }
}

/// A line typed into the terminal while the sign-in waits.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Paste {
    Blank,
    NotARedirect,
    ForeignAttempt,
    Accepted(Redirect),
}

/// Reads a pasted redirect address, with or without angle brackets around it.
pub(crate) fn read_paste(line: &str, expected_state: &str) -> Paste {
    let text = line.trim().trim_start_matches('<').trim_end_matches('>');
    if text.is_empty() {
        return Paste::Blank;
    }
    let Some(redirect) = text
        .parse::<http::Uri>()
        .ok()
        .and_then(|uri| Query::<Redirect>::try_from_uri(&uri).ok())
        .map(|Query(redirect)| redirect)
        .filter(|redirect| redirect.code.is_some() || redirect.error.is_some())
    else {
        return Paste::NotARedirect;
    };
    if redirect.state.as_deref() == Some(expected_state) {
        Paste::Accepted(redirect)
    } else {
        Paste::ForeignAttempt
    }
}

/// The loopback listener the issuer redirects to.
pub(crate) struct Callback {
    pub(crate) redirect_uri: String,
    expected_state: String,
    received: mpsc::Receiver<Redirect>,
    stop: oneshot::Sender<()>,
}

#[derive(Clone)]
struct CallbackState {
    expected_state: String,
    sender: mpsc::Sender<Redirect>,
}

impl Callback {
    /// Listens on `127.0.0.1:<port>`; port 0 picks a free one.
    pub(crate) async fn listen(port: u16, expected_state: &str) -> Result<Self, SignInError> {
        let listener = tokio::net::TcpListener::bind((CALLBACK_HOST, port))
            .await
            .map_err(|source| SignInError::CallbackUnavailable { port, source })?;
        let bound_port = listener
            .local_addr()
            .map_err(|source| SignInError::CallbackUnavailable { port, source })?
            .port();
        let (sender, received) = mpsc::channel(4);
        let (stop, stopped) = oneshot::channel::<()>();
        let state = CallbackState {
            expected_state: expected_state.to_owned(),
            sender,
        };
        let app = Router::new()
            .route(CALLBACK_PATH, get(receive))
            .with_state(state);
        tokio::spawn(serve_until_stopped(listener, app, stopped));
        Ok(Self {
            redirect_uri: format!("http://{CALLBACK_HOST}:{bound_port}{CALLBACK_PATH}"),
            expected_state: expected_state.to_owned(),
            received,
            stop,
        })
    }

    /// Waits for the redirect of this attempt, from the browser or pasted, then stops listening.
    pub(crate) async fn wait(
        mut self,
        pasted: mpsc::Receiver<String>,
        show: impl FnMut(String),
        timeout: Duration,
    ) -> Result<Redirect, SignInError> {
        let first = tokio::time::timeout(timeout, self.first_redirect(pasted, show)).await;
        let _ = self.stop.send(());
        first.map_err(|_| SignInError::TimedOut)
    }

    async fn first_redirect(
        &mut self,
        mut pasted: mpsc::Receiver<String>,
        mut show: impl FnMut(String),
    ) -> Redirect {
        let mut paste_open = true;
        loop {
            tokio::select! {
                Some(redirect) = self.received.recv() => return redirect,
                line = pasted.recv(), if paste_open => match line.map(|line| read_paste(&line, &self.expected_state)) {
                    None => paste_open = false,
                    Some(Paste::Accepted(redirect)) => return redirect,
                    Some(Paste::Blank) => {}
                    Some(Paste::ForeignAttempt) => show(PASTE_FOREIGN_ATTEMPT.into()),
                    Some(Paste::NotARedirect) => show(PASTE_NOT_A_REDIRECT.into()),
                },
                else => std::future::pending::<()>().await,
            }
        }
    }
}

async fn serve_until_stopped(
    listener: tokio::net::TcpListener,
    app: Router,
    stopped: oneshot::Receiver<()>,
) {
    let shutdown = async {
        let _ = stopped.await;
    };
    let _ = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;
}

/// Answers the browser; only a redirect with this attempt's state reaches the command.
async fn receive(
    State(state): State<CallbackState>,
    Query(redirect): Query<Redirect>,
) -> Html<&'static str> {
    if redirect.state.as_deref() != Some(state.expected_state.as_str()) {
        return Html(PAGE_FOREIGN_ATTEMPT);
    }
    let page = if redirect.code.is_some() {
        PAGE_RECEIVED
    } else {
        PAGE_NOT_COMPLETED
    };
    let _ = state.sender.send(redirect).await;
    Html(page)
}

/// Lines typed into the terminal, read on a thread of their own so a blocked read never holds
/// the runtime open. The channel closes at end of input.
fn terminal_lines() -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel(4);
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if sender.blocking_send(line).is_err() {
                break;
            }
        }
    });
    receiver
}

/// Opens the address in the default browser; returns whether one started.
fn open_in_system_browser(url: &str) -> bool {
    let Some(mut command) = system_browser_command() else {
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

/// The platform's command that opens an address. On Linux, only with a graphical display.
fn system_browser_command() -> Option<std::process::Command> {
    if cfg!(target_os = "macos") {
        return Some(std::process::Command::new("open"));
    }
    if cfg!(windows) {
        let mut command = std::process::Command::new("rundll32");
        command.arg("url.dll,FileProtocolHandler");
        return Some(command);
    }
    let has_display =
        std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some();
    has_display.then(|| std::process::Command::new("xdg-open"))
}

/// Prints the address only, for `--no-browser`.
fn print_only(_url: &str) -> bool {
    false
}
