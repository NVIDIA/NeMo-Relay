// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pinned per-user TLS identity for authenticated bootstrap hook delivery.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

const IDENTITY_FILE: &str = "hook-tls-identity.json";
/// Certificate an administrator publishes in the system config directory for a daemon that runs
/// under another account, such as a system service account.
pub(crate) const DAEMON_CERTIFICATE_FILE: &str = "hook-tls-certificate.pem";
const CERTIFICATE_LABEL: &str = "CERTIFICATE";

#[derive(Deserialize, Serialize)]
struct IdentityRecord {
    certificate_der: Vec<u8>,
    private_key_der: Vec<u8>,
}

pub(crate) struct RelayTlsIdentity {
    record: IdentityRecord,
}

impl RelayTlsIdentity {
    pub(crate) fn load_or_create() -> Result<Self, String> {
        // Gateways share this identity even when they listen on different ports.
        // Serialize the check and publication so no live server's pin is replaced.
        let state = crate::bootstrap::state::state_dir()?;
        let _lock = crate::bootstrap::state::lock_endpoint(&state, IDENTITY_FILE)?;
        let path = identity_path()?;
        if path.exists() {
            return Self::load();
        }
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()])
            .map_err(|error| format!("failed to generate Relay TLS identity: {error}"))?;
        let record = IdentityRecord {
            certificate_der: certified.cert.der().to_vec(),
            private_key_der: certified.key_pair.serialize_der(),
        };
        let bytes = serde_json::to_vec(&record)
            .map_err(|error| format!("failed to encode Relay TLS identity: {error}"))?;
        crate::filesystem::atomic_write_private(&path, &bytes)?;
        Ok(Self { record })
    }

    pub(crate) fn load() -> Result<Self, String> {
        let path = identity_path()?;
        let bytes = crate::filesystem::bounded::read_bounded_regular_file(
            &path,
            "Relay hook TLS identity",
        )?;
        let record = serde_json::from_slice::<IdentityRecord>(&bytes)
            .map_err(|error| format!("invalid Relay TLS identity {}: {error}", path.display()))?;
        if record.certificate_der.is_empty() || record.private_key_der.is_empty() {
            return Err(format!(
                "Relay TLS identity {} is incomplete",
                path.display()
            ));
        }
        Ok(Self { record })
    }

    /// Loads this account's identity, or `None` when it has never been created.
    pub(crate) fn load_if_present() -> Result<Option<Self>, String> {
        match std::fs::symlink_metadata(identity_path()?) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            _ => Self::load().map(Some),
        }
    }

    /// The public certificate as one PEM `CERTIFICATE` block. The private key is never exported.
    pub(crate) fn certificate_pem(&self) -> String {
        let encoded =
            base64::engine::general_purpose::STANDARD.encode(&self.record.certificate_der);
        let mut pem = format!("-----BEGIN {CERTIFICATE_LABEL}-----\n");
        for line in encoded.as_bytes().chunks(64) {
            pem.push_str(std::str::from_utf8(line).expect("base64 output is ASCII"));
            pem.push('\n');
        }
        pem.push_str(&format!("-----END {CERTIFICATE_LABEL}-----\n"));
        pem
    }

    pub(crate) fn server_config(&self) -> Result<Arc<rustls::ServerConfig>, String> {
        ensure_crypto_provider();
        let certificate = CertificateDer::from(self.record.certificate_der.clone());
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            self.record.private_key_der.clone(),
        ));
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate], key)
            .map(Arc::new)
            .map_err(|error| format!("invalid Relay TLS server identity: {error}"))
    }

    pub(crate) fn client_config(&self) -> Result<Arc<rustls::ClientConfig>, String> {
        pinned_client_config(vec![self.record.certificate_der.clone()])
    }
}

/// Pinned client trust for a local HTTP daemon address.
///
/// A daemon run by the same user serves this account's own identity. A daemon run under another
/// account, such as a system service, serves an identity this user cannot read, so an administrator
/// publishes its certificate at [`daemon_certificate_path`]. Only these certificates are trusted;
/// the platform trust store is never consulted.
pub(crate) fn daemon_client_config() -> Result<Arc<rustls::ClientConfig>, String> {
    let mut certificates = Vec::new();
    if let Some(identity) = RelayTlsIdentity::load_if_present()? {
        certificates.push(identity.record.certificate_der);
    }
    let published = daemon_certificate_path();
    if let Some(certificate) = load_published_daemon_certificate(&published)? {
        certificates.push(certificate);
    }
    if certificates.is_empty() {
        return Err(format!(
            "no pinned certificate for the daemon: {} does not exist and no daemon certificate is published at {}; run the hook as the daemon's user, or publish the daemon's certificate with `nemo-relay daemon tls-certificate`",
            identity_path()?.display(),
            published.display()
        ));
    }
    pinned_client_config(certificates)
}

/// Where an administrator publishes the certificate of a daemon that runs under another account.
pub(crate) fn daemon_certificate_path() -> PathBuf {
    crate::configuration::system_config_dir().join(DAEMON_CERTIFICATE_FILE)
}

fn load_published_daemon_certificate(path: &Path) -> Result<Option<Vec<u8>>, String> {
    let description = "published daemon TLS certificate";
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "failed to inspect {description} {}: {error}",
                path.display()
            ));
        }
    };
    // Check the type first: on Linux a symlink's own mode is 0777, which would otherwise be
    // reported as a permission problem instead of the real one.
    if !metadata.file_type().is_file() {
        return Err(format!(
            "{description} {} must be a regular file",
            path.display()
        ));
    }
    // A user who can replace this file could make hooks trust a listener that squats the port.
    ensure_administrator_controlled(path, &metadata, description)?;
    if let Some(directory) = path.parent() {
        let metadata = std::fs::metadata(directory).map_err(|error| {
            format!(
                "failed to inspect the directory of the {description} {}: {error}",
                directory.display()
            )
        })?;
        ensure_administrator_controlled(
            directory,
            &metadata,
            "directory of the published daemon TLS certificate",
        )?;
    }
    let bytes = crate::filesystem::bounded::read_bounded_regular_file(path, description)?;
    let mut certificates = decode_pem_blocks(&bytes, CERTIFICATE_LABEL)
        .map_err(|error| format!("invalid {description} {}: {error}", path.display()))?;
    if certificates.len() != 1 {
        return Err(format!(
            "{description} {} must contain exactly one {CERTIFICATE_LABEL} block",
            path.display()
        ));
    }
    Ok(Some(certificates.remove(0)))
}

#[cfg(unix)]
fn ensure_administrator_controlled(
    path: &Path,
    metadata: &std::fs::Metadata,
    description: &str,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    // SAFETY: geteuid has no preconditions and cannot fail.
    let current_user = unsafe { libc::geteuid() };
    if metadata.uid() != 0 && metadata.uid() != current_user {
        return Err(format!(
            "{description} {} must be owned by root or the current user",
            path.display()
        ));
    }
    if metadata.mode() & 0o022 != 0 {
        return Err(format!(
            "{description} {} must not be writable by group or other users",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_administrator_controlled(
    _path: &Path,
    _metadata: &std::fs::Metadata,
    _description: &str,
) -> Result<(), String> {
    // As for the other system-layer files, the documented ACL on the Windows system config
    // directory gives users read access only (docs/daemon/configuration.mdx).
    Ok(())
}

fn pinned_client_config(certificates: Vec<Vec<u8>>) -> Result<Arc<rustls::ClientConfig>, String> {
    ensure_crypto_provider();
    let mut roots = rustls::RootCertStore::empty();
    for certificate in certificates {
        roots
            .add(CertificateDer::from(certificate))
            .map_err(|error| format!("invalid pinned Relay TLS certificate: {error}"))?;
    }
    Ok(Arc::new(
        rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth(),
    ))
}

/// Decodes every PEM block with `label`, ignoring text outside the blocks.
pub(crate) fn decode_pem_blocks(bytes: &[u8], label: &str) -> Result<Vec<Vec<u8>>, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("TLS PEM is not UTF-8: {error}"))?;
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let mut blocks = Vec::new();
    let mut remainder = text;
    while let Some((_, after_begin)) = remainder.split_once(&begin) {
        let Some((encoded, after_end)) = after_begin.split_once(&end) else {
            return Err(format!("TLS PEM has an unterminated {label} block"));
        };
        let compact = encoded
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
            .collect::<String>();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(compact)
            .map_err(|_| format!("TLS PEM contains invalid {label} base64"))?;
        if decoded.is_empty() {
            return Err(format!("TLS PEM contains an empty {label} block"));
        }
        blocks.push(decoded);
        remainder = after_end;
    }
    Ok(blocks)
}

fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

fn identity_path() -> Result<PathBuf, String> {
    Ok(crate::bootstrap::state::state_dir()?.join(IDENTITY_FILE))
}

/// Upgrade an HTTP connection to TLS while limiting concurrent handshakes.
pub(crate) fn upgrade(
    mut request: axum::http::Request<axum::body::Body>,
    tls: Arc<rustls::ServerConfig>,
    app: axum::Router,
    handshake_permits: Arc<Semaphore>,
) -> axum::response::Response {
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    if request
        .headers()
        .get(header::UPGRADE)
        .and_then(|value| value.to_str().ok())
        != Some("nemo-relay-tls")
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let Ok(handshake_permit) = handshake_permits.try_acquire_owned() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let upgrade = hyper::upgrade::on(&mut request);
    tokio::spawn(async move {
        let Ok(upgraded) = upgrade.await else {
            return;
        };
        let acceptor = tokio_rustls::TlsAcceptor::from(tls);
        let handshake = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            acceptor.accept(hyper_util::rt::TokioIo::new(upgraded)),
        )
        .await;
        drop(handshake_permit);
        let Ok(Ok(encrypted)) = handshake else {
            return;
        };
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(
                hyper_util::rt::TokioIo::new(encrypted),
                hyper_util::service::TowerToHyperService::new(app),
            )
            .await;
    });
    axum::http::Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(header::CONNECTION, "upgrade")
        .header(header::UPGRADE, "nemo-relay-tls")
        .header(header::CONTENT_LENGTH, "0")
        .body(axum::body::Body::empty())
        .expect("TLS upgrade headers are valid")
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/gateway_tls_tests.rs"]
mod tests;
