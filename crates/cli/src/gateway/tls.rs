// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pinned per-user TLS identity for authenticated bootstrap hook delivery.

use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};

const IDENTITY_FILE: &str = "hook-tls-identity.json";

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

    pub(crate) fn server_config(&self) -> Result<Arc<rustls::ServerConfig>, String> {
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
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(CertificateDer::from(self.record.certificate_der.clone()))
            .map_err(|error| format!("invalid pinned Relay TLS certificate: {error}"))?;
        Ok(Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        ))
    }
}

fn identity_path() -> Result<std::path::PathBuf, String> {
    Ok(crate::bootstrap::state::state_dir()?.join(IDENTITY_FILE))
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/gateway_tls_tests.rs"]
mod tests;
