// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TLS certificates and private keys for OTLP endpoints.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use serde::{Deserialize, Serialize};
use tonic::transport::{Certificate, ClientTlsConfig, Identity};

use super::otel::{
    ExporterInitializationError, ExporterInitializationResult as Result, OpenTelemetryError,
};

/// Optional certificate and private key file paths for one OTLP endpoint.
///
/// Use absolute paths or paths relative to the process working directory.
/// Relay reads the PEM files during preflight and again when it creates the
/// exporter. Recreate the exporter to load new certificates or keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetryTlsConfig {
    /// Path to a PEM CA bundle. Replaces the system's trusted CAs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certificate: Option<String>,
    /// Path to a PEM client certificate chain. Also set `private_key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_certificate: Option<String>,
    /// Path to a PEM private key. Also set `client_certificate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub private_key: Option<String>,
}

fn invalid(field: &str, reason: &str) -> ExporterInitializationError {
    ExporterInitializationError::InvalidTlsConfiguration(format!("tls.{field}: {reason}"))
}

fn invalid_identity() -> ExporterInitializationError {
    invalid(
        "client_certificate/private_key",
        "invalid PEM client identity",
    )
}

fn normalize_private_key(pem: &[u8]) -> Result<Vec<u8>> {
    // Preserve reqwest's last-key selection, but give both transports one key.
    let key = PrivateKeyDer::pem_slice_iter(pem)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| invalid_identity())?
        .pop()
        .ok_or_else(invalid_identity)?;
    let label = match &key {
        PrivateKeyDer::Pkcs1(_) => "RSA PRIVATE KEY",
        PrivateKeyDer::Pkcs8(_) => "PRIVATE KEY",
        PrivateKeyDer::Sec1(_) => "EC PRIVATE KEY",
        _ => return Err(invalid_identity()),
    };
    let encoded = STANDARD.encode(key.secret_der());
    let mut normalized = format!("-----BEGIN {label}-----\n").into_bytes();
    for line in encoded.as_bytes().chunks(64) {
        normalized.extend_from_slice(line);
        normalized.push(b'\n');
    }
    normalized.extend_from_slice(format!("-----END {label}-----\n").as_bytes());
    Ok(normalized)
}

impl OpenTelemetryTlsConfig {
    /// Uses TLS variables for this signal before the shared TLS variables.
    pub(crate) fn from_environment(signal: &str) -> Result<Self> {
        let value = |suffix: &str| -> Result<Option<String>> {
            for name in [
                format!("OTEL_EXPORTER_OTLP_{signal}_{suffix}"),
                format!("OTEL_EXPORTER_OTLP_{suffix}"),
            ] {
                match std::env::var(&name) {
                    Ok(value) if value.is_empty() => continue,
                    Ok(value) => return Ok(Some(value)),
                    Err(std::env::VarError::NotPresent) => continue,
                    Err(_) => return Err(invalid(suffix, "environment path is not UTF-8")),
                }
            }
            Ok(None)
        };
        Ok(Self {
            certificate: value("CERTIFICATE")?,
            client_certificate: value("CLIENT_CERTIFICATE")?,
            private_key: value("CLIENT_KEY")?,
        })
    }

    /// Checks the URL scheme and loads the same TLS material as exporter startup.
    pub(crate) fn validate(&self, endpoint: &str) -> Result<()> {
        let mut pem = self.load(endpoint)?;
        if pem.certificate.is_some() || pem.client_certificate.is_some() {
            // Build without connecting so PEM parsing and key compatibility
            // checks stay aligned with both HTTP and gRPC startup.
            let _ = pem.http_client(None)?;
        }
        Ok(())
    }

    fn validate_shape(&self, endpoint: &str) -> Result<()> {
        for (field, path) in [
            ("certificate", &self.certificate),
            ("client_certificate", &self.client_certificate),
            ("private_key", &self.private_key),
        ] {
            if path.as_ref().is_some_and(|path| path.trim().is_empty()) {
                return Err(invalid(field, "must be a nonblank PEM file path"));
            }
        }
        if self.client_certificate.is_some() != self.private_key.is_some() {
            return Err(invalid(
                "client_certificate/private_key",
                "client certificate and private key must be supplied together",
            ));
        }
        if (self.certificate.is_some() || self.client_certificate.is_some())
            && !endpoint
                .trim()
                .parse::<reqwest::Url>()
                .is_ok_and(|url| url.scheme() == "https")
        {
            return Err(invalid(
                "certificate",
                "TLS settings require an HTTPS endpoint",
            ));
        }
        Ok(())
    }

    fn read(path: &Option<String>, field: &str) -> Result<Option<Vec<u8>>> {
        path.as_ref()
            .map(|path| {
                std::fs::read(path).map_err(|error| ExporterInitializationError::TlsFileRead {
                    field: field.to_owned(),
                    kind: error.kind(),
                })
            })
            .transpose()
    }

    fn load(&self, endpoint: &str) -> Result<TlsPem> {
        self.validate_shape(endpoint)?;
        Ok(TlsPem {
            certificate: Self::read(&self.certificate, "certificate")?,
            client_certificate: Self::read(&self.client_certificate, "client_certificate")?,
            private_key: Self::read(&self.private_key, "private_key")?,
        })
    }

    pub(crate) fn http_client(
        &self,
        endpoint: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<reqwest::Client> {
        self.load(endpoint)?.http_client(timeout)
    }

    pub(crate) fn grpc_config(&self, endpoint: &str) -> Result<ClientTlsConfig> {
        let mut pem = self.load(endpoint)?;
        // Validate the same snapshot and normalize its private key for both
        // transports. Never return backend errors containing PEM material.
        let _ = pem.http_client(None)?;
        // Tonic uses the process provider. Preserve an embedding application's
        // choice, and select Relay's ring provider when none has been installed.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut config = ClientTlsConfig::new();
        if let Some(ca) = pem.certificate {
            config = config.ca_certificate(Certificate::from_pem(ca));
        } else {
            config = config.with_native_roots();
        }
        if let Some(cert) = pem.client_certificate {
            config = config.identity(Identity::from_pem(
                cert,
                pem.private_key.unwrap_or_default(),
            ));
        }
        Ok(config)
    }
}

struct TlsPem {
    certificate: Option<Vec<u8>>,
    client_certificate: Option<Vec<u8>>,
    private_key: Option<Vec<u8>>,
}

impl TlsPem {
    fn http_client(&mut self, timeout: Option<std::time::Duration>) -> Result<reqwest::Client> {
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        if let Some(pem) = &self.certificate {
            let mut roots = rustls::RootCertStore::empty();
            for certificate in CertificateDer::pem_slice_iter(pem) {
                let certificate =
                    certificate.map_err(|_| invalid("certificate", "invalid PEM CA bundle"))?;
                roots
                    .add(certificate)
                    .map_err(|_| invalid("certificate", "invalid PEM CA bundle"))?;
            }
            let certificates = reqwest::Certificate::from_pem_bundle(pem)
                .map_err(|_| invalid("certificate", "invalid PEM CA bundle"))?;
            if certificates.is_empty() {
                return Err(invalid(
                    "certificate",
                    "PEM CA bundle contains no certificates",
                ));
            }
            builder = builder.tls_certs_only(certificates);
        }
        if let Some(pem) = self.private_key.as_deref() {
            self.private_key = Some(normalize_private_key(pem)?);
        }
        if let Some(cert) = &self.client_certificate {
            // PEM decoding alone does not validate DER or certificate/key pairing.
            let certificates = CertificateDer::pem_slice_iter(cert)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| invalid_identity())?;
            let key =
                PrivateKeyDer::from_pem_slice(self.private_key.as_deref().unwrap_or_default())
                    .map_err(|_| invalid_identity())?;
            let provider = rustls::crypto::CryptoProvider::get_default()
                .cloned()
                .unwrap_or_else(|| rustls::crypto::ring::default_provider().into());
            rustls::sign::CertifiedKey::from_der(certificates, key, &provider)
                .map_err(|_| invalid_identity())?;
            let mut pem = cert.clone();
            pem.push(b'\n');
            pem.extend(self.private_key.as_deref().unwrap_or_default());
            builder = builder
                .identity(reqwest::Identity::from_pem(&pem).map_err(|_| invalid_identity())?);
        }
        // The client builder also initializes platform trust and transport state;
        // its flattened error does not establish a configuration-validation cause.
        builder.build().map_err(|_| OpenTelemetryError::ExporterBuild(
            "tls.certificate/client_certificate/private_key: invalid TLS trust or client identity".to_string(),
        ).into())
    }
}
