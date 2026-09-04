// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ephemeral, daemon-pinned TLS identities for network-reachable workers.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use base64::Engine;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair,
    KeyUsagePurpose,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};

use super::transport::{PooledClient, pooled_worker_h2c_client};
use crate::error::CliError;

const MAX_ROOT_CERTIFICATE_BYTES: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_IDLE_CONNECTIONS_PER_HOST: usize = 256;
const MAX_CACHED_TLS_WORKER_POOLS: usize = 256;

/// The daemon's process-wide connection pools for authenticated worker data-plane traffic.
///
/// Cleartext loopback workers share one HTTP/2 prior-knowledge pool. Network-reachable workers
/// share a pool only when they present the same pinned TLS root; the bounded LRU cache prevents
/// unbounded retention while active targets keep their selected pool alive through [`Arc`].
pub(crate) struct WorkerClientPool {
    cleartext_h2c: Arc<PooledClient>,
    tls: Mutex<TlsClientCache>,
}

impl WorkerClientPool {
    pub(crate) fn new() -> Result<Self, CliError> {
        Self::with_tls_capacity(MAX_CACHED_TLS_WORKER_POOLS)
    }

    fn with_tls_capacity(capacity: usize) -> Result<Self, CliError> {
        assert!(capacity > 0, "worker TLS pool capacity must be positive");
        Ok(Self {
            cleartext_h2c: Arc::new(
                pooled_worker_h2c_client().map_err(|error| CliError::Launch(error.to_string()))?,
            ),
            tls: Mutex::new(TlsClientCache::new(capacity)),
        })
    }

    /// Selects a shared pool without weakening the per-root TLS trust boundary.
    pub(crate) fn client(
        &self,
        tls_root_certificate: Option<&str>,
    ) -> Result<Arc<PooledClient>, CliError> {
        let Some(root_certificate) = tls_root_certificate else {
            return Ok(Arc::clone(&self.cleartext_h2c));
        };
        let root_der = decode_worker_tls_root(root_certificate)?;
        let root_id = WorkerTlsRootId::from_der(&root_der);
        let mut cache = lock(&self.tls);
        if let Some(client) = cache.get(root_id) {
            return Ok(client);
        }
        let client = Arc::new(pooled_worker_tls_client_from_der(root_der)?);
        cache.insert(root_id, Arc::clone(&client));
        Ok(client)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct WorkerTlsRootId([u8; 32]);

impl WorkerTlsRootId {
    fn from_der(der: &[u8]) -> Self {
        Self(Sha256::digest(der).into())
    }
}

struct TlsClientCache {
    capacity: usize,
    entries: HashMap<WorkerTlsRootId, Arc<PooledClient>>,
    recency: VecDeque<WorkerTlsRootId>,
}

impl TlsClientCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            recency: VecDeque::with_capacity(capacity),
        }
    }

    fn get(&mut self, root_id: WorkerTlsRootId) -> Option<Arc<PooledClient>> {
        let client = self.entries.get(&root_id).cloned()?;
        self.touch(root_id);
        Some(client)
    }

    fn insert(&mut self, root_id: WorkerTlsRootId, client: Arc<PooledClient>) {
        if self.entries.len() == self.capacity
            && let Some(evicted) = self.recency.pop_front()
        {
            self.entries.remove(&evicted);
        }
        self.entries.insert(root_id, client);
        self.recency.push_back(root_id);
    }

    fn touch(&mut self, root_id: WorkerTlsRootId) {
        if let Some(index) = self
            .recency
            .iter()
            .position(|candidate| *candidate == root_id)
        {
            self.recency.remove(index);
        }
        self.recency.push_back(root_id);
    }
}

/// A worker-only server identity and the public trust anchor sent to the daemon.
pub(crate) struct WorkerTlsIdentity {
    server_config: Arc<rustls::ServerConfig>,
    root_certificate: String,
}

impl WorkerTlsIdentity {
    /// Generates a private root and a server leaf whose SAN matches the advertised worker host.
    /// The root private key is discarded before this value is returned.
    pub(crate) fn generate(advertised_host: &str) -> Result<Self, CliError> {
        let advertised_host = advertised_host.trim_matches(['[', ']']);
        if advertised_host.is_empty() || advertised_host == "0.0.0.0" {
            return Err(CliError::Config(
                "worker TLS requires a concrete advertised host or IP".into(),
            ));
        }

        let mut root_params = CertificateParams::new(Vec::<String>::new())
            .map_err(|error| tls_error("create worker root parameters", error))?;
        root_params
            .distinguished_name
            .push(DnType::CommonName, "NeMo Relay ephemeral worker root");
        root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        root_params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        let root_key =
            KeyPair::generate().map_err(|error| tls_error("generate worker root key", error))?;
        let root = root_params
            .self_signed(&root_key)
            .map_err(|error| tls_error("sign worker root certificate", error))?;

        let mut leaf_params = CertificateParams::new(vec![advertised_host.to_owned()])
            .map_err(|error| tls_error("create worker leaf parameters", error))?;
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, "nemo-relay worker");
        leaf_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        leaf_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let leaf_key =
            KeyPair::generate().map_err(|error| tls_error("generate worker leaf key", error))?;
        let leaf = leaf_params
            .signed_by(&leaf_key, &root, &root_key)
            .map_err(|error| tls_error("sign worker leaf certificate", error))?;

        let root_der = root.der().to_vec();
        let certificate_chain = vec![leaf.der().clone(), root.der().clone()];
        let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(leaf_key.serialize_der()));
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certificate_chain, private_key)
            .map_err(|error| tls_error("build worker TLS server", error))?;
        server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        Ok(Self {
            server_config: Arc::new(server_config),
            root_certificate: base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(root_der),
        })
    }

    pub(crate) fn server_config(&self) -> Arc<rustls::ServerConfig> {
        Arc::clone(&self.server_config)
    }

    pub(crate) fn root_certificate(&self) -> &str {
        &self.root_certificate
    }
}

/// Builds one long-lived daemon-to-worker pool pinned to the worker's ephemeral root.
#[cfg(test)]
pub(crate) fn pooled_worker_tls_client(root_certificate: &str) -> Result<PooledClient, CliError> {
    pooled_worker_tls_client_from_der(decode_worker_tls_root(root_certificate)?)
}

fn decode_worker_tls_root(root_certificate: &str) -> Result<Vec<u8>, CliError> {
    let encoded = root_certificate.as_bytes();
    if encoded.len() > encoded_certificate_limit() {
        return Err(CliError::Unauthorized(
            "worker TLS root certificate exceeds its size limit".into(),
        ));
    }
    let der = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| CliError::Unauthorized("worker TLS root certificate is invalid".into()))?;
    if der.is_empty() || der.len() > MAX_ROOT_CERTIFICATE_BYTES {
        return Err(CliError::Unauthorized(
            "worker TLS root certificate has an invalid size".into(),
        ));
    }
    Ok(der)
}

fn pooled_worker_tls_client_from_der(der: Vec<u8>) -> Result<PooledClient, CliError> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from(der))
        .map_err(|_| CliError::Unauthorized("worker TLS root certificate is invalid".into()))?;
    let client_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let mut http = HttpConnector::new();
    http.enforce_http(false);
    http.set_nodelay(true);
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let connector = HttpsConnectorBuilder::new()
        .with_tls_config(client_config)
        .https_only()
        .enable_http1()
        .enable_http2()
        .wrap_connector(http);
    let mut builder = Client::builder(TokioExecutor::new());
    builder.timer(TokioTimer::new());
    builder.pool_idle_timeout(Duration::from_secs(120));
    builder.pool_max_idle_per_host(MAX_IDLE_CONNECTIONS_PER_HOST);
    builder.http2_keep_alive_interval(Duration::from_secs(15));
    builder.http2_keep_alive_timeout(Duration::from_secs(5));
    builder.http2_keep_alive_while_idle(true);
    Ok(builder.build(connector))
}

const fn encoded_certificate_limit() -> usize {
    MAX_ROOT_CERTIFICATE_BYTES.div_ceil(3) * 4
}

fn tls_error(context: &str, error: impl std::fmt::Display) -> CliError {
    CliError::Launch(format!("failed to {context}: {error}"))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_tls_tests.rs"]
mod tests;
