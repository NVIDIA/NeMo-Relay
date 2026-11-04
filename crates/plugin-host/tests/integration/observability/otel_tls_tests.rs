// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::observability_test_support::otel::ExporterInitializationError;
use crate::observability_test_support::otel::OpenTelemetryError;
use nemo_relay_runtime::api::event::{
    BaseEvent, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION,
    MarkEvent, ScopeCategory, ScopeEvent,
};
use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, KeyUsagePurpose,
};
use serde_json::json;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::{TlsAcceptor, rustls};
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

struct Certificates {
    _directory: tempfile::TempDir,
    ca: String,
    server_certificate: String,
    server_key: String,
    tls: OpenTelemetryTlsConfig,
}

impl Certificates {
    fn new() -> Self {
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        let ca_key = KeyPair::generate().unwrap();
        let ca = ca_params.self_signed(&ca_key).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(vec!["127.0.0.1".into()]).unwrap();
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let server = params.signed_by(&server_key, &ca, &ca_key).unwrap();
        let client_key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
        let client = params.signed_by(&client_key, &ca, &ca_key).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let write = |name: &str, contents: String| {
            let path = directory.path().join(name);
            std::fs::write(&path, contents).unwrap();
            Some(path.to_str().unwrap().to_string())
        };
        let tls = OpenTelemetryTlsConfig {
            certificate: write("ca.pem", ca.pem()),
            client_certificate: write("client.pem", client.pem()),
            private_key: write("client-key.pem", client_key.serialize_pem()),
        };
        Self {
            _directory: directory,
            ca: ca.pem(),
            server_certificate: server.pem(),
            server_key: server_key.serialize_pem(),
            tls,
        }
    }
}

// Preflight is synchronous and must work without a Tokio runtime or collector.
#[test]
fn tls_preflight_checks_files_and_identity_for_every_signal_and_transport() {
    let certificates = Certificates::new();
    assert_tls_preflight(&certificates.tls, true);
    assert_tls_preflight(&OpenTelemetryTlsConfig::default(), true);
    assert_tls_preflight(
        &OpenTelemetryTlsConfig {
            certificate: certificates.tls.certificate.clone(),
            ..Default::default()
        },
        true,
    );
    assert_tls_preflight(
        &OpenTelemetryTlsConfig {
            certificate: None,
            ..certificates.tls.clone()
        },
        true,
    );
    for field in ["certificate", "client_certificate", "private_key"] {
        for contents in [
            None,
            Some(""),
            Some("SECRET not PEM"),
            Some("-----BEGIN CERTIFICATE-----\nSECRET invalid base64\n-----END CERTIFICATE-----"),
        ] {
            let path = certificates._directory.path().join("invalid.pem");
            if let Some(contents) = contents {
                std::fs::write(&path, contents).unwrap();
            }
            let mut tls = certificates.tls.clone();
            let path = Some(path.to_str().unwrap().to_string());
            match field {
                "certificate" => tls.certificate = path,
                "client_certificate" => tls.client_certificate = path,
                _ => tls.private_key = path,
            }
            assert_tls_preflight(&tls, false);
            if contents.is_some() {
                std::fs::remove_file(certificates._directory.path().join("invalid.pem")).unwrap();
            }
        }
    }
    let other = Certificates::new();
    let mut tls = certificates.tls.clone();
    tls.private_key = other.tls.private_key.clone();
    assert_tls_preflight(&tls, false);

    // Startup must recheck files even after a successful preflight.
    std::fs::write(
        certificates.tls.private_key.as_ref().unwrap(),
        "SECRET invalid key",
    )
    .unwrap();
    assert_tls_preflight(&certificates.tls, false);
    assert!(
        certificates
            .tls
            .http_client("https://127.0.0.1:4318", None)
            .is_err()
    );
    assert!(
        certificates
            .tls
            .grpc_config("https://127.0.0.1:4317")
            .is_err()
    );
}

fn assert_tls_preflight(tls: &OpenTelemetryTlsConfig, valid: bool) {
    for transport in ["http_binary", "grpc"] {
        for signal in ["traces", "logs", "metrics"] {
            let mut otel = json!({"enabled":true});
            let mut endpoint =
                json!({"endpoint":"https://127.0.0.1:4318", "transport":transport, "tls":tls});
            if signal == "traces" {
                endpoint["type"] = json!("full");
                otel["endpoints"] = json!([endpoint]);
            } else {
                otel[signal] = json!({"enabled":true,"endpoints":[endpoint]});
            };
            let config = serde_json::from_value(json!({
                "version":1,
                "components":[{"kind":OBSERVABILITY_PLUGIN_KIND,"config":{"version":4,"opentelemetry":otel}}]
            })).unwrap();
            let report = nemo_relay_plugin_host::dynamic::validate_exact(config).config;
            if valid {
                assert!(
                    report.diagnostics.is_empty(),
                    "{signal}/{transport}: {report:?}"
                );
            } else {
                assert_eq!(
                    report.diagnostics.len(),
                    1,
                    "{signal}/{transport}: {report:?}"
                );
                let diagnostic = &report.diagnostics[0];
                assert_eq!(diagnostic.level, DiagnosticLevel::Error);
                assert_eq!(diagnostic.field.as_deref(), Some("endpoints[0].tls"));
                assert!(diagnostic.message.contains("tls."));
                assert!(!diagnostic.message.contains("SECRET"));
            }
        }
    }
}

#[derive(Default)]
struct CollectorCounts {
    connections: AtomicUsize,
    requests: AtomicUsize,
    signals: AtomicU8,
}

impl CollectorCounts {
    fn record(&self, signal: &str) {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let bit = match signal {
            "trace" | "traces" => 1,
            "logs" => 2,
            "metrics" => 4,
            _ => panic!("unexpected signal"),
        };
        self.signals.fetch_or(bit, Ordering::SeqCst);
    }

    fn load(&self, ordering: Ordering) -> usize {
        self.requests.load(ordering)
    }
}

#[derive(Clone)]
struct Collector(Arc<CollectorCounts>);
macro_rules! grpc_collector {
    ($module:ident, $service_module:ident, $service:ident, $request:ident, $response:ident) => {
        #[tonic::async_trait]
        impl opentelemetry_proto::tonic::collector::$module::v1::$service_module::$service
            for Collector
        {
            async fn export(
                &self,
                _: tonic::Request<opentelemetry_proto::tonic::collector::$module::v1::$request>,
            ) -> std::result::Result<
                tonic::Response<opentelemetry_proto::tonic::collector::$module::v1::$response>,
                tonic::Status,
            > {
                self.0.record(stringify!($module));
                Ok(tonic::Response::new(Default::default()))
            }
        }
    };
}
// Fully qualified service modules avoid trait-name collisions between signals.
use opentelemetry_proto::tonic::collector::{
    logs::v1::logs_service_server, metrics::v1::metrics_service_server,
    trace::v1::trace_service_server,
};
grpc_collector!(
    trace,
    trace_service_server,
    TraceService,
    ExportTraceServiceRequest,
    ExportTraceServiceResponse
);
grpc_collector!(
    logs,
    logs_service_server,
    LogsService,
    ExportLogsServiceRequest,
    ExportLogsServiceResponse
);
grpc_collector!(
    metrics,
    metrics_service_server,
    MetricsService,
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse
);

async fn collector(
    certificates: &Certificates,
    grpc: bool,
    require_client: bool,
) -> (String, Arc<CollectorCounts>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("https://{}", listener.local_addr().unwrap());
    let received = Arc::new(CollectorCounts::default());
    let count = received.clone();
    let _ = rustls::crypto::ring::default_provider().install_default();
    let task = if grpc {
        let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(
            &certificates.server_certificate,
            &certificates.server_key,
        ));
        if require_client {
            tls = tls.client_ca_root(Certificate::from_pem(&certificates.ca));
        }
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .tls_config(tls)
                .unwrap()
                .add_service(trace_service_server::TraceServiceServer::new(Collector(
                    count.clone(),
                )))
                .add_service(logs_service_server::LogsServiceServer::new(Collector(
                    count.clone(),
                )))
                .add_service(metrics_service_server::MetricsServiceServer::new(
                    Collector(count),
                ))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        })
    } else {
        use rustls::pki_types::pem::PemObject;
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(
                rustls::pki_types::CertificateDer::from_pem_slice(certificates.ca.as_bytes())
                    .unwrap(),
            )
            .unwrap();
        let builder = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap();
        let builder = if require_client {
            builder.with_client_cert_verifier(
                rustls::server::WebPkiClientVerifier::builder_with_provider(
                    Arc::new(roots),
                    Arc::new(rustls::crypto::ring::default_provider()),
                )
                .build()
                .unwrap(),
            )
        } else {
            builder.with_no_client_auth()
        };
        let tls = builder
            .with_single_cert(
                vec![
                    rustls::pki_types::CertificateDer::from_pem_slice(
                        certificates.server_certificate.as_bytes(),
                    )
                    .unwrap(),
                ],
                rustls::pki_types::PrivateKeyDer::from_pem_slice(
                    certificates.server_key.as_bytes(),
                )
                .unwrap(),
            )
            .unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(tls));
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                count.connections.fetch_add(1, Ordering::SeqCst);
                let acceptor = acceptor.clone();
                let count = count.clone();
                tokio::spawn(async move {
                    let Ok(mut stream) = acceptor.accept(stream).await else {
                        return;
                    };
                    let Some(request) = read_collector_request(&mut stream).await else {
                        return;
                    };
                    let request_line = String::from_utf8_lossy(&request);
                    let signal = ["traces", "logs", "metrics"]
                        .into_iter()
                        .find(|signal| request_line.starts_with(&format!("POST /v1/{signal} ")))
                        .unwrap();
                    count.record(signal);
                    let json = String::from_utf8_lossy(&request).contains("application/json");
                    let response = if json {
                        &b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}"[..]
                    } else {
                        &b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Type: application/x-protobuf\r\nConnection: close\r\n\r\n"[..]
                    };
                    stream.write_all(response).await.unwrap();
                    let _ = stream.shutdown().await;
                });
            }
        })
    };
    (endpoint, received, task)
}

#[test]
fn wrong_ca_trace_export_during_shutdown_reports_tls_and_releases_host() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        let certificates = Certificates::new();
        let other = Certificates::new();
        let (endpoint, count, server) = collector(&certificates, false, false).await;
        for (trusted, file_headers) in [false, true].into_iter().flat_map(|trusted| [false, true].map(|file_headers| (trusted, file_headers))) {
            let before = count.load(Ordering::SeqCst);
            let endpoint = endpoint.clone();
            let ca = if trusted { &certificates } else { &other }
                .tls.certificate.clone().unwrap();
            let token = certificates._directory.path().join("token");
            std::fs::write(&token, "secret-token").unwrap();
            tokio::task::spawn_blocking(move || {
                use nemo_relay_plugin_host::dynamic::PluginHostActivation;
                let config = serde_json::from_value(json!({
                    "components": [{"kind": "observability", "config": {
                        "version": 4,
                        "opentelemetry": {"enabled": true, "traces": [{
                            "type": "full", "endpoint": if trusted { endpoint.clone() } else { format!("{endpoint}/private-path?secret-query") },
                            "transport": "http_binary", "tls": {"certificate": ca},
                            "scheduled_delay_millis": 600_000,
                            "header_file": if file_headers { json!({"x-token": token}) } else { json!({}) }
                        }]}
                    }}]
                })).unwrap();
                let mut activation = futures::executor::block_on(
                    PluginHostActivation::initialize_exact(config),
                ).unwrap();
                let callbacks: Vec<_> = nemo_relay_runtime::api::runtime::global_context()
                    .read().unwrap().event_subscribers.values().cloned().collect();
                let uuid = uuid::Uuid::now_v7();
                for category in [ScopeCategory::Start, ScopeCategory::End] {
                    let event = Event::Scope(ScopeEvent::new(
                        BaseEvent::builder().uuid(uuid).name("tls.trace").build(),
                        category, Vec::new(), EventCategory::agent(), None,
                    ));
                    for callback in &callbacks { callback(&event); }
                }
                // Do not force-flush: the first export must fail inside
                // shutdown, rather than taking the preexisting-failure path.
                let result = activation.close();
                assert!(!activation.is_active(), "one close must release ownership: {result:?}");
                let report = activation.report();
                if trusted {
                    result.unwrap();
                    assert!(report.config.runtime_diagnostics.is_empty());
                } else {
                    assert!(result.unwrap_err().to_string().contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER));
                    let diagnostic = report.config.runtime_diagnostics.iter()
                        .find(|diagnostic| diagnostic.code == "otel.tls_failed")
                        .expect("actionable TLS diagnostic");
                    assert!(diagnostic.message.contains(&endpoint));
                    assert!(diagnostic.message.contains("TLS certificate verification failed"));
                    for secret in ["private-path", "secret-query", "secret-token"] {
                        assert!(!diagnostic.message.contains(secret));
                    }
                    assert_eq!(diagnostic.count, 1);
                }
                let mut next = futures::executor::block_on(
                    PluginHostActivation::initialize_exact(Default::default()),
                ).expect("host must be available without a second close");
                next.close().unwrap();
            }).await.unwrap();
            assert_eq!(count.load(Ordering::SeqCst), before + usize::from(trusted));
        }
        server.abort();
    });
}

async fn read_collector_request(
    stream: &mut tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
) -> Option<Vec<u8>> {
    let mut request = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let size = stream.read(&mut buffer).await.unwrap();
        if size == 0 {
            return None;
        }
        request.extend_from_slice(&buffer[..size]);
        if let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + length {
                break;
            }
        }
    }
    Some(request)
}
