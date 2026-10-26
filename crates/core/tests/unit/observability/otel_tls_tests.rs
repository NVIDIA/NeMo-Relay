// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::event::{
    BaseEvent, DataSchema, EventCategory, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION,
    MarkEvent, ScopeCategory, ScopeEvent,
};
use crate::observability::otel::ExporterInitializationError;
use crate::observability::otel::OpenTelemetryError;
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

#[test]
fn tls_rejects_invalid_configuration_and_identity_without_disclosing_pem() {
    let certificates = Certificates::new();
    let endpoint = "https://127.0.0.1:4317";
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _entered = runtime.enter();
    certificates.tls.grpc_config(endpoint).unwrap();
    certificates.tls.http_client(endpoint, None).unwrap();
    OpenTelemetryTlsConfig::default()
        .http_client("http://localhost:4318", None)
        .unwrap();
    for tls in [
        OpenTelemetryTlsConfig {
            certificate: Some(" ".into()),
            ..Default::default()
        },
        OpenTelemetryTlsConfig {
            private_key: certificates.tls.private_key.clone(),
            ..Default::default()
        },
        OpenTelemetryTlsConfig {
            client_certificate: certificates.tls.client_certificate.clone(),
            ..Default::default()
        },
        OpenTelemetryTlsConfig {
            certificate: Some("/nonexistent/relay-ca.pem".into()),
            ..Default::default()
        },
    ] {
        assert!(tls.http_client(endpoint, None).is_err());
        assert!(tls.grpc_config(endpoint).is_err());
    }
    assert!(
        certificates
            .tls
            .http_client("http://127.0.0.1", None)
            .is_err()
    );
    assert!(certificates.tls.grpc_config("http://127.0.0.1").is_err());
    let key_path = certificates.tls.private_key.as_ref().unwrap();
    std::fs::write(key_path, KeyPair::generate().unwrap().serialize_pem()).unwrap();
    assert!(
        certificates.tls.http_client(endpoint, None).is_err(),
        "mismatched identity must fail at construction"
    );
    std::fs::write(key_path, "SECRET INVALID PRIVATE KEY").unwrap();
    let error = certificates
        .tls
        .grpc_config(endpoint)
        .unwrap_err()
        .to_string();
    assert!(error.contains("tls.client_certificate/private_key"));
    assert!(!error.contains("SECRET"));
    std::fs::write(
        certificates.tls.certificate.as_ref().unwrap(),
        "not a certificate",
    )
    .unwrap();
    assert!(
        certificates
            .tls
            .http_client(endpoint, None)
            .unwrap_err()
            .to_string()
            .contains("tls.certificate")
    );
    assert!(
        serde_json::from_value::<OpenTelemetryTlsConfig>(json!({"client_key":"typo"})).is_err()
    );
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
            let report = crate::plugin::dynamic::validate_exact(config).config;
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

#[test]
fn tls_attributes_invalid_material_to_trust_or_identity_without_disclosing_pem() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _entered = runtime.enter();
    let endpoint = "https://127.0.0.1:4318";
    let invalid_certificate = "-----BEGIN CERTIFICATE-----\nU0VDUkVU\n-----END CERTIFICATE-----\n";
    let invalid_key = "-----BEGIN PRIVATE KEY-----\nU0VDUkVU\n-----END PRIVATE KEY-----\n";
    for (field, contents, diagnostic) in [
        (
            "client_certificate",
            "SECRET INVALID CERTIFICATE".to_string(),
            "tls.client_certificate/private_key: invalid PEM client identity",
        ),
        (
            "private_key",
            "SECRET INVALID KEY".to_string(),
            "tls.client_certificate/private_key: invalid PEM client identity",
        ),
        (
            "client_certificate",
            invalid_certificate.to_string(),
            "tls.client_certificate/private_key: invalid PEM client identity",
        ),
        (
            "private_key",
            invalid_key.to_string(),
            "tls.client_certificate/private_key: invalid PEM client identity",
        ),
        (
            "private_key",
            KeyPair::generate().unwrap().serialize_pem(),
            "tls.client_certificate/private_key: invalid PEM client identity",
        ),
        (
            "certificate",
            invalid_certificate.to_string(),
            "tls.certificate: invalid PEM CA bundle",
        ),
    ] {
        let certificates = Certificates::new();
        let path = match field {
            "client_certificate" => &certificates.tls.client_certificate,
            "private_key" => &certificates.tls.private_key,
            _ => &certificates.tls.certificate,
        };
        std::fs::write(path.as_ref().unwrap(), &contents).unwrap();
        for grpc in [false, true] {
            let result = if grpc {
                certificates.tls.grpc_config(endpoint).map(|_| ())
            } else {
                certificates.tls.http_client(endpoint, None).map(|_| ())
            };
            let error = result.unwrap_err().to_string();
            assert_eq!(
                error,
                format!("failed to build the OTLP exporter: {diagnostic}"),
                "field={field}, grpc={grpc}"
            );
            assert!(!error.contains("SECRET"));
            assert!(!error.contains(&contents));
        }
    }
}

#[test]
fn tls_validates_the_last_private_key_in_a_pem_bundle() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        for grpc in [false, true] {
            let certificates = Certificates::new();
            let key_path = certificates.tls.private_key.as_ref().unwrap();
            let matching = std::fs::read_to_string(key_path).unwrap();
            let mismatched = KeyPair::generate().unwrap().serialize_pem();
            let (endpoint, count, server) = collector(&certificates, grpc, true).await;
            let transport = if grpc { OtlpTransport::Grpc } else { OtlpTransport::HttpBinary };

            std::fs::write(key_path, format!("{mismatched}\n{matching}")).unwrap();
            let export_endpoint = endpoint.clone();
            let tls = certificates.tls.clone();
            tokio::task::spawn_blocking(move || export_all(&export_endpoint, transport, tls, "code"))
                .await
                .unwrap();
            assert!(count.load(Ordering::SeqCst) >= 3);
            assert_eq!(count.signals.load(Ordering::SeqCst), 7);

            std::fs::write(key_path, format!("{matching}\n{mismatched}")).unwrap();
            let result = if grpc {
                certificates.tls.grpc_config(&endpoint).map(|_| ())
            } else {
                certificates.tls.http_client(&endpoint, None).map(|_| ())
            };
            assert_eq!(
                result.unwrap_err().to_string(),
                "failed to build the OTLP exporter: tls.client_certificate/private_key: invalid PEM client identity"
            );
            server.abort();
        }
    });
}

#[test]
fn tls_schema_aliases_editor_and_signal_inheritance() {
    let config: OpenTelemetrySectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "endpoints": [{"type":"full", "endpoint":"https://collector.example:4317", "transport":"grpc", "tls": {
            "certificate":"ca.pem", "client_certificate":"client.pem", "private_key":"key.pem"
        }}]
    })).unwrap();
    let serialized = serde_json::to_value(&config).unwrap();
    assert_eq!(serialized["traces"][0]["tls"]["private_key"], "key.pem");
    for signal in ["logs", "metrics"] {
        let derived = derive_signal_endpoint(signal, 0, &config.endpoints[0]).unwrap();
        assert_eq!(derived.tls, config.endpoints[0].tls);
        let explicit: OpenTelemetrySignalEndpointConfig =
            serde_json::from_value(json!({"endpoint":"https://collector.example:4317"})).unwrap();
        let resolved =
            resolve_signal_endpoints(signal, Some(&vec![explicit]), &config.endpoints).unwrap();
        assert!(
            resolved.endpoints[0]
                .value
                .as_active()
                .unwrap()
                .tls
                .is_none()
        );
    }
    for schema in [
        OpenTelemetryEndpointConfig::editor_schema(),
        OpenTelemetrySignalEndpointConfig::editor_schema(),
    ] {
        let entry = schema.field("tls").unwrap();
        assert!(entry.optional);
        let nested = entry.nested_schema.unwrap()();
        assert_eq!(nested.fields.len(), 3);
        assert!(nested.field("private_key").unwrap().optional);
    }
    #[cfg(feature = "schema")]
    {
        let schema =
            serde_json::to_value(schemars::schema_for!(OpenTelemetrySectionConfig)).unwrap();
        assert!(
            schema["definitions"]["OpenTelemetryTlsConfig"]["properties"]["private_key"]
                .is_object()
        );
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

fn export_all(endpoint: &str, transport: OtlpTransport, tls: OpenTelemetryTlsConfig, mode: &str) {
    let (trace, logs, metrics) = match mode {
        "automatic" | "automatic_json" => (
            OpenTelemetrySubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .unwrap(),
            OpenTelemetryLogSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .unwrap(),
            OpenTelemetryMetricSubscriber::new_from_automatic_configuration_for_plugin(
                PlaintextPolicy::default(),
            )
            .unwrap(),
        ),
        "plugin" => {
            let token =
                std::path::Path::new(tls.certificate.as_ref().unwrap()).with_file_name("token");
            std::fs::write(&token, "tls-token").unwrap();
            let trace: OpenTelemetryEndpointConfig = serde_json::from_value(json!({"type":"full", "endpoint":endpoint, "transport": if transport == OtlpTransport::Grpc {"grpc"} else {"http_binary"}, "tls":tls, "header_file":{"x-tls-token":token}})).unwrap();
            let log = derive_signal_endpoint("logs", 0, &trace).unwrap();
            let metric = derive_signal_endpoint("metrics", 0, &trace).unwrap();
            (
                OpenTelemetrySubscriber::new(
                    build_otel_config(0, trace, &PlaintextPolicy::default()).unwrap(),
                )
                .unwrap(),
                OpenTelemetryLogSubscriber::new(
                    build_log_config(
                        0,
                        log,
                        &serde_json::from_value(json!({})).unwrap(),
                        LogSeverity::Info,
                        &PlaintextPolicy::default(),
                    )
                    .unwrap(),
                )
                .unwrap(),
                OpenTelemetryMetricSubscriber::new(
                    build_metric_config(
                        0,
                        metric,
                        &serde_json::from_value(json!({})).unwrap(),
                        MetricTemporality::Cumulative,
                        &PlaintextPolicy::default(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
        }
        _ => (
            OpenTelemetrySubscriber::new(
                CoreOpenTelemetryConfig::new(OpenTelemetryType::Full, endpoint)
                    .with_transport(transport)
                    .with_tls(tls.clone()),
            )
            .unwrap(),
            OpenTelemetryLogSubscriber::new(
                CoreOpenTelemetryLogConfig::new(endpoint)
                    .with_transport(transport)
                    .with_tls(tls.clone()),
            )
            .unwrap(),
            OpenTelemetryMetricSubscriber::new(
                CoreOpenTelemetryMetricConfig::new(endpoint)
                    .with_transport(transport)
                    .with_tls(tls),
            )
            .unwrap(),
        ),
    };
    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("tls.export").build(),
        None,
        None,
    ));
    let uuid = uuid::Uuid::now_v7();
    for category in [ScopeCategory::Start, ScopeCategory::End] {
        trace.subscriber()(&Event::Scope(ScopeEvent::new(
            BaseEvent::builder().uuid(uuid).name("tls.trace").build(),
            category,
            Vec::new(),
            EventCategory::agent(),
            None,
        )));
    }
    logs.subscriber()(&mark);
    metrics.subscriber()(&Event::Mark(MarkEvent::new(BaseEvent::builder().name("tls.metric").data(json!({"measurements":[{"name":"tls.requests", "kind":"counter", "value_type":"u64", "value":1}]})).data_schema(DataSchema::builder().name(METRIC_DATA_SCHEMA_NAME).version(METRIC_DATA_SCHEMA_VERSION).build()).build(), None, None)));
    trace.force_flush().unwrap();
    logs.force_flush().unwrap();
    metrics.force_flush().unwrap();
    trace.shutdown().unwrap();
    logs.shutdown().unwrap();
    metrics.shutdown().unwrap();
}

fn configure_export_environment(
    mode: &str,
    grpc: bool,
    endpoint: &str,
    tls: &OpenTelemetryTlsConfig,
) -> Vec<(&'static str, Option<std::ffi::OsString>)> {
    let mut env = Vec::new();
    if mode.starts_with("automatic") {
        for (name, value) in [
            ("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint.to_owned()),
            (
                "OTEL_EXPORTER_OTLP_PROTOCOL",
                if grpc {
                    "grpc"
                } else if mode == "automatic_json" {
                    "http/json"
                } else {
                    "http/protobuf"
                }
                .into(),
            ),
            (
                "OTEL_EXPORTER_OTLP_CERTIFICATE",
                tls.certificate.clone().unwrap(),
            ),
        ] {
            env.push((name, std::env::var_os(name)));
            unsafe {
                std::env::set_var(name, value);
            }
        }
        for (name, value) in [
            (
                "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE",
                &tls.client_certificate,
            ),
            ("OTEL_EXPORTER_OTLP_CLIENT_KEY", &tls.private_key),
        ] {
            env.push((name, std::env::var_os(name)));
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(name, value);
                } else {
                    std::env::remove_var(name);
                }
            }
        }
    }
    env
}

#[test]
fn tls_and_mtls_export_all_signals_through_code_plugins_and_environment() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        for (grpc, require_client) in [false, true].into_iter().flat_map(|grpc| [false, true].map(|require_client| (grpc, require_client))) {
            let certificates = Certificates::new();
            let (endpoint, count, server) = collector(&certificates, grpc, require_client).await;
            let transport = if grpc { OtlpTransport::Grpc } else { OtlpTransport::HttpBinary };
            let mut tls = certificates.tls.clone();
            if !require_client { tls.client_certificate = None; tls.private_key = None; }
            for mode in ["code", "plugin", "automatic", "automatic_json"] {
                if grpc && mode == "automatic_json" { continue; }
                count.signals.store(0, Ordering::SeqCst);
                let before = count.load(Ordering::SeqCst);
                    let env = configure_export_environment(mode, grpc, &endpoint, &tls);
                let endpoint = endpoint.clone();
                let tls = tls.clone();
                let result = tokio::task::spawn_blocking(move || export_all(&endpoint, transport, tls, mode)).await;
                for (name, value) in env {
                    unsafe { if let Some(value) = value { std::env::set_var(name, value); } else { std::env::remove_var(name); } }
                }
                result.unwrap();
                assert!(count.load(Ordering::SeqCst) >= before + 3, "each signal must reach the collector: {mode}, grpc={grpc}, mtls={require_client}");
                assert_eq!(count.signals.load(Ordering::SeqCst), 7);
            }
            server.abort();
        }
    });
}

#[test]
fn tls_endpoint_identities_stay_isolated_from_each_other_and_environment() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        for grpc in [false, true] {
            let first = Certificates::new();
            let second = Certificates::new();
            let (a, count_a, server_a) = collector(&first, grpc, true).await;
            let (b, count_b, server_b) = collector(&second, grpc, true).await;
            let transport = if grpc {
                OtlpTransport::Grpc
            } else {
                OtlpTransport::HttpBinary
            };
            let variables = [
                "OTEL_EXPORTER_OTLP_CERTIFICATE",
                "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE",
                "OTEL_EXPORTER_OTLP_CLIENT_KEY",
            ];
            let previous: Vec<_> = variables
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect();
            for name in variables {
                unsafe {
                    std::env::set_var(name, "/must-not-be-read/ambient-credential.pem");
                }
            }
            let tls_a = first.tls.clone();
            let tls_b = second.tls.clone();
            let (result_a, result_b) = tokio::join!(
                tokio::task::spawn_blocking(move || export_all(&a, transport, tls_a, "code")),
                tokio::task::spawn_blocking(move || export_all(&b, transport, tls_b, "plugin")),
            );
            for (name, value) in previous {
                unsafe {
                    if let Some(value) = value {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
            result_a.unwrap();
            result_b.unwrap();
            assert!(count_a.load(Ordering::SeqCst) >= 3);
            assert!(count_b.load(Ordering::SeqCst) >= 3);
            assert_eq!(count_a.signals.load(Ordering::SeqCst), 7);
            assert_eq!(count_b.signals.load(Ordering::SeqCst), 7);
            server_a.abort();
            server_b.abort();
        }
    });
}

#[test]
fn tls_failure_stops_later_http_trace_connections_until_exporter_is_recreated() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
        let correct_tls = certificates.tls.clone();
        tokio::task::spawn_blocking(move || {
            let config = CoreOpenTelemetryConfig::new(OpenTelemetryType::Full, &endpoint)
                .with_scheduled_delay(Duration::from_secs(600));
            let failed =
                OpenTelemetrySubscriber::new(config.clone().with_tls(other.tls.clone())).unwrap();
            for batch in 0..3 {
                let uuid = uuid::Uuid::now_v7();
                for category in [ScopeCategory::Start, ScopeCategory::End] {
                    failed.subscriber()(&Event::Scope(ScopeEvent::new(
                        BaseEvent::builder()
                            .uuid(uuid)
                            .name(format!("tls.batch.{batch}"))
                            .build(),
                        category,
                        Vec::new(),
                        EventCategory::agent(),
                        None,
                    )));
                }
                let error = failed.force_flush().unwrap_err().to_string();
                assert!(error.contains("TLS certificate verification failed"));
                assert_eq!(
                    count.connections.load(Ordering::SeqCst),
                    1,
                    "later batches must not open another connection"
                );
                assert_eq!(count.load(Ordering::SeqCst), 0);
                assert_eq!(
                    failed
                        .runtime_diagnostics()
                        .get("otel.tls_failed")
                        .unwrap()
                        .count,
                    1
                );
            }
            failed.shutdown().unwrap();

            let recovered = OpenTelemetrySubscriber::new(config.with_tls(correct_tls)).unwrap();
            let uuid = uuid::Uuid::now_v7();
            for category in [ScopeCategory::Start, ScopeCategory::End] {
                recovered.subscriber()(&Event::Scope(ScopeEvent::new(
                    BaseEvent::builder()
                        .uuid(uuid)
                        .name("tls.recreated")
                        .build(),
                    category,
                    Vec::new(),
                    EventCategory::agent(),
                    None,
                )));
            }
            recovered.force_flush().unwrap();
            recovered.shutdown().unwrap();
            assert_eq!(count.load(Ordering::SeqCst), 1);
            assert_eq!(count.connections.load(Ordering::SeqCst), 2);
        })
        .await
        .unwrap();
        server.abort();
    });
}

#[test]
fn wrong_ca_trace_export_during_shutdown_reports_tls_and_releases_host() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
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
                use crate::plugin::dynamic::PluginHostActivation;
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
                let callbacks: Vec<_> = crate::api::runtime::global_context()
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

#[test]
fn tls_rejects_untrusted_servers_and_missing_or_wrong_client_identity() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        for grpc in [false, true] {
            let certificates = Certificates::new();
            let other = Certificates::new();
            let (endpoint, count, server) = collector(&certificates, grpc, true).await;
            for tls in [
                OpenTelemetryTlsConfig {
                    certificate: other.tls.certificate.clone(),
                    ..certificates.tls.clone()
                },
                OpenTelemetryTlsConfig {
                    certificate: certificates.tls.certificate.clone(),
                    ..Default::default()
                },
                OpenTelemetryTlsConfig {
                    certificate: certificates.tls.certificate.clone(),
                    ..other.tls.clone()
                },
            ] {
                let endpoint = endpoint.clone();
                tokio::task::spawn_blocking(move || {
                    let logs = OpenTelemetryLogSubscriber::new(
                        CoreOpenTelemetryLogConfig::new(endpoint)
                            .with_transport(if grpc {
                                OtlpTransport::Grpc
                            } else {
                                OtlpTransport::HttpBinary
                            })
                            .with_tls(tls)
                            .with_timeout(Duration::from_millis(500)),
                    )
                    .unwrap();
                    logs.subscriber()(&Event::Mark(MarkEvent::new(
                        BaseEvent::builder().name("rejected").build(),
                        None,
                        None,
                    )));
                    assert!(logs.force_flush().is_err());
                    let _ = logs.shutdown();
                })
                .await
                .unwrap();
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    0,
                    "rejected identity must not reach the collector"
                );
            }
            server.abort();
        }
    });
}

#[test]
fn tls_environment_uses_signal_specific_paths_before_generic_paths() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let values = [
        ("OTEL_EXPORTER_OTLP_CERTIFICATE", "generic-ca.pem"),
        (
            "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE",
            "generic-client.pem",
        ),
        ("OTEL_EXPORTER_OTLP_CLIENT_KEY", "generic-key.pem"),
        ("OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE", "logs-ca.pem"),
        (
            "OTEL_EXPORTER_OTLP_LOGS_CLIENT_CERTIFICATE",
            "logs-client.pem",
        ),
        ("OTEL_EXPORTER_OTLP_LOGS_CLIENT_KEY", "logs-key.pem"),
    ];
    let previous: Vec<_> = values
        .iter()
        .map(|(name, _)| (*name, std::env::var_os(name)))
        .collect();
    for (name, value) in values {
        unsafe {
            std::env::set_var(name, value);
        }
    }
    let tls = OpenTelemetryTlsConfig::from_environment("LOGS").unwrap();
    let traces = OpenTelemetryTlsConfig::from_environment("TRACES").unwrap();
    for (name, value) in previous {
        unsafe {
            if let Some(value) = value {
                std::env::set_var(name, value);
            } else {
                std::env::remove_var(name);
            }
        }
    }
    assert_eq!(tls.certificate.as_deref(), Some("logs-ca.pem"));
    assert_eq!(tls.client_certificate.as_deref(), Some("logs-client.pem"));
    assert_eq!(tls.private_key.as_deref(), Some("logs-key.pem"));
    assert_eq!(traces.certificate.as_deref(), Some("generic-ca.pem"));
    assert_eq!(traces.private_key.as_deref(), Some("generic-key.pem"));
}

#[test]
fn tls_automatic_signals_use_independent_trust_roots_and_client_identities() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    runtime.block_on(async {
        for grpc in [false, true] {
            let certificates = [
                Certificates::new(),
                Certificates::new(),
                Certificates::new(),
            ];
            let mut previous = Vec::new();
            let mut collectors = Vec::new();
            for (signal, certificates) in
                ["TRACES", "LOGS", "METRICS"].into_iter().zip(&certificates)
            {
                let (endpoint, count, task) = collector(certificates, grpc, true).await;
                collectors.push((count, task));
                for (suffix, value) in [
                    (
                        "ENDPOINT",
                        if grpc {
                            endpoint
                        } else {
                            format!("{endpoint}/v1/{}", signal.to_ascii_lowercase())
                        },
                    ),
                    (
                        "PROTOCOL",
                        if grpc { "grpc" } else { "http/protobuf" }.into(),
                    ),
                    ("CERTIFICATE", certificates.tls.certificate.clone().unwrap()),
                    (
                        "CLIENT_CERTIFICATE",
                        certificates.tls.client_certificate.clone().unwrap(),
                    ),
                    ("CLIENT_KEY", certificates.tls.private_key.clone().unwrap()),
                ] {
                    let name = format!("OTEL_EXPORTER_OTLP_{signal}_{suffix}");
                    previous.push((name.clone(), std::env::var_os(&name)));
                    unsafe {
                        std::env::set_var(name, value);
                    }
                }
            }
            let result = tokio::task::spawn_blocking(move || {
                export_all(
                    "",
                    OtlpTransport::HttpBinary,
                    OpenTelemetryTlsConfig::default(),
                    "automatic",
                )
            })
            .await;
            for (name, value) in previous {
                unsafe {
                    if let Some(value) = value {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
            result.unwrap();
            for (count, task) in collectors {
                assert!(
                    count.load(Ordering::SeqCst) > 0,
                    "each automatic signal must use its own identity"
                );
                task.abort();
            }
        }
    });
}

#[test]
fn tls_failures_preserve_validation_and_file_read_causes_without_paths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("private-certificate.pem");
    let tls = OpenTelemetryTlsConfig {
        certificate: Some(path.to_string_lossy().into_owned()),
        ..Default::default()
    };
    let missing = tls.validate("https://collector.example:4318").unwrap_err();
    assert!(matches!(
        missing,
        ExporterInitializationError::TlsFileRead {
            kind: std::io::ErrorKind::NotFound,
            ..
        }
    ));
    assert_eq!(
        missing.to_string(),
        "failed to build the OTLP exporter: tls.certificate: cannot read PEM file"
    );
    assert!(!missing.to_string().contains("private-certificate"));

    std::fs::write(&path, "private-invalid-PEM").unwrap();
    let invalid = tls.validate("https://collector.example:4318").unwrap_err();
    assert!(matches!(
        invalid,
        ExporterInitializationError::InvalidTlsConfiguration(_)
    ));
    assert!(!invalid.to_string().contains("private-invalid-PEM"));
    assert!(!invalid.to_string().contains("private-certificate"));
    assert!(matches!(
        tls.validate("http://collector.example:4318").unwrap_err(),
        ExporterInitializationError::InvalidTlsConfiguration(_),
    ));
}

#[test]
fn public_exporter_errors_preserve_variants_and_tls_display() {
    // An exhaustive match also guards the pre-existing public variant set.
    fn exporter_payload(error: OpenTelemetryError) -> String {
        match error {
            OpenTelemetryError::ExporterBuild(message) => message,
            OpenTelemetryError::InvalidGrpcHeader { .. }
            | OpenTelemetryError::InvalidHeader { .. }
            | OpenTelemetryError::GlobalHeaderEnvironmentUnsupported { .. }
            | OpenTelemetryError::TraceProvider(_)
            | OpenTelemetryError::LogProvider(_)
            | OpenTelemetryError::MetricProvider(_)
            | OpenTelemetryError::InvalidAttributeMappings(_)
            | OpenTelemetryError::InvalidMetadataPromotionPrefixes(_)
            | OpenTelemetryError::Core(_) => {
                panic!("expected the historical ExporterBuild variant")
            }
        }
    }
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("private-certificate.pem");
    let endpoint = "https://collector.example:4318";
    for transport in [OtlpTransport::HttpBinary, OtlpTransport::Grpc] {
        for invalid_pem in [false, true] {
            if invalid_pem {
                std::fs::write(&path, "invalid PEM").unwrap();
            }
            let tls = OpenTelemetryTlsConfig {
                certificate: Some(path.to_string_lossy().into_owned()),
                ..Default::default()
            };
            let expected = tls.validate(endpoint).unwrap_err().to_string();
            let errors = [
                OpenTelemetrySubscriber::new(
                    CoreOpenTelemetryConfig::new(OpenTelemetryType::Full, endpoint)
                        .with_transport(transport)
                        .with_tls(tls.clone()),
                )
                .err()
                .unwrap(),
                OpenTelemetryLogSubscriber::new(
                    CoreOpenTelemetryLogConfig::new(endpoint)
                        .with_transport(transport)
                        .with_tls(tls.clone()),
                )
                .err()
                .unwrap(),
                OpenTelemetryMetricSubscriber::new(
                    CoreOpenTelemetryMetricConfig::new(endpoint)
                        .with_transport(transport)
                        .with_tls(tls),
                )
                .err()
                .unwrap(),
            ];
            for error in errors {
                assert_eq!(error.to_string(), expected);
                let payload = exporter_payload(error);
                if !invalid_pem {
                    assert_eq!(payload, "tls.certificate: cannot read PEM file");
                }
            }
            if invalid_pem {
                std::fs::remove_file(&path).unwrap();
            }
        }
    }
    for endpoint in ["not an endpoint", "http://collector.example:4318"] {
        // Logs and metrics enforce plaintext policy when attaching credentials.
        for error in [
            OpenTelemetrySubscriber::new(CoreOpenTelemetryConfig::new(
                OpenTelemetryType::Full,
                endpoint,
            ))
            .err()
            .unwrap(),
            OpenTelemetryLogSubscriber::new(
                CoreOpenTelemetryLogConfig::new(endpoint).with_header("authorization", "test"),
            )
            .err()
            .unwrap(),
            OpenTelemetryMetricSubscriber::new(
                CoreOpenTelemetryMetricConfig::new(endpoint).with_header("authorization", "test"),
            )
            .err()
            .unwrap(),
        ] {
            assert!(!exporter_payload(error).is_empty());
        }
    }
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
