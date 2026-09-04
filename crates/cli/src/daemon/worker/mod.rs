// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Daemon-attached per-user worker runtime.

mod control;
mod managed;
mod runtime;

#[cfg(test)]
pub(crate) use runtime::{TestWorkerHandle, test_router};

use std::net::{Ipv4Addr, SocketAddr};

use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

use super::common::address::{daemon_url, worker_advertised_address, worker_socket};
use super::common::control::{MAX_CONTROL_BODY_BYTES, WorkerBootstrap};
use super::common::state::load_or_create_machine_identity;
use super::common::worker_tls::WorkerTlsIdentity;
use crate::error::CliError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) daemon_address: String,
    pub(crate) bind: Ipv4Addr,
    pub(crate) port: Option<u16>,
    pub(crate) advertise_address: Option<String>,
}

pub(crate) async fn run(options: Options) -> Result<(), CliError> {
    let daemon = daemon_url(&options.daemon_address)?;
    let daemon_origin = daemon.as_str().trim_end_matches('/').to_owned();
    let bootstrap = read_bootstrap().await?;
    let bind = validate_bootstrap(&options, &bootstrap)?;
    let identity = load_or_create_machine_identity()?;
    let listener = TcpListener::bind(bind).await.map_err(|error| {
        CliError::Launch(format!(
            "failed to bind daemon worker listener {bind}: {error}"
        ))
    })?;
    let local = listener.local_addr()?;
    let advertised = worker_advertised_address(local, options.advertise_address.as_deref())?;
    let worker_tls = if local.ip().is_unspecified() {
        Some(WorkerTlsIdentity::generate(
            options
                .advertise_address
                .as_deref()
                .expect("an unspecified worker bind requires an advertised address"),
        )?)
    } else {
        None
    };
    let endpoint = format!(
        "{}://{advertised}",
        if worker_tls.is_some() {
            "https"
        } else {
            "http"
        }
    );
    let worker_tls_root = worker_tls
        .as_ref()
        .map(|identity| identity.root_certificate().to_owned());
    let tls_config = worker_tls.as_ref().map(WorkerTlsIdentity::server_config);
    let worker_id = uuid::Uuid::now_v7().to_string();

    let managed = crate::configuration::resolve_managed_worker_config()?;
    let dynamic_plugins = crate::plugins::lifecycle::active_dynamic_plugin_components(
        Some(&managed.plugin_config_path),
        &managed.resolved,
    )?;
    let registration = control::register(
        &daemon_origin,
        &identity,
        &worker_id,
        &endpoint,
        bootstrap,
        worker_tls_root.clone(),
    )
    .await?;

    runtime::serve(
        listener,
        runtime::RuntimeOptions {
            daemon_origin,
            identity,
            worker_id,
            endpoint,
            worker_tls_root,
            tls_config,
            config: managed.resolved.gateway,
            dynamic_plugins,
            registration,
        },
    )
    .await
}

async fn read_bootstrap() -> Result<WorkerBootstrap, CliError> {
    let limit = u64::try_from(MAX_CONTROL_BODY_BYTES)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut encoded = Vec::new();
    tokio::io::stdin()
        .take(limit)
        .read_to_end(&mut encoded)
        .await
        .map_err(|error| {
            CliError::Launch(format!(
                "failed to read daemon worker activation channel: {error}"
            ))
        })?;
    if encoded.len() > MAX_CONTROL_BODY_BYTES {
        return Err(CliError::Launch(format!(
            "daemon worker activation exceeded {MAX_CONTROL_BODY_BYTES} bytes"
        )));
    }
    if encoded.is_empty() {
        return Err(CliError::Unauthorized(
            "daemon worker requires a protected activation grant on standard input".into(),
        ));
    }
    serde_json::from_slice(&encoded)
        .map_err(|_| CliError::Unauthorized("daemon worker activation grant was invalid".into()))
}

fn validate_bootstrap(
    options: &Options,
    bootstrap: &WorkerBootstrap,
) -> Result<SocketAddr, CliError> {
    let requested_port = options.port.unwrap_or(0);
    if options.bind != bootstrap.bind_ip
        || requested_port != bootstrap.port
        || options.advertise_address != bootstrap.advertise_address
    {
        return Err(CliError::Unauthorized(
            "daemon worker network options do not match the activation grant".into(),
        ));
    }
    worker_socket(options.bind, options.port)
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_tests.rs"]
mod tests;
