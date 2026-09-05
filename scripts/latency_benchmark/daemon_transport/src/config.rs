// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail, ensure};
use http::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::metadata::BinarySpec;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Protocol {
    Http1,
    Http2,
}

impl fmt::Display for Protocol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http1 => formatter.write_str("http1"),
            Self::Http2 => formatter.write_str("http2"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Openai,
    Anthropic,
}

impl fmt::Display for Provider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Openai => formatter.write_str("openai"),
            Self::Anthropic => formatter.write_str("anthropic"),
        }
    }
}

impl Provider {
    pub fn path(self) -> &'static str {
        match self {
            Self::Openai => "/v1/responses",
            Self::Anthropic => "/v1/messages",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Topology {
    Direct,
    DaemonPassThrough,
    WorkerOnly,
    DaemonWorker,
}

impl fmt::Display for Topology {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => formatter.write_str("direct"),
            Self::DaemonPassThrough => formatter.write_str("daemon-pass-through"),
            Self::WorkerOnly => formatter.write_str("worker-only"),
            Self::DaemonWorker => formatter.write_str("daemon-worker"),
        }
    }
}

impl FromStr for Topology {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "direct" => Ok(Self::Direct),
            "daemon-pass-through" => Ok(Self::DaemonPassThrough),
            "worker-only" => Ok(Self::WorkerOnly),
            "daemon-worker" => Ok(Self::DaemonWorker),
            _ => bail!("unknown topology {value:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TargetSpec {
    pub topology: Topology,
    pub url: String,
}

impl FromStr for TargetSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (name, url) = value.split_once('=').context("target must use NAME=URL")?;
        let topology = Topology::from_str(name)?;
        ensure!(
            topology != Topology::Direct,
            "direct is supplied with --direct-url"
        );
        validate_url(url)?;
        Ok(Self {
            topology,
            url: url.trim_end_matches('/').to_owned(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct TargetHeader {
    pub topology: Topology,
    pub name: HeaderName,
    pub value: HeaderValue,
}

impl FromStr for TargetHeader {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (target_and_header, environment_name) = value
            .split_once('=')
            .context("header binding must use TARGET:HEADER=ENV_NAME")?;
        let (target, header) = target_and_header
            .split_once(':')
            .context("header binding must use TARGET:HEADER=ENV_NAME")?;
        ensure!(
            !environment_name.is_empty(),
            "header environment name is empty"
        );
        let topology = Topology::from_str(target)?;
        let name = HeaderName::from_str(header).context("invalid HTTP header name")?;
        let raw_value = env::var(environment_name)
            .with_context(|| format!("environment variable {environment_name} is not set"))?;
        let value = HeaderValue::from_str(&raw_value).context("invalid HTTP header value")?;
        Ok(Self {
            topology,
            name,
            value,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ProcessSpec {
    pub name: String,
    pub pid: u32,
}

impl FromStr for ProcessSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (name, pid) = value.split_once('=').context("process must use NAME=PID")?;
        ensure!(!name.is_empty(), "process name is empty");
        Ok(Self {
            name: name.to_owned(),
            pid: pid.parse().context("invalid process ID")?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixConfig {
    pub protocols: Vec<Protocol>,
    pub providers: Vec<Provider>,
    pub response_bytes: Vec<usize>,
    pub events: usize,
    pub concurrency: Vec<usize>,
    pub warmup_seconds: u64,
    pub duration_seconds: u64,
    pub event_delay_micros: u64,
    pub cancel_every: usize,
    pub slow_streams: usize,
    pub slow_event_delay_millis: u64,
}

impl MatrixConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.protocols.is_empty(), "protocols must not be empty");
        ensure!(!self.providers.is_empty(), "providers must not be empty");
        ensure!(
            !self.response_bytes.is_empty(),
            "response_bytes must not be empty"
        );
        ensure!(
            !self.concurrency.is_empty(),
            "concurrency must not be empty"
        );
        ensure!(self.events >= 128, "events must be at least 128");
        ensure!(
            self.duration_seconds > 0,
            "duration_seconds must be positive"
        );
        ensure!(
            self.response_bytes.iter().all(|size| *size > 0),
            "response sizes must be positive"
        );
        ensure!(
            self.concurrency.iter().all(|value| *value > 0),
            "concurrency must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Target {
    pub topology: Topology,
    pub url: String,
    pub headers: Vec<(HeaderName, HeaderValue)>,
}

#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub matrix: MatrixConfig,
    pub targets: Vec<Target>,
    pub processes: Vec<ProcessSpec>,
    pub binaries: Vec<BinarySpec>,
    pub output: PathBuf,
}

impl LoadOptions {
    #[allow(clippy::too_many_arguments)]
    pub fn from_file(
        path: &Path,
        direct_url: String,
        targets: Vec<TargetSpec>,
        headers: Vec<TargetHeader>,
        processes: Vec<ProcessSpec>,
        binaries: Vec<BinarySpec>,
        output: PathBuf,
    ) -> Result<Self> {
        validate_url(&direct_url)?;
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let matrix: MatrixConfig = toml::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        matrix.validate()?;

        let mut seen = BTreeSet::from([Topology::Direct]);
        let mut target_map = BTreeMap::from([(
            Topology::Direct,
            Target {
                topology: Topology::Direct,
                url: direct_url.trim_end_matches('/').to_owned(),
                headers: Vec::new(),
            },
        )]);
        for target in targets {
            ensure!(
                target.topology != Topology::WorkerOnly,
                "worker-only cannot be supplied as a direct target because production workers require a broker-private credential; use --worker-binary"
            );
            ensure!(
                seen.insert(target.topology),
                "duplicate target {}",
                target.topology
            );
            target_map.insert(
                target.topology,
                Target {
                    topology: target.topology,
                    url: target.url,
                    headers: Vec::new(),
                },
            );
        }
        for header in headers {
            let target = target_map
                .get_mut(&header.topology)
                .with_context(|| format!("header refers to absent target {}", header.topology))?;
            ensure!(
                !target.headers.iter().any(|(name, _)| name == header.name),
                "duplicate header {} for {}",
                header.name,
                header.topology
            );
            target.headers.push((header.name, header.value));
        }

        Ok(Self {
            matrix,
            targets: target_map.into_values().collect(),
            processes,
            binaries,
            output,
        })
    }

    pub fn smoke(output: PathBuf, direct_url: String) -> Self {
        Self {
            matrix: MatrixConfig {
                protocols: vec![Protocol::Http1, Protocol::Http2],
                providers: vec![Provider::Openai, Provider::Anthropic],
                response_bytes: vec![16 * 1024],
                events: 128,
                concurrency: vec![1],
                warmup_seconds: 0,
                duration_seconds: 1,
                event_delay_micros: 0,
                cancel_every: 0,
                slow_streams: 0,
                slow_event_delay_millis: 0,
            },
            targets: vec![Target {
                topology: Topology::Direct,
                url: direct_url,
                headers: Vec::new(),
            }],
            processes: Vec::new(),
            binaries: Vec::new(),
            output,
        }
    }
}

fn validate_url(url: &str) -> Result<()> {
    let uri = http::Uri::from_str(url).context("invalid target URL")?;
    ensure!(
        matches!(uri.scheme_str(), Some("http" | "https")),
        "URL must use http or https"
    );
    ensure!(uri.authority().is_some(), "URL must contain an authority");
    ensure!(
        uri.path_and_query()
            .is_none_or(|value| value.as_str() == "/"),
        "target URL must not contain a path or query"
    );
    Ok(())
}
