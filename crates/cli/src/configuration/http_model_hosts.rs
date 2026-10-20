// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit host trust for credential-free personal-gateway HTTP upstreams.

use crate::error::CliError;
use serde::Serialize;
use std::net::IpAddr;

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct HttpModelHosts(Vec<String>);

impl HttpModelHosts {
    pub(crate) fn parse(hosts: Vec<String>) -> Result<Self, CliError> {
        let mut normalized = Vec::with_capacity(hosts.len());
        for (index, host) in hosts.iter().enumerate() {
            let host = normalize_host(host).ok_or_else(|| CliError::Config(format!(
                "upstream.insecure_http_model_hosts[{index}] must be a bare DNS hostname or canonical IP address without a scheme, port, path, user information, or wildcard"
            )))?;
            if !normalized.contains(&host) {
                normalized.push(host);
            }
        }
        Ok(Self(normalized))
    }
    pub(crate) fn allows(&self, url: &reqwest::Url) -> bool {
        url.host_str()
            .and_then(|host| normalize_host(host.trim_matches(['[', ']'])))
            .is_some_and(|host| self.0.contains(&host))
    }
    pub(crate) fn allows_base_url(&self, value: &str, url: &reqwest::Url) -> bool {
        self.allows(url)
            && value
                .parse::<http::Uri>()
                .ok()
                .and_then(|uri| {
                    uri.authority()
                        .and_then(|a| normalize_host(a.host().trim_matches(['[', ']'])))
                })
                .is_some_and(|host| self.0.contains(&host))
    }
}

pub(crate) fn remote_http(url: &reqwest::Url) -> bool {
    url.scheme() == "http"
        && url.host_str().is_some_and(|host| {
            !host.eq_ignore_ascii_case("localhost")
                && !host
                    .trim_matches(['[', ']'])
                    .parse::<IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        })
}

fn normalize_host(host: &str) -> Option<String> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(ip.to_string());
    }
    let host = host.strip_suffix('.').unwrap_or(host);
    if host.is_empty()
        || host.len() > 253
        || !host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label.starts_with(|c: char| c.is_ascii_alphanumeric())
                && label.ends_with(|c: char| c.is_ascii_alphanumeric())
                && label
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-')
        })
    {
        return None;
    }
    // Do not reinterpret apparent DNS names as alternate IPv4 forms.
    let url = reqwest::Url::parse(&format!("http://{host}")).ok()?;
    url.host_str()
        .filter(|parsed| parsed.eq_ignore_ascii_case(host))
        .map(str::to_ascii_lowercase)
}
