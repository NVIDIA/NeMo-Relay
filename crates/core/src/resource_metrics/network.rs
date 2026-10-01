// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Instant;

use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CountUnit, DataUnit, NetworkInterfaceMetrics, NetworkMetrics,
    NetworkTrafficMetrics, ResourceMeasurement, ResourceMeasurementScope, ResourceUnit,
};
use sysinfo::Networks;

#[derive(Clone, Copy)]
struct Counters {
    received: u64,
    transmitted: u64,
    received_packets: u64,
    transmitted_packets: u64,
    receive_errors: u64,
    transmit_errors: u64,
    sampled_at: Instant,
}

impl Counters {
    fn from_data(data: &sysinfo::NetworkData, sampled_at: Instant) -> Self {
        Self {
            received: data.total_received(),
            transmitted: data.total_transmitted(),
            received_packets: data.total_packets_received(),
            transmitted_packets: data.total_packets_transmitted(),
            receive_errors: data.total_errors_on_received(),
            transmit_errors: data.total_errors_on_transmitted(),
            sampled_at,
        }
    }
}

pub(crate) struct NetworkSampler {
    networks: Networks,
    previous: HashMap<String, Counters>,
}

impl Default for NetworkSampler {
    fn default() -> Self {
        Self {
            networks: Networks::new(),
            previous: HashMap::new(),
        }
    }
}

fn rate(
    current: u64,
    previous: Option<Counters>,
    get: impl Fn(Counters) -> u64,
    now: Instant,
) -> Option<ResourceMeasurement<f64, BandwidthUnit>> {
    let previous = previous?;
    let elapsed = now
        .checked_duration_since(previous.sampled_at)?
        .as_secs_f64();
    if elapsed <= 0.0 {
        return None;
    }
    let delta = current.checked_sub(get(previous))?;
    let value = delta as f64 / elapsed;
    value
        .is_finite()
        .then(|| ResourceMeasurement::new(value, BandwidthUnit::BytesPerSecond))
}

fn sum_integer(
    records: &[NetworkInterfaceMetrics],
    get: impl Fn(&NetworkTrafficMetrics) -> Option<u64>,
) -> Option<u64> {
    if records.is_empty() {
        return None;
    }
    records.iter().try_fold(0_u64, |total, record| {
        total.checked_add(get(&record.traffic)?)
    })
}

fn sum_rate(
    records: &[NetworkInterfaceMetrics],
    get: impl Fn(&NetworkTrafficMetrics) -> Option<f64>,
) -> Option<f64> {
    if records.is_empty() {
        return None;
    }
    let value = records
        .iter()
        .try_fold(0.0, |total, record| Some(total + get(&record.traffic)?))?;
    value.is_finite().then_some(value)
}

impl NetworkSampler {
    pub(crate) fn sample(&mut self, selectors: &[String]) -> NetworkMetrics {
        self.networks.refresh(true);
        let now = Instant::now();
        let mut next = HashMap::new();
        let mut interfaces = Vec::new();
        for (name, data) in &self.networks {
            if !selectors.is_empty() && !selectors.iter().any(|selector| selector == name) {
                continue;
            }
            let counters = Counters::from_data(data, now);
            let previous = self.previous.get(name).copied();
            next.insert(name.clone(), counters);
            let traffic = NetworkTrafficMetrics {
                received_data: Some(ResourceMeasurement::new(counters.received, DataUnit::Bytes)),
                transmitted_data: Some(ResourceMeasurement::new(
                    counters.transmitted,
                    DataUnit::Bytes,
                )),
                receive_throughput: rate(
                    counters.received,
                    previous,
                    |sample| sample.received,
                    now,
                ),
                transmit_throughput: rate(
                    counters.transmitted,
                    previous,
                    |sample| sample.transmitted,
                    now,
                ),
                received_packets: Some(ResourceMeasurement::new(
                    counters.received_packets,
                    CountUnit::Packets,
                )),
                transmitted_packets: Some(ResourceMeasurement::new(
                    counters.transmitted_packets,
                    CountUnit::Packets,
                )),
                receive_errors: Some(ResourceMeasurement::new(
                    counters.receive_errors,
                    CountUnit::Errors,
                )),
                transmit_errors: Some(ResourceMeasurement::new(
                    counters.transmit_errors,
                    CountUnit::Errors,
                )),
            };
            interfaces.push(NetworkInterfaceMetrics {
                name: name.clone(),
                traffic,
            });
        }
        self.previous = next;
        interfaces.sort_by(|a, b| a.name.cmp(&b.name));
        fn integer<U: ResourceUnit>(
            interfaces: &[NetworkInterfaceMetrics],
            get: fn(&NetworkTrafficMetrics) -> Option<u64>,
            unit: U,
        ) -> Option<
            ResourceMeasurement<nemo_relay_types::api::resource_metrics::ResourceMetricValue, U>,
        > {
            sum_integer(interfaces, get).map(|value| ResourceMeasurement::new(value, unit))
        }
        let floating = |get: fn(&NetworkTrafficMetrics) -> Option<f64>| {
            sum_rate(&interfaces, get)
                .map(|value| ResourceMeasurement::new(value, BandwidthUnit::BytesPerSecond))
        };
        let system = NetworkTrafficMetrics {
            received_data: integer(
                &interfaces,
                |traffic| {
                    traffic.received_data.as_ref().and_then(|m| match m.value {
                        nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(
                            value,
                        ) => Some(value),
                        _ => None,
                    })
                },
                DataUnit::Bytes,
            ),
            transmitted_data: integer(
                &interfaces,
                |traffic| {
                    traffic.transmitted_data.as_ref().and_then(|m| match m.value { nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(value) => Some(value), _ => None })
                },
                DataUnit::Bytes,
            ),
            receive_throughput: floating(|traffic| {
                traffic.receive_throughput.as_ref().map(|m| m.value)
            }),
            transmit_throughput: floating(|traffic| {
                traffic.transmit_throughput.as_ref().map(|m| m.value)
            }),
            received_packets: integer(
                &interfaces,
                |traffic| {
                    traffic.received_packets.as_ref().and_then(|m| match m.value { nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(value) => Some(value), _ => None })
                },
                CountUnit::Packets,
            ),
            transmitted_packets: integer(
                &interfaces,
                |traffic| {
                    traffic.transmitted_packets.as_ref().and_then(|m| match m.value { nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(value) => Some(value), _ => None })
                },
                CountUnit::Packets,
            ),
            receive_errors: integer(
                &interfaces,
                |traffic| {
                    traffic.receive_errors.as_ref().and_then(|m| match m.value {
                        nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(
                            value,
                        ) => Some(value),
                        _ => None,
                    })
                },
                CountUnit::Errors,
            ),
            transmit_errors: integer(
                &interfaces,
                |traffic| {
                    traffic.transmit_errors.as_ref().and_then(|m| match m.value { nemo_relay_types::api::resource_metrics::ResourceMetricValue::Integer(value) => Some(value), _ => None })
                },
                CountUnit::Errors,
            ),
        };
        NetworkMetrics {
            measurement_scope: ResourceMeasurementScope::Global,
            system,
            interfaces,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics/network_tests.rs"]
mod tests;
