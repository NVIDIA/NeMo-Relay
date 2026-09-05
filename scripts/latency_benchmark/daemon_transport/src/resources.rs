// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::oneshot;

use crate::config::ProcessSpec;

#[derive(Clone, Debug, Default)]
struct Aggregate {
    baseline_rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    cpu_total: f64,
    cpu_samples: u64,
}

#[derive(Debug, Serialize)]
pub struct ResourceRecord {
    pid: u32,
    baseline_rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    rss_growth_bytes: Option<u64>,
    rss_growth_per_active_stream_bytes: Option<f64>,
    average_cpu_percent: Option<f64>,
}

pub struct ResourceSampler {
    processes: Vec<ProcessSpec>,
    state: Arc<Mutex<BTreeMap<u32, Aggregate>>>,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl ResourceSampler {
    pub async fn start(mut processes: Vec<ProcessSpec>) -> Self {
        processes.push(ProcessSpec {
            name: "load-driver".to_owned(),
            pid: std::process::id(),
        });
        let state = Arc::new(Mutex::new(BTreeMap::new()));
        update(&processes, &state).await;
        let (stop, mut stopped) = oneshot::channel();
        let task_processes = processes.clone();
        let task_state = Arc::clone(&state);
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    _ = interval.tick() => update(&task_processes, &task_state).await,
                    _ = &mut stopped => break,
                }
            }
        });
        Self {
            processes,
            state,
            stop: Some(stop),
            task,
        }
    }

    pub async fn finish(mut self, active_streams: usize) -> BTreeMap<String, ResourceRecord> {
        update(&self.processes, &self.state).await;
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
        let state = self.state.lock().expect("resource state mutex poisoned");
        self.processes
            .iter()
            .map(|process| {
                let aggregate = state.get(&process.pid).cloned().unwrap_or_default();
                let growth = aggregate
                    .baseline_rss_bytes
                    .zip(aggregate.peak_rss_bytes)
                    .map(|(baseline, peak)| peak.saturating_sub(baseline));
                let cpu = (aggregate.cpu_samples > 0)
                    .then(|| aggregate.cpu_total / aggregate.cpu_samples as f64);
                (
                    process.name.clone(),
                    ResourceRecord {
                        pid: process.pid,
                        baseline_rss_bytes: aggregate.baseline_rss_bytes,
                        peak_rss_bytes: aggregate.peak_rss_bytes,
                        rss_growth_bytes: growth,
                        rss_growth_per_active_stream_bytes: growth
                            .map(|value| value as f64 / active_streams.max(1) as f64),
                        average_cpu_percent: cpu,
                    },
                )
            })
            .collect()
    }
}

async fn update(processes: &[ProcessSpec], state: &Arc<Mutex<BTreeMap<u32, Aggregate>>>) {
    let samples = sample_processes(processes).await;
    let mut state = state.lock().expect("resource state mutex poisoned");
    for (pid, (rss, cpu)) in samples {
        let aggregate = state.entry(pid).or_default();
        aggregate.baseline_rss_bytes.get_or_insert(rss);
        aggregate.peak_rss_bytes = Some(aggregate.peak_rss_bytes.unwrap_or(0).max(rss));
        aggregate.cpu_total += cpu;
        aggregate.cpu_samples += 1;
    }
}

#[cfg(unix)]
async fn sample_processes(processes: &[ProcessSpec]) -> BTreeMap<u32, (u64, f64)> {
    if processes.is_empty() {
        return BTreeMap::new();
    }
    let pids = processes
        .iter()
        .map(|process| process.pid.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let output = tokio::process::Command::new("ps")
        .args(["-o", "pid=", "-o", "rss=", "-o", "%cpu=", "-p", &pids])
        .output()
        .await;
    let Ok(output) = output else {
        return BTreeMap::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse().ok()?;
            let rss_kib: u64 = fields.next()?.parse().ok()?;
            let cpu = fields.next()?.replace(',', ".").parse().ok()?;
            Some((pid, (rss_kib.saturating_mul(1024), cpu)))
        })
        .collect()
}

#[cfg(not(unix))]
async fn sample_processes(_processes: &[ProcessSpec]) -> BTreeMap<u32, (u64, f64)> {
    BTreeMap::new()
}
