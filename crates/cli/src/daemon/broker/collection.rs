// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded copied pages. Worker evidence is never replayed into local recording.
use serde::{
    Deserialize,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

pub(crate) const SOURCES: usize = 64;
pub(crate) const PAGE_BYTES: usize = 128 * 1024;
pub(crate) const POLLS: usize = 4;
pub(crate) const RESPONSE_BYTES: usize = 256 * 1024;
const MAX_JSON_NODES: usize = 8192;
const MAX_JSON_DEPTH: usize = 24;

pub(crate) const fn owned_bytes() -> usize {
    // Compile-time checks keep the conservative fixed metadata allowances honest.
    assert!(std::mem::size_of::<Slot>() + 512 <= 1024);
    assert!(
        std::mem::size_of::<Option<crate::daemon::diagnostics::DiagnosticOffer>>()
            + std::mem::size_of::<Option<crate::daemon::diagnostics::DiagnosticReadGrant>>()
            + 43
            + 256
            + 2
            <= 512
    );
    assert!(std::mem::size_of::<Collector>() + 2 * std::mem::size_of::<CacheKey>() + 256 < 4096);
    // Includes ingress, worst-case escaped-string parser scratch, output slots,
    // session negotiation metadata at the existing 4096-session cap, and fixed indices.
    SOURCES * PAGE_BYTES
        + POLLS * (PAGE_BYTES + 2 * PAGE_BYTES + RESPONSE_BYTES)
        + 4096 * 512
        + SOURCES * 1024
        + 64 * 1024
}

pub(super) struct Collector {
    pub cache: Mutex<Cache>,
    pub ingress: BufferPool,
    pub responses: BufferPool,
    closed: std::sync::atomic::AtomicBool,
}
impl Collector {
    pub(super) fn new(ceiling: usize) -> Result<Self, ()> {
        if owned_bytes() > ceiling {
            return Err(());
        }
        Ok(Self {
            cache: Mutex::new(Cache::new()?),
            ingress: BufferPool::new(POLLS, PAGE_BYTES)?,
            responses: BufferPool::new(POLLS, RESPONSE_BYTES)?,
            closed: false.into(),
        })
    }
    pub(super) fn close(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
    }
    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Reusable buffers stay charged until their last response/body owner drops.
#[derive(Clone)]
pub(super) struct BufferPool {
    backing: Arc<Mutex<Vec<Vec<u8>>>>,
    capacity: usize,
}
impl BufferPool {
    pub(super) fn new(count: usize, capacity: usize) -> Result<Self, ()> {
        let mut backing = Vec::new();
        backing.try_reserve_exact(count).map_err(|_| ())?;
        for _ in 0..count {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(capacity).map_err(|_| ())?;
            backing.push(bytes);
        }
        Ok(Self {
            backing: Arc::new(Mutex::new(backing)),
            capacity,
        })
    }
    pub(super) fn reserve(&self) -> Option<Buffer> {
        let bytes = self
            .backing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()?;
        Some(Buffer {
            bytes,
            pool: self.clone(),
        })
    }
}
pub(super) struct Buffer {
    pub bytes: Vec<u8>,
    pool: BufferPool,
}
impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.pool.capacity.saturating_sub(self.bytes.len()) {
            return Err(io::ErrorKind::WriteZero.into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl AsRef<[u8]> for Buffer {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.bytes.clear();
        self.pool
            .backing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(std::mem::take(&mut self.bytes));
    }
}

/// Streaming validation avoids constructing a remote owned snapshot object graph.
/// serde_json's escaped-string scratch is bounded by the capped input, with at
/// most two times the input charged for geometric growth. Recursion and node
/// counts have independent bounds; there is no decompressor or response .json().
pub(super) fn validate_page(bytes: &[u8], source: uuid::Uuid) -> Result<(), ()> {
    if bytes.len() > PAGE_BYTES {
        return Err(());
    }
    let mut remaining = MAX_JSON_NODES;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    Walk {
        remaining: &mut remaining,
        depth: 0,
    }
    .deserialize(&mut decoder)
    .map_err(|_| ())?;
    decoder.end().map_err(|_| ())?;
    drop(decoder); // Release escaped-string scratch before the header pass.
    #[derive(Deserialize)]
    struct Source {
        instance_id: uuid::Uuid,
    }
    #[derive(Deserialize)]
    struct Header {
        schema_version: u32,
        source: Source,
        sequence: nemo_relay::diagnostics::DiagnosticNumber,
        captured_at: nemo_relay::diagnostics::ObservationTime,
    }
    let header: Header = serde_json::from_slice(bytes).map_err(|_| ())?;
    let _ = (header.sequence, header.captured_at);
    if header.schema_version != nemo_relay::diagnostics::DIAGNOSTIC_SCHEMA_VERSION
        || header.source.instance_id != source
    {
        return Err(());
    }
    Ok(())
}
struct Walk<'a> {
    remaining: &'a mut usize,
    depth: usize,
}
impl<'de> DeserializeSeed<'de> for Walk<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        if self.depth > MAX_JSON_DEPTH || *self.remaining == 0 {
            return Err(de::Error::custom("diagnostic page structural limit"));
        }
        *self.remaining -= 1;
        deserializer.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Walk<'_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded JSON")
    }
    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq
            .next_element_seed(Walk {
                remaining: self.remaining,
                depth: self.depth + 1,
            })?
            .is_some()
        {}
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map
            .next_key_seed(Walk {
                remaining: self.remaining,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            map.next_value_seed(Walk {
                remaining: self.remaining,
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}

use super::super::common::identity::TokenDigest;
use nemo_relay::diagnostics::DiagnosticNumber;
use serde::Serialize;
use tokio::time::Instant;

#[derive(Clone)]
pub(super) struct CacheKey {
    pub worker_id: String,
    pub generation: String,
    pub source: uuid::Uuid,
    control: TokenDigest,
}
impl CacheKey {
    pub(super) fn new(
        worker_id: &str,
        generation: &str,
        source: uuid::Uuid,
        control: TokenDigest,
    ) -> Option<Self> {
        if worker_id.is_empty()
            || worker_id.len() > 256
            || generation.is_empty()
            || generation.len() > 256
        {
            return None;
        }
        Some(Self {
            worker_id: worker_id.into(),
            generation: generation.to_owned(),
            source,
            control,
        })
    }
    pub(super) fn matches(&self, other: &Self) -> bool {
        self.worker_id == other.worker_id
            && self.generation == other.generation
            && self.source == other.source
            && self.control.matches(&other.control)
    }
}
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum PollOutcome {
    Pending,
    Succeeded,
    Unreachable,
    Timeout,
    Unsupported,
    Disabled,
    Unauthorized,
    Busy,
    Oversized,
    InvalidPage,
    EnvelopeTooSmall,
}
impl PollOutcome {
    pub(super) fn terminal(self) -> bool {
        matches!(self, Self::Unsupported | Self::Disabled)
    }
}
struct Slot {
    key: Option<CacheKey>,
    bytes: Vec<u8>,
    receipt: Option<Instant>,
    received_unix_ms: u64,
    ordinal: u64,
    polling: bool,
    outcome: PollOutcome,
    retry_at: Instant,
    failures: u32,
}
pub(super) struct Ticket {
    key: CacheKey,
    index: usize,
    ordinal: u64,
}
impl Ticket {
    pub(super) fn key(&self) -> &CacheKey {
        &self.key
    }
}

pub(super) struct Cache {
    slots: Vec<Slot>,
    ordinal: u64,
    evictions: u64,
}
impl Cache {
    pub(super) fn new() -> Result<Self, ()> {
        let mut slots = Vec::new();
        slots.try_reserve_exact(SOURCES).map_err(|_| ())?;
        for _ in 0..SOURCES {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(PAGE_BYTES).map_err(|_| ())?;
            slots.push(Slot {
                key: None,
                bytes,
                receipt: None,
                received_unix_ms: 0,
                ordinal: 0,
                polling: false,
                outcome: PollOutcome::Pending,
                retry_at: Instant::now(),
                failures: 0,
            });
        }
        Ok(Self {
            slots,
            ordinal: 0,
            evictions: 0,
        })
    }
    pub(super) fn begin(&mut self, key: CacheKey, now: Instant) -> Option<Ticket> {
        let index = match self
            .slots
            .iter()
            .position(|s| s.key.as_ref().is_some_and(|k| k.matches(&key)))
        {
            Some(index) => {
                let slot = &self.slots[index];
                if slot.polling || slot.retry_at > now || slot.outcome.terminal() {
                    return None;
                }
                index
            }
            None => self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.polling)
                .min_by_key(|(_, s)| (s.key.is_some(), s.ordinal))
                .map(|(i, _)| i)?,
        };
        self.ordinal = self.ordinal.checked_add(1)?;
        let slot = &mut self.slots[index];
        if !slot.key.as_ref().is_some_and(|k| k.matches(&key)) {
            if slot.key.is_some() {
                self.evictions = self.evictions.saturating_add(1);
            }
            slot.bytes.clear();
            slot.receipt = None;
            slot.received_unix_ms = 0;
            slot.outcome = PollOutcome::Pending;
            slot.failures = 0;
        }
        slot.key = Some(key.clone());
        slot.ordinal = self.ordinal;
        slot.polling = true;
        Some(Ticket {
            key,
            index,
            ordinal: self.ordinal,
        })
    }
    /// Caller holds the current-session fence while publishing. Invalidated
    /// requests are abandoned, never attached to a replacement source/session.
    pub(super) fn complete(
        &mut self,
        ticket: Ticket,
        result: Result<Buffer, PollOutcome>,
        now: Instant,
        wall_ms: u64,
    ) {
        let slot = &mut self.slots[ticket.index];
        if slot.ordinal != ticket.ordinal
            || !slot.key.as_ref().is_some_and(|k| k.matches(&ticket.key))
        {
            return;
        }
        slot.polling = false;
        match result {
            Ok(mut buffer) => {
                std::mem::swap(&mut slot.bytes, &mut buffer.bytes);
                slot.receipt = Some(now);
                slot.received_unix_ms = wall_ms;
                slot.outcome = PollOutcome::Succeeded;
                slot.failures = 0;
            }
            Err(reason) => {
                slot.outcome = reason;
                slot.failures = slot.failures.saturating_add(1);
            }
        }
        // Jitter is deterministic from the request ordinal; no queue or extra task.
        let delay = 5u64.saturating_mul(1 << slot.failures.min(4)).min(60);
        slot.retry_at =
            now + std::time::Duration::from_millis(delay * 1000 + ticket.ordinal % 1000);
    }
    pub(super) fn abandon(&mut self, ticket: Ticket) {
        let slot = &mut self.slots[ticket.index];
        if slot.ordinal == ticket.ordinal {
            slot.polling = false;
            slot.key = None;
            slot.bytes.clear();
            slot.receipt = None;
        }
    }
    pub(super) fn write_page(
        &self,
        key: &CacheKey,
        now: Instant,
        output: &mut Buffer,
    ) -> Result<(), ()> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.key.as_ref().is_some_and(|k| k.matches(key)));
        let retained = slot.filter(|s| s.receipt.is_some());
        #[derive(Serialize)]
        struct Assignment<'a> {
            worker_id: &'a str,
            generation: &'a str,
            source: uuid::Uuid,
        }
        #[derive(Serialize)]
        struct Collection {
            availability: &'static str,
            latest_poll: PollOutcome,
            age_millis: Option<DiagnosticNumber>,
            received_unix_millis: Option<DiagnosticNumber>,
            request_ordinal: Option<DiagnosticNumber>,
            cache_evictions_total: DiagnosticNumber,
            retained_sources: DiagnosticNumber,
        }
        let age = retained.and_then(|s| s.receipt).map(|tick| {
            now.saturating_duration_since(tick)
                .as_millis()
                .min(u128::from(u64::MAX)) as u64
        });
        let collection = Collection {
            availability: match age {
                None => "missing",
                Some(ms) if ms > 15_000 => "stale",
                Some(_) => "fresh",
            },
            latest_poll: slot.map_or(PollOutcome::Pending, |s| s.outcome),
            age_millis: age.map(Into::into),
            received_unix_millis: retained.map(|s| s.received_unix_ms.into()),
            request_ordinal: slot.map(|s| s.ordinal.into()),
            cache_evictions_total: self.evictions.into(),
            retained_sources: (self.slots.iter().filter(|s| s.receipt.is_some()).count() as u64)
                .into(),
        };
        output.write_all(b"{\"assignment\":").map_err(|_| ())?;
        serde_json::to_writer(
            &mut *output,
            &Assignment {
                worker_id: &key.worker_id,
                generation: &key.generation,
                source: key.source,
            },
        )
        .map_err(|_| ())?;
        output.write_all(b",\"collection\":").map_err(|_| ())?;
        serde_json::to_writer(&mut *output, &collection).map_err(|_| ())?;
        output.write_all(b",\"snapshot\":").map_err(|_| ())?;
        output
            .write_all(retained.map_or(b"null".as_slice(), |s| s.bytes.as_slice()))
            .map_err(|_| ())?;
        output.write_all(b"}").map_err(|_| ())
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/collection_cache_tests.rs"]
mod tests;
