// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-backed [`StorageBackend`] implementation.
//!
//! Provides [`RedisBackend`] which persists runs, plans, trie envelopes, and
//! accumulator state in Redis using atomic JSON blob operations.
//!
//! # Key layout
//!
//! All keys are prefixed with the configurable `key_prefix`:
//!
//! | Kind           | Key pattern                             | Value           |
//! |----------------|-----------------------------------------|-----------------|
//! | Run record     | `{prefix}runs:{agent_id}:{run_id}`     | JSON RunRecord  |
//! | Run index      | `{prefix}runs_index:{agent_id}`         | LIST of run UUIDs |
//! | Execution plan | `{prefix}plan:{agent_id}`               | JSON ExecutionPlan |
//! | Trie envelope  | `{prefix}trie:{agent_id}`               | JSON TrieEnvelope |
//! | Accumulators   | `{prefix}accumulators:{agent_id}`       | JSON AccumulatorState |

use std::future::Future;
use std::pin::Pin;

use redis::aio::ConnectionManager;

use crate::error::{AdaptiveError, Result};
use crate::storage::traits::{StorageBackend, StorageBackendDyn};
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::RunRecord;

/// A Redis-backed storage backend for cross-process shared state.
///
/// Uses [`ConnectionManager`] which is `Clone` (internally `Arc`-based) and
/// automatically reconnects on transient failures. Trie persistence uses an
/// atomic single JSON blob `SET` — no partial update is possible.
pub struct RedisBackend {
    conn: ConnectionManager,
    key_prefix: String,
}

impl RedisBackend {
    /// Connect to Redis and return a new `RedisBackend`.
    ///
    /// # Arguments
    ///
    /// * `url` — Redis connection URL (e.g. `redis://127.0.0.1:6379`).
    /// * `key_prefix` — String prepended to every Redis key (e.g. `"nemo_flow:"`).
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveError::Storage`] if the client cannot be created or the
    /// connection cannot be established.
    pub async fn new(url: &str, key_prefix: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| AdaptiveError::Storage(format!("redis client: {e}")))?;
        let conn = client
            .get_connection_manager()
            .await
            .map_err(|e| AdaptiveError::Storage(format!("redis connection: {e}")))?;
        Ok(Self {
            conn,
            key_prefix: key_prefix.into(),
        })
    }

    /// Build the key for a kind + agent_id pair.
    fn key(&self, kind: &str, agent_id: &str) -> String {
        format!("{}{}:{}", self.key_prefix, kind, agent_id)
    }

    /// Build the key for a specific run record.
    fn run_key(&self, agent_id: &str, run_id: &uuid::Uuid) -> String {
        format!("{}runs:{}:{}", self.key_prefix, agent_id, run_id)
    }
}

impl StorageBackend for RedisBackend {
    fn store_run(&self, record: &RunRecord) -> impl Future<Output = Result<()>> + Send {
        let mut conn = self.conn.clone();
        let run_key = self.run_key(&record.agent_id, &record.id);
        let index_key = self.key("runs_index", &record.agent_id);
        let json = serde_json::to_string(record);
        let run_id_str = record.id.to_string();

        async move {
            let json = json.map_err(AdaptiveError::Serialization)?;
            redis::pipe()
                .atomic()
                .cmd("SET")
                .arg(&run_key)
                .arg(&json)
                .cmd("RPUSH")
                .arg(&index_key)
                .arg(&run_id_str)
                .exec_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis store_run pipeline: {e}")))?;
            Ok(())
        }
    }

    fn load_plan(
        &self,
        agent_id: &str,
    ) -> impl Future<Output = Result<Option<ExecutionPlan>>> + Send {
        let mut conn = self.conn.clone();
        let key = self.key("plan", agent_id);

        async move {
            let maybe_json: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis GET plan: {e}")))?;
            match maybe_json {
                Some(json) => {
                    let plan = serde_json::from_str(&json).map_err(AdaptiveError::Serialization)?;
                    Ok(Some(plan))
                }
                None => Ok(None),
            }
        }
    }

    fn list_runs(&self, agent_id: &str) -> impl Future<Output = Result<Vec<RunRecord>>> + Send {
        let mut conn = self.conn.clone();
        let index_key = self.key("runs_index", agent_id);
        let prefix = self.key_prefix.clone();
        let agent_id_owned = agent_id.to_string();

        async move {
            let ids: Vec<String> = redis::cmd("LRANGE")
                .arg(&index_key)
                .arg(0i64)
                .arg(-1i64)
                .query_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis LRANGE runs: {e}")))?;

            let mut records = Vec::with_capacity(ids.len());
            for id_str in &ids {
                let key = format!("{}runs:{}:{}", prefix, agent_id_owned, id_str);
                let maybe_json: Option<String> = redis::cmd("GET")
                    .arg(&key)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| AdaptiveError::Storage(format!("redis GET run: {e}")))?;
                if let Some(json) = maybe_json {
                    let record: RunRecord =
                        serde_json::from_str(&json).map_err(AdaptiveError::Serialization)?;
                    records.push(record);
                }
            }
            Ok(records)
        }
    }
}

impl StorageBackendDyn for RedisBackend {
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.store_run(record))
    }

    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        Box::pin(self.load_plan(agent_id))
    }

    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(self.list_runs(agent_id))
    }

    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let mut conn = self.conn.clone();
        let key = self.key("trie", agent_id);
        let json = serde_json::to_string(envelope);

        Box::pin(async move {
            let json = json.map_err(AdaptiveError::Serialization)?;
            redis::cmd("SET")
                .arg(&key)
                .arg(&json)
                .exec_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis SET trie: {e}")))?;
            Ok(())
        })
    }

    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        let mut conn = self.conn.clone();
        let key = self.key("trie", agent_id);

        Box::pin(async move {
            let maybe_json: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis GET trie: {e}")))?;
            match maybe_json {
                Some(json) => {
                    let envelope =
                        serde_json::from_str(&json).map_err(AdaptiveError::Serialization)?;
                    Ok(Some(envelope))
                }
                None => Ok(None),
            }
        })
    }

    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let mut conn = self.conn.clone();
        let key = self.key("accumulators", agent_id);
        let json = serde_json::to_string(state);

        Box::pin(async move {
            let json = json.map_err(AdaptiveError::Serialization)?;
            redis::cmd("SET")
                .arg(&key)
                .arg(&json)
                .exec_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis SET accumulators: {e}")))?;
            Ok(())
        })
    }

    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        let mut conn = self.conn.clone();
        let key = self.key("accumulators", agent_id);

        Box::pin(async move {
            let maybe_json: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .map_err(|e| AdaptiveError::Storage(format!("redis GET accumulators: {e}")))?;
            match maybe_json {
                Some(json) => {
                    let state =
                        serde_json::from_str(&json).map_err(AdaptiveError::Serialization)?;
                    Ok(Some(state))
                }
                None => Ok(None),
            }
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/redis_tests.rs"]
mod tests;
