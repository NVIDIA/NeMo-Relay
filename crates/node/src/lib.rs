//! NAPI-RS bindings for NVAgentRT, exposing the agent runtime framework to Node.js.
//!
//! This crate provides JavaScript/TypeScript access to scope management, tool and LLM
//! lifecycle operations, guardrails, intercepts, and event subscriptions via NAPI-RS.
//! Doc comments on `#[napi]` items are emitted into the generated `index.d.ts` TypeScript
//! definitions.

#![allow(dead_code)]

mod api;
mod callable;
mod convert;
mod stream;
mod types;
