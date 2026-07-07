<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay Go Binding

The Go binding exposes NeMo Relay runtime APIs through CGo and the raw
`nemo-relay-ffi` library. Use it when a Go application or integration needs the
same scope, middleware, lifecycle event, and observability model used by the
Rust runtime.

This binding is experimental and source-first. Rust, Python, and Node.js are the
primary supported surfaces.

## Why Use It?

- **Use NeMo Relay from Go**: Group agent, tool, and LLM work into the same
  scope and lifecycle model as the Rust runtime.
- **Bridge through CGo and FFI**: Consume the shared runtime through the
  repository-maintained `nemo-relay-ffi` layer.
- **Observe runtime behavior**: Register subscribers for scope, tool, LLM,
  and mark events emitted by the runtime.
- **Evaluate an experimental binding**: Use the source-first Go surface when
  a Go integration needs NeMo Relay semantics.

## What You Get

- **Scope, tool, and LLM helpers**: Managed lifecycle APIs backed by the
  shared Rust runtime.
- **Middleware APIs**: Guardrails and intercepts for request rewriting,
  blocking, sanitization, and execution wrapping, including mark and scope
  event sanitizers at global, scope-local, and plugin-context levels.
- **Event subscribers**: Runtime lifecycle callbacks for observability and
  diagnostics.
- **Convenience subpackages**: Short imports for scopes, tools, LLM calls,
  guardrails, intercepts, subscribers, plugins, and adaptive helpers.
- **Local source-first workflow**: Build the FFI library locally, then test or
  consume the Go module from the checkout.

## Installation

Build the FFI library from a repository checkout before using the Go binding:

```bash
git clone https://github.com/NVIDIA/NeMo-Relay.git
cd NeMo-Relay
cargo build --release -p nemo-relay-ffi
```

For a Go application that consumes a local checkout, point the module at the
checked-out binding:

```bash
go mod edit -replace github.com/NVIDIA/NeMo-Relay/go/nemo_relay=../NeMo-Relay/go/nemo_relay
go get github.com/NVIDIA/NeMo-Relay/go/nemo_relay
```

## Getting Started

Run the binding tests from the repository checkout to verify the CGo link path
and the FFI library:

```bash
cd go/nemo_relay
go test ./...
```

Then import the package from application code:

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	nemo "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/scope"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/tools"
)

func main() {
	if err := nemo.RegisterSubscriber("printer", func(event nemo.Event) {
		fmt.Printf("%s %s\n", event.Kind(), event.Name())
		fmt.Println(string(event.JSON()))
	}); err != nil {
		log.Fatal(err)
	}
	defer nemo.DeregisterSubscriber("printer")
	defer nemo.FlushSubscribers()

	handle, err := scope.Push("demo-agent", nemo.ScopeTypeAgent)
	if err != nil {
		log.Fatal(err)
	}
	defer scope.Pop(handle)

	if err := scope.Event("initialized"); err != nil {
		log.Fatal(err)
	}

	result, err := tools.Execute("search", json.RawMessage(`{"query":"hello"}`), func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(result))
}
```
