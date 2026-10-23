---
name: contribute-integration
description: Add or change a third-party framework integration that attaches NeMo Relay to framework tool, LLM, or lifecycle boundaries. Do not use for core runtime or binding-only changes.
license: Apache-2.0
---


# Contribute A Framework Integration

Use this skill when contributing an integration with a framework or plugin such
as LangChain, LangGraph, Deep Agents, Google ADK, or OpenClaw through its public
APIs.

## Default Guidance

- Keep NeMo Relay optional
- Use stable, documented framework or plugin APIs
- Wrap tool and LLM paths at the correct framework boundary
- Preserve the framework's original behavior when NeMo Relay is absent

## Checklist

- [ ] Integration pattern follows `docs/integrate-into-frameworks/adding-scopes.mdx`
- [ ] Integration uses public framework or plugin APIs
- [ ] Managed tool adapters return `ToolExecutionResult` to Relay and unwrap
      `.result` only at the framework boundary; opaque annotations are
      preserved through forwarding execution intercepts
- [ ] Relevant integration tests or smoke path pass
- [ ] Docs updated if activation or usage changed

## References

- `docs/integrate-into-frameworks/about.mdx`
