<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance

NeMo Relay keeps runtime overhead focused around the work that is active for the current scope and call.

## Runtime Model

These points summarize the runtime behaviors that matter most for performance-sensitive
paths.

- Scope stacks define active ownership and scope-local visibility.
- Middleware registries are priority ordered and lazily sorted.
- Managed tool and LLM helpers resolve visible middleware before executing the user callback.
- Subscribers receive emitted events after runtime work creates them.

## Practical Guidance

Use these practices when applying the concept in application or integration code.

- Prefer scope-local middleware for request-specific behavior so cleanup happens when the scope closes.
- Keep subscriber callbacks lightweight or move expensive export work out of the hot path.
- Use execution intercepts when you need to wrap real execution and sanitize guardrails when you only need to change emitted observability payloads.
- Use binding-native typed wrappers and codecs when provider payload conversion would otherwise be repeated at many call sites.

## Related Topics

Use these links to continue into adjacent concepts and workflows.

- [Architecture](../about/architecture.md)
- [Middleware](../about/concepts/middleware.md)
- [Subscribers](../about/concepts/subscribers.md)
- [Typed Wrappers And Codecs](../integrate-frameworks/using-codecs.md)
