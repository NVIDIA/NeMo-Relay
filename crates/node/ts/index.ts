// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json, JsonCompatible, JsonObject, JsonObjectCompatible } from '../binding.js';

export * from '../binding.js';

/** Selection, typed content snapshots, and text extraction over normalized annotations. */
export * as codecs from './codecs.js';

export class LlmRequest<
  THeaders extends JsonObjectCompatible<THeaders> = JsonObject,
  TContent extends JsonCompatible<TContent> = Json,
> {
  readonly headers: THeaders;
  readonly content: TContent;

  constructor(headers: THeaders, content: TContent) {
    if (headers === null || typeof headers !== 'object' || Array.isArray(headers)) {
      throw new TypeError('LlmRequest headers must be a JSON object');
    }
    this.headers = headers;
    this.content = content;
  }
}
