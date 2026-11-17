// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { codecs, OpenAIChatCodec } from 'nemo-relay-node';
import type { JsonValue } from 'nemo-relay-node';

interface Annotation {
  messages: { role: string; content: string }[];
}
const annotation: Annotation = { messages: [{ role: 'user', content: 'hello' }] };
const selected: codecs.MessageView | null = codecs.lastMessage(annotation);
const history: codecs.MessageView = codecs.message(annotation, -1);
const instructions: codecs.MessageView | null = codecs.instructionsView(annotation);
const system: string | null = codecs.systemPrompt(annotation);
const user: string | null = codecs.lastUserMessage(annotation);
const decoded = new OpenAIChatCodec().decodeResponse({ choices: [] });
const output: codecs.MessageView = codecs.output(decoded);
const response: string | null = codecs.responseText(decoded);
const standalone = codecs.MessageView.fromContent([{ type: 'text', text: 'hello' }]);
const item = codecs.MessageView.fromMessage({ role: 'assistant', content: 'hello' });

for (const entry of history.parts()) {
  const location: codecs.ContentLocation = entry.location;
  const index: number | null = location.messageIndex;
  // @ts-expect-error Locations are read-only.
  location.index = 1;
  switch (entry.kind) {
    case 'text':
    case 'refusal': {
      const text: string = entry.text;
      // @ts-expect-error Text entries have no tool invocation.
      entry.toolCall;
      void text;
      break;
    }
    case 'tool_call': {
      const name: string = entry.toolCall.name;
      const payload: codecs.PayloadView = entry.toolCall.arguments;
      if (payload.kind === 'string') {
        const value: string = payload.value;
        void value;
      }
      if (payload.kind === 'null') {
        const value: null = payload.value;
        void value;
      }
      // @ts-expect-error Tool names are read-only.
      entry.toolCall.name = 'changed';
      void name;
      break;
    }
    case 'tool_result': {
      const result: codecs.ToolResultView = entry.toolResult;
      const value: JsonValue = result.payload.value;
      // @ts-expect-error Tool results are excluded from explicit text entries.
      entry.text;
      void value;
      break;
    }
    case 'provider_native': {
      const provider: string = entry.provider;
      const nativeKind: string = entry.nativeKind;
      void provider;
      void nativeKind;
      break;
    }
    default: {
      const value: JsonValue = entry.value;
      void value;
    }
  }
  void index;
}
for (const text of output.textParts()) {
  const value: string = text;
  void value;
}
// @ts-expect-error Content accepts JSON-compatible values, not functions.
codecs.MessageView.fromContent(() => 'invalid');
// @ts-expect-error Message views are selected through factories.
new codecs.MessageView();
void [selected, instructions, system, user, response, standalone, item];
