// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `nemo_relay` agent tool: how the DeepSeek Harness agent inspects and
 * changes NeMo Relay when the user asks it to.
 *
 * Every action calls an operation the `/relay` command also uses (the status
 * snapshot, `GuardrailPolicy`, `PhoenixControl`), so the agent and the user
 * change the same state through the same validation and get the same errors.
 *
 * Read-only actions and changes that only add protection run directly.
 * Changes that weaken protection, launch Phoenix, send traces to it, or stop
 * sending them need the user's approval: `classifyAgentAction` decides, and
 * `agentToolPreExecute` turns that decision into DeepSeek Harness's approval
 * prompt from a `tools/pre-execute` listener. Slash commands stay user-only,
 * so the agent cannot route around the prompt.
 */
import { runRelayCommand, sessionLink, type RelayCommandContext } from './commands.js';
import type { FailMode } from './config.js';
import type { JsonValue, PreToolDecision } from './dsh-types.js';
import {
  validateRules,
  type FailModeChangeResult,
  type GuardrailRule,
  type GuardrailsSnapshot,
  type RulesChangeResult,
} from './policy.js';
import { formatPhoenix, formatRulesSummary, type PhoenixSnapshot } from './status.js';

export type { FailModeChangeResult } from './policy.js';

export const AGENT_TOOL_NAME = 'nemo_relay';

/** Every action the tool accepts, in the order the schema lists them. */
export const AGENT_ACTIONS = [
  'status',
  'rules.list',
  'rules.test',
  'rules.add',
  'rules.update',
  'rules.enable',
  'rules.disable',
  'rules.remove',
  'rules.reset',
  'guardrails.failMode',
  'phoenix.status',
  'phoenix.start',
  'phoenix.connect',
  'phoenix.disconnect',
  'phoenix.stop',
  'phoenix.link',
] as const;

export type AgentAction = (typeof AGENT_ACTIONS)[number];

/** Arguments after DeepSeek Harness validated them against the tool's parameter schema. */
export type AgentToolArgs = {
  action: AgentAction;
  id?: string;
  tool?: string;
  match?: string;
  flags?: string;
  reason?: string;
  arguments?: Record<string, unknown>;
  mode?: FailMode;
};

/** The tool's value: concise text for the model plus optional JSON details. */
export type AgentToolResult = { action: AgentAction; text: string; data?: JsonValue };

/** The host operations behind the tool. */
export type AgentToolOperations = {
  /** The `/relay` command's operations for the calling session (`undefined` outside a session). */
  context(sessionId: string | undefined): RelayCommandContext;
  /** Persist and apply `guardrails.failMode`; omit it when the host cannot change the fail mode. */
  setFailMode?: ((mode: FailMode) => Promise<FailModeChangeResult>) | undefined;
};

type TextBlock = { type: 'text'; text: string };

/** The `defineTool` options this module passes (see `@deepseek-ai/dsh-tools`). */
export type AgentToolDefinitionOptions = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  output: {
    schema: Record<string, unknown>;
    render(args: AgentToolArgs, value: AgentToolResult): TextBlock[];
  };
  execute(args: AgentToolArgs, exec: unknown): Promise<AgentToolResult>;
  isConcurrencySafe(args: AgentToolArgs): boolean;
  presentCall(args: AgentToolArgs): { card: 'generic'; title: string; kind: 'read' | 'other'; rawInput: AgentToolArgs };
};

/** `defineTool` from `@deepseek-ai/dsh-tools`, as this module calls it. */
export type DefineTool = (options: AgentToolDefinitionOptions) => unknown;

const DESCRIPTION =
  'Inspect and change NeMo Relay in this app: status, the guardrail rules that block tool calls, and Arize Phoenix ' +
  'tracing. Changes that weaken protection, and every Phoenix change, ask the user first. Load the nemo-relay skill ' +
  'for rule syntax and examples.';

const RULE_FIELDS = 'rules.add/update';

/** Parameter schema in the `@deepseek-ai/dsh-tools` DSL; detail lives in the nemo-relay skill. */
const PARAMETERS = {
  action: { type: 'string', required: true, enum: [...AGENT_ACTIONS] },
  id: { type: 'string', description: 'Rule id (rules.add, update, enable, disable, remove).' },
  tool: {
    type: 'string',
    description: `${RULE_FIELDS}: tool name, comma list, /regex/flags or *. rules.test: the tool to check.`,
  },
  match: { type: 'string', description: `${RULE_FIELDS}: regex tested against the call arguments.` },
  flags: { type: 'string', description: `${RULE_FIELDS}: regex flags from imsu; default i.` },
  reason: { type: 'string', description: `${RULE_FIELDS}: what the agent is told when the rule blocks a call.` },
  arguments: {
    type: 'object',
    additionalProperties: true,
    description: 'rules.test: the arguments of the call to check.',
  },
  mode: {
    type: 'string',
    enum: ['open', 'closed'],
    description: 'guardrails.failMode: closed blocks calls the rules cannot check.',
  },
};

const OUTPUT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    action: { type: 'string', required: true },
    text: { type: 'string', required: true },
    data: { type: 'json' },
  },
};

const READ_ONLY = new Set<string>(['status', 'rules.list', 'rules.test', 'phoenix.status', 'phoenix.link']);

/**
 * Build the `nemo_relay` tool definition.
 *
 * @param defineTool - `defineTool` from the running DeepSeek Harness's `@deepseek-ai/dsh-tools`.
 * @param operations - The host operations the tool calls.
 * @returns The registry-ready definition `defineTool` returns.
 */
export function createAgentTool(defineTool: DefineTool, operations: AgentToolOperations): unknown {
  return defineTool({
    name: AGENT_TOOL_NAME,
    description: DESCRIPTION,
    parameters: PARAMETERS,
    output: {
      schema: OUTPUT_SCHEMA,
      render: (_args, value) => [{ type: 'text', text: renderResult(value) }],
    },
    execute: (args, exec) => runAgentAction(args, operations, sessionIdOf(exec)),
    isConcurrencySafe: (args) => READ_ONLY.has(args.action),
    presentCall: (args) => ({
      card: 'generic',
      title: callTitle(args),
      kind: READ_ONLY.has(args.action) ? 'read' : 'other',
      rawInput: args,
    }),
  });
}

/* -------------------------------------------------------------------------- */
/* Approval                                                                    */
/* -------------------------------------------------------------------------- */

export type AgentActionClassification = {
  /** True when DeepSeek Harness must ask the user before the call runs. */
  needsApproval: boolean;
  /** Why, in one English sentence; the logged approval reason. */
  reason: string;
  /** The approval prompt in English and Chinese; present exactly when `needsApproval`. */
  displayReason?: { en: string; zh: string };
};

/**
 * Decide whether a `nemo_relay` call needs the user's approval. Read-only
 * actions and changes that can only block more calls run directly; anything
 * that can let more calls through, launch Phoenix, send traces to it, or stop
 * sending them (which reduces what the user sees of the agent) asks.
 * Arguments this function does not recognize ask too, so a new or malformed
 * action can never skip the prompt.
 *
 * @param args - The call's arguments, as the `tools/pre-execute` listener receives them.
 * @returns The decision and its reason.
 */
export function classifyAgentAction(args: unknown): AgentActionClassification {
  const record = isRecord(args) ? args : {};
  const action = typeof record.action === 'string' ? record.action : undefined;
  const rule = ruleNames(record.id);
  switch (action) {
    case 'status':
    case 'rules.list':
    case 'rules.test':
    case 'phoenix.status':
    case 'phoenix.link':
      return direct('Reads NeMo Relay state without changing it.');
    case 'rules.add':
      return direct('Adds an enabled rule, which can only block more calls.');
    case 'rules.enable':
      return direct('Turns a rule on, which can only block more calls.');
    case 'guardrails.failMode':
      if (record.mode === 'closed') {
        return direct('Fail-closed blocks the calls the rules cannot check, which can only block more calls.');
      }
      if (record.mode === 'open') {
        return ask(
          'Fail-open lets calls through when the rules cannot be checked.',
          'The agent wants NeMo Relay to allow tool calls whenever its rules cannot be checked (fail open).',
          '智能体请求让 NeMo Relay 在无法检查规则时放行工具调用（失败放行）。',
        );
      }
      break;
    case 'rules.update': {
      const fields = ['tool', 'match', 'flags', 'reason'].filter((field) => typeof record[field] === 'string');
      const changes = fields.length === 0 ? '' : ` (${fields.join(', ')})`;
      return ask(
        `Changes rule ${rule.en}, which can stop it from blocking calls it blocks now.`,
        `The agent wants to change NeMo Relay rule ${rule.en}${changes}. The changed rule may stop blocking calls it blocks now.`,
        `智能体请求修改 NeMo Relay 规则${rule.zh}${changes}。修改后，该规则可能不再拦截它目前拦截的调用。`,
      );
    }
    case 'rules.disable':
      return ask(
        `Turns rule ${rule.en} off, so calls it blocks are allowed.`,
        `The agent wants to turn off NeMo Relay rule ${rule.en}. Calls it blocks will be allowed.`,
        `智能体请求关闭 NeMo Relay 规则${rule.zh}。它拦截的调用将被放行。`,
      );
    case 'rules.remove':
      return ask(
        `Deletes rule ${rule.en}, so calls it blocks are allowed.`,
        `The agent wants to delete NeMo Relay rule ${rule.en}. Calls it blocks will be allowed.`,
        `智能体请求删除 NeMo Relay 规则${rule.zh}。它拦截的调用将被放行。`,
      );
    case 'rules.reset':
      return ask(
        'Replaces every rule with the shipped defaults, deleting added and changed rules.',
        'The agent wants to restore the default NeMo Relay rules. Rules added or changed since then will be lost.',
        '智能体请求恢复 NeMo Relay 的默认规则。此后添加或修改的规则将会丢失。',
      );
    case 'phoenix.start':
      return ask(
        "May download and launch Arize Phoenix, then sends this app's traces to it.",
        "The agent wants to start Arize Phoenix and send this app's traces to it. This may download Phoenix and run a local server.",
        '智能体请求启动 Arize Phoenix 并将本应用的追踪数据发送给它。这可能会下载 Phoenix 并运行一个本地服务。',
      );
    case 'phoenix.connect':
      return ask(
        "Sends this app's traces to the configured Phoenix server.",
        "The agent wants to send this app's traces to the Phoenix server at the configured URL.",
        '智能体请求将本应用的追踪数据发送到已配置的 Phoenix 服务器。',
      );
    case 'phoenix.disconnect':
      return ask(
        'Stops sending traces to Phoenix, so later activity of the agent does not appear there.',
        "The agent wants to stop sending this app's traces to Phoenix. What it does afterwards will not appear there.",
        '智能体请求停止向 Phoenix 发送本应用的追踪数据。此后它的操作将不会出现在 Phoenix 中。',
      );
    case 'phoenix.stop':
      return ask(
        'Stops sending traces to Phoenix and stops a Phoenix server this app started, so later activity of the agent does not appear there.',
        "The agent wants to stop sending this app's traces to Phoenix and to stop the Phoenix server if this app started it. What it does afterwards will not appear there.",
        '智能体请求停止向 Phoenix 发送本应用的追踪数据，并停止由本应用启动的 Phoenix 服务。此后它的操作将不会出现在 Phoenix 中。',
      );
  }
  return ask(
    'The call does not name an action NeMo Relay recognizes.',
    'The agent called NeMo Relay with an action it does not recognize.',
    '智能体以 NeMo Relay 无法识别的操作调用了它。',
  );
}

/**
 * `tools/pre-execute` listener body: asks the user before a `nemo_relay` call
 * that `classifyAgentAction` gates. Other tools and every decision a later
 * listener makes (deny, cancel, or its own ask) pass through unchanged.
 *
 * @param exec - The pending call.
 * @param next - The rest of the waterfall.
 * @returns The decision for this call.
 */
export async function agentToolPreExecute(
  exec: { readonly name: string; readonly arguments: unknown },
  next: () => Promise<PreToolDecision>,
): Promise<PreToolDecision> {
  if (exec.name !== AGENT_TOOL_NAME) {
    return next();
  }
  const decision = await next();
  if (decision.kind !== 'allow') {
    return decision;
  }
  const verdict = classifyAgentAction(exec.arguments);
  if (!verdict.needsApproval) {
    return decision;
  }
  return {
    kind: 'ask',
    reason: verdict.reason,
    ...(verdict.displayReason === undefined ? {} : { displayReason: verdict.displayReason }),
  };
}

function direct(reason: string): AgentActionClassification {
  return { needsApproval: false, reason };
}

function ask(reason: string, en: string, zh: string): AgentActionClassification {
  return { needsApproval: true, reason, displayReason: { en, zh } };
}

/** The rule an action names, quoted for each prompt language. */
function ruleNames(id: unknown): { en: string; zh: string } {
  const name =
    typeof id === 'string'
      ? id
          .replace(/[\u0000-\u001f\u007f"“”]/g, '')
          .trim()
          .slice(0, 64)
      : '';
  return name.length === 0 ? { en: '(no id given)', zh: '（未提供 id）' } : { en: `"${name}"`, zh: `“${name}”` };
}

/* -------------------------------------------------------------------------- */
/* Registration                                                                */
/* -------------------------------------------------------------------------- */

/** Step-time activation condition for `ctx.tools.register(definition, { when })`. */
export type ToolActivationCondition = (context: {
  readonly agent?: unknown;
  readonly signal?: AbortSignal;
}) => boolean | Promise<boolean>;

/** The part of a DeepSeek Harness plugin context `registerAgentTool` uses. */
export type AgentToolHost = {
  readonly tools?:
    | {
        register(definition: unknown, options?: { when?: ToolActivationCondition }): unknown;
      }
    | undefined;
};

export type RegisterAgentToolDeps = {
  operations: AgentToolOperations;
  /** Activate the tool only for steps where this returns true; without it the tool is always active. */
  when?: ToolActivationCondition | undefined;
  warn?: ((message: string) => void) | undefined;
  /** Load `@deepseek-ai/dsh-tools`; tests replace it. */
  importTools?: (() => Promise<unknown>) | undefined;
};

/** Disposer of a registration; `ready` settles to whether the tool was registered and never rejects. */
export type AgentToolRegistration = (() => void) & { readonly ready: Promise<boolean> };

/** Resolved by DeepSeek Harness from its own installation; a variable keeps TypeScript from resolving it. */
const DSH_TOOLS_MODULE: string = '@deepseek-ai/dsh-tools';

/**
 * Register `nemo_relay` on `ctx.tools`. `defineTool` comes from the running
 * DeepSeek Harness when its `@deepseek-ai/dsh-tools` can be imported, and from
 * {@link defineToolLocally} otherwise; when `ctx.tools` is unavailable, nothing
 * is registered and nothing throws.
 *
 * @param ctx - A context that provides the `tools` service.
 * @param deps - The operations the tool calls, plus optional activation and logging.
 * @returns A disposer that unregisters the tool, or cancels a registration still loading.
 */
export function registerAgentTool(ctx: AgentToolHost, deps: RegisterAgentToolDeps): AgentToolRegistration {
  let disposed = false;
  let unregister: (() => void) | undefined;
  const warn = (message: string): void => {
    try {
      deps.warn?.(message);
    } catch {
      // Logging is best effort.
    }
  };
  const ready = (async (): Promise<boolean> => {
    try {
      const tools = ctx.tools;
      if (typeof tools?.register !== 'function') {
        return false;
      }
      // An installed bundle cannot resolve DeepSeek Harness's own packages: its code runs from the bundle's
      // directory, whose node_modules holds only the bundle's dependencies. The registry accepts the plain
      // definition `defineTool` would build, so build it here when the import fails.
      const defineTool =
        (await loadDefineTool(deps.importTools ?? (() => import(DSH_TOOLS_MODULE)))) ?? defineToolLocally;
      if (disposed) {
        return false;
      }
      const definition = createAgentTool(defineTool, deps.operations);
      const dispose =
        deps.when === undefined ? tools.register(definition) : tools.register(definition, { when: deps.when });
      if (typeof dispose === 'function') {
        unregister = dispose as () => void;
      }
      return true;
    } catch (error) {
      warn(`NeMo Relay could not register the nemo_relay tool: ${messageOf(error)}`);
      return false;
    }
  })();
  const dispose = (): void => {
    disposed = true;
    const current = unregister;
    unregister = undefined;
    try {
      current?.();
    } catch (error) {
      warn(`NeMo Relay could not unregister the nemo_relay tool: ${messageOf(error)}`);
    }
  };
  return Object.assign(dispose, { ready });
}

async function loadDefineTool(load: () => Promise<unknown>): Promise<DefineTool | undefined> {
  try {
    const module = await load();
    const defineTool = isRecord(module) ? module.defineTool : undefined;
    return typeof defineTool === 'function' ? (defineTool as DefineTool) : undefined;
  } catch {
    return undefined;
  }
}

/** A JSON Schema node of the subset this tool's schemas use. */
type JsonSchemaNode = { [key: string]: unknown };

/**
 * Build the registry definition the way `defineTool` from
 * `@deepseek-ai/dsh-tools` does, for an installed bundle that cannot import it:
 * the property-map parameters and output specs become JSON Schema (per-property
 * `required: true` flags become the parent's `required` list; the `json` type
 * becomes an annotation-only schema), and `execute` checks the arguments
 * against the parameter schema before it runs. The tool registry checks the
 * output against `output.schema` itself.
 */
export const defineToolLocally: DefineTool = (options) => {
  const parameters = compileSpec({ type: 'object', properties: options.parameters });
  const outputSchema = compileSpec(options.output.schema);
  const check = (args: unknown): string[] => checkValue(parameters, args, 'arguments');
  return {
    name: options.name,
    description: options.description,
    parameters,
    output: {
      schema: outputSchema,
      render: (args: AgentToolArgs, value: AgentToolResult) => options.output.render(args, value),
    },
    async execute(args: AgentToolArgs, exec: unknown) {
      const problems = check(args);
      if (problems.length > 0) {
        throw new Error(`Invalid nemo_relay arguments: ${problems.join('; ')}`);
      }
      return options.execute(args, exec);
    },
    isConcurrencySafe: (args: AgentToolArgs) => options.isConcurrencySafe(args),
    presentCall: (args: AgentToolArgs) => (check(args).length > 0 ? undefined : options.presentCall(args)),
  };
};

/** Compile one property-map spec node to JSON Schema. */
function compileSpec(spec: JsonSchemaNode): JsonSchemaNode {
  if (spec.type === 'json') {
    const { type: _json, required: _required, ...annotations } = spec;
    return annotations;
  }
  const { required: _required, properties, items, ...rest } = spec;
  const node: JsonSchemaNode = { ...rest };
  if (isRecord(properties)) {
    const compiled: Record<string, JsonSchemaNode> = {};
    const required: string[] = [];
    for (const [name, child] of Object.entries(properties)) {
      if (!isRecord(child)) {
        continue;
      }
      compiled[name] = compileSpec(child);
      if (child.required === true) {
        required.push(name);
      }
    }
    node.properties = compiled;
    if (required.length > 0) {
      node.required = required;
    }
  }
  if (isRecord(items)) {
    node.items = compileSpec(items);
  }
  return node;
}

/** Problems of `value` against the JSON Schema subset `compileSpec` produces (types, enums, required, extras). */
function checkValue(schema: JsonSchemaNode, value: unknown, path: string): string[] {
  const problems: string[] = [];
  const type = schema.type;
  const kind = Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value;
  if (typeof type === 'string') {
    const matches =
      type === 'integer'
        ? typeof value === 'number' && Number.isInteger(value)
        : type === 'number'
          ? typeof value === 'number' && Number.isFinite(value)
          : type === 'object'
            ? isRecord(value)
            : kind === type;
    if (!matches) {
      return [`${path} must be ${type === 'object' ? 'an object' : `a ${type}`}`];
    }
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    problems.push(`${path} must be one of ${schema.enum.map((entry) => JSON.stringify(entry)).join(', ')}`);
  }
  if (isRecord(value) && isRecord(schema.properties)) {
    const properties = schema.properties as Record<string, JsonSchemaNode>;
    for (const name of Array.isArray(schema.required) ? (schema.required as string[]) : []) {
      if (value[name] === undefined) {
        problems.push(`${path}.${name} is required`);
      }
    }
    for (const [name, child] of Object.entries(value)) {
      const property = properties[name];
      if (property === undefined) {
        if (schema.additionalProperties === false) {
          problems.push(`${path}.${name} is not allowed`);
        }
      } else if (child !== undefined) {
        problems.push(...checkValue(property, child, `${path}.${name}`));
      }
    }
  }
  return problems;
}

/* -------------------------------------------------------------------------- */
/* Actions                                                                     */
/* -------------------------------------------------------------------------- */

async function runAgentAction(
  args: AgentToolArgs,
  operations: AgentToolOperations,
  sessionId: string | undefined,
): Promise<AgentToolResult> {
  const context = operations.context(sessionId);
  const { action } = args;
  switch (action) {
    case 'status':
      return { action, text: await commandText('status', context) };
    case 'rules.list': {
      context.publish();
      const guardrails = context.policy.snapshot();
      return { action, text: rulesSummary(guardrails), data: rulesData(guardrails) };
    }
    case 'rules.test':
      return testCall(args, context);
    case 'rules.add':
      return addRule(args, context);
    case 'rules.update':
      return updateRule(args, context);
    case 'rules.enable':
    case 'rules.disable':
      return toggleRule(args, context, action === 'rules.enable');
    case 'rules.remove':
      return removeRule(args, context);
    case 'rules.reset': {
      const current = context.policy.snapshot();
      const result = await context.policy.reset(current.revision);
      return saved(action, context, result, 'Restored the default rules.', (guardrails) => ({
        rules: guardrails.rules,
      }));
    }
    case 'guardrails.failMode':
      return changeFailMode(args, operations, context);
    case 'phoenix.status': {
      const text = await commandText('phoenix', context);
      return { action, text, data: phoenixData(context.snapshot().phoenix, context.sessionId) };
    }
    case 'phoenix.start':
    case 'phoenix.connect':
    case 'phoenix.disconnect':
    case 'phoenix.stop':
      return phoenixAction(action, context);
    case 'phoenix.link':
      return phoenixLink(context);
    default:
      throw new Error(`Unknown action "${String(action)}". Use one of: ${AGENT_ACTIONS.join(', ')}.`);
  }
}

/** The `/relay` command's reply text, which publishes the status first. */
async function commandText(input: string, context: RelayCommandContext): Promise<string> {
  const reply = await runRelayCommand(input, context);
  if (reply.kind === 'error') {
    throw new Error(reply.text);
  }
  return reply.text;
}

function testCall(args: AgentToolArgs, context: RelayCommandContext): AgentToolResult {
  const tool = args.tool?.trim() ?? '';
  if (tool.length === 0) {
    throw new Error('rules.test needs "tool": the name of the tool whose call to check.');
  }
  const verdict = context.policy.dryRun(tool, args.arguments ?? {});
  context.publish();
  if (verdict.verdict === 'block') {
    return {
      action: 'rules.test',
      text: `Blocked by rule "${verdict.ruleId}". The agent would be told: ${verdict.reason}`,
      data: { verdict: 'block', ruleId: verdict.ruleId, reason: verdict.reason },
    };
  }
  const timedOut = verdict.timedOut ?? [];
  const text = !context.policy.enabled
    ? 'Allowed: the guardrails are off (guardrails.enabled is false).'
    : timedOut.length > 0
      ? `Allowed, but these rules could not be checked (out of time, or arguments too large): ${timedOut.join(', ')}.`
      : 'Allowed: no enabled rule matches this call.';
  return { action: 'rules.test', text, data: { verdict: 'allow', ...(timedOut.length > 0 ? { timedOut } : {}) } };
}

async function addRule(args: AgentToolArgs, context: RelayCommandContext): Promise<AgentToolResult> {
  const id = requireId(args);
  const candidate = {
    id,
    tool: args.tool ?? '',
    match: args.match ?? '',
    ...(args.flags === undefined ? {} : { flags: args.flags }),
    ...(args.reason === undefined ? {} : { reason: args.reason }),
    enabled: true,
  };
  const current = context.policy.snapshot();
  // The stored id is the normalized one, such as "No force push" -> "No-force-push"; an id without
  // a usable character is numbered by the rule's position, so it is normalized in place.
  const ruleId = validateRules([...current.rules, candidate]).rules.at(-1)?.id ?? id;
  const existing = findRule(current.rules, ruleId);
  if (existing !== undefined) {
    throw new Error(
      `A rule named "${existing.id}" already exists; use rules.update to change it, or choose another id.`,
    );
  }
  // Appended, so existing rules keep their place and their share of the per-call time budget.
  const result = await context.policy.save(current.revision, [...current.rules, candidate]);
  const stored = result.ok ? (result.guardrails.rules.at(-1)?.id ?? ruleId) : ruleId;
  return saved('rules.add', context, result, `Added rule "${stored}".`, (guardrails) => ruleData(guardrails, stored));
}

async function updateRule(args: AgentToolArgs, context: RelayCommandContext): Promise<AgentToolResult> {
  const current = context.policy.snapshot();
  const rule = existingRule(current, requireId(args));
  const changes: Partial<GuardrailRule> = {};
  for (const field of ['tool', 'match', 'flags', 'reason'] as const) {
    const value = args[field];
    if (value !== undefined) {
      changes[field] = value;
    }
  }
  if (Object.keys(changes).length === 0) {
    throw new Error('rules.update needs at least one of "tool", "match", "flags" or "reason" to change.');
  }
  const next = current.rules.map((entry) => (entry === rule ? { ...entry, ...changes } : entry));
  const result = await context.policy.save(current.revision, next);
  return saved('rules.update', context, result, `Updated rule "${rule.id}".`, (guardrails) =>
    ruleData(guardrails, rule.id),
  );
}

async function toggleRule(
  args: AgentToolArgs,
  context: RelayCommandContext,
  enabled: boolean,
): Promise<AgentToolResult> {
  const action = enabled ? 'rules.enable' : 'rules.disable';
  const current = context.policy.snapshot();
  const rule = existingRule(current, requireId(args));
  const state = enabled ? 'on' : 'off';
  if (rule.enabled === enabled) {
    context.publish();
    return {
      action,
      text: `Rule "${rule.id}" is already ${state}. ${formatRulesSummary(current)}`,
      data: toJson({ revision: current.revision, rule }),
    };
  }
  const next = current.rules.map((entry) => (entry === rule ? { ...entry, enabled } : entry));
  const result = await context.policy.save(current.revision, next);
  return saved(action, context, result, `Turned rule "${rule.id}" ${state}.`, (guardrails) =>
    ruleData(guardrails, rule.id),
  );
}

async function removeRule(args: AgentToolArgs, context: RelayCommandContext): Promise<AgentToolResult> {
  const current = context.policy.snapshot();
  const rule = existingRule(current, requireId(args));
  const result = await context.policy.save(
    current.revision,
    current.rules.filter((entry) => entry !== rule),
  );
  return saved('rules.remove', context, result, `Deleted rule "${rule.id}".`, () => ({ removed: { ...rule } }));
}

/** Turn a rules change into the tool result, or into an error the model can act on. */
function saved(
  action: AgentAction,
  context: RelayCommandContext,
  result: RulesChangeResult,
  done: string,
  details: (guardrails: GuardrailsSnapshot) => Record<string, unknown>,
): AgentToolResult {
  context.publish();
  if (!result.ok) {
    throw new Error(
      result.code === 'stale'
        ? 'The rules changed while this edit was being saved, so nothing was saved. Run the action again.'
        : result.error,
    );
  }
  return {
    action,
    text: `${done} ${formatRulesSummary(result.guardrails)}`,
    data: toJson({ revision: result.revision, ...details(result.guardrails) }),
  };
}

async function changeFailMode(
  args: AgentToolArgs,
  operations: AgentToolOperations,
  context: RelayCommandContext,
): Promise<AgentToolResult> {
  const { mode } = args;
  if (mode !== 'open' && mode !== 'closed') {
    throw new Error('guardrails.failMode needs "mode": "open" or "closed".');
  }
  if (context.policy.snapshot().failMode === mode) {
    context.publish();
    return { action: 'guardrails.failMode', text: `The guardrails already fail ${mode}.`, data: { failMode: mode } };
  }
  if (operations.setFailMode === undefined) {
    throw new Error(
      'This NeMo Relay version cannot change the fail mode while it runs. Ask the user to set guardrails.failMode in the NeMo Relay plugin configuration.',
    );
  }
  const result = await operations.setFailMode(mode);
  context.publish();
  if (!result.ok) {
    throw new Error(result.error);
  }
  const meaning = result.failMode === 'closed' ? 'blocks' : 'allows';
  return {
    action: 'guardrails.failMode',
    text: `The guardrails now fail ${result.failMode}: NeMo Relay ${meaning} a call when its rules cannot be checked.`,
    data: { failMode: result.failMode },
  };
}

async function phoenixAction(
  action: 'phoenix.start' | 'phoenix.connect' | 'phoenix.disconnect' | 'phoenix.stop',
  context: RelayCommandContext,
): Promise<AgentToolResult> {
  const control = context.phoenix;
  if (control === undefined) {
    context.publish();
    throw new Error(
      'Phoenix is turned off in the NeMo Relay plugin configuration (phoenix.enabled: false); ask the user to turn it on.',
    );
  }
  const verb = action.slice('phoenix.'.length) as 'start' | 'connect' | 'disconnect' | 'stop';
  const result = await control[verb]();
  context.publish();
  if (!result.ok) {
    throw new Error(result.error ?? formatPhoenix(result.phoenix));
  }
  const link = sessionLink(result.phoenix, context.sessionId);
  return {
    action,
    text: [formatPhoenix(result.phoenix), ...(link === undefined ? [] : [`This session: ${link}`])].join('\n'),
    data: phoenixData(result.phoenix, context.sessionId),
  };
}

function phoenixLink(context: RelayCommandContext): AgentToolResult {
  context.publish();
  if (context.sessionId === undefined) {
    throw new Error('This call does not belong to a session, so there is no session link.');
  }
  const phoenix = context.snapshot().phoenix;
  const link = sessionLink(phoenix, context.sessionId);
  if (link === undefined) {
    throw new Error(`Phoenix is not receiving this session's traces. ${formatPhoenix(phoenix)}`);
  }
  return { action: 'phoenix.link', text: `This session in Phoenix: ${link}`, data: { url: link } };
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

function requireId(args: AgentToolArgs): string {
  const id = args.id?.trim() ?? '';
  if (id.length === 0) {
    throw new Error(`${args.action} needs "id", the rule's id from rules.list.`);
  }
  return id;
}

/** A rule by id: exact match first, then case-insensitive, as ids are unique regardless of case. */
function findRule(rules: readonly GuardrailRule[], id: string): GuardrailRule | undefined {
  return rules.find((rule) => rule.id === id) ?? rules.find((rule) => rule.id.toLowerCase() === id.toLowerCase());
}

function existingRule(guardrails: GuardrailsSnapshot, id: string): GuardrailRule {
  const rule = findRule(guardrails.rules, id);
  if (rule === undefined) {
    const ids = guardrails.rules.map((entry) => entry.id);
    throw new Error(`No rule is named "${id}". Rule ids: ${ids.length === 0 ? '(none)' : ids.join(', ')}.`);
  }
  return rule;
}

function ruleData(guardrails: GuardrailsSnapshot, id: string): Record<string, unknown> {
  const rule = findRule(guardrails.rules, id);
  return rule === undefined ? {} : { rule };
}

function rulesSummary(guardrails: GuardrailsSnapshot): string {
  return guardrails.loadError === undefined
    ? formatRulesSummary(guardrails)
    : `${formatRulesSummary(guardrails)}\n${guardrails.loadError}`;
}

function rulesData(guardrails: GuardrailsSnapshot): JsonValue {
  return toJson({
    enabled: guardrails.enabled,
    failMode: guardrails.failMode,
    revision: guardrails.revision,
    rules: guardrails.rules,
    ...(guardrails.errors === undefined ? {} : { errors: guardrails.errors }),
  });
}

/** Phoenix state for results: the snapshot with this session's link in place of the link template. */
function phoenixData(phoenix: PhoenixSnapshot, sessionId: string | undefined): JsonValue {
  const { sessionUrl: _template, ...rest } = phoenix;
  const link = sessionLink(phoenix, sessionId);
  return toJson({ ...rest, ...(link === undefined ? {} : { sessionLink: link }) });
}

function renderResult(value: AgentToolResult): string {
  return value.data === undefined ? value.text : `${value.text}\n${JSON.stringify(value.data)}`;
}

const TITLES: Record<AgentAction, string> = {
  status: 'NeMo Relay status',
  'rules.list': 'List NeMo Relay rules',
  'rules.test': 'Test NeMo Relay rules',
  'rules.add': 'Add NeMo Relay rule',
  'rules.update': 'Change NeMo Relay rule',
  'rules.enable': 'Turn on NeMo Relay rule',
  'rules.disable': 'Turn off NeMo Relay rule',
  'rules.remove': 'Delete NeMo Relay rule',
  'rules.reset': 'Restore default NeMo Relay rules',
  'guardrails.failMode': 'Set NeMo Relay fail mode',
  'phoenix.status': 'Phoenix status',
  'phoenix.start': 'Start Phoenix',
  'phoenix.connect': 'Connect to Phoenix',
  'phoenix.disconnect': 'Disconnect from Phoenix',
  'phoenix.stop': 'Stop Phoenix',
  'phoenix.link': 'Phoenix link for this session',
};

/** The tool card title: the action, plus the rule, tool, or mode it applies to. */
function callTitle(args: AgentToolArgs): string {
  const title = TITLES[args.action] ?? 'NeMo Relay';
  const subject =
    args.action === 'guardrails.failMode'
      ? args.mode
      : args.action === 'rules.test'
        ? args.tool
        : String(args.action).startsWith('rules.')
          ? args.id
          : undefined;
  return subject === undefined || subject.trim().length === 0 ? title : `${title}: ${subject.trim()}`;
}

function sessionIdOf(exec: unknown): string | undefined {
  const id = (exec as { agent?: { session?: { id?: unknown } } } | null | undefined)?.agent?.session?.id;
  return typeof id === 'string' && id.length > 0 ? id : undefined;
}

/** A plain JSON copy: the tool registry accepts only lossless JSON values. */
function toJson(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
