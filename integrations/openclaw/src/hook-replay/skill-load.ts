// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Conservative skill-load detection for OpenClaw hook payloads. */

export const SKILL_LOADS_METADATA_KEY = 'nemo_relay.skill_loads';

export type SkillLoadDetection = {
  skillName: string;
  source: 'skill_tool' | 'structured_read' | 'shell_read';
};

/** Detect complete SKILL.md loads without retaining paths or commands. */
export function detectSkillLoads(toolName: string, params: unknown): SkillLoadDetection[] {
  const normalizedTool = normalizeIdentifier(toolName);
  let source: SkillLoadDetection['source'];
  let names: string[];

  if (normalizedTool === 'skill' || normalizedTool === 'skillview') {
    source = 'skill_tool';
    names = namedStrings(params, new Set(['skill', 'skillname', 'name']));
  } else if (isStructuredReader(normalizedTool) && !hasPartialReadControls(params)) {
    source = 'structured_read';
    names = namedValues(params, new Set(['path', 'filepath', 'filename', 'file', 'paths']))
      .flatMap(pathStrings)
      .map(skillNameFromPath)
      .filter((name): name is string => name !== undefined);
  } else if (isShellTool(normalizedTool)) {
    source = 'shell_read';
    names = namedStrings(params, new Set(['command', 'cmd']))
      .flatMap(completeReaderPaths)
      .map(skillNameFromPath)
      .filter((name): name is string => name !== undefined);
  } else {
    return [];
  }

  return [...new Set(names)].map((skillName) => ({ skillName, source }));
}

function isStructuredReader(toolName: string): boolean {
  return ['read', 'readfile', 'readtextfile', 'readmultiplefiles', 'fileread'].some(
    (reader) => toolName === reader || toolName.endsWith(reader),
  );
}

function isShellTool(toolName: string): boolean {
  return [
    'bash',
    'shell',
    'shellcommand',
    'exec',
    'execcommand',
    'execute',
    'terminal',
    'runcommand',
    'runshellcommand',
    'shellexec',
    'powershell',
  ].includes(toolName);
}

function hasPartialReadControls(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some(hasPartialReadControls);
  }
  if (!isRecord(value)) {
    return false;
  }
  return Object.entries(value).some(([rawKey, item]) => {
    const key = normalizeIdentifier(rawKey);
    const partial =
      (key === 'offset' && typeof item === 'number' && item !== 0) ||
      (['limit', 'range', 'head', 'tail', 'startline', 'endline', 'linestart', 'lineend'].includes(key) &&
        item !== null &&
        item !== undefined);
    return partial || hasPartialReadControls(item);
  });
}

function namedStrings(value: unknown, keys: Set<string>): string[] {
  return namedValues(value, keys)
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.trim())
    .filter(Boolean);
}

function namedValues(value: unknown, keys: Set<string>): unknown[] {
  if (Array.isArray(value)) {
    return value.flatMap((item) => namedValues(item, keys));
  }
  if (!isRecord(value)) {
    return [];
  }
  const values: unknown[] = [];
  for (const [rawKey, item] of Object.entries(value)) {
    if (keys.has(normalizeIdentifier(rawKey))) {
      values.push(item);
    }
    values.push(...namedValues(item, keys));
  }
  return values;
}

function pathStrings(value: unknown): string[] {
  if (typeof value === 'string') {
    return [value];
  }
  return Array.isArray(value) ? value.flatMap(pathStrings) : [];
}

function skillNameFromPath(path: string): string | undefined {
  const components = path
    .trim()
    .replace(/^['"]|['"]$/g, '')
    .split(/[\\/]/)
    .filter(Boolean);
  if (components.length < 2 || components.at(-1)?.toLowerCase() !== 'skill.md') {
    return undefined;
  }
  const parent = components.at(-2);
  return parent && parent !== '.' && parent !== '..' && !parent.endsWith(':') ? parent : undefined;
}

function completeReaderPaths(command: string): string[] {
  const words = tokenizeSimpleCommand(command);
  if (!words?.length) {
    return [];
  }
  const executable = words[0]?.split(/[\\/]/).at(-1)?.toLowerCase().replace(/\.exe$/, '');
  const args = words.slice(1);
  if (executable === 'cat') {
    return positionalPaths(args, []);
  }
  if (executable === 'bat' || executable === 'batcat') {
    return positionalPaths(args, ['-r', '--line-range']);
  }
  if (executable === 'get-content') {
    return powershellContentPaths(args);
  }
  return [];
}

function positionalPaths(args: string[], rejectedFlags: string[]): string[] {
  if (args.some((arg) => rejectedFlags.some((flag) => arg.toLowerCase() === flag || arg.startsWith(`${flag}=`)))) {
    return [];
  }
  return args.filter((arg) => !arg.startsWith('-'));
}

function powershellContentPaths(args: string[]): string[] {
  const rejected = ['-totalcount', '-tail', '-head', '-first', '-last'];
  if (args.some((arg) => rejected.some((flag) => arg.toLowerCase() === flag || arg.startsWith(`${flag}:`)))) {
    return [];
  }
  const paths: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg?.toLowerCase() === '-path' || arg?.toLowerCase() === '-literalpath') {
      const path = args[index + 1];
      if (path) {
        paths.push(path);
      }
      index += 1;
    } else if (arg && !arg.startsWith('-')) {
      paths.push(arg);
    }
  }
  return paths;
}

function tokenizeSimpleCommand(command: string): string[] | undefined {
  const words: string[] = [];
  let current = '';
  let quote: '"' | "'" | undefined;
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index];
    if (quote) {
      if (character === quote) {
        quote = undefined;
      } else if (quote === '"' && character === '\\') {
        const next = command[index + 1];
        if (next && ['\\', '"', '$', '`', '\n'].includes(next)) {
          index += 1;
          current += next;
        } else {
          current += character;
        }
      } else {
        current += character;
      }
    } else if (character === '"' || character === "'") {
      quote = character;
    } else if ('|&;<>`\n\r'.includes(character ?? '') || (character === '$' && command[index + 1] === '(')) {
      return undefined;
    } else if (/\s/.test(character ?? '')) {
      if (current) {
        words.push(current);
        current = '';
      }
    } else if (character === '\\' && /[\s\\'\"]/.test(command[index + 1] ?? '')) {
      index += 1;
      current += command[index];
    } else {
      current += character;
    }
  }
  if (quote) return undefined;
  if (current) words.push(current);
  return words.length ? words : undefined;
}

function normalizeIdentifier(value: string): string {
  return [...value]
    .filter((character) => /[a-z0-9]/i.test(character))
    .join('')
    .toLowerCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
