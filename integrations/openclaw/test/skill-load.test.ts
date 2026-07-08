// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { detectSkillLoads } from '../src/hook-replay/skill-load.js';

describe('skill-load detection', () => {
  it('detects every first-class skill tool name field', () => {
    for (const [toolName, params, skillName] of [
      ['Skill', { skill: 'review' }, 'review'],
      ['skill_view', { skill_name: 'testing' }, 'testing'],
      ['skill-view', { request: { name: 'authoring' } }, 'authoring'],
    ] as const) {
      assert.deepEqual(detectSkillLoads(toolName, params), [{ skillName, source: 'skill_tool' }]);
    }
  });

  it('rejects missing, empty, non-string, and unsupported first-class calls', () => {
    for (const [toolName, params] of [
      ['Skill', {}],
      ['Skill', { skill: '  ' }],
      ['skill_view', { name: 7 }],
      ['skill_catalog', { skill: 'review' }],
    ] as const) {
      assert.deepEqual(detectSkillLoads(toolName, params), []);
    }
  });

  it('detects structured readers, recursive fields, path styles, and per-call deduplication', () => {
    assert.deepEqual(
      detectSkillLoads('mcp__filesystem__read_multiple_files', {
        request: {
          paths: [
            '/skills/review/SKILL.md',
            'C:\\skills\\testing\\skill.MD',
            'relative/skills/authoring/SKILL.md',
            '/skills/review/SKILL.md',
          ],
        },
      }),
      [
        { skillName: 'review', source: 'structured_read' },
        { skillName: 'testing', source: 'structured_read' },
        { skillName: 'authoring', source: 'structured_read' },
      ],
    );
    for (const [toolName, params] of [
      ['Read', { path: '/skills/review/SKILL.md' }],
      ['read_file', { file_path: '/skills/review/SKILL.md' }],
      ['read_text_file', { filepath: '/skills/review/SKILL.md' }],
      ['file_read', { filename: '/skills/review/SKILL.md' }],
      ['mcp__filesystem__read_file', { file: '/skills/review/SKILL.md' }],
    ] as const) {
      assert.deepEqual(detectSkillLoads(toolName, params), [
        { skillName: 'review', source: 'structured_read' },
      ]);
    }
  });

  it('allows zero offset and rejects every partial structured-read control', () => {
    assert.deepEqual(detectSkillLoads('Read', { path: '/skills/review/SKILL.md', offset: 0 }), [
      { skillName: 'review', source: 'structured_read' },
    ]);
    for (const partial of [
      { offset: 1 },
      { offset: -1 },
      { limit: 20 },
      { range: '1:20' },
      { head: true },
      { tail: 20 },
      { start_line: 1 },
      { end_line: 20 },
      { line_start: 1 },
      { line_end: 20 },
      { options: { limit: 20 } },
    ]) {
      assert.deepEqual(detectSkillLoads('Read', { path: '/skills/review/SKILL.md', ...partial }), []);
    }
  });

  it('rejects ordinary paths, missing parents, and non-read tools', () => {
    for (const [toolName, params] of [
      ['Read', { path: '/skills/review/README.md' }],
      ['Read', { path: 'SKILL.md' }],
      ['Read', { path: '/SKILL.md' }],
      ['Read', { path: 'C:\\SKILL.md' }],
      ['Read', { path: '/skills/./SKILL.md' }],
      ['Read', { path: '/skills/../SKILL.md' }],
      ['write_file', { path: '/skills/review/SKILL.md' }],
      ['edit_file', { path: '/skills/review/SKILL.md' }],
    ] as const) {
      assert.deepEqual(detectSkillLoads(toolName, params), []);
    }
  });

  it('detects complete cat, bat, batcat, and PowerShell commands', () => {
    assert.deepEqual(detectSkillLoads('exec_command', { cmd: "cat '/skills/review/SKILL.md'" }), [
      { skillName: 'review', source: 'shell_read' },
    ]);
    assert.deepEqual(
      detectSkillLoads('terminal', {
        command: '"C:\\Tools\\bat.exe" --plain C:\\skills\\review\\SKILL.md',
      }),
      [{ skillName: 'review', source: 'shell_read' }],
    );
    assert.deepEqual(
      detectSkillLoads('run_shell_command', {
        command: 'batcat /skills/review/SKILL.md /skills/testing/SKILL.md /skills/review/SKILL.md',
      }),
      [
        { skillName: 'review', source: 'shell_read' },
        { skillName: 'testing', source: 'shell_read' },
      ],
    );
    for (const command of [
      "Get-Content -Raw -LiteralPath 'C:\\skills\\review\\SKILL.md'",
      'Get-Content -Encoding utf8 -Path C:\\skills\\review\\SKILL.md',
      'C:\\Windows\\System32\\Get-Content.exe C:\\skills\\review\\SKILL.md',
    ]) {
      assert.deepEqual(detectSkillLoads('powershell', { command }), [
        { skillName: 'review', source: 'shell_read' },
      ]);
    }
  });

  it('rejects partial, transformed, redirected, substituted, and compound shell commands', () => {
    for (const command of [
      "sed -n '1,20p' /skills/review/SKILL.md",
      'head /skills/review/SKILL.md',
      'tail /skills/review/SKILL.md',
      'bat -r 1:20 /skills/review/SKILL.md',
      'bat --line-range 1:20 /skills/review/SKILL.md',
      'bat --line-range=1:20 /skills/review/SKILL.md',
      'Get-Content -TotalCount 20 /skills/review/SKILL.md',
      'Get-Content -Tail 20 /skills/review/SKILL.md',
      'Get-Content -Head 20 /skills/review/SKILL.md',
      'Get-Content -First 20 /skills/review/SKILL.md',
      'Get-Content -Last 20 /skills/review/SKILL.md',
      'cat /skills/review/SKILL.md | head',
      'cat /skills/review/SKILL.md > /tmp/copy',
      'cat /skills/review/SKILL.md < /tmp/input',
      'cat /skills/review/SKILL.md && echo done',
      'cat /skills/review/SKILL.md || echo failed',
      'cat /skills/review/SKILL.md; echo done',
      'cat /skills/review/SKILL.md\necho done',
      'cat $(find /skills -name SKILL.md)',
      'cat `find /skills -name SKILL.md`',
    ]) {
      assert.deepEqual(detectSkillLoads('shell', { command }), []);
    }
  });

  it('rejects malformed, unknown, and non-command shell inputs', () => {
    for (const [toolName, params] of [
      ['shell', { command: "cat '/skills/review/SKILL.md" }],
      ['shell', { command: 'cp /skills/review/SKILL.md /tmp' }],
      ['shell', { command: '' }],
      ['shell', { command: 7 }],
      ['shell', { script: 'cat /skills/review/SKILL.md' }],
      ['python', { command: 'cat /skills/review/SKILL.md' }],
    ] as const) {
      assert.deepEqual(detectSkillLoads(toolName, params), []);
    }
  });
});
