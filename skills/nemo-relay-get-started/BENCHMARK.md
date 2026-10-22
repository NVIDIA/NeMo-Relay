# Skill Benchmark: nemo-relay-get-started

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-relay-get-started`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 22 evaluation tasks (21 positive, 1 negative)
- Dataset digest: `sha256:477cc9bd7b45256aa6d8f837938c05e6d79c539ed8bce8bf1908e266f61f3e9f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 66.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 95.5% → 88.6% (-6.9 points) | 81.8% → 84.1% (+2.3 points) |
| Correctness | 35.5% → 86.4% (+50.9 points) | 54.6% → 78.2% (+23.6 points) |
| Discoverability | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 81.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.4% → 68.0% (+31.6 points) | 35.7% → 56.5% (+20.8 points) |
| Efficiency | 78.0% — baseline ran, but no comparable score was available; uplift unavailable | 34.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 8,501,714 | 2,841,312 | +5,660,402 | +199.22% | skill 22/22; base 22/22 |
| claude-code | nemo-relay-get-started-001 | 234,300 | 90,328 | +143,972 | +159.39% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-002 | 501,566 | 89,724 | +411,842 | +459.01% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-003 | 1,281,565 | 120,761 | +1,160,804 | +961.24% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-004 | 669,151 | 29,060 | +640,091 | +2202.65% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-005 | 163,588 | 89,706 | +73,882 | +82.36% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-006 | 566,837 | 121,225 | +445,612 | +367.59% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-007 | 325,431 | 59,026 | +266,405 | +451.34% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-008 | 259,030 | 153,136 | +105,894 | +69.15% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-009 | 128,810 | 120,904 | +7,906 | +6.54% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-010 | 191,136 | 181,637 | +9,499 | +5.23% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-011 | 329,840 | 151,120 | +178,720 | +118.26% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-012 | 60,403 | 30,582 | +29,821 | +97.51% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-013 | 299,576 | 89,462 | +210,114 | +234.86% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-014 | 377,437 | 60,903 | +316,534 | +519.73% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-015 | 804,985 | 153,982 | +651,003 | +422.78% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-016 | 352,848 | 122,166 | +230,682 | +188.83% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-017 | 448,571 | 280,717 | +167,854 | +59.79% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-018 | 437,991 | 420,860 | +17,131 | +4.07% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-019 | 439,503 | 120,656 | +318,847 | +264.26% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-020 | 133,817 | 90,144 | +43,673 | +48.45% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-021 | 327,526 | 174,764 | +152,762 | +87.41% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-022 | 167,803 | 90,449 | +77,354 | +85.52% | skill 1/1; base 1/1 |
| codex | All cases | 7,500,771 | 12,726,457 | N/A | N/A | skill 21/22; base 21/22 |
| codex | nemo-relay-get-started-001 | 32,351 | 18,475 | +13,876 | +75.11% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-002 | N/A | 75,390 | N/A | N/A | skill 0/1; base 1/1 |
| codex | nemo-relay-get-started-003 | 1,297,787 | 392,320 | +905,467 | +230.80% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-004 | 13,340 | 13,284 | +56 | +0.42% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-005 | 571,101 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemo-relay-get-started-006 | 39,240 | 13,493 | +25,747 | +190.82% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-007 | 56,325 | 38,983 | +17,342 | +44.49% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-008 | 139,829 | 76,289 | +63,540 | +83.29% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-009 | 81,040 | 1,280,482 | -1,199,442 | -93.67% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-010 | 79,819 | 13,524 | +66,295 | +490.20% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-011 | 39,096 | 52,962 | -13,866 | -26.18% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-012 | 25,411 | 39,926 | -14,515 | -36.35% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-013 | 43,618 | 66,226 | -22,608 | -34.14% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-014 | 80,456 | 41,991 | +38,465 | +91.60% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-015 | 1,450,454 | 712,199 | +738,255 | +103.66% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-016 | 2,211,957 | 1,903,076 | +308,881 | +16.23% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-017 | 46,635 | 29,892 | +16,743 | +56.01% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-018 | 92,424 | 7,200,869 | -7,108,445 | -98.72% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-019 | 1,064,010 | 497,376 | +566,634 | +113.92% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-020 | 28,229 | 69,149 | -40,920 | -59.18% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-021 | 68,457 | 101,834 | -33,377 | -32.78% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-022 | 39,192 | 88,717 | -49,525 | -55.82% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 16,002,485 | 15,567,769 | N/A | N/A | skill 43/44; base 43/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in agent-runtimes-try-now.md (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-relay-get-started/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nemo-relay-get-started/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
