# Skill Benchmark: nemo-relay-install

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-relay-install`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 15 evaluation tasks (14 positive, 1 negative)
- Dataset digest: `sha256:ac9a9cc622e67035055f7153d383fd5ea4ef6039a1dfb0eeb0bd7180f89dc9c1` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.4% — baseline ran, but no comparable score was available; uplift unavailable | 71.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 86.7% → 80.0% (-6.7 points) | 80.0% → 66.7% (-13.3 points) |
| Correctness | 46.7% → 81.3% (+34.6 points) | 57.3% → 89.3% (+32.0 points) |
| Discoverability | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 78.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.96% → 61.39% (+21.43 points) | 35.0% → 56.6% (+21.6 points) |
| Efficiency | 73.9% — baseline ran, but no comparable score was available; uplift unavailable | 65.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,455,650 | 2,371,194 | +2,084,456 | +87.91% | skill 15/15; base 15/15 |
| claude-code | nemo-relay-install-001 | 162,667 | 225,307 | -62,640 | -27.80% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-002 | 230,842 | 90,805 | +140,037 | +154.22% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-003 | 336,754 | 121,416 | +215,338 | +177.36% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-004 | 515,190 | 122,927 | +392,263 | +319.10% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-005 | 332,291 | 151,985 | +180,306 | +118.63% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-006 | 92,279 | 343,396 | -251,117 | -73.13% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-007 | 92,164 | 90,299 | +1,865 | +2.07% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-008 | 456,799 | 91,221 | +365,578 | +400.76% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-009 | 580,468 | 123,825 | +456,643 | +368.78% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-010 | 135,037 | 127,415 | +7,622 | +5.98% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-011 | 155,895 | 155,074 | +821 | +0.53% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-012 | 502,936 | 164,712 | +338,224 | +205.34% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-013 | 126,432 | 153,225 | -26,793 | -17.49% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-014 | 328,036 | 123,537 | +204,499 | +165.54% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-015 | 407,860 | 286,050 | +121,810 | +42.58% | skill 1/1; base 1/1 |
| codex | All cases | 4,473,150 | 6,359,481 | -1,886,331 | -29.66% | skill 15/15; base 15/15 |
| codex | nemo-relay-install-001 | 65,084 | 30,402 | +34,682 | +114.08% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-002 | 192,352 | 68,701 | +123,651 | +179.98% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-003 | 203,514 | 103,025 | +100,489 | +97.54% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-004 | 456,736 | 250,390 | +206,346 | +82.41% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-005 | 338,220 | 311,924 | +26,296 | +8.43% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-006 | 28,301 | 452,435 | -424,134 | -93.74% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-007 | 173,181 | 32,304 | +140,877 | +436.10% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-008 | 18,458 | 18,347 | +111 | +0.61% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-009 | 469,707 | 3,213,997 | -2,744,290 | -85.39% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-010 | 114,387 | 77,146 | +37,241 | +48.27% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-011 | 145,451 | 153,515 | -8,064 | -5.25% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-012 | 1,287,056 | 664,730 | +622,326 | +93.62% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-013 | 279,119 | 113,393 | +165,726 | +146.15% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-014 | 454,994 | 654,532 | -199,538 | -30.49% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-015 | 246,590 | 214,640 | +31,950 | +14.89% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,928,800 | 8,730,675 | +198,125 | +2.27% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-relay-install/SKILL.md`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: https://api.github.com/ (`SKILL.md:45`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
