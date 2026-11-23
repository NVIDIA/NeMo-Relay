# Skill Benchmark: nemo-relay-install

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-relay-install`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 15 evaluation tasks (14 positive, 1 negative)
- Dataset digest: `sha256:ac9a9cc622e67035055f7153d383fd5ea4ef6039a1dfb0eeb0bd7180f89dc9c1` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-relay-install | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-relay-install | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-relay-install | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-relay-install | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-relay-install | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nemo-relay-install`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 78.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.3% → 90.0% (-3.3 points) | 83.3% → 73.3% (-10.0 points) |
| Correctness | 41.3% → 96.0% (+54.7 points) | 46.7% → 85.3% (+38.6 points) |
| Discoverability | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.7% → 76.0% (+44.3 points) | 32.8% → 70.6% (+37.8 points) |
| Efficiency | 74.9% — baseline ran, but no comparable score was available; uplift unavailable | 78.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 95.4% (-2.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.3% → 86.9% (-1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,378,471 | 2,140,861 | +1,237,610 | +57.81% | skill 15/15; base 15/15 |
| claude-code | nemo-relay-install-001 | 234,256 | 256,040 | -21,784 | -8.51% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-002 | 232,412 | 106,927 | +125,485 | +117.36% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-003 | 194,035 | 106,915 | +87,120 | +81.49% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-004 | 508,798 | 143,774 | +365,024 | +253.89% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-005 | 145,887 | 107,288 | +38,599 | +35.98% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-006 | 147,646 | 305,442 | -157,796 | -51.66% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-007 | 109,794 | 107,269 | +2,525 | +2.35% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-008 | 186,684 | 35,515 | +151,169 | +425.65% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-009 | 198,527 | 180,336 | +18,191 | +10.09% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-010 | 154,621 | 145,785 | +8,836 | +6.06% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-011 | 355,406 | 106,658 | +248,748 | +233.22% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-012 | 242,026 | 107,436 | +134,590 | +125.27% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-013 | 149,996 | 142,811 | +7,185 | +5.03% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-014 | 235,901 | 144,635 | +91,266 | +63.10% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-install-015 | 282,482 | 144,030 | +138,452 | +96.13% | skill 1/1; base 1/1 |
| codex | All cases | 1,776,699 | 1,619,259 | +157,440 | +9.72% | skill 15/15; base 15/15 |
| codex | nemo-relay-install-001 | 103,566 | 21,679 | +81,887 | +377.72% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-002 | 43,382 | 40,540 | +2,842 | +7.01% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-003 | 135,274 | 62,970 | +72,304 | +114.82% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-004 | 450,898 | 54,693 | +396,205 | +724.42% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-005 | 100,926 | 66,540 | +34,386 | +51.68% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-006 | 27,731 | 322,621 | -294,890 | -91.40% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-007 | 27,743 | 62,918 | -35,175 | -55.91% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-008 | 13,382 | 13,267 | +115 | +0.87% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-009 | 102,036 | 83,759 | +18,277 | +21.82% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-010 | 64,229 | 64,221 | +8 | +0.01% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-011 | 58,261 | 58,794 | -533 | -0.91% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-012 | 262,407 | 385,163 | -122,756 | -31.87% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-013 | 47,469 | 54,428 | -6,959 | -12.79% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-014 | 117,846 | 286,877 | -169,031 | -58.92% | skill 1/1; base 1/1 |
| codex | nemo-relay-install-015 | 221,549 | 40,789 | +180,760 | +443.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,155,170 | 3,760,120 | +1,395,050 | +37.10% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

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
