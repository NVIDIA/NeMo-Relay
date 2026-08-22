## Description: <br>
Use this skill when assessing, extending, or implementing NeMo Relay support in an agent harness or agent framework, including coding agents and orchestration runtimes, when the host lacks Relay support or an existing integration needs deeper coverage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers assessing, extending, or implementing NeMo Relay observability and middleware support within agent frameworks and orchestration runtimes that lack coverage or need deeper integration. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Assess The Host](references/assess-host.md) <br>
- [Concurrency And Lifecycle](references/concurrency-and-lifecycle.md) <br>
- [Host Attachment And Hook Patterns](references/host-attachment-patterns.md) <br>
- [Implement The Integration](references/implement-integration.md) <br>
- [Qualify The Integration](references/qualify-integration.md) <br>
- [NeMo Relay GitHub Repository](https://github.com/NVIDIA/NeMo-Relay/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Configuration instructions] <br>
**Output Format:** [Markdown with structured decision records and code examples] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 15 evaluation tasks (12 positive, 3 negative) using the skill-evaluator-dataset-snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Assesses whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 58% → 92% (+34 points) | 64% → 92% (+28 points) |
| Security | 93% → 100% (+7 points) | 87% → 100% (+13 points) |
| Correctness | 60% → 92% (+32 points) | 76% → 96% (+20 points) |
| Discoverability | 46% → 99% (+53 points) | 50% → 94% (+43 points) |
| Effectiveness | 45% → 71% (+26 points) | 55% → 76% (+22 points) |
| Efficiency | 45% → 97% (+52 points) | 53% → 96% (+43 points) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
7eb33b6 (source: git SHA, committed 2026-08-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
