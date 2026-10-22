## Description: <br>
Use this skill when choosing or running NeMo Relay installation for an agent runtime, application language, framework integration, or plugin bundle, or checking a runtime's native Relay readiness. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to select and install the appropriate current stable NeMo Relay package or executable for a chosen agent runtime, application language, framework integration, or plugin bundle, and to verify that the installation is available. <br>

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
- [CLI Installation](references/cli-install.md) <br>
- [Language Packages](references/language-packages.md) <br>
- [Maintained Integrations](references/maintained-integrations.md) <br>
- [NeMo Relay Installation Guide](https://docs.nvidia.com/nemo/relay/getting-started/installation) <br>
- [NeMo Relay Support Matrix](https://docs.nvidia.com/nemo/relay/reference/support-matrix) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (14 positive, 1 negative), each attempt run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, covering unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the task, combining goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 77.4% — baseline ran, but no comparable score was available; uplift unavailable | 71.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 86.7% → 80.0% (-6.7 points) | 80.0% → 66.7% (-13.3 points) |
| Correctness | 46.7% → 81.3% (+34.6 points) | 57.3% → 89.3% (+32.0 points) |
| Discoverability | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 78.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.96% → 61.39% (+21.43 points) | 35.0% → 56.6% (+21.6 points) |
| Efficiency | 73.9% — baseline ran, but no comparable score was available; uplift unavailable | 65.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
2e0b348a (source: git SHA, committed 2026-10-20) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
