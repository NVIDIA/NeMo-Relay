## Description: <br>
Use this skill when first-time NeMo Relay users want to try Relay, choose the least-complex supported quick start, or verify initial value through the CLI, a maintained integration, or direct Python, Node.js, or Rust instrumentation before production setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who are new to NeMo Relay and want to quickly evaluate it by following the least-complex try-now path for their existing environment, whether through the CLI, a maintained framework integration, or direct language-level instrumentation. <br>

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
- [CLI Try-Now Reference](references/cli-try-now.md) <br>
- [Built-In Integrations Try-Now Reference](references/built-in-integrations-try-now.md) <br>
- [Manual Language Try-Now Reference](references/manual-language-try-now.md) <br>
- [NeMo Relay CLI Overview](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/about) <br>
- [Supported Integrations](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/about) <br>
- [Language Quick Starts](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start) <br>
- [Plugin Configuration](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/about) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 15 evaluation tasks (14 positive skill-activation, 1 negative) in the astra-sandbox environment via NVSkills-Eval external profile. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 97% (+0%) | 97% (+10%) |
| Correctness | 8 | 89% (+61%) | 76% (+38%) |
| Discoverability | 8 | 82% (+50%) | 74% (+35%) |
| Effectiveness | 8 | 74% (+50%) | 65% (+32%) |
| Efficiency | 8 | 72% (+34%) | 68% (+29%) |

## Skill Version(s): <br>
0.6.0-alpha.20260715 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
