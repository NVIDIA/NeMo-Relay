// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisory diagnostics for structurally suspicious agent invocations.

use crate::agents::CodingAgent;

pub(crate) const POSSIBLE_DUPLICATE_AGENT_EXECUTABLE: &str = "possible_duplicate_agent_executable";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvocationForm {
    Run,
    Shortcut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DuplicateAgentExecutable<'a> {
    agent: CodingAgent,
    form: InvocationForm,
    command: &'a [String],
}

impl<'a> DuplicateAgentExecutable<'a> {
    pub(crate) fn detect(
        agent: CodingAgent,
        command: &'a [String],
        form: InvocationForm,
    ) -> Option<Self> {
        let executable = command.first()?;
        (CodingAgent::infer(executable) == Some(agent)).then_some(Self {
            agent,
            form,
            command,
        })
    }

    pub(crate) fn log(&self) {
        let agent = self.agent.as_arg();
        log::warn!(
            target: "nemo_relay.cli",
            event = "agent_invocation_warning",
            diagnostic_code = POSSIBLE_DUPLICATE_AGENT_EXECUTABLE,
            agent = agent,
            duplicate_executable = agent,
            confidence = "high",
            action = "continued",
            command_modified = false,
            arguments_redacted = true;
            "Possible duplicate agent executable after `--`"
        );
    }

    pub(crate) fn format_warning(&self) -> String {
        format!(
            "WARNING: Possible duplicate agent executable after `--`.\n\
               Diagnostic: {POSSIBLE_DUPLICATE_AGENT_EXECUTABLE}\n\
               Duplicate executable: {}\n\
               Observed: {}\n\
               Recommended: {}\n\
               Inspect without launching: {}\n\
               Relay will continue without modifying the command.",
            self.agent.as_arg(),
            self.observed_command(),
            self.recommended_command(),
            self.dry_run_command(),
        )
    }

    fn observed_command(&self) -> String {
        let mut command = self.relay_prefix();
        command.push("--".into());
        command.push(self.agent.as_arg().into());
        if self.command.len() > 1 {
            command.push("<arguments redacted>".into());
        }
        render_command(&command)
    }

    fn recommended_command(&self) -> String {
        let mut command = self.relay_prefix();
        command.push("--".into());
        if self.command.len() > 1 {
            command.push("<arguments redacted>".into());
        }
        render_command(&command)
    }

    fn dry_run_command(&self) -> String {
        let mut command = self.relay_prefix();
        command.push("--dry-run".into());
        command.push("--".into());
        command.push(self.agent.as_arg().into());
        if self.command.len() > 1 {
            command.push("<arguments redacted>".into());
        }
        render_command(&command)
    }

    fn relay_prefix(&self) -> Vec<String> {
        match self.form {
            InvocationForm::Run => vec![
                "nemo-relay".into(),
                "run".into(),
                "--agent".into(),
                self.agent.as_arg().into(),
            ],
            InvocationForm::Shortcut => {
                vec!["nemo-relay".into(), self.agent.as_arg().into()]
            }
        }
    }
}

fn render_command(command: &[String]) -> String {
    command
        .iter()
        .map(|argument| crate::process::shell_quote_arg_for_platform(argument, cfg!(windows)))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
#[path = "../../../tests/coverage/shared/invocation_diagnostic_tests.rs"]
mod tests;
