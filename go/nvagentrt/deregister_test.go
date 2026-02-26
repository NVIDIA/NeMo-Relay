package nvagentrt

import "testing"

// ============================================================================
// Deregister nonexistent
// ============================================================================

func TestDeregisterNonexistentGuardrails(t *testing.T) {
	// These should not panic, but they may return errors
	DeregisterToolSanitizeRequestGuardrail("nonexistent")
	DeregisterToolSanitizeResponseGuardrail("nonexistent")
	DeregisterToolConditionalExecutionGuardrail("nonexistent")
	DeregisterLlmSanitizeRequestGuardrail("nonexistent")
	DeregisterLlmSanitizeResponseGuardrail("nonexistent")
	DeregisterLlmConditionalExecutionGuardrail("nonexistent")
}

func TestDeregisterNonexistentIntercepts(t *testing.T) {
	DeregisterToolRequestIntercept("nonexistent")
	DeregisterToolResponseIntercept("nonexistent")
	DeregisterToolExecutionIntercept("nonexistent")
	DeregisterLlmRequestIntercept("nonexistent")
	DeregisterLlmResponseIntercept("nonexistent")
	DeregisterLlmStreamResponseIntercept("nonexistent")
	DeregisterLlmExecutionIntercept("nonexistent")
	DeregisterLlmStreamExecutionIntercept("nonexistent")
}
