# Failure modes: base model vs run 1

- **base**: `vanillux2-base-k3-20260926T140340Z`, 267 finished trials
- **sft-run1**: `vanillux2-sft-k3-20260926T230224Z`, 267 finished trials

Every number below is computed from harbor's per-trial files with the fixed rules at the end of this page. No model or human judgement is involved.

## 1. Why each trial ended (one primary cause per trial)

| primary cause | base | sft-run1 |
|---|---|---|
| Solved | 3 (1%) | 4 (1%) |
| Verifier error/timeout (infra) | 1 (0%) | 4 (1%) |
| Agent timeout (90 min) | 4 (1%) | 77 (29%) |
| Other agent exception | 0 (0%) | 0 (0%) |
| Submitted, tests failed | 212 (79%) | 14 (5%) |
| Hit 64-step limit | 5 (2%) | 35 (13%) |
| Hit format-error limit | 0 (0%) | 0 (0%) |
| Stopped early: context window exceeded | 42 (16%) | 133 (50%) |
| **trials with ≥1 format-error turn** | 47 (18%) | 124 (46%) |

## 2. Everything that went wrong, counted per occurrence

| event | base | sft-run1 |
|---|---|---|
| agent turns (total) | 2806 | 7797 |
| format error: Truncated: <tool_call> never closed | 4 | 32 |
| format error: Invalid JSON: raw control char at end of command | 0 | 2 |
| format error: Invalid JSON: raw control char mid-command | 32 | 33 |
| format error: Invalid JSON: invalid \escape | 90 | 49 |
| format error: Invalid JSON: other | 7 | 66 |
| format error: Closed call with valid JSON | 0 | 0 |
| format error: No <tool_call> in text | 30 | 99 |
| format-error turn ≥45,000 chars (≈16k-token cap) | 11 | 91 |
| format-error turn with unique-line ratio < 0.5 (repetition loop) | 9 | 86 |
| LLM time spent on turns that became format errors | 17.0 h of 100.3 h | 144.9 h of 266.8 h |
| agent commands killed at the 120 s limit (exit 124) | 121 | 540 |
| `TimeoutExpired` inside the verifier's tests | 54 | 51 |

Sections 3 and 4 of the generated report (per-task examples and format-error excerpts) are left out of this copy. Regenerate them with `scripts/failure_mode_bucketing.py` from the Harbor job directories.

## Rules

Primary cause, first match wins:

1. **Solved**: reward == 1
2. **Verifier error/timeout (infra)**: no verifier reward, or exception VerifierTimeoutError
3. **Agent timeout (90 min)**: exception AgentTimeoutError
4. **Other agent exception**: any other exception_type
5. **Submitted, tests failed**: agent ran a command containing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, reward 0
6. **Hit 64-step limit**: 64 recorded steps, never submitted
7. **Hit format-error limit**: 64 format-error turns
8. **Stopped early: context window exceeded**: none of the above: the agent loop only exits this way on ContextWindowExceededError

A format error is an assistant turn with no tool call that the agent answered with its 'Format error' message. Its kind, first match wins:

1. **Truncated: <tool_call> never closed**: text opens <tool_call> but has no </tool_call>
2. **Invalid JSON: raw control char at end of command**: json error 'Invalid control character' within the last 10 chars of the call
3. **Invalid JSON: raw control char mid-command**: 'Invalid control character' elsewhere
4. **Invalid JSON: invalid \escape**: json error 'Invalid \escape'
5. **Invalid JSON: other**: any other json.JSONDecodeError
6. **Closed call with valid JSON**: server still returned no tool_calls (e.g. wrong tool name)
7. **No <tool_call> in text**: prose/thinking only, or a call the server parsed but the agent rejected
