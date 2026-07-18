"""ARCGame env registration for Verlog (TCP-driven Unity backend)."""

import os
import sys

from .clean_lang_wrapper import ARCGameCleanLangWrapper
from .llm_agents_wrapper import ARCGameLLMAgentsWrapper


def _ensure_smoke_on_path() -> None:
    arc_path = os.environ.get("ARC_GAME_PATH")
    if arc_path and arc_path not in sys.path:
        sys.path.insert(0, arc_path)


# ── Shared mechanics preamble (state block, entities, rules) ───────
# Kept as a module constant so both the XML-tag prompt and the tool-call
# prompt speak in the exact same voice — the ONLY difference between the
# two arms should be the HOW-TO-ACT paragraph and the closing example.
# Anything else in the preamble that changes muddies an A/B comparison.
_TOOL_HOW_TO_ACT = """HOW TO ACT — call the provided FUNCTIONS. Each turn you may emit any number of
`<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks. The available functions and
their argument schemas are listed above (build, hire, train, staff, deconstruct, task, transfer).
Prefer the stable task tokens (BUDGET_DAILY, FOOD_C01, ...) shown in each task's `id` field when
calling `task`; the raw integer taskId also works but is less stable across turns.

STAFF ONLY buildings listed in `available.needStaff`, passing the EXACT name shown there. A
building not in `needStaff` is either already fully staffed or not yet built — staffing it is
rejected.

EXECUTION ORDER: your calls resolve in a fixed commonsense order each turn — deconstruct, build,
hire, train, staff, transfer — NOT the textual order. Hiring and staffing in the SAME turn works;
staffing a building you just built THIS turn does NOT (it's still UnderConstruction — wait until
it appears in `needStaff` next round).

RESPOND with one short line of reasoning inside <think>...</think> (if in thinking mode), then
your tool calls. Example (2-turn plan):
<tool_call>{"name": "build", "arguments": {"type": "shelter", "site_id": 3}}</tool_call>
<tool_call>{"name": "hire", "arguments": {"kind": "untrained", "count": 4}}</tool_call>
<tool_call>{"name": "task", "arguments": {"task_id": "BUDGET_DAILY", "choice_id": 1}}</tool_call>
"""


def get_instruction_prompt(env=None, mission: str = "ARC disaster response") -> str:
    """System prompt.

    Two modes, selected by ARC_ACTION_MODE:
      - "text" (default): benchmark's minimal_cmd_v3 prompt with <build>...</build> XML grammar,
        matched to what `parse_commands` accepts today. Same surface the benchmark scores.
      - "tool_calls":     drop the XML grammar section, tell the model to call the tool schemas
        that Qwen3's chat template renders as a <tools> block. Used when
        `multi_turn.tool_config_path` is set. Mechanics preamble is IDENTICAL to
        the text prompt so the A/B isolates format only.
    """
    _ensure_smoke_on_path()
    from llm_smoke_test import (  # type: ignore
        CMD_MINIMAL_SYSTEM_PROMPT,
        cmd_system_prompt,
    )

    mode = os.environ.get("ARC_ACTION_MODE", "text").strip().lower()
    if mode == "tool_calls":
        # Strip the XML "HOW TO ACT" paragraph + closing REASONING example
        # from the benchmark prompt and splice in the tool-call directive.
        # Split at the well-known anchor so we keep the shared mechanics
        # preamble byte-identical to the text arm.
        anchor = "HOW TO ACT — emit COMMAND TAGS."
        preamble, _rest = CMD_MINIMAL_SYSTEM_PROMPT.split(anchor, 1)
        return preamble.rstrip() + "\n\n" + _TOOL_HOW_TO_ACT
    return cmd_system_prompt(manual_transfers=False, variant="minimal")
