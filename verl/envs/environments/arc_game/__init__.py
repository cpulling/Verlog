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
# _TOOL_HOW_TO_ACT deleted: the tool-mode directive now lives ONLY in cora_prompts
# (tool_system_prompt(wire_format="hermes")). This copy had drifted from it -- and was
# the MORE correct of the two on execution order -- which is exactly why one copy.



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
        cmd_system_prompt,
        tool_system_prompt,
    )

    mode = os.environ.get("ARC_ACTION_MODE", "text").strip().lower()
    if mode == "tool_calls":
        # veRL renders the schemas as a <tools> block and expects hermes-style
        # <tool_call>{...}</tool_call> emissions; wire_format="hermes" swaps only that
        # opening paragraph. Everything else -- mechanics preamble AND action rules --
        # is byte-identical to the benchmark's typed arm.
        return tool_system_prompt(manual_transfers=False, variant="minimal",
                                  wire_format="hermes")
    return cmd_system_prompt(manual_transfers=False, variant="minimal")
