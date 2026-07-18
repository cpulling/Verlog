"""ARCGame env registration for Verlog (TCP-driven Unity backend)."""

import os
import sys

from .clean_lang_wrapper import ARCGameCleanLangWrapper
from .llm_agents_wrapper import ARCGameLLMAgentsWrapper


def _ensure_smoke_on_path() -> None:
    arc_path = os.environ.get("ARC_GAME_PATH")
    if arc_path and arc_path not in sys.path:
        sys.path.insert(0, arc_path)


def get_instruction_prompt(env=None, mission: str = "ARC disaster response") -> str:
    """System prompt: same one the benchmark's `minimal_cmd_v3` cell uses.

    Reuses llm_smoke_test.cmd_system_prompt(variant='minimal') so the RL agent
    trains against the same action surface (state-only obs + <build>/<hire>/
    <staff>/<task> command tags) that the benchmark scores.
    """
    _ensure_smoke_on_path()
    from llm_smoke_test import cmd_system_prompt  # type: ignore
    return cmd_system_prompt(manual_transfers=False, variant="minimal")
