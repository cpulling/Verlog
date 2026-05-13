"""ARCGame env registration for Verlog (TCP-driven Unity backend)."""

from .clean_lang_wrapper import ARCGameCleanLangWrapper
from .llm_agents_wrapper import ARCGameLLMAgentsWrapper


def get_instruction_prompt(env=None, mission: str = "ARC disaster response") -> str:
    """Static prompt prepended to each rollout.

    The action list varies each turn, so the prompt only describes the *format*
    of the response and the high-level objective. The numbered action list is
    appended to each observation by ARCGameCleanLangWrapper.
    """
    return (
        "You are an agent managing disaster relief operations in the ARC "
        "disaster-response simulator. Each turn you will receive the current "
        "situation (day, segment, satisfaction, budget) and a numbered list of "
        "available actions. Your job is to pick the single action that best "
        "balances immediate need with long-term sustainability.\n\n"
        f"Mission: {mission}.\n\n"
        "Response format:\n"
        "REASONING: <1-2 sentences explaining your choice>\n"
        "ACTION: <single integer index from the available action list>\n\n"
        "Output exactly one ACTION line per turn."
    ).strip()
