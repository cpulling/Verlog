"""The CORA game for Verlog: rl.CoraEnv (ARC_Game repo) behind cora_adapter.CoraVerlogEnv."""


def get_instruction_prompt(env=None, mission: str = "") -> str:
    """The system prompt: CoraEnv's, the same prompt pack the CORA benchmark uses."""
    return env.get_instruction_prompt()
