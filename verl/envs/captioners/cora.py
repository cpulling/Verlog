from verl.envs.captioners.base import BaseCaptioner


class CoraCaptioner(BaseCaptioner):
    """The CORA turn contract (ARC_Game docs/ARCHITECTURE.md): the system prompt and the turn's user
    message, exactly as the CORA benchmark sends them — no history, prefix or answer-format footer
    (the tool schema rides the chat template's tools block)."""

    def __init__(self, prompt_builder, env_name=None):
        super().__init__(prompt_builder)

    def get_obs(self, obs):
        return [{"role": "system", "content": self.prompt_builder.system_prompt},
                {"role": "user", "content": obs["text"]["long_term_context"]}]

    def update_action(self, full_action, executed_action):
        pass
