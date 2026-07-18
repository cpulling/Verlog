"""ARCGame clean language wrapper — `minimal_cmd_v3` observation.

Renders each turn as the same compact state block the benchmark's
`bench_format_ablation/compact_minimal_cmd_v3` cell uses:
  * state-only obs (no enumerated menu) via `summarize_commands`
  * compact text serialization via `render_state_compact`

The system prompt (cmd grammar + minimal PIMMUR variant) is provided
separately by `arc_game/__init__.py:get_instruction_prompt`, and the
LLM's tag output is parsed by `parse_commands` in the llm_agents wrapper.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym


def _ensure_smoke_on_path() -> None:
    arc_path = os.environ.get("ARC_GAME_PATH")
    if arc_path and arc_path not in sys.path:
        sys.path.insert(0, arc_path)


class ARCGameCleanLangWrapper(gym.Wrapper):
    """Text-rendering wrapper over ARCGameGymEnv (cmd/compact/minimal)."""

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._last_state: dict = {}
        # Kept for backwards-compat with the BALROG helper API. Under the cmd
        # format actions are emitted as free-form tags, not integer indices,
        # so this list has no useful contents each turn — leave it empty.
        self.language_action_space: list[str] = []
        self.progression: float = 0.0

    @property
    def max_steps(self):
        return getattr(self.env, "max_episode_steps", 100)

    @property
    def default_action(self) -> str:
        # No-op turn under the cmd format is the empty string (the base env
        # treats an empty action CSV as a genuine no-op — see step()).
        return ""

    def get_text_action(self, action):
        return str(action)

    @property
    def valid_actions(self):
        return getattr(self.env, "valid_actions", []) or []

    def _build_text_obs(self) -> dict:
        _ensure_smoke_on_path()
        from llm_smoke_test import (  # type: ignore
            render_state_compact,
            summarize_commands,
        )
        state_obs = summarize_commands(self.env)
        long_term_context = render_state_compact(state_obs)
        return {"long_term_context": long_term_context, "short_term_context": ""}

    # ── gym API ────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_state = obs if isinstance(obs, dict) else {}
        self.progression = 0.0
        out = {"text": self._build_text_obs(), "image": None,
               "game_state": self._last_state}
        return out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_state = obs if isinstance(obs, dict) else self._last_state
        if reward > 0:
            self.progression = max(self.progression, min(1.0, self.progression + 0.1))
        out = {"text": self._build_text_obs(), "image": None,
               "game_state": self._last_state}
        return out, reward, terminated, truncated, info

    def get_stats(self):
        return {"progression": self.progression}
