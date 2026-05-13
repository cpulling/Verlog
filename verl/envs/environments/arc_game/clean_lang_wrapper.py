"""
ARCGame clean language wrapper.

Converts ARCGameGymEnv observations into a text-only prompt and exposes a
dynamic numeric-index action space ("0", "1", ..., str(N-1)) where N is the
number of valid actions for the current state.

The underlying env is verb/state-based and produces large nested dicts; this
wrapper flattens the most important fields (day, satisfaction, budget) and
renders the action list in the same numbered format that ARC's own llm_query
uses, so prompts are consistent with the agent_router stack.
"""

from __future__ import annotations

import gymnasium as gym


def _summarize_state(game_state: dict) -> str:
    session = game_state.get("sessionInfo", {}) or {}
    sat_budget = game_state.get("satisfactionAndBudget", {}) or {}
    day = session.get("currentDay", "?")
    segment = session.get("currentTimeSegment", "?")
    satisfaction = sat_budget.get("satisfaction", "?")
    budget = sat_budget.get("budget", "?")
    if isinstance(budget, (int, float)):
        budget_str = f"${budget:,}"
    else:
        budget_str = str(budget)
    return (
        f"Day {day}, Segment {segment}. "
        f"Satisfaction: {satisfaction}. Budget: {budget_str}."
    )


def _format_actions(valid_actions: list) -> str:
    if not valid_actions:
        return "(no valid actions)"
    lines = []
    for i, a in enumerate(valid_actions):
        action_type = a.get("actionType", "?")
        description = a.get("description", "?")
        cost = a.get("cost", 0)
        lines.append(f"{i}. [{action_type}] {description} (cost: ${cost})")
    return "\n".join(lines)


class ARCGameCleanLangWrapper(gym.Wrapper):
    """Text-rendering wrapper over ARCGameGymEnv.

    The underlying env already returns a game_state dict from `reset()` and
    `step()`; this wrapper attaches the rendered `obs["text"]` and tracks
    `language_action_space` based on the current `valid_actions` so that
    BALROG-style helpers (get_text_action / default_action) keep working.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._last_state: dict = {}
        self.language_action_space: list[str] = []
        self.progression: float = 0.0

    @property
    def max_steps(self):
        # Underlying ARC env carries this as `max_episode_steps`.
        return getattr(self.env, "max_episode_steps", 100)

    @property
    def default_action(self) -> str:
        return "0" if self.language_action_space else ""

    def get_text_action(self, action):
        # If called with an int-like, return the indexed description; otherwise echo.
        try:
            idx = int(action)
            if 0 <= idx < len(getattr(self.env, "valid_actions", []) or []):
                return self.env.valid_actions[idx].get("description", str(idx))
        except (TypeError, ValueError):
            pass
        return str(action)

    @property
    def valid_actions(self):
        # Expose the underlying ARCGameGymEnv's dynamic action list so the
        # LLM-agents wrapper can validate parsed indices without relying on
        # gym.Wrapper attribute proxying.
        return getattr(self.env, "valid_actions", []) or []

    def _refresh_action_space(self) -> None:
        n = len(self.valid_actions)
        self.language_action_space = [str(i) for i in range(n)]

    def _build_text_obs(self, game_state: dict) -> dict:
        valid_actions = getattr(self.env, "valid_actions", []) or []
        long_term_context = (
            f"Current situation:\n{_summarize_state(game_state)}\n\n"
            f"Available actions:\n{_format_actions(valid_actions)}"
        )
        return {"long_term_context": long_term_context, "short_term_context": ""}

    # ── gym API ────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_state = obs if isinstance(obs, dict) else {}
        self.progression = 0.0
        self._refresh_action_space()
        out = {"text": self._build_text_obs(self._last_state), "image": None,
               "game_state": self._last_state}
        return out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_state = obs if isinstance(obs, dict) else self._last_state
        if reward > 0:
            self.progression = max(self.progression, min(1.0, self.progression + 0.1))
        self._refresh_action_space()
        out = {"text": self._build_text_obs(self._last_state), "image": None,
               "game_state": self._last_state}
        return out, reward, terminated, truncated, info

    def get_stats(self):
        return {"progression": self.progression}
