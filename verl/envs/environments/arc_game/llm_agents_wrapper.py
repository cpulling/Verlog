"""
ARCGame LLM-agents wrapper.

Parses LLM outputs of the form `ACTION: <idx>` (or just a bare integer) and
forwards them through ARCGameGymEnv's CSV action string interface. Tracks
behavioral metrics in the same shape BALROG/Verlog expects.
"""

from __future__ import annotations

import re

import gymnasium as gym


_ACTION_HEADER_RE = re.compile(r"action\s*[:#]\s*", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


class ARCGameLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.env = env
        self.format_penalty = float(kwargs.get("format_penalty", 0.0))
        self.binary_reward = bool(kwargs.get("binary_reward", False))
        self._last_long_term: str | None = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self._last_long_term = None
        return self.env.reset(**kwargs)

    def step(self, action, is_valid: bool = True):
        # `action` here is the CSV index string produced by extract_action below.
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not is_valid:
            reward = -self.format_penalty
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward

        new_long_term = (obs.get("text") or {}).get("long_term_context")
        is_loop = (
            self._last_long_term is not None and new_long_term == self._last_long_term
        )
        self._last_long_term = new_long_term

        metrics = info.get("metrics", {}) or {}
        metrics["behavior/loop_rate"] = 1.0 if is_loop else 0.0
        info["metrics"] = metrics
        return obs, reward * 1.0, terminated, truncated, info

    def extract_action(self, action: str):
        """Parse the model's raw output into an ARC action index string."""
        full_action = str(action)
        text = full_action

        header_match = _ACTION_HEADER_RE.search(text)
        if header_match:
            text = text[header_match.end():]

        m = _INT_RE.search(text)
        if m is None:
            executed = self.env.default_action if hasattr(self.env, "default_action") else ""
            is_valid = False
        else:
            idx = int(m.group(0))
            n = len(getattr(self.env, "valid_actions", []) or [])
            if 0 <= idx < n:
                executed = str(idx)
                is_valid = True
            else:
                executed = self.env.default_action if hasattr(self.env, "default_action") else ""
                is_valid = False

        valid_count = 1.0 if is_valid else 0.0
        metrics = {
            "behavior/valid_action_ratio": valid_count,
            "behavior/plan_length": float(len(_INT_RE.findall(full_action))),
            "behavior/backtrack_length": float(
                sum(full_action.lower().count(w) for w in [
                    "however", "different", "but", "wait", "won't", "can't",
                    "cannot", "another"
                ])
            ),
        }
        return full_action, executed, is_valid, metrics
