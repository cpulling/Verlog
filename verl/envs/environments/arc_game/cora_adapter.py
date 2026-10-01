"""Verlog's view of the CORA game: a thin adapter over rl.CoraEnv (ARC_Game repo).

Everything game-facing — the system prompt, the tool schema, the observation, resolving and
executing the policy's tool calls, the reward (change in score) — is CoraEnv's, the same code the
CORA benchmark plays through. This adapter only speaks Verlog's env protocol:

    BalrogEnv.step(action):  extract_action(action) -> step(executed, is_valid) -> obs, reward, ...

and adds the RL-side pieces: optional reward shaping for malformed / rejected calls, and the
per-step metrics Verlog logs (consistent keys on every step for verl's protocol concat).
"""
from __future__ import annotations

import gymnasium as gym

# Verlog's tool_agent_loop hands over {"raw_text", "content", "tool_calls": [(name, args_json)]}
# when the model emitted <tool_call> blocks, or the decoded text when it emitted none.
ACTION_TYPES = ("build", "hire", "train", "staff", "deconstruct", "task", "transfer")


class CoraVerlogEnv(gym.Env):
    def __init__(self, cora_env, format_penalty: float = 0.0, semantic_penalty: float = 0.0,
                 format_bonus: float = 0.0):
        """format_penalty: subtracted on a turn with no calls or with a malformed/invalid call.
        semantic_penalty: subtracted per call the game refused. format_bonus: added on a turn
        whose calls all executed; when set, a turn's total penalty is capped at -format_bonus."""
        super().__init__()
        self.cora = cora_env
        # The policy acts through tool calls; these spaces only satisfy gymnasium.Wrapper.
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Sequence(gym.spaces.Text(max_length=4096))
        self.format_penalty, self.semantic_penalty, self.format_bonus = \
            float(format_penalty), float(semantic_penalty), float(format_bonus)
        self._ep = {}
        # BALROG helper API that Verlog's wrappers read; tool calls have no discrete action space.
        self.language_action_space: list = []
        self.default_action: list = []

    # ── Verlog protocol ──
    @property
    def system_prompt(self) -> str:
        return self.cora.system_prompt

    @property
    def max_steps(self) -> int:
        return self.cora.config.max_steps

    def get_instruction_prompt(self, *args, **kwargs) -> str:
        return self.cora.system_prompt

    def reset(self, **kwargs):
        user, info = self.cora.reset()
        self._ep = {"turns": 0, "score_return": 0.0, **{f"n_{t}": 0 for t in ACTION_TYPES},
                    "calls": 0, "executed": 0, "refused": 0, "invalid": 0, "malformed_turns": 0}
        info["metrics"] = self._metrics(info, {}, terminal=False)
        return self._obs(user), info

    def extract_action(self, action):
        """(raw model output, the tool calls, None, {}): validity is decided by execution in step()."""
        if isinstance(action, dict) and "tool_calls" in action:      # cora.executor parses the arguments
            return str(action.get("raw_text") or ""), list(action["tool_calls"]), None, {}
        return str(action), [], None, {}

    def step(self, calls, is_valid=None):
        user, reward, terminated, truncated, info = self.cora.step(calls)
        statuses = [c["status"] for c in info["calls"]]
        executed, refused, invalid = (statuses.count(s) for s in ("executed", "refused", "invalid"))
        all_ok = bool(statuses) and executed == len(statuses)
        shaping = 0.0
        if all_ok and self.format_bonus:
            shaping = self.format_bonus
        elif not all_ok:
            if (not statuses or invalid or info["malformed"]) and self.format_penalty:
                shaping -= self.format_penalty
            if refused and self.semantic_penalty:
                shaping -= self.semantic_penalty * refused
            if self.format_bonus:
                shaping = max(shaping, -self.format_bonus)
        ep = self._ep
        ep["turns"] += 1
        ep["score_return"] += float(reward)
        ep["calls"] += len(statuses)
        ep["executed"] += executed; ep["refused"] += refused; ep["invalid"] += invalid
        ep["malformed_turns"] += int(info["malformed"])
        for c in info["calls"]:
            if c["status"] == "executed" and c["tool"] in ACTION_TYPES:
                ep[f"n_{c['tool']}"] += 1
        turn = {"calls": len(statuses), "executed": executed, "refused": refused, "invalid": invalid,
                "malformed": info["malformed"], "all_ok": all_ok, "shaping": shaping}
        info["metrics"] = self._metrics(info, turn, terminal=bool(terminated or truncated))
        info.pop("call_results", None)          # objects; info["calls"] holds their dicts
        return self._obs(user), float(reward) + shaping, terminated, truncated, info

    def get_stats(self):
        return {}

    def close(self):
        self.cora.close()

    # ── helpers ──
    @staticmethod
    def _obs(user: str) -> dict:
        return {"text": {"long_term_context": user, "short_term_context": ""}, "image": None}

    def _metrics(self, info: dict, turn: dict, terminal: bool) -> dict:
        ep, done = self._ep, 1.0 if terminal else 0.0
        score = float(info.get("score", 0.0) or 0.0)
        m = {k: float(v) for k, v in (info.get("metrics") or {}).items()}        # game/* (GameEnv)
        m.update({
            "behavior/tool_calls": float(turn.get("calls", 0)),
            "behavior/calls_executed": float(turn.get("executed", 0)),
            "behavior/calls_refused": float(turn.get("refused", 0)),
            "behavior/calls_invalid": float(turn.get("invalid", 0)),
            "behavior/malformed_turn": float(bool(turn.get("malformed"))),
            "behavior/empty_turn": float(turn.get("calls", 1) == 0),
            "behavior/all_calls_executed": float(bool(turn.get("all_ok"))),
            "behavior/shaping": float(turn.get("shaping", 0.0)),
            "episode/done": done,
            "episode/final_score": score * done,
            "episode/score_return": float(ep.get("score_return", 0.0)) * done,
            "episode/turns": float(ep.get("turns", 0)) * done,
            "episode/calls_executed": float(ep.get("executed", 0)) * done,
            "episode/calls_refused": float(ep.get("refused", 0)) * done,
            "episode/calls_invalid": float(ep.get("invalid", 0)) * done,
            "episode/malformed_turns": float(ep.get("malformed_turns", 0)) * done,
        })
        for t in ACTION_TYPES:
            m[f"episode/executed/{t}"] = float(ep.get(f"n_{t}", 0)) * done
        return m

