"""Smoke test for the CORA env in Verlog, through the trainer's own path (no model, no GPU).

get_balrog_env (registry -> CoraVerlogEnv -> rl.CoraEnv, with the cora captioner) launches the
headless build, resets, and plays a few scripted turns of tool calls, as tool_agent_loop hands
them over, including an empty turn and a malformed call. Prints the messages, rewards and metric
keys so a run can be checked before a GPU job.

    ARC_GAME_PATH=.../ARC_Game_New python smoke_arc_game_env.py [--port 9970] [--steps 4]
    ARC_GAME_PATH=... python smoke_arc_game_env.py --backend sim --policy rules --episodes 2 --seed 7000

--backend sim plays ARC_Game's surrogate (no Unity build, no GPU). --policy rules plays whole games
with ARC_Game's rules baseline through the trainer's path and checks that the rewards sum to the
final score, so the adapter, the reward and the episode boundaries are checked end to end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf  # type: ignore

from verl.envs.env import get_balrog_env

TURNS = [
    [("hire", json.dumps({"kind": "untrained", "count": 2}))],
    [],                                                            # an empty turn
    [("build", json.dumps({"type": "shelter", "site_id": 9999}))],  # refused/invalid site
    [("no_such_tool", "{}"), ("train", "{not json")],              # malformed
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9970)
    ap.add_argument("--steps", type=int, default=len(TURNS))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--backend", choices=("unity", "sim"), default="unity")
    ap.add_argument("--policy", choices=("scripted", "rules"), default="scripted",
                    help="scripted: the TURNS above; rules: whole games with ARC_Game's rules baseline")
    ap.add_argument("--episodes", type=int, default=1, help="--policy rules: games to play")
    ap.add_argument("--seed", type=int, default=None,
                    help="CoraEnvConfig.seed: every game plays this scenario (default: a fresh one per game, as in training)")
    # Defaults match train_arc_game_qwen3_4B_tools_v3.sbatch.
    ap.add_argument("--format-penalty", type=float, default=0.05)
    ap.add_argument("--semantic-penalty", type=float, default=0.05)
    ap.add_argument("--format-bonus", type=float, default=0.10)
    a = ap.parse_args()
    if not os.environ.get("ARC_GAME_PATH"):
        sys.exit("Set ARC_GAME_PATH to the ARC_Game_New directory (the one containing rl/ and cora/).")
    # The trainer's own envs defaults (ppo_trainer.yaml), with this run's overrides.
    trainer = OmegaConf.load(_REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml")
    config = OmegaConf.create({"envs": OmegaConf.to_container(trainer.envs)})
    config.envs.merge_with({"env_name": "arc_game", "task": "ARC disaster response", "num_envs": 1,
                            "captioner": {"type": "cora"}})
    config.envs.arc_game_kwargs.merge_with({"unity_port": a.port, "max_steps": a.max_steps,
                                            "format_penalty": a.format_penalty, "semantic_penalty": a.semantic_penalty,
                                            "format_bonus": a.format_bonus,
                                            "backend": a.backend, "seed": a.seed})
    env = get_balrog_env(config)
    if a.policy == "rules":
        try:
            return play_rules(env, a)
        finally:
            env.close()
    try:
        messages, info = env.reset()
        print(f"[smoke] reset: prompt_sha={info.get('prompt_sha')} scenario={info.get('scenario')}")
        print(f"[smoke] system prompt: {len(messages[0]['content'])} chars; "
              f"user message (first 400):\n{messages[1]['content'][:400]}\n")
        keys = sorted(info["metrics"])          # reset reports the same keys as every step
        for i in range(a.steps):
            calls = TURNS[i % len(TURNS)]
            action = {"raw_text": f"turn {i}", "content": "", "tool_calls": calls} if calls else f"turn {i}: no calls"
            messages, reward, term, trunc, info = env.step(action)
            m = info["metrics"]
            print(f"[smoke] step {i + 1}: calls={[(c['tool'], c['status']) for c in info['calls']]} "
                  f"malformed={info['malformed']} reward={reward:+.4f} shaping={m['behavior/shaping']:+.2f}")
            if sorted(m) != keys:
                print(f"[smoke] METRIC KEYS CHANGED: {set(keys) ^ set(m)}")
                return 1
            if term or trunc:
                print(f"[smoke] episode end: final_score={m['episode/final_score']:.4f} "
                      f"turns={m['episode/turns']:.0f}; auto-reset user message ready: "
                      f"{messages[1]['content'][:40]!r}")
                break
        print(f"[smoke] {len(keys)} metric keys, consistent across steps: {keys}")
    finally:
        env.close()
    return 0


def _cora(env):
    """The CoraVerlogEnv under Verlog's wrappers."""
    while not hasattr(env, "cora"):
        env = env.env
    return env


def play_rules(env, a) -> int:
    """Whole games with the rules baseline, as tool calls in tool_agent_loop's shape; the rewards
    (no shaping fires when every call executes) must sum to the game's final score."""
    from bench.baselines import POLICIES
    from bench.baselines.common import tool_calls
    from cora.env import DECISIONS
    adapter, ok = _cora(env), True
    for ep in range(a.episodes):
        messages, info = env.reset()
        scenario = (info.get("scenario") or {}).get("seed", "?")
        game = adapter.cora.game
        total, shaping, m = 0.0, 0.0, {}
        for i in range(a.max_steps):
            calls = [(n, json.dumps(args)) for n, args in tool_calls(game, POLICIES["rules"](game, i, DECISIONS))]
            messages, reward, term, trunc, info = env.step(
                {"raw_text": "", "content": "", "tool_calls": calls} if calls else "no calls")
            m = info["metrics"]
            total += reward; shaping += m["behavior/shaping"]
            if term or trunc:
                break
        final, ret = m["episode/final_score"], m["episode/score_return"]
        exact = abs((total - shaping) - final) < 1e-6 and abs(ret - final) < 1e-6
        ok &= exact and bool(term or trunc)
        print(f"[smoke] game {ep + 1}: scenario={scenario} turns={m['episode/turns']:.0f} "
              f"final_score={final:.4f} reward_sum={total - shaping:.4f} shaping={shaping:+.3f} "
              f"executed={m['episode/calls_executed']:.0f} refused={m['episode/calls_refused']:.0f} "
              f"invalid={m['episode/calls_invalid']:.0f} {'OK' if exact else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
