"""
Standalone smoke test for the ARCGame Verlog env wrapper.

Builds the env via `verl.envs.environments.make_env` to exercise the same
registration path the trainer uses, then resets and runs 5 dummy steps.

Expects:
  - A running ARC headless build with gym TCP on port 9876.
  - ARC_GAME_PATH set, OR the sibling `../ARC_Game/ARC_Game_New` layout used in
    local dev.

This script does NOT load a model and does NOT exercise Verlog's trainer; it's
a pure env-plumbing check meant to run on CPU/Mac before porting to a GPU
cluster.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make Verlog importable when this script is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf  # type: ignore

from verl.envs.environments import make_env
from verl.envs.environments.env_wrapper import EnvWrapper


def build_config():
    return OmegaConf.create(
        {
            "envs": {
                "env_name": "arc_game",
                "task": "ARC disaster response",
                "arc_game_kwargs": {
                    "unity_port": 9876,
                    "max_episode_steps": 5,
                    "max_days": 30,
                    "auto_start_unity": False,
                    "format_penalty": 0.5,
                    "binary_reward": False,
                },
            }
        }
    )


def main() -> int:
    if not os.environ.get("ARC_GAME_PATH"):
        guess = (_REPO_ROOT.parent / "ARC_Game" / "ARC_Game_New").resolve()
        if guess.exists():
            os.environ["ARC_GAME_PATH"] = str(guess)
            print(f"[smoke] ARC_GAME_PATH defaulted to {guess}")

    config = build_config()
    base_env = make_env(config.envs.env_name, config.envs.task, config)
    env = EnvWrapper(base_env, config.envs.env_name, config.envs.task)

    inst_prompt = env.get_instruction_prompt(instructions=config.envs.task)
    print(f"[smoke] instruction_prompt:\n{inst_prompt}\n")

    obs, info = env.reset()
    print(f"[smoke] reset OK. valid_actions={info.get('valid_action_count')}, "
          f"satisfaction={info.get('satisfaction')}, budget={info.get('budget')}")
    text_obs = (obs or {}).get("text", {})
    preview = (text_obs.get("long_term_context") or "")[:500]
    print(f"[smoke] obs preview (first 500 chars):\n{preview}\n...")

    total_reward = 0.0
    for step_idx in range(5):
        raw_llm_output = f"REASONING: dummy step {step_idx}\nACTION: 0"
        full, executed, is_valid, ext_metrics = env.extract_action(raw_llm_output)
        obs, reward, terminated, truncated, info = env.step(executed, is_valid)
        total_reward += reward
        print(
            f"[smoke] step {step_idx+1}: parsed={executed!r} (is_valid={is_valid}) "
            f"reward={reward:+.2f} satisfaction={info.get('satisfaction')} "
            f"loop_rate={info.get('metrics', {}).get('behavior/loop_rate')} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            break

    print(f"[smoke] done. total_reward={total_reward:+.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
