"""
Inference-only smoke test: drive the ARCGame Verlog env with Qwen2.5-0.5B.

Loads the model with transformers on MPS (Mac) or CUDA (cluster), prepends the
instruction prompt + obs, samples a response, parses an ACTION index, and
steps the env. No gradient updates, no Verlog trainer — just a sanity check
that the env wrapper produces prompts the model can act on.

Expects:
  - ARC headless build running on TCP port 9876.
  - HuggingFace cache populated (or network access on first run).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.envs.environments import make_env
from verl.envs.environments.env_wrapper import EnvWrapper


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_env(unity_port: int, max_steps: int):
    if not os.environ.get("ARC_GAME_PATH"):
        guess = (_REPO_ROOT.parent / "ARC_Game" / "ARC_Game_New").resolve()
        if guess.exists():
            os.environ["ARC_GAME_PATH"] = str(guess)

    config = OmegaConf.create(
        {
            "envs": {
                "env_name": "arc_game",
                "task": "ARC disaster response",
                "arc_game_kwargs": {
                    "unity_port": unity_port,
                    "max_episode_steps": max_steps,
                    "auto_start_unity": False,
                    "format_penalty": 0.5,
                    "binary_reward": False,
                },
            }
        }
    )
    base_env = make_env(config.envs.env_name, config.envs.task, config)
    return EnvWrapper(base_env, config.envs.env_name, config.envs.task)


def build_prompt(env, obs) -> str:
    instr = env.get_instruction_prompt(instructions="ARC disaster response")
    long_term = (obs or {}).get("text", {}).get("long_term_context", "")
    return f"{instr}\n\n{long_term}\n\nYour response:"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = _pick_device()
    print(f"[smoke] device={device}")

    print(f"[smoke] loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s")

    env = build_env(args.port, args.steps)
    obs, info = env.reset()
    print(f"[smoke] reset OK. valid_actions={info.get('valid_action_count')}, "
          f"satisfaction={info.get('satisfaction')}, budget={info.get('budget')}")

    total_reward = 0.0
    for step_idx in range(args.steps):
        prompt = build_prompt(env, obs)
        messages = [{"role": "user", "content": prompt}]
        templated = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(templated, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        full, executed, is_valid, ext_metrics = env.extract_action(response)
        obs, reward, terminated, truncated, info = env.step(executed, is_valid)
        total_reward += reward

        print(
            f"\n[smoke] step {step_idx+1}: "
            f"sat={info.get('satisfaction')} budget={info.get('budget')} "
            f"reward={reward:+.2f} is_valid={is_valid} executed={executed!r}"
        )
        print(f"[smoke]   response: {response.strip()[:200]!r}")
        if terminated or truncated:
            break

    print(f"\n[smoke] done. total_reward={total_reward:+.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
