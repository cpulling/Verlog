#!/usr/bin/env python
"""Build the offline dataset the critic is pretrained on and replays during PPO (verl.utils.critic_replay).

The rows are turns of games across the whole range of play, weak to strong, each with its
Monte-Carlo return-to-go (score delta + the same reward shaping PPO sees):

  model    rollouts of earlier PPO runs (--rollouts, Verlog's <experiment>_rollouts dumps)
  search   the search agents' warm-up games in those same dumps
  weak     do nothing, answer every task with its first open choice, random legal turns
  mixed    a policy that changes mid-game, and rules or low-budget search with random turns mixed in

A turn is about 1 s of critic time and turns of one game are correlated, so a game gives only
--turns-per-game random turns. Prompts are built as ToolAgentLoop builds them; responses are the
Hermes text of tool_agent_loop._scripted_turn. Run with ARC_GAME_PATH and CORA_STUDY_RULES set.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

TOOL_CONFIG = "verl/envs/environments/arc_game/tool_config/arc_tools.yaml"
GENERATED = {            # policy: (group, games)
    "noop": ("weak", 30), "tasks_first": ("weak", 60), "random": ("weak", 150),
    "rules_then_random": ("mixed", 80), "random_then_rules": ("mixed", 80),
    "rules_eps": ("mixed", 80), "mcts30_eps": ("mixed", 40),
}
USER = re.compile(r"\nuser\n(State:.*?Act by calling the tools\.)\nassistant", re.S)


def render(calls):
    return "\n".join('<tool_call>\n{"name": %s, "arguments": %s}\n</tool_call>' % (json.dumps(n), json.dumps(a))
                     for n, a in calls)


def first_open_choices(user):
    calls, tid = [], None
    for ln in user.splitlines():
        m = re.match(r"\s+\[([^\]]+)\]\s+\w+\s+\"", ln)
        if m:
            tid = m.group(1)
            continue
        c = re.match(r"\s{4}(\d+):\s*(.*)", ln)
        if c and tid and "UNAVAILABLE" not in c.group(2):
            calls.append(("task", {"task_id": tid, "choice_id": int(c.group(1))}))
            tid = None
    return calls


def play_game(args):
    """One generated game: [{user, response, reward}] and its final score."""
    policy, seed = args
    from oracle.search import Agent, _random_calls
    from rl import CoraEnv, CoraEnvConfig
    from verl.envs.environments.arc_game.cora_adapter import CoraVerlogEnv
    rr = random.Random(seed)
    env = CoraVerlogEnv(CoraEnv(CoraEnvConfig(backend="sim", prompt=os.environ["CRITIC_PROMPT_PACK"], max_steps=29)),
                        format_penalty=0.05, semantic_penalty=0.0, format_bonus=0.0, empty_penalty=0.0)
    obs, _ = env.reset()
    rules = Agent("rules", 0, "fair", seed=seed)
    mcts = Agent("mcts", 30, "fair", seed=seed)
    switch = rr.randrange(3, 26)
    rand = lambda g: [(t, dict(a)) for t, a in _random_calls(g, rr)]
    turns, turn, info = [], 0, {}
    while True:
        user = obs["text"]["long_term_context"]
        game = env.cora.game
        if policy == "noop":
            calls = []
        elif policy == "tasks_first":
            calls = first_open_choices(user)
        elif policy == "random":
            calls = rand(game)
        elif policy == "rules_then_random":
            calls = rules.calls(game, turn) if turn < switch else rand(game)
        elif policy == "random_then_rules":
            calls = rand(game) if turn < switch else rules.calls(game, turn)
        elif policy == "rules_eps":
            calls = rand(game) if rr.random() < 0.3 else rules.calls(game, turn)
        elif policy == "mcts30_eps":
            calls = rand(game) if rr.random() < 0.3 else mcts.calls(game, turn)
        else:
            raise ValueError(policy)
        text = render(calls)
        _, executed, _, _ = env.extract_action({"raw_text": text, "content": "", "tool_calls": [(n, json.dumps(a)) for n, a in calls]})
        obs, reward, term, trunc, info = env.step(executed)
        turns.append({"user": user, "response": text, "reward": float(reward)})
        turn += 1
        if term or trunc:
            break
    score = float(env.cora.game.game_state and info.get("score", 0.0) or 0.0)
    env.cora.close()
    return policy, turns, score


def init_worker(arc_path):
    sys.path.insert(0, arc_path)
    sys.path.insert(0, os.getcwd())


def from_dumps(rollout_dir, steps):
    """The games of a Verlog rollout dump: 30 rows each, the last one the bootstrap row."""
    games = []
    for step in steps:
        path = Path(rollout_dir) / f"{step}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path)]
        for i in range(0, len(rows), 30):
            turns = []
            for r in rows[i:i + 29]:
                m = USER.search(r["input"])
                if m is None:
                    raise ValueError(f"no user turn in {path} row {i}")
                turns.append({"user": m.group(1), "response": r["output"], "reward": float(r["score"])})
            games.append(turns)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rollouts", nargs="*", default=[], help="dir:group:first:last, group = model | search")
    ap.add_argument("--max-dump-games", type=int, default=300, help="per group, evenly over the dumps")
    ap.add_argument("--turns-per-game", type=int, default=7)
    ap.add_argument("--holdout", type=float, default=0.05, help="share of games kept out of training")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the generated games")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--prompt-pack", default="", help="ARC_Game prompt pack (default: cora.prompts.DEFAULT_PACK, the benchmark's)")
    ap.add_argument("--seed", type=int, default=1106)
    a = ap.parse_args()
    arc_path = os.environ["ARC_GAME_PATH"]
    init_worker(arc_path)
    from cora.prompts import DEFAULT_PACK
    from rl import CoraEnv, CoraEnvConfig
    from verl.tools.utils.tool_registry import initialize_tools_from_config
    a.prompt_pack = a.prompt_pack or DEFAULT_PACK
    os.environ["CRITIC_PROMPT_PACK"] = a.prompt_pack        # the workers' envs (spawned after this)
    cora = CoraEnv(CoraEnvConfig(backend="sim", prompt=a.prompt_pack))
    system = cora.system_prompt
    tools = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in initialize_tools_from_config(TOOL_CONFIG)]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(a.seed)

    games = []                         # (group, policy, turns)
    by_group = collections.defaultdict(list)
    for spec in a.rollouts:
        path, group, first, last = spec.rsplit(":", 3)
        by_group[group] += [(Path(path).name, g) for g in from_dumps(path, range(int(first), int(last) + 1))]
    for group, gs in by_group.items():
        rng.shuffle(gs)
        games += [(group, name, g) for name, g in gs[:a.max_dump_games]]
    jobs = [(p, rng.randrange(10_000, 2**31 - 1)) for p, (_, n) in GENERATED.items() for _ in range(max(1, int(n * a.scale)))]
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(arc_path,)) as pool:
        scores = collections.defaultdict(list)
        for policy, turns, score in pool.imap_unordered(play_game, jobs, chunksize=2):
            games.append((GENERATED[policy][0], policy, turns))
            scores[policy].append(sum(t["reward"] for t in turns))
    for p, v in sorted(scores.items()):
        print(f"[generate] {p}: {len(v)} games, mean return {np.mean(v):.3f} (min {min(v):.3f}, max {max(v):.3f})", flush=True)

    out = collections.defaultdict(list)
    for g, (group, policy, turns) in enumerate(games):
        returns = np.cumsum([t["reward"] for t in turns][::-1])[::-1]
        held = rng.random() < a.holdout
        for k in sorted(rng.sample(range(len(turns)), min(a.turns_per_game, len(turns)))):
            t = turns[k]
            prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t["user"]}],
                                             tools=tools, add_generation_prompt=True, tokenize=True, enable_thinking=False)
            response = tok.encode(t["response"], add_special_tokens=False) + [tok.eos_token_id]
            for key, v in (("prompt_ids", prompt), ("response_ids", response), ("ret", float(returns[k])), ("group", group),
                           ("policy", policy), ("game", g), ("turn", k), ("holdout", held)):
                out[key].append(v)
    torch.save(dict(out), a.out)
    # What the rows were rendered with: the trainer refuses a dataset whose prompt differs from the run's.
    with open(a.out + ".meta.json", "w") as fh:
        json.dump({"prompt_pack": a.prompt_pack, "prompt_sha": cora.prompt_sha, "tokenizer": a.tokenizer,
                   "rollouts": a.rollouts, "seed": a.seed}, fh, indent=1)
    c = collections.Counter(out["group"])
    print(f"[generate] {len(games)} games, {len(out['ret'])} rows {dict(c)}, return {np.mean(out['ret']):.3f} +- {np.std(out['ret']):.3f}, "
          f"{sum(out['holdout'])} held out -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
