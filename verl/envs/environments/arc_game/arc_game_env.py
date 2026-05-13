"""ARCGame env factory for Verlog.

Connects to a Unity gym TCP server (GymServerManager) at a configurable port
and wraps `ARCGameGymEnv` with the BALROG-style language wrappers Verlog
expects (clean-language wrapper + LLM-agents wrapper).

`ARCGameGymEnv` lives outside the Verlog tree, in the ARC_Game repo. Point at
it with the env var `ARC_GAME_PATH` (absolute path to the directory containing
`arc_game_gym_env_tcp.py`). The factory falls back to a sibling
`../ARC_Game/ARC_Game_New` if the env var is unset, which matches the layout
used during local development on the author's machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from verl.envs.environments.arc_game import (
    ARCGameCleanLangWrapper,
    ARCGameLLMAgentsWrapper,
)


def _resolve_arc_game_path() -> Path:
    env_path = os.environ.get("ARC_GAME_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # Fallback: sibling repo layout used in local dev.
    repo_root = Path(__file__).resolve().parents[4]  # .../Verlog
    sibling = repo_root.parent / "ARC_Game" / "ARC_Game_New"
    return sibling.resolve()


def make_arc_game_env(env_name, task, config, render_mode: Optional[str] = None):
    arc_path = _resolve_arc_game_path()
    if not (arc_path / "arc_game_gym_env_tcp.py").exists():
        raise FileNotFoundError(
            f"ARCGameGymEnv not found at {arc_path}. Set ARC_GAME_PATH to the "
            f"directory containing arc_game_gym_env_tcp.py."
        )
    if str(arc_path) not in sys.path:
        sys.path.insert(0, str(arc_path))

    # Import lazily so that other Verlog envs aren't penalised when this module
    # is imported just for registry side-effects.
    from arc_game_gym_env_tcp import ARCGameGymEnv  # type: ignore

    arc_cfg = getattr(config.envs, "arc_game_kwargs", None) or {}
    if hasattr(arc_cfg, "to_container"):
        arc_cfg = arc_cfg.to_container(resolve=True)
    arc_cfg = dict(arc_cfg)

    unity_port = arc_cfg.pop("unity_port", 9876)
    max_episode_steps = arc_cfg.pop("max_episode_steps", 100)
    max_days = arc_cfg.pop("max_days", 30)
    auto_start_unity = arc_cfg.pop("auto_start_unity", False)
    unity_exe_path = arc_cfg.pop("unity_exe_path", None)
    connection_timeout = arc_cfg.pop("connection_timeout", 30.0)

    env = ARCGameGymEnv(
        unity_exe_path=unity_exe_path,
        unity_port=unity_port,
        max_days=max_days,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        auto_start_unity=auto_start_unity,
        connection_timeout=connection_timeout,
    )

    env = ARCGameCleanLangWrapper(env, **arc_cfg)
    env = ARCGameLLMAgentsWrapper(env, **arc_cfg)
    return env
