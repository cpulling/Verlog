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
import socket
import sys
from pathlib import Path
from typing import Optional

from verl.envs.environments.arc_game import (
    ARCGameCleanLangWrapper,
    ARCGameLLMAgentsWrapper,
)

# Per-process env cache. Verlog's env-creator pattern calls make_arc_game_env
# more than once per Ray worker (probe + real); with the modulo-N port counter,
# the second call wraps back to the same slot and spawns a second Unity on the
# same port. Since the fixed Unity build enforces single-client, that path
# hangs the whole rollout. Memoising per process guarantees exactly one env,
# one Unity, one client per Ray worker. Keyed by (base_port, num_envs) so a
# worker configured with more envs than workers still gets distinct slots on
# repeat calls (the port counter differentiates by call ordinal within slot).
_ENV_CACHE: dict[tuple[int, int], object] = {}
_PROCESS_SLOT_CACHE: dict[tuple[int, int], int] = {}


def _resolve_arc_game_path() -> Path:
    env_path = os.environ.get("ARC_GAME_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # Fallback: sibling repo layout used in local dev.
    repo_root = Path(__file__).resolve().parents[4]  # .../Verlog
    sibling = repo_root.parent / "ARC_Game" / "ARC_Game_New"
    return sibling.resolve()


def _port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """Return True iff something is currently listening on host:port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except (ConnectionRefusedError, OSError):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _claim_port_slot(base_port: int, num_envs: int) -> int:
    """Atomically assign a unique port slot to this env instance.

    Verlog spawns one AgentLoopWorker per env across separate Ray actors;
    each worker hits this factory once and needs its own Unity instance.
    We coordinate via a file counter under /tmp so the assignment is
    cross-process safe without changing Verlog plumbing.

    Cached per-process: repeat calls from the same Ray worker return the
    same slot instead of advancing the counter, so probe + real factory
    calls converge on ONE port (and, with the env cache below, ONE Unity).
    """
    cached = _PROCESS_SLOT_CACHE.get((base_port, num_envs))
    if cached is not None:
        return cached
    if num_envs <= 1:
        _PROCESS_SLOT_CACHE[(base_port, num_envs)] = base_port
        return base_port
    import fcntl
    state_dir = Path(os.environ.get("ARC_GAME_PORT_STATE_DIR", "/tmp"))
    state_dir.mkdir(parents=True, exist_ok=True)
    state = state_dir / f"arc_game_port_counter_{base_port}_{num_envs}.txt"
    # Open-create-rw, fcntl flock for atomic read-modify-write.
    fd = os.open(state, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 64).decode() or "0"
        try:
            n = int(raw.strip())
        except ValueError:
            n = 0
        # Skip past ports already occupied on this host so distinct workers
        # never race for the same slot after a partial cleanup.
        for _ in range(num_envs):
            slot = n % num_envs
            n += 1
            if not _port_is_open("127.0.0.1", base_port + slot):
                break
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(n).encode())
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    port = base_port + slot
    _PROCESS_SLOT_CACHE[(base_port, num_envs)] = port
    return port


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

    base_port = arc_cfg.pop("unity_port", 9876)
    num_envs = int(getattr(config.envs, "num_envs", 1) or 1)

    cache_key = (base_port, num_envs)
    cached = _ENV_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unity_port = _claim_port_slot(base_port, num_envs)
    max_episode_steps = arc_cfg.pop("max_episode_steps", 100)
    max_days = arc_cfg.pop("max_days", 30)
    auto_start_unity = arc_cfg.pop("auto_start_unity", False)
    unity_exe_path = arc_cfg.pop("unity_exe_path", None)
    connection_timeout = arc_cfg.pop("connection_timeout", 30.0)

    # Per-Unity log file (one per worker port) when a log dir is configured.
    unity_log_path = None
    log_dir = arc_cfg.pop(
        "unity_log_dir",
        os.environ.get("ARC_GAME_UNITY_LOG_DIR"),
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        unity_log_path = os.path.join(log_dir, f"unity_{unity_port}.log")

    env = ARCGameGymEnv(
        unity_exe_path=unity_exe_path,
        unity_port=unity_port,
        max_days=max_days,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        auto_start_unity=auto_start_unity,
        connection_timeout=connection_timeout,
        unity_log_path=unity_log_path,
    )

    env = ARCGameCleanLangWrapper(env, **arc_cfg)
    env = ARCGameLLMAgentsWrapper(env, **arc_cfg)
    _ENV_CACHE[cache_key] = env
    return env
