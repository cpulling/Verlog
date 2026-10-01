"""ARCGame env factory for Verlog: one CoraEnv (ARC_Game repo, rl/cora_env.py) per Ray worker,
wrapped by cora_adapter.CoraVerlogEnv.

The ARC_Game repo is found via ARC_GAME_PATH (the directory containing rl/ and cora/), falling back
to a sibling ../ARC_Game/ARC_Game_New checkout. Each worker claims its own Unity port so concurrent
workers never share a game.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Optional

from verl.envs.environments.arc_game.cora_adapter import CoraVerlogEnv

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
    """config.envs.arc_game_kwargs: CoraEnvConfig fields (prompt, ablation, manual_transfers,
    obs_encoding, max_steps, seed, map_config, param_config, unity_exe, unity_log_dir) plus
    unity_port (the base port; workers take base..base+num_envs-1) and the reward-shaping knobs
    format_penalty / semantic_penalty / format_bonus."""
    arc_path = _resolve_arc_game_path()
    if not (arc_path / "rl" / "cora_env.py").exists():
        raise FileNotFoundError(f"CoraEnv not found under {arc_path}. Set ARC_GAME_PATH to the "
                                f"ARC_Game_New directory (the one containing rl/ and cora/).")
    if str(arc_path) not in sys.path:
        sys.path.insert(0, str(arc_path))
    # Imported lazily so other Verlog envs don't pay for it at registry import.
    from rl import CoraEnv, CoraEnvConfig  # type: ignore

    kw = getattr(config.envs, "arc_game_kwargs", None) or {}
    if hasattr(kw, "to_container"):
        kw = kw.to_container(resolve=True)
    kw = dict(kw)
    base_port = int(kw.pop("unity_port", 9876))
    num_envs = int(getattr(config.envs, "num_envs", 1) or 1)
    cache_key = (base_port, num_envs)
    if cache_key in _ENV_CACHE:
        return _ENV_CACHE[cache_key]
    port = _claim_port_slot(base_port, num_envs)
    shaping = {k: kw.pop(k) for k in ("format_penalty", "semantic_penalty", "format_bonus") if k in kw}
    log_dir = kw.pop("unity_log_dir", os.environ.get("ARC_GAME_UNITY_LOG_DIR"))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        kw["unity_log"] = os.path.join(log_dir, f"unity_{port}.log")
    env = CoraVerlogEnv(CoraEnv(CoraEnvConfig(port=port, **kw)), **shaping)
    _ENV_CACHE[cache_key] = env
    return env
