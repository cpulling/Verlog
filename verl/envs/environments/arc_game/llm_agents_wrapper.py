"""
ARCGame LLM-agents wrapper.

Sits on top of ARCGameCleanLangWrapper / ARCGameGymEnv. Two responsibilities:

  1. Parse the LLM's `ACTIONS: i,j,k` line into a CSV index string the inner
     env understands, and translate the inner env's step() result into the
     Verlog metric shape (info["metrics"], behavior/*, episode/*).

  2. Drive a fresh Unity process per RL episode. ARCGameGymEnv.reset() only
     reads game state — Unity owns the simulation, so between episodes we
     terminate the current Unity process and respawn it. This is the only
     mechanism that fully resets the singletons (GlobalClock, TaskSystem,
     SatisfactionAndBudget) that survive a SceneManager.LoadScene because
     of DontDestroyOnLoad.

The reward returned by the inner env is already `compute_score()`-based
(Satisfaction − CostEfficiency, see ARC_Game_New/arc_game_gym_env_tcp.py).
This wrapper passes it through unchanged so the Unity-side scoring stays
the single source of truth.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from contextlib import suppress

import gymnasium as gym


def _ensure_smoke_on_path() -> None:
    arc_path = os.environ.get("ARC_GAME_PATH")
    if arc_path and arc_path not in sys.path:
        sys.path.insert(0, arc_path)

_REASONING_RE = re.compile(r"REASONING:\s*(.+)")
_HAS_CMD_TAG_RE = re.compile(
    r"<\s*(build|hire|train|staff|task|deconstruct|transfer)\s*>",
    re.IGNORECASE,
)

_ACTION_TYPES = (
    "construction",
    "deconstruction",
    "worker",
    "worker_assignment",
    "resource_transfer",
)


def _actions_at(valid_actions, csv: str):
    out = []
    if not csv:
        return out
    for tok in str(csv).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok)
        except ValueError:
            continue
        if 0 <= idx < len(valid_actions or []):
            out.append(valid_actions[idx])
    return out


def _innermost(env):
    while hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _wait_port_free(host: str, port: int, timeout: float):
    """Block until nothing is bound to host:port (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.25)
        try:
            s.connect((host, port))
            s.close()
            time.sleep(0.2)
        except (ConnectionRefusedError, OSError):
            return True
    return False


def _wait_port_open(host: str, port: int, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.25)
    return False


class _EpisodeState:
    __slots__ = (
        "turns", "actions_requested", "exec_attempted", "exec_success",
        "invalid_prompt_turns", "invalid_game_turns", "loop_turns",
        "type_counts", "sat_at_start", "budget_at_start",
        "score_return", "score_at_start",
    )

    def __init__(self):
        self.turns = 0
        self.actions_requested = 0
        self.exec_attempted = 0
        self.exec_success = 0
        self.invalid_prompt_turns = 0
        self.invalid_game_turns = 0
        self.loop_turns = 0
        self.type_counts = {t: 0 for t in _ACTION_TYPES}
        self.sat_at_start = None
        self.budget_at_start = None
        self.score_return = 0.0
        self.score_at_start = 0.0


class ARCGameLLMAgentsWrapper(gym.Wrapper):
    """Final Verlog-facing wrapper.

    Config knobs (kwargs):
      restart_unity_each_episode (bool, default True): respawn Unity in reset().
      unity_startup_log_dir (str, optional): write per-Unity logs here.
      format_penalty (float, default 0.0): subtract per turn when the LLM's
        output fails validation. Validation fires on ANY of: no command tag;
        parser error (unknown build type, no such site, unknown task id,
        unstaffable building, ...); or a well-formed tag whose body specifies
        parameters that don't map to a legal current-turn action.
      semantic_penalty (float, default 0.0): subtract for each command that
        parsed successfully but Unity rejected at execution time
        (`exec_attempted - exec_success`). Additive to format_penalty.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.env = env
        self.restart_unity_each_episode = bool(
            kwargs.get("restart_unity_each_episode", True)
        )
        self.unity_startup_log_dir = kwargs.get(
            "unity_startup_log_dir",
            os.environ.get("ARC_GAME_UNITY_LOG_DIR"),
        )
        self.format_penalty = float(kwargs.get("format_penalty", 0.0))
        self.semantic_penalty = float(kwargs.get("semantic_penalty", 0.0))
        # Flat bonus per turn for HOLISTIC correctness: is_valid AND ≥1 command
        # extracted AND every command executed by Unity. Symmetric with penalties
        # (total format-negative capped at -format_bonus) so worst-case format
        # outcome = -R, best = +R. Prevents partial-credit reward hacking and
        # "silence is safe" collapse. No positive bonuses for partial compliance.
        self.format_bonus = float(kwargs.get("format_bonus", 0.0))
        # ARCGameGymEnv hardcodes a 60s per-request socket timeout, but the
        # Unity-side `HandleAdvanceTime` has a 60s safety cap of its own when
        # the round's running-edge is missed (sub-frame race with
        # captureDeltaTime). 60s vs 60s preempts Unity; bump to a generous
        # ceiling so Unity always wins the race.
        self.unity_request_timeout = float(kwargs.get("unity_request_timeout", 180.0))
        self._first_reset = True
        self._last_long_term: str | None = None
        self._ep = _EpisodeState()
        self._bump_socket_timeout()
        self._traj_log_path = os.environ.get("ARC_TRAJECTORY_LOG")
        self._last_raw_response: str = ""
        self._last_executed: str = ""
        self._last_is_valid: bool = False
        self._traj_turn_counter: int = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    # -- Unity lifecycle (touches the user's gym env via documented attrs)

    def _bump_socket_timeout(self):
        inner = _innermost(self.env)
        sock = getattr(inner, "sock", None)
        if sock is not None:
            with suppress(Exception):
                sock.settimeout(self.unity_request_timeout)

    def _restart_unity(self):
        """Kill+respawn the underlying Unity process the env owns.

        We rely on attributes ARCGameGymEnv already exposes:
          unity_exe_path, unity_port, auto_start_unity, unity_process, sock,
          connection_timeout
        No edits to the user's gym env file.
        """
        inner = _innermost(self.env)
        if not getattr(inner, "auto_start_unity", False):
            return  # caller is driving Unity by hand; don't touch it.
        exe = getattr(inner, "unity_exe_path", None)
        if not exe:
            return
        port = int(getattr(inner, "unity_port", 9876))

        # Close socket.
        sock = getattr(inner, "sock", None)
        if sock is not None:
            with suppress(Exception):
                sock.close()
            inner.sock = None
            inner.connected = False

        # Terminate previous Unity process if still alive. Kill the whole
        # process group (Popen was launched with start_new_session=True in
        # the fixed base env) so any child Unity helpers die too — otherwise
        # a restart-per-episode run leaks a Unity per episode.
        proc = getattr(inner, "unity_process", None)
        if proc is not None:
            pid = proc.pid
            try:
                import os as _os
                pgid = _os.getpgid(pid)
            except (ProcessLookupError, PermissionError):
                pgid = pid
            except Exception:
                pgid = pid
            with suppress(Exception):
                import os as _os
                import signal as _sig
                _os.killpg(pgid, _sig.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(Exception):
                    import os as _os
                    import signal as _sig
                    _os.killpg(pgid, _sig.SIGKILL)
            except Exception:
                pass
            inner.unity_process = None

        # Wait for the OS to release the port before respawning so the new
        # Unity's TcpListener doesn't EADDRINUSE.
        _wait_port_free("127.0.0.1", port, timeout=10.0)

        # Spawn fresh Unity. Reuse the inner env's unity_log_path when set
        # so restarts overwrite the same file as the initial boot; otherwise
        # fall back to our own log_dir.
        log_path = getattr(inner, "unity_log_path", None)
        if not log_path and self.unity_startup_log_dir:
            os.makedirs(self.unity_startup_log_dir, exist_ok=True)
            log_path = os.path.join(self.unity_startup_log_dir, f"unity_{port}.log")
        cmd = [
            str(exe), "-batchmode", "-nographics",
            "-gym-server", "-gym-port", str(port),
            "-logFile", log_path if log_path else "-",
        ]
        inner.unity_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Register the respawn in the base env's orphan-sweep registry so
        # the module-level atexit hook reaps it if we die uncleanly.
        with suppress(Exception):
            from arc_game_gym_env_tcp import _registry_append  # type: ignore
            _registry_append(inner.unity_process.pid, port)

        # Wait for the new gym server to start listening, then connect.
        if not _wait_port_open(
            "127.0.0.1", port, timeout=float(getattr(inner, "connection_timeout", 30.0))
        ):
            raise ConnectionError(
                f"Unity gym server did not open port {port} after restart"
            )
        inner._connect_socket()
        self._bump_socket_timeout()

    def _reset_game_via_rpc(self):
        """Send `reset_game` to Unity so it re-initializes singletons in-process,
        avoiding a full Unity restart. Requires the HandleResetGame handler in
        the Unity build (added Jul 2026)."""
        inner = _innermost(self.env)
        if inner.sock is None or not getattr(inner, "connected", False):
            return
        response = inner._send_request({"type": "reset_game"})
        rtype = response.get("type", "")
        if rtype in ("error",) or (rtype and "error" in rtype.lower()):
            raise RuntimeError(
                f"reset_game failed: {response.get('message') or response}"
            )

    # -- gym API --------------------------------------------------------

    def reset(self, **kwargs):
        if not self._first_reset:
            if self.restart_unity_each_episode:
                self._restart_unity()
            else:
                self._reset_game_via_rpc()
        self._first_reset = False
        self._last_long_term = None
        self._ep = _EpisodeState()

        obs, info = self.env.reset(**kwargs)
        self._ep.sat_at_start = float(info.get("satisfaction", 50.0) or 0.0)
        self._ep.budget_at_start = float(info.get("budget", 10000.0) or 0.0)
        # compute_score baseline from the inner env's bookkeeping.
        inner = _innermost(self.env)
        self._ep.score_at_start = float(getattr(inner, "previous_score", 0.0) or 0.0)

        info["metrics"] = self._metrics(info, terminal=False)
        return obs, info

    def step(self, action, is_valid: bool = True):
        ep = self._ep
        pre_valid_actions = list(getattr(self.env, "valid_actions", []) or [])
        chosen_list = _actions_at(pre_valid_actions, action)

        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_step_sec = time.perf_counter() - t0

        # Loop detection via long-term context string.
        new_long_term = (obs.get("text") or {}).get("long_term_context")
        is_loop = (
            self._last_long_term is not None and new_long_term == self._last_long_term
        )
        self._last_long_term = new_long_term

        exec_results = info.get("execution_results", []) or []
        exec_count = len(exec_results)
        exec_success = sum(
            1 for r in exec_results if isinstance(r, dict) and r.get("success")
        )
        game_valid = 1.0 if exec_count and exec_success == exec_count else 0.0

        # Per-type accounting on the successful prefix (Unity stops on first
        # failure inside its execute_action loop).
        successful_prefix = chosen_list[:exec_success] if exec_count else []
        for chosen in successful_prefix:
            a_type = chosen.get("action_type") or chosen.get("actionType") or ""
            if a_type in ep.type_counts:
                ep.type_counts[a_type] += 1

        # Bookkeeping.
        ep.turns += 1
        ep.actions_requested += len(chosen_list)
        ep.exec_attempted += exec_count
        ep.exec_success += exec_success
        if not is_valid:
            ep.invalid_prompt_turns += 1
        if exec_count and exec_success < exec_count:
            ep.invalid_game_turns += 1
        if is_loop:
            ep.loop_turns += 1
        ep.score_return += float(reward)

        # Reward = Unity's compute_score delta + format shaping.
        # Two-mode format shaping:
        #   (a) holistic-correct turn (is_valid AND exec_count>0 AND all commands
        #       succeeded)  →  +format_bonus  (flat, no partial credit)
        #   (b) otherwise: attribute per-violation penalties (flat format_penalty
        #       for parser reject; per-rejected-command semantic_penalty). Total
        #       format-negative capped at -format_bonus so silence is not safe.
        reward = float(reward)
        rejected = exec_count - exec_success if exec_count else 0
        holistic_ok = bool(is_valid) and exec_count > 0 and exec_success == exec_count
        format_delta = 0.0
        if holistic_ok and self.format_bonus > 0:
            format_delta += self.format_bonus
        elif not holistic_ok:
            if not is_valid and self.format_penalty:
                format_delta -= self.format_penalty
            if rejected > 0 and self.semantic_penalty:
                format_delta -= self.semantic_penalty * float(rejected)
            if self.format_bonus > 0 and format_delta < -self.format_bonus:
                format_delta = -self.format_bonus
        reward += format_delta

        # Build metrics, merge in behavior/* from extract_action and game/*
        # scalars emitted by the inner ARC env (info["metrics"] in
        # arc_game_gym_env_tcp.py).
        metrics = self._metrics(info, terminal=bool(terminated or truncated))
        prior = info.get("metrics", {}) or {}
        for k, v in prior.items():
            if k.startswith("behavior/") or k.startswith("game/"):
                metrics[k] = v
        metrics["behavior/loop_rate"] = 1.0 if is_loop else 0.0
        metrics["behavior/valid_action_prompt"] = 1.0 if is_valid else 0.0
        metrics["behavior/valid_action_game"] = game_valid
        metrics["arc/snap/env_step_sec"] = float(env_step_sec)
        info["metrics"] = metrics

        if self._traj_log_path:
            with suppress(Exception):
                self._traj_turn_counter += 1
                port_id = int(getattr(_innermost(self.env), "unity_port", 0))
                pre_ctx = self._last_long_term or ""
                record = {
                    "pid": os.getpid(),
                    "port": port_id,
                    "turn": self._traj_turn_counter,
                    "pre_step_context": pre_ctx,
                    "model_raw_response": self._last_raw_response,
                    "executed": self._last_executed,
                    "is_valid_format": self._last_is_valid,
                    "exec_attempted": exec_count,
                    "exec_success": exec_success,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "satisfaction": float(info.get("satisfaction", 0.0) or 0.0),
                    "budget": float(info.get("budget", 0.0) or 0.0),
                    "score": float(info.get("score", 0.0) or 0.0),
                    "post_step_context": new_long_term or "",
                }
                with open(self._traj_log_path, "a") as _f:
                    import json as _json
                    _f.write(_json.dumps(record) + "\n")

        # Auto-reset on done: verl's tool_agent_loop.py only calls env.reset()
        # during validation, never during training rollouts. Without this, when a
        # game terminates the loop keeps stepping on a finished Unity → prior runs
        # showed 92% of "episodes" started on days 2-8 instead of day 1. Handle
        # the reset HERE so the OBSERVATION returned is the fresh day-1 state,
        # while keeping terminated/truncated True so the trainer's GAE properly
        # marks the episode boundary and bootstraps V(s') as 0.
        if bool(terminated) or bool(truncated):
            try:
                fresh_obs, _fresh_info = self.reset()
                obs = fresh_obs
            except Exception as e:
                print(
                    f"[llm_agents_wrapper] auto-reset after done=True failed: {e}",
                    flush=True,
                )

        return obs, reward, terminated, truncated, info

    # -- metric scaffold (always-same keys for verl's protocol concat) ----

    def _metrics(self, info: dict, terminal: bool) -> dict:
        ep = self._ep
        sat = float(info.get("satisfaction", 0.0) or 0.0)
        budget = float(info.get("budget", 0.0) or 0.0)
        score = float(info.get("score", 0.0) or 0.0)
        sat_score = float(info.get("satisfaction_score", 0.0) or 0.0)
        cost_eff = float(info.get("cost_efficiency", 0.0) or 0.0)
        rm = info.get("reward_metrics") or {}
        done_flag = 1.0 if terminal else 0.0

        m = {
            # snapshots
            "arc/snap/satisfaction": sat,
            "arc/snap/budget": budget,
            "arc/snap/day": float(info.get("day", 0) or 0),
            "arc/snap/segment": float(info.get("segment", 0) or 0),
            "arc/snap/valid_action_count": float(info.get("valid_action_count", 0) or 0),
            "arc/snap/score": score,
            "arc/snap/satisfaction_score": sat_score,
            "arc/snap/cost_efficiency": cost_eff,
            # episode-running cumulatives (reset on env.reset())
            "arc/cum/turns": float(ep.turns),
            "arc/cum/actions_requested": float(ep.actions_requested),
            "arc/cum/exec_attempted": float(ep.exec_attempted),
            "arc/cum/exec_success": float(ep.exec_success),
            "arc/cum/invalid_prompt_turns": float(ep.invalid_prompt_turns),
            "arc/cum/invalid_game_turns": float(ep.invalid_game_turns),
            "arc/cum/loop_turns": float(ep.loop_turns),
            "arc/cum/score_return": float(ep.score_return),
            # Unity-side raw cumulatives from rewardMetrics (already running
            # totals per episode on Unity's side).
            "arc/cum/foodFulfilled": float(rm.get("foodFulfilled", 0) or 0),
            "arc/cum/foodResolved": float(rm.get("foodResolved", 0) or 0),
            "arc/cum/lodgingFulfilled": float(rm.get("lodgingFulfilled", 0) or 0),
            "arc/cum/lodgingResolved": float(rm.get("lodgingResolved", 0) or 0),
            "arc/cum/foodSpend": float(rm.get("foodSpend", 0) or 0),
            "arc/cum/lodgingSpend": float(rm.get("lodgingSpend", 0) or 0),
            "arc/cum/workerSpend": float(rm.get("workerSpend", 0) or 0),
            "arc/cum/cumWorkingWorkers": float(rm.get("cumWorkingWorkers", 0) or 0),
            "arc/cum/cumTrainingWorkers": float(rm.get("cumTrainingWorkers", 0) or 0),
            "arc/cum/cumIdleWorkers": float(rm.get("cumIdleWorkers", 0) or 0),
        }
        for t in _ACTION_TYPES:
            m[f"arc/cum/action_type/{t}"] = float(ep.type_counts[t])

        # episode finals (non-zero only on terminal turn; recover means by
        # dividing by episode/done in WandB).
        m["episode/done"] = done_flag
        m["episode/final_satisfaction"] = sat * done_flag
        m["episode/final_budget"] = budget * done_flag
        m["episode/final_day"] = float(info.get("day", 0) or 0) * done_flag
        m["episode/final_score"] = score * done_flag
        m["episode/final_satisfaction_score"] = sat_score * done_flag
        m["episode/final_cost_efficiency"] = cost_eff * done_flag
        m["episode/turns_taken"] = float(ep.turns) * done_flag
        m["episode/score_return"] = float(ep.score_return) * done_flag
        m["episode/satisfaction_delta_total"] = (
            sat - (ep.sat_at_start or 0.0)
        ) * done_flag
        m["episode/total_actions_requested"] = float(ep.actions_requested) * done_flag
        m["episode/total_exec_attempted"] = float(ep.exec_attempted) * done_flag
        m["episode/total_exec_success"] = float(ep.exec_success) * done_flag
        m["episode/total_invalid_prompt_turns"] = float(ep.invalid_prompt_turns) * done_flag
        m["episode/total_invalid_game_turns"] = float(ep.invalid_game_turns) * done_flag
        for t in _ACTION_TYPES:
            m[f"episode/total_action_type/{t}"] = float(ep.type_counts[t]) * done_flag

        # behavior/* always-present slots so list_of_dict_to_dict_of_list
        # in verl/protocol.py sees consistent keys across all (turn, env) pairs.
        m.setdefault("behavior/valid_action_prompt", 0.0)
        m.setdefault("behavior/valid_action_game", 0.0)
        m.setdefault("behavior/loop_rate", 0.0)
        m.setdefault("behavior/valid_action_ratio", 0.0)
        m.setdefault("behavior/plan_length", 0.0)
        m.setdefault("behavior/backtrack_length", 0.0)
        m.setdefault("behavior/has_tag", 0.0)
        m.setdefault("behavior/empty_plan", 0.0)
        m.setdefault("behavior/cmd_parse_errors", 0.0)
        m.setdefault("behavior/cmd_tags_parsed", 0.0)
        m.setdefault("arc/snap/env_step_sec", 0.0)
        return m

    # -- LLM output parsing --------------------------------------------

    def extract_action(self, action: str):
        """Parse the LLM's cmd-tag response and prep the base env for step().

        Uses the benchmark's own `parse_commands` (llm_smoke_test) so the RL
        agent trains against the same action surface as the `minimal_cmd_v3`
        benchmark cell. `parse_commands` returns:
            {actions: [idx...], choices: [{taskId,choiceId}...],
             parsed: [...], errors: [...]}
        Task choices are submitted directly via the base env's
        `select_task_choice` (they aren't part of the action CSV); the
        returned CSV covers everything env.step() will execute.
        """
        _ensure_smoke_on_path()
        from llm_smoke_test import parse_commands  # type: ignore

        full_action = str(action)
        inner = _innermost(self.env)

        with suppress(Exception):
            pc = parse_commands(full_action, inner)
        pc = pc if isinstance(pc, dict) else {"actions": [], "choices": [],
                                              "parsed": [], "errors": []}
        actions = pc.get("actions") or []
        choices = pc.get("choices") or []
        parsed = pc.get("parsed") or []
        errors = list(pc.get("errors") or [])

        # `parse_commands` semantically validates build/hire/staff/deconstruct/
        # transfer against the round's enumerated actions (unknown types, sites,
        # buildings, routes → parser errors). It does NOT validate <task> ids or
        # their choices — enforce that here so a hallucinated `<task>12,2</task>`
        # is treated as invalid rather than silently dropped by Unity.
        active_tasks = {}
        with suppress(Exception):
            for t in (inner.game_state.get("allActiveTasks") or []):
                tid = int(t.get("taskId"))
                active_tasks[tid] = {
                    int(c.get("choiceId")) for c in (t.get("choices") or [])
                    if c.get("choiceId") is not None
                }
        validated_choices = []
        for c in choices:
            try:
                tid, cid = int(c["taskId"]), int(c["choiceId"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"task: malformed choice {c!r}")
                continue
            if tid not in active_tasks:
                errors.append(f"task: no active task with id {tid}")
                continue
            if cid not in active_tasks[tid]:
                errors.append(f"task: task {tid} has no choice {cid}")
                continue
            validated_choices.append({"taskId": tid, "choiceId": cid})
        choices = validated_choices

        # A response is well-formed iff it (a) contains at least one command
        # tag, (b) had no parser or param-validation errors, AND (c) actually
        # yielded at least one executable action or valid task choice. Empty
        # output, pure prose, or well-formed nonsense all fail.
        has_tag = bool(_HAS_CMD_TAG_RE.search(full_action))
        is_valid = bool(has_tag and not errors and (actions or choices))

        # Commit task choices to Unity before env.step (which advances time).
        # select_task_choice is idempotent per (taskId, choiceId).
        for c in choices:
            with suppress(Exception):
                inner.select_task_choice(int(c["taskId"]), int(c["choiceId"]))

        executed = ",".join(str(int(i)) for i in actions)

        low = full_action.lower()
        metrics = {
            "behavior/valid_action_ratio": 1.0 if is_valid else 0.0,
            "behavior/plan_length": float(len(actions) + len(choices)),
            "behavior/backtrack_length": float(sum(low.count(w) for w in (
                "however", "different", "but", "wait", "won't", "can't",
                "cannot", "another"
            ))),
            "behavior/cmd_parse_errors": float(len(errors)),
            "behavior/cmd_tags_parsed": float(len(parsed)),
            "behavior/has_tag": 1.0 if has_tag else 0.0,
            "behavior/empty_plan": 1.0 if (has_tag and not errors and not (actions or choices)) else 0.0,
        }
        self._last_raw_response = full_action
        self._last_executed = executed
        self._last_is_valid = bool(is_valid)
        return full_action, executed, is_valid, metrics


