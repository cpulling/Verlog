# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # CORA: forward GUIDED_REGEX to workers. Ray does not auto-inherit driver
    # env vars, and the ARC agent_loop reads GUIDED_REGEX from os.environ
    # (env-var-based so regex metachars don't need Hydra-escaping).
    if os.environ.get("GUIDED_REGEX"):
        runtime_env["env_vars"]["GUIDED_REGEX"] = os.environ["GUIDED_REGEX"]
    # CORA: forward ARC_STABLE_TASK_TOKENS to workers. obs_encoder reads this
    # at import time to decide whether to render tasks with stable (BUDGET_DAILY,
    # FOOD_C01, ...) tokens instead of the drifting integer taskId.
    if os.environ.get("ARC_STABLE_TASK_TOKENS"):
        runtime_env["env_vars"]["ARC_STABLE_TASK_TOKENS"] = os.environ["ARC_STABLE_TASK_TOKENS"]
    # CORA: forward ARC_ACTION_MODE to workers. arc_game.__init__.get_instruction_prompt
    # reads this per-call to choose between the XML-tag system prompt (default) and the
    # tool-call system prompt (when set to "tool_calls").
    if os.environ.get("ARC_ACTION_MODE"):
        runtime_env["env_vars"]["ARC_ACTION_MODE"] = os.environ["ARC_ACTION_MODE"]
    return runtime_env
