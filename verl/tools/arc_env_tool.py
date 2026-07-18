# Copyright 2026 CORA project
#
# Licensed under the Apache License, Version 2.0 (the "License").

from verl.tools.base_tool import BaseTool


class ArcEnvTool(BaseTool):
    """Schema-only tool stub for the ARC gym env.

    Our ToolAgentLoop runs a gym loop where env.step() is the executor, not
    per-tool Python classes. This class exists so the YAML config can carry
    OpenAI-format function schemas that Qwen3's chat template renders into a
    Hermes <tool_call> system block. execute() is intentionally a no-op —
    tool calls are parsed by HermesToolParser and dispatched by the env
    wrapper (llm_agents_wrapper._step_from_tool_calls).
    """

    pass
