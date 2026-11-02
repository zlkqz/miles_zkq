"""Pydantic request/response models for the Miles agent server."""

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    base_url: str
    model: str
    sampling_params: dict[str, Any] = {}
    api_key: str = "dummy"

    instance_id: str = ""
    agent_name: str = "mini-swe-agent"
    max_seq_len: int | None = None
    session_server_id: str | None = None
    session_server_instance_id: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = {}
    eval_report: dict[str, Any] = {}
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlushRequest(BaseModel):
    session_server_id: str | None = None
    session_server_instance_id: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,100}$")


class FlushAllRequest(BaseModel):
    cancel_tasks: bool = True
    kill_containers: bool = True
    prune: bool = False
