"""远端 episode 协议：GPU 只传配方参数，路径、代理和资源上限由远端服务持有。"""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

EPISODE_ENV = (
    "SETA_FULL_PASS_BONUS",
    "SETA_RECORD_TERMINAL_SESSION",
    "HARBOR_AGENT_MAX_ITERATIONS",
    "HARBOR_MAX_SEQ_LEN",
    "AGENT_MAX_INPUT_TOKENS",
    "AGENT_MAX_OUTPUT_TOKENS",
    "AGENT_MODEL_NAME",
    "AGENT_TIMEOUT",
    "AGENT_TRIAL_TIMEOUT",
    "HARBOR_TIMEOUT_MULTIPLIER",
    "HARBOR_VERIFIER_TIMEOUT_SEC",
    "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER",
    "HARBOR_RESPONSE_LENGTH_POLICY",
    "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE",
    "HARBOR_TERMINUS_2_LINEAR_HISTORY",
)


@dataclass(frozen=True)
class Endpoint:
    url: str
    max_concurrent: int
    token_file: str = ""

    def headers(self) -> dict[str, str]:
        if not self.token_file:
            return {}
        token = Path(self.token_file).read_text().strip()
        if not token:
            raise ValueError(f"远端服务 token 文件为空：{self.token_file}")
        return {"Authorization": f"Bearer {token}"}


def load_endpoints(path: str) -> tuple[Endpoint, ...]:
    data = json.loads(Path(path).read_text())
    endpoints = []
    for row in data["endpoints"]:
        url = row["url"].rstrip("/")
        parsed = urlsplit(url)
        limit = row["max_concurrent"]
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path:
            raise ValueError(f"远端服务 URL 必须是 HTTP(S) 根地址：{url}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("每个远端的 max_concurrent 必须是正整数")
        endpoints.append(Endpoint(url, limit, row.get("token_file", "")))
    if not endpoints or len({e.url for e in endpoints}) != len(endpoints):
        raise ValueError("至少配置一个远端，且 URL 不能重复")
    return tuple(endpoints)
