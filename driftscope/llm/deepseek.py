from __future__ import annotations

import json
import os
import time
from typing import Any

import certifi
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from driftscope.config.loader import DriftScopeConfig, load_default_config
from driftscope.llm.openrouter import _parse_json_payload


class DeepSeekStructuredLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_sec: int = 30,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = _normalize_deepseek_model(model)
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, max_retries)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)

    @classmethod
    def from_env(
        cls,
        *,
        config: DriftScopeConfig | None = None,
        model: str | None = None,
    ) -> "DeepSeekStructuredLLM":
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
        cfg = config or load_default_config()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return cls(
            api_key=api_key,
            model=model or os.getenv("DEEPSEEK_MODEL", cfg.llm.default_model),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout_sec=cfg.llm.timeout_sec,
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", str(cfg.llm.max_retries))),
            retry_backoff_sec=float(os.getenv("DEEPSEEK_RETRY_BACKOFF_SEC", "1.0")),
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel | dict[str, Any]:
        schema = response_model.model_json_schema()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Return valid JSON only. "
                    "The word JSON is intentionally present for DeepSeek JSON Output mode. "
                    f"The JSON must satisfy this schema: {json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            body = self._post_json(payload)
            response_payload = json.loads(body)
            content = response_payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            content_text = str(content)
            try:
                parsed = _parse_json_payload(content_text)
            except ValueError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                messages.append({"role": "assistant", "content": content_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Return the same information as valid JSON only. "
                            "Do not use markdown. Escape any double quotes inside string values."
                        ),
                    }
                )
                continue
            try:
                return response_model.model_validate(parsed)
            except ValidationError:
                return parsed
        raise last_error or ValueError("DeepSeek returned invalid JSON")

    def _post_json(self, payload: dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DriftScope/0.1",
            "Connection": "close",
        }
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                    verify=certifi.where(),
                )
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                status = exc.response.status_code if exc.response is not None else "unknown"
                raise RuntimeError(f"DeepSeek request failed with HTTP {status}: {details}") from exc
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_sec * attempt)
        raise RuntimeError(
            f"DeepSeek request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


def _normalize_deepseek_model(model: str) -> str:
    normalized = model.strip()
    aliases = {
        "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
        "deepseekv4flash": "deepseek-v4-flash",
        "v4flash": "deepseek-v4-flash",
        "deepseek-v4-flash": "deepseek-v4-flash",
    }
    return aliases.get(normalized.lower(), normalized)
