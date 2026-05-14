from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any, get_args, get_origin

import certifi
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from driftscope.config.loader import DriftScopeConfig, load_default_config

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class OpenRouterStructuredLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "DriftScope",
        site_url: str = "http://localhost",
        timeout_sec: int = 30,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = _normalize_model_name(model)
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.site_url = site_url
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, max_retries)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)

    @classmethod
    def from_env(
        cls,
        *,
        config: DriftScopeConfig | None = None,
        model: str | None = None,
    ) -> "OpenRouterStructuredLLM":
        load_dotenv()
        cfg = config or load_default_config()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return cls(
            api_key=api_key,
            model=_normalize_model_name(model or os.getenv("OPENROUTER_MODEL", cfg.llm.default_model)),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            app_name=os.getenv("OPENROUTER_APP_NAME", "DriftScope"),
            site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            timeout_sec=cfg.llm.timeout_sec,
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", str(cfg.llm.max_retries))),
            retry_backoff_sec=float(os.getenv("OPENROUTER_RETRY_BACKOFF_SEC", "1.0")),
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
                "content": f"{system_prompt}\n\nReturn valid JSON only.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": _json_schema_response_format(response_model, schema),
            "messages": messages,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            body = self._post_json(payload)
            response_payload = json.loads(body)
            content_text = _message_content_text(response_payload)
            try:
                parsed = _parse_json_payload(content_text)
            except ValueError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": content_text,
                    }
                )
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
            except ValidationError as exc:
                last_error = exc
                coerced = _coerce_single_required_answer(parsed, response_model)
                if coerced is not None:
                    try:
                        return response_model.model_validate(coerced)
                    except ValidationError as coerced_exc:
                        last_error = coerced_exc
                if attempt >= self.max_retries:
                    return parsed
                messages.append(
                    {
                        "role": "assistant",
                        "content": content_text,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was valid JSON but did not match the required output shape. "
                            f"{_compact_model_shape(response_model)} "
                            "Return JSON only. Do not add task-specific top-level keys."
                        ),
                    }
                )
                continue
        raise last_error or ValueError("OpenRouter returned invalid JSON")

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
        }
        body = self._post_json(payload)
        response_payload = json.loads(body)
        return _message_content_text(response_payload).strip()

    def _post_json(self, payload: dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "User-Agent": f"{self.app_name}/0.1",
            "Connection": "close",
        }
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        active_payload = payload
        used_relaxed_payload = False
        attempt = 1
        while attempt <= self.max_retries:
            try:
                response = requests.post(
                    url,
                    json=active_payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                    verify=certifi.where(),
                )
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                status = exc.response.status_code if exc.response is not None else None
                if (
                    not used_relaxed_payload
                    and _looks_like_parameter_routing_error(status, details)
                ):
                    active_payload = _relaxed_openrouter_payload(payload)
                    used_relaxed_payload = True
                    continue
                # Transient upstream errors (Cloudflare 502, OpenRouter 503/504,
                # rate-limit 429) are retryable with backoff. Without retrying,
                # one flaky request kills an entire 16-turn batch and the LLM's
                # work is silently lost.
                if status in {429, 502, 503, 504} and attempt < self.max_retries:
                    last_error = exc
                    time.sleep(self.retry_backoff_sec * attempt)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"OpenRouter request failed with HTTP {status if status is not None else 'unknown'}: {details[:300]}"
                ) from exc
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_sec * attempt)
                attempt += 1
                continue
            attempt += 1
        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if not stripped:
        raise ValueError("OpenRouter returned empty content")
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        stripped = match.group(1).strip()
    candidates = [stripped]
    extracted = _extract_json_candidate(stripped)
    if extracted and extracted != stripped:
        candidates.append(extracted)

    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        repaired = _repair_json_candidate(candidate)
        if repaired and repaired not in seen:
            seen.add(repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                last_error = exc

    snippet = stripped[:240].replace("\n", "\\n")
    raise ValueError(f"OpenRouter returned invalid JSON: {last_error}; content starts with: {snippet}")


def _message_content_text(response_payload: dict[str, Any]) -> str:
    content = response_payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _json_schema_response_format(
    response_model: type[BaseModel],
    schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _schema_name(response_model),
            "strict": True,
            "schema": _make_schema_openai_strict(schema),
        },
    }


def _make_schema_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Patch a Pydantic-generated JSON schema for OpenAI strict mode.

    Under ``strict: True``, OpenAI requires every property to appear in the
    ``required`` array and every object to set ``additionalProperties: false``.
    Pydantic's auto-generated schema lists only fields without defaults in
    ``required``, which causes a 400 when the request routes to OpenAI/Azure.
    We patch the schema in-memory (deep copy so we don't mutate Pydantic's
    cached schema) by walking every nested object and forcing the constraint.

    Optional fields keep their nullability via the ``anyOf: [{type: ...},
    {type: "null"}]`` union Pydantic already emits — strict mode accepts that
    so long as the field is listed in ``required``.
    """
    patched = copy.deepcopy(schema)
    _enforce_strict_object(patched)
    return patched


def _enforce_strict_object(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False
        # OpenAI strict mode forbids `default` on properties even when the
        # property is listed in `required`. Pydantic emits `default: null`
        # for `Optional[X] = None` fields; drop it to satisfy the validator.
        node.pop("default", None)
        for key in ("properties", "$defs", "definitions"):
            sub = node.get(key)
            if isinstance(sub, dict):
                for value in sub.values():
                    _enforce_strict_object(value)
        for key in ("items", "additionalItems"):
            if key in node:
                _enforce_strict_object(node[key])
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            sub = node.get(key)
            if isinstance(sub, list):
                for value in sub:
                    _enforce_strict_object(value)
    elif isinstance(node, list):
        for value in node:
            _enforce_strict_object(value)


def _schema_name(response_model: type[BaseModel]) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", response_model.__name__).strip("_")
    return name or "StructuredResponse"


def _compact_model_shape(response_model: type[BaseModel]) -> str:
    fields = response_model.model_fields
    parts = []
    for name, field in fields.items():
        required = "required" if field.is_required() else "optional"
        parts.append(f"{name} ({_compact_type_name(field.annotation)}, {required})")
    return "Use exactly these top-level keys: " + "; ".join(parts) + "."


def _compact_type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return "array"
    if args and type(None) in args:
        non_null = [arg for arg in args if arg is not type(None)]
        if len(non_null) == 1:
            return f"{_compact_type_name(non_null[0])} or null"
        return "value or null"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    return "value"


def _coerce_single_required_answer(
    parsed: dict[str, Any],
    response_model: type[BaseModel],
) -> dict[str, Any] | None:
    fields = response_model.model_fields
    required = [name for name, field in fields.items() if field.is_required()]
    if (
        required != ["answer"]
        or set(fields) != {"answer"}
        or "answer" in parsed
        or len(parsed) != 1
    ):
        return None
    value = next(iter(parsed.values()))
    if isinstance(value, (dict, list)):
        answer = json.dumps(value, ensure_ascii=False)
    else:
        answer = str(value)
    return {"answer": answer}


def _looks_like_parameter_routing_error(status: int | None, details: str) -> bool:
    if status not in {400, 404}:
        return False
    lowered = details.lower()
    return (
        ("no endpoints found" in lowered and "requested parameters" in lowered)
        or "require_parameters" in lowered
        or ("invalid schema for response_format" in lowered)
        or ("response_format" in lowered and "not supported" in lowered)
    )


def _relaxed_openrouter_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fallback for OpenRouter routes that reject strict provider parameters.

    Some providers behind OpenRouter do not advertise support for strict JSON
    schema or `provider.require_parameters`. When routing fails before a model
    is selected, retry with broad JSON mode so the caller can still parse and
    validate the returned object locally.
    """
    relaxed = dict(payload)
    relaxed.pop("provider", None)
    relaxed.pop("reasoning", None)
    relaxed["response_format"] = {"type": "json_object"}
    return relaxed


def _extract_json_candidate(content: str) -> str | None:
    start_positions = [pos for pos in (content.find("{"), content.find("[")) if pos >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = content[start:].strip()

    balanced = _extract_balanced_json_prefix(candidate)
    if balanced:
        return balanced
    return candidate


def _extract_balanced_json_prefix(content: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    started = False

    for index, char in enumerate(content):
        if not started:
            if char.isspace():
                continue
            if char not in "{[":
                return None
            started = True
            stack.append("}" if char == "{" else "]")
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            stack.append("}")
            continue
        if char == "[":
            stack.append("]")
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return content[: index + 1]

    return None


def _repair_json_candidate(content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return None

    builder: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                builder.append(char)
                escaped = False
                continue
            if char == "\\":
                builder.append(char)
                escaped = True
                continue
            if char == '"':
                if _looks_like_string_terminator(stripped, index):
                    builder.append(char)
                    in_string = False
                else:
                    # OpenRouter occasionally emits bare quotes inside string values,
                    # e.g. `"candidate_content": "The musical "Hadestown" ..."`
                    # Repair those by escaping them instead of closing the string.
                    builder.append('\\"')
                continue
            if char in "\r\n":
                builder.append("\\n")
                continue
            builder.append(char)
            continue

        if char == '"':
            builder.append(char)
            in_string = True
            continue
        if char == "{":
            stack.append("}")
            builder.append(char)
            continue
        if char == "[":
            stack.append("]")
            builder.append(char)
            continue
        if char in "}]":
            if stack and char == stack[-1]:
                stack.pop()
                builder.append(char)
            continue
        builder.append(char)

    if in_string:
        if escaped:
            builder.append("\\")
        builder.append('"')

    repaired = "".join(builder)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    repaired = repaired.rstrip(", \n\r\t")
    if stack:
        repaired += "".join(reversed(stack))
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def _looks_like_string_terminator(content: str, index: int) -> bool:
    cursor = index + 1
    while cursor < len(content) and content[cursor].isspace():
        cursor += 1
    if cursor >= len(content):
        return True
    return content[cursor] in {",", "}", "]", ":"}


def _normalize_model_name(model: str) -> str:
    normalized = model.strip()
    aliases = {
        "gpt4omini": "openai/gpt-4o-mini",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt5nano": "openai/gpt-5-nano",
        "gpt-5-nano": "openai/gpt-5-nano",
        "gpt41nano": "openai/gpt-4.1-nano",
        "gpt-4.1-nano": "openai/gpt-4.1-nano",
        "deepseekv4flash": "deepseek/deepseek-v4-flash",
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
        "v4flash": "deepseek/deepseek-v4-flash",
    }
    return aliases.get(normalized.lower(), normalized)
