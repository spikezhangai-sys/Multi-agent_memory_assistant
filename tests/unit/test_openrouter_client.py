import json

from pydantic import BaseModel
import requests

from driftscope.llm.openrouter import OpenRouterStructuredLLM, _parse_json_payload


class DemoResponse(BaseModel):
    answer: str


class DemoDecision(BaseModel):
    answer: str
    cited_memory_ids: list[str] = []
    context_only_ids: list[str] = []
    abstained: bool = False
    abstain_reason: str | None = None


class FakeResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class FakeHTTPErrorResponse:
    def __init__(self, *, status_code: int, payload: dict):
        self.text = json.dumps(payload)
        self.status_code = status_code

    def raise_for_status(self) -> None:
        raise requests.exceptions.HTTPError("request failed", response=self)


def test_openrouter_client_parses_json_response(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
    )

    def fake_post(*args, **kwargs):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": "ok"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert result.answer == "ok"


def test_openrouter_client_generates_raw_text(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
    )
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "25",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_text(
        system_prompt="Answer directly",
        user_prompt="How many?",
    )

    assert result == "25"
    assert "response_format" not in captured
    assert "provider" not in captured


def test_openrouter_client_requests_strict_json_schema(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="deepseekv4flash",
    )
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": "ok"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert result.answer == "ok"
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert "provider" not in captured
    assert "reasoning" not in captured
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"


def test_openrouter_client_relaxes_payload_when_strict_routing_fails(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
    )
    captured = []

    def fake_post(*args, **kwargs):
        captured.append(dict(kwargs["json"]))
        if len(captured) == 1:
            return FakeHTTPErrorResponse(
                status_code=404,
                payload={
                    "error": {
                        "message": "No endpoints found that can handle the requested parameters.",
                    }
                },
            )
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": "ok"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert result.answer == "ok"
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert "provider" not in captured[0]
    assert "reasoning" not in captured[0]
    assert captured[1]["response_format"] == {"type": "json_object"}
    assert "provider" not in captured[1]
    assert "reasoning" not in captured[1]


def test_openrouter_client_repairs_valid_json_with_wrong_shape(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
    )
    captured = []

    def fake_post(*args, **kwargs):
        captured.append(dict(kwargs["json"]))
        if len(captured) in {1, 3}:
            return FakeHTTPErrorResponse(
                status_code=404,
                payload={
                    "error": {
                        "message": "No endpoints found that can handle the requested parameters.",
                    }
                },
            )
        if len(captured) == 2:
            content = {"to_watch_list_count": 25}
        else:
            content = {
                "answer": "25",
                "cited_memory_ids": ["m1"],
                "context_only_ids": [],
                "abstained": False,
                "abstain_reason": None,
            }
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(content),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt='{"query":"How many?","evidence":[{"id":"m1","content":"25"}]}',
        response_model=DemoDecision,
    )

    assert isinstance(result, DemoDecision)
    assert result.answer == "25"
    assert result.cited_memory_ids == ["m1"]
    assert captured[2]["response_format"]["type"] == "json_schema"
    repair_prompt = captured[2]["messages"][-1]["content"]
    assert "answer (string, required)" in repair_prompt
    assert "cited_memory_ids (array, optional)" in repair_prompt


def test_openrouter_client_retries_transient_ssl_error(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
        max_retries=2,
        retry_backoff_sec=0,
    )
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.SSLError("EOF occurred in violation of protocol")
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": "ok"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert calls["count"] == 2
    assert result.answer == "ok"


def test_openrouter_client_returns_raw_payload_on_validation_mismatch(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
        max_retries=1,
    )

    def fake_post(*args, **kwargs):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"unexpected": "shape"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoDecision,
    )

    assert result == {"unexpected": "shape"}


def test_openrouter_client_retries_when_model_returns_invalid_json(monkeypatch) -> None:
    client = OpenRouterStructuredLLM(
        api_key="test-key",
        model="gpt4omini",
        max_retries=2,
    )
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"answer" "ok"}',
                            }
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": "ok"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("driftscope.llm.openrouter.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert calls["count"] == 2
    assert result.answer == "ok"


def test_parse_json_payload_repairs_truncated_string_and_closing_braces() -> None:
    payload = '{"answer":"ok","meta":{"note":"unterminated'

    parsed = _parse_json_payload(payload)

    assert parsed["answer"] == "ok"
    assert parsed["meta"]["note"] == "unterminated"


def test_parse_json_payload_extracts_json_from_prefixed_text() -> None:
    payload = 'Here is the JSON:\n{"answer":"ok","extra":["a","b"]}\nThanks.'

    parsed = _parse_json_payload(payload)

    assert parsed["answer"] == "ok"
    assert parsed["extra"] == ["a", "b"]


def test_parse_json_payload_repairs_bare_quotes_inside_string_values() -> None:
    payload = """
    {
      "proposals": [
        {
          "source_turn_index": 1,
          "intent": "store_fact",
          "candidate_content": "The musical "Hadestown" won several awards."
        }
      ]
    }
    """

    parsed = _parse_json_payload(payload)

    assert parsed["proposals"][0]["candidate_content"] == 'The musical "Hadestown" won several awards.'
