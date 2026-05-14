import json

from pydantic import BaseModel
import requests

from driftscope.llm.deepseek import DeepSeekStructuredLLM


class DemoResponse(BaseModel):
    answer: str


class FakeResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def test_deepseek_client_parses_json_response(monkeypatch) -> None:
    client = DeepSeekStructuredLLM(
        api_key="test-key",
        model="deepseekv4flash",
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

    monkeypatch.setattr("driftscope.llm.deepseek.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert result.answer == "ok"


def test_deepseek_client_uses_official_json_output(monkeypatch) -> None:
    client = DeepSeekStructuredLLM(
        api_key="test-key",
        model="deepseek/deepseek-v4-flash",
    )
    captured = {}

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
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

    monkeypatch.setattr("driftscope.llm.deepseek.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert result.answer == "ok"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["response_format"] == {"type": "json_object"}
    assert "provider" not in captured
    assert "reasoning" not in captured


def test_deepseek_client_retries_transient_ssl_error(monkeypatch) -> None:
    client = DeepSeekStructuredLLM(
        api_key="test-key",
        model="deepseek-v4-flash",
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

    monkeypatch.setattr("driftscope.llm.deepseek.requests.post", fake_post)
    result = client.generate_structured(
        system_prompt="Return JSON",
        user_prompt="{}",
        response_model=DemoResponse,
    )

    assert calls["count"] == 2
    assert result.answer == "ok"
