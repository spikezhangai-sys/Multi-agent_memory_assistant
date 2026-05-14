from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class StructuredLLM(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel | dict[str, Any]:
        ...

