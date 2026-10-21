"""Unified LLM accounting helper for baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Optional

from client import model_client


class BaselineLLMTracker:
    """Call the model client while collecting per-operation usage."""

    def __init__(self) -> None:
        self.llm_call_count = 0
        self.calls_by_operation: Counter[str] = Counter()
        self._usage_available_by_operation: Counter[str] = Counter()
        self._usage_unavailable_by_operation: Counter[str] = Counter()
        self._tokens_by_operation: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.usage_available_call_count = 0
        self.usage_unavailable_call_count = 0

    def call(
        self,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        self.llm_call_count += 1
        self.calls_by_operation[operation] += 1
        raw = model_client.run_llm(
            system_prompt,
            user_prompt,
            model=model,
            response_format=response_format,
            return_usage=True,
        )
        text = getattr(raw, "text", raw)
        usage = getattr(raw, "usage", None)
        if usage is None:
            self.usage_unavailable_call_count += 1
            self._usage_unavailable_by_operation[operation] += 1
            return str(text or "")

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens <= 0 and input_tokens + output_tokens > 0:
            total_tokens = input_tokens + output_tokens

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.usage_available_call_count += 1
        self._usage_available_by_operation[operation] += 1
        bucket = self._tokens_by_operation[operation]
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += total_tokens
        return str(text or "")

    def token_usage_payload(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_available_call_count": self.usage_available_call_count,
            "usage_unavailable_call_count": self.usage_unavailable_call_count,
            "by_operation": {
                operation: {
                    "input_tokens": self._tokens_by_operation[operation][
                        "input_tokens"
                    ],
                    "output_tokens": self._tokens_by_operation[operation][
                        "output_tokens"
                    ],
                    "total_tokens": self._tokens_by_operation[operation][
                        "total_tokens"
                    ],
                    "call_count": self._usage_available_by_operation[operation],
                    "usage_available_call_count": self._usage_available_by_operation[
                        operation
                    ],
                    "usage_unavailable_call_count": (
                        self._usage_unavailable_by_operation[operation]
                    ),
                }
                for operation in sorted(self.calls_by_operation)
            },
        }

    def calls_by_operation_payload(self) -> Dict[str, int]:
        return dict(sorted(self.calls_by_operation.items()))
