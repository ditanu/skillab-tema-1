import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import time

import httpx

from agent import QAAgent
from memory import build_conversation_memory
from prompt_cache import PromptCacheMetrics


DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = "ollama/mistral:7b"
DEFAULT_SESSION_ID = "default"


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] | None = None
    usage: dict | None = None
    latency_seconds: float = 0.0


class LiteLLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 2,
        prompt_caching: bool = False,
        cache_metrics: PromptCacheMetrics | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_caching = prompt_caching
        self.cache_metrics = cache_metrics

    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("input_schema") or tool["parameters"],
                    },
                }
                for tool in tools
            ]

        started_at = time.perf_counter()
        response = self._post_with_retries(payload)
        latency_seconds = time.perf_counter() - started_at

        data = response.json()
        message = self._extract_message(data)
        usage = data.get("usage") or {}

        if self.cache_metrics is not None:
            self.cache_metrics.record(usage=usage, latency_seconds=latency_seconds)

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls"),
            usage=usage,
            latency_seconds=latency_seconds,
        )

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        prepared = []

        for message in messages:
            prepared_message = dict(message)
            cache_control = prepared_message.pop("cache_control", None)

            if (
                self.prompt_caching
                and
                cache_control
                and prepared_message.get("role") == "system"
                and isinstance(prepared_message.get("content"), str)
            ):
                prepared_message["content"] = [
                    {
                        "type": "text",
                        "text": prepared_message["content"],
                        "cache_control": cache_control,
                    }
                ]

            prepared.append(prepared_message)

        return prepared

    @staticmethod
    def _extract_message(data: dict) -> dict:
        if data.get("error"):
            raise RuntimeError(f"LLM returned an error response: {data['error']}")

        choices = data.get("choices")

        if not choices:
            raise RuntimeError(f"LLM response did not contain any choices: {data}")

        message = choices[0].get("message")

        if not message:
            raise RuntimeError(f"LLM response choice did not contain a message: {data}")

        return message

    def _post_with_retries(self, payload: dict) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, json=payload, timeout=self.timeout)
            except httpx.TimeoutException as error:
                raise RuntimeError(
                    f"LLM request timed out after {self.timeout:g}s. "
                    "Dacă folosești Ollama local, modelul poate fi încă la prima încărcare "
                    "sau prea lent pentru apelul curent. Încearcă din nou, crește "
                    "LITELLM_TIMEOUT sau folosește un model mai mic."
                ) from error
            except httpx.RequestError as error:
                raise RuntimeError(
                    "Nu pot contacta serverul LiteLLM. Verifică dacă infrastructura este pornită "
                    "și dacă URL-ul este corect. "
                    f"Detalii: {error}"
                ) from error

            if response.status_code != 429:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    raise RuntimeError(
                        f"LLM request failed with HTTP {response.status_code}: "
                        f"{response.text}"
                    ) from error

                return response

            if attempt == self.max_retries:
                raise RuntimeError(
                    "LLM request failed with HTTP 429 Too Many Requests. "
                    "Verifică disponibilitatea modelului sau încearcă din nou mai târziu. "
                    f"Detalii server: {response.text}"
                )

            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2 ** attempt
            time.sleep(delay)

        raise RuntimeError("LLM request failed unexpectedly.")


def main():
    parser = argparse.ArgumentParser(description="Run the local QAAgent.")
    parser.add_argument("question", nargs="?", default="Care este programul de suport?")
    parser.add_argument(
        "--model",
        default=os.getenv("LITELLM_MODEL", DEFAULT_MODEL),
        help="Modelul trimis către serverul LiteLLM/OpenAI-compatible.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL),
        help="URL-ul serverului OpenAI-compatible, de exemplu http://localhost:4000/v1.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("LITELLM_TIMEOUT", "120")),
        help="Timeout HTTP în secunde pentru fiecare apel către model.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("LITELLM_MAX_RETRIES", "2")),
        help="Numărul de retry-uri pentru răspunsuri HTTP 429.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("AGENT_SESSION_ID", DEFAULT_SESSION_ID),
        help="Identificatorul conversației folosit pentru memorie persistentă.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help=(
            "URL PostgreSQL pentru long-term memory. "
            "Exemplu: postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab"
        ),
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=int(os.getenv("AGENT_MEMORY_WINDOW", "10")),
        help="Numărul maxim de mesaje recente injectate în context.",
    )
    parser.add_argument(
        "--anthropic-prompt-cache",
        action="store_true",
        default=os.getenv("ANTHROPIC_PROMPT_CACHE", "").lower() in {"1", "true", "yes"},
        help="Activează cache_control: ephemeral pentru prefixul static Anthropic.",
    )
    parser.add_argument(
        "--show-cache-metrics",
        action="store_true",
        default=os.getenv("SHOW_CACHE_METRICS", "").lower() in {"1", "true", "yes"},
        help="Afișează tokenii economisiți și latența pentru prompt caching.",
    )
    parser.add_argument(
        "--fixed-context-file",
        default=os.getenv("FIXED_CONTEXT_FILE"),
        help="Fișier cu context static inclus în system prompt și eligibil pentru prompt caching.",
    )
    args = parser.parse_args()
    cache_metrics = PromptCacheMetrics()
    fixed_context = _read_fixed_context(args.fixed_context_file)

    llm = LiteLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        prompt_caching=args.anthropic_prompt_cache,
        cache_metrics=cache_metrics,
    )
    memory = build_conversation_memory(
        database_url=args.database_url,
        window=args.memory_window,
    )
    agent = QAAgent(llm, memory=memory, fixed_context=fixed_context)
    print(agent.answer(args.question, session_id=args.session_id))

    if args.show_cache_metrics:
        print()
        print(cache_metrics.format_report())


def _read_fixed_context(path: str | None) -> str:
    if not path:
        return ""

    return Path(path).read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
