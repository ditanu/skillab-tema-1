import argparse
from dataclasses import dataclass
import os
import time

import httpx

from agent import QAAgent


DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = "ollama/mistral:7b"


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] | None = None


class LiteLLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
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

        response = self._post_with_retries(payload)

        message = self._extract_message(response.json())
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls"),
        )

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
    args = parser.parse_args()

    llm = LiteLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    agent = QAAgent(llm)
    print(agent.answer(args.question))


if __name__ == "__main__":
    main()
