import argparse
import os
from pathlib import Path

from agent import QAAgent
from memory import InMemoryConversationMemory
from prompt_cache import PromptCacheMetrics
from run_agent import DEFAULT_BASE_URL, LiteLLMClient


DEFAULT_CONTEXT = (
    "Contract de prestari servicii intre TechCorp SRL si furnizorul ACME. "
    "Termenii includ livrare in 30 de zile, penalizare 0.1% pe zi de intarziere, "
    "TVA 19% aplicat la valoarea neta, plata in 15 zile de la factura. "
    "Clauzele de confidentialitate si forta majora sunt aplicabile. "
) * 90


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testeaza Anthropic prompt caching cu cache_control: ephemeral."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LITELLM_MODEL", "anthropic/claude-sonnet-4-20250514"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--fixed-context-file",
        default=os.getenv("FIXED_CONTEXT_FILE"),
        help="Optional. Daca lipseste, scriptul foloseste un context static mare.",
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LITELLM_TIMEOUT", "120")))
    args = parser.parse_args()

    fixed_context = DEFAULT_CONTEXT

    if args.fixed_context_file:
        fixed_context = Path(args.fixed_context_file).read_text(encoding="utf-8")

    metrics = PromptCacheMetrics()
    llm = LiteLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        prompt_caching=True,
        cache_metrics=metrics,
    )
    agent = QAAgent(
        llm,
        memory=InMemoryConversationMemory(),
        fixed_context=fixed_context,
    )

    print("Apel 1: astept cache_creation_input_tokens > 0")
    print(agent.answer("Care sunt termenii principali din contextul fix?", session_id="prompt-cache-test"))
    print()
    print(metrics.format_report())

    print("\nApel 2: astept cache_read_input_tokens > 0")
    print(agent.answer("Exista penalizare pentru intarziere?", session_id="prompt-cache-test"))
    print()
    print(metrics.format_report())


if __name__ == "__main__":
    main()
