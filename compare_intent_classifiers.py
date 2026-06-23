from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

from intent_classifier import (
    DEFAULT_DATASET,
    DEFAULT_MODEL_PATH,
    LABELS,
    IntentExample,
    load_examples,
    load_model,
    predict_intent,
    split_examples,
    train_intent_classifier,
)
from run_agent import DEFAULT_BASE_URL, DEFAULT_MODEL, LiteLLMClient


LLM_INTENT_SYSTEM_PROMPT = f"""
Clasifica intentia query-ului user-ului.

Raspunde cu exact unul dintre aceste label-uri:
{", ".join(LABELS)}

Definitii:
- search: user-ul vrea sa caute informatii sau sa gaseasca raspunsuri in knowledge base.
- extract: user-ul vrea sa extraga campuri, entitati sau valori structurate din text/document.
- summarize: user-ul vrea un rezumat, o sinteza sau o concluzie scurta.

Nu explica alegerea.
""".strip()


@dataclass
class PredictionResult:
    expected: str
    predicted: str
    latency_seconds: float
    cost_usd: float = 0.0

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LLM intent classification with TF-IDF + LogisticRegression."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--retrain", action="store_true", help="Reantreneaza modelul inainte de benchmark.")
    parser.add_argument("--skip-llm", action="store_true", help="Ruleaza doar classifier-ul local.")
    parser.add_argument("--model", default=os.getenv("LITELLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LITELLM_TIMEOUT", "120")))
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=float(os.getenv("LLM_INPUT_COST_PER_1M", "3.0")),
        help="Cost USD per 1M input tokens pentru estimare.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=float(os.getenv("LLM_OUTPUT_COST_PER_1M", "15.0")),
        help="Cost USD per 1M output tokens pentru estimare.",
    )
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    _, eval_examples = split_examples(examples)

    if args.retrain or not os.path.exists(args.model_path):
        train_report = train_intent_classifier(examples, args.model_path)
        print("Training report:")
        print(json.dumps({k: v for k, v in train_report.items() if k != "classification_report"}, indent=2))
        print(train_report["classification_report"])

    model = load_model(args.model_path)
    local_results = run_local_classifier(model, eval_examples)
    llm_results = []

    if not args.skip_llm:
        llm = LiteLLMClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
        )
        llm_results = run_llm_classifier(
            llm=llm,
            examples=eval_examples,
            input_cost_per_1m=args.input_cost_per_1m,
            output_cost_per_1m=args.output_cost_per_1m,
        )

    print_report(local_results, llm_results)


def run_local_classifier(model, examples: list[IntentExample]) -> list[PredictionResult]:
    results = []

    for example in examples:
        prediction = predict_intent(model, example.query)
        results.append(
            PredictionResult(
                expected=example.label,
                predicted=prediction.label,
                latency_seconds=prediction.latency_seconds,
                cost_usd=0.0,
            )
        )

    return results


def run_llm_classifier(
    llm: LiteLLMClient,
    examples: list[IntentExample],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> list[PredictionResult]:
    results = []

    for example in examples:
        started_at = time.perf_counter()
        response = llm.invoke(
            [
                {"role": "system", "content": LLM_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": example.query},
            ],
            tools=None,
        )
        latency = response.latency_seconds or (time.perf_counter() - started_at)
        predicted = normalize_label(response.content)
        cost = estimate_cost(
            usage=response.usage or {},
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        results.append(
            PredictionResult(
                expected=example.label,
                predicted=predicted,
                latency_seconds=latency,
                cost_usd=cost,
            )
        )

    return results


def normalize_label(text: str) -> str:
    cleaned = text.strip().lower()

    for label in LABELS:
        if cleaned == label or cleaned.startswith(label):
            return label

    for label in LABELS:
        if label in cleaned:
            return label

    return cleaned


def estimate_cost(
    usage: dict,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> float:
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_tokens_details", {}).get("accepted_prediction_tokens")
        or 0
    )
    return (input_tokens / 1_000_000 * input_cost_per_1m) + (
        output_tokens / 1_000_000 * output_cost_per_1m
    )


def summarize(results: list[PredictionResult]) -> dict:
    if not results:
        return {}

    latencies = [result.latency_seconds for result in results]

    return {
        "examples": len(results),
        "accuracy": sum(result.correct for result in results) / len(results),
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p95_latency_ms": percentile(latencies, 0.95) * 1000,
        "total_cost_usd": sum(result.cost_usd for result in results),
    }


def percentile(values: list[float], rank: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * rank))
    return ordered[index]


def print_report(
    local_results: list[PredictionResult],
    llm_results: list[PredictionResult],
) -> None:
    local = summarize(local_results)
    llm = summarize(llm_results)

    print("\nComparison:")
    print(json.dumps({"sklearn_classifier": local, "llm": llm or "skipped"}, indent=2))

    if llm_results:
        latency_speedup = llm["avg_latency_ms"] / local["avg_latency_ms"]
        cost_savings = llm["total_cost_usd"] - local["total_cost_usd"]
        accuracy_delta = local["accuracy"] - llm["accuracy"]
        print()
        print(f"Latency speedup classifier vs LLM: {latency_speedup:.1f}x")
        print(f"Cost saved on this benchmark: ${cost_savings:.6f}")
        print(f"Accuracy delta classifier - LLM: {accuracy_delta:+.3f}")

        misses = [
            (index, local_results[index], llm_results[index])
            for index in range(len(local_results))
            if not local_results[index].correct or not llm_results[index].correct
        ]

        if misses:
            print("\nMisclassifications:")
            for index, local_result, llm_result in misses:
                print(
                    f"- row {index + 1}: expected={local_result.expected}, "
                    f"sklearn={local_result.predicted}, llm={llm_result.predicted}"
                )


if __name__ == "__main__":
    main()
