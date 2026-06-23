from __future__ import annotations

import csv
import pickle
import time
from dataclasses import dataclass
from pathlib import Path


LABELS = ("search", "extract", "summarize")
DEFAULT_DATASET = Path("data/intent_queries.csv")
DEFAULT_MODEL_PATH = Path("models/intent_classifier.pkl")


@dataclass
class IntentExample:
    query: str
    label: str


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    latency_seconds: float


def load_examples(path: str | Path = DEFAULT_DATASET) -> list[IntentExample]:
    examples = []

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            label = row["label"].strip()

            if label not in LABELS:
                raise ValueError(f"Label invalid: {label!r}. Valori permise: {LABELS}")

            examples.append(IntentExample(query=row["query"].strip(), label=label))

    return examples


def split_examples(
    examples: list[IntentExample],
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[list[IntentExample], list[IntentExample]]:
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state,
        stratify=[example.label for example in examples],
    )

    return list(train), list(test)


def train_intent_classifier(
    examples: list[IntentExample],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline

    train_examples, test_examples = split_examples(examples)
    train_x = [example.query for example in train_examples]
    train_y = [example.label for example in train_examples]
    test_x = [example.query for example in test_examples]
    test_y = [example.label for example in test_examples]

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    started_at = time.perf_counter()
    pipeline.fit(train_x, train_y)
    train_latency = time.perf_counter() - started_at

    predictions = pipeline.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    return {
        "model_path": str(model_path),
        "train_size": len(train_x),
        "test_size": len(test_x),
        "train_latency_seconds": train_latency,
        "accuracy": accuracy,
        "classification_report": classification_report(
            test_y,
            predictions,
            labels=list(LABELS),
            zero_division=0,
        ),
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict_intent(model, query: str) -> IntentPrediction:
    started_at = time.perf_counter()
    label = model.predict([query])[0]
    probabilities = model.predict_proba([query])[0]
    latency = time.perf_counter() - started_at
    confidence = max(probabilities)

    return IntentPrediction(
        label=label,
        confidence=confidence,
        latency_seconds=latency,
    )


def train_from_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression intent classifier.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    report = train_intent_classifier(
        examples=load_examples(args.dataset),
        model_path=args.model_path,
    )

    print(f"Saved model: {report['model_path']}")
    print(f"Train size: {report['train_size']} | Test size: {report['test_size']}")
    print(f"Train latency: {report['train_latency_seconds']:.4f}s")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(report["classification_report"])


if __name__ == "__main__":
    train_from_cli()
